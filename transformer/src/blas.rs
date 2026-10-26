//! Optional: a BLAS library for the large f32 matrix products (PLAN.md, open
//! decision 8). BLAS (Basic Linear Algebra Subprograms) is the standard
//! interface for them; `cblas_sgemm` is its f32 matrix product.
//!
//! Built only with `cargo build --release --features blas`:
//! - on macOS it links Apple's Accelerate framework, whose BLAS uses the
//!   chip's matrix hardware
//! - elsewhere it links whatever CBLAS library the build is given, such as
//!   `RUSTFLAGS="-l blas"` on Linux (how this code is tested off a Mac)
//!
//! Without the feature, every product uses the Rust loops in tensor.rs, and
//! this module does nothing. With it, the library is used only for f32
//! products with at least PARALLEL_MIN_WORK multiply-adds, called from
//! outside our thread pool: the big products of a training step, not the
//! many small ones attention runs side by side in the pool (the library
//! uses its own threads). Results are equal to the Rust loops' to rounding,
//! not bit for bit, since the library sums in its own order.

use crate::tensor::PARALLEL_MIN_WORK;
use std::any::TypeId;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "blas")]
mod imp {
    use std::os::raw::c_int;

    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;

    #[cfg_attr(target_os = "macos", link(name = "Accelerate", kind = "framework"))]
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }

    pub const BUILT: bool = true;

    /// C (m × n) = A·B, or Aᵀ·B if `trans_a`; all row-major. A is m × k, or
    /// k × m when transposed; B is k × n.
    pub fn sgemm(trans_a: bool, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == m * k && b.len() == k * n && c.len() == m * n);
        let fits = |x: usize| c_int::try_from(x).expect("matrix too large for BLAS");
        let lda = if trans_a { m } else { k };
        // Safety: the slices have exactly the sizes the call reads and writes.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                if trans_a { TRANS } else { NO_TRANS },
                NO_TRANS,
                fits(m),
                fits(n),
                fits(k),
                1.0,
                a.as_ptr(),
                fits(lda),
                b.as_ptr(),
                fits(n),
                0.0,
                c.as_mut_ptr(),
                fits(n),
            );
        }
    }
}

#[cfg(not(feature = "blas"))]
mod imp {
    pub const BUILT: bool = false;
    pub fn sgemm(_: bool, _: usize, _: usize, _: usize, _: &[f32], _: &[f32], _: &mut [f32]) {
        unreachable!("built without the blas feature")
    }
}

/// Switched off by `bench` to time the Rust loops in the same binary.
static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Whether the library is built in and switched on.
pub fn active() -> bool {
    imp::BUILT && ENABLED.load(Ordering::Relaxed)
}

/// What computes the big matrix products, for the logs.
pub fn backend() -> &'static str {
    match (imp::BUILT, cfg!(target_os = "macos")) {
        (false, _) => "Rust loops (src/tensor.rs)",
        (true, true) => "Apple Accelerate (BLAS)",
        (true, false) => "a CBLAS library",
    }
}

/// C = A·B (or Aᵀ·B) with the library, if it applies: f32, large enough,
/// and called from outside our thread pool. Returns whether it did.
pub fn try_gemm<T: 'static>(
    trans_a: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &[T],
    b: &[T],
    c: &mut [T],
) -> bool {
    if !active()
        || m * n * k < PARALLEL_MIN_WORK
        || rayon::current_thread_index().is_some()
        || TypeId::of::<T>() != TypeId::of::<f32>()
    {
        return false;
    }
    // Safety: T is f32 (checked just above), so these are the same slices.
    let (a, b, c) = unsafe {
        (
            std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()),
            std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()),
            std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()),
        )
    };
    imp::sgemm(trans_a, m, n, k, a, b, c);
    true
}

#[cfg(all(test, feature = "blas"))]
mod tests {
    use crate::rng::Rng;
    use crate::tensor::Mat;

    fn rand(r: usize, c: usize, seed: u64) -> Mat<f32> {
        let mut rng = Rng::new(seed);
        Mat::from_vec(r, c, (0..r * c).map(|_| rng.normal() as f32).collect())
    }

    /// The library's products against the Rust loops (computed inside a
    /// thread pool, where the library is never used), within f32 rounding.
    #[test]
    fn library_matches_the_rust_loops() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for (r, k, c) in [(300, 200, 150), (64, 1024, 257), (1000, 64, 300)] {
            let (a, b) = (rand(r, k, 1), rand(k, c, 2));
            let fast = a.matmul(&b);
            let slow = pool.install(|| a.matmul(&b));
            let at = rand(k, r, 3); // for Aᵀ·B: A is k × r
            let fast_tn = at.matmul_tn(&b);
            let slow_tn = pool.install(|| at.matmul_tn(&b));
            for (x, y) in fast
                .data
                .iter()
                .zip(&slow.data)
                .chain(fast_tn.data.iter().zip(&slow_tn.data))
            {
                // Sums of k products of normal numbers: errors of order
                // √k · 1e-7 · √k, far below 1e-3 absolute here.
                assert!(
                    (x - y).abs() < 1e-3 * (1.0 + y.abs()),
                    "{x} vs {y} at {r}×{k}×{c}"
                );
            }
            assert_ne!(fast.data, vec![0.0; fast.data.len()]); // it really computed
        }
    }
}
