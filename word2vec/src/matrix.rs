//! Shared building blocks: a weight matrix that many threads can update at
//! once, a small random number generator, and vector helpers.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// A rows x dim matrix of f32 weights, shared by all training threads.
///
/// word2vec trains with "Hogwild!" updates: every thread reads and writes the
/// shared weights with no locks at all. Two threads occasionally update the
/// same row at the same moment and one update is lost, but with tens of
/// thousands of rows such collisions are rare and the noise is harmless to
/// SGD. The original C code does this with plain memory writes. In Rust a
/// plain unsynchronized write would be undefined behavior, so each weight is
/// stored as the bits of an f32 in an `AtomicU32` and accessed with `Relaxed`
/// loads and stores. On x86 and ARM those compile to ordinary loads and
/// stores, so this costs nothing and keeps the program well-defined.
pub struct Matrix {
    data: Vec<AtomicU32>,
    dim: usize,
}

impl Matrix {
    /// A matrix of zeros.
    pub fn zeros(rows: usize, dim: usize) -> Matrix {
        let data = (0..rows * dim).map(|_| AtomicU32::new(0)).collect();
        Matrix { data, dim }
    }

    /// A matrix of values drawn uniformly from [-scale, scale).
    pub fn uniform(rows: usize, dim: usize, scale: f32, rng: &mut Rng) -> Matrix {
        let data = (0..rows * dim)
            .map(|_| AtomicU32::new(((rng.unit() * 2.0 - 1.0) * scale).to_bits()))
            .collect();
        Matrix { data, dim }
    }

    pub fn rows(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Copy row `r` into `out`.
    pub fn read_row(&self, r: usize, out: &mut [f32]) {
        let row = &self.data[r * self.dim..(r + 1) * self.dim];
        for (o, w) in out.iter_mut().zip(row) {
            *o = f32::from_bits(w.load(Relaxed));
        }
    }

    /// Row `r` += `delta`. Each element is a separate load and store, not an
    /// atomic read-modify-write: if another thread writes the same weight in
    /// between, one of the two updates is lost (Hogwild!, see above).
    pub fn add_row(&self, r: usize, delta: &[f32]) {
        let row = &self.data[r * self.dim..(r + 1) * self.dim];
        for (w, d) in row.iter().zip(delta) {
            w.store((f32::from_bits(w.load(Relaxed)) + d).to_bits(), Relaxed);
        }
    }
}

/// SplitMix64: a tiny, fast, good-quality pseudo-random generator. Each
/// thread has its own, so no locking is needed and runs are reproducible
/// for a given seed and thread count.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform integer in 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A uniform f32 in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ln(1 + e^x), computed without overflow. The loss terms are
/// -ln sigmoid(x) = softplus(-x) and -ln(1 - sigmoid(x)) = softplus(x).
pub fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_reproducible_and_in_range() {
        let (mut a, mut b) = (Rng::new(9), Rng::new(9));
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
            let u = a.unit();
            assert!((0.0..1.0).contains(&u));
            assert!(a.below(7) < 7);
            b.unit();
            b.below(7);
        }
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }

    #[test]
    fn rows_read_back_what_was_added() {
        let m = Matrix::zeros(3, 2);
        m.add_row(1, &[1.5, -2.0]);
        m.add_row(1, &[0.5, 1.0]);
        let mut row = [0.0; 2];
        m.read_row(1, &mut row);
        assert_eq!(row, [2.0, -1.0]);
        m.read_row(0, &mut row);
        assert_eq!(row, [0.0, 0.0]);
        assert_eq!((m.rows(), m.dim()), (3, 2));
    }

    #[test]
    fn uniform_stays_within_scale() {
        let m = Matrix::uniform(50, 4, 0.25, &mut Rng::new(3));
        let mut row = [0.0; 4];
        for r in 0..m.rows() {
            m.read_row(r, &mut row);
            assert!(row.iter().all(|x| (-0.25..0.25).contains(x)));
        }
    }

    /// softplus(x) = ln(1 + e^x), also where e^x would overflow; and the loss
    /// identity it serves: -ln sigmoid(x) = softplus(-x).
    #[test]
    fn softplus_and_sigmoid() {
        for x in [-20.0f64, -1.0, 0.0, 0.5, 10.0] {
            assert!((softplus(x) - (1.0 + x.exp()).ln()).abs() < 1e-12);
            let s = f64::from(sigmoid(x as f32));
            assert!((-s.ln() - softplus(-x)).abs() < 1e-6);
        }
        assert_eq!(softplus(1000.0), 1000.0);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, -5.0, 6.0]), 12.0);
    }
}
