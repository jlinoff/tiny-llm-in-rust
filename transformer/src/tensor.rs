//! A row-major matrix of floats, generic over the float type: f32 for
//! training (half the memory, about twice the speed), f64 for the gradient
//! checks (finite differences in f32 are too noisy to catch subtle errors).
//!
//! The matrix multiply is split into small blocks of rows that `rayon`'s
//! worker threads take as they become free ("work stealing"), so the fast
//! performance cores don't sit waiting for the slower efficiency cores. Each
//! output element is still summed in the same order whatever the number of
//! threads, so results don't depend on it.

use rayon::prelude::*;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// The float operations the model needs, for f32 and f64.
pub trait Float:
    Copy
    + Debug
    + Default
    + PartialOrd
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
{
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn tanh(self) -> Self;
    fn zero() -> Self {
        Self::from_f64(0.0)
    }
    fn one() -> Self {
        Self::from_f64(1.0)
    }
    fn max(self, other: Self) -> Self {
        if other > self {
            other
        } else {
            self
        }
    }
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
        }
    };
}
impl_float!(f32);
impl_float!(f64);

#[derive(Clone, Debug, PartialEq)]
pub struct Mat<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

/// Rows per block of work in the matrix multiply: small enough that every
/// core gets many blocks (so slow cores don't hold up the rest), large
/// enough that handing out a block costs little.
const BLOCK_ROWS: usize = 8;

impl<T: Float> Mat<T> {
    pub fn zeros(rows: usize, cols: usize) -> Mat<T> {
        Mat {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Mat<T> {
        assert_eq!(rows * cols, data.len(), "shape doesn't match data");
        Mat { rows, cols, data }
    }

    pub fn row(&self, r: usize) -> &[T] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn transpose(&self) -> Mat<T> {
        let mut out = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for (c, &v) in self.row(r).iter().enumerate() {
                out.data[c * self.rows + r] = v;
            }
        }
        out
    }

    /// C = self · other, in parallel over blocks of rows of C. Each element
    /// of C is summed over k in order 0, 1, 2, …, so the result is the same
    /// for any number of threads.
    pub fn matmul(&self, other: &Mat<T>) -> Mat<T> {
        assert_eq!(self.cols, other.rows, "matmul shape mismatch");
        let (k, m) = (self.cols, other.cols);
        let mut out = Mat::zeros(self.rows, m);
        if m == 0 || k == 0 {
            return out;
        }
        out.data
            .par_chunks_mut(BLOCK_ROWS * m)
            .zip(self.data.par_chunks(BLOCK_ROWS * k))
            .for_each(|(c_block, a_block)| {
                for (c_row, a_row) in c_block.chunks_exact_mut(m).zip(a_block.chunks_exact(k)) {
                    for (&a, b_row) in a_row.iter().zip(other.data.chunks_exact(m)) {
                        for (c, &b) in c_row.iter_mut().zip(b_row) {
                            *c += a * b;
                        }
                    }
                }
            });
        out
    }

    /// C = selfᵀ · other, without forming the transpose. Used for weight
    /// gradients (dW = Xᵀ·dY), where the shared dimension is every token in
    /// the batch: computed as a sum of one small update per token,
    /// dW += x_rᵀ · dy_r, so dW stays in the cache while X and dY are read
    /// once. The tokens are split into fixed blocks whose partial sums are
    /// computed in parallel and then added in block order, so the result
    /// doesn't depend on the number of threads.
    pub fn matmul_tn(&self, other: &Mat<T>) -> Mat<T> {
        assert_eq!(self.rows, other.rows, "matmul_tn shape mismatch");
        let (n, m) = (self.cols, other.cols);
        const TOKENS_PER_BLOCK: usize = 64;
        let blocks = self.rows.div_ceil(TOKENS_PER_BLOCK);
        let partials: Vec<Mat<T>> = (0..blocks)
            .into_par_iter()
            .map(|blk| {
                let mut acc = Mat::zeros(n, m);
                let end = ((blk + 1) * TOKENS_PER_BLOCK).min(self.rows);
                for r in blk * TOKENS_PER_BLOCK..end {
                    let (x, dy) = (self.row(r), other.row(r));
                    for (i, &a) in x.iter().enumerate() {
                        for (c, &b) in acc.row_mut(i).iter_mut().zip(dy) {
                            *c += a * b;
                        }
                    }
                }
                acc
            })
            .collect();
        let mut out = Mat::zeros(n, m);
        for p in &partials {
            out.add_assign(p);
        }
        out
    }

    /// self += other, element by element.
    pub fn add_assign(&mut self, other: &Mat<T>) {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "shape mismatch"
        );
        // Element by element, in parallel.
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, &b)| *a += b);
    }

    /// Apply f to every element, in parallel (each element is independent,
    /// so the result doesn't depend on the number of threads).
    pub fn map(&self, f: impl Fn(T) -> T + Sync) -> Mat<T> {
        Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.par_iter().map(|&x| f(x)).collect(),
        }
    }

    /// Combine with another matrix of the same shape, element by element, in
    /// parallel.
    pub fn zip_map(&self, other: &Mat<T>, f: impl Fn(T, T) -> T + Sync) -> Mat<T> {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "shape mismatch"
        );
        Mat {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .par_iter()
                .zip(&other.data)
                .map(|(&x, &y)| f(x, y))
                .collect(),
        }
    }

    /// Convert to another float type (f32 ↔ f64).
    #[cfg(test)]
    pub fn convert<U: Float>(&self) -> Mat<U> {
        Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| U::from_f64(x.to_f64())).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(rows: usize, cols: usize, seed: f64) -> Mat<f64> {
        let data = (0..rows * cols)
            .map(|i| ((i as f64 * 0.37 + seed).sin() * 3.0).round() / 2.0)
            .collect();
        Mat::from_vec(rows, cols, data)
    }

    #[test]
    fn matmul_matches_the_definition_exactly() {
        // 19 rows: not a multiple of the block size, so the last block is short.
        let (a, b) = (sample(19, 7, 0.1), sample(7, 5, 0.9));
        let c = a.matmul(&b);
        for r in 0..19 {
            for col in 0..5 {
                let mut want = 0.0;
                for p in 0..7 {
                    want += a.data[r * 7 + p] * b.data[p * 5 + col];
                }
                assert_eq!(c.data[r * 5 + col], want);
            }
        }
    }

    #[test]
    fn matmul_is_the_same_for_any_number_of_threads() {
        let (a, b) = (sample(100, 33, 0.2), sample(33, 17, 0.4));
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| a.matmul(&b));
        let many = rayon::ThreadPoolBuilder::new()
            .num_threads(7)
            .build()
            .unwrap()
            .install(|| a.matmul(&b));
        assert_eq!(one, many);
    }

    #[test]
    fn matmul_tn_equals_transpose_then_matmul() {
        // 150 rows: more than two blocks, the last one short.
        let (a, b) = (sample(150, 6, 0.5), sample(150, 4, 0.7));
        let want = a.transpose().matmul(&b);
        let got = a.matmul_tn(&b);
        for (x, y) in want.data.iter().zip(&got.data) {
            assert!((x - y).abs() < 1e-9 * (1.0 + x.abs()));
        }
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| a.matmul_tn(&b));
        let many = rayon::ThreadPoolBuilder::new()
            .num_threads(5)
            .build()
            .unwrap()
            .install(|| a.matmul_tn(&b));
        assert_eq!(one, many); // identical, not just close
    }

    #[test]
    fn transpose_twice_is_identity() {
        let a = sample(4, 9, 0.3);
        assert_eq!(a.transpose().transpose(), a);
        assert_eq!(a.transpose().data[2 * 4 + 1], a.data[9 + 2]); // (2, 1) of the transpose is (1, 2)
    }
}
