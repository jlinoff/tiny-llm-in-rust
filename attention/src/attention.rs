//! One attention head, with every intermediate result kept so it can be
//! printed. This is the computation described in PLAN.md, section "How
//! attention works: queries, keys and values".

use crate::matrix::Mat;

/// The three weight matrices of one head, each d × head_size.
pub struct Head {
    pub wq: Mat,
    pub wk: Mat,
    pub wv: Mat,
}

/// Everything one head computes for an input X (T tokens × d).
pub struct Trace {
    /// Queries, keys, values: T × head_size, row i belongs to token i.
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    /// Q · Kᵀ / √(head_size): T × T, entry (i, j) = how well token i's
    /// query matches token j's key.
    pub scores: Mat,
    /// The scores with every j > i (a later token) set to −∞.
    pub masked: Mat,
    /// Softmax of each masked row: T × T, each row sums to 1.
    pub weights: Mat,
    /// Weights · V: T × head_size, row i = token i's weighted sum of values.
    pub out: Mat,
}

impl Head {
    /// Run the head on X, keeping every step.
    pub fn forward(&self, x: &Mat) -> Trace {
        attend(x, &self.wq, &self.wk, &self.wv)
    }
}

/// One head's attention on X with the given weights, keeping every step.
pub fn attend(x: &Mat, wq: &Mat, wk: &Mat, wv: &Mat) -> Trace {
    let head_size = wq.cols;
    // Step 1: project every token three ways.
    let q = x.matmul(wq);
    let k = x.matmul(wk);
    let v = x.matmul(wv);
    // Step 2: score every query against every key, scaled by 1/√(head size)
    // so the scores don't grow with the head size and saturate the softmax.
    let scores = q
        .matmul(&k.transpose())
        .scale(1.0 / (head_size as f64).sqrt());
    // Step 3: the causal mask; a token may look only at itself and earlier.
    let masked = causal_mask(&scores);
    // Step 4: softmax each row into weights that sum to 1.
    let weights = softmax_rows(&masked);
    // Step 5: each token's output is the weighted sum of the values.
    let out = weights.matmul(&v);
    Trace {
        q,
        k,
        v,
        scores,
        masked,
        weights,
        out,
    }
}

/// Set every entry above the diagonal (j > i) to −∞.
pub fn causal_mask(scores: &Mat) -> Mat {
    let mut m = scores.clone();
    for i in 0..m.rows {
        for j in i + 1..m.cols {
            m.set(i, j, f64::NEG_INFINITY);
        }
    }
    m
}

/// Softmax of each row: weight_j = e^(z_j) / Σ e^(z_k).
///
/// The row's largest value is subtracted first. That changes nothing
/// mathematically (a constant added to every score cancels out), but keeps
/// e^z from overflowing. −∞ entries become exactly 0.
pub fn softmax_rows(m: &Mat) -> Mat {
    let mut out = m.clone();
    for r in 0..out.rows {
        let row = out.row_mut(r);
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for z in row.iter_mut() {
            *z = (*z - max).exp();
            sum += *z;
        }
        for z in row.iter_mut() {
            *z /= sum;
        }
    }
    out
}

/// Layer normalization of each row, without the learned scale and shift:
/// subtract the row's mean and divide by its standard deviation.
pub fn layer_norm_rows(x: &Mat) -> Mat {
    let mut out = x.clone();
    for r in 0..out.rows {
        let row = out.row_mut(r);
        let n = row.len() as f64;
        let mean = row.iter().sum::<f64>() / n;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = (var + 1e-5).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) / std;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9; // floating point: compare with a tolerance

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// The worked example from PLAN.md, section "How attention works".
    fn plan_example() -> (Mat, Head) {
        let x = Mat::from_rows(&[&[1., 0., 1., 0.], &[0., 1., 0., 1.], &[1., 1., 0., 0.]]);
        let head = Head {
            wq: Mat::from_rows(&[&[1., 0.], &[0., 1.], &[1., 0.], &[0., 0.]]),
            wk: Mat::from_rows(&[&[0., 1.], &[1., 0.], &[0., 0.], &[1., 1.]]),
            wv: Mat::from_rows(&[&[1., 0.], &[0., 1.], &[0., 0.], &[0., 0.]]),
        };
        (x, head)
    }

    #[test]
    fn reproduces_the_plan_example() {
        let (x, head) = plan_example();
        let t = head.forward(&x);
        assert_eq!(t.q, Mat::from_rows(&[&[2., 0.], &[0., 1.], &[1., 1.]]));
        assert_eq!(t.k, Mat::from_rows(&[&[0., 1.], &[2., 1.], &[1., 1.]]));
        assert_eq!(t.v, Mat::from_rows(&[&[1., 0.], &[0., 1.], &[1., 1.]]));
        // "the" scores "cat" at 4/√2 = 2.83, "cat" scores "the" at 1/√2 = 0.71.
        assert!(close(t.scores.get(0, 1), 4.0 / 2f64.sqrt(), TOL));
        assert!(close(t.scores.get(1, 0), 1.0 / 2f64.sqrt(), TOL));
        // The row for "sat": 0.14, 0.58, 0.28 as printed in the plan.
        let sat = t.weights.row(2);
        assert!(
            close(sat[0], 0.14, 0.005) && close(sat[1], 0.58, 0.005) && close(sat[2], 0.28, 0.005)
        );
        // sat's output: [0.42 0.86].
        assert!(close(t.out.get(2, 0), 0.42, 0.005) && close(t.out.get(2, 1), 0.86, 0.005));
    }

    #[test]
    fn weights_rows_sum_to_one_and_mask_gives_zero() {
        let (x, head) = plan_example();
        let w = head.forward(&x).weights;
        for i in 0..w.rows {
            assert!(close(w.row(i).iter().sum::<f64>(), 1.0, TOL));
            for j in i + 1..w.cols {
                assert_eq!(w.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn softmax_handles_huge_scores() {
        let m = Mat::from_rows(&[&[1000., 1001., f64::NEG_INFINITY]]);
        let w = softmax_rows(&m);
        assert!(w.data.iter().all(|v| v.is_finite()));
        assert!(close(w.get(0, 1) / w.get(0, 0), std::f64::consts::E, 1e-9));
        assert_eq!(w.get(0, 2), 0.0);
    }

    #[test]
    fn layer_norm_gives_mean_zero_variance_one() {
        let x = Mat::from_rows(&[&[1., 2., 3., 10.], &[-5., 0., 5., 0.]]);
        let n = layer_norm_rows(&x);
        for r in 0..n.rows {
            let row = n.row(r);
            let mean = row.iter().sum::<f64>() / 4.0;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
            assert!(close(mean, 0.0, 1e-12) && close(var, 1.0, 1e-5));
        }
    }
}
