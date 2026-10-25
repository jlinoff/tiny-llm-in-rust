//! Adam: gradient descent that keeps running averages of each weight's
//! gradient (m) and squared gradient (v), and scales each step by them,
//! so every weight moves at a sensible rate.

use crate::matrix::Mat;

pub const B1: f64 = 0.9;
pub const B2: f64 = 0.999;

/// One Adam update of one weight matrix w, with gradient g, at step t
/// (counting from 1). m and v are its running averages, same shape as w.
pub fn adam_update(m: &mut Mat, v: &mut Mat, w: &mut Mat, g: &Mat, t: i32, lr: f64) {
    let (c1, c2) = (1.0 - B1.powi(t), 1.0 - B2.powi(t));
    for i in 0..w.data.len() {
        let gi = g.data[i];
        m.data[i] = B1 * m.data[i] + (1.0 - B1) * gi;
        v.data[i] = B2 * v.data[i] + (1.0 - B2) * gi * gi;
        w.data[i] -= lr * (m.data[i] / c1) / ((v.data[i] / c2).sqrt() + 1e-8);
    }
}
