//! Copied from tiny-diffusion-in-rust (diffusion/src/optim.rs) for the
//! classifier (PLAN.md, Drawing), which it trains.
//!
//! Adam, the optimizer (Kingma and Ba 2015): each
//! weight moves by its gradient's running average divided by the running
//! average of its square's root, so every weight takes steps of about the
//! learning rate whatever its gradient's scale. No weight decay here: the
//! classifier's training is short and checked on held-out images.

use crate::tensor::Mat;

pub struct Adam {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    m: Vec<Mat<f32>>,
    v: Vec<Mat<f32>>,
    step: i32,
}

impl Adam {
    /// Running averages for parameters of these shapes, starting at zero.
    pub fn new(params: &[Mat<f32>]) -> Adam {
        let zeros = || params.iter().map(|p| Mat::zeros(p.rows, p.cols)).collect();
        Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: zeros(),
            v: zeros(),
            step: 0,
        }
    }

    /// One update of every parameter, at learning rate `lr`. The running
    /// averages start at zero, so early on they underestimate; dividing by
    /// 1 − β^step corrects that ("bias correction").
    pub fn update(&mut self, params: &mut [Mat<f32>], grads: &[Mat<f32>], lr: f32) {
        assert_eq!(params.len(), grads.len());
        self.step += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let c1 = 1.0 - b1.powi(self.step);
        let c2 = 1.0 - b2.powi(self.step);
        for ((p, g), (m, v)) in params
            .iter_mut()
            .zip(grads)
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            for i in 0..p.data.len() {
                let gi = g.data[i];
                m.data[i] = b1 * m.data[i] + (1.0 - b1) * gi;
                v.data[i] = b2 * v.data[i] + (1.0 - b2) * gi * gi;
                let mh = m.data[i] / c1;
                let vh = v.data[i] / c2;
                p.data[i] -= lr * mh / (vh.sqrt() + self.eps);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first step moves every weight by almost exactly the learning
    /// rate, against its gradient's sign, whatever the gradient's size.
    #[test]
    fn first_step_is_the_learning_rate() {
        let mut p = vec![Mat::from_vec(1, 3, vec![1.0, 1.0, 1.0])];
        let g = vec![Mat::from_vec(1, 3, vec![1000.0, -0.001, 3.0])];
        Adam::new(&p).update(&mut p, &g, 0.1);
        let want = [0.9, 1.1, 0.9];
        for (a, b) in p[0].data.iter().zip(want) {
            assert!((a - b).abs() < 1e-4, "{:?}", p[0].data);
        }
    }

    /// It minimizes a quadratic: (x − 3)², from x = 0.
    #[test]
    fn finds_a_minimum() {
        let mut p = vec![Mat::from_vec(1, 1, vec![0.0])];
        let mut adam = Adam::new(&p);
        for _ in 0..2000 {
            let g = vec![Mat::from_vec(1, 1, vec![2.0 * (p[0].data[0] - 3.0)])];
            adam.update(&mut p, &g, 0.01);
        }
        assert!((p[0].data[0] - 3.0).abs() < 1e-2, "{}", p[0].data[0]);
    }
}
