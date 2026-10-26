//! LoRA, low-rank adaptation (Hu et al. 2021): fine-tuning that freezes the
//! pretrained weights and learns, for each chosen weight matrix W
//! (d_in × d_out), a low-rank update:
//!
//!   W' = W + (α/r) · A·B       A: d_in × r, B: r × d_out, r small
//!
//! A·B has rank at most r: a sum of r outer products. A starts random and B
//! at zero, so training starts exactly at the pretrained model. Only A and
//! B are trained (and only they get Adam's state), and the result, an
//! adapter, is a small file. `merge` folds it into W, giving an ordinary
//! model of the original shape.
//!
//! In our tape the frozen weights still get gradients computed (they're
//! simply never used): the savings here are in the trained weights, the
//! optimizer's state and the checkpoint, not in backpropagation's work.

use crate::config::ModelConfig;
use crate::gpt::Params;
use crate::rng::Rng;
use crate::tensor::{Float, Mat};

/// The weight matrices LoRA may adapt, as named within a block.
pub const TARGETS: [&str; 6] = [
    "attn.w_q", "attn.w_k", "attn.w_v", "attn.w_o", "ffn.w1", "ffn.w2",
];

/// The adapter: an A and a B for each adapted matrix of each block.
#[derive(Clone)]
pub struct Lora<T> {
    pub rank: usize,
    pub alpha: f64,
    /// Full weight names, such as "blocks.0.attn.w_q", in order.
    pub targets: Vec<String>,
    pub a: Vec<Mat<T>>,
    pub b: Vec<Mat<T>>,
}

impl<T: Float> Lora<T> {
    /// A fresh adapter for `kinds` (from TARGETS) in every block: A random
    /// with standard deviation 1/√d_in, B zero.
    pub fn new(
        c: &ModelConfig,
        base: &Params<T>,
        rank: usize,
        alpha: f64,
        kinds: &[String],
        seed: u64,
    ) -> Lora<T> {
        let mut rng = Rng::new(seed);
        let (mut targets, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..c.layers {
            for k in kinds {
                let name = format!("blocks.{i}.{k}");
                let w = &base.mats[base.position(&name)];
                let std = 1.0 / (w.rows as f64).sqrt();
                a.push(Mat::from_vec(
                    w.rows,
                    rank,
                    (0..w.rows * rank)
                        .map(|_| T::from_f64(std * rng.normal()))
                        .collect(),
                ));
                b.push(Mat::zeros(rank, w.cols));
                targets.push(name);
            }
        }
        Lora {
            rank,
            alpha,
            targets,
            a,
            b,
        }
    }

    /// The scale α/r applied to A·B.
    pub fn scale(&self) -> T {
        T::from_f64(self.alpha / self.rank as f64)
    }

    /// The adapter as named matrices ("lora.blocks.0.attn.w_q.a", "….b", …),
    /// the form the training loop and the checkpoint use.
    pub fn to_params(&self) -> Params<T> {
        let mut names = Vec::new();
        let mut mats = Vec::new();
        for (k, t) in self.targets.iter().enumerate() {
            names.push(format!("lora.{t}.a"));
            mats.push(self.a[k].clone());
            names.push(format!("lora.{t}.b"));
            mats.push(self.b[k].clone());
        }
        Params { names, mats }
    }

    /// Back from named matrices (as written by `to_params`).
    pub fn from_params(p: &Params<T>, rank: usize, alpha: f64) -> Result<Lora<T>, String> {
        if p.names.len() % 2 != 0 {
            return Err("an adapter has an A and a B for each matrix".into());
        }
        let (mut targets, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
        for pair in p.names.chunks(2).zip(p.mats.chunks(2)) {
            let ((na, nb), (ma, mb)) = ((&pair.0[0], &pair.0[1]), (&pair.1[0], &pair.1[1]));
            let t = na
                .strip_prefix("lora.")
                .and_then(|n| n.strip_suffix(".a"))
                .ok_or(format!("unexpected adapter tensor {na}"))?;
            if *nb != format!("lora.{t}.b") || ma.cols != rank || mb.rows != rank {
                return Err(format!("adapter tensors for {t} don't match rank {rank}"));
            }
            targets.push(t.to_string());
            a.push(ma.clone());
            b.push(mb.clone());
        }
        Ok(Lora {
            rank,
            alpha,
            targets,
            a,
            b,
        })
    }

    /// The model with the adapter folded in: W + (α/r)·A·B for each target.
    /// The same operations, in the same order, as the forward pass with the
    /// adapter, so the merged model's predictions are identical.
    pub fn merge(&self, base: &Params<T>) -> Params<T> {
        let mut out = base.clone();
        let s = self.scale();
        for (k, t) in self.targets.iter().enumerate() {
            let ab = self.a[k].matmul(&self.b[k]).map(|x| x * s);
            out.mats[base.position(t)].add_assign(&ab);
        }
        out
    }

    /// Check the adapter fits the base model: every target exists, with
    /// A's rows and B's columns matching its shape.
    pub fn check(&self, base: &Params<T>) -> Result<(), String> {
        for (k, t) in self.targets.iter().enumerate() {
            let i = base
                .names
                .iter()
                .position(|n| n == t)
                .ok_or(format!("the base model has no weight {t}"))?;
            let w = &base.mats[i];
            if self.a[k].rows != w.rows || self.b[k].cols != w.cols {
                return Err(format!(
                    "adapter for {t} doesn't fit its {} × {} shape",
                    w.rows, w.cols
                ));
            }
        }
        Ok(())
    }

    /// Numbers in the adapter.
    pub fn size(&self) -> usize {
        self.a.iter().chain(&self.b).map(|m| m.data.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::forward_with;
    use crate::model;

    fn tiny() -> ModelConfig {
        ModelConfig {
            d_model: 8,
            layers: 2,
            heads: 2,
            context: 6,
            d_ff: 16,
            seed: 3,
        }
    }

    fn all_kinds() -> Vec<String> {
        TARGETS.iter().map(|s| s.to_string()).collect()
    }

    fn base(c: &ModelConfig, vocab: usize) -> Params<f64> {
        let mut p = Params::<f64>::from_tensors(&model::init(c, vocab));
        let mut rng = Rng::new(17);
        for m in &mut p.mats {
            for x in &mut m.data {
                *x += 0.3 * rng.normal();
            }
        }
        p
    }

    fn batch(vocab: usize, n: usize, seed: u64) -> (Vec<usize>, Vec<Option<usize>>) {
        let mut rng = Rng::new(seed);
        let ids: Vec<usize> = (0..n + 1).map(|_| rng.below(vocab)).collect();
        (
            ids[..n].to_vec(),
            ids[1..].iter().map(|&t| Some(t)).collect(),
        )
    }

    /// With B = 0 the adapted model is exactly the base model.
    #[test]
    fn a_fresh_adapter_changes_nothing() {
        let (c, vocab) = (tiny(), 11);
        let p = base(&c, vocab);
        let l = Lora::new(&c, &p, 2, 4.0, &all_kinds(), 5);
        let (x, y) = batch(vocab, 6, 1);
        let plain = forward_with(&p, None, &c, &x, &y, 6);
        let adapted = forward_with(&p, Some(&l), &c, &x, &y, 6);
        assert_eq!(
            plain.tape.value(plain.logits),
            adapted.tape.value(adapted.logits)
        );
    }

    /// Every number of A and B against finite differences.
    #[test]
    fn adapter_gradients_match_finite_differences() {
        let (c, vocab) = (tiny(), 11);
        let p = base(&c, vocab);
        let mut l = Lora::new(&c, &p, 2, 4.0, &all_kinds(), 5);
        let mut rng = Rng::new(9);
        for m in &mut l.b {
            for x in &mut m.data {
                *x = 0.2 * rng.normal(); // B ≠ 0, so A's gradient isn't zero
            }
        }
        let (x, y) = batch(vocab, 6, 2);
        let mut f = forward_with(&p, Some(&l), &c, &x, &y, 6);
        f.tape.backward(f.loss);
        let grads: Vec<Mat<f64>> = f.lora_leaves.iter().map(|&v| f.tape.take_grad(v)).collect();
        let loss_at = |l: &Lora<f64>| {
            let f = forward_with(&p, Some(l), &c, &x, &y, 6);
            f.tape.value(f.loss).data[0]
        };
        let eps = 1e-6;
        let mut worst: f64 = 0.0;
        for (k, g) in grads.iter().enumerate() {
            for i in 0..g.data.len() {
                let mut probe = l.clone();
                let m = if k % 2 == 0 {
                    &mut probe.a[k / 2]
                } else {
                    &mut probe.b[k / 2]
                };
                let orig = m.data[i];
                m.data[i] = orig + eps;
                let up = loss_at(&probe);
                let m = if k % 2 == 0 {
                    &mut probe.a[k / 2]
                } else {
                    &mut probe.b[k / 2]
                };
                m.data[i] = orig - eps;
                let down = loss_at(&probe);
                let numeric = (up - down) / (2.0 * eps);
                worst = worst
                    .max((numeric - g.data[i]).abs() / (numeric.abs() + g.data[i].abs() + 1e-3));
            }
        }
        assert!(worst < 1e-6, "worst relative error {worst}");
    }

    /// The merged model predicts exactly what the base plus adapter does,
    /// in f32 too (the same operations in the same order).
    #[test]
    fn merging_is_exact() {
        let (c, vocab) = (tiny(), 11);
        let p = Params::<f32>::from_tensors(&model::init(&c, vocab));
        let mut l = Lora::new(&c, &p, 3, 6.0, &all_kinds(), 5);
        let mut rng = Rng::new(4);
        for m in &mut l.b {
            for x in &mut m.data {
                *x = (0.1 * rng.normal()) as f32;
            }
        }
        let (x, y) = batch(vocab, 6, 3);
        let adapted = forward_with(&p, Some(&l), &c, &x, &y, 6);
        let merged = l.merge(&p);
        let plain = forward_with(&merged, None, &c, &x, &y, 6);
        assert_eq!(
            adapted.tape.value(adapted.logits),
            plain.tape.value(plain.logits)
        );
    }

    #[test]
    fn round_trip_and_size() {
        let (c, vocab) = (tiny(), 11);
        let p = base(&c, vocab);
        let l = Lora::new(
            &c,
            &p,
            2,
            4.0,
            &["attn.w_q".to_string(), "ffn.w1".to_string()],
            5,
        );
        // 2 blocks × (w_q: 8×2 + 2×8, w1: 8×2 + 2×16)
        assert_eq!(l.size(), 2 * (16 + 16 + 16 + 32));
        let back = Lora::from_params(&l.to_params(), 2, 4.0).unwrap();
        assert_eq!(back.targets, l.targets);
        assert_eq!(back.a, l.a);
        assert!(back.check(&p).is_ok());
        assert!(Lora::from_params(&l.to_params(), 3, 4.0).is_err());
    }
}
