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
//! Optionally (`lora_embeddings`), the adapter also trains a change to some
//! rows of the token embedding E, which LoRA otherwise leaves frozen:
//!
//!   E'[i] = E[i] + D[k]       for the k-th chosen row i; D starts at zero
//!
//! "markers" chooses the special tokens' rows (the chat markers, which
//! pretraining never taught, and end of text), "all" every row. E is also
//! the output layer (logits = LN_f(x)·Eᵀ), so a changed row changes both
//! how the token is read and how likely it is to be predicted. This is not
//! low-rank: a plain, full change to the chosen rows.
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

/// The name of the embedding change D in the adapter's tensors.
pub const EMBEDDING_DELTA: &str = "lora.token_embedding.delta";

/// The choices for `lora_embeddings`.
pub const EMBEDDING_MODES: [&str; 3] = ["none", "markers", "all"];

/// The adapter: an A and a B for each adapted matrix of each block, and
/// optionally a change D to some rows of the token embedding.
#[derive(Clone)]
pub struct Lora<T> {
    pub rank: usize,
    pub alpha: f64,
    /// Full weight names, such as "blocks.0.attn.w_q", in order.
    pub targets: Vec<String>,
    pub a: Vec<Mat<T>>,
    pub b: Vec<Mat<T>>,
    /// The token embedding's rows that D changes, in D's row order (empty:
    /// no D).
    pub emb_rows: Vec<usize>,
    /// D: emb_rows.len() × d_model, or None.
    pub emb_delta: Option<Mat<T>>,
}

impl<T: Float> Lora<T> {
    /// A fresh adapter for `kinds` (from TARGETS) in every block: A random
    /// with standard deviation 1/√d_in, B zero; and, if `emb_rows` isn't
    /// empty, D zero for those rows of the token embedding.
    pub fn new(
        c: &ModelConfig,
        base: &Params<T>,
        rank: usize,
        alpha: f64,
        kinds: &[String],
        emb_rows: &[usize],
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
        let emb_delta = (!emb_rows.is_empty()).then(|| {
            Mat::zeros(
                emb_rows.len(),
                base.mats[base.position("token_embedding")].cols,
            )
        });
        Lora {
            rank,
            alpha,
            targets,
            a,
            b,
            emb_rows: emb_rows.to_vec(),
            emb_delta,
        }
    }

    /// The scale α/r applied to A·B.
    pub fn scale(&self) -> T {
        T::from_f64(self.alpha / self.rank as f64)
    }

    /// The adapter as named matrices ("lora.blocks.0.attn.w_q.a", "….b", …,
    /// then D if there is one), the form the training loop and the
    /// checkpoint use.
    pub fn to_params(&self) -> Params<T> {
        let mut names = Vec::new();
        let mut mats = Vec::new();
        for (k, t) in self.targets.iter().enumerate() {
            names.push(format!("lora.{t}.a"));
            mats.push(self.a[k].clone());
            names.push(format!("lora.{t}.b"));
            mats.push(self.b[k].clone());
        }
        if let Some(d) = &self.emb_delta {
            names.push(EMBEDDING_DELTA.to_string());
            mats.push(d.clone());
        }
        Params { names, mats }
    }

    /// Back from named matrices (as written by `to_params`). `emb_rows`:
    /// the embedding rows D changes, recorded in the adapter's metadata
    /// (empty if it has no D).
    pub fn from_params(
        p: &Params<T>,
        rank: usize,
        alpha: f64,
        emb_rows: &[usize],
    ) -> Result<Lora<T>, String> {
        let (mut names, mut mats) = (p.names.as_slice(), p.mats.as_slice());
        let mut emb_delta = None;
        if names.last().map(String::as_str) == Some(EMBEDDING_DELTA) {
            let d = mats.last().expect("as many matrices as names");
            if d.rows != emb_rows.len() {
                return Err(format!(
                    "the adapter's embedding change has {} rows, its metadata lists {}",
                    d.rows,
                    emb_rows.len()
                ));
            }
            emb_delta = Some(d.clone());
            names = &names[..names.len() - 1];
            mats = &mats[..mats.len() - 1];
        } else if !emb_rows.is_empty() {
            return Err(
                "the adapter's metadata lists embedding rows but it has no embedding change".into(),
            );
        }
        if names.len() % 2 != 0 {
            return Err("an adapter has an A and a B for each matrix".into());
        }
        let (mut targets, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
        for pair in names.chunks(2).zip(mats.chunks(2)) {
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
            emb_rows: emb_rows.to_vec(),
            emb_delta,
        })
    }

    /// D spread over the whole embedding table (V × d, zero in the rows it
    /// doesn't change), the way both the forward pass and `merge` compute
    /// it: D itself when it covers every row in order, otherwise S·D with S
    /// the V × k one-hot matrix putting D's k-th row at row emb_rows[k]
    /// (adding zeros, so exact). Returns S, or None when D is used as it is.
    pub fn selector(&self, vocab: usize) -> Option<Mat<T>> {
        if self.emb_rows.len() == vocab && self.emb_rows.iter().enumerate().all(|(k, &i)| k == i) {
            return None;
        }
        let k = self.emb_rows.len();
        let mut s = Mat::zeros(vocab, k);
        for (j, &i) in self.emb_rows.iter().enumerate() {
            s.data[i * k + j] = T::one();
        }
        Some(s)
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
        if let Some(d) = &self.emb_delta {
            let e = base.position("token_embedding");
            let full = match self.selector(base.mats[e].rows) {
                None => d.clone(),
                Some(sel) => sel.matmul(d),
            };
            out.mats[e].add_assign(&full);
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
        if let Some(d) = &self.emb_delta {
            let e = &base.mats[base.position("token_embedding")];
            let mut seen = vec![false; e.rows];
            for &i in &self.emb_rows {
                if i >= e.rows || std::mem::replace(&mut seen[i], true) {
                    return Err(format!(
                        "embedding row {i} is out of range or listed twice (V = {})",
                        e.rows
                    ));
                }
            }
            if d.cols != e.cols {
                return Err(format!(
                    "the embedding change is {} wide, the model's embedding {}",
                    d.cols, e.cols
                ));
            }
        }
        Ok(())
    }

    /// Numbers in the adapter.
    pub fn size(&self) -> usize {
        self.a
            .iter()
            .chain(&self.b)
            .chain(&self.emb_delta)
            .map(|m| m.data.len())
            .sum()
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

    /// With B = 0 and D = 0 the adapted model is exactly the base model.
    #[test]
    fn a_fresh_adapter_changes_nothing() {
        let (c, vocab) = (tiny(), 11);
        let p = base(&c, vocab);
        let l = Lora::new(&c, &p, 2, 4.0, &all_kinds(), &[10, 2], 5);
        let (x, y) = batch(vocab, 6, 1);
        let plain = forward_with(&p, None, &c, &x, &y, 6);
        let adapted = forward_with(&p, Some(&l), &c, &x, &y, 6);
        assert_eq!(
            plain.tape.value(plain.logits),
            adapted.tape.value(adapted.logits)
        );
    }

    /// Every number of A, B and D against finite differences. The input
    /// includes token 10, whose row D changes, so D is used both as input
    /// and in the output layer.
    #[test]
    fn adapter_gradients_match_finite_differences() {
        let (c, vocab) = (tiny(), 11);
        let p = base(&c, vocab);
        let mut l = Lora::new(&c, &p, 2, 4.0, &all_kinds(), &[10, 2], 5);
        let mut rng = Rng::new(9);
        for m in l.b.iter_mut().chain(&mut l.emb_delta) {
            for x in &mut m.data {
                *x = 0.2 * rng.normal(); // B ≠ 0, so A's gradient isn't zero
            }
        }
        let (mut x, y) = batch(vocab, 6, 2);
        x[1] = 10;
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
            // The k-th trained matrix, in to_params order: A, B, …, then D.
            fn pick(l: &mut Lora<f64>, k: usize) -> &mut Mat<f64> {
                if k == 2 * l.a.len() {
                    l.emb_delta.as_mut().unwrap()
                } else if k % 2 == 0 {
                    &mut l.a[k / 2]
                } else {
                    &mut l.b[k / 2]
                }
            }
            for i in 0..g.data.len() {
                let mut probe = l.clone();
                let orig = pick(&mut probe, k).data[i];
                pick(&mut probe, k).data[i] = orig + eps;
                let up = loss_at(&probe);
                pick(&mut probe, k).data[i] = orig - eps;
                let down = loss_at(&probe);
                let numeric = (up - down) / (2.0 * eps);
                worst = worst
                    .max((numeric - g.data[i]).abs() / (numeric.abs() + g.data[i].abs() + 1e-3));
            }
        }
        assert_eq!(grads.len(), 2 * l.a.len() + 1); // D's gradient was checked too
        assert!(worst < 1e-6, "worst relative error {worst}");
    }

    /// The merged model predicts exactly what the base plus adapter does,
    /// in f32 too (the same operations in the same order): with no
    /// embedding change, with some rows, and with every row.
    #[test]
    fn merging_is_exact() {
        let (c, vocab) = (tiny(), 11);
        let p = Params::<f32>::from_tensors(&model::init(&c, vocab));
        let all: Vec<usize> = (0..vocab).collect();
        for rows in [&[][..], &[7, 3][..], &all[..]] {
            let mut l = Lora::new(&c, &p, 3, 6.0, &all_kinds(), rows, 5);
            let mut rng = Rng::new(4);
            for m in l.b.iter_mut().chain(&mut l.emb_delta) {
                for x in &mut m.data {
                    *x = (0.1 * rng.normal()) as f32;
                }
            }
            let (mut x, y) = batch(vocab, 6, 3);
            x[0] = 7;
            let adapted = forward_with(&p, Some(&l), &c, &x, &y, 6);
            let merged = l.merge(&p);
            let plain = forward_with(&merged, None, &c, &x, &y, 6);
            assert_eq!(
                adapted.tape.value(adapted.logits),
                plain.tape.value(plain.logits)
            );
            // Only the chosen rows of the embedding changed.
            let e = p.position("token_embedding");
            for r in 0..vocab {
                assert_eq!(
                    merged.mats[e].row(r) == p.mats[e].row(r),
                    !rows.contains(&r),
                    "row {r}"
                );
            }
        }
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
            &[4, 9, 0],
            5,
        );
        // 2 blocks × (w_q: 8×2 + 2×8, w1: 8×2 + 2×16), and D: 3 rows × 8
        assert_eq!(l.size(), 2 * (16 + 16 + 16 + 32) + 3 * 8);
        let back = Lora::from_params(&l.to_params(), 2, 4.0, &[4, 9, 0]).unwrap();
        assert_eq!(back.targets, l.targets);
        assert_eq!(back.a, l.a);
        assert_eq!(back.emb_delta, l.emb_delta);
        assert!(back.check(&p).is_ok());
        assert!(Lora::from_params(&l.to_params(), 3, 4.0, &[4, 9, 0]).is_err());
        assert!(Lora::from_params(&l.to_params(), 2, 4.0, &[4, 9]).is_err()); // rows don't match D
        let mut bad = back.clone();
        bad.emb_rows = vec![4, 9, 11]; // row 11 doesn't exist (V = 11)
        assert!(bad.check(&p).is_err());
        bad.emb_rows = vec![4, 9, 4]; // listed twice
        assert!(bad.check(&p).is_err());
    }
}
