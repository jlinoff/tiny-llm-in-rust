//! The model's forward pass, written on the tape (forward code only; the
//! tape works out the gradients). For a batch of B sequences of T tokens,
//! one row per token:
//!
//!   x      = E[token] + P[position]                  the residual stream
//!   for each block:
//!     h    = x + (attention(LN₁(x)) · W_o + b_o)      Q, K, V = LN₁(x)·W + b
//!     x    = h + (GELU(LN₂(h)·W₁ + b₁)·W₂ + b₂)
//!   logits = LN_f(x) · Eᵀ                            output tied to E
//!   loss   = mean cross-entropy against the next tokens

use crate::config::ModelConfig;
use crate::safetensors::Tensor;
use crate::tape::{Tape, Var};
use crate::tensor::{Float, Mat};

/// The weights as matrices, by name, in the checkpoint's order. Vectors
/// (biases, γ, β) are 1 × n matrices.
#[derive(Clone)]
pub struct Params<T> {
    pub names: Vec<String>,
    pub mats: Vec<Mat<T>>,
}

impl<T: Float> Params<T> {
    pub fn from_tensors(tensors: &[Tensor]) -> Params<T> {
        let mut names = Vec::new();
        let mut mats = Vec::new();
        for t in tensors {
            let (rows, cols) = match t.shape.as_slice() {
                [n] => (1, *n),
                [r, c] => (*r, *c),
                s => panic!("tensor {}: unexpected shape {s:?}", t.name),
            };
            names.push(t.name.clone());
            mats.push(Mat::from_vec(
                rows,
                cols,
                t.data.iter().map(|&x| T::from_f64(x as f64)).collect(),
            ));
        }
        Params { names, mats }
    }

    fn index(&self, name: &str) -> usize {
        self.names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no weight named {name}"))
    }
}

/// One forward pass on a tape: the loss, the logits, and each weight's
/// leaf (in `Params` order, for reading the gradients after `backward`).
#[allow(dead_code)] // leaves: read by the training loop; logits: by generate
pub struct Forward<T: Float> {
    pub tape: Tape<T>,
    pub leaves: Vec<Var>,
    pub logits: Var,
    pub loss: Var,
}

/// `inputs`: B·T token IDs, sequence after sequence; `targets`: the token
/// that should follow each one (None for no loss there).
pub fn forward<T: Float>(
    p: &Params<T>,
    c: &ModelConfig,
    inputs: &[usize],
    targets: &[Option<usize>],
    seq: usize,
) -> Forward<T> {
    assert!(
        seq >= 1 && seq <= c.context,
        "sequence length must be 1 to context"
    );
    assert_eq!(inputs.len() % seq, 0, "inputs must be whole sequences");
    let mut t = Tape::new();
    let leaves: Vec<Var> = p.mats.iter().map(|m| t.leaf(m.clone())).collect();
    let w = |name: &str| leaves[p.index(name)];

    // x = E[token] + P[position]
    let e = w("token_embedding");
    let tok = t.gather(e, inputs);
    let positions: Vec<usize> = (0..inputs.len()).map(|i| i % seq).collect();
    let pos = t.gather(w("position_embedding"), &positions);
    let mut x = t.add(tok, pos);

    for i in 0..c.layers {
        let b = |s: &str| w(&format!("blocks.{i}.{s}"));
        // Attention sublayer.
        let z = t.layer_norm(x, b("ln1.gamma"), b("ln1.beta"));
        let mut proj = |wn: &str, bn: &str| {
            let m = t.matmul(z, b(wn));
            t.add_row(m, b(bn))
        };
        let q = proj("attn.w_q", "attn.b_q");
        let k = proj("attn.w_k", "attn.b_k");
        let v = proj("attn.w_v", "attn.b_v");
        let att = t.attention(q, k, v, seq, c.heads);
        let o = t.matmul(att, b("attn.w_o"));
        let o = t.add_row(o, b("attn.b_o"));
        x = t.add(x, o);
        // FFN sublayer.
        let z = t.layer_norm(x, b("ln2.gamma"), b("ln2.beta"));
        let h = t.matmul(z, b("ffn.w1"));
        let h = t.add_row(h, b("ffn.b1"));
        let h = t.gelu(h);
        let f = t.matmul(h, b("ffn.w2"));
        let f = t.add_row(f, b("ffn.b2"));
        x = t.add(x, f);
    }

    let z = t.layer_norm(x, w("final_ln.gamma"), w("final_ln.beta"));
    let logits = t.matmul_nt(z, e);
    let loss = t.cross_entropy(logits, targets);
    Forward {
        tape: t,
        leaves,
        logits,
        loss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model;
    use crate::rng::Rng;

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

    /// Random weights with larger values than the real initialization, so
    /// every part of the model contributes visibly to the gradients.
    fn random_params(c: &ModelConfig, vocab: usize) -> Params<f64> {
        let mut p = Params::<f64>::from_tensors(&model::init(c, vocab));
        let mut rng = Rng::new(17);
        for m in &mut p.mats {
            for x in &mut m.data {
                *x += 0.3 * rng.normal();
            }
        }
        p
    }

    fn batch(vocab: usize, b: usize, seq: usize, seed: u64) -> (Vec<usize>, Vec<Option<usize>>) {
        let mut rng = Rng::new(seed);
        let ids: Vec<usize> = (0..b * (seq + 1))
            .map(|_| (rng.next_u64() % vocab as u64) as usize)
            .collect();
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for s in 0..b {
            let w = &ids[s * (seq + 1)..(s + 1) * (seq + 1)];
            inputs.extend(&w[..seq]);
            targets.extend(w[1..].iter().map(|&t| Some(t)));
        }
        (inputs, targets)
    }

    /// Every weight of a small two-block model, against finite differences.
    #[test]
    fn whole_model_gradients_match_finite_differences() {
        let (c, vocab, seq) = (tiny(), 11, 5);
        let p = random_params(&c, vocab);
        let (inputs, targets) = batch(vocab, 2, seq, 5);
        let mut f = forward(&p, &c, &inputs, &targets, seq);
        f.tape.backward(f.loss);
        let grads: Vec<Mat<f64>> = f.leaves.iter().map(|&l| f.tape.take_grad(l)).collect();
        let loss_at = |p: &Params<f64>| {
            let f = forward(p, &c, &inputs, &targets, seq);
            f.tape.value(f.loss).data[0]
        };
        let eps = 1e-6;
        let mut worst: f64 = 0.0;
        let mut probe = p.clone();
        let mut checked = 0;
        for (k, g) in grads.iter().enumerate() {
            for i in 0..g.data.len() {
                let orig = probe.mats[k].data[i];
                probe.mats[k].data[i] = orig + eps;
                let up = loss_at(&probe);
                probe.mats[k].data[i] = orig - eps;
                let down = loss_at(&probe);
                probe.mats[k].data[i] = orig;
                let numeric = (up - down) / (2.0 * eps);
                let a = g.data[i];
                worst = worst.max((numeric - a).abs() / (numeric.abs() + a.abs() + 1e-3));
                checked += 1;
            }
        }
        let n: usize = p.mats.iter().map(|m| m.data.len()).sum();
        assert_eq!(checked, n); // every single weight
        assert!(worst < 1e-6, "worst relative error {worst}");
    }

    #[test]
    fn initial_loss_is_about_ln_v() {
        // Small initial weights give near-equal scores for every token, so
        // the loss starts near ln V (a uniform guess).
        let c = ModelConfig {
            d_model: 32,
            layers: 2,
            heads: 4,
            context: 16,
            d_ff: 128,
            seed: 1,
        };
        let vocab = 300;
        let p = Params::<f32>::from_tensors(&model::init(&c, vocab));
        let (inputs, targets) = batch(vocab, 4, 16, 9);
        let f = forward(&p, &c, &inputs, &targets, 16);
        let loss = f.tape.value(f.loss).data[0] as f64;
        assert!((loss - (vocab as f64).ln()).abs() < 0.05, "loss {loss}");
    }

    #[test]
    fn a_later_token_never_changes_earlier_predictions() {
        let (c, vocab, seq) = (tiny(), 11, 6);
        let p = random_params(&c, vocab);
        let (mut inputs, targets) = batch(vocab, 1, seq, 2);
        let before = forward(&p, &c, &inputs, &targets, seq);
        inputs[4] = (inputs[4] + 1) % vocab;
        let after = forward(&p, &c, &inputs, &targets, seq);
        let (lb, la) = (
            before.tape.value(before.logits),
            after.tape.value(after.logits),
        );
        for r in 0..4 {
            assert_eq!(lb.row(r), la.row(r), "position {r} saw a later token");
        }
        assert_ne!(lb.row(4), la.row(4));
    }
}
