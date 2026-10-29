//! A Llama-style open-weight model, for inference (PLAN.md, Order of work,
//! the endpoint): the architecture of SmolLM, Llama, Mistral, Qwen and many
//! others, read from Hugging Face's files (`config.json`,
//! `model.safetensors`). It differs from our GPT-2-style model in four
//! pieces (PLAN.md, "What production models add"):
//!
//!   x = E[token]                                   no position table
//!   for each block:
//!     h = x + attention(RMSNorm(x))                 RoPE on q and k; no biases;
//!                                                   grouped-query attention
//!     x = h + W_down·(SiLU(W_gate·z) ⊙ (W_up·z)),  z = RMSNorm(h)   (SwiGLU)
//!   logits = RMSNorm(x) · Eᵀ                       tied, as ours
//!
//! - **RMSNorm**: x / √(mean(x²) + ε) · w, layer norm without the mean
//!   subtraction and the shift.
//! - **RoPE** (rotary position embedding): position enters by rotating
//!   pairs of query and key components through an angle proportional to
//!   the position, so a query–key dot product depends on their distance.
//!   This follows Hugging Face's layout: component j pairs with j + h/2.
//! - **SwiGLU**: a gated feed-forward network; SiLU(x) = x·sigmoid(x).
//! - **Grouped-query attention**: query heads share key/value heads (9 and
//!   3 in SmolLM-135M: each key/value head serves 3 query heads), which
//!   shrinks the KV cache.
//!
//! Hugging Face stores each matrix as (out × in), PyTorch's convention;
//! they're transposed on loading to our x·W layout.

use crate::generate::affine;
use crate::safetensors;
use crate::tensor::Mat;
use std::path::Path;

/// The settings from `config.json` this implementation uses.
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub context: usize,
    pub eps: f32,
    pub rope_theta: f64,
}

impl LlamaConfig {
    pub fn load(path: &Path) -> Result<LlamaConfig, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("can't read {}: {e}", path.display()))?;
        let j: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let bad = |what: String| format!("{}: {what}", path.display());
        let num = |k: &str| {
            j[k].as_u64()
                .map(|v| v as usize)
                .ok_or(bad(format!("{k} missing or not a whole number")))
        };
        // What this implementation supports, checked rather than assumed.
        if j["model_type"] != "llama" {
            return Err(bad(format!(
                "model_type {} (only llama is supported)",
                j["model_type"]
            )));
        }
        if j["hidden_act"] != "silu" {
            return Err(bad(format!("hidden_act {} (only silu)", j["hidden_act"])));
        }
        for k in ["attention_bias", "mlp_bias"] {
            if j[k] == true {
                return Err(bad(format!("{k} isn't supported")));
            }
        }
        if !j["rope_scaling"].is_null() {
            return Err(bad("rope_scaling isn't supported".into()));
        }
        if j["tie_word_embeddings"] != true {
            return Err(bad(
                "only models whose output layer is tied to the embeddings are supported".into(),
            ));
        }
        let (d, heads) = (num("hidden_size")?, num("num_attention_heads")?);
        let c = LlamaConfig {
            d,
            layers: num("num_hidden_layers")?,
            heads,
            kv_heads: num("num_key_value_heads")?,
            head_dim: j["head_dim"].as_u64().map_or(d / heads, |v| v as usize),
            d_ff: num("intermediate_size")?,
            vocab: num("vocab_size")?,
            context: num("max_position_embeddings")?,
            eps: j["rms_norm_eps"]
                .as_f64()
                .ok_or(bad("rms_norm_eps missing".into()))? as f32,
            rope_theta: j["rope_theta"].as_f64().unwrap_or(10000.0),
        };
        if c.heads % c.kv_heads != 0 || c.head_dim % 2 != 0 {
            return Err(bad(
                "heads must be a multiple of kv_heads, head_dim even".into()
            ));
        }
        Ok(c)
    }

    pub fn parameters(&self) -> usize {
        let (d, kv) = (self.d, self.kv_heads * self.head_dim);
        let q = self.heads * self.head_dim;
        self.vocab * d + self.layers * (2 * d + d * q + 2 * d * kv + q * d + 3 * d * self.d_ff) + d
    }
}

struct Block {
    attn_norm: Mat<f32>,
    q: Mat<f32>,
    k: Mat<f32>,
    v: Mat<f32>,
    o: Mat<f32>,
    ffn_norm: Mat<f32>,
    gate: Mat<f32>,
    up: Mat<f32>,
    down: Mat<f32>,
}

pub struct Llama {
    pub c: LlamaConfig,
    embed: Mat<f32>,
    /// The embedding table transposed (d × V), for the output layer's
    /// product.
    embed_t: Mat<f32>,
    blocks: Vec<Block>,
    norm: Mat<f32>,
    /// RoPE's angle factors θ^(−2j/h), j = 0 … h/2 − 1.
    inv_freq: Vec<f64>,
}

impl Llama {
    /// Load `config.json` and `model.safetensors` from a model's folder.
    pub fn load(dir: &Path) -> Result<Llama, String> {
        let c = LlamaConfig::load(&dir.join("config.json"))?;
        let f = safetensors::read(&dir.join("model.safetensors"))?;
        let mut by_name: std::collections::HashMap<String, safetensors::Tensor> =
            f.tensors.into_iter().map(|t| (t.name.clone(), t)).collect();
        let mut take = |name: &str, shape: &[usize]| -> Result<Mat<f32>, String> {
            let t = by_name
                .remove(name)
                .ok_or(format!("the model has no tensor {name}"))?;
            if t.shape != shape {
                return Err(format!("{name}: shape {:?}, expected {shape:?}", t.shape));
            }
            let (r, cols) = if shape.len() == 1 {
                (1, shape[0])
            } else {
                (shape[0], shape[1])
            };
            Ok(Mat::from_vec(r, cols, t.data))
        };
        let (d, q, kv) = (c.d, c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let embed = take("model.embed_tokens.weight", &[c.vocab, d])?;
        let mut blocks = Vec::new();
        for i in 0..c.layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            // (out × in) → (in × out)
            let mut lin =
                |s: &str, out: usize, inp: usize| take(&p(s), &[out, inp]).map(|m| m.transpose());
            blocks.push(Block {
                q: lin("self_attn.q_proj.weight", q, d)?,
                k: lin("self_attn.k_proj.weight", kv, d)?,
                v: lin("self_attn.v_proj.weight", kv, d)?,
                o: lin("self_attn.o_proj.weight", d, q)?,
                gate: lin("mlp.gate_proj.weight", c.d_ff, d)?,
                up: lin("mlp.up_proj.weight", c.d_ff, d)?,
                down: lin("mlp.down_proj.weight", d, c.d_ff)?,
                attn_norm: take(&p("input_layernorm.weight"), &[d])?,
                ffn_norm: take(&p("post_attention_layernorm.weight"), &[d])?,
            });
        }
        let norm = take("model.norm.weight", &[d])?;
        if let Some(extra) = by_name.keys().next() {
            return Err(format!(
                "unexpected tensor {extra} (and {} more)",
                by_name.len() - 1
            ));
        }
        let h = c.head_dim;
        let inv_freq = (0..h / 2)
            .map(|j| 1.0 / c.rope_theta.powf((2 * j) as f64 / h as f64))
            .collect();
        Ok(Llama {
            embed_t: embed.transpose(),
            embed,
            blocks,
            norm,
            inv_freq,
            c,
        })
    }
}

/// x / √(mean(x²) + ε) · w, for every row.
pub fn rms_norm(x: &Mat<f32>, w: &Mat<f32>, eps: f32) -> Mat<f32> {
    let mut out = Mat::zeros(x.rows, x.cols);
    for r in 0..x.rows {
        let row = x.row(r);
        let ms = row.iter().map(|&v| v * v).sum::<f32>() / row.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for ((o, &v), &g) in out.row_mut(r).iter_mut().zip(row).zip(&w.data) {
            *o = g * (v * inv);
        }
    }
    out
}

/// Rotate one head's vector for position `pos` (RoPE, Hugging Face's
/// layout): component j and j + h/2 turn by the angle pos·θ^(−2j/h).
pub fn rope(v: &mut [f32], pos: usize, inv_freq: &[f64]) {
    let half = v.len() / 2;
    for (j, &f) in inv_freq.iter().enumerate() {
        let angle = pos as f64 * f;
        let (sin, cos) = (angle.sin() as f32, angle.cos() as f32);
        let (a, b) = (v[j], v[j + half]);
        v[j] = a * cos - b * sin;
        v[j + half] = b * cos + a * sin;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn no_bias(cols: usize) -> Mat<f32> {
    Mat::zeros(1, cols)
}

/// Inference with a KV cache, as `generate::Generator` does for our model.
pub struct LlamaGenerator<'a> {
    m: &'a Llama,
    /// Per block: the keys (after RoPE) and values of every position so far.
    keys: Vec<Mat<f32>>,
    values: Vec<Mat<f32>>,
    len: usize,
}

impl<'a> LlamaGenerator<'a> {
    pub fn new(m: &'a Llama) -> LlamaGenerator<'a> {
        let kv = m.c.kv_heads * m.c.head_dim;
        LlamaGenerator {
            keys: (0..m.c.layers)
                .map(|_| Mat::zeros(m.c.context, kv))
                .collect(),
            values: (0..m.c.layers)
                .map(|_| Mat::zeros(m.c.context, kv))
                .collect(),
            len: 0,
            m,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Run new tokens through the model; the logits of every new row
    /// (`every_row`) or of the last only.
    pub fn feed(&mut self, tokens: &[usize], every_row: bool) -> Mat<f32> {
        let (m, c) = (self.m, &self.m.c);
        let (n, start, h) = (tokens.len(), self.len, c.head_dim);
        assert!(
            n >= 1 && start + n <= c.context,
            "within the context of {}",
            c.context
        );
        let group = c.heads / c.kv_heads;
        let scale = 1.0 / (h as f32).sqrt();
        let mut x = Mat::zeros(n, c.d);
        for (r, &t) in tokens.iter().enumerate() {
            x.row_mut(r).copy_from_slice(m.embed.row(t));
        }
        for (i, b) in m.blocks.iter().enumerate() {
            let z = rms_norm(&x, &b.attn_norm, c.eps);
            let mut q = affine(&z, &b.q, &no_bias(b.q.cols));
            let mut k = affine(&z, &b.k, &no_bias(b.k.cols));
            let v = affine(&z, &b.v, &no_bias(b.v.cols));
            for r in 0..n {
                let pos = start + r;
                for hd in 0..c.heads {
                    rope(&mut q.row_mut(r)[hd * h..(hd + 1) * h], pos, &m.inv_freq);
                }
                for hd in 0..c.kv_heads {
                    rope(&mut k.row_mut(r)[hd * h..(hd + 1) * h], pos, &m.inv_freq);
                }
                self.keys[i].row_mut(pos).copy_from_slice(k.row(r));
                self.values[i].row_mut(pos).copy_from_slice(v.row(r));
            }
            // Causal attention, each query head reading its group's
            // key/value head.
            let mut att = Mat::zeros(n, c.heads * h);
            for r in 0..n {
                let t = start + r;
                for hd in 0..c.heads {
                    let kvh = hd / group;
                    let qv = &q.row(r)[hd * h..(hd + 1) * h];
                    let mut w: Vec<f32> = (0..=t)
                        .map(|j| {
                            let kv = &self.keys[i].row(j)[kvh * h..(kvh + 1) * h];
                            qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale
                        })
                        .collect();
                    let max = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut total = 0.0;
                    for s in w.iter_mut() {
                        *s = (*s - max).exp();
                        total += *s;
                    }
                    let out = &mut att.row_mut(r)[hd * h..(hd + 1) * h];
                    for (j, s) in w.iter().enumerate() {
                        let a = s / total;
                        let vv = &self.values[i].row(j)[kvh * h..(kvh + 1) * h];
                        for (o, &val) in out.iter_mut().zip(vv) {
                            *o += a * val;
                        }
                    }
                }
            }
            let o = att.matmul(&b.o);
            add_in(&mut x, &o);
            // SwiGLU feed-forward network.
            let z = rms_norm(&x, &b.ffn_norm, c.eps);
            let g = z.matmul(&b.gate);
            let u = z.matmul(&b.up);
            let gu = g.zip_map(&u, |a, b| silu(a) * b);
            add_in(&mut x, &gu.matmul(&b.down));
        }
        self.len += n;
        let first = if every_row { 0 } else { n - 1 };
        let last = Mat::from_vec(n - first, c.d, x.data[first * c.d..].to_vec());
        rms_norm(&last, &m.norm, c.eps).matmul(&m.embed_t)
    }
}

fn add_in(x: &mut Mat<f32>, y: &Mat<f32>) {
    for (a, &b) in x.data.iter_mut().zip(&y.data) {
        *a += b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_by_hand() {
        let x = Mat::from_vec(1, 4, vec![1.0, -1.0, 3.0, -3.0]); // mean square 5
        let w = Mat::from_vec(1, 4, vec![1.0, 2.0, 1.0, 1.0]);
        let y = rms_norm(&x, &w, 0.0);
        let s = 1.0 / 5f32.sqrt();
        for (a, b) in y.data.iter().zip([s, -2.0 * s, 3.0 * s, -3.0 * s]) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    /// RoPE keeps lengths, and a query–key dot product depends only on the
    /// distance between their positions.
    #[test]
    fn rope_rotates_by_relative_position() {
        let inv: Vec<f64> = (0..4)
            .map(|j| 1.0 / 10000f64.powf(2.0 * j as f64 / 8.0))
            .collect();
        let (q0, k0) = (
            [0.3f32, -1.2, 0.5, 0.9, -0.4, 0.1, 0.7, -0.6],
            [1.0f32, 0.2, -0.3, 0.4, 0.8, -0.9, 0.05, 0.6],
        );
        let dot = |pq: usize, pk: usize| {
            let (mut q, mut k) = (q0, k0);
            rope(&mut q, pq, &inv);
            rope(&mut k, pk, &inv);
            q.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>()
        };
        assert!((dot(7, 3) - dot(14, 10)).abs() < 1e-4); // both 4 apart
        assert!((dot(7, 3) - dot(7, 5)).abs() > 1e-3); // a different distance
        let mut q = q0;
        rope(&mut q, 9, &inv);
        let len = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
        assert!((len(&q) - len(&q0)).abs() < 1e-5);
    }
}
