//! Step 6, generate: inference. The trained model writes text one token at
//! a time, each new token appended to the input for the next.
//!
//! Two phases (PLAN.md, step 6):
//!   prefill  the prompt's tokens go through the model together, as a
//!            matrix of rows, like training without the backward pass
//!   decode   one new token at a time: a single row through the model
//!
//! The KV cache. Attention at position t needs the keys and values of every
//! position 0…t. Those never change once computed (a later token never
//! changes an earlier one's), so each layer keeps them in a cache, and a new
//! token costs one row of work instead of rerunning the whole sequence.
//!
//! This is a separate path from training's tape (no gradients, no stored
//! intermediates), but it does exactly the same arithmetic in the same
//! order, so its predictions are bit-for-bit equal to the training model's
//! (the test `cache_gives_exactly_the_training_forward_pass` checks this).

use crate::config::ModelConfig;
use crate::gpt::Params;
use crate::rng::Rng;
use crate::tape::{gelu, layer_norm_row};
use crate::tensor::{Float, Mat, PARALLEL_MIN_WORK};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The config file's [generate] section. Every key is optional.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GenerateConfig {
    /// Stop after this many new tokens (or earlier, at end of text).
    #[serde(default = "d_max_tokens")]
    pub max_tokens: usize,
    /// Divides the scores before the softmax: below 1 sharpens the
    /// distribution toward the likeliest tokens, above 1 flattens it; 0
    /// always picks the single likeliest token (greedy).
    #[serde(default = "d_temperature")]
    pub temperature: f64,
    /// Sample only from the k likeliest tokens; 0 means all of them.
    #[serde(default = "d_top_k")]
    pub top_k: usize,
    /// Seed for the sampling; the same seed and prompt give the same text.
    #[serde(default = "d_seed")]
    pub seed: u64,
}

fn d_max_tokens() -> usize {
    200
}
fn d_temperature() -> f64 {
    0.8
}
fn d_top_k() -> usize {
    40
}
fn d_seed() -> u64 {
    1
}

impl Default for GenerateConfig {
    fn default() -> Self {
        GenerateConfig {
            max_tokens: d_max_tokens(),
            temperature: d_temperature(),
            top_k: d_top_k(),
            seed: d_seed(),
        }
    }
}

#[derive(Deserialize)]
struct File {
    #[serde(default)]
    generate: GenerateConfig,
}

impl GenerateConfig {
    /// The [generate] section of the config file (all defaults if absent).
    pub fn load(path: &Path) -> Result<GenerateConfig, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("can't read config {}: {e}", path.display()))?;
        let f: File =
            toml::from_str(&text).map_err(|e| format!("config {}: {e}", path.display()))?;
        f.generate.check()?;
        Ok(f.generate)
    }

    pub fn check(&self) -> Result<(), String> {
        if !(self.temperature >= 0.0 && self.temperature.is_finite()) {
            return Err("[generate] temperature must be 0 or more".into());
        }
        if self.max_tokens == 0 {
            return Err("[generate] max_tokens must be at least 1".into());
        }
        Ok(())
    }
}

/// Where one block's weights are in `Params`.
struct BlockIdx {
    ln1: (usize, usize),
    q: (usize, usize),
    k: (usize, usize),
    v: (usize, usize),
    o: (usize, usize),
    ln2: (usize, usize),
    w1: (usize, usize),
    w2: (usize, usize),
}

/// The model plus its KV cache.
pub struct Generator<'a, T: Float> {
    p: &'a Params<T>,
    c: &'a ModelConfig,
    emb: usize,
    pos: usize,
    final_ln: (usize, usize),
    blocks: Vec<BlockIdx>,
    /// Per block: the keys and values of positions 0…len−1, context × d.
    keys: Vec<Mat<T>>,
    values: Vec<Mat<T>>,
    len: usize,
}

impl<'a, T: Float> Generator<'a, T> {
    pub fn new(p: &'a Params<T>, c: &'a ModelConfig) -> Self {
        let at = |name: &str| {
            p.names
                .iter()
                .position(|n| n == name)
                .unwrap_or_else(|| panic!("no weight named {name}"))
        };
        let pair = |i: usize, w: &str, b: &str| {
            (
                at(&format!("blocks.{i}.{w}")),
                at(&format!("blocks.{i}.{b}")),
            )
        };
        let blocks = (0..c.layers)
            .map(|i| BlockIdx {
                ln1: pair(i, "ln1.gamma", "ln1.beta"),
                q: pair(i, "attn.w_q", "attn.b_q"),
                k: pair(i, "attn.w_k", "attn.b_k"),
                v: pair(i, "attn.w_v", "attn.b_v"),
                o: pair(i, "attn.w_o", "attn.b_o"),
                ln2: pair(i, "ln2.gamma", "ln2.beta"),
                w1: pair(i, "ffn.w1", "ffn.b1"),
                w2: pair(i, "ffn.w2", "ffn.b2"),
            })
            .collect();
        let cache = || {
            (0..c.layers)
                .map(|_| Mat::zeros(c.context, c.d_model))
                .collect()
        };
        Generator {
            p,
            c,
            emb: at("token_embedding"),
            pos: at("position_embedding"),
            final_ln: (at("final_ln.gamma"), at("final_ln.beta")),
            blocks,
            keys: cache(),
            values: cache(),
            len: 0,
        }
    }

    /// Positions in the cache so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Empty the cache (to start again from new tokens).
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Run `tokens` through the model at the next positions, adding their
    /// keys and values to the cache. Returns the scores (logits) for the
    /// token after each of them if `every_row`, else only after the last:
    /// one row of V per token.
    pub fn feed(&mut self, tokens: &[usize], every_row: bool) -> Mat<T> {
        let (n, d) = (tokens.len(), self.c.d_model);
        assert!(n >= 1, "feed at least one token");
        assert!(
            self.len + n <= self.c.context,
            "the cache holds {} positions",
            self.c.context
        );
        let m = &self.p.mats;
        let start = self.len;

        // x = E[token] + P[position], one row per new token.
        let mut x = Mat::zeros(n, d);
        for (r, &tok) in tokens.iter().enumerate() {
            let (e, p) = (m[self.emb].row(tok), m[self.pos].row(start + r));
            for (xv, (&a, &b)) in x.row_mut(r).iter_mut().zip(e.iter().zip(p)) {
                *xv = a + b;
            }
        }

        for (i, b) in self.blocks.iter().enumerate() {
            // Attention sublayer.
            let z = norm(&x, &m[b.ln1.0], &m[b.ln1.1]);
            let q = affine(&z, &m[b.q.0], &m[b.q.1]);
            let k = affine(&z, &m[b.k.0], &m[b.k.1]);
            let v = affine(&z, &m[b.v.0], &m[b.v.1]);
            for r in 0..n {
                self.keys[i].row_mut(start + r).copy_from_slice(k.row(r));
                self.values[i].row_mut(start + r).copy_from_slice(v.row(r));
            }
            let att = attend(&q, &self.keys[i], &self.values[i], start, self.c.heads);
            add_in(&mut x, &affine(&att, &m[b.o.0], &m[b.o.1]));
            // FFN sublayer.
            let z = norm(&x, &m[b.ln2.0], &m[b.ln2.1]);
            let h = affine(&z, &m[b.w1.0], &m[b.w1.1]).map(gelu);
            add_in(&mut x, &affine(&h, &m[b.w2.0], &m[b.w2.1]));
        }
        self.len += n;

        // Scores against every token's embedding (the tied output layer),
        // for the rows wanted.
        let first = if every_row { 0 } else { n - 1 };
        let rows = n - first;
        let last = Mat::from_vec(rows, d, x.data[first * d..].to_vec());
        let z = norm(&last, &m[self.final_ln.0], &m[self.final_ln.1]);
        let e = &m[self.emb];
        let vocab = e.rows;
        let mut logits = Mat::zeros(rows, vocab);
        for r in 0..rows {
            let zr = z.row(r);
            let score = |(t, out): (usize, &mut T)| {
                let mut s = T::zero();
                for (&a, &b) in zr.iter().zip(e.row(t)) {
                    s += a * b;
                }
                *out = s;
            };
            if vocab * d < PARALLEL_MIN_WORK {
                logits.row_mut(r).iter_mut().enumerate().for_each(score);
            } else {
                logits.row_mut(r).par_iter_mut().enumerate().for_each(score);
            }
        }
        logits
    }
}

/// Layer norm of every row.
fn norm<T: Float>(x: &Mat<T>, g: &Mat<T>, b: &Mat<T>) -> Mat<T> {
    let mut xhat = vec![T::zero(); x.cols];
    let mut out = Mat::zeros(x.rows, x.cols);
    for r in 0..x.rows {
        layer_norm_row(x.row(r), &g.data, &b.data, &mut xhat, out.row_mut(r));
    }
    out
}

/// x·W + b.
fn affine<T: Float>(x: &Mat<T>, w: &Mat<T>, b: &Mat<T>) -> Mat<T> {
    let mut y = x.matmul(w);
    for r in 0..y.rows {
        for (v, &c) in y.row_mut(r).iter_mut().zip(&b.data) {
            *v += c;
        }
    }
    y
}

/// x += y (the residual connection).
fn add_in<T: Float>(x: &mut Mat<T>, y: &Mat<T>) {
    for (a, &b) in x.data.iter_mut().zip(&y.data) {
        *a += b;
    }
}

/// Causal multi-head attention for the new rows `q`, at positions start,
/// start + 1, …, against the cached keys and values of positions 0 up to
/// each row's own. One unit of work per (row, head), in parallel.
fn attend<T: Float>(
    q: &Mat<T>,
    keys: &Mat<T>,
    values: &Mat<T>,
    start: usize,
    heads: usize,
) -> Mat<T> {
    let d = q.cols;
    let hs = d / heads;
    let scale = T::one() / T::from_f64(hs as f64).sqrt();
    let unit = |u: usize| {
        let (r, h) = (u / heads, u % heads);
        let t = start + r; // this row's position: it sees 0…t
        let qv = &q.row(r)[h * hs..(h + 1) * hs];
        let mut w = Vec::with_capacity(t + 1);
        let mut max = T::from_f64(f64::NEG_INFINITY);
        for j in 0..=t {
            let kv = &keys.row(j)[h * hs..(h + 1) * hs];
            let mut s = T::zero();
            for c in 0..hs {
                s += qv[c] * kv[c];
            }
            let s = s * scale;
            w.push(s);
            max = max.max(s);
        }
        let mut total = T::zero();
        for s in w.iter_mut() {
            *s = (*s - max).exp();
            total += *s;
        }
        let mut out = vec![T::zero(); hs];
        for (j, s) in w.iter().enumerate() {
            let a = *s / total;
            let vv = &values.row(j)[h * hs..(h + 1) * hs];
            for (o, &b) in out.iter_mut().zip(vv) {
                *o += a * b;
            }
        }
        out
    };
    // Parallel only when there's enough work (see PARALLEL_MIN_WORK).
    let work = q.rows * (start + q.rows) * d * 2;
    let parts: Vec<Vec<T>> = if work < PARALLEL_MIN_WORK {
        (0..q.rows * heads).map(unit).collect()
    } else {
        (0..q.rows * heads).into_par_iter().map(unit).collect()
    };
    let mut att = Mat::zeros(q.rows, d);
    for (u, part) in parts.into_iter().enumerate() {
        let (r, h) = (u / heads, u % heads);
        att.row_mut(r)[h * hs..(h + 1) * hs].copy_from_slice(&part);
    }
    att
}

/// Choose the next token from the scores. Temperature 0: the likeliest
/// (lowest ID on a tie). Otherwise keep the `top_k` likeliest (0: all),
/// turn their scores into probabilities with softmax(score / temperature),
/// and draw one.
pub fn sample(logits: &[f64], temperature: f64, top_k: usize, rng: &mut Rng) -> usize {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    // Highest score first; ties by ID, so the order never depends on the sort.
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    if temperature == 0.0 {
        return order[0];
    }
    if top_k > 0 && top_k < order.len() {
        order.truncate(top_k);
    }
    let top = logits[order[0]];
    let weights: Vec<f64> = order
        .iter()
        .map(|&t| ((logits[t] - top) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let mut r = rng.unit() * total;
    for (&t, &w) in order.iter().zip(&weights) {
        if r < w {
            return t;
        }
        r -= w;
    }
    *order.last().unwrap() // only reached through rounding
}

/// Turns a stream of token bytes into text. A token can end partway
/// through a UTF-8 character, so bytes wait here until they complete one.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Add bytes; returns the text completed so far.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..good]).unwrap());
                    match e.error_len() {
                        // Incomplete character at the end: wait for more.
                        None => {
                            self.pending.drain(..good);
                            return out;
                        }
                        // Bytes that can never be valid: show U+FFFD.
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..good + bad);
                        }
                    }
                }
            }
        }
    }

    /// Whatever is left at the end (an unfinished character).
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// How a generation went, for the speed report.
pub struct Report {
    pub generated: Vec<usize>,
    pub stopped_at_eos: bool,
    pub prompt_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    /// Times the cache filled up and was rebuilt from the recent tokens.
    pub rebuilds: usize,
}

/// Generate from `prompt` (token IDs, at least one). `emit` gets each new
/// token as it's chosen. Stops at `eos` (not emitted) or after max_tokens.
///
/// Past the context length: the model has no positions beyond T − 1, so
/// when the cache is full it's rebuilt from the most recent 3/4 of T
/// tokens (a sliding window), and generation continues from there. The
/// keys and values can't simply be shifted, because each was computed with
/// its old position's embedding.
pub fn generate<T: Float>(
    g: &mut Generator<T>,
    cfg: &GenerateConfig,
    prompt: &[usize],
    eos: Option<usize>,
    mut emit: impl FnMut(usize),
) -> Report {
    assert!(!prompt.is_empty(), "the prompt needs at least one token");
    let context = g.c.context;
    let keep = (context - context / 4).max(1);
    let mut rng = Rng::new(cfg.seed);
    // A prompt longer than the context: only its last T tokens are seen.
    let mut history: Vec<usize> = prompt.to_vec();
    let window = &history[history.len().saturating_sub(context)..];
    g.reset();
    let t0 = std::time::Instant::now();
    let mut logits = g.feed(window, false);
    let prefill_secs = t0.elapsed().as_secs_f64();

    let t1 = std::time::Instant::now();
    let mut generated = Vec::new();
    let mut stopped_at_eos = false;
    let mut rebuilds = 0;
    while generated.len() < cfg.max_tokens {
        let scores: Vec<f64> = logits.row(0).iter().map(|v| v.to_f64()).collect();
        let next = sample(&scores, cfg.temperature, cfg.top_k, &mut rng);
        if Some(next) == eos {
            stopped_at_eos = true;
            break;
        }
        generated.push(next);
        history.push(next);
        emit(next);
        if generated.len() == cfg.max_tokens {
            break;
        }
        if g.len() == context {
            g.reset();
            rebuilds += 1;
            logits = g.feed(&history[history.len() - keep..], false);
        } else {
            logits = g.feed(&[next], false);
        }
    }
    Report {
        generated,
        stopped_at_eos,
        prompt_tokens: prompt.len(),
        prefill_secs,
        decode_secs: t1.elapsed().as_secs_f64(),
        rebuilds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::forward;
    use crate::model;

    fn tiny() -> ModelConfig {
        ModelConfig {
            d_model: 16,
            layers: 2,
            heads: 4,
            context: 12,
            d_ff: 32,
            seed: 3,
        }
    }

    fn random_params<T: Float>(c: &ModelConfig, vocab: usize) -> Params<T> {
        let mut p = Params::<T>::from_tensors(&model::init(c, vocab));
        let mut rng = Rng::new(17);
        for m in &mut p.mats {
            for x in &mut m.data {
                *x += T::from_f64(0.3 * rng.normal());
            }
        }
        p
    }

    /// The key test: feeding a sequence through the cache (a prefill of 3
    /// tokens, then one at a time) gives, position by position, exactly the
    /// scores of the training model's forward pass over the whole sequence.
    fn cache_matches<T: Float>() {
        let (c, vocab) = (tiny(), 23);
        let p = random_params::<T>(&c, vocab);
        let mut rng = Rng::new(5);
        let tokens: Vec<usize> = (0..c.context).map(|_| rng.below(vocab)).collect();
        let f = forward(&p, &c, &tokens, &vec![None; c.context], c.context);
        let want = f.tape.value(f.logits);

        let mut g = Generator::new(&p, &c);
        let mut got = g.feed(&tokens[..3], true).data;
        for &t in &tokens[3..] {
            got.extend(g.feed(&[t], true).data);
        }
        assert_eq!(got, want.data, "cache and forward pass differ");

        // Asking only for the last row gives that same row.
        g.reset();
        let last = g.feed(&tokens, false);
        assert_eq!(last.data, want.row(c.context - 1));
    }

    #[test]
    fn cache_gives_exactly_the_training_forward_pass() {
        cache_matches::<f64>();
        cache_matches::<f32>();
    }

    #[test]
    fn greedy_and_top_1_pick_the_likeliest() {
        let logits = [0.1, 2.0, -1.0, 2.0, 0.5];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&logits, 0.0, 0, &mut rng), 1); // tie: lower ID
        for _ in 0..100 {
            assert_eq!(sample(&logits, 1.0, 1, &mut rng), 1);
        }
    }

    #[test]
    fn sampling_follows_the_softmax() {
        let logits = [1.0, 0.0, -1.0, 0.5];
        let probs: Vec<f64> = {
            let e: Vec<f64> = logits.iter().map(|l: &f64| l.exp()).collect();
            let s: f64 = e.iter().sum();
            e.iter().map(|x| x / s).collect()
        };
        let mut rng = Rng::new(9);
        let n = 200_000;
        let mut counts = [0usize; 4];
        for _ in 0..n {
            counts[sample(&logits, 1.0, 0, &mut rng)] += 1;
        }
        for i in 0..4 {
            let f = counts[i] as f64 / n as f64;
            // Within 5 standard deviations of the expected frequency.
            let sd = (probs[i] * (1.0 - probs[i]) / n as f64).sqrt();
            assert!(
                (f - probs[i]).abs() < 5.0 * sd,
                "token {i}: {f} vs {}",
                probs[i]
            );
        }
        // Top-2 never picks the other two.
        for _ in 0..1000 {
            let t = sample(&logits, 1.0, 2, &mut rng);
            assert!(t == 0 || t == 3);
        }
    }

    #[test]
    fn generation_is_reproducible_and_runs_past_the_context() {
        let (c, vocab) = (tiny(), 23);
        let p = random_params::<f32>(&c, vocab);
        let cfg = GenerateConfig {
            max_tokens: 40, // more than three context lengths of 12
            temperature: 1.0,
            top_k: 0,
            seed: 4,
        };
        let mut g = Generator::new(&p, &c);
        let a = generate(&mut g, &cfg, &[1, 2], None, |_| {});
        let b = generate(&mut g, &cfg, &[1, 2], None, |_| {});
        assert_eq!(a.generated.len(), 40);
        assert_eq!(a.generated, b.generated);
        assert!(a.rebuilds >= 3, "{} rebuilds", a.rebuilds);
        let other = generate(
            &mut g,
            &GenerateConfig {
                seed: 5,
                ..cfg.clone()
            },
            &[1, 2],
            None,
            |_| {},
        );
        assert_ne!(a.generated, other.generated);
    }

    #[test]
    fn stops_at_end_of_text() {
        let (c, vocab) = (tiny(), 23);
        let p = random_params::<f32>(&c, vocab);
        let mut g = Generator::new(&p, &c);
        let cfg = GenerateConfig {
            temperature: 0.0,
            ..GenerateConfig::default()
        };
        // Greedy: whatever the first choice is, call it end of text.
        let first = generate(
            &mut g,
            &GenerateConfig {
                max_tokens: 1,
                ..cfg.clone()
            },
            &[3],
            None,
            |_| {},
        );
        let r = generate(&mut g, &cfg, &[3], Some(first.generated[0]), |_| {});
        assert!(r.stopped_at_eos);
        assert!(r.generated.is_empty());
    }

    #[test]
    fn utf8_waits_for_whole_characters() {
        let mut s = Utf8Stream::default();
        let e_acute = "é".as_bytes(); // two bytes
        assert_eq!(s.push(b"caf"), "caf");
        assert_eq!(s.push(&e_acute[..1]), "");
        assert_eq!(s.push(&e_acute[1..]), "é");
        assert_eq!(s.push(&[0xFF, b'x']), "\u{FFFD}x");
        assert_eq!(s.push(&e_acute[..1]), "");
        assert_eq!(s.finish(), "\u{FFFD}");
    }

    #[test]
    fn config_section_is_optional_and_checked() {
        let f: File = toml::from_str("[model]\nd_model = 1\n").unwrap();
        assert_eq!(f.generate, GenerateConfig::default());
        let f: File = toml::from_str("[generate]\ntemperature = 0.5\ntop_k = 0\n").unwrap();
        assert_eq!(f.generate.temperature, 0.5);
        assert_eq!(f.generate.max_tokens, 200);
        assert!(toml::from_str::<File>("[generate]\ntemprature = 0.5\n").is_err());
        assert!(GenerateConfig {
            temperature: -1.0,
            ..GenerateConfig::default()
        }
        .check()
        .is_err());
    }
}
