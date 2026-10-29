//! Looking inside a trained model (PLAN.md, "Looking inside the model").
//!
//! Three views, all from one plain forward pass that keeps what training
//! and generation throw away:
//!
//! - **Head scores.** Each head's previous-token score (how much a token
//!   attends to the one just before it) and induction score, on the
//!   standard test: random tokens, then the same tokens again. In the
//!   second copy an induction head attends to the token that followed the
//!   current token's earlier occurrence, which is the token to predict.
//!   The test also gives the loss on each copy: about chance on the first,
//!   far lower on the second if the model copies from its context.
//! - **One head's pattern** on real text: where each position looks most.
//! - **The logit lens.** The residual stream after each block, read through
//!   the final layer norm and the output layer as if the model stopped
//!   there: how the prediction forms, block by block.
//!
//! The forward pass is the same arithmetic as generation's (a test checks
//! its predictions against training's), one sequence at a time, with every
//! head's attention weights kept.

use crate::config::ModelConfig;
use crate::generate::{affine, norm};
use crate::gpt::Params;
use crate::rng::Rng;
use crate::tape::gelu;
use crate::tensor::{Float, Mat};

/// What one forward pass leaves behind.
pub struct Trace<T> {
    /// The residual stream: after the embeddings, then after each block
    /// (layers + 1 matrices, one row per token).
    pub residual: Vec<Mat<T>>,
    /// attention[block][head]: n × n; row t holds how position t weighs
    /// positions 0…t (zero beyond t, the causal mask).
    pub attention: Vec<Vec<Mat<T>>>,
}

/// The forward pass over one sequence (at most the context length).
pub fn trace<T: Float>(p: &Params<T>, c: &ModelConfig, tokens: &[usize]) -> Trace<T> {
    trace_ablated(p, c, tokens, None)
}

/// The forward pass with one head switched off ("ablated"): its output is
/// zero, as if the head weren't there; everything else is unchanged.
pub fn trace_ablated<T: Float>(
    p: &Params<T>,
    c: &ModelConfig,
    tokens: &[usize],
    ablate: Option<(usize, usize)>,
) -> Trace<T> {
    let (n, d, heads) = (tokens.len(), c.d_model, c.heads);
    assert!(n >= 1 && n <= c.context, "1 to context tokens");
    let w = |name: &str| &p.mats[p.position(name)];
    let (e, pos) = (w("token_embedding"), w("position_embedding"));
    let mut x = Mat::zeros(n, d);
    for (r, &tok) in tokens.iter().enumerate() {
        for (xv, (&a, &b)) in x
            .row_mut(r)
            .iter_mut()
            .zip(e.row(tok).iter().zip(pos.row(r)))
        {
            *xv = a + b;
        }
    }
    let mut residual = vec![x.clone()];
    let mut attention = Vec::new();
    let hs = d / heads;
    let scale = T::one() / T::from_f64(hs as f64).sqrt();
    for i in 0..c.layers {
        let b = |s: &str| w(&format!("blocks.{i}.{s}"));
        let z = norm(&x, b("ln1.gamma"), b("ln1.beta"));
        let q = affine(&z, b("attn.w_q"), b("attn.b_q"));
        let k = affine(&z, b("attn.w_k"), b("attn.b_k"));
        let v = affine(&z, b("attn.w_v"), b("attn.b_v"));
        let mut att = Mat::zeros(n, d);
        let mut weights = Vec::new();
        for h in 0..heads {
            let cols = h * hs..(h + 1) * hs;
            let mut a = Mat::zeros(n, n);
            for t in 0..n {
                let qv = &q.row(t)[cols.clone()];
                let mut max = T::from_f64(f64::NEG_INFINITY);
                for j in 0..=t {
                    let kv = &k.row(j)[cols.clone()];
                    let mut s = T::zero();
                    for (&x1, &x2) in qv.iter().zip(kv) {
                        s += x1 * x2;
                    }
                    let s = s * scale;
                    a.row_mut(t)[j] = s;
                    max = max.max(s);
                }
                let mut total = T::zero();
                for s in &mut a.row_mut(t)[..=t] {
                    *s = (*s - max).exp();
                    total += *s;
                }
                for s in &mut a.row_mut(t)[..=t] {
                    *s = *s / total;
                }
                if ablate == Some((i, h)) {
                    continue; // the head's weights are kept, its output dropped
                }
                let out = &mut att.row_mut(t)[cols.clone()];
                for j in 0..=t {
                    let wt = a.row(t)[j];
                    for (o, &vv) in out.iter_mut().zip(&v.row(j)[cols.clone()]) {
                        *o += wt * vv;
                    }
                }
            }
            weights.push(a);
        }
        attention.push(weights);
        let o = affine(&att, b("attn.w_o"), b("attn.b_o"));
        add_in(&mut x, &o);
        let z = norm(&x, b("ln2.gamma"), b("ln2.beta"));
        let hdn = affine(&z, b("ffn.w1"), b("ffn.b1")).map(gelu);
        add_in(&mut x, &affine(&hdn, b("ffn.w2"), b("ffn.b2")));
        residual.push(x.clone());
    }
    Trace {
        residual,
        attention,
    }
}

fn add_in<T: Float>(x: &mut Mat<T>, y: &Mat<T>) {
    for (a, &b) in x.data.iter_mut().zip(&y.data) {
        *a += b;
    }
}

/// Next-token scores from a residual stream: the final layer norm, then
/// the tied output layer (each token's embedding). One row per position.
pub fn read_out<T: Float>(p: &Params<T>, x: &Mat<T>) -> Mat<T> {
    let w = |name: &str| &p.mats[p.position(name)];
    let z = norm(x, w("final_ln.gamma"), w("final_ln.beta"));
    z.matmul(&w("token_embedding").transpose())
}

/// −ln p(target) under softmax(scores), in f64.
fn nll<T: Float>(scores: &[T], target: usize) -> f64 {
    let max = scores
        .iter()
        .map(|s| s.to_f64())
        .fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = scores.iter().map(|s| (s.to_f64() - max).exp()).sum();
    max + total.ln() - scores[target].to_f64()
}

/// The index of the largest score.
pub fn argmax<T: Float>(scores: &[T]) -> usize {
    let mut best = 0;
    for (i, s) in scores.iter().enumerate() {
        if *s > scores[best] {
            best = i;
        }
    }
    best
}

/// One stage of the logit lens: the loss and top-1 accuracy of the
/// predictions read from the residual stream at that stage.
pub struct LensRow {
    pub loss: f64,
    pub accuracy: f64,
    /// Scores at the last position (the next token after the text).
    pub last: Vec<f64>,
}

/// The logit lens over a traced sequence: one row per stage (after the
/// embeddings, after each block), each position predicting the next token.
pub fn logit_lens<T: Float>(p: &Params<T>, tr: &Trace<T>, tokens: &[usize]) -> Vec<LensRow> {
    let n = tokens.len();
    tr.residual
        .iter()
        .map(|x| {
            let s = read_out(p, x);
            let (mut loss, mut right) = (0.0, 0usize);
            for t in 0..n.saturating_sub(1) {
                loss += nll(s.row(t), tokens[t + 1]);
                right += usize::from(argmax(s.row(t)) == tokens[t + 1]);
            }
            let m = (n.saturating_sub(1)).max(1) as f64;
            LensRow {
                loss: loss / m,
                accuracy: right as f64 / m,
                last: s.row(n - 1).iter().map(|v| v.to_f64()).collect(),
            }
        })
        .collect()
}

/// The repeated-random-tokens test, averaged over several sequences.
pub struct HeadScores {
    /// prev[block][head]: mean attention to the immediately preceding token.
    pub prev: Vec<Vec<f64>>,
    /// induction[block][head]: in the second copy, mean attention to the
    /// token after the current token's earlier occurrence.
    pub induction: Vec<Vec<f64>>,
    /// The model's loss predicting the first copy's tokens, and the second's.
    pub loss_first: f64,
    pub loss_second: f64,
    /// Random tokens per copy, and sequences averaged.
    pub length: usize,
    pub sequences: usize,
}

/// Head scores on `sequences` sequences of `start` then L random tokens
/// (drawn from `pool`) then the same L again, with 1 + 2L = the context (or
/// one less). L = (context − 1) / 2.
pub fn head_scores<T: Float>(
    p: &Params<T>,
    c: &ModelConfig,
    start: usize,
    pool: &[usize],
    sequences: usize,
    seed: u64,
) -> HeadScores {
    let l = (c.context - 1) / 2;
    let (mut prev, mut ind) = (
        vec![vec![0.0; c.heads]; c.layers],
        vec![vec![0.0; c.heads]; c.layers],
    );
    let (mut first, mut second) = (0.0, 0.0);
    for sq in 0..sequences {
        let mut rng = Rng::stream(seed, sq as u64);
        let r: Vec<usize> = (0..l).map(|_| pool[rng.below(pool.len())]).collect();
        let mut tokens = vec![start];
        tokens.extend(&r);
        tokens.extend(&r);
        let tr = trace(p, c, &tokens);
        // Positions: 0 is `start`; the first copy is 1…L, the second L+1…2L.
        for (i, block) in tr.attention.iter().enumerate() {
            for (h, a) in block.iter().enumerate() {
                let pv: f64 = (1..tokens.len()).map(|t| a.row(t)[t - 1].to_f64()).sum();
                prev[i][h] += pv / (tokens.len() - 1) as f64;
                // Second copy, position L+1+j holds r[j]; its earlier
                // occurrence is at 1+j, followed by r[j+1] at 2+j.
                let iv: f64 = (0..l - 1).map(|j| a.row(l + 1 + j)[2 + j].to_f64()).sum();
                ind[i][h] += iv / (l - 1) as f64;
            }
        }
        let s = read_out(p, tr.residual.last().expect("at least the embeddings"));
        // Predicting r[j+1] from position 1+j (first copy), and from L+1+j.
        first += (0..l - 1).map(|j| nll(s.row(1 + j), r[j + 1])).sum::<f64>() / (l - 1) as f64;
        second += (0..l - 1)
            .map(|j| nll(s.row(l + 1 + j), r[j + 1]))
            .sum::<f64>()
            / (l - 1) as f64;
    }
    let k = sequences as f64;
    for row in prev.iter_mut().chain(ind.iter_mut()) {
        for v in row.iter_mut() {
            *v /= k;
        }
    }
    HeadScores {
        prev,
        induction: ind,
        loss_first: first / k,
        loss_second: second / k,
        length: l,
        sequences,
    }
}

/// One head's part in copying a repeated real text.
pub struct CopyHead {
    pub block: usize,
    pub head: usize,
    /// In the second copy: mean attention to the token after the current
    /// token's earlier occurrence (the induction pattern).
    pub attention: f64,
    /// How much the loss on the second copy rises with the head ablated.
    pub ablation: f64,
}

/// Copying on real text: `story` (no start token) repeated twice after
/// `start`, cut so the whole fits the context. Returns the losses on the
/// first and second copies, and every head's induction attention and
/// ablation effect on the second copy.
pub fn copy_heads<T: Float>(
    p: &Params<T>,
    c: &ModelConfig,
    start: usize,
    story: &[usize],
) -> (f64, f64, usize, Vec<CopyHead>) {
    let l = story.len().min((c.context - 1) / 2);
    let s = &story[..l];
    let mut tokens = vec![start];
    tokens.extend(s);
    tokens.extend(s);
    // Predicting s[j+1] from position 1+j (first copy) or L+1+j (second).
    let loss = |tr: &Trace<T>, offset: usize| {
        let out = read_out(p, tr.residual.last().expect("at least the embeddings"));
        (0..l - 1)
            .map(|j| nll(out.row(offset + 1 + j), s[j + 1]))
            .sum::<f64>()
            / (l - 1) as f64
    };
    let base = trace(p, c, &tokens);
    let (first, second) = (loss(&base, 0), loss(&base, l));
    let mut heads = Vec::new();
    for (b, block) in base.attention.iter().enumerate() {
        for (h, a) in block.iter().enumerate() {
            let attention = (0..l - 1)
                .map(|j| a.row(l + 1 + j)[2 + j].to_f64())
                .sum::<f64>()
                / (l - 1) as f64;
            let ablated = trace_ablated(p, c, &tokens, Some((b, h)));
            heads.push(CopyHead {
                block: b,
                head: h,
                attention,
                ablation: loss(&ablated, l) - second,
            });
        }
    }
    (first, second, l, heads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::forward;
    use crate::model;

    fn tiny() -> ModelConfig {
        ModelConfig {
            d_model: 8,
            layers: 2,
            heads: 2,
            context: 9,
            d_ff: 16,
            seed: 3,
        }
    }

    fn params(c: &ModelConfig, vocab: usize) -> Params<f64> {
        let mut p = Params::<f64>::from_tensors(&model::init(c, vocab));
        let mut rng = Rng::new(5);
        for m in &mut p.mats {
            for x in &mut m.data {
                *x += 0.3 * rng.normal();
            }
        }
        p
    }

    /// The probe's predictions equal training's forward pass, so what it
    /// shows is what the model really computes.
    #[test]
    fn trace_matches_the_training_forward_pass() {
        let (c, vocab) = (tiny(), 13);
        let p = params(&c, vocab);
        let tokens = vec![1, 5, 2, 12, 0, 7, 7];
        let targets: Vec<Option<usize>> = vec![None; tokens.len()];
        let f = forward(&p, &c, &tokens, &targets, tokens.len());
        let want = f.tape.value(f.logits);
        let tr = trace(&p, &c, &tokens);
        let got = read_out(&p, tr.residual.last().unwrap());
        for (a, b) in want.data.iter().zip(&got.data) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
        // The last stage of the logit lens is the model's own loss.
        let lens = logit_lens(&p, &tr, &tokens);
        assert_eq!(lens.len(), c.layers + 1);
        let tg: Vec<Option<usize>> = tokens[1..].iter().map(|&t| Some(t)).chain([None]).collect();
        let f = forward(&p, &c, &tokens, &tg, tokens.len());
        let loss = f.tape.value(f.loss).data[0];
        assert!((lens[c.layers].loss - loss).abs() < 1e-10);
    }

    /// Every row of every head's weights sums to 1 and is zero past the
    /// diagonal (the causal mask); the scores are averages of such weights.
    #[test]
    fn attention_weights_are_causal_distributions() {
        let (c, vocab) = (tiny(), 13);
        let p = params(&c, vocab);
        let tr = trace(&p, &c, &[3, 1, 4, 1, 5, 9, 2]);
        for a in tr.attention.iter().flatten() {
            for t in 0..a.rows {
                let row = a.row(t);
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(row[t + 1..].iter().all(|&w| w == 0.0));
            }
        }
        // Ablating a head changes the output; ablating nothing doesn't.
        let toks = [3, 1, 4, 1, 5, 9, 2];
        let plain = trace(&p, &c, &toks);
        let none = trace_ablated(&p, &c, &toks, None);
        let one = trace_ablated(&p, &c, &toks, Some((1, 0)));
        assert_eq!(plain.residual[2], none.residual[2]);
        assert_ne!(plain.residual[2], one.residual[2]);
        assert_eq!(plain.residual[1], one.residual[1]); // block 0 is untouched
        let (_, _, l, ch) = copy_heads(&p, &c, 0, &[3, 1, 4, 1, 5, 9]);
        assert_eq!((l, ch.len()), (4, c.layers * c.heads)); // cut to (9 − 1) / 2
        let pool: Vec<usize> = (1..vocab).collect();
        let hs = head_scores(&p, &c, 0, &pool, 3, 1);
        assert_eq!(hs.length, 4); // (9 − 1) / 2
        for v in hs.prev.iter().chain(&hs.induction).flatten() {
            assert!((0.0..=1.0).contains(v));
        }
    }
}
