//! Training attention heads on tasks whose right attention pattern is known
//! in advance: at every position, predict the token(s) at fixed distances
//! back (the "lags"). With one lag, the right answer is to attend to exactly
//! that position, so we can watch the attention weights move from uniform
//! to that pattern. With two lags, one head isn't enough (see heads.rs).
//!
//! The model, kept minimal so the prediction must come through attention:
//!
//!   x_t     = E[token_t] + P[t]            token embedding + position embedding
//!   O_k     = softmax(mask(Q_k·K_kᵀ/√h))·V_k   head k (Q_k = X·W_q,k, and so on)
//!   O       = [O_1 O_2 …]                  the heads' outputs, side by side
//!   logits  = O·U                          one score per possible answer
//!   loss    = −ln p(correct answer), averaged over positions with a target
//!
//! With one lag, an answer is a token; with lags (1, 2), it is a pair of
//! tokens, one of V² classes. A real attention sublayer multiplies O by one
//! more matrix, W_o, to bring it back to width d for the residual stream.
//! Here U follows directly, and two matrices in a row are equivalent to one
//! (W_o·U), so U plays both parts. W_o arrives with the residual stream.
//!
//! There's no residual connection, FFN or layer normalization: they'd let
//! the model route around attention, and they'd lengthen the gradient
//! formulas. They come in the full block.

use crate::attention::{attend, causal_mask, softmax_rows, Trace};
use crate::matrix::{print_weights, Mat};
use crate::rng::Rng;
use crate::tape::{Tape, Var};

pub struct Config {
    pub vocab: usize,
    pub len: usize,
    pub dim: usize,
    pub head_size: usize,
    /// Number of heads.
    pub heads: usize,
    /// The distances back whose tokens are to be predicted, together.
    pub lags: Vec<usize>,
    pub steps: usize,
    pub lr: f64,
    pub batch: usize,
    pub seed: u64,
    pub show: Vec<usize>,
}

/// The model's weights, by name. Gradients and Adam's running averages use
/// the same structure, which keeps the update loops simple.
#[derive(Clone)]
pub struct Params {
    /// Token embeddings, vocab × d.
    pub e: Mat,
    /// Position embeddings, len × d: what lets a head find "two back".
    pub p: Mat,
    pub heads: Vec<HeadWeights>,
    /// Output layer, (heads × head_size) × classes.
    pub u: Mat,
}

/// One head's query, key and value matrices, each d × head_size.
#[derive(Clone)]
pub struct HeadWeights {
    pub wq: Mat,
    pub wk: Mat,
    pub wv: Mat,
}

impl Params {
    pub fn new(c: &Config, rng: &mut Rng) -> Params {
        let e = Mat::random_normal(c.vocab, c.dim, 1.0, rng);
        let p = Mat::random_normal(c.len, c.dim, 1.0, rng);
        let heads = (0..c.heads)
            .map(|_| HeadWeights {
                // Small query and key weights: attention starts near uniform.
                wq: Mat::random_normal(c.dim, c.head_size, 0.02, rng),
                wk: Mat::random_normal(c.dim, c.head_size, 0.02, rng),
                wv: Mat::random_normal(c.dim, c.head_size, 0.2, rng),
            })
            .collect();
        let u = Mat::random_normal(c.heads * c.head_size, classes(c), 0.2, rng);
        Params { e, p, heads, u }
    }

    fn zeros_like(&self) -> Params {
        let z = |m: &Mat| Mat::zeros(m.rows, m.cols);
        Params {
            e: z(&self.e),
            p: z(&self.p),
            heads: self
                .heads
                .iter()
                .map(|h| HeadWeights {
                    wq: z(&h.wq),
                    wk: z(&h.wk),
                    wv: z(&h.wv),
                })
                .collect(),
            u: z(&self.u),
        }
    }

    /// Every weight matrix, in a fixed order: E, P, each head's W_q, W_k,
    /// W_v, then U.
    fn all(&self) -> Vec<&Mat> {
        let mut v = vec![&self.e, &self.p];
        for h in &self.heads {
            v.extend([&h.wq, &h.wk, &h.wv]);
        }
        v.push(&self.u);
        v
    }

    fn all_mut(&mut self) -> Vec<&mut Mat> {
        let mut v = vec![&mut self.e, &mut self.p];
        for h in &mut self.heads {
            v.extend([&mut h.wq, &mut h.wk, &mut h.wv]);
        }
        v.push(&mut self.u);
        v
    }
}

/// Everything the forward pass computes for one sequence.
pub struct Forward {
    pub x: Mat,
    /// One trace per head.
    pub attn: Vec<Trace>,
    /// The heads' outputs side by side.
    pub out: Mat,
    pub probs: Mat,
    pub loss: f64,
    /// Correct predictions at the positions that have a target.
    pub correct: usize,
}

/// Number of possible answers: V for one lag, V² for two, and so on.
pub fn classes(c: &Config) -> usize {
    c.vocab.pow(c.lags.len() as u32)
}

/// The target at position t, if every lag reaches back that far: the
/// tokens at t − lag for each lag, combined into one class number, with
/// the first lag's token as the most significant digit (base V).
fn target(tokens: &[usize], t: usize, lags: &[usize], vocab: usize) -> Option<usize> {
    lags.iter().try_fold(0, |acc, &lag| {
        t.checked_sub(lag).map(|s| acc * vocab + tokens[s])
    })
}

/// The first position with a target.
fn first(lags: &[usize]) -> usize {
    lags.iter().copied().max().unwrap_or(0)
}

/// Copy columns [from, from + width) of m.
fn columns(m: &Mat, from: usize, width: usize) -> Mat {
    let mut out = Mat::zeros(m.rows, width);
    for i in 0..m.rows {
        out.row_mut(i)
            .copy_from_slice(&m.row(i)[from..from + width]);
    }
    out
}

/// Place matrices side by side.
fn side_by_side(ms: &[&Mat]) -> Mat {
    let rows = ms[0].rows;
    let mut out = Mat::zeros(rows, ms.iter().map(|m| m.cols).sum());
    for i in 0..rows {
        let mut at = 0;
        for m in ms {
            out.row_mut(i)[at..at + m.cols].copy_from_slice(m.row(i));
            at += m.cols;
        }
    }
    out
}

fn embed(w: &Params, tokens: &[usize]) -> Mat {
    let mut x = Mat::zeros(tokens.len(), w.e.cols);
    for (t, &tok) in tokens.iter().enumerate() {
        let row = x.row_mut(t);
        for ((r, e), p) in row.iter_mut().zip(w.e.row(tok)).zip(w.p.row(t)) {
            *r = e + p;
        }
    }
    x
}

pub fn forward(w: &Params, tokens: &[usize], lags: &[usize]) -> Forward {
    let x = embed(w, tokens);
    let attn: Vec<Trace> = w
        .heads
        .iter()
        .map(|h| attend(&x, &h.wq, &h.wk, &h.wv))
        .collect();
    let out = side_by_side(&attn.iter().map(|a| &a.out).collect::<Vec<_>>());
    let probs = softmax_rows(&out.matmul(&w.u));
    let vocab = w.e.rows;
    let lag = first(lags);
    let (mut loss, mut correct) = (0.0, 0);
    for t in lag..tokens.len() {
        let tgt = target(tokens, t, lags, vocab).unwrap();
        loss -= probs.get(t, tgt).ln();
        let row = probs.row(t);
        let best = (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b });
        correct += usize::from(best == tgt);
    }
    loss /= (tokens.len() - lag) as f64;
    Forward {
        x,
        attn,
        out,
        probs,
        loss,
        correct,
    }
}

/// Backpropagation: the gradient of the loss with respect to every weight,
/// working backward through the forward pass one step at a time.
pub fn backward(w: &Params, tokens: &[usize], lags: &[usize], f: &Forward) -> Params {
    let n = (tokens.len() - first(lags)) as f64;
    let vocab = w.e.rows;
    let mut g = w.zeros_like();

    // Loss → logits. For softmax followed by −ln p(target), the gradient
    // with respect to the logits is (probabilities − one-hot target).
    let mut d_logits = f.probs.clone();
    for t in 0..tokens.len() {
        match target(tokens, t, lags, vocab) {
            None => d_logits.row_mut(t).fill(0.0), // no target: no loss
            Some(tgt) => {
                let v = d_logits.get(t, tgt);
                d_logits.set(t, tgt, v - 1.0);
            }
        }
    }
    let d_logits = d_logits.scale(1.0 / n);

    // logits = O·U
    g.u = f.out.transpose().matmul(&d_logits);
    let d_out_all = d_logits.matmul(&w.u.transpose());

    // O = [O_1 O_2 …]: each head gets its own columns of the gradient, and
    // the heads' contributions to the gradient for X add up.
    let mut d_x = Mat::zeros(f.x.rows, f.x.cols);
    let xt = f.x.transpose();
    for (k, (hw, a)) in w.heads.iter().zip(&f.attn).enumerate() {
        let hs = hw.wq.cols;
        let d_out = columns(&d_out_all, k * hs, hs);
        let (dq, dk, dv) = head_backward(a, &d_out, hs as f64);
        let gh = &mut g.heads[k];
        // Q, K, V = X·W_q, X·W_k, X·W_v
        gh.wq = xt.matmul(&dq);
        gh.wk = xt.matmul(&dk);
        gh.wv = xt.matmul(&dv);
        d_x.add_scaled(&dq.matmul(&hw.wq.transpose()), 1.0);
        d_x.add_scaled(&dk.matmul(&hw.wk.transpose()), 1.0);
        d_x.add_scaled(&dv.matmul(&hw.wv.transpose()), 1.0);
    }

    // X = E[token] + P: each row's gradient goes to its token's embedding
    // and to its position's embedding.
    for (t, &tok) in tokens.iter().enumerate() {
        for (ge, dx) in g.e.row_mut(tok).iter_mut().zip(d_x.row(t)) {
            *ge += dx;
        }
        g.p.row_mut(t).copy_from_slice(d_x.row(t));
    }
    g
}

/// One head, backward: from the gradient for its output O to the gradients
/// for its Q, K and V.
fn head_backward(a: &Trace, d_out: &Mat, h: f64) -> (Mat, Mat, Mat) {
    // O = A·V, where A holds the attention weights
    let d_weights = d_out.matmul(&a.v.transpose());
    let d_v = a.weights.transpose().matmul(d_out);

    // A = softmax(S) row by row: dS_ij = A_ij (dA_ij − Σ_k A_ik dA_ik).
    // Masked positions have A_ij = 0, so they get no gradient.
    let mut d_scores = Mat::zeros(a.weights.rows, a.weights.cols);
    for i in 0..a.weights.rows {
        let (ai, dai) = (a.weights.row(i), d_weights.row(i));
        let dot: f64 = ai.iter().zip(dai).map(|(x, y)| x * y).sum();
        for (j, s) in d_scores.row_mut(i).iter_mut().enumerate() {
            *s = ai[j] * (dai[j] - dot);
        }
    }

    // S = Q·Kᵀ / √h
    let d_q = d_scores.matmul(&a.k).scale(1.0 / h.sqrt());
    let d_k = d_scores.transpose().matmul(&a.q).scale(1.0 / h.sqrt());
    (d_q, d_k, d_v)
}

/// The same model written on the automatic differentiation tape: forward
/// code only, no backward code at all. The tape records each operation and
/// derives the gradients. Returns the loss and every weight's gradient,
/// which must match `backward`'s (see `tape_agreement`).
pub fn tape_gradients(w: &Params, tokens: &[usize], lags: &[usize]) -> (f64, Params) {
    let mut t = Tape::new();
    let e = t.leaf(w.e.clone());
    let p = t.leaf(w.p.clone());
    let heads: Vec<[Var; 3]> = w
        .heads
        .iter()
        .map(|h| {
            [
                t.leaf(h.wq.clone()),
                t.leaf(h.wk.clone()),
                t.leaf(h.wv.clone()),
            ]
        })
        .collect();
    let u = t.leaf(w.u.clone());

    // x_t = E[token_t] + P[t]
    let tok = t.gather(e, tokens);
    let positions: Vec<usize> = (0..tokens.len()).collect();
    let pos = t.gather(p, &positions);
    let x = t.add(tok, pos);

    // Each head: softmax(mask(Q·Kᵀ/√h))·V
    let mut outs = Vec::new();
    for (&[wq, wk, wv], hw) in heads.iter().zip(&w.heads) {
        let q = t.matmul(x, wq);
        let k = t.matmul(x, wk);
        let v = t.matmul(x, wv);
        let kt = t.transpose(k);
        let scores = t.matmul(q, kt);
        let scores = t.scale(scores, 1.0 / (hw.wq.cols as f64).sqrt());
        let weights = t.causal_softmax(scores);
        outs.push(t.matmul(weights, v));
    }

    // The heads side by side, the output layer, and the loss.
    let o = t.concat(&outs);
    let logits = t.matmul(o, u);
    let vocab = w.e.rows;
    let targets: Vec<Option<usize>> = (0..tokens.len())
        .map(|i| target(tokens, i, lags, vocab))
        .collect();
    let loss = t.softmax_cross_entropy(logits, &targets);

    t.backward(loss);
    let grads = Params {
        e: t.grad(e),
        p: t.grad(p),
        heads: heads
            .iter()
            .map(|&[wq, wk, wv]| HeadWeights {
                wq: t.grad(wq),
                wk: t.grad(wk),
                wv: t.grad(wv),
            })
            .collect(),
        u: t.grad(u),
    };
    (t.value(loss).data[0], grads)
}

/// Compare the tape's gradients with the hand-written backpropagation:
/// the largest difference over every weight, relative to the largest
/// gradient. The two add the same numbers in slightly different orders,
/// so expect rounding-level agreement (around 10⁻¹⁶), not exact equality.
pub fn tape_agreement(w: &Params, tokens: &[usize], lags: &[usize]) -> f64 {
    let hand = backward(w, tokens, lags, &forward(w, tokens, lags));
    let (_, tape) = tape_gradients(w, tokens, lags);
    let (mut diff, mut size): (f64, f64) = (0.0, 0.0);
    for (a, b) in hand.all().into_iter().zip(tape.all()) {
        for (x, y) in a.data.iter().zip(&b.data) {
            diff = diff.max((x - y).abs());
            size = size.max(x.abs());
        }
    }
    diff / size
}

/// Check backpropagation against finite differences: nudge single weights
/// by ±ε, measure the change in the loss directly, and compare with the
/// computed gradient. Returns the worst relative error.
pub fn gradient_check(w: &Params, tokens: &[usize], lags: &[usize], rng: &mut Rng) -> f64 {
    let f = forward(w, tokens, lags);
    let g = backward(w, tokens, lags, &f);
    let eps = 1e-6;
    let mut probe = w.clone();
    let mut worst: f64 = 0.0;
    for k in 0..w.all().len() {
        for _ in 0..5 {
            let len = probe.all()[k].data.len();
            let i = rng.below(len);
            let orig = probe.all()[k].data[i];
            probe.all_mut()[k].data[i] = orig + eps;
            let up = forward(&probe, tokens, lags).loss;
            probe.all_mut()[k].data[i] = orig - eps;
            let down = forward(&probe, tokens, lags).loss;
            probe.all_mut()[k].data[i] = orig;
            let numeric = (up - down) / (2.0 * eps);
            let analytic = g.all()[k].data[i];
            let scale = numeric.abs() + analytic.abs();
            if scale > 1e-10 {
                worst = worst.max((numeric - analytic).abs() / scale);
            }
        }
    }
    worst
}

/// Adam: gradient descent that keeps running averages of each weight's
/// gradient (m) and squared gradient (v), and scales each step by them,
/// so every weight moves at a sensible rate.
pub struct Adam {
    m: Params,
    v: Params,
    t: i32,
}

impl Adam {
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;

    pub fn new(w: &Params) -> Adam {
        Adam {
            m: w.zeros_like(),
            v: w.zeros_like(),
            t: 0,
        }
    }

    fn step(&mut self, w: &mut Params, g: &Params, lr: f64) {
        self.t += 1;
        let (c1, c2) = (1.0 - Self::B1.powi(self.t), 1.0 - Self::B2.powi(self.t));
        let (ms, vs, ws) = (self.m.all_mut(), self.v.all_mut(), w.all_mut());
        for (((m, v), w), g) in ms.into_iter().zip(vs).zip(ws).zip(g.all()) {
            for i in 0..w.data.len() {
                let gi = g.data[i];
                m.data[i] = Self::B1 * m.data[i] + (1.0 - Self::B1) * gi;
                v.data[i] = Self::B2 * v.data[i] + (1.0 - Self::B2) * gi * gi;
                w.data[i] -= lr * (m.data[i] / c1) / ((v.data[i] / c2).sqrt() + 1e-8);
            }
        }
    }
}

/// One training step: average the gradient over a batch of fresh random
/// sequences and let Adam update the weights. Returns the batch's mean
/// loss and its accuracy.
pub fn train_step(w: &mut Params, adam: &mut Adam, c: &Config, rng: &mut Rng) -> (f64, f64) {
    let mut grad = w.zeros_like();
    let (mut loss, mut correct) = (0.0, 0);
    for _ in 0..c.batch {
        let tokens = random_tokens(c, rng);
        let f = forward(w, &tokens, &c.lags);
        let g = backward(w, &tokens, &c.lags, &f);
        loss += f.loss;
        correct += f.correct;
        for (acc, gi) in grad.all_mut().into_iter().zip(g.all()) {
            acc.add_scaled(gi, 1.0 / c.batch as f64);
        }
    }
    adam.step(w, &grad, c.lr);
    let acc = correct as f64 / (c.batch * (c.len - first(&c.lags))) as f64;
    (loss / c.batch as f64, acc)
}

pub fn random_tokens(c: &Config, rng: &mut Rng) -> Vec<usize> {
    (0..c.len).map(|_| rng.below(c.vocab)).collect()
}

pub fn letter(tok: usize) -> char {
    (b'a' + tok as u8) as char
}

pub fn word(tokens: &[usize]) -> String {
    tokens.iter().map(|&t| letter(t)).collect()
}

/// Row and column labels such as "3:e" (position 3, token e).
pub fn labels(tokens: &[usize]) -> Vec<String> {
    tokens
        .iter()
        .enumerate()
        .map(|(t, &tok)| format!("{}:{}", t + 1, letter(tok)))
        .collect()
}

/// Accuracy over fresh random sequences.
pub fn evaluate(w: &Params, c: &Config, rng: &mut Rng, n: usize) -> f64 {
    let mut correct = 0;
    for _ in 0..n {
        correct += forward(w, &random_tokens(c, rng), &c.lags).correct;
    }
    correct as f64 / (n * (c.len - first(&c.lags))) as f64
}

/// The index of the largest entry.
pub fn argmax(row: &[f64]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

/// Score spread (largest minus smallest) of a matrix.
pub fn spread(m: &Mat) -> f64 {
    let lo = m.data.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = m.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    hi - lo
}

/// Scores between two sets of vectors as seen by one head: (A·W_q)(B·W_k)ᵀ/√h.
pub fn head_scores(hw: &HeadWeights, a: &Mat, b: &Mat) -> Mat {
    let h = (hw.wq.cols as f64).sqrt();
    a.matmul(&hw.wq)
        .matmul(&b.matmul(&hw.wk).transpose())
        .scale(1.0 / h)
}

/// A head's attention computed from the position embeddings alone (leaving
/// out the token part of each x_t): where it looks, regardless of content.
pub fn position_attention(w: &Params, hw: &HeadWeights) -> (Mat, Mat) {
    let pos = head_scores(hw, &w.p, &w.p);
    (softmax_rows(&causal_mask(&pos)), pos)
}

/// `train`: one head, one lag.
pub fn run(c: &Config) {
    let lag = c.lags[0];
    let mut rng = Rng::new(c.seed);
    let mut w = Params::new(c, &mut rng);
    let demo = random_tokens(c, &mut rng);
    let demo_labels = labels(&demo);

    println!("=== Training one head: predict the token {lag} back ===");
    println!(
        "Random sequences of {} tokens over {} symbols (a–{}); d = {}, head size {}.",
        c.len,
        c.vocab,
        letter(c.vocab - 1),
        c.dim,
        c.head_size
    );
    println!("Model: token + position embedding → one attention head → output layer.");
    println!(
        "Adam, learning rate {}, {} sequences per step, {} steps, seed {}.\n",
        c.lr, c.batch, c.steps, c.seed
    );
    println!("The right attention pattern is known in advance: row t should put its");
    println!("weight on position t − {lag}. Rows 1–{lag} have nothing that far back to predict.");
    println!(
        "Chance-level loss is ln {} = {:.3}.\n",
        c.vocab,
        (c.vocab as f64).ln()
    );

    let worst = gradient_check(&w, &demo, &c.lags, &mut rng);
    println!(
        "Gradient check (backpropagation vs finite differences, {} weights): \
         worst relative error {worst:.1e} {}\n",
        5 * w.all().len(),
        if worst < 1e-5 {
            "— OK"
        } else {
            "— WARNING: gradients disagree"
        }
    );

    let agree = tape_agreement(&w, &demo, &c.lags);
    println!(
        "Automatic differentiation tape vs hand-written backpropagation: largest\n\
         difference {agree:.1e} of the largest gradient {}\n",
        if agree < 1e-12 {
            "— they agree"
        } else {
            "— WARNING: they disagree"
        }
    );

    let snapshot = |w: &Params, step: usize, loss: f64, acc: f64| {
        let f = forward(w, &demo, &c.lags);
        print_weights(
            &format!(
                "step {step}: loss {loss:.3}, accuracy {:.0}%  — attention on the sequence {}",
                100.0 * acc,
                word(&demo)
            ),
            &f.attn[0].weights,
            &demo_labels,
            2,
        );
    };

    // Step 0: before any training.
    let mut eval_rng = Rng::new(c.seed ^ 0xE7A1);
    if c.show.contains(&0) {
        let f0 = forward(&w, &demo, &c.lags);
        snapshot(&w, 0, f0.loss, evaluate(&w, c, &mut eval_rng, 50));
    }
    let mut adam = Adam::new(&w);
    for step in 1..=c.steps {
        let (loss, acc) = train_step(&mut w, &mut adam, c, &mut rng);
        if c.show.contains(&step) {
            snapshot(&w, step, loss, acc);
        }
    }

    println!("=== After training ===");
    println!(
        "Accuracy on 500 fresh sequences: {:.1}%\n",
        100.0 * evaluate(&w, c, &mut eval_rng, 500)
    );
    explain(&w, c);
}

/// Show how the head solved the task: where to look came from positions
/// (the QK circuit), what to copy from token identity (the OV circuit).
fn explain(w: &Params, c: &Config) {
    let hw = &w.heads[0];
    let (pos_attn, pos) = position_attention(w, hw);
    let tok = head_scores(hw, &w.e, &w.e);
    let pos_labels: Vec<String> = (1..=c.len).map(|t| t.to_string()).collect();
    print_weights(
        "Where to look (QK): attention computed from the position embeddings alone",
        &pos_attn,
        &pos_labels,
        2,
    );
    println!(
        "Score spread from positions: {:.2}; from token identities: {:.2}.",
        spread(&pos),
        spread(&tok)
    );
    println!("The head learned to look by position and to ignore which token is where.\n");

    // What to copy: a token's value, sent through the output layer.
    let ov = w.e.matmul(&hw.wv).matmul(&w.u);
    let copies = (0..c.vocab).filter(|&t| argmax(ov.row(t)) == t).count();
    println!("What to copy (OV): sending each token's embedding through W_v and U, the");
    println!(
        "highest-scoring output is the token itself for {copies} of {} tokens.",
        c.vocab
    );
    println!("So W_q and W_k learned *where* to look, and W_v and U *what* to pass on:");
    println!("the QK/OV split described in PLAN.md, appearing on its own.");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(steps: usize) -> Config {
        Config {
            vocab: 8,
            len: 8,
            dim: 16,
            head_size: 16,
            heads: 1,
            lags: vec![2],
            steps,
            lr: 0.001,
            batch: 16,
            seed: 1,
            show: vec![],
        }
    }

    #[test]
    fn backpropagation_matches_finite_differences() {
        let c = config(0);
        let mut rng = Rng::new(7);
        let w = Params::new(&c, &mut rng);
        let tokens = random_tokens(&c, &mut rng);
        assert!(gradient_check(&w, &tokens, &c.lags, &mut rng) < 1e-5);
    }

    #[test]
    fn tape_reproduces_the_hand_written_gradients() {
        let c = config(0);
        let mut rng = Rng::new(11);
        for _ in 0..5 {
            let w = Params::new(&c, &mut rng);
            let tokens = random_tokens(&c, &mut rng);
            assert!(tape_agreement(&w, &tokens, &c.lags) < 1e-12);
            // Same loss too.
            let (loss, _) = tape_gradients(&w, &tokens, &c.lags);
            assert!((loss - forward(&w, &tokens, &c.lags).loss).abs() < 1e-12);
        }
    }

    #[test]
    fn training_learns_the_lag_task() {
        let c = config(400);
        let mut rng = Rng::new(c.seed);
        let mut w = Params::new(&c, &mut rng);
        let mut adam = Adam::new(&w);
        for _ in 0..c.steps {
            train_step(&mut w, &mut adam, &c, &mut rng);
        }
        assert!(evaluate(&w, &c, &mut Rng::new(99), 200) > 0.98);
        // Every row with a target puts most of its weight lag positions back.
        let f = forward(&w, &random_tokens(&c, &mut rng), &c.lags);
        for t in 2..c.len {
            assert!(f.attn[0].weights.get(t, t - 2) > 0.8, "row {t}");
        }
    }
}

#[cfg(test)]
mod target_tests {
    use super::*;

    #[test]
    fn targets_combine_the_lagged_tokens_in_base_v() {
        let tokens = [3, 1, 2, 0, 1];
        // One lag: the token itself.
        assert_eq!(target(&tokens, 4, &[2], 4), Some(2));
        // Lags (1, 2) at t = 4: tokens 0 (one back) and 2 (two back) → 0·4 + 2.
        assert_eq!(target(&tokens, 4, &[1, 2], 4), Some(2));
        // At t = 2: tokens 1 and 3 → 1·4 + 3 = 7.
        assert_eq!(target(&tokens, 2, &[1, 2], 4), Some(7));
        // Not far enough in for the larger lag: no target.
        assert_eq!(target(&tokens, 1, &[1, 2], 4), None);
    }
}
