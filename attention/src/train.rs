//! Training one attention head on a task whose right attention pattern is
//! known in advance: at every position, predict the token `lag` positions
//! back. The right answer is to attend to exactly that position, so we can
//! watch the attention weights move from uniform to that pattern.
//!
//! The model, kept minimal so the prediction must come through attention:
//!
//!   x_t     = E[token_t] + P[t]          token embedding + position embedding
//!   Q, K, V = X·W_q, X·W_k, X·W_v        one head
//!   O       = softmax(mask(Q·Kᵀ/√h))·V   the attention output
//!   logits  = O·U                        one score per possible token
//!   loss    = −ln p(correct token), averaged over positions t ≥ lag
//!
//! There's no residual connection, FFN or layer normalization: they'd let
//! the model route around attention, and they'd lengthen the gradient
//! formulas. They come in the full block.

use crate::attention::{attend, causal_mask, softmax_rows, Trace};
use crate::matrix::{print_weights, Mat};
use crate::rng::Rng;

pub struct Config {
    pub vocab: usize,
    pub len: usize,
    pub dim: usize,
    pub head_size: usize,
    pub lag: usize,
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
    /// Position embeddings, len × d: what lets the head find "two back".
    pub p: Mat,
    pub wq: Mat,
    pub wk: Mat,
    pub wv: Mat,
    /// Output layer, head_size × vocab.
    pub u: Mat,
}

impl Params {
    fn new(c: &Config, rng: &mut Rng) -> Params {
        Params {
            e: Mat::random_normal(c.vocab, c.dim, 1.0, rng),
            p: Mat::random_normal(c.len, c.dim, 1.0, rng),
            // Small query and key weights: attention starts near uniform.
            wq: Mat::random_normal(c.dim, c.head_size, 0.02, rng),
            wk: Mat::random_normal(c.dim, c.head_size, 0.02, rng),
            wv: Mat::random_normal(c.dim, c.head_size, 0.2, rng),
            u: Mat::random_normal(c.head_size, c.vocab, 0.2, rng),
        }
    }

    fn zeros_like(&self) -> Params {
        let z = |m: &Mat| Mat::zeros(m.rows, m.cols);
        Params {
            e: z(&self.e),
            p: z(&self.p),
            wq: z(&self.wq),
            wk: z(&self.wk),
            wv: z(&self.wv),
            u: z(&self.u),
        }
    }

    fn all(&self) -> [&Mat; 6] {
        [&self.e, &self.p, &self.wq, &self.wk, &self.wv, &self.u]
    }

    fn all_mut(&mut self) -> [&mut Mat; 6] {
        [
            &mut self.e,
            &mut self.p,
            &mut self.wq,
            &mut self.wk,
            &mut self.wv,
            &mut self.u,
        ]
    }
}

/// Everything the forward pass computes for one sequence.
pub struct Forward {
    pub x: Mat,
    pub attn: Trace,
    pub probs: Mat,
    pub loss: f64,
    /// Correct predictions at the positions that have a target.
    pub correct: usize,
}

/// The target at position t: the token `lag` back, if there is one.
fn target(tokens: &[usize], t: usize, lag: usize) -> Option<usize> {
    t.checked_sub(lag).map(|s| tokens[s])
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

pub fn forward(w: &Params, tokens: &[usize], lag: usize) -> Forward {
    let x = embed(w, tokens);
    let attn = attend(&x, &w.wq, &w.wk, &w.wv);
    let probs = softmax_rows(&attn.out.matmul(&w.u));
    let (mut loss, mut correct) = (0.0, 0);
    for t in lag..tokens.len() {
        let tgt = target(tokens, t, lag).unwrap();
        loss -= probs.get(t, tgt).ln();
        let row = probs.row(t);
        let best = (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b });
        correct += usize::from(best == tgt);
    }
    loss /= (tokens.len() - lag) as f64;
    Forward {
        x,
        attn,
        probs,
        loss,
        correct,
    }
}

/// Backpropagation: the gradient of the loss with respect to every weight,
/// working backward through the forward pass one step at a time.
pub fn backward(w: &Params, tokens: &[usize], lag: usize, f: &Forward) -> Params {
    let n = (tokens.len() - lag) as f64;
    let h = w.wq.cols as f64;
    let a = &f.attn;
    let mut g = w.zeros_like();

    // Loss → logits. For softmax followed by −ln p(target), the gradient
    // with respect to the logits is (probabilities − one-hot target).
    let mut d_logits = f.probs.clone();
    for t in 0..tokens.len() {
        match target(tokens, t, lag) {
            None => d_logits.row_mut(t).fill(0.0), // no target: no loss
            Some(tgt) => {
                let v = d_logits.get(t, tgt);
                d_logits.set(t, tgt, v - 1.0);
            }
        }
    }
    let d_logits = d_logits.scale(1.0 / n);

    // logits = O·U
    g.u = a.out.transpose().matmul(&d_logits);
    let d_out = d_logits.matmul(&w.u.transpose());

    // O = A·V, where A holds the attention weights
    let d_weights = d_out.matmul(&a.v.transpose());
    let d_v = a.weights.transpose().matmul(&d_out);

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

    // Q, K, V = X·W_q, X·W_k, X·W_v
    let xt = f.x.transpose();
    g.wq = xt.matmul(&d_q);
    g.wk = xt.matmul(&d_k);
    g.wv = xt.matmul(&d_v);
    let mut d_x = d_q.matmul(&w.wq.transpose());
    d_x.add_scaled(&d_k.matmul(&w.wk.transpose()), 1.0);
    d_x.add_scaled(&d_v.matmul(&w.wv.transpose()), 1.0);

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

/// Check backpropagation against finite differences: nudge single weights
/// by ±ε, measure the change in the loss directly, and compare with the
/// computed gradient. Returns the worst relative error.
pub fn gradient_check(w: &Params, tokens: &[usize], lag: usize, rng: &mut Rng) -> f64 {
    let f = forward(w, tokens, lag);
    let g = backward(w, tokens, lag, &f);
    let eps = 1e-6;
    let mut probe = w.clone();
    let mut worst: f64 = 0.0;
    for k in 0..6 {
        for _ in 0..5 {
            let len = probe.all()[k].data.len();
            let i = rng.below(len);
            let orig = probe.all()[k].data[i];
            probe.all_mut()[k].data[i] = orig + eps;
            let up = forward(&probe, tokens, lag).loss;
            probe.all_mut()[k].data[i] = orig - eps;
            let down = forward(&probe, tokens, lag).loss;
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
struct Adam {
    m: Params,
    v: Params,
    t: i32,
}

impl Adam {
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;

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

fn random_tokens(c: &Config, rng: &mut Rng) -> Vec<usize> {
    (0..c.len).map(|_| rng.below(c.vocab)).collect()
}

fn letter(tok: usize) -> char {
    (b'a' + tok as u8) as char
}

/// Row and column labels such as "3:e" (position 3, token e).
fn labels(tokens: &[usize]) -> Vec<String> {
    tokens
        .iter()
        .enumerate()
        .map(|(t, &tok)| format!("{}:{}", t + 1, letter(tok)))
        .collect()
}

/// Accuracy over fresh random sequences.
fn evaluate(w: &Params, c: &Config, rng: &mut Rng, n: usize) -> f64 {
    let mut correct = 0;
    for _ in 0..n {
        correct += forward(w, &random_tokens(c, rng), c.lag).correct;
    }
    correct as f64 / (n * (c.len - c.lag)) as f64
}

pub fn run(c: &Config) {
    let mut rng = Rng::new(c.seed);
    let mut w = Params::new(c, &mut rng);
    let demo = random_tokens(c, &mut rng);
    let demo_labels = labels(&demo);

    println!(
        "=== Training one head: predict the token {} back ===",
        c.lag
    );
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
    println!(
        "weight on position t − {}. Rows 1–{} have nothing that far back to predict.",
        c.lag, c.lag
    );
    println!(
        "Chance-level loss is ln {} = {:.3}.\n",
        c.vocab,
        (c.vocab as f64).ln()
    );

    let worst = gradient_check(&w, &demo, c.lag, &mut rng);
    println!(
        "Gradient check (backpropagation vs finite differences, 30 weights): \
         worst relative error {worst:.1e} {}\n",
        if worst < 1e-5 {
            "— OK"
        } else {
            "— WARNING: gradients disagree"
        }
    );

    let mut adam = Adam {
        m: w.zeros_like(),
        v: w.zeros_like(),
        t: 0,
    };
    let snapshot = |w: &Params, step: usize, loss: f64, acc: f64| {
        let f = forward(w, &demo, c.lag);
        print_weights(
            &format!(
                "step {step}: loss {loss:.3}, accuracy {:.0}%  — attention on the sequence {}",
                100.0 * acc,
                demo.iter().map(|&t| letter(t)).collect::<String>()
            ),
            &f.attn.weights,
            &demo_labels,
            2,
        );
    };

    // Step 0: before any training.
    let mut eval_rng = Rng::new(c.seed ^ 0xE7A1);
    let f0 = forward(&w, &demo, c.lag);
    if c.show.contains(&0) {
        snapshot(&w, 0, f0.loss, evaluate(&w, c, &mut eval_rng, 50));
    }
    for step in 1..=c.steps {
        let mut grad = w.zeros_like();
        let mut loss = 0.0;
        let mut correct = 0;
        for _ in 0..c.batch {
            let tokens = random_tokens(c, &mut rng);
            let f = forward(&w, &tokens, c.lag);
            let g = backward(&w, &tokens, c.lag, &f);
            loss += f.loss;
            correct += f.correct;
            for (acc, gi) in grad.all_mut().into_iter().zip(g.all()) {
                acc.add_scaled(gi, 1.0 / c.batch as f64);
            }
        }
        adam.step(&mut w, &grad, c.lr);
        if c.show.contains(&step) {
            let acc = correct as f64 / (c.batch * (c.len - c.lag)) as f64;
            snapshot(&w, step, loss / c.batch as f64, acc);
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
    let h = (c.head_size as f64).sqrt();
    let scores = |a: &Mat, b: &Mat| {
        a.matmul(&w.wq)
            .matmul(&b.matmul(&w.wk).transpose())
            .scale(1.0 / h)
    };
    let spread = |m: &Mat| {
        let lo = m.data.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = m.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    // Scores between position embeddings alone (leaving out the token parts
    // of each x_t), and between token embeddings alone.
    let pos = scores(&w.p, &w.p);
    let tok = scores(&w.e, &w.e);
    let pos_labels: Vec<String> = (1..=c.len).map(|t| t.to_string()).collect();
    print_weights(
        "Where to look (QK): attention computed from the position embeddings alone",
        &softmax_rows(&causal_mask(&pos)),
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
    let ov = w.e.matmul(&w.wv).matmul(&w.u);
    let copies = (0..c.vocab)
        .filter(|&t| {
            let row = ov.row(t);
            (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b }) == t
        })
        .count();
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
            lag: 2,
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
        assert!(gradient_check(&w, &tokens, c.lag, &mut rng) < 1e-5);
    }

    #[test]
    fn training_learns_the_lag_task() {
        let c = config(400);
        let mut rng = Rng::new(c.seed);
        let mut w = Params::new(&c, &mut rng);
        let mut adam = Adam {
            m: w.zeros_like(),
            v: w.zeros_like(),
            t: 0,
        };
        for _ in 0..c.steps {
            let mut grad = w.zeros_like();
            for _ in 0..c.batch {
                let tokens = random_tokens(&c, &mut rng);
                let g = backward(&w, &tokens, c.lag, &forward(&w, &tokens, c.lag));
                for (acc, gi) in grad.all_mut().into_iter().zip(g.all()) {
                    acc.add_scaled(gi, 1.0 / c.batch as f64);
                }
            }
            adam.step(&mut w, &grad, c.lr);
        }
        assert!(evaluate(&w, &c, &mut Rng::new(99), 200) > 0.98);
        // Every row with a target puts most of its weight lag positions back.
        let f = forward(&w, &random_tokens(&c, &mut rng), c.lag);
        for t in c.lag..c.len {
            assert!(f.attn.weights.get(t, t - c.lag) > 0.8, "row {t}");
        }
    }
}
