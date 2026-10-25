//! Item 5: one complete transformer block, written as forward code only on
//! the automatic differentiation tape: the residual stream, layer norm,
//! several heads joined by W_o, and the FFN.
//!
//!   x      = E[token] + P[t]                    the residual stream starts
//!   h      = x + W_o·[heads](LN₁(x))            attention sublayer
//!   y      = h + W₂·GELU(W₁·LN₂(h) + b₁) + b₂   FFN sublayer
//!   logits = LN_f(y)·U
//!
//! The task: predict (token one back + token two back) mod V. Attention has
//! to bring the two digits to the current position; adding them mod V is a
//! nonlinear function of the pair, the kind of computation the FFN is for.

use crate::matrix::{print_table, print_weights, Mat};
use crate::optim::adam_update;
use crate::rng::Rng;
use crate::tape::{Tape, Var};

#[derive(Clone)]
pub struct Config {
    pub vocab: usize,
    pub len: usize,
    pub dim: usize,
    /// Number of blocks, applied one after another.
    pub layers: usize,
    /// Heads per block.
    pub heads: usize,
    pub d_ff: usize,
    pub steps: usize,
    pub lr: f64,
    pub batch: usize,
    pub seed: u64,
    /// Include the FFN sublayer (false for the ablation).
    pub ffn: bool,
    pub task: Task,
}

/// What the model is trained to predict.
#[derive(Clone, Copy)]
pub enum Task {
    /// (token one back + token two back) mod V, at each position from 3 on.
    SumMod,
    /// Next-token prediction on sequences A A tail, where A is k random
    /// tokens, k from `min` to `max`. Accuracy is measured where the next
    /// token is predictable: inside the second copy of A.
    Induction { min: usize, max: usize },
}

/// One training or test example: the tokens, the target at each position
/// (None where there is no loss), and which positions count for accuracy.
pub struct Example {
    pub tokens: Vec<usize>,
    pub targets: Vec<Option<usize>>,
    pub scored: Vec<bool>,
}

impl Task {
    pub fn sample(&self, c: &Config, rng: &mut Rng) -> Example {
        match *self {
            Task::SumMod => {
                let tokens: Vec<usize> = (0..c.len).map(|_| rng.below(c.vocab)).collect();
                let targets: Vec<Option<usize>> =
                    (0..c.len).map(|t| target(&tokens, t, c.vocab)).collect();
                let scored = targets.iter().map(Option::is_some).collect();
                Example {
                    tokens,
                    targets,
                    scored,
                }
            }
            Task::Induction { min, max } => {
                // A has k distinct tokens, so "the token that followed the
                // earlier copy of this one" is always well defined.
                let k = min + rng.below(max - min + 1);
                let mut a: Vec<usize> = Vec::with_capacity(k);
                while a.len() < k {
                    let tok = rng.below(c.vocab);
                    if !a.contains(&tok) {
                        a.push(tok);
                    }
                }
                // A A starts at a random point, after filler, so a token's
                // position says nothing about where the earlier copy is.
                // Filler never uses A's tokens, so there's no false match.
                let start = rng.below(c.len - 2 * k + 1);
                let mut filler = || loop {
                    let tok = rng.below(c.vocab);
                    if !a.contains(&tok) {
                        break tok;
                    }
                };
                let mut tokens: Vec<usize> = (0..start).map(|_| filler()).collect();
                tokens.extend(&a);
                tokens.extend(&a);
                while tokens.len() < c.len {
                    tokens.push(filler());
                }
                // Next-token prediction everywhere; the last position has
                // nothing after it.
                let targets = (0..c.len)
                    .map(|t| (t + 1 < c.len).then(|| tokens[t + 1]))
                    .collect();
                // Predictable: the positions inside the second copy of A
                // whose next token is also in it.
                let second = start + k;
                let scored = (0..c.len)
                    .map(|t| t >= second && t + 1 < second + k)
                    .collect();
                Example {
                    tokens,
                    targets,
                    scored,
                }
            }
        }
    }
}

/// One block's weights.
#[derive(Clone)]
pub struct Layer {
    pub ln1_g: Mat,
    pub ln1_b: Mat,
    /// Per head: W_q, W_k, W_v, each d × h.
    pub heads: Vec<[Mat; 3]>,
    /// The output matrix W_o, d × d: joined heads back to the residual stream.
    pub wo: Mat,
    pub ln2_g: Mat,
    pub ln2_b: Mat,
    pub w1: Mat,
    pub b1: Mat,
    pub w2: Mat,
    pub b2: Mat,
}

/// The model's weights, by name.
#[derive(Clone)]
pub struct Weights {
    pub e: Mat,
    pub p: Mat,
    pub layers: Vec<Layer>,
    pub lnf_g: Mat,
    pub lnf_b: Mat,
    pub u: Mat,
}

impl Weights {
    pub fn new(c: &Config, rng: &mut Rng) -> Weights {
        let d = c.dim;
        let h = d / c.heads;
        let ones = |n: usize| {
            let mut m = Mat::zeros(1, n);
            m.data.fill(1.0);
            m
        };
        let s = 1.0 / (d as f64).sqrt();
        let e = Mat::random_normal(c.vocab, d, 1.0, rng);
        let p = Mat::random_normal(c.len, d, 1.0, rng);
        let layers = (0..c.layers)
            .map(|_| Layer {
                ln1_g: ones(d),
                ln1_b: Mat::zeros(1, d),
                heads: (0..c.heads)
                    .map(|_| {
                        [
                            Mat::random_normal(d, h, 0.02, rng),
                            Mat::random_normal(d, h, 0.02, rng),
                            Mat::random_normal(d, h, s, rng),
                        ]
                    })
                    .collect(),
                wo: Mat::random_normal(d, d, s, rng),
                ln2_g: ones(d),
                ln2_b: Mat::zeros(1, d),
                w1: Mat::random_normal(d, c.d_ff, s, rng),
                b1: Mat::zeros(1, c.d_ff),
                w2: Mat::random_normal(c.d_ff, d, 1.0 / (c.d_ff as f64).sqrt(), rng),
                b2: Mat::zeros(1, d),
            })
            .collect();
        let u = Mat::random_normal(d, c.vocab, s, rng);
        Weights {
            e,
            p,
            layers,
            lnf_g: ones(d),
            lnf_b: Mat::zeros(1, d),
            u,
        }
    }

    /// Every weight matrix, in a fixed order: E, P, each block's (LN₁, the
    /// heads' W_q, W_k, W_v, W_o, LN₂, the FFN), LN_f, U.
    pub fn all(&self) -> Vec<&Mat> {
        let mut v = vec![&self.e, &self.p];
        for l in &self.layers {
            v.extend([&l.ln1_g, &l.ln1_b]);
            for h in &l.heads {
                v.extend(h.iter());
            }
            v.extend([&l.wo, &l.ln2_g, &l.ln2_b, &l.w1, &l.b1, &l.w2, &l.b2]);
        }
        v.extend([&self.lnf_g, &self.lnf_b, &self.u]);
        v
    }

    /// Which of the matrices in `all()` the model uses: without the FFN
    /// sublayer, each block's LN₂ and FFN weights exist but never take part.
    pub fn used(&self, ffn: bool) -> Vec<bool> {
        let mut v = vec![true, true];
        for l in &self.layers {
            v.extend([true, true]);
            v.extend(std::iter::repeat_n(true, 3 * l.heads.len()));
            v.push(true); // W_o
            v.extend([ffn; 6]); // LN₂ (2), W₁, b₁, W₂, b₂
        }
        v.extend([true, true, true]);
        v
    }

    pub fn all_mut(&mut self) -> Vec<&mut Mat> {
        let mut v = vec![&mut self.e, &mut self.p];
        for l in &mut self.layers {
            v.extend([&mut l.ln1_g, &mut l.ln1_b]);
            for h in &mut l.heads {
                v.extend(h.iter_mut());
            }
            v.extend([
                &mut l.wo,
                &mut l.ln2_g,
                &mut l.ln2_b,
                &mut l.w1,
                &mut l.b1,
                &mut l.w2,
                &mut l.b2,
            ]);
        }
        v.extend([&mut self.lnf_g, &mut self.lnf_b, &mut self.u]);
        v
    }
}

/// The target at position t for the sum task: (one back + two back) mod V.
pub fn target(tokens: &[usize], t: usize, vocab: usize) -> Option<usize> {
    (t >= 2).then(|| (tokens[t - 1] + tokens[t - 2]) % vocab)
}

/// What to leave out at run time, to see what each part contributes: one
/// block's attention addition, or one block's FFN addition.
#[derive(Clone, Copy, Default)]
pub struct Knockout {
    pub attention: Option<usize>,
    pub ffn: Option<usize>,
}

/// The forward pass, recorded on a tape.
pub struct Run {
    pub tape: Tape,
    pub leaves: Vec<Var>,
    pub loss: Var,
    pub logits: Var,
    /// Each block's heads' attention weights.
    pub attn: Vec<Vec<Var>>,
    /// The residual stream's parts: x (embeddings), and what each block's
    /// attention and FFN added.
    pub x: Var,
    pub attn_out: Vec<Var>,
    pub ffn_out: Vec<Option<Var>>,
}

pub fn run(w: &Weights, c: &Config, ex: &Example, ko: Knockout) -> Run {
    let mut t = Tape::new();
    let leaves: Vec<Var> = w.all().into_iter().map(|m| t.leaf(m.clone())).collect();
    // Walk the leaves in Weights::all() order.
    let mut next = leaves.iter().copied();
    let mut take = || next.next().unwrap();
    let (e, p) = (take(), take());

    // x = E[token] + P[t]: the residual stream starts here.
    let tok = t.gather(e, &ex.tokens);
    let positions: Vec<usize> = (0..ex.tokens.len()).collect();
    let pos = t.gather(p, &positions);
    let x = t.add(tok, pos);

    let mut stream = x;
    let (mut attn, mut attn_out, mut ffn_out) = (Vec::new(), Vec::new(), Vec::new());
    for (li, layer) in w.layers.iter().enumerate() {
        let (ln1_g, ln1_b) = (take(), take());
        let heads: Vec<[Var; 3]> = layer
            .heads
            .iter()
            .map(|_| [take(), take(), take()])
            .collect();
        let (wo, ln2_g, ln2_b, w1, b1, w2, b2) =
            (take(), take(), take(), take(), take(), take(), take());

        // Attention sublayer: h = x + W_o·[heads](LN₁(x))
        let z = t.layer_norm(stream, ln1_g, ln1_b, 1e-5);
        let hs = (c.dim / heads.len()) as f64;
        let mut weights = Vec::new();
        let mut outs = Vec::new();
        for &[wq, wk, wv] in &heads {
            let q = t.matmul(z, wq);
            let k = t.matmul(z, wk);
            let v = t.matmul(z, wv);
            let kt = t.transpose(k);
            let s = t.matmul(q, kt);
            let s = t.scale(s, 1.0 / hs.sqrt());
            let a = t.causal_softmax(s);
            weights.push(a);
            outs.push(t.matmul(a, v));
        }
        let joined = t.concat(&outs);
        let a_out = t.matmul(joined, wo);
        if ko.attention != Some(li) {
            stream = t.add(stream, a_out);
        }
        attn.push(weights);
        attn_out.push(a_out);

        // FFN sublayer: y = h + W₂·GELU(W₁·LN₂(h) + b₁) + b₂
        if c.ffn {
            let z2 = t.layer_norm(stream, ln2_g, ln2_b, 1e-5);
            let a1 = t.matmul(z2, w1);
            let a1 = t.add_row(a1, b1);
            let g = t.gelu(a1);
            let f = t.matmul(g, w2);
            let f = t.add_row(f, b2);
            if ko.ffn != Some(li) {
                stream = t.add(stream, f);
            }
            ffn_out.push(Some(f));
        } else {
            ffn_out.push(None);
        }
    }

    // Output: logits = LN_f(y)·U
    let (lnf_g, lnf_b, u) = (take(), take(), take());
    let zf = t.layer_norm(stream, lnf_g, lnf_b, 1e-5);
    let logits = t.matmul(zf, u);
    let loss = t.softmax_cross_entropy(logits, &ex.targets);
    Run {
        tape: t,
        leaves,
        loss,
        logits,
        attn,
        x,
        attn_out,
        ffn_out,
    }
}

/// The index of the largest entry.
fn argmax(row: &[f64]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

/// Correct predictions at the scored positions, and how many there are.
pub fn score(r: &Run, ex: &Example) -> (usize, usize) {
    let l = r.tape.value(r.logits);
    let (mut correct, mut total) = (0, 0);
    for (t, &counted) in ex.scored.iter().enumerate() {
        if counted {
            correct += usize::from(Some(argmax(l.row(t))) == ex.targets[t]);
            total += 1;
        }
    }
    (correct, total)
}

pub fn evaluate(w: &Weights, c: &Config, rng: &mut Rng, n: usize, ko: Knockout) -> f64 {
    let (mut correct, mut total) = (0, 0);
    for _ in 0..n {
        let ex = c.task.sample(c, rng);
        let (a, b) = score(&run(w, c, &ex, ko), &ex);
        correct += a;
        total += b;
    }
    correct as f64 / total.max(1) as f64
}

pub struct Trainer {
    m: Vec<Mat>,
    v: Vec<Mat>,
    t: i32,
}

impl Trainer {
    pub fn new(w: &Weights) -> Trainer {
        let z: Vec<Mat> = w.all().iter().map(|m| Mat::zeros(m.rows, m.cols)).collect();
        Trainer {
            m: z.clone(),
            v: z,
            t: 0,
        }
    }

    /// One step: gradients from the tape, averaged over a batch, then Adam.
    /// Returns the batch's mean loss and accuracy.
    pub fn step(&mut self, w: &mut Weights, c: &Config, rng: &mut Rng) -> (f64, f64) {
        let mut grad: Vec<Mat> = w.all().iter().map(|m| Mat::zeros(m.rows, m.cols)).collect();
        let (mut loss, mut correct, mut total) = (0.0, 0, 0);
        for _ in 0..c.batch {
            let ex = c.task.sample(c, rng);
            let mut r = run(w, c, &ex, Knockout::default());
            r.tape.backward(r.loss);
            for (g, &leaf) in grad.iter_mut().zip(&r.leaves) {
                g.add_scaled(&r.tape.grad(leaf), 1.0 / c.batch as f64);
            }
            loss += r.tape.value(r.loss).data[0];
            let (a, b) = score(&r, &ex);
            correct += a;
            total += b;
        }
        self.t += 1;
        for (((m, v), w), g) in self
            .m
            .iter_mut()
            .zip(&mut self.v)
            .zip(w.all_mut())
            .zip(&grad)
        {
            adam_update(m, v, w, g, self.t, c.lr);
        }
        (loss / c.batch as f64, correct as f64 / total.max(1) as f64)
    }
}

fn digits(tokens: &[usize]) -> String {
    tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Labels such as "3:7" (position 3, digit 7).
fn labels(tokens: &[usize]) -> Vec<String> {
    tokens
        .iter()
        .enumerate()
        .map(|(t, d)| format!("{}:{d}", t + 1))
        .collect()
}

/// Finite-difference check of the whole block's tape gradients, on a few
/// entries of every weight matrix. Returns the worst relative error.
pub fn gradient_check(w: &Weights, c: &Config, ex: &Example, rng: &mut Rng) -> (f64, usize) {
    let mut r = run(w, c, ex, Knockout::default());
    r.tape.backward(r.loss);
    let grads: Vec<Mat> = r.leaves.iter().map(|&l| r.tape.grad(l)).collect();
    let loss_at = |w: &Weights| {
        let r = run(w, c, ex, Knockout::default());
        r.tape.value(r.loss).data[0]
    };
    let eps = 1e-6;
    let mut probe = w.clone();
    let used = w.used(c.ffn);
    let (mut worst, mut checked): (f64, usize) = (0.0, 0);
    for (k, grad) in grads.iter().enumerate() {
        if !used[k] {
            continue; // not part of this model: its gradient is 0
        }
        for _ in 0..3 {
            let len = probe.all()[k].data.len();
            let i = rng.below(len);
            let orig = probe.all()[k].data[i];
            probe.all_mut()[k].data[i] = orig + eps;
            let up = loss_at(&probe);
            probe.all_mut()[k].data[i] = orig - eps;
            let down = loss_at(&probe);
            probe.all_mut()[k].data[i] = orig;
            let numeric = (up - down) / (2.0 * eps);
            let analytic = grad.data[i];
            let scale = numeric.abs() + analytic.abs();
            if scale > 1e-9 {
                worst = worst.max((numeric - analytic).abs() / scale);
            }
            checked += 1;
        }
    }
    (worst, checked)
}

/// Train quietly; returns the weights and the accuracy on fresh sequences.
pub fn train_quietly(c: &Config, seed: u64) -> (Weights, f64) {
    let c = Config { seed, ..c.clone() };
    let mut rng = Rng::new(seed);
    let mut w = Weights::new(&c, &mut rng);
    let mut tr = Trainer::new(&w);
    for _ in 0..c.steps {
        tr.step(&mut w, &c, &mut rng);
    }
    let acc = evaluate(
        &w,
        &c,
        &mut Rng::new(seed ^ 0xE7A1),
        500,
        Knockout::default(),
    );
    (w, acc)
}

/// Average length of each part of the residual stream, over the rows with
/// a target: the embeddings x, what attention added, what the FFN added.
fn stream_sizes(w: &Weights, c: &Config, rng: &mut Rng, n: usize) -> [f64; 3] {
    let norm = |m: &Mat, t: usize| m.row(t).iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut sizes = [0.0; 3];
    let mut rows = 0;
    for _ in 0..n {
        let ex = c.task.sample(c, rng);
        let r = run(w, c, &ex, Knockout::default());
        for t in 2..ex.tokens.len() {
            sizes[0] += norm(r.tape.value(r.x), t);
            sizes[1] += norm(r.tape.value(r.attn_out[0]), t);
            if let Some(f) = r.ffn_out[0] {
                sizes[2] += norm(r.tape.value(f), t);
            }
            rows += 1;
        }
    }
    sizes.map(|s| s / rows as f64)
}

pub fn run_command(c: &Config, seeds: u64) {
    let d = c.dim;
    let h = d / c.heads;
    println!(
        "=== One complete block: predict (one back + two back) mod {} ===",
        c.vocab
    );
    println!(
        "Random sequences of {} digits, 0–{}. At each position from 3 on, the answer",
        c.len,
        c.vocab - 1
    );
    println!(
        "is the sum of the two previous digits mod {}; chance is 1 in {}.\n",
        c.vocab, c.vocab
    );
    println!("  x      = E[token] + P[t]                    the residual stream starts");
    println!("  h      = x + W_o·[heads](LN₁(x))            attention sublayer");
    println!("  y      = h + W₂·GELU(W₁·LN₂(h) + b₁) + b₂   FFN sublayer");
    println!("  logits = LN_f(y)·U\n");
    println!(
        "d = {d}, {} heads of size {h}, d_ff = {}. Written as forward code only: the",
        c.heads, c.d_ff
    );
    println!("automatic differentiation tape works out every gradient.\n");

    let mut rng = Rng::new(c.seed);
    let mut w = Weights::new(c, &mut rng);
    let n: usize = w.all().iter().map(|m| m.data.len()).sum();
    let (v, t) = (c.vocab, c.len);
    let formula = v * d + t * d + (4 * d * d + 2 * d * c.d_ff + 9 * d + c.d_ff) + 2 * d;
    println!("Parameters: {n}. PLAN.md's formula for one GPT-2 block gives {formula};");
    println!(
        "this block has no attention biases (−4d = −{}) and a separate output layer",
        4 * d
    );
    println!(
        "instead of weight tying (+V·d = +{}): {formula} − {} + {} = {}.\n",
        v * d,
        4 * d,
        v * d,
        formula - 4 * d + v * d
    );

    let ex = c.task.sample(c, &mut rng);
    let demo = ex.tokens.clone();
    let (worst, checked) = gradient_check(&w, c, &ex, &mut Rng::new(c.seed ^ 0x6C));
    println!(
        "Gradient check (tape vs finite differences, {checked} weights in all {} matrices):",
        w.all().len()
    );
    println!(
        "worst relative error {worst:.1e} {}\n",
        if worst < 1e-5 {
            "— OK"
        } else {
            "— WARNING: gradients disagree"
        }
    );

    println!(
        "Training: Adam, learning rate {}, {} sequences per step, seed {}.\n",
        c.lr, c.batch, c.seed
    );
    println!("   step    loss   accuracy");
    let mut tr = Trainer::new(&w);
    let every = (c.steps / 8).max(1);
    let (mut loss_sum, mut acc_sum, mut k) = (0.0, 0.0, 0);
    for step in 1..=c.steps {
        let (loss, acc) = tr.step(&mut w, c, &mut rng);
        loss_sum += loss;
        acc_sum += acc;
        k += 1;
        if step % every == 0 || step == c.steps {
            println!(
                "  {step:>5}   {:.3}   {:>5.1}%",
                loss_sum / k as f64,
                100.0 * acc_sum / k as f64
            );
            (loss_sum, acc_sum, k) = (0.0, 0.0, 0);
        }
    }
    println!("  (averages since the previous line)\n");

    let mut eval = Rng::new(c.seed ^ 0xE7A1);
    let full = evaluate(&w, c, &mut eval, 500, Knockout::default());
    println!("=== After training ===");
    println!("Accuracy on 500 fresh sequences: {:.1}%\n", 100.0 * full);

    let r = run(&w, c, &ex, Knockout::default());
    for (k, &a) in r.attn[0].iter().enumerate() {
        print_weights(
            &format!(
                "head {}: attention on the sequence {}",
                k + 1,
                digits(&demo)
            ),
            r.tape.value(a),
            &labels(&demo),
            2,
        );
    }
    let preds = r.tape.value(r.logits);
    let mut line = String::new();
    for t in 2..demo.len() {
        let row = preds.row(t);
        let best = (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b });
        line += &format!("  {}+{}→{}", demo[t - 2], demo[t - 1], best);
    }
    println!("Its predictions on that sequence (two back + one back → answer):\n{line}\n");

    let sizes = stream_sizes(&w, c, &mut Rng::new(c.seed ^ 0x5E), 200);
    print_table(
        "The residual stream: average length of each part, in rows with a target",
        &Mat::from_rows(&[&sizes]),
        &["length".to_string()],
        &[
            "x (embeddings)".into(),
            "+ attention".into(),
            "+ FFN".into(),
        ],
        2,
    );
    println!("Each sublayer adds its result to the stream rather than replacing it.\n");

    let mut eval = Rng::new(c.seed ^ 0xE7A1);
    let no_attn = evaluate(
        &w,
        c,
        &mut eval,
        500,
        Knockout {
            attention: Some(0),
            ffn: None,
        },
    );
    let mut eval = Rng::new(c.seed ^ 0xE7A1);
    let no_ffn = evaluate(
        &w,
        c,
        &mut eval,
        500,
        Knockout {
            attention: None,
            ffn: Some(0),
        },
    );
    println!("Knocking out one sublayer at test time (its addition to the stream left");
    println!("out; nothing retrained):\n");
    println!("  full block:          {:>5.1}%", 100.0 * full);
    println!("  without attention:   {:>5.1}%", 100.0 * no_attn);
    println!("  without the FFN:     {:>5.1}%", 100.0 * no_ffn);
    println!("  chance:              {:>5.1}%\n", 100.0 / c.vocab as f64);
    println!("Both are needed. Only attention can bring the earlier digits to the");
    println!("current position; the FFN does the arithmetic on them. Attention moves");
    println!("information between tokens, the FFN computes with it.\n");

    compare(c, seeds, full);
}

/// The same task, trained with and without the FFN, over several seeds.
fn compare(c: &Config, seeds: u64, first: f64) {
    println!("=== Trained without an FFN ===");
    println!("The same block with the FFN sublayer removed, trained from scratch, against");
    println!(
        "the full block, over {seeds} seeds (accuracy on 500 fresh sequences after {} steps):\n",
        c.steps
    );
    let no_ffn = Config {
        ffn: false,
        ..c.clone()
    };
    let list: Vec<u64> = (c.seed..c.seed + seeds).collect();
    let results: Vec<(f64, f64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = list
            .iter()
            .map(|&seed| {
                let (c, no_ffn) = (c, &no_ffn);
                scope.spawn(move || {
                    let a = if seed == c.seed {
                        first
                    } else {
                        train_quietly(c, seed).1
                    };
                    (a, train_quietly(no_ffn, seed).1)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!("  seed   full block   without FFN");
    for (seed, (a, b)) in list.iter().zip(&results) {
        println!("  {seed:>4}   {:>8.1}%   {:>9.1}%", 100.0 * a, 100.0 * b);
    }
    let mean = |f: fn(&(f64, f64)) -> f64| {
        100.0 * results.iter().map(f).sum::<f64>() / results.len() as f64
    };
    println!(
        "  mean   {:>8.1}%   {:>9.1}%\n",
        mean(|r| r.0),
        mean(|r| r.1)
    );
    println!("Without an FFN the block still gets partway, often far: attention is itself");
    println!("nonlinear, because its weights depend on which tokens are where, and the");
    println!("final layer norm adds a little more. But it's unreliable. With the FFN the");
    println!("block learns the task every time, and, as the knockout shows, hands the");
    println!("arithmetic to it.");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            vocab: 10,
            len: 8,
            dim: 32,
            layers: 1,
            heads: 2,
            d_ff: 128,
            steps: 1000,
            lr: 0.003,
            batch: 16,
            seed: 1,
            ffn: true,
            task: Task::SumMod,
        }
    }

    #[test]
    fn whole_block_gradients_match_finite_differences() {
        let c = config();
        let mut rng = Rng::new(4);
        let w = Weights::new(&c, &mut rng);
        let ex = c.task.sample(&c, &mut rng);
        let (worst, checked) = gradient_check(&w, &c, &ex, &mut rng);
        assert_eq!(checked, 3 * w.all().len());
        assert!(worst < 1e-5, "{worst}");
    }

    #[test]
    fn parameter_count_matches_the_plan_formula() {
        let c = config();
        let w = Weights::new(&c, &mut Rng::new(1));
        let n: usize = w.all().iter().map(|m| m.data.len()).sum();
        let (v, t, d, f) = (c.vocab, c.len, c.dim, c.d_ff);
        // PLAN.md's formula for one block, minus attention biases, plus a
        // separate output layer.
        let formula = v * d + t * d + (4 * d * d + 2 * d * f + 9 * d + f) + 2 * d;
        assert_eq!(n, formula - 4 * d + v * d);
    }

    #[test]
    fn targets_are_the_sum_of_the_two_previous_digits() {
        let tokens = [7, 4, 9, 1];
        assert_eq!(target(&tokens, 1, 10), None);
        assert_eq!(target(&tokens, 2, 10), Some(1)); // 7 + 4 = 11
        assert_eq!(target(&tokens, 3, 10), Some(3)); // 4 + 9 = 13
    }

    #[test]
    fn the_block_learns_the_task_and_needs_both_sublayers() {
        let c = config();
        let (w, acc) = train_quietly(&c, c.seed);
        assert!(acc > 0.95, "accuracy {acc}");
        let ko =
            |attention, ffn| evaluate(&w, &c, &mut Rng::new(3), 200, Knockout { attention, ffn });
        assert!(ko(Some(0), None) < 0.3, "without attention");
        assert!(ko(None, Some(0)) < 0.4, "without the FFN");
    }
}
