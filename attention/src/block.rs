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
    pub heads: usize,
    pub d_ff: usize,
    pub steps: usize,
    pub lr: f64,
    pub batch: usize,
    pub seed: u64,
    /// Include the FFN sublayer (false for the ablation).
    pub ffn: bool,
}

/// The block's weights, by name.
#[derive(Clone)]
pub struct Weights {
    pub e: Mat,
    pub p: Mat,
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
        Weights {
            e: Mat::random_normal(c.vocab, d, 1.0, rng),
            p: Mat::random_normal(c.len, d, 1.0, rng),
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
            lnf_g: ones(d),
            lnf_b: Mat::zeros(1, d),
            u: Mat::random_normal(d, c.vocab, s, rng),
        }
    }

    pub fn all(&self) -> Vec<&Mat> {
        let mut v = vec![&self.e, &self.p, &self.ln1_g, &self.ln1_b];
        for h in &self.heads {
            v.extend(h.iter());
        }
        v.extend([
            &self.wo,
            &self.ln2_g,
            &self.ln2_b,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
            &self.lnf_g,
            &self.lnf_b,
            &self.u,
        ]);
        v
    }

    pub fn all_mut(&mut self) -> Vec<&mut Mat> {
        let mut v = vec![&mut self.e, &mut self.p, &mut self.ln1_g, &mut self.ln1_b];
        for h in &mut self.heads {
            v.extend(h.iter_mut());
        }
        v.extend([
            &mut self.wo,
            &mut self.ln2_g,
            &mut self.ln2_b,
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
            &mut self.lnf_g,
            &mut self.lnf_b,
            &mut self.u,
        ]);
        v
    }
}

/// The target at position t: (token one back + token two back) mod V.
pub fn target(tokens: &[usize], t: usize, vocab: usize) -> Option<usize> {
    (t >= 2).then(|| (tokens[t - 1] + tokens[t - 2]) % vocab)
}

/// What to leave out at run time, to see what each sublayer contributes.
#[derive(Clone, Copy, Default)]
pub struct Knockout {
    pub attention: bool,
    pub ffn: bool,
}

/// The forward pass, recorded on a tape.
pub struct Run {
    pub tape: Tape,
    pub leaves: Vec<Var>,
    pub loss: Var,
    pub logits: Var,
    /// Each head's attention weights.
    pub attn: Vec<Var>,
    /// The residual stream's parts: x (embeddings), what attention added,
    /// what the FFN added.
    pub x: Var,
    pub attn_out: Var,
    pub ffn_out: Option<Var>,
}

pub fn run(w: &Weights, c: &Config, tokens: &[usize], ko: Knockout) -> Run {
    let mut t = Tape::new();
    let leaves: Vec<Var> = w.all().into_iter().map(|m| t.leaf(m.clone())).collect();
    // The leaves in Weights::all() order.
    let (e, p, ln1_g, ln1_b) = (leaves[0], leaves[1], leaves[2], leaves[3]);
    let nh = w.heads.len();
    let heads: Vec<[Var; 3]> = (0..nh)
        .map(|k| [leaves[4 + 3 * k], leaves[5 + 3 * k], leaves[6 + 3 * k]])
        .collect();
    let r = &leaves[4 + 3 * nh..];
    let (wo, ln2_g, ln2_b, w1, b1, w2, b2, lnf_g, lnf_b, u) =
        (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]);

    // x = E[token] + P[t]: the residual stream starts here.
    let tok = t.gather(e, tokens);
    let positions: Vec<usize> = (0..tokens.len()).collect();
    let pos = t.gather(p, &positions);
    let x = t.add(tok, pos);

    // Attention sublayer: h = x + W_o·[heads](LN₁(x))
    let z = t.layer_norm(x, ln1_g, ln1_b, 1e-5);
    let hs = (c.dim / nh) as f64;
    let mut attn = Vec::new();
    let mut outs = Vec::new();
    for &[wq, wk, wv] in &heads {
        let q = t.matmul(z, wq);
        let k = t.matmul(z, wk);
        let v = t.matmul(z, wv);
        let kt = t.transpose(k);
        let s = t.matmul(q, kt);
        let s = t.scale(s, 1.0 / hs.sqrt());
        let a = t.causal_softmax(s);
        attn.push(a);
        outs.push(t.matmul(a, v));
    }
    let joined = t.concat(&outs);
    let attn_out = t.matmul(joined, wo);
    let h = if ko.attention { x } else { t.add(x, attn_out) };

    // FFN sublayer: y = h + W₂·GELU(W₁·LN₂(h) + b₁) + b₂
    let (y, ffn_out) = if c.ffn {
        let z2 = t.layer_norm(h, ln2_g, ln2_b, 1e-5);
        let a1 = t.matmul(z2, w1);
        let a1 = t.add_row(a1, b1);
        let g = t.gelu(a1);
        let f = t.matmul(g, w2);
        let f = t.add_row(f, b2);
        let y = if ko.ffn { h } else { t.add(h, f) };
        (y, Some(f))
    } else {
        (h, None)
    };

    // Output: logits = LN_f(y)·U
    let zf = t.layer_norm(y, lnf_g, lnf_b, 1e-5);
    let logits = t.matmul(zf, u);
    let targets: Vec<Option<usize>> = (0..tokens.len())
        .map(|i| target(tokens, i, c.vocab))
        .collect();
    let loss = t.softmax_cross_entropy(logits, &targets);
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

pub fn random_tokens(c: &Config, rng: &mut Rng) -> Vec<usize> {
    (0..c.len).map(|_| rng.below(c.vocab)).collect()
}

/// Correct predictions and the number of positions with a target.
pub fn score(r: &Run, tokens: &[usize], c: &Config) -> (usize, usize) {
    let l = r.tape.value(r.logits);
    let mut correct = 0;
    for t in 2..tokens.len() {
        let row = l.row(t);
        let best = (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b });
        correct += usize::from(Some(best) == target(tokens, t, c.vocab));
    }
    (correct, tokens.len() - 2)
}

pub fn evaluate(w: &Weights, c: &Config, rng: &mut Rng, n: usize, ko: Knockout) -> f64 {
    let (mut correct, mut total) = (0, 0);
    for _ in 0..n {
        let tokens = random_tokens(c, rng);
        let (a, b) = score(&run(w, c, &tokens, ko), &tokens, c);
        correct += a;
        total += b;
    }
    correct as f64 / total as f64
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
    pub fn step(&mut self, w: &mut Weights, c: &Config, rng: &mut Rng) -> (f64, f64) {
        let mut grad: Vec<Mat> = w.all().iter().map(|m| Mat::zeros(m.rows, m.cols)).collect();
        let (mut loss, mut correct, mut total) = (0.0, 0, 0);
        for _ in 0..c.batch {
            let tokens = random_tokens(c, rng);
            let mut r = run(w, c, &tokens, Knockout::default());
            r.tape.backward(r.loss);
            for (g, &leaf) in grad.iter_mut().zip(&r.leaves) {
                g.add_scaled(&r.tape.grad(leaf), 1.0 / c.batch as f64);
            }
            loss += r.tape.value(r.loss).data[0];
            let (a, b) = score(&r, &tokens, c);
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
        (loss / c.batch as f64, correct as f64 / total as f64)
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
fn gradient_check(w: &Weights, c: &Config, tokens: &[usize], rng: &mut Rng) -> (f64, usize) {
    let mut r = run(w, c, tokens, Knockout::default());
    r.tape.backward(r.loss);
    let grads: Vec<Mat> = r.leaves.iter().map(|&l| r.tape.grad(l)).collect();
    let loss_at = |w: &Weights| {
        let r = run(w, c, tokens, Knockout::default());
        r.tape.value(r.loss).data[0]
    };
    let eps = 1e-6;
    let mut probe = w.clone();
    let (mut worst, mut checked): (f64, usize) = (0.0, 0);
    for (k, grad) in grads.iter().enumerate() {
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
fn train_quietly(c: &Config, seed: u64) -> (Weights, f64) {
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
        let tokens = random_tokens(c, rng);
        let r = run(w, c, &tokens, Knockout::default());
        for t in 2..tokens.len() {
            sizes[0] += norm(r.tape.value(r.x), t);
            sizes[1] += norm(r.tape.value(r.attn_out), t);
            if let Some(f) = r.ffn_out {
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

    let demo = random_tokens(c, &mut rng);
    let (worst, checked) = gradient_check(&w, c, &demo, &mut Rng::new(c.seed ^ 0x6C));
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

    let r = run(&w, c, &demo, Knockout::default());
    for (k, &a) in r.attn.iter().enumerate() {
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
            attention: true,
            ffn: false,
        },
    );
    let mut eval = Rng::new(c.seed ^ 0xE7A1);
    let no_ffn = evaluate(
        &w,
        c,
        &mut eval,
        500,
        Knockout {
            attention: false,
            ffn: true,
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
            heads: 2,
            d_ff: 128,
            steps: 1000,
            lr: 0.003,
            batch: 16,
            seed: 1,
            ffn: true,
        }
    }

    #[test]
    fn whole_block_gradients_match_finite_differences() {
        let c = config();
        let mut rng = Rng::new(4);
        let w = Weights::new(&c, &mut rng);
        let tokens = random_tokens(&c, &mut rng);
        let (worst, checked) = gradient_check(&w, &c, &tokens, &mut rng);
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
        assert!(ko(true, false) < 0.3, "without attention");
        assert!(ko(false, true) < 0.4, "without the FFN");
    }
}
