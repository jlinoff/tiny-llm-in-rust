//! Item 6: two blocks forming an induction head.
//!
//! The task is plain next-token prediction on sequences like
//!
//!   filler  A  A  filler        A = k distinct random letters, k from 4 to 7
//!
//! The model is never told about the repeat. Inside the second copy of A,
//! the next letter is predictable, but only by finding where the current
//! letter appeared before and copying what came after it. A starts at a
//! random point, so position alone can't find the earlier copy: the search
//! has to be by content. That takes two blocks:
//!
//!   block 1: a previous-token head writes "the letter before me was X" into
//!            each position's vector (the residual stream)
//!   block 2: an induction head at a letter X looks for the position whose
//!            "letter before me" is X, i.e. the one right after the earlier
//!            X, and copies its letter forward
//!
//! By default the blocks are attention-only (no FFN): the task needs
//! copying, not computing, and with FFNs the circuit forms more slowly.

use crate::block::{gradient_check, run, score, Config, Example, Knockout, Trainer, Weights};
use crate::matrix::print_weights;
use crate::rng::Rng;

pub struct Options {
    /// Seeds in the comparison over seeds, starting from the main seed.
    pub seeds: u64,
}

fn letter(t: usize) -> char {
    (b'a' + t as u8) as char
}

fn word(tokens: &[usize]) -> String {
    tokens.iter().map(|&t| letter(t)).collect()
}

fn labels(tokens: &[usize]) -> Vec<String> {
    tokens
        .iter()
        .enumerate()
        .map(|(i, &t)| format!("{}:{}", i + 1, letter(t)))
        .collect()
}

/// Where the repeat is, from the scored positions: (start of the first copy
/// of A, k). The scored rows are the second copy's positions second … second
/// + k − 2.
fn repeat_of(ex: &Example) -> (usize, usize) {
    let second = ex.scored.iter().position(|&s| s).unwrap();
    let k = ex.scored.iter().filter(|&&s| s).count() + 1;
    (second - k, k)
}

/// Accuracy on the repeated part over n fresh sequences.
fn evaluate(w: &Weights, c: &Config, rng: &mut Rng, n: usize, ko: Knockout) -> f64 {
    let (mut correct, mut total) = (0, 0);
    for _ in 0..n {
        let ex = c.task.sample(c, rng);
        let (a, b) = score(&run(w, c, &ex, ko), &ex);
        correct += a;
        total += b;
    }
    correct as f64 / total.max(1) as f64
}

/// The training result for one seed: final accuracy, and the first window
/// of 250 steps whose average accuracy reached 50% (the phase change).
#[derive(Clone, Copy)]
struct Outcome {
    accuracy: f64,
    phase_change: Option<usize>,
}

const WINDOW: usize = 250;

/// Train; call `report(step, mean loss, mean accuracy)` after every window
/// of 250 steps. Returns the weights and the outcome.
fn train(c: &Config, mut report: impl FnMut(usize, f64, f64)) -> (Weights, Outcome) {
    let mut rng = Rng::new(c.seed);
    let mut w = Weights::new(c, &mut rng);
    let mut tr = Trainer::new(&w);
    let (mut loss_sum, mut acc_sum) = (0.0, 0.0);
    let mut phase_change = None;
    for step in 1..=c.steps {
        let (loss, acc) = tr.step(&mut w, c, &mut rng);
        loss_sum += loss;
        acc_sum += acc;
        if step % WINDOW == 0 {
            let (l, a) = (loss_sum / WINDOW as f64, acc_sum / WINDOW as f64);
            if phase_change.is_none() && a >= 0.5 {
                phase_change = Some(step);
            }
            report(step, l, a);
            (loss_sum, acc_sum) = (0.0, 0.0);
        }
    }
    let accuracy = evaluate(
        &w,
        c,
        &mut Rng::new(c.seed ^ 0xE7A1),
        500,
        Knockout::default(),
    );
    (
        w,
        Outcome {
            accuracy,
            phase_change,
        },
    )
}

pub fn run_command(c: &Config, o: &Options) {
    let (lo, hi) = match c.task {
        crate::block::Task::Induction { min, max } => (min, max),
        _ => unreachable!("induction runs the induction task"),
    };
    let d = c.dim;
    println!("=== Two blocks forming an induction head ===");
    println!(
        "Next-token prediction on {} letters (a–{}), in sequences of {} like",
        c.vocab,
        letter(c.vocab - 1),
        c.len
    );
    println!("\n    filler  A  A  filler\n");
    println!("where A is {lo}–{hi} distinct random letters and A A starts at a random point.");
    println!("The model is never told about the repeat. Inside the second copy of A the");
    println!("next letter is predictable, but only by finding where the current letter");
    println!("appeared before and copying what came after it. Accuracy is measured there;");
    println!(
        "elsewhere the next letter is random (chance is 1 in {} = {:.1}%).\n",
        c.vocab,
        100.0 / c.vocab as f64
    );
    println!(
        "Model: {} blocks, {} head{} each (d = {d}), {}.",
        c.layers,
        c.heads,
        if c.heads == 1 { "" } else { "s" },
        if c.ffn {
            "with FFNs"
        } else {
            "attention only (no FFN)"
        }
    );

    let mut rng = Rng::new(c.seed);
    let w0 = Weights::new(c, &mut rng);
    let used = w0.used(c.ffn);
    let n: usize = w0
        .all()
        .iter()
        .zip(&used)
        .filter(|(_, &u)| u)
        .map(|(m, _)| m.data.len())
        .sum();
    let per_block = 4 * d * d
        + 2 * d
        + if c.ffn {
            2 * d * c.d_ff + c.d_ff + 3 * d
        } else {
            0
        };
    println!(
        "Parameters: {n} = embeddings {} + {} blocks × {per_block} + final layer norm {} + output layer {}.",
        c.vocab * d + c.len * d,
        c.layers,
        2 * d,
        d * c.vocab
    );
    let ex = c.task.sample(c, &mut Rng::new(c.seed ^ 0xD3));
    let (worst, checked) = gradient_check(&w0, c, &ex, &mut Rng::new(c.seed ^ 0x6C));
    println!(
        "Gradient check (tape vs finite differences, {checked} weights): worst relative error {worst:.1e} {}\n",
        if worst < 1e-5 { "— OK" } else { "— WARNING: gradients disagree" }
    );

    println!(
        "Training: Adam, learning rate {}, {} sequences per step, seed {}.\n",
        c.lr, c.batch, c.seed
    );
    println!("   step    loss   accuracy on the repeat");
    let mut marked = false;
    let (w, out) = train(c, |step, loss, acc| {
        let note = if !marked && acc >= 0.5 {
            marked = true;
            "   ← the jump: the induction head has formed"
        } else {
            ""
        };
        println!("  {step:>5}   {loss:.3}   {:>5.1}%{note}", 100.0 * acc);
    });
    println!("  (averages over each {WINDOW} steps)\n");
    println!("The loss falls only slightly for a long time: the plateau. Then accuracy");
    println!("jumps within a few hundred steps, the \"phase change\" seen when induction");
    println!("heads form in real models (Olsson et al., 2022).\n");

    println!("=== After training ===");
    println!(
        "Accuracy on the repeat, 500 fresh sequences: {:.1}%\n",
        100.0 * out.accuracy
    );
    heads_explained(&w, c);
    knockouts(&w, c, out.accuracy);
    compare(c, o, out);
}

/// What each head does: averages over many sequences, then one example.
fn heads_explained(w: &Weights, c: &Config) {
    let mut rng = Rng::new(c.seed ^ 0xA77);
    let (mut prev, mut even, mut prev_rows) = (0.0, 0.0, 0);
    let (mut ind, mut ind_rows) = (0.0, 0);
    for _ in 0..300 {
        let ex = c.task.sample(c, &mut rng);
        let (start, k) = repeat_of(&ex);
        let r = run(w, c, &ex, Knockout::default());
        // Block 1: weight on the previous token, over the rows of A A.
        let a1 = r.tape.value(r.attn[0][0]);
        for t in start + 1..start + 2 * k {
            prev += a1.get(t, t - 1);
            even += 1.0 / (t + 1) as f64; // row t spread evenly over t + 1 places
            prev_rows += 1;
        }
        // Last block: on the scored rows, weight on the position right after
        // the earlier copy of the current letter: t − k + 1.
        let a2 = r.tape.value(r.attn[c.layers - 1][0]);
        for (t, &s) in ex.scored.iter().enumerate() {
            if s {
                ind += a2.get(t, t + 1 - k);
                ind_rows += 1;
            }
        }
    }
    let prev = prev / prev_rows as f64;
    let even = even / prev_rows as f64;
    let ind = ind / ind_rows as f64;
    println!("What the heads do, averaged over 300 fresh sequences:\n");
    println!("  block 1's head puts {prev:.2} of its weight on the previous letter, in the rows");
    println!("  of A A (spreading evenly would give {even:.2})");
    println!(
        "  block {}'s head puts {ind:.2} of its weight on the letter right after the earlier",
        c.layers
    );
    println!("  copy of the current one (rows inside the second copy)\n");

    let ex = c.task.sample(c, &mut Rng::new(c.seed ^ 0xBEE));
    let (start, k) = repeat_of(&ex);
    let r = run(w, c, &ex, Knockout::default());
    println!(
        "For example, the sequence {}: A = {} (positions {}–{}), repeated at {}–{}.\n",
        word(&ex.tokens),
        word(&ex.tokens[start..start + k]),
        start + 1,
        start + k,
        start + k + 1,
        start + 2 * k
    );
    for (li, heads) in r.attn.iter().enumerate() {
        for (hi, &a) in heads.iter().enumerate() {
            let name = if c.heads == 1 {
                format!("block {}'s head", li + 1)
            } else {
                format!("block {}, head {}", li + 1, hi + 1)
            };
            print_weights(&name, r.tape.value(a), &labels(&ex.tokens), 2);
        }
    }
    let l = r.tape.value(r.logits);
    let mut line = String::new();
    for (t, &s) in ex.scored.iter().enumerate() {
        if s {
            let row = l.row(t);
            let best = (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b });
            let ok = Some(best) == ex.targets[t];
            line += &format!(
                "  {}:{}→{}{}",
                t + 1,
                letter(ex.tokens[t]),
                letter(best),
                if ok { "✓" } else { "✗" }
            );
        }
    }
    println!("Predictions in the second copy (position:letter → predicted next letter):");
    println!("{line}\n");
    println!(
        "Read block {}'s rows inside the second copy: each one's weight sits on the",
        c.layers
    );
    println!("position just after the earlier copy of its own letter, a stripe parallel");
    println!("to the diagonal, k − 1 places to its left. Rows with nothing to find park");
    println!("their weight on the first positions. Block 1's head is looser than a textbook");
    println!("previous-token head: it puts the most weight on the previous letter and less");
    println!("on each one before, so each position's vector carries a blurred record of");
    println!("the letters just before it, sharpest for the one right before. That is");
    println!("enough for block 2 to match on, as the knockouts show.\n");
}

fn knockouts(w: &Weights, c: &Config, full: f64) {
    println!("Knocking out one head at test time (its block's attention addition left");
    println!("out; nothing retrained):\n");
    println!("  both heads:            {:>5.1}%", 100.0 * full);
    for li in 0..c.layers {
        let acc = evaluate(
            w,
            c,
            &mut Rng::new(c.seed ^ 0xE7A1),
            500,
            Knockout {
                attention: Some(li),
                ffn: None,
            },
        );
        println!("  without block {}'s head: {:>5.1}%", li + 1, 100.0 * acc);
    }
    println!(
        "  chance:                {:>5.1}%\n",
        100.0 / c.vocab as f64
    );
    println!("Both are needed: block 2's search works only on what block 1 wrote into the");
    println!("residual stream. This is composition: one layer's output becomes the next");
    println!("layer's key.\n");
}

/// One block with the same number of heads, and two blocks over several seeds.
fn compare(c: &Config, o: &Options, first: Outcome) {
    let one = Config {
        layers: 1,
        heads: c.heads * c.layers,
        ..c.clone()
    };
    let seeds: Vec<u64> = (c.seed + 1..c.seed + o.seeds).collect();
    let (one_out, others): (Outcome, Vec<Outcome>) = std::thread::scope(|scope| {
        let h1 = scope.spawn(|| train(&one, |_, _, _| {}).1);
        let hs: Vec<_> = seeds
            .iter()
            .map(|&seed| {
                let cs = Config { seed, ..c.clone() };
                scope.spawn(move || train(&cs, |_, _, _| {}).1)
            })
            .collect();
        (
            h1.join().unwrap(),
            hs.into_iter().map(|h| h.join().unwrap()).collect(),
        )
    });

    println!("=== One block with {} heads instead ===", one.heads);
    println!(
        "Same task, same training, the same number of heads in a single block: {:.1}%",
        100.0 * one_out.accuracy
    );
    println!(
        "on the repeat (chance {:.1}%, two blocks {:.1}%). A single layer can match the",
        100.0 / c.vocab as f64,
        100.0 * first.accuracy
    );
    println!("current letter against earlier letters, but what it finds is the earlier");
    println!("copy itself, not the letter after it: no position yet holds \"the letter");
    println!("before me\". That takes one layer to write and a second layer to read.\n");

    println!("=== Two blocks over {} seeds ===", o.seeds);
    println!("When the jump happens varies a lot (first {WINDOW}-step window averaging ≥ 50%):\n");
    println!("  seed   accuracy   jump by step");
    let fmt = |o: &Outcome| o.phase_change.map_or("none".to_string(), |s| s.to_string());
    println!(
        "  {:>4}   {:>7.1}%   {:>8}",
        c.seed,
        100.0 * first.accuracy,
        fmt(&first)
    );
    for (seed, out) in seeds.iter().zip(&others) {
        println!(
            "  {seed:>4}   {:>7.1}%   {:>8}",
            100.0 * out.accuracy,
            fmt(out)
        );
    }
    println!("\nThe plateau's length depends on the seed; the jump, when it comes, is fast.");
    println!("A run with no jump may simply need more steps (--steps).");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Task;

    fn config() -> Config {
        Config {
            vocab: 16,
            len: 20,
            dim: 32,
            layers: 2,
            heads: 1,
            d_ff: 128,
            steps: 4000,
            lr: 0.005,
            batch: 16,
            seed: 1,
            ffn: false,
            task: Task::Induction { min: 4, max: 7 },
        }
    }

    #[test]
    fn sequences_are_filler_a_a_filler() {
        let c = config();
        let mut rng = Rng::new(8);
        for _ in 0..200 {
            let ex = c.task.sample(&c, &mut rng);
            let (start, k) = repeat_of(&ex);
            assert!((4..=7).contains(&k));
            let a = &ex.tokens[start..start + k];
            // A has distinct letters, and is repeated right after itself.
            for (i, x) in a.iter().enumerate() {
                assert!(!a[i + 1..].contains(x), "A repeats a letter");
            }
            assert_eq!(a, &ex.tokens[start + k..start + 2 * k]);
            // Filler never uses A's letters.
            for (t, x) in ex.tokens.iter().enumerate() {
                if t < start || t >= start + 2 * k {
                    assert!(!a.contains(x), "filler uses a letter of A");
                }
            }
            // Targets are the next token; the scored ones are predictable
            // from the first copy.
            for t in 0..c.len - 1 {
                assert_eq!(ex.targets[t], Some(ex.tokens[t + 1]));
                if ex.scored[t] {
                    assert_eq!(ex.targets[t], Some(ex.tokens[t + 1 - k]));
                }
            }
            assert_eq!(ex.targets[c.len - 1], None);
        }
    }

    #[test]
    fn two_block_gradients_match_finite_differences() {
        let c = config();
        let mut rng = Rng::new(6);
        let w = Weights::new(&c, &mut rng);
        let ex = c.task.sample(&c, &mut rng);
        let (worst, checked) = gradient_check(&w, &c, &ex, &mut rng);
        // Only the weights in use: no LN₂ or FFN in attention-only blocks.
        let used = w.used(c.ffn).iter().filter(|&&u| u).count();
        assert_eq!(checked, 3 * used);
        assert!(worst < 1e-5, "{worst}");
    }

    #[test]
    fn with_ffns_every_weight_is_checked() {
        let c = Config {
            ffn: true,
            ..config()
        };
        let mut rng = Rng::new(6);
        let w = Weights::new(&c, &mut rng);
        let ex = c.task.sample(&c, &mut rng);
        let (worst, checked) = gradient_check(&w, &c, &ex, &mut rng);
        assert_eq!(checked, 3 * w.all().len());
        assert!(worst < 1e-5, "{worst}");
    }

    /// Slow (about 20 seconds in release): run with
    /// `cargo test --release -- --ignored`.
    #[test]
    #[ignore]
    fn two_blocks_form_an_induction_head_and_one_block_cannot() {
        let c = config();
        let (w, out) = train(&c, |_, _, _| {});
        assert!(out.accuracy > 0.9, "two blocks: {}", out.accuracy);
        assert!(out.phase_change.is_some());
        let ko = |li| {
            evaluate(
                &w,
                &c,
                &mut Rng::new(3),
                200,
                Knockout {
                    attention: Some(li),
                    ffn: None,
                },
            )
        };
        assert!(ko(0) < 0.3 && ko(1) < 0.3, "knockouts");
        let one = Config {
            layers: 1,
            heads: 2,
            ..c
        };
        let (_, out1) = train(&one, |_, _, _| {});
        assert!(out1.accuracy < 0.3, "one block: {}", out1.accuracy);
    }
}
