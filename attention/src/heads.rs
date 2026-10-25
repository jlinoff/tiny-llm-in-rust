//! Item 4: several heads. The task: at every position, predict the tokens
//! one, two and three back together, as one answer among V³ classes.
//!
//! One head can do this, by splitting its attention unevenly between the
//! positions, so that the blend of their values still says which token came
//! from where. But that solution is slow to find and sometimes not found.
//! Several heads divide the work: each takes one position, crisply. This
//! file trains several heads in detail, shows how they divided the work,
//! then compares one head of the same total size over many seeds.

use crate::matrix::{print_weights, Mat};
use crate::rng::Rng;
use crate::train::{
    argmax, classes, evaluate, forward, gradient_check, labels, letter, position_attention,
    random_tokens, train_step, word, Adam, Config, Params,
};

pub struct Options {
    /// Seeds to compare one head with several, starting from the main seed.
    pub seeds: u64,
}

/// "one back", "two back", or "5 back".
fn back(lag: usize) -> String {
    match lag {
        1 => "one back".into(),
        2 => "two back".into(),
        3 => "three back".into(),
        n => format!("{n} back"),
    }
}

fn lag_list(lags: &[usize]) -> String {
    let v: Vec<String> = lags.iter().map(|&l| back(l)).collect();
    v.join(", ")
}

pub fn run(c: &Config, o: &Options) {
    let lo = *c.lags.iter().max().unwrap();
    let n_classes = classes(c);
    println!(
        "=== Several heads: predict the tokens {} ===",
        lag_list(&c.lags)
    );
    println!(
        "Random sequences of {} tokens over {} symbols (a–{}); d = {}.",
        c.len,
        c.vocab,
        letter(c.vocab - 1),
        c.dim
    );
    println!(
        "Each answer is the {} tokens together, so there are {}^{} = {} possible",
        c.lags.len(),
        c.vocab,
        c.lags.len(),
        n_classes
    );
    println!(
        "answers; chance-level loss is ln {} = {:.3}. Rows 1–{lo} have no target.",
        n_classes,
        (n_classes as f64).ln()
    );
    println!(
        "Model: token + position embedding → {} heads of size {} → their outputs",
        c.heads, c.head_size
    );
    println!("side by side → output layer.");
    println!(
        "Adam, learning rate {}, {} sequences per step, {} steps, seed {}.\n",
        c.lr, c.batch, c.steps, c.seed
    );

    let mut rng = Rng::new(c.seed);
    let mut w = Params::new(c, &mut rng);
    let demo = random_tokens(c, &mut rng);
    let demo_labels = labels(&demo);

    // Its own random numbers, so training matches train_quietly's exactly.
    let worst = gradient_check(&w, &demo, &c.lags, &mut Rng::new(c.seed ^ 0x6C));
    println!(
        "Gradient check ({} weights, including each head's): worst relative error \
         {worst:.1e} {}\n",
        5 * (3 + 3 * c.heads),
        if worst < 1e-5 {
            "— OK"
        } else {
            "— WARNING: gradients disagree"
        }
    );

    let snapshot = |w: &Params, step: usize, loss: f64, acc: f64| {
        println!(
            "--- step {step}: loss {loss:.3}, accuracy {:.0}%  — attention on the sequence {} ---\n",
            100.0 * acc,
            word(&demo)
        );
        let f = forward(w, &demo, &c.lags);
        for (k, a) in f.attn.iter().enumerate() {
            print_weights(&format!("head {}", k + 1), &a.weights, &demo_labels, 2);
        }
    };

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
    division_of_labor(&w, c);
    compare(c, o);
}

/// How the heads divided the work: where each looks (QK, from the position
/// embeddings alone) and which part of the answer each one's values push
/// toward (OV).
fn division_of_labor(w: &Params, c: &Config) {
    let lo = *c.lags.iter().max().unwrap();
    let rows = c.len - lo;
    println!("How the heads divided the work:\n");
    for (k, hw) in w.heads.iter().enumerate() {
        // Where it looks: share of each row's position-only attention that
        // falls on each lag, averaged over the rows with a target.
        let (pa, _) = position_attention(w, hw);
        let share: Vec<f64> = c
            .lags
            .iter()
            .map(|&lag| (lo..c.len).map(|t| pa.get(t, t - lag)).sum::<f64>() / rows as f64)
            .collect();
        let looks = argmax(&share);

        // What it passes on: send each token's embedding through this head's
        // W_v and its rows of U, and see which part of the answer the scores
        // depend on. For each part (slot), average the scores over the
        // answers with each token in that slot; the slot whose averages vary
        // most with the token is the one this head informs.
        let hs = hw.wv.cols;
        let u_k = rows_of(&w.u, k * hs, hs);
        let ov = w.e.matmul(&hw.wv).matmul(&u_k); // vocab × classes
        let slots = c.lags.len();
        let mut slot_spread = vec![0.0; slots];
        let mut slot_copies = vec![0; slots];
        for (s, (spread, copies)) in slot_spread.iter_mut().zip(&mut slot_copies).enumerate() {
            let place = c.vocab.pow((slots - 1 - s) as u32);
            for tok in 0..c.vocab {
                // Mean score for each letter in slot s.
                let mut by_letter = vec![0.0; c.vocab];
                for (class, &v) in ov.row(tok).iter().enumerate() {
                    by_letter[(class / place) % c.vocab] += v;
                }
                let (lo_v, hi_v) = by_letter
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                        (a.min(v), b.max(v))
                    });
                *spread += (hi_v - lo_v) / (c.vocab * c.vocab) as f64;
                *copies += usize::from(argmax(&by_letter) == tok);
            }
        }
        let informs = argmax(&slot_spread);
        println!(
            "  head {}: looks {} ({:.2} of its weight, from positions alone); its values",
            k + 1,
            back(c.lags[looks]),
            share[looks]
        );
        println!(
            "          push toward answers whose {} token is the one it saw ({} of {} tokens)",
            back(c.lags[informs]),
            slot_copies[informs],
            c.vocab
        );
    }
    println!("\nEach head finds one position and passes that token on: where to look");
    println!("(QK) and what to pass on (OV) line up head by head.\n");
}

/// Rows [from, from + n) of m.
fn rows_of(m: &Mat, from: usize, n: usize) -> Mat {
    let mut out = Mat::zeros(n, m.cols);
    for i in 0..n {
        out.row_mut(i).copy_from_slice(m.row(from + i));
    }
    out
}

/// The training result for one seed: final accuracy, and the first step at
/// which accuracy on fresh sequences reached 99% (checked every 25 steps).
#[derive(Clone, Copy)]
struct Outcome {
    accuracy: f64,
    reached: Option<usize>,
}

/// Train with this seed, quietly. Returns the weights, the outcome and the
/// demo sequence that run() would show for this seed.
fn train_quietly(c: &Config, seed: u64) -> (Params, Outcome, Vec<usize>) {
    let c = Config {
        seed,
        show: vec![],
        lags: c.lags.clone(),
        ..*c
    };
    let mut rng = Rng::new(seed);
    let mut w = Params::new(&c, &mut rng);
    let demo = random_tokens(&c, &mut rng); // drawn as in run()
    let mut adam = Adam::new(&w);
    let mut check = Rng::new(seed ^ 0x99);
    let mut reached = None;
    for step in 1..=c.steps {
        train_step(&mut w, &mut adam, &c, &mut rng);
        if reached.is_none() && step % 25 == 0 && evaluate(&w, &c, &mut check, 50) >= 0.99 {
            reached = Some(step);
        }
    }
    let accuracy = evaluate(&w, &c, &mut Rng::new(seed ^ 0xE7A1), 500);
    (w, Outcome { accuracy, reached }, demo)
}

/// One head with the same total size, on the same task: first in detail for
/// the main seed, then both models over many seeds.
fn compare(c: &Config, o: &Options) {
    let one = Config {
        heads: 1,
        head_size: c.heads * c.head_size,
        show: vec![],
        lags: c.lags.clone(),
        ..*c
    };
    println!(
        "=== The same task with one head of size {} ===",
        one.head_size
    );
    println!(
        "Same total size ({} × {} = {}), same seed, same training.\n",
        c.heads, c.head_size, one.head_size
    );
    let (w1, out1, demo) = train_quietly(&one, c.seed);
    let f = forward(&w1, &demo, &one.lags);
    print_weights(
        &format!(
            "one head, after {} steps: accuracy {:.1}%  — attention on the sequence {}",
            c.steps,
            100.0 * out1.accuracy,
            word(&demo)
        ),
        &f.attn[0].weights,
        &labels(&demo),
        2,
    );
    let lo = *c.lags.iter().max().unwrap();
    let (pa, _) = position_attention(&w1, &w1.heads[0]);
    let split: Vec<String> = c
        .lags
        .iter()
        .map(|&lag| {
            let share = (lo..c.len).map(|t| pa.get(t, t - lag)).sum::<f64>() / (c.len - lo) as f64;
            format!("{share:.2} {}", back(lag))
        })
        .collect();
    println!("A single head can still solve this, by splitting each row's attention");
    println!("unevenly between the positions it needs, so that the blend of their");
    println!("values still says which token came from where. On average (from positions");
    println!("alone) this head puts {}.", split.join(", "));
    println!("The pattern is a compromise: slower to find, and harder to read.");
    if out1.accuracy < 0.99 {
        println!("Here it didn't fully get there: look for a row that put all its weight");
        println!("on one position and so lost the other token. Training doesn't recover");
        println!("from that: the run is stuck in a local minimum.");
    }

    println!(
        "\n=== One head vs {} heads over {} seeds ===",
        c.heads, o.seeds
    );
    println!(
        "Accuracy on 500 fresh sequences after {} steps, and the first step at",
        c.steps
    );
    println!("which accuracy on 50 fresh sequences reached 99% (checked every 25 steps;");
    println!("\"never\" if it didn't). The 50-sequence check is noisier, so a run can pass");
    println!("it and still end just under 99% on the 500.\n");
    println!(
        "  seed   1 head of size {:<3}        {} heads of size {}",
        one.head_size, c.heads, c.head_size
    );
    let fmt = |o: &Outcome| {
        format!(
            "{:5.1}%  {:>5}",
            100.0 * o.accuracy,
            o.reached.map_or("never".to_string(), |s| s.to_string())
        )
    };
    // The seeds are independent, so train them in parallel; each result
    // depends only on its seed, never on the order the threads run in.
    let seeds: Vec<u64> = (c.seed..c.seed + o.seeds).collect();
    let results: Vec<(Outcome, Outcome)> = std::thread::scope(|scope| {
        let handles: Vec<_> = seeds
            .iter()
            .map(|&seed| {
                let (one, c) = (&one, c);
                scope.spawn(move || {
                    let a = if seed == c.seed {
                        out1
                    } else {
                        train_quietly(one, seed).1
                    };
                    (a, train_quietly(c, seed).1)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut solved1, mut solved_n) = (0, 0);
    let (mut sum1, mut sum_n, mut n1, mut nn) = (0, 0, 0, 0);
    for (&seed, &(a, b)) in seeds.iter().zip(&results) {
        println!("  {seed:>4}   {}               {}", fmt(&a), fmt(&b));
        for (o, solved, sum, n) in [
            (&a, &mut solved1, &mut sum1, &mut n1),
            (&b, &mut solved_n, &mut sum_n, &mut nn),
        ] {
            if o.accuracy >= 0.99 {
                *solved += 1;
            }
            if let Some(s) = o.reached {
                *sum += s;
                *n += 1;
            }
        }
    }
    let mean = |sum: usize, n: usize| {
        if n == 0 {
            "—".to_string()
        } else {
            format!("{:.0}", sum as f64 / n as f64)
        }
    };
    println!(
        "\n  solved (≥ 99%): one head {solved1} of {s}, {h} heads {solved_n} of {s}",
        s = o.seeds,
        h = c.heads
    );
    println!(
        "  mean steps to 99%, where reached: one head {}, {} heads {}",
        mean(sum1, n1),
        c.heads,
        mean(sum_n, nn)
    );
    println!("\nSeveral heads don't add a capability one head entirely lacks here: a");
    println!("single head can blend the lookups. What they add is a solution that is");
    println!("easy to find: each head does one lookup at full strength, instead of one");
    println!("head balancing them all. In the words of the transformer paper: \"with a");
    println!("single attention head, averaging inhibits this.\"");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::Config;

    fn config() -> Config {
        Config {
            vocab: 4,
            len: 8,
            dim: 18,
            head_size: 6,
            heads: 3,
            lags: vec![1, 2, 3],
            steps: 400,
            lr: 0.003,
            batch: 16,
            seed: 1,
            show: vec![],
        }
    }

    #[test]
    fn backpropagation_with_several_heads_matches_finite_differences() {
        let c = config();
        let mut rng = Rng::new(3);
        let w = Params::new(&c, &mut rng);
        let tokens = random_tokens(&c, &mut rng);
        assert!(gradient_check(&w, &tokens, &c.lags, &mut rng) < 1e-5);
    }

    #[test]
    fn three_heads_learn_one_lag_each() {
        let c = config();
        let (w, out, _) = train_quietly(&c, c.seed);
        assert!(out.accuracy > 0.99, "accuracy {}", out.accuracy);
        // Each head's position-only attention favors a different lag.
        let mut taken: Vec<usize> = w
            .heads
            .iter()
            .map(|hw| {
                let (pa, _) = position_attention(&w, hw);
                let share: Vec<f64> = c
                    .lags
                    .iter()
                    .map(|&lag| (3..c.len).map(|t| pa.get(t, t - lag)).sum::<f64>())
                    .collect();
                c.lags[argmax(&share)]
            })
            .collect();
        taken.sort();
        assert_eq!(taken, c.lags);
    }
}
