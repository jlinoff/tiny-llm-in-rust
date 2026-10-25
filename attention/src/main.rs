//! mini_attention: self-attention, small enough to print every number.
//!
//! Subcommands, in the order PLAN.md suggests working through them:
//!   example   the worked example from PLAN.md, with hand-set weights
//!   random    an untrained head: random weights, near-uniform attention
//!   train     train one head on a task with a known answer, and watch it learn

mod attention;
mod matrix;
mod rng;
mod train;

use clap::{Parser, Subcommand};

use attention::{layer_norm_rows, Head};
use matrix::{index_labels, print_table, print_weights, Mat};
use rng::Rng;

#[derive(Parser)]
#[command(
    name = "mini_attention",
    version,
    about = "Self-attention, small enough to print every number.",
    after_help = "Start with `mini_attention example`, then `random`, then `train`."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// The worked example from PLAN.md: 3 tokens, d = 4, one head of size 2,
    /// hand-set weights. Prints every step.
    Example,
    /// An untrained head: random embeddings and random weights. Shows that
    /// small random weights give near-uniform attention.
    Random {
        /// The text to attend over, one token per word
        #[arg(long, default_value = "the cat sat on the mat")]
        text: String,
        /// Vector width d
        #[arg(short, long, default_value_t = 16)]
        dim: usize,
        /// Head size [default: d]
        #[arg(long)]
        head_size: Option<usize>,
        /// Standard deviation of the random weights (GPT-2 used 0.02)
        #[arg(long, default_value_t = 0.02)]
        std: f64,
        /// Random seed
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Also print Q, K, V and the outputs
        #[arg(short, long)]
        verbose: bool,
    },
    /// Train one head to predict the token LAG positions back, and watch its
    /// attention move from uniform to the right pattern.
    Train {
        /// How far back the token to predict is
        #[arg(long, default_value_t = 2)]
        lag: usize,
        /// Training steps
        #[arg(long, default_value_t = 400)]
        steps: usize,
        /// Adam learning rate
        #[arg(long, default_value_t = 0.001)]
        lr: f64,
        /// Sequences per step
        #[arg(long, default_value_t = 16)]
        batch: usize,
        /// Number of distinct tokens (2–26, shown as letters)
        #[arg(long, default_value_t = 8)]
        vocab: usize,
        /// Sequence length
        #[arg(long, default_value_t = 8)]
        len: usize,
        /// Vector width d
        #[arg(short, long, default_value_t = 16)]
        dim: usize,
        /// Head size [default: d]
        #[arg(long)]
        head_size: Option<usize>,
        /// Random seed
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Steps at which to print the attention, comma-separated
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "0,50,100,150,200,300,400"
        )]
        show: Vec<usize>,
    },
}

fn main() {
    match Cli::parse().command {
        Command::Example => example(),
        Command::Random {
            text,
            dim,
            head_size,
            std,
            seed,
            verbose,
        } => random(&text, dim, head_size.unwrap_or(dim), std, seed, verbose),
        Command::Train {
            lag,
            steps,
            lr,
            batch,
            vocab,
            len,
            dim,
            head_size,
            seed,
            show,
        } => {
            let c = train::Config {
                vocab,
                len,
                dim,
                head_size: head_size.unwrap_or(dim),
                lag,
                steps,
                lr,
                batch,
                seed,
                show,
            };
            if !(2..=26).contains(&c.vocab)
                || c.lag == 0
                || c.lag >= c.len
                || c.dim == 0
                || c.head_size == 0
                || c.batch == 0
                || c.lr.is_nan()
                || c.lr <= 0.0
            {
                eprintln!(
                    "mini_attention: error: need --vocab 2–26, 1 ≤ --lag < --len, \
                     and positive --dim, --head-size, --batch and --lr"
                );
                std::process::exit(2);
            }
            train::run(&c);
        }
    }
}

fn labels(words: &[&str]) -> Vec<String> {
    words.iter().map(|w| w.to_string()).collect()
}

/// The worked example from PLAN.md, section "How attention works".
fn example() {
    let tokens = labels(&["the", "cat", "sat"]);
    let dims = index_labels(4);
    let hcols = index_labels(2);
    let x = Mat::from_rows(&[&[1., 0., 1., 0.], &[0., 1., 0., 1.], &[1., 1., 0., 0.]]);
    let head = Head {
        wq: Mat::from_rows(&[&[1., 0.], &[0., 1.], &[1., 0.], &[0., 0.]]),
        wk: Mat::from_rows(&[&[0., 1.], &[1., 0.], &[0., 0.], &[1., 1.]]),
        wv: Mat::from_rows(&[&[1., 0.], &[0., 1.], &[0., 0.], &[0., 0.]]),
    };

    println!("=== The worked example from PLAN.md ===");
    println!("3 tokens, d = 4, one head of size 2. The weights are hand-picked small");
    println!("integers so the arithmetic is easy to follow; learned ones are messier.\n");
    print_table(
        "X: the tokens' vectors (one row per token)",
        &x,
        &tokens,
        &dims,
        0,
    );
    print_table("W_q (d × head size)", &head.wq, &dims, &hcols, 0);
    print_table("W_k", &head.wk, &dims, &hcols, 0);
    print_table("W_v", &head.wv, &dims, &hcols, 0);

    let t = head.forward(&x);
    println!("--- Step 1: project every token three ways ---\n");
    print_table(
        "Q = X · W_q   (what each token looks for)",
        &t.q,
        &tokens,
        &hcols,
        0,
    );
    print_table(
        "K = X · W_k   (what each token offers to be matched on)",
        &t.k,
        &tokens,
        &hcols,
        0,
    );
    print_table(
        "V = X · W_v   (what each token passes on)",
        &t.v,
        &tokens,
        &hcols,
        0,
    );
    println!("--- Step 2: score every query against every key, ÷ √(head size) = √2 ---\n");
    print_table(
        "scores = Q · Kᵀ / √2   (row: query of; column: key of)",
        &t.scores,
        &tokens,
        &tokens,
        2,
    );
    println!("--- Step 3: the causal mask: later tokens set to −∞ ---\n");
    print_table("masked scores", &t.masked, &tokens, &tokens, 2);
    println!("--- Step 4: softmax each row into weights that sum to 1 ---\n");
    print_weights("weights", &t.weights, &tokens, 2);
    println!("--- Step 5: each token's output is the weighted sum of the values ---\n");
    print_table("output = weights · V", &t.out, &tokens, &hcols, 2);
    let w = t.weights.row(2);
    println!(
        "sat:  {:.2} × [1 0]  +  {:.2} × [0 1]  +  {:.2} × [1 1]  =  [{:.2}  {:.2}]",
        w[0],
        w[1],
        w[2],
        t.out.get(2, 0),
        t.out.get(2, 1)
    );
    println!("      (the's value)    (cat's value)     (sat's value)");
}

/// An untrained head: random embeddings and random weights.
fn random(text: &str, dim: usize, head_size: usize, std: f64, seed: u64, verbose: bool) {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() || dim == 0 || head_size == 0 {
        eprintln!("mini_attention: error: need some text, and --dim and --head-size of at least 1");
        std::process::exit(2);
    }
    let tokens = labels(&words);
    let mut rng = Rng::new(seed);

    // One random embedding per distinct word, then layer normalization, as
    // a token's vector would be on entering block 1 (without position
    // information, so a repeated word gets an identical vector).
    let mut vocab: Vec<&str> = Vec::new();
    for w in &words {
        if !vocab.contains(w) {
            vocab.push(w);
        }
    }
    let table = Mat::random_normal(vocab.len(), dim, 1.0, &mut rng);
    let mut x = Mat::zeros(words.len(), dim);
    for (i, w) in words.iter().enumerate() {
        let id = vocab.iter().position(|v| v == w).unwrap();
        x.row_mut(i).copy_from_slice(table.row(id));
    }
    let x = layer_norm_rows(&x);
    let head = Head {
        wq: Mat::random_normal(dim, head_size, std, &mut rng),
        wk: Mat::random_normal(dim, head_size, std, &mut rng),
        wv: Mat::random_normal(dim, head_size, std, &mut rng),
    };

    println!("=== An untrained head ===");
    println!(
        "{} tokens, d = {dim}, head size {head_size}; weights drawn from normal(0, {std}²), seed {seed}.",
        words.len()
    );
    println!("Each token's vector: a random embedding, layer-normalized.\n");

    let t = head.forward(&x);
    if verbose {
        let hcols = index_labels(head_size);
        print_table("Q", &t.q, &tokens, &hcols, 3);
        print_table("K", &t.k, &tokens, &hcols, 3);
        print_table("V", &t.v, &tokens, &hcols, 3);
    }
    print_table(
        "scores = Q · Kᵀ / √(head size)",
        &t.scores,
        &tokens,
        &tokens,
        3,
    );
    print_weights(
        "weights (after the causal mask and softmax)",
        &t.weights,
        &tokens,
        3,
    );
    if verbose {
        print_table(
            "output = weights · V",
            &t.out,
            &tokens,
            &index_labels(head_size),
            3,
        );
    }

    // Compare with perfectly uniform attention: token i spreads 1/(i+1) over
    // itself and the i tokens before it.
    let max_score = t.scores.data.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let max_dev = (0..t.weights.rows)
        .flat_map(|i| (0..=i).map(move |j| (i, j)))
        .map(|(i, j)| (t.weights.get(i, j) - 1.0 / (i + 1) as f64).abs())
        .fold(0.0f64, f64::max);
    println!("largest |score|: {max_score:.4}");
    println!("largest difference from uniform attention (row i: 1/(i+1) each): {max_dev:.4}");
    if vocab.len() < words.len() {
        println!("\nRepeated words have identical rows: without position information, the same");
        println!("word gets the same vector wherever it appears, so attention can't tell the");
        println!("copies apart. This is why the model is given each token's position (see");
        println!("\"Position information\" in PLAN.md).");
    }
    if std <= 0.1 {
        println!("\nWith small weights every score is near 0, so each row is nearly uniform:");
        println!("an untrained head averages all earlier tokens. Try --std 1 to see why");
        println!("models start with small weights: large ones give sharp, arbitrary attention.");
    } else {
        println!("\nWith large weights the scores are large, so the softmax is sharp and the");
        println!("attention is arbitrary: the head has strong preferences it never learned.");
    }
}
