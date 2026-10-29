//! mini_w2v: a minimal word2vec, written for understanding.
//!
//! word2vec (Mikolov et al., 2013) learns a vector for every word from one
//! idea: a word is characterized by the words around it. Each word starts as
//! a small random vector; training repeatedly nudges the vectors so that a
//! word's vector can predict its neighbors. Words used in similar contexts
//! end up with similar vectors.
//!
//! The stages, one module each:
//!   corpus  read text, split into words, build the vocabulary (V words)
//!   train   two V x d tables; skip-gram or CBOW with negative sampling
//!   query   nearest neighbors; save vectors in word2vec text format

mod corpus;
mod matrix;
mod query;
mod train;

use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;

use corpus::Corpus;
use query::Normalized;
use train::{Arch, Config, Model, Trainer};

const EPILOG: &str = "\
The word2vec phase:
  input     training text: the corpus files (each file is one document)
  process   learn one vector per vocabulary word from its contexts
  output    the word vectors (-o FILE), one line per word, in the standard
            word2vec text format; explore them with the scripts/ directory

examples:
  mini_w2v ../data/text8 -p king,cat,paris          train, watch three words
  mini_w2v ../data/text8 --model cbow -o cbow.txt   train with CBOW
  mini_w2v ../data/text8 -e 0 -p king -o rand.txt   random vectors, no training
  mini_w2v ../data/text8 -t 1 --seed 7              reproducible run";

#[derive(Parser)]
#[command(
    name = "mini_w2v",
    version,
    about = "Train word vectors with word2vec (skip-gram or CBOW, negative sampling).",
    after_help = EPILOG
)]
struct Args {
    /// UTF-8 text file(s) to train on; each file is one document
    #[arg(required = true)]
    corpus: Vec<PathBuf>,

    /// Where to write the trained word vectors (word2vec text format)
    #[arg(short, long, default_value = "vectors.txt")]
    output: PathBuf,

    /// Training task: skipgram predicts context words from the center word;
    /// cbow predicts the center word from its context
    #[arg(short, long, value_enum, default_value_t = Arch::SkipGram)]
    model: Arch,

    /// Vector dimension d
    #[arg(short, long, default_value_t = 100)]
    dim: usize,

    /// Maximum context window: words on each side of the center word
    #[arg(short, long, default_value_t = 5)]
    window: usize,

    /// Negative samples per prediction
    #[arg(short = 'k', long, default_value_t = 5)]
    negative: usize,

    /// Passes over the corpus (0 = save the random initial vectors)
    #[arg(short, long, default_value_t = 5)]
    epochs: usize,

    /// Ignore words that occur fewer than this many times
    #[arg(long, default_value_t = 5)]
    min_count: u64,

    /// Subsampling threshold for frequent words (0 disables subsampling)
    #[arg(long, default_value_t = 1e-3)]
    sample: f64,

    /// Starting learning rate [default: 0.025 for skipgram, 0.05 for cbow]
    #[arg(long)]
    alpha: Option<f64>,

    /// Worker threads [default: all CPU cores]; with 1 thread a given
    /// seed always gives identical results
    #[arg(short, long)]
    threads: Option<usize>,

    /// Random seed
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Comma-separated words whose nearest neighbors are shown after
    /// initialization and after every epoch
    #[arg(short, long, value_delimiter = ',')]
    probe: Vec<String>,

    /// Number of neighbors shown for each probe word
    #[arg(long, default_value_t = 8)]
    top: usize,

    /// Don't show the progress line during training
    #[arg(short, long)]
    quiet: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("mini_w2v: error: {msg}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &Args) -> Result<(), String> {
    if args.dim == 0 || args.window == 0 {
        return Err("--dim and --window must be at least 1".into());
    }
    let threads = args
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    if threads == 0 {
        return Err("--threads must be at least 1".into());
    }
    let alpha = args.alpha.unwrap_or(match args.model {
        Arch::SkipGram => 0.025,
        Arch::Cbow => 0.05,
    });
    // Fail now, not after a long training run, if the output can't be written.
    check_writable(&args.output)?;

    // --- Phase input: the corpus and its vocabulary ---
    println!("=== Corpus ===");
    let t0 = Instant::now();
    let corpus = Corpus::load(&args.corpus, args.min_count)?;
    let vocab = &corpus.vocab;
    for doc in &corpus.documents {
        println!("  {:<28} {:>12} words", doc.path.display(), sep(doc.words));
    }
    let words_read: u64 = corpus.documents.iter().map(|d| d.words).sum();
    println!(
        "  {} words read, {} distinct ({:.1} s)",
        sep(words_read),
        sep(corpus.distinct as u64),
        t0.elapsed().as_secs_f64()
    );
    println!(
        "  vocabulary: V = {} words occurring at least {} times ({} corpus words kept)",
        sep(vocab.len() as u64),
        args.min_count,
        sep(corpus.tokens.len() as u64)
    );
    let top: Vec<String> = (0..vocab.len().min(10))
        .map(|i| format!("{} ({})", vocab.words[i], sep(vocab.counts[i])))
        .collect();
    println!("  most frequent: {}", top.join(", "));

    // --- The model: two randomly initialized tables ---
    let model = Model::new(vocab.len(), args.dim, args.seed);
    let cfg = Config {
        arch: args.model,
        window: args.window,
        negative: args.negative,
        epochs: args.epochs,
        alpha,
        sample: args.sample,
        threads,
        seed: args.seed,
        quiet: args.quiet,
    };
    let trainer = Trainer::new(&cfg, &model, &corpus);
    let weights = sep((vocab.len() * args.dim) as u64);
    println!("\n=== Model ===");
    println!(
        "  input  (word vectors):    V x d = {} x {} = {} weights, uniform in [-{}, {})",
        sep(vocab.len() as u64),
        args.dim,
        weights,
        0.5 / args.dim as f64,
        0.5 / args.dim as f64
    );
    println!(
        "  output (context vectors): V x d = {} x {} = {} weights, all zero",
        sep(vocab.len() as u64),
        args.dim,
        weights
    );
    println!(
        "  {}, window {}, {} negative samples, {} epochs, lr {} -> {}, {} thread{}",
        match cfg.arch {
            Arch::SkipGram => "skipgram",
            Arch::Cbow => "cbow",
        },
        cfg.window,
        cfg.negative,
        cfg.epochs,
        alpha,
        alpha * 1e-4,
        threads,
        if threads == 1 { "" } else { "s" }
    );
    if cfg.sample > 0.0 {
        println!(
            "  subsampling (t = {}): about {:.0}% of word occurrences kept each epoch",
            cfg.sample,
            100.0 * trainer.expected_keep_fraction()
        );
    }

    let probes = resolve_probes(&args.probe, vocab);
    show_probes(
        "after initialization (random vectors)",
        &probes,
        &model,
        vocab,
        args.top,
    );

    // --- Process: training ---
    if cfg.epochs > 0 {
        println!("\n=== Training ===");
        println!("  loss per prediction: \"running\" is measured during the epoch while the");
        println!(
            "  weights change; \"fixed\" is measured after it on the same 100,000 predictions"
        );
        println!("  with the weights frozen, so it is comparable from epoch to epoch.");
        println!("  before training: fixed {:.3}", trainer.eval_loss());
    }
    let start = Instant::now();
    for epoch in 1..=cfg.epochs {
        let t = Instant::now();
        let stats = trainer.run_epoch(epoch);
        let secs = t.elapsed().as_secs_f64();
        println!(
            "  epoch {epoch}/{}: loss running {:.3}, fixed {:.3}; {} words after subsampling, {:.1} s ({:.0}k words/s)",
            cfg.epochs,
            stats.loss / stats.predictions.max(1) as f64,
            trainer.eval_loss(),
            sep(stats.words_kept),
            secs,
            corpus.tokens.len() as f64 / secs / 1000.0
        );
        show_probes(
            &format!("after epoch {epoch}"),
            &probes,
            &model,
            vocab,
            args.top,
        );
    }
    if cfg.epochs > 0 {
        println!(
            "  total training time {:.1} s",
            start.elapsed().as_secs_f64()
        );
    }

    // --- Phase output: the word vectors ---
    query::save_vectors(&args.output, vocab, &model.input)
        .map_err(|e| format!("can't write {}: {e}", args.output.display()))?;
    println!(
        "\nSaved {} word vectors of dimension {} to {}",
        sep(vocab.len() as u64),
        args.dim,
        args.output.display()
    );
    Ok(())
}

/// Check that `path` can be written, without changing an existing file:
/// open it for writing without truncating, and if it didn't exist before,
/// remove the empty file that the check created.
fn check_writable(path: &Path) -> Result<(), String> {
    if path.is_dir() {
        return Err(format!("output {} is a directory", path.display()));
    }
    let existed = path.exists();
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| format!("can't write output {}: {e}", path.display()))?;
    if !existed {
        let _ = fs::remove_file(path);
    }
    Ok(())
}

/// Look up the probe words, reporting any that aren't in the vocabulary.
fn resolve_probes(words: &[String], vocab: &corpus::Vocab) -> Vec<(String, usize)> {
    let mut found = Vec::new();
    for w in words {
        let w = w.trim().to_lowercase();
        match vocab.id(&w) {
            Some(id) => found.push((w, id as usize)),
            None => println!("  (probe word {w:?} is not in the vocabulary; skipped)"),
        }
    }
    found
}

fn show_probes(
    when: &str,
    probes: &[(String, usize)],
    model: &Model,
    vocab: &corpus::Vocab,
    top: usize,
) {
    if probes.is_empty() {
        return;
    }
    println!("  nearest neighbors {when}:");
    let normed = Normalized::new(&model.input);
    for (word, id) in probes {
        let list: Vec<String> = normed
            .nearest(*id, top)
            .iter()
            .map(|&(n, sim)| format!("{} {:.2}", vocab.words[n], sim))
            .collect();
        println!("    {word:>10}: {}", list.join(", "));
    }
}

/// Format an integer with thousands separators: 17005207 -> "17,005,207".
fn sep(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
