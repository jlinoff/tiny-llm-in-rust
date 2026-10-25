//! transformer: the LLM flow's steps 3–6 (PLAN.md, "The flows"). So far:
//!   build     step 3: create the model with random weights
//!   train     step 4: train it on token IDs (resumable)
//!   inspect   show what a checkpoint holds, and check it against its config
//!   bench     measure this machine's matrix-multiply speed

mod bench;
mod config;
mod data;
mod gpt;
mod model;
mod rng;
mod safetensors;
mod tape;
mod tensor;
mod tokenizer;
mod train;

use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use config::ModelConfig;
use tokenizer::{hex, TokenizerInfo};

/// Written into every checkpoint's metadata, so a file says what made it.
const FORMAT: &str = "mini_transformer checkpoint 1";

#[derive(Parser)]
#[command(
    name = "transformer",
    version,
    about = "A small GPT-style language model: the LLM flow's steps 3–6."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Step 3: create the model, with random weights, from the tokenizer (for
    /// V) and the config's [model] section; write it as a checkpoint.
    Build {
        /// The tokenizer file (tok.bpe), for V and its fingerprint
        #[arg(long)]
        tokenizer: PathBuf,
        /// The config file (config.toml); only [model] is read
        #[arg(long)]
        config: PathBuf,
        /// The checkpoint to write (model-init.safetensors)
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Step 4: train the model on token IDs (corpus.ids). Resumes if the
    /// input checkpoint holds Adam's state (a checkpoint written by train).
    Train {
        /// The checkpoint to start from (model-init.safetensors, or a
        /// training checkpoint to resume)
        #[arg(long)]
        model: PathBuf,
        /// The token IDs (corpus.ids), made by the model's tokenizer
        #[arg(long)]
        data: PathBuf,
        /// The config file; only [training] is read
        #[arg(long)]
        config: PathBuf,
        /// The checkpoint to write (model.safetensors); may be the same file
        /// as --model, to resume in place
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Show a checkpoint's metadata and tensors, and check them against its
    /// stored config.
    Inspect {
        /// The checkpoint (.safetensors)
        checkpoint: PathBuf,
        /// Also list every tensor
        #[arg(short, long)]
        verbose: bool,
    },
    /// Measure this machine's f32 matrix-multiply speed, to estimate training
    /// time.
    Bench {
        /// A checkpoint, to estimate for its parameter count
        #[arg(long)]
        checkpoint: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Build {
            tokenizer,
            config,
            output,
        } => build(&tokenizer, &config, &output),
        Command::Train {
            model,
            data,
            config,
            output,
        } => train_command(&model, &data, &config, &output),
        Command::Inspect {
            checkpoint,
            verbose,
        } => inspect(&checkpoint, verbose),
        Command::Bench { checkpoint } => bench_command(checkpoint.as_deref()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("transformer: error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Stop now, before any work, if `path` can't be written (the project's
/// convention). An existing file is left unchanged; a new one is removed.
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

fn check_readable(path: &Path, what: &str) -> Result<(), String> {
    fs::File::open(path)
        .map(|_| ())
        .map_err(|e| format!("can't read {what} {}: {e}", path.display()))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("can't read {}: {e}", path.display()))?;
    Ok(hex(&Sha256::digest(&bytes)))
}

/// SHA-256 of the weights alone (names, shapes, values), independent of the
/// metadata, so two builds can be compared across machines.
fn weights_fingerprint(tensors: &[safetensors::Tensor]) -> String {
    let mut h = Sha256::new();
    for t in tensors {
        h.update(t.name.as_bytes());
        for d in &t.shape {
            h.update((*d as u64).to_le_bytes());
        }
        for v in &t.data {
            h.update(v.to_le_bytes());
        }
    }
    hex(&h.finalize())
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

fn build(tok_path: &Path, config_path: &Path, output: &Path) -> Result<(), String> {
    // Check every input and output first, before doing any work.
    check_readable(tok_path, "tokenizer")?;
    check_readable(config_path, "config")?;
    for input in [tok_path, config_path] {
        if same_file(input, output) {
            return Err(format!(
                "output {} would overwrite an input",
                output.display()
            ));
        }
    }
    check_writable(output)?;

    let config = ModelConfig::load(config_path)?;
    let tok = TokenizerInfo::load(tok_path)?;
    let v = tok.vocab_size;
    println!("=== transformer build ===");
    println!(
        "Tokenizer {}: V = {v} (256 bytes + {} merges + {} special: {})",
        tok_path.display(),
        tok.merges,
        tok.specials.len(),
        tok.specials.join(", ")
    );
    println!("  fingerprint {}", hex(&tok.fingerprint));
    println!(
        "Config {} [model]: d_model = {}, layers = {}, heads = {} (head size {}), context = {}, d_ff = {}, seed = {}",
        config_path.display(),
        config.d_model,
        config.layers,
        config.heads,
        config.head_size(),
        config.context,
        config.d_ff,
        config.seed
    );

    let tensors = model::init(&config, v);
    let n: usize = tensors.iter().map(|t| t.data.len()).sum();
    let expected = model::formula(&config, v);
    if n != expected {
        return Err(format!(
            "internal error: {n} parameters, but the formula gives {expected}"
        ));
    }
    println!(
        "\nParameters: {n}, matching PLAN.md's formula V·d + T·d + L·(4d² + 2·d·d_ff + 9d + d_ff) + 2d."
    );
    let emb = v * config.d_model + config.context * config.d_model;
    println!(
        "  embeddings {emb} ({:.0}%), blocks {} ({:.0}%), final layer norm {}",
        100.0 * emb as f64 / n as f64,
        n - emb - 2 * config.d_model,
        100.0 * (n - emb - 2 * config.d_model) as f64 / n as f64,
        2 * config.d_model
    );

    let mut meta = BTreeMap::new();
    meta.insert("format".into(), FORMAT.into());
    meta.insert("kind".into(), "init".into());
    meta.insert("model".into(), config.to_toml());
    meta.insert("vocab_size".into(), v.to_string());
    meta.insert("parameters".into(), n.to_string());
    meta.insert("architecture".into(), model::ARCHITECTURE.into());
    meta.insert("init".into(), model::INIT.into());
    meta.insert("tokenizer_fingerprint".into(), hex(&tok.fingerprint));
    meta.insert("tokenizer_file".into(), tok_path.display().to_string());
    meta.insert("tokenizer_sha256".into(), sha256_file(tok_path)?);
    meta.insert("config_file".into(), config_path.display().to_string());
    meta.insert("config_sha256".into(), sha256_file(config_path)?);
    meta.insert(
        "producer".into(),
        format!("transformer build {}", env!("CARGO_PKG_VERSION")),
    );
    safetensors::write(output, &tensors, &meta)?;

    // Verify: read the file back and compare everything, bit for bit.
    let back = safetensors::read(output)?;
    if back.tensors != tensors || back.metadata != meta {
        return Err(format!(
            "{}: read-back doesn't match what was written",
            output.display()
        ));
    }
    let size = fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    println!(
        "\nWrote {} ({:.1} MB, {} tensors); read back and verified.",
        output.display(),
        size as f64 / 1e6,
        tensors.len()
    );
    println!("  weights fingerprint {}", weights_fingerprint(&tensors));
    println!("  (same seed and shapes on another machine should give the same fingerprint)");
    Ok(())
}

/// Metadata keys that describe one training run; the rest of an input
/// checkpoint's metadata (model, tokenizer, …) is carried forward.
const RUN_KEYS: [&str; 11] = [
    "kind",
    "training",
    "step",
    "tokens_seen",
    "train_loss",
    "valid_loss",
    "data_file",
    "data_sha256",
    "parent_file",
    "parent_sha256",
    "producer",
];

fn train_command(
    model_path: &Path,
    data_path: &Path,
    config_path: &Path,
    output: &Path,
) -> Result<(), String> {
    // Check every input and output first, before doing any work. The output
    // may be the input checkpoint (resuming in place: it's read in full
    // first, and replaced by a rename), but not the data or the config.
    check_readable(model_path, "checkpoint")?;
    check_readable(data_path, "data")?;
    check_readable(config_path, "config")?;
    for input in [data_path, config_path] {
        if same_file(input, output) {
            return Err(format!(
                "output {} would overwrite an input",
                output.display()
            ));
        }
    }
    check_writable(output)?;

    let ckpt = safetensors::read(model_path)?;
    let meta = |k: &str| {
        ckpt.metadata
            .get(k)
            .cloned()
            .ok_or(format!("{}: no {k} in the metadata", model_path.display()))
    };
    if meta("format")? != FORMAT {
        return Err(format!(
            "{}: not a mini_transformer checkpoint",
            model_path.display()
        ));
    }
    let config = ModelConfig::from_toml(&meta("model")?)?;
    let vocab: usize = meta("vocab_size")?
        .parse()
        .map_err(|_| "vocab_size isn't a number")?;
    let problems = model::check_layout(&ckpt.tensors, &config, vocab);
    if !problems.is_empty() {
        return Err(format!("{}: {}", model_path.display(), problems.join("; ")));
    }
    let corpus = data::Corpus::load(data_path)?;
    train::check_data(&corpus, vocab, &meta("tokenizer_fingerprint")?)?;
    let tcfg = train::TrainingConfig::load(config_path)?;
    let state = train::State::from_checkpoint(&ckpt)?;
    if state.step >= tcfg.steps {
        return Err(format!(
            "{} is already at step {}; raise [training] steps to train further",
            model_path.display(),
            state.step
        ));
    }

    let n: usize = state.params.mats.iter().map(|m| m.data.len()).sum();
    let seq = config.context;
    let n_valid = (corpus.ids.len() as f64 * tcfg.validation_fraction).round() as usize;
    let n_train = corpus.ids.len() - n_valid;
    let tokens_per_step = tcfg.batch_size * seq;
    let remaining = (tcfg.steps - state.step) as usize * tokens_per_step;
    println!("=== transformer train ===");
    println!(
        "Model {}: {n} parameters, context {seq}, V = {vocab}",
        model_path.display()
    );
    if state.step > 0 {
        println!(
            "  resuming at step {} ({} tokens seen so far)",
            state.step, state.tokens
        );
    }
    println!(
        "Data {}: {} tokens: {n_train} for training, {n_valid} held out for validation",
        data_path.display(),
        corpus.ids.len()
    );
    println!(
        "Training: steps {} to {}, {} sequences × {seq} tokens = {tokens_per_step} tokens per step",
        state.step + 1,
        tcfg.steps,
        tcfg.batch_size
    );
    println!(
        "  {remaining} tokens to go, about {:.1} passes over the training data",
        remaining as f64 / n_train.max(1) as f64
    );
    println!(
        "  AdamW: peak learning rate {} after {} warm-up steps, cosine down to {} of it;",
        tcfg.learning_rate, tcfg.warmup_steps, tcfg.min_lr_fraction
    );
    println!(
        "  weight decay {}; gradient clipped at length {}",
        tcfg.weight_decay, tcfg.grad_clip
    );
    println!(
        "  chance-level loss: ln {vocab} = {:.3}\n",
        (vocab as f64).ln()
    );

    let mut base = ckpt.metadata.clone();
    for k in RUN_KEYS {
        base.remove(k);
    }
    let prov = train::Provenance {
        base,
        training_toml: toml::to_string(&tcfg).expect("a flat struct always serializes"),
        data_file: data_path.display().to_string(),
        data_sha256: sha256_file(data_path)?,
        parent_file: model_path.display().to_string(),
        parent_sha256: sha256_file(model_path)?,
    };
    let start = std::time::Instant::now();
    let state = train::run(state, &config, &tcfg, &corpus.ids, output, &prov, |line| {
        println!("{line}")
    })?;
    let secs = start.elapsed().as_secs_f64();

    // Verify: read the final checkpoint back and compare.
    let tensors = state.tensors();
    if safetensors::read(output)?.tensors != tensors {
        return Err(format!(
            "{}: read-back doesn't match what was written",
            output.display()
        ));
    }
    println!(
        "\nTrained to step {} in {:.1} minutes ({:.0} tokens per second overall).",
        state.step,
        secs / 60.0,
        remaining as f64 / secs
    );
    println!(
        "Wrote {} with Adam's state (resumable); read back and verified.",
        output.display()
    );
    let weights = &tensors[..state.params.mats.len()];
    println!("  weights fingerprint {}", weights_fingerprint(weights));
    Ok(())
}

fn inspect(path: &Path, verbose: bool) -> Result<(), String> {
    let f = safetensors::read(path)?;
    let meta = |k: &str| f.metadata.get(k).map(String::as_str);
    println!("=== {} ===", path.display());
    if meta("format") != Some(FORMAT) {
        println!(
            "(not a mini_transformer checkpoint: format {:?})",
            meta("format")
        );
    }
    for (k, v) in &f.metadata {
        if k == "model" {
            continue;
        }
        println!("  {k:<22} {v}");
    }
    let n: usize = f.tensors.iter().map(|t| t.data.len()).sum();
    println!("  {:<22} {}", "tensors", f.tensors.len());
    println!("  {:<22} {n} numbers", "total");
    if verbose {
        println!();
        for t in &f.tensors {
            println!("  {:<28} {:?}", t.name, t.shape);
        }
    }
    let (Some(model_text), Some(v)) = (meta("model"), meta("vocab_size")) else {
        return Err("no stored [model] or vocab_size: can't check the layout".into());
    };
    let config = ModelConfig::from_toml(model_text)?;
    let v: usize = v.parse().map_err(|_| "vocab_size isn't a number")?;
    println!("\nStored [model]:");
    for line in model_text.lines() {
        println!("  {line}");
    }
    let problems = model::check_layout(&f.tensors, &config, v);
    if problems.is_empty() {
        println!(
            "\nEvery tensor matches the stored config's layout ({} expected parameters).",
            model::formula(&config, v)
        );
        println!("  weights fingerprint {}", weights_fingerprint(&f.tensors));
        Ok(())
    } else {
        for p in &problems {
            println!("  problem: {p}");
        }
        Err(format!(
            "{} problem(s) with the tensor layout",
            problems.len()
        ))
    }
}

fn bench_command(checkpoint: Option<&Path>) -> Result<(), String> {
    let model = match checkpoint {
        None => None,
        Some(p) => {
            let f = safetensors::read(p)?;
            let meta = |k: &str| {
                f.metadata
                    .get(k)
                    .cloned()
                    .ok_or(format!("{}: no {k} in the metadata", p.display()))
            };
            let config = ModelConfig::from_toml(&meta("model")?)?;
            let vocab: usize = meta("vocab_size")?
                .parse()
                .map_err(|_| "vocab_size isn't a number")?;
            let problems = model::check_layout(&f.tensors, &config, vocab);
            if !problems.is_empty() {
                return Err(format!("{}: {}", p.display(), problems.join("; ")));
            }
            let tensors = f
                .tensors
                .into_iter()
                .filter(|t| !t.name.starts_with("adam."))
                .collect();
            Some(bench::Model {
                config,
                vocab,
                tensors,
            })
        }
    };
    bench::run(model.as_ref());
    Ok(())
}
