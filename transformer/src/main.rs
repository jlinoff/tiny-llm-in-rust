//! transformer: the LLM flow's steps 3–6 (PLAN.md, "The flows"). So far:
//!   build     step 3: create the model with random weights
//!   train     step 4: train it on token IDs (resumable)
//!   finetune  step 5: train it further on conversations (chat.ids), all
//!             weights or (--lora-rank) a low-rank adapter
//!   merge     fold a LoRA adapter into its base model
//!   generate  step 6: write text from a prompt (inference)
//!   evaluate  measure how often generated stories use requested words
//!   inspect   show what a checkpoint holds, and check it against its config
//!   bench     measure this machine's matrix-multiply speed

mod bench;
mod blas;
mod config;
mod data;
mod evaluate;
mod generate;
mod gpt;
mod hf_tokenizer;
mod llama;
mod lora;
mod model;
mod probe;
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
    /// Step 5: fine-tune a trained model on conversations (chat.ids, made by
    /// chat/make_chat.py and mini_bpe.py --allow-special). Only the replies
    /// are scored. Resumes if the input is a fine-tuning checkpoint.
    Finetune {
        /// The pretrained checkpoint (model.safetensors), or a fine-tuning
        /// checkpoint to resume
        #[arg(long)]
        model: PathBuf,
        /// The conversations' token IDs (chat.ids)
        #[arg(long)]
        data: PathBuf,
        /// The tokenizer (tok.bpe), for the turn markers' IDs
        #[arg(long)]
        tokenizer: PathBuf,
        /// The config file; only [finetune] is read
        #[arg(long)]
        config: PathBuf,
        /// The checkpoint to write (chat.safetensors; with LoRA, the adapter,
        /// such as chat-lora.safetensors)
        #[arg(short, long)]
        output: PathBuf,
        /// LoRA: train only a rank-R adapter, the model frozen (overrides
        /// [finetune] lora_rank; 0 fine-tunes every weight)
        #[arg(long)]
        lora_rank: Option<usize>,
        /// LoRA's α: the update is scaled by α/R (overrides [finetune]
        /// lora_alpha; default 2R)
        #[arg(long)]
        lora_alpha: Option<f64>,
        /// LoRA: also train a change to these rows of the token embedding:
        /// none, markers (the special tokens') or all (overrides [finetune]
        /// lora_embeddings; default none)
        #[arg(long)]
        lora_embeddings: Option<String>,
        /// Override [finetune] learning_rate
        #[arg(long)]
        learning_rate: Option<f64>,
        /// LoRA: resume this adapter (made by finetune from the same --model)
        #[arg(long)]
        adapter: Option<PathBuf>,
    },
    /// Fold a LoRA adapter into its base model, giving an ordinary checkpoint
    /// of the original shape (for generate and evaluate).
    Merge {
        /// The base model the adapter was trained on (model.safetensors)
        #[arg(long)]
        model: PathBuf,
        /// The adapter (chat-lora.safetensors)
        #[arg(long)]
        adapter: PathBuf,
        /// The merged checkpoint to write
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Step 6: write text that continues a prompt, one token at a time
    /// (inference). Works with any checkpoint.
    Generate {
        /// The checkpoint (model.safetensors, or chat.safetensors later)
        #[arg(long)]
        model: PathBuf,
        /// The tokenizer the model was trained with (tok.bpe)
        #[arg(long)]
        tokenizer: PathBuf,
        /// The config file; only [generate] is read (all optional)
        #[arg(long)]
        config: PathBuf,
        /// The text to continue (or with --chat, the request); empty starts
        /// a new document
        #[arg(long, default_value = "")]
        prompt: String,
        /// Treat the prompt as a user's request: wrap it in the turn markers
        /// and stop at <|end|> (for a fine-tuned model)
        #[arg(long)]
        chat: bool,
        /// Override [generate] max_tokens
        #[arg(long)]
        max_tokens: Option<usize>,
        /// Override [generate] temperature (0: always the likeliest token)
        #[arg(long)]
        temperature: Option<f64>,
        /// Override [generate] top_k (0: no cutoff)
        #[arg(long)]
        top_k: Option<usize>,
        /// Override [generate] seed
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Generate with a Llama-style open-weight model downloaded from
    /// Hugging Face (SmolLM, for example), or check our implementation
    /// against reference numbers from Hugging Face's own code
    Llama {
        /// The model's folder: config.json, model.safetensors, tokenizer.json
        #[arg(long)]
        model_dir: PathBuf,
        /// The text to continue
        #[arg(long, default_value = "")]
        prompt: String,
        /// Compare with tools/llama_reference.py's output instead of generating
        #[arg(long)]
        check: Option<PathBuf>,
        /// Stop after this many new tokens
        #[arg(long, default_value_t = 100)]
        max_tokens: usize,
        /// Sampling temperature (0: always the likeliest token)
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        /// Sample only from the k likeliest tokens (0: no cutoff)
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// Seed for the sampling
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Look inside a model: each head's previous-token and induction
    /// scores, one head's attention on a prompt, and the logit lens (what
    /// each block would predict).
    Probe {
        /// The checkpoint (any: model-init, a training checkpoint, model)
        #[arg(long)]
        model: PathBuf,
        /// The tokenizer the model was built with (tok.bpe)
        #[arg(long)]
        tokenizer: PathBuf,
        /// The text for the logit lens and the head's pattern
        #[arg(
            long,
            default_value = "Once upon a time, there was a little girl named Lily. She liked to play with her red ball in the park. One day, Lily lost her ball. She looked and looked, but she could not find it. Then her friend Tom came. Tom said, \"Lily, I can help you find your"
        )]
        prompt: String,
        /// The head to show on the prompt, as BLOCK.HEAD counting from 0
        /// (default: the strongest induction head)
        #[arg(long)]
        head: Option<String>,
        /// Random sequences for the head scores
        #[arg(long, default_value_t = 8)]
        sequences: usize,
        /// Seed for the random sequences
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Measure how often generated stories use the words their requests ask
    /// for (chat-eval.tsv, made by chat/make_chat.py), against a control.
    Evaluate {
        /// The checkpoint (chat.safetensors, or model.safetensors to compare)
        #[arg(long)]
        model: PathBuf,
        /// The tokenizer the model was built with (tok.bpe)
        #[arg(long)]
        tokenizer: PathBuf,
        /// The config file; [generate] sets the sampling
        #[arg(long)]
        config: PathBuf,
        /// The evaluation requests (chat-eval.tsv)
        #[arg(long)]
        requests: PathBuf,
        /// Also run the paired-loss check on these conversations
        /// (chat-eval.txt: the same requests, with their stories)
        #[arg(long)]
        paired: Option<PathBuf>,
        /// Send each request as a conversation (turn markers); without it,
        /// as plain text for the model to continue
        #[arg(long)]
        chat: bool,
        /// Use only the first N requests
        #[arg(long)]
        count: Option<usize>,
        /// Print this many of the stories
        #[arg(long, default_value_t = 3)]
        show: usize,
        /// Override [generate] max_tokens
        #[arg(long)]
        max_tokens: Option<usize>,
        /// Override [generate] temperature
        #[arg(long)]
        temperature: Option<f64>,
        /// Override [generate] top_k
        #[arg(long)]
        top_k: Option<usize>,
        /// Override [generate] seed (request i uses seed + i)
        #[arg(long)]
        seed: Option<u64>,
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
        Command::Finetune {
            model,
            data,
            tokenizer,
            config,
            output,
            lora_rank,
            lora_alpha,
            lora_embeddings,
            learning_rate,
            adapter,
        } => finetune_command(
            &model,
            &data,
            &tokenizer,
            &config,
            &output,
            FinetuneOptions {
                lora_rank,
                lora_alpha,
                lora_embeddings,
                learning_rate,
                adapter,
            },
        ),
        Command::Merge {
            model,
            adapter,
            output,
        } => merge_command(&model, &adapter, &output),
        Command::Evaluate {
            model,
            tokenizer,
            config,
            requests,
            paired,
            chat,
            count,
            show,
            max_tokens,
            temperature,
            top_k,
            seed,
        } => evaluate_command(
            &model,
            &tokenizer,
            &config,
            &requests,
            paired.as_deref(),
            chat,
            count,
            show,
            Overrides {
                max_tokens,
                temperature,
                top_k,
                seed,
            },
        ),
        Command::Llama {
            model_dir,
            prompt,
            check,
            max_tokens,
            temperature,
            top_k,
            seed,
        } => match check {
            Some(r) => llama_check(&model_dir, &r),
            None => llama_generate(&model_dir, &prompt, max_tokens, temperature, top_k, seed),
        },
        Command::Probe {
            model,
            tokenizer,
            prompt,
            head,
            sequences,
            seed,
        } => probe_command(
            &model,
            &tokenizer,
            &prompt,
            head.as_deref(),
            sequences,
            seed,
        ),
        Command::Generate {
            model,
            tokenizer,
            config,
            prompt,
            chat,
            max_tokens,
            temperature,
            top_k,
            seed,
        } => generate_command(
            &model,
            &tokenizer,
            &config,
            &prompt,
            chat,
            Overrides {
                max_tokens,
                temperature,
                top_k,
                seed,
            },
        ),
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

/// The sampling line of a training log. `passes`: the whole run's items
/// divided by the training set. With random sampling, how many items a run
/// of that many passes never sees: each item is picked a Poisson-distributed
/// number of times with mean `passes`, so none at all with chance e^−passes.
fn sampling_text(
    t: &train::TrainingConfig,
    passes: f64,
    items: &str,
    random: &str,
    what: &str,
) -> String {
    if t.sampling == "epochs" {
        format!(
            "sampling = epochs: {items}, each once per epoch, in a new shuffled order each epoch"
        )
    } else {
        format!(
            "sampling = random: {random}; by the end about {:.0}% of the {what} in none",
            100.0 * (-passes).exp()
        )
    }
}

/// A byte count for people: "512 bytes", "37 kB", "1.4 MB".
fn size_text(bytes: usize) -> String {
    match bytes {
        0..=999 => format!("{bytes} bytes"),
        1_000..=999_999 => format!("{:.0} kB", bytes as f64 / 1e3),
        _ => format!("{:.1} MB", bytes as f64 / 1e6),
    }
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
    // Check the data is big enough for the split and the windows now,
    // before printing anything or doing any work.
    data::split(&corpus.ids, tcfg.validation_fraction, config.context)
        .map_err(|e| format!("{}: {e}", data_path.display()))?;

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
    println!("  large matrix products: {}", blas::backend());
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
        "  {remaining} tokens to go, about {} passes over the training data",
        train::passes(remaining as f64 / n_train.max(1) as f64)
    );
    println!(
        "  {}",
        sampling_text(
            &tcfg,
            (tcfg.steps as usize * tcfg.batch_size * seq) as f64 / n_train.max(1) as f64,
            &format!(
                "{} non-overlapping windows",
                data::epoch_windows(n_train, seq)
            ),
            "windows at random starting points",
            "training tokens"
        )
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
        kind: "train",
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

/// A checkpoint read and checked: a mini_transformer checkpoint whose
/// tensors match its stored [model].
struct Loaded {
    path: PathBuf,
    ckpt: safetensors::File,
    config: ModelConfig,
    vocab: usize,
}

impl Loaded {
    fn meta(&self, k: &str) -> Result<String, String> {
        self.ckpt
            .metadata
            .get(k)
            .cloned()
            .ok_or(format!("{}: no {k} in the metadata", self.path.display()))
    }

    /// The weights alone (without Adam's state), as f32 matrices.
    fn params(&self) -> gpt::Params<f32> {
        let weights: Vec<safetensors::Tensor> = self
            .ckpt
            .tensors
            .iter()
            .filter(|t| !t.name.starts_with("adam."))
            .cloned()
            .collect();
        gpt::Params::<f32>::from_tensors(&weights)
    }

    /// "untrained", "trained to step N" or "fine-tuned for N steps".
    fn describe(&self) -> String {
        let step = self.ckpt.metadata.get("step");
        let rank = self
            .ckpt
            .metadata
            .get("lora_rank")
            .cloned()
            .unwrap_or_default();
        match (self.ckpt.metadata.get("kind").map(String::as_str), step) {
            (Some("finetune"), Some(s)) => format!("fine-tuned for {s} steps"),
            (Some("merged"), Some(s)) => {
                let emb = match self
                    .ckpt
                    .metadata
                    .get("lora_embeddings")
                    .map(String::as_str)
                {
                    Some("markers") => " (plus the special tokens' embeddings)",
                    Some("all") => " (plus the token embedding)",
                    _ => "",
                };
                format!("fine-tuned with a rank-{rank} LoRA adapter{emb} for {s} steps, merged")
            }
            (_, Some(s)) => format!("trained to step {s}"),
            _ => "untrained".into(),
        }
    }
}

fn load_checkpoint(path: &Path) -> Result<Loaded, String> {
    let ckpt = safetensors::read(path)?;
    let mut l = Loaded {
        path: path.to_path_buf(),
        ckpt,
        config: ModelConfig {
            d_model: 1,
            layers: 1,
            heads: 1,
            context: 1,
            d_ff: 1,
            seed: 0,
        },
        vocab: 0,
    };
    if l.meta("format")? != FORMAT {
        return Err(format!(
            "{}: not a mini_transformer checkpoint",
            path.display()
        ));
    }
    if l.ckpt.metadata.get("kind").map(String::as_str) == Some("lora") {
        return Err(format!(
            "{} is a LoRA adapter, not a whole model: fold it into its base model first (transformer merge)",
            path.display()
        ));
    }
    l.config = ModelConfig::from_toml(&l.meta("model")?)?;
    l.vocab = l
        .meta("vocab_size")?
        .parse()
        .map_err(|_| "vocab_size isn't a number")?;
    let problems = model::check_layout(&l.ckpt.tensors, &l.config, l.vocab);
    if !problems.is_empty() {
        return Err(format!("{}: {}", path.display(), problems.join("; ")));
    }
    Ok(l)
}

/// The tokenizer, checked to be the one the model was built with.
fn load_tokenizer(tok_path: &Path, model: &Loaded) -> Result<TokenizerInfo, String> {
    let tok = TokenizerInfo::load(tok_path)?;
    if hex(&tok.fingerprint) != model.meta("tokenizer_fingerprint")? {
        return Err(format!(
            "{} isn't the tokenizer {} was built with (fingerprints differ)",
            tok_path.display(),
            model.path.display()
        ));
    }
    Ok(tok)
}

/// The special tokens a conversation needs.
fn markers(tok: &TokenizerInfo) -> Result<data::Markers, String> {
    let id = |name: &str| {
        tok.special_id(name).map(|i| i as u32).ok_or(format!(
            "the tokenizer has no {name}: retrain it with the current mini_bpe.py, which adds the turn markers"
        ))
    };
    Ok(data::Markers {
        eos: id("<|endoftext|>")?,
        user: id("<|user|>")?,
        assistant: id("<|assistant|>")?,
        end: id("<|end|>")?,
    })
}

/// Command-line values that replace the config's [generate] ones.
struct Overrides {
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_k: Option<usize>,
    seed: Option<u64>,
}

fn sampling_config(
    config_path: &Path,
    over: &Overrides,
) -> Result<generate::GenerateConfig, String> {
    let mut cfg = generate::GenerateConfig::load(config_path)?;
    cfg.max_tokens = over.max_tokens.unwrap_or(cfg.max_tokens);
    cfg.temperature = over.temperature.unwrap_or(cfg.temperature);
    cfg.top_k = over.top_k.unwrap_or(cfg.top_k);
    cfg.seed = over.seed.unwrap_or(cfg.seed);
    cfg.check()?;
    Ok(cfg)
}

fn describe_sampling(cfg: &generate::GenerateConfig) -> String {
    format!(
        "temperature {}, top-k {}, seed {}, at most {} tokens",
        cfg.temperature,
        if cfg.top_k == 0 {
            "off".to_string()
        } else {
            cfg.top_k.to_string()
        },
        cfg.seed,
        cfg.max_tokens
    )
}

/// The prompt's token IDs, and the tokens that end generation.
///
/// Plain: EOS, then the text. Each training document ended with EOS, so EOS
/// is also what comes just before a document's start: putting it first
/// tells the model a new document begins. Generation stops at EOS.
///
/// Chat: EOS, <|user|>, the request, <|assistant|>: exactly how each
/// fine-tuning conversation began, so the model's next tokens are its
/// reply. Generation stops at <|end|> (or EOS).
fn prompt_ids(
    tok: &TokenizerInfo,
    encoder: &tokenizer::Encoder,
    prompt: &str,
    chat: bool,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    if chat {
        let m = markers(tok)?;
        let mut ids = vec![m.eos as usize, m.user as usize];
        ids.extend(encoder.encode(prompt)?);
        ids.push(m.assistant as usize);
        return Ok((ids, vec![m.end as usize, m.eos as usize]));
    }
    let eos = tok.special_id("<|endoftext|>");
    let mut ids: Vec<usize> = eos.into_iter().collect();
    ids.extend(encoder.encode(prompt)?);
    if ids.is_empty() {
        return Err("an empty prompt needs a tokenizer with <|endoftext|>".into());
    }
    Ok((ids, eos.into_iter().collect()))
}

fn llama_load(dir: &Path) -> Result<(llama::Llama, hf_tokenizer::HfTokenizer), String> {
    for f in ["config.json", "model.safetensors", "tokenizer.json"] {
        check_readable(&dir.join(f), f)?;
    }
    let tok = hf_tokenizer::HfTokenizer::load(&dir.join("tokenizer.json"))?;
    let start = std::time::Instant::now();
    let m = llama::Llama::load(dir)?;
    if tok.vocab_size() != m.c.vocab {
        return Err(format!(
            "the tokenizer has {} tokens, the model {}",
            tok.vocab_size(),
            m.c.vocab
        ));
    }
    let c = &m.c;
    println!(
        "Model {}: {} parameters; {} blocks, width {}, {} query heads sharing {} key/value heads (head size {}), FFN {}, V = {}, context {}",
        dir.display(),
        c.parameters(),
        c.layers,
        c.d,
        c.heads,
        c.kv_heads,
        c.head_dim,
        c.d_ff,
        c.vocab,
        c.context
    );
    println!(
        "  loaded in {:.1} s; large matrix products: {}",
        start.elapsed().as_secs_f64(),
        blas::backend()
    );
    Ok((m, tok))
}

fn llama_generate(
    dir: &Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_k: usize,
    seed: u64,
) -> Result<(), String> {
    use std::io::Write;
    println!("=== transformer llama ===");
    let (m, tok) = llama_load(dir)?;
    let eos = tok.special("<|endoftext|>");
    let mut ids = tok.encode(prompt)?;
    if ids.is_empty() {
        ids.push(eos.ok_or("an empty prompt needs <|endoftext|> to start from")?);
    }
    if ids.len() >= m.c.context {
        return Err(format!(
            "the prompt is {} tokens; the context is {}",
            ids.len(),
            m.c.context
        ));
    }
    println!(
        "Sampling: temperature {temperature}, top-k {top_k}, seed {seed}, at most {max_tokens} tokens; prompt {} tokens\n",
        ids.len()
    );
    print!("{prompt}");
    let mut g = llama::LlamaGenerator::new(&m);
    let mut rng = rng::Rng::new(seed);
    let mut out = generate::Utf8Stream::default();
    let t0 = std::time::Instant::now();
    let mut logits = g.feed(&ids, false);
    let prefill = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let mut made = 0;
    let mut stopped = false;
    while made < max_tokens && g.len() < m.c.context {
        let scores: Vec<f64> = logits.row(0).iter().map(|&v| v as f64).collect();
        let next = generate::sample(&scores, temperature, top_k, &mut rng);
        made += 1;
        if Some(next) == eos {
            stopped = true;
            break;
        }
        print!("{}", out.push(&tok.token_bytes(next)));
        std::io::stdout().flush().ok();
        if g.len() >= m.c.context {
            break;
        }
        logits = g.feed(&[next], false);
    }
    print!("{}", out.finish());
    let decode = t1.elapsed().as_secs_f64();
    println!(
        "\n\n{}",
        if stopped {
            format!("Stopped at <|endoftext|> after {made} tokens.")
        } else {
            format!("Stopped after {made} tokens.")
        }
    );
    println!(
        "  prefill: {} tokens in {:.0} ms; decode: {made} tokens in {:.1} s ({:.0} tokens/s)",
        ids.len(),
        prefill * 1e3,
        decode,
        made as f64 / decode.max(1e-9)
    );
    Ok(())
}

/// Compare with Hugging Face's numbers (tools/llama_reference.py): token
/// IDs, every position's next-token log-probability, the last position's
/// scores, and a greedy continuation.
fn llama_check(dir: &Path, reference: &Path) -> Result<(), String> {
    println!("=== transformer llama --check ===");
    check_readable(reference, "reference")?;
    let (m, tok) = llama_load(dir)?;
    let text =
        fs::read_to_string(reference).map_err(|e| format!("{}: {e}", reference.display()))?;
    let r: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", reference.display()))?;
    let ids_of = |v: &serde_json::Value| -> Vec<usize> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut failures = 0;

    // 1. The tokenizer.
    let cases = r["tokenize"].as_array().ok_or("no tokenize cases")?;
    let mut same = 0;
    for c in cases {
        let text = c["text"].as_str().unwrap_or_default();
        let want = ids_of(&c["ids"]);
        let got = tok.encode(text)?;
        if got == want {
            same += 1;
        } else {
            failures += 1;
            println!("  tokenizer differs on {text:?}:\n    ours {got:?}\n    theirs {want:?}");
        }
        if tok.decode(&got) != text {
            println!("  (decoding {text:?} doesn't give it back)");
        }
    }
    println!(
        "\nTokenizer: identical IDs for {same} of {} texts",
        cases.len()
    );

    // 2. The model.
    println!("\nModel, prompt by prompt:");
    println!(
        "  {:>6} {:>22} {:>22} {:>16}",
        "tokens", "largest log-prob diff", "largest logit diff", "greedy same for"
    );
    let (mut worst_lp, mut worst_logit) = (0.0f64, 0.0f64);
    for p in r["prompts"].as_array().ok_or("no prompts")? {
        let text = p["text"].as_str().unwrap_or_default();
        let ids = ids_of(&p["ids"]);
        if tok.encode(text)? != ids {
            failures += 1;
            println!("  prompt {text:?}: our token IDs differ; skipped");
            continue;
        }
        let mut g = llama::LlamaGenerator::new(&m);
        let all = g.feed(&ids, true);
        let mut lp_diff = 0.0f64;
        for (t, want) in p["next_logprob"]
            .as_array()
            .ok_or("no next_logprob")?
            .iter()
            .enumerate()
        {
            let row = all.row(t);
            let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
            let z: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
            let ours = row[ids[t + 1]] as f64 - max - z.ln();
            lp_diff = lp_diff.max((ours - want.as_f64().unwrap_or(f64::NAN)).abs());
        }
        let last = all.row(ids.len() - 1);
        let logit_diff = p["last_logits"]
            .as_array()
            .ok_or("no last_logits")?
            .iter()
            .zip(last)
            .map(|(w, &o)| (o as f64 - w.as_f64().unwrap_or(f64::NAN)).abs())
            .fold(0.0, f64::max);
        // Greedy continuation, through the KV cache, token by token.
        let want = ids_of(&p["greedy"]);
        let mut logits = tensor::Mat::from_vec(1, last.len(), last.to_vec());
        let mut agree = 0;
        for &w in &want {
            let next = probe::argmax(logits.row(0));
            if next != w {
                break;
            }
            agree += 1;
            logits = g.feed(&[next], false);
        }
        worst_lp = worst_lp.max(lp_diff);
        worst_logit = worst_logit.max(logit_diff);
        println!(
            "  {:>6} {lp_diff:>22.2e} {logit_diff:>22.2e} {:>10} of {}",
            ids.len(),
            agree,
            want.len()
        );
        if agree < want.len() {
            println!(
                "    greedy text diverges after {agree} tokens: {:?}",
                tok.decode(&want[..agree])
            );
        }
    }
    let ok = failures == 0 && worst_lp < 1e-3 && worst_logit < 1e-2;
    println!(
        "\n{}: largest log-probability difference {worst_lp:.2e}, largest logit difference {worst_logit:.2e}",
        if ok { "Matches the reference" } else { "Doesn't match the reference" }
    );
    if ok {
        Ok(())
    } else {
        Err("see the differences above".into())
    }
}

/// A token for display: its text with a leading space as ␣ and a newline
/// as ↵, special tokens by name.
fn token_text(enc: &tokenizer::Encoder, id: usize) -> String {
    String::from_utf8_lossy(enc.token_bytes(id))
        .replace(' ', "␣")
        .replace('\n', "↵")
}

fn probe_command(
    model_path: &Path,
    tok_path: &Path,
    prompt: &str,
    head: Option<&str>,
    sequences: usize,
    seed: u64,
) -> Result<(), String> {
    check_readable(model_path, "checkpoint")?;
    check_readable(tok_path, "tokenizer")?;
    if sequences == 0 {
        return Err("--sequences must be at least 1".into());
    }
    let model = load_checkpoint(model_path)?;
    let tok = load_tokenizer(tok_path, &model)?;
    let enc = tok.encoder()?;
    let m = markers(&tok)?;
    let (mut ids, _) = prompt_ids(&tok, &enc, prompt, false)?;
    let params = model.params();
    let c = &model.config;
    if ids.len() > c.context {
        ids.drain(..ids.len() - c.context);
    }
    let (layers, heads) = (c.layers, c.heads);
    let pick = match head {
        None => None,
        Some(h) => {
            let (b, hd) = h
                .split_once('.')
                .and_then(|(b, hd)| Some((b.parse::<usize>().ok()?, hd.parse::<usize>().ok()?)))
                .ok_or(format!("--head {h}: write BLOCK.HEAD, such as 1.3"))?;
            if b >= layers || hd >= heads {
                return Err(format!(
                    "--head {h}: the model has blocks 0–{} and heads 0–{}",
                    layers - 1,
                    heads - 1
                ));
            }
            Some((b, hd))
        }
    };

    println!("=== transformer probe ===");
    println!(
        "Model {}: {} parameters, {} blocks × {} heads, V = {}, {}",
        model_path.display(),
        model::formula(c, model.vocab),
        layers,
        heads,
        model.vocab,
        model.describe()
    );

    // 1. Head scores on repeated random words: the tokens that are a space
    // and a whole lowercase word of three letters or more ("␣ball"), the
    // kind of token a story model has seen most; random fragments
    // ("getable") would test it on input unlike anything it learned.
    let pool: Vec<usize> = (256..256 + tok.merges)
        .filter(|&t| {
            let b = enc.token_bytes(t);
            b.len() >= 4 && b[0] == b' ' && b[1..].iter().all(|c| c.is_ascii_lowercase())
        })
        .collect();
    if pool.len() < 10 {
        return Err("the tokenizer has too few whole-word tokens for the head test".into());
    }
    let hs = probe::head_scores(&params, c, m.eos as usize, &pool, sequences, seed);
    println!(
        "\nHeads, on {} sequences of {} random words repeated twice (from the tokenizer's {} whole-word tokens; PLAN.md, \"Looking inside the model\"):",
        hs.sequences,
        hs.length,
        pool.len()
    );
    println!(
        "  loss on the first copy:  {:.3}   (nothing to copy yet; chance is ln {} = {:.3})",
        hs.loss_first,
        model.vocab,
        (model.vocab as f64).ln()
    );
    println!(
        "  loss on the second copy: {:.3}   (lower: the model copies from its context)",
        hs.loss_second
    );
    let table = |title: &str, v: &[Vec<f64>]| {
        println!("\n  {title}");
        print!("  {:>9}", "");
        for h in 0..heads {
            print!(" {:>6}", format!("head {h}"));
        }
        println!();
        for (b, row) in v.iter().enumerate() {
            print!("  {:>9}", format!("block {b}"));
            for x in row {
                print!(" {x:>6.2}");
            }
            println!();
        }
    };
    table(
        "previous-token score: attention to the token just before",
        &hs.prev,
    );
    table(
        "induction score: in the second copy, attention to the token after the current token's earlier occurrence",
        &hs.induction,
    );
    let best = |v: &[Vec<f64>]| {
        let mut b = (0, 0, f64::NEG_INFINITY);
        for (i, row) in v.iter().enumerate() {
            for (h, &x) in row.iter().enumerate() {
                if x > b.2 {
                    b = (i, h, x);
                }
            }
        }
        b
    };
    let (pb, ph, pv) = best(&hs.prev);
    let (ib, ih, iv) = best(&hs.induction);
    println!(
        "\n  strongest previous-token head: {pb}.{ph} ({pv:.2}); strongest induction head: {ib}.{ih} ({iv:.2})"
    );
    println!(
        "  (a head attending evenly to all t + 1 positions would score about 1/(t + 1): {:.3} on average here)",
        (1..=2 * hs.length).map(|t| 1.0 / (t + 1) as f64).sum::<f64>() / (2 * hs.length) as f64
    );

    // 1b. Copying on real text: the prompt repeated, each head ablated.
    let (first, second, len, mut ch) = probe::copy_heads(&params, c, m.eos as usize, &ids[1..]);
    println!(
        "\nCopying real text: the prompt's {len} tokens, then the same again; every head switched off (ablated) in turn"
    );
    println!("  loss on the first copy:  {first:.3}");
    println!(
        "  loss on the second copy: {second:.3}   (lower than the first if the model copies from it)"
    );
    ch.sort_by(|a, b| b.copying().total_cmp(&a.copying()));
    println!(
        "\n  heads ranked by their part in the repeat: the loss rise on the second copy minus on the first,"
    );
    println!(
        "  without the head (a head important for language in general raises both; top 10 of {})",
        ch.len()
    );
    println!(
        "  {:>5} {:>5} {:>10} {:>12} {:>12} {:>10}",
        "rank", "head", "repeat", "rise, 2nd", "rise, 1st", "induction"
    );
    for (r, h) in ch.iter().take(10).enumerate() {
        println!(
            "  {:>5} {:>5} {:>+10.3} {:>+12.3} {:>+12.3} {:>10.2}",
            r + 1,
            format!("{}.{}", h.block, h.head),
            h.copying(),
            h.rise_second,
            h.rise_first,
            h.attention
        );
    }
    let most = ch
        .iter()
        .max_by(|a, b| a.rise_second.total_cmp(&b.rise_second))
        .expect("at least one head");
    let top_att = ch
        .iter()
        .max_by(|a, b| a.attention.total_cmp(&b.attention))
        .expect("at least one head");
    println!(
        "  most important overall: head {}.{} (rises {:+.3} and {:+.3})",
        most.block, most.head, most.rise_first, most.rise_second
    );
    println!(
        "  strongest induction attention on this text: head {}.{} ({:.2}), repeat effect {:+.3}",
        top_att.block,
        top_att.head,
        top_att.attention,
        top_att.copying()
    );
    println!(
        "  (induction: attention, in the second copy, to the token after the current token's earlier occurrence)"
    );

    // 2. The logit lens on the prompt.
    let tr = probe::trace(&params, c, &ids);
    let lens = probe::logit_lens(&params, &tr, &ids);
    println!(
        "\nLogit lens on the prompt ({} tokens): the residual stream after each stage, read through the final layer norm and output layer",
        ids.len()
    );
    println!(
        "  {:<22} {:>6} {:>9}   top 3 next tokens after the prompt",
        "after", "loss", "top-1"
    );
    for (k, row) in lens.iter().enumerate() {
        let name = if k == 0 {
            "the embeddings".to_string()
        } else if k == layers {
            format!("block {} (the output)", k - 1)
        } else {
            format!("block {}", k - 1)
        };
        let mut order: Vec<usize> = (0..row.last.len()).collect();
        order.sort_by(|&a, &b| row.last[b].total_cmp(&row.last[a]));
        let max = row.last[order[0]];
        let total: f64 = row.last.iter().map(|s| (s - max).exp()).sum();
        let top: Vec<String> = order[..3]
            .iter()
            .map(|&t| {
                format!(
                    "{} {:.0}%",
                    token_text(&enc, t),
                    100.0 * (row.last[t] - max).exp() / total
                )
            })
            .collect();
        println!(
            "  {name:<22} {:>6.3} {:>8.1}%   {}",
            row.loss,
            100.0 * row.accuracy,
            top.join(", ")
        );
    }

    // 3. One head's pattern on the prompt.
    let (b, h) = pick.unwrap_or((ib, ih));
    let a = &tr.attention[b][h];
    println!(
        "\nHead {b}.{h}{} on the prompt: where each position looks most (weight), from position 1",
        if pick.is_none() {
            " (the strongest induction head)"
        } else {
            ""
        }
    );
    for t in 1..ids.len() {
        let row = &a.row(t)[..=t];
        let j = probe::argmax(row);
        println!(
            "  [{t:>3}] {:<14} → [{j:>3}] {:<14} {:.2}",
            token_text(&enc, ids[t]),
            token_text(&enc, ids[j]),
            row[j]
        );
    }
    Ok(())
}

fn generate_command(
    model_path: &Path,
    tok_path: &Path,
    config_path: &Path,
    prompt: &str,
    chat: bool,
    over: Overrides,
) -> Result<(), String> {
    use std::io::Write;
    // Check every input first, before doing any work.
    check_readable(model_path, "checkpoint")?;
    check_readable(tok_path, "tokenizer")?;
    check_readable(config_path, "config")?;
    let cfg = sampling_config(config_path, &over)?;
    let model = load_checkpoint(model_path)?;
    let tok = load_tokenizer(tok_path, &model)?;
    let encoder = tok.encoder()?;
    let (ids, stop) = prompt_ids(&tok, &encoder, prompt, chat)?;
    let params = model.params();
    let config = &model.config;

    let n: usize = params.mats.iter().map(|m| m.data.len()).sum();
    println!(
        "=== transformer generate{} ===",
        if chat { " --chat" } else { "" }
    );
    println!(
        "Model {}: {n} parameters, context {}, V = {}, {}",
        model_path.display(),
        config.context,
        model.vocab,
        model.describe()
    );
    println!("Sampling: {}", describe_sampling(&cfg));
    println!(
        "Prompt: {} token(s), {}\n",
        ids.len(),
        if chat {
            "<|endoftext|> <|user|> request <|assistant|>"
        } else if stop.is_empty() {
            "the text"
        } else {
            "starting with <|endoftext|>"
        }
    );

    let mut out = std::io::stdout().lock();
    if chat {
        let _ = write!(out, "User: {prompt}\nAssistant: ");
    } else {
        let _ = write!(out, "{prompt}");
    }
    let _ = out.flush();
    let mut text = generate::Utf8Stream::default();
    let mut g = generate::Generator::new(&params, config);
    let report = generate::generate(&mut g, &cfg, &ids, &stop, |t| {
        let s = text.push(encoder.token_bytes(t));
        let _ = write!(out, "{s}");
        let _ = out.flush();
    });
    let _ = writeln!(out, "{}", text.finish());
    drop(out);

    let made = report.generated.len();
    println!(
        "\n{}",
        match report.stopped_at {
            Some(t) => format!(
                "Stopped at {} after {made} tokens.",
                String::from_utf8_lossy(encoder.token_bytes(t))
            ),
            None => format!("Stopped at the limit of {made} tokens."),
        }
    );
    let prompt_in = report.prompt_tokens.min(config.context);
    println!(
        "  prefill: {prompt_in} tokens in {:.1} ms ({:.0} tokens/s)",
        report.prefill_secs * 1e3,
        prompt_in as f64 / report.prefill_secs.max(1e-9)
    );
    if made > 0 {
        println!(
            "  decode:  {made} tokens in {:.2} s ({:.0} tokens/s){}",
            report.decode_secs,
            made as f64 / report.decode_secs.max(1e-9),
            if report.rebuilds > 0 {
                format!(
                    ", including {} rebuild(s) of the cache past the context",
                    report.rebuilds
                )
            } else {
                String::new()
            }
        );
    }
    Ok(())
}

/// Command-line values that replace or add to the config's [finetune].
struct FinetuneOptions {
    lora_rank: Option<usize>,
    lora_alpha: Option<f64>,
    lora_embeddings: Option<String>,
    learning_rate: Option<f64>,
    adapter: Option<PathBuf>,
}

fn finetune_command(
    model_path: &Path,
    data_path: &Path,
    tok_path: &Path,
    config_path: &Path,
    output: &Path,
    opts: FinetuneOptions,
) -> Result<(), String> {
    // Check every input and output first, before doing any work. As with
    // train, the output may be the input checkpoint (resuming in place).
    check_readable(model_path, "checkpoint")?;
    check_readable(data_path, "data")?;
    check_readable(tok_path, "tokenizer")?;
    check_readable(config_path, "config")?;
    if let Some(a) = &opts.adapter {
        check_readable(a, "adapter")?;
    }
    for input in [data_path, tok_path, config_path, model_path] {
        // The model may be the output only when resuming full fine-tuning
        // in place; a LoRA adapter must not overwrite its base.
        if input == model_path && (opts.lora_rank.unwrap_or(0) == 0 && opts.adapter.is_none()) {
            continue;
        }
        if same_file(input, output) {
            return Err(format!(
                "output {} would overwrite an input",
                output.display()
            ));
        }
    }
    check_writable(output)?;

    let model = load_checkpoint(model_path)?;
    let tok = load_tokenizer(tok_path, &model)?;
    let m = markers(&tok)?;
    let corpus = data::Corpus::load(data_path)?;
    train::check_data(&corpus, model.vocab, &model.meta("tokenizer_fingerprint")?)?;
    let mut tcfg = train::TrainingConfig::load_section(config_path, "finetune")?;
    if let Some(r) = opts.lora_rank {
        tcfg.lora_rank = r;
    }
    if opts.lora_alpha.is_some() {
        tcfg.lora_alpha = opts.lora_alpha;
    }
    if let Some(e) = opts.lora_embeddings {
        tcfg.lora_embeddings = e;
    }
    if let Some(lr) = opts.learning_rate {
        tcfg.learning_rate = lr;
    }
    tcfg.check()?;
    let lora = tcfg.lora_rank > 0;
    if opts.adapter.is_some() && !lora {
        return Err("--adapter resumes a LoRA run: give --lora-rank too".into());
    }
    if !lora && tcfg.lora_embeddings != "none" {
        return Err(
            "lora_embeddings is part of a LoRA adapter: give --lora-rank too (full fine-tuning trains every embedding row anyway)"
                .into(),
        );
    }
    let emb_rows = embedding_rows(&tcfg.lora_embeddings, &m, model.vocab);
    let config = model.config.clone();
    let seq = config.context;
    let convs = data::Conversations::parse(&corpus.ids, &m, seq)
        .map_err(|e| format!("{}: {e}", data_path.display()))?;
    let n_rows = convs.rows.len();
    let n_valid = (n_rows as f64 * tcfg.validation_fraction).round() as usize;
    let n_train = n_rows - n_valid;
    if n_train == 0 {
        return Err(format!(
            "{}: no conversation to train on fits in the context of {seq} tokens",
            data_path.display()
        ));
    }

    // Starting from a pretrained checkpoint: fine-tuning is a new run, with
    // its own step count and a fresh optimizer (Adam's averages describe
    // pretraining's gradients, not these). From a fine-tuning checkpoint:
    // resume.
    let parent_kind = model.ckpt.metadata.get("kind").cloned().unwrap_or_default();
    let model_sha = sha256_file(model_path)?;
    let mut state = if lora {
        lora_state(
            &model,
            &model_sha,
            &tcfg,
            &emb_rows,
            opts.adapter.as_deref(),
        )?
    } else {
        train::State::from_checkpoint(&model.ckpt)?
    };
    if !lora && parent_kind != "finetune" {
        state.step = 0;
        state.tokens = 0;
        for a in state.m.iter_mut().chain(state.v.iter_mut()) {
            a.data.iter_mut().for_each(|x| *x = 0.0);
        }
    }
    if state.step >= tcfg.steps {
        return Err(format!(
            "{} is already at fine-tuning step {}; raise [finetune] steps to train further",
            model_path.display(),
            state.step
        ));
    }

    let reply_tokens: usize = convs
        .rows
        .iter()
        .zip(&convs.reply_at)
        .map(|(r, &a)| r.len() - 1 - a)
        .sum();
    let all_tokens: usize = convs.rows.iter().map(|r| r.len() - 1).sum();
    let n: usize = state.params.mats.iter().map(|w| w.data.len()).sum();
    let n_model = model::formula(&config, model.vocab);
    println!(
        "=== transformer finetune{} ===",
        if lora { " (LoRA)" } else { "" }
    );
    println!(
        "Model {}: {n_model} parameters, context {seq}, V = {}, {}",
        model_path.display(),
        model.vocab,
        model.describe()
    );
    println!("  large matrix products: {}", blas::backend());
    if lora {
        println!(
            "LoRA adapter: rank {}, α {} (updates scaled by {}), on {} in each of {} blocks",
            tcfg.lora_rank,
            tcfg.alpha(),
            tcfg.alpha() / tcfg.lora_rank as f64,
            tcfg.targets().join(", "),
            config.layers
        );
        if !emb_rows.is_empty() {
            println!(
                "  and a change to {} of the token embedding's {} rows ({}), {} numbers",
                emb_rows.len(),
                model.vocab,
                if tcfg.lora_embeddings == "markers" {
                    "the special tokens'"
                } else {
                    "all"
                },
                emb_rows.len() * config.d_model
            );
        }
        println!(
            "  {n} trained numbers ({:.2}% of the model's); the model's own weights are frozen",
            100.0 * n as f64 / n_model as f64
        );
    }
    if state.step > 0 {
        println!("  resuming at fine-tuning step {}", state.step);
    }
    println!(
        "Data {}: {n_rows} conversations ({} longer than the context left out): {n_train} for training, {n_valid} held out",
        data_path.display(),
        convs.dropped
    );
    println!(
        "  only the replies are scored: {reply_tokens} of {all_tokens} tokens ({:.0}%)",
        evaluate::percent(reply_tokens, all_tokens)
    );
    println!(
        "Fine-tuning: steps {} to {}, {} conversations per step (each padded to {seq} tokens), about {} passes",
        state.step + 1,
        tcfg.steps,
        tcfg.batch_size,
        train::passes(((tcfg.steps - state.step) as usize * tcfg.batch_size) as f64 / n_train as f64)
    );
    println!(
        "  {}",
        sampling_text(
            &tcfg,
            (tcfg.steps as usize * tcfg.batch_size) as f64 / n_train as f64,
            &format!("{n_train} conversations"),
            "conversations drawn at random",
            "conversations"
        )
    );
    println!(
        "  AdamW: peak learning rate {} after {} warm-up steps, cosine down to {} of it;",
        tcfg.learning_rate, tcfg.warmup_steps, tcfg.min_lr_fraction
    );
    println!(
        "  weight decay {}; gradient clipped at length {}",
        tcfg.weight_decay, tcfg.grad_clip
    );

    let valid_rows: Vec<usize> = (n_train..n_rows)
        .take(tcfg.eval_batches * tcfg.batch_size)
        .collect();
    let valid: Vec<data::Batch> = valid_rows
        .chunks(tcfg.batch_size)
        .map(|c| convs.batch(c, seq, m.eos))
        .collect();
    if let Some(l) = train::validation_loss(&state, &config, &valid, seq) {
        println!("  validation loss on the replies before fine-tuning: {l:.3}");
    }
    println!();

    let mut base = model.ckpt.metadata.clone();
    if parent_kind == "train" {
        // Keep a record of the pretraining run this started from.
        for (from, to) in [
            ("training", "pretraining"),
            ("step", "pretraining_step"),
            ("valid_loss", "pretraining_valid_loss"),
            ("data_file", "pretraining_data_file"),
            ("data_sha256", "pretraining_data_sha256"),
        ] {
            if let Some(v) = model.ckpt.metadata.get(from) {
                base.insert(to.into(), v.clone());
            }
        }
    }
    for k in RUN_KEYS {
        base.remove(k);
    }
    if lora {
        base.insert("lora_rank".into(), tcfg.lora_rank.to_string());
        base.insert("lora_alpha".into(), tcfg.alpha().to_string());
        base.insert("lora_targets".into(), tcfg.targets().join(","));
        base.insert("lora_embeddings".into(), tcfg.lora_embeddings.clone());
        base.insert("lora_embedding_rows".into(), rows_text(&emb_rows));
        base.insert("base_file".into(), model_path.display().to_string());
        base.insert("base_sha256".into(), model_sha.clone());
    }
    let prov = train::Provenance {
        kind: if lora { "lora" } else { "finetune" },
        base,
        training_toml: toml::to_string(&tcfg).expect("a flat struct always serializes"),
        data_file: data_path.display().to_string(),
        data_sha256: sha256_file(data_path)?,
        parent_file: model_path.display().to_string(),
        parent_sha256: model_sha,
    };
    let start = std::time::Instant::now();
    let order = (tcfg.sampling == "epochs").then(|| data::EpochOrder::new(n_train, tcfg.seed));
    let batches = |step: u64| {
        convs.train_batch(
            n_train,
            tcfg.batch_size,
            seq,
            tcfg.seed,
            step,
            m.eos,
            order.as_ref(),
        )
    };
    let first = state.step;
    let state = train::run_with(
        state,
        &config,
        &tcfg,
        batches,
        &valid,
        output,
        &prov,
        |line| println!("{line}"),
    )?;
    let secs = start.elapsed().as_secs_f64();

    let tensors = state.tensors();
    if safetensors::read(output)?.tensors != tensors {
        return Err(format!(
            "{}: read-back doesn't match what was written",
            output.display()
        ));
    }
    println!(
        "\nFine-tuned to step {} in {:.1} minutes ({:.1} s per step).",
        state.step,
        secs / 60.0,
        secs / (state.step - first) as f64
    );
    println!(
        "Wrote {} with Adam's state (resumable); read back and verified.",
        output.display()
    );
    if lora {
        println!(
            "  a LoRA adapter, {n} numbers ({} of weights, {} with Adam's state); to use it:",
            size_text(4 * n),
            size_text(12 * n)
        );
        println!(
            "  transformer merge --model {} --adapter {} -o merged.safetensors",
            model_path.display(),
            output.display()
        );
    }
    let weights = &tensors[..state.params.mats.len()];
    println!("  weights fingerprint {}", weights_fingerprint(weights));
    Ok(())
}

/// The training state for a LoRA run: a fresh adapter on the model, or the
/// adapter given to resume (which must have been made from this very model).
/// The token embedding rows `lora_embeddings` chooses: none, the special
/// tokens' (end of text and the three chat markers), or all of them.
fn embedding_rows(mode: &str, m: &data::Markers, vocab: usize) -> Vec<usize> {
    match mode {
        "markers" => [m.eos, m.user, m.assistant, m.end]
            .iter()
            .map(|&i| i as usize)
            .collect(),
        "all" => (0..vocab).collect(),
        _ => Vec::new(),
    }
}

/// Rows as recorded in an adapter's metadata: "4092,4093,4094,4095", or
/// "0-4095" for every row in order.
fn rows_text(rows: &[usize]) -> String {
    if !rows.is_empty() && rows.iter().enumerate().all(|(k, &i)| k == i) {
        format!("0-{}", rows.len() - 1)
    } else {
        rows.iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// The embedding rows recorded in an adapter's metadata (none if absent:
/// adapters made before lora_embeddings existed).
fn adapter_rows(md: &std::collections::BTreeMap<String, String>) -> Result<Vec<usize>, String> {
    let text = md
        .get("lora_embedding_rows")
        .map(String::as_str)
        .unwrap_or("");
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let bad = || format!("bad lora_embedding_rows {text:?}");
    if let Some((a, b)) = text.split_once('-') {
        let (a, b): (usize, usize) = (a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?);
        return Ok((a..=b).collect());
    }
    text.split(',')
        .map(|x| x.parse().map_err(|_| bad()))
        .collect()
}

fn lora_state(
    model: &Loaded,
    model_sha: &str,
    tcfg: &train::TrainingConfig,
    emb_rows: &[usize],
    adapter: Option<&Path>,
) -> Result<train::State, String> {
    let base = model.params();
    let (rank, alpha) = (tcfg.lora_rank, tcfg.alpha());
    let state = match adapter {
        None => {
            let l = lora::Lora::new(
                &model.config,
                &base,
                rank,
                alpha,
                &tcfg.targets(),
                emb_rows,
                tcfg.seed,
            );
            let zeros = || {
                l.to_params()
                    .mats
                    .iter()
                    .map(|m| tensor::Mat::zeros(m.rows, m.cols))
                    .collect()
            };
            train::State {
                params: l.to_params(),
                frozen: None,
                m: zeros(),
                v: zeros(),
                step: 0,
                tokens: 0,
            }
        }
        Some(path) => {
            let f = safetensors::read(path)?;
            let meta = |k: &str| f.metadata.get(k).cloned().unwrap_or_default();
            if meta("kind") != "lora" {
                return Err(format!("{} is not a LoRA adapter", path.display()));
            }
            if meta("base_sha256") != model_sha {
                return Err(format!(
                    "{} was trained on a different model than {} (checksums differ)",
                    path.display(),
                    model.path.display()
                ));
            }
            if meta("lora_rank") != rank.to_string() || meta("lora_alpha") != alpha.to_string() {
                return Err(format!(
                    "{} has rank {} and α {}; give the same --lora-rank and --lora-alpha",
                    path.display(),
                    meta("lora_rank"),
                    meta("lora_alpha")
                ));
            }
            if adapter_rows(&f.metadata)? != emb_rows {
                return Err(format!(
                    "{} has lora_embeddings = {:?}; give the same setting to resume it",
                    path.display(),
                    f.metadata
                        .get("lora_embeddings")
                        .map(String::as_str)
                        .unwrap_or("none")
                ));
            }
            train::State::from_checkpoint(&f)?
        }
    };
    lora::Lora::from_params(&state.params, rank, alpha, emb_rows)?.check(&base)?;
    Ok(train::State {
        frozen: Some(train::Frozen {
            base,
            rank,
            alpha,
            emb_rows: emb_rows.to_vec(),
        }),
        ..state
    })
}

fn merge_command(model_path: &Path, adapter_path: &Path, output: &Path) -> Result<(), String> {
    check_readable(model_path, "checkpoint")?;
    check_readable(adapter_path, "adapter")?;
    for input in [model_path, adapter_path] {
        if same_file(input, output) {
            return Err(format!(
                "output {} would overwrite an input",
                output.display()
            ));
        }
    }
    check_writable(output)?;
    let model = load_checkpoint(model_path)?;
    let f = safetensors::read(adapter_path)?;
    let meta = |k: &str| f.metadata.get(k).cloned().unwrap_or_default();
    if meta("kind") != "lora" {
        return Err(format!("{} is not a LoRA adapter", adapter_path.display()));
    }
    if meta("base_sha256") != sha256_file(model_path)? {
        return Err(format!(
            "{} was trained on a different model than {} (checksums differ)",
            adapter_path.display(),
            model_path.display()
        ));
    }
    let rank: usize = meta("lora_rank").parse().map_err(|_| "bad lora_rank")?;
    let alpha: f64 = meta("lora_alpha").parse().map_err(|_| "bad lora_alpha")?;
    let adapter_tensors: Vec<safetensors::Tensor> = f
        .tensors
        .iter()
        .filter(|t| !t.name.starts_with("adam."))
        .cloned()
        .collect();
    let l = lora::Lora::from_params(
        &gpt::Params::<f32>::from_tensors(&adapter_tensors),
        rank,
        alpha,
        &adapter_rows(&f.metadata)?,
    )?;
    let base = model.params();
    l.check(&base)?;
    let merged = l.merge(&base);
    // The merged weights, with the base checkpoint's names and shapes.
    let tensors: Vec<safetensors::Tensor> = model
        .ckpt
        .tensors
        .iter()
        .filter(|t| !t.name.starts_with("adam."))
        .zip(&merged.mats)
        .map(|(t, m)| safetensors::Tensor::new(&t.name, &t.shape, m.data.clone()))
        .collect();
    let mut md = f.metadata.clone();
    md.insert("kind".into(), "merged".into());
    md.insert("adapter_file".into(), adapter_path.display().to_string());
    md.insert("adapter_sha256".into(), sha256_file(adapter_path)?);
    md.insert(
        "producer".into(),
        format!("transformer merge {}", env!("CARGO_PKG_VERSION")),
    );
    safetensors::write(output, &tensors, &md)?;
    let back = safetensors::read(output)?;
    if back.tensors != tensors || back.metadata != md {
        return Err(format!(
            "{}: read-back doesn't match what was written",
            output.display()
        ));
    }
    println!("=== transformer merge ===");
    println!(
        "Base {}: {} parameters; adapter {}: rank {rank}, α {alpha}, {} numbers on {} matrices",
        model_path.display(),
        model::formula(&model.config, model.vocab),
        adapter_path.display(),
        l.size(),
        l.targets.len()
    );
    println!(
        "Wrote {}: W + (α/r)·A·B for each adapted matrix; read back and verified.",
        output.display()
    );
    println!("  weights fingerprint {}", weights_fingerprint(&tensors));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_command(
    model_path: &Path,
    tok_path: &Path,
    config_path: &Path,
    requests_path: &Path,
    paired_path: Option<&Path>,
    chat: bool,
    count: Option<usize>,
    show: usize,
    over: Overrides,
) -> Result<(), String> {
    check_readable(model_path, "checkpoint")?;
    check_readable(tok_path, "tokenizer")?;
    check_readable(config_path, "config")?;
    check_readable(requests_path, "requests")?;
    if let Some(p) = paired_path {
        check_readable(p, "conversations")?;
    }
    let cfg = sampling_config(config_path, &over)?;
    let model = load_checkpoint(model_path)?;
    let tok = load_tokenizer(tok_path, &model)?;
    let encoder = tok.encoder()?;
    let text = fs::read_to_string(requests_path)
        .map_err(|e| format!("can't read {}: {e}", requests_path.display()))?;
    let mut requests =
        evaluate::parse_requests(&text).map_err(|e| format!("{}: {e}", requests_path.display()))?;
    if let Some(n) = count {
        requests.truncate(n);
    }
    if requests.len() < 2 {
        return Err("need at least 2 requests (each is the other's control)".into());
    }
    let params = model.params();
    println!(
        "=== transformer evaluate{} ===",
        if chat { " --chat" } else { "" }
    );
    println!(
        "Model {}: {}, V = {}",
        model_path.display(),
        model.describe(),
        model.vocab
    );
    println!(
        "Requests {}: {}, sent {}",
        requests_path.display(),
        requests.len(),
        if chat {
            "as conversations: <|user|> request <|assistant|>"
        } else {
            "as plain text for the model to continue (no turn markers)"
        }
    );
    if let Some(p) = paired_path {
        let text = fs::read_to_string(p).map_err(|e| format!("can't read {}: {e}", p.display()))?;
        let mut convs =
            evaluate::parse_conversations(&text).map_err(|e| format!("{}: {e}", p.display()))?;
        convs.truncate(requests.len());
        if convs.len() != requests.len()
            || convs
                .iter()
                .zip(&requests)
                .any(|(c, r)| c.request != r.text)
        {
            return Err(format!(
                "{} doesn't hold the same requests, in the same order, as {}",
                p.display(),
                requests_path.display()
            ));
        }
        let mut g = generate::Generator::new(&params, &model.config);
        let mut pt = evaluate::Paired::default();
        for (i, c) in convs.iter().enumerate() {
            let other = &requests[(i + 1) % requests.len()];
            let prompt = |req: &str| -> Result<Vec<usize>, String> {
                let text = if chat {
                    req.to_string()
                } else {
                    format!("{req}\n")
                };
                Ok(prompt_ids(&tok, &encoder, &text, chat)?.0)
            };
            let story = encoder.encode(&c.story)?;
            let lens: Vec<usize> = story
                .iter()
                .map(|&t| encoder.token_bytes(t).len())
                .collect();
            let spans: Vec<(usize, usize)> = requests[i]
                .words
                .iter()
                .flat_map(|w| evaluate::word_spans(&c.story, w))
                .collect();
            let words = evaluate::tokens_in_spans(&lens, &spans);
            match (
                evaluate::story_losses(&mut g, &prompt(&c.request)?, &story),
                evaluate::story_losses(&mut g, &prompt(&other.text)?, &story),
            ) {
                (Some(own), Some(oth)) => pt.add(&own, &oth, &words),
                _ => pt.skipped += 1,
            }
        }
        let n = pt.stories.max(1) as f64;
        let nw = pt.word_tokens.max(1) as f64;
        println!(
            "Paired loss: each held-out story's loss given its own request, and given the next request instead ({} stories{}):",
            pt.stories,
            if pt.skipped > 0 {
                format!(", {} longer than the context left out", pt.skipped)
            } else {
                String::new()
            }
        );
        println!(
            "  all story tokens:      own {:.4}, other {:.4}, difference {:+.4}; own lower for {} of {} stories (sign test p = {:.2e})",
            pt.own / n,
            pt.other / n,
            (pt.own - pt.other) / n,
            pt.own_lower,
            pt.stories,
            evaluate::sign_test(pt.own_lower, pt.stories)
        );
        let dw = (pt.own_word - pt.other_word) / nw;
        println!(
            "  requested-word tokens: own {:.3}, other {:.3}, difference {:+.3}; own lower for {} of {} stories (p = {:.2e})",
            pt.own_word / nw,
            pt.other_word / nw,
            dw,
            pt.own_word_lower,
            pt.word_stories,
            evaluate::sign_test(pt.own_word_lower, pt.word_stories)
        );
        println!(
            "  so the own request makes the requested words' tokens {:.2} times as likely, on average ({} tokens)\n",
            (-dw).exp(),
            pt.word_tokens
        );
    }
    println!(
        "Sampling: {} (request i uses seed + i)\n",
        describe_sampling(&cfg)
    );

    let mut g = generate::Generator::new(&params, &model.config);
    let mut tally = evaluate::Tally::default();
    let start = std::time::Instant::now();
    for (i, r) in requests.iter().enumerate() {
        let prompt = if chat {
            r.text.clone()
        } else {
            format!("{}\n", r.text)
        };
        let (ids, stop) = prompt_ids(&tok, &encoder, &prompt, chat)?;
        let cfg_i = generate::GenerateConfig {
            seed: cfg.seed + i as u64,
            ..cfg.clone()
        };
        let rep = generate::generate(&mut g, &cfg_i, &ids, &stop, |_| {});
        let bytes: Vec<u8> = rep
            .generated
            .iter()
            .flat_map(|&t| encoder.token_bytes(t).iter().copied())
            .collect();
        let story = String::from_utf8_lossy(&bytes);
        let control = &requests[(i + 1) % requests.len()].words;
        tally.add(
            &story,
            &r.words,
            control,
            rep.stopped_at.is_some(),
            rep.generated.len(),
        );
        if i < show {
            let marked: Vec<String> = r
                .words
                .iter()
                .map(|w| {
                    let yes = evaluate::contains_word(&story, w);
                    format!("{w} {}", if yes { "✓" } else { "✗" })
                })
                .collect();
            println!(
                "[{}] {}\n{}\n  ({})\n",
                i + 1,
                r.text,
                story.trim(),
                marked.join(", ")
            );
        }
    }
    let t = &tally;
    println!(
        "{} stories in {:.1} s, {:.0} tokens on average; {:.0}% ended by themselves ({}).",
        t.stories,
        start.elapsed().as_secs_f64(),
        t.tokens as f64 / t.stories as f64,
        evaluate::percent(t.stopped, t.stories),
        if chat { "<|end|>" } else { "<|endoftext|>" }
    );
    println!(
        "Requested words used:   {:>5.1}%  ({} of {}); all three in {:.1}% of stories",
        evaluate::percent(t.words_found, t.words_asked),
        t.words_found,
        t.words_asked,
        evaluate::percent(t.all_found, t.stories)
    );
    println!(
        "Control (another request's words, same stories): {:.1}%  ({} of {})",
        evaluate::percent(t.control_found, t.control_asked),
        t.control_found,
        t.control_asked
    );
    println!(
        "The difference, {:+.1} points, is what following the request adds.",
        evaluate::percent(t.words_found, t.words_asked)
            - evaluate::percent(t.control_found, t.control_asked)
    );
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
    if meta("kind") == Some("lora") {
        let rank: usize = meta("lora_rank")
            .and_then(|r| r.parse().ok())
            .ok_or("no lora_rank")?;
        let alpha: f64 = meta("lora_alpha")
            .and_then(|a| a.parse().ok())
            .ok_or("no lora_alpha")?;
        let weights: Vec<safetensors::Tensor> = f
            .tensors
            .iter()
            .filter(|t| !t.name.starts_with("adam."))
            .cloned()
            .collect();
        let l = lora::Lora::from_params(
            &gpt::Params::<f32>::from_tensors(&weights),
            rank,
            alpha,
            &adapter_rows(&f.metadata)?,
        )?;
        let emb = if l.emb_rows.is_empty() {
            String::new()
        } else {
            format!(" and {} embedding rows", l.emb_rows.len())
        };
        println!(
            "\nA LoRA adapter: rank {rank}, α {alpha}, {} numbers on {} matrices{emb}; fold it into its base model with transformer merge.",
            l.size(),
            l.targets.len()
        );
        println!("  weights fingerprint {}", weights_fingerprint(&weights));
        return Ok(());
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
