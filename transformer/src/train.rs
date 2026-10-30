//! Step 4, train: the training loop. Step 5, finetune, runs the same loop
//! on conversations (`run_with`).
//!
//! Each step: take a batch of random windows from the training data, run
//! the model forward and backward (the tape works out every gradient),
//! limit the gradient's overall size, and let AdamW update every weight.
//! Every so often: report the loss, measure the loss on held-out validation
//! data, and write a checkpoint that holds everything needed to resume.

use crate::config::ModelConfig;
use crate::data::{self, Batch, Corpus};
use crate::gpt::{forward, forward_with, Params};
use crate::lora::Lora;
use crate::safetensors::{self, Tensor};
use crate::tape::Var;
use crate::tensor::Mat;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

/// The `[training]` section of config.toml. Only `learning_rate`, `steps`
/// and `batch_size` are required; the rest have the defaults below.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    /// Peak learning rate, reached after the warm-up.
    pub learning_rate: f64,
    /// Train until this step (counting from the start, across resumes).
    pub steps: u64,
    /// Sequences per step; each is the model's full context length.
    pub batch_size: usize,
    /// Steps over which the learning rate rises from 0 to its peak.
    #[serde(default = "d_warmup")]
    pub warmup_steps: u64,
    /// After the warm-up, the learning rate follows a cosine curve down to
    /// this fraction of the peak at the last step.
    #[serde(default = "d_min_lr")]
    pub min_lr_fraction: f64,
    /// AdamW's weight decay, applied to the matrices and embeddings only
    /// (not to biases or layer norm's γ and β).
    #[serde(default = "d_wd")]
    pub weight_decay: f64,
    /// If the gradient's overall length (all weights together) exceeds
    /// this, it's scaled down to it. 0 turns clipping off.
    #[serde(default = "d_clip")]
    pub grad_clip: f64,
    #[serde(default = "d_beta1")]
    pub beta1: f64,
    #[serde(default = "d_beta2")]
    pub beta2: f64,
    /// Seed for choosing the batches.
    #[serde(default = "d_seed")]
    pub seed: u64,
    /// The last fraction of the data, held out for validation.
    #[serde(default = "d_valid")]
    pub validation_fraction: f64,
    #[serde(default = "d_log")]
    pub log_every: u64,
    #[serde(default = "d_eval")]
    pub eval_every: u64,
    /// Validation batches per evaluation.
    #[serde(default = "d_eval_batches")]
    pub eval_batches: usize,
    #[serde(default = "d_ckpt")]
    pub checkpoint_every: u64,
    /// Fine-tuning only: LoRA's rank r; 0 (the default) fine-tunes every
    /// weight. With r > 0, the pretrained weights are frozen and only a
    /// low-rank adapter is trained (src/lora.rs).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub lora_rank: usize,
    /// LoRA's α: the adapter's update is scaled by α/r. Default 2r.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_alpha: Option<f64>,
    /// The matrices to adapt in every block, from lora::TARGETS. Default
    /// (empty): attention's four, attn.w_q, attn.w_k, attn.w_v, attn.w_o.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lora_targets: Vec<String>,
    /// Which rows of the token embedding the adapter also changes (with a
    /// full, not low-rank, change D): "none" (the default), "markers" (the
    /// special tokens', <|endoftext|> and the chat markers) or "all".
    #[serde(default = "d_embeddings", skip_serializing_if = "is_none_mode")]
    pub lora_embeddings: String,
    /// How each step picks its training items (windows of the corpus, or
    /// conversations): "random" (the default), independently at random,
    /// so some are picked twice and others never; or "epochs", every item
    /// once per epoch in a shuffled order (windows then don't overlap).
    #[serde(default = "d_sampling", skip_serializing_if = "is_random")]
    pub sampling: String,
}

fn d_sampling() -> String {
    "random".into()
}

fn is_random(s: &str) -> bool {
    s == "random"
}

fn d_embeddings() -> String {
    "none".into()
}

fn is_none_mode(s: &str) -> bool {
    s == "none"
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

fn d_warmup() -> u64 {
    100
}
fn d_min_lr() -> f64 {
    0.1
}
fn d_wd() -> f64 {
    0.1
}
fn d_clip() -> f64 {
    1.0
}
fn d_beta1() -> f64 {
    0.9
}
fn d_beta2() -> f64 {
    0.95
}
fn d_seed() -> u64 {
    1
}
fn d_valid() -> f64 {
    0.05
}
fn d_log() -> u64 {
    50
}
fn d_eval() -> u64 {
    250
}
fn d_eval_batches() -> usize {
    10
}
fn d_ckpt() -> u64 {
    500
}

impl TrainingConfig {
    /// The `[training]` section of the config file.
    pub fn load(path: &Path) -> Result<TrainingConfig, String> {
        Self::load_section(path, "training")
    }

    /// A section with the same keys: `[training]`, or `[finetune]` for step 5.
    pub fn load_section(path: &Path, section: &str) -> Result<TrainingConfig, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("can't read config {}: {e}", path.display()))?;
        let mut table: toml::Table =
            toml::from_str(&text).map_err(|e| format!("config {}: {e}", path.display()))?;
        let value = table
            .remove(section)
            .ok_or(format!("config {}: no [{section}] section", path.display()))?;
        let t: TrainingConfig = value
            .try_into()
            .map_err(|e| format!("config {} [{section}]: {e}", path.display()))?;
        t.check()
            .map_err(|e| e.replace("[training]", &format!("[{section}]")))?;
        if section == "training" && (t.lora_rank > 0 || t.lora_embeddings != "none") {
            return Err(format!(
                "config {} [training]: lora_rank and lora_embeddings are for fine-tuning only",
                path.display()
            ));
        }
        Ok(t)
    }

    pub fn check(&self) -> Result<(), String> {
        if self.learning_rate.is_nan() || self.learning_rate <= 0.0 {
            return Err("[training] learning_rate must be positive".into());
        }
        if self.batch_size == 0 || self.steps == 0 {
            return Err("[training] steps and batch_size must be at least 1".into());
        }
        if !(0.0..1.0).contains(&self.validation_fraction) {
            return Err(
                "[training] validation_fraction must be from 0 up to (not including) 1".into(),
            );
        }
        if !(0.0..=1.0).contains(&self.min_lr_fraction) {
            return Err("[training] min_lr_fraction must be from 0 to 1".into());
        }
        if self.log_every == 0 || self.eval_every == 0 || self.checkpoint_every == 0 {
            return Err(
                "[training] log_every, eval_every and checkpoint_every must be at least 1".into(),
            );
        }
        if self.lora_alpha.is_some_and(|a| a.is_nan() || a <= 0.0) {
            return Err("[training] lora_alpha must be positive".into());
        }
        if let Some(t) = self
            .lora_targets
            .iter()
            .find(|t| !crate::lora::TARGETS.contains(&t.as_str()))
        {
            return Err(format!(
                "[training] lora_targets: unknown matrix {t:?} (choose from {})",
                crate::lora::TARGETS.join(", ")
            ));
        }
        if self.sampling != "random" && self.sampling != "epochs" {
            return Err(format!(
                "[training] sampling: {:?} isn't random or epochs",
                self.sampling
            ));
        }
        if !crate::lora::EMBEDDING_MODES.contains(&self.lora_embeddings.as_str()) {
            return Err(format!(
                "[training] lora_embeddings: {:?} isn't one of {}",
                self.lora_embeddings,
                crate::lora::EMBEDDING_MODES.join(", ")
            ));
        }
        Ok(())
    }

    /// LoRA's α (default 2r).
    pub fn alpha(&self) -> f64 {
        self.lora_alpha.unwrap_or(2.0 * self.lora_rank as f64)
    }

    /// The matrices LoRA adapts in each block.
    pub fn targets(&self) -> Vec<String> {
        if self.lora_targets.is_empty() {
            ["attn.w_q", "attn.w_k", "attn.w_v", "attn.w_o"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            self.lora_targets.clone()
        }
    }

    /// The learning rate at a step (counting from 1): a linear warm-up, then
    /// a cosine decay to min_lr_fraction of the peak at the last step.
    pub fn lr(&self, step: u64) -> f64 {
        let peak = self.learning_rate;
        if step <= self.warmup_steps {
            return peak * step as f64 / self.warmup_steps.max(1) as f64;
        }
        let span = (self.steps.saturating_sub(self.warmup_steps)).max(1) as f64;
        let progress = ((step - self.warmup_steps) as f64 / span).min(1.0);
        let floor = peak * self.min_lr_fraction;
        floor + (peak - floor) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

/// Everything training needs: the weights, Adam's running averages for
/// each weight (m, the gradient; v, its square), and how far it has got.
pub struct State {
    /// The weights being trained: the whole model, or with LoRA the adapter.
    pub params: Params<f32>,
    /// With LoRA: the frozen pretrained model the adapter applies to.
    pub frozen: Option<Frozen>,
    pub m: Vec<Mat<f32>>,
    pub v: Vec<Mat<f32>>,
    pub step: u64,
    pub tokens: u64,
}

/// LoRA's frozen base model and the adapter's shape.
pub struct Frozen {
    pub base: Params<f32>,
    pub rank: usize,
    pub alpha: f64,
    /// The token embedding's rows the adapter changes (empty: none).
    pub emb_rows: Vec<usize>,
}

impl State {
    /// The forward pass on a batch, with the adapter if there is one.
    /// Returns it with the leaves of the weights being trained.
    fn forward(
        &self,
        c: &ModelConfig,
        batch: &Batch,
        seq: usize,
    ) -> (crate::gpt::Forward<f32>, Vec<Var>) {
        match &self.frozen {
            None => {
                let f = forward(&self.params, c, &batch.inputs, &batch.targets, seq);
                let leaves = f.leaves.clone();
                (f, leaves)
            }
            Some(fr) => {
                let l = Lora::from_params(&self.params, fr.rank, fr.alpha, &fr.emb_rows)
                    .expect("the adapter was checked when loaded");
                let f = forward_with(&fr.base, Some(&l), c, &batch.inputs, &batch.targets, seq);
                let leaves = f.lora_leaves.clone();
                (f, leaves)
            }
        }
    }

    /// From a checkpoint's tensors: resume if it holds Adam's state
    /// (`adam.m.<name>`, `adam.v.<name>`), otherwise start from step 0.
    pub fn from_checkpoint(f: &safetensors::File) -> Result<State, String> {
        let weights: Vec<Tensor> = f
            .tensors
            .iter()
            .filter(|t| !t.name.starts_with("adam."))
            .cloned()
            .collect();
        let params = Params::<f32>::from_tensors(&weights);
        let find = |prefix: &str| -> Option<Vec<Mat<f32>>> {
            params
                .names
                .iter()
                .zip(&params.mats)
                .map(|(n, w)| {
                    f.tensors
                        .iter()
                        .find(|t| t.name == format!("{prefix}{n}"))
                        .map(|t| Mat::from_vec(w.rows, w.cols, t.data.clone()))
                })
                .collect()
        };
        let zeros = || {
            params
                .mats
                .iter()
                .map(|w| Mat::zeros(w.rows, w.cols))
                .collect()
        };
        let num = |k: &str| {
            f.metadata
                .get(k)
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0)
        };
        Ok(match (find("adam.m."), find("adam.v.")) {
            (Some(m), Some(v)) => State {
                frozen: None,
                step: num("step"),
                tokens: num("tokens_seen"),
                params,
                m,
                v,
            },
            _ => State {
                frozen: None,
                m: zeros(),
                v: zeros(),
                step: 0,
                tokens: 0,
                params,
            },
        })
    }

    /// The weights followed by Adam's state, as named tensors.
    pub fn tensors(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        let to_tensor = |name: &str, m: &Mat<f32>| {
            let shape = if m.rows == 1 {
                vec![m.cols]
            } else {
                vec![m.rows, m.cols]
            };
            Tensor::new(name, &shape, m.data.clone())
        };
        for (n, w) in self.params.names.iter().zip(&self.params.mats) {
            out.push(to_tensor(n, w));
        }
        for (prefix, mats) in [("adam.m.", &self.m), ("adam.v.", &self.v)] {
            for (n, a) in self.params.names.iter().zip(mats) {
                out.push(to_tensor(&format!("{prefix}{n}"), a));
            }
        }
        out
    }
}

/// One training step on one batch. Returns the loss and the gradient's
/// overall length before clipping.
pub fn step(
    state: &mut State,
    c: &ModelConfig,
    t: &TrainingConfig,
    batch: &Batch,
    seq: usize,
) -> (f64, f64) {
    let (mut f, leaves) = state.forward(c, batch, seq);
    let loss = f.tape.value(f.loss).data[0] as f64;
    f.tape.backward(f.loss);
    let grads: Vec<Mat<f32>> = leaves.iter().map(|&l| f.tape.take_grad(l)).collect();

    // The gradient's overall length, all weights together (summed in a fixed
    // order, in f64), and clipping.
    let norm = grads
        .iter()
        .flat_map(|g| g.data.iter())
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    let clip = if t.grad_clip > 0.0 && norm > t.grad_clip {
        t.grad_clip / norm
    } else {
        1.0
    };

    // AdamW. With bias correction for the averages' start at 0.
    state.step += 1;
    state.tokens += batch.inputs.len() as u64;
    let k = state.step as i32;
    let lr = t.lr(state.step);
    let (b1, b2) = (t.beta1, t.beta2);
    let (c1, c2) = (1.0 - b1.powi(k), 1.0 - b2.powi(k));
    for (((w, m), v), g) in state
        .params
        .mats
        .iter_mut()
        .zip(&mut state.m)
        .zip(&mut state.v)
        .zip(&grads)
    {
        // Decay the matrices and embeddings; not biases or γ, β (1 × n).
        let decay = if w.rows > 1 { t.weight_decay } else { 0.0 };
        for i in 0..w.data.len() {
            let gi = g.data[i] as f64 * clip;
            let mi = b1 * m.data[i] as f64 + (1.0 - b1) * gi;
            let vi = b2 * v.data[i] as f64 + (1.0 - b2) * gi * gi;
            m.data[i] = mi as f32;
            v.data[i] = vi as f32;
            let wi = w.data[i] as f64;
            let update = (mi / c1) / ((vi / c2).sqrt() + 1e-8) + decay * wi;
            w.data[i] = (wi - lr * update) as f32;
        }
    }
    (loss, norm)
}

/// The mean loss over the validation batches (forward only).
pub fn validation_loss(
    state: &State,
    c: &ModelConfig,
    batches: &[Batch],
    seq: usize,
) -> Option<f64> {
    if batches.is_empty() {
        return None;
    }
    let total: f64 = batches
        .iter()
        .map(|b| {
            let (f, _) = state.forward(c, b, seq);
            f.tape.value(f.loss).data[0] as f64
        })
        .sum();
    Some(total / batches.len() as f64)
}

/// What a checkpoint records about how it was made.
pub struct Provenance {
    /// "train" or "finetune": the checkpoint's kind, and the command that
    /// wrote it.
    pub kind: &'static str,
    /// The input checkpoint's metadata, carried forward (model, tokenizer…).
    pub base: BTreeMap<String, String>,
    pub training_toml: String,
    pub data_file: String,
    pub data_sha256: String,
    pub parent_file: String,
    pub parent_sha256: String,
}

/// Write a checkpoint safely: to a temporary file, then renamed over the
/// output, so an interrupted write never leaves a damaged checkpoint.
pub fn save(
    path: &Path,
    state: &State,
    prov: &Provenance,
    train_loss: Option<f64>,
    valid_loss: Option<f64>,
) -> Result<(), String> {
    let mut meta = prov.base.clone();
    meta.insert("kind".into(), prov.kind.into());
    meta.insert("training".into(), prov.training_toml.clone());
    meta.insert("step".into(), state.step.to_string());
    meta.insert("tokens_seen".into(), state.tokens.to_string());
    if let Some(l) = train_loss {
        meta.insert("train_loss".into(), format!("{l:.6}"));
    }
    if let Some(l) = valid_loss {
        meta.insert("valid_loss".into(), format!("{l:.6}"));
    }
    meta.insert("data_file".into(), prov.data_file.clone());
    meta.insert("data_sha256".into(), prov.data_sha256.clone());
    meta.insert("parent_file".into(), prov.parent_file.clone());
    meta.insert("parent_sha256".into(), prov.parent_sha256.clone());
    meta.insert(
        "producer".into(),
        format!("transformer {} {}", prov.kind, env!("CARGO_PKG_VERSION")),
    );
    let tensors = state.tensors();
    let tmp = path.with_extension("safetensors.tmp");
    safetensors::write(&tmp, &tensors, &meta)?;
    std::fs::rename(&tmp, path)
        .map_err(|e| format!("can't rename {} to {}: {e}", tmp.display(), path.display()))?;
    Ok(())
}

/// Check that the data belongs to the model: same vocabulary size, made by
/// the same tokenizer (fingerprints equal).
pub fn check_data(corpus: &Corpus, vocab: usize, fingerprint_hex: &str) -> Result<(), String> {
    if corpus.vocab_size != vocab {
        return Err(format!(
            "the data's V ({}) differs from the model's ({vocab})",
            corpus.vocab_size
        ));
    }
    let fp = crate::tokenizer::hex(&corpus.fingerprint);
    if fp != fingerprint_hex {
        return Err(format!(
            "the data was encoded by a different tokenizer (fingerprint {fp}, model's {fingerprint_hex})"
        ));
    }
    Ok(())
}

/// A duration for the log, to the precision that matters: "2h05m", "14m",
/// "40s", or "—" for none left.
pub fn duration(secs: f64) -> String {
    let s = secs.round() as u64;
    match s {
        0 => "—".into(),
        1..=59 => format!("{s}s"),
        60..=3599 => format!("{}m", (s + 30) / 60),
        _ => {
            let m = (s + 30) / 60;
            format!("{}h{:02}m", m / 60, m % 60)
        }
    }
}

/// A number of passes over the data, for the log: one decimal from 0.1 up,
/// two significant digits below ("0.0037", where one decimal would say 0.0).
pub fn passes(x: f64) -> String {
    if x >= 0.1 || x <= 0.0 || !x.is_finite() {
        return format!("{x:.1}");
    }
    let decimals = (-x.log10()).floor() as usize + 2;
    format!("{x:.decimals$}")
}

/// Report progress after a window of steps.
pub struct Window {
    pub loss_sum: f64,
    pub norm_sum: f64,
    pub steps: u64,
    pub start: Instant,
}

impl Window {
    pub fn new() -> Window {
        Window {
            loss_sum: 0.0,
            norm_sum: 0.0,
            steps: 0,
            start: Instant::now(),
        }
    }
}

/// Pretraining (step 4): train on random windows of `corpus_ids` until
/// `t.steps`; `report` receives each log line. Returns the final state.
pub fn run(
    state: State,
    c: &ModelConfig,
    t: &TrainingConfig,
    corpus_ids: &[u32],
    output: &Path,
    prov: &Provenance,
    report: impl FnMut(&str),
) -> Result<State, String> {
    let seq = c.context;
    let split = data::split(corpus_ids, t.validation_fraction, seq)?;
    let valid = data::valid_batches(split.valid, t.eval_batches, t.batch_size, seq);
    let order = (t.sampling == "epochs")
        .then(|| data::EpochOrder::new(data::epoch_windows(split.train.len(), seq), t.seed));
    let batches =
        |step: u64| data::train_batch(split.train, t.batch_size, seq, t.seed, step, order.as_ref());
    run_with(state, c, t, batches, &valid, output, prov, report)
}

/// The training loop, for any source of batches: `batches(step)` gives the
/// batch for a step (it must depend only on the step, so a resumed run
/// sees the same batches); `valid` is measured every `t.eval_every` steps.
#[allow(clippy::too_many_arguments)]
pub fn run_with(
    mut state: State,
    c: &ModelConfig,
    t: &TrainingConfig,
    batches: impl Fn(u64) -> Batch,
    valid: &[Batch],
    output: &Path,
    prov: &Provenance,
    mut report: impl FnMut(&str),
) -> Result<State, String> {
    let seq = c.context;
    let tokens_per_step = (t.batch_size * seq) as f64;
    let mut win = Window::new();
    let mut last_loss = None;
    let mut last_valid = None;
    report("   step    loss   valid      lr  grad norm  tokens/s  remaining");
    while state.step < t.steps {
        let batch = batches(state.step + 1);
        let (loss, norm) = step(&mut state, c, t, &batch, seq);
        if !loss.is_finite() {
            return Err(format!("the loss became {loss} at step {}", state.step));
        }
        win.loss_sum += loss;
        win.norm_sum += norm;
        win.steps += 1;
        let s = state.step;
        let last = s == t.steps;
        let evaluate = s.is_multiple_of(t.eval_every) || last;
        if evaluate {
            last_valid = validation_loss(&state, c, valid, seq);
        }
        if s.is_multiple_of(t.log_every) || last || evaluate {
            let mean = win.loss_sum / win.steps as f64;
            let secs = win.start.elapsed().as_secs_f64();
            let tps = tokens_per_step * win.steps as f64 / secs;
            // From this window's pace (it includes any validation run in it).
            let remaining = duration((t.steps - s) as f64 * secs / win.steps as f64);
            let v = if evaluate {
                last_valid.map_or("     —".into(), |l| format!("{l:7.3}"))
            } else {
                "       ".into()
            };
            report(&format!(
                "  {s:>5}   {mean:.3} {v}  {:.2e}  {:9.3}  {tps:8.0}  {remaining:>9}",
                t.lr(s),
                win.norm_sum / win.steps as f64
            ));
            last_loss = Some(mean);
            win = Window::new();
        }
        if s.is_multiple_of(t.checkpoint_every) || last {
            save(output, &state, prov, last_loss, last_valid)?;
        }
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model;

    fn tiny_model() -> ModelConfig {
        ModelConfig {
            d_model: 32,
            layers: 2,
            heads: 2,
            context: 16,
            d_ff: 64,
            seed: 2,
        }
    }

    fn training(steps: u64) -> TrainingConfig {
        toml::from_str::<TrainingConfig>(&format!(
            "learning_rate = 0.003\nsteps = {steps}\nbatch_size = 8\nwarmup_steps = 10\nvalidation_fraction = 0.0\nlog_every = 1000\neval_every = 1000\ncheckpoint_every = 1000\n"
        ))
        .unwrap()
    }

    fn fresh(c: &ModelConfig, vocab: usize) -> State {
        let f = safetensors::File {
            tensors: model::init(c, vocab),
            metadata: BTreeMap::new(),
        };
        State::from_checkpoint(&f).unwrap()
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(duration(0.2), "—");
        assert_eq!(duration(40.0), "40s");
        assert_eq!(duration(14.0 * 60.0 + 10.0), "14m");
        assert_eq!(duration(2.0 * 3600.0 + 5.0 * 60.0), "2h05m");
        assert_eq!(duration(23.0 * 3600.0 + 59.0 * 60.0 + 50.0), "24h00m");
    }

    #[test]
    fn passes_show_small_fractions() {
        assert_eq!(passes(0.27), "0.3");
        assert_eq!(passes(1.6), "1.6");
        assert_eq!(passes(0.0037), "0.0037");
        assert_eq!(passes(0.05), "0.050");
        assert_eq!(passes(0.0), "0.0");
    }

    #[test]
    fn learning_rate_schedule() {
        let t = toml::from_str::<TrainingConfig>(
            "learning_rate = 0.01\nsteps = 1100\nbatch_size = 1\nwarmup_steps = 100\n",
        )
        .unwrap();
        assert!((t.lr(50) - 0.005).abs() < 1e-12); // halfway up the warm-up
        assert!((t.lr(100) - 0.01).abs() < 1e-12); // the peak
        assert!((t.lr(600) - 0.0055).abs() < 1e-12); // halfway down: (peak + floor) / 2
        assert!((t.lr(1100) - 0.001).abs() < 1e-12); // the floor, 10% of the peak
    }

    /// A tiny model on a short repeating text should memorize it: the loss
    /// must fall from about ln V to near 0.
    #[test]
    fn a_tiny_model_memorizes_a_repeating_text() {
        let c = tiny_model();
        let vocab = 12;
        let ids: Vec<u32> = (0..400)
            .map(|i| [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5][i % 11])
            .collect();
        let t = training(300);
        let mut s = fresh(&c, vocab);
        let mut losses = Vec::new();
        for step_no in 1..=t.steps {
            let b = data::train_batch(&ids, 8, 16, t.seed, step_no, None);
            losses.push(step(&mut s, &c, &t, &b, 16).0);
        }
        assert!(
            (losses[0] - (vocab as f64).ln()).abs() < 0.1,
            "start {}",
            losses[0]
        );
        assert!(
            *losses.last().unwrap() < 0.2,
            "end {}",
            losses.last().unwrap()
        );
    }

    /// Training 20 steps without stopping gives exactly the same weights and
    /// Adam state as 10 steps, a checkpoint, a resume and 10 more: the
    /// checkpoint holds everything training needs.
    #[test]
    fn resuming_from_a_checkpoint_is_exact() {
        let c = tiny_model();
        let vocab = 30;
        let ids: Vec<u32> = (0..2000u32)
            .map(|i| (i * 7 + i / 13) % vocab as u32)
            .collect();
        let dir =
            std::env::temp_dir().join(format!("mini_transformer_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let prov = Provenance {
            kind: "train",
            base: BTreeMap::new(),
            training_toml: String::new(),
            data_file: "test".into(),
            data_sha256: String::new(),
            parent_file: String::new(),
            parent_sha256: String::new(),
        };

        let straight = run(
            fresh(&c, vocab),
            &c,
            &training(20),
            &ids,
            &dir.join("a.safetensors"),
            &prov,
            |_| {},
        )
        .unwrap();

        let half = dir.join("b.safetensors");
        run(
            fresh(&c, vocab),
            &c,
            &training(10),
            &ids,
            &half,
            &prov,
            |_| {},
        )
        .unwrap();
        let loaded = safetensors::read(&half).unwrap();
        let resumed_state = State::from_checkpoint(&loaded).unwrap();
        assert_eq!(resumed_state.step, 10);
        let resumed = run(
            resumed_state,
            &c,
            &training(20),
            &ids,
            &dir.join("c.safetensors"),
            &prov,
            |_| {},
        )
        .unwrap();

        assert_eq!(straight.step, resumed.step);
        assert_eq!(straight.tokens, resumed.tokens);
        assert_eq!(straight.tensors(), resumed.tensors()); // bit for bit
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// LoRA training (with an embedding change) changes only the adapter,
    /// lowers the loss, and resumes
    /// exactly: 10 steps + checkpoint + 10 steps = 20 steps, bit for bit.
    #[test]
    fn lora_trains_the_adapter_only_and_resumes_exactly() {
        let c = tiny_model();
        let vocab = 30;
        let ids: Vec<u32> = (0..2000u32)
            .map(|i| (i * 7 + i / 13) % vocab as u32)
            .collect();
        let pretrained = run(
            fresh(&c, vocab),
            &c,
            &training(40),
            &ids,
            &std::env::temp_dir().join(format!("mt_lora_base_{}", std::process::id())),
            &test_prov(),
            |_| {},
        )
        .unwrap();
        let base = pretrained.params.clone();
        let lora_state = || {
            let l = Lora::new(
                &c,
                &base,
                2,
                4.0,
                &["attn.w_q".into(), "attn.w_v".into()],
                &[29, 3], // and two embedding rows, out of order
                1,
            );
            let p = l.to_params();
            let zeros = || p.mats.iter().map(|m| Mat::zeros(m.rows, m.cols)).collect();
            State {
                m: zeros(),
                v: zeros(),
                params: p,
                frozen: Some(Frozen {
                    base: base.clone(),
                    rank: 2,
                    alpha: 4.0,
                    emb_rows: vec![29, 3],
                }),
                step: 0,
                tokens: 0,
            }
        };
        let dir = std::env::temp_dir().join(format!("mt_lora_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut prov = test_prov();
        prov.kind = "lora";
        let before = validation_loss(
            &lora_state(),
            &c,
            &[data::train_batch(&ids, 8, 16, 9, 1, None)],
            16,
        )
        .unwrap();
        let straight = run(
            lora_state(),
            &c,
            &training(20),
            &ids,
            &dir.join("a"),
            &prov,
            |_| {},
        )
        .unwrap();
        let after = validation_loss(
            &straight,
            &c,
            &[data::train_batch(&ids, 8, 16, 9, 1, None)],
            16,
        )
        .unwrap();
        assert!(after < before, "loss {before} -> {after}");
        assert_eq!(straight.frozen.as_ref().unwrap().base.mats, base.mats); // untouched
        let d = straight.params.mats.last().unwrap(); // the embedding change trained too
        assert_eq!(
            straight.params.names.last().unwrap(),
            crate::lora::EMBEDDING_DELTA
        );
        assert!(d.data.iter().any(|&x| x != 0.0));

        let half = dir.join("b");
        run(lora_state(), &c, &training(10), &ids, &half, &prov, |_| {}).unwrap();
        let mut resumed = State::from_checkpoint(&safetensors::read(&half).unwrap()).unwrap();
        resumed.frozen = lora_state().frozen;
        assert_eq!(resumed.step, 10);
        let resumed = run(
            resumed,
            &c,
            &training(20),
            &ids,
            &dir.join("c"),
            &prov,
            |_| {},
        )
        .unwrap();
        assert_eq!(straight.tensors(), resumed.tensors()); // adapter and Adam's state, bit for bit
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn test_prov() -> Provenance {
        Provenance {
            kind: "train",
            base: BTreeMap::new(),
            training_toml: String::new(),
            data_file: "test".into(),
            data_sha256: String::new(),
            parent_file: String::new(),
            parent_sha256: String::new(),
        }
    }

    /// A golden regression test for training as a whole: a fixed model, seed
    /// and corpus (tests/data/tiny.ids, a real tokenized text) must reproduce
    /// the recorded losses. The other tests check pieces (gradients, the
    /// schedule, resuming); this one catches a change that quietly makes
    /// training different, such as a learning-rate or batching slip, which
    /// those would all pass. The tolerance allows for summation-order
    /// rounding (another BLAS library, another thread count) and nothing
    /// more: a real change to training moves these by far more.
    ///
    /// If a change to training is intended, rerun with
    /// `cargo test golden -- --nocapture`, check the new losses are
    /// plausible, and update GOLDEN.
    #[test]
    fn golden_losses_on_the_fixture_corpus() {
        const GOLDEN: [(u64, f64); 4] = [
            (1, 5.757092),
            (20, 4.652846),
            (40, 4.190655),
            (60, 4.424769),
        ];
        const TOLERANCE: f64 = 2e-3;
        let corpus = Corpus::decode(include_bytes!("../tests/data/tiny.ids")).unwrap();
        let c = tiny_model();
        let t = training(60);
        let mut s = fresh(&c, corpus.vocab_size);
        let mut losses = Vec::new();
        for step_no in 1..=t.steps {
            let b = data::train_batch(&corpus.ids, 8, c.context, t.seed, step_no, None);
            losses.push(step(&mut s, &c, &t, &b, c.context).0);
        }
        let got: Vec<(u64, f64)> = GOLDEN
            .iter()
            .map(|&(n, _)| (n, losses[n as usize - 1]))
            .collect();
        println!("losses at steps {got:?}");
        for (&(n, want), &(_, have)) in GOLDEN.iter().zip(&got) {
            assert!(
                (have - want).abs() < TOLERANCE,
                "step {n}: loss {have:.6}, golden {want:.6} (all: {got:?})"
            );
        }
    }

    #[test]
    fn refuses_data_from_another_tokenizer() {
        let corpus = Corpus::decode(include_bytes!("../tests/data/tiny.ids")).unwrap();
        let fp = crate::tokenizer::hex(&corpus.fingerprint);
        assert!(check_data(&corpus, 317, &fp).is_ok());
        assert!(check_data(&corpus, 318, &fp).is_err());
        assert!(check_data(&corpus, 317, &"0".repeat(64)).is_err());
    }
}
