//! The model and its training: two embedding tables, negative sampling,
//! subsampling of frequent words, and the skip-gram and CBOW updates.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use crate::corpus::Corpus;
use crate::matrix::{dot, sigmoid, softplus, Matrix, Rng};

/// Which prediction task to train on.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Arch {
    /// Predict each context word from the center word.
    #[value(name = "skipgram")]
    SkipGram,
    /// Predict the center word from the average of its context words.
    #[value(name = "cbow")]
    Cbow,
}

pub struct Config {
    pub arch: Arch,
    pub window: usize,
    pub negative: usize,
    pub epochs: usize,
    pub alpha: f64,
    pub sample: f64,
    pub threads: usize,
    pub seed: u64,
    pub quiet: bool,
}

/// The model's weights: two V x d tables.
///
/// `input` holds the word vectors. Row i is the embedding of word i, the same
/// role as an LLM's embedding table, and it is what gets saved at the end.
/// `output` holds a second "context" vector per word, used only to score
/// predictions during training and then discarded.
pub struct Model {
    pub input: Matrix,
    pub output: Matrix,
}

impl Model {
    /// Fresh weights, as in the original word2vec: input vectors uniform in
    /// [-0.5/d, 0.5/d), output vectors all zero. The input side must be
    /// random to break symmetry: if every row started equal, every word
    /// would get identical updates and the rows would stay equal forever.
    /// The output side can start at zero because the random input side
    /// already makes every word's gradient different.
    pub fn new(vocab_size: usize, dim: usize, seed: u64) -> Model {
        let mut rng = Rng::new(seed);
        Model {
            input: Matrix::uniform(vocab_size, dim, 0.5 / dim as f32, &mut rng),
            output: Matrix::zeros(vocab_size, dim),
        }
    }
}

/// Draws "negative" words for negative sampling.
///
/// Each word's probability is proportional to count^0.75. The power 0.75
/// (found empirically by Mikolov et al.) flattens the distribution: frequent
/// words are still drawn most often, but rare words more often than their raw
/// frequency. Sampling uses a large lookup table in which each word fills a
/// share of slots equal to its probability, so a draw is one random index.
pub struct NegativeSampler {
    table: Vec<u32>,
}

impl NegativeSampler {
    const TABLE_SIZE: usize = 10_000_000;

    pub fn new(counts: &[u64]) -> NegativeSampler {
        let weights: Vec<f64> = counts.iter().map(|&c| (c as f64).powf(0.75)).collect();
        let total: f64 = weights.iter().sum();
        let mut table = Vec::with_capacity(Self::TABLE_SIZE);
        let mut cumulative = 0.0;
        for (id, w) in weights.iter().enumerate() {
            cumulative += w / total;
            let end = ((cumulative * Self::TABLE_SIZE as f64) as usize).min(Self::TABLE_SIZE);
            table.resize(end.max(table.len()), id as u32);
        }
        table.resize(Self::TABLE_SIZE, (counts.len() - 1) as u32);
        NegativeSampler { table }
    }

    pub fn sample(&self, rng: &mut Rng) -> u32 {
        self.table[rng.below(self.table.len())]
    }
}

/// Probability of keeping each occurrence of each word (subsampling).
///
/// Very frequent words ("the", "of") carry little information about their
/// neighbors and dominate the training pairs. Each occurrence of a word with
/// corpus frequency f is kept with probability (sqrt(f/t) + 1) * t/f, capped
/// at 1, where t is the `sample` threshold. Words rarer than t are always
/// kept; "the" is mostly dropped. This is the formula in the original C code.
/// `sample` = 0 keeps everything.
pub fn keep_probabilities(counts: &[u64], sample: f64) -> Vec<f32> {
    let total: u64 = counts.iter().sum();
    counts
        .iter()
        .map(|&c| {
            if sample <= 0.0 {
                return 1.0;
            }
            let f = c as f64 / total as f64;
            (((f / sample).sqrt() + 1.0) * sample / f).min(1.0) as f32
        })
        .collect()
}

/// The loss for one scored word: -ln sigmoid(score) for the true target
/// (label 1), -ln(1 - sigmoid(score)) = -ln sigmoid(-score) for a negative.
fn prediction_loss(score: f32, label: f32) -> f64 {
    if label == 1.0 {
        softplus(-f64::from(score))
    } else {
        softplus(f64::from(score))
    }
}

/// What one epoch did, summed over threads.
#[derive(Default)]
pub struct EpochStats {
    /// Sum of the negative-sampling loss over all predictions.
    pub loss: f64,
    /// Number of predictions (one per positive target).
    pub predictions: u64,
    /// Words that survived subsampling.
    pub words_kept: u64,
}

impl EpochStats {
    fn add(&mut self, other: EpochStats) {
        self.loss += other.loss;
        self.predictions += other.predictions;
        self.words_kept += other.words_kept;
    }
}

/// Runs training epochs over a corpus, updating a model in place.
pub struct Trainer<'a> {
    cfg: &'a Config,
    model: &'a Model,
    corpus: &'a Corpus,
    sampler: NegativeSampler,
    keep: Vec<f32>,
    /// Words processed so far, across all threads and epochs. Drives the
    /// learning-rate schedule and the progress display.
    progress: AtomicU64,
    total_work: u64,
    start: Instant,
}

/// Per-thread scratch space, so the inner loop never allocates.
struct Scratch {
    /// The hidden layer: the vector that makes the prediction.
    h: Vec<f32>,
    /// Accumulated gradient for the hidden layer.
    grad_h: Vec<f32>,
    /// A row of the output table, then reused as that row's update.
    u: Vec<f32>,
    /// The current sentence after subsampling.
    sentence: Vec<u32>,
}

impl<'a> Trainer<'a> {
    pub fn new(cfg: &'a Config, model: &'a Model, corpus: &'a Corpus) -> Trainer<'a> {
        Trainer {
            cfg,
            model,
            corpus,
            sampler: NegativeSampler::new(&corpus.vocab.counts),
            keep: keep_probabilities(&corpus.vocab.counts, cfg.sample),
            progress: AtomicU64::new(0),
            total_work: (cfg.epochs * corpus.tokens.len()) as u64,
            start: Instant::now(),
        }
    }

    /// Expected fraction of word occurrences kept by subsampling.
    pub fn expected_keep_fraction(&self) -> f64 {
        let counts = &self.corpus.vocab.counts;
        let kept: f64 = counts
            .iter()
            .zip(&self.keep)
            .map(|(&c, &p)| c as f64 * p as f64)
            .sum();
        kept / self.corpus.vocab.total() as f64
    }

    /// One pass over the corpus. Sentences are dealt round-robin to the
    /// threads, which all update the same shared model (Hogwild!).
    pub fn run_epoch(&self, epoch: usize) -> EpochStats {
        let mut total = EpochStats::default();
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..self.cfg.threads)
                .map(|t| s.spawn(move || self.worker(epoch, t)))
                .collect();
            for h in handles {
                total.add(h.join().expect("training thread panicked"));
            }
        });
        if !self.cfg.quiet {
            eprint!("\r{:79}\r", ""); // erase the progress line
        }
        total
    }

    /// Learning rate: decays linearly from alpha to alpha/10,000 over the
    /// whole run, as in the original word2vec.
    fn learning_rate(&self, done: u64) -> f32 {
        let remaining = 1.0 - done as f64 / self.total_work as f64;
        (self.cfg.alpha * remaining.max(1e-4)) as f32
    }

    fn worker(&self, epoch: usize, thread: usize) -> EpochStats {
        let dim = self.model.input.dim();
        let mut rng =
            Rng::new(self.cfg.seed ^ ((epoch as u64) << 32) ^ ((thread as u64 + 1) * 0x5851_F42D));
        let mut scratch = Scratch {
            h: vec![0.0; dim],
            grad_h: vec![0.0; dim],
            u: vec![0.0; dim],
            sentence: Vec::with_capacity(crate::corpus::MAX_SENTENCE),
        };
        let mut stats = EpochStats::default();

        let sentences = &self.corpus.sentences;
        for (n, &(start, end)) in sentences
            .iter()
            .enumerate()
            .skip(thread)
            .step_by(self.cfg.threads)
        {
            let done = self.progress.fetch_add((end - start) as u64, Relaxed);
            let alpha = self.learning_rate(done);
            if thread == 0 && !self.cfg.quiet && n.is_multiple_of(64 * self.cfg.threads) {
                self.show_progress(epoch, done, alpha);
            }

            // Subsample: drop some occurrences of frequent words.
            scratch.sentence.clear();
            for &w in &self.corpus.tokens[start..end] {
                if rng.unit() < self.keep[w as usize] {
                    scratch.sentence.push(w);
                }
            }
            stats.words_kept += scratch.sentence.len() as u64;

            for pos in 0..scratch.sentence.len() {
                // Dynamic window: the effective size is uniform in
                // 1..=window, so nearer words are used more often.
                let span = self.cfg.window - rng.below(self.cfg.window);
                let lo = pos.saturating_sub(span);
                let hi = (pos + span).min(scratch.sentence.len() - 1);
                match self.cfg.arch {
                    Arch::SkipGram => {
                        for c in lo..=hi {
                            if c != pos {
                                let (center, context) =
                                    (scratch.sentence[pos], scratch.sentence[c]);
                                stats.loss +=
                                    self.skip_gram(center, context, alpha, &mut scratch, &mut rng);
                                stats.predictions += 1;
                            }
                        }
                    }
                    Arch::Cbow => {
                        if hi > lo {
                            stats.loss += self.cbow(pos, lo, hi, alpha, &mut scratch, &mut rng);
                            stats.predictions += 1;
                        }
                    }
                }
            }
        }
        stats
    }

    /// Skip-gram: use the center word's vector to predict one context word.
    ///
    /// (The C implementation swaps the roles, using the context word to
    /// predict the center word. Over a whole corpus every pair appears both
    /// ways, so the two are equivalent.)
    fn skip_gram(
        &self,
        center: u32,
        context: u32,
        alpha: f32,
        s: &mut Scratch,
        rng: &mut Rng,
    ) -> f64 {
        self.model.input.read_row(center as usize, &mut s.h);
        s.grad_h.fill(0.0);
        let loss = self.negative_sampling(context, alpha, s, rng);
        self.model.input.add_row(center as usize, &s.grad_h);
        loss
    }

    /// CBOW: average the context words' vectors and use that to predict the
    /// center word. Every context word then receives the full gradient.
    fn cbow(
        &self,
        pos: usize,
        lo: usize,
        hi: usize,
        alpha: f32,
        s: &mut Scratch,
        rng: &mut Rng,
    ) -> f64 {
        s.h.fill(0.0);
        let mut row = std::mem::take(&mut s.u);
        for c in (lo..=hi).filter(|&c| c != pos) {
            self.model.input.read_row(s.sentence[c] as usize, &mut row);
            for (h, r) in s.h.iter_mut().zip(&row) {
                *h += r;
            }
        }
        s.u = row;
        let n = (hi - lo) as f32; // number of context words (pos excluded)
        for h in s.h.iter_mut() {
            *h /= n;
        }
        s.grad_h.fill(0.0);
        let loss = self.negative_sampling(s.sentence[pos], alpha, s, rng);
        for c in (lo..=hi).filter(|&c| c != pos) {
            self.model.input.add_row(s.sentence[c] as usize, &s.grad_h);
        }
        loss
    }

    /// One prediction trained by negative sampling.
    ///
    /// The hidden vector h (already in `s.h`) should score high against the
    /// true target's output vector and low against `negative` random words:
    /// push sigmoid(h . u_target) toward 1 and sigmoid(h . u_neg) toward 0.
    /// For each of these words, with label 1 for the target and 0 for a
    /// negative, the gradient step is
    ///     g       = alpha * (label - sigmoid(h . u))
    ///     grad_h += g * u        (applied to h by the caller)
    ///     u      += g * h
    /// This replaces a softmax over the whole vocabulary, which would cost
    /// V dot products per prediction, with 1 + negative of them.
    /// Returns the loss: -ln sigmoid(h . u_target) - sum ln sigmoid(-h . u_neg).
    fn negative_sampling(&self, target: u32, alpha: f32, s: &mut Scratch, rng: &mut Rng) -> f64 {
        let mut loss = 0.0;
        for k in 0..=self.cfg.negative {
            let Some((word, label)) = self.draw(k, target, rng) else {
                continue;
            };
            self.model.output.read_row(word as usize, &mut s.u);
            let score = dot(&s.h, &s.u);
            loss += prediction_loss(score, label);
            let g = alpha * (label - sigmoid(score));
            for (gh, u) in s.grad_h.iter_mut().zip(&s.u) {
                *gh += g * u;
            }
            for (u, h) in s.u.iter_mut().zip(&s.h) {
                *u = g * h; // s.u now holds this row's update
            }
            self.model.output.add_row(word as usize, &s.u);
        }
        loss
    }

    /// The k-th word scored in a prediction: for k = 0 the true target with
    /// label 1, otherwise a random negative with label 0. A negative that
    /// happens to equal the target is skipped (None).
    fn draw(&self, k: usize, target: u32, rng: &mut Rng) -> Option<(u32, f32)> {
        if k == 0 {
            return Some((target, 1.0));
        }
        let w = self.sampler.sample(rng);
        (w != target).then_some((w, 0.0))
    }

    /// Average loss on a fixed sample of predictions, with the weights frozen.
    ///
    /// The loss reported during training is measured *while* the weights
    /// change. With a high learning rate the model briefly adapts to the
    /// article it is reading (text8 is Wikipedia articles in order), so a
    /// word repeated within an article soon becomes easy to predict, and the
    /// running loss benefits. As the learning rate falls, that short-term
    /// adaptation fades and the running loss can rise even while the vectors
    /// improve. This measurement removes the effect: the same predictions,
    /// with the same negative samples, are scored after every epoch and
    /// nothing is updated. It is word2vec's equivalent of an evaluation loss.
    pub fn eval_loss(&self) -> f64 {
        const PREDICTIONS: usize = 100_000;
        const SEED: u64 = 0x5EED;
        let tokens = &self.corpus.tokens;
        if tokens.len() < 2 {
            return f64::NAN;
        }
        let dim = self.model.input.dim();
        let (mut h, mut row) = (vec![0.0; dim], vec![0.0; dim]);
        let mut rng = Rng::new(SEED);
        let mut loss = 0.0;
        let mut done = 0;
        while done < PREDICTIONS {
            // A random position, subsampled as in training.
            let pos = rng.below(tokens.len());
            if rng.unit() >= self.keep[tokens[pos] as usize] {
                continue;
            }
            let span = self.cfg.window - rng.below(self.cfg.window);
            let lo = pos.saturating_sub(span);
            let hi = (pos + span).min(tokens.len() - 1);
            let target = match self.cfg.arch {
                Arch::SkipGram => {
                    // One context word, chosen at random from the window.
                    let mut c = lo + rng.below(hi - lo);
                    if c >= pos {
                        c += 1;
                    }
                    self.model.input.read_row(tokens[pos] as usize, &mut h);
                    tokens[c]
                }
                Arch::Cbow => {
                    h.fill(0.0);
                    for c in (lo..=hi).filter(|&c| c != pos) {
                        self.model.input.read_row(tokens[c] as usize, &mut row);
                        h.iter_mut().zip(&row).for_each(|(h, r)| *h += r);
                    }
                    let n = (hi - lo) as f32;
                    h.iter_mut().for_each(|h| *h /= n);
                    tokens[pos]
                }
            };
            for k in 0..=self.cfg.negative {
                if let Some((word, label)) = self.draw(k, target, &mut rng) {
                    self.model.output.read_row(word as usize, &mut row);
                    loss += prediction_loss(dot(&h, &row), label);
                }
            }
            done += 1;
        }
        loss / PREDICTIONS as f64
    }

    fn show_progress(&self, epoch: usize, done: u64, alpha: f32) {
        let pct = 100.0 * done as f64 / self.total_work as f64;
        let rate = done as f64 / self.start.elapsed().as_secs_f64().max(1e-9);
        eprint!(
            "\r  epoch {}/{}  overall {:5.1}%  lr {:.6}  {:6.0}k words/s ",
            epoch,
            self.cfg.epochs,
            pct,
            alpha,
            rate / 1000.0
        );
        let _ = std::io::stderr().flush();
    }
}
