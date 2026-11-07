//! Copied from tiny-diffusion-in-rust (diffusion/src/classify.rs), unchanged
//! but for this note, so drawing is measured with the same instrument
//! (PLAN.md, Drawing, part 3); "stage 4" below is that project's.
//!
//! The measuring instrument of stage 4 (its PLAN.md, Stage 4, What it adds): a
//! classifier that says which digit an image looks like, and how sure it
//! is, so generated images can be counted rather than judged by eye.
//!
//! ```text
//! h      = GELU(pixels·W₁ + b₁)        784 → 256
//! h      = GELU(h·W₂ + b₂)             256 → 256
//! logits = h·W₃ + b₃                   256 → 10, one score per class
//! p      = softmax(logits)             a probability for each class
//! ```
//!
//! Trained by cross-entropy on the data set's labelled images, and checked
//! on its separate test images. Generic over the float type like the tape:
//! f32 to train, f64 for the gradient check.

// The measures (agreement, confident, variety) are used by part 3's
// commands, which come after the tokenizer checks that use the rest.
#![allow(dead_code)]

use crate::image::Images;
use crate::optim::Adam;
use crate::rng::Rng;
use crate::safetensors::{File, Tensor};
use crate::tape::{gelu, Tape, Var};
use crate::tensor::{Float, Mat};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassifierConfig {
    /// Values per image: 784 for MNIST.
    pub inputs: usize,
    /// Neurons per hidden layer.
    pub hidden: usize,
    /// Hidden layers.
    pub layers: usize,
    /// Classes: 10 digits.
    pub classes: usize,
}

impl ClassifierConfig {
    pub fn parameters(&self) -> usize {
        let (i, h, c) = (self.inputs, self.hidden, self.classes);
        (i * h + h) + (self.layers - 1) * (h * h + h) + (h * c + c)
    }
}

/// The weights, in order: W₁, b₁, each further hidden layer's W and b, then
/// the output layer's.
pub struct Classifier<T> {
    pub cfg: ClassifierConfig,
    pub params: Vec<Mat<T>>,
}

impl<T: Float> Classifier<T> {
    /// Random weights, each W normal with standard deviation 1/√(inputs);
    /// biases zero.
    pub fn new(cfg: ClassifierConfig, seed: u64) -> Classifier<T> {
        assert!(cfg.layers >= 1, "at least one hidden layer");
        let mut rng = Rng::new(seed);
        let mut normal = |rows: usize, cols: usize| {
            let s = 1.0 / (rows as f64).sqrt();
            Mat::from_vec(
                rows,
                cols,
                (0..rows * cols)
                    .map(|_| T::from_f64(s * rng.normal()))
                    .collect(),
            )
        };
        let (i, h, c) = (cfg.inputs, cfg.hidden, cfg.classes);
        let mut params = vec![normal(i, h), Mat::zeros(1, h)];
        for _ in 1..cfg.layers {
            params.push(normal(h, h));
            params.push(Mat::zeros(1, h));
        }
        params.push(normal(h, c));
        params.push(Mat::zeros(1, c));
        Classifier { cfg, params }
    }

    /// The scores (logits) on a tape, and the parameters' variables.
    pub fn forward(&self, tape: &mut Tape<T>, images: Mat<T>) -> (Var, Vec<Var>) {
        let vars: Vec<Var> = self.params.iter().map(|p| tape.leaf(p.clone())).collect();
        let mut h = tape.leaf(images);
        let layers = vars.len() / 2;
        for l in 0..layers {
            let z = tape.matmul(h, vars[2 * l]);
            h = tape.add_row(z, vars[2 * l + 1]);
            if l + 1 < layers {
                h = tape.gelu(h);
            }
        }
        (h, vars)
    }

    /// The same scores without a tape.
    pub fn logits(&self, images: &Mat<T>) -> Mat<T> {
        let layers = self.params.len() / 2;
        let mut h = images.clone();
        for l in 0..layers {
            let mut z = h.matmul(&self.params[2 * l]);
            let b = &self.params[2 * l + 1];
            for r in 0..z.rows {
                for (x, &y) in z.row_mut(r).iter_mut().zip(&b.data) {
                    *x += y;
                }
            }
            h = if l + 1 < layers { z.map(gelu) } else { z };
        }
        h
    }
}

/// Each row's scores turned into probabilities that sum to 1.
pub fn softmax(logits: &Mat<f32>) -> Mat<f32> {
    let mut p = logits.clone();
    for r in 0..p.rows {
        let row = p.row_mut(r);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for x in row.iter_mut() {
            *x = (*x - max).exp();
            total += *x;
        }
        for x in row.iter_mut() {
            *x /= total;
        }
    }
    p
}

/// The most likely class and its probability, for each row.
pub fn best(probs: &Mat<f32>) -> Vec<(usize, f32)> {
    (0..probs.rows)
        .map(|r| {
            probs
                .row(r)
                .iter()
                .copied()
                .enumerate()
                .fold(
                    (0, f32::NEG_INFINITY),
                    |a, (i, p)| if p > a.1 { (i, p) } else { a },
                )
        })
        .collect()
}

pub struct ClassifierTraining {
    pub steps: u64,
    pub batch: usize,
    pub learning_rate: f32,
    pub warmup: u64,
    pub seed: u64,
}

impl ClassifierTraining {
    /// Warm-up, then cosine decay to a tenth, as for the diffusion models.
    pub fn lr(&self, step: u64) -> f32 {
        let peak = self.learning_rate;
        if step <= self.warmup {
            return peak * step as f32 / self.warmup.max(1) as f32;
        }
        let p = (step - self.warmup) as f32 / (self.steps - self.warmup).max(1) as f32;
        peak * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * p.min(1.0)).cos()))
    }
}

impl Classifier<f32> {
    /// Train on `set` (images and their labels), random batches; returns
    /// each step's loss and calls `report(step, loss)` after each.
    pub fn train(
        &mut self,
        set: &Images,
        c: &ClassifierTraining,
        mut report: impl FnMut(u64, f32),
    ) -> Vec<f32> {
        assert_eq!(set.shape.values(), self.cfg.inputs, "image size");
        let mut adam = Adam::new(&self.params);
        let mut losses = Vec::with_capacity(c.steps as usize);
        for step in 1..=c.steps {
            let mut rng = Rng::stream(c.seed, step);
            let rows: Vec<usize> = (0..c.batch).map(|_| rng.below(set.pixels.rows)).collect();
            let targets: Vec<Option<usize>> =
                rows.iter().map(|&r| Some(set.labels[r] as usize)).collect();
            let mut tape = Tape::new();
            let (logits, vars) = self.forward(&mut tape, set.batch(&rows));
            let loss = tape.cross_entropy(logits, &targets);
            let value = tape.value(loss).data[0];
            tape.backward(loss);
            let grads: Vec<Mat<f32>> = vars.iter().map(|&v| tape.take_grad(v)).collect();
            adam.update(&mut self.params, &grads, c.lr(step));
            losses.push(value);
            report(step, value);
        }
        losses
    }

    /// Probabilities for each image, in chunks to bound the memory.
    pub fn probabilities(&self, images: &Mat<f32>) -> Mat<f32> {
        let mut out = Vec::with_capacity(images.rows * self.cfg.classes);
        for start in (0..images.rows).step_by(1000) {
            let end = (start + 1000).min(images.rows);
            let chunk = Mat::from_vec(
                end - start,
                images.cols,
                images.data[start * images.cols..end * images.cols].to_vec(),
            );
            out.extend(softmax(&self.logits(&chunk)).data);
        }
        Mat::from_vec(images.rows, self.cfg.classes, out)
    }

    /// The fraction of `set` classified as its label says.
    pub fn accuracy(&self, set: &Images) -> f64 {
        let b = best(&self.probabilities(&set.pixels));
        let right = b
            .iter()
            .zip(&set.labels)
            .filter(|((c, _), &l)| *c == l as usize)
            .count();
        right as f64 / set.labels.len().max(1) as f64
    }

    pub fn to_file(&self, mut metadata: BTreeMap<String, String>) -> File {
        let c = self.cfg;
        for (k, v) in [
            ("inputs", c.inputs),
            ("hidden", c.hidden),
            ("layers", c.layers),
            ("classes", c.classes),
        ] {
            metadata.insert(k.into(), v.to_string());
        }
        metadata.insert("kind".into(), "classifier".into());
        let layers = self.params.len() / 2;
        let tensors = self
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let l = i / 2;
                let name = if l + 1 == layers {
                    "out".to_string()
                } else {
                    format!("hidden{}", l + 1)
                };
                let part = if i % 2 == 0 { "w" } else { "b" };
                Tensor::new(&format!("{name}.{part}"), &[p.rows, p.cols], p.data.clone())
            })
            .collect();
        File { tensors, metadata }
    }

    pub fn from_file(f: &File) -> Result<Classifier<f32>, String> {
        if f.metadata.get("kind").map(String::as_str) != Some("classifier") {
            return Err("not a classifier file (make classifier writes one)".into());
        }
        let num = |k: &str| -> Result<usize, String> {
            f.metadata
                .get(k)
                .ok_or(format!("the classifier file has no {k:?} in its metadata"))?
                .parse()
                .map_err(|_| format!("the classifier file's {k:?} isn't a number"))
        };
        let cfg = ClassifierConfig {
            inputs: num("inputs")?,
            hidden: num("hidden")?,
            layers: num("layers")?,
            classes: num("classes")?,
        };
        let shell = Classifier::<f32>::new(cfg, 0);
        if f.tensors.len() != shell.params.len() {
            return Err(format!(
                "the classifier file has {} tensors, its shape needs {}",
                f.tensors.len(),
                shell.params.len()
            ));
        }
        let mut params = Vec::new();
        for (t, p) in f.tensors.iter().zip(&shell.params) {
            if t.shape != [p.rows, p.cols] {
                return Err(format!("tensor {} has shape {:?}", t.name, t.shape));
            }
            params.push(Mat::from_vec(p.rows, p.cols, t.data.clone()));
        }
        Ok(Classifier { cfg, params })
    }
}

/// What the instrument says about a set of generated images (PLAN.md,
/// Stage 4: agreement, confident, variety, overshoot).
#[derive(Debug, Clone, PartialEq)]
pub struct Measures {
    pub n: usize,
    /// How many images the classifier puts in each class.
    pub counts: Vec<usize>,
    /// The fraction classified as some class with probability above 0.9.
    pub confident: f64,
    /// The fraction classified as the requested class, when there was one.
    pub agreement: Option<f64>,
    /// For each margin m of `MARGINS`, the fraction of pixel values more
    /// than m beyond −1 to 1, before clipping (beyond −1 − m or 1 + m).
    pub overshoot: Vec<(f32, f64)>,
}

/// The margins past the pixel range at which overshoot is counted. A
/// margin of 0 counts every value outside −1 to 1, which on MNIST mostly
/// counts tiny errors around the background's exact −1 (PLAN.md, Stage 4
/// results, prediction 3); the larger margins count real overshoot.
pub const MARGINS: [f32; 4] = [0.0, 0.1, 0.25, 0.5];

/// The probability above which a classification counts as confident.
pub const CONFIDENT: f32 = 0.9;

/// Measure `images` (raw, unclipped values) with `clf`; `requested` gives
/// the class each image was asked to be, if any.
pub fn measure(clf: &Classifier<f32>, images: &Mat<f32>, requested: Option<&[usize]>) -> Measures {
    let n = images.rows;
    let b = best(&clf.probabilities(images));
    let mut counts = vec![0; clf.cfg.classes];
    for &(c, _) in &b {
        counts[c] += 1;
    }
    let confident = b.iter().filter(|(_, p)| *p > CONFIDENT).count() as f64 / n.max(1) as f64;
    let agreement = requested.map(|req| {
        assert_eq!(req.len(), n, "one requested class per image");
        b.iter().zip(req).filter(|((c, _), r)| c == *r).count() as f64 / n.max(1) as f64
    });
    let total = images.data.len().max(1) as f64;
    let overshoot = MARGINS
        .iter()
        .map(|&m| {
            let k = images.data.iter().filter(|x| x.abs() > 1.0 + m).count();
            (m, k as f64 / total)
        })
        .collect();
    Measures {
        n,
        counts,
        confident,
        agreement,
        overshoot,
    }
}

/// The average Euclidean distance between two different images of a set:
/// how varied it is (0 if every image is the same).
pub fn spread(images: &Mat<f32>) -> f64 {
    let n = images.rows;
    if n < 2 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            s += images
                .row(i)
                .iter()
                .zip(images.row(j))
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
        }
    }
    s / (n * (n - 1) / 2) as f64
}

/// Variety: the spread of generated images of one class divided by the
/// spread of real images of that class (1 is the real data's variety).
pub fn variety(generated: &Mat<f32>, real: &Mat<f32>) -> f64 {
    spread(generated) / spread(real).max(1e-12)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // index loops over parameters and their gradients
mod tests {
    use super::*;
    use crate::image::shapes;

    fn small() -> ClassifierConfig {
        ClassifierConfig {
            inputs: 6,
            hidden: 5,
            layers: 2,
            classes: 3,
        }
    }

    #[test]
    fn parameter_count_matches_plan_md() {
        let mnist = ClassifierConfig {
            inputs: 784,
            hidden: 256,
            layers: 2,
            classes: 10,
        };
        assert_eq!(mnist.parameters(), 269_322);
        let m = Classifier::<f32>::new(mnist, 1);
        let total: usize = m.params.iter().map(|p| p.data.len()).sum();
        assert_eq!(total, mnist.parameters());
    }

    #[test]
    fn gradients_match_finite_differences() {
        let m0 = Classifier::<f64>::new(small(), 2);
        let mut m = m0;
        let mut rng = Rng::new(3);
        for p in m.params.iter_mut() {
            for x in p.data.iter_mut() {
                *x += 0.3 * rng.normal();
            }
        }
        let x = Mat::from_vec(4, 6, (0..24).map(|_| rng.normal()).collect());
        let targets = [Some(0), Some(2), None, Some(1)];
        let loss_of = |m: &Classifier<f64>| {
            let mut t = Tape::new();
            let (l, _) = m.forward(&mut t, x.clone());
            let loss = t.cross_entropy(l, &targets);
            t.value(loss).data[0]
        };
        let mut t = Tape::new();
        let (l, vars) = m.forward(&mut t, x.clone());
        let loss = t.cross_entropy(l, &targets);
        t.backward(loss);
        let grads: Vec<Mat<f64>> = vars.iter().map(|&v| t.take_grad(v)).collect();
        let eps = 1e-6;
        let mut worst: f64 = 0.0;
        for k in 0..m.params.len() {
            for i in 0..m.params[k].data.len() {
                m.params[k].data[i] += eps;
                let up = loss_of(&m);
                m.params[k].data[i] -= 2.0 * eps;
                let down = loss_of(&m);
                m.params[k].data[i] += eps;
                let numeric = (up - down) / (2.0 * eps);
                let a = grads[k].data[i];
                worst = worst.max((numeric - a).abs() / (numeric.abs() + a.abs() + 1e-3));
            }
        }
        assert!(worst < 1e-6, "{worst}");
    }

    #[test]
    fn logits_equal_the_tape_forward_pass() {
        let m = Classifier::<f32>::new(small(), 4);
        let x = Mat::from_vec(3, 6, (0..18).map(|i| (i as f32 * 0.37).sin()).collect());
        let mut t = Tape::new();
        let (l, _) = m.forward(&mut t, x.clone());
        let a = t.value(l).data.clone();
        for (p, q) in a.iter().zip(&m.logits(&x).data) {
            assert!((p - q).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_rows_are_probabilities() {
        let p = softmax(&Mat::from_vec(
            2,
            3,
            vec![1.0, 2.0, 3.0, 1000.0, 0.0, -1000.0],
        ));
        for r in 0..2 {
            let s: f32 = p.row(r).iter().sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
        assert_eq!(best(&p)[0].0, 2);
        assert_eq!(best(&p)[1], (0, 1.0));
    }

    fn cfg(steps: u64) -> ClassifierTraining {
        ClassifierTraining {
            steps,
            batch: 32,
            learning_rate: 3e-3,
            warmup: 10,
            seed: 1,
        }
    }

    fn shapes_cfg() -> ClassifierConfig {
        ClassifierConfig {
            inputs: 12 * 12,
            hidden: 64,
            layers: 2,
            classes: 3,
        }
    }

    /// The classifier learns to tell squares, discs and crosses apart, and
    /// is tested on shapes it never saw.
    #[test]
    fn learns_shapes_and_generalizes() {
        let train = shapes(5000, 12, 1, 5);
        let test = shapes(500, 12, 1, 6);
        let mut m = Classifier::new(shapes_cfg(), 1);
        let before = m.accuracy(&test);
        m.train(&train, &cfg(1500), |_, _| {});
        let after = m.accuracy(&test);
        // 0.92 when recorded; chance is 1/3.
        assert!(after > 0.85, "accuracy {before} -> {after}");
    }

    /// A golden test: a fixed seed reproduces recorded losses. If a change
    /// to training is meant, rerun with `cargo test golden -- --nocapture`
    /// and update the values.
    #[test]
    fn golden_classifier_losses() {
        const GOLDEN: [f32; 3] = [1.108289, 1.006367, 0.958591];
        let train = shapes(500, 12, 1, 5);
        let mut m = Classifier::new(shapes_cfg(), 1);
        let l = m.train(&train, &cfg(40), |_, _| {});
        let got = [l[0], l[19], l[39]];
        println!("classifier golden: {got:?}");
        for (g, w) in got.iter().zip(GOLDEN) {
            assert!((g - w).abs() < 2e-3, "{got:?}, golden {GOLDEN:?}");
        }
    }

    #[test]
    fn files_round_trip() {
        let m = Classifier::<f32>::new(small(), 7);
        let f = m.to_file(BTreeMap::new());
        let back = Classifier::from_file(&f).unwrap();
        assert_eq!(back.cfg, m.cfg);
        assert_eq!(back.params, m.params);
        let mut wrong = f;
        wrong.metadata.remove("kind");
        assert!(Classifier::from_file(&wrong).is_err());
    }

    #[test]
    fn measures_by_hand() {
        // A classifier whose scores are its input's first three values.
        let mut m = Classifier::<f32>::new(
            ClassifierConfig {
                inputs: 3,
                hidden: 3,
                layers: 1,
                classes: 3,
            },
            1,
        );
        m.params[0] = Mat::from_vec(3, 3, vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]);
        m.params[2] = Mat::from_vec(3, 3, vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]);
        // GELU(x) ≈ x for large x, ≈ 0 for x ≤ 0: image 0 is confidently
        // class 0, image 1 class 2 (with a value outside −1 to 1), image 2
        // undecided.
        let imgs = Mat::from_vec(3, 3, vec![9., 0., 0., 0., 0., 9., 0.1, 0.1, 0.1]);
        let ms = measure(&m, &imgs, Some(&[0, 1, 0]));
        assert_eq!(ms.counts.iter().sum::<usize>(), 3);
        assert_eq!(ms.counts[2], 1);
        assert!((ms.confident - 2.0 / 3.0).abs() < 1e-9, "{ms:?}");
        assert!((ms.agreement.unwrap() - 2.0 / 3.0).abs() < 1e-9, "{ms:?}");
        // Two values outside −1 to 1 (both 9), both far outside.
        assert_eq!(ms.overshoot[0].0, 0.0);
        for &(_, f) in &ms.overshoot {
            assert!((f - 2.0 / 9.0).abs() < 1e-9, "{ms:?}");
        }
        // A value just past the edge counts at margin 0 only.
        let edge = Mat::from_vec(1, 3, vec![-1.05, 0.0, 0.0]);
        let o = measure(&m, &edge, None).overshoot;
        assert!((o[0].1 - 1.0 / 3.0).abs() < 1e-9 && o[1].1 == 0.0, "{o:?}");
        assert_eq!(measure(&m, &imgs, None).agreement, None);
    }

    #[test]
    fn spread_and_variety() {
        let same = Mat::from_vec(3, 2, vec![1., 2., 1., 2., 1., 2.]);
        assert_eq!(spread(&same), 0.0);
        let two = Mat::from_vec(2, 2, vec![0., 0., 3., 4.]);
        assert!((spread(&two) - 5.0).abs() < 1e-12);
        assert!((variety(&two, &two) - 1.0).abs() < 1e-12);
        assert_eq!(variety(&same, &two), 0.0);
    }
}
