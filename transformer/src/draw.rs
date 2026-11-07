//! Drawing (PLAN.md, Drawing): images as sentences of tokens, so this
//! project's transformer can learn to make them.
//!
//! ```text
//! position:   0      1    2    3   …   783    784
//! token:     <7>    p1   p2   p3   …   p783   p784
//!
//! inputs   (positions 0 to 783): the class token, then pixels p1 to p783
//! targets  (positions 1 to 784): pixels p1 to p784
//! ```
//!
//! An image's tokens come first in the vocabulary (4 brightness levels for
//! part 1, 256 codebook entries for part 2), then the 10 class tokens
//! `<0>`–`<9>`, then `<none>`, "any digit", for guidance.

use crate::config::ModelConfig;
use crate::generate::{sample, Generator};
use crate::gpt::Params;
use crate::image::{to_byte, Images};
use crate::rng::Rng;
use crate::tensor::{Float, Mat};
use rayon::prelude::*;

/// The brightness levels a pixel is rounded to (part 1).
pub const LEVELS: usize = 4;

/// The class tokens: one per digit.
pub const CLASSES: usize = 10;

/// A vocabulary of `image` image tokens, then the class tokens, then
/// `<none>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vocab {
    pub image: usize,
}

impl Vocab {
    /// The token for "draw digit c".
    pub fn class(&self, c: usize) -> u32 {
        assert!(c < CLASSES, "digit {c}");
        (self.image + c) as u32
    }

    /// `<none>`: "draw any digit".
    pub fn none(&self) -> u32 {
        (self.image + CLASSES) as u32
    }

    /// Every token: image tokens, classes and `<none>`.
    pub fn size(&self) -> usize {
        self.image + CLASSES + 1
    }

    /// The name of a token, for logs: `L2` (a level), `#17` (an entry),
    /// `<7>` or `<none>`.
    pub fn name(&self, t: u32, levels: bool) -> String {
        let t = t as usize;
        if t < self.image {
            if levels {
                format!("L{t}")
            } else {
                format!("#{t}")
            }
        } else if t < self.image + CLASSES {
            format!("<{}>", t - self.image)
        } else {
            "<none>".into()
        }
    }
}

/// The pixel model's vocabulary: 4 levels, 10 classes, `<none>`: 15.
pub const PIXELS: Vocab = Vocab { image: LEVELS };

/// The fingerprint part 1's token files and models carry, in place of a
/// tokenizer's: the SHA-256 of a description of the encoding, so a change
/// to it (more levels, other values) makes old files unreadable.
pub fn pixel_fingerprint() -> [u8; 32] {
    use sha2::{Digest, Sha256};
    Sha256::digest(
        b"mini_transformer drawing, part 1: pixels in raster order, levels 0 85 170 255; then <0>-<9>, <none>",
    )
    .into()
}

/// The level of a pixel (−1 black to 1 white): its byte value, 0 to 255,
/// rounded to the nearest of 0, 85, 170 and 255, so black stays black and
/// white stays white. The boundaries fall between 42 and 43, 127 and 128,
/// and 212 and 213.
pub fn level_of(x: f32) -> u32 {
    (to_byte(x) as u32 + 42) / 85
}

/// The brightness a level decodes to: 0, 85, 170 or 255 as a byte, on the
/// −1 to 1 scale.
pub fn level_value(level: u32) -> f32 {
    assert!((level as usize) < LEVELS, "level {level}");
    (85 * level) as f32 / 127.5 - 1.0
}

/// An image's sentence: its class token, then its pixels' levels in raster
/// order (row by row from the top left).
pub fn pixel_sentence(image: &[f32], label: usize) -> Vec<u32> {
    let mut s = Vec::with_capacity(image.len() + 1);
    s.push(PIXELS.class(label));
    s.extend(image.iter().map(|&x| level_of(x)));
    s
}

/// The image a sentence's pixel tokens (everything after the class token)
/// describe.
pub fn pixel_image(tokens: &[u32]) -> Vec<f32> {
    tokens.iter().map(|&t| level_value(t)).collect()
}

/// A set's sentences, image after image, each 1 + pixels tokens.
pub fn pixel_sentences(set: &Images) -> Vec<u32> {
    (0..set.pixels.rows)
        .flat_map(|r| pixel_sentence(set.pixels.row(r), set.labels[r] as usize))
        .collect()
}

/// A set turned into its sentences and back: what part 1's tokens keep of
/// it, its pixels rounded to 4 levels.
pub fn quantized(set: &Images) -> Images {
    let len = set.pixels.cols + 1;
    let pixels = pixel_sentences(set)
        .chunks(len)
        .flat_map(|s| pixel_image(&s[1..]))
        .collect();
    Images {
        shape: set.shape,
        pixels: Mat::from_vec(set.pixels.rows, set.pixels.cols, pixels),
        labels: set.labels.clone(),
    }
}

/// An image's sentence for part 2: its class token, then its patches'
/// codebook entries, as `vocab` numbers them (entries first).
pub fn codebook_sentences(
    entries: &[u32],
    per_image: usize,
    labels: &[u8],
    vocab: Vocab,
) -> Vec<u32> {
    assert_eq!(entries.len(), per_image * labels.len());
    let mut s = Vec::with_capacity(entries.len() + labels.len());
    for (ids, &l) in entries.chunks(per_image).zip(labels) {
        s.push(vocab.class(l as usize));
        s.extend_from_slice(ids);
    }
    s
}

/// How often each image token appears, as fractions of `tokens`.
pub fn shares(tokens: impl Iterator<Item = u32>, image_tokens: usize) -> Vec<f64> {
    let mut counts = vec![0u64; image_tokens];
    let mut n = 0u64;
    for t in tokens {
        counts[t as usize] += 1;
        n += 1;
    }
    counts.iter().map(|&c| c as f64 / n.max(1) as f64).collect()
}

/// The entropy of a distribution, in nats: what predicting a token costs
/// when only how common each one is is known (PLAN.md, Drawing,
/// prediction 2's reference line).
pub fn entropy(p: &[f64]) -> f64 {
    -p.iter()
        .filter(|&&q| q > 0.0)
        .map(|q| q * q.ln())
        .sum::<f64>()
}

// ----- sampling (PLAN.md, Drawing, parts 1 and 4)

/// Classifier-free guidance's combination of two rows of logits,
/// ℓ_none + w·(ℓ_class − ℓ_none): exactly ℓ_class at w = 1 and exactly
/// ℓ_none at w = 0 (returned as they are, not computed, so no rounding).
pub fn guide(class: &[f64], none: &[f64], w: f64) -> Vec<f64> {
    if w == 1.0 {
        class.to_vec()
    } else if w == 0.0 {
        none.to_vec()
    } else {
        class
            .iter()
            .zip(none)
            .map(|(c, n)| n + w * (c - n))
            .collect()
    }
}

/// One image to draw: the digit (None: any digit, the `<none>` token), the
/// guidance scale, and which random stream it draws with.
#[derive(Clone, Copy, Debug)]
pub struct Ask {
    pub class: Option<usize>,
    pub w: f64,
    pub stream: u64,
}

/// One image's `n` tokens, after its class token. Each is drawn at
/// temperature 1 with no cutoff from the image tokens only (a class token
/// in the middle of an image means nothing), its random numbers from
/// `stream` of `seed`, so an image doesn't depend on what else is drawn
/// with it. With guidance (a digit asked for and w ≠ 1) a second copy of
/// the model reads the same tokens after `<none>`, and the two sets of
/// logits are combined by `guide`.
pub fn draw_one<T: Float>(
    p: &Params<T>,
    c: &ModelConfig,
    vocab: Vocab,
    n: usize,
    ask: Ask,
    seed: u64,
) -> Vec<u32> {
    assert!(n <= c.context, "{n} tokens need a context of {n}");
    let mut rng = Rng::stream(seed, ask.stream);
    let first = ask.class.map_or(vocab.none(), |k| vocab.class(k));
    let guided = ask.class.is_some() && ask.w != 1.0;
    let mut g = Generator::new(p, c);
    let mut u = guided.then(|| Generator::new(p, c));
    let row =
        |m: &Mat<T>| -> Vec<f64> { m.row(0)[..vocab.image].iter().map(|v| v.to_f64()).collect() };
    let mut lc = row(&g.feed(&[first as usize], false));
    let mut ln = u
        .as_mut()
        .map(|u| row(&u.feed(&[vocab.none() as usize], false)));
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let logits = match &ln {
            Some(l) => guide(&lc, l, ask.w),
            None => lc.clone(),
        };
        let t = sample(&logits, 1.0, 0, &mut rng);
        out.push(t as u32);
        if out.len() < n {
            lc = row(&g.feed(&[t], false));
            if let Some(u) = u.as_mut() {
                ln = Some(row(&u.feed(&[t], false)));
            }
        }
    }
    out
}

/// Many images at once, one per core at a time: each gives exactly the
/// tokens `draw_one` gives it alone.
pub fn draw_many<T: Float>(
    p: &Params<T>,
    c: &ModelConfig,
    vocab: Vocab,
    n: usize,
    asks: &[Ask],
    seed: u64,
) -> Vec<Vec<u32>> {
    asks.par_iter()
        .map(|&a| draw_one(p, c, vocab, n, a, seed))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Shape;

    /// A byte value on the −1 to 1 scale, as MNIST is read.
    fn px(byte: u8) -> f32 {
        byte as f32 / 127.5 - 1.0
    }

    #[test]
    fn levels_at_every_boundary() {
        for (byte, level) in [
            (0, 0),
            (42, 0),
            (43, 1),
            (85, 1),
            (127, 1),
            (128, 2),
            (170, 2),
            (212, 2),
            (213, 3),
            (255, 3),
        ] {
            assert_eq!(level_of(px(byte)), level, "byte {byte}");
        }
        // Black decodes to black, white to white, the grays to 85 and 170.
        for (level, byte) in [(0, 0u8), (1, 85), (2, 170), (3, 255)] {
            assert_eq!(level_value(level), px(byte), "level {level}");
            assert_eq!(level_of(level_value(level)), level);
        }
        assert_eq!(level_value(0), -1.0);
        assert_eq!(level_value(3), 1.0);
    }

    #[test]
    fn the_vocabulary() {
        assert_eq!(PIXELS.size(), 15);
        assert_eq!(
            (PIXELS.class(0), PIXELS.class(9), PIXELS.none()),
            (4, 13, 14)
        );
        let codebook = Vocab { image: 256 };
        assert_eq!(codebook.size(), 267);
        assert_eq!(codebook.none(), 266);
        assert_eq!(PIXELS.name(2, true), "L2");
        assert_eq!(codebook.name(17, false), "#17");
        assert_eq!(PIXELS.name(11, true), "<7>");
        assert_eq!(PIXELS.name(14, true), "<none>");
    }

    /// A 2 × 2 image, by hand: black, dark gray, light gray, white.
    #[test]
    fn a_sentence_by_hand() {
        let img = [px(0), px(60), px(150), px(250)];
        let s = pixel_sentence(&img, 7);
        assert_eq!(s, vec![11, 0, 1, 2, 3]);
        assert_eq!(pixel_image(&s[1..]), vec![px(0), px(85), px(170), px(255)]);
    }

    /// An MNIST-sized image makes 785 tokens, the class first, and decodes
    /// to its reconstruction.
    #[test]
    fn an_image_is_785_tokens() {
        let img: Vec<f32> = (0..784).map(|i| px((i * 37 % 256) as u8)).collect();
        let s = pixel_sentence(&img, 3);
        assert_eq!(s.len(), 785);
        assert_eq!(s[0], PIXELS.class(3));
        assert!(s[1..].iter().all(|&t| (t as usize) < LEVELS));
        let set = Images {
            shape: Shape {
                height: 28,
                width: 28,
                channels: 1,
            },
            pixels: Mat::from_vec(1, 784, img),
            labels: vec![3],
        };
        assert_eq!(pixel_image(&s[1..]), quantized(&set).pixels.data);
    }

    #[test]
    fn codebook_sentences_put_the_class_first() {
        let v = Vocab { image: 256 };
        let s = codebook_sentences(&[1, 2, 3, 4, 5, 6], 3, &[7, 0], v);
        assert_eq!(s, vec![263, 1, 2, 3, 256, 4, 5, 6]);
    }

    #[test]
    fn guidance_by_hand() {
        let (c, n) = ([2.0, 0.0, -1.0], [1.0, 1.0, 1.0]);
        assert_eq!(guide(&c, &n, 1.0), c.to_vec());
        assert_eq!(guide(&c, &n, 0.0), n.to_vec());
        // 1 + 3·(2 − 1) = 4, 1 + 3·(0 − 1) = −2, 1 + 3·(−1 − 1) = −5.
        assert_eq!(guide(&c, &n, 3.0), vec![4.0, -2.0, -5.0]);
        assert_eq!(guide(&c, &n, 0.5), vec![1.5, 0.5, 0.0]);
    }

    fn tiny() -> (Params<f32>, ModelConfig) {
        let c = ModelConfig {
            d_model: 16,
            layers: 1,
            heads: 2,
            context: 9,
            d_ff: 32,
            seed: 4,
        };
        (
            Params::from_tensors(&crate::model::init(&c, PIXELS.size())),
            c,
        )
    }

    /// Drawn together, each image gets exactly the tokens it gets alone,
    /// with and without guidance; the tokens are image tokens; the same
    /// stream gives the same image, another stream another.
    #[test]
    fn many_at_once_equals_one_at_a_time() {
        let (p, c) = tiny();
        let asks: Vec<Ask> = (0..8)
            .map(|i| Ask {
                class: if i == 7 { None } else { Some(i % 10) },
                w: [1.0, 3.0, 0.0, 1.5][i % 4],
                stream: i as u64,
            })
            .collect();
        let many = draw_many(&p, &c, PIXELS, 9, &asks, 5);
        for (a, got) in asks.iter().zip(&many) {
            assert_eq!(*got, draw_one(&p, &c, PIXELS, 9, *a, 5), "{a:?}");
            assert_eq!(got.len(), 9);
            assert!(got.iter().all(|&t| (t as usize) < LEVELS));
        }
        let a = Ask {
            class: Some(3),
            w: 1.0,
            stream: 1,
        };
        let b = Ask { stream: 2, ..a };
        assert_eq!(
            draw_one(&p, &c, PIXELS, 9, a, 5),
            draw_one(&p, &c, PIXELS, 9, a, 5)
        );
        assert_ne!(
            draw_one(&p, &c, PIXELS, 9, a, 5),
            draw_one(&p, &c, PIXELS, 9, b, 5)
        );
    }

    /// At w = 0 a request for a digit draws exactly what `<none>` does (and
    /// at w = 1 it is the unguided request itself, by `guide`'s test).
    #[test]
    fn guidance_at_one_and_zero() {
        let (p, c) = tiny();
        let ask = |class, w| Ask {
            class,
            w,
            stream: 3,
        };
        let none = draw_one(&p, &c, PIXELS, 9, ask(None, 1.0), 1);
        assert_eq!(draw_one(&p, &c, PIXELS, 9, ask(Some(4), 0.0), 1), none);
        let strong = draw_one(&p, &c, PIXELS, 9, ask(Some(4), 8.0), 1);
        assert_eq!(strong.len(), 9);
    }

    #[test]
    fn shares_and_entropy() {
        let p = shares([0u32, 0, 1, 3].into_iter(), 4);
        assert_eq!(p, vec![0.5, 0.25, 0.0, 0.25]);
        // −(½ ln ½ + 2 · ¼ ln ¼) = 1.5 ln 2.
        assert!((entropy(&p) - 1.5 * 2f64.ln()).abs() < 1e-12);
        assert_eq!(entropy(&[1.0, 0.0]), 0.0);
    }
}
