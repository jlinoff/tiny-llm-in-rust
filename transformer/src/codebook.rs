//! The codebook (PLAN.md, Drawing, part 2): K typical 4 × 4 patches,
//! learned from the training images by k-means, so an image becomes one
//! token per patch: 49 for MNIST.
//!
//! ```text
//! start:  K patches from the data, each new one picked with probability
//!         proportional to its squared distance from the nearest already
//!         picked (k-means++), so no entry starts as a copy of another
//! repeat: give each patch to its nearest entry;
//!         move each entry to the average of the patches it was given
//!         (an entry given none stays where it is)
//! until:  no patch changes entry, or the repetitions run out
//! ```
//!
//! Each repetition can only lower the total squared distance from the
//! patches to their entries: giving a patch to a nearer entry lowers its
//! term, and an average is the point nearest, in total squared distance,
//! to the points averaged. Distances are computed directly, ties go to the
//! lower-numbered entry, and sums are added in a fixed order, so a seed
//! gives the same codebook, byte for byte, however many threads run.

use crate::image::{patchify, unpatchify, Images, Shape};
use crate::rng::Rng;
use crate::safetensors::{File, Tensor};
use crate::tensor::Mat;
use crate::tokenizer::hex;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Patches per parallel chunk: fixed, so the order sums are added in
/// doesn't depend on the number of threads.
const CHUNK: usize = 4096;

pub struct Codebook {
    /// The patch side, in pixels: 4.
    pub patch: usize,
    /// One entry per row, K × patch² values on the −1 to 1 scale.
    pub entries: Mat<f32>,
}

/// The squared distance between two patches.
fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The nearest entry to a patch, and its squared distance; ties go to the
/// lower-numbered entry.
fn nearest(entries: &Mat<f32>, p: &[f32]) -> (u32, f32) {
    let mut best = (0u32, f32::INFINITY);
    for k in 0..entries.rows {
        let d = dist2(entries.row(k), p);
        if d < best.1 {
            best = (k as u32, d);
        }
    }
    best
}

/// Each patch's nearest entry, and the total squared distance (in f64,
/// added chunk by chunk in order).
fn assign(entries: &Mat<f32>, patches: &Mat<f32>) -> (Vec<u32>, f64) {
    let rows: Vec<usize> = (0..patches.rows).collect();
    let parts: Vec<(Vec<u32>, f64)> = rows
        .par_chunks(CHUNK)
        .map(|chunk| {
            let mut ids = Vec::with_capacity(chunk.len());
            let mut total = 0.0f64;
            for &r in chunk {
                let (k, d) = nearest(entries, patches.row(r));
                ids.push(k);
                total += d as f64;
            }
            (ids, total)
        })
        .collect();
    let mut ids = Vec::with_capacity(patches.rows);
    let mut total = 0.0;
    for (i, t) in parts {
        ids.extend(i);
        total += t;
    }
    (ids, total)
}

/// k-means++: K starting entries picked from the patches.
fn start(patches: &Mat<f32>, k: usize, seed: u64) -> Mat<f32> {
    let (n, dim) = (patches.rows, patches.cols);
    assert!(k >= 1 && n >= k, "{k} entries need at least {k} patches");
    let mut rng = Rng::new(seed);
    let mut entries = Mat::zeros(k, dim);
    entries
        .row_mut(0)
        .copy_from_slice(patches.row(rng.below(n)));
    // Each patch's squared distance to its nearest entry so far.
    let mut d2: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|r| dist2(patches.row(r), entries.row(0)))
        .collect();
    for j in 1..k {
        let total: f64 = d2.iter().map(|&d| d as f64).sum();
        let pick = if total > 0.0 {
            let mut u = rng.unit() * total;
            let mut pick = n - 1;
            for (r, &d) in d2.iter().enumerate() {
                u -= d as f64;
                if u < 0.0 {
                    pick = r;
                    break;
                }
            }
            pick
        } else {
            rng.below(n) // fewer distinct patches than entries
        };
        entries.row_mut(j).copy_from_slice(patches.row(pick));
        let e = entries.row(j).to_vec();
        d2.par_iter_mut().enumerate().for_each(|(r, d)| {
            *d = d.min(dist2(patches.row(r), &e));
        });
    }
    entries
}

/// Each entry moved to the average of its patches (sums in f64, in order);
/// an entry with none stays.
fn update(entries: &mut Mat<f32>, patches: &Mat<f32>, ids: &[u32]) {
    let dim = patches.cols;
    let mut sums = vec![0.0f64; entries.rows * dim];
    let mut counts = vec![0u64; entries.rows];
    for (r, &k) in ids.iter().enumerate() {
        let k = k as usize;
        counts[k] += 1;
        for (s, &x) in sums[k * dim..(k + 1) * dim].iter_mut().zip(patches.row(r)) {
            *s += x as f64;
        }
    }
    for k in 0..entries.rows {
        if counts[k] > 0 {
            for (e, s) in entries.row_mut(k).iter_mut().zip(&sums[k * dim..]) {
                *e = (*s / counts[k] as f64) as f32;
            }
        }
    }
}

/// What one repetition did, for the log and the tests.
pub struct Repetition {
    /// The total squared distance after the patches were reassigned, before
    /// the entries moved.
    pub total: f64,
    /// How many patches changed entry.
    pub changed: usize,
}

impl Codebook {
    /// Learn `k` entries from `patches` (one per row), at most `repetitions`
    /// times round; `report` is called after each.
    pub fn learn(
        patches: &Mat<f32>,
        patch: usize,
        k: usize,
        repetitions: usize,
        seed: u64,
        mut report: impl FnMut(usize, &Repetition),
    ) -> (Codebook, Vec<Repetition>) {
        assert_eq!(patches.cols, patch * patch, "patches of {patch} × {patch}");
        let mut entries = start(patches, k, seed);
        let mut ids: Vec<u32> = Vec::new();
        let mut log = Vec::new();
        for i in 1..=repetitions {
            let (new, total) = assign(&entries, patches);
            let changed = if ids.is_empty() {
                new.len()
            } else {
                new.iter().zip(&ids).filter(|(a, b)| a != b).count()
            };
            ids = new;
            let rep = Repetition { total, changed };
            report(i, &rep);
            log.push(rep);
            if changed == 0 {
                break;
            }
            update(&mut entries, patches, &ids);
        }
        (Codebook { patch, entries }, log)
    }

    pub fn size(&self) -> usize {
        self.entries.rows
    }

    /// Each image's patches as entry numbers: images.rows × (patches per
    /// image), image after image, each image's patches in raster order.
    pub fn encode(&self, images: &Mat<f32>, shape: Shape) -> Vec<u32> {
        assign(&self.entries, &patchify(images, shape, self.patch)).0
    }

    /// Images from their entry numbers (the inverse of `encode`, up to the
    /// rounding to entries).
    pub fn decode(&self, ids: &[u32], shape: Shape) -> Mat<f32> {
        let dim = self.entries.cols;
        let mut patches = Mat::zeros(ids.len(), dim);
        for (r, &k) in ids.iter().enumerate() {
            patches
                .row_mut(r)
                .copy_from_slice(self.entries.row(k as usize));
        }
        unpatchify(&patches, shape, self.patch)
    }

    /// A set rebuilt from its codebook entries: what part 2's tokens keep
    /// of it.
    pub fn reconstructed(&self, set: &Images) -> Images {
        Images {
            shape: set.shape,
            pixels: self.decode(&self.encode(&set.pixels, set.shape), set.shape),
            labels: set.labels.clone(),
        }
    }

    /// The SHA-256 of the entries' bytes and the patch size: token files
    /// carry it, so they can't be decoded with another codebook.
    pub fn fingerprint(&self) -> String {
        let mut h = Sha256::new();
        h.update((self.patch as u64).to_le_bytes());
        h.update((self.entries.rows as u64).to_le_bytes());
        for x in &self.entries.data {
            h.update(x.to_le_bytes());
        }
        hex(&h.finalize())
    }

    pub fn to_file(&self, mut metadata: BTreeMap<String, String>) -> File {
        metadata.insert("kind".into(), "codebook".into());
        metadata.insert("patch".into(), self.patch.to_string());
        metadata.insert("fingerprint".into(), self.fingerprint());
        let e = &self.entries;
        File {
            tensors: vec![Tensor::new("entries", &[e.rows, e.cols], e.data.clone())],
            metadata,
        }
    }

    pub fn from_file(f: &File) -> Result<Codebook, String> {
        if f.metadata.get("kind").map(String::as_str) != Some("codebook") {
            return Err("not a codebook file (make draw-codebook writes one)".into());
        }
        let patch: usize = f
            .metadata
            .get("patch")
            .and_then(|p| p.parse().ok())
            .ok_or("the codebook file has no patch size")?;
        let t = f
            .tensors
            .iter()
            .find(|t| t.name == "entries")
            .ok_or("the codebook file has no entries")?;
        if t.shape.len() != 2 || t.shape[1] != patch * patch {
            return Err(format!("entries of shape {:?} for patch {patch}", t.shape));
        }
        let c = Codebook {
            patch,
            entries: Mat::from_vec(t.shape[0], t.shape[1], t.data.clone()),
        };
        match f.metadata.get("fingerprint") {
            Some(fp) if *fp == c.fingerprint() => Ok(c),
            Some(fp) => Err(format!(
                "the codebook's fingerprint {fp} doesn't match its entries ({})",
                c.fingerprint()
            )),
            None => Err("the codebook file has no fingerprint".into()),
        }
    }

    /// Check that token files made with codebook `fingerprint` can be read
    /// with this one. (Used once drawing's token files are written, with
    /// training; tested now.)
    #[allow(dead_code)]
    pub fn check(&self, fingerprint: &str) -> Result<(), String> {
        if fingerprint == self.fingerprint() {
            Ok(())
        } else {
            Err(format!(
                "these tokens were made with codebook {fingerprint}, not this one ({})",
                self.fingerprint()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::shapes;

    /// The patches of a few synthetic 12 × 12 shapes.
    fn patches() -> Mat<f32> {
        let s = shapes(40, 12, 1, 2);
        patchify(&s.pixels, s.shape, 4)
    }

    /// After each repetition every patch is with its nearest entry (checked
    /// by brute force), and the total never rises.
    #[test]
    fn each_repetition_assigns_the_nearest_and_lowers_the_total() {
        let p = patches();
        let mut entries = start(&p, 8, 3);
        let mut last = f64::INFINITY;
        for _ in 0..10 {
            let (ids, total) = assign(&entries, &p);
            for (r, &k) in ids.iter().enumerate() {
                let mine = dist2(entries.row(k as usize), p.row(r));
                for j in 0..entries.rows {
                    let other = dist2(entries.row(j), p.row(r));
                    assert!(
                        mine <= other,
                        "patch {r}: entry {k} at {mine}, {j} at {other}"
                    );
                    if other == mine {
                        assert!(k as usize <= j, "ties go to the lower entry");
                    }
                }
            }
            assert!(total <= last * (1.0 + 1e-12), "{total} after {last}");
            last = total;
            update(&mut entries, &p, &ids);
        }
        let (_, log) = Codebook::learn(&p, 4, 8, 50, 3, |_, _| {});
        for w in log.windows(2) {
            assert!(w[1].total <= w[0].total * (1.0 + 1e-12));
        }
        assert_eq!(log.last().unwrap().changed, 0, "it settles");
    }

    /// The same seed gives the same codebook, byte for byte.
    #[test]
    fn a_seed_gives_the_same_codebook() {
        let p = patches();
        let (a, _) = Codebook::learn(&p, 4, 8, 20, 5, |_, _| {});
        let (b, _) = Codebook::learn(&p, 4, 8, 20, 5, |_, _| {});
        assert_eq!(a.entries, b.entries);
        assert_eq!(a.fingerprint(), b.fingerprint());
        let (c, _) = Codebook::learn(&p, 4, 8, 20, 6, |_, _| {});
        assert_ne!(a.fingerprint(), c.fingerprint());
    }

    /// k-means++ never starts two entries on the same patch while distinct
    /// patches remain.
    #[test]
    fn starting_entries_are_distinct() {
        let e = start(&patches(), 8, 1);
        for i in 0..e.rows {
            for j in i + 1..e.rows {
                assert_ne!(e.row(i), e.row(j), "entries {i} and {j}");
            }
        }
    }

    /// A patch equal to an entry encodes to that entry, and decodes back
    /// exactly; an image made of entries survives the round trip.
    #[test]
    fn entries_encode_to_themselves() {
        let (c, _) = Codebook::learn(&patches(), 4, 8, 20, 1, |_, _| {});
        let shape = Shape {
            height: 4,
            width: 8,
            channels: 1,
        };
        // An image of two patches: entry 5 on the left, entry 2 on the right.
        let img = c.decode(&[5, 2], shape);
        assert_eq!(c.encode(&img, shape), vec![5, 2]);
        assert_eq!(c.decode(&c.encode(&img, shape), shape), img);
    }

    #[test]
    fn files_round_trip_and_fingerprints_are_checked() {
        let (c, _) = Codebook::learn(&patches(), 4, 8, 20, 1, |_, _| {});
        let f = c.to_file(BTreeMap::new());
        let back = Codebook::from_file(&f).unwrap();
        assert_eq!(back.entries, c.entries);
        assert!(back.check(&c.fingerprint()).is_ok());
        let (other, _) = Codebook::learn(&patches(), 4, 8, 20, 2, |_, _| {});
        assert!(
            other.check(&c.fingerprint()).is_err(),
            "tokens from another codebook"
        );
        let mut tampered = f;
        tampered.tensors[0].data[0] += 0.5;
        assert!(Codebook::from_file(&tampered).is_err());
    }
}
