//! Training data: a `.ids` file written by tokenizer/mini_bpe.py --encode
//! (step 2), and the batches cut from it.
//!
//! The file: a 64-byte header, then the token IDs.
//!   offset  size  field
//!        0     8  magic b"MBPEIDS\0"
//!        8     4  format version (1)
//!       12     4  bytes per ID: 2 or 4
//!       16     4  V, the tokenizer's vocabulary size
//!       20     4  reserved (0)
//!       24     8  number of IDs
//!       32    32  tokenizer fingerprint
//!       64     -  the IDs
//! All integers unsigned, little-endian.

use crate::rng::Rng;
use std::path::Path;
use std::sync::Mutex;

pub struct Corpus {
    pub ids: Vec<u32>,
    pub vocab_size: usize,
    pub fingerprint: [u8; 32],
}

impl Corpus {
    pub fn load(path: &Path) -> Result<Corpus, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("can't read {}: {e}", path.display()))?;
        Self::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn decode(bytes: &[u8]) -> Result<Corpus, String> {
        if bytes.len() < 64 {
            return Err("too short for the 64-byte header".into());
        }
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        if &bytes[..8] != b"MBPEIDS\0" || u32_at(8) != 1 {
            return Err("not a mini_bpe token-ID file (version 1)".into());
        }
        let width = u32_at(12) as usize;
        let vocab_size = u32_at(16) as usize;
        let count = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
        let fingerprint: [u8; 32] = bytes[32..64].try_into().unwrap();
        if width != 2 && width != 4 {
            return Err(format!("{width} bytes per ID (expected 2 or 4)"));
        }
        let body = &bytes[64..];
        if body.len() != count * width {
            return Err(format!(
                "header says {count} IDs of {width} bytes, file holds {} bytes",
                body.len()
            ));
        }
        let ids: Vec<u32> = if width == 2 {
            body.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                .collect()
        } else {
            body.chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        if let Some(&bad) = ids.iter().find(|&&t| t as usize >= vocab_size) {
            return Err(format!("token ID {bad} is not below V = {vocab_size}"));
        }
        Ok(Corpus {
            ids,
            vocab_size,
            fingerprint,
        })
    }
}

/// The corpus split into a training part and a validation part (the last
/// fraction of the tokens, held out, never trained on).
pub struct Split<'a> {
    pub train: &'a [u32],
    pub valid: &'a [u32],
}

pub fn split(ids: &[u32], valid_fraction: f64, window: usize) -> Result<Split<'_>, String> {
    let n_valid = ((ids.len() as f64) * valid_fraction).round() as usize;
    let n_train = ids.len() - n_valid;
    if n_train < window + 1 {
        return Err(format!(
            "the training part has {n_train} tokens; each window needs {}",
            window + 1
        ));
    }
    if n_valid > 0 && n_valid < window + 1 {
        return Err(format!(
            "the validation part has {n_valid} tokens; each window needs {} (lower validation_fraction to 0, or use more data)",
            window + 1
        ));
    }
    Ok(Split {
        train: &ids[..n_train],
        valid: &ids[n_train..],
    })
}

/// A batch: `inputs` and `targets` of B·T tokens, sequence after sequence;
/// each target is the token after its input.
pub struct Batch {
    pub inputs: Vec<usize>,
    pub targets: Vec<Option<usize>>,
}

fn batch_at(data: &[u32], starts: &[usize], seq: usize) -> Batch {
    let mut inputs = Vec::with_capacity(starts.len() * seq);
    let mut targets = Vec::with_capacity(starts.len() * seq);
    for &s in starts {
        let w = &data[s..s + seq + 1];
        inputs.extend(w[..seq].iter().map(|&t| t as usize));
        targets.extend(w[1..].iter().map(|&t| Some(t as usize)));
    }
    Batch { inputs, targets }
}

/// The training batch for a given step: `batch` windows of `seq` + 1 tokens.
/// With no `order` (`sampling = "random"`), at random starting points,
/// drawn independently: some stretches are picked twice, others never.
/// With an `order` (`sampling = "epochs"`), the stream is cut into
/// non-overlapping windows (starting at 0, seq, 2·seq, …) taken in a
/// shuffled order, every window once per epoch. Either way the choice
/// depends only on the seed and the step, so a resumed run trains on
/// exactly the same batches.
pub fn train_batch(
    data: &[u32],
    batch: usize,
    seq: usize,
    seed: u64,
    step: u64,
    order: Option<&EpochOrder>,
) -> Batch {
    let starts: Vec<usize> = match order {
        None => {
            let mut rng = Rng::stream(seed, step);
            let span = data.len() - seq; // valid starts: 0 … len − seq − 1
            (0..batch).map(|_| rng.below(span)).collect()
        }
        Some(o) => o.picks(batch, step).iter().map(|&w| w * seq).collect(),
    };
    batch_at(data, &starts, seq)
}

/// The number of non-overlapping windows of `seq` + 1 tokens (one token
/// shared between neighbours, as each window's last target is the next
/// one's first input) in `len` tokens: the items of an epoch.
pub fn epoch_windows(len: usize, seq: usize) -> usize {
    len.saturating_sub(1) / seq
}

/// Epochs: `items` things (windows, or conversations) taken in a shuffled
/// order, every one once per epoch, a new order each epoch. The order of
/// epoch e is a fixed function of the seed and e, so any step's items can
/// be found without replaying the steps before it (resuming is exact). The
/// current epoch's order is kept, so finding it costs one shuffle per epoch.
pub struct EpochOrder {
    items: usize,
    seed: u64,
    current: Mutex<Option<(u64, Vec<u32>)>>,
}

impl EpochOrder {
    pub fn new(items: usize, seed: u64) -> EpochOrder {
        assert!(
            items > 0 && items <= u32::MAX as usize,
            "an epoch needs 1 to 2³² − 1 items"
        );
        EpochOrder {
            items,
            seed,
            current: Mutex::new(None),
        }
    }

    /// The `batch` items for a step (counting from 1): positions
    /// (step − 1)·batch onward in the sequence of epochs.
    pub fn picks(&self, batch: usize, step: u64) -> Vec<usize> {
        let first = (step - 1) * batch as u64;
        (0..batch as u64).map(|k| self.item(first + k)).collect()
    }

    /// The item at position g, counting from 0 across all epochs.
    pub fn item(&self, g: u64) -> usize {
        let (epoch, i) = (g / self.items as u64, (g % self.items as u64) as usize);
        let mut cur = self
            .current
            .lock()
            .expect("no other thread panics holding it");
        if cur.as_ref().map(|(e, _)| *e) != Some(epoch) {
            *cur = Some((epoch, self.shuffled(epoch)));
        }
        cur.as_ref().expect("just set").1[i] as usize
    }

    /// 0 … items − 1 in a random order (Fisher-Yates), from the seed and
    /// the epoch; its own stream of random numbers, apart from the ones
    /// random sampling uses for each step.
    fn shuffled(&self, epoch: u64) -> Vec<u32> {
        let mut rng = Rng::stream(self.seed ^ 0x6570_6f63_6873, epoch); // "epochs"
        let mut v: Vec<u32> = (0..self.items as u32).collect();
        for i in (1..v.len()).rev() {
            v.swap(i, rng.below(i + 1));
        }
        v
    }
}

/// Fixed validation batches: windows spread evenly over the held-out part,
/// the same every time, so validation losses are comparable across steps.
pub fn valid_batches(data: &[u32], batches: usize, batch: usize, seq: usize) -> Vec<Batch> {
    if data.len() < seq + 1 || batches == 0 {
        return Vec::new();
    }
    let n = batches * batch;
    let span = data.len() - seq;
    let starts: Vec<usize> = (0..n).map(|i| i * span / n).collect();
    starts
        .chunks(batch)
        .map(|c| batch_at(data, c, seq))
        .collect()
}

/// The special tokens that structure a conversation.
#[derive(Clone, Copy, Debug)]
pub struct Markers {
    pub eos: u32,
    pub user: u32,
    pub assistant: u32,
    pub end: u32,
}

/// Conversations for fine-tuning (step 5), from a `.ids` file written with
/// --allow-special: documents separated by EOS, each
///   <|user|> request <|assistant|> reply <|end|>
/// Each becomes one row of a batch: EOS first (as in generate, it marks
/// the start of a document), then the conversation.
pub struct Conversations {
    /// Each row: EOS, then the conversation's tokens.
    pub rows: Vec<Vec<u32>>,
    /// Per row: the position of <|assistant|>. The targets scored are the
    /// tokens after it, the reply and its <|end|>.
    pub reply_at: Vec<usize>,
    /// Conversations longer than the context, left out.
    pub dropped: usize,
}

impl Conversations {
    pub fn parse(ids: &[u32], m: &Markers, context: usize) -> Result<Conversations, String> {
        let mut c = Conversations {
            rows: Vec::new(),
            reply_at: Vec::new(),
            dropped: 0,
        };
        for (k, doc) in ids.split(|&t| t == m.eos).enumerate() {
            if doc.is_empty() {
                continue; // after the final EOS
            }
            let count = |t: u32| doc.iter().filter(|&&x| x == t).count();
            let a = doc.iter().position(|&t| t == m.assistant);
            let ok = doc[0] == m.user
                && *doc.last().unwrap() == m.end
                && count(m.user) == 1
                && count(m.assistant) == 1
                && count(m.end) == 1;
            let Some(a) = a.filter(|_| ok) else {
                return Err(format!(
                    "conversation {} isn't <|user|> … <|assistant|> … <|end|> (was the file encoded with --allow-special?)",
                    k + 1
                ));
            };
            // A row of n + 1 tokens gives n (input, target) pairs, so a
            // conversation of up to `context` tokens fits.
            if doc.len() > context {
                c.dropped += 1;
                continue;
            }
            let mut row = Vec::with_capacity(doc.len() + 1);
            row.push(m.eos);
            row.extend_from_slice(doc);
            c.rows.push(row);
            c.reply_at.push(a + 1);
        }
        Ok(c)
    }

    /// A batch of the given conversations (indexes into `rows`), each
    /// padded to `seq`. Only the reply's tokens are targets: input position
    /// i predicts row[i + 1], scored when i ≥ the <|assistant|> position.
    pub fn batch(&self, which: &[usize], seq: usize, pad: u32) -> Batch {
        let mut inputs = Vec::with_capacity(which.len() * seq);
        let mut targets = Vec::with_capacity(which.len() * seq);
        for &r in which {
            let (row, a) = (&self.rows[r], self.reply_at[r]);
            for i in 0..seq {
                inputs.push(*row.get(i).unwrap_or(&pad) as usize);
                targets.push(if i >= a && i + 1 < row.len() {
                    Some(row[i + 1] as usize)
                } else {
                    None
                });
            }
        }
        Batch { inputs, targets }
    }

    /// The training batch for a step: `batch` conversations from rows
    /// 0…n_train − 1, drawn at random, or with an `order` (epochs) each once
    /// per epoch in a shuffled order; depending only on the seed and the
    /// step.
    #[allow(clippy::too_many_arguments)]
    pub fn train_batch(
        &self,
        n_train: usize,
        batch: usize,
        seq: usize,
        seed: u64,
        step: u64,
        pad: u32,
        order: Option<&EpochOrder>,
    ) -> Batch {
        let which: Vec<usize> = match order {
            None => {
                let mut rng = Rng::stream(seed, step);
                (0..batch).map(|_| rng.below(n_train)).collect()
            }
            Some(o) => o.picks(batch, step),
        };
        self.batch(&which, seq, pad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every item exactly once per epoch, a different order each epoch, the
    /// same order whatever step it's asked from (so resuming is exact).
    #[test]
    fn epochs_take_every_item_once() {
        let o = EpochOrder::new(10, 3);
        let mut seen = Vec::new();
        for step in 1..=5 {
            seen.extend(o.picks(4, step)); // 20 items: two epochs
        }
        for epoch in seen.chunks(10) {
            let mut e = epoch.to_vec();
            e.sort();
            assert_eq!(e, (0..10).collect::<Vec<_>>());
        }
        assert_ne!(seen[..10], seen[10..]); // a new order each epoch
        assert_ne!(seen[..10], (0..10).collect::<Vec<_>>()[..]); // shuffled
        let fresh = EpochOrder::new(10, 3);
        assert_eq!(fresh.picks(4, 4), seen[12..16]); // straight to step 4
        assert_eq!(fresh.picks(4, 1), seen[..4]); // and back
    }

    #[test]
    fn epoch_windows_tile_the_stream() {
        let data: Vec<u32> = (0..23).collect();
        assert_eq!(epoch_windows(data.len(), 5), 4); // 0–5, 5–10, 10–15, 15–20
        let o = EpochOrder::new(4, 1);
        let mut firsts: Vec<usize> = Vec::new();
        for step in 1..=2 {
            let b = train_batch(&data, 2, 5, 1, step, Some(&o));
            firsts.extend(b.inputs.chunks(5).map(|w| w[0]));
            for (w, t) in b.inputs.chunks(5).zip(b.targets.chunks(5)) {
                assert_eq!(t[0], Some(w[0] + 1)); // targets shifted by one
            }
        }
        firsts.sort();
        assert_eq!(firsts, vec![0, 5, 10, 15]);
    }

    /// tests/data/tiny.ids was written by mini_bpe.py --encode with
    /// tiny.bpe; tiny.ids.expected holds what Python reads back.
    #[test]
    fn reads_a_file_written_by_mini_bpe() {
        let c = Corpus::decode(include_bytes!("../tests/data/tiny.ids")).unwrap();
        let expected = include_str!("../tests/data/tiny.ids.expected");
        let mut parts = expected.split_whitespace();
        let count: usize = parts.next().unwrap().parse().unwrap();
        let v: usize = parts.next().unwrap().parse().unwrap();
        let fp = parts.next().unwrap();
        assert_eq!(c.ids.len(), count);
        assert_eq!(c.vocab_size, v);
        assert_eq!(crate::tokenizer::hex(&c.fingerprint), fp);
        assert_eq!(&c.ids[..4], &[84, 104, 101, 272]);
        assert_eq!(*c.ids.last().unwrap(), 316); // EOS at the end of the document
    }

    #[test]
    fn rejects_damaged_files() {
        let good = include_bytes!("../tests/data/tiny.ids").to_vec();
        assert!(Corpus::decode(&good[..good.len() - 1]).is_err());
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(Corpus::decode(&bad_magic).is_err());
        let mut bad_id = good.clone();
        let n = bad_id.len();
        bad_id[n - 2..].copy_from_slice(&5000u16.to_le_bytes()); // ≥ V
        assert!(Corpus::decode(&bad_id).is_err());
    }

    #[test]
    fn batches_are_shifted_by_one_and_reproducible() {
        let data: Vec<u32> = (0..100).collect();
        let b = train_batch(&data, 3, 8, 7, 42, None);
        for s in 0..3 {
            for t in 0..8 {
                assert_eq!(b.targets[s * 8 + t], Some(b.inputs[s * 8 + t] + 1));
            }
        }
        let again = train_batch(&data, 3, 8, 7, 42, None);
        assert_eq!(b.inputs, again.inputs);
        let other = train_batch(&data, 3, 8, 7, 43, None);
        assert_ne!(b.inputs, other.inputs);
        // The last window may end exactly at the last token, never past it.
        for _ in 0..1000 {
            let b = train_batch(&data[..10], 1, 9, 1, 5, None);
            assert_eq!(b.inputs[0], 0);
        }
    }

    /// The windows over many steps must be about as varied as truly random
    /// picks. (A bug in seeding each step once made 600 steps × 16 windows
    /// visit only about 1,200 distinct places in 720,000.)
    #[test]
    fn windows_over_many_steps_are_varied() {
        let data: Vec<u32> = (0..720_000).collect(); // a window's first token is its start
        let mut starts = std::collections::HashSet::new();
        for step in 1..=600u64 {
            let b = train_batch(&data, 16, 128, 1, step, None);
            for w in 0..16 {
                starts.insert(b.inputs[w * 128]);
            }
        }
        // 9,600 random picks from 719,872 places: about 64 repeats expected.
        assert!(
            starts.len() > 9_450,
            "only {} distinct starts",
            starts.len()
        );
    }

    const M: Markers = Markers {
        eos: 0,
        user: 1,
        assistant: 2,
        end: 3,
    };

    #[test]
    fn conversations_score_only_the_reply() {
        // <|user|> 10 11 <|assistant|> 20 21 22 <|end|>, then a longer one.
        let ids = [
            1, 10, 11, 2, 20, 21, 22, 3, 0, 1, 10, 2, 20, 21, 22, 23, 24, 25, 3, 0,
        ];
        let c = Conversations::parse(&ids, &M, 8).unwrap();
        assert_eq!(c.rows.len(), 1); // the second (10 tokens) doesn't fit in 8
        assert_eq!(c.dropped, 1);
        assert_eq!(c.rows[0], vec![0, 1, 10, 11, 2, 20, 21, 22, 3]);
        let b = c.batch(&[0], 10, 0);
        assert_eq!(b.inputs, vec![0, 1, 10, 11, 2, 20, 21, 22, 3, 0]);
        // Targets: nothing for the request; the reply and <|end|>; nothing
        // after (padding).
        let none = None;
        assert_eq!(
            b.targets,
            vec![
                none,
                none,
                none,
                none,
                Some(20),
                Some(21),
                Some(22),
                Some(3),
                none,
                none
            ]
        );
    }

    #[test]
    fn malformed_conversations_are_refused() {
        for bad in [
            &[10, 2, 20, 3, 0][..],      // no <|user|>
            &[1, 10, 20, 3, 0][..],      // no <|assistant|>
            &[1, 10, 2, 20, 0][..],      // no <|end|>
            &[1, 2, 1, 2, 20, 3, 0][..], // two turns
        ] {
            assert!(Conversations::parse(bad, &M, 100).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn validation_is_the_held_out_tail() {
        let data: Vec<u32> = (0..1000).collect();
        let s = split(&data, 0.1, 16).unwrap();
        assert_eq!(s.train.len(), 900);
        assert_eq!(s.valid[0], 900);
        let vb = valid_batches(s.valid, 2, 4, 16);
        assert_eq!(vb.len(), 2);
        assert!(vb.iter().all(|b| b.inputs.iter().all(|&t| t >= 900)));
        assert!(split(&data[..20], 0.1, 16).is_err());
    }
}
