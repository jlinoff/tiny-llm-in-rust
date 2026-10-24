//! Reading the corpus: splitting text into words, building the vocabulary,
//! and storing the corpus as a sequence of word IDs.
//!
//! This is word2vec's equivalent of the tokenizer phase, but much simpler:
//! the "tokens" are whole words, and the vocabulary is just every word that
//! occurs at least `min_count` times. Its size, V, sets the number of rows in
//! both embedding tables.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Training processes the corpus in "sentences" of at most this many words
/// (the same limit as the original C word2vec). A context window never
/// crosses a sentence boundary. text8 has no punctuation or line breaks, so
/// this is just a convenient unit of work to hand to a thread.
pub const MAX_SENTENCE: usize = 1000;

/// Split text into lowercase words: maximal runs of alphanumeric characters.
/// Everything else (spaces, punctuation, apostrophes) is a separator, so
/// "Don't stop!" gives "don", "t", "stop". Unlike mini_bpe this is lossy on
/// purpose: word2vec models words, not the exact text.
pub fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
}

/// The vocabulary: word <-> ID, most frequent word first (ID 0).
pub struct Vocab {
    /// `words[id]` is the word with that ID.
    pub words: Vec<String>,
    /// `counts[id]` is how many times that word occurs in the corpus.
    pub counts: Vec<u64>,
    index: HashMap<String, u32>,
}

impl Vocab {
    /// Number of words in the vocabulary: V.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// The ID of `word`, if it is in the vocabulary.
    pub fn id(&self, word: &str) -> Option<u32> {
        self.index.get(word).copied()
    }

    /// Total occurrences of vocabulary words in the corpus.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }
}

/// One input file, as read.
pub struct Document {
    pub path: PathBuf,
    pub words: u64,
}

/// The whole training corpus, held in memory as word IDs.
pub struct Corpus {
    pub vocab: Vocab,
    pub documents: Vec<Document>,
    /// Number of distinct words seen, before the `min_count` cut.
    pub distinct: usize,
    /// Every corpus word that made it into the vocabulary, as IDs, in order.
    /// Rare words are dropped entirely, so a window simply spans over them.
    pub tokens: Vec<u32>,
    /// Ranges into `tokens`, each at most MAX_SENTENCE long and never
    /// crossing a document boundary.
    pub sentences: Vec<(usize, usize)>,
}

impl Corpus {
    /// Read the corpus files and build the vocabulary in one pass.
    ///
    /// Each distinct word gets a provisional ID as it is first seen, and the
    /// files are stored as provisional IDs. Once all counts are known, the
    /// words below `min_count` are dropped, the rest are sorted by frequency,
    /// and the stored IDs are translated to their final values.
    pub fn load(paths: &[PathBuf], min_count: u64) -> Result<Corpus, String> {
        let mut provisional: HashMap<String, u32> = HashMap::new();
        let mut seen_words: Vec<String> = Vec::new();
        let mut seen_counts: Vec<u64> = Vec::new();
        let mut doc_ids: Vec<Vec<u32>> = Vec::new();
        let mut documents = Vec::new();

        for path in paths {
            let text = read_text(path)?;
            let mut ids = Vec::new();
            for word in words(&text) {
                let id = *provisional.entry(word).or_insert_with_key(|w| {
                    seen_words.push(w.clone());
                    seen_counts.push(0);
                    (seen_words.len() - 1) as u32
                });
                seen_counts[id as usize] += 1;
                ids.push(id);
            }
            documents.push(Document {
                path: path.clone(),
                words: ids.len() as u64,
            });
            doc_ids.push(ids);
        }

        // Keep words with count >= min_count, most frequent first; ties are
        // broken alphabetically so the vocabulary is deterministic.
        let mut kept: Vec<u32> = (0..seen_words.len() as u32)
            .filter(|&id| seen_counts[id as usize] >= min_count)
            .collect();
        kept.sort_by(|&a, &b| {
            seen_counts[b as usize]
                .cmp(&seen_counts[a as usize])
                .then_with(|| seen_words[a as usize].cmp(&seen_words[b as usize]))
        });
        if kept.is_empty() {
            return Err(format!(
                "no word occurs at least {min_count} times; lower --min-count"
            ));
        }

        let mut final_id = vec![u32::MAX; seen_words.len()];
        let mut vocab = Vocab {
            words: Vec::with_capacity(kept.len()),
            counts: Vec::with_capacity(kept.len()),
            index: HashMap::with_capacity(kept.len()),
        };
        for (new, &old) in kept.iter().enumerate() {
            final_id[old as usize] = new as u32;
            let word = std::mem::take(&mut seen_words[old as usize]);
            vocab.index.insert(word.clone(), new as u32);
            vocab.words.push(word);
            vocab.counts.push(seen_counts[old as usize]);
        }

        let mut tokens = Vec::new();
        let mut sentences = Vec::new();
        for ids in doc_ids {
            let doc_start = tokens.len();
            tokens.extend(
                ids.into_iter()
                    .map(|id| final_id[id as usize])
                    .filter(|&id| id != u32::MAX),
            );
            let mut start = doc_start;
            while start < tokens.len() {
                let end = (start + MAX_SENTENCE).min(tokens.len());
                sentences.push((start, end));
                start = end;
            }
        }

        Ok(Corpus {
            vocab,
            documents,
            distinct: provisional.len(),
            tokens,
            sentences,
        })
    }
}

fn read_text(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|e| format!("can't read corpus file {}: {e}", path.display()))?;
    String::from_utf8(bytes)
        .map_err(|_| format!("corpus file {} is not valid UTF-8 text", path.display()))
}
