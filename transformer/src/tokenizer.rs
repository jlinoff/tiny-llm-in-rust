//! Reading what `build` needs from a tokenizer file (`tok.bpe`, written by
//! tokenizer/mini_bpe.py): the vocabulary size V and the fingerprint.
//!
//! The format: comment lines start with '#'; one "pattern …" line; one
//! "<left_id> <right_id>" line per merge, in order; one "special <name>" line
//! per special token. Text after '#' on a data line is a comment.
//! V = 256 bytes + merges + special tokens.

use sha2::{Digest, Sha256};
use std::path::Path;

pub struct TokenizerInfo {
    pub vocab_size: usize,
    pub merges: usize,
    pub specials: Vec<String>,
    /// The pre-tokenizer pattern and the merges in learned order, for
    /// encoding and decoding (see `Encoder`).
    pub pattern: String,
    pub merge_list: Vec<(usize, usize)>,
    /// SHA-256 of the pattern, merges and special tokens, computed exactly as
    /// mini_bpe.py's Tokenizer.fingerprint does, so the two always agree.
    pub fingerprint: [u8; 32],
}

impl TokenizerInfo {
    pub fn load(path: &Path) -> Result<TokenizerInfo, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("can't read tokenizer {}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("tokenizer {}: {e}", path.display()))
    }

    pub fn parse(text: &str) -> Result<TokenizerInfo, String> {
        let mut pattern: Option<&str> = None;
        let mut merges: Vec<(usize, usize)> = Vec::new();
        let mut specials: Vec<String> = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let lineno = i + 1;
            if let Some(p) = line.strip_prefix("pattern ") {
                pattern = Some(p);
                continue;
            }
            let data: Vec<&str> = line.split('#').next().unwrap().split_whitespace().collect();
            match data.as_slice() {
                [] => {}
                ["special", name] => {
                    if specials.iter().any(|s| s == name) {
                        return Err(format!("line {lineno}: special token {name} repeated"));
                    }
                    specials.push(name.to_string());
                }
                [a, b] => {
                    let next = 256 + merges.len();
                    let parse = |s: &str| {
                        s.parse::<usize>().ok().filter(|&v| v < next).ok_or(format!(
                            "line {lineno}: token id {s} out of range (below {next})"
                        ))
                    };
                    merges.push((parse(a)?, parse(b)?));
                }
                _ => return Err(format!("line {lineno}: expected '<left_id> <right_id>'")),
            }
        }
        let pattern = pattern.ok_or("no 'pattern' line")?;
        // The canonical text, exactly as in mini_bpe.py.
        let mut canon = String::new();
        canon.push_str(pattern);
        canon.push('\n');
        for (a, b) in &merges {
            canon.push_str(&format!("{a} {b}\n"));
        }
        for name in &specials {
            canon.push_str(&format!("special {name}\n"));
        }
        let fingerprint: [u8; 32] = Sha256::digest(canon.as_bytes()).into();
        Ok(TokenizerInfo {
            vocab_size: 256 + merges.len() + specials.len(),
            merges: merges.len(),
            specials,
            fingerprint,
            pattern: pattern.to_string(),
            merge_list: merges,
        })
    }
}

/// Encoding text to token IDs and decoding IDs back, exactly as
/// mini_bpe.py does: split the text into chunks with the pre-tokenizer
/// pattern, then in each chunk repeatedly apply the earliest-learned merge
/// present, until none applies. Decoding concatenates each token's bytes.
pub struct Encoder {
    /// Compiled from the pattern after `python_classes`. The pattern's
    /// `\s+(?!\S)` is a lookahead, which Rust's standard `regex` doesn't
    /// support; `fancy_regex` adds lookaround.
    regex: fancy_regex::Regex,
    /// Each merged pair → the ID it creates; a smaller ID is an earlier merge.
    ranks: std::collections::HashMap<(u32, u32), u32>,
    /// Each token's bytes (special tokens decode to their names).
    vocab: Vec<Vec<u8>>,
}

/// Rewrite a pattern written for Python's `re` so Rust matches the same
/// characters. Python and Rust define the shorthand classes differently:
/// - `\w`: Python's is `str.isalnum()` plus `_`, that is letters (L) and
///   numbers (N, so "²", "½" and "Ⅻ" too) and `_`; Rust's adds combining
///   marks (M) and connector punctuation (Pc, such as "‿"), and leaves out
///   numbers that aren't decimal digits.
/// - `\d`: both are decimal digits (Nd).
/// - `\s`: Python's also includes the control characters U+001C–U+001F
///   (file, group, record and unit separators); Rust's doesn't.
///
/// Each is replaced by an explicit class with Python's meaning. A doubled
/// backslash (a literal backslash) is left alone.
pub fn python_classes(pattern: &str) -> String {
    let mut out = String::new();
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('w') => out.push_str(r"[\p{L}\p{N}_]"),
            Some('W') => out.push_str(r"[^\p{L}\p{N}_]"),
            Some('d') => out.push_str(r"\p{Nd}"),
            Some('D') => out.push_str(r"\P{Nd}"),
            Some('s') => out.push_str(r"[\s\x{1C}-\x{1F}]"),
            Some('S') => out.push_str(r"[^\s\x{1C}-\x{1F}]"),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

impl TokenizerInfo {
    pub fn encoder(&self) -> Result<Encoder, String> {
        let regex = fancy_regex::Regex::new(&python_classes(&self.pattern))
            .map_err(|e| format!("tokenizer pattern: {e}"))?;
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut ranks = std::collections::HashMap::new();
        for (k, &(a, b)) in self.merge_list.iter().enumerate() {
            let id = (256 + k) as u32;
            ranks.insert((a as u32, b as u32), id);
            let mut bytes = vocab[a].clone();
            bytes.extend_from_slice(&vocab[b]);
            vocab.push(bytes);
        }
        for name in &self.specials {
            vocab.push(name.as_bytes().to_vec());
        }
        Ok(Encoder {
            regex,
            ranks,
            vocab,
        })
    }

    /// The ID of a special token, if the tokenizer has it.
    pub fn special_id(&self, name: &str) -> Option<usize> {
        self.specials
            .iter()
            .position(|s| s == name)
            .map(|i| 256 + self.merges + i)
    }
}

impl Encoder {
    /// The pre-tokenizer's chunks of `text`.
    pub fn chunks<'t>(&self, text: &'t str) -> Result<Vec<&'t str>, String> {
        self.regex
            .find_iter(text)
            .map(|m| {
                m.map(|m| m.as_str())
                    .map_err(|e| format!("pre-tokenizer: {e}"))
            })
            .collect()
    }

    /// One chunk's IDs: repeatedly merge the pair with the earliest merge,
    /// all its non-overlapping occurrences left to right, until none is left.
    fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut seq: Vec<u32> = chunk.bytes().map(u32::from).collect();
        while seq.len() > 1 {
            let best = seq
                .windows(2)
                .filter_map(|w| self.ranks.get(&(w[0], w[1])).map(|&id| (id, (w[0], w[1]))))
                .min();
            let Some((id, pair)) = best else { break };
            let mut out = Vec::with_capacity(seq.len());
            let mut i = 0;
            while i < seq.len() {
                if i + 1 < seq.len() && (seq[i], seq[i + 1]) == pair {
                    out.push(id);
                    i += 2;
                } else {
                    out.push(seq[i]);
                    i += 1;
                }
            }
            seq = out;
        }
        seq
    }

    /// Text → token IDs. Never produces a special token's ID: literal
    /// special-token text is encoded as ordinary bytes.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        let mut ids = Vec::new();
        for chunk in self.chunks(text)? {
            ids.extend(self.encode_chunk(chunk).into_iter().map(|t| t as usize));
        }
        Ok(ids)
    }

    /// One token's bytes (a token may be part of a UTF-8 character).
    pub fn token_bytes(&self, id: usize) -> &[u8] {
        &self.vocab[id]
    }

    /// Token IDs → text (invalid UTF-8, from a cut-off character, is shown
    /// with the replacement character). generate streams through
    /// `Utf8Stream` instead; this is for the tests' round trips.
    #[cfg(test)]
    pub fn decode(&self, ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.vocab[i].iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// tests/data/tiny.bpe was written by mini_bpe.py; tiny.bpe.expected holds
    /// the V and fingerprint that Python computes for it.
    #[test]
    fn fingerprint_matches_python() {
        let info = TokenizerInfo::parse(include_str!("../tests/data/tiny.bpe")).unwrap();
        let expected = include_str!("../tests/data/tiny.bpe.expected");
        let mut parts = expected.split_whitespace();
        let v: usize = parts.next().unwrap().parse().unwrap();
        let fp = parts.next().unwrap();
        assert_eq!(info.vocab_size, v);
        assert_eq!(hex(&info.fingerprint), fp);
        assert_eq!(info.specials, vec!["<|endoftext|>".to_string()]);
    }

    #[test]
    fn comments_dont_change_the_fingerprint_but_merges_do() {
        let text = include_str!("../tests/data/tiny.bpe");
        let base = TokenizerInfo::parse(text).unwrap().fingerprint;
        let recommented = text.replace("# vocab_size", "# a different comment, vocab_size");
        assert_eq!(
            TokenizerInfo::parse(&recommented).unwrap().fingerprint,
            base
        );
        // Drop the last merge line (the one before the special token).
        let lines: Vec<&str> = text.lines().collect();
        let last_merge = lines
            .iter()
            .rposition(|l| !l.starts_with('#') && !l.starts_with("special"))
            .unwrap();
        let fewer: String = lines
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != last_merge)
            .map(|(_, l)| format!("{l}\n"))
            .collect();
        let info = TokenizerInfo::parse(&fewer).unwrap();
        assert_ne!(info.fingerprint, base);
        assert_eq!(info.vocab_size, 316);
    }

    /// tests/data/encode_cases.json holds, for each test string, the chunks
    /// and IDs that mini_bpe.py produces with tiny.bpe: ASCII, contractions,
    /// numbers, underscores, whitespace runs, and several scripts.
    #[test]
    fn encoding_matches_python() {
        let info = TokenizerInfo::parse(include_str!("../tests/data/tiny.bpe")).unwrap();
        let enc = info.encoder().unwrap();
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../tests/data/encode_cases.json")).unwrap();
        for case in &cases {
            let text = case["text"].as_str().unwrap();
            let chunks: Vec<&str> = case["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap())
                .collect();
            let ids: Vec<usize> = case["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| i.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(enc.chunks(text).unwrap(), chunks, "chunks of {text:?}");
            assert_eq!(enc.encode(text).unwrap(), ids, "IDs of {text:?}");
            assert_eq!(enc.decode(&ids), text, "decoding {text:?}");
        }
    }

    #[test]
    fn literal_special_token_text_is_not_the_special_token() {
        let info = TokenizerInfo::parse(include_str!("../tests/data/tiny.bpe")).unwrap();
        let enc = info.encoder().unwrap();
        let eos = info.special_id("<|endoftext|>").unwrap();
        let ids = enc.encode("a <|endoftext|> b").unwrap();
        assert!(!ids.contains(&eos));
        assert_eq!(enc.decode(&[eos]), "<|endoftext|>");
    }

    #[test]
    fn rejects_out_of_range_merges() {
        assert!(TokenizerInfo::parse("pattern x\n0 256\n").is_err()); // 256 not yet made
        assert!(TokenizerInfo::parse("0 1\n").is_err()); // no pattern
    }
}
