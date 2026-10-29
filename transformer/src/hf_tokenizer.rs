//! A Hugging Face byte-level BPE tokenizer, read from its `tokenizer.json`
//! (PLAN.md, Order of work, the endpoint). The same algorithm as ours
//! (tokenizer/mini_bpe.py), with three differences of bookkeeping:
//!
//! - **IDs come from a vocabulary table**, not from "256 bytes, then merges
//!   in order": the special tokens are 0…16, and every other token's ID is
//!   whatever the table says.
//! - **Tokens are written in GPT-2's printable byte alphabet**: each of the
//!   256 byte values stands for one printable character, so "Ġ" is the
//!   space byte and a merge reads "Ġ t" for " " + "t". The mapping is
//!   GPT-2's `bytes_to_unicode`: the printable ASCII and Latin-1 bytes stand
//!   for themselves, the other 68 bytes for the characters 256 onward.
//! - **Pre-tokenization in two stages**: first every digit becomes its own
//!   piece ("Digits", individual), then GPT-2's pattern splits each piece
//!   ("ByteLevel"), as our pre-tokenizer does.
//!
//! Special tokens ("added tokens") are matched whole in the text first,
//! longest first, and never split.

use std::collections::HashMap;
use std::path::Path;

/// GPT-2's pre-tokenizer pattern (the one `tokenizer.json` names with
/// `"use_regex": true`).
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// GPT-2's byte → character table, and its inverse.
fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut printable: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
    printable.extend(0xA1..=0xAC);
    printable.extend(0xAE..=0xFF);
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..256u32 {
        table[b as usize] = if printable.contains(&b) {
            char::from_u32(b).expect("Latin-1")
        } else {
            next += 1;
            char::from_u32(next - 1).expect("below 0x200")
        };
    }
    let inverse = table
        .iter()
        .enumerate()
        .map(|(b, &c)| (c, b as u8))
        .collect();
    (table, inverse)
}

pub struct HfTokenizer {
    vocab: HashMap<String, usize>,
    /// ID → token string (in the byte alphabet, or a special token's text).
    tokens: Vec<String>,
    /// (left, right) → rank: a lower rank merges first.
    ranks: HashMap<(String, String), usize>,
    /// Special tokens: (text, ID), longest text first.
    specials: Vec<(String, usize)>,
    pattern: fancy_regex::Regex,
    alphabet: [char; 256],
    inverse: HashMap<char, u8>,
}

impl HfTokenizer {
    pub fn load(path: &Path) -> Result<HfTokenizer, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("can't read {}: {e}", path.display()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let bad = |what: &str| format!("{}: {what}", path.display());
        let model = &json["model"];
        if model["type"] != "BPE" {
            return Err(bad("not a BPE tokenizer"));
        }
        // What this implementation handles: plain byte-level BPE.
        for key in [
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "unk_token",
            "dropout",
        ] {
            if !model[key].is_null() {
                return Err(bad(&format!("model.{key} isn't supported")));
            }
        }
        if model["byte_fallback"] == true || model["ignore_merges"] == true {
            return Err(bad("byte_fallback and ignore_merges aren't supported"));
        }
        if !json["normalizer"].is_null() || !json["post_processor"].is_null() {
            return Err(bad("a normalizer or post-processor isn't supported"));
        }
        let pre = json["pre_tokenizer"].to_string();
        if !pre.contains("\"individual_digits\":true")
            || !pre.contains("ByteLevel")
            || !pre.contains("\"add_prefix_space\":false")
        {
            return Err(bad(&format!(
                "pre-tokenizer {pre}: expected Digits (individual) then ByteLevel without a prefix space"
            )));
        }

        let mut vocab = HashMap::new();
        for (tok, id) in model["vocab"].as_object().ok_or(bad("no vocabulary"))? {
            vocab.insert(
                tok.clone(),
                id.as_u64().ok_or(bad("a non-numeric ID"))? as usize,
            );
        }
        let mut tokens = vec![String::new(); vocab.len()];
        for (tok, &id) in &vocab {
            *tokens.get_mut(id).ok_or(bad("IDs aren't 0…V − 1"))? = tok.clone();
        }
        let mut ranks = HashMap::new();
        for (rank, m) in model["merges"]
            .as_array()
            .ok_or(bad("no merges"))?
            .iter()
            .enumerate()
        {
            let (a, b) = match m {
                serde_json::Value::String(s) => {
                    s.split_once(' ').ok_or(bad("a merge without a space"))?
                }
                _ => return Err(bad("merges must be strings \"a b\"")),
            };
            ranks.insert((a.to_string(), b.to_string()), rank);
        }
        let mut specials = Vec::new();
        for a in json["added_tokens"]
            .as_array()
            .ok_or(bad("no added_tokens"))?
        {
            let (content, id) = (a["content"].as_str(), a["id"].as_u64());
            let (Some(content), Some(id)) = (content, id) else {
                return Err(bad("an added token without content or id"));
            };
            specials.push((content.to_string(), id as usize));
        }
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let (alphabet, inverse) = byte_alphabet();
        Ok(HfTokenizer {
            vocab,
            tokens,
            ranks,
            specials,
            pattern: fancy_regex::Regex::new(GPT2_PATTERN).expect("a valid pattern"),
            alphabet,
            inverse,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The ID of a special token, by its text.
    pub fn special(&self, text: &str) -> Option<usize> {
        self.specials
            .iter()
            .find(|(s, _)| s == text)
            .map(|&(_, id)| id)
    }

    /// Text → token IDs, special tokens' text included as those tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest special token in what's left (longest on a tie).
            let next = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|at| (at, s.len(), *id)))
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            let (plain, after) = match next {
                Some((at, len, id)) => {
                    let (p, a) = (&rest[..at], &rest[at + len..]);
                    self.encode_plain(p, &mut ids)?;
                    ids.push(id);
                    rest = a;
                    continue;
                }
                None => (rest, ""),
            };
            self.encode_plain(plain, &mut ids)?;
            rest = after;
        }
        Ok(ids)
    }

    /// Text with no special tokens: digits apart, then GPT-2's pattern,
    /// then BPE on each piece.
    fn encode_plain(&self, text: &str, ids: &mut Vec<usize>) -> Result<(), String> {
        for piece in split_digits(text) {
            for m in self.pattern.find_iter(piece) {
                let chunk = m.map_err(|e| format!("pre-tokenizer: {e}"))?.as_str();
                for tok in self.bpe(chunk) {
                    ids.push(
                        *self
                            .vocab
                            .get(&tok)
                            .ok_or(format!("token {tok:?} isn't in the vocabulary"))?,
                    );
                }
            }
        }
        Ok(())
    }

    /// One chunk's bytes, as alphabet characters, merged by rank: at each
    /// step the adjacent pair with the lowest rank is merged, everywhere it
    /// occurs, until no pair has a rank.
    fn bpe(&self, chunk: &str) -> Vec<String> {
        let mut parts: Vec<String> = chunk
            .bytes()
            .map(|b| self.alphabet[b as usize].to_string())
            .collect();
        loop {
            let best = parts
                .windows(2)
                .filter_map(|w| self.ranks.get(&(w[0].clone(), w[1].clone())))
                .min()
                .copied();
            let Some(rank) = best else { break };
            let mut merged = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len()
                    && self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) == Some(&rank)
                {
                    merged.push(format!("{}{}", parts[i], parts[i + 1]));
                    i += 2;
                } else {
                    merged.push(parts[i].clone());
                    i += 1;
                }
            }
            parts = merged;
        }
        parts
    }

    /// One token's bytes (a special token's are its text).
    pub fn token_bytes(&self, id: usize) -> Vec<u8> {
        if self.specials.iter().any(|&(_, s)| s == id) {
            return self.tokens[id].as_bytes().to_vec();
        }
        self.tokens[id]
            .chars()
            .map(|c| *self.inverse.get(&c).unwrap_or(&b'?'))
            .collect()
    }

    /// Token IDs → text (a cut-off UTF-8 character shows as �).
    pub fn decode(&self, ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.token_bytes(i)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Split text so that every digit is a piece of its own ("Digits" with
/// individual digits): "ab12 c" → "ab", "1", "2", " c".
fn split_digits(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if c.is_numeric() {
            if start < i {
                out.push(&text[start..i]);
            }
            out.push(&text[i..i + c.len_utf8()]);
            start = i + c.len_utf8();
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_byte_alphabet_is_gpt2s() {
        let (table, inverse) = byte_alphabet();
        assert_eq!(table[b' ' as usize], 'Ġ'); // space → U+0120
        assert_eq!(table[b'\n' as usize], 'Ċ'); // newline → U+010A
        assert_eq!(table[b'a' as usize], 'a');
        assert_eq!(inverse.len(), 256); // a bijection
    }

    #[test]
    fn digits_stand_alone() {
        assert_eq!(split_digits("ab12 c3"), vec!["ab", "1", "2", " c", "3"]);
        assert_eq!(split_digits("no digits"), vec!["no digits"]);
    }
}
