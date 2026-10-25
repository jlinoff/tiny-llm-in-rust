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
        })
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

    #[test]
    fn rejects_out_of_range_merges() {
        assert!(TokenizerInfo::parse("pattern x\n0 256\n").is_err()); // 256 not yet made
        assert!(TokenizerInfo::parse("0 1\n").is_err()); // no pattern
    }
}
