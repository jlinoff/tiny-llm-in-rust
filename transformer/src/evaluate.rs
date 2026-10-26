//! Measuring fine-tuning (step 5): does the model use the words it's asked
//! for? Each evaluation request names three words (chat/make_chat.py writes
//! them, from stories neither training step saw). For each generated story:
//! how many of its three requested words appear in it.
//!
//! The control: the same stories, checked for the words of a *different*
//! request (the next one in the file). A model that ignores requests scores
//! the same on both, so the difference is what following the request adds,
//! beyond words like "ball" simply being common in stories.
//!
//! The paired-loss check (--paired) is more sensitive: no sampling, just
//! the model's loss on each held-out story given its own request, and
//! given another request. If the model uses the request at all, its own
//! request makes the story more predictable, above all at the requested
//! words.

use crate::generate::Generator;
use crate::tensor::Float;

/// Where `text` contains `word` as a whole word, ignoring ASCII case: the
/// byte ranges. "Ball" and "ball." count; "balloon" doesn't. Inflected
/// forms ("balls") don't count either: a strict, simple test. (The
/// requested words are plain lowercase ASCII, chat/make_chat.py's [a-z]+.)
pub fn word_spans(text: &str, word: &str) -> Vec<(usize, usize)> {
    let (t, w) = (text.as_bytes(), word.as_bytes());
    let is_letter = |c: Option<char>| c.is_some_and(|c| c.is_alphabetic());
    let mut spans = Vec::new();
    if w.is_empty() || w.len() > t.len() {
        return spans;
    }
    for start in 0..=t.len() - w.len() {
        let end = start + w.len();
        if text.is_char_boundary(start)
            && text.is_char_boundary(end)
            && t[start..end].eq_ignore_ascii_case(w)
            && !is_letter(text[..start].chars().next_back())
            && !is_letter(text[end..].chars().next())
        {
            spans.push((start, end));
        }
    }
    spans
}

/// Does `text` contain `word` as a whole word (see `word_spans`)?
pub fn contains_word(text: &str, word: &str) -> bool {
    !word_spans(text, word).is_empty()
}

/// One held-out conversation from chat-eval.txt: a request and the story
/// it was made from.
pub struct Conversation {
    pub request: String,
    pub story: String,
}

/// Parse chat-eval.txt: conversations separated by `<|endoftext|>` lines,
/// each `<|user|>request<|assistant|>story<|end|>`.
pub fn parse_conversations(text: &str) -> Result<Vec<Conversation>, String> {
    let mut out = Vec::new();
    let mut docs = vec![String::new()];
    for line in text.lines() {
        if line.trim() == "<|endoftext|>" {
            docs.push(String::new());
        } else {
            let d = docs.last_mut().unwrap();
            if !d.is_empty() {
                d.push('\n');
            }
            d.push_str(line);
        }
    }
    for (k, d) in docs
        .iter()
        .map(|d| d.trim())
        .filter(|d| !d.is_empty())
        .enumerate()
    {
        let bad = || {
            format!(
                "conversation {} isn't <|user|>…<|assistant|>…<|end|>",
                k + 1
            )
        };
        let rest = d.strip_prefix("<|user|>").ok_or_else(bad)?;
        let rest = rest.strip_suffix("<|end|>").ok_or_else(bad)?;
        let (request, story) = rest.split_once("<|assistant|>").ok_or_else(bad)?;
        out.push(Conversation {
            request: request.to_string(),
            story: story.to_string(),
        });
    }
    Ok(out)
}

/// The loss (−ln of the probability given) on each story token, after
/// `prompt`. None if prompt and story don't fit in the context.
pub fn story_losses<T: Float>(
    g: &mut Generator<T>,
    prompt: &[usize],
    story: &[usize],
) -> Option<Vec<f64>> {
    let tokens: Vec<usize> = prompt.iter().chain(story).copied().collect();
    if prompt.is_empty() || story.is_empty() || tokens.len() - 1 > g.context() {
        return None;
    }
    g.reset();
    let logits = g.feed(&tokens[..tokens.len() - 1], true);
    Some(
        (0..story.len())
            .map(|k| {
                let row = logits.row(prompt.len() - 1 + k);
                let max = row
                    .iter()
                    .map(|v| v.to_f64())
                    .fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = row.iter().map(|v| (v.to_f64() - max).exp()).sum();
                (total.ln() + max) - row[story[k]].to_f64()
            })
            .collect(),
    )
}

/// Which story tokens overlap the given byte ranges of the story's text.
/// `token_lens`: each token's length in bytes (their concatenation is the
/// story's text).
pub fn tokens_in_spans(token_lens: &[usize], spans: &[(usize, usize)]) -> Vec<bool> {
    let mut at = 0;
    token_lens
        .iter()
        .map(|&len| {
            let (s, e) = (at, at + len);
            at = e;
            spans.iter().any(|&(a, b)| s < b && a < e)
        })
        .collect()
}

/// Totals for the paired-loss check.
#[derive(Default)]
pub struct Paired {
    pub stories: usize,
    pub skipped: usize,
    pub own: f64,
    pub other: f64,
    pub own_lower: usize,
    pub word_stories: usize,
    pub word_tokens: usize,
    pub own_word: f64,
    pub other_word: f64,
    pub own_word_lower: usize,
}

impl Paired {
    /// Add one story's token losses under its own request and another's;
    /// `words` marks the tokens of its requested words.
    pub fn add(&mut self, own: &[f64], other: &[f64], words: &[bool]) {
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (o, x) = (mean(own), mean(other));
        self.stories += 1;
        self.own += o;
        self.other += x;
        self.own_lower += usize::from(o < x);
        let pick = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(words)
                .filter(|(_, &w)| w)
                .map(|(&l, _)| l)
                .collect()
        };
        let (ow, xw) = (pick(own), pick(other));
        if !ow.is_empty() {
            self.word_stories += 1;
            self.word_tokens += ow.len();
            self.own_word += ow.iter().sum::<f64>();
            self.other_word += xw.iter().sum::<f64>();
            self.own_word_lower += usize::from(mean(&ow) < mean(&xw));
        }
    }
}

/// The sign test: if each of `n` comparisons were a fair coin flip, the
/// probability of a split at least as uneven as `k` of `n` (two-sided).
pub fn sign_test(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut pmf = vec![0.0f64; n + 1];
    pmf[0] = 0.5f64.powi(n as i32);
    for i in 1..=n {
        pmf[i] = pmf[i - 1] * (n - i + 1) as f64 / i as f64;
    }
    let extreme = k.max(n - k);
    let tail: f64 = pmf[extreme..].iter().sum();
    (2.0 * tail).min(1.0)
}

/// One evaluation request: its three words and the request text.
pub struct Request {
    pub words: Vec<String>,
    pub text: String,
}

/// Parse chat-eval.tsv: each line, the words (space-separated), a tab, the
/// request.
pub fn parse_requests(text: &str) -> Result<Vec<Request>, String> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let Some((words, request)) = line.split_once('\t') else {
            return Err(format!(
                "line {}: expected words, a tab, then the request",
                n + 1
            ));
        };
        let words: Vec<String> = words.split_whitespace().map(String::from).collect();
        if words.is_empty() || request.trim().is_empty() {
            return Err(format!("line {}: no words or no request", n + 1));
        }
        out.push(Request {
            words,
            text: request.to_string(),
        });
    }
    Ok(out)
}

/// The running totals.
#[derive(Default)]
pub struct Tally {
    pub stories: usize,
    pub words_asked: usize,
    pub words_found: usize,
    pub all_found: usize,
    pub control_asked: usize,
    pub control_found: usize,
    pub stopped: usize,
    pub tokens: usize,
}

impl Tally {
    /// Add one story: the words requested for it, another request's words
    /// (the control), whether it ended by itself, and its length in tokens.
    pub fn add(
        &mut self,
        story: &str,
        words: &[String],
        control: &[String],
        stopped: bool,
        tokens: usize,
    ) {
        let found = words.iter().filter(|w| contains_word(story, w)).count();
        self.stories += 1;
        self.words_asked += words.len();
        self.words_found += found;
        self.all_found += usize::from(found == words.len());
        self.control_asked += control.len();
        self.control_found += control.iter().filter(|w| contains_word(story, w)).count();
        self.stopped += usize::from(stopped);
        self.tokens += tokens;
    }
}

pub fn percent(part: usize, whole: usize) -> f64 {
    100.0 * part as f64 / whole.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_and_token_overlap() {
        let s = "A Pond, a pond! ponds";
        assert_eq!(word_spans(s, "pond"), vec![(2, 6), (10, 14)]);
        // Tokens "A", " Po", "nd", ",", " a", " pond", "!", " ponds".
        let lens = [1, 3, 2, 1, 2, 5, 1, 6];
        assert_eq!(
            tokens_in_spans(&lens, &word_spans(s, "pond")),
            vec![false, true, true, false, false, true, false, false]
        );
    }

    #[test]
    fn conversations_parse() {
        let t = "<|user|>Use dog.<|assistant|>A dog.\nThe end.<|end|>\n<|endoftext|>\n<|user|>Hi<|assistant|>Yo<|end|>\n";
        let c = parse_conversations(t).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].request, "Use dog.");
        assert_eq!(c[0].story, "A dog.\nThe end.");
        assert!(parse_conversations("<|user|>no reply<|end|>").is_err());
    }

    #[test]
    fn sign_test_values() {
        assert!((sign_test(13, 16) - 0.021_270_751_953_125).abs() < 1e-12);
        assert_eq!(sign_test(8, 16), 1.0);
        assert!(sign_test(150, 200) < 1e-9);
    }

    /// The story losses from the cache equal the training forward pass's
    /// per-position cross-entropy (here: its mean, the tape's loss).
    #[test]
    fn story_losses_match_the_training_loss() {
        use crate::config::ModelConfig;
        use crate::gpt::{forward, Params};
        let c = ModelConfig {
            d_model: 16,
            layers: 2,
            heads: 2,
            context: 12,
            d_ff: 32,
            seed: 4,
        };
        let p = Params::<f64>::from_tensors(&crate::model::init(&c, 20));
        let tokens = [3usize, 7, 1, 9, 9, 2, 5, 11, 4];
        let (prompt, story) = tokens.split_at(4);
        let mut g = Generator::new(&p, &c);
        let losses = story_losses(&mut g, prompt, story).unwrap();
        let inputs = &tokens[..tokens.len() - 1];
        let targets: Vec<Option<usize>> = (0..inputs.len())
            .map(|i| {
                if i + 1 >= 4 {
                    Some(tokens[i + 1])
                } else {
                    None
                }
            })
            .collect();
        let f = forward(&p, &c, inputs, &targets, inputs.len());
        let want = f.tape.value(f.loss).data[0];
        let got = losses.iter().sum::<f64>() / losses.len() as f64;
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn whole_words_only() {
        let s = "The Ball rolled. A balloon flew; the dog's ball-game.";
        assert!(contains_word(s, "ball"));
        assert!(contains_word(s, "BALLOON"));
        assert!(contains_word(s, "dog"));
        assert!(!contains_word(s, "loon"));
        assert!(!contains_word(s, "rolle"));
        assert!(!contains_word("balls", "ball"));
        assert!(contains_word("café ball", "café"));
        assert!(!contains_word("", "ball"));
    }

    #[test]
    fn requests_parse_and_bad_lines_are_refused() {
        let r =
            parse_requests("dog ball park\tWrite a story with the words dog, ball and park.\n\n")
                .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].words, ["dog", "ball", "park"]);
        assert!(r[0].text.starts_with("Write"));
        assert!(parse_requests("no tab here\n").is_err());
        assert!(parse_requests("\tjust a request\n").is_err());
    }

    #[test]
    fn tally_counts_requested_and_control_words() {
        let mut t = Tally::default();
        let w = |s: &str| s.split(' ').map(String::from).collect::<Vec<_>>();
        t.add(
            "the dog found a ball",
            &w("dog ball park"),
            &w("cat ball sun"),
            true,
            10,
        );
        t.add(
            "a cat in the park",
            &w("cat sun park"),
            &w("dog ball park"),
            false,
            20,
        );
        assert_eq!((t.words_found, t.words_asked), (4, 6));
        assert_eq!(t.all_found, 0);
        assert_eq!((t.control_found, t.control_asked), (2, 6));
        assert_eq!((t.stopped, t.tokens), (1, 30));
    }
}
