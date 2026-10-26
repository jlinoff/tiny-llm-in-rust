//! Measuring fine-tuning (step 5): does the model use the words it's asked
//! for? Each evaluation request names three words (chat/make_chat.py writes
//! them, from stories neither training step saw). For each generated story:
//! how many of its three requested words appear in it.
//!
//! The control: the same stories, checked for the words of a *different*
//! request (the next one in the file). A model that ignores requests scores
//! the same on both, so the difference is what following the request adds,
//! beyond words like "ball" simply being common in stories.

/// Does `text` contain `word` as a whole word, ignoring case? "Ball" and
/// "ball." count; "balloon" doesn't. Inflected forms ("balls") don't count
/// either: a strict, simple test.
pub fn contains_word(text: &str, word: &str) -> bool {
    let text = text.to_lowercase();
    let word = word.to_lowercase();
    let is_letter = |c: Option<char>| c.is_some_and(|c| c.is_alphabetic());
    let mut from = 0;
    while let Some(i) = text[from..].find(&word) {
        let start = from + i;
        let end = start + word.len();
        if !is_letter(text[..start].chars().next_back()) && !is_letter(text[end..].chars().next()) {
            return true;
        }
        from = start + word.len().max(1);
    }
    false
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
