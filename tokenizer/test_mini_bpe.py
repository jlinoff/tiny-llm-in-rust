#!/usr/bin/env python3
"""Unit tests for mini_bpe.py.

    python3 -m unittest discover -s tokenizer -v      (from the top of the repository)
    make test                                         (runs these with everything else)

Standard library only, like mini_bpe.py. Three kinds of check:

- invariants that hold for any training text: decoding inverts encoding,
  the fast trainer gives exactly the reference trainer's merges, training is
  deterministic, files round-trip;
- agreement with the fixtures in transformer/tests/data, which the Rust tests
  also read. Those were written by mini_bpe.py; if a change here makes these
  tests fail, mini_bpe.py no longer produces what the Rust tokenizer was
  checked against, and the fixtures (and the Rust side) need a deliberate
  update, not a silent one;
- special tokens: never produced from ordinary text, only on request.
"""
# Each test's name says what it checks, so they carry no docstrings.
# pylint: disable=missing-function-docstring
import json
import os
import random
import tempfile
import unittest

import mini_bpe

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "transformer", "tests", "data")

# A small training text with repetition, contractions, digits, punctuation,
# whitespace runs and non-ASCII letters: enough for a few dozen merges.
TEXT = ("Once upon a time, there was a little cat. The cat sat on the mat. "
        "The little cat didn't like the rain; it's wet! In 2026 the cat "
        "saw 12 birds and 3 dogs.\n\nThe café was naïve about the cat.  "
        "Then the cat sat again, and again, and again.\n") * 3


def fixture(name):
    """The path of a file in transformer/tests/data."""
    return os.path.join(FIXTURES, name)


def trained(num_merges=40, texts=None):
    """A Tokenizer trained on TEXT (or `texts`), with the default specials."""
    merges, vocab = mini_bpe.train(texts or [TEXT], num_merges, verbose=False)
    return mini_bpe.Tokenizer(merges, vocab, mini_bpe.PATTERN,
                              list(mini_bpe.DEFAULT_SPECIALS))


class Invariants(unittest.TestCase):
    """Properties that hold whatever the training text."""

    def test_decode_inverts_encode(self):
        tok = trained()
        for text in [TEXT, "", "a", "   ", "\n\n\t x", "unseen words zqx",
                     "emoji 🙂🚀 and 東京 and Ελληνικά", "x_y__z", "1234567890"]:
            ids = mini_bpe.encode(text, tok.merges, tok.pattern)
            self.assertEqual(mini_bpe.decode(ids, tok.vocab), text)

    def test_pretokenizer_covers_every_character(self):
        # A character no alternative matched would be silently dropped.
        rng = random.Random(1)
        alphabet = "ab Z9_.,'\"\t\n!?é東🙂-"
        for _ in range(200):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(30)))
            self.assertEqual("".join(mini_bpe.pretokenize(text)), text)

    def test_fast_trainer_matches_reference(self):
        texts = [TEXT, "the the the cat cat", "a b a b a b"]
        fast = mini_bpe.train(texts, 60, verbose=False)
        reference = mini_bpe.train_reference(texts, 60, verbose=False)
        self.assertEqual(fast, reference)

    def test_fast_trainer_matches_reference_to_the_end(self):
        # Run until nothing repeats, on texts full of ties, so the stopping
        # rule and the tie-break (first occurrence wins) both matter.
        rng = random.Random(7)
        samples = [["aaabdaaabac"], ["ab ab ba ba ab"], ["x y x y z z"]]
        samples += [["".join(rng.choice("abc ") for _ in range(rng.randrange(5, 60)))
                     for _ in range(rng.randrange(1, 4))] for _ in range(40)]
        for texts in samples:
            with self.subTest(texts=texts):
                self.assertEqual(mini_bpe.train(texts, 10_000, verbose=False),
                                 mini_bpe.train_reference(texts, 10_000, verbose=False))

    def test_training_is_deterministic(self):
        self.assertEqual(trained().merges, trained().merges)

    def test_merges_shorten_the_encoding(self):
        tok = trained()
        ids = mini_bpe.encode(TEXT, tok.merges, tok.pattern)
        self.assertLess(len(ids), len(TEXT.encode("utf-8")) * 0.7)

    def test_training_stops_when_nothing_repeats(self):
        merges, _ = mini_bpe.train(["abcdefg"], 100, verbose=False)
        self.assertEqual(merges, [])

    def test_merge_ids_follow_the_bytes(self):
        tok = trained()
        self.assertEqual([new for _, new in tok.merges],
                         list(range(256, 256 + len(tok.merges))))
        for (a, b), new in tok.merges:
            self.assertEqual(tok.vocab[new], tok.vocab[a] + tok.vocab[b])


class Files(unittest.TestCase):
    """The .bpe and .ids files round-trip."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with

    def tearDown(self):
        self.dir.cleanup()

    def test_tokenizer_file_round_trips(self):
        tok = trained()
        path = os.path.join(self.dir.name, "t.bpe")
        mini_bpe.save_tokenizer(path, tok, [("test", "0" * 64)])
        back = mini_bpe.load_tokenizer(path)
        self.assertEqual(back.merges, tok.merges)
        self.assertEqual(back.specials, tok.specials)
        self.assertEqual(back.pattern.pattern, tok.pattern.pattern)
        self.assertEqual(back.fingerprint(), tok.fingerprint())

    def test_ids_file_round_trips(self):
        tok = trained()
        ids = mini_bpe.encode(TEXT, tok.merges, tok.pattern)
        path = os.path.join(self.dir.name, "t.ids")
        self.assertEqual(mini_bpe.write_ids(path, ids, tok.vocab_size,
                                            tok.fingerprint()), 2)
        self.assertEqual(mini_bpe.read_ids(path),
                         (ids, tok.vocab_size, tok.fingerprint()))

    def test_large_vocabularies_use_four_bytes(self):
        path = os.path.join(self.dir.name, "big.ids")
        ids = [0, 70000, 5]
        self.assertEqual(mini_bpe.write_ids(path, ids, 70001, b"\0" * 32), 4)
        self.assertEqual(mini_bpe.read_ids(path)[0], ids)

    def test_fingerprint_ignores_comments(self):
        tok = trained()
        a = os.path.join(self.dir.name, "a.bpe")
        b = os.path.join(self.dir.name, "b.bpe")
        mini_bpe.save_tokenizer(a, tok, [("one source", "1" * 64)])
        mini_bpe.save_tokenizer(b, tok, [("another", "2" * 64)])
        self.assertEqual(mini_bpe.load_tokenizer(a).fingerprint(),
                         mini_bpe.load_tokenizer(b).fingerprint())

    def test_fingerprint_depends_on_what_defines_the_tokenizer(self):
        tok = trained()
        base = tok.fingerprint()
        variants = [
            mini_bpe.Tokenizer(tok.merges[:-1], tok.vocab, tok.pattern, tok.specials),
            mini_bpe.Tokenizer(tok.merges, tok.vocab, tok.pattern, tok.specials[:-1]),
            mini_bpe.Tokenizer(tok.merges, tok.vocab, tok.pattern,
                               tok.specials[1:] + tok.specials[:1]),
            mini_bpe.Tokenizer(tok.merges, tok.vocab, mini_bpe.re.compile(r"\S+|\s+"),
                               tok.specials),
        ]
        for v in variants:
            self.assertNotEqual(v.fingerprint(), base)

    def test_bad_tokenizer_file_is_refused(self):
        path = os.path.join(self.dir.name, "bad.bpe")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("pattern x\n300 5\n")  # 300 doesn't exist yet
        with self.assertRaises(ValueError):
            mini_bpe.load_tokenizer(path)


class Fixtures(unittest.TestCase):
    """Agreement with transformer/tests/data, which the Rust tests check too."""

    def test_tiny_bpe_size_and_fingerprint(self):
        tok = mini_bpe.load_tokenizer(fixture("tiny.bpe"))
        with open(fixture("tiny.bpe.expected"), encoding="utf-8") as fh:
            size, fp = fh.read().split()
        self.assertEqual(tok.vocab_size, int(size))
        self.assertEqual(tok.fingerprint().hex(), fp)

    def test_encode_cases(self):
        tok = mini_bpe.load_tokenizer(fixture("tiny.bpe"))
        with open(fixture("encode_cases.json"), encoding="utf-8") as fh:
            cases = json.load(fh)
        self.assertGreater(len(cases), 10)
        for case in cases:
            with self.subTest(text=case["text"]):
                self.assertEqual(mini_bpe.pretokenize(case["text"], tok.pattern),
                                 case["chunks"])
                self.assertEqual(mini_bpe.encode(case["text"], tok.merges, tok.pattern),
                                 case["ids"])

    def test_tiny_ids(self):
        ids, size, fp = mini_bpe.read_ids(fixture("tiny.ids"))
        with open(fixture("tiny.ids.expected"), encoding="utf-8") as fh:
            want = fh.read().split(" ", 3)
        self.assertEqual(len(ids), int(want[0]))
        self.assertEqual(size, int(want[1]))
        self.assertEqual(fp.hex(), want[2])
        first, last = want[3].strip().split("] [")
        self.assertEqual(ids[:12], json.loads(first + "]"))
        self.assertEqual(ids[-3:], json.loads("[" + last))


class Specials(unittest.TestCase):
    """Special tokens come only from code or from marked-up conversations."""

    def test_ids_follow_the_merges(self):
        tok = trained()
        first = 256 + len(tok.merges)
        self.assertEqual([tok.special_id(n) for n in tok.specials],
                         list(range(first, first + len(tok.specials))))
        self.assertEqual(tok.vocab_size, first + len(tok.specials))
        self.assertIsNone(tok.special_id("<|nonsense|>"))

    def test_ordinary_text_never_yields_a_special_token(self):
        tok = trained()
        text = "a story <|endoftext|> with <|user|> markers <|end|>"
        ids = mini_bpe.encode(text, tok.merges, tok.pattern)
        self.assertTrue(all(i < 256 + len(tok.merges) for i in ids))

    def test_markers_become_special_tokens_on_request(self):
        tok = trained()
        text = "<|user|>Write a story.<|assistant|>Once upon a time.<|end|>"
        ids = mini_bpe.encode_with_specials(text, tok)
        for name in ("<|user|>", "<|assistant|>", "<|end|>"):
            self.assertEqual(ids.count(tok.special_id(name)), 1)
        self.assertEqual(mini_bpe.decode_with_specials(ids, tok), text)

    def test_end_of_sequence_is_not_a_text_marker(self):
        # EOS is inserted by code between documents, never read from text.
        tok = trained()
        ids = mini_bpe.encode_with_specials("end <|endoftext|>", tok)
        self.assertNotIn(tok.special_id(mini_bpe.EOS), ids)


class Documents(unittest.TestCase):
    """Splitting a corpus into documents at separator lines."""

    def test_split_documents(self):
        text = "one\nstory\n<|endoftext|>\ntwo\n<|endoftext|>\n<|endoftext|>\nthree"
        docs, seps = mini_bpe.split_documents(text, "<|endoftext|>")
        self.assertEqual(docs, ["one\nstory", "two", "three"])
        self.assertEqual(seps, 3)

    def test_separator_must_fill_the_line(self):
        docs, seps = mini_bpe.split_documents("a <|endoftext|> b\n", "<|endoftext|>")
        self.assertEqual((docs, seps), (["a <|endoftext|> b"], 0))

    def test_windows_line_endings(self):
        docs, _ = mini_bpe.split_documents("a\r\n<|endoftext|>\r\nb\r\n", "<|endoftext|>")
        self.assertEqual(docs, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
