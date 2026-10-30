#!/usr/bin/env python3
"""Unit tests for make_chat.py.

    python3 -m unittest discover -s chat -v      (from the top of the repository)
    make test                                    (runs these with everything else)

The property that matters most: every request is honest. Each story really
contains the three words its request names, listed in the order they first
appear, and the evaluation stories never overlap the training ones.
"""
# Each test's name says what it checks, so they carry no docstrings.
# pylint: disable=missing-function-docstring
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest

import make_chat

HERE = os.path.dirname(os.path.abspath(__file__))
STORIES = os.path.join(HERE, "..", "tools", "ci", "stories.txt")
SEP = make_chat.SEP


def words_in(story):
    """The set of candidate-shaped words in a story."""
    return set(make_chat.WORD.findall(story))


class Splitting(unittest.TestCase):
    """Stories separated by marker lines, read as text or as bytes."""

    TEXT = f"one\nstory\n{SEP}\n\n{SEP}\n  two  \n {SEP} \nthree\n"

    def test_split_stories(self):
        self.assertEqual(make_chat.split_stories(self.TEXT, SEP),
                         ["one\nstory", "two", "three"])

    def test_byte_spans_agree_with_text_split(self):
        raw = self.TEXT.encode("utf-8")
        spans = make_chat.story_spans(raw, SEP)
        self.assertEqual([make_chat.decode_story(raw, sp) for sp in spans],
                         make_chat.split_stories(self.TEXT, SEP))

    def test_marker_inside_a_line_does_not_split(self):
        self.assertEqual(make_chat.split_stories(f"a {SEP} b", SEP), [f"a {SEP} b"])


class Words(unittest.TestCase):
    """Which words may be requested."""

    STORIES = ["the dog ran to the park", "the dog saw a ball",
               "the cat saw the dog", "a Big cat ate", "the fish swam"]

    def test_min_stories_and_skip_common(self):
        words, counts = make_chat.candidate_words(self.STORIES, 2, 1)
        self.assertEqual(counts["the"], 4)
        self.assertNotIn("the", words)          # the single most widespread word
        self.assertEqual(words, {"dog", "saw", "cat"})  # each in at least 2 stories

    def test_short_and_capitalized_words_are_not_candidates(self):
        self.assertEqual(words_in("A Big ox ate an apple"), {"ate", "apple"})


class Requests(unittest.TestCase):
    """Each request names three words its story contains."""

    STORY = "The frog sat by the pond. The frog jumped. A bell rang near the egg."
    WORDS = {"frog", "pond", "bell", "egg", "rang", "jumped"}

    def test_request_words_are_in_the_story_in_order(self):
        rng = random.Random(3)
        for _ in range(50):
            picked, request = make_chat.request_for(self.STORY, self.WORDS, rng)
            self.assertEqual(len(set(picked)), 3)
            self.assertTrue(set(picked) <= words_in(self.STORY))
            first = [re.search(rf"\b{w}\b", self.STORY).start() for w in picked]
            self.assertEqual(first, sorted(first))
            for w in picked:
                self.assertIn(w, request)

    def test_too_few_words_gives_no_request(self):
        self.assertIsNone(make_chat.request_for("The frog sat by the pond.",
                                                self.WORDS, random.Random(1)))

    def test_requests_are_reproducible(self):
        a = make_chat.request_for(self.STORY, self.WORDS, random.Random(5))
        b = make_chat.request_for(self.STORY, self.WORDS, random.Random(5))
        self.assertEqual(a, b)

    def test_conversation_format(self):
        self.assertEqual(make_chat.conversation("Ask.", "Answer."),
                         "<|user|>Ask.<|assistant|>Answer.<|end|>")

    def test_stories_without_enough_words_are_skipped(self):
        convs, skipped = make_chat.conversations([self.STORY, "no words here"],
                                                 self.WORDS, random.Random(1))
        self.assertEqual((len(convs), skipped), (1, 1))
        self.assertTrue(convs[0].endswith(self.STORY + "<|end|>"))


class EndToEnd(unittest.TestCase):
    """The script on the CI stories (tools/ci/stories.txt)."""

    def test_outputs_are_honest_and_held_out(self):  # pylint: disable=too-many-locals
        with tempfile.TemporaryDirectory() as d:
            out = {k: os.path.join(d, f) for k, f in
                   [("chat", "chat.txt"), ("tsv", "eval.tsv"), ("eval", "eval.txt")]}
            subprocess.run(
                [sys.executable, os.path.join(HERE, "make_chat.py"), STORIES,
                 "-o", out["chat"], "--eval", out["tsv"], "--eval-chat", out["eval"],
                 "--min-stories", "2", "--skip-common", "20", "--eval-fraction", "0.25"],
                check=True, capture_output=True)
            with open(STORIES, encoding="utf-8") as fh:
                stories = make_chat.split_stories(fh.read(), SEP)
            n_eval = round(len(stories) * 0.25)
            train_stories, held = stories[:-n_eval], stories[-n_eval:]
            with open(out["chat"], encoding="utf-8") as fh:
                convs = make_chat.split_stories(fh.read(), SEP)
            with open(out["eval"], encoding="utf-8") as fh:
                evals = make_chat.split_stories(fh.read(), SEP)
            with open(out["tsv"], encoding="utf-8") as fh:
                rows = [line.rstrip("\n").split("\t") for line in fh]
        self.assertTrue(convs and evals)
        self.assertEqual(len(rows), len(evals))
        pattern = re.compile(r"<\|user\|>(.*)<\|assistant\|>(.*)<\|end\|>", re.S)
        for conv in convs:
            request, story = pattern.fullmatch(conv).groups()
            self.assertIn(story, train_stories)
        for (words, request), conv in zip(rows, evals):
            asked, story = pattern.fullmatch(conv).groups()
            self.assertEqual(asked, request)
            self.assertIn(story, held)          # evaluation stories are held out
            self.assertTrue(set(words.split()) <= words_in(story))


if __name__ == "__main__":
    unittest.main()
