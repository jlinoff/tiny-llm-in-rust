#!/usr/bin/env python3
"""Unit tests for acronyms.py: what counts as defined, and what is skipped.

    python3 -m unittest discover -s tools -v      (or make test-python)
"""
# Each test's name says what it checks, so they carry no docstrings.
# pylint: disable=missing-function-docstring
import os
import tempfile
import unittest

import acronyms


def problems(text, known=()):
    """The (line, word) findings for `text`, written to a temporary file."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(text)
    try:
        return acronyms.check(fh.name, set(known))
    finally:
        os.remove(fh.name)


class Definitions(unittest.TestCase):
    """What counts as a definition at first use."""

    def test_an_undefined_acronym_is_reported_once(self):
        self.assertEqual(problems("We use XYZ here.\nAnd XYZ again.\n"), [(1, "XYZ")])

    def test_the_usual_shapes_of_a_definition(self):
        for text in ["byte-pair encoding (BPE) here",
                     "BPE (byte-pair encoding) here",
                     "**BPE** (byte-pair encoding) here",
                     "a BPE library (the same interface)",
                     "a GPT-style model (GPT: see below)",
                     "- **BPE**: the algorithm",
                     "a large language model, or BPE, here"]:
            self.assertEqual(problems(text + "\n"), [], text)

    def test_a_definition_may_wrap_onto_the_next_lines(self):
        self.assertEqual(problems("part of a UTF-8\ncharacter (UTF-8, the\nstandard)\n"), [])

    def test_only_the_first_use_must_be_defined(self):
        self.assertEqual(problems("XYZ first.\n\n\n\nThen XYZ (defined late).\n"), [(1, "XYZ")])

    def test_plurals_and_numbered_names_follow_their_base(self):
        self.assertEqual(problems("an LLM (large language model); LLMs; GPT (x) and GPT-2\n"), [])

    def test_a_heading_defers_to_the_text_below(self):
        self.assertEqual(problems("## About XYZ\n\nXYZ (the thing) is ...\n"), [])
        self.assertEqual(problems("## About XYZ\n\nXYZ is ...\n"), [(3, "XYZ")])


class Skipped(unittest.TestCase):
    """What the check leaves out."""

    def test_code_links_toc_and_bibliography_are_skipped(self):
        text = ("```\nXYZ in a block\n```\n"
                "   ```\n   ABC in an indented block\n   ```\n"
                "an `XYZ` span, a [link](#XYZ-anchor), PLAN.md, a file\n"
                "<!-- toc -->\n- [XYZ](#xyz)\n<!-- /toc -->\n"
                "## Bibliography\n- Author, Q. (2020). QRS.\n")
        self.assertEqual(problems(text), [])

    def test_the_allowlist(self):
        self.assertEqual(problems("Built on GitHub.\n", known={"GitHub"}), [])
        self.assertEqual(problems("Built on GitHub.\n"), [(1, "GitHub")])

    def test_allowlist_comments(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as fh:
            fh.write("# a comment\nGitHub   # a reason\n\nMB\n")
        try:
            self.assertEqual(acronyms.load_known(fh.name), {"GitHub", "MB"})
        finally:
            os.remove(fh.name)


if __name__ == "__main__":
    unittest.main()
