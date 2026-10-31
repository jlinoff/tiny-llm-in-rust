#!/usr/bin/env python3
"""Unit tests for join_shards.py's checks, which need no torch or safetensors.

    python3 -m unittest discover -s tools -v      (or make test-python)
"""
# Each test's name says what it checks, so they carry no docstrings.
# pylint: disable=missing-function-docstring
import json
import os
import tempfile
import unittest

import join_shards


class Merge(unittest.TestCase):
    """Merging shards, and the checks before anything is written."""

    def test_merges_every_shard(self):
        got = join_shards.merge([{"a": 1, "b": 2}, {"c": 3}], ["a", "b", "c"])
        self.assertEqual(got, {"a": 1, "b": 2, "c": 3})

    def test_no_shards_is_an_error_not_an_empty_model(self):
        with self.assertRaisesRegex(ValueError, "no shards"):
            join_shards.merge([])

    def test_a_tensor_in_two_shards_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "b is in more than one shard"):
            join_shards.merge([{"a": 1, "b": 2}, {"b": 3}])

    def test_a_missing_shard_is_caught_by_the_index(self):
        with self.assertRaisesRegex(ValueError, r"missing \['c'\]"):
            join_shards.merge([{"a": 1, "b": 2}], ["a", "b", "c"])

    def test_an_extra_tensor_is_caught_by_the_index(self):
        with self.assertRaisesRegex(ValueError, r"extra \['z'\]"):
            join_shards.merge([{"a": 1, "z": 9}], ["a"])

    def test_without_an_index_only_duplicates_are_checked(self):
        self.assertEqual(join_shards.merge([{"a": 1}, {"b": 2}]), {"a": 1, "b": 2})


class Files(unittest.TestCase):
    """Finding the shards and reading the index."""

    def test_finds_shards_in_order_and_reads_the_index(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors",
                         "model.safetensors", "other.safetensors"]:
                with open(os.path.join(d, name), "w", encoding="utf-8"):
                    pass
            self.assertEqual([os.path.basename(p) for p in join_shards.find_shards(d)],
                             ["model-00001-of-00002.safetensors",
                              "model-00002-of-00002.safetensors"])
            self.assertIsNone(join_shards.read_index(d))
            with open(os.path.join(d, "model.safetensors.index.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"metadata": {}, "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors"}}, fh)
            self.assertEqual(sorted(join_shards.read_index(d)), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
