#!/usr/bin/env python3
"""
train_gensim.py — train word vectors with gensim, as a reference for mini_w2v.

gensim is a widely used Python library whose word2vec is a careful port of the
original C code (its inner loops are compiled). Training it on the same corpus
with the same settings gives a baseline: if mini_w2v's vectors score about as
well on the analogy test (explore.py evaluate), its implementation is sound.

The corpus is read the same way as mini_w2v: each file is one document, split
into lowercase words at every character that isn't a letter or digit, and cut
into "sentences" of at most 1,000 words that never cross a document boundary.
(For plain ASCII text such as text8 the split is identical; for unusual Unicode
characters the two languages' definitions of "alphanumeric" differ slightly.)

The options and defaults match mini_w2v's.

Usage:
  ./train_gensim.py ../../data/text8 -o gensim.txt
  ./train_gensim.py ../../data/text8 --model cbow -o gensim_cbow.txt
"""
import argparse
import os
import re
import sys
import time

from gensim.models import Word2Vec

MAX_SENTENCE = 1000
WORD = re.compile(r"[^\W_]+")  # runs of letters and digits (\w minus "_")


def read_sentences(paths):
    """Return the corpus as a list of sentences (lists of words)."""
    sentences = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                words = WORD.findall(fh.read().lower())
        except (OSError, UnicodeDecodeError) as err:
            sys.exit(f"train_gensim.py: error: can't read {path!r}: {err}")
        sentences += [words[i:i + MAX_SENTENCE]
                      for i in range(0, len(words), MAX_SENTENCE)]
    return sentences


class EpochReporter:  # pylint: disable=too-few-public-methods
    """gensim callback: print the elapsed time after every epoch."""

    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch = 0
        self.start = time.time()

    def on_epoch_end(self, _model):
        """Called by gensim at the end of each epoch."""
        self.epoch += 1
        print(f"  epoch {self.epoch}/{self.epochs}: "
              f"{time.time() - self.start:.1f} s elapsed")

    def on_epoch_begin(self, _model):
        """Called by gensim at the start of each epoch (unused)."""

    def on_train_begin(self, _model):
        """Called by gensim before training (unused)."""

    def on_train_end(self, _model):
        """Called by gensim after training (unused)."""


def main():
    """Parse arguments, train, and save the vectors."""
    ap = argparse.ArgumentParser(
        description="Train word vectors with gensim using mini_w2v's "
                    "settings, as a reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1].join(["usage examples:", ""]))
    ap.add_argument("corpus", nargs="+",
                    help="UTF-8 text file(s); each file is one document")
    ap.add_argument("-o", "--output", default="gensim.txt",
                    help="output vectors file (default gensim.txt)")
    ap.add_argument("-m", "--model", choices=["skipgram", "cbow"],
                    default="skipgram", help="training task (default skipgram)")
    ap.add_argument("-d", "--dim", type=int, default=100,
                    help="vector dimension d (default 100)")
    ap.add_argument("-w", "--window", type=int, default=5,
                    help="maximum context window (default 5)")
    ap.add_argument("-k", "--negative", type=int, default=5,
                    help="negative samples per prediction (default 5)")
    ap.add_argument("-e", "--epochs", type=int, default=5,
                    help="passes over the corpus (default 5)")
    ap.add_argument("--min-count", type=int, default=5,
                    help="ignore words rarer than this (default 5)")
    ap.add_argument("--sample", type=float, default=1e-3,
                    help="subsampling threshold (default 0.001)")
    ap.add_argument("--alpha", type=float,
                    help="starting learning rate (default 0.025 skipgram, "
                         "0.05 cbow)")
    ap.add_argument("-t", "--threads", type=int, default=os.cpu_count(),
                    help="worker threads (default: all CPU cores)")
    ap.add_argument("--seed", type=int, default=1, help="random seed")
    args = ap.parse_args()

    skipgram = args.model == "skipgram"
    alpha = args.alpha or (0.025 if skipgram else 0.05)

    start = time.time()
    sentences = read_sentences(args.corpus)
    n_words = sum(len(s) for s in sentences)
    print(f"Read {n_words:,} words in {len(sentences):,} sentences "
          f"({time.time() - start:.1f} s)")

    start = time.time()
    model = Word2Vec(
        sentences=sentences, sg=int(skipgram), vector_size=args.dim,
        window=args.window, negative=args.negative, hs=0,
        min_count=args.min_count, sample=args.sample, alpha=alpha,
        min_alpha=alpha * 1e-4, epochs=args.epochs, workers=args.threads,
        seed=args.seed, cbow_mean=1, callbacks=[EpochReporter(args.epochs)])
    print(f"Trained {args.model}: V = {len(model.wv):,}, d = {args.dim}, "
          f"{args.epochs} epochs, {args.threads} threads, "
          f"{time.time() - start:.1f} s")

    model.wv.save_word2vec_format(args.output, binary=False)
    print(f"Saved vectors to {args.output}")


if __name__ == "__main__":
    main()
