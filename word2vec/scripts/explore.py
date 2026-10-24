#!/usr/bin/env python3
"""
explore.py — look inside a set of word vectors (word2vec text format).

Subcommands:
  neighbors  the nearest words to each given word (cosine similarity)
  analogy    a : b :: c : ?  answered by vector arithmetic, b - a + c
  evaluate   accuracy on Google's analogy test set (19,544 questions)

The analogy and evaluate commands report results two ways:
  excluding the input words  the standard convention, used in the original
                             word2vec papers and by gensim
  including the input words  the honest version: often the nearest vector
                             to king - man + woman is just "king"
                             (see Nissim, van Noord & van der Goot, 2020,
                             "Fair is Better than Sensational")

Usage:
  ./explore.py neighbors vectors.txt king cat paris
  ./explore.py analogy vectors.txt man king woman      # king - man + woman
  ./explore.py evaluate vectors.txt
"""
import argparse
import sys
import time

import numpy as np
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


def load(path):
    """Load vectors and return (KeyedVectors, unit-length vector matrix)."""
    start = time.time()
    try:
        kv = KeyedVectors.load_word2vec_format(path, binary=False)
    except (OSError, ValueError) as err:
        sys.exit(f"explore.py: error: can't load vectors from {path!r}: {err}")
    print(f"Loaded {len(kv.index_to_key):,} vectors of dimension "
          f"{kv.vector_size} from {path} ({time.time() - start:.1f} s)\n")
    return kv, kv.get_normed_vectors()


def lookup(kv, word):
    """Return the index of `word` (lowercased), or exit with a message."""
    key = word.lower()
    if key not in kv.key_to_index:
        sys.exit(f"explore.py: error: {word!r} is not in the vocabulary")
    return kv.key_to_index[key]


def top_k(scores, k, exclude=()):
    """Indices of the k highest scores, best first, skipping `exclude`."""
    scores = scores.copy()
    scores[list(exclude)] = -np.inf
    best = np.argpartition(-scores, k)[:k]
    return best[np.argsort(-scores[best])]


def cmd_neighbors(args):
    """Print the nearest neighbors of each word."""
    kv, unit = load(args.vectors)
    for word in args.words:
        i = lookup(kv, word)
        scores = unit @ unit[i]
        best = top_k(scores, args.top, exclude=[i])
        print(f"{word:>12}: " + ", ".join(
            f"{kv.index_to_key[j]} {scores[j]:.3f}" for j in best))


def cmd_analogy(args):
    """Answer a : b :: c : ? both with and without the input words."""
    kv, unit = load(args.vectors)
    ia, ib, ic = (lookup(kv, w) for w in (args.a, args.b, args.c))
    target = unit[ib] - unit[ia] + unit[ic]
    target /= np.linalg.norm(target)
    scores = unit @ target
    print(f"{args.a} : {args.b} :: {args.c} : ?     "
          f"(nearest to {args.b} - {args.a} + {args.c})\n")
    for title, exclude in [("excluding the input words (standard)",
                            [ia, ib, ic]),
                           ("including the input words", [])]:
        print(f"  {title}:")
        for rank, j in enumerate(top_k(scores, args.top, exclude), 1):
            mark = "   <- input word" if j in (ia, ib, ic) else ""
            print(f"    {rank:2}. {kv.index_to_key[j]:<16} "
                  f"{scores[j]:.3f}{mark}")
        print()


def read_questions(kv, restrict):
    """Read Google's analogy questions, keeping those whose four words are
    all among the `restrict` most frequent words. Returns a list of
    (section, array of shape (n, 4) of word indices) and the total count."""
    sections, current, rows, total = [], None, [], 0
    with open(datapath("questions-words.txt"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(":"):
                if current is not None:
                    sections.append((current, np.array(rows, dtype=int)))
                current, rows = line[1:].strip(), []
                continue
            total += 1
            ids = [kv.key_to_index.get(w.lower()) for w in line.split()]
            if all(i is not None and i < restrict for i in ids):
                rows.append(ids)
    sections.append((current, np.array(rows, dtype=int)))
    return sections, total


def solve(unit, questions, restrict, exclude_inputs):
    """Return a boolean array: is b - a + c nearest to d for each question?"""
    vocab = unit[:restrict]
    correct = np.zeros(len(questions), dtype=bool)
    for start in range(0, len(questions), 1000):
        q = questions[start:start + 1000]
        target = vocab[q[:, 1]] - vocab[q[:, 0]] + vocab[q[:, 2]]
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        scores = target @ vocab.T
        if exclude_inputs:
            rows = np.arange(len(q))[:, None]
            scores[rows, q[:, :3]] = -np.inf
        correct[start:start + len(q)] = scores.argmax(axis=1) == q[:, 3]
    return correct


def print_row(name, counts):
    """Print one results row from counts = [questions, correct excluding
    inputs, correct including inputs]."""
    n, excl, incl = counts
    print(f"  {name:<28} {n:>9,}  {100 * excl / n:>11.1f}%  "
          f"{100 * incl / n:>11.1f}%")


def cmd_evaluate(args):
    """Score the vectors on Google's analogy test set."""
    kv, unit = load(args.vectors)
    restrict = min(args.restrict, len(kv.index_to_key))
    sections, total = read_questions(kv, restrict)
    asked = sum(len(q) for _, q in sections)
    print(f"Google analogy test set: {asked:,} of {total:,} questions have "
          f"all four words among the {restrict:,} most frequent words.\n")
    if not asked:
        print("  (no questions could be asked)")
        return

    print(f"  {'section':<28} {'questions':>9}  {'excl. inputs':>12}  "
          f"{'incl. inputs':>12}")
    # The first five sections are semantic (capitals, currencies, family
    # ...); the "gram" sections are syntactic (plurals, tenses ...).
    totals = {"semantic": [0, 0, 0], "syntactic": [0, 0, 0],
              "TOTAL": [0, 0, 0]}
    for name, questions in sections:
        if len(questions) == 0:
            continue
        counts = [len(questions),
                  solve(unit, questions, restrict, True).sum(),
                  solve(unit, questions, restrict, False).sum()]
        print_row(name, counts)
        kind = "syntactic" if name.startswith("gram") else "semantic"
        for key in (kind, "TOTAL"):
            totals[key] = [a + b for a, b in zip(totals[key], counts)]
    print()
    for kind, counts in totals.items():
        if counts[0]:
            print_row(kind, counts)


def main():
    """Parse the command line and run a subcommand."""
    ap = argparse.ArgumentParser(
        description="Explore word vectors stored in word2vec text format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1].join(["usage examples:", ""]))
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("neighbors", help="nearest words to each given word")
    p.add_argument("vectors", help="word vectors file (word2vec text format)")
    p.add_argument("words", nargs="+", help="words to look up")
    p.add_argument("--top", type=int, default=10,
                   help="neighbors per word (default 10)")
    p.set_defaults(func=cmd_neighbors)

    p = sub.add_parser("analogy", help="a : b :: c : ?  via b - a + c")
    p.add_argument("vectors", help="word vectors file (word2vec text format)")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("c")
    p.add_argument("--top", type=int, default=5,
                   help="candidates shown (default 5)")
    p.set_defaults(func=cmd_analogy)

    p = sub.add_parser("evaluate",
                       help="accuracy on Google's analogy test set")
    p.add_argument("vectors", help="word vectors file (word2vec text format)")
    p.add_argument("--restrict", type=int, default=30000,
                   help="only use the N most frequent words, as is standard "
                        "(default 30000)")
    p.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
