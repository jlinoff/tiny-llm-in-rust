#!/usr/bin/env python3
"""
make_chat.py — build fine-tuning conversations from stories (PLAN.md, step 5).

For each story, pick three uncommon words that occur in it, and write a
request around them: "Write a story with the words dog, ball and park." The
conversation is that request followed by the story itself, as the reply:

    <|user|>Write a story with the words dog, ball and park.<|assistant|>Once upon a time …<|end|>

The requests are honest: each story really does contain its three words. A
fine-tuned model can then be measured: how often do the requested words
appear in the stories it writes? That's what the evaluation requests are
for. They come from the last stories of the file (the part pretraining held
out as its validation data), so neither pretraining nor fine-tuning ever
saw them.

Which words: lowercase words of at least three letters (capitalized words
are mostly names), that occur in enough stories for the model to know them
(--min-stories), leaving out the most common ones (--skip-common), which are
function words and story staples like "the", "she", "happy" and "day".

Outputs:
  chat.txt       training conversations, separated by '<|endoftext|>' lines,
                 ready for: mini_bpe.py --load tok.bpe --encode chat.txt
                            -o chat.ids --split-on '<|endoftext|>' --allow-special
  chat-eval.tsv  evaluation requests, one per line: the three words
                 (space-separated), a tab, then the request

Usage:
  python3 make_chat.py data/TinyStories-valid.txt -o out/chat.txt --eval out/chat-eval.tsv
"""
import argparse
import os
import random
import re
import sys
from collections import Counter

SEP = "<|endoftext|>"
MARKERS = ("<|user|>", "<|assistant|>", "<|end|>")
WORD = re.compile(r"\b[a-z]{3,}\b")

# Every template lists the words, so any word fits grammatically (a noun,
# a verb or an adjective). Several phrasings, so the model learns to
# respond to a request rather than to one exact sentence.
TEMPLATES = [
    "Write a story with the words {a}, {b} and {c}.",
    "Tell me a story that uses the words {a}, {b} and {c}.",
    "Can you write a story with the words {a}, {b} and {c} in it?",
    "Please tell me a story. Use the words {a}, {b} and {c}.",
    "I want a story with these words: {a}, {b} and {c}.",
]


def split_stories(text, sep):
    """The stories in `text`, separated by lines consisting exactly of `sep`
    (blank stories skipped), each stripped of surrounding whitespace."""
    stories, current = [], []
    for line in text.splitlines():
        if line.strip() == sep:
            stories.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)
    stories.append("\n".join(current).strip())
    return [s for s in stories if s]


def candidate_words(stories, min_stories, skip_common):
    """Words eligible for requests: in at least `min_stories` stories, but
    not among the `skip_common` most widespread. Returns (set of words,
    Counter of the number of stories each word occurs in)."""
    in_stories = Counter()
    for s in stories:
        in_stories.update(set(WORD.findall(s)))
    common = {w for w, _ in in_stories.most_common(skip_common)}
    words = {w for w, n in in_stories.items()
             if n >= min_stories and w not in common}
    return words, in_stories


def request_for(story, words, rng):
    """(three words, request) for a story, or None if it has fewer than three
    candidate words. The words are listed in the order they first appear in
    the story."""
    found = sorted(set(WORD.findall(story)) & words)
    if len(found) < 3:
        return None
    picked = rng.sample(found, 3)
    picked.sort(key=lambda w: re.search(rf"\b{w}\b", story).start())
    a, b, c = picked
    return picked, rng.choice(TEMPLATES).format(a=a, b=b, c=c)


def check_outputs(ap, args):
    """Stop before any work if the input can't be read or an output can't
    be written, or an output would overwrite the input."""
    if not os.path.isfile(args.stories) or not os.access(args.stories, os.R_OK):
        ap.error(f"can't read {args.stories!r}")
    for path in (args.output, args.eval):
        if os.path.exists(path) and os.path.samefile(path, args.stories):
            ap.error(f"output {path!r} would overwrite the input")
        parent = os.path.dirname(os.path.abspath(path))
        if not os.access(parent, os.W_OK) or os.path.isdir(path):
            ap.error(f"can't write {path!r}")
    if args.output == args.eval:
        ap.error("--output and --eval must be different files")
    if not 0 < args.eval_fraction < 1:
        ap.error("--eval-fraction must be between 0 and 1")


def build_parser():
    """The command line."""
    ap = argparse.ArgumentParser(
        description="Build fine-tuning conversations from stories: a request "
                    "naming three words from each story, answered by the story.")
    ap.add_argument("stories", help="UTF-8 text file of stories")
    ap.add_argument("-o", "--output", required=True,
                    help="the training conversations (chat.txt)")
    ap.add_argument("--eval", required=True,
                    help="the evaluation requests (chat-eval.tsv)")
    ap.add_argument("--split-on", default=SEP, metavar="SEP",
                    help=f"line separating the stories (default {SEP})")
    ap.add_argument("--eval-fraction", type=float, default=0.05,
                    help="the last fraction of the stories, kept for "
                         "evaluation (default 0.05, pretraining's held-out part)")
    ap.add_argument("--eval-count", type=int, default=200,
                    help="evaluation requests to write (default 200)")
    ap.add_argument("--min-stories", type=int, default=20,
                    help="a requested word must occur in at least this many "
                         "stories (default 20)")
    ap.add_argument("--skip-common", type=int, default=300,
                    help="leave out the words occurring in the most stories, "
                         "this many of them (default 300)")
    ap.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    return ap


def conversations(stories, words, rng):
    """The training conversations, and how many stories had too few
    candidate words to make one."""
    convs, skipped = [], 0
    for story in stories:
        got = request_for(story, words, rng)
        if got is None:
            skipped += 1
            continue
        convs.append(f"{MARKERS[0]}{got[1]}{MARKERS[1]}{story}{MARKERS[2]}")
    return convs, skipped


def write_outputs(args, convs, evals):
    """Write both files, then read the conversations back to verify them."""
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(f"\n{args.split_on}\n".join(convs) + "\n")
    with open(args.eval, "w", encoding="utf-8") as fh:
        for picked, request in evals:
            fh.write(f"{' '.join(picked)}\t{request}\n")
    with open(args.output, encoding="utf-8") as fh:
        if split_stories(fh.read(), args.split_on) != convs:
            sys.exit(f"make_chat.py: {args.output}: read-back doesn't match")


def main():
    """Read the stories, write the conversations and evaluation requests."""
    ap = build_parser()
    args = ap.parse_args()
    check_outputs(ap, args)
    with open(args.stories, encoding="utf-8") as fh:
        stories = split_stories(fh.read(), args.split_on)
    if any(m in s for s in stories for m in MARKERS + (SEP,)):
        ap.error("a story contains a turn marker or the separator")
    n_eval = round(len(stories) * args.eval_fraction)
    words, in_stories = candidate_words(stories, args.min_stories,
                                        args.skip_common)
    rng = random.Random(args.seed)
    convs, skipped = conversations(stories[:-n_eval], words, rng)
    evals = [g for g in (request_for(s, words, rng) for s in stories[-n_eval:])
             if g][:args.eval_count]
    if not convs or len(evals) < 2:
        ap.error(f"only {len(words)} candidate words, too few to make "
                 f"requests: lower --skip-common or --min-stories")
    write_outputs(args, convs, evals)

    print(f"{args.stories}: {len(stories):,} stories; the last {n_eval:,} "
          f"held out for evaluation")
    print(f"Candidate words: {len(words):,} (in at least {args.min_stories} "
          f"stories, leaving out the {args.skip_common} most widespread, "
          f"from {', '.join(w for w, _ in in_stories.most_common(8))} …)")
    print(f"Wrote {args.output}: {len(convs):,} conversations ({skipped:,} "
          f"stories skipped, fewer than three candidate words); read back and "
          f"verified")
    print(f"Wrote {args.eval}: {len(evals):,} evaluation requests, e.g.")
    for _, request in evals[:3]:
        print(f"  {request}")


if __name__ == "__main__":
    main()
