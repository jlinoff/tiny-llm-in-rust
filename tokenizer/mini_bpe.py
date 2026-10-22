#!/usr/bin/env python3
"""
mini_bpe.py — a minimal byte-level BPE tokenizer, for understanding, not speed.

Same three stages as a real LLM tokenizer:
  1. Pre-tokenize: split text into chunks with a regex (words, numbers,
     punctuation runs, whitespace). Merges never cross chunk boundaries.
  2. Train: start from the 256 byte values, repeatedly merge the most
     frequent adjacent pair, record each merge in order.
  3. Encode/decode: apply merges in learned order; decoding just
     concatenates the bytes back — lossless by construction.

Each input file is treated as one "document". Its SHA-256 checksum is
computed so exact duplicate documents can be recognized. Without --dedup,
duplicates are reported but still counted (inflating their pair counts);
with --dedup, only the first copy is used — the exact-match stage of the
deduplication real training pipelines perform.

Usage:
  python3 mini_bpe.py corpus.txt                  # train, then type lines to tokenize
  python3 mini_bpe.py a.txt b.txt c.txt           # train on several documents
  python3 mini_bpe.py *.txt --dedup               # drop exact duplicate documents
  python3 mini_bpe.py corpus.txt -n 500           # 500 merges
  python3 mini_bpe.py corpus.txt -t "Some text."  # after training, tokenize this string
  python3 mini_bpe.py corpus.txt -f tests.txt     # after training, tokenize each line
                                                  # of tests.txt (test input, not training)
  python3 mini_bpe.py corpus.txt -q               # don't print every merge

Run with -h for all options.
"""
import argparse
import hashlib
import re
import sys
from collections import Counter

# GPT-2-style pre-tokenizer pattern (stdlib `re` version).
#   contractions | optional space + letters | optional space + digits
#   | optional space + other symbols (punctuation) | whitespace
PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


def pretokenize(text):
    return PATTERN.findall(text)


def show(tok_bytes):
    """Render a token unambiguously for display.

    Space -> '␣' (U+2423), newline -> '\\n', tab -> '\\t'. Backslashes and
    literal '␣' characters in the text are escaped first, so every rendered
    string maps back to exactly one byte sequence. Bytes that aren't valid
    UTF-8 on their own (part of a multi-byte character) show as '\\xNN'.
    """
    out = []
    i = 0
    while i < len(tok_bytes):
        # Decode one UTF-8 character at a time; fall back to \xNN for a
        # byte that doesn't start a complete character within this token.
        for width in (1, 2, 3, 4):
            try:
                ch = tok_bytes[i:i + width].decode("utf-8")
                break
            except UnicodeDecodeError:
                continue
        else:
            out.append(f"\\x{tok_bytes[i]:02x}")
            i += 1
            continue
        i += width
        out.append(DISPLAY.get(ch, ch))
    return "".join(out)


def fmt(token_list):
    """Show a sequence of tokens as [tok][tok][tok]."""
    return "".join(f"[{show(t)}]" for t in token_list)


DISPLAY = {
    " ": "␣",        # visible space ␣
    "\n": "\\n",
    "\t": "\\t",
    "\\": "\\\\",
    "␣": "\\u2423",  # a literal ␣ in the text stays distinguishable
}


def train(texts, num_merges, verbose=True):
    # Word-frequency table: each chunk as a tuple of byte-token IDs, summed
    # over all documents. Each document is pre-tokenized separately, so no
    # chunk (and hence no merge) spans two documents.
    # IDs 0..255 are the raw bytes; each merge creates ID 256, 257, ...
    chunks = Counter()
    for text in texts:
        chunks.update(tuple(c.encode("utf-8")) for c in pretokenize(text))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []  # ordered list of ((a, b), new_id)

    for step in range(num_merges):
        # Count adjacent pairs, weighted by how often each chunk occurs.
        pairs = Counter()
        for seq, n in chunks.items():
            for pair in zip(seq, seq[1:]):
                pairs[pair] += n
        if not pairs:
            break
        best, count = pairs.most_common(1)[0]
        if count < 2:
            break  # nothing repeats; further merges would be noise
        new_id = 256 + len(merges)
        merges.append((best, new_id))
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]

        # Replace the pair everywhere it occurs.
        new_chunks = Counter()
        for seq, n in chunks.items():
            new_chunks[merge_seq(seq, best, new_id)] += n
        chunks = new_chunks

        if verbose:
            print(f"merge {step+1:3d}: {show(vocab[best[0]]):>10} + "
                  f"{show(vocab[best[1]]):<10} -> {show(vocab[new_id]):<14}"
                  f" (count {count})")
    return merges, vocab


def merge_seq(seq, pair, new_id):
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


def encode(text, merges):
    ids = []
    for chunk in pretokenize(text):
        seq = tuple(chunk.encode("utf-8"))
        for pair, new_id in merges:  # apply merges in learned order
            seq = merge_seq(seq, pair, new_id)
        ids.extend(seq)
    return ids


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8")


def report(text, merges, vocab):
    ids = encode(text, merges)
    toks = fmt([vocab[i] for i in ids])
    ok = decode(ids, vocab) == text
    nbytes = len(text.encode("utf-8"))
    print(f"{text!r}\n  {len(ids)} tokens ({nbytes} bytes, "
          f"{nbytes / max(len(ids), 1):.2f} bytes/token): {toks}\n"
          f"  ids: {ids}\n  lossless round-trip: {ok}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Train a tiny byte-level BPE tokenizer on one or more "
                    "text files (documents) and use it to tokenize text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Two kinds of input:
  training  the corpus files; the tokenizer's merges are learned from these
  test      text tokenized AFTER training, to see how the learned tokenizer
            splits it: -t strings first, then -f lines; if neither is given,
            you're prompted to type lines interactively

examples:
  %(prog)s sample.txt                       train, then type test lines
  %(prog)s sample.txt -t "The cat sat."     train, tokenize one string
  %(prog)s sample.txt -f tests.txt -q       train, tokenize each line of tests.txt
  %(prog)s sample.txt news*.txt --dedup     train on several documents, dedup
""")
    ap.add_argument("corpus", nargs="+",
                    help="UTF-8 text file(s) to train the tokenizer on; "
                         "each file is one document")
    ap.add_argument("--dedup", action="store_true",
                    help="skip documents whose SHA-256 checksum matches an "
                         "earlier document (exact-duplicate removal)")
    ap.add_argument("-n", "--merges", type=int, default=200,
                    help="maximum number of merges to learn (default 200)")
    ap.add_argument("-t", "--text", action="append", default=[],
                    metavar="STRING",
                    help="after training, tokenize STRING with the learned "
                         "tokenizer and show the result (not used for "
                         "training); may be repeated")
    ap.add_argument("-f", "--test-file", metavar="FILE",
                    help="after training, tokenize each non-blank line of "
                         "FILE separately and show the results; like one -t "
                         "per line (not used for training)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="don't print each merge during training")
    args = ap.parse_args()

    # --- Load documents, checksum each, optionally drop exact duplicates ---
    print(f"=== Documents ({'dedup ON' if args.dedup else 'dedup off'}) ===")
    texts = []
    first_seen = {}  # sha256 hex -> first file with that content
    for path in args.corpus:
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as e:
            ap.error(f"can't read corpus file {path!r}: {e.strerror}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            ap.error(f"corpus file {path!r} is not valid UTF-8 text")

        digest = hashlib.sha256(raw).hexdigest()
        if digest in first_seen:
            if args.dedup:
                status = f"DUPLICATE of {first_seen[digest]} — skipped"
            else:
                status = f"DUPLICATE of {first_seen[digest]} — counted again"
                texts.append(text)
        else:
            first_seen[digest] = path
            status = "kept"
            texts.append(text)
        print(f"  {path:<24} {len(raw):>9,} bytes  sha256 {digest[:16]}…  "
              f"{status}")

    if not any(texts):
        ap.error("no text to train on (all documents empty or skipped)")

    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    chunks = [c for t in texts for c in pretokenize(t)]
    print(f"\n=== Corpus: {len(texts)} of {len(args.corpus)} documents — "
          f"{total_bytes:,} bytes, {len(chunks):,} chunks, "
          f"{len(set(chunks)):,} unique ===")
    print("First 25 chunks:", fmt([c.encode() for c in chunks[:25]]), "\n")

    print(f"=== Training: up to {args.merges} merges ===")
    merges, vocab = train(texts, args.merges, verbose=not args.quiet)
    n_tokens = sum(len(encode(t, merges)) for t in texts)
    print(f"\nLearned {len(merges)} merges; vocabulary = {len(vocab)} tokens "
          f"(256 bytes + merges).")
    print(f"Corpus encodes to {n_tokens:,} tokens "
          f"({total_bytes / n_tokens:.2f} bytes/token).")
    if args.quiet and merges:
        print("Last 10 tokens learned:",
              fmt([vocab[m[1]] for m in merges[-10:]]))

    tests = list(args.text)
    if args.test_file:
        try:
            with open(args.test_file, encoding="utf-8") as fh:
                tests += [line.rstrip("\n") for line in fh if line.strip()]
        except OSError as e:
            ap.error(f"can't read test file {args.test_file!r}: {e.strerror}")

    print("\n=== Encoding (␣ = space) ===")
    if tests:
        for t in tests:
            report(t, merges, vocab)
    else:
        print("Type a line to tokenize (Ctrl-D or empty line to quit).")
        while True:
            try:
                line = input("> ")
            except EOFError:
                print()
                break
            if not line:
                break
            report(line, merges, vocab)


if __name__ == "__main__":
    main()
