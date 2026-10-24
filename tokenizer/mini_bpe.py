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
  python3 mini_bpe.py corpus.txt --save tok.bpe   # write the trained tokenizer
  python3 mini_bpe.py --load tok.bpe -t "Text."   # use a saved tokenizer, no training
  python3 mini_bpe.py --load tok.bpe --encode corpus.txt -o corpus.ids
                                                  # encode a text file into token IDs

Run with -h for all options.
"""
import argparse
import hashlib
import os
import re
import struct
import sys
from array import array
from collections import Counter

# GPT-2-style pre-tokenizer pattern (stdlib `re` version).
#   contractions | optional space + letters | optional space + digits
#   | optional space + other symbols (punctuation) | whitespace
# Every character must be matched by some alternative: a character that
# none matches would be silently dropped. "Other symbols" is spelled
# [^\s\w]|_ because Python's \w includes "_", so [^\s\w] alone misses it.
PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)


def pretokenize(text, pattern=PATTERN):
    """Split text into chunks (words, numbers, punctuation runs, whitespace).

    BPE merges never cross a chunk boundary.
    """
    return pattern.findall(text)


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


def count_chunks(texts):
    """Build the word-frequency table the merges are learned from.

    Each chunk becomes a tuple of byte-token IDs (0-255), counted over all
    documents. Each document is pre-tokenized separately, so no chunk (and
    hence no merge) spans two documents.
    """
    chunks = Counter()
    for text in texts:
        chunks.update(tuple(c.encode("utf-8")) for c in pretokenize(text))
    return chunks


def count_pairs(chunks):
    """Count adjacent token pairs, weighted by how often each chunk occurs."""
    pairs = Counter()
    for seq, n in chunks.items():
        for pair in zip(seq, seq[1:]):
            pairs[pair] += n
    return pairs


def apply_merge(chunks, pair, new_id):
    """Replace every occurrence of `pair` in every chunk with `new_id`."""
    merged = Counter()
    for seq, n in chunks.items():
        merged[merge_seq(seq, pair, new_id)] += n
    return merged


def train(texts, num_merges, verbose=True):
    """Learn up to `num_merges` BPE merges from a list of documents.

    Starts from the 256 byte values and repeatedly merges the most frequent
    adjacent pair. Stops early once no pair occurs more than once. Returns
    (merges, vocab): merges is the ordered list of ((left_id, right_id),
    new_id), and vocab maps every token ID to its bytes.
    """
    chunks = count_chunks(texts)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    for step in range(num_merges):
        pairs = count_pairs(chunks)
        if not pairs:
            break
        best, count = pairs.most_common(1)[0]
        if count < 2:
            break  # nothing repeats; further merges would be noise
        new_id = 256 + len(merges)
        merges.append((best, new_id))
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        chunks = apply_merge(chunks, best, new_id)

        if verbose:
            print(f"merge {step+1:3d}: {show(vocab[best[0]]):>10} + "
                  f"{show(vocab[best[1]]):<10} -> {show(vocab[new_id]):<14}"
                  f" (count {count})")
    return merges, vocab


def merge_seq(seq, pair, new_id):
    """Return `seq` with each non-overlapping occurrence of `pair`, scanned
    left to right, replaced by `new_id`."""
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


def encode_chunk(chunk_bytes, ranks):
    """Encode one chunk: repeatedly apply the earliest-learned merge whose
    pair occurs in it, until none does.

    This gives exactly the same tokens as applying every merge in learned
    order (a pair can only occur once the merges that built its two parts
    have been applied), but costs time in proportion to the merges that
    actually apply, not to the total number of merges.
    """
    seq = tuple(chunk_bytes)
    while len(seq) > 1:
        best = min(zip(seq, seq[1:]), key=lambda p: ranks.get(p, sys.maxsize))
        if best not in ranks:
            break
        seq = merge_seq(seq, best, ranks[best])
    return seq


def encode(text, merges, pattern=PATTERN, cache=None):
    """Convert text to token IDs.

    `ranks` maps each merged pair to the ID it creates; since IDs are
    assigned in learning order, a smaller ID means an earlier merge. Each
    distinct chunk is encoded once and remembered in `cache`, which matters
    for a corpus, where the same words recur constantly.
    """
    ranks = dict(merges)  # (left, right) -> new ID
    cache = {} if cache is None else cache
    ids = []
    for match in pattern.finditer(text):
        chunk = match.group()
        got = cache.get(chunk)
        if got is None:
            got = cache[chunk] = encode_chunk(chunk.encode("utf-8"), ranks)
        ids.extend(got)
    return ids


def decode(ids, vocab):
    """Convert token IDs back to text by concatenating their bytes."""
    return b"".join(vocab[i] for i in ids).decode("utf-8")


def report(text, merges, vocab, pattern=PATTERN):
    """Tokenize `text` and print its tokens, IDs, compression, and a check
    that decoding reproduces it exactly."""
    ids = encode(text, merges, pattern)
    toks = fmt([vocab[i] for i in ids])
    ok = decode(ids, vocab) == text
    nbytes = len(text.encode("utf-8"))
    print(f"{text!r}\n  {len(ids)} tokens ({nbytes} bytes, "
          f"{nbytes / max(len(ids), 1):.2f} bytes/token): {toks}\n"
          f"  ids: {ids}\n  lossless round-trip: {ok}\n")


FORMAT_VERSION = "mini_bpe tokenizer v1"


def save_tokenizer(path, merges, vocab, sources):
    """Write the tokenizer (the output of the tokenizer phase) to a text file.

    A trained tokenizer is fully defined by two things:
      1. the pre-tokenizer pattern (how text is split into chunks), and
      2. the ordered list of merges.
    IDs 0-255 are the raw bytes, and the merge on data line k (counting from
    0) creates token ID 256 + k, so the vocabulary is rebuilt from the merges
    and needn't be stored. Text after '#' on a merge line is a comment.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {FORMAT_VERSION}\n")
        fh.write(f"# vocab_size: {len(vocab)}  "
                 f"(256 bytes + {len(merges)} merges)\n")
        for src, digest in sources:
            fh.write(f"# trained on: {src}  sha256 {digest}\n")
        fh.write("# Data lines: one merge per line, '<left_id> <right_id>'.\n")
        fh.write("# The merge on data line k (from 0) creates token id 256 + k.\n")
        fh.write(f"pattern {PATTERN.pattern}\n")
        for (a, b), new_id in merges:
            fh.write(f"{a} {b}  # {new_id}: [{show(vocab[a])}] + "
                     f"[{show(vocab[b])}] -> [{show(vocab[new_id])}]\n")


def load_tokenizer(path):
    """Read a file written by save_tokenizer; return (merges, vocab, pattern)."""
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    pattern = None
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if line.startswith("pattern "):
                pattern = line[len("pattern "):]
                continue
            data = line.split("#", 1)[0].split()
            if not data:
                continue  # comment or blank line
            if len(data) != 2:
                raise ValueError(f"line {lineno}: expected '<left_id> <right_id>'")
            a, b = int(data[0]), int(data[1])
            new_id = 256 + len(merges)
            if not (0 <= a < new_id and 0 <= b < new_id):
                raise ValueError(f"line {lineno}: token id out of range "
                                 f"(must be below {new_id})")
            merges.append(((a, b), new_id))
            vocab[new_id] = vocab[a] + vocab[b]
    if pattern is None:
        raise ValueError("no 'pattern' line found")
    return merges, vocab, pattern


def tokenizer_fingerprint(merges, pattern_text):
    """SHA-256 of what defines the tokenizer: its pattern and ordered merges.

    Comments and provenance in a .bpe file don't affect tokenization, so
    they're left out: two files with the same pattern and merges have the
    same fingerprint. The canonical text hashed is the pattern, a newline,
    then one "left right" line per merge.
    """
    text = pattern_text + "\n" + "".join(f"{a} {b}\n" for (a, b), _ in merges)
    return hashlib.sha256(text.encode("utf-8")).digest()


# The token-ID file (.ids): a 64-byte header, then the IDs.
#   offset  size  field
#        0     8  magic b"MBPEIDS\0"
#        8     4  format version (1)
#       12     4  bytes per ID: 2 if V <= 65,536, else 4
#       16     4  V, the tokenizer's vocabulary size
#       20     4  reserved (0)
#       24     8  number of IDs
#       32    32  tokenizer fingerprint (see tokenizer_fingerprint)
#       64     -  the IDs
# All integers are unsigned and little-endian.
IDS_MAGIC = b"MBPEIDS\0"
IDS_VERSION = 1
IDS_HEADER = struct.Struct("<8sIIIIQ32s")


def write_ids(path, ids, vocab_size, fingerprint):
    """Write token IDs to `path` in the .ids format; return bytes per ID."""
    width = 2 if vocab_size <= 1 << 16 else 4
    arr = array("H" if width == 2 else "I", ids)
    if arr.itemsize != width:  # 'I' is 4 bytes on every common platform
        raise RuntimeError(f"array type has {arr.itemsize}-byte items, "
                           f"expected {width}")
    if sys.byteorder == "big":
        arr.byteswap()
    with open(path, "wb") as fh:
        fh.write(IDS_HEADER.pack(IDS_MAGIC, IDS_VERSION, width, vocab_size,
                                 0, len(ids), fingerprint))
        fh.write(arr.tobytes())
    return width


def read_ids(path):
    """Read a .ids file; return (ids, vocab_size, fingerprint)."""
    with open(path, "rb") as fh:
        header = fh.read(IDS_HEADER.size)
        if len(header) < IDS_HEADER.size:
            raise ValueError("file too short for the header")
        magic, version, width, vocab_size, _, count, fingerprint = \
            IDS_HEADER.unpack(header)
        if magic != IDS_MAGIC or version != IDS_VERSION or width not in (2, 4):
            raise ValueError("not a mini_bpe token-ID file (version 1)")
        arr = array("H" if width == 2 else "I")
        arr.frombytes(fh.read())
    if sys.byteorder == "big":
        arr.byteswap()
    if len(arr) != count:
        raise ValueError(f"header says {count} IDs, file holds {len(arr)}")
    return arr.tolist(), vocab_size, fingerprint


def check_writable(ap, path, what):
    """Stop with an error now, before any work is done, if `path` can't be
    written. An existing file is left unchanged (it's opened for writing
    without truncation); if the file didn't exist, the empty file the check
    created is removed."""
    if os.path.isdir(path):
        ap.error(f"{what} {path!r} is a directory")
    existed = os.path.exists(path)
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o666))
    except OSError as e:
        ap.error(f"can't write {what} {path!r}: {e.strerror}")
    if not existed:
        os.remove(path)


def read_corpus(ap, paths, dedup):
    """Load documents, checksum each, optionally drop exact duplicates.

    Returns (texts, sources): the texts to train on, and (path, sha256) for
    each of them, recorded in the saved tokenizer as provenance.
    """
    print(f"=== Documents ({'dedup ON' if dedup else 'dedup off'}) ===")
    texts, sources = [], []
    first_seen = {}  # sha256 hex -> first file with that content
    for path in paths:
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
        if digest in first_seen and dedup:
            status = f"DUPLICATE of {first_seen[digest]} — skipped"
        else:
            if digest in first_seen:
                status = f"DUPLICATE of {first_seen[digest]} — counted again"
            else:
                first_seen[digest] = path
                status = "kept"
            texts.append(text)
            sources.append((path, digest))
        print(f"  {path:<24} {len(raw):>9,} bytes  sha256 {digest[:16]}…  "
              f"{status}")

    if not any(texts):
        ap.error("no text to train on (all documents empty or skipped)")
    return texts, sources


def build_parser():
    """Define the command-line interface."""
    ap = argparse.ArgumentParser(
        description="Train a tiny byte-level BPE tokenizer on one or more "
                    "text files (documents), or load a saved one, and use it "
                    "to tokenize text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
The tokenizer phase:
  input     training text: the corpus files (each file is one document)
  process   learn merges from the corpus
  output    the trained tokenizer: --save FILE writes it (pre-tokenizer
            pattern + ordered merges); --load FILE reads it back instead of
            training. Its vocabulary size is what a model's embedding table
            is sized from.

The encode phase:
  --encode TEXTFILE -o FILE turns a text file into token IDs with the trained
  or loaded tokenizer, and writes them in a binary .ids file that records
  which tokenizer produced them. This is what the model trains on.

Test input (tokenized after training or loading, never used for training):
  -t strings first, then -f lines. If neither is given and there's no --save
  or --encode, you're prompted to type lines interactively.

Output files (--save, -o) are checked before any work starts, so a bad path
fails immediately.

examples:
  %(prog)s sample.txt                       train, then type test lines
  %(prog)s sample.txt -t "The cat sat."     train, tokenize one string
  %(prog)s sample.txt -f tests.txt -q       train, tokenize each line of tests.txt
  %(prog)s sample.txt news*.txt --dedup     train on several documents, dedup
  %(prog)s sample.txt -q --save tok.bpe     train and save the tokenizer
  %(prog)s --load tok.bpe -t "The cat sat." load a saved tokenizer, no training
  %(prog)s --load tok.bpe --encode corpus.txt -o corpus.ids
                                            encode a corpus into token IDs
""")
    ap.add_argument("corpus", nargs="*",
                    help="UTF-8 text file(s) to train the tokenizer on; each "
                         "file is one document (omit when using --load)")
    ap.add_argument("--dedup", action="store_true",
                    help="skip documents whose SHA-256 checksum matches an "
                         "earlier document (exact-duplicate removal)")
    ap.add_argument("-n", "--merges", type=int, metavar="N",
                    help="maximum number of merges to learn (default 200)")
    ap.add_argument("--save", metavar="FILE",
                    help="after training, write the tokenizer to FILE")
    ap.add_argument("--load", metavar="FILE",
                    help="load a tokenizer saved with --save instead of "
                         "training (no corpus files)")
    ap.add_argument("--encode", metavar="TEXTFILE",
                    help="after training or loading, encode TEXTFILE into "
                         "token IDs and write them to the -o file")
    ap.add_argument("-o", "--output", metavar="FILE",
                    help="where --encode writes the token IDs (binary .ids "
                         "format, described in the README)")
    ap.add_argument("-t", "--text", action="append", default=[],
                    metavar="STRING",
                    help="after training or loading, tokenize STRING and show "
                         "the result (not used for training); may be repeated")
    ap.add_argument("-f", "--test-file", metavar="FILE",
                    help="after training or loading, tokenize each non-blank "
                         "line of FILE separately and show the results; like "
                         "one -t per line (not used for training)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="don't print each merge during training")
    return ap


def load_mode(ap, args):
    """Load a saved tokenizer. Returns (merges, vocab, pattern)."""
    conflicts = [name for name, given in [
        ("corpus files", args.corpus), ("--dedup", args.dedup),
        ("-n/--merges", args.merges is not None), ("--save", args.save)]
        if given]
    if conflicts:
        ap.error(f"--load uses a saved tokenizer instead of training; "
                 f"don't combine it with {', '.join(conflicts)}")
    try:
        merges, vocab, pattern_text = load_tokenizer(args.load)
    except OSError as e:
        ap.error(f"can't read tokenizer file {args.load!r}: {e.strerror}")
    except ValueError as e:
        ap.error(f"{args.load!r} is not a valid tokenizer file: {e}")

    pattern = PATTERN
    if pattern_text != PATTERN.pattern:
        print("Note: the file's pre-tokenizer pattern differs from this "
              "script's default; using the file's pattern.")
        pattern = re.compile(pattern_text)
    print(f"=== Tokenizer loaded from {args.load} ===")
    print(f"{len(merges)} merges; vocabulary = {len(vocab)} tokens "
          f"(256 bytes + merges).")
    if merges:
        print("Last 10 tokens learned:",
              fmt([vocab[m[1]] for m in merges[-10:]]))
    return merges, vocab, pattern


def train_mode(ap, args):
    """Train a tokenizer on the corpus files (and save it if asked).
    Returns (merges, vocab)."""
    if not args.corpus:
        ap.error("give corpus file(s) to train on, or --load FILE")
    num_merges = 200 if args.merges is None else args.merges
    texts, sources = read_corpus(ap, args.corpus, args.dedup)

    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    chunks = [c for t in texts for c in pretokenize(t)]
    print(f"\n=== Corpus: {len(texts)} of {len(args.corpus)} documents — "
          f"{total_bytes:,} bytes, {len(chunks):,} chunks, "
          f"{len(set(chunks)):,} unique ===")
    print("First 25 chunks:", fmt([c.encode() for c in chunks[:25]]), "\n")

    print(f"=== Training: up to {num_merges} merges ===")
    merges, vocab = train(texts, num_merges, verbose=not args.quiet)
    n_tokens = sum(len(encode(t, merges)) for t in texts)
    print(f"\nLearned {len(merges)} merges; vocabulary = {len(vocab)} "
          f"tokens (256 bytes + merges).")
    print(f"Corpus encodes to {n_tokens:,} tokens "
          f"({total_bytes / n_tokens:.2f} bytes/token).")
    if args.quiet and merges:
        print("Last 10 tokens learned:",
              fmt([vocab[m[1]] for m in merges[-10:]]))

    if args.save:
        try:
            save_tokenizer(args.save, merges, vocab, sources)
        except OSError as e:
            ap.error(f"can't write tokenizer file {args.save!r}: "
                     f"{e.strerror}")
        print(f"Saved tokenizer to {args.save} (vocab_size {len(vocab)}).")
    return merges, vocab


def read_tests(ap, args):
    """Collect test strings: -t strings first, then non-blank -f lines."""
    tests = list(args.text)
    if args.test_file:
        try:
            with open(args.test_file, encoding="utf-8") as fh:
                tests += [line.rstrip("\n") for line in fh if line.strip()]
        except OSError as e:
            ap.error(f"can't read test file {args.test_file!r}: {e.strerror}")
    return tests


def check_outputs(ap, args):
    """Validate the options that read and write files, and check that the
    text to encode is readable and every output path writable, before any
    training or encoding starts."""
    if args.output and not args.encode:
        ap.error("-o/--output is only used with --encode")
    if args.encode and not args.output:
        ap.error("--encode needs -o FILE for the token IDs")
    if args.encode and not (os.path.isfile(args.encode)
                            and os.access(args.encode, os.R_OK)):
        ap.error(f"can't read text file {args.encode!r}")
    outputs = [(args.save, "tokenizer file"), (args.output, "output file")]
    paths = [p for p, _ in outputs if p]
    if len(paths) != len({os.path.realpath(p) for p in paths}):
        ap.error("--save and -o must be different files")
    for path, what in outputs:
        if path:
            check_writable(ap, path, what)


def encode_mode(ap, args, merges, vocab, pattern):
    """Encode the --encode text file into token IDs and write the -o file."""
    try:
        with open(args.encode, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        ap.error(f"can't read text file {args.encode!r}: {e.strerror}")
    except UnicodeDecodeError:
        ap.error(f"text file {args.encode!r} is not valid UTF-8 text")

    print(f"\n=== Encode: {args.encode} -> {args.output} ===")
    ids = encode(text, merges, pattern)
    if decode(ids, vocab) != text:  # can't happen: the encoding is lossless
        ap.error("round-trip check failed: decoding didn't reproduce the text")
    fingerprint = tokenizer_fingerprint(merges, pattern.pattern)
    try:
        width = write_ids(args.output, ids, len(vocab), fingerprint)
    except OSError as e:
        ap.error(f"can't write {args.output!r}: {e.strerror}")
    nbytes = len(text.encode("utf-8"))
    print(f"  {nbytes:,} bytes of text -> {len(ids):,} tokens "
          f"({nbytes / max(len(ids), 1):.2f} bytes/token); "
          f"lossless round-trip: True")
    print(f"  wrote {IDS_HEADER.size}-byte header + {len(ids):,} IDs x "
          f"{width} bytes; V = {len(vocab):,}; "
          f"tokenizer fingerprint {fingerprint.hex()[:16]}…")


def run_tests(tests, merges, vocab, pattern, prompt):
    """Tokenize each test string; if there are none and `prompt` is true,
    ask for lines interactively."""
    if not tests and not prompt:
        return
    print("\n=== Encoding (␣ = space) ===")
    if tests:
        for t in tests:
            report(t, merges, vocab, pattern)
        return
    print("Type a line to tokenize (Ctrl-D or empty line to quit).")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            break
        if not line:
            break
        report(line, merges, vocab, pattern)


def main():
    """Train or load a tokenizer, then encode a file and/or tokenize the
    test input."""
    ap = build_parser()
    args = ap.parse_args()
    check_outputs(ap, args)
    tests = read_tests(ap, args)
    if args.load:
        merges, vocab, pattern = load_mode(ap, args)
    else:
        merges, vocab = train_mode(ap, args)
        pattern = PATTERN
    if args.encode:
        encode_mode(ap, args, merges, vocab, pattern)
    # Prompt only when the run has nothing else to do: no files to write.
    run_tests(tests, merges, vocab, pattern,
              prompt=not (args.save or args.encode))


if __name__ == "__main__":
    main()
