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

Each input file is treated as one "document", unless --split-on SEP is
given: then every line consisting exactly of SEP separates documents within
a file (TinyStories, for example, puts "<|endoftext|>" on a line between
stories). The separator lines themselves are dropped; they're never encoded
as text. Each file's SHA-256 checksum is computed so exact duplicate files
can be recognized. Without --dedup,
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
  python3 mini_bpe.py --load tok.bpe --encode a.txt b.txt -o ab.ids
                                                  # two documents, each ending in EOS
  python3 mini_bpe.py --load tok.bpe --encode stories.txt -o s.ids --split-on '<|endoftext|>'
                                                  # one document per story, each ending in EOS

Run with -h for all options.
"""
import argparse
import hashlib
import heapq
import os
import re
import struct
import sys
import time
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass

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


def report_merge(step, vocab, pair, new_id, count):
    """Print one line of the merge log."""
    print(f"merge {step+1:3d}: {show(vocab[pair[0]]):>10} + "
          f"{show(vocab[pair[1]]):<10} -> {show(vocab[new_id]):<14}"
          f" (count {count})")


def train_reference(texts, num_merges, verbose=True):
    """The simple BPE trainer, kept as the readable reference.

    After every merge it recounts every pair in every chunk, which makes the
    algorithm easy to see but slow: the cost of each merge is proportional to
    the whole corpus. train() produces exactly the same merges, faster.

    Starts from the 256 byte values and repeatedly merges the most frequent
    adjacent pair; ties go to the pair that occurs first in the corpus (the
    first chunk, then the leftmost position). Stops early once no pair occurs
    more than once. Returns (merges, vocab): merges is the ordered list of
    ((left_id, right_id), new_id), and vocab maps every token ID to its bytes.
    """
    chunks = count_chunks(texts)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    for step in range(num_merges):
        pairs = count_pairs(chunks)
        if not pairs:
            break
        # most_common(1) returns the first maximum in insertion order, i.e.
        # the pair whose first occurrence comes earliest in the corpus.
        best, count = pairs.most_common(1)[0]
        if count < 2:
            break  # nothing repeats; further merges would be noise
        new_id = 256 + len(merges)
        merges.append((best, new_id))
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        chunks = apply_merge(chunks, best, new_id)
        if verbose:
            report_merge(step, vocab, best, new_id, count)
    return merges, vocab


class IncrementalPairs:
    """Pair counts that are updated, not recounted, after each merge.

    A merge only changes the chunks that contain the merged pair, so only
    those chunks are re-examined: their old pairs are subtracted from the
    counts and their new pairs added. To find the most frequent pair quickly,
    every pair's current ranking key is kept in a heap. Entries are never
    removed when a key changes; a new entry is pushed instead, and old ones
    are recognized as stale and skipped when they reach the top.

    A pair's ranking key is (-count, first chunk, byte offset): highest count
    first, ties to the earliest first occurrence, so the merges are identical
    to train_reference(). Chunks are numbered in the order they first appear,
    and the byte offset (where the pair's left token starts in the chunk's
    original bytes) orders positions within a chunk; unlike a token index, it
    doesn't shift as merges shorten the chunk.
    """

    def __init__(self, texts):
        chunks = count_chunks(texts)  # insertion order = first appearance
        self.chunks = [(tuple(seq), n) for seq, n in chunks.items()]
        self.tok_len = {i: 1 for i in range(256)}  # token ID -> bytes
        self.counts = Counter()  # pair -> occurrences, weighted by freq
        self.where = defaultdict(set)  # pair -> chunks containing it
        for ci, (seq, n) in enumerate(self.chunks):
            for pair in zip(seq, seq[1:]):
                self.counts[pair] += n
                self.where[pair].add(ci)
        self.first = {p: min(cs) for p, cs in self.where.items()}
        self.heap = []
        self.pushed = {}  # pair -> the key of its newest heap entry
        for pair in self.counts:
            self._push(pair)

    def _key(self, pair):
        """(-count, first chunk, byte offset of the first occurrence)."""
        seq, offset = self.chunks[self.first[pair]][0], 0
        for a, b in zip(seq, seq[1:]):
            if (a, b) == pair:
                return (-self.counts[pair], self.first[pair], offset)
            offset += self.tok_len[a]
        raise AssertionError(f"pair {pair} missing from its first chunk")

    def _push(self, pair):
        key = self._key(pair)
        if self.pushed.get(pair) != key:
            self.pushed[pair] = key
            heapq.heappush(self.heap, (*key, pair))

    def best(self):
        """Return (pair, count) for the next merge, or (None, 0) if none."""
        while self.heap:
            *key, pair = self.heap[0]
            if self.pushed.get(pair) == tuple(key):
                return pair, -key[0]
            heapq.heappop(self.heap)  # stale: the pair's key has changed
        return None, 0

    def merge(self, pair, new_id):
        """Replace `pair` with `new_id` everywhere and update the counts."""
        self.tok_len[new_id] = self.tok_len[pair[0]] + self.tok_len[pair[1]]
        touched, recheck = set(), set()
        for ci in list(self.where[pair]):
            old, n = self.chunks[ci]
            new = merge_seq(old, pair, new_id)
            old_pairs = Counter(zip(old, old[1:]))
            new_pairs = Counter(zip(new, new[1:]))
            for p, k in old_pairs.items():
                self.counts[p] -= k * n
            for p, k in new_pairs.items():
                self.counts[p] += k * n
            for p in old_pairs.keys() - new_pairs.keys():
                self.where[p].discard(ci)
                if self.first.get(p) == ci:
                    recheck.add(p)  # lost its first chunk
            for p in new_pairs.keys() - old_pairs.keys():
                self.where[p].add(ci)
                if p not in self.first or ci < self.first[p]:
                    self.first[p] = ci
            self.chunks[ci] = (new, n)
            touched |= old_pairs.keys() | new_pairs.keys()
        for p in touched:
            if self.counts[p] <= 0:
                for table in (self.counts, self.where, self.first, self.pushed):
                    table.pop(p, None)
                continue
            if p in recheck:
                self.first[p] = min(self.where[p])
            self._push(p)


def train(texts, num_merges, verbose=True):
    """Learn up to `num_merges` BPE merges from a list of documents.

    Gives exactly the same merges as train_reference(), but updates the pair
    counts incrementally (see IncrementalPairs) instead of recounting the
    whole corpus after every merge. Returns (merges, vocab).
    """
    pairs = IncrementalPairs(texts)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for step in range(num_merges):
        best, count = pairs.best()
        if best is None or count < 2:
            break  # nothing repeats; further merges would be noise
        new_id = 256 + len(merges)
        merges.append((best, new_id))
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        pairs.merge(best, new_id)
        if verbose:
            report_merge(step, vocab, best, new_id, count)
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


# Special tokens: token IDs reserved by hand for structure, not text. They
# get the IDs after the merges and never come out of encoding text: if a
# document contains the characters "<|endoftext|>", they're encoded as
# ordinary bytes. Only code inserts a special token's ID.
EOS = "<|endoftext|>"  # end of sequence: marks the end of each document
DEFAULT_SPECIALS = [EOS]


@dataclass
class Tokenizer:
    """Everything that defines a trained tokenizer.

    merges:   ordered list of ((left_id, right_id), new_id)
    vocab:    token ID -> bytes, for the 256 bytes and every merge
    pattern:  the compiled pre-tokenizer regex
    specials: special-token names; the i-th has ID 256 + len(merges) + i
    """
    merges: list
    vocab: dict
    pattern: "re.Pattern[str]"
    specials: list

    @property
    def vocab_size(self):
        """V: bytes + merges + special tokens."""
        return 256 + len(self.merges) + len(self.specials)

    def special_id(self, name):
        """The ID of a special token, or None if the tokenizer lacks it."""
        if name not in self.specials:
            return None
        return 256 + len(self.merges) + self.specials.index(name)

    def fingerprint(self):
        """SHA-256 of what defines the tokenizer: its pattern, ordered merges
        and special tokens.

        Comments and provenance in a .bpe file don't affect tokenization, so
        they're left out: two files with the same pattern, merges and special
        tokens have the same fingerprint. The canonical text hashed is the
        pattern, a newline, one "left right" line per merge, then one
        "special NAME" line per special token.
        """
        text = (self.pattern.pattern + "\n"
                + "".join(f"{a} {b}\n" for (a, b), _ in self.merges)
                + "".join(f"special {name}\n" for name in self.specials))
        return hashlib.sha256(text.encode("utf-8")).digest()


def save_tokenizer(path, tok, sources):
    """Write the tokenizer (the output of the tokenizer phase) to a text file.

    A trained tokenizer is fully defined by three things:
      1. the pre-tokenizer pattern (how text is split into chunks),
      2. the ordered list of merges, and
      3. the special tokens.
    IDs 0-255 are the raw bytes, the merge on data line k (counting from 0)
    creates token ID 256 + k, and the special tokens take the IDs after the
    merges, in order. So the vocabulary is rebuilt from the file and needn't
    be stored. Text after '#' on a data line is a comment.
    """
    vocab = tok.vocab
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {FORMAT_VERSION}\n")
        fh.write(f"# vocab_size: {tok.vocab_size}  (256 bytes + "
                 f"{len(tok.merges)} merges + {len(tok.specials)} special)\n")
        for src, digest in sources:
            fh.write(f"# trained on: {src}  sha256 {digest}\n")
        fh.write("# Merge lines: '<left_id> <right_id>'; the merge on merge "
                 "line k (from 0)\n# creates token id 256 + k. Special "
                 "lines: 'special <name>', IDs after the merges.\n")
        fh.write(f"pattern {tok.pattern.pattern}\n")
        for (a, b), new_id in tok.merges:
            fh.write(f"{a} {b}  # {new_id}: [{show(vocab[a])}] + "
                     f"[{show(vocab[b])}] -> [{show(vocab[new_id])}]\n")
        for name in tok.specials:
            fh.write(f"special {name}  # {tok.special_id(name)}\n")


def load_tokenizer(path):
    """Read a file written by save_tokenizer; return a Tokenizer."""
    vocab = {i: bytes([i]) for i in range(256)}
    merges, specials = [], []
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
            if data[0] == "special":
                if len(data) != 2 or data[1] in specials:
                    raise ValueError(f"line {lineno}: expected 'special "
                                     f"<name>' with a new, space-free name")
                specials.append(data[1])
                continue
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
    return Tokenizer(merges, vocab, re.compile(pattern), specials)


# The token-ID file (.ids): a 64-byte header, then the IDs.
#   offset  size  field
#        0     8  magic b"MBPEIDS\0"
#        8     4  format version (1)
#       12     4  bytes per ID: 2 if V <= 65,536, else 4
#       16     4  V, the tokenizer's vocabulary size
#       20     4  reserved (0)
#       24     8  number of IDs
#       32    32  tokenizer fingerprint (see Tokenizer.fingerprint)
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


def split_documents(text, sep):
    """Split text into documents at lines consisting exactly of `sep`.

    The separator lines are dropped, along with one newline at the end of
    each document (so a story ends with its last word, not a line break).
    Empty documents, such as from two separators in a row, are skipped.
    Returns (documents, number of separator lines).
    """
    docs, current, seps = [], [], 0
    for line in text.splitlines(keepends=True):
        if line.rstrip("\r\n") == sep:
            seps += 1
            docs.append("".join(current))
            current = []
        else:
            current.append(line)
    docs.append("".join(current))
    docs = [d[:-2] if d.endswith("\r\n") else d[:-1] if d.endswith("\n") else d
            for d in docs]
    return [d for d in docs if d], seps


def add_documents(texts, text, split_on):
    """Append a file's documents to `texts`: the whole file, or its parts
    between separator lines. Returns a note for the report."""
    if split_on is None:
        texts.append(text)
        return ""
    docs, seps = split_documents(text, split_on)
    texts += docs
    return f", {len(docs):,} documents ({seps:,} separators)"


def read_corpus(ap, paths, dedup, split_on=None):
    """Load documents, checksum each file, optionally drop exact duplicates.

    With `split_on`, each file is split into documents at its separator
    lines (see split_documents). Returns (texts, sources): the documents to
    train on, and (path, sha256) for each file, recorded in the saved
    tokenizer as provenance.
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
            status += add_documents(texts, text, split_on)
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
    ap.add_argument("--encode", metavar="TEXTFILE", nargs="+",
                    help="after training or loading, encode the text "
                         "file(s) into token IDs, each file a document "
                         "followed by EOS, and write them to the -o file")
    ap.add_argument("--split-on", metavar="SEP",
                    help="split each corpus or --encode file into documents "
                         "at lines consisting exactly of SEP (dropping those "
                         "lines), e.g. '<|endoftext|>' for TinyStories; "
                         "without it, each file is one document")
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
    ap.add_argument("--reference-trainer", action="store_true",
                    help="train with the simple, slow reference trainer "
                         "(same merges; for comparison and reading)")
    return ap


def describe(tok):
    """One line describing the vocabulary."""
    return (f"vocabulary V = {tok.vocab_size} tokens (256 bytes + "
            f"{len(tok.merges)} merges + {len(tok.specials)} special: "
            f"{', '.join(tok.specials) or 'none'})")


def load_mode(ap, args):
    """Load a saved tokenizer. Returns a Tokenizer."""
    conflicts = [name for name, given in [
        ("corpus files", args.corpus), ("--dedup", args.dedup),
        ("-n/--merges", args.merges is not None), ("--save", args.save)]
        if given]
    if conflicts:
        ap.error(f"--load uses a saved tokenizer instead of training; "
                 f"don't combine it with {', '.join(conflicts)}")
    try:
        tok = load_tokenizer(args.load)
    except (OSError, re.error) as e:
        ap.error(f"can't read tokenizer file {args.load!r}: {e}")
    except ValueError as e:
        ap.error(f"{args.load!r} is not a valid tokenizer file: {e}")

    if tok.pattern.pattern != PATTERN.pattern:
        print("Note: the file's pre-tokenizer pattern differs from this "
              "script's default; using the file's pattern.")
    print(f"=== Tokenizer loaded from {args.load} ===")
    print(f"{len(tok.merges)} merges; {describe(tok)}.")
    if tok.merges:
        print("Last 10 tokens learned:",
              fmt([tok.vocab[m[1]] for m in tok.merges[-10:]]))
    return tok


def train_mode(ap, args):
    """Train a tokenizer on the corpus files (and save it if asked).
    Returns a Tokenizer."""
    if not args.corpus:
        ap.error("give corpus file(s) to train on, or --load FILE")
    num_merges = 200 if args.merges is None else args.merges
    texts, sources = read_corpus(ap, args.corpus, args.dedup, args.split_on)

    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    chunks = [c for t in texts for c in pretokenize(t)]
    print(f"\n=== Corpus: {len(texts):,} "
          f"{'documents' if args.split_on else f'of {len(args.corpus)} documents'} — "
          f"{total_bytes:,} bytes, {len(chunks):,} chunks, "
          f"{len(set(chunks)):,} unique ===")
    print("First 25 chunks:", fmt([c.encode() for c in chunks[:25]]), "\n")

    print(f"=== Training: up to {num_merges} merges ===")
    trainer = train_reference if args.reference_trainer else train
    start = time.time()
    merges, vocab = trainer(texts, num_merges, verbose=not args.quiet)
    elapsed = time.time() - start
    tok = Tokenizer(merges, vocab, PATTERN, list(DEFAULT_SPECIALS))
    n_tokens = sum(len(encode(t, merges)) for t in texts)
    print(f"\nLearned {len(merges)} merges in {elapsed:.1f} s"
          f"{' (reference trainer)' if args.reference_trainer else ''}; "
          f"{describe(tok)}.")
    print(f"Corpus encodes to {n_tokens:,} tokens "
          f"({total_bytes / n_tokens:.2f} bytes/token).")
    if args.quiet and merges:
        print("Last 10 tokens learned:",
              fmt([vocab[m[1]] for m in merges[-10:]]))

    if args.save:
        try:
            save_tokenizer(args.save, tok, sources)
        except OSError as e:
            ap.error(f"can't write tokenizer file {args.save!r}: "
                     f"{e.strerror}")
        print(f"Saved tokenizer to {args.save} (vocab_size {tok.vocab_size}).")
    return tok


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
    texts to encode are readable and every output path writable, before any
    training or encoding starts."""
    if args.output and not args.encode:
        ap.error("-o/--output is only used with --encode")
    if args.split_on is not None and (not args.split_on.strip()
                                      or "\n" in args.split_on
                                      or "\r" in args.split_on):
        ap.error("--split-on needs a non-blank separator without line breaks")
    if args.encode and not args.output:
        ap.error("--encode needs -o FILE for the token IDs")
    for path in args.encode or []:
        if not (os.path.isfile(path) and os.access(path, os.R_OK)):
            ap.error(f"can't read text file {path!r}")
    outputs = [(args.save, "tokenizer file"), (args.output, "output file")]
    paths = [p for p, _ in outputs if p]
    if len(paths) != len({os.path.realpath(p) for p in paths}):
        ap.error("--save and -o must be different files")
    for path, what in outputs:
        if path:
            check_writable(ap, path, what)


def read_text(ap, path):
    """Read a UTF-8 text file, or stop with an error."""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as e:
        ap.error(f"can't read text file {path!r}: {e.strerror}")
    except UnicodeDecodeError:
        ap.error(f"text file {path!r} is not valid UTF-8 text")
    return None  # not reached: ap.error exits


def encode_file(ap, path, tok, cache, split_on):
    """Encode one --encode file: its documents, each followed by EOS, with
    a round-trip check per document. Prints a line about the file.
    Returns (IDs, number of documents, bytes of document text)."""
    text = read_text(ap, path)
    docs, seps = (([text], None) if split_on is None
                  else split_documents(text, split_on))
    ids, nbytes = [], 0
    for i, doc_text in enumerate(docs):
        doc = encode(doc_text, tok.merges, tok.pattern, cache)
        # Round-trip check per document: the IDs between EOS tokens must
        # decode to exactly this document. Encoding text never produces the
        # EOS ID, so the document boundaries are unambiguous.
        if decode(doc, tok.vocab) != doc_text:
            ap.error(f"round-trip check failed for {path!r}"
                     f"{'' if seps is None else f', document {i + 1}'}: "
                     f"decoding didn't reproduce the text")
        ids += doc + [tok.special_id(EOS)]
        nbytes += len(doc_text.encode("utf-8"))
    head = f"  {path:<24} {len(text.encode('utf-8')):>12,} bytes -> "
    if seps is None:
        print(f"{head}{len(ids) - 1:>11,} tokens + EOS")
    else:
        print(f"{head}{len(docs):,} documents ({seps:,} separator lines "
              f"dropped), {len(ids) - len(docs):,} tokens + one EOS each")
    return ids, len(docs), nbytes


def encode_mode(ap, args, tok):
    """Encode the --encode text files into token IDs and write the -o file.

    Each file is one document, or with --split-on, each part of a file
    between separator lines (the separator lines are dropped, not encoded).
    Each document's tokens are followed by the EOS token, so the model can
    learn where a document ends, both to keep unrelated documents apart
    during training and to know when to stop generating. Literal EOS text
    inside a document is still encoded as ordinary bytes: only the named
    separator, on a line of its own, marks a boundary.
    """
    eos = tok.special_id(EOS)
    if eos is None:
        ap.error(f"the tokenizer has no {EOS} (EOS) token; it was saved by "
                 f"an older version of mini_bpe.py, so retrain it")
    print(f"\n=== Encode: {' '.join(args.encode)} -> {args.output} ===")
    ids, cache, nbytes, ndocs = [], {}, 0, 0
    for path in args.encode:
        file_ids, docs, doc_bytes = encode_file(ap, path, tok, cache,
                                                args.split_on)
        ids += file_ids
        ndocs += docs
        nbytes += doc_bytes
    fingerprint = tok.fingerprint()
    try:
        width = write_ids(args.output, ids, tok.vocab_size, fingerprint)
    except OSError as e:
        ap.error(f"can't write {args.output!r}: {e.strerror}")
    print(f"  {ndocs:,} document(s), {nbytes:,} bytes of text -> "
          f"{len(ids):,} IDs including {ndocs:,} EOS (ID {eos}); "
          f"{nbytes / max(len(ids) - ndocs, 1):.2f} bytes/token; "
          f"lossless round-trip of every document: True")
    print(f"  wrote {IDS_HEADER.size}-byte header + {len(ids):,} IDs x "
          f"{width} bytes; V = {tok.vocab_size:,}; "
          f"tokenizer fingerprint {fingerprint.hex()[:16]}…")


def run_tests(tests, tok, prompt):
    """Tokenize each test string; if there are none and `prompt` is true,
    ask for lines interactively."""
    if not tests and not prompt:
        return
    print("\n=== Encoding (␣ = space) ===")
    if tests:
        for t in tests:
            report(t, tok.merges, tok.vocab, tok.pattern)
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
        report(line, tok.merges, tok.vocab, tok.pattern)


def main():
    """Train or load a tokenizer, then encode a file and/or tokenize the
    test input."""
    ap = build_parser()
    args = ap.parse_args()
    check_outputs(ap, args)
    tests = read_tests(ap, args)
    tok = load_mode(ap, args) if args.load else train_mode(ap, args)
    if args.encode:
        encode_mode(ap, args, tok)
    # Prompt only when the run has nothing else to do: no files to write.
    run_tests(tests, tok, prompt=not (args.save or args.encode))


if __name__ == "__main__":
    main()
