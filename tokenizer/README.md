# mini_bpe

A minimal byte-level BPE (byte-pair encoding) tokenizer in about 650 lines of
plain Python, written to make one question concrete: **where does an LLM's list
of tokens come from?**

It follows the same stages as a production LLM tokenizer, at toy scale and with
everything visible:

1. **Pre-tokenize** — split text into chunks with a regex: words with their
   leading space, numbers, punctuation runs, whitespace. Merges never cross a
   chunk boundary.
2. **Train** — start from the 256 byte values, repeatedly merge the most
   frequent adjacent pair, and record each merge in order. The vocabulary is
   the 256 bytes plus one new token per merge.
3. **Encode / decode** — apply the merges to new text, giving token IDs;
   decoding concatenates the tokens' bytes back. Tokenization is **lossless**:
   decoding reproduces the input exactly, including every space and newline.

In the plan for this project ([`../PLAN.md`](../PLAN.md)), this script performs
two steps of the LLM flow:

- **Step 1, tokenize**
  - *Input:* training text, often a representative sample of the corpus
  - *Output:* the trained tokenizer, written with `--save`: the pre-tokenizer
    pattern plus the ordered merges. See
    [Saving and loading the tokenizer](#saving-and-loading-the-tokenizer).
- **Step 2, encode**
  - *Inputs:* a text file and the tokenizer
  - *Output:* the text as token IDs, written with `--encode FILE -o FILE` in a
    binary `.ids` file. See [Encoding a corpus](#encoding-a-corpus).

A model sizes its embedding table from the tokenizer's vocabulary size, V (one
row per token), and trains on the token IDs, which select those rows.

It is for understanding, not production use. Training is slow, it has no special
tokens, and its regex approximates GPT-style pre-tokenization, so its splits
won't exactly match any real model's.

## Requirements

Python 3.8 or later. Standard library only; nothing to install.

## Quick start

```
chmod +x mini_bpe.py
./mini_bpe.py sample.txt
```

This trains on `sample.txt`, prints each merge as it is learned, then prompts
you to type sentences and shows how the learned tokenizer splits them. An empty
line or Ctrl-D quits.

## Usage

```
./mini_bpe.py [options] corpus [corpus ...]     # train a tokenizer
./mini_bpe.py --load FILE [options]             # use a saved tokenizer
```

There are two kinds of input:

- **Training input**: the corpus files. The merges are learned from these.
  Each file is treated as one document.
- **Test input**: text tokenized *after* training (or loading), to see how the
  tokenizer splits it. It is never used for training. `-t` strings are
  processed first, then `-f` lines. If neither is given and the run writes no
  files (no `--save` or `--encode`), you're prompted to type lines.

Options:

- `-n N`, `--merges N`: maximum number of merges to learn (default 200)
- `--dedup`: skip documents whose SHA-256 checksum (a standard cryptographic
  hash: identical files always give the same 64-hex-digit value, different
  files practically never do) matches an earlier document
- `--save FILE`: after training, write the tokenizer to FILE
- `--load FILE`: load a tokenizer saved with `--save` instead of training. It
  can't be combined with corpus files, `-n`, `--dedup`, or `--save`.
- `--encode TEXTFILE`: after training or loading, encode TEXTFILE into token
  IDs; requires `-o`
- `-o FILE`, `--output FILE`: where `--encode` writes the token IDs
- `-t STRING`, `--text STRING`: after training or loading, tokenize STRING;
  may be repeated
- `-f FILE`, `--test-file FILE`: after training or loading, tokenize each
  non-blank line of FILE separately
- `-q`, `--quiet`: don't print each merge during training
- `-h`, `--help`: show help

**Files are checked before any work starts.** The output files (`--save`,
`-o`) must be writable and the `--encode` file readable; otherwise the run
stops at once, instead of after a long training run. The check doesn't modify
an existing output file.

Examples:

```
./mini_bpe.py sample.txt -t "The cat sat."     # train, tokenize one string
./mini_bpe.py sample.txt -f tests.txt -q       # train, tokenize each line of tests.txt
./mini_bpe.py sample.txt -n 1000 -q            # more merges, no merge log
./mini_bpe.py sample.txt news*.txt --dedup     # several documents, drop exact duplicates
./mini_bpe.py sample.txt -q --save tok.bpe     # train and save the tokenizer
./mini_bpe.py --load tok.bpe -f tests.txt      # reuse it later, no training
./mini_bpe.py --load tok.bpe --encode corpus.txt -o corpus.ids   # encode a corpus
```

## Reading the output

**Documents.** One line per corpus file, with its size, the first 16 hex
digits of its SHA-256 checksum, and its status: `kept`, or `DUPLICATE of …`
(counted again, or skipped with `--dedup`).

**Corpus.** Total bytes, number of pre-tokenized chunks, and how many are
unique, followed by the first 25 chunks.

**Merge log.**

```
merge   3:         ␣t + he         -> ␣the           (count 11)
```

At merge 3, the pair `␣t` followed by `he` was the most frequent adjacent pair
in the corpus, occurring 11 times, so it became the new token `␣the`. The count
is a raw occurrence count, recomputed after every merge because each merge
changes the text. The winning count never increases from one merge to the next.
Training stops early when no pair occurs more than once, which is why a small
corpus learns fewer merges than `-n` allows.

**Summary.** The number of merges learned, the vocabulary size (256 + merges),
and **bytes per token**: how well the tokenizer compresses the corpus. Higher
means fewer, longer tokens.

**Encoding.** For each test string: the tokens, their integer IDs, bytes per
token, and a check that decoding reproduces the input exactly.

### Token display

Tokens are shown in brackets, `[The][␣cat][.]`, with these markers:

| shown | meaning |
|---|---|
| `␣` | a space (U+2423, "visible space") |
| `\n`, `\t` | newline, tab |
| `\\` | a literal backslash |
| `\u2423` | a literal `␣` character in the text |
| `\xNN` | a single byte that is part of a multi-byte UTF-8 character, e.g. `æ` → `[\xc3][\xa6]` |

Every displayed form maps back to exactly one byte sequence.

## Saving and loading the tokenizer

`--save FILE` writes the trained tokenizer to a plain-text file, and
`--load FILE` reads it back so you can tokenize without retraining. Loading
gives exactly the same tokens and IDs as the original training run.

A trained tokenizer is fully defined by two things, and the file holds both:

1. **The pre-tokenizer pattern.** The merges only reproduce the same tokens if
   text is split into the same chunks first.
2. **The ordered merge list.** Order matters: encoding applies the merges in
   the order they were learned.

The vocabulary itself isn't stored, because it can be rebuilt: IDs 0–255 are
the raw bytes, and the merge on data line *k* (counting from 0) creates token ID
256 + *k*. Example (`./mini_bpe.py sample.txt news.txt -q --save tok.bpe`):

```
# mini_bpe tokenizer v1
# vocab_size: 364  (256 bytes + 108 merges)
# trained on: sample.txt  sha256 8e5a595442e28fa0e4ee1a7d…
# trained on: news.txt  sha256 cf7a91850aa826969c1fb760…
# Data lines: one merge per line, '<left_id> <right_id>'.
# The merge on data line k (from 0) creates token id 256 + k.
pattern '(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+
32 116  # 256: [␣] + [t] -> [␣t]
104 101  # 257: [h] + [e] -> [he]
…
```

Lines starting with `#` and text after `#` on a merge line are comments for
human readers: the vocabulary size, the training documents with their
checksums (provenance), and each merge in readable form. Only the `pattern`
line and the pairs of IDs are used when loading. If the saved pattern differs
from the script's built-in one, the file's pattern is used.

**Connection to the next phase.** The `vocab_size` line is V, the number the
model needs: its embedding table is allocated as V × d (d is a model setting)
and filled with small random values, one row per token ID. Real tokenizers ship
the same way — a merge list and vocabulary file distributed with the model —
because the two are locked together: token ID 300 means `west` only under this
tokenizer. Train a different tokenizer and every row of the embedding table
would point at the wrong token.

## Encoding a corpus

`--encode TEXTFILE -o FILE` turns a text file into token IDs and writes them to
a binary file, conventionally named `.ids`. This is step 2 of the LLM flow: the
model trains on these integers and never sees the text. With the tokenizer
from the example above (trained on `sample.txt` and `news.txt`):

```
./mini_bpe.py --load tok.bpe --encode sample.txt -o sample.ids
```
```
=== Encode: sample.txt -> sample.ids ===
  682 bytes of text -> 305 tokens (2.24 bytes/token); lossless round-trip: True
  wrote 64-byte header + 305 IDs x 2 bytes; V = 364; tokenizer fingerprint e72abeff2f40696c…
```

After encoding, the IDs are decoded again and compared with the input; if they
differ the run stops with an error and writes nothing.

**The `.ids` format.** A 64-byte header, then the IDs. All integers are
unsigned and little-endian:

```
offset  size  field
     0     8  magic: the bytes "MBPEIDS" followed by a zero byte
     8     4  format version (1)
    12     4  bytes per ID: 2 if V <= 65,536, otherwise 4
    16     4  V, the tokenizer's vocabulary size
    20     4  reserved (0)
    24     8  number of IDs
    32    32  tokenizer fingerprint
    64     -  the IDs
```

**The tokenizer fingerprint** is a SHA-256 hash of what defines the tokenizer:
its pattern, a newline, and then one `left right` line per merge, in order.
Comments and provenance in the `.bpe` file are left out, so two tokenizer
files with the same pattern and merges have the same fingerprint. The
fingerprint lets a later stage check that an `.ids` file was made with the
tokenizer it's about to use, because IDs are meaningless under any other.

**Speed.** Encoding repeatedly applies the earliest-learned merge present in
each chunk, which gives exactly the same tokens as applying all the merges in
order but costs time only for the merges that actually apply, and each distinct
chunk is encoded once and remembered. Encoding all of text8 (100 MB) with a
1,000-merge tokenizer took 25 seconds, round-trip check included.

## Losslessness, and a bug it caught

The round-trip check found a bug that had been present since the first
version: the underscore `_` was silently dropped. The pre-tokenizer only keeps
characters that one of its alternatives matches, and in Python's `re`, `\w`
includes `_`, so neither the letters class `[^\W\d_]` nor the punctuation class
`[^\s\w]` matched it. (GPT-2's original pattern uses Unicode classes, in
which `_` counts as punctuation.) The tell-tale sign was a run with no merges
reporting 681 tokens for 682 bytes, which a byte-level tokenizer can't do.

The pattern's punctuation class is now `(?:[^\s\w]|_)`, and every one of the
1,112,064 Unicode characters has been checked to pass through pre-tokenization
intact. A tokenizer saved by an older version still carries the old pattern;
encoding text with underscores using it fails the round-trip check, so retrain
it.

## Data files

- `corpus.txt`: a small test corpus.
- `sample.txt`: a small mixed corpus: simple English sentences with repeated
  words (cat, dog, sat), shared suffixes (lowest, newest, slowest),
  punctuation, contractions, numbers, and two short Python functions to show
  how code affects the vocabulary (indentation, `()`, `):`, and underscores).
- `news.txt`: a cookie-banner notice, standing in for web boilerplate.
- `news_copy1.txt`, `news_copy2.txt`: exact copies of `news.txt`. They have
  the same checksum, so `--dedup` removes them.
- `news_edited.txt`: `news.txt` with one word changed ("improve" →
  "enhance"). Its checksum is completely different, so exact deduplication
  keeps it.

## Experiments to try

**Vocabulary size.** Run with `-n 1`, `10`, `50`, `100`, and so on, and watch
bytes per token rise quickly and then level off. Early merges are letter pairs,
middle merges are common words and suffixes, and late merges on a small corpus
memorize individual words (overfitting).

```
for n in 0 10 50 100 200 500; do
  echo -n "n=$n: "; ./mini_bpe.py sample.txt -n $n -q -t x | grep 'Corpus encodes'
done
```

**Unseen and rare words.** Tokenize words that don't appear in the corpus.
`slowest` reuses `west` learned from *lowest*/*newest*; `Thermodynamics` falls
back to single letters; `hwæt` falls back to raw bytes. None of them fails.

**Duplicates.** Compare the merge order with and without `--dedup`:

```
./mini_bpe.py sample.txt news*.txt -q
./mini_bpe.py sample.txt news*.txt -q --dedup
```

Counted duplicates push the banner's words (`␣website`, `␣cookies`) ahead of
ordinary words like `␣dog` and `␣cat`. That's the distortion deduplication
prevents. `news_edited.txt` survives `--dedup` because exact hashing only
catches identical copies; real pipelines add near-duplicate detection
(typically MinHash with locality-sensitive hashing) for that.

**Corpus composition.** Remove the Python functions from a copy of `sample.txt`
and retrain: the code tokens disappear and the merges go to English. Or train
on a larger text, such as a Project Gutenberg book, with `-n 1000 -q`, and
compare the bytes per token for the same `-f` test file. The script can also
train on its own source as a code corpus: `./mini_bpe.py mini_bpe.py -n 300`.

**Tokenizers are not interchangeable.** Save two tokenizers trained on
different corpora and tokenize the same text with each:

```
./mini_bpe.py sample.txt -q --save a.bpe -t x > /dev/null
./mini_bpe.py news.txt -q --save b.bpe -t x > /dev/null
./mini_bpe.py --load a.bpe -t "The website uses cookies."
./mini_bpe.py --load b.bpe -t "The website uses cookies."
```

The same text gets different tokens *and* different IDs. A model trained with
one tokenizer can't use the other: its embedding rows are indexed by the first
tokenizer's IDs.

## Performance

Training is the slow part: roughly O(merges × unique chunks) in pure Python,
because it recounts every pair after every merge. Learning 1,000 merges from
2 MB of text took about 1¾ minutes; a ~1 MB book with 1,000 merges takes a
minute or two. Use `-q` to skip the merge log. Encoding is fast (see
[Encoding a corpus](#encoding-a-corpus)).

## Further reading

- R. Sennrich, B. Haddow, A. Birch, "Neural Machine Translation of Rare Words
  with Subword Units" (2016), the paper that introduced BPE for neural
  language models.
- A. Karpathy, "Let's build the GPT Tokenizer" (video lecture), a longer
  from-scratch walkthrough, including how tokenization explains many odd LLM
  behaviors.
