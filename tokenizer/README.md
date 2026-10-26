# mini_bpe

A minimal byte-level BPE (byte-pair encoding) tokenizer in about 850 lines of
plain Python, written to make one question concrete: **where does an LLM's list
of tokens come from?**

It follows the same stages as a production LLM tokenizer, at toy scale and with
everything visible:

1. **Pre-tokenize** — split text into chunks with a regex: words with their
   leading space, numbers, punctuation runs, whitespace. Merges never cross a
   chunk boundary.
2. **Train** — start from the 256 byte values, repeatedly merge the most
   frequent adjacent pair, and record each merge in order. The vocabulary is
   the 256 bytes plus one new token per merge (plus EOS; see
   [Special tokens](#special-tokens)).
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
  - *Inputs:* one or more text files (documents) and the tokenizer
  - *Output:* the text as token IDs, each document followed by EOS, written
    with `--encode FILE ... -o FILE` in a binary `.ids` file. See [Encoding a corpus](#encoding-a-corpus).

A model sizes its embedding table from the tokenizer's vocabulary size, V (one
row per token), and trains on the token IDs, which select those rows.

It is for understanding, not production use. It's pure Python, its only special
token is EOS, and its regex approximates GPT-style pre-tokenization, so its splits
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
- `--encode TEXTFILE ...`: after training or loading, encode one or more text
  files into token IDs, each file a document followed by EOS; requires `-o`
- `--split-on SEP`: split each corpus or `--encode` file into documents at
  lines consisting exactly of SEP, dropping those lines (see
  [Files holding many documents](#files-holding-many-documents)). Without
  it, each file is one document.
- `-o FILE`, `--output FILE`: where `--encode` writes the token IDs
- `-t STRING`, `--text STRING`: after training or loading, tokenize STRING;
  may be repeated
- `-f FILE`, `--test-file FILE`: after training or loading, tokenize each
  non-blank line of FILE separately
- `-q`, `--quiet`: don't print each merge during training
- `--reference-trainer`: train with the simple, slow trainer instead (same
  merges; see [Performance](#performance))
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
./mini_bpe.py --load tok.bpe --encode a.txt b.txt -o ab.ids      # two documents
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

A trained tokenizer is fully defined by three things, and the file holds all
of them:

1. **The pre-tokenizer pattern.** The merges only reproduce the same tokens if
   text is split into the same chunks first.
2. **The ordered merge list.** Order matters: encoding applies the merges in
   the order they were learned.
3. **The special tokens** (see [Special tokens](#special-tokens)).

The vocabulary itself isn't stored, because it can be rebuilt: IDs 0–255 are
the raw bytes, the merge on merge line *k* (counting from 0) creates token ID
256 + *k*, and the special tokens take the IDs after the merges, in order.
Example (`./mini_bpe.py sample.txt news.txt -q --save tok.bpe`):

```
# mini_bpe tokenizer v1
# vocab_size: 365  (256 bytes + 108 merges + 1 special)
# trained on: sample.txt  sha256 8e5a595442e28fa0e4ee1a7d…
# trained on: news.txt  sha256 cf7a91850aa826969c1fb760…
# Merge lines: '<left_id> <right_id>'; the merge on merge line k (from 0)
# creates token id 256 + k. Special lines: 'special <name>', IDs after the merges.
pattern '(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+
32 116  # 256: [␣] + [t] -> [␣t]
104 101  # 257: [h] + [e] -> [he]
…
special <|endoftext|>  # 364
```

Lines starting with `#` and text after `#` on a data line are comments for
human readers: the vocabulary size, the training documents with their
checksums (provenance), and each merge or special token's ID in readable form.
Only the `pattern` line, the pairs of IDs and the `special` lines are used when
loading. If the saved pattern differs
from the script's built-in one, the file's pattern is used.

**Connection to the next phase.** The `vocab_size` line is V, the number the
model needs: its embedding table is allocated as V × d (d is a model setting)
and filled with small random values, one row per token ID. Real tokenizers ship
the same way — a merge list and vocabulary file distributed with the model —
because the two are locked together: token ID 300 means `west` only under this
tokenizer. Train a different tokenizer and every row of the embedding table
would point at the wrong token.

## Special tokens

A special token is a token ID reserved by hand for structure, not text. The
tokenizer currently has one, **EOS** ("end of sequence"), named
`<|endoftext|>` as in GPT-2. Training always reserves it, with the first ID
after the merges, so V = 256 + merges + 1.

EOS marks the end of each document. In training it keeps unrelated documents
apart, so the model doesn't learn to continue a recipe with a news article;
in generation, the model *predicting* EOS is how it signals that it's
finished.

**Text never becomes a special token.** If a document contains the
characters `<|endoftext|>`, they're encoded as ordinary bytes; only the encoder
itself inserts the EOS ID, at document boundaries. Otherwise any text could
fake a document boundary, a tokenizer-level form of injection.

The file format holds a list of special tokens, so the conversation markers
needed for fine-tuning (see `../PLAN.md`) can be added later. A tokenizer saved
before EOS existed has no `special` line: it still works for `-t` and `-f`,
but `--encode` refuses it, so retrain it.

## Encoding a corpus

`--encode TEXTFILE ... -o FILE` turns one or more text files into token IDs and
writes them to a binary file, conventionally named `.ids`. This is step 2 of
the LLM flow: the model trains on these integers and never sees the text.

Each file is one document, and each document's tokens are followed by EOS,
including the last one's, so the model also learns how a document ends. With
the tokenizer from the example above (trained on `sample.txt` and `news.txt`):

```
./mini_bpe.py --load tok.bpe --encode sample.txt news.txt -o docs.ids
```
```
=== Encode: sample.txt news.txt -> docs.ids ===
  sample.txt                        682 bytes ->         305 tokens + EOS
  news.txt                          137 bytes ->          58 tokens + EOS
  2 document(s), 819 bytes of text -> 365 IDs including 2 EOS (ID 364); 2.26 bytes/token; lossless round-trip: True
  wrote 64-byte header + 365 IDs x 2 bytes; V = 365; tokenizer fingerprint 3e9260d88f88bc69…
```

Each document's IDs are decoded again and compared with its file; if any
differ, the run stops with an error and writes nothing. Because text never
produces the EOS ID, splitting the IDs at EOS recovers the documents exactly.

### Files holding many documents

Some corpora put many documents in one file, with a separator line between
them. TinyStories, for example, puts `<|endoftext|>` on a line of its own
between stories (21,989 of them in its validation file). Encoding such a file
as one document would be wrong twice over: the stories would run together,
and each separator would be encoded as about a dozen ordinary tokens of text
(the encoder never turns literal special-token text into the EOS ID; see
[Special tokens](#special-tokens)). `--split-on SEP` handles this:

```
./mini_bpe.py --load tok.bpe --encode TinyStories-valid.txt -o valid.ids --split-on '<|endoftext|>'
```

- Every line consisting exactly of SEP ends a document; the separator lines
  are dropped, not encoded.
- One newline at the end of each document is dropped too, so a story ends
  with its last word and then EOS, not with a line break.
- Empty documents (two separators in a row) are skipped.
- Each document is round-trip checked on its own, and followed by EOS.
- SEP anywhere else, such as inside a line of a story, is still ordinary
  text: you name the separator on the command line, so the injection rule in
  [Special tokens](#special-tokens) still holds.

Use the same option when training the tokenizer on such a file, so that no
merges are spent on pieces of the separator. Without `--split-on`, encoding
is unchanged (byte-identical output).

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
its pattern, a newline, one `left right` line per merge in order, then one
`special NAME` line per special token. Comments and provenance in the `.bpe`
file are left out, so two tokenizer files with the same pattern, merges and
special tokens have the same fingerprint. The
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

There are two trainers, and they produce exactly the same merges:

- **The reference trainer** (`--reference-trainer`, function
  `train_reference`) is the simple version: after every merge it recounts
  every pair in every chunk. It's the easiest to read, but each merge costs
  time in proportion to the whole corpus.
- **The default trainer** (function `train`, class `IncrementalPairs`) keeps
  the pair counts between merges. A merge only changes the chunks that
  contain the merged pair, so only those are re-examined: their old pairs are
  subtracted from the counts and their new pairs added. The most frequent pair
  is found with a heap (a priority queue). Ties between equally frequent pairs
  go to the pair that occurs first in the corpus, the same rule the reference
  trainer follows, which is what makes the merges identical.

They were checked against each other on English text, code, multilingual
text with emoji, and 40 random texts over two- to five-letter alphabets
(lots of ties and overlapping pairs such as `aaaa`): the merges matched every
time.

Measured on a 2-processor-core machine:

```
text                 merges   reference     default
README.md                1,002     2.4 s       0.10 s
text8, first 2 MB        1,000   102   s       3.2  s
text8, first 20 MB       4,000   ~27 min*     17   s
```

\* Extrapolated from the 2 MB run, not measured.

The 20 MB tokenizer averages 4.08 bytes per token on text8, about what
production tokenizers reach on English, using a peak of 450 MB of memory.
Use `-q` to skip the merge log. Encoding is fast (see
[Encoding a corpus](#encoding-a-corpus)).

## Further reading

- R. Sennrich, B. Haddow, A. Birch, "Neural Machine Translation of Rare Words
  with Subword Units" (2016), the paper that introduced BPE for neural
  language models.
- A. Karpathy, "Let's build the GPT Tokenizer" (video lecture), a longer
  from-scratch walkthrough, including how tokenization explains many odd LLM
  behaviors.
