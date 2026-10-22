# mini_bpe

A minimal byte-level BPE (byte-pair encoding) tokenizer in under 300 lines of
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
3. **Encode / decode** — apply the merges in learned order to new text;
   decoding concatenates the bytes back. Tokenization is **lossless**: decoding
   reproduces the input exactly, including every space and newline.

It is for understanding, not production use. It's slow, it has no special
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
./mini_bpe.py [options] corpus [corpus ...]
```

There are two kinds of input:

- **Training input**: the corpus files. The merges are learned from these.
  Each file is treated as one document.
- **Test input**: text tokenized *after* training, to see how the learned
  tokenizer splits it. It is not used for training. `-t` strings are
  processed first, then `-f` lines. If neither is given, you're prompted to
  type lines.

| option | meaning |
|---|---|
| `-n N`, `--merges N` | maximum number of merges to learn (default 200) |
| `-t STRING`, `--text STRING` | after training, tokenize STRING; may be repeated |
| `-f FILE`, `--test-file FILE` | after training, tokenize each non-blank line of FILE separately |
| `--dedup` | skip documents whose SHA-256 checksum matches an earlier document |
| `-q`, `--quiet` | don't print each merge during training |
| `-h`, `--help` | show help |

Examples:

```
./mini_bpe.py sample.txt -t "The cat sat."     # train, tokenize one string
./mini_bpe.py sample.txt -f tests.txt -q       # train, tokenize each line of tests.txt
./mini_bpe.py sample.txt -n 1000 -q            # more merges, no merge log
./mini_bpe.py sample.txt news*.txt --dedup     # several documents, drop exact duplicates
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

## Data files

| file | purpose |
|---|---|
| `corpus.txt` | Small test corpus. |
| `sample.txt` | Small mixed corpus: simple English sentences with repeated words (cat, dog, sat), shared suffixes (lowest, newest, slowest), punctuation, contractions, numbers, and two short Python functions to show how code affects the vocabulary (indentation, `()`, `):`). |
| `news.txt` | A cookie-banner notice, standing in for web boilerplate. |
| `news_copy1.txt`, `news_copy2.txt` | Exact copies of `news.txt`. They have the same checksum, so `--dedup` removes them. |
| `news_edited.txt` | `news.txt` with one word changed ("improve" → "enhance"). Its checksum is completely different, so exact deduplication keeps it. |

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

## Performance

Training is roughly O(merges × unique chunks) in pure Python. A ~1 MB book
with 1000 merges takes a minute or two; use `-q` to skip the merge log.

## Further reading

- R. Sennrich, B. Haddow, A. Birch, "Neural Machine Translation of Rare Words
  with Subword Units" (2016), the paper that introduced BPE for neural
  language models.
- A. Karpathy, "Let's build the GPT Tokenizer" (video lecture), a longer
  from-scratch walkthrough, including how tokenization explains many odd LLM
  behaviors.
