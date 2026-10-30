# mini_w2v

A minimal word2vec in Rust, written for understanding. word2vec (Mikolov et
al., 2013) learns one vector per word from a single idea: a word is
characterized by the words around it. Every word starts as a small random
vector; training repeatedly nudges the vectors so that a word's vector can
predict its neighbors. Words used in similar contexts end up with similar
vectors, and the result supports nearest-neighbor queries (*cat* → *dog*,
*pig*, *meow*) and, famously, analogies (*king* − *man* + *woman* ≈ *queen*).

In this project, word2vec is the historical side branch (see
[`../PLAN.md`](../PLAN.md)): it shows, in isolation, how an embedding table
learns meaning. That same mechanism gives an LLM's embedding layer its
structure. [How it relates to LLMs](#how-it-relates-to-llms) covers what
changed.

## Terms

Used throughout; each uses only terms defined before it.

- **vocabulary**, **V**: the set of distinct words the model learns vectors
  for, and its size. Here, every word occurring at least `--min-count` times.
- **d**: the length of each word's vector (`--dim`).
- **embedding table**: a V × d table with one row (one vector) per word.
  word2vec has two: the **input** table (the word vectors, saved at the end)
  and the **output** table (a second "context" vector per word, used only
  during training).
- **center word**, **context words**: the word being looked at, and the words
  within a few positions of it (the **window**).
- **skip-gram**: the training task that predicts each context word from the
  center word.
- **CBOW (continuous bag of words)**: the training task that predicts the
  center word from the average of its context words.
- **negative sampling**: scoring the correct word against a few randomly drawn
  wrong words ("negatives") instead of against all V words.
- **epoch**: one full pass over the corpus.
- **learning rate**: how large each weight adjustment is.
- **loss**: a number measuring how wrong the predictions are; training lowers
  it.
- **text8**: a standard benchmark corpus, the first 100 MB of English
  Wikipedia, cleaned to lowercase letters and spaces (17 million words).

## Requirements

- Rust 1.87 or later (`cargo`). The only dependency is `clap`, for the command
  line.
- For the exploration scripts: Python 3 with gensim (a widely used Python
  library that includes its own word2vec) and NumPy:
  `pip install -r scripts/requirements.txt`.

## Getting text8

```
cd ../data
curl -O http://mattmahoney.net/dc/text8.zip && unzip text8.zip && rm text8.zip
```

If that site is unreachable, gensim's mirror on GitHub has the same file:

```
curl -L -o text8.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/text8.gz
gunzip text8.gz
```

## Quick start

```
cargo build --release
./target/release/mini_w2v ../data/text8 -p king,cat,paris,computer,three
```

This reads text8, builds the vocabulary, creates the two random tables, prints
the nearest neighbors of the probe words while the vectors are still random,
then trains for 5 epochs, printing the losses and the probe words' neighbors
after each one. The vectors are saved to `vectors.txt`.

On text8 with the default settings, training took 8½ to 12½ minutes on a
machine with 2 processor cores (511 s and 759 s in two identical runs; that
machine's speed varied), and less with more cores. CBOW (`-m cbow`) is about
4½ times faster.

## Usage

```
mini_w2v [options] corpus [corpus ...]
```

Each corpus file is one document; a context window never crosses from one
document into the next.

- `-o FILE`, `--output FILE`: where to save the word vectors (default
  `vectors.txt`)
- `-m MODEL`, `--model MODEL`: `skipgram` (default) or `cbow`
- `-d N`, `--dim N`: vector length d (default 100)
- `-w N`, `--window N`: maximum context words on each side (default 5)
- `-k N`, `--negative N`: negative samples per prediction (default 5)
- `-e N`, `--epochs N`: passes over the corpus (default 5). `-e 0` saves the
  random initial vectors without training.
- `--min-count N`: ignore words occurring fewer than N times (default 5)
- `--sample T`: subsampling threshold for frequent words (default 0.001;
  0 turns subsampling off; see [How it works](#how-it-works))
- `--alpha R`: starting learning rate (default 0.025 for skip-gram, 0.05 for
  CBOW)
- `-t N`, `--threads N`: worker threads (default: all processor cores). With
  `-t 1`, a given `--seed` always gives identical results.
- `--seed N`: random seed (default 1)
- `-p WORDS`, `--probe WORDS`: comma-separated words whose nearest neighbors
  are shown after initialization and after every epoch
- `--top N`: neighbors shown per probe word (default 8)
- `-q`, `--quiet`: don't show the progress line
- `-h`, `--help`: show help

Examples:

```
mini_w2v ../data/text8 -p king,cat,paris          # train, watch three words
mini_w2v ../data/text8 --model cbow -o cbow.txt   # train with CBOW
mini_w2v ../data/text8 -e 0 -p king -o rand.txt   # random vectors, no training
mini_w2v ../data/text8 -t 1 --seed 7              # reproducible run
```

**Small corpora need `--sample 0`.** Subsampling is relative to the corpus: on
a few sentences it discards most of the words, leaving almost nothing to train
on. It also needs `--min-count 1`, so that words seen only a few times are
kept.

## Reading the output

**Corpus.** Words per file, the total, and the vocabulary: V words occurring at
least `--min-count` times, and the most frequent ten.

**Model.** The two V × d tables and how they start: the input table uniform in
[−0.5/d, 0.5/d), the output table all zero. Then the training settings, and
what fraction of words subsampling keeps.

**Probes.** Each probe word's nearest neighbors by cosine similarity (1 means
the vectors point the same way, 0 unrelated). After initialization they are
random: unrelated words with similarities around 0.35 to 0.45. That is not a
sign of meaning. Any one pair of random 100-dimensional vectors has a
similarity near 0, with a standard deviation of about 1/√d = 0.1, but the
nearest neighbor is the best of the other 71,289 words, and the largest of
that many such values lies about √(2 ln 71,289) ≈ 4.7 standard deviations
out (about 0.27 at d = 300). After training, the neighbors are meaningful:
with the default d = 100, close neighbors score about 0.7 to 0.8; with
d = 300, about 0.5 to 0.6, even though the neighbor lists are better.
Absolute similarities shrink as d grows, so compare rankings, not values,
between models of different sizes.

**Training.** One line per epoch with two losses:

```
  epoch 3/5: loss running 2.145, fixed 2.272; 12,507,033 words after subsampling, 154.7 s (108k words/s)
```

- **running**: the average loss over the epoch's predictions, measured as they
  happen, while the weights change.
- **fixed**: the average loss on the same 100,000 predictions (same words,
  same negatives), measured after the epoch with the weights frozen.

Only the fixed loss is comparable from epoch to epoch. See
[Why two losses](#why-two-losses).

Before any training, the loss is about 6 × ln 2 ≈ 4.159: the output table
starts at zero, so every score is 0, and each of the 1 + 5 words in a
prediction is a coin flip (loss ln 2 each).

## How it works

The source is short, one concept per file:

| file | contents |
|---|---|
| `src/corpus.rs` | split text into words, build the vocabulary, store the corpus as word IDs |
| `src/train.rs` | the two tables, negative sampling, subsampling, the skip-gram and CBOW updates |
| `src/matrix.rs` | the shared weight matrix, the random number generator, vector helpers |
| `src/query.rs` | nearest neighbors; saving vectors |
| `src/main.rs` | the command line and the printed report |

### The prediction and its update (negative sampling)

For skip-gram, each (center, context) pair is one prediction. Let h be the
center word's input vector and u the output vector of a word being scored.
The model's estimate that the word really is a context word is
σ(h · u), where σ(x) = 1 / (1 + e^(−x)) squashes a score into (0, 1).

Each prediction scores the true context word (label 1) and k randomly drawn
negatives (label 0). For each of those k + 1 words:

```
g       = alpha * (label - σ(h · u))      how wrong, scaled by the learning rate
grad_h += g * u                           collect the change for h
u      += g * h                           move u toward h (label 1) or away (label 0)
```

and finally h += grad_h. The loss being reduced is
−ln σ(h · u_true) − Σ ln σ(−h · u_negative).

Scoring k + 1 words instead of all V (71,290 for text8) is what makes training
feasible: a full softmax would cost V dot products per prediction instead of 6.

CBOW is the same update with h the *average* of the context words' input
vectors and the center word as the true target; every context word then
receives the whole of grad_h.

### The details that make it work

- **Negatives are drawn with probability ∝ count^0.75.** The power 0.75 (found
  by experiment in the original paper) flattens the word-frequency
  distribution: frequent words are still drawn most, but rare words more often
  than their raw share. Drawing uses a 10-million-slot table in which each word
  fills a share of slots equal to its probability.
- **Subsampling of frequent words.** Each occurrence of a word with corpus
  frequency f is kept with probability (√(f/t) + 1) · t/f, at most 1, where t
  is `--sample`. Words rarer than t are always kept; "the" is mostly dropped.
  Frequent words say little about their neighbors and would otherwise dominate
  the pairs. On text8, about 75% of word occurrences survive each epoch.
- **Dynamic window.** For each center word the window size is drawn uniformly
  from 1 to `--window`, so nearer words are used more often.
- **Learning-rate decay.** The rate falls linearly from `--alpha` to
  alpha/10,000 over the whole run.
- **Initialization.** Input vectors random and small; output vectors zero. The
  input side must be random to break symmetry: if every row started equal,
  every word would get identical updates and stay identical forever.
- **Hogwild parallelism.** All threads update the shared tables with no locks.
  Two threads occasionally update the same row at once and one update is lost;
  with tens of thousands of rows this is rare and harmless. The original C code
  uses plain memory writes; in Rust that would be undefined behavior, so each
  weight is an `AtomicU32` holding an f32's bits, read and written with
  `Relaxed` ordering, which compiles to ordinary loads and stores.

### Why two losses

An early version reported only the running loss, and on text8 it rose in the
later epochs while the neighbor lists kept improving. Ten epochs on the first
20 MB of text8 show both measurements:

```
epoch   running   fixed
  0        -      4.148
  1      2.373    2.597
  2      2.217    2.475
  3      2.192    2.404
  4      2.162    2.396
  5      2.162    2.332
  6      2.148    2.305
  7      2.162    2.258
  8      2.170    2.238
  9      2.181    2.226
 10      2.214    2.183
```

The fixed loss falls every epoch: the model keeps improving. The running loss
stops falling after epoch 6 and then rises.

The likely explanation is that text8 is Wikipedia articles in order, and the
running loss is measured *while the weights change*. With a high learning
rate, the model adapts to the article it is reading: a name that appears 20
times in one article becomes easy to predict after its first few appearances,
and the running loss gets the credit. As the learning rate falls toward zero,
that short-term adaptation fades, so the running loss rises even though the
vectors are better. Two details fit this: early on the running loss is *below*
the fixed loss (the adaptation helps), and by the last epoch, when the
learning rate is nearly zero, it is above.

The lesson carries over directly to LLMs: a loss measured during training is
not the same as a loss measured on fixed data with the weights frozen, and
only the second can be compared over time. LLM training reports a separate
evaluation loss on held-out text for this reason.

(The fixed loss starts at 4.148 rather than 6 × ln 2 ≈ 4.159 because a
negative sample that happens to equal the true word is skipped, so a few
predictions score fewer than six words.)

## Exploring the vectors

The vectors file uses the standard word2vec text format (a header line "V d",
then one line per word), so other tools read it too. `scripts/explore.py` has
three commands:

```
cd scripts
./explore.py neighbors ../vectors.txt king cat paris
./explore.py analogy   ../vectors.txt man king woman     # king - man + woman
./explore.py evaluate  ../vectors.txt                    # Google analogy test set
```

**Analogies, honestly.** `analogy` shows the answers two ways. The standard
convention, used in the original papers and by gensim, excludes the three input
words from the candidates. Without that exclusion, the nearest vector to
king − man + woman is often just *king*, because the arithmetic moves the vector
only part of the way. Nissim, van Noord and van der Goot (2020) argue the
convention makes the analogies look cleaner than they are; `evaluate` reports
the accuracy both ways so you can see the size of the effect.

**A reference implementation.** `scripts/train_gensim.py` trains gensim's
word2vec on the same corpus with the same settings, for comparison:

```
./train_gensim.py ../../data/text8 -o ../gensim.txt
./explore.py evaluate ../gensim.txt
```

## Results on text8

Measured with the default settings (d = 100, window 5, 5 negatives, 5
epochs) on a machine with 2 processor cores.

**Analogy accuracy** on Google's test set (`explore.py evaluate`), using the
30,000 most frequent words, as is standard:

```
                      training   excluding inputs           including inputs
                      time       total  semantic syntactic  total
mini_w2v skip-gram    759 s      38.0%   29.9%    42.1%      9.3%
gensim skip-gram      419 s      37.4%   30.8%    40.6%      9.6%
mini_w2v CBOW         168 s      37.3%   27.4%    42.3%     10.2%
untrained (-e 0)        -         0.0%    0.0%     0.0%      0.0%
```

- **mini_w2v matches gensim.** The differences are within run-to-run
  variation. (gensim's evaluator, run on the same files, gives the same
  38.0% and 37.4%.) "Semantic" questions are capitals, currencies and family
  relations; "syntactic" ones are plurals, tenses and comparatives.
- **CBOW is 4½ times faster,** slightly weaker on semantic analogies and
  slightly stronger on syntactic ones, as the original paper found.
- **Excluding the input words matters enormously:** without that convention,
  accuracy falls from about 38% to about 9%. Most "correct" answers are
  second-nearest after one of the question's own words.
- **gensim is faster,** about 1.8 times here. Likely reasons: its inner loops
  call optimized vector routines (BLAS) and look up σ in a precomputed table,
  while mini_w2v's atomic weights keep the compiler from vectorizing the loads
  and stores. mini_w2v keeps the loops plain on purpose.

**Nearest neighbors** after training (mini_w2v skip-gram, cosine similarity):

```
      king: prince 0.81, canute 0.76, pretender 0.75, haakon 0.75, valdemar 0.74
       cat: felis 0.69, guppy 0.69, prionailurus 0.68, ass 0.68, meow 0.67, dog 0.67
     paris: universelle 0.75, brussels 0.74, montparnasse 0.74, seine 0.72
  computer: computers 0.81, hardware 0.79, computing 0.78, networked 0.74
     three: four 0.98, five 0.98, six 0.96, two 0.96, seven 0.94
```

(*Felis* and *Prionailurus* are cat genera; Canute, Haakon and Valdemar were
Scandinavian kings.)

**The famous analogy: a negative result.** On text8, king − man + woman does
*not* reliably give *queen*. In this run its top answers were *dowager,
sigismund, alexandra, daughter, isabella*, with *queen* 11th. Individual
analogies are a coin toss at 38% accuracy:

```
                                   rank of the expected answer
                                   mini_w2v    gensim
man : king :: woman : queen           11          1
france : paris :: italy : rome        11         12
france : paris :: germany : berlin     1          1
man : he :: woman : she                1          1
big : bigger :: small : smaller        1          2
walk : walked :: swim : swam           4         61
```

The two implementations score the same overall but succeed on different
questions, so which particular analogies come out first depends on the random
details of the run. The famous examples came
from vectors trained on about 100 billion words of news text, roughly 6,000
times the size of text8.

## Experiments to try

- **Watch meaning emerge.** Run with `-p` and compare the neighbors after
  initialization, epoch 1 and epoch 5.
- **Random vectors.** `-e 0 -o rand.txt`, then `explore.py evaluate rand.txt`:
  the accuracy of untrained vectors is essentially zero.
- **Skip-gram versus CBOW.** CBOW is several times faster; skip-gram is
  usually better on rare words and on the semantic analogies. Compare with
  `evaluate`.
- **Vector size.** Try `-d 50` and `-d 300`.
- **Negatives.** `-k 0` scores only the true word, so nothing pushes vectors
  apart. After one epoch on 20 MB of text8, every vector points the same way:
  *king*'s nearest neighbors are *by, on, s, in, of*, all with similarity 1.00.
  `-k 15` is slower and sometimes better.
- **Subsampling off.** `--sample 0`: slower, and frequent words crowd the
  neighbor lists.

## Tests

`cargo test --release` (or `make test-rust`) checks: splitting text into
words; the vocabulary's order (most frequent first, ties alphabetical) and
the `--min-count` cut; sentences never longer than 1,000 words nor crossing
a file; the random generator, the shared matrix and the loss functions;
subsampling (rare words always kept); negative sampling following
count^0.75; cosine nearest neighbors; and, for skip-gram and CBOW, that
training on two topics that never share a window lowers the loss and gives
every word only neighbors from its own topic.

## How it relates to LLMs

- **The same table.** word2vec's input table is the same kind of object as an
  LLM's embedding table: V × d, random at first, given its structure by
  training. Its output table corresponds to an LLM's output layer (the
  "unembedding"), which turns a vector into a score for every token.
- **What sits between them.** word2vec has nothing between the two tables. An
  LLM puts a stack of transformer layers there.
- **Static versus contextual.** A word2vec vector is fixed: *bank* has one
  vector whether the text is about rivers or loans. In an LLM the embedding is
  only the starting point; the attention layers rebuild each token's vector
  from its context.
- **Separate versus joint training.** word2vec trains its vectors with their
  own objective and then stops. An LLM's embedding table is trained together
  with all its other weights, on next-token prediction.
- **Words versus tokens.** word2vec's vocabulary is whole words counted from
  the corpus; an LLM's comes from a separately trained tokenizer (see
  `../tokenizer/`).

## References

- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in
  Vector Space." [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases
  and their Compositionality." NeurIPS.
  [arXiv:1310.4546](https://arxiv.org/abs/1310.4546). Negative sampling and
  subsampling, as implemented here.
- Goldberg, Y., & Levy, O. (2014). "word2vec Explained."
  [arXiv:1402.3722](https://arxiv.org/abs/1402.3722). A clear derivation of
  the objective.
- Niu, F., et al. (2011). "Hogwild!: A Lock-Free Approach to Parallelizing
  Stochastic Gradient Descent." NeurIPS.
  [arXiv:1106.5730](https://arxiv.org/abs/1106.5730)
- Nissim, M., van Noord, R., & van der Goot, R. (2020). "Fair Is Better than
  Sensational: Man Is to Doctor as Woman Is to Doctor." *Computational
  Linguistics* 46(2). [arXiv:1905.09866](https://arxiv.org/abs/1905.09866)

The full bibliography is in [`../PLAN.md`](../PLAN.md#bibliography).
