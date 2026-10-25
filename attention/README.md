# mini_attention

Self-attention, small enough to print every number, with heads trained
while you watch. This is the learning step
before `../transformer/`: it computes exactly what PLAN.md describes in "The
transformer block", section "How attention works: queries, keys and values",
and prints each intermediate table so you can follow it.

Terms (query, key, value, causal mask, softmax, head) are defined in
[`../PLAN.md`](../PLAN.md#terms).

## Build

```
cargo build --release
cargo test --release
```

Rust 1.85 or later; the only dependency is `clap`, for the command line.
All arithmetic uses 64-bit floats: at these sizes speed doesn't matter, and
the extra precision keeps the checks clean.

## What one head computes (`example`, `random`)

These are items 1 and 2 of the list in PLAN.md, "Where each part is
built".

### `mini_attention example`

The worked example from PLAN.md: 3 tokens ("the cat sat"), d = 4, one head of
size 2, with hand-picked integer weights. It prints X, the three weight
matrices, and then every step: Q, K and V; the scores Q · Kᵀ / √2; the
causal mask; the softmax weights; and the outputs. The numbers match the
plan exactly; for example, the row for "sat":

```
weights
          the    cat    sat
  the    1.00      ·      ·
  cat    0.50   0.50      ·
  sat    0.14   0.58   0.28
  (· = masked: a later token, weight exactly 0)

sat:  0.14 × [1 0]  +  0.58 × [0 1]  +  0.28 × [1 1]  =  [0.42  0.86]
```

In the weights tables, `·` marks a position the causal mask removed: its
score was −∞, so its weight is exactly 0 (e^−∞ = 0). A printed `0.000` is
different: an allowed token whose weight is real but too small to show.

### `mini_attention random`

An untrained head: random embeddings (layer-normalized, as a vector entering
block 1 would be) and random weights drawn from normal(0, std²). Options:

- `--text "..."`: the words to attend over (default "the cat sat on the mat")
- `-d N`, `--dim N`: vector width d (default 16)
- `--head-size N`: head size (default d)
- `--std X`: standard deviation of the weights (default 0.02, as in GPT-2)
- `--seed N`: random seed (default 1)
- `-v`, `--verbose`: also print Q, K, V and the outputs

What to look for:

- **Small weights give near-uniform attention.** With the default 0.02,
  every score is near 0 (under 0.01 with the defaults), so each row is close
  to uniform: 1, ½, ⅓, ¼, … An untrained head simply averages all earlier
  tokens; training teaches it to be selective.
- **Large weights give arbitrary attention.** With `--std 1`, scores reach
  ±20 or more, the softmax becomes sharp, and most rows put nearly all their
  weight on one token chosen by chance. This is why models start with small
  weights.
- **Repeated words are indistinguishable.** In "the cat sat on the mat", both
  "the" rows are identical: with no position information, the same word has
  the same vector wherever it appears. This is why the model is given each
  token's position (PLAN.md, "Position information").

## Watching a head learn (`train`)

Item 3 of the list in PLAN.md, "Where each part is built".

### `mini_attention train`

Trains one head on a task whose right attention pattern is known in advance:
in random sequences of letters, predict at every position the token `lag`
positions back (default 2). The only way to do that is to attend to position
t − 2, so you can watch each row of the attention weights move from uniform
to a single 1.00 two places left of the diagonal.

The model is as small as it can be while still having to use attention:

```
x_t     = E[token_t] + P[t]          token embedding + position embedding
Q, K, V = X·W_q, X·W_k, X·W_v        one head
O       = softmax(mask(Q·Kᵀ/√h))·V   the attention output
logits  = O·U                        one score per possible token
loss    = −ln p(correct token), averaged over positions t ≥ lag
```

There's no residual connection, FFN or layer normalization yet. With a
residual connection the model could pass information around the head; here
every prediction has to come through attention. Those parts come in the
full block.

Training is ordinary backpropagation and Adam, written out by hand in
`src/train.rs` (about 60 lines for the backward pass), one step per formula
of the forward pass. Before training, the program checks the hand-written
gradients against finite differences: it nudges 30 individual weights by
±10⁻⁶, measures how the loss changes, and compares. The worst relative error
is about 10⁻⁷; a mistake in the backward pass would show up as 10⁻¹ or
worse.

Options:

- `--lag N`: how far back the token to predict is (default 2)
- `--steps N`: training steps (default 400)
- `--lr X`: Adam learning rate (default 0.001)
- `--batch N`: sequences per step (default 16)
- `--vocab N`: number of distinct tokens, 2–26, shown as letters (default 8)
- `--len N`: sequence length (default 8)
- `-d N`, `--dim N`: vector width d (default 16)
- `--head-size N`: head size (default d)
- `--seed N`: random seed (default 1)
- `--show LIST`: steps at which to print the attention (default
  `0,50,100,150,200,300,400`)

What to look for, with the defaults:

- **Step 0**: near-uniform attention (1, ½, ⅓, …), as in `random`, and a loss
  of about 2.1, which is chance (ln 8 = 2.079: a uniform guess among 8 letters).
- **Steps 50–200**: the band at t − 2 appears, unevenly. Some rows find it
  before others, and accuracy climbs from 18% to over 90%.
- **Step 300 onward**: nearly every row with a target puts 0.9 or more on
  t − 2; loss is under 0.1 and accuracy is 100%.
- **Rows 1 and 2 never settle.** Nothing lies two positions before them, so
  they have no target, contribute nothing to the loss, and get no gradient.
  Their weights are whatever the training of the other rows leaves behind
  (row 2 ends at 0.74/0.26). Real models have the same situation at the start
  of every context, without the excuse.

After training it explains how the head solved the task:

- **Where to look (QK)**: the attention computed from the position embeddings
  alone, (P·W_q)(P·W_k)ᵀ/√h, masked and softmaxed, already shows the band.
  Score spread from positions is about 13.5, against about 1.4 from token
  identities: the head looks by position and ignores which letter is where.
- **What to copy (OV)**: sending each token's embedding through W_v and then
  U gives the highest score to that same token, for all 8 tokens. The value
  and output weights learned to pass the token along unchanged.

So W_q and W_k learned where to look and W_v and U what to pass on, the QK
and OV circuits of PLAN.md, separating without being told to. Try
`--lag 3`, or `--lag 1` (which is the "previous token" head real models
learn early in training), or a smaller `--head-size`.

## Several heads (`heads`)

Item 4 of the list in PLAN.md, "Where each part is built".

### `mini_attention heads`

The task: at every position, predict the tokens one, two *and* three back,
together, as one answer among V³ = 64 (with 4 letters, a–d). The model is
`train`'s, with several heads side by side: each head has its own W_q, W_k
and W_v, the heads' outputs are placed side by side, and U maps them to
the 64 scores.

A real attention sublayer has one more matrix here, W_o, which takes the
heads' joined outputs back to width d for the residual stream. Here U
follows directly, and two matrices in a row are equivalent to one
(W_o·U), so U does both jobs. W_o appears with the residual stream in
item 5.

It runs in three parts:

1. **Three heads, trained in detail.** A gradient check (now covering every
   head), then snapshots of all three heads' attention. At step 0 they're
   uniform. At step 100 they're still mixed, each row leaning on whatever
   it found first. By step 250 each head has taken one lag at 0.9–1.0, and
   accuracy is 100%.
2. **How they divided the work.** For each head: which lag its attention
   favors, computed from the position embeddings alone (QK), and which part
   of the answer its values push toward (OV), found by sending each letter's
   embedding through that head's W_v and its rows of U. With seed 1, head 1
   takes three back, head 2 one back, head 3 two back. The order is
   arbitrary and changes with the seed; nothing assigns lags to heads.
   For every head, where it looks and what it passes on agree.
3. **One head of the same total size** (18, against 3 × 6), on the same
   task with the same training, first in detail and then over 20 seeds.

**The result is not "one head can't do this."** I expected it to fail, and
it doesn't. It spreads each row's attention over the three positions
*unevenly*: averaged over 500 fresh sequences, 0.45, 0.33 and 0.20, so its
output, roughly 0.45·v(a) + 0.33·v(b) + 0.20·v(c), is a blend in which each
letter comes in at a different strength. That is enough for U to tell which
letter came from where. What one head lacks is ease. Over seeds 1–20 with
the defaults:

- one head of size 18: solved (≥ 99% accuracy) 15 of 20 seeds, reaching 99%
  after about 880 steps on average
- three heads of size 6: solved 20 of 20, after about 230 steps

So several heads aren't needed for a new capability here. They make the
solution easy to find and easy to read: each head does one lookup at full
strength, instead of one head keeping three in balance. The transformer
paper (Vaswani et al. 2017) gives the same reason: "With a single attention
head, averaging inhibits this." There's a converse finding too: after
training, many of a large model's heads can be removed with little loss
(Michel et al. 2019, in PLAN.md's bibliography). One reading, consistent with
what `heads` shows, is that several heads matter more for learning than for
the finished model.

With two lags (`--lags 1,2 --vocab 6 -d 16 --head-size 8 --steps 800`) the
gap is smaller: one head solves 17 of 20 seeds, two heads 19 of 20.

The failures are worth looking at. When a run gets stuck (seed 1's single
head with two lags, for example), one row has put all its weight on a single
position and lost the other token, so that row's predictions are right only
about 1 time in 6. Training doesn't recover: it's a local minimum. The
weak point is that each position has its own learned position vector, so
each row's pattern is learned somewhat on its own.

Options (the comparison trains 2 × `--seeds` models, in parallel):

- `--lags LIST`: the distances back to predict together (default `1,2,3`)
- `--heads N`: number of heads (default: one per lag)
- `--head-size N`: size of each head (default 6); the single head in the
  comparison has size heads × head-size
- `--vocab N`: number of distinct tokens, 2–26 (default 4). There are
  vocab^lags possible answers, at most 10,000.
- `--len N`, `-d N`, `--steps N` (default 1500), `--lr X` (default 0.003),
  `--batch N`, `--seed N`: as for `train`
- `--seeds N`: seeds in the comparison, starting from `--seed` (default 20)
- `--show LIST`: steps at which to print the attention (default `0,100,250`)

The default run takes a few seconds on a multi-core machine (40 training
runs of 1,500 steps; the matrix products for the 64 output scores dominate).

## Tests

`cargo test` checks:

- the worked example reproduces the plan's tables (exact for the integer
  Q, K and V; within a tolerance for the floating-point results, per the
  project's comparison convention)
- every row of weights sums to 1, and masked entries are exactly 0
- the softmax stays finite for very large scores (it subtracts each row's
  largest score first)
- layer normalization gives mean 0 and variance 1
- the hand-written backward pass matches finite differences (relative
  error under 10⁻⁵)
- training on the lag-2 task reaches over 98% accuracy, with every row that
  has a target putting over 0.8 of its weight two positions back
- targets for several lags combine the tokens correctly (base V)
- backpropagation with three heads matches finite differences
- three heads trained on lags 1, 2 and 3 reach over 99% accuracy, each
  favoring a different lag

## Source

- `src/attention.rs`: one head (`Head::forward` keeps every step), the
  causal mask, the softmax, layer normalization, and the tests
- `src/matrix.rs`: a minimal matrix type and table printing
- `src/train.rs`: the training model (any number of heads and lags),
  backpropagation, Adam, the gradient check, and `train`'s explanation
- `src/heads.rs`: `heads`: the division-of-labor analysis and the
  one-head comparison
- `src/rng.rs`: a seedable random number generator, with normal values
- `src/main.rs`: the command line

## Progress

As planned in PLAN.md ("Where each part is built"):

1. ~~Hand-set weights reproducing the worked example~~ (`example`)
2. ~~Random weights~~ (`random`)
3. ~~Weights trained on a task whose right attention pattern is known in
   advance, to watch W_q and W_k learn it~~ (`train`)
4. ~~Several heads~~ (`heads`)
5. One complete block: the FFN, residual connections and layer normalization
   (`block`)
6. Two blocks forming an induction head (`induction`)
