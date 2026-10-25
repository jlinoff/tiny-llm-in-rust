# mini_attention

Self-attention, small enough to print every number, and one head trained
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

## Source

- `src/attention.rs`: one head (`Head::forward` keeps every step), the
  causal mask, the softmax, layer normalization, and the tests
- `src/matrix.rs`: a minimal matrix type and table printing
- `src/train.rs`: the training model, backpropagation, Adam, the gradient
  check, and the after-training explanation
- `src/rng.rs`: a seedable random number generator, with normal values
- `src/main.rs`: the command line

## Progress

As planned in PLAN.md ("Where each part is built"):

1. ~~Hand-set weights reproducing the worked example~~ (`example`)
2. ~~Random weights~~ (`random`)
3. ~~Weights trained on a task whose right attention pattern is known in
   advance, to watch W_q and W_k learn it~~ (`train`)
4. Several heads
5. One complete block: the FFN, residual connections and layer normalization
