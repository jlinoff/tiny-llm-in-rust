# mini_attention

Self-attention, small enough to print every number, with heads and then a
complete transformer block trained while you watch. This is the learning step
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

**Reproducibility.** On one machine, every run is deterministic: the same
seed gives the same numbers, whatever order the parallel comparisons run in.
Across machines, results can differ slightly: `exp` and `tanh` come from
each platform's maths library, and Apple's and Linux's may round the last
bit differently. Runs that converge firmly absorb that, and every figure
quoted below for `train`, `heads` and `block`'s main run matches between a
Mac and Linux. Fragile runs can amplify it: `block`'s runs without an FFN
give different per-seed figures on the two (see that section). Read the
seed comparisons for their spread, not their exact numbers.

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

A second check follows: the same gradients computed by the automatic
differentiation tape (see below), from forward code alone. They agree with
the hand-written ones to about 10⁻¹⁷ of the largest gradient, which is
rounding: the two add the same numbers in slightly different orders.

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

## One complete block (`block`)

Item 5 of the list in PLAN.md, "Where each part is built": the pre-norm
transformer block, with everything around attention that the earlier items
left out.

```
x      = E[token] + P[t]                    the residual stream starts
h      = x + W_o·[heads](LN₁(x))            attention sublayer
y      = h + W₂·GELU(W₁·LN₂(h) + b₁) + b₂   FFN sublayer
logits = LN_f(y)·U
```

- **Residual stream:** each sublayer *adds* its result to x instead of
  replacing it.
- **Layer norm** (LN₁, LN₂, LN_f), each with a learned scale and shift,
  applied to the copy entering each sublayer and before the output layer.
- **W_o**, which takes the heads' joined outputs back to width d for the
  stream. (In `heads`, U absorbed it; here it can't, because the result is
  added to x.)
- **The FFN:** d → d_ff → GELU → d, on each token separately.

It is the first model written as forward code only: the automatic
differentiation tape (below) works out every gradient. The output starts
with the parameter count, 13,536, worked out from PLAN.md's formula, and a
finite-difference check covering all 20 weight matrices.

The task: random sequences of digits 0–9; at each position from 3 on,
predict (one back + two back) mod 10. Attention has to bring the two
digits to the current position, and adding them mod 10 is a nonlinear
function of the pair, the kind of computation the FFN is for.

What it shows, with the defaults:

- **Training:** 10% (chance) at first, 55% by step 500, 100% by step 750.
- **Attention:** head 1 mostly looks one back and head 2 two back, though
  some rows mix the two; the predictions on the demo sequence are printed
  as "two back + one back → answer".
- **The residual stream:** the average length of each part: the embeddings
  (7.6), what attention added (21.9), what the FFN added (30.8).
- **Knockouts** (one sublayer's addition left out at test time, nothing
  retrained): full block 100%; without attention 9.4%; without the FFN
  12.7%; chance 10%. Both are needed: attention moves information between
  tokens, and the FFN computes with it.
- **Trained without an FFN**, from scratch, over seeds 1–6: roughly
  70–100%, with a mean in the low-to-mid 80s, against 100% on every seed
  with the FFN. The per-seed figures differ between machines (on Linux,
  72.4/99.7/81.5/85.2/100.0/76.2%; on a Mac, 71.8/97.3/79.4/85.5/100.0/
  77.0%), and changing the learning rate by 3 × 10⁻¹⁶ moves them by up to 12
  points (seed 5: 100% to 88%), while the full block's run doesn't change
  in any printed digit. The block without an FFN lands in a different
  partial solution depending on last-bit rounding; with the FFN, the
  solution is robust.

That last result is again not the clean "can't" I expected. Attention is
itself nonlinear: its weights depend on which tokens are where, so a head
can blend the two digits differently for different pairs. The final layer
norm adds a little more. That gets a block without an FFN partway, often
far, but not reliably. With an FFN, the block learns the task every time,
and the knockout shows it hands the arithmetic to the FFN.

Options (the comparison trains 2 × `--seeds` − 1 more models, in parallel):

- `--vocab N`: number of digits, 2–100 (default 10)
- `--len N`: sequence length, at least 3 (default 8)
- `-d N`, `--dim N`: vector width d (default 32)
- `--heads N`: number of heads, dividing d (default 2)
- `--d-ff N`: the FFN's inner width (default 128, 4 × d)
- `--steps N` (default 2000), `--lr X` (default 0.003), `--batch N`,
  `--seed N`: as for `train`
- `--seeds N`: seeds in the with/without-FFN comparison (default 6)

With 5 digits (`--vocab 5`), a block without an FFN learns the task too:
only 25 pairs, few enough for attention's own nonlinearity.

The default run takes about 30 seconds on two cores, less on more.

## The automatic differentiation tape (`src/tape.rs`)

Not a subcommand: the machinery that computes gradients from now on (open
decision 7 in PLAN.md). `train` and `heads` have hand-written backward
passes; from item 5 on (`block`), models are written as forward code only,
and the tape works out the gradients.

How it works. Every operation in the forward pass is recorded on the tape as
it runs: its result, and which earlier entries it used. `backward` then walks
the tape in reverse, starting from the loss (whose gradient with respect to
itself is 1). Each entry applies its own backward rule: given the gradient
with respect to its result, it adds its share to its inputs' gradients. That
is the chain rule, one operation at a time.

```
forward (recorded):  E, P ─► gather ─► add ─► matmul ─► … ─► softmax_cross_entropy ─► loss
backward (replayed):  ◄──────────────── each entry passes gradient to its inputs ◄── 1
```

The backward rules are still written by hand, but each is local and short,
for example:

- matmul, C = A·B: dA = dC·Bᵀ, dB = Aᵀ·dC
- causal softmax, by row: dA_ij = C_ij (dC_ij − Σ_k C_ik dC_ik)
- gather (an embedding lookup): each row of dC goes back to the row it came
  from, adding up when a row was used more than once

It records whole matrices, not single numbers, so the tape for one sequence
is a few dozen entries. Some operations are **fused**: softmax followed by
cross-entropy is one operation, whose backward rule (probabilities −
one-hot) is much simpler than chaining the two rules.

Layer normalization isn't an operation of its own: `layer_norm` builds it
from the primitives (row mean, subtract, square, square root, divide), so
the tape derives its gradient. A test checks that against the textbook
closed form, dx = (d − mean(d) − x̂·mean(d ⊙ x̂)) / σ with d = dy ⊙ γ.

How it's checked:

- every operation's backward rule against finite differences, on its own
- layer norm against the closed form, as above
- the whole model of `train` and `heads` against their hand-written
  backward passes: the tape must reproduce every weight's gradient to within
  10⁻¹² (in practice about 10⁻¹⁷), and the loss
- the checks themselves: breaking matmul's backward rule by 1%, or the
  square root's, makes the tests fail

The hand-written backward passes stay: they're the references the tape is
checked against.

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
- the tape: each operation's backward rule against finite differences;
  layer norm against its closed-form gradient; an entry used twice gets
  both contributions; and the tape reproduces the hand-written gradients of
  `train`'s model and of three heads
- `block`: the whole block's tape gradients against finite differences, in
  every weight matrix; the parameter count against PLAN.md's formula; the
  targets; and that it learns the task (over 95%) and fails without
  either sublayer

## Source

- `src/attention.rs`: one head (`Head::forward` keeps every step), the
  causal mask, the softmax, layer normalization, and the tests
- `src/matrix.rs`: a minimal matrix type and table printing
- `src/train.rs`: the training model (any number of heads and lags),
  backpropagation, Adam, the gradient check, and `train`'s explanation
- `src/heads.rs`: `heads`: the division-of-labor analysis and the
  one-head comparison
- `src/block.rs`: `block`: one complete transformer block on the tape, the
  knockouts and the with/without-FFN comparison
- `src/tape.rs`: the automatic differentiation tape and its tests
- `src/optim.rs`: the Adam update, shared by `train`, `heads` and `block`
- `src/rng.rs`: a seedable random number generator, with normal values
- `src/main.rs`: the command line

## Progress

As planned in PLAN.md ("Where each part is built"):

1. ~~Hand-set weights reproducing the worked example~~ (`example`)
2. ~~Random weights~~ (`random`)
3. ~~Weights trained on a task whose right attention pattern is known in
   advance, to watch W_q and W_k learn it~~ (`train`)
4. ~~Several heads~~ (`heads`)
5. ~~One complete block: the FFN, residual connections and layer
   normalization~~ (`block`)
6. Two blocks forming an induction head (`induction`)
