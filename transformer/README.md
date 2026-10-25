# transformer

The real model: a small GPT-style language model, and the LLM flow's steps 3
to 6 in PLAN.md (build, train, finetune, generate). So far: step 3 is done,
and step 4 (train) has its forward and backward passes; the training loop
comes next.

Terms (tensor, checkpoint, safetensors, layer norm, …) are defined in
[`../PLAN.md`](../PLAN.md#terms).

## Build

```
cargo build --release
cargo test --release
```

Rust 1.85 or later. Dependencies: `clap` (command line), `serde` with `toml`
(the config) and `serde_json` (the safetensors header), `sha2` (the
tokenizer fingerprint), and `rayon` (threads). The binary is
`target/release/transformer`.

Unlike attention/, which uses 64-bit floats so every number prints cleanly,
the model uses 32-bit floats (f32): half the memory and about twice the
speed, and plenty of precision for training (PLAN.md, "Conventions for all
code").

## `transformer build` (step 3)

```
transformer build --tokenizer tok.bpe --config config.toml -o model-init.safetensors
```

Creates the model with random weights and writes it as a checkpoint. It
reads V and the tokenizer's fingerprint from `tok.bpe`, and only the
`[model]` section of `config.toml`:

- `d_model`: the model width d
- `layers`: transformer blocks
- `heads`: attention heads per block (must divide `d_model`)
- `context`: the most tokens the model sees at once (T)
- `d_ff`: the FFN's inner width
- `seed`: for the random weights

The design is GPT-2's, the one PLAN.md's parameter formula describes:

- learned position embeddings
- pre-norm layer normalization with a scale (γ) and shift (β)
- a bias on every matrix
- GELU in the FFN
- the output layer tied to the token embedding table

So there is no separate `output` tensor.

The initial weights follow GPT-2 too:

- embeddings and matrices: normal(0, 0.02)
- the two matrices that write into the residual stream (`attn.w_o`,
  `ffn.w2`): normal(0, 0.02/√(2L)), so the stream doesn't grow with depth
- biases: 0
- layer norm's γ: 1, and β: 0

What `build` does, in order:

1. Checks that the inputs are readable and the output writable, and that the
   output isn't one of the inputs, before any work (the project's
   convention).
2. Creates the weights and checks their count against PLAN.md's formula,
   N = V·d + T·d + L·(4d² + 2·d·d_ff + 9d + d_ff) + 2d.
3. Writes the checkpoint, reads it back and compares it bit for bit.
4. Prints a **weights fingerprint**, a SHA-256 of the names, shapes and
   values. The same seed and shapes should give the same fingerprint on any
   machine. The random numbers use `ln` and `cos`, which can differ in the
   last bit between platforms, but rounding to f32 should absorb that.
   Comparing fingerprints checks it.

### The checkpoint

A [safetensors](../PLAN.md#open-decisions) file (open decision 3): an 8-byte
header length, a JSON header naming each tensor with its dtype, shape and
byte offsets, then the raw little-endian f32 numbers. Written and read by
hand in `src/safetensors.rs`. Tensor names follow the scheme in decision 3:
`token_embedding`, `position_embedding`, `blocks.i.ln1.gamma`,
`blocks.i.attn.w_q`, …, `final_ln.beta`, each matrix stored input × output.

The header's `__metadata__` holds:

- `format`, `kind` (`init` for build's output), `producer`
- `model`: the `[model]` section as TOML text. Later steps read the shapes
  from here, never from `config.toml`, so the two can't disagree.
- `vocab_size`, `parameters`, `architecture`, `init`
- `tokenizer_fingerprint` (the same SHA-256 that mini_bpe.py computes, and
  that `.ids` files carry), `tokenizer_file`, `tokenizer_sha256`
- `config_file`, `config_sha256`

The file is checked to work with Hugging Face's own `safetensors` library
(version 0.8.0) both ways. That library reads build's output correctly,
tensors and metadata. In the other direction, `tests/data/hf_written.safetensors`
was written by that library and is read correctly by ours (a test).

## `transformer inspect`

```
transformer inspect model-init.safetensors [-v]
```

Prints the metadata and the stored `[model]`, then checks every tensor
against the layout that config implies: every expected name present, with
the right shape, and nothing unexpected. It ends with the weights
fingerprint. `-v` lists every tensor.

## The forward and backward passes (step 4, in progress)

`src/gpt.rs` writes the model's forward pass for a batch of B sequences of T
tokens, one row per token, on an automatic differentiation tape
(`src/tape.rs`), the same idea as attention/'s, so there is no hand-written
backward pass for the model:

```
x      = E[token] + P[position]
for each block:
  h    = x + (attention(LN₁(x)) · W_o + b_o)        Q, K, V = LN₁(x)·W + b
  x    = h + (GELU(LN₂(h)·W₁ + b₁)·W₂ + b₂)
logits = LN_f(x) · Eᵀ                              the output tied to E
loss   = mean cross-entropy against the next tokens
```

The tape differs from attention/'s in three ways:

- **Generic over the float type.** It's the same code for f32 (training)
  and f64 (gradient checks: finite differences in f32 are too noisy to
  catch subtle errors).
- **Batched.** A batch is one (B·T) × d matrix, so per-token operations
  handle every token at once.
- **Fused operations,** each with one hand-written backward rule:
  - layer norm, with the closed-form gradient attention/'s tape verified
  - multi-head causal attention over the whole batch, one parallel task
    per (sequence, head)
  - softmax with cross-entropy

**Threads.** Matrix products split their output into blocks of 8 rows,
which `rayon`'s worker threads take as they become free (work stealing), so
fast performance cores don't wait for slower efficiency cores. Weight
gradients (dW = Xᵀ·dY, summed over every token in the batch) are computed as
a sum of per-token updates in fixed blocks of 64 tokens, which keeps dW in
the cache. Element-by-element and row-by-row operations (GELU, biases,
residual additions, layer norm) run in parallel too. Without that, a
profile showed GELU alone taking 20% of a step on one thread, a share that
grows with more cores (Amdahl's law). Every result is the same whatever the
number of threads: each element is summed in a fixed order (tests check
this).

**Checks.**

- Every operation's gradient against finite differences in f64.
- Then every weight of a small two-block model at once: embeddings, both
  blocks' attention and FFN, all layer norms and biases, and the tied
  output layer.
- The initial loss is about ln V, a uniform guess.
- A later token never changes an earlier prediction.
- The tests catch deliberate errors: removing attention's 1/√h scale, or
  the softmax rule's −rowsum term, fails them.

## `transformer bench`

```
transformer bench [--checkpoint model-init.safetensors]
```

Measures this machine's f32 matrix-multiply speed, in GFLOP/s, at the shapes
a training step uses, on 1 thread and on all of them. It then estimates the
training speed: training costs about 6 floating-point operations per
parameter per token, so tokens per second ≈ speed / (6 · N). With
`--checkpoint`, N is that model's parameter count.

On a 2-core Linux machine it measures about 16 GFLOP/s per core, 33 on both.
The matrix multiply is straightforward (a row-by-row loop the compiler
vectorizes, rows split across threads); a tuned kernel, or Apple's
Accelerate library, could be several times faster. These measurements
decide whether that's worth doing (PLAN.md, open decision 6).

With `--checkpoint`, `bench` also times a real training step for that
model: forward and backward on a batch of 16 sequences of up to 128 tokens,
in f32, on all threads (the optimizer isn't included yet). For PLAN.md's
starting shape with a small vocabulary, a 2-core Linux machine takes about
870 ms per step: 2,350 tokens per second, 36% of its matrix-multiply speed.

## Tests

`cargo test` checks:

- the config: `[model]` round-trips through TOML; heads must divide
  `d_model`; unknown keys are rejected
- the tokenizer: for `tests/data/tiny.bpe` (written by mini_bpe.py), V and
  the fingerprint equal what Python computes
  (`tests/data/tiny.bpe.expected`); comments don't change the fingerprint,
  merges do; bad merge lines are rejected
- safetensors: an exact round trip, the header padding, rejection of
  truncated or padded files, and reading a file written by Hugging Face's
  library
- the model: the parameter count equals the formula (970,624 for PLAN.md's
  starting model; 124,439,808 for GPT-2 small); initialization is
  reproducible, with the planned standard deviations; a layout check names
  a missing tensor and a wrong shape
- the matrix multiplies (`matmul`, and `matmul_tn` for Aᵀ·B) equal the
  definition, and give identical results for 1 and several threads
- the tape: each operation against finite differences in f64; attention is
  causal and keeps sequences separate; f32 and f64 agree
- the model: every weight of a two-block model against finite differences;
  the initial loss is about ln V; a later token never changes an earlier
  prediction

## Source

- `src/main.rs`: the command line, `build` and `inspect`
- `src/config.rs`: `config.toml`'s `[model]` section
- `src/tokenizer.rs`: V and the fingerprint from `tok.bpe`
- `src/safetensors.rs`: the checkpoint format
- `src/model.rs`: weight names, shapes, the parameter formula,
  initialization and the layout check
- `src/tensor.rs`: the matrix type (f32 or f64) and the threaded multiplies
- `src/tape.rs`: the batched automatic differentiation tape
- `src/gpt.rs`: the model's forward pass, from a checkpoint's tensors
- `src/bench.rs`: the matrix-multiply benchmark and the training-step timing
- `src/rng.rs`: the seeded random number generator (from attention/)

## Next

The rest of step 4, `transformer train`:

- read `corpus.ids`, checking its tokenizer fingerprint against the
  checkpoint's
- the training loop with Adam, reporting loss and tokens per second
- write `model.safetensors` with Adam's state, so training can resume
