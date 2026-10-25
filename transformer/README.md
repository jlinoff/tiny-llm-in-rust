# transformer

The real model: a small GPT-style language model, and the LLM flow's steps 3
to 6 in PLAN.md (build, train, finetune, generate). So far, step 3 is done.

Terms (tensor, checkpoint, safetensors, layer norm, …) are defined in
[`../PLAN.md`](../PLAN.md#terms).

## Build

```
cargo build --release
cargo test --release
```

Rust 1.85 or later. Dependencies: `clap` (command line), `serde` with `toml`
(the config) and `serde_json` (the safetensors header), and `sha2` (the
tokenizer fingerprint). The binary is `target/release/transformer`.

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
- the benchmark's threaded matrix multiply equals the definition exactly,
  for 1 to 8 threads

## Source

- `src/main.rs`: the command line, `build` and `inspect`
- `src/config.rs`: `config.toml`'s `[model]` section
- `src/tokenizer.rs`: V and the fingerprint from `tok.bpe`
- `src/safetensors.rs`: the checkpoint format
- `src/model.rs`: weight names, shapes, the parameter formula,
  initialization and the layout check
- `src/bench.rs`: the matrix-multiply benchmark
- `src/rng.rs`: the seeded random number generator (from attention/)

## Next

Step 4, `transformer train`: the f32 forward and backward passes for the
whole model, checked against attention/'s f64 reference versions, reading
`corpus.ids` and writing `model.safetensors` with Adam's state.
