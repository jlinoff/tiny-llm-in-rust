# transformer

The real model: a small GPT-style language model, and the LLM flow's steps 3
to 6 in PLAN.md (build, train, finetune, generate). All four are done, plus
`evaluate` (measuring fine-tuning), `inspect` and `bench`. Step 6 was built
before step 5 on purpose (PLAN.md, "Order of work").

Terms (tensor, checkpoint, safetensors, layer norm, …) are defined in
[`../PLAN.md`](../PLAN.md#terms).

## Build

```
cargo build --release
cargo test --release
```

Rust 1.85 or later. Dependencies: `clap` (command line), `serde` with `toml`
(the config) and `serde_json` (the safetensors header), `sha2` (the
tokenizer fingerprint), `fancy-regex` (the tokenizer's pre-tokenizer
pattern, which needs a lookahead), and `rayon` (threads). The binary is
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

## `transformer train` (step 4)

```
transformer train --model model-init.safetensors --data corpus.ids --config config.toml -o model.safetensors
```

Trains the model on token IDs. It reads only the `[training]` section of
`config.toml` (see the comments there): the learning rate, the number of
steps and the batch size are required, and everything else has a default.

Before any work, it checks:

- the inputs are readable and the output writable
- the data file is made by the model's tokenizer: same V, and the
  fingerprint in `corpus.ids` equal to the one the checkpoint recorded at
  build time

The output may be the input checkpoint itself (resuming in place); it may
not be the data or the config.

**What it does each step.**

- Takes a batch of `batch_size` windows of `context` + 1 tokens, at random
  places in the training data. The last `validation_fraction` of the data
  (5%) is held out and never trained on.
- Runs the model forward and backward (the tape works out every gradient).
- Measures the gradient's overall length. If it's over `grad_clip`, scales
  the gradient down to that length, a guard against one unusual batch
  throwing the weights off.
- Updates every weight with **AdamW**: Adam, plus a small pull of each
  matrix and embedding toward zero (weight decay; biases and layer norm's
  γ and β are left alone).

The learning rate rises linearly over `warmup_steps`, then falls along a
cosine curve to `min_lr_fraction` of its peak at the last step.

It prints a line every `log_every` steps:

- the step
- the mean training loss since the last line
- the validation loss, every `eval_every` steps, measured on fixed windows
  from the held-out data, so the numbers are comparable over time
- the learning rate
- the mean gradient length before clipping
- tokens per second

**Checkpoints and resuming.** Every `checkpoint_every` steps, and at the
end, it writes the output checkpoint: the weights, Adam's state for every
weight (`adam.m.<name>`, `adam.v.<name>`), the step count and the tokens
seen. It writes to a temporary file first and then renames it, so an
interrupted write never leaves a damaged checkpoint.

The metadata carries forward everything from the input (the model, the
tokenizer, the config and their checksums) and adds:

- the `[training]` settings
- the step and the losses
- the data file and its SHA-256
- the parent checkpoint and its SHA-256, so a chain of checkpoints records
  its own history

To train further, raise `steps` in the config and pass the checkpoint
back as `--model`. A resumed run is exact: each step's batch depends only on
the seed and the step number, and Adam's state is saved, so stopping and
resuming gives bit-for-bit the same weights as not stopping (a test checks
this).

**A bug worth knowing about.** The first version picked each step's windows
with a random-number generator seeded by `seed ^ step × 0x9E37…`. That
constant is also the amount by which SplitMix64 steps its own state, so step
k + 1's numbers nearly repeated step k's, in exactly the low bits used to
choose a window. In 600 steps of 16 windows, training visited only about
1,200 distinct places out of 720,000, partly memorizing them. The symptom
was a training loss well below the validation loss from the very start, and
rising later. Each step now gets its own generator through SplitMix64's
scrambling function (`Rng::stream`), and a test requires the windows of
many steps to be as varied as random picks.

## `transformer finetune` (step 5)

```
transformer finetune --model model.safetensors --data chat.ids --tokenizer tok.bpe --config config.toml -o chat.safetensors
```

Post-training: the same training loop as `train` (`train::run_with`), on
conversations instead of random windows of text, with `[finetune]`'s
settings (the same keys as `[training]`). `chat.ids` comes from
`../chat/make_chat.py`, encoded with `mini_bpe.py --allow-special`: each
document is `<|user|> request <|assistant|> story <|end|>`.

- **One conversation per row**, starting with EOS, padded to the context
  length. Conversations longer than the context are left out, and counted.
- **Loss masking**: only the reply's tokens (the story and its `<|end|>`)
  are scored. The request is context, never a target. The loss is the mean
  over the scored tokens.
- **A fresh optimizer** when starting from a pretrained checkpoint: step 0,
  and Adam's averages reset. A fine-tuning checkpoint resumes exactly, as
  in `train`.
- **Before and after**: the validation loss on the held-out conversations'
  replies is printed before the first step, then as training goes.
- **Provenance**: `chat.safetensors` keeps the pretraining run's settings,
  step, validation loss and data (`pretraining_*` in its metadata), and its
  own as usual, with `kind = finetune`.
- **Checks**: the tokenizer must be the model's and have the turn markers;
  the data must come from the same tokenizer; every conversation must have
  the shape above (one request, one reply).

## `transformer evaluate`

```
transformer evaluate --model chat.safetensors --tokenizer tok.bpe --config config.toml --requests chat-eval.tsv --chat
transformer evaluate --model model.safetensors --tokenizer tok.bpe --config config.toml --requests chat-eval.tsv
make evaluate
```

Measures fine-tuning. For each request in `chat-eval.tsv` (from stories
neither training step saw), it generates a story with `[generate]`'s
sampling (request i uses seed + i) and counts how many of the three
requested words appear, as whole words ignoring case ("ball" counts in
"Ball." but not in "balloon" or "balls").

The control: the same stories checked for the words of a *different*
request (the next one). Words like "ball" are common in stories anyway, so
the control measures how often a story contains three such words by chance.
The difference between the two is what following the request adds. It also
reports how many stories ended by themselves (at `<|end|>` with `--chat`,
`<|endoftext|>` without) and their average length, and prints the first few
stories with each word marked ✓ or ✗ (`--show N`).

With `--chat`, each request is sent as a conversation. Without it, the
request is plain text for the model to continue: the baseline for the
pretrained model, which has never seen a turn marker. `make evaluate` runs
both: the pretrained model without `--chat`, the fine-tuned one with it.

## `transformer generate` (step 6)

```
transformer generate --model model.safetensors --tokenizer tok.bpe --config config.toml --prompt "Once upon a time"
make generate PROMPT='The dog' GEN_ARGS='--temperature 0.5 --seed 7'
```

With `--chat`, the prompt is a request to a fine-tuned model: it's sent as
`<|endoftext|> <|user|> request <|assistant|>`, exactly how each fine-tuning
conversation began, and generation stops at `<|end|>`. The markers are added
by the code: marker text typed into a prompt stays ordinary text.

```
transformer generate --model chat.safetensors --tokenizer tok.bpe --config config.toml --chat --prompt "Write a story with the words frog, bell and egg."
make chat PROMPT='Write a story with the words frog, bell and egg.'
```

Inference: the model continues the prompt one token at a time, printing
each as it's chosen, then reports the speed of the two phases, prefill (the
prompt, all at once) and decode (one token at a time). Any checkpoint works:
`model-init.safetensors` gives random tokens, a trained one gives text.

- **Settings**: `[generate]` in config.toml, all optional: `max_tokens`
  (200), `temperature` (0.8; 0 is greedy, always the likeliest token),
  `top_k` (40; 0 for no cutoff), `seed` (1). Each can be overridden on the
  command line (`--max-tokens`, `--temperature`, `--top-k`, `--seed`).
  The same seed and prompt give the same text.
- **The prompt** is encoded exactly as mini_bpe.py encodes (see Tests),
  with `<|endoftext|>` put first: every training document ended with it,
  so it marks the start of a new one. An empty prompt starts a story from
  nothing. Generation stops when the model picks `<|endoftext|>`.
- **The tokenizer** must be the one the model was built with: its
  fingerprint is checked against the checkpoint's.
- **The KV cache** (`src/generate.rs`): each block keeps the keys and values
  of every position so far, so a new token computes only its own row. This
  path doesn't use the tape, but it does the same arithmetic in the same
  order, and a test requires its predictions to equal the training forward
  pass's exactly, bit for bit, in f32 and f64. (Changing `x / total` to
  `x · (1/total)` in the attention weights is enough to fail it.)
- **Past the context length** there are no more positions, so when the
  cache is full it's rebuilt from the last ¾ of the context and generation
  continues: a sliding window. The cached keys and values can't simply be
  shifted, because each includes its old position's embedding. The report
  counts the rebuilds.
- **Output bytes**: a token can end partway through a UTF-8 character, so
  output waits until each character is complete.

Greedy decoding (`--temperature 0`) tends to fall into a loop, repeating a
phrase forever: once a phrase is the likeliest continuation of itself,
nothing breaks the cycle. Sampling with a temperature avoids it.

Speed, on the 2-core Linux machine with a 0.9M-parameter model: about 2,200
tokens per second within the context, about half that past it (the
rebuilds). Producing one token is a matrix-vector product per weight
matrix, where each weight is read once and used once, so it runs at about
a quarter of the matrix-matrix speed. Small products run on one thread:
handing about 30 tiny jobs per token to the thread pool cost more than the
work and halved the speed (`PARALLEL_MIN_WORK` in `src/tensor.rs`).

## `transformer inspect`

```
transformer inspect model-init.safetensors [-v]
```

Prints the metadata and the stored `[model]`, then checks every tensor
against the layout that config implies: every expected name present, with
the right shape, and nothing unexpected. It ends with the weights
fingerprint. `-v` lists every tensor.

## The forward and backward passes (step 4)

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
  merges do; bad merge lines are rejected; encoding 19 texts gives exactly
  Python's chunks and IDs (`tests/data/encode_cases.json`), including
  accents, combining marks, other scripts, emoji, and the characters where
  Python's `\w`, `\d` and `\s` differ from Rust's (see below); literal
  `<|endoftext|>` text is not the special token
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
- data: reading a `.ids` file written by mini_bpe.py (V, fingerprint,
  IDs); rejecting damaged files; batches shifted by one, reproducible, and
  as varied as random picks over many steps; the validation split is the
  held-out tail
- training: the learning-rate schedule; a tiny model memorizes a repeating
  text (loss from ln V to under 0.2); data from another tokenizer is
  refused; and 10 steps + checkpoint + resume + 10 steps gives exactly the
  weights and Adam state of 20 steps
- fine-tuning data: conversations parse into rows with only the reply and
  `<|end|>` scored, too-long ones are left out and counted, and malformed
  ones (no request, no reply, no end, two turns) are refused
- evaluation: whole-word matching, parsing `chat-eval.tsv`, and the tally of
  requested and control words
- generation: the KV cache gives exactly the training forward pass's
  predictions (above); greedy and top-1 pick the likeliest token; sampling
  frequencies match the softmax (200,000 draws, within 5 standard
  deviations) and top-k never picks outside the k; generation is
  reproducible for a seed and runs past the context; it stops at end of
  text; output waits for whole UTF-8 characters; `[generate]` is optional
  and checked

**Python's regular expressions vs Rust's.** The pre-tokenizer pattern was
written for Python's `re`, and the same pattern means something slightly
different in Rust, found by the encoding test. `\w` in Python is letters,
any kind of number (so "²", "½", "Ⅻ") and `_`; Rust's adds combining marks
(so "e" plus a combining accent stays one chunk instead of two) and
connector punctuation such as "‿". Python's `\s` also includes four old
control characters, U+001C–U+001F. `python_classes` in `src/tokenizer.rs`
rewrites each shorthand into an explicit Unicode class with Python's
meaning before compiling. A remaining caveat: Python and the Rust crate
each carry their own copy of the Unicode tables, so a character added in a
newer Unicode version could still be classed differently.

## Source

- `src/main.rs`: the command line: `build`, `train`, `finetune`, `generate`,
  `evaluate`, `inspect`, `bench`
- `src/config.rs`: `config.toml`'s `[model]` section
- `src/tokenizer.rs`: V and the fingerprint from `tok.bpe`; encoding and
  decoding, matching mini_bpe.py
- `src/safetensors.rs`: the checkpoint format
- `src/model.rs`: weight names, shapes, the parameter formula,
  initialization and the layout check
- `src/tensor.rs`: the matrix type (f32 or f64) and the threaded multiplies
- `src/tape.rs`: the batched automatic differentiation tape
- `src/gpt.rs`: the model's forward pass, from a checkpoint's tensors
- `src/data.rs`: reading `corpus.ids`, the validation split, and batches;
  conversations for fine-tuning, with loss masking
- `src/train.rs`: `[training]` and `[finetune]`, AdamW, the schedule, the
  training loop for any batch source, checkpoints, resuming
- `src/generate.rs`: `[generate]`, the KV-cache inference path, sampling,
  and the generation loop
- `src/evaluate.rs`: word matching and the tally for `evaluate`
- `src/bench.rs`: the matrix-multiply benchmark and the training-step timing
- `src/rng.rs`: the seeded random number generator (from attention/)

## Next

The full TinyStories run: a bigger model, about a day or more of training,
then fine-tuning it the same way.
