# Plan: building the pieces of an LLM

The goal is understanding, not performance: each stage is small enough to read,
and each runs on its own so its output can be inspected. A driver script
(working name `simple`) chains the LLM stages into a complete flow.

## Why Rust

Training is slow even for small models, because it's billions of tiny
arithmetic updates. word2vec on text8 (17 million words) makes about 375
million predictions over 5 epochs, each updating 7 vectors of 100 numbers:
over 2.5 billion vector updates per run. In plain Python, the interpreter's
overhead on every one of those operations dominates, likely making it 50–100×
slower (an estimate, not a measurement). NumPy only helps if the updates are
batched into large array operations, which hides the algorithm behind array
manipulation. In Rust, the inner loops are written as plain loops that read
like the math and run at native speed: the full text8 run takes about 8½
minutes on 2 CPU cores, and less on a machine with more cores.

The tokenizer stays in Python because it's fast enough at this scale. It can
move to Rust later if there's a compelling reason.

## The flows

There are two separate flows:

1. **The LLM flow**: the main goal. It builds a small GPT-style language
   model, from text to a trained model that generates text, and then
   fine-tunes it to follow a conversation format.
2. **The word2vec flow**: a historical precursor (2013) that learns word
   vectors and nothing else. It shares no files with the LLM flow; it's here
   because it shows, in isolation, how an embedding table learns meaning.

### 1. The LLM flow

```
simple:
  step          input(s)                                       output
  1. tokenize   corpus.txt                                 ──► tok.bpe          # contains: pattern, merges, special tokens; V
  2. encode     corpus.txt + tok.bpe                       ──► corpus.ids       # contains: the corpus as token IDs
  3. build      tok.bpe + config [model]                   ──► model-init.ckpt  # contains: all weights, random; the model settings
  4. train      model-init.ckpt + corpus.ids + [training]  ──► model.ckpt       # contains: the same weights, pretrained
  5. finetune   model.ckpt + chat.ids + [finetune]         ──► chat.ckpt        # contains: the same weights, tuned on conversations
  6. generate   chat.ckpt + tok.bpe + prompt + [generate]  ──► text
```

Key: each step reads the files to the left of `──►` and writes the one file to
its right. Text after `#` is a comment describing what that file contains.
`[model]`, `[training]`, `[finetune]` and `[generate]` are sections of the configuration
file (see [Configuration](#configuration)). `chat.ids` is step 2 run on
`chat.txt`, a file of example conversations.

Steps 1–4 are **pretraining**; step 5 is **post-training** (see
[Post-training](#post-training)). Step 6 works after either: with
`model.ckpt` it continues text, with `chat.ckpt` it answers in turns.

The files are the interfaces between stages, so their
formats must be defined exactly. This matters doubly because the tokenizer is
in Python and the later stages are in Rust.

#### What each stage does

| stage | input | output | notes |
|---|---|---|---|
| **tokenize** | training text, often a representative sample of the corpus | `tok.bpe`: pre-tokenizer pattern, ordered merges, special tokens, V | Done: `tokenizer/mini_bpe.py --save`, except special tokens (to do; needed by step 5). Trained once, then frozen. |
| **encode** | text + `tok.bpe` | token IDs | Turns text into the integers the model consumes. Real pipelines pre-tokenize the whole corpus into a binary ID file once, so training never touches text. |
| **build** | V (from `tok.bpe`) + config `[model]`: d, layers, heads, context length, seed | `model-init.ckpt` | Creates the **whole** model: the V × d embedding table, every attention and feed-forward layer, the output layer, all randomly initialized. Writes the `[model]` settings into the checkpoint. One seed gives one starting model, so experiments are reproducible. (Precedent: EleutherAI's Pythia models publish their step-0, untrained checkpoints.) |
| **train** | `model-init.ckpt` + `corpus.ids` + config `[training]`: learning rate, steps, batch size | `model.ckpt` | Repeatedly: predict the next token ID, compare with the real one, adjust every weight, including the embedding rows. Reads the model's shape from the checkpoint, never from the config. The tokenizer is not involved; it already did its work in stage 2. |
| **finetune** | `model.ckpt` + `chat.ids` + config `[finetune]` | `chat.ckpt` | The same training loop as stage 4, continued from the pretrained model on example conversations, usually with a lower learning rate. |
| **generate** | `model.ckpt` or `chat.ckpt` + `tok.bpe` + a prompt + config `[generate]`: temperature, length | text | Encode the prompt, predict a token ID, append it, repeat; decode the IDs back to text. |

#### What crosses each boundary

In flow order:

- **tokenize → encode:** `tok.bpe`, everything needed to turn text into token
  IDs: the pre-tokenizer pattern, the ordered merges, and the special tokens.
- **encode → train and finetune:** `corpus.ids` and `chat.ids`, which hold
  **only token IDs**. This is where text stops: the model never sees a
  character of the corpus, nor the tokenizer's pattern or merges, only the
  integers.
- **tokenize → build:** only V (the number of rows in the embedding table and
  columns in the output layer). The IDs themselves carry the rest of the
  connection: token ID 261 means `The` only under this `tok.bpe`, so a model
  checkpoint is only valid with the tokenizer it was built from, and the ID
  files only with the tokenizer that encoded them.
- **config → build:** the `[model]` settings, read once. From then on the
  checkpoint carries them.
- **build → train → finetune → generate:** the checkpoint. Training and
  fine-tuning change the values, never the shapes.
- **tokenize → generate:** `tok.bpe` again, used at both ends of generation:
  encoding the prompt into IDs, and decoding the predicted IDs back into text.
  The tokenizer is the first thing the flow uses and the last.

### 2. The word2vec flow

```
word2vec:
  step          input(s)                    output
  1. train      corpus.txt + settings   ──► vectors.txt   # contains: one d-dimensional vector per vocabulary word
  2. explore    vectors.txt + words     ──► (printed)     # nearest neighbors, analogies, analogy-test accuracy
```

Same key as above. "Settings" are the vector size d, window size, number of
epochs, and so on. Step 2 prints its results rather than writing a file.

word2vec is a single program (`mini_w2v`), as it was designed in 2013, so the
LLM flow's first four stages all happen inside its step 1 and nothing is saved
between them:

| LLM flow stage | word2vec equivalent (inside step 1) |
|---|---|
| tokenize | Count whole words; keep those occurring at least `--min-count` times. That count is V. Nothing is saved; V is printed. |
| encode | The corpus is held in memory as word IDs. |
| build | Create two V × d tables: input (the word vectors), random; output (context vectors), zero. |
| train | Predict nearby words instead of the next token: skip-gram predicts each context word from the center word, CBOW the center word from its context. |
| generate | None: word2vec can't produce text. Its product is the trained input table itself, written to `vectors.txt`. |

## Configuration

One file, `config.toml`, holds the settings for every LLM stage, with one
section per stage. Each stage reads only its own section:

```
[model]        # read by build:    d, layers, heads, context length, d_ff, seed
[training]     # read by train:    learning rate, steps, batch size
[finetune]     # read by finetune: learning rate, steps
[generate]     # read by generate: temperature, maximum length
```

**Terminology.** In machine learning, **parameters** means the learned
weights (a "7-billion-parameter model" has 7 billion of them). The settings we
choose, the independent variables of an experiment, are called
**hyperparameters**. The config file holds hyperparameters; the checkpoint
holds parameters.

**The `[model]` hyperparameters** define the model's shape:

| name | meaning |
|---|---|
| d (`d_model`) | the **model width** or hidden size: the length of every token's vector throughout the network |
| layers | how many transformer blocks (attention + feed-forward) are stacked |
| heads | how many attention heads each layer splits into; d must divide evenly by it |
| context length | the most tokens the model can see at once |
| d_ff | the inner width of each feed-forward layer, conventionally 4 × d |
| seed | the random seed for the initial weights |

Starting values will be chosen for a model that trains in minutes on a laptop
(roughly d = 128, 4 layers, 4 heads, context 128) and then varied as
experiments.

**Guaranteeing consistency.** d and the other `[model]` values must be the
same in every stage. Rather than every stage rereading the config (someone
could edit it between build and train), **build copies `[model]` into the
checkpoint**, and train, finetune and generate read the shapes from the
checkpoint only. After build, there is exactly one source of truth, so a
mismatch can't happen. Hugging Face does the same: every model ships a
`config.json` beside its weights.

**V is not in the config.** It comes from `tok.bpe`. Writing it in a second
place would only create another way for the two to disagree.

## Post-training

Steps 1–4 produce a **pretrained** model: it continues text. Ask it "What is
the capital of France?" and it may well continue with "What is the capital of
Spain?", because lists of questions are common in its training text. Turning a
text-continuer into an assistant is **post-training**, which has two common
stages:

1. **Supervised fine-tuning (SFT)**, step 5: keep training the pretrained
   model on example conversations (a prompt and a good response). It's the
   same training loop with different data and a lower learning rate. The model
   learns the format of a conversation: whose turn it is, and when a response
   ends.
2. **Preference tuning** (RLHF, or the simpler DPO): train the model toward
   responses that people, or a reward model trained on people's choices,
   preferred over alternatives. Out of scope for a toy model, but it's the
   step that shapes a production assistant's behavior most.

**Requirement on the tokenizer: special tokens.** Conversations need markers
that never occur in ordinary text, such as `<|user|>`, `<|assistant|>` and
`<|end|>`. Each is a single reserved token ID, added by hand rather than
learned from frequency. `mini_bpe.py` has none yet. They must be reserved
before step 3, because each one needs a row in the embedding table; adding
them after the model is built means growing the table.

**Expectations.** A toy model trained for minutes will produce text that is
locally plausible but often meaningless, and fine-tuning it will mostly teach
it the turn-taking format, not helpfulness. That's still the point: the
mechanics are the same as in a production model, only the scale differs.

## Where the stages live

```
llm/
├── PLAN.md        # this file
├── data/          # corpora, e.g. text8 (git-ignored)
├── tokenizer/     # text <-> token IDs
├── word2vec/      # word vectors, the historical precursor
├── attention/     # self-attention on its own
└── transformer/   # the model itself
```

| directory | flow steps | status |
|---|---|---|
| `tokenizer/` | LLM 1 tokenize, 2 encode | step 1 done (`mini_bpe.py`) except special tokens; step 2 to do |
| `word2vec/` | word2vec 1 train, 2 explore | trainer working; scripts and README to do |
| `attention/` | none: a learning step before `transformer/` | not started |
| `transformer/` | LLM 3 build, 4 train, 5 finetune, 6 generate | not started |

## word2vec: the historical side branch

word2vec (2013) is not part of the LLM flow. It shows the oldest form of the
idea the embedding table relies on, that a word's meaning can be learned from
the words around it, with nothing else in the way:

- Its internal steps mirror the LLM flow's first four stages (see "The
  word2vec flow" above).
- Its vocabulary is whole words counted from the corpus, not `tok.bpe`.
- Its trained table *is* its final product. It's a legitimate phase output,
  usable for nearest neighbors and analogies.
- Its two tables correspond to an LLM's embedding (input) and unembedding
  (output) layers. An LLM puts the transformer layers between them, and its
  vectors become contextual: "bank" is represented differently in "river
  bank" and "bank loan". word2vec's vectors are static.

**Why it matters for self-attention and feed-forward layers.** word2vec isn't
an ancestor of either mechanism; it's the foundation they build on and the
limitation they were invented to overcome:

- **The representation.** It established that a word can be a dense learned
  vector whose dot product with another measures relatedness. Self-attention
  is built from exactly that operation: each token scores every other token by
  a dot product, then mixes their vectors according to those scores.
- **The limitation.** One fixed vector per word can't tell "river bank" from
  "bank loan". Self-attention exists to fix this: it rebuilds each token's
  vector from its context, layer after layer, and the feed-forward layers
  transform each of those context-aware vectors further.
- **The training principle.** It showed that useful meaning emerges just from
  predicting words from their neighbors in raw, unlabeled text. LLM
  pretraining applies the same self-supervised idea, predicting the next
  token, at vastly larger scale.

Status: Rust trainer (skip-gram and CBOW, negative sampling, Hogwild threads)
written and trained on text8. Sensible neighbors emerge (king → prince, queen;
three → four, five). Before it's delivered: investigate why the reported loss
rises slightly in later epochs, finish the Python exploration scripts (analogy
with and without excluding the input words; Google analogy test set), compare
against gensim, and write the README.

## Open decisions

1. **Where encoding happens in Rust.** Stages 4–6 need token IDs, and stage
   6 must also decode IDs to text. Options:
   - (a) Python pre-encodes the corpus to `corpus.ids` (stage 2), and Rust
     only ever sees IDs. Generate then needs decode in Rust, which is
     easy: decoding is concatenating each token's bytes.
   - (b) Port `encode` to Rust as well. Small, but note that the pattern's
     lookahead `\s+(?!\S)` isn't supported by Rust's `regex` crate; it would
     need `fancy-regex`, or the pattern would need rewriting.

   Leaning toward (a): it matches how real pipelines work and keeps a single
   implementation of encoding.
2. **The `corpus.ids` format.** A simple binary file of little-endian
   integers (u16 is enough while V < 65,536), with a small header, is what
   nanoGPT does.
3. **The checkpoint format.** A custom binary format with a header, or
   safetensors (simple, well-specified, readable from both Rust and Python,
   which allows inspecting weights in Python).

   *A "checkpoint" is a saved snapshot of a model's weights, taken during
   training or at the end. `.ckpt` in the flow above is a placeholder meaning
   "a checkpoint, format to be decided". Once a format is chosen, the files
   should use its extension (for safetensors: `model-init.safetensors`,
   `model.safetensors`).*

   `.ckpt` is a convention, not a standard. TensorFlow historically saved
   `model.ckpt-…` files, PyTorch Lightning uses `.ckpt`, and early Stable
   Diffusion models were distributed as `.ckpt`. In each case the extension
   says nothing about the format inside. Common extensions for model weights:

   | extension | what it is |
   |---|---|
   | `.safetensors` | Hugging Face's format: a JSON header giving each tensor's name, type, and shape, followed by the raw numbers. Now the default on Hugging Face. |
   | `.pt` / `.pth` | PyTorch's native save format, built on Python's pickle |
   | `.bin` | older Hugging Face files (`pytorch_model.bin`), also pickle |
   | `.gguf` | llama.cpp's format for running models locally |
   | `.ckpt` | TensorFlow and Lightning checkpoints, old Stable Diffusion files |

   **Security note:** pickle-based files (`.pt`, `.bin`, old `.ckpt`) can
   execute arbitrary code when loaded, because unpickling can reconstruct
   objects by calling functions. Malicious model files have been found on
   public model hubs. safetensors was created mainly to close that hole: it
   contains only data, so loading it can't run anything. That's a point in
   its favor here.
4. **The config format.** The design is settled (see
   [Configuration](#configuration)); the syntax isn't. TOML is easy to read and
   edit by hand and supports comments; JSON is what Hugging Face uses for the
   copy stored with a model.
5. **The driver.** A shell script or a small Makefile. A Makefile would only
   re-run the stages whose inputs changed.
6. **The training data.** A pretraining corpus small enough to train on in
   minutes, but large enough to learn something: text8 (17 million words) is
   one candidate. For fine-tuning, a few hundred to a few thousand short
   example conversations: hand-written, generated, or from a small public
   dataset.

## Order of work

1. Finish word2vec (above).
2. Tokenizer: reserve special tokens, then stage 2 (`--encode` writing
   `corpus.ids`).
3. `attention/`: self-attention alone, on toy inputs.
4. `transformer/`: build, then train, then generate, then finetune.
5. The `simple` driver, and fold this plan into `llm/README.md`.
