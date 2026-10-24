# Plan: building the pieces of an LLM (large language model)

This project builds a small LLM piece by piece. The goal
is understanding, not performance: each stage is small enough to read, and each
runs on its own so its output can be inspected. A Makefile (a build script run
by the `make` tool) chains the LLM stages into a complete flow, with one target
per step: `make tokenize`, `make encode`, and so on.

Technical terms are defined in [Terms](#terms), just below; tool and product
names are explained where they first appear.

## Terms

Grouped so that each definition uses only terms defined above it.

**Text and tokens**

- **corpus**: the body of text a model is trained on.
- **token**: a unit of text the model works with: a word, part of a word, a
  punctuation mark, or a single byte. Each token has a **token ID**, an integer.
- **tokenizer**: the component that converts text to token IDs (**encoding**)
  and token IDs back to text (**decoding**).
- **BPE (byte-pair encoding)**: the algorithm that learns a tokenizer. It starts
  from the 256 byte values and repeatedly **merges** the most frequent adjacent
  pair into a new token. The ordered list of merges is the tokenizer's main
  content. `mini_bpe.py` implements it, and `.bpe` is our extension for its
  saved tokenizer.
- **pre-tokenizer pattern**: the regular expression that first splits text into
  chunks (words, numbers, runs of punctuation), so that merges never cross a
  chunk boundary.
- **special tokens**: tokens added by hand rather than learned, to mark
  structure rather than text, such as the end of a document (**EOS**, "end of
  sequence") or the start of a user's turn in a conversation (more in
  [Special tokens](#special-tokens)).
- **vocabulary**, **V**: the set of all tokens, and its size.

**The model**

- **vector**: a list of numbers; here, the numbers that represent a token.
- **d (model width)**: the length of every token's vector (more in
  [Configuration](#configuration)).
- **layer**: one stage of computation that the vectors pass through.
- **embedding table**: a V × d table with one row (one vector) per token. It is
  the model's first layer: a token ID selects its row.
- **weights**, **parameters**: the numbers inside the model that are adjusted
  as it learns. **Hyperparameters** are the settings we choose (more in
  [Configuration](#configuration)).
- **self-attention**: a layer in which each token's vector takes in
  information from the tokens before it, weighting each by a relevance score
  (a dot product of their vectors). An **attention head** is one independent
  copy of this computation; each attention layer runs several side by side.
- **feed-forward layer**: a layer that transforms each token's vector on its
  own, the same way at every position.
- **transformer block**: one self-attention layer followed by one feed-forward
  layer. A **transformer** is a model built from a stack of these blocks.
- **output layer** (or **unembedding**): the final layer, which turns a vector
  into V scores, one for each possible next token.
- **context length**: the most tokens the model can take in at once.
- **GPT (generative pre-trained transformer)**: a transformer trained to
  predict the next token. "GPT-style" means this design, which most LLMs use.

**Training**

- **training**: repeatedly make a prediction, measure the error, and adjust the
  weights slightly to reduce it (gradient descent).
- **learning rate**: how large each adjustment is.
- **step**, **batch size**: one adjustment; the number of examples used to
  compute it.
- **epoch**: one full pass over the training data.
- **seed**: the starting value of a pseudo-random number generator; the same
  seed reproduces the same "random" numbers.
- **pretraining**: training on a large corpus to predict the next token.
  **Post-training**: further training that turns the result into an assistant
  (see [Post-training](#post-training)). **Fine-tuning**: continuing to train
  an already trained model on new data.
- **temperature**: a text-generation setting. Low values pick the most likely
  next token; higher values sample more varied ones.
- **checkpoint**: a saved file of a model's weights (more in open decision 3).

**word2vec**

- **word2vec**: a 2013 method that learns one vector per word, the historical
  precursor covered by this project's second flow.
- **skip-gram**, **CBOW (continuous bag of words)**: word2vec's two training
  tasks. Skip-gram predicts each nearby word from the center word; CBOW
  predicts the center word from its neighbors.
- **negative sampling**: word2vec's shortcut. Instead of scoring all V words
  for each prediction, it scores the correct word against a few randomly drawn
  wrong ones.
- **Hogwild**: training with many threads that update the shared weights
  without locks, accepting the rare collision.

## Why Rust

Training is slow even for small models, because it's billions of tiny
arithmetic updates. word2vec on text8 (a standard benchmark corpus: 17 million
words of Wikipedia text) makes about 375
million predictions over 5 epochs, each updating 7 vectors of 100 numbers:
over 2.5 billion vector updates per run. In plain Python, the interpreter's
overhead on every one of those operations dominates, likely making it 50–100×
slower (an estimate, not a measurement). NumPy (Python's numerical array
library) only helps if the updates are
batched into large array operations, which hides the algorithm behind array
manipulation. In Rust, the inner loops are written as plain loops that read
like the math and run at native speed: the full text8 run takes about 8½
minutes on 2 processor cores, and less on a machine with more cores.

The tokenizer stays in Python because it's fast enough at this scale. Its
first trainer recounted every pair after every merge and would have needed
about half an hour for 4,000 merges on 20 MB; the current one updates only
what each merge changes and takes 17 seconds, with identical results. Fixing
the algorithm gained more than a Rust port of the old one would have (an
estimated 30 to 100 times), without the risk of two pre-tokenizers
disagreeing. It can still move to Rust if there's a compelling reason.

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
LLM flow:
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

Commands marked *proposed* belong to programs not yet written; their names
and options may change when the code is. `transformer` is the planned program
in `transformer/`, with one subcommand per step.

##### Step 1: tokenize

- **Input:**
  - training text, often a representative sample of the corpus
- **Output:**
  - `tok.bpe`: the pre-tokenizer pattern, the ordered merges, the special
    tokens, and V
- **Command:**

  ```
  python3 tokenizer/mini_bpe.py corpus.txt -n 1000 --save tok.bpe
  ```

  `-n` is the maximum number of merges, `--save` names the output. Special
  tokens will need one more option, proposed as
  `--special "<|user|>,<|assistant|>,<|end|>"`.
- **Status:** done (`tokenizer/mini_bpe.py --save`), including EOS; the turn
  markers for step 5 are still to do

**What it does:** learns the tokenizer from the training text, and reserves
IDs for the special tokens after the merges. It runs once; the result is then
frozen and used unchanged by every later step that needs it.

##### Step 2: encode

- **Inputs:**
  - text: `corpus.txt` (one or more documents) for step 4, `chat.txt` for
    step 5
  - `tok.bpe`
- **Output:**
  - `corpus.ids` (or `chat.ids`): the text as token IDs
- **Command:**

  ```
  python3 tokenizer/mini_bpe.py --load tok.bpe --encode corpus.txt -o corpus.ids
  python3 tokenizer/mini_bpe.py --load tok.bpe --encode chat.txt   -o chat.ids
  ```

  `--load` reads the saved tokenizer, `--encode` names the text, `-o` the
  output.
- **Status:** done (`tokenizer/mini_bpe.py --encode`)

**What it does:** turns text into the integers the model consumes. Real
pipelines pre-tokenize the whole corpus into a binary ID file once, so training
never touches text. The `.ids` file has a 64-byte header recording V and a
fingerprint of the tokenizer (a hash of its pattern and merges), so later steps
can refuse IDs made by a different tokenizer. After encoding, the IDs are
decoded and compared with the input. The format is described in
`tokenizer/README.md`. Encoding is also where special tokens enter the data:
each input file is a document, and EOS is appended after each one (done); each
message of `chat.txt` will be wrapped in turn markers (planned; see
[Special tokens](#special-tokens)).

##### Step 3: build

- **Inputs:**
  - V, from `tok.bpe`
  - config `[model]`: d, layers, heads, context length, d_ff, seed
- **Output:**
  - `model-init.ckpt`: all the model's weights, random, plus the `[model]`
    settings
- **Command** (proposed):

  ```
  transformer build --tokenizer tok.bpe --config config.toml -o model-init.ckpt
  ```
- **Status:** not started

**What it does:** creates the **whole** model: the V × d embedding table,
every attention and feed-forward layer, and the output layer, all randomly
initialized. One seed gives one starting model, so experiments are
reproducible. (Precedent: EleutherAI, a nonprofit AI research group, publishes
the step-0, untrained checkpoints of its Pythia models.)

##### Step 4: train

- **Inputs:**
  - `model-init.ckpt`
  - `corpus.ids`
  - config `[training]`: learning rate, steps, batch size
- **Output:**
  - `model.ckpt`: the same weights, pretrained
- **Command** (proposed):

  ```
  transformer train --model model-init.ckpt --data corpus.ids --config config.toml -o model.ckpt
  ```
- **Status:** not started

**What it does:** repeatedly predicts the next token ID, compares the
prediction with the real one, and adjusts every weight slightly, including
the embedding rows. It reads the model's shape from the checkpoint, never from
the config. The tokenizer is not involved; it already did its work in step 2.

##### Step 5: finetune

- **Inputs:**
  - `model.ckpt`
  - `chat.ids`
  - config `[finetune]`: learning rate, steps
- **Output:**
  - `chat.ckpt`: the same weights, tuned on conversations
- **Command** (proposed):

  ```
  transformer finetune --model model.ckpt --data chat.ids --config config.toml -o chat.ckpt
  ```
- **Status:** not started

**What it does:** runs the same training loop as step 4, continuing from the
pretrained model on example conversations, usually with a lower learning rate.
This is post-training (see [Post-training](#post-training)).

##### Step 6: generate

- **Inputs:**
  - `chat.ckpt` (or `model.ckpt`)
  - `tok.bpe`
  - a prompt
  - config `[generate]`: temperature, maximum length
- **Output:**
  - text
- **Command** (proposed):

  ```
  transformer generate --model chat.ckpt --tokenizer tok.bpe --config config.toml --prompt "Hello"
  ```

  The output text is printed rather than written to a file.
- **Status:** not started

**What it does:** encodes the prompt, predicts a token ID, appends it, and
repeats; then decodes the IDs back to text. With `model.ckpt` it continues the
prompt; with `chat.ckpt` it answers as the assistant.

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
LLM flow's first four steps all happen inside its step 1 and nothing is saved
between them. Each LLM step and its word2vec equivalent:

- **tokenize:** count whole words; keep those occurring at least
  `--min-count` times. That count is V. Nothing is saved; V is printed.

- **encode:** the corpus is held in memory as word IDs.

- **build:** create two V × d tables: input (the word vectors), random; output
  (context vectors), zero.

- **train:** predict nearby words instead of the next token. Skip-gram predicts
  each context word from the center word; CBOW predicts the center word from
  its context.

- **finetune:** none.

- **generate:** none. word2vec can't produce text. Its product is the trained
  input table itself, written to `vectors.txt`.

## Special tokens

A special token is a token ID reserved by hand for structure, not text. It
never comes out of encoding ordinary text. Common kinds:

- **EOS (end of sequence, or end of document).** GPT-2's is `<|endoftext|>`.
  In training, documents are joined into one stream with EOS between them, so
  the model learns where one ends and an unrelated one begins. In generation,
  the model *predicting* EOS is how it signals that it's finished; without
  it, generation runs until a length limit.
- **BOS (beginning of sequence).** Some models start every input with one.
- **PAD (padding).** Filler that makes the sequences in a training batch the
  same length; the loss ignores it.
- **Turn markers.** Mark who is speaking in a conversation, such as
  `<|user|>`, `<|assistant|>`, and an end-of-turn token. The end-of-turn token
  is what makes a chat model stop after its answer instead of also writing the
  user's next line.

What makes them special:

1. **Not learned from frequency.** They're appended after the merges, so
   V = 256 + merges + number of special tokens.
2. **An ordinary embedding row.** Each gets a row in the embedding table,
   random at first and trained like any other; the model learns what EOS
   "means" from where it appears.
3. **Never produced from text.** If the text contains the literal characters
   `<|endoftext|>`, the encoder treats them as ordinary bytes. Otherwise any
   text could fake a document boundary or a change of speaker, a
   tokenizer-level form of injection. Special IDs are inserted only by code.

Where they appear in the flow:

- **Step 1, tokenize:** defined. The list is saved in `tok.bpe`, each gets a
  reserved ID, and they count toward V and the fingerprint. EOS
  (`<|endoftext|>`) is always reserved (done).
- **Step 2, encode:** inserted. EOS after each document in `corpus.ids`
  (done); turn markers around each message when `chat.txt` is encoded (to
  do).
- **Step 3, build:** each needs its row in the embedding table, which is why
  they must be defined before the model is built.
- **Step 6, generate:** used. Generation stops at EOS or end-of-turn, and the
  prompt is wrapped in the same turn markers.

## Configuration

One file, `config.toml`, holds the settings for every LLM stage, with one
section per stage. (TOML, "Tom's Obvious, Minimal Language", is a simple
format for settings files.) Each stage reads only its own section:

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
mismatch can't happen. Hugging Face, the main public hub for sharing models,
does the same: every model ships a `config.json` beside its weights. (JSON,
JavaScript Object Notation, is a common text format for structured data.)

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
2. **Preference tuning**: train the model toward responses that people
   preferred over alternatives. The two main methods are RLHF
   (reinforcement learning from human feedback), which first trains a
   **reward model** (a second model that predicts which response people would
   prefer) and then trains the LLM to score well on it, and the simpler DPO
   (direct preference optimization), which learns from the preference pairs
   directly. Out of scope for a toy model, but it's the
   step that shapes a production assistant's behavior most.

**Requirement on the tokenizer: special tokens.** Conversations need markers
that never occur in ordinary text, such as `<|user|>`, `<|assistant|>` and
`<|end|>`; see [Special tokens](#special-tokens).

**Expectations.** A toy model trained for minutes will produce text that is
locally plausible but often meaningless, and fine-tuning it will mostly teach
it the turn-taking format, not helpfulness. That's still the point: the
mechanics are the same as in a production model, only the scale differs.

## Conventions for all code

Rules every program in the project follows:

- **Check files before doing any work.** Every output file must be writable
  and every input file readable *before* any training or encoding starts, so
  a bad path fails in a fraction of a second instead of after a long run. The
  check must not modify an existing output file, and must not leave an empty
  file behind if the run later fails. (Implemented in `mini_bpe.py`,
  `mini_w2v` and `train_gensim.py`.)
- **Verify what can be verified.** When an output can be checked cheaply,
  check it: `--encode` decodes its IDs and compares them with the input
  before writing anything. That check found a bug in which underscores were
  silently dropped by the tokenizer (see `tokenizer/README.md`).
- **Record what produced a file.** Files passed between steps say what made
  them: the tokenizer lists its training documents with checksums, and an
  `.ids` file carries the tokenizer's fingerprint.

## Where the stages live

The rule: **one directory per component, one command per step.** A component
is code that several steps share. The tokenizer's code serves steps 1, 2 and 6;
the model's code serves steps 3–6. Splitting the steps into separate
directories would mean copying that code or reaching across directories for it.

```
llm/
├── PLAN.md        # this file
├── Makefile       # one target per step: make tokenize, make encode, ...
├── out/           # generated files: tok.bpe, *.ids, *.ckpt (git-ignored; make clean removes it)
├── data/          # corpora and example conversations, e.g. text8, chat.txt (git-ignored)
├── tokenizer/     # text <-> token IDs: tokenize, encode, decode
├── word2vec/      # word vectors, the historical precursor
├── attention/     # self-attention on its own
└── transformer/   # the model: build, train, finetune, generate
```

| directory | flow steps | status |
|---|---|---|
| `tokenizer/` | LLM 1 tokenize, 2 encode; decode for 6 | steps 1 and 2 done (`mini_bpe.py`), with EOS; turn markers to do |
| `word2vec/` | word2vec 1 train, 2 explore | done |
| `attention/` | none: a learning step before `transformer/` | not started |
| `transformer/` | LLM 3 build, 4 train, 5 finetune, 6 generate | not started |

The Makefile in `llm/` drives the flow, with one target per step. Each target
knows which files it needs, so `make train`, for example, first reruns any
earlier step whose inputs have changed, and skips the ones that are up to
date:

```
make tokenize  →  make encode  →  make build  →  make train  →  make finetune  →  make generate
```

Generated files go in `out/` (so `make clean` is simply `rm -rf out`), and
corpora stay in `data/`. Make tracks files, not variables: after changing a
setting such as `MERGES`, force a rebuild with `make -B <target>`. The commands
the targets run, one per step:

```
# 1 tokenize
python3 tokenizer/mini_bpe.py corpus.txt -n 1000 --save tok.bpe
# 2 encode (the corpus for step 4, the conversations for step 5)
python3 tokenizer/mini_bpe.py --load tok.bpe --encode corpus.txt -o corpus.ids
python3 tokenizer/mini_bpe.py --load tok.bpe --encode chat.txt   -o chat.ids
# 3 build
transformer build    --tokenizer tok.bpe --config config.toml -o model-init.ckpt
# 4 train (end of pretraining)
transformer train    --model model-init.ckpt --data corpus.ids --config config.toml -o model.ckpt
# 5 finetune (post-training)
transformer finetune --model model.ckpt --data chat.ids --config config.toml -o chat.ckpt
# 6 generate
transformer generate --model chat.ckpt --tokenizer tok.bpe --config config.toml --prompt "Hello"
```

As in [What each stage does](#what-each-stage-does), everything after step 2
is proposed. The `tokenize` and `encode` targets exist so far; the others are added as
each step is built. (The paths above are shortened: in the Makefile, the
corpus is `data/corpus.txt` and generated files are in `out/`.)

**Where post-training lives.** The example conversations (`chat.txt`) are in
`data/`. The special tokens are reserved by `tokenizer/` in `tok.bpe`, and
`chat.txt` is encoded with the same `--encode` as the corpus. Fine-tuning
itself is `transformer finetune`, the same training loop as `train`, so it
shares all the model code. Preference tuning, if it's ever added, would be
another `transformer` subcommand for the same reason.

**A naming note.** The tokenizer script keeps the name `mini_bpe.py`, not
`tokenize.py`: `tokenize` is a module in Python's standard library, and a
script of that name would shadow it. Running it puts the script's own
directory first on the import path, so standard-library code that imports
`tokenize`, including Python's error-reporting machinery (`linecache`,
`traceback`), would load the script instead.

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

Status: done. The Rust trainer (`mini_w2v`: skip-gram and CBOW, negative
sampling, Hogwild threads) scores 38.0% on Google's analogy test set (about
19,500 questions such as Athens : Greece :: Baghdad : ?) after training on
text8, matching gensim (a widely used Python library with its own word2vec) at
37.4%. Details, and the honest caveats, are in `word2vec/README.md`:

- **Two losses.** The loss measured during training rose in later epochs
  while a loss measured on a fixed sample with frozen weights kept falling;
  the trainer now reports both. The same distinction, training loss versus
  evaluation loss, will matter for the LLM.
- **Analogies.** Excluding the question's own words (the standard convention)
  is worth most of the score: without it, accuracy falls to about 9%. On text8,
  king − man + woman gave *queen* only 11th; individual analogies are a coin
  toss at this corpus size.

## Open decisions

1. **Where encoding happens in Rust.** Stages 4–6 need token IDs, and stage
   6 must also decode IDs to text. Options:
   - (a) Python pre-encodes the corpus to `corpus.ids` (stage 2), and Rust
     only ever sees IDs. Generate then needs decode in Rust, which is
     easy: decoding is concatenating each token's bytes.
   - (b) Port `encode` to Rust as well. Small, but note that the pattern's
     lookahead `\s+(?!\S)` isn't supported by `regex`, the standard Rust
     library ("crate") for regular expressions; it would need the
     `fancy-regex` crate, which supports lookahead, or the pattern would need
     rewriting.

   *Decided:* (a). It matches how real pipelines work and keeps a single
   implementation of encoding.
2. **The `corpus.ids` format.** *Decided:* a 64-byte header (magic, version,
   bytes per ID, V, count, tokenizer fingerprint), then the IDs as
   little-endian integers: 2 bytes each while V ≤ 65,536, otherwise 4. This
   follows nanoGPT (Andrej Karpathy's minimal GPT training code), which
   stores plain 16-bit IDs, plus the header. Specified in
   `tokenizer/README.md`.
3. **The checkpoint format.** A custom binary format with a header, or
   safetensors (Hugging Face's weights format, described below: simple,
   well-specified, readable from both Rust and Python, which allows inspecting
   weights in Python).

   *A "checkpoint" is a saved snapshot of a model's weights, taken during
   training or at the end. `.ckpt` in the flow above is a placeholder meaning
   "a checkpoint, format to be decided". Once a format is chosen, the files
   should use its extension (for safetensors: `model-init.safetensors`,
   `model.safetensors`).*

   `.ckpt` is a convention, not a standard. TensorFlow (Google's deep-learning
   framework) historically saved `model.ckpt-…` files; PyTorch Lightning (a
   training library built on PyTorch, Meta's deep-learning framework) uses
   `.ckpt`; and early versions of Stable Diffusion (an image-generation model)
   were distributed as `.ckpt`. In each case the extension
   says nothing about the format inside. Common extensions for model weights:

   | extension | what it is |
   |---|---|
   | `.safetensors` | Hugging Face's format: a JSON header giving each tensor's name, type, and shape, followed by the raw numbers. Now the default on Hugging Face. |
   | `.pt` / `.pth` | PyTorch's native save format, built on pickle (Python's built-in format for saving arbitrary objects) |
   | `.bin` | older Hugging Face files (`pytorch_model.bin`), also pickle |
   | `.gguf` | the format of llama.cpp, a popular program for running models on your own computer |
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
5. **The driver.** *Decided:* a Makefile, with one target per step (see
   [Where the stages live](#where-the-stages-live)). Unlike a shell script, it
   reruns only the steps whose inputs changed.
6. **The training data.** A pretraining corpus small enough to train on in
   minutes, but large enough to learn something: text8 (17 million words) is
   one candidate. For fine-tuning, a few hundred to a few thousand short
   example conversations: hand-written, generated, or from a small public
   dataset.

## Order of work

1. ~~Finish word2vec.~~ Done.
2. Tokenizer: ~~stage 2 (`--encode` writing `corpus.ids`)~~ and ~~EOS after
   each document~~ done. Turn markers when post-training needs them.
3. `attention/`: self-attention alone, on toy inputs.
4. `transformer/`: build, then train, then generate, then finetune.
5. Throughout: add each step's Makefile target as the step is built
   (`tokenize` exists). At the end, fold this plan into `llm/README.md`.

## Bibliography

Grouped by stage of the plan. Each entry says why it matters here. ★ marks a
suggested first-reading path, roughly one paper per stage. Most papers are free
on arXiv, the open preprint server used throughout machine learning.

Venue abbreviations: NeurIPS (Conference on Neural Information Processing
Systems), ICLR (International Conference on Learning Representations), ICML
(International Conference on Machine Learning), JMLR (Journal of Machine
Learning Research). ACL is the Association for Computational Linguistics; its
conferences ACL, NAACL (North American chapter), EACL (European chapter) and
EMNLP (Empirical Methods in Natural Language Processing), and its journals TACL
(Transactions of the ACL) and Computational Linguistics, are the main venues
for language research.

### Tokenization (`tokenizer/`)

- Gage, P. (1994). "A New Algorithm for Data Compression." *C Users Journal*
  12(2). The original byte-pair encoding, invented for compression, not
  language.
- ★ Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation
  of Rare Words with Subword Units." ACL.
  [arXiv:1508.07909](https://arxiv.org/abs/1508.07909). Brought BPE to neural
  language models; the algorithm `mini_bpe.py` implements.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).
  "Language Models are Unsupervised Multitask Learners." OpenAI technical
  report (GPT-2). Introduced byte-level BPE and the pre-tokenizer regex that
  `mini_bpe.py`'s pattern approximates.
- Kudo, T., & Richardson, J. (2018). "SentencePiece: A simple and language
  independent subword tokenizer and detokenizer for Neural Text Processing."
  EMNLP. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226). The `▁`
  space-marker convention and the main alternative toolkit.
- Kudo, T. (2018). "Subword Regularization: Improving Neural Network
  Translation Models with Multiple Subword Candidates." ACL.
  [arXiv:1804.10959](https://arxiv.org/abs/1804.10959). The unigram tokenizer,
  which prunes a large vocabulary down instead of merging up.
- Petrov, A., La Malfa, E., Torr, P., & Bibi, A. (2023). "Language Model
  Tokenizers Introduce Unfairness Between Languages." NeurIPS.
  [arXiv:2305.15425](https://arxiv.org/abs/2305.15425). The "token tax": the
  same text costs some languages several times more tokens.
- Land, S., & Bartolo, M. (2024). "Fishing for Magikarp: Automatically
  Detecting Under-trained Tokens in Large Language Models." EMNLP.
  [arXiv:2405.05417](https://arxiv.org/abs/2405.05417). Glitch tokens such as
  " SolidGoldMagikarp": what happens when the tokenizer's sample and the
  training corpus differ.
- Xue, L., et al. (2022). "ByT5: Towards a Token-Free Future with Pre-trained
  Byte-to-Byte Models." TACL.
  [arXiv:2105.13626](https://arxiv.org/abs/2105.13626). The alternative: skip
  learned tokenization and feed the model raw bytes.

### Word vectors (`word2vec/`)

- Harris, Z. S. (1954). "Distributional Structure." *Word* 10(2–3), and Firth,
  J. R. (1957). "A Synopsis of Linguistic Theory, 1930–1955." The
  distributional hypothesis behind all of this: "You shall know a word by the
  company it keeps" (Firth).
- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). "A Neural
  Probabilistic Language Model." JMLR 3. The first neural language model with
  a learned embedding table, ten years before word2vec.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). "Efficient Estimation
  of Word Representations in Vector Space."
  [arXiv:1301.3781](https://arxiv.org/abs/1301.3781). The first word2vec
  paper: skip-gram and CBOW, and the analogy test set.
- ★ Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).
  "Distributed Representations of Words and Phrases and their
  Compositionality." NeurIPS.
  [arXiv:1310.4546](https://arxiv.org/abs/1310.4546). Negative sampling and
  subsampling of frequent words, as implemented in `mini_w2v`.
- Mikolov, T., Yih, W., & Zweig, G. (2013). "Linguistic Regularities in
  Continuous Space Word Representations." NAACL. The source of
  king − man + woman ≈ queen.
- ★ Goldberg, Y., & Levy, O. (2014). "word2vec Explained: Deriving Mikolov et
  al.'s Negative-Sampling Word-Embedding Method."
  [arXiv:1402.3722](https://arxiv.org/abs/1402.3722). A short, clear
  derivation of the objective the original papers leave terse.
- Levy, O., & Goldberg, Y. (2014). "Neural Word Embedding as Implicit Matrix
  Factorization." NeurIPS. Shows skip-gram with negative sampling is
  implicitly factorizing a matrix of word co-occurrence statistics (shifted
  pointwise mutual information). Likely to appeal to a physicist.
- Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors
  for Word Representation." EMNLP. The main alternative, built directly from
  co-occurrence counts.
- Niu, F., Recht, B., Ré, C., & Wright, S. J. (2011). "Hogwild!: A Lock-Free
  Approach to Parallelizing Stochastic Gradient Descent." NeurIPS.
  [arXiv:1106.5730](https://arxiv.org/abs/1106.5730). Why lock-free parallel
  updates work, as used in `mini_w2v`.
- Nissim, M., van Noord, R., & van der Goot, R. (2020). "Fair Is Better than
  Sensational: Man Is to Doctor as Woman Is to Doctor." *Computational
  Linguistics* 46(2). [arXiv:1905.09866](https://arxiv.org/abs/1905.09866).
  The caveat on analogy arithmetic: the standard evaluation excludes the input
  words.

### Attention and the transformer (`attention/`, `transformer/`)

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by
  Jointly Learning to Align and Translate." ICLR.
  [arXiv:1409.0473](https://arxiv.org/abs/1409.0473). The origin of attention,
  added to an older recurrent translation model.
- ★ Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). The transformer:
  self-attention, attention heads, feed-forward layers, positional
  information.
- Peters, M. E., et al. (2018). "Deep Contextualized Word Representations."
  NAACL. [arXiv:1802.05365](https://arxiv.org/abs/1802.05365). ELMo: the step
  from word2vec's static vectors to vectors that depend on context.
- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018).
  "Improving Language Understanding by Generative Pre-Training." OpenAI
  technical report (GPT-1). Pretraining a transformer to predict the next
  token: the design `transformer/` follows.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT:
  Pre-training of Deep Bidirectional Transformers for Language Understanding."
  NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805). The other
  branch: predict hidden words from both sides, CBOW's descendant.
- Brown, T. B., et al. (2020). "Language Models are Few-Shot Learners."
  NeurIPS. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165). GPT-3: what
  the same design does at scale.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for
  Image Recognition." CVPR (Conference on Computer Vision and Pattern
  Recognition). [arXiv:1512.03385](https://arxiv.org/abs/1512.03385), and Ba,
  J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization."
  [arXiv:1607.06450](https://arxiv.org/abs/1607.06450). Two pieces every
  transformer block uses to make deep stacks trainable.
- Press, O., & Wolf, L. (2017). "Using the Output Embedding to Improve
  Language Models." EACL. [arXiv:1608.05859](https://arxiv.org/abs/1608.05859).
  Weight tying: using the embedding table as the output layer too.
- Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic
  Optimization." ICLR. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980). The
  optimizer that nearly every transformer is trained with.
- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). "The Curious
  Case of Neural Text Degeneration." ICLR.
  [arXiv:1904.09751](https://arxiv.org/abs/1904.09751). How to choose the next
  token in `generate`: temperature, top-k, and nucleus sampling.
- Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). "Transformer
  Feed-Forward Layers Are Key-Value Memories." EMNLP.
  [arXiv:2012.14913](https://arxiv.org/abs/2012.14913). What the feed-forward
  layers appear to be doing.
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer
  Circuits." Anthropic, transformer-circuits.pub. Reading a transformer as
  linear algebra; demanding but rewarding.

### Training data and small models

- Eldan, R., & Li, Y. (2023). "TinyStories: How Small Can Language Models Be
  and Still Speak Coherent English?"
  [arXiv:2305.07759](https://arxiv.org/abs/2305.07759). Models of a few million
  parameters, trained on simple stories, write coherent text. A strong
  candidate corpus for open decision 6.
- Biderman, S., et al. (2023). "Pythia: A Suite for Analyzing Large Language
  Models Across Training and Scaling." ICML.
  [arXiv:2304.01373](https://arxiv.org/abs/2304.01373). The models whose
  step-0 checkpoints are the precedent for the build stage.
- Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models."
  [arXiv:2001.08361](https://arxiv.org/abs/2001.08361), and Hoffmann, J., et
  al. (2022). "Training Compute-Optimal Large Language Models."
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556). How model size, data
  and compute trade off; useful for choosing toy-model sizes.

### Post-training (`transformer finetune`)

- Christiano, P., et al. (2017). "Deep Reinforcement Learning from Human
  Preferences." NeurIPS. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741).
  The origin of RLHF.
- ★ Ouyang, L., et al. (2022). "Training Language Models to Follow
  Instructions with Human Feedback." NeurIPS.
  [arXiv:2203.02155](https://arxiv.org/abs/2203.02155). InstructGPT: the full
  post-training recipe, supervised fine-tuning then RLHF.
- Bai, Y., et al. (2022). "Training a Helpful and Harmless Assistant with
  Reinforcement Learning from Human Feedback."
  [arXiv:2204.05862](https://arxiv.org/abs/2204.05862). Anthropic's account of
  RLHF for an assistant.
- Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language
  Model is Secretly a Reward Model." NeurIPS.
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290). DPO: preference tuning
  without a separate reward model.
- Zhou, C., et al. (2023). "LIMA: Less Is More for Alignment." NeurIPS.
  [arXiv:2305.11206](https://arxiv.org/abs/2305.11206). Fine-tuning on only
  1,000 carefully chosen conversations works well, encouraging for a small
  `chat.txt`.

### Not papers, but excellent for this project

- Karpathy, A. "Let's build the GPT Tokenizer" (2024) and "Let's build GPT:
  from scratch, in code, spelled out" (2023), video lectures on YouTube, and
  his nanoGPT code on GitHub. The closest existing match to this project.
- Alammar, J. "The Illustrated Transformer" (2018), a blog post. The standard
  visual introduction to attention.
