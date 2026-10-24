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
- **layer**: one stage of computation that the vectors pass through. In a
  transformer, "layer" means one whole transformer block (defined below), as
  in "a 12-layer model"; the two parts inside a block are its **sublayers**.
  The embedding layer and the output layer are named for being the first and
  last stages; they are not blocks.
- **embedding table**: a V × d table with one row (one vector) per token. It is
  the model's first layer: a token ID selects its row.
- **weights**, **parameters**: the numbers inside the model that are adjusted
  as it learns. **Hyperparameters** are the settings we choose (more in
  [Configuration](#configuration)).
- **self-attention**: a sublayer in which each token's vector takes in
  information from the tokens before it, weighting each by a relevance score
  (a dot product of their vectors). An **attention head** is one independent
  copy of this computation, working on a slice of each token's vector; each
  attention sublayer, in every block, runs several side by side (more in
  [How heads fit in](#how-heads-fit-in)). Inside attention, each token's
  vector is turned into a **query** (what it looks for), a **key** (what it
  offers to be matched on) and a **value** (what it passes on); see
  [How attention works](#how-attention-works-queries-keys-and-values).
- **contextual embedding**: a token's vector after attention has mixed in
  information from its context, so the same word gets different vectors in
  different sentences. A row of the embedding table, by contrast, is the
  same wherever the token appears.
- **feed-forward network** (**FFN**; often loosely "feed-forward layer"): a
  sublayer that transforms each token's vector on its own, the same way at
  every position.
  "Feed-forward" means information passes straight through, with no loops.
- **residual connection**: adding a sublayer's input back to its output, so
  the sublayer computes a *change* to each vector rather than a replacement.
- **layer normalization**: rescaling each token's vector to a standard size
  (mean 0, variance 1, then a learned scale and shift), which keeps values
  stable as they pass through many layers.
- **transformer block**, or **layer**: one self-attention sublayer and one
  FFN sublayer, each with a residual connection and layer normalization. A
  **transformer** is a model built from a stack of these blocks (more in
  [The transformer block](#the-transformer-block)).
- **output layer** (or **unembedding**, or **LM head** for "language-model
  head"): the final layer, which turns a vector into V scores, one for each
  possible next token. This "head" is unrelated to attention heads.
- **softmax**: turns a list of scores z₁ … zₙ into positive weights that sum
  to 1: weightᵢ = e^zᵢ / (e^z₁ + … + e^zₙ). Higher scores get exponentially
  more weight. It's the Boltzmann distribution, with the scores playing the
  role of negative energies. It turns attention scores into weights, and the
  output layer's V scores into next-token probabilities.
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
- **temperature**: a text-generation setting T: the output scores are divided
  by T before the softmax, exactly as temperature enters the Boltzmann
  distribution. Low values make the most likely next token dominate; higher
  values flatten the probabilities, so sampling picks more varied tokens.
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
every block's attention and FFN sublayers, and the output layer, all randomly
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

## The transformer block

The model built in steps 3–6 is a stack of identical **transformer blocks**,
also called **layers** ("a 4-layer model" has 4 blocks), between the embedding
table and the output layer:

```
token IDs → embedding table (+ position information) → block 1 → block 2 → … → block N
          → layer normalization → output layer → V scores for the next token
```

Each block updates every token's vector in two stages, its two sublayers.
In the form used by GPT-2 and most modern LLMs, called "pre-norm":

```
h = x + attention(normalize(x))     # sublayer 1: tokens exchange information
y = h + FFN(normalize(h))           # sublayer 2: each token is processed on its own
```

The same thing as a picture, following one token's vector through one block:

```
                one token's vector (d numbers), from the block below
                                  │
╔═════════════════════════════════╪═════════════════════════════════╗
║ ONE TRANSFORMER BLOCK           │                                 ║
║ (= one layer)                   │                                 ║
║                ┌────────────────┤  the residual stream            ║
║                │                ▼                                 ║
║                │       layer normalization                        ║
║  sublayer 1    │                ▼                                 ║
║                │        self-attention  ◄──── also reads the      ║
║                │                │             earlier tokens'     ║
║                │                ▼             vectors             ║
║                └──────────────► + add the change back             ║
║                                 │                                 ║
║                ┌────────────────┤                                 ║
║                │                ▼                                 ║
║                │       layer normalization                        ║
║  sublayer 2    │                ▼                                 ║
║                │   feed-forward network (FFN)                     ║
║                │      this token only                             ║
║                │                ▼                                 ║
║                └──────────────► + add the change back             ║
║                                 │                                 ║
╚═════════════════════════════════╪═════════════════════════════════╝
                                  ▼
                       to the next block (layer)
```

Here x holds one d-dimensional vector per token, "normalize" is layer
normalization, and the `x +` and `h +` are the residual connections. (The
original 2017 transformer normalized *after* each addition instead; putting
normalization first trains more stably in deep stacks.)

#### The four parts

- **Self-attention: tokens exchange information.** The only place in the
  block where one token's vector is affected by the others. Each token's
  vector is turned, by three learned matrices, into a **query** ("what am I
  looking for?"), a **key** ("what do I contain?") and a **value** ("what do
  I pass on?"). Each token scores every earlier token by the dot product of
  its query with their keys, divided by √(head size) to keep the scores in a
  workable range. Softmax (see [Terms](#terms)) turns those scores into
  positive weights that sum to 1, and the token's output is the weighted sum of the values. The
  **causal mask** sets the scores for later tokens to −∞ before the softmax,
  so a token can never use the future it's being trained to predict:

  ```
  attention scores for a four-token input
  (row = the token looking, column = the token looked at)

              The    cat    sat     on
      The      ■      ·      ·      ·        ■ = allowed
      cat      ■      ■      ·      ·        · = masked: set to −∞, so its
      sat      ■      ■      ■      ·            softmax weight is exactly 0
      on       ■      ■      ■      ■

  each row → softmax → weights summing to 1 → weighted sum of those tokens' values
  ```

  Several **heads** do this side by side, each in a slice of d with its own
  matrices, so different heads can track different relationships; their
  outputs are joined and passed through one more learned matrix.
- **Feed-forward network (FFN): each token on its own.** Two learned matrix
  multiplications with a nonlinearity between them: d → d_ff (usually 4 × d) →
  nonlinearity (GELU, a smooth version of "keep positive values, zero the
  rest") → d. It never looks at other tokens. It holds two-thirds of a block's
  parameters (8d² against attention's 4d²), and studies such as Geva et al.
  (2021) find that much of a model's stored knowledge appears to live there.
- **Residual connections.** Each stage *adds* its result to the vectors it
  received. A token's vector becomes a running sum that every block reads and
  writes, often called the **residual stream**. This also gives the training
  signal a direct path back through a deep stack (He et al., 2016).
- **Layer normalization.** Applied to the copy of the vector entering each
  sublayer, it gives every sublayer inputs at a consistent scale however many
  blocks they have passed through (Ba et al., 2016).

#### How attention works: queries, keys and values

This expands the self-attention summary above, with a worked example small
enough to check by hand.

**How the matrices are constructed.** W_q, W_k and W_v are weights, exactly
like the embedding table: created with random values in step 3 (build),
fixed during a forward pass, where they're simply multiplied with the input,
and nudged a little by training after every batch. Nothing about "queries" or
"keys" is designed in; those are names for the roles the matrices end up
playing once trained. Each head has three matrices of size d × (head size);
with d = 128 and 4 heads, 128 × 32 each. In practice the heads' matrices are
stored side by side as one d × d matrix, hence the "4d²" (W_q, W_k, W_v and
the output matrix W_o) in [The four parts](#the-four-parts). Every block has
its own four matrices; they aren't shared between blocks.

**A worked example.** Three tokens, d = 4, one head of size 2. The rows of X
are the three tokens' vectors after layer normalization. The weights are
hand-picked small integers so the arithmetic is easy to follow; learned ones
are messier.

```
          X (3 tokens × d=4)       W_q (4×2)    W_k (4×2)    W_v (4×2)
the    [ 1  0  1  0 ]              [ 1  0 ]     [ 0  1 ]     [ 1  0 ]
cat    [ 0  1  0  1 ]              [ 0  1 ]     [ 1  0 ]     [ 0  1 ]
sat    [ 1  1  0  0 ]              [ 1  0 ]     [ 0  0 ]     [ 0  0 ]
                                   [ 0  0 ]     [ 1  1 ]     [ 0  0 ]
```

*Step 1: project every token three ways.* Ordinary matrix products, each
token's row times each matrix. The same input vector gives what the token
looks for (its query q), what it offers to be matched on (its key k), and
what it hands over (its value v):

```
          Q = X·W_q    K = X·W_k    V = X·W_v
the        [ 2  0 ]     [ 0  1 ]     [ 1  0 ]
cat        [ 0  1 ]     [ 2  1 ]     [ 0  1 ]
sat        [ 1  1 ]     [ 1  1 ]     [ 1  1 ]
```

*Step 2: score every query against every key.* scores = Q · Kᵀ: entry (i, j)
is the dot product of token i's query with token j's key, then divided by
√(head size) = √2:

```
                    key of:  the    cat    sat
query of  the               0.00   2.83   1.41
          cat               0.71   0.71   0.71
          sat               0.71   2.12   1.41
```

*Step 3: apply the causal mask.* A token may look only at itself and earlier
tokens, so everything above the diagonal becomes −∞:

```
          the    cat    sat
the       0.00    −∞     −∞
cat       0.71   0.71    −∞
sat       0.71   2.12   1.41
```

*Step 4: softmax each row* into weights that sum to 1; the −∞ entries become
exactly 0:

```
          the    cat    sat
the       1.00   0      0        "the" can only see itself
cat       0.50   0.50   0        equal scores give equal weights
sat       0.14   0.58   0.28     "sat" attends mostly to "cat"
```

*Step 5: each token's output is the weighted sum of the values:*

```
sat:  0.14 × [1 0]  +  0.58 × [0 1]  +  0.28 × [1 1]  =  [0.42  0.86]
      (the's value)    (cat's value)     (sat's value)
```

That's the whole mechanism: three matrix multiplications, one more for the
scores, a mask, a softmax and a weighted sum. Everything except the softmax
is ordinary linear algebra.

**Who does what.**

```
Q · K  →  where to look   (the weights: how much each earlier token matters)
V      →  what you get    (the information actually passed along)
```

- **Q and K decide the weights; V doesn't.** The weights come from pairing
  the current token's query with the key of itself and each earlier token. V
  plays no part in deciding where to look; it's what gets delivered once the
  weights are fixed. Elhage et al. (2021) call these the "QK circuit" and the
  "OV circuit" (V with the output matrix W_o) and analyze them separately: a
  head can find tokens by one property and copy a different one from them.
- **Each row is one token looking back.** The weights are the attention the
  *current* token pays to each earlier token (and itself): "sat"'s row,
  0.14, 0.58, 0.28, is sat's attention budget, always summing to 1.
- **Triples and pairs.** Each token has three vectors, a triple (q, k, v).
  The *pairs* are (query of the token looking, key of the token looked at),
  one per cell of the score triangle.

**How a token is updated.** A token isn't updated by its own value vector,
but by the weighted sum of the value vectors of the tokens it attends to
(possibly including itself); its own value is what it offers to others. For
"sat", the update is [0.42 0.86], drawn mostly from "cat". Then:

```
join all heads' updates → multiply by W_o → ADD to the token's vector in the residual stream
```

The addition is the residual connection: "sat" keeps everything it had and
gains information from its context. In short:

- **Q · K:** whom each token listens to
- **V:** what each token says when listened to
- **the weighted sum, then W_o, added to the stream:** how each token's
  vector changes as a result

This is exactly the step word2vec lacks. A word2vec vector for "sat" is the
same in every sentence; after attention, "sat" in "the cat sat" carries
information about the cat, and after a few more layers it can encode who
sat, where and when. That is what a **contextual embedding** is.

**Why three separate matrices.** Expand a score:

```
score(i, j) = q_i · k_j = (x_i W_q) · (x_j W_k) = x_i · (W_q W_kᵀ) x_j
```

The score is a **learned bilinear form** xᵢᵀ M xⱼ, with M = W_q W_kᵀ a d × d
matrix of rank at most the head size: each head learns its own definition of
"relevant". Using the raw vectors instead, xᵢ · xⱼ, would measure only
similarity, word2vec-style, and would be symmetric. With a learned M:

- **relevance can differ from similarity:** a verb can attend to its subject
  although their vectors aren't alike, because M maps verb-like queries onto
  subject-like keys;
- **relevance can be asymmetric:** in the example, "the" scores "cat" at
  2.83, but "cat" scores "the" at only 0.71.

W_v is separate from both because *what you match on* and *what you pass
along* are different questions: a token can be found by its grammatical role
while contributing its meaning.

**How they become useful.** Training never sees "queries" or "keys", only
the final next-token loss. The gradient flows back through the weighted sum,
the softmax and the dot products into all three matrices, and they drift
toward whatever makes predictions better. The names come from database
lookup, and attention is a **soft lookup**: a dictionary `d[key]` returns
the value whose key matches *exactly*; attention returns a *blend* of every
value, weighted by how well each key matches the query.

The `attention/` program will print exactly these tables (Q, K, V, scores,
mask, weights, outputs) for small inputs, first with hand-set weights like
these and later with trained ones.

#### How heads fit in

A head is a part of **one attention sublayer, inside one block**, not a pass
through the model. With the starting sizes from
[Configuration](#configuration) (d = 128, 4 layers, 4 heads):

```
model
└── 4 transformer blocks (layers), applied one after another
    └── each block
        ├── attention sublayer
        │   ├── head 1 ─┐  each head: its own query, key and value matrices,
        │   ├── head 2  │  working on a 32-number slice (128 / 4) of every
        │   ├── head 3  │  token's vector, attending over ALL the tokens
        │   └── head 4 ─┘
        │        → the 4 heads' outputs are joined (4 × 32 = 128 numbers)
        │          and passed through one more learned matrix
        └── FFN sublayer
```

That model has 4 × 4 = 16 heads in all, four in each block's attention
sublayer. Each head is a complete, independent attention computation: its own
queries, keys and values, scores, causal mask and softmax.

**Why several heads:** each can learn to track a different relationship. In
trained models, researchers have found heads that consistently attend to the
previous token, heads that link a pronoun to its noun, and "induction heads"
(Elhage et al., 2021; studied in depth by Olsson et al., 2022) that spot a repeated pattern and predict its continuation. A single head per
block could learn only one weighting at a time.

**Every block processes all the tokens, in parallel.** The blocks aren't
applied to one token at a time: each block updates every token's vector.
What's per-token is the FFN, which handles each vector separately; attention,
including every head, is exactly the part that looks across tokens.

**Two meanings of "head".** The output layer is often called the **LM head**
(language-model head), meaning the final layer of the whole model, which
turns vectors into token scores. It has nothing to do with attention heads.

#### What runs in parallel

- **Heads within an attention sublayer: parallel.** They're independent until
  their outputs are joined. In practice they aren't even separate
  computations: one matrix multiplication produces the queries for all heads,
  and "splitting into heads" just views the result as slices.
- **Blocks: sequential.** Each block needs the previous block's output, so
  depth costs time in a way width doesn't.
- **Tokens during training: parallel.** Every position in a sequence is
  processed at once, predicting every next token simultaneously; the causal
  mask is what makes this legitimate, since no position can see a later one.
  This parallelism is the main reason transformers replaced recurrent
  networks, which had to read text one token at a time.
- **Tokens during generation: sequential.** A token can't be generated until
  the one before it exists, so generation makes one pass through the model per
  token. The **KV cache** (key-value cache) keeps each earlier token's keys and
  values, so each pass computes only the newest token.

All of these are optimizations: in exact arithmetic, computing the heads
together or one at a time, or the tokens at once or one by one, gives the same
result. In floating-point arithmetic the results are equal only to rounding
(see [Conventions for all code](#conventions-for-all-code)). Our code may
simply loop over the heads, which is clearer and fast enough at toy sizes.

#### Why repeated normalization doesn't lose information

With 2N normalizations in an N-layer model, it might seem that information is
whittled away. It isn't, for three reasons:

1. **The residual stream itself is never normalized.** In `h = x +
   attention(normalize(x))`, normalization produces only the sublayer's
   *input*; `x` itself flows past untouched to the `+`, where the sublayer's
   output is added. In the diagram, the normalization boxes sit on the side
   branches, never on the main vertical line. The stream carries its full,
   unnormalized information from the embedding to the top of the stack, each
   block only adding to it, and it is normalized just once, at the very end,
   before the output layer.
2. **What normalizing a copy discards is deliberately little.** Layer
   normalization subtracts the vector's mean and divides by its standard
   deviation: it removes 2 of the d numbers' worth of information, the
   vector's overall offset and overall scale. Its *direction*, the other
   d − 2 degrees of freedom, is untouched, and that's where the meaning is.
   (word2vec showed the same: cosine similarity, which ignores length, was the
   right measure of relatedness.) Two learned parameters per dimension, a
   scale and a shift, are then applied, so a sublayer that needs a particular
   scale can learn it.
3. **It improves numerical precision.** Keeping sublayer inputs near
   magnitude 1 is where floating-point numbers are most precise, and keeps
   them away from overflow (too large to represent) and underflow (too small),
   which matters even more when models compute with 16-bit numbers.

Why normalize at all: without it, the sublayers' input magnitudes would drift
as blocks keep adding to the stream, and each sublayer would have to cope with
inputs ranging from tiny to huge. Normalizing the copy means every sublayer
always sees the same scale, which keeps training stable.

The **post-norm** design of the original 2017 transformer *did* normalize the
stream itself after every sublayer, repeatedly discarding its scale. It
worked, but deep stacks were harder to train, which is one reason pre-norm
took over. Many recent models (Llama, for example) use **RMSNorm** (root mean
square normalization), which only divides by the vector's scale without
subtracting the mean, discarding even less.

#### Position information

Attention by itself ignores order: shuffling the
earlier tokens wouldn't change a token's weighted sum. So the model is told
each token's position. GPT-2 adds a learned **position embedding** (one
vector per position, up to the context length) to each token's embedding;
many newer models instead rotate the queries and keys by an angle that
depends on position (RoPE, rotary position embedding). Which we use is a
decision for `transformer/`.

#### Where each part is built

- **`attention/`**: self-attention alone, on toy inputs, small enough to print
  every number: one head first (queries, keys, values, the scores, the
  softmax weights, the causal mask), then several heads. It ends with a look
  ahead at one complete block, adding the FFN, residual connections and
  layer normalization, so they appear before `transformer/` without being
  the focus.
- **`transformer/`**: the real model. Step 3 builds the embedding table,
  position information, N blocks and the output layer; step 4 trains them
  all together.

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
| d_ff | the inner width of each FFN sublayer, conventionally 4 × d |
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
- **Exact comparisons for integers, tolerances for floating point.** Token
  IDs, merges and other integer data must match exactly, as the tokenizer's
  tests require. Floating-point results must be compared with a tolerance
  (for example, a relative difference below 10⁻⁵ for 32-bit floats), because
  floating-point addition isn't associative: (a + b) + c and a + (b + c) can
  differ in the last bit, since each addition rounds. Two computations that
  are identical in exact arithmetic, such as a looped and a batched dot
  product, add in different orders and agree only to about 6 or 7 significant
  digits. A consequence for reproducibility: if the order of additions varies
  between runs (as with some parallel or GPU operations), two runs with the
  same seed can drift apart, and over thousands of training steps the
  differences can grow; keeping the order fixed avoids that. None of this is
  a flaw in the results: models train in 16-bit floats (about 3 significant
  digits) and run in 8- or 4-bit formats, because the learned function is
  robust to small perturbations and training is already noisy.
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
├── attention/     # self-attention on its own, then a look at one full block
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

**Why it matters for self-attention and feed-forward networks.** word2vec isn't
an ancestor of either mechanism; it's the foundation they build on and the
limitation they were invented to overcome:

- **The representation.** It established that a word can be a dense learned
  vector whose dot product with another measures relatedness. Self-attention
  is built from exactly that operation: each token scores every other token by
  a dot product, then mixes their vectors according to those scores.
- **The limitation.** One fixed vector per word can't tell "river bank" from
  "bank loan". Self-attention exists to fix this: it rebuilds each token's
  vector from its context, block after block, and the FFN sublayers
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
3. `attention/`: self-attention alone, on toy inputs, ending with one
   complete block (see [The transformer block](#the-transformer-block)).
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
  self-attention, attention heads, feed-forward networks, positional
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
  linear algebra; demanding but rewarding. Introduces the residual stream
  view and induction heads.
- Olsson, C., et al. (2022). "In-context Learning and Induction Heads."
  Anthropic, transformer-circuits.pub.
  [arXiv:2209.11895](https://arxiv.org/abs/2209.11895). How heads that copy
  repeated patterns form during training.

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
