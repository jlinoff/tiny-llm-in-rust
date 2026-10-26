# chat

`make_chat.py` builds the fine-tuning data for step 5 of the LLM flow
(`../PLAN.md`, "Step 5: finetune"): conversations made from the corpus's own
stories.

A model that has read only children's stories can't answer general
questions, so its assistant is a storyteller. For each story, the script
picks three uncommon words that occur in it and writes a request around
them. The story itself is the reply:

```
<|user|>Write a story with the words dog, ball and park.<|assistant|>Once upon a time …<|end|>
```

The requests are honest, since each story really contains its three words.
That makes fine-tuning measurable: how often do the requested words appear
in the stories the model writes? (`transformer evaluate`, in
`../transformer/`.)

## Usage

```
python3 make_chat.py data/TinyStories-valid.txt -o out/chat.txt --eval out/chat-eval.tsv
```

Python 3.8 or later, standard library only. `make finetune` runs it (with
`CHAT_ARGS` for extra options), then encodes the result with
`mini_bpe.py --allow-special`.

Options:

- `-o FILE`: the training conversations (`chat.txt`), separated by
  `<|endoftext|>` lines, with the turn markers written as text
- `--eval FILE`: the evaluation requests (`chat-eval.tsv`), one per line:
  the three words (space-separated), a tab, the request
- `--split-on SEP`: the line between stories (default `<|endoftext|>`)
- `--eval-fraction F`: the last fraction of the stories, kept for evaluation
  (default 0.05). That's the same tail pretraining held out as validation
  data, approximately (pretraining holds out the last 5% of the tokens), so
  neither training step sees the evaluation stories.
- `--eval-count N`: evaluation requests to write (default 200)
- `--min-stories N`: a requested word must occur in at least N stories, so
  the model has seen it (default 20)
- `--skip-common N`: leave out the N words that occur in the most stories
  (default 300): function words and story staples such as "the", "she",
  "happy" and "day", which would appear anyway
- `--seed N`: random seed (default 1); the same seed gives the same files

## How the words are chosen

- **Lowercase words of three letters or more.** Capitalized words are mostly
  names (Lily, Tom), which a request shouldn't hinge on.
- **Known, but not everywhere.** At least `--min-stories` stories, and not
  among the `--skip-common` most widespread.
- **Three at random** from the story's eligible words, listed in the order
  they first appear in the story.
- **Several phrasings.** Five templates ("Write a story with the words …",
  "Can you write a story with the words … in it?", …), so the model learns
  to respond to a request rather than to one exact sentence. Every template
  lists the words, so any word fits grammatically, whether it's a noun, a
  verb or an adjective.

A story with fewer than three eligible words is skipped, and the count is
reported. If there are too few eligible words to make requests at all (a
small or unusual corpus), the script stops with an error: lower
`--skip-common` or `--min-stories`.

## Checks

The input is checked to be readable and both outputs writable before any
work, and no story may already contain a turn marker or the separator. After
writing, `chat.txt` is read back and compared with what was meant to be
written.
