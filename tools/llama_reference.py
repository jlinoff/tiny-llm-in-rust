#!/usr/bin/env python3
"""Reference numbers from Hugging Face's own code, for `transformer llama --check`.

Runs a Llama-style model with the `transformers` library (which needs PyTorch)
and writes what our implementation must reproduce (PLAN.md, Order of work,
the endpoint):

- token IDs for a set of tricky strings (digits, whitespace, Unicode,
  contractions, special tokens), from the model's own tokenizer;
- for a few prompts: the log-probability of each actual next token at every
  position, the full scores (logits) at the last position, and a 40-token
  greedy continuation.

    python3 tools/llama_reference.py MODEL_DIR -o reference.json
    transformer llama --model-dir MODEL_DIR --check reference.json

Needs `pip install transformers torch` (in a virtual environment).
"""

import argparse
import json
import sys

TOKENIZE = [
    "Once upon a time, there was a little girl named Lily.",
    "Numbers: 3.14159, 2026-10-28, 1,000,000 and 42nd.",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\n\n\nthree of them",
    "Contractions: I'm, you're, they've, it's, we'll, he'd, don't.",
    "Unicode: café, naïve, Zürich, 東京, Ελληνικά, emoji 🙂🚀.",
    "Code: fn main() { println!(\"{}\", x[i] + 1); } // ok",
    "Special tokens: <|endoftext|> and <|im_start|>user<|im_end|>",
    "Fullwidth and other digits: ４２ ٣ ½ Ⅻ",
    "",
    "a",
    "!!!???...,,,",
]

PROMPTS = [
    "Once upon a time there was an itty bitty llm on m4 macbook pro that "
    "was trying to run inference like the big boys",
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Tom and Sue went to the park. Tom saw a red ball. Tom and",
]


def main():
    """Run the model on the test texts and prompts, and write the JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("model_dir",
                    help="the model's folder (config.json, model.safetensors, tokenizer.json)")
    ap.add_argument("-o", "--output", required=True, help="the JSON file to write")
    ap.add_argument("--continuation", type=int, default=40, help="greedy tokens per prompt")
    a = ap.parse_args()

    import torch  # pylint: disable=import-outside-toplevel
    from transformers import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=import-outside-toplevel

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32)
    model.eval()
    out = {"model_dir": a.model_dir, "tokenize": [], "prompts": []}
    for text in TOKENIZE:
        out["tokenize"].append({"text": text, "ids": tok(text, add_special_tokens=False).input_ids})
    with torch.no_grad():
        for text in PROMPTS:
            ids = tok(text, add_special_tokens=False).input_ids
            logits = model(torch.tensor([ids])).logits[0]
            logp = torch.log_softmax(logits, dim=-1)
            next_logp = [logp[t, ids[t + 1]].item() for t in range(len(ids) - 1)]
            gen = model.generate(torch.tensor([ids]), max_new_tokens=a.continuation,
                                 do_sample=False)[0, len(ids):].tolist()
            out["prompts"].append({
                "text": text,
                "ids": ids,
                "next_logprob": next_logp,
                "last_logits": logits[-1].tolist(),
                "greedy": gen,
            })
            print(f"{len(ids):3} tokens, greedy: {tok.decode(gen)[:60]!r}", file=sys.stderr)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"wrote {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
