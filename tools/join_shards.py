#!/usr/bin/env python3
"""
join_shards.py — merge a downloaded model's weight shards into the single
model.safetensors file that `transformer llama` reads (PLAN.md, Trying a new
model). Large models are published in shards (model-00001-of-00002.safetensors,
…) with an index, model.safetensors.index.json, naming every tensor's shard.

    cd hugging_face && pipenv run python3 ../tools/join_shards.py models/SmolLM-1.7B

(or `make download-big` in hugging_face/, which downloads and merges).

Checks before writing: at least one shard, no tensor in two shards, and the
merged tensors exactly those the index lists. Writes to a temporary file and
renames it, so an interrupted run never leaves a partial model.safetensors
for make to mistake for a finished one.

Needs the safetensors and torch packages (in hugging_face/'s pipenv); the
checks themselves are plain Python, tested without them
(tools/test_join_shards.py).
"""
import glob
import json
import os
import sys


def find_shards(folder):
    """The shard files in `folder`, in order."""
    return sorted(glob.glob(os.path.join(folder, "model-*-of-*.safetensors")))


def merge(parts, listed=None):
    """Merge the tensors of each shard (a list of name -> tensor dicts) into
    one dict, refusing a name that appears twice. If `listed` (the names the
    index gives) is set, the merged names must be exactly those. Returns the
    merged dict; raises ValueError on any problem."""
    if not parts:
        raise ValueError("no shards (model-*-of-*.safetensors)")
    merged = {}
    for part in parts:
        for name, value in part.items():
            if name in merged:
                raise ValueError(f"{name} is in more than one shard")
            merged[name] = value
    if listed is not None and set(listed) != set(merged):
        missing = sorted(set(listed) - set(merged))
        extra = sorted(set(merged) - set(listed))
        raise ValueError(f"the shards don't match the index: missing {missing[:5]}"
                         f"{' …' if len(missing) > 5 else ''}, extra {extra[:5]}"
                         f"{' …' if len(extra) > 5 else ''}")
    return merged


def read_index(folder):
    """The tensor names model.safetensors.index.json lists, or None if the
    folder has no index."""
    path = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return list(json.load(fh)["weight_map"])


def main():
    """Merge the shards in the folder given, after checking them."""
    if len(sys.argv) != 2:
        sys.exit("usage: join_shards.py MODEL_DIR")
    folder = sys.argv[1]
    from safetensors.torch import load_file, save_file  # pylint: disable=import-outside-toplevel
    shards = find_shards(folder)
    try:
        merged = merge([load_file(p) for p in shards], read_index(folder))
    except ValueError as e:
        sys.exit(f"join_shards.py: {folder}: {e}")
    out = os.path.join(folder, "model.safetensors")
    save_file(merged, out + ".tmp", metadata={"format": "pt"})
    os.replace(out + ".tmp", out)
    print(f"join_shards.py: {len(shards)} shards, {len(merged)} tensors -> {out}")
    print("The shard files can now be deleted.")


if __name__ == "__main__":
    main()
