#!/usr/bin/env python3
"""
acronyms.py — check that every acronym in a Markdown file is defined (or
points to a definition) where it's first used.

An acronym here is a word with two or more capital letters: BPE, KV, GFLOP,
DiT, LoRA, UTF-8. At its first use, it counts as defined when, on that line
or the next two (a definition can wrap), it is written next to a parenthesis,
the usual shapes of a definition, or in bold, as a Terms entry is:

    byte-pair encoding (BPE)          the expansion first
    BPE (byte-pair encoding)          the acronym first
    BPE library (the same interface)  a word or two, then the parenthesis
    GPT-style model (GPT: see below)  a pointer to the definition
    **token ID**, an integer          a Terms entry
    large language model, or LLM      the expansion, then "or"

A name with a number (GPT-2) counts as defined once its base (GPT) is.

A plural (LLMs) is checked as its singular (LLM). Code (``` blocks and
`inline` spans), link targets, the generated table of contents and the
bibliography (authors, titles and venues) are skipped.

Names that need no definition (GitHub, macOS, TinyStories, a chip's name)
go in an allowlist, one per line with an optional "# reason":

    python3 tools/acronyms.py PLAN.md README.md --known tools/acronyms-known.txt

Prints one line per acronym used undefined, and exits with status 1 if
there are any, so `make lint` and CI fail until each is defined or, if
it's a name, added to the list. It can tell whether a definition is
missing, not whether it's a good one.
"""
import argparse
import re
import sys

WORD = re.compile(
    r"(?<![\w/.-])([A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*(?:-\d+(?:\.\d+)?[A-Z]?)?)(?![\w/]|\.\w)")
STOP = re.compile(r"^#{1,6} (Bibliography|References)\b")
HEADING = re.compile(r"^#{1,6} ")


def load_known(path):
    """The allowlist: one name per line; text after # is a comment."""
    if not path:
        return set()
    with open(path, encoding="utf-8") as fh:
        return {line.split("#", 1)[0].strip() for line in fh} - {""}


def prose_lines(text):
    """(line number, text) for the lines to check: code blocks, the table of
    contents and everything from the bibliography on are left out, and
    inline code and link targets blanked."""
    code = toc = False
    for n, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("```"):
            code = not code
            continue
        if line.strip() == "<!-- toc -->":
            toc = True
        if line.strip() == "<!-- /toc -->":
            toc = False
            continue
        if code or toc:
            continue
        if STOP.match(line):
            return
        line = re.sub(r"`[^`]*`", " ", line)          # inline code
        line = re.sub(r"\]\([^)]*\)", "]", line)      # link targets
        line = re.sub(r"https?://\S+", " ", line)     # bare URLs
        yield n, line


def defined_on(text, word):
    """Whether `word` appears next to a parenthesis, or in bold, in `text`."""
    w = re.escape(word)
    return bool(
        re.search(rf"\(\s*\**{w}\b", text)                          # (BPE  (**BPE
        or re.search(rf"\b{w}s?\**(?:[-\s]+[\w.'\"-]+){{0,2}}\s*\(", text)  # BPE (  BPE library (
        or re.search(rf"\b{w}s?\**\s*\)", text)                     # … (… BPE)
        or re.search(rf"\*\*[^*]*\b{w}s?\b[^*]*\*\*", text)            # **… BPE …**
        or re.search(rf",\s+or\s+\**{w}\b", text)                    # language model, or LLM
    )


def check(path, known):
    """The acronyms first used in `path` without a definition: (line, word)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    seen, problems = set(), []
    lines = list(prose_lines(text))
    for k, (n, line) in enumerate(lines):
        window = " ".join(text for _, text in lines[k:k + 3])
        for m in WORD.finditer(line):
            word = m.group(1)
            base = word[:-1] if word.endswith("s") and word[:-1] in seen | known else word
            stem = re.sub(r"-[\d.]+[A-Z]?$", "", base)  # GPT-2 -> GPT
            if base in seen or base in known or word in known or stem in seen | known:
                continue
            if defined_on(window if not HEADING.match(line) else line, base):
                seen.add(base)
            elif HEADING.match(line):
                # A heading names a section; unless it defines the acronym,
                # the text below must, so judge the next use instead.
                continue
            else:
                seen.add(base)
                problems.append((n, base))
    return problems


def main():
    """Check each file; report and fail on any undefined acronym."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 2)[1])
    ap.add_argument("files", nargs="+", help="Markdown files to check")
    ap.add_argument("--known", help="allowlist of names that need no definition")
    args = ap.parse_args()
    known = load_known(args.known)
    total = 0
    for path in args.files:
        for n, word in check(path, known):
            print(f"{path}:{n}: {word} is used before it's defined "
                  "(define it here, or add it to the allowlist if it's a name)")
            total += 1
    if total:
        print(f"acronyms.py: {total} undefined", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
