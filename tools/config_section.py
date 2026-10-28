#!/usr/bin/env python3
"""Copy one section of a config file to its own file, for make's dependencies.

Each step of the flow reads one section of config.toml: build reads [model],
train [training], finetune [finetune]. If the model files depended on the
whole config file, editing any section (even [generate], or a comment) would
retrain everything. Instead each depends on a file holding only its section,
written by this script:

    python3 tools/config_section.py config.toml finetune out/config/finetune.toml

The section is normalized (comments, blank lines and spacing outside quoted
strings removed), so only a change of value counts. The output is rewritten
only when that normalized text changes; otherwise its time stamp is left
alone, and make sees nothing new. A new output gets the config file's own
time stamp, so adding this to an existing setup doesn't make models built
from an unchanged config look out of date.
"""

import argparse
import os
import re
import sys

HEADER = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")


def normalize(line):
    """The line without its comment and without whitespace outside quotes."""
    out = []
    quote = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        elif not ch.isspace():
            out.append(ch)
    return "".join(out)


def section_text(text, section):
    """The normalized lines of [section], one per line ("" if it's absent)."""
    lines = []
    inside = False
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            inside = m.group(1).strip() == section
            continue
        if inside:
            norm = normalize(line)
            if norm:
                lines.append(norm)
    return "".join(f"{x}\n" for x in lines)


def main():
    """Write the section's file if it's new or its values changed."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("config", help="the config file, such as config.toml")
    ap.add_argument("section", help="the section's name, such as finetune")
    ap.add_argument("output", help="the file to write, such as out/config/finetune.toml")
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as f:
        want = section_text(f.read(), a.section)
    try:
        with open(a.output, encoding="utf-8") as f:
            have = f.read()
    except FileNotFoundError:
        have = None
    if have == want:
        return 0
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    tmp = a.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(want)
    if have is None:
        st = os.stat(a.config)
        os.utime(tmp, (st.st_atime, st.st_mtime))
    os.replace(tmp, a.output)
    if have is not None:
        print(f"[{a.section}] of {a.config} changed: {a.output} updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
