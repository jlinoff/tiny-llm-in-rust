#!/usr/bin/env python3
"""
toc.py — keep a Markdown file's table of contents up to date, and check its
internal links.

The table of contents goes between two marker lines in the file:

    <!-- toc -->
    <!-- /toc -->

Everything between them is regenerated from the file's headings (## to
#####; the # title is left out, and lines inside ``` code blocks are not
headings). Each entry links to its heading's anchor, made the way GitHub
makes them: lowercase, punctuation removed, spaces turned into hyphens, and
-1, -2, … added to repeats.

It also checks every internal link, [text](#anchor), against the headings,
so a renamed heading can't silently break a link elsewhere in the file.

Usage:
  python3 tools/toc.py PLAN.md            # update the table of contents, check links
  python3 tools/toc.py --check PLAN.md    # change nothing; fail if out of date or broken
"""
import argparse
import re
import sys

START, END = "<!-- toc -->", "<!-- /toc -->"
HEADING = re.compile(r"^(#{2,5}) (.+?)\s*$")
LINK = re.compile(r"\]\(#([^)\s]+)\)")


def slug(text):
    """GitHub's anchor for a heading: lowercase; drop everything but letters,
    digits, spaces, hyphens and underscores (Markdown code and link syntax
    first reduced to its text); each space becomes a hyphen."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](url) -> text
    text = text.replace("`", "").lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def headings(lines):
    """(level, text, anchor) for each heading outside code blocks."""
    out, seen, fenced = [], {}, False
    for line in lines:
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        m = None if fenced else HEADING.match(line)
        if m:
            base = slug(m.group(2))
            n = seen.get(base, 0)
            seen[base] = n + 1
            out.append((len(m.group(1)), m.group(2), base if n == 0 else f"{base}-{n}"))
    return out


def toc_lines(heads):
    """The table of contents: a nested list, indented by depth in the
    outline (how many enclosing headings), so a #### directly under a ##
    is indented one step, not two."""
    out, stack = [], []
    for level, text, anchor in heads:
        while stack and stack[-1] >= level:
            stack.pop()
        out.append(f"{'  ' * len(stack)}- [{text.replace('`', '')}](#{anchor})")
        stack.append(level)
    return out


def main():
    """Update or check the table of contents and the internal links."""
    ap = argparse.ArgumentParser(description="Update a Markdown table of contents "
                                             "and check internal links.")
    ap.add_argument("file", help="the Markdown file (PLAN.md)")
    ap.add_argument("--check", action="store_true",
                    help="change nothing; exit 1 if the table of contents is "
                         "out of date or a link is broken")
    args = ap.parse_args()
    with open(args.file, encoding="utf-8") as fh:
        text = fh.read()
    lines = text.split("\n")
    if lines.count(START) != 1 or lines.count(END) != 1 \
            or lines.index(START) > lines.index(END):
        sys.exit(f"toc.py: {args.file} needs one '{START}' line, then one '{END}' line")
    a, b = lines.index(START), lines.index(END)
    heads = headings(lines)
    new = lines[:a + 1] + toc_lines(heads) + lines[b:]

    anchors = {h[2] for h in heads}
    broken = sorted({m.group(1) for m in LINK.finditer(text)} - anchors)
    for anchor in broken:
        print(f"toc.py: {args.file}: broken link to #{anchor}", file=sys.stderr)

    stale = new != lines
    if args.check:
        if stale:
            print(f"toc.py: {args.file}: table of contents is out of date "
                  f"(run without --check to update it)", file=sys.stderr)
        sys.exit(1 if stale or broken else 0)
    if stale:
        with open(args.file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(new))
    print(f"{args.file}: {len(heads)} headings, table of contents "
          f"{'updated' if stale else 'already up to date'}; "
          f"{len(set(LINK.findall(text)))} internal link targets, {len(broken)} broken")
    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()
