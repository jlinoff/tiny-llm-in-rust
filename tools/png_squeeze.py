#!/usr/bin/env python3
"""Recompress PNG files in place, losslessly, before they go into docs/.

The transformer writes PNGs with stored (uncompressed) zlib blocks: simple
to write and to check, but a grid of digits comes out about 600 KB. This
rewrites each file's image data compressed at zlib's best level, keeping
every other chunk and every pixel as it was: about 35 KB.

    python3 tools/png_squeeze.py docs/drawing/pixels-w1.png [more.png ...]
"""

import struct
import sys
import zlib

SIGNATURE = b"\x89PNG\r\n\x1a\n"


def chunks(data):
    """The (type, body) of each chunk in a PNG file's bytes."""
    if not data.startswith(SIGNATURE):
        raise ValueError("not a PNG file")
    i, out = len(SIGNATURE), []
    while i < len(data):
        (n,) = struct.unpack(">I", data[i:i + 4])
        out.append((data[i + 4:i + 8], data[i + 8:i + 8 + n]))
        i += 12 + n
    return out


def chunk(kind, body):
    """One chunk: length, type, body, CRC."""
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def squeeze(data):
    """The same PNG with its image data recompressed (all IDAT chunks
    joined into one)."""
    parts = chunks(data)
    pixels = zlib.decompress(b"".join(b for k, b in parts if k == b"IDAT"))
    out, done = [SIGNATURE], False
    for kind, body in parts:
        if kind == b"IDAT":
            if not done:
                out.append(chunk(b"IDAT", zlib.compress(pixels, 9)))
                done = True
        else:
            out.append(chunk(kind, body))
    return b"".join(out)


def pixels_of(data):
    """The decompressed image data, for checking nothing changed."""
    return zlib.decompress(b"".join(b for k, b in chunks(data) if k == b"IDAT"))


def main(paths):
    """Squeeze each file, checking its pixels survive."""
    for path in paths:
        with open(path, "rb") as f:
            before = f.read()
        after = squeeze(before)
        if pixels_of(after) != pixels_of(before):
            sys.exit(f"{path}: the pixels changed; left as it was")
        with open(path, "wb") as f:
            f.write(after)
        print(f"{path}: {len(before):,} bytes to {len(after):,}")


if __name__ == "__main__":
    main(sys.argv[1:])
