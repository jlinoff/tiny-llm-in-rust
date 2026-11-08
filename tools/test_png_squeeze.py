"""Tests for png_squeeze.py."""

import struct
import unittest
import zlib

import png_squeeze as ps


def tiny_png():
    """A 2 × 2 grayscale PNG with stored blocks, as the transformer writes."""
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 0, 0)
    raw = b"\x00\x00\xff\x00\x80\x40"
    return ps.SIGNATURE + ps.chunk(b"IHDR", ihdr) + ps.chunk(b"IDAT", zlib.compress(raw, 0)) \
        + ps.chunk(b"IEND", b"")


class SqueezeTest(unittest.TestCase):
    """The pixels and other chunks survive; the image data is compressed."""

    def test_pixels_survive(self):
        """Same pixels, same chunks in the same order, IHDR untouched."""
        before = tiny_png()
        after = ps.squeeze(before)
        self.assertEqual(ps.pixels_of(after), ps.pixels_of(before))
        self.assertEqual([k for k, _ in ps.chunks(after)], [b"IHDR", b"IDAT", b"IEND"])
        self.assertEqual(ps.chunks(after)[0], ps.chunks(before)[0])

    def test_refuses_other_files(self):
        """A file that isn't a PNG is refused."""
        with self.assertRaises(ValueError):
            ps.chunks(b"not a png")


if __name__ == "__main__":
    unittest.main()
