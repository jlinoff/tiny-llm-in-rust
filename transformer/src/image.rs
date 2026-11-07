//! Copied from tiny-diffusion-in-rust (diffusion/src/image.rs) for drawing
//! (PLAN.md, Drawing), keeping only what drawing uses: image sets, MNIST's
//! IDX files, cutting images into patches and back, and grids of images as
//! PNG files.
//!
//! An image set holds `n` images of height × width pixels with `channels`
//! values each (1 for grayscale, 3 for color), as an n × (h·w·c) matrix:
//! one row per image, pixels in rows from the top, each pixel's channels
//! together. Values run from −1 (black) to 1 (white). Nothing here assumes
//! 28 × 28.

// Copied whole: some of it is used by later parts of drawing (patches by
// the codebook, `shapes` by tests and fixtures) before the commands do.
#![allow(dead_code)]

use crate::rng::Rng;
use crate::tensor::Mat;
use std::fs;
use std::io::Write;
use std::path::Path;

/// The shape of every image in a set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

impl Shape {
    pub fn values(&self) -> usize {
        self.height * self.width * self.channels
    }
}

/// A set of images, with the class of each (a digit, a kind of clothing, a
/// kind of shape).
pub struct Images {
    pub shape: Shape,
    pub pixels: Mat<f32>,
    pub labels: Vec<u8>,
}

impl Images {
    /// The images at rows `rows`, as a batch.
    pub fn batch(&self, rows: &[usize]) -> Mat<f32> {
        let v = self.shape.values();
        let mut data = Vec::with_capacity(rows.len() * v);
        for &r in rows {
            data.extend_from_slice(self.pixels.row(r));
        }
        Mat::from_vec(rows.len(), v, data)
    }
}

// ----- patches

/// The number of patches (tokens) an image of this shape makes.
pub fn patch_count(shape: Shape, patch: usize) -> usize {
    (shape.height / patch) * (shape.width / patch)
}

pub fn check_patch(shape: Shape, patch: usize) -> Result<(), String> {
    if patch == 0 || !shape.height.is_multiple_of(patch) || !shape.width.is_multiple_of(patch) {
        return Err(format!(
            "patch size {patch} doesn't divide the {}×{} images",
            shape.height, shape.width
        ));
    }
    Ok(())
}

/// Images (n × h·w·c) to patches: one row per patch, (n·T) × (p·p·c), the
/// patches of image 0 first, each image's patches in rows from the top
/// left, each patch's pixels in rows, each pixel's channels together.
pub fn patchify(images: &Mat<f32>, shape: Shape, patch: usize) -> Mat<f32> {
    let (h, w, c, p) = (shape.height, shape.width, shape.channels, patch);
    let (ph, pw) = (h / p, w / p);
    let per = p * p * c;
    let mut out = Mat::zeros(images.rows * ph * pw, per);
    for n in 0..images.rows {
        let img = images.row(n);
        for by in 0..ph {
            for bx in 0..pw {
                let row = out.row_mut((n * ph + by) * pw + bx);
                for y in 0..p {
                    for x in 0..p {
                        let src = ((by * p + y) * w + (bx * p + x)) * c;
                        let dst = (y * p + x) * c;
                        row[dst..dst + c].copy_from_slice(&img[src..src + c]);
                    }
                }
            }
        }
    }
    out
}

/// The inverse of `patchify`.
pub fn unpatchify(patches: &Mat<f32>, shape: Shape, patch: usize) -> Mat<f32> {
    let (h, w, c, p) = (shape.height, shape.width, shape.channels, patch);
    let (ph, pw) = (h / p, w / p);
    let n = patches.rows / (ph * pw);
    let mut out = Mat::zeros(n, shape.values());
    for i in 0..n {
        for by in 0..ph {
            for bx in 0..pw {
                let src_row = patches.row((i * ph + by) * pw + bx).to_vec();
                let img = out.row_mut(i);
                for y in 0..p {
                    for x in 0..p {
                        let dst = ((by * p + y) * w + (bx * p + x)) * c;
                        let src = (y * p + x) * c;
                        img[dst..dst + c].copy_from_slice(&src_row[src..src + c]);
                    }
                }
            }
        }
    }
    out
}

// ----- data sets

/// MNIST or Fashion-MNIST, from the decompressed IDX files the Makefile
/// downloads (`*-images-idx3-ubyte`, `*-labels-idx1-ubyte`). IDX: a 4-byte
/// magic number (0x0803 for images, 0x0801 for labels), then each dimension
/// as a big-endian 32-bit count, then the bytes, one per pixel or label.
pub fn read_idx(images: &Path, labels: &Path) -> Result<Images, String> {
    let read = |p: &Path| fs::read(p).map_err(|e| format!("can't read {}: {e}", p.display()));
    let (img, lab) = (read(images)?, read(labels)?);
    let be = |b: &[u8], i: usize| -> usize {
        u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]) as usize
    };
    if img.len() < 16 || be(&img, 0) != 0x0803 {
        return Err(format!("{} isn't an IDX image file", images.display()));
    }
    if lab.len() < 8 || be(&lab, 0) != 0x0801 {
        return Err(format!("{} isn't an IDX label file", labels.display()));
    }
    let (n, h, w) = (be(&img, 1), be(&img, 2), be(&img, 3));
    if img.len() != 16 + n * h * w || lab.len() != 8 + n || be(&lab, 1) != n {
        return Err(format!(
            "{} and {} don't hold {n} images and labels",
            images.display(),
            labels.display()
        ));
    }
    let pixels = img[16..].iter().map(|&b| b as f32 / 127.5 - 1.0).collect();
    Ok(Images {
        shape: Shape {
            height: h,
            width: w,
            channels: 1,
        },
        pixels: Mat::from_vec(n, h * w, pixels),
        labels: lab[8..].to_vec(),
    })
}

/// Write images and labels as IDX files (for tests, and anyone wanting a
/// small data set in the same format).
pub fn write_idx(set: &Images, images: &Path, labels: &Path) -> Result<(), String> {
    assert_eq!(set.shape.channels, 1, "IDX images are grayscale");
    let n = set.pixels.rows as u32;
    let mut img = Vec::new();
    for v in [0x0803, n, set.shape.height as u32, set.shape.width as u32] {
        img.extend_from_slice(&v.to_be_bytes());
    }
    img.extend(set.pixels.data.iter().map(|&x| to_byte(x)));
    let mut lab = Vec::new();
    for v in [0x0801, n] {
        lab.extend_from_slice(&v.to_be_bytes());
    }
    lab.extend_from_slice(&set.labels);
    fs::write(images, img).map_err(|e| format!("can't write {}: {e}", images.display()))?;
    fs::write(labels, lab).map_err(|e| format!("can't write {}: {e}", labels.display()))
}

/// A synthetic set, for any size and either channel count, so larger images
/// can be tried with no download: on a dark background, one shape of a
/// random kind, size and position, a filled square (label 0), a disc (1) or
/// a cross (2), white, or in color a random bright color.
pub fn shapes(n: usize, size: usize, channels: usize, seed: u64) -> Images {
    let shape = Shape {
        height: size,
        width: size,
        channels,
    };
    let mut rng = Rng::new(seed);
    let mut pixels = Vec::with_capacity(n * shape.values());
    let mut labels = Vec::with_capacity(n);
    let s = size as f64;
    for _ in 0..n {
        let kind = rng.below(3) as u8;
        let r = s * (0.15 + 0.15 * rng.unit()); // half-size
        let cx = r + (s - 2.0 * r) * rng.unit();
        let cy = r + (s - 2.0 * r) * rng.unit();
        let color: Vec<f32> = (0..channels)
            .map(|_| {
                if channels == 1 {
                    1.0
                } else {
                    (0.2 + 0.8 * rng.unit()) as f32 * 2.0 - 1.0
                }
            })
            .collect();
        for y in 0..size {
            for x in 0..size {
                let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                let inside = match kind {
                    0 => dx.abs() <= r && dy.abs() <= r,
                    1 => dx * dx + dy * dy <= r * r,
                    _ => {
                        (dx.abs() <= r && dy.abs() <= r / 3.0)
                            || (dy.abs() <= r && dx.abs() <= r / 3.0)
                    }
                };
                for &v in &color {
                    pixels.push(if inside { v } else { -1.0 });
                }
            }
        }
        labels.push(kind);
    }
    Images {
        shape,
        pixels: Mat::from_vec(n, shape.values(), pixels),
        labels,
    }
}

// ----- PNG

/// CRC-32 (the cyclic redundancy check PNG uses for each chunk; polynomial
/// 0xEDB88320, reflected), bit by bit: slow, plain, and fast enough here.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
        }
    }
    !c
}

/// Adler-32, the checksum ending a zlib stream.
pub fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in bytes {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// A zlib stream holding `data` in "stored" (uncompressed) deflate blocks
/// of at most 65,535 bytes: valid for every PNG reader, just not small.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = if data.is_empty() {
        vec![&[]]
    } else {
        data.chunks(65_535).collect()
    };
    for (i, block) in blocks.iter().enumerate() {
        out.push(u8::from(i + 1 == blocks.len())); // BFINAL, BTYPE = 00
        let len = block.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// One image (pixel values from −1 to 1, h × w × c, c = 1 or 3) as PNG
/// bytes: 8 bits per channel, grayscale or RGB.
pub fn png(pixels: &[f32], height: usize, width: usize, channels: usize) -> Vec<u8> {
    assert!(channels == 1 || channels == 3, "grayscale or RGB");
    assert_eq!(pixels.len(), height * width * channels);
    let mut raw = Vec::with_capacity(height * (1 + width * channels));
    for y in 0..height {
        raw.push(0); // filter type 0: the row as it is
        let row = &pixels[y * width * channels..(y + 1) * width * channels];
        raw.extend(row.iter().map(|&x| to_byte(x)));
    }
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, if channels == 1 { 0 } else { 2 }, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &zlib_stored(&raw));
    chunk(b"IEND", &[]);
    out
}

/// Images in a grid, `cols` across, each scaled up by `zoom` (pixels
/// repeated) with a 1-pixel gap at that zoom: one picture of many samples.
/// `rows_of_images` is n × (h·w·c).
pub fn grid(images: &Mat<f32>, shape: Shape, cols: usize, zoom: usize) -> (Vec<f32>, usize, usize) {
    let (h, w, c) = (shape.height, shape.width, shape.channels);
    let cols = cols.max(1).min(images.rows.max(1));
    let rows = images.rows.div_ceil(cols);
    let gap = zoom.max(1);
    let gh = rows * (h * zoom + gap) + gap;
    let gw = cols * (w * zoom + gap) + gap;
    let mut out = vec![0.0f32; gh * gw * c]; // the gaps: mid grey
    for (k, img) in (0..images.rows).map(|k| (k, images.row(k))) {
        let (oy, ox) = (
            gap + (k / cols) * (h * zoom + gap),
            gap + (k % cols) * (w * zoom + gap),
        );
        for y in 0..h * zoom {
            for x in 0..w * zoom {
                let src = ((y / zoom) * w + x / zoom) * c;
                let dst = ((oy + y) * gw + ox + x) * c;
                out[dst..dst + c].copy_from_slice(&img[src..src + c]);
            }
        }
    }
    (out, gh, gw)
}

/// Write images as one PNG grid.
pub fn write_grid(
    path: &Path,
    images: &Mat<f32>,
    shape: Shape,
    cols: usize,
    zoom: usize,
) -> Result<(), String> {
    let (px, h, w) = grid(images, shape, cols, zoom);
    let mut f =
        fs::File::create(path).map_err(|e| format!("can't write {}: {e}", path.display()))?;
    f.write_all(&png(&px, h, w, shape.channels))
        .map_err(|e| format!("can't write {}: {e}", path.display()))
}

// ----- bytes

/// A value in −1 to 1 as a byte, 0 to 255, the way MNIST stores it.
pub fn to_byte(x: f32) -> u8 {
    ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shp(h: usize, w: usize, c: usize) -> Shape {
        Shape {
            height: h,
            width: w,
            channels: c,
        }
    }

    /// Patches and back, bit for bit, for several sizes, patch sizes and
    /// channel counts (Decision 5: nothing fixed at 28 × 28).
    #[test]
    fn patches_round_trip_at_any_size() {
        let mut rng = Rng::new(1);
        for (h, w, c, p) in [
            (28, 28, 1, 4),
            (28, 28, 1, 7),
            (16, 16, 3, 4),
            (32, 32, 3, 8),
            (8, 12, 1, 2),
        ] {
            let s = shp(h, w, c);
            let imgs = Mat::from_vec(
                3,
                s.values(),
                (0..3 * s.values()).map(|_| rng.normal() as f32).collect(),
            );
            let pt = patchify(&imgs, s, p);
            assert_eq!((pt.rows, pt.cols), (3 * patch_count(s, p), p * p * c));
            assert_eq!(unpatchify(&pt, s, p), imgs, "{h}×{w}×{c}, patch {p}");
        }
        assert!(check_patch(shp(28, 28, 1), 5).is_err());
        assert!(check_patch(shp(28, 28, 1), 7).is_ok());
    }

    /// A patch really is a square of neighbouring pixels.
    #[test]
    fn a_patch_is_a_square_of_pixels() {
        let s = shp(4, 4, 1);
        let imgs = Mat::from_vec(1, 16, (0..16).map(|v| v as f32).collect());
        let pt = patchify(&imgs, s, 2);
        assert_eq!(pt.row(0), &[0.0, 1.0, 4.0, 5.0]); // top left
        assert_eq!(pt.row(1), &[2.0, 3.0, 6.0, 7.0]); // top right
        assert_eq!(pt.row(3), &[10.0, 11.0, 14.0, 15.0]); // bottom right
    }

    #[test]
    fn checksums_against_known_values() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926); // the standard check value
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(crc32(b""), 0);
        assert_eq!(adler32(b""), 1);
    }

    /// Read our own PNG back: the signature, each chunk's CRC, the header,
    /// and the pixels out of the stored zlib blocks.
    #[test]
    fn png_round_trip() {
        for c in [1, 3] {
            let (h, w) = (5, 300);
            let px: Vec<f32> = (0..h * w * c)
                .map(|i| ((i % 256) as f32) / 127.5 - 1.0)
                .collect();
            let bytes = png(&px, h, w, c);
            assert_eq!(
                &bytes[..8],
                &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
            );
            let mut i = 8;
            let mut idat = Vec::new();
            while i < bytes.len() {
                let len = u32::from_be_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
                let body = &bytes[i + 4..i + 8 + len];
                let crc = u32::from_be_bytes(bytes[i + 8 + len..i + 12 + len].try_into().unwrap());
                assert_eq!(crc32(body), crc);
                match &body[..4] {
                    b"IHDR" => {
                        assert_eq!(
                            u32::from_be_bytes(body[4..8].try_into().unwrap()) as usize,
                            w
                        );
                        assert_eq!(
                            u32::from_be_bytes(body[8..12].try_into().unwrap()) as usize,
                            h
                        );
                        assert_eq!(body[13], if c == 1 { 0 } else { 2 });
                    }
                    b"IDAT" => idat.extend_from_slice(&body[4..]),
                    _ => {}
                }
                i += 12 + len;
            }
            // Unpack the stored blocks.
            let mut raw = Vec::new();
            let mut j = 2;
            loop {
                let last = idat[j] & 1 == 1;
                let len = u16::from_le_bytes([idat[j + 1], idat[j + 2]]) as usize;
                assert_eq!(
                    u16::from_le_bytes([idat[j + 3], idat[j + 4]]),
                    !(len as u16)
                );
                raw.extend_from_slice(&idat[j + 5..j + 5 + len]);
                j += 5 + len;
                if last {
                    break;
                }
            }
            assert_eq!(
                u32::from_be_bytes(idat[j..j + 4].try_into().unwrap()),
                adler32(&raw)
            );
            for y in 0..h {
                let row = &raw[y * (1 + w * c)..(y + 1) * (1 + w * c)];
                assert_eq!(row[0], 0);
                for x in 0..w * c {
                    assert_eq!(row[1 + x], to_byte(px[y * w * c + x]));
                }
            }
        }
    }

    #[test]
    fn large_images_use_several_stored_blocks() {
        let z = zlib_stored(&vec![7u8; 150_000]);
        // Three blocks of 65,535 or fewer bytes, each with a 5-byte header.
        assert_eq!(z.len(), 2 + 150_000 + 3 * 5 + 4);
    }

    #[test]
    fn idx_round_trip_and_checks() {
        let dir = std::env::temp_dir().join(format!("mini_transformer_idx_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let set = shapes(7, 12, 1, 3);
        let (i, l) = (dir.join("img"), dir.join("lab"));
        write_idx(&set, &i, &l).unwrap();
        let back = read_idx(&i, &l).unwrap();
        assert_eq!(back.shape, set.shape);
        assert_eq!(back.labels, set.labels);
        for (a, b) in back.pixels.data.iter().zip(&set.pixels.data) {
            assert!((a - b).abs() < 1.0 / 127.0, "{a} {b}");
        }
        assert!(read_idx(&l, &i).is_err()); // swapped
        fs::write(&i, &fs::read(&i).unwrap()[..100]).unwrap();
        assert!(read_idx(&i, &l).is_err()); // truncated
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn shapes_at_any_size_and_channels() {
        for (size, c) in [(16, 1), (28, 1), (32, 3), (64, 3)] {
            let s = shapes(30, size, c, 4);
            assert_eq!(s.shape, shp(size, size, c));
            assert_eq!((s.pixels.rows, s.pixels.cols), (30, size * size * c));
            assert!(s.pixels.data.iter().all(|v| (-1.0..=1.0).contains(v)));
            assert!(s.labels.iter().all(|&l| l < 3));
            // Every image has a shape on it: some pixels lit.
            for r in 0..30 {
                assert!(s.pixels.row(r).iter().any(|&v| v > -0.5));
            }
        }
        assert_eq!(shapes(5, 16, 1, 9).pixels, shapes(5, 16, 1, 9).pixels);
    }
}
