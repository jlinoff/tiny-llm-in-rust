//! Reading and writing safetensors files (PLAN.md, open decision 3), by hand.
//!
//! The format:
//!   - 8 bytes: N, the header's length, an unsigned little-endian integer
//!   - N bytes: a JSON object. Each tensor's name maps to its dtype, shape
//!     and data_offsets ([start, end) in the byte buffer); the optional key
//!     "__metadata__" maps to a string-to-string map. It may end in spaces.
//!   - the byte buffer: every tensor's numbers, little-endian, row-major,
//!     packed with no gaps or overlaps
//!
//! Only f32 tensors are written or read here; that's all this project uses.

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// One named tensor: an n-dimensional array of f32, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(name: &str, shape: &[usize], data: Vec<f32>) -> Tensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "tensor {name}: shape {shape:?} doesn't match {} numbers",
            data.len()
        );
        Tensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            data,
        }
    }
}

/// The file's contents: its tensors (in data order) and its metadata.
#[derive(Debug, PartialEq)]
pub struct File {
    pub tensors: Vec<Tensor>,
    pub metadata: BTreeMap<String, String>,
}

/// Encode tensors and metadata as safetensors bytes.
pub fn encode(tensors: &[Tensor], metadata: &BTreeMap<String, String>) -> Vec<u8> {
    let mut header = Map::new();
    if !metadata.is_empty() {
        header.insert("__metadata__".into(), json!(metadata));
    }
    let mut offset = 0;
    for t in tensors {
        assert!(t.name != "__metadata__", "reserved tensor name");
        assert!(
            !header.contains_key(&t.name),
            "duplicate tensor name {}",
            t.name
        );
        let end = offset + 4 * t.data.len();
        header.insert(
            t.name.clone(),
            json!({ "dtype": "F32", "shape": t.shape, "data_offsets": [offset, end] }),
        );
        offset = end;
    }
    let mut text = Value::Object(header).to_string().into_bytes();
    // Pad the header with spaces so the byte buffer starts 8-byte aligned.
    while !(8 + text.len()).is_multiple_of(8) {
        text.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + text.len() + offset);
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(&text);
    for t in tensors {
        for v in &t.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Decode safetensors bytes, checking everything the format promises.
pub fn decode(bytes: &[u8]) -> Result<File, String> {
    let n = bytes
        .get(..8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize)
        .ok_or("file too short for the header length")?;
    let header_bytes = bytes.get(8..8 + n).ok_or("file shorter than its header")?;
    if header_bytes.first() != Some(&b'{') {
        return Err("header doesn't start with '{'".into());
    }
    let header: Map<String, Value> = serde_json::from_slice(header_bytes)
        .map_err(|e| format!("header isn't valid JSON: {e}"))?;
    let buffer = &bytes[8 + n..];

    let mut metadata = BTreeMap::new();
    let mut entries: Vec<(usize, usize, Tensor)> = Vec::new();
    for (name, v) in &header {
        if name == "__metadata__" {
            let m = v.as_object().ok_or("__metadata__ isn't an object")?;
            for (k, val) in m {
                let s = val.as_str().ok_or(format!("metadata {k} isn't a string"))?;
                metadata.insert(k.clone(), s.to_string());
            }
            continue;
        }
        let bad = |what: &str| format!("tensor {name}: {what}");
        if v["dtype"] != "F32" {
            return Err(bad(&format!(
                "dtype {} (only F32 is supported)",
                v["dtype"]
            )));
        }
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .ok_or_else(|| bad("no shape"))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|d| d as usize)
                    .ok_or_else(|| bad("bad shape"))
            })
            .collect::<Result<_, _>>()?;
        let offs = v["data_offsets"]
            .as_array()
            .ok_or_else(|| bad("no data_offsets"))?;
        let get = |i: usize| {
            offs.get(i)
                .and_then(Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| bad("bad data_offsets"))
        };
        let (start, end) = (get(0)?, get(1)?);
        let count: usize = shape.iter().product();
        if end < start || end - start != 4 * count {
            return Err(bad("data_offsets don't match its shape"));
        }
        let raw = buffer
            .get(start..end)
            .ok_or_else(|| bad("data past the end of the file"))?;
        let data = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        entries.push((
            start,
            end,
            Tensor {
                name: name.clone(),
                shape,
                data,
            },
        ));
    }
    // The tensors must cover the buffer exactly: no gaps, no overlaps.
    entries.sort_by_key(|e| e.0);
    let mut at = 0;
    for (start, end, t) in &entries {
        if *start != at {
            return Err(format!(
                "tensor {}: gap or overlap in the byte buffer",
                t.name
            ));
        }
        at = *end;
    }
    if at != buffer.len() {
        return Err(format!("{} bytes after the last tensor", buffer.len() - at));
    }
    Ok(File {
        tensors: entries.into_iter().map(|e| e.2).collect(),
        metadata,
    })
}

pub fn write(
    path: &Path,
    tensors: &[Tensor],
    metadata: &BTreeMap<String, String>,
) -> Result<(), String> {
    fs::write(path, encode(tensors, metadata))
        .map_err(|e| format!("can't write {}: {e}", path.display()))
}

pub fn read(path: &Path) -> Result<File, String> {
    let bytes = fs::read(path).map_err(|e| format!("can't read {}: {e}", path.display()))?;
    decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (Vec<Tensor>, BTreeMap<String, String>) {
        let tensors = vec![
            Tensor::new(
                "a",
                &[2, 3],
                vec![1.0, -2.5, 3.25, 0.0, f32::MIN_POSITIVE, 1e30],
            ),
            Tensor::new("b.c", &[4], vec![0.1, 0.2, 0.3, 0.4]),
            Tensor::new("scalar", &[], vec![7.0]),
        ];
        let mut meta = BTreeMap::new();
        meta.insert("format".into(), "test".into());
        meta.insert("multi".into(), "line one\nline two = \"quoted\"".into());
        (tensors, meta)
    }

    #[test]
    fn round_trip_is_exact() {
        let (tensors, meta) = sample();
        let bytes = encode(&tensors, &meta);
        let f = decode(&bytes).unwrap();
        assert_eq!(f.tensors, tensors); // bit-for-bit: f32 → bytes → f32
        assert_eq!(f.metadata, meta);
    }

    #[test]
    fn header_is_padded_so_data_is_aligned() {
        let (tensors, meta) = sample();
        let bytes = encode(&tensors, &meta);
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!((8 + n) % 8, 0);
    }

    #[test]
    fn rejects_truncated_and_padded_files() {
        let (tensors, meta) = sample();
        let bytes = encode(&tensors, &meta);
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(decode(&longer).is_err());
        assert!(decode(&bytes[..5]).is_err());
    }
}

#[cfg(test)]
mod interop {
    use super::*;

    /// tests/data/hf_written.safetensors was written by Hugging Face's own
    /// library (safetensors 0.8.0, Python): w = arange(6)·0.5 − 1 as 2 × 3,
    /// b = [3.25, −0.125], and one metadata entry.
    #[test]
    fn reads_a_file_written_by_the_official_library() {
        let f = decode(include_bytes!("../tests/data/hf_written.safetensors")).unwrap();
        let get = |n: &str| f.tensors.iter().find(|t| t.name == n).unwrap();
        assert_eq!(get("w").shape, vec![2, 3]);
        assert_eq!(get("w").data, vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]);
        assert_eq!(get("b").data, vec![3.25, -0.125]);
        assert_eq!(f.metadata["note"], "written by safetensors 0.8.0 (Python)");
    }
}
