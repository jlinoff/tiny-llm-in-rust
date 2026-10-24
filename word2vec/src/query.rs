//! Using the trained vectors: nearest neighbors and saving to a file.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::corpus::Vocab;
use crate::matrix::{dot, Matrix};

/// A copy of the matrix with every row scaled to unit length, so that a dot
/// product of two rows is their cosine similarity.
pub struct Normalized {
    data: Vec<f32>,
    dim: usize,
}

impl Normalized {
    pub fn new(m: &Matrix) -> Normalized {
        let dim = m.dim();
        let mut data = vec![0.0; m.rows() * dim];
        for (r, row) in data.chunks_mut(dim).enumerate() {
            m.read_row(r, row);
            let norm = dot(row, row).sqrt();
            if norm > 0.0 {
                row.iter_mut().for_each(|x| *x /= norm);
            }
        }
        Normalized { data, dim }
    }

    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.dim..(r + 1) * self.dim]
    }

    /// The `top` words most similar to word `query` by cosine similarity,
    /// excluding the word itself, most similar first.
    pub fn nearest(&self, query: usize, top: usize) -> Vec<(usize, f32)> {
        let q = self.row(query);
        let mut scored: Vec<(usize, f32)> = (0..self.data.len() / self.dim)
            .filter(|&r| r != query)
            .map(|r| (r, dot(q, self.row(r))))
            .collect();
        let top = top.min(scored.len());
        if top == 0 {
            return Vec::new();
        }
        scored.select_nth_unstable_by(top - 1, |a, b| b.1.total_cmp(&a.1));
        scored.truncate(top);
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }
}

/// Write the word vectors in the standard word2vec text format: a header
/// line "V d", then one line per word, "word v1 v2 ... vd". gensim, among
/// others, reads this format directly.
pub fn save_vectors(path: &Path, vocab: &Vocab, m: &Matrix) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "{} {}", m.rows(), m.dim())?;
    let mut row = vec![0.0; m.dim()];
    for (id, word) in vocab.words.iter().enumerate() {
        m.read_row(id, &mut row);
        write!(out, "{word}")?;
        for x in &row {
            write!(out, " {x:.6}")?;
        }
        writeln!(out)?;
    }
    out.flush()
}
