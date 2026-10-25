//! A minimal dense matrix of f64 values, stored row by row, with only the
//! operations attention needs. Sizes here are tiny (a few tokens, a few
//! dozen dimensions), so clarity matters more than speed.

use crate::rng::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    /// Element (r, c) is data[r * cols + c].
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Build a matrix from a list of equal-length rows.
    pub fn from_rows(rows: &[&[f64]]) -> Mat {
        let cols = rows.first().map_or(0, |r| r.len());
        assert!(rows.iter().all(|r| r.len() == cols), "ragged rows");
        Mat {
            rows: rows.len(),
            cols,
            data: rows.iter().flat_map(|r| r.iter().copied()).collect(),
        }
    }

    /// A matrix of independent normal(0, std²) values.
    pub fn random_normal(rows: usize, cols: usize, std: f64, rng: &mut Rng) -> Mat {
        let data = (0..rows * cols).map(|_| std * rng.normal()).collect();
        Mat { rows, cols, data }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Matrix product self · other.
    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows, "matmul shape mismatch");
        let mut out = Mat::zeros(self.rows, other.cols);
        // Row r of the result is the sum over k of self[r][k] × row k of
        // other: the inner loop runs along contiguous rows.
        for (out_row, a_row) in out
            .data
            .chunks_exact_mut(other.cols.max(1))
            .zip(self.data.chunks_exact(self.cols.max(1)))
        {
            for (&a, b_row) in a_row.iter().zip(other.data.chunks_exact(other.cols.max(1))) {
                if a == 0.0 {
                    continue;
                }
                for (o, &b) in out_row.iter_mut().zip(b_row) {
                    *o += a * b;
                }
            }
        }
        out
    }

    pub fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// self += s · other, element by element.
    pub fn add_scaled(&mut self, other: &Mat, s: f64) {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "shape mismatch"
        );
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += s * b;
        }
    }

    /// Every element multiplied by s.
    pub fn scale(&self, s: f64) -> Mat {
        Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * s).collect(),
        }
    }
}

/// Print a matrix as a labeled table. −∞ is printed as "−∞".
pub fn print_table(
    title: &str,
    m: &Mat,
    row_labels: &[String],
    col_labels: &[String],
    decimals: usize,
) {
    print_cells(title, m, row_labels, col_labels, decimals, false);
}

/// Print attention weights: like print_table, but the positions the causal
/// mask removed (above the diagonal) are shown as "·" rather than as 0, so
/// that a printed 0.000 always means "allowed, but too small to show".
pub fn print_weights(title: &str, m: &Mat, labels: &[String], decimals: usize) {
    print_cells(title, m, labels, labels, decimals, true);
    println!("  (· = masked: a later token, weight exactly 0)\n");
}

fn print_cells(
    title: &str,
    m: &Mat,
    row_labels: &[String],
    col_labels: &[String],
    decimals: usize,
    masked_above_diagonal: bool,
) {
    let width = (decimals + 5).max(
        col_labels
            .iter()
            .map(|l| l.chars().count())
            .max()
            .unwrap_or(0)
            + 1,
    );
    let label_width = row_labels
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(0)
        .max(4);
    println!("{title}");
    print!("  {:label_width$}", "");
    for l in col_labels {
        print!("{l:>width$}");
    }
    println!();
    for (r, label) in row_labels.iter().enumerate().take(m.rows) {
        print!("  {label:<label_width$}");
        for c in 0..m.cols {
            let v = m.get(r, c);
            let s = if masked_above_diagonal && c > r {
                "·".to_string()
            } else if v == f64::NEG_INFINITY {
                "−∞".to_string()
            } else {
                format!("{v:.decimals$}")
            };
            print!("{s:>width$}");
        }
        println!();
    }
    if !masked_above_diagonal {
        println!();
    }
}

/// Labels "1", "2", ... for columns or rows that have no names.
pub fn index_labels(n: usize) -> Vec<String> {
    (1..=n).map(|i| i.to_string()).collect()
}
