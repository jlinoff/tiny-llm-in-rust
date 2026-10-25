//! A small reverse-mode automatic differentiation tape, on whole matrices.
//!
//! How it works. Every operation in the forward pass (a matrix product, a
//! softmax, an addition) is recorded on the tape as it runs: its result, and
//! which earlier entries it used. To compute gradients, `backward` starts
//! from the loss, whose gradient with respect to itself is 1, and walks the
//! tape in reverse. Each entry applies its own backward rule: given the
//! gradient of the loss with respect to its result, it adds its share to
//! the gradients of its inputs. That is the chain rule, applied one
//! operation at a time; after the walk, every input holds its gradient.
//!
//! So the backward rules are still written by hand, but each is local: one
//! operation, a few lines, checked on its own against finite differences
//! (see the tests). A model then needs only its forward pass.
//!
//! Entries are referred to by `Var`, an index into the tape. A fresh tape
//! is built for every sequence: the weights go in as leaves (`leaf`), the
//! forward pass records its operations, `backward` runs once, and the
//! leaves' gradients are read out with `grad`.
//!
//! Shapes follow the rest of the crate: one row per token. "Row vector"
//! means a 1 × m matrix (a bias, say) added or multiplied into every row;
//! "column" means an n × 1 matrix with one number per row (a row's mean).

use crate::matrix::Mat;

/// A reference to one entry on the tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(usize);

/// One recorded operation, with the inputs its backward rule needs.
enum Op {
    /// An input or a weight: nothing to pass the gradient on to.
    Leaf,
    /// C = A·B
    MatMul(Var, Var),
    /// C = Aᵀ
    Transpose(Var),
    /// C = A + B, same shape
    Add(Var, Var),
    /// C = A ⊙ B, element by element, same shape
    Mul(Var, Var),
    /// C = s·A
    Scale(Var, f64),
    /// C = A + s, for every element
    AddScalar(Var),
    /// C = A + b, the row vector b added to every row
    AddRow(Var, Var),
    /// C = A ⊙ g, the row vector g multiplying every row
    MulRow(Var, Var),
    /// C = A − c, the column c subtracted from every column
    SubCol(Var, Var),
    /// C = A / c, every column divided by the column c
    DivCol(Var, Var),
    /// C = the mean of each row of A, as a column
    RowMean(Var),
    /// C = √A, element by element
    Sqrt(Var),
    /// C = GELU(A), element by element
    Gelu(Var),
    /// C = the sum of every element of A, as 1 × 1 (used by the tests to
    /// reduce a result to a loss)
    #[allow(dead_code)]
    Sum(Var),
    /// C = rows idx[0], idx[1], … of A (an embedding lookup)
    Gather(Var, Vec<usize>),
    /// C = [A₁ A₂ …], placed side by side
    Concat(Vec<Var>),
    /// C = softmax of each row of A after the causal mask: row i keeps
    /// columns 0..=i, and the later ones get weight exactly 0
    CausalSoftmax(Var),
    /// C = mean over the rows with a target of −ln softmax(A)[row][target],
    /// as 1 × 1. Fused: its backward rule (probabilities − one-hot) is far
    /// simpler than chaining softmax's and the logarithm's. Keeps the
    /// probabilities for the backward pass.
    SoftmaxCrossEntropy(Var, Vec<Option<usize>>, Mat),
}

struct Node {
    value: Mat,
    op: Op,
}

#[derive(Default)]
pub struct Tape {
    nodes: Vec<Node>,
    grads: Vec<Option<Mat>>,
}

/// GELU, the tanh form GPT-2 uses: x·Φ(x), with Φ approximated.
const GELU_C: f64 = 0.797_884_560_802_865_4; // √(2/π)
const GELU_A: f64 = 0.044_715;

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (GELU_C * (x + GELU_A * x * x * x)).tanh())
}

fn gelu_slope(x: f64) -> f64 {
    let t = (GELU_C * (x + GELU_A * x * x * x)).tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * GELU_C * (1.0 + 3.0 * GELU_A * x * x)
}

/// Apply f to every element of a.
fn map(a: &Mat, f: impl Fn(f64) -> f64) -> Mat {
    let mut out = a.clone();
    out.data.iter_mut().for_each(|v| *v = f(*v));
    out
}

/// Combine a and b element by element.
fn zip(a: &Mat, b: &Mat, f: impl Fn(f64, f64) -> f64) -> Mat {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols), "shape mismatch");
    let mut out = a.clone();
    for (o, &y) in out.data.iter_mut().zip(&b.data) {
        *o = f(*o, y);
    }
    out
}

/// The sum of each column, as a row vector.
fn col_sums(a: &Mat) -> Mat {
    let mut out = Mat::zeros(1, a.cols);
    for r in 0..a.rows {
        for (o, v) in out.data.iter_mut().zip(a.row(r)) {
            *o += v;
        }
    }
    out
}

/// The sum of each row, as a column.
fn row_sums(a: &Mat) -> Mat {
    let mut out = Mat::zeros(a.rows, 1);
    for r in 0..a.rows {
        out.data[r] = a.row(r).iter().sum();
    }
    out
}

/// Combine every row of a with the row vector b.
fn with_row(a: &Mat, b: &Mat, f: impl Fn(f64, f64) -> f64) -> Mat {
    assert_eq!((b.rows, b.cols), (1, a.cols), "row vector shape mismatch");
    let mut out = a.clone();
    for r in 0..a.rows {
        for (o, &y) in out.row_mut(r).iter_mut().zip(&b.data) {
            *o = f(*o, y);
        }
    }
    out
}

/// Combine every column of a with the column c.
fn with_col(a: &Mat, c: &Mat, f: impl Fn(f64, f64) -> f64) -> Mat {
    assert_eq!((c.rows, c.cols), (a.rows, 1), "column shape mismatch");
    let mut out = a.clone();
    for r in 0..a.rows {
        let y = c.data[r];
        out.row_mut(r).iter_mut().for_each(|o| *o = f(*o, y));
    }
    out
}

impl Tape {
    pub fn new() -> Tape {
        Tape::default()
    }

    fn push(&mut self, value: Mat, op: Op) -> Var {
        self.nodes.push(Node { value, op });
        Var(self.nodes.len() - 1)
    }

    pub fn value(&self, v: Var) -> &Mat {
        &self.nodes[v.0].value
    }

    /// The gradient of the loss with respect to v, after `backward`.
    pub fn grad(&self, v: Var) -> Mat {
        let m = &self.nodes[v.0].value;
        self.grads
            .get(v.0)
            .and_then(|g| g.clone())
            .unwrap_or_else(|| Mat::zeros(m.rows, m.cols))
    }

    // ----- forward operations: each computes its result and records itself

    pub fn leaf(&mut self, m: Mat) -> Var {
        self.push(m, Op::Leaf)
    }

    pub fn matmul(&mut self, a: Var, b: Var) -> Var {
        let v = self.value(a).matmul(self.value(b));
        self.push(v, Op::MatMul(a, b))
    }

    pub fn transpose(&mut self, a: Var) -> Var {
        let v = self.value(a).transpose();
        self.push(v, Op::Transpose(a))
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let v = zip(self.value(a), self.value(b), |x, y| x + y);
        self.push(v, Op::Add(a, b))
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let v = zip(self.value(a), self.value(b), |x, y| x * y);
        self.push(v, Op::Mul(a, b))
    }

    pub fn scale(&mut self, a: Var, s: f64) -> Var {
        let v = self.value(a).scale(s);
        self.push(v, Op::Scale(a, s))
    }

    pub fn add_scalar(&mut self, a: Var, s: f64) -> Var {
        let v = map(self.value(a), |x| x + s);
        self.push(v, Op::AddScalar(a))
    }

    pub fn add_row(&mut self, a: Var, b: Var) -> Var {
        let v = with_row(self.value(a), self.value(b), |x, y| x + y);
        self.push(v, Op::AddRow(a, b))
    }

    pub fn mul_row(&mut self, a: Var, g: Var) -> Var {
        let v = with_row(self.value(a), self.value(g), |x, y| x * y);
        self.push(v, Op::MulRow(a, g))
    }

    pub fn sub_col(&mut self, a: Var, c: Var) -> Var {
        let v = with_col(self.value(a), self.value(c), |x, y| x - y);
        self.push(v, Op::SubCol(a, c))
    }

    pub fn div_col(&mut self, a: Var, c: Var) -> Var {
        let v = with_col(self.value(a), self.value(c), |x, y| x / y);
        self.push(v, Op::DivCol(a, c))
    }

    pub fn row_mean(&mut self, a: Var) -> Var {
        let m = self.value(a);
        let v = row_sums(m).scale(1.0 / m.cols as f64);
        self.push(v, Op::RowMean(a))
    }

    pub fn sqrt(&mut self, a: Var) -> Var {
        let v = map(self.value(a), f64::sqrt);
        self.push(v, Op::Sqrt(a))
    }

    pub fn gelu(&mut self, a: Var) -> Var {
        let v = map(self.value(a), gelu);
        self.push(v, Op::Gelu(a))
    }

    #[allow(dead_code)] // used by the tests
    pub fn sum(&mut self, a: Var) -> Var {
        let s: f64 = self.value(a).data.iter().sum();
        self.push(Mat::from_rows(&[&[s]]), Op::Sum(a))
    }

    pub fn gather(&mut self, table: Var, idx: &[usize]) -> Var {
        let t = self.value(table);
        let mut v = Mat::zeros(idx.len(), t.cols);
        for (r, &i) in idx.iter().enumerate() {
            v.row_mut(r).copy_from_slice(t.row(i));
        }
        self.push(v, Op::Gather(table, idx.to_vec()))
    }

    pub fn concat(&mut self, parts: &[Var]) -> Var {
        let rows = self.value(parts[0]).rows;
        let cols = parts.iter().map(|&p| self.value(p).cols).sum();
        let mut v = Mat::zeros(rows, cols);
        for r in 0..rows {
            let mut at = 0;
            for &p in parts {
                let m = self.value(p);
                v.row_mut(r)[at..at + m.cols].copy_from_slice(m.row(r));
                at += m.cols;
            }
        }
        self.push(v, Op::Concat(parts.to_vec()))
    }

    pub fn causal_softmax(&mut self, a: Var) -> Var {
        let s = self.value(a);
        let mut v = Mat::zeros(s.rows, s.cols);
        for i in 0..s.rows {
            let row = &s.row(i)[..=i.min(s.cols - 1)];
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = row.iter().map(|x| (x - max).exp()).sum();
            for (j, &x) in row.iter().enumerate() {
                v.set(i, j, (x - max).exp() / total);
            }
        }
        self.push(v, Op::CausalSoftmax(a))
    }

    pub fn softmax_cross_entropy(&mut self, logits: Var, targets: &[Option<usize>]) -> Var {
        let l = self.value(logits);
        assert_eq!(targets.len(), l.rows, "one target (or None) per row");
        let mut probs = Mat::zeros(l.rows, l.cols);
        let (mut loss, mut n) = (0.0, 0);
        for (i, target) in targets.iter().enumerate() {
            let row = l.row(i);
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = row.iter().map(|x| (x - max).exp()).sum();
            for (p, x) in probs.row_mut(i).iter_mut().zip(row) {
                *p = (x - max).exp() / total;
            }
            if let Some(t) = *target {
                loss -= probs.get(i, t).ln();
                n += 1;
            }
        }
        let loss = Mat::from_rows(&[&[loss / n.max(1) as f64]]);
        self.push(
            loss,
            Op::SoftmaxCrossEntropy(logits, targets.to_vec(), probs),
        )
    }

    // ----- composites: built from the operations above, so they need no
    // backward rule of their own

    /// Layer normalization of each row, then a learned scale γ and shift β
    /// (row vectors): y = γ ⊙ (x − mean) / √(variance + ε) + β.
    pub fn layer_norm(&mut self, x: Var, gamma: Var, beta: Var, eps: f64) -> Var {
        let mean = self.row_mean(x);
        let centered = self.sub_col(x, mean);
        let squared = self.mul(centered, centered);
        let var = self.row_mean(squared);
        let var_eps = self.add_scalar(var, eps);
        let std = self.sqrt(var_eps);
        let normed = self.div_col(centered, std);
        let scaled = self.mul_row(normed, gamma);
        self.add_row(scaled, beta)
    }

    // ----- backward

    /// Add g to the gradient of v.
    fn accumulate(&mut self, v: Var, g: &Mat) {
        match &mut self.grads[v.0] {
            Some(acc) => acc.add_scaled(g, 1.0),
            slot @ None => *slot = Some(g.clone()),
        }
    }

    /// Compute the gradient of `loss` (a 1 × 1 entry) with respect to every
    /// entry on the tape, walking it in reverse.
    pub fn backward(&mut self, loss: Var) {
        let l = self.value(loss);
        assert_eq!((l.rows, l.cols), (1, 1), "backward needs a 1 × 1 loss");
        self.grads = vec![None; self.nodes.len()];
        self.grads[loss.0] = Some(Mat::from_rows(&[&[1.0]]));
        for i in (0..=loss.0).rev() {
            let Some(g) = self.grads[i].take() else {
                continue; // the loss doesn't depend on this entry
            };
            for (input, contribution) in self.backward_rule(i, &g) {
                self.accumulate(input, &contribution);
            }
            self.grads[i] = Some(g);
        }
    }

    /// The backward rule of entry i: given g, the gradient of the loss with
    /// respect to its result C, the gradient contributions to its inputs.
    fn backward_rule(&self, i: usize, g: &Mat) -> Vec<(Var, Mat)> {
        let c = &self.nodes[i].value;
        let val = |v: Var| &self.nodes[v.0].value;
        match &self.nodes[i].op {
            Op::Leaf => vec![],
            // C = A·B:  dA = dC·Bᵀ,  dB = Aᵀ·dC
            Op::MatMul(a, b) => vec![
                (*a, g.matmul(&val(*b).transpose())),
                (*b, val(*a).transpose().matmul(g)),
            ],
            // C = Aᵀ:  dA = dCᵀ
            Op::Transpose(a) => vec![(*a, g.transpose())],
            // C = A + B:  both receive dC
            Op::Add(a, b) => vec![(*a, g.clone()), (*b, g.clone())],
            // C = A ⊙ B:  dA = dC ⊙ B,  dB = dC ⊙ A
            Op::Mul(a, b) => vec![
                (*a, zip(g, val(*b), |x, y| x * y)),
                (*b, zip(g, val(*a), |x, y| x * y)),
            ],
            // C = s·A:  dA = s·dC
            Op::Scale(a, s) => vec![(*a, g.scale(*s))],
            // C = A + s:  dA = dC
            Op::AddScalar(a) => vec![(*a, g.clone())],
            // C = A + b (every row):  dA = dC,  db = column sums of dC
            Op::AddRow(a, b) => vec![(*a, g.clone()), (*b, col_sums(g))],
            // C = A ⊙ g (every row):  dA = dC ⊙ g,  dg = column sums of dC ⊙ A
            Op::MulRow(a, r) => vec![
                (*a, with_row(g, val(*r), |x, y| x * y)),
                (*r, col_sums(&zip(g, val(*a), |x, y| x * y))),
            ],
            // C = A − c (every column):  dA = dC,  dc = −(row sums of dC)
            Op::SubCol(a, col) => vec![(*a, g.clone()), (*col, row_sums(g).scale(-1.0))],
            // C = A / c (every column):  dA = dC / c,
            //                            dc = −(row sums of dC ⊙ A) / c²
            Op::DivCol(a, col) => {
                let cv = val(*col);
                let da = with_col(g, cv, |x, y| x / y);
                let dc = with_col(&row_sums(&zip(g, val(*a), |x, y| x * y)), cv, |s, y| {
                    -s / (y * y)
                });
                vec![(*a, da), (*col, dc)]
            }
            // C = row means of A (m columns):  dA[i][j] = dC[i] / m
            Op::RowMean(a) => {
                let m = val(*a);
                let mut da = Mat::zeros(m.rows, m.cols);
                for r in 0..m.rows {
                    da.row_mut(r).fill(g.data[r] / m.cols as f64);
                }
                vec![(*a, da)]
            }
            // C = √A:  dA = dC / (2C)
            Op::Sqrt(a) => vec![(*a, zip(g, c, |x, y| x / (2.0 * y)))],
            // C = GELU(A):  dA = dC ⊙ GELU′(A)
            Op::Gelu(a) => vec![(*a, zip(g, val(*a), |x, y| x * gelu_slope(y)))],
            // C = sum of A:  every element receives dC
            Op::Sum(a) => {
                let m = val(*a);
                vec![(*a, map(&Mat::zeros(m.rows, m.cols), |_| g.data[0]))]
            }
            // C = rows idx of A:  each row of dC goes back to the row it came
            // from (adding up when a row was used more than once)
            Op::Gather(a, idx) => {
                let m = val(*a);
                let mut da = Mat::zeros(m.rows, m.cols);
                for (r, &src) in idx.iter().enumerate() {
                    for (d, x) in da.row_mut(src).iter_mut().zip(g.row(r)) {
                        *d += x;
                    }
                }
                vec![(*a, da)]
            }
            // C = [A₁ A₂ …]:  each part receives its own columns of dC
            Op::Concat(parts) => {
                let mut at = 0;
                parts
                    .iter()
                    .map(|&p| {
                        let w = val(p).cols;
                        let mut dp = Mat::zeros(g.rows, w);
                        for r in 0..g.rows {
                            dp.row_mut(r).copy_from_slice(&g.row(r)[at..at + w]);
                        }
                        at += w;
                        (p, dp)
                    })
                    .collect()
            }
            // C = softmax by row (masked):  dA_ij = C_ij (dC_ij − Σ_k C_ik dC_ik).
            // Masked entries have C_ij = 0, so they get no gradient.
            Op::CausalSoftmax(a) => {
                let mut da = Mat::zeros(c.rows, c.cols);
                for r in 0..c.rows {
                    let (cr, gr) = (c.row(r), g.row(r));
                    let dot: f64 = cr.iter().zip(gr).map(|(x, y)| x * y).sum();
                    for (j, d) in da.row_mut(r).iter_mut().enumerate() {
                        *d = cr[j] * (gr[j] - dot);
                    }
                }
                vec![(*a, da)]
            }
            // C = mean −ln p(target):  dA = (probabilities − one-hot) / n on
            // rows with a target, 0 on the others, times dC
            Op::SoftmaxCrossEntropy(a, targets, probs) => {
                let n = targets.iter().filter(|t| t.is_some()).count().max(1) as f64;
                let mut da = probs.clone();
                for (r, t) in targets.iter().enumerate() {
                    match t {
                        None => da.row_mut(r).fill(0.0),
                        Some(t) => {
                            let p = da.get(r, *t);
                            da.set(r, *t, p - 1.0);
                        }
                    }
                }
                vec![(*a, da.scale(g.data[0] / n))]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Finite-difference check: build the loss from leaves holding `inputs`,
    /// compare the tape's gradient for every input element with
    /// (loss(x + ε) − loss(x − ε)) / 2ε. Returns the worst relative error.
    fn check(inputs: &[Mat], build: impl Fn(&mut Tape, &[Var]) -> Var) -> f64 {
        let run = |ms: &[Mat]| {
            let mut t = Tape::new();
            let vars: Vec<Var> = ms.iter().map(|m| t.leaf(m.clone())).collect();
            let loss = build(&mut t, &vars);
            (t, vars, loss)
        };
        let (mut t, vars, loss) = run(inputs);
        t.backward(loss);
        let eps = 1e-6;
        let mut worst: f64 = 0.0;
        for (k, m) in inputs.iter().enumerate() {
            let analytic = t.grad(vars[k]);
            for i in 0..m.data.len() {
                let mut probe = inputs.to_vec();
                probe[k].data[i] += eps;
                let (tu, _, lu) = run(&probe);
                probe[k].data[i] -= 2.0 * eps;
                let (td, _, ld) = run(&probe);
                let numeric = (tu.value(lu).data[0] - td.value(ld).data[0]) / (2.0 * eps);
                let a = analytic.data[i];
                let scale = numeric.abs() + a.abs();
                if scale > 1e-9 {
                    worst = worst.max((numeric - a).abs() / scale);
                }
            }
        }
        worst
    }

    fn rand(rows: usize, cols: usize, seed: u64) -> Mat {
        Mat::random_normal(rows, cols, 1.0, &mut Rng::new(seed))
    }

    /// Positive values, for √ and division.
    fn positive(rows: usize, cols: usize, seed: u64) -> Mat {
        map(&rand(rows, cols, seed), |x| 1.0 + x.abs())
    }

    /// Reduce any result to a 1 × 1 loss: sum(C ⊙ R) for a fixed random R,
    /// so every element of C gets a different weight.
    fn reduce(t: &mut Tape, c: Var, seed: u64) -> Var {
        let (rows, cols) = (t.value(c).rows, t.value(c).cols);
        let r = t.leaf(rand(rows, cols, seed));
        let p = t.mul(c, r);
        t.sum(p)
    }

    const TOL: f64 = 1e-6;

    #[test]
    fn matmul_and_transpose() {
        let e = check(&[rand(3, 4, 1), rand(4, 2, 2)], |t, v| {
            let c = t.matmul(v[0], v[1]);
            let c = t.transpose(c);
            reduce(t, c, 9)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn elementwise_operations() {
        let e = check(&[rand(3, 4, 1), rand(3, 4, 2)], |t, v| {
            let a = t.add(v[0], v[1]);
            let m = t.mul(a, v[1]);
            let s = t.scale(m, -0.7);
            let s = t.add_scalar(s, 0.3);
            let g = t.gelu(s);
            reduce(t, g, 9)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn row_and_column_broadcasts() {
        let e = check(
            &[
                rand(3, 4, 1),
                rand(1, 4, 2),
                rand(1, 4, 3),
                positive(3, 1, 4),
            ],
            |t, v| {
                let a = t.add_row(v[0], v[1]);
                let a = t.mul_row(a, v[2]);
                let mean = t.row_mean(a);
                let a = t.sub_col(a, mean);
                let a = t.div_col(a, v[3]);
                reduce(t, a, 9)
            },
        );
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn square_root() {
        let e = check(&[positive(3, 4, 1)], |t, v| {
            let s = t.sqrt(v[0]);
            reduce(t, s, 9)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn gather_with_repeated_rows_and_concat() {
        let e = check(&[rand(5, 3, 1), rand(4, 2, 2)], |t, v| {
            let g = t.gather(v[0], &[4, 1, 1, 0]); // row 1 used twice
            let c = t.concat(&[g, v[1]]);
            reduce(t, c, 9)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn causal_softmax() {
        let e = check(&[rand(4, 4, 1)], |t, v| {
            let a = t.causal_softmax(v[0]);
            reduce(t, a, 9)
        });
        assert!(e < TOL, "{e}");
        // Masked entries are exactly 0 and each row sums to 1.
        let mut t = Tape::new();
        let s = t.leaf(rand(4, 4, 1));
        let a = t.causal_softmax(s);
        let w = t.value(a);
        for i in 0..4 {
            assert!((w.row(i).iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(w.row(i)[i + 1..].iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn softmax_cross_entropy_skips_rows_without_targets() {
        let targets = [None, Some(2), Some(0), None, Some(1)];
        let e = check(&[rand(5, 3, 1)], |t, v| {
            t.softmax_cross_entropy(v[0], &targets)
        });
        assert!(e < TOL, "{e}");
        let mut t = Tape::new();
        let l = t.leaf(rand(5, 3, 1));
        let loss = t.softmax_cross_entropy(l, &targets);
        t.backward(loss);
        let g = t.grad(l);
        assert!(g.row(0).iter().chain(g.row(3)).all(|&x| x == 0.0));
    }

    /// Layer norm is built from the primitive operations, so the tape derives
    /// its gradient. Compare with the closed-form gradient: with x̂ the
    /// normalized row, σ its standard deviation and d = dy ⊙ γ,
    ///   dx = (d − mean(d) − x̂ · mean(d ⊙ x̂)) / σ
    #[test]
    fn layer_norm_matches_the_closed_form_gradient() {
        let (x, gamma, beta, r) = (rand(3, 5, 1), rand(1, 5, 2), rand(1, 5, 3), rand(3, 5, 4));
        let eps = 1e-5;
        let mut t = Tape::new();
        let (xv, gv, bv, rv) = (
            t.leaf(x.clone()),
            t.leaf(gamma.clone()),
            t.leaf(beta),
            t.leaf(r.clone()),
        );
        let y = t.layer_norm(xv, gv, bv, eps);
        let p = t.mul(y, rv);
        let loss = t.sum(p);
        t.backward(loss);
        let dx = t.grad(xv);
        for i in 0..x.rows {
            let row = x.row(i);
            let n = row.len() as f64;
            let mean = row.iter().sum::<f64>() / n;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let sigma = (var + eps).sqrt();
            let xhat: Vec<f64> = row.iter().map(|v| (v - mean) / sigma).collect();
            let d: Vec<f64> = (0..row.len())
                .map(|j| r.get(i, j) * gamma.data[j])
                .collect();
            let md = d.iter().sum::<f64>() / n;
            let mdx = d.iter().zip(&xhat).map(|(a, b)| a * b).sum::<f64>() / n;
            for j in 0..row.len() {
                let closed = (d[j] - md - xhat[j] * mdx) / sigma;
                assert!((closed - dx.get(i, j)).abs() < 1e-12, "row {i} col {j}");
            }
        }
        // And the whole thing passes the finite-difference check.
        let e = check(&[rand(3, 5, 1), rand(1, 5, 2), rand(1, 5, 3)], |t, v| {
            let y = t.layer_norm(v[0], v[1], v[2], eps);
            reduce(t, y, 9)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn an_entry_used_twice_gets_both_contributions() {
        // loss = sum(A ⊙ A) has gradient 2A.
        let a = rand(2, 3, 1);
        let mut t = Tape::new();
        let v = t.leaf(a.clone());
        let sq = t.mul(v, v);
        let loss = t.sum(sq);
        t.backward(loss);
        let g = t.grad(v);
        for (x, y) in g.data.iter().zip(&a.data) {
            assert!((x - 2.0 * y).abs() < 1e-15);
        }
    }
}
