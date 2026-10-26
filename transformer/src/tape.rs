//! Reverse-mode automatic differentiation on whole matrices, for batches
//! (PLAN.md, open decision 7). The same idea as attention/src/tape.rs: each
//! operation records its result and inputs as the forward pass runs, and
//! `backward` walks the tape in reverse, each entry applying its own
//! backward rule (the chain rule, one operation at a time).
//!
//! What's different here:
//! - Generic over the float type: f32 for training, f64 for the gradient
//!   checks in the tests.
//! - A batch of B sequences of T tokens is one (B·T) × d matrix, one row per
//!   token. Every per-token operation (matrix products, layer norm, GELU,
//!   biases) works on all B·T rows at once.
//! - Fused operations, each with one hand-written backward rule, where
//!   building it from small pieces would make the tape long and slow:
//!   layer norm (its closed-form gradient, the one attention/'s tape checked
//!   against), multi-head causal attention over the whole batch, and
//!   softmax with cross-entropy.

use crate::tensor::{Float, Mat};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(usize);

enum Op<T> {
    /// An input or a weight.
    Leaf,
    /// C = A·B
    MatMul(Var, Var),
    /// C = A·Bᵀ (the tied output layer: hidden states · token_embeddingᵀ)
    MatMulNT(Var, Var),
    /// C = A + B, same shape (the residual connections)
    Add(Var, Var),
    /// C = A + b, the row vector b added to every row (biases)
    AddRow(Var, Var),
    /// C = s·A for a fixed number s (LoRA's α/r)
    Scale(Var, T),
    /// Layer norm of each row with scale γ and shift β (row vectors). Keeps
    /// the normalized rows x̂ and each row's 1/σ for the backward pass.
    LayerNorm {
        x: Var,
        gamma: Var,
        beta: Var,
        xhat: Mat<T>,
        inv_std: Vec<T>,
    },
    /// C = GELU(A), element by element
    Gelu(Var),
    /// C = rows idx[0], idx[1], … of A (embedding lookups)
    Gather(Var, Vec<usize>),
    /// Multi-head causal self-attention over a batch: Q, K, V are (B·T) × d,
    /// the heads side by side (head k in columns k·h … (k+1)·h − 1). Keeps
    /// every head's attention weights, B·H blocks of T × T.
    Attention {
        q: Var,
        k: Var,
        v: Var,
        seq: usize,
        heads: usize,
        weights: Vec<Mat<T>>,
    },
    /// Mean over rows with a target of −ln softmax(row)[target], as 1 × 1.
    /// Keeps the probabilities: the backward rule is (probabilities −
    /// one-hot) / n.
    CrossEntropy {
        logits: Var,
        targets: Vec<Option<usize>>,
        probs: Mat<T>,
    },
    /// Σ A ⊙ W for a fixed W, as 1 × 1: used by the tests to turn any result
    /// into a single number.
    #[allow(dead_code)]
    WeightedSum(Var, Mat<T>),
}

struct Node<T> {
    value: Mat<T>,
    op: Op<T>,
}

pub struct Tape<T: Float> {
    nodes: Vec<Node<T>>,
    grads: Vec<Option<Mat<T>>>,
}

const LN_EPS: f64 = 1e-5;
const GELU_C: f64 = 0.797_884_560_802_865_4; // √(2/π)
const GELU_A: f64 = 0.044_715;

/// Layer norm of one row: x̂ = (x − mean)/√(variance + ε) into `xhat`,
/// γ·x̂ + β into `out`; returns 1/√(variance + ε). Shared with generate's
/// inference path, so both do exactly the same arithmetic.
pub(crate) fn layer_norm_row<T: Float>(
    row: &[T],
    g: &[T],
    b: &[T],
    xhat: &mut [T],
    out: &mut [T],
) -> T {
    let n = T::from_f64(row.len() as f64);
    let mut mean = T::zero();
    for &v in row {
        mean += v;
    }
    mean = mean / n;
    let mut var = T::zero();
    for &v in row {
        var += (v - mean) * (v - mean);
    }
    let inv = T::one() / (var / n + T::from_f64(LN_EPS)).sqrt();
    for j in 0..row.len() {
        let h = (row[j] - mean) * inv;
        xhat[j] = h;
        out[j] = g[j] * h + b[j];
    }
    inv
}

pub(crate) fn gelu<T: Float>(x: T) -> T {
    let (c, a, half) = (T::from_f64(GELU_C), T::from_f64(GELU_A), T::from_f64(0.5));
    half * x * (T::one() + (c * (x + a * x * x * x)).tanh())
}

fn gelu_slope<T: Float>(x: T) -> T {
    let (c, a, half) = (T::from_f64(GELU_C), T::from_f64(GELU_A), T::from_f64(0.5));
    let t = (c * (x + a * x * x * x)).tanh();
    half * (T::one() + t)
        + half * x * (T::one() - t * t) * c * (T::one() + T::from_f64(3.0) * a * x * x)
}

/// Column sums, as a 1 × m row vector.
fn col_sums<T: Float>(a: &Mat<T>) -> Mat<T> {
    let mut out = Mat::zeros(1, a.cols);
    for r in 0..a.rows {
        for (o, &v) in out.data.iter_mut().zip(a.row(r)) {
            *o += v;
        }
    }
    out
}

/// Copy block (rows r0.., cols c0..) of size rows × cols out of m.
fn block<T: Float>(m: &Mat<T>, r0: usize, c0: usize, rows: usize, cols: usize) -> Mat<T> {
    let mut out = Mat::zeros(rows, cols);
    for r in 0..rows {
        out.row_mut(r)
            .copy_from_slice(&m.row(r0 + r)[c0..c0 + cols]);
    }
    out
}

/// Write b into m at (r0, c0).
fn put_block<T: Float>(m: &mut Mat<T>, b: &Mat<T>, r0: usize, c0: usize) {
    for r in 0..b.rows {
        m.row_mut(r0 + r)[c0..c0 + b.cols].copy_from_slice(b.row(r));
    }
}

impl<T: Float> Default for Tape<T> {
    fn default() -> Self {
        Tape {
            nodes: Vec::new(),
            grads: Vec::new(),
        }
    }
}

impl<T: Float> Tape<T> {
    pub fn new() -> Tape<T> {
        Tape::default()
    }

    fn push(&mut self, value: Mat<T>, op: Op<T>) -> Var {
        self.nodes.push(Node { value, op });
        Var(self.nodes.len() - 1)
    }

    pub fn value(&self, v: Var) -> &Mat<T> {
        &self.nodes[v.0].value
    }

    /// The gradient of the loss with respect to v, after `backward`; taken
    /// out of the tape (it's only needed once).
    #[allow(dead_code)] // read by the training loop (step 4, stage 2)
    pub fn take_grad(&mut self, v: Var) -> Mat<T> {
        let m = &self.nodes[v.0].value;
        let (rows, cols) = (m.rows, m.cols);
        self.grads
            .get_mut(v.0)
            .and_then(Option::take)
            .unwrap_or_else(|| Mat::zeros(rows, cols))
    }

    // ----- forward operations

    pub fn leaf(&mut self, m: Mat<T>) -> Var {
        self.push(m, Op::Leaf)
    }

    pub fn matmul(&mut self, a: Var, b: Var) -> Var {
        let v = self.value(a).matmul(self.value(b));
        self.push(v, Op::MatMul(a, b))
    }

    pub fn matmul_nt(&mut self, a: Var, b: Var) -> Var {
        let v = self.value(a).matmul(&self.value(b).transpose());
        self.push(v, Op::MatMulNT(a, b))
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let mut v = self.value(a).clone();
        v.add_assign(self.value(b));
        self.push(v, Op::Add(a, b))
    }

    pub fn add_row(&mut self, a: Var, b: Var) -> Var {
        let bias = self.value(b);
        assert_eq!(
            (bias.rows, bias.cols),
            (1, self.value(a).cols),
            "bias shape"
        );
        let mut v = self.value(a).clone();
        let cols = v.cols;
        v.data.par_chunks_mut(cols).for_each(|row| {
            for (x, &y) in row.iter_mut().zip(&bias.data) {
                *x += y;
            }
        });
        self.push(v, Op::AddRow(a, b))
    }

    pub fn scale(&mut self, a: Var, s: T) -> Var {
        let v = self.value(a).map(|x| x * s);
        self.push(v, Op::Scale(a, s))
    }

    pub fn layer_norm(&mut self, x: Var, gamma: Var, beta: Var) -> Var {
        let xm = self.value(x);
        let (g, b) = (&self.value(gamma).data, &self.value(beta).data);
        let cols = xm.cols;
        let mut xhat = Mat::zeros(xm.rows, cols);
        let mut out = Mat::zeros(xm.rows, cols);
        // Each row is independent: in parallel, one row per task.
        let inv_std: Vec<T> = xhat
            .data
            .par_chunks_mut(cols)
            .zip(out.data.par_chunks_mut(cols))
            .zip(xm.data.par_chunks(cols))
            .map(|((hrow, orow), row)| layer_norm_row(row, g, b, hrow, orow))
            .collect();
        self.push(
            out,
            Op::LayerNorm {
                x,
                gamma,
                beta,
                xhat,
                inv_std,
            },
        )
    }

    pub fn gelu(&mut self, a: Var) -> Var {
        let v = self.value(a).map(gelu);
        self.push(v, Op::Gelu(a))
    }

    pub fn gather(&mut self, table: Var, idx: &[usize]) -> Var {
        let t = self.value(table);
        let mut v = Mat::zeros(idx.len(), t.cols);
        for (r, &i) in idx.iter().enumerate() {
            v.row_mut(r).copy_from_slice(t.row(i));
        }
        self.push(v, Op::Gather(table, idx.to_vec()))
    }

    /// Multi-head causal attention. q, k, v: (B·T) × d, with T = `seq`.
    pub fn attention(&mut self, q: Var, k: Var, v: Var, seq: usize, heads: usize) -> Var {
        let (qm, km, vm) = (self.value(q), self.value(k), self.value(v));
        let d = qm.cols;
        assert!(qm.rows % seq == 0 && d % heads == 0, "attention shapes");
        let (batch, hs) = (qm.rows / seq, d / heads);
        let scale = T::one() / T::from_f64(hs as f64).sqrt();
        // One unit of work per (sequence, head), in parallel.
        let units: Vec<(Mat<T>, Mat<T>)> = (0..batch * heads)
            .into_par_iter()
            .map(|u| {
                let (b, h) = (u / heads, u % heads);
                let (qb, kb, vb) = (
                    block(qm, b * seq, h * hs, seq, hs),
                    block(km, b * seq, h * hs, seq, hs),
                    block(vm, b * seq, h * hs, seq, hs),
                );
                // Scores and causal softmax, row by row: row i sees 0..=i.
                let mut w = Mat::zeros(seq, seq);
                for i in 0..seq {
                    let mut max = T::from_f64(f64::NEG_INFINITY);
                    for j in 0..=i {
                        let mut s = T::zero();
                        for c in 0..hs {
                            s += qb.data[i * hs + c] * kb.data[j * hs + c];
                        }
                        let s = s * scale;
                        w.data[i * seq + j] = s;
                        max = max.max(s);
                    }
                    let mut total = T::zero();
                    for j in 0..=i {
                        let e = (w.data[i * seq + j] - max).exp();
                        w.data[i * seq + j] = e;
                        total += e;
                    }
                    for j in 0..=i {
                        w.data[i * seq + j] = w.data[i * seq + j] / total;
                    }
                }
                let o = w.matmul(&vb);
                (o, w)
            })
            .collect();
        let mut out = Mat::zeros(qm.rows, d);
        let mut weights = Vec::with_capacity(units.len());
        for (u, (o, w)) in units.into_iter().enumerate() {
            put_block(&mut out, &o, (u / heads) * seq, (u % heads) * hs);
            weights.push(w);
        }
        self.push(
            out,
            Op::Attention {
                q,
                k,
                v,
                seq,
                heads,
                weights,
            },
        )
    }

    pub fn cross_entropy(&mut self, logits: Var, targets: &[Option<usize>]) -> Var {
        let l = self.value(logits);
        assert_eq!(targets.len(), l.rows, "one target (or None) per row");
        let mut probs = Mat::zeros(l.rows, l.cols);
        let (mut loss, mut n) = (0.0f64, 0usize);
        for (r, t) in targets.iter().enumerate() {
            let row = l.row(r);
            let max = row
                .iter()
                .copied()
                .fold(T::from_f64(f64::NEG_INFINITY), T::max);
            let mut total = T::zero();
            for (p, &x) in probs.row_mut(r).iter_mut().zip(row) {
                *p = (x - max).exp();
                total += *p;
            }
            for p in probs.row_mut(r) {
                *p = *p / total;
            }
            if let Some(t) = *t {
                loss -= probs.data[r * l.cols + t].to_f64().ln();
                n += 1;
            }
        }
        let v = Mat::from_vec(1, 1, vec![T::from_f64(loss / n.max(1) as f64)]);
        self.push(
            v,
            Op::CrossEntropy {
                logits,
                targets: targets.to_vec(),
                probs,
            },
        )
    }

    #[allow(dead_code)]
    pub fn weighted_sum(&mut self, a: Var, w: Mat<T>) -> Var {
        let am = self.value(a);
        assert_eq!((am.rows, am.cols), (w.rows, w.cols), "weights shape");
        let mut s = T::zero();
        for (&x, &y) in am.data.iter().zip(&w.data) {
            s += x * y;
        }
        self.push(Mat::from_vec(1, 1, vec![s]), Op::WeightedSum(a, w))
    }

    // ----- backward

    fn accumulate(&mut self, v: Var, g: Mat<T>) {
        match &mut self.grads[v.0] {
            Some(acc) => acc.add_assign(&g),
            slot @ None => *slot = Some(g),
        }
    }

    /// Compute every entry's gradient with respect to `loss` (1 × 1).
    pub fn backward(&mut self, loss: Var) {
        assert_eq!(
            (self.value(loss).rows, self.value(loss).cols),
            (1, 1),
            "backward needs a 1 × 1 loss"
        );
        self.grads = (0..self.nodes.len()).map(|_| None).collect();
        self.grads[loss.0] = Some(Mat::from_vec(1, 1, vec![T::one()]));
        for i in (0..=loss.0).rev() {
            let Some(g) = self.grads[i].take() else {
                continue;
            };
            for (input, contribution) in self.backward_rule(i, &g) {
                self.accumulate(input, contribution);
            }
            self.grads[i] = Some(g);
        }
    }

    fn backward_rule(&self, i: usize, g: &Mat<T>) -> Vec<(Var, Mat<T>)> {
        let val = |v: Var| &self.nodes[v.0].value;
        match &self.nodes[i].op {
            Op::Leaf => vec![],
            // C = A·B:  dA = dC·Bᵀ,  dB = Aᵀ·dC
            Op::MatMul(a, b) => vec![
                (*a, g.matmul(&val(*b).transpose())),
                (*b, val(*a).matmul_tn(g)),
            ],
            // C = A·Bᵀ:  dA = dC·B,  dB = dCᵀ·A
            Op::MatMulNT(a, b) => {
                vec![(*a, g.matmul(val(*b))), (*b, g.matmul_tn(val(*a)))]
            }
            Op::Add(a, b) => vec![(*a, g.clone()), (*b, g.clone())],
            // C = A + b (every row):  dA = dC,  db = column sums of dC
            Op::AddRow(a, b) => vec![(*a, g.clone()), (*b, col_sums(g))],
            // C = s·A:  dA = s·dC
            Op::Scale(a, s) => {
                let s = *s;
                vec![(*a, g.map(|x| x * s))]
            }
            // y = γ ⊙ x̂ + β:  dγ = Σ dy ⊙ x̂,  dβ = Σ dy, and with d = dy ⊙ γ,
            // dx = (d − mean(d) − x̂ · mean(d ⊙ x̂)) / σ, row by row
            Op::LayerNorm {
                x,
                gamma,
                beta,
                xhat,
                inv_std,
            } => {
                let gm = &val(*gamma).data;
                let cols = xhat.cols;
                let n = T::from_f64(cols as f64);
                let mut dx = Mat::zeros(xhat.rows, cols);
                // dx row by row, in parallel (rows are independent).
                dx.data
                    .par_chunks_mut(cols)
                    .zip(g.data.par_chunks(cols))
                    .zip(xhat.data.par_chunks(cols))
                    .zip(inv_std.par_iter())
                    .for_each(|(((dxr, gr), hr), &inv)| {
                        let (mut md, mut mdh) = (T::zero(), T::zero());
                        for j in 0..cols {
                            let d = gr[j] * gm[j];
                            md += d;
                            mdh += d * hr[j];
                        }
                        let (md, mdh) = (md / n, mdh / n);
                        for j in 0..cols {
                            dxr[j] = (gr[j] * gm[j] - md - hr[j] * mdh) * inv;
                        }
                    });
                // dγ and dβ sum over rows, in a fixed order.
                let dgamma = col_sums(&g.zip_map(xhat, |a, b| a * b));
                vec![(*x, dx), (*gamma, dgamma), (*beta, col_sums(g))]
            }
            // C = GELU(A):  dA = dC ⊙ GELU′(A)
            Op::Gelu(a) => vec![(*a, g.zip_map(val(*a), |d, x| d * gelu_slope(x)))],
            // Each row of dC goes back to the row it came from, adding up.
            Op::Gather(a, idx) => {
                let am = val(*a);
                let mut da = Mat::zeros(am.rows, am.cols);
                for (r, &src) in idx.iter().enumerate() {
                    for (d, &x) in da.row_mut(src).iter_mut().zip(g.row(r)) {
                        *d += x;
                    }
                }
                vec![(*a, da)]
            }
            // Per (sequence, head), with A its weights and dO its slice of dC:
            //   dV = Aᵀ·dO;  dA = dO·Vᵀ;  dS = A ⊙ (dA − rowsum(A ⊙ dA));
            //   dQ = dS·K / √h;  dK = dSᵀ·Q / √h
            Op::Attention {
                q,
                k,
                v,
                seq,
                heads,
                weights,
            } => {
                let (qm, km, vm) = (val(*q), val(*k), val(*v));
                let (seq, heads) = (*seq, *heads);
                let d = qm.cols;
                let hs = d / heads;
                let scale = T::one() / T::from_f64(hs as f64).sqrt();
                let units: Vec<(Mat<T>, Mat<T>, Mat<T>)> = weights
                    .par_iter()
                    .enumerate()
                    .map(|(u, w)| {
                        let (r0, c0) = ((u / heads) * seq, (u % heads) * hs);
                        let (qb, kb, vb) = (
                            block(qm, r0, c0, seq, hs),
                            block(km, r0, c0, seq, hs),
                            block(vm, r0, c0, seq, hs),
                        );
                        let dout = block(g, r0, c0, seq, hs);
                        let dv = w.transpose().matmul(&dout);
                        let da = dout.matmul(&vb.transpose());
                        let mut ds = Mat::zeros(seq, seq);
                        for i in 0..seq {
                            let mut dot = T::zero();
                            for j in 0..=i {
                                dot += w.data[i * seq + j] * da.data[i * seq + j];
                            }
                            for j in 0..=i {
                                ds.data[i * seq + j] =
                                    w.data[i * seq + j] * (da.data[i * seq + j] - dot) * scale;
                            }
                        }
                        let dq = ds.matmul(&kb);
                        let dk = ds.transpose().matmul(&qb);
                        (dq, dk, dv)
                    })
                    .collect();
                let mut dq = Mat::zeros(qm.rows, d);
                let mut dk = Mat::zeros(km.rows, d);
                let mut dv = Mat::zeros(vm.rows, d);
                for (u, (bq, bk, bv)) in units.iter().enumerate() {
                    let (r0, c0) = ((u / heads) * seq, (u % heads) * hs);
                    put_block(&mut dq, bq, r0, c0);
                    put_block(&mut dk, bk, r0, c0);
                    put_block(&mut dv, bv, r0, c0);
                }
                vec![(*q, dq), (*k, dk), (*v, dv)]
            }
            // dLogits = (probabilities − one-hot) / n on rows with a target.
            Op::CrossEntropy {
                logits,
                targets,
                probs,
            } => {
                let n = targets.iter().filter(|t| t.is_some()).count().max(1);
                let scale = g.data[0] / T::from_f64(n as f64);
                let mut d = probs.clone();
                for (r, t) in targets.iter().enumerate() {
                    match t {
                        None => d.row_mut(r).fill(T::zero()),
                        Some(t) => {
                            let p = d.data[r * d.cols + t];
                            d.data[r * d.cols + t] = p - T::one();
                        }
                    }
                }
                vec![(*logits, d.map(|x| x * scale))]
            }
            Op::WeightedSum(a, w) => vec![(*a, w.map(|x| x * g.data[0]))],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn rand(rows: usize, cols: usize, seed: u64) -> Mat<f64> {
        let mut rng = Rng::new(seed);
        Mat::from_vec(rows, cols, (0..rows * cols).map(|_| rng.normal()).collect())
    }

    /// Finite-difference check in f64: build a loss from leaves holding
    /// `inputs`, compare the tape's gradient for every input element with
    /// (loss(x + ε) − loss(x − ε)) / 2ε. Returns the worst relative error.
    fn check(inputs: &[Mat<f64>], build: impl Fn(&mut Tape<f64>, &[Var]) -> Var) -> f64 {
        let run = |ms: &[Mat<f64>]| {
            let mut t = Tape::new();
            let vars: Vec<Var> = ms.iter().map(|m| t.leaf(m.clone())).collect();
            let loss = build(&mut t, &vars);
            (t, vars, loss)
        };
        let (mut t, vars, loss) = run(inputs);
        t.backward(loss);
        let grads: Vec<Mat<f64>> = vars.iter().map(|&v| t.take_grad(v)).collect();
        let eps = 1e-6;
        let mut worst: f64 = 0.0;
        for (k, m) in inputs.iter().enumerate() {
            for i in 0..m.data.len() {
                let mut probe = inputs.to_vec();
                probe[k].data[i] += eps;
                let (tu, _, lu) = run(&probe);
                probe[k].data[i] -= 2.0 * eps;
                let (td, _, ld) = run(&probe);
                let numeric = (tu.value(lu).data[0] - td.value(ld).data[0]) / (2.0 * eps);
                let a = grads[k].data[i];
                // Relative error, with a floor of 1e-3 in the denominator:
                // for a tiny gradient, the finite difference's own rounding
                // (about 1e-10) would otherwise look like a large relative
                // error. A wrong backward rule gives errors far above 1e-3.
                worst = worst.max((numeric - a).abs() / (numeric.abs() + a.abs() + 1e-3));
            }
        }
        worst
    }

    fn reduce(t: &mut Tape<f64>, c: Var) -> Var {
        let (r, cl) = (t.value(c).rows, t.value(c).cols);
        t.weighted_sum(c, rand(r, cl, 99))
    }

    const TOL: f64 = 1e-6;

    #[test]
    fn matmuls() {
        let e = check(&[rand(5, 4, 1), rand(4, 3, 2), rand(6, 3, 3)], |t, v| {
            let c = t.matmul(v[0], v[1]); // 5 × 3
            let d = t.matmul_nt(c, v[2]); // 5 × 6
            reduce(t, d)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn scale() {
        let e = check(&[rand(3, 4, 1), rand(4, 2, 2)], |t, v| {
            let m = t.matmul(v[0], v[1]);
            let s = t.scale(m, 2.5);
            reduce(t, s)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn add_bias_gelu() {
        let e = check(&[rand(4, 5, 1), rand(4, 5, 2), rand(1, 5, 3)], |t, v| {
            let a = t.add(v[0], v[1]);
            let b = t.add_row(a, v[2]);
            let g = t.gelu(b);
            reduce(t, g)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn layer_norm() {
        let e = check(&[rand(4, 6, 1), rand(1, 6, 2), rand(1, 6, 3)], |t, v| {
            let y = t.layer_norm(v[0], v[1], v[2]);
            reduce(t, y)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn gather_with_repeats() {
        let e = check(&[rand(5, 3, 1)], |t, v| {
            let g = t.gather(v[0], &[4, 1, 1, 0, 1]);
            reduce(t, g)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn attention_over_a_batch_with_several_heads() {
        // B = 2 sequences of T = 4 tokens, d = 6, 3 heads of size 2.
        let e = check(&[rand(8, 6, 1), rand(8, 6, 2), rand(8, 6, 3)], |t, v| {
            let o = t.attention(v[0], v[1], v[2], 4, 3);
            reduce(t, o)
        });
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn attention_is_causal_and_sequences_dont_mix() {
        // Changing a later token, or a token of the other sequence, must not
        // change a row's output.
        let (q, k, v) = (rand(8, 4, 1), rand(8, 4, 2), rand(8, 4, 3));
        let out = |k: &Mat<f64>, v: &Mat<f64>| {
            let mut t = Tape::new();
            let (a, b, c) = (t.leaf(q.clone()), t.leaf(k.clone()), t.leaf(v.clone()));
            let o = t.attention(a, b, c, 4, 2);
            t.value(o).clone()
        };
        let base = out(&k, &v);
        let (mut k2, mut v2) = (k.clone(), v.clone());
        for c in 0..4 {
            k2.data[3 * 4 + c] += 1.0; // last token of sequence 1
            v2.data[3 * 4 + c] -= 2.0;
        }
        let changed = out(&k2, &v2);
        for r in 0..3 {
            assert_eq!(base.row(r), changed.row(r), "row {r} saw a later token");
        }
        for r in 4..8 {
            assert_eq!(
                base.row(r),
                changed.row(r),
                "row {r} saw the other sequence"
            );
        }
        assert_ne!(base.row(3), changed.row(3));
    }

    #[test]
    fn cross_entropy_skips_rows_without_targets() {
        let targets = [Some(2), None, Some(0), Some(1)];
        let e = check(&[rand(4, 3, 1)], |t, v| t.cross_entropy(v[0], &targets));
        assert!(e < TOL, "{e}");
    }

    #[test]
    fn f32_and_f64_agree() {
        // The same computation in both precisions agrees to f32's precision.
        let (x, w) = (rand(6, 5, 1), rand(5, 4, 2));
        let run64 = {
            let mut t = Tape::<f64>::new();
            let (a, b) = (t.leaf(x.clone()), t.leaf(w.clone()));
            let c = t.matmul(a, b);
            let g = t.gelu(c);
            t.value(g).clone()
        };
        let run32 = {
            let mut t = Tape::<f32>::new();
            let (a, b) = (t.leaf(x.convert()), t.leaf(w.convert()));
            let c = t.matmul(a, b);
            let g = t.gelu(c);
            t.value(g).convert::<f64>()
        };
        for (a, b) in run64.data.iter().zip(&run32.data) {
            assert!((a - b).abs() < 1e-5 * (1.0 + a.abs()), "{a} vs {b}");
        }
    }
}
