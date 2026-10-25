//! The model's weights: their names and shapes (PLAN.md, open decision 3),
//! and their random initial values (step 3, build).
//!
//! The design is GPT-2's, which PLAN.md's parameter formula describes:
//! learned position embeddings, pre-norm layer normalization with a scale
//! and a shift, a bias on every matrix, GELU in the FFN, and an output layer
//! tied to the token embedding table (so there's no separate `output`).

use crate::config::ModelConfig;
use crate::rng::Rng;
use crate::safetensors::Tensor;

/// A one-line description stored in every checkpoint.
pub const ARCHITECTURE: &str = "GPT-2 style: learned position embeddings, pre-norm layer \
     normalization (scale and shift), biases on every matrix, GELU (tanh form) in the FFN, \
     output layer tied to token_embedding";

/// How the initial weights are drawn, stored in the checkpoint.
pub const INIT: &str = "matrices and embeddings normal(0, 0.02); attn.w_o and ffn.w2 \
     normal(0, 0.02/sqrt(2 * layers)); biases 0; layer norm gamma 1, beta 0";

/// What a weight is, which decides its initial value.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    /// Embeddings and most matrices: normal(0, 0.02).
    Normal,
    /// Matrices that write into the residual stream (W_o, W₂): scaled down by
    /// 1/√(2L), so the stream's size doesn't grow with the number of blocks.
    Residual,
    Zero,
    One,
}

/// Every weight's name and shape, in the order they're stored, for a model
/// with this shape and vocabulary size V.
pub fn layout(c: &ModelConfig, vocab: usize) -> Vec<(String, Vec<usize>)> {
    layout_with_kinds(c, vocab)
        .into_iter()
        .map(|(n, s, _)| (n, s))
        .collect()
}

fn layout_with_kinds(c: &ModelConfig, vocab: usize) -> Vec<(String, Vec<usize>, Kind)> {
    let (d, f) = (c.d_model, c.d_ff);
    let mut v = vec![
        ("token_embedding".to_string(), vec![vocab, d], Kind::Normal),
        (
            "position_embedding".to_string(),
            vec![c.context, d],
            Kind::Normal,
        ),
    ];
    for i in 0..c.layers {
        let b = |s: &str| format!("blocks.{i}.{s}");
        v.extend([
            (b("ln1.gamma"), vec![d], Kind::One),
            (b("ln1.beta"), vec![d], Kind::Zero),
            // Each d × d: the heads side by side, head k in columns
            // k·h … (k + 1)·h − 1.
            (b("attn.w_q"), vec![d, d], Kind::Normal),
            (b("attn.b_q"), vec![d], Kind::Zero),
            (b("attn.w_k"), vec![d, d], Kind::Normal),
            (b("attn.b_k"), vec![d], Kind::Zero),
            (b("attn.w_v"), vec![d, d], Kind::Normal),
            (b("attn.b_v"), vec![d], Kind::Zero),
            (b("attn.w_o"), vec![d, d], Kind::Residual),
            (b("attn.b_o"), vec![d], Kind::Zero),
            (b("ln2.gamma"), vec![d], Kind::One),
            (b("ln2.beta"), vec![d], Kind::Zero),
            (b("ffn.w1"), vec![d, f], Kind::Normal),
            (b("ffn.b1"), vec![f], Kind::Zero),
            (b("ffn.w2"), vec![f, d], Kind::Residual),
            (b("ffn.b2"), vec![d], Kind::Zero),
        ]);
    }
    v.extend([
        ("final_ln.gamma".to_string(), vec![d], Kind::One),
        ("final_ln.beta".to_string(), vec![d], Kind::Zero),
    ]);
    v
}

/// PLAN.md's parameter formula for this design (Configuration, "Counting
/// the parameters"): N = V·d + T·d + L·(4d² + 2·d·d_ff + 9d + d_ff) + 2d.
pub fn formula(c: &ModelConfig, vocab: usize) -> usize {
    let (d, f, l, t) = (c.d_model, c.d_ff, c.layers, c.context);
    vocab * d + t * d + l * (4 * d * d + 2 * d * f + 9 * d + f) + 2 * d
}

/// The random initial weights (step 3, build), reproducible from the seed.
pub fn init(c: &ModelConfig, vocab: usize) -> Vec<Tensor> {
    let mut rng = Rng::new(c.seed);
    let residual_std = 0.02 / ((2 * c.layers) as f64).sqrt();
    layout_with_kinds(c, vocab)
        .into_iter()
        .map(|(name, shape, kind)| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = match kind {
                Kind::Normal => (0..n).map(|_| (0.02 * rng.normal()) as f32).collect(),
                Kind::Residual => (0..n)
                    .map(|_| (residual_std * rng.normal()) as f32)
                    .collect(),
                Kind::Zero => vec![0.0; n],
                Kind::One => vec![1.0; n],
            };
            Tensor::new(&name, &shape, data)
        })
        .collect()
}

/// Check a checkpoint's tensors against the layout its stored config
/// implies: every expected name present with the right shape, and nothing
/// else. Returns a list of problems (empty if it all matches).
pub fn check_layout(tensors: &[Tensor], c: &ModelConfig, vocab: usize) -> Vec<String> {
    let mut problems = Vec::new();
    let expected = layout(c, vocab);
    for (name, shape) in &expected {
        match tensors.iter().find(|t| &t.name == name) {
            None => problems.push(format!("missing {name}")),
            Some(t) if &t.shape != shape => {
                problems.push(format!("{name}: shape {:?}, expected {shape:?}", t.shape))
            }
            Some(_) => {}
        }
    }
    for t in tensors {
        let known = expected.iter().any(|(n, _)| n == &t.name);
        let optimizer = t.name.starts_with("adam.");
        if !known && !optimizer {
            problems.push(format!("unexpected tensor {}", t.name));
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig {
            d_model: 128,
            layers: 4,
            heads: 4,
            context: 128,
            d_ff: 512,
            seed: 1,
        }
    }

    #[test]
    fn parameter_count_matches_the_plan_formula() {
        let c = config();
        let n: usize = init(&c, 1257).iter().map(|t| t.data.len()).sum();
        assert_eq!(n, formula(&c, 1257));
        assert_eq!(n, 970_624); // the figure PLAN.md gives for the starting model
    }

    #[test]
    fn gpt2_small_formula() {
        let c = ModelConfig {
            d_model: 768,
            layers: 12,
            heads: 12,
            context: 1024,
            d_ff: 3072,
            seed: 1,
        };
        assert_eq!(formula(&c, 50257), 124_439_808);
    }

    #[test]
    fn init_is_reproducible_and_has_the_planned_statistics() {
        let c = config();
        let (a, b) = (init(&c, 1257), init(&c, 1257));
        assert_eq!(a, b);
        let std = |t: &Tensor| {
            let n = t.data.len() as f64;
            let mean = t.data.iter().map(|&x| x as f64).sum::<f64>() / n;
            (t.data
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / n)
                .sqrt()
        };
        let get = |n: &str| a.iter().find(|t| t.name == n).unwrap();
        assert!((std(get("token_embedding")) - 0.02).abs() < 0.001);
        assert!((std(get("blocks.0.attn.w_o")) - 0.02 / 8f64.sqrt()).abs() < 0.0005);
        assert!(get("blocks.0.ln1.gamma").data.iter().all(|&x| x == 1.0));
        assert!(get("blocks.3.ffn.b1").data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn check_layout_names_the_problem() {
        let c = config();
        let mut t = init(&c, 1257);
        assert!(check_layout(&t, &c, 1257).is_empty());
        t.retain(|x| x.name != "blocks.2.ffn.w1");
        t[0].shape = vec![1000, 128];
        t[0].data.truncate(1000 * 128);
        let p = check_layout(&t, &c, 1257);
        assert!(p.iter().any(|s| s.contains("missing blocks.2.ffn.w1")));
        assert!(p.iter().any(|s| s.starts_with("token_embedding: shape")));
    }
}
