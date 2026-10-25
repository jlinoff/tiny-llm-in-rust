//! config.toml: one file, one section per step. `build` reads only
//! `[model]`; the other sections belong to later steps and are ignored here.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// The `[model]` hyperparameters: the model's shape (PLAN.md,
/// "Configuration"). V is not here: it comes from the tokenizer.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// The model width: the length of every token's vector.
    pub d_model: usize,
    /// Number of transformer blocks.
    pub layers: usize,
    /// Attention heads per block; must divide d_model.
    pub heads: usize,
    /// The most tokens the model sees at once (T).
    pub context: usize,
    /// The FFN's inner width, conventionally 4 × d_model.
    pub d_ff: usize,
    /// Seed for the random initial weights.
    pub seed: u64,
}

#[derive(Deserialize)]
struct File {
    model: ModelConfig,
}

impl ModelConfig {
    /// Read `[model]` from a config file, and check it makes sense.
    pub fn load(path: &Path) -> Result<ModelConfig, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("can't read config {}: {e}", path.display()))?;
        let file: File =
            toml::from_str(&text).map_err(|e| format!("config {}: {e}", path.display()))?;
        file.model.check()?;
        Ok(file.model)
    }

    pub fn check(&self) -> Result<(), String> {
        let positive = [
            ("d_model", self.d_model),
            ("layers", self.layers),
            ("heads", self.heads),
            ("context", self.context),
            ("d_ff", self.d_ff),
        ];
        for (name, v) in positive {
            if v == 0 {
                return Err(format!("[model] {name} must be at least 1"));
            }
        }
        if self.d_model % self.heads != 0 {
            return Err(format!(
                "[model] heads ({}) must divide d_model ({})",
                self.heads, self.d_model
            ));
        }
        Ok(())
    }

    pub fn head_size(&self) -> usize {
        self.d_model / self.heads
    }

    /// The `[model]` section as TOML text: what the checkpoint stores, so
    /// later steps read the shapes from it, never from the config file.
    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("a flat struct of numbers always serializes")
    }

    pub fn from_toml(text: &str) -> Result<ModelConfig, String> {
        let c: ModelConfig = toml::from_str(text).map_err(|e| format!("stored [model]: {e}"))?;
        c.check()?;
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ModelConfig {
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
    fn round_trips_through_toml() {
        let c = sample();
        assert_eq!(ModelConfig::from_toml(&c.to_toml()).unwrap(), c);
    }

    #[test]
    fn rejects_heads_that_dont_divide_d_model() {
        let c = ModelConfig {
            heads: 3,
            ..sample()
        };
        assert!(c.check().is_err());
    }

    #[test]
    fn rejects_unknown_keys_in_model() {
        let text =
            "d_model = 8\nlayers = 1\nheads = 1\ncontext = 4\nd_ff = 32\nseed = 1\ndmodel = 9\n";
        assert!(ModelConfig::from_toml(text).is_err());
    }
}
