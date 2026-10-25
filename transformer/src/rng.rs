//! A small, seedable random number generator, so every run is reproducible.

/// SplitMix64: fast, good-quality, and any seed works.
pub struct Rng(u64);

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64's scrambling function: every bit of the input affects every
/// bit of the output.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        mix(self.0)
    }

    /// A generator for one of many independent streams, such as one per
    /// training step. The stream number goes through the scrambling function
    /// before it's combined with the seed. Combining them directly (seed ^
    /// stream × GOLDEN) would be a mistake: the generator itself steps its
    /// state by GOLDEN, so stream k + 1 would start almost where stream k's
    /// second number came from, and the low bits (which `below` uses)
    /// would repeat between streams.
    pub fn stream(seed: u64, stream: u64) -> Rng {
        Rng(mix(seed ^ mix(stream.wrapping_add(GOLDEN))))
    }

    /// Uniform in (0, 1]: never exactly 0, so ln() below is safe.
    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// A uniform integer in 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A standard normal value (mean 0, variance 1), by the Box-Muller method.
    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
