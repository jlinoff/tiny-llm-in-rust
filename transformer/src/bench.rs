//! `transformer bench`: how fast this machine multiplies f32 matrices, at the
//! shapes the model uses. Training time is roughly 6·N·D floating-point
//! operations (N parameters, D tokens) divided by this speed, so measuring
//! it turns PLAN.md's time estimates into figures for this computer.

use std::time::Instant;

/// C = A·B for row-major A (n × k) and B (k × m), on the rows from..to of
/// A and C. The inner loop runs along contiguous rows of B and C, which the
/// compiler turns into vector (SIMD) instructions.
fn matmul_rows(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    k: usize,
    m: usize,
    rows: std::ops::Range<usize>,
) {
    for (ci, r) in rows.enumerate() {
        let out = &mut c[ci * m..(ci + 1) * m];
        out.fill(0.0);
        for (p, &x) in a[r * k..(r + 1) * k].iter().enumerate() {
            for (o, &y) in out.iter_mut().zip(&b[p * m..(p + 1) * m]) {
                *o += x * y;
            }
        }
    }
}

/// C = A·B using `threads` threads, each taking a block of rows.
pub fn matmul(a: &[f32], b: &[f32], n: usize, k: usize, m: usize, threads: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * m];
    let per = n.div_ceil(threads.max(1));
    std::thread::scope(|s| {
        for (i, chunk) in c.chunks_mut(per * m).enumerate() {
            let start = i * per;
            let end = start + chunk.len() / m;
            s.spawn(move || matmul_rows(a, b, chunk, k, m, start..end));
        }
    });
    c
}

/// GFLOP/s for an n × k by k × m product, repeated for about `seconds`.
fn measure(n: usize, k: usize, m: usize, threads: usize, seconds: f64) -> f64 {
    let a: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * m).map(|i| (i % 5) as f32 * 0.1).collect();
    let flops = 2.0 * (n * k * m) as f64; // one multiply and one add each
    let _ = matmul(&a, &b, n, k, m, threads); // warm up
    let start = Instant::now();
    let mut reps = 0;
    while start.elapsed().as_secs_f64() < seconds {
        std::hint::black_box(matmul(&a, &b, n, k, m, threads));
        reps += 1;
    }
    flops * reps as f64 / start.elapsed().as_secs_f64() / 1e9
}

pub fn run(params: Option<usize>) {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("=== Matrix multiply speed, f32 ===");
    println!("{cores} hardware threads available.\n");
    // Shapes from a training step: 16 sequences × 128 tokens = 2,048 rows,
    // times the model's matrices (d = 128: d × d and d × 4d; d = 256 too).
    let shapes = [
        (2048, 128, 128),
        (2048, 128, 512),
        (2048, 512, 128),
        (2048, 256, 1024),
    ];
    println!("  shape (rows × k × cols)     1 thread     {cores} threads");
    let mut best: f64 = 0.0;
    for &(n, k, m) in &shapes {
        let one = measure(n, k, m, 1, 0.5);
        let all = measure(n, k, m, cores, 0.5);
        best = best.max(all);
        println!("  {n:>5} × {k:>4} × {m:>4}      {one:>7.1} GFLOP/s  {all:>7.1} GFLOP/s");
    }
    println!();
    println!("Training costs about 6 floating-point operations per parameter per token");
    println!("(2 for the forward pass, 4 for backpropagation). The matrix products are");
    println!("most of it; the rest (softmax, layer norm, the optimizer) adds overhead, so");
    println!("treat these as best cases.");
    let n = params.unwrap_or(970_624);
    let label = if params.is_some() {
        "this config"
    } else {
        "PLAN.md's starting model"
    };
    let tokens_per_sec = best * 1e9 / (6.0 * n as f64);
    println!("\nAt {best:.0} GFLOP/s, {label} ({n} parameters) would train at up to about");
    println!(
        "{:.0} tokens per second: 20 million tokens in about {:.1} hours.",
        tokens_per_sec,
        20e6 / tokens_per_sec / 3600.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threaded_matmul_matches_the_definition() {
        let (n, k, m) = (7, 5, 3);
        let a: Vec<f32> = (0..n * k).map(|i| i as f32 - 10.0).collect();
        let b: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.5).collect();
        for threads in [1, 2, 3, 8] {
            let c = matmul(&a, &b, n, k, m, threads);
            for r in 0..n {
                for col in 0..m {
                    let want: f32 = (0..k).map(|p| a[r * k + p] * b[p * m + col]).sum();
                    // Same additions in the same order: exactly equal.
                    assert_eq!(c[r * m + col], want, "threads {threads} at {r},{col}");
                }
            }
        }
    }
}
