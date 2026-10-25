//! `transformer bench`: how fast this machine multiplies f32 matrices, at the
//! shapes the model uses, with the model's own multi-threaded multiply. Training time is roughly 6·N·D floating-point
//! operations (N parameters, D tokens) divided by this speed, so measuring
//! it turns PLAN.md's time estimates into figures for this computer.

use crate::config::ModelConfig;
use crate::gpt::{forward, Params};
use crate::safetensors::Tensor;
use crate::tensor::Mat;
use std::time::Instant;

/// GFLOP/s for an n × k by k × m product on `threads` threads, repeated for
/// about `seconds`. Uses the model's own matrix multiply (src/tensor.rs).
fn measure(n: usize, k: usize, m: usize, threads: usize, seconds: f64) -> f64 {
    let a = Mat::<f32>::from_vec(n, k, (0..n * k).map(|i| (i % 7) as f32 * 0.1).collect());
    let b = Mat::<f32>::from_vec(k, m, (0..k * m).map(|i| (i % 5) as f32 * 0.1).collect());
    let flops = 2.0 * (n * k * m) as f64; // one multiply and one add each
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let _ = a.matmul(&b); // warm up
        let start = Instant::now();
        let mut reps = 0;
        while start.elapsed().as_secs_f64() < seconds {
            std::hint::black_box(a.matmul(&b));
            reps += 1;
        }
        flops * reps as f64 / start.elapsed().as_secs_f64() / 1e9
    })
}

/// A checkpoint's model, for timing a real training step.
pub struct Model {
    pub config: ModelConfig,
    pub vocab: usize,
    pub tensors: Vec<Tensor>,
}

pub fn run(model: Option<&Model>) {
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
    println!("(2 for the forward pass, 4 for backpropagation).");

    let Some(m) = model else {
        let n = 970_624;
        let tps = best * 1e9 / (6.0 * n as f64);
        println!(
            "\nAt {best:.0} GFLOP/s, PLAN.md's starting model ({n} parameters) would train at"
        );
        println!(
            "up to about {tps:.0} tokens per second, if the matrix products were all the work."
        );
        println!("Give --checkpoint to time a real training step instead.");
        return;
    };

    // A real step: forward and backward on a random batch, f32, all threads.
    let n: usize = m.tensors.iter().map(|t| t.data.len()).sum();
    let seq = m.config.context.min(128);
    let batch = 16;
    let p = Params::<f32>::from_tensors(&m.tensors);
    let mut rng = crate::rng::Rng::new(1);
    let ids: Vec<usize> = (0..batch * seq)
        .map(|_| (rng.next_u64() % m.vocab as u64) as usize)
        .collect();
    let targets: Vec<Option<usize>> = ids.iter().map(|&t| Some(t)).collect();
    let step = || {
        let mut f = forward(&p, &m.config, &ids, &targets, seq);
        f.tape.backward(f.loss);
        std::hint::black_box(f.tape.value(f.loss).data[0]);
    };
    step(); // warm up
    let start = Instant::now();
    let mut reps = 0;
    while start.elapsed().as_secs_f64() < 3.0 {
        step();
        reps += 1;
    }
    let secs = start.elapsed().as_secs_f64() / reps as f64;
    let tps = (batch * seq) as f64 / secs;
    let effective = 6.0 * n as f64 * tps / 1e9;
    println!("\nA real training step (forward and backward, no optimizer) for this model:");
    println!(
        "  {n} parameters, {batch} sequences × {seq} tokens: {:.0} ms per step",
        secs * 1000.0
    );
    println!(
        "  {tps:.0} tokens per second, about {effective:.0} GFLOP/s of useful work ({:.0}% of {best:.0})",
        100.0 * effective / best
    );
    println!(
        "  20 million tokens would take about {:.1} hours.",
        20e6 / tps / 3600.0
    );
}
