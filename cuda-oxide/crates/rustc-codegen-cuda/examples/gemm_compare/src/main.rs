/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust BF16 GEMM versus a live cuBLAS reference, including Ampere Tensor Cores.

mod cublas;
mod kernels;

use cublas::Cublas;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use half::bf16;
use std::io::Write;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const ABS_TOL: f32 = 0.02;
const REL_TOL: f32 = 0.01;
const SEED_A: u32 = 0x1234_5678;
const SEED_B: u32 = 0x8765_4321;

#[derive(Clone, Copy, Debug)]
struct Problem {
    m: u32,
    n: u32,
    k: u32,
}

impl Problem {
    fn new(m: u32, n: u32, k: u32) -> Result<Self> {
        if [m, n, k].iter().any(|&x| x == 0 || x > i32::MAX as u32) {
            return Err("dimensions must be positive and fit the cuBLAS i32 ABI".into());
        }
        for (x, y) in [(m, k), (k, n), (m, n)] {
            x.checked_mul(y)
                .ok_or("matrix element counts must fit u32 kernel offsets")?;
        }
        Ok(Self { m, n, k })
    }

    fn flops(self) -> f64 {
        2.0 * self.m as f64 * self.n as f64 * self.k as f64
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kernel {
    Cublas,
    Naive,
    Tiled,
    Tensor,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Self::Cublas => "cublas",
            Self::Naive => "naive",
            Self::Tiled => "tiled",
            Self::Tensor => "tensor",
        }
    }
}

struct Options {
    problems: Vec<Problem>,
    kernels: Vec<Kernel>,
    warmup: usize,
    iterations: usize,
    rounds: usize,
    cooldown_ms: u64,
    verify_only: bool,
    csv: Option<String>,
}

fn options() -> Result<Option<Options>> {
    let mut o = Options {
        problems: [256, 512, 1024]
            .map(|s| Problem::new(s, s, s).unwrap())
            .to_vec(),
        kernels: vec![Kernel::Cublas, Kernel::Naive, Kernel::Tiled, Kernel::Tensor],
        warmup: 10,
        iterations: 30,
        rounds: 3,
        cooldown_ms: 100,
        verify_only: false,
        csv: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!(
                "gemm_compare: BF16 A[M,K] * B[K,N] -> C[M,N], FP32 accumulation\n\
                --sizes 256,512,1024   Square benchmark sizes\n\
                --shape M,N,K         One rectangular benchmark\n\
                --kernels naive,tiled,tensor  Rust kernels (cuBLAS always runs)\n\
                --warmup 10 --iterations 30 --rounds 3\n\
                --cooldown-ms 100      Pause before each timed batch\n\
                --verify-only         Check edge cases and requested sizes, no timing\n\
                --csv PATH            Write measurements with GPU/cuBLAS metadata"
            );
            return Ok(None);
        }
        if arg == "--verify-only" {
            o.verify_only = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--sizes" => {
                o.problems = value
                    .split(',')
                    .map(|s| {
                        let size = s.parse()?;
                        Problem::new(size, size, size)
                    })
                    .collect::<Result<_>>()?;
            }
            "--shape" => {
                let dims = value
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<Vec<u32>, _>>()?;
                if dims.len() != 3 {
                    return Err("--shape needs M,N,K".into());
                }
                o.problems = vec![Problem::new(dims[0], dims[1], dims[2])?];
            }
            "--kernels" => {
                o.kernels = vec![Kernel::Cublas];
                for name in value.split(',') {
                    let kernel = match name {
                        "naive" => Kernel::Naive,
                        "tiled" => Kernel::Tiled,
                        "tensor" => Kernel::Tensor,
                        _ => return Err(format!("unknown Rust kernel: {name}").into()),
                    };
                    if !o.kernels.contains(&kernel) {
                        o.kernels.push(kernel);
                    }
                }
            }
            "--warmup" => o.warmup = value.parse()?,
            "--iterations" => o.iterations = value.parse()?,
            "--rounds" => o.rounds = value.parse()?,
            "--cooldown-ms" => o.cooldown_ms = value.parse()?,
            "--csv" => o.csv = Some(value),
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    if o.iterations == 0 || o.rounds == 0 {
        return Err("iterations and rounds must be positive".into());
    }
    if o.verify_only && o.csv.is_some() {
        return Err("--csv requires a timed run".into());
    }
    Ok(Some(o))
}

fn input(len: usize, mut state: u32) -> Vec<u16> {
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = (state & 0xffff) as f32 / 32768.0 - 1.0;
            bf16::from_f32(value).to_bits()
        })
        .collect()
}

#[derive(Clone, Copy, Default)]
struct Errors {
    max_abs: f32,
    max_rel: f32,
}

fn check_value(
    label: &str,
    i: usize,
    expected: f32,
    actual: f32,
    errors: &mut Errors,
) -> Result<()> {
    let abs = (expected - actual).abs();
    if !expected.is_finite() || !actual.is_finite() || abs > ABS_TOL + REL_TOL * expected.abs() {
        return Err(format!("{label} mismatch at {i}: expected {expected}, got {actual}").into());
    }
    errors.max_abs = errors.max_abs.max(abs);
    errors.max_rel = errors.max_rel.max(abs / expected.abs().max(1e-6));
    Ok(())
}

fn compare(label: &str, expected: &[u16], actual: &[u16]) -> Result<Errors> {
    if expected.len() != actual.len() {
        return Err("output sizes differ".into());
    }
    let mut errors = Errors::default();
    for (i, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        check_value(
            label,
            i,
            bf16::from_bits(e).to_f32(),
            bf16::from_bits(a).to_f32(),
            &mut errors,
        )?;
    }
    Ok(errors)
}

fn check_cpu(p: Problem, a: &[u16], b: &[u16], reference: &[u16]) -> Result<()> {
    let all = p.m as u64 * p.n as u64 * p.k as u64 <= 2_000_000;
    let count = if all { reference.len() } else { 33 };
    let mut errors = Errors::default();
    for sample in 0..count {
        let i = if all {
            sample
        } else {
            sample * (reference.len() - 1) / (count - 1)
        };
        let row = i / p.n as usize;
        let col = i % p.n as usize;
        let mut sum = 0.0f64;
        for kk in 0..p.k as usize {
            sum += bf16::from_bits(a[row * p.k as usize + kk]).to_f64()
                * bf16::from_bits(b[kk * p.n as usize + col]).to_f64();
        }
        let expected = bf16::from_f64(sum).to_f32();
        check_value(
            "cuBLAS vs CPU",
            i,
            expected,
            bf16::from_bits(reference[i]).to_f32(),
            &mut errors,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch(
    kernel: Kernel,
    module: &kernels::gemm::LoadedModule,
    blas: &Cublas,
    stream: &CudaStream,
    p: Problem,
    a: &DeviceBuffer<u16>,
    b: &DeviceBuffer<u16>,
    c: &mut DeviceBuffer<u16>,
) -> Result<()> {
    if kernel == Kernel::Cublas {
        return blas.run(p, a, b, c);
    }
    let (tile, block) = if kernel == Kernel::Tensor {
        (32, (128, 1, 1))
    } else {
        (16, (16, 16, 1))
    };
    let config = LaunchConfig {
        grid_dim: (p.n.div_ceil(tile), p.m.div_ceil(tile), 1),
        block_dim: block,
        shared_mem_bytes: 0,
    };
    let ap = a.cu_deviceptr() as *const u16;
    let bp = b.cu_deviceptr() as *const u16;
    let cp = c.cu_deviceptr() as *mut u16;
    // SAFETY: exact block shapes match the kernel contracts. Buffers cover
    // M*K/K*N/M*N, are disjoint, and live through the final synchronization.
    // Edge CTAs are zero-padded, and all complete warps participate in MMA.
    unsafe {
        match kernel {
            Kernel::Naive => module.naive(stream, config, p.m, p.n, p.k, ap, bp, cp)?,
            Kernel::Tiled => module.tiled(stream, config, p.m, p.n, p.k, ap, bp, cp)?,
            Kernel::Tensor => module.tensor(stream, config, p.m, p.n, p.k, ap, bp, cp)?,
            Kernel::Cublas => unreachable!(),
        }
    }
    Ok(())
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

#[allow(clippy::too_many_arguments)]
fn run_problem(
    p: Problem,
    o: &Options,
    ctx: &std::sync::Arc<CudaContext>,
    stream: &CudaStream,
    module: &kernels::gemm::LoadedModule,
    blas: &Cublas,
    timed: bool,
    csv: &mut Option<std::fs::File>,
) -> Result<()> {
    let a = input((p.m * p.k) as usize, SEED_A);
    let b = input((p.k * p.n) as usize, SEED_B);
    let ad = DeviceBuffer::from_host(stream, &a)?;
    let bd = DeviceBuffer::from_host(stream, &b)?;
    // NaN detects missing stores, including stores omitted on partial tiles.
    let poison = vec![0x7fc0u16; (p.m * p.n) as usize];
    let mut outputs = o
        .kernels
        .iter()
        .map(|_| DeviceBuffer::from_host(stream, &poison))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    blas.run(p, &ad, &bd, &mut outputs[0])?;
    let reference = outputs[0].to_host_vec(stream)?;
    check_cpu(p, &a, &b, &reference)?;
    let mut errors = vec![Errors::default()];
    for (i, &kernel) in o.kernels.iter().enumerate().skip(1) {
        launch(kernel, module, blas, stream, p, &ad, &bd, &mut outputs[i])?;
        errors.push(compare(
            kernel.name(),
            &reference,
            &outputs[i].to_host_vec(stream)?,
        )?);
    }
    println!(
        "PASS: {}x{}x{} — all {} Rust kernels checked against cuBLAS; CPU reference checked",
        p.m,
        p.n,
        p.k,
        o.kernels.len() - 1
    );
    if !timed {
        return Ok(());
    }
    let start = ctx.new_event(Some(cuda_bindings::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let stop = ctx.new_event(Some(cuda_bindings::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let mut samples = vec![Vec::with_capacity(o.rounds); o.kernels.len()];
    for round in 0..o.rounds {
        // Rotate order between rounds to reduce systematic thermal/order bias.
        for offset in 0..o.kernels.len() {
            let i = (round + offset) % o.kernels.len();
            let kernel = o.kernels[i];
            for _ in 0..o.warmup {
                launch(kernel, module, blas, stream, p, &ad, &bd, &mut outputs[i])?;
            }
            stream.synchronize()?;
            std::thread::sleep(std::time::Duration::from_millis(o.cooldown_ms));
            start.record(stream)?;
            for _ in 0..o.iterations {
                launch(kernel, module, blas, stream, p, &ad, &bd, &mut outputs[i])?;
            }
            stop.record(stream)?;
            let ms = start.elapsed_ms(&stop)? as f64 / o.iterations as f64;
            if !ms.is_finite() || ms <= 0.0 {
                return Err("invalid CUDA event timing".into());
            }
            samples[i].push(ms);
        }
    }
    let medians: Vec<_> = samples.iter_mut().map(|s| median(s)).collect();
    println!("  kernel        median ms     TFLOPS   % cuBLAS   max abs error");
    for (i, &kernel) in o.kernels.iter().enumerate() {
        // Also check the final timed result; beta=0 must make repeated launches
        // independent of the previous output buffer contents.
        compare(kernel.name(), &reference, &outputs[i].to_host_vec(stream)?)?;
        let ms = medians[i];
        let tflops = p.flops() / (ms * 1e9);
        let percent = 100.0 * medians[0] / ms;
        println!(
            "  {:<10} {:>11.6} {:>10.4} {:>9.2} {:>15.6}",
            kernel.name(),
            ms,
            tflops,
            percent,
            errors[i].max_abs
        );
        if let Some(file) = csv {
            let gpu = ctx.device_name()?.replace('"', "\"\"");
            let (major, minor) = ctx.compute_capability()?;
            writeln!(
                file,
                "\"{gpu}\",sm_{major}{minor},{},{},{},{},{},bf16,fp32_reduction,{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.6},{:.4},{},{}",
                blas.version,
                p.m,
                p.n,
                p.k,
                kernel.name(),
                o.warmup,
                o.iterations,
                o.rounds,
                o.cooldown_ms,
                SEED_A,
                SEED_B,
                ms,
                samples[i][0],
                samples[i][samples[i].len() - 1],
                tflops,
                percent,
                errors[i].max_abs,
                errors[i].max_rel
            )?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let Some(o) = options()? else {
        return Ok(());
    };
    let ctx = CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    if major < 8 {
        println!(
            "skipping: BF16 Tensor Core comparison requires sm_80+ (device sm_{major}{minor})"
        );
        return Ok(());
    }
    let stream = ctx.default_stream();
    let module = kernels::gemm::load(&ctx)?;
    let blas = Cublas::new(stream.clone())?;
    println!(
        "GPU: {} (sm_{major}{minor}); cuBLAS: {}",
        ctx.device_name()?,
        blas.version
    );
    println!("BF16 input/output, FP32 accumulate/reduce; A/B/C row-major; alpha=1, beta=0");
    println!(
        "{} warmup, {} iterations, {} rounds, {} ms cooldown; median batch mean, CUDA events",
        o.warmup, o.iterations, o.rounds, o.cooldown_ms
    );
    let mut csv = if let Some(path) = &o.csv {
        let mut file = std::fs::File::create(path)?;
        writeln!(
            file,
            "gpu,arch,cublas_version,m,n,k,kernel,dtype,math_mode,warmup,iterations,rounds,cooldown_ms,seed_a,seed_b,median_ms,min_ms,max_ms,tflops,percent_cublas,max_abs_error,max_rel_error"
        )?;
        Some(file)
    } else {
        None
    };
    // Aligned, rectangular, tiny, and ragged cases cover fragment mapping,
    // K tails, and output edge stores. The CPU checks all their output cells.
    for (m, n, k) in [
        (1, 1, 1),
        (16, 8, 16),
        (32, 32, 32),
        (17, 19, 23),
        (63, 65, 37),
        (128, 96, 31),
        (1, 1024, 1024),
        (17, 19, 1025),
    ] {
        run_problem(
            Problem::new(m, n, k)?,
            &o,
            &ctx,
            &stream,
            &module,
            &blas,
            false,
            &mut None,
        )?;
    }
    for &p in &o.problems {
        run_problem(
            p,
            &o,
            &ctx,
            &stream,
            &module,
            &blas,
            !o.verify_only,
            &mut csv,
        )?;
    }
    println!("SUCCESS: Rust GEMM comparison completed");
    Ok(())
}
