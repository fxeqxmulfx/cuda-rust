# gemm_compare

An initial Rust versus cuBLAS GEMM comparison for Ampere, including the RTX
3050 Laptop (`sm_86`, 4 GiB). All paths compute `C = A * B` from the same
row-major BF16 buffers, accumulate in FP32, and round the output to BF16.
`alpha = 1`, `beta = 0`; no transpose or host-side pre-tiling is timed.

The progression and use of a live cuBLAS reference are inspired by
[fast.cu](https://github.com/pranjalssh/fast.cu). Its H100 series uses
Hopper-specific instructions, so those kernels are not portable to an RTX
3050. This example implements Ampere kernels using the existing cuda-oxide
intrinsics; it does not vendor upstream code or reuse upstream TFLOPS claims.

| Path | Implementation | Output tile | Threads |
| --- | --- | --- | ---: |
| `naive` | One scalar FP32 dot product per thread | 16 x 16 | 256 |
| `tiled` | Cooperative BF16 shared-memory loads, scalar FP32 math | 16 x 16 | 256 |
| `tensor` | `ldmatrix` plus BF16 `mma.sync.m16n8k16`, four warps | 32 x 32 | 128 |
| `cublas` | Live `cublasGemmEx`, BF16 in/out, FP32 compute and reduction | Library selected | Library selected |

The Tensor Core kernel is a starting point: one shared stage, no `cp.async`,
double buffering, or register tiling across multiple K steps. Its first purpose
is to establish a correct, measurable Ampere baseline for further work.

## Run

From `cuda-oxide/`, with the pinned nightly and CUDA development environment:

```bash
cargo oxide run gemm_compare --arch sm_86

# Check all edge cases and a rectangular matrix without timing.
cargo oxide run gemm_compare --arch sm_86 -- --verify-only --shape 127,95,63

# 4096/8192 comparison with all three Rust kernels and cuBLAS.
cargo oxide run gemm_compare --arch sm_86 -- \
    --sizes 4096,8192 --warmup 2 --iterations 3 --rounds 3 \
    --csv gemm-large-results.csv
```

Default sizes are 256, 512, and 1024; default timing is 10 warmup launches,
30 measured launches per round, three rounds, and a 100 ms pause before each
batch. cuBLAS is always included. `--help` lists all options.

The executable dynamically loads `libcublas.so`. Set `GEMM_CUBLAS_LIBRARY`
to its full path if the toolkit libraries are outside the loader's search
path. No separate C/C++ baseline helper is required. Compilation does not
require cuBLAS to be installed; execution does. BF16 Tensor Core math requires
`sm_80+`.

## Correctness and measurement

- A and B are deterministic pseudorandom BF16 values in `[-1, 1]`. Rust and
  cuBLAS use the same device inputs, CUDA context, and CUDA stream.
- The row-major cuBLAS call computes `C^T = B^T * A^T` with swapped M/N and
  leading dimensions N/K/N. Rectangular checks catch incorrect leading
  dimensions that a square-only sweep could hide.
- cuBLAS uses `CUBLAS_COMPUTE_32F` with
  `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`. This keeps Tensor Cores
  enabled while requiring FP32 intermediate reductions. The default cuBLAS
  math mode may reduce split-K chunks in BF16 when the output is BF16; that
  changes the numerical workload. See the
  [cuBLAS numerical behavior documentation](https://docs.nvidia.com/cuda/cublas/index.html#gemm-algorithms-numerical-behavior).
- Before timing, every Rust output element is checked against cuBLAS. Output
  buffers start as NaNs so missing stores fail. All eight validation
  problems are also checked in full against an independent FP64 CPU dot
  product rounded to BF16. Larger problems check 33 CPU reference elements.
- Comparisons reject non-finite values and require
  `abs(actual - reference) <= 0.02 + 0.01 * abs(reference)`. The tolerance
  allows different FP32 reduction orders and BF16 output rounding; the report
  records observed maximum absolute and relative errors.
- CUDA events cover the launch batch, including gaps between launches and
  excluding input generation, allocation,
  transfers, module loading, cuBLAS handle creation, and correctness checks.
  The reported time is the median of the rounds' mean kernel times. Kernel
  order rotates between rounds. Final timed outputs are checked again.
- `TFLOPS = 2*M*N*K / (milliseconds * 1e9)`;
  `% cuBLAS = 100 * cuBLAS_time / Rust_time`. A value above 100 means the
  Rust kernel ran faster in this measurement.
- CSV rows include GPU, architecture, cuBLAS version, shape, kernel, timing
  parameters, both input seeds, math mode, timing range, throughput, and
  observed errors. Laptop power limits and clock/temperature changes still
  affect results; compare reruns under the same power and cooling conditions.

## Initial RTX 3050 results

Measured on 2026-10-10 with the code in this example, based on `experiments`
commit `4ab506b64edca98412cecf937a687ddfb46e321c`:

- NVIDIA GeForce RTX 3050 Laptop GPU, `sm_86`, 4096 MiB.
- NVIDIA driver 615.71.09; CUDA toolkit 13.4.92; cuBLAS version 130800.
- Pinned Rust `nightly-2026-10-09`, optimized build.
- Default protocol: 10 warmups, 30 launches per round, three rounds,
  100 ms cooldown; CUDA event timings, FP32 cuBLAS reductions enabled.

[Raw CSV with timing ranges and correctness errors](results/rtx3050-2026-10-10.csv).

| M=N=K | cuBLAS TFLOPS | Naive TFLOPS | Tiled TFLOPS | Tensor TFLOPS | Tensor / cuBLAS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2.908 | 0.340 | 0.321 | 1.698 | 58.38% |
| 512 | 8.826 | 0.357 | 0.334 | 2.626 | 29.75% |
| 1024 | 14.517 | 0.406 | 0.419 | 3.671 | 25.29% |

This is an initial machine-specific snapshot. GPU clocks and power limits were
not locked. In the 1024 run, the three cuBLAS batch means ranged from 0.148 to
0.193 ms; the reported median was 0.148 ms. The CSV retains these ranges.

The Tensor Core path is roughly nine times faster than the naive kernel at
1024, and still roughly four times slower than cuBLAS. Shared-memory tiling
alone did not produce a consistent improvement over the scalar baseline.
Larger register tiles, packed/coalesced global loads, and a `cp.async` pipeline
are useful next experiments with this harness.

## RTX 3050 results at 4096 and 8192

Measured on the same GPU, driver, toolkit, and cuBLAS version as above on
2026-10-10. All three Rust kernels were included. This larger-shape sweep uses
two warmup launches and three measured launches per round, with three rounds
and 100 ms cooldown. The scalar kernels take roughly 2.8–3.0 seconds per
launch at 8192, so this profile uses fewer repetitions than the initial sweep.
GPU clocks and power limits were not locked.

[Raw CSV with timing ranges and correctness errors](results/rtx3050-4096-8192-2026-10-10.csv).

| M=N=K | cuBLAS TFLOPS | Naive TFLOPS | Tiled TFLOPS | Tensor TFLOPS | Tensor / cuBLAS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 15.760 | 0.378 | 0.404 | 3.345 | 21.23% |
| 8192 | 14.311 | 0.371 | 0.395 | 3.112 | 21.74% |

Median Tensor Core / cuBLAS batch-mean times were 41.086 / 8.721 ms at 4096
and 353.362 / 76.831 ms at 8192. The Tensor Core baseline sustained roughly
21–22% of cuBLAS throughput at these sizes.

Every Rust output element was checked against cuBLAS before timing and after
the last timed launch, including all 67,108,864 outputs at 8192. Both shapes
also passed the 33-point independent FP64 CPU reference check. Maximum
absolute Tensor Core error relative to cuBLAS was zero for both shapes; scalar
kernels passed the documented tolerance. This sweep completed successfully.
CSV field counts, timing ranges, FLOP calculations, and cuBLAS ratios were
checked separately.

## Validation of this snapshot

- `cargo oxide build gemm_compare --arch sm_86`: passed.
- `cargo clippy --locked --manifest-path
  crates/rustc-codegen-cuda/examples/gemm_compare/Cargo.toml -- -D warnings`:
  passed; formatting and `git diff --check` passed.
- Eight built-in edge/long-K cases plus 127x95x63: all outputs checked against
  cuBLAS, with full FP64 CPU reference checks.
- Default 256/512/1024 sweep: all Rust output elements checked against cuBLAS
  before timing and after the last timed launch; 33 CPU reference points per
  benchmark shape. CSV schema and FLOP calculations also checked.
- Compute Sanitizer `memcheck`: zero errors; `racecheck`: zero hazards,
  errors, or warnings, including the long-K cases and 127x95x63.
- SPDX header guard and the example's `cargo deny` checks passed.
- The repository's `check-example-smoketest-contract.sh` fails on the existing
  `block_pipeline` example lacking a success marker at the base commit. The
  same failure was reproduced in the source checkout. `gemm_compare` prints
  both `PASS` and `SUCCESS`; it introduces no additional guard failure.
