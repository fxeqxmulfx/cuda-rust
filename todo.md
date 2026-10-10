<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TODO: Efficient FP8 Emulation

Plan based on the review of the current implementation and vLLM Marlin.
Implementation and benchmarking have not started; expected improvements must
be confirmed by measurements.

GEMM goal: keep weights in FP8, load and unpack them inside the kernel, compute
using FP16/BF16 Tensor Cores, and coordinate unpacking with work distribution
and the load pipeline. Evaluate standalone `convert` optimization separately.

## Current Implementation and Constraints

- [ ] Record baseline performance for `convert` and GEMM with FP8 weights on
  `sm_80`/`sm_86`, and pin a specific vLLM revision for comparison.
- [ ] Review [FP8 lowering](cutile-rs/cutile-ir/src/emulation.rs): on `sm_80..sm_89`,
  storage becomes `i8`, while decoding always constructs FP32 and separately
  computes normal, subnormal, and special values. Even conversion to FP16/BF16
  follows this path.
- [ ] Preserve bytes, shapes, strides, pointers, constants, and zero padding;
  preserve diagnostics for unsupported FP8 operations.
- [ ] Preserve the native path on `sm_90+` and compatibility with supported
  Tile IR versions. Select new optimizations according to target capabilities.
- [ ] Account for the existing
  [persistent GEMM](cutile-rs/cutile-examples/examples/persistent_gemm.rs): it
  distributes whole output tiles and traverses all of K for each tile. This
  example has no striped work distribution with partial sums.

## Specialized Decoding

- [ ] Add a dedicated E5M2 → FP16 path: extend the byte, shift by 8 bits, and
  reinterpret. The exponent layout and bias match FP16, so intermediate FP32
  and separate subnormal computation are unnecessary. Check zeros, subnormals,
  Inf, and NaN representation requirements.
- [ ] Evaluate Marlin's E4M3FN → FP16 approach: masks and shifts construct FP16,
  then multiplication by 256 corrects the exponent bias. Check exactness for
  all finite values, including intermediate subnormals and FTZ behavior.
- [ ] Handle E4M3FN NaNs. Marlin's fast decoder formula maps `0x7f`/`0xff` to
  ±480; the general `convert` operation must return NaN.
- [ ] For FP8 → BF16/FP32/FP64, compare specialized decoding to FP16 followed
  by hardware conversion against direct specialized paths. All finite
  E4M3FN/E5M2 values are exactly representable in FP16.
- [ ] Separately evaluate Marlin's direct E4M3FN → BF16 path with the `2^120`
  correction: check intermediate subnormals and the scale range.

## Packing and Scaling

- [ ] Process multiple values together: four FP8 values in a 32-bit register
  become two FP16/BF16 pairs. Evaluate final instruction count and register
  usage alongside the number of Tile IR operations.
- [ ] Evaluate Tile IR 13.3+ `pack/unpack`: these operations require 1D tiles,
  so reshaping and restoring the original shape may be necessary. Retain a
  compatible path for Tile IR 13.2.
- [ ] Verify that packing does not introduce expensive cross-thread shuffles
  and that the compiler actually emits beneficial packed instructions.
- [ ] For specialized GEMM, prearrange weights and scales into a layout suited
  to loads and MMA. Preserve the general FP8 storage format.
- [ ] For constant scales, evaluate folding the exponent correction into
  scales during weight preparation: `2^8` for FP16 and `2^120` for BF16.
  Check overflow, underflow, and rounding changes; provide a correct fallback.
- [ ] For per-output-channel scales, evaluate scaling the FP32 accumulator
  before reduction/writeback. For grouped scales, retain scaling within the
  corresponding K segment; moving it outside the full K sum is invalid.

## Persistent Scheduler and Striped Work Distribution

- [ ] Choose the number of persistent blocks using the SM count and kernel
  resources. Tune BM/BN/BK, thread count, and occupancy for different M/N/K shapes.
- [ ] Implement Marlin's hybrid distribution: process the main workload as
  whole output tiles, then divide the remaining work into approximately equal
  K stripes that may cross boundaries between adjacent output tiles.
- [ ] Check SM utilization for small M and few output tiles, as well as an
  incomplete final wave of blocks.
- [ ] Align stripe boundaries with quantization groups and MMA granularity;
  minimize split output tiles and cross-block reductions.
- [ ] Choose how to combine partial sums: FP32 workspace followed by a merge,
  reduction with cross-block synchronization, or suitable atomics. Measure
  overhead and verify numerical accuracy for each selected mode.
- [ ] Account for the `MappedPartitionMut` contract: it guarantees disjoint
  writes. Multiple blocks contributing to one result require separate partial
  outputs and a merge, or an explicitly justified synchronization/atomic path.
- [ ] For cross-block waits, verify forward progress, memory visibility,
  workspace initialization/reset, and freedom from deadlocks.
- [ ] Evaluate distributing K among warps within a block with local reduction:
  this may increase active warp count without enlarging the output tile.

## Load, Unpack, and MMA Pipeline

- [ ] Fuse FP8 loads, decoding, scaling, and MMA into one kernel. Eliminate
  the intermediate unpacked matrix in global memory wherever a separate
  `api::convert` is currently called before GEMM.
- [ ] Overlap global-to-shared loads with computation; use a separate buffered
  pipeline from shared memory to registers.
- [ ] Schedule unpacking so integer/vector instructions can overlap Tensor
  Core MMA; limit the lifetime of unpacked fragments and register usage.
- [ ] Tune pipeline depth together with tile sizes and shared memory usage.
  Consider `cp.async` on Ampere; evaluate TMA only on GPUs that support it.
- [ ] Reuse activations through L2 and registers; evaluate caching policies
  for streamed weights according to their actual reuse.
- [ ] Check load vectorization and alignment, shared-memory bank conflicts,
  static offsets, and appropriate loop unrolling.
- [ ] Verify that the Tile compiler turns persistent traversal and
  `load_pipelined` hints into the desired pipeline. Before choosing a path,
  compare Tile and SIMT (`cuda-oxide`) support for manual instruction scheduling.

## Encoding to FP8

- [ ] Optimize [encoding](cutile-rs/cutile-ir/src/emulation.rs) separately:
  Marlin's fast dequantization does not implement FP32/FP64 → FP8.
- [ ] Evaluate subnormal rounding by adding a carefully chosen floating-point
  constant in place of the long chain of variable integer shifts; use PyTorch
  implementations as algorithm references.
- [ ] Preserve nearest-even rounding, finite saturation, NaNs, and signed
  zero for both FP8 formats. In particular, E5M2 overflow/Inf must also saturate
  to a finite value under the current contract.
- [ ] Preserve FP64 → FP8 accuracy: intermediate narrowing to FP32/FP16 can
  cause double rounding. Retain the current FP64 path until a replacement is
  verified.

## Validation and Acceptance Criteria

- [ ] Use existing [IR tests](cutile-rs/cutile-ir/tests/fp8_emulation.rs) for
  capabilities, serialization, dominance, and preservation of the native path.
- [ ] Use [GPU dtype tests](cutile-rs/cutile/tests/dtype_float_ops.rs): all 256
  encodings of both FP8 formats to FP16/BF16/FP32/FP64; ±0, subnormals, NaN/Inf;
  midpoints and neighboring values for nearest-even rounding and saturation.
  Also check integer conversions and FP8 → FP8.
- [ ] For the scheduler, verify complete work coverage without omissions or
  duplicates, split tiles, scale-group boundaries, padding, and partial-sum
  reductions.
- [ ] Compare regular FP16/BF16 GEMM, separate convert + GEMM, and fused FP8
  GEMM. Measure the contributions of decoding, packing, scheduling, and
  pipelining separately.
- [ ] Include small M (1/8/16/32), medium and large M, different N/K, short and
  long K, and shapes with incomplete block waves.
- [ ] Measure latency, bandwidth, SM/Tensor Core utilization, register usage,
  spills, shared memory, occupancy, and stalls; inspect PTX/SASS and Nsight
  Compute profiles.
- [ ] Document accuracy and the range of workloads that benefit. On Ampere,
  computation remains FP16/BF16; FP8 reduces weight storage and load volume.
  Evaluate compute-bound workloads separately. INT4 Marlin results must not
  be treated as FP8 measurements.

## Sources

- [vLLM: packed FP8 → FP16/BF16 dequantization](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/marlin/dequant.h#L292).
- [vLLM: weight preparation and folding exponent bias into scales](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py#L30).
- [vLLM: configuration selection and persistent grid](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/marlin/marlin.cu#L181).
- [vLLM: striped scheduler](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/marlin/marlin_template.h#L318),
  [pipeline](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/marlin/marlin_template.h#L1434),
  [reduction](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/marlin/marlin_template.h#L1636).
- [Marlin: work distribution, caching, and instruction scheduling](https://github.com/IST-DASLab/marlin#techniques).
- [vLLM: vectorized FP8 conversions for the KV cache](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh).
- [PyTorch: E5M2 via FP16](https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_e5m2.h#L62).
- [PyTorch: E4M3FN rounding](https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_e4m3fn.h#L164).
