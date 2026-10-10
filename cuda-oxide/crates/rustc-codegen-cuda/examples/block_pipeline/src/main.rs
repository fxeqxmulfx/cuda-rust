// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::shared::BlockTile;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (D <= 128))]
    #[block_pipeline(rows = 4, iterations = passes)]
    pub fn sum<const D: usize>(
        passes: usize,
        #[tile_source(columns = D, stride = D, offset = block * 3)] a: &[f32],
        #[tile_source(columns = D, stride = D, offset = block * 5)] b: &[f32],
        mut out: DisjointSlice<f32, thread::StripedWarp<D>>,
    ) {
        let init = || {
            thread::sync_threads();
            [0.0; 4]
        };
        let step = |state: &mut [f32; 4],
                    _iteration: usize,
                    a: BlockTile<'_, f32>,
                    b: BlockTile<'_, f32>| {
            thread::sync_threads();
            let lane = thread::threadIdx_x() as usize % 32;
            if lane.is_multiple_of(7) {
                return;
            }
            for c in 0..D.div_ceil(32) {
                for row in 0..4 {
                    state[c] += a.load(row, lane + c * 32) + b.load(row, lane + c * 32);
                }
            }
        };
        let mut finish = |state: [f32; 4]| {
            for c in 0..D.div_ceil(32) {
                if let Some(index) =
                    thread::striped_index::<D>(thread::threadIdx_x() as usize % 32 + c * 32)
                {
                    if let Some(value) = out.get_mut(index) {
                        *value = state[c];
                    }
                }
            }
        };
    }
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    #[block_pipeline(rows = R + 1, iterations = passes)]
    pub fn swizzled<const D: usize, const R: usize>(
        passes: usize,
        #[tile_source(columns = D, stride = D, offset = (block * 3 + 1) / (1 - 1 / (passes + 1)), layout = xor32)]
        a: &[f32],
        #[tile_source(columns = D, stride = D, offset = (block * 3 + 1) / (1 - 1 / (passes + 1)))]
        b: &[f32],
        mut out: DisjointSlice<f32, thread::StripedWarp<D>>,
    ) {
        let init = || [0.0; 4];
        let step = |state: &mut [f32; 4],
                    _iteration: usize,
                    a: BlockTile<'_, f32>,
                    b: BlockTile<'_, f32>| {
            let lane = thread::threadIdx_x() as usize % 32;
            for c in 0..D / 32 {
                for row in 0..R + 1 {
                    state[c] += a.load(row, lane + c * 32) + b.load(row, lane + c * 32);
                }
            }
        };
        let mut finish = |state: [f32; 4]| {
            for c in 0..D / 32 {
                if let Some(index) =
                    thread::striped_index::<D>(thread::threadIdx_x() as usize % 32 + c * 32)
                {
                    if let Some(value) = out.get_mut(index) {
                        *value = state[c];
                    }
                }
            }
        };
    }
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1)]
    pub fn view<const C: usize>(
        input: &[f32],
        offset: usize,
        rows: usize,
        stride: usize,
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let tile = BlockTile::new(input, offset, rows, C, stride);
        let mut sum = 0.0;
        for row in 0..rows {
            sum += tile.load(row, index.get() % C);
        }
        if let Some(value) = out.get_mut(index) {
            *value = sum;
        }
    }
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1)]
    pub fn striped<const D: usize>(mut out: DisjointSlice<f32, thread::StripedWarp<D>>) {
        for column in 0..D + 13 {
            if let Some(index) = thread::striped_index::<D>(column) {
                let raw = index.get();
                if let Some(value) = out.get_mut(index) {
                    *value = (raw + 1) as f32;
                }
            }
        }
    }
}
fn check<const D: usize>(
    context: &std::sync::Arc<CudaContext>,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = context.new_stream()?;
    let module = kernels::load(context)?;
    for length in [0, 1, 7, 129, 517] {
        let a: Vec<f32> = (0..length).map(|i| (i % 19) as f32).collect();
        let b: Vec<f32> = (0..length).map(|i| -(i as f32 % 7.0)).collect();
        let a_gpu = DeviceBuffer::from_host(&stream, &a)?;
        let b_gpu = DeviceBuffer::from_host(&stream, &b)?;
        for passes in [0, 1, 2, 3, 8] {
            let mut out = DeviceBuffer::zeroed(&stream, 3 * 4 * D)?;
            let bytes = (2 * 4 * 2 * D * 4) as u32;
            let config = LaunchConfig1D::new(3, 128, bytes);
            let launch = module.prepare_sum::<D>(config)?;
            module.sum(&stream, &launch, passes, &a_gpu, &b_gpu, &mut out)?;
            for (i, value) in out.to_host_vec(&stream)?.into_iter().enumerate() {
                let column = i % D;
                let block = i / D / 4;
                let mut expected = 0.0;
                if !(column % 32).is_multiple_of(7) {
                    for row in 0..passes * 4 {
                        expected += a.get(block * 3 + row * D + column).copied().unwrap_or(0.0)
                            + b.get(block * 5 + row * D + column).copied().unwrap_or(0.0);
                    }
                }
                assert_eq!(
                    value, expected,
                    "D={D} len={length} passes={passes} index={i}"
                );
            }
            assert!(
                module
                    .prepare_sum::<D>(LaunchConfig1D::new(3, 64, bytes))
                    .is_err()
            );
            if D > 0 {
                assert!(
                    module
                        .prepare_sum::<D>(LaunchConfig1D::new(3, 128, bytes - 1))
                        .is_err()
                );
            }
        }
    }
    Ok(())
}
fn check_swizzled<const D: usize, const R: usize>(
    context: &std::sync::Arc<CudaContext>,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = context.new_stream()?;
    let module = kernels::load(context)?;
    for length in [0, 1, 7, 129, 517, 4099] {
        let input: Vec<f32> = (0..length).map(|i| (i % 37) as f32).collect();
        let device = DeviceBuffer::from_host(&stream, &input)?;
        let other: Vec<f32> = input[..length / 2]
            .iter()
            .map(|value| -2.0 * value)
            .collect();
        let other_device = DeviceBuffer::from_host(&stream, &other)?;
        for passes in [0, 1, 2, 3, 8] {
            let bytes = (2 * (R + 1) * 2 * D * 4) as u32;
            let launch = module.prepare_swizzled::<D, R>(LaunchConfig1D::new(3, 128, bytes))?;
            let mut out = DeviceBuffer::zeroed(&stream, 3 * 4 * D)?;
            module.swizzled(&stream, &launch, passes, &device, &other_device, &mut out)?;
            for (i, actual) in out.to_host_vec(&stream)?.into_iter().enumerate() {
                let column = i % D;
                let block = i / D / 4;
                let expected: f32 = (0..passes * (R + 1))
                    .map(|row| {
                        input
                            .get(block * 3 + 1 + row * D + column)
                            .copied()
                            .unwrap_or(0.0)
                            + other
                                .get(block * 3 + 1 + row * D + column)
                                .copied()
                                .unwrap_or(0.0)
                    })
                    .sum();
                assert_eq!(
                    actual, expected,
                    "swizzled D={D} R={R} length={length} passes={passes} i={i}"
                );
            }
            assert!(
                module
                    .prepare_swizzled::<D, R>(LaunchConfig1D::new(3, 128, bytes - 1))
                    .is_err()
            );
        }
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    check::<0>(&context)?;
    check::<1>(&context)?;
    check::<32>(&context)?;
    check::<65>(&context)?;
    check::<128>(&context)?;
    check_swizzled::<32, 0>(&context)?;
    check_swizzled::<32, 3>(&context)?;
    check_swizzled::<32, 31>(&context)?;
    check_swizzled::<64, 7>(&context)?;
    check_swizzled::<64, 63>(&context)?;
    let stream = context.new_stream()?;
    let module = kernels::load(&context)?;
    for threads in [1u32, 17, 31, 32, 33, 48, 64, 127, 128] {
        let mut out = DeviceBuffer::zeroed(&stream, 3 * threads.div_ceil(32) as usize * 65)?;
        let launch = module.prepare_striped::<65>(LaunchConfig1D::new(3, threads, 0))?;
        module.striped(&stream, &launch, &mut out)?;
        for (i, actual) in out.to_host_vec(&stream)?.into_iter().enumerate() {
            let warp = i / 65 % threads.div_ceil(32) as usize;
            let lane = i % 65 % 32;
            let expected = if warp * 32 + lane < threads as usize {
                (i + 1) as f32
            } else {
                0.0
            };
            assert_eq!(actual, expected, "threads={threads} i={i}");
        }
    }
    let input: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let device = DeviceBuffer::from_host(&stream, &input)?;
    let cases = [
        (0, 4, 32),
        (5, 3, 40),
        (100, 7, 33),
        (299, 2, 0),
        (300, 2, 1),
        (usize::MAX, 2, 1),
        (7, 3, usize::MAX / 2 + 1),
        (usize::MAX - 40, 3, 20),
    ];
    for (offset, rows, stride) in cases {
        let mut out = DeviceBuffer::zeroed(&stream, 96)?;
        let launch = module.prepare_view::<40>(LaunchConfig1D::new(3, 32, 0))?;
        module.view(&stream, &launch, &device, offset, rows, stride, &mut out)?;
        for (i, actual) in out.to_host_vec(&stream)?.into_iter().enumerate() {
            let expected: f32 = (0..rows)
                .map(|row| {
                    let index = row as u128 * stride as u128 + offset as u128 + (i % 40) as u128;
                    if index < 300 {
                        input[index as usize]
                    } else {
                        0.0
                    }
                })
                .sum();
            assert_eq!(
                actual, expected,
                "view offset={offset} stride={stride} i={i}"
            );
        }
    }
    println!("275 pipeline cases, 9 partial-warp cases and 8 view cases passed");
    Ok(())
}
