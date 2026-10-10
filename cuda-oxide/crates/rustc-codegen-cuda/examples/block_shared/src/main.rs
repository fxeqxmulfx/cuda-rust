// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry-staged read-only tiles, including tails and divergent user returns.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn neighbor_u32(#[block_shared(32)] input: &[u32], mut out: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let lane = index.get() % 32;
        if lane >= input.len() {
            return;
        }
        if let Some(value) = out.get_mut(index) {
            *value = input[(lane + 1) % input.len()];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, requires = (stride >= 1))]
    pub fn shared<const REPEATS: usize>(
        stride: usize,
        #[block_shared(128, offset = (block / stride) * REPEATS * 16)] input: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= out.len() || i.is_multiple_of(7) || input.is_empty() {
            return;
        }
        if let Some(value) = out.get_mut(index) {
            let mut sum = 0.0;
            for pass in 0..REPEATS {
                for local in 0..input.len() {
                    sum += input[(local + i + pass) % input.len()] * (pass + 1) as f32;
                }
            }
            *value = sum;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, requires = (stride >= 1))]
    pub fn global<const REPEATS: usize>(stride: usize, input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        let begin = (thread::blockIdx_x() as usize / stride) * 128;
        if i >= out.len() || i.is_multiple_of(7) || begin >= input.len() {
            return;
        }
        let input = &input[begin..(begin + 128).min(input.len())];
        if let Some(value) = out.get_mut(index) {
            let mut sum = 0.0;
            for pass in 0..REPEATS {
                for local in 0..input.len() {
                    sum += input[(local + i + pass) % input.len()] * (pass + 1) as f32;
                }
            }
            *value = sum;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    // SAFETY: this executable embeds exactly the kernels declared above. The
    // non-generic neighbor_u32 export has a unique name and the generated ABI.
    // Legacy non-generic symbols do not yet carry generic TypeId provenance.
    let module = unsafe { kernels::load(&context)? };
    for count in [0, 1, 31, 127, 128, 129, 257, 4096] {
        let data: Vec<f32> = (0..count)
            .map(|i| ((i * 7 % 29) as f32 - 14.0) / 32.0)
            .collect();
        let input = DeviceBuffer::from_host(&stream, &data)?;
        for block in [32, 64, 128] {
            for stride in [1, 2, 3] {
                let mut shared = DeviceBuffer::zeroed(&stream, count + 17)?;
                let mut global = DeviceBuffer::zeroed(&stream, count + 17)?;
                let config =
                    LaunchConfig1D::new((count + 17).div_ceil(block as usize) as u32, block, 0);
                let launch = module.prepare_shared::<8>(config)?;
                module.shared(&stream, &launch, stride, &input, &mut shared)?;
                let launch = module.prepare_global::<8>(config)?;
                module.global(&stream, &launch, stride, &input, &mut global)?;
                let shared = shared.to_host_vec(&stream)?;
                let global = global.to_host_vec(&stream)?;
                for (i, (&actual, &control)) in shared.iter().zip(&global).enumerate() {
                    let begin = i / block as usize / stride * 128;
                    let expected = if i.is_multiple_of(7) || begin >= count {
                        0.0
                    } else {
                        data[begin..(begin + 128).min(count)]
                            .iter()
                            .map(|&v| f64::from(v) * 36.0)
                            .sum()
                    };
                    assert_eq!(
                        f64::from(actual),
                        expected,
                        "count={count} block={block} stride={stride} i={i}"
                    );
                    assert_eq!(actual.to_bits(), control.to_bits());
                }
                println!(
                    "count={count} block={block} stride={stride}: PASS (tails, neighbor reads, early returns)"
                );
            }
        }
    }
    for count in [1_usize, 31, 32, 33, 65] {
        let data: Vec<u32> = (0..count).map(|i| i as u32 * 17 + 3).collect();
        let input = DeviceBuffer::from_host(&stream, &data)?;
        let mut output = DeviceBuffer::zeroed(&stream, count + 5)?;
        let launch = module.prepare_neighbor_u32(LaunchConfig1D::new(
            (count + 5).div_ceil(32) as u32,
            32,
            0,
        ))?;
        module.neighbor_u32(&stream, &launch, &input, &mut output)?;
        let result = output.to_host_vec(&stream)?;
        for (i, &value) in result.iter().enumerate() {
            let begin = i / 32 * 32;
            let len = count.saturating_sub(begin).min(32);
            let expected = if i >= count {
                0
            } else {
                data[begin + (i % 32 + 1) % len]
            };
            assert_eq!(value, expected);
        }
        println!("u32 count={count}: PASS (non-generic safe launcher)");
    }
    Ok(())
}
