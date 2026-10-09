// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

#[cuda_host::cuda_module]
mod sized {
    use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), requires = (
        N >= 1, width <= N, input.len() == output.len()))]
    pub fn copy<const N: usize>(
        width: usize,
        input: &[[f32; N]],
        mut output: DisjointSlice<[f32; N]>,
    ) {
        let index = thread::index_1d();
        let source = index.get();
        if let Some(row) = output.get_mut(index) {
            row[..width].copy_from_slice(&input[source][..width]);
        }
    }
}

#[cuda_device::kernel]
#[cuda_device::launch_contract(domain = 1, block = (64, 1, 1), requires = (
    input.len() >= N, output.len() >= N))]
pub fn standalone<const N: u32>(input: &[f32], mut output: cuda_device::DisjointSlice<f32>) {
    let _ = (input, &mut output);
}

fn checked_launch(
    module: &sized::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<[f32; 4]>,
    output: &mut DeviceBuffer<[f32; 4]>,
) {
    let launch = module.prepare_copy::<4>(LaunchConfig1D::new(1, 64, 0)).unwrap();
    module.copy(stream, &launch, 4, input, output).unwrap();
}

fn main() {}
