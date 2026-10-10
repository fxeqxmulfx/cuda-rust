// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D, LaunchContractError};
use cuda_device::{DisjointSlice, kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), requires = (input.len() == out.len()))]
    pub fn copy(input: &[u32], mut out: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(value) = out.get_mut(index) {
            *value = input[i];
        }
    }
}

fn checked_launch(module: &kernels::LoadedModule, stream: &CudaStream,
                  input: &DeviceBuffer<[u32; 2]>, output: &mut DeviceBuffer<[u32; 2]>)
    -> Result<(), LaunchContractError>
{
    let input = input.as_chunks::<u32>().unwrap();
    let mut output = output.as_chunks_mut::<u32>().unwrap();
    let launch = module.prepare_copy(LaunchConfig1D::new(1, 32, 0))?;
    module.copy(stream, &launch, &input, &mut output)
}
fn main() {}
