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
    #[launch_contract(domain = 1, block = (32, 1, 1), requires = (out.len() == input.len()))]
    pub fn copy<const N: usize>(#[block_shared(32)] input: &[u32], mut out: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(value) = out.get_mut(index) { *value = input[i % 32] + N as u32; }
    }
}
fn checked(module: &kernels::LoadedModule, stream: &CudaStream, input: &DeviceBuffer<u32>, output: &mut DeviceBuffer<u32>) -> Result<(), LaunchContractError> {
    let launch = module.prepare_copy::<1>(LaunchConfig1D::new(1, 32, 0))?;
    module.copy(stream, &launch, input, output)
}
fn main() {}
