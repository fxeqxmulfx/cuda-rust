// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::{CudaContext, CudaModule};
use std::sync::Arc;

#[cuda_host::cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, kernel, launch_contract};

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn copy<const N: usize>(mut output: DisjointSlice<[u32; N]>) {
        let _ = &mut output;
    }
}

fn bind_foreign(module: Arc<CudaModule>, context: &Arc<CudaContext>) {
    let _ = kernels::from_module(module);
    let _ = kernels::load_named(context, "another-package");
}

fn main() {}
