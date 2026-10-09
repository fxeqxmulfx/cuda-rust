// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression for generic PTX instantiated in a library and its consumer.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use std::sync::Arc;

#[cuda_host::cuda_module]
pub mod checked {
    use cuda_device::{DisjointSlice, device, kernel, launch_contract, thread};

    #[device]
    fn increment(value: u32) -> u32 {
        value + 1
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), requires = (
        N >= 1, width <= N, input.len() == output.len()))]
    pub fn copy_rows<const N: usize>(
        width: usize,
        input: &[[u32; N]],
        mut output: DisjointSlice<[u32; N]>,
    ) {
        let index = thread::index_1d();
        let source = index.get();
        if let Some(row) = output.get_mut(index) {
            for channel in 0..width {
                row[channel] = increment(input[source][channel]);
            }
        }
    }
}

pub fn verify_rows<const N: usize>(context: &Arc<CudaContext>) {
    let stream = context.default_stream();
    let module = checked::load(context).expect("load checked embedded generic module");
    let input = vec![[N as u32; N]; 65];
    let source = DeviceBuffer::from_host(&stream, &input).expect("upload rows");
    let mut output = DeviceBuffer::zeroed(&stream, input.len()).expect("allocate rows");
    let launch = module
        .prepare_copy_rows::<N>(LaunchConfig1D::new(2, 64, 0))
        .expect("prepare copy");
    module
        .copy_rows(&stream, &launch, N, &source, &mut output)
        .expect("copy rows");
    assert_eq!(
        output.to_host_vec(&stream).expect("download rows"),
        vec![[N as u32 + 1; N]; input.len()]
    );
    assert!(
        module
            .copy_rows(&stream, &launch, N + 1, &source, &mut output)
            .is_err()
    );
    let mut short = DeviceBuffer::zeroed(&stream, input.len() - 1).expect("allocate short rows");
    assert!(
        module
            .copy_rows(&stream, &launch, N, &source, &mut short)
            .is_err()
    );
}

/// Concrete library code instantiates this entry in the library artifact.
pub fn verify_library_specialization(context: &Arc<CudaContext>) {
    verify_rows::<4>(context);
}
