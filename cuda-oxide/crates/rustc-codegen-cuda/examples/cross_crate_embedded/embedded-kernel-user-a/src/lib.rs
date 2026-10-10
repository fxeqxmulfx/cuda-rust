/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Library crate that launches the generic `scale` kernel of
//! `embedded-kernel-lib` but does not define it.
//!
//! The crates `embedded-kernel-user-a` and `embedded-kernel-user-b` both
//! compile this file. Each puts `scale::<f64>` in its own PTX bundle, thus
//! the merged load gets that entry from two bundles.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use embedded_kernel_lib::kernels;

/// Multiply each element of `input` by `factor` with `scale::<f64>`.
pub fn scale_f64(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    factor: f64,
    input: &[f64],
) -> Result<Vec<f64>, DriverError> {
    let in_dev = DeviceBuffer::from_host(stream, input)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(stream, input.len())?;
    let cfg = LaunchConfig::for_num_elems(input.len() as u32);
    // SAFETY: the launch has one thread for each element of the buffers.
    unsafe { module.scale::<f64>(stream, cfg, factor, &in_dev, &mut out_dev) }?;
    out_dev.to_host_vec(stream)
}
