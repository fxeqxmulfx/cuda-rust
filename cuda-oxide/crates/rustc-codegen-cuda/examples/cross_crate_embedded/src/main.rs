/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-crate embedded bundle regression test (issue #222).
//!
//! This example exercises the code path that issue #222 fixed: loading a
//! generic kernel from a library crate via the embedded artifact bundle API
//! rather than from a PTX file on disk.
//!
//! When `embedded_kernel_lib::kernels::load(&ctx)` is called, the macro-generated
//! `load` function uses `load_ptx_bundles_merged_from_anchor` (because the
//! module contains generic kernels). This merges the PTX bundles with
//! specializations of `scale`, here the binary crate's bundle where the
//! monomorphized PTX for `scale::<f32>` and `scale::<i32>` actually lives,
//! and the bundles of two libraries with `scale::<f64>`.
//!
//! Before the fix this would panic with:
//!   DriverError(500, "named symbol not found")
//! because `load_embedded_module("embedded-kernel-lib")` only searched that library's
//! own bundle, which has no monomorphized entry points.
//!
//! Run: cargo oxide run cross_crate_embedded

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use embedded_kernel_lib::kernels;

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    // Embedded loading path: this is what issue #222 was about.
    let module = kernels::load(&ctx).expect("load embedded module");

    const N: usize = 256;
    let cfg = LaunchConfig::for_num_elems(N as u32);
    let mut errors = 0usize;

    // scale::<f32>
    {
        let factor: f32 = 2.5;
        let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let in_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.scale::<f32>(&stream, cfg, factor, &in_dev, &mut out_dev) }
            .expect("scale::<f32> launch");
        let out = out_dev.to_host_vec(&stream).unwrap();
        for (i, (&got, &x)) in out.iter().zip(input.iter()).enumerate() {
            if (got - x * factor).abs() > 1e-5 {
                if errors < 5 {
                    eprintln!(
                        "  FAIL scale::<f32>[{}]: got {} want {}",
                        i,
                        got,
                        x * factor
                    );
                }
                errors += 1;
            }
        }
    }

    // scale::<i32>
    {
        let factor: i32 = 3;
        let input: Vec<i32> = (0..N as i32).collect();
        let in_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.scale::<i32>(&stream, cfg, factor, &in_dev, &mut out_dev) }
            .expect("scale::<i32> launch");
        let out = out_dev.to_host_vec(&stream).unwrap();
        for (i, (&got, &x)) in out.iter().zip(input.iter()).enumerate() {
            if got != x * factor {
                if errors < 5 {
                    eprintln!(
                        "  FAIL scale::<i32>[{}]: got {} want {}",
                        i,
                        got,
                        x * factor
                    );
                }
                errors += 1;
            }
        }
    }

    // scale::<f64>, launched by two libraries that do not define the kernel.
    // Only the bundles of these libraries have this specialization.
    {
        let input: Vec<f64> = (0..N).map(|i| i as f64).collect();
        let a = embedded_kernel_user_a::scale_f64(&module, &stream, 1.5, &input)
            .expect("library A scale::<f64> launch");
        let b = embedded_kernel_user_b::scale_f64(&module, &stream, 0.5, &input)
            .expect("library B scale::<f64> launch");
        for (i, &x) in input.iter().enumerate() {
            if a[i] != x * 1.5 || b[i] != x * 0.5 {
                if errors < 5 {
                    eprintln!("  FAIL library scale::<f64>[{}]: {} {}", i, a[i], b[i]);
                }
                errors += 1;
            }
        }
    }

    if errors == 0 {
        println!("SUCCESS: cross-crate generic kernels load correctly via embedded bundles");
    } else {
        eprintln!("FAIL: {} errors", errors);
        std::process::exit(1);
    }
}
