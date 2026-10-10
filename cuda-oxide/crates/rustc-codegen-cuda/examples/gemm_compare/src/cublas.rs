/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Minimal host-only cuBLAS binding. Runtime loading keeps GPU-less builds
//! independent of a link-time cuBLAS installation.

use crate::{Problem, Result};
use cuda_core::{CudaStream, DeviceBuffer};
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::Arc;

type Handle = *mut c_void;
type Create = unsafe extern "C" fn(*mut Handle) -> c_int;
type Destroy = unsafe extern "C" fn(Handle) -> c_int;
type SetStream = unsafe extern "C" fn(Handle, *mut c_void) -> c_int;
type SetMathMode = unsafe extern "C" fn(Handle, c_int) -> c_int;
type GetVersion = unsafe extern "C" fn(Handle, *mut c_int) -> c_int;
type Gemm = unsafe extern "C" fn(
    Handle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;

fn check(status: c_int, operation: &str) -> Result<()> {
    if status != 0 {
        return Err(format!("{operation} failed with cuBLAS status {status}").into());
    }
    Ok(())
}

pub struct Cublas {
    handle: Handle,
    gemm: Gemm,
    destroy: Destroy,
    pub version: i32,
    stream: Arc<CudaStream>,
    // Function pointers stay valid until after the handle has been destroyed.
    _library: Library,
}

impl Cublas {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let path = std::env::var_os("GEMM_CUBLAS_LIBRARY").unwrap_or_else(|| "libcublas.so".into());
        // SAFETY: load the CUDA library with its documented C ABI. An explicit
        // path is available for installations outside the loader's search path.
        let library = unsafe { Library::new(&path) }.map_err(|e| {
            format!(
                "cannot load {}: {e}; set GEMM_CUBLAS_LIBRARY",
                path.to_string_lossy()
            )
        })?;
        // SAFETY: signatures match cublas_api.h; the library owns these symbols.
        let (create, destroy, set_stream, set_math_mode, get_version, gemm) = unsafe {
            (
                *library.get::<Create>(b"cublasCreate_v2\0")?,
                *library.get::<Destroy>(b"cublasDestroy_v2\0")?,
                *library.get::<SetStream>(b"cublasSetStream_v2\0")?,
                *library.get::<SetMathMode>(b"cublasSetMathMode\0")?,
                *library.get::<GetVersion>(b"cublasGetVersion_v2\0")?,
                *library.get::<Gemm>(b"cublasGemmEx\0")?,
            )
        };
        let mut handle = std::ptr::null_mut();
        // SAFETY: writable handle slot; the CUDA context is current.
        check(unsafe { create(&mut handle) }, "cublasCreate")?;
        let mut blas = Self {
            handle,
            gemm,
            destroy,
            version: 0,
            stream,
            _library: library,
        };
        // SAFETY: a live handle and the same context/stream used by Rust kernels.
        check(
            unsafe { set_stream(blas.handle, blas.stream.cu_stream().cast()) },
            "cublasSetStream",
        )?;
        // BF16 output must not permit BF16 reductions of split-K chunks.
        // Keep Tensor Cores enabled but require FP32 intermediate reduction.
        // SAFETY: live handle; 16 is DISALLOW_REDUCED_PRECISION_REDUCTION.
        check(
            unsafe { set_math_mode(blas.handle, 16) },
            "cublasSetMathMode",
        )?;
        // SAFETY: a live handle and a writable version slot.
        check(
            unsafe { get_version(blas.handle, &mut blas.version) },
            "cublasGetVersion",
        )?;
        Ok(blas)
    }

    pub fn run(
        &self,
        p: Problem,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &mut DeviceBuffer<u16>,
    ) -> Result<()> {
        // C(row-major) = A * B is C^T(column-major) = B^T * A^T.
        // Thus the cuBLAS call swaps A/B and M/N, with lda=N, ldb=K, ldc=N.
        // CUDA_R_16BF=14, CUBLAS_COMPUTE_32F=68, CUBLAS_GEMM_DEFAULT=-1.
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // SAFETY: all buffers cover the validated matrix extents and remain
        // alive through stream synchronization. Scalars use host pointer mode,
        // the cuBLAS default; both inputs and output are BF16, compute is FP32.
        check(
            unsafe {
                (self.gemm)(
                    self.handle,
                    0,
                    0,
                    p.n as i32,
                    p.m as i32,
                    p.k as i32,
                    (&alpha as *const f32).cast(),
                    b.cu_deviceptr() as *const c_void,
                    14,
                    p.n as i32,
                    a.cu_deviceptr() as *const c_void,
                    14,
                    p.k as i32,
                    (&beta as *const f32).cast(),
                    c.cu_deviceptr() as *mut c_void,
                    14,
                    p.n as i32,
                    68,
                    -1,
                )
            },
            "cublasGemmEx",
        )
    }
}

impl Drop for Cublas {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        // SAFETY: this handle was created once and the library is still loaded.
        let status = unsafe { (self.destroy)(self.handle) };
        if status != 0 {
            eprintln!("cublasDestroy failed with status {status}");
        }
    }
}
