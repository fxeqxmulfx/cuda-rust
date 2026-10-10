/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::{SharedArray, kernel, thread, wmma};
use cuda_host::cuda_module;

#[cuda_module]
pub mod gemm {
    use super::*;

    /// One scalar dot product per thread, BF16 inputs/output, FP32 accumulation.
    ///
    /// # Safety
    /// Launch with a 16x16 block and a ceil-divided NxM grid. A, B, and C
    /// must cover M*K, K*N, and M*N elements and C must not alias A or B.
    #[kernel]
    pub unsafe fn naive(m: u32, n: u32, k: u32, a: *const u16, b: *const u16, c: *mut u16) {
        let row = thread::blockIdx_y() * 16 + thread::threadIdx_y();
        let col = thread::blockIdx_x() * 16 + thread::threadIdx_x();
        if row >= m || col >= n {
            return;
        }
        let mut acc = 0.0f32;
        let mut kk = 0u32;
        while kk < k {
            // SAFETY: row < M, col < N, kk < K and the host sized the buffers.
            unsafe {
                acc += bf16_to_f32(*a.add((row * k + kk) as usize))
                    * bf16_to_f32(*b.add((kk * n + col) as usize));
            }
            kk += 1;
        }
        // SAFETY: each thread owns its unique in-bounds (row, col).
        unsafe { *c.add((row * n + col) as usize) = f32_to_bf16_rne(acc) };
    }

    /// A 16x16 shared-memory SIMT tile, with zero padding at all matrix edges.
    ///
    /// # Safety
    /// Same buffer/launch requirements as `naive`. All 256 threads must
    /// participate in both barriers, including threads outside the output.
    #[kernel]
    pub unsafe fn tiled(m: u32, n: u32, k: u32, a: *const u16, b: *const u16, c: *mut u16) {
        static mut A_TILE: SharedArray<u16, 256> = SharedArray::UNINIT;
        static mut B_TILE: SharedArray<u16, 256> = SharedArray::UNINIT;
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let row = thread::blockIdx_y() * 16 + ty;
        let col = thread::blockIdx_x() * 16 + tx;
        // SAFETY: these pointers refer to this CTA's static shared arrays.
        let sa = unsafe { SharedArray::as_raw_mut_ptr(&raw mut A_TILE) };
        let sb = unsafe { SharedArray::as_raw_mut_ptr(&raw mut B_TILE) };
        let mut acc = 0.0f32;
        let mut base = 0u32;
        while base < k {
            // SAFETY: unique shared slot per thread; global loads are bounded.
            unsafe {
                *sa.add((ty * 16 + tx) as usize) = if row < m && base + tx < k {
                    *a.add((row * k + base + tx) as usize)
                } else {
                    0
                };
                *sb.add((ty * 16 + tx) as usize) = if base + ty < k && col < n {
                    *b.add(((base + ty) * n + col) as usize)
                } else {
                    0
                };
            }
            thread::sync_threads();
            let mut i = 0u32;
            while i < 16 {
                // SAFETY: the preceding barrier published all 256 slots.
                unsafe {
                    acc += bf16_to_f32(*sa.add((ty * 16 + i) as usize))
                        * bf16_to_f32(*sb.add((i * 16 + tx) as usize));
                }
                i += 1;
            }
            thread::sync_threads();
            base += 16;
        }
        if row < m && col < n {
            // SAFETY: unique, in-bounds output cell.
            unsafe { *c.add((row * n + col) as usize) = f32_to_bf16_rne(acc) };
        }
    }

    /// Four warps compute a 32x32 output tile using ldmatrix and BF16 mma.sync.
    /// Each warp owns a 16x16 tile (two m16n8k16 accumulators).
    ///
    /// # Safety
    /// Launch with exactly 128x1x1 threads and ceil(N/32)xceil(M/32) CTAs.
    /// Buffers have the same requirements as `naive`. Requires sm_80+.
    /// All lanes participate in every ldmatrix, MMA, and CTA barrier.
    #[kernel]
    pub unsafe fn tensor(m: u32, n: u32, k: u32, a: *const u16, b: *const u16, c: *mut u16) {
        static mut A_TILE: SharedArray<u16, 512, 16> = SharedArray::UNINIT;
        static mut B_TILE: SharedArray<u16, 512, 16> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let warp = tid / 32;
        let warp_m = (warp / 2) * 16;
        let warp_n = (warp % 2) * 16;
        let block_m = thread::blockIdx_y() * 32;
        let block_n = thread::blockIdx_x() * 32;
        // SAFETY: CTA-local shared arrays, explicitly aligned to 16 bytes.
        let sa = unsafe { SharedArray::as_raw_mut_ptr(&raw mut A_TILE) };
        let sb = unsafe { SharedArray::as_raw_mut_ptr(&raw mut B_TILE) };
        let mut acc0 = [0.0f32; 4];
        let mut acc1 = [0.0f32; 4];
        let mut base = 0u32;
        while base < k {
            let mut i = tid;
            while i < 512 {
                let ar = block_m + i / 16;
                let ak = base + i % 16;
                let bk = base + i / 32;
                let bc = block_n + i % 32;
                // SAFETY: cooperative stores are disjoint; loads are bounded.
                unsafe {
                    *sa.add(i as usize) = if ar < m && ak < k {
                        *a.add((ar * k + ak) as usize)
                    } else {
                        0
                    };
                    *sb.add(i as usize) = if bk < k && bc < n {
                        *b.add((bk * n + bc) as usize)
                    } else {
                        0
                    };
                }
                i += 128;
            }
            thread::sync_threads();
            // SAFETY: every lane supplies an aligned row in initialized shared
            // memory. A's four 8x8 subtiles are top-left, bottom-left,
            // top-right, bottom-right. B's two subtiles are stacked vertically;
            // .trans converts its row-major shared layout to MMA's B fragments.
            unsafe {
                let a_offset = (warp_m + lane % 16) * 16 + (lane / 16) * 8;
                let b_offset = (lane % 16) * 32 + warp_n;
                let af = wmma::ldmatrix_x4(sa.add(a_offset as usize).cast());
                let bf0 = wmma::ldmatrix_x2_trans(sb.add(b_offset as usize).cast());
                let bf1 = wmma::ldmatrix_x2_trans(sb.add((b_offset + 8) as usize).cast());
                acc0 = wmma::mma_m16n8k16_f32_bf16(acc0, af, bf0);
                acc1 = wmma::mma_m16n8k16_f32_bf16(acc1, af, bf1);
            }
            // No warp may overwrite the next stage before other warps finish.
            thread::sync_threads();
            base += 16;
        }
        let mut j = 0u32;
        while j < 4 {
            let row = block_m + warp_m + lane / 4 + (j / 2) * 8;
            let col = block_n + warp_n + (lane % 4) * 2 + j % 2;
            if row < m && col < n {
                // SAFETY: the MMA fragment map gives each lane unique cells.
                unsafe { *c.add((row * n + col) as usize) = f32_to_bf16_rne(acc0[j as usize]) };
            }
            if row < m && col + 8 < n {
                // SAFETY: this is the second, disjoint m16n8 output fragment.
                unsafe {
                    *c.add((row * n + col + 8) as usize) = f32_to_bf16_rne(acc1[j as usize]);
                }
            }
            j += 1;
        }
    }
}
