/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared memory support for CUDA kernels.
//!
//! Shared memory is a fast, block-scoped memory space that all threads
//! in a block can access. Use it for:
//! - Tile-based algorithms (GEMM, convolutions)
//! - Reductions and scans
//! - Inter-thread communication within a block
//!
//! # Usage
//!
//! Declare shared memory as `static mut` inside a kernel:
//!
//! ```rust,ignore
//! use cuda_device::{kernel, thread, SharedArray};
//!
//! #[kernel]
//! pub fn tiled_kernel(data: &[f32], mut out: DisjointSlice<f32>) {
//!     static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
//!
//!     let tid = thread::threadIdx_x() as usize;
//!
//!     unsafe {
//!         // Each thread loads one element
//!         TILE[tid] = data[global_idx];
//!     }
//!
//!     thread::sync_threads();
//!
//!     unsafe {
//!         // Now safe to read what other threads wrote
//!         let neighbor = TILE[(tid + 1) % 256];
//!     }
//! }
//! ```
//!
//! # Safety
//!
//! All shared memory access requires `unsafe` because:
//! - Memory is **uninitialized** at kernel start
//! - Multiple threads access concurrently (potential races)
//!
//! Use `thread::sync_threads()` to ensure all writes are visible before reading.
//!
//! # Block Scope
//!
//! - All threads in a block see the **same** shared memory
//! - Different blocks have **independent** copies
//! - Memory does not persist between kernel launches
//!
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

/// Block-scoped shared memory array.
///
/// Declare as `static mut` inside a kernel function. The cuda-oxide compiler
/// recognizes this type and allocates memory in GPU shared memory space
/// (LLVM address space 3).
///
/// # Example
///
/// ```rust,ignore
/// static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
///
/// unsafe {
///     TILE[tid] = value;
/// }
/// thread::sync_threads();
/// unsafe {
///     let v = TILE[other_tid];
/// }
/// ```
///
/// # Type Parameters
///
/// - `T`: Element type (f32, f64, i32, etc.)
/// - `N`: Array size (must be known at compile time)
/// - `ALIGN`: Alignment in bytes (0 = natural alignment, 128 for TMA destinations)
///
/// # Alignment
///
/// By default (`ALIGN = 0`), shared memory uses natural alignment based on the
/// element type. For TMA (Tensor Memory Accelerator) destinations, use `ALIGN = 128`
/// to meet the 128-byte alignment requirement:
///
/// ```rust,ignore
/// // Regular shared memory - natural alignment
/// static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
///
/// // TMA destination - 128-byte alignment required
/// static mut TMA_TILE: SharedArray<f32, 256, 128> = SharedArray::UNINIT;
/// ```
///
/// # Soundness: `!Sync`
///
/// `SharedArray` is intentionally `!Sync`. GPU shared memory is concurrently
/// accessed by all threads in a block, and correctness depends on hardware
/// barriers (`sync_threads` / `bar.sync`) that are invisible to Rust's type
/// system. If the type were `Sync`, it would promise that `&SharedArray` can
/// be safely shared across threads — but concurrent reads after concurrent
/// writes are only safe *after* a barrier, which the compiler cannot verify.
///
/// This is achieved via `PhantomData<UnsafeCell<...>>` (`UnsafeCell` is `!Sync`).
/// The type remains `Send` when `T: Send`.
///
/// All access is through `static mut` (which independently requires `unsafe`),
/// so this primarily guards against future abstractions that might incorrectly
/// rely on a `Sync` bound.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct SharedArray<T, const N: usize, const ALIGN: usize = 0> {
    // PhantomData<UnsafeCell<...>> makes this type !Sync: concurrent access
    // requires external synchronization (sync_threads / bar.sync) that the
    // Rust type system cannot see or enforce. See "Soundness: !Sync" above.
    _marker: PhantomData<UnsafeCell<[T; N]>>,
}

impl<T, const N: usize, const ALIGN: usize> SharedArray<T, N, ALIGN> {
    /// Marker constant for uninitialized shared memory.
    ///
    /// Use this to initialize `static mut` declarations:
    /// ```rust,ignore
    /// static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    /// ```
    pub const UNINIT: Self = Self {
        _marker: PhantomData,
    };

    /// Returns the array length.
    #[inline(always)]
    pub const fn len() -> usize {
        N
    }

    /// Returns true if the array is empty.
    #[inline(always)]
    pub const fn is_empty() -> bool {
        N == 0
    }

    /// Returns the alignment in bytes.
    /// Returns 0 if natural alignment is used.
    #[inline(always)]
    pub const fn alignment() -> usize {
        ALIGN
    }

    /// Returns a raw pointer to the shared memory array.
    ///
    /// This is useful for operations that require a pointer, such as
    /// distributed shared memory (DSMEM) address mapping.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid within a CUDA kernel. The caller must:
    /// - Ensure the pointed-to memory has been initialized
    /// - Use appropriate synchronization before reading data written by other threads
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use with map_shared_rank for DSMEM
    /// let remote_ptr = unsafe { cluster::map_shared_rank(SHMEM.as_ptr(), neighbor_rank) };
    /// ```
    #[inline(never)]
    pub fn as_ptr(&self) -> *const T {
        unreachable!("SharedArray::as_ptr called outside CUDA kernel context")
    }

    /// Returns a mutable raw pointer to the shared memory array.
    ///
    /// This is useful for operations that require a mutable pointer.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid within a CUDA kernel. The caller must:
    /// - Ensure no other thread is concurrently accessing the memory
    /// - Use appropriate synchronization after writing
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Initialize first element
    /// unsafe { SHMEM.as_mut_ptr().write(value) };
    /// ```
    #[inline(never)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        unreachable!("SharedArray::as_mut_ptr called outside CUDA kernel context")
    }

    /// Returns a mutable raw pointer to the shared memory array without
    /// creating a Rust reference to the complete allocation.
    ///
    /// This is the appropriate entry point when multiple CUDA threads derive
    /// disjoint raw pointers from one `static mut SharedArray`. Unlike
    /// [`Self::as_mut_ptr`], the raw receiver does not require each thread to
    /// create an overlapping `&mut SharedArray`.
    ///
    /// # Safety
    ///
    /// `shared` must point to a `static mut SharedArray` in the current CUDA
    /// kernel, normally obtained with `&raw mut`. The returned pointer is valid
    /// only within that kernel. Callers must keep concurrent accesses disjoint
    /// and use the CUDA synchronization required before reading data written by
    /// another thread.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut SCRATCH: SharedArray<f32, 256> = SharedArray::UNINIT;
    /// let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    /// ```
    #[inline(never)]
    pub unsafe fn as_raw_mut_ptr(shared: *mut Self) -> *mut T {
        let _ = shared;
        unreachable!("SharedArray::as_raw_mut_ptr called outside CUDA kernel context")
    }
}

impl<T, const N: usize, const ALIGN: usize> Index<usize> for SharedArray<T, N, ALIGN> {
    type Output = T;

    /// Access shared memory element by index.
    ///
    /// # Panics (in host code)
    ///
    /// This function should never be called on the host. If called outside
    /// a CUDA kernel context, it will panic.
    ///
    /// # Safety (in device code)
    ///
    /// While the `Index` trait method is not marked unsafe, accessing shared
    /// memory through a `static mut` requires an `unsafe` block. The caller
    /// must ensure:
    /// - Index is within bounds (`idx < N`)
    /// - The element has been initialized by some thread
    /// - Appropriate synchronization (`sync_threads`) if reading data written
    ///   by another thread
    #[inline(never)]
    fn index(&self, _idx: usize) -> &Self::Output {
        // This body never executes on GPU - the compiler replaces it with
        // a load from addrspace(3) memory.
        unreachable!("SharedArray::index called outside CUDA kernel context")
    }
}

impl<T, const N: usize, const ALIGN: usize> IndexMut<usize> for SharedArray<T, N, ALIGN> {
    /// Access shared memory element mutably by index.
    ///
    /// # Panics (in host code)
    ///
    /// This function should never be called on the host. If called outside
    /// a CUDA kernel context, it will panic.
    ///
    /// # Safety (in device code)
    ///
    /// While the `IndexMut` trait method is not marked unsafe, accessing shared
    /// memory through a `static mut` requires an `unsafe` block. The caller
    /// must ensure:
    /// - Index is within bounds (`idx < N`)
    /// - No other thread is concurrently accessing the same index (data race)
    #[inline(never)]
    fn index_mut(&mut self, _idx: usize) -> &mut Self::Output {
        // This body never executes on GPU - the compiler replaces it with
        // a store to addrspace(3) memory.
        unreachable!("SharedArray::index_mut called outside CUDA kernel context")
    }
}

// SharedArray<T, N> auto-trait summary:
//   Copy + Clone: yes (derived) — only meaningful inside kernels (ZST marker)
//   Send: yes (when T: Send) — auto-derived
//   Sync: NO (!Sync via UnsafeCell) — concurrent access requires GPU barriers
//         that the Rust type system cannot represent

// ============================================================================
// DynamicSharedArray - Runtime-sized shared memory
// ============================================================================

/// Compile-time minimum alignment marker for a kernel's dynamic shared memory.
///
/// This is injected by `#[launch_contract]`. cuda-oxide removes the call before
/// code generation, so it adds no kernel hot-path instructions. Kernel authors
/// should use the attribute rather than calling this function directly.
#[doc(hidden)]
#[inline(never)]
pub fn __dynamic_shared_alignment<const ALIGN: usize>() {
    // The MIR importer removes this marker after recording ALIGN.
}

/// Dynamic (runtime-sized) shared memory with configurable alignment.
///
/// Unlike [`SharedArray`] which has a compile-time known size, `DynamicSharedArray`
/// allows the shared memory size to be specified at kernel launch time via
/// `LaunchConfig::shared_mem_bytes`.
///
/// This enables CUTLASS-style patterns where the same kernel PTX can be used
/// with different shared memory configurations.
///
/// # Type Parameters
///
/// - `T`: Element type for the returned pointer
/// - `ALIGN`: Base alignment in bytes (default: 16, matching nvcc)
///
/// # Alignment
///
/// The `ALIGN` parameter controls the alignment of the `extern __shared__`
/// declaration in PTX:
///
/// ```rust,ignore
/// // Default alignment (16 bytes, matches nvcc for char[])
/// let smem: *mut f32 = DynamicSharedArray::<f32>::get();
/// // PTX: .extern .shared .align 16 .b8 __dynamic_smem[];
///
/// // TMA-compatible alignment (128 bytes required for TMA operations)
/// let tma_smem: *mut f32 = DynamicSharedArray::<f32, 128>::get();
/// // PTX: .extern .shared .align 128 .b8 __dynamic_smem[];
///
/// // Higher alignment for specific cache patterns
/// let aligned_smem: *mut f32 = DynamicSharedArray::<f32, 256>::get();
/// // PTX: .extern .shared .align 256 .b8 __dynamic_smem[];
/// ```
///
/// Common alignment values:
/// - `16` (default): Matches nvcc, suitable for most use cases
/// - `128`: Required for TMA (Tensor Memory Accelerator) operations
/// - `256`: Cache-line friendly for some architectures
///
/// # Usage
///
/// ```rust,ignore
/// use cuda_device::{kernel, DynamicSharedArray, DisjointSlice};
///
/// #[kernel]
/// pub fn flexible_kernel(data: &[f32], mut out: DisjointSlice<f32>) {
///     // Get typed pointer to dynamic shared memory
///     let smem: *mut f32 = DynamicSharedArray::<f32>::get();
///
///     // Or partition the memory for multiple arrays
///     let smem_a: *mut f32 = DynamicSharedArray::<f32>::get();
///     let smem_b: *mut f32 = DynamicSharedArray::<f32>::offset(1024); // After 1024 bytes
///
///     // Access elements (unsafe - no bounds checking)
///     unsafe {
///         *smem_a.add(thread::threadIdx_x() as usize) = 1.0;
///     }
/// }
///
/// // For TMA operations, use 128-byte alignment:
/// #[kernel]
/// pub fn tma_kernel(tensor_map: *const TmaDescriptor, mut out: DisjointSlice<f32>) {
///     let tma_buffer: *mut u8 = DynamicSharedArray::<u8, 128>::get_raw();
///     // TMA copy to tma_buffer...
/// }
/// ```
///
/// # Host-side Launch
///
/// Specify the shared memory size in the launch configuration:
///
/// ```rust,ignore
/// // SAFETY: argument list matches `flexible_kernel`'s signature.
/// unsafe {
///     cuda_launch! {
///         kernel: flexible_kernel,
///         config: LaunchConfig {
///             grid_dim: (blocks, 1, 1),
///             block_dim: (256, 1, 1),
///             shared_mem_bytes: 2048,  // 512 f32s total
///         },
///         // ...
///     }
/// }
/// ```
///
/// # Memory Partitioning
///
/// All calls to `DynamicSharedArray::get()` and `DynamicSharedArray::offset()` within
/// a kernel reference the **same** underlying memory. Use byte offsets to
/// partition the memory for multiple arrays:
///
/// ```rust,ignore
/// // First array: 256 f32s (1024 bytes) starting at offset 0
/// let array_a: *mut f32 = DynamicSharedArray::<f32>::get();
///
/// // Second array: 256 f32s starting at offset 1024
/// let array_b: *mut f32 = DynamicSharedArray::<f32>::offset(1024);
///
/// // Launch with: shared_mem_bytes = 2048
/// ```
///
/// # Safety
///
/// All operations on dynamic shared memory are inherently unsafe:
/// - No compile-time bounds checking (size is determined at launch)
/// - Memory is uninitialized at kernel start
/// - Multiple threads access concurrently
///
/// The caller must ensure:
/// - `shared_mem_bytes` in launch config is sufficient for all accesses
/// - Proper synchronization (`sync_threads`) between reads and writes
/// - Correct alignment when using `offset()`
///
/// # Soundness: `!Sync`
///
/// Like [`SharedArray`], this type is `!Sync` because concurrent access to
/// GPU shared memory requires hardware barriers that are outside the Rust
/// type system's knowledge. See [`SharedArray`] docs for the full rationale.
#[derive(Copy, Clone)]
pub struct DynamicSharedArray<T, const ALIGN: usize = 16>(PhantomData<UnsafeCell<T>>);

impl<T, const ALIGN: usize> DynamicSharedArray<T, ALIGN> {
    /// Returns the alignment in bytes for this dynamic shared memory.
    #[inline(always)]
    pub const fn alignment() -> usize {
        ALIGN
    }
    /// Get a typed pointer to the start of dynamic shared memory.
    ///
    /// All threads in a block get the same base address. The pointer
    /// is cast to the specified type `T`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let smem: *mut f32 = DynamicSharedArray::<f32>::get();
    /// unsafe {
    ///     *smem.add(tid) = value;
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if called outside a CUDA kernel context (on host).
    #[inline(never)]
    pub fn get() -> *mut T {
        unreachable!("DynamicSharedArray::get called outside CUDA kernel context")
    }

    /// Get a raw byte pointer to the start of dynamic shared memory.
    ///
    /// This is useful for manual memory management or when working
    /// with heterogeneous data layouts.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let raw: *mut u8 = DynamicSharedArray::<u8>::get_raw();
    /// // Cast to specific types as needed
    /// let floats = raw as *mut f32;
    /// let ints = raw.add(1024) as *mut i32;
    /// ```
    #[inline(never)]
    pub fn get_raw() -> *mut u8 {
        unreachable!("DynamicSharedArray::get_raw called outside CUDA kernel context")
    }

    /// Get a typed pointer at the specified byte offset into dynamic shared memory.
    ///
    /// This is used to partition dynamic shared memory into multiple arrays.
    /// The offset is in **bytes**, not elements.
    ///
    /// # Arguments
    ///
    /// * `byte_offset` - Offset in bytes from the start of dynamic shared memory
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // First array at offset 0
    /// let array_a: *mut f32 = DynamicSharedArray::<f32>::get();
    ///
    /// // Second array at offset 1024 bytes (256 f32s after array_a)
    /// let array_b: *mut f32 = DynamicSharedArray::<f32>::offset(1024);
    /// ```
    ///
    /// # Alignment
    ///
    /// Ensure `byte_offset` is properly aligned for type `T`. For example,
    /// `f32` requires 4-byte alignment, `f64` requires 8-byte alignment.
    ///
    /// The base of dynamic shared memory is aligned to `ALIGN` bytes (default 16),
    /// so to get an N-byte aligned pointer, ensure `byte_offset % N == 0`.
    #[inline(never)]
    pub fn offset(byte_offset: usize) -> *mut T {
        // The byte_offset is used by the compiler to compute the pointer.
        // This prevents the compiler from optimizing away the argument.
        let _ = byte_offset;
        unreachable!("DynamicSharedArray::offset called outside CUDA kernel context")
    }
}

/// Convert a generic-address pointer into its raw `.shared` window offset.
///
/// The CUDA C++ `__cvta_generic_to_shared_offset` analog (PTX `cvta.to.shared`).
/// Rust-visible pointer addresses (`ptr as usize`, `ptr::addr`) are CUDA
/// generic addresses; hardware SMEM descriptors (WGMMA and tcgen05 matrix
/// descriptors, whose low bits encode `(start_address >> 4) & 0x3FFF`) are
/// defined on the space-local shared offset instead. Use this to derive
/// descriptor base addresses; do not pass a raw `ptr as u64` there.
///
/// ```rust,ignore
/// static mut SMEM_A: SharedArray<f16, 2048> = SharedArray::UNINIT;
/// let base = unsafe { cvta_generic_to_shared_offset(&raw const SMEM_A as *const u8) };
/// let desc = build_smem_descriptor(base, LBO_BYTES, SBO_BYTES, SWIZZLE_NONE);
/// ```
///
/// # Safety
///
/// `ptr` must be a generic pointer to shared memory (for example one
/// derived from a `SharedArray` static). Converting a pointer that does not
/// point into shared memory yields an unspecified offset.
//
// The importer intercepts calls by the exact rendered def-path
// `cuda_device::shared::cvta_generic_to_shared_offset`. A `pub use` at the crate
// root would change rustc's visible path for this item and silently break
// interception, so import it through this module.
#[inline(never)]
pub unsafe fn cvta_generic_to_shared_offset(ptr: *const u8) -> u64 {
    let _ = ptr;
    unreachable!("cvta_generic_to_shared_offset called outside CUDA kernel context")
}

/// Convert a generic address into its 32-bit `.shared::cta` state-space address.
///
/// Unlike [`cvta_generic_to_shared_offset`], which returns the `u64` carrier
/// used to construct WGMMA and tcgen05 descriptors, this form matches the
/// `u32` address operand consumed by CTA-shared instructions such as `ldmatrix`.
/// Convert a shared base once, then keep byte-offset and swizzle arithmetic in
/// `u32` when the complete addressed allocation fits the CTA shared window.
///
/// # Safety
///
/// `ptr` must be a generic pointer to memory in the current CTA's shared-memory
/// window. Converting any other pointer produces an unspecified address.
//
// Keep this uninlined: mir-importer intercepts the exact rendered def-path.
#[inline(never)]
pub unsafe fn cvta_generic_to_shared_u32(ptr: *const u8) -> u32 {
    let _ = ptr;
    unreachable!("cvta_generic_to_shared_u32 called outside CUDA kernel context")
}

include!("generated/shared_sreg.rs");

mod tile_element {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for i64 {}
    impl Sealed for u64 {}
}

/// Primitive immutable elements accepted by compiler-staged tiles.
///
/// Sealing keeps aliases and shadowed primitive names from changing the
/// initialization or immutability contract established by the macro.
#[doc(hidden)]
pub trait __TileElement: tile_element::Sealed + Copy {}
impl __TileElement for f32 {}
impl __TileElement for f64 {}
impl __TileElement for i32 {}
impl __TileElement for u32 {}
impl __TileElement for i64 {}
impl __TileElement for u64 {}

/// Lifetime guard for a compiler-initialized immutable block tile.
///
/// This implementation detail is public only for proc-macro expansion.
/// Safe kernel code obtains a borrowed slice with `#[block_shared(N)]`.
#[doc(hidden)]
pub struct __ReadOnlyTile<T: __TileElement> {
    pointer: *const T,
    len: usize,
}

impl<T: __TileElement> __ReadOnlyTile<T> {
    /// # Safety
    /// `pointer` must address `len` initialized, properly aligned elements in
    /// the current block's shared memory. All block threads must have finished
    /// initialization and synchronized. No writes may occur while this guard or
    /// its derived borrows are live. The guard must not escape the invocation.
    #[inline(always)]
    pub unsafe fn from_initialized(pointer: *const T, len: usize) -> Self {
        Self { pointer, len }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the constructor establishes initialization, alignment,
        // visibility, and immutability. The borrow cannot outlive this guard.
        if self.len == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(self.pointer, self.len) }
        }
    }
}

/// A bounded read-only matrix view valid for one pipeline callback.
///
/// Out-of-range rows, columns, offsets, and arithmetic overflow return `None`.
/// The same view supports ordinary global-memory helpers and entry-staged tiles.
pub struct BlockTile<'a, T> {
    data: &'a [T],
    offset: usize,
    rows: usize,
    columns: usize,
    stride: usize,
    xor32: bool,
}
impl<'a, T> BlockTile<'a, T> {
    #[inline(always)]
    pub fn new(data: &'a [T], offset: usize, rows: usize, columns: usize, stride: usize) -> Self {
        Self {
            data,
            offset,
            rows,
            columns,
            stride,
            xor32: false,
        }
    }
    /// Permute columns within each 32-element group to avoid shared-bank
    /// conflicts when a warp reads either rows or columns. Rows must be stored
    /// with the same XOR permutation; complete 32-column groups are required.
    #[inline(always)]
    pub fn with_xor32(mut self) -> Self {
        assert!(
            self.columns.is_multiple_of(32),
            "xor32 columns must be divisible by 32"
        );
        self.xor32 = true;
        self
    }
    #[inline(always)]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.data.get(self.index(row, column)?)
    }

    /// Read one element, returning the default value for unavailable coordinates
    /// or arithmetic overflow. This avoids a reference-valued intermediate in
    /// numerical kernels while retaining the same checked bounds as `get`.
    #[inline(always)]
    pub fn load(&self, row: usize, column: usize) -> T
    where
        T: Copy + Default,
    {
        if let Some(index) = self.index(row, column)
            && index < self.data.len()
        {
            self.data[index]
        } else {
            T::default()
        }
    }

    #[inline(always)]
    fn index(&self, row: usize, column: usize) -> Option<usize> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        let column = if self.xor32 {
            (column & !31) + ((column ^ row) & 31)
        } else {
            column
        };
        row.checked_mul(self.stride)?
            .checked_add(self.offset)?
            .checked_add(column)
    }
}

#[cfg(test)]
mod block_tile_tests {
    use super::BlockTile;
    #[test]
    fn xor32_preserves_elements_and_unique_row_and_column_banks() {
        for columns in [32, 64, 128] {
            let mut data = [0usize; 8192];
            let mut written = [false; 8192];
            for row in 0..64 {
                for column in 0..columns {
                    let physical = row * columns + (column & !31) + ((column ^ row) & 31);
                    assert!(!written[physical]);
                    written[physical] = true;
                    data[physical] = row * columns + column;
                }
            }
            assert!(written[..64 * columns].iter().all(|&value| value));
            let tile = BlockTile::new(&data, 0, 64, columns, columns).with_xor32();
            for row in 0..64 {
                for column in 0..columns {
                    assert_eq!(tile.get(row, column), Some(&(row * columns + column)));
                    assert_eq!(tile.load(row, column), row * columns + column);
                }
                for group in 0..columns / 32 {
                    let mut banks = [false; 32];
                    for lane in 0..32 {
                        let column = group * 32 + lane;
                        let bank = (column ^ row) & 31;
                        assert!(!banks[bank]);
                        banks[bank] = true;
                    }
                }
            }
            for column in 0..columns {
                for group in 0..2 {
                    let mut banks = [false; 32];
                    for lane in 0..32 {
                        let row = group * 32 + lane;
                        let bank = (column ^ row) & 31;
                        assert!(!banks[bank]);
                        banks[bank] = true;
                    }
                }
            }
            assert_eq!(tile.get(64, 0), None);
            assert_eq!(tile.get(0, columns), None);
            assert_eq!(tile.load(64, 0), 0);
            assert_eq!(tile.load(0, columns), 0);
        }
    }

    #[test]
    #[should_panic(expected = "xor32 columns must be divisible by 32")]
    fn xor32_rejects_incomplete_column_groups() {
        BlockTile::new(&[0u32; 65], 0, 1, 65, 65).with_xor32();
    }

    #[test]
    fn rows_columns_tail_and_overflow_are_checked() {
        let data = [1u32, 2, 3, 4, 5];
        let view = BlockTile::new(&data, 1, 3, 2, 2);
        assert_eq!(view.get(0, 0), Some(&2));
        assert_eq!(view.get(1, 1), Some(&5));
        assert_eq!(view.get(2, 0), None);
        assert_eq!(view.load(2, 0), 0);
        assert_eq!(view.load(1, 1), 5);
        assert_eq!(BlockTile::new(&data, usize::MAX, 2, 2, 2).load(0, 1), 0);
        assert_eq!(view.get(0, 2), None);
        assert_eq!(view.get(3, 0), None);
        assert_eq!(BlockTile::new(&data, usize::MAX, 2, 2, 2).get(0, 1), None);
        assert_eq!(
            BlockTile::new(&data, 0, usize::MAX, 2, usize::MAX).get(2, 0),
            None
        );
    }
}
