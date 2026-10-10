<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Ready: Changes in the `experiments` Branch

These changes are in `experiments` and not in `main`. [`todo.md`](todo.md)
lists the fixes that are not done.

## Features

### Const Parameters in Launch Contracts

The `requires` relations of `#[launch_contract]` accept unsigned integer const
parameters, for example `width <= N`.

Commit: `dba950f7`.

### Borrowed Device Buffer Views

`DeviceBuffer::as_chunks`, `as_chunks_mut` and `with_chunks` borrow a buffer
with a different element type. They check the size and the alignment. Checked
kernel launchers accept these views.

Commit: `760d4aa3`.

### Safe Load of Generic Modules

`load` and `load_async` are safe for a module that has only generic kernels and
no constants. The load merges the PTX bundles of the image that contains the
artifact anchor of the module. It checks the mapped file and rejects two
definitions of one entry. Entry names with `_TID_` are reserved for generic
specializations.

Commits: `080c5b8b`, `5db1b0f8`.

### Staged Block Tiles

`#[block_shared(N)]` on an immutable numeric slice parameter copies a tile of
the slice to shared memory before the user code. The user code gets a
read-only slice of the tile. The tile reads use `ld.shared`, and the copy has
no bounds check.

Commits: `fe3f4bb7`, `99674beb`, `88cd281d`, `5043589d`.

### Async Tile Pipelines

On sm_80 and later, `#[block_pipeline]` copies the `#[tile_source]` slices with
`cp.async` into two shared-memory slots. The `step` closure gets each tile as a
`BlockTile`.

- `layout = xor32` puts the columns of each row in an XOR order. Warp reads of
  a row or a column then use different banks.
- `BlockTile::load` returns a copy of the element, or the default value out of
  bounds.
- Sources with the same columns, stride and offset share the copy indices.
- The copy calculates the limits one time for each tile. `cp.async` fills the
  elements out of bounds with zero.
- When the rows, the stride and the tile address are 16-byte aligned, a tile
  without the XOR layout uses 16-byte `cp.async.cg`.

`DisjointSlice<T, thread::StripedWarp<N>>` and `thread::striped_index` give
warp-striped writes.

Commits: `f6702b7c`, `c9c1706a`, `c80ea119`, `65a073c7`, `c74a9823`,
`b13e2b56`, `2aa79806`.

### FP8 on Pre-Hopper GPUs (cutile)

On sm_80 to sm_89, cutile stores E4M3FN and E5M2 values as `i8` and converts
them with integer and float operations. Arithmetic on FP8 values gives an
error. sm_90 and later targets keep native FP8.

Commit: `eaae51de`.

## Fixes

### PTX Merge

- The merged module has one `.version` and one `.target`. The `.version` is the
  highest version. The `.target` covers the targets of all bundles.
- The merge renames the device functions and the module-scope variables of
  each bundle. Thus equal names in two bundles do not collide.
- A `.extern` declaration links to the `.visible` definition in another bundle.
- Only the first bundle with DWARF sections keeps them. Before, ptxas did not
  assemble a merged module with different abbreviation tables.

Commits: `bd060dae`, `b861e347`, `17e16979`, `5791e688`, `0d17fdd5`.

### Inline Policy of Generic Device Functions

`#[device]` adds `#[inline(never)]` to a generic function only when the
function has no `inline` attribute, also in `cfg_attr`. Before, rustc ignored
the `inline` attribute of the user.

Commit: `2e7b9847`.

### Integer Warp Reductions on Ampere

An integer `redux.sync` in a module with an unrelated `.f32` instruction got
the `sm_100a` requirement of the `.f32` form. Now only the `.f32` form gets it.

Commit: `fe2795cd`.

### cutile Rustdoc Links and Benchmark Smoke Tests

- Repair the public rustdoc links. Commit: `8208b01b`.
- The benchmark runner gives `--features smoke-test`. GEMM smoke tests use 256
  and 512 square matrices. Commit: `f8f89296`.

## Maintenance

- Rust 1.99.0 stable and nightly-2026-10-09 (rustc 1.101.0). Commit:
  `7133c184`.
- LLVM 23.1.3 provenance of the intrinsics catalog. Commit: `d84ebf06`.
- Rust dependencies in all workspaces, with Syn 3.
  `CUDA_TILE_USE_LLVM_INSTALL_DIR` selects a prebuilt LLVM for `cuda-tile-rs`.
  Commit: `e8f6af97`.
- CUDA 13.4.2, LLVM 23, Nix, GitHub Actions and documentation tools. Commit:
  `126d8619`.
