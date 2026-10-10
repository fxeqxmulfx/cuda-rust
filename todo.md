<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TODO: Fixes for the `experiments` Branch

Make one commit with a test for each fix.

## Bugs

- [x] `just clippy` and `just check` fail because of `clippy::replace_box` in
  the [pipeline tests](cuda-oxide/crates/cuda-macros/src/tests/pipeline.rs).
  Write `*input.block = syn::parse2(body).unwrap();`.
- [ ] A generic kernel specialization is lost when only a library that does not
  define the kernel instantiates it. No symbol refers to the bundle anchor of
  that library, so the linker does not extract the bundle. The merged module
  then does not contain the entry. Make the linker keep the bundle of each
  crate that instantiates a specialization. When the linker keeps them, two
  such libraries with the same specialization give the same entry two times.
  Then the merge gives `ConflictingEntry`.
- [ ] The `load` of a generic module merges the PTX bundles of all crates in
  the image. Thus two crates with a non-generic kernel of the same name give
  `ConflictingEntry`. Also, the driver compiles the PTX of all crates for each
  of these modules. Merge only the bundles that the module needs.
- [ ] Only the first merged bundle with DWARF sections keeps them, because
  ptxas reads all compile units with the first abbreviation table. The other
  bundles keep only line information. Combine the abbreviation tables.
- [ ] CI uses only LLVM 23. In `main`, the unit tests use LLVM 21, the
  documented minimum, and the example build uses LLVM 22. Add jobs for these
  versions, or change the documented minimum.

## Zero Cost

Examine the PTX of the `block_shared` and `block_pipeline` examples before and
after each device fix.

- [x] [`__ReadOnlyTile::as_slice`](cuda-oxide/crates/cuda-device/src/shared.rs)
  returns `&[]` for an empty tile. The `select` between that global address
  and the shared pointer changes the tile reads to generic `ld`. Remove the
  branch: `from_raw_parts` accepts a length of zero with a non-null, aligned
  pointer. A test without the branch gave `ld.shared` for all four loads of
  `block_shared`.
- [x] The `#[block_shared]` copy loop in
  [`BlockShared::prologue`](cuda-oxide/crates/cuda-macros/src/kernel/shared.rs)
  does a bounds check with a `trap` for each element. The loop condition
  already gives `base + lane < len`. Use `get_unchecked`.
- [x] The pipeline copy loop in
  [`BlockPipeline::body`](cuda-oxide/crates/cuda-macros/src/kernel/pipeline.rs)
  does checked 64-bit arithmetic for each element. After the first tile, each
  element gets `mul.hi.u64`, `mul.lo.s64` and two overflow checks. Check the
  start and the end of each tile one time. Then compare only `src < len` for
  each element, or nothing when all of the tile is in the limits.
- [x] The pipeline copies 4 or 8 bytes with each `cp.async.ca`. Add a 16-byte
  `cp.async.cg` path when `columns` and `stride` are multiples of 16 bytes.
  Check the alignment of the source address one time for each tile.
- [x] Each [`striped_index`](cuda-oxide/crates/cuda-device/src/thread.rs) call
  checks `thread >= threads`, which cannot occur, and checks that the lane owns
  the column. It also calculates the warp base again. Add an iterator over the
  columns of the lane that calculates the base one time.
- [x] [`BlockTile::index`](cuda-oxide/crates/cuda-device/src/shared.rs) does
  `checked_mul` and `checked_add` for each access. In the staged pipeline, LLVM
  removes them, but views with a runtime offset or stride keep them. Check the
  limits one time in `BlockTile::new`.
- [x] Host, one time for each module load:
  [`EditScript::replace`](cuda-oxide/crates/ptx-parse/src/edit.rs) finds
  overlaps with a linear search, so a merge is O(n²) in the number of edits.
  Keep the edits sorted and use a binary search.
- [x] Host, one time for each module load:
  [`register_merged_module_entries`](cuda-oxide/crates/cuda-host/src/entry_registry.rs)
  parses the merged PTX again. Use the entry names that the merge found.
- [ ] [FP8 emulation](cutile-rs/cutile-ir/src/emulation.rs) changes FP8
  conversions on sm_80–sm_89 to software conversions that the source does not
  show. Use the emulation only when the user selects it. Without it, keep the
  capability error.
