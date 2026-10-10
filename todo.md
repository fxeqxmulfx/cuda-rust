<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TODO: Fixes for the `experiments` Branch

Make one commit with a test for each fix.

## Bugs

- [x] With `lto = "fat"` or `lto = "thin"`, the build of `cross_crate_embedded`
  fails: "failed to get bitcode from object file for LTO (Can't find section
  .llvmbc)". rustc reads LLVM bitcode from each Rust object of the rlibs. The
  artifact objects (`*.embed.o`) are Rust objects without bitcode. `main` has
  the same problem. Keep the artifact objects out of the LTO input. Then
  examine the LTO modes that `retain_specialization_artifacts` skips.
- [x] The DWARF merge adds the file index offset of a bundle to its
  `DW_AT_decl_file` and `DW_AT_call_file` values. LLVM writes small values with
  `DW_FORM_data1`. If a value is more than 255 after the offset, the bundle
  keeps only its line information. Thus in a module with more than 255 `.file`
  directives, the last bundles can lose their DWARF. Give a larger form to
  these values, and change the unit lengths and the references.

## Zero Cost

- [x] `DisjointSlice::get_mut` returns `Option<&mut T>`. `None` is a null
  pointer, thus the caller compares the reference with null. After the bounds
  check, the PTX still compares the slice pointer with null. In the PTX of
  `vecadd`, this is `setp.eq.b64` with `cvta.to.global` of 0. `main` has the
  same compare. Remove it.

## Maintenance

- [ ] In `cuda-oxide/`, the cargo alias `oxide` in `.cargo/config.toml` has the
  same name as an installed `cargo-oxide` subcommand. Cargo then gives a
  warning that this will become an error.
