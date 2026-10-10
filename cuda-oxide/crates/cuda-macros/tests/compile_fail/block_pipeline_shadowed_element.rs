// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code, unused_variables, non_camel_case_types)]
use cuda_device::{kernel, launch_contract};
use cuda_device::shared::BlockTile;
type f32 = *const u32;
#[kernel]
#[launch_contract(domain = 1, block = (128, 1, 1))]
#[block_pipeline(rows = 4, iterations = 2)]
fn invalid<const D: usize>(#[tile_source(columns = D, stride = D, offset = 0)] input: &[f32]) {
    let init = || ();
    let step = |_: &mut (), _: usize, _: BlockTile<'_, f32>| {};
    let finish = |_: ()| {};
}
fn main() {}
