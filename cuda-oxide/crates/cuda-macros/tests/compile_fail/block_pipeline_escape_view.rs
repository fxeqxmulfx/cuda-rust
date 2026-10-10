// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code, unused_variables)]
use cuda_device::{kernel, launch_contract};
use cuda_device::shared::BlockTile;
#[kernel]
#[launch_contract(domain = 1, block = (128, 1, 1))]
#[block_pipeline(rows = 4, iterations = 2)]
fn invalid<const D: usize>(#[tile_source(columns = D, stride = D, offset = 0)] input: &[f32]) {
    let init = || None;
    let step = |state: &mut Option<BlockTile<'_, f32>>, _iteration: usize, tile: BlockTile<'_, f32>| { *state = Some(tile); };
    let finish = |_state: Option<BlockTile<'_, f32>>| {};
}
fn main() {}
