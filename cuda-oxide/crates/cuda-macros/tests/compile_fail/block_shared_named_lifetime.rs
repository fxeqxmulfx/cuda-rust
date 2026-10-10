// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code, unused_variables)]
use cuda_device::{kernel, launch_contract};

#[kernel]
#[launch_contract(domain = 1, block = (32, 1, 1))]
fn invalid<'a>(#[block_shared(128)] input: &'a [f32]) {}
fn main() {}
