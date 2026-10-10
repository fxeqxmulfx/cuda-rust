// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compare ordinary inline code, indexed generic device helpers, and array iterators.
//! Indexed helpers must match ordinary code bit for bit; iterator rounding is checked against f64.
#![allow(clippy::needless_range_loop)]
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, device, kernel, launch_contract, thread};
use cuda_host::cuda_module;
use std::time::Instant;

#[inline(always)]
fn ordinary<const W: usize>(channel: usize, coefficients: &[f32], values: &[f32]) -> f32 {
    let mut sum = 0.0;
    let whole = coefficients.len() / 4 * 4;
    for row in (0..whole).step_by(4) {
        let mut products = [0.0; 4];
        for local in 0..4 {
            products[local] = coefficients[row + local] * values[(row + local) * W + channel];
        }
        for local in 0..4 {
            sum += products[local];
        }
    }
    for row in whole..coefficients.len() {
        sum += coefficients[row] * values[row * W + channel];
    }
    sum
}

#[device]
fn device_default<const W: usize>(channel: usize, coefficients: &[f32], values: &[f32]) -> f32 {
    let mut sum = 0.0;
    let whole = coefficients.len() / 4 * 4;
    for row in (0..whole).step_by(4) {
        let mut products = [0.0; 4];
        for local in 0..4 {
            products[local] = coefficients[row + local] * values[(row + local) * W + channel];
        }
        for local in 0..4 {
            sum += products[local];
        }
    }
    for row in whole..coefficients.len() {
        sum += coefficients[row] * values[row * W + channel];
    }
    sum
}

#[device]
#[inline(always)]
fn device_indexed<const W: usize>(channel: usize, coefficients: &[f32], values: &[f32]) -> f32 {
    let mut sum = 0.0;
    let whole = coefficients.len() / 4 * 4;
    for row in (0..whole).step_by(4) {
        let mut products = [0.0; 4];
        for local in 0..4 {
            products[local] = coefficients[row + local] * values[(row + local) * W + channel];
        }
        for local in 0..4 {
            sum += products[local];
        }
    }
    for row in whole..coefficients.len() {
        sum += coefficients[row] * values[row * W + channel];
    }
    sum
}

#[device]
#[inline(always)]
fn device_iterator<const W: usize>(channel: usize, coefficients: &[f32], values: &[f32]) -> f32 {
    let mut sum = 0.0;
    let whole = coefficients.len() / 4 * 4;
    for row in (0..whole).step_by(4) {
        let mut products = [0.0; 4];
        for local in 0..4 {
            products[local] = coefficients[row + local] * values[(row + local) * W + channel];
        }
        for product in products {
            sum += product;
        }
    }
    for row in whole..coefficients.len() {
        sum += coefficients[row] * values[row * W + channel];
    }
    sum
}

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (W >= 1, values.len() == coefficients.len() * W))]
    pub fn ordinary<const W: usize>(
        coefficients: &[f32],
        values: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let channel = index.get() % W;
        if let Some(value) = out.get_mut(index) {
            *value = super::ordinary::<W>(channel, coefficients, values);
        }
    }
    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (W >= 1, values.len() == coefficients.len() * W))]
    pub fn device_default<const W: usize>(
        coefficients: &[f32],
        values: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let channel = index.get() % W;
        if let Some(value) = out.get_mut(index) {
            *value = super::device_default::<W>(channel, coefficients, values);
        }
    }
    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (W >= 1, values.len() == coefficients.len() * W))]
    pub fn device_indexed<const W: usize>(
        coefficients: &[f32],
        values: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let channel = index.get() % W;
        if let Some(value) = out.get_mut(index) {
            *value = super::device_indexed::<W>(channel, coefficients, values);
        }
    }
    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (W >= 1, values.len() == coefficients.len() * W))]
    pub fn device_iterator<const W: usize>(
        coefficients: &[f32],
        values: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let channel = index.get() % W;
        if let Some(value) = out.get_mut(index) {
            *value = super::device_iterator::<W>(channel, coefficients, values);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const W: usize = 128;
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let module = kernels::load(&context)?;
    for rows in [1, 3, 4, 5, 255, 256, 257, 4096] {
        let coefficients: Vec<f32> = (0..rows)
            .map(|row| ((row * 7 % 23) as f32 - 11.0) * 0.03)
            .collect();
        let values: Vec<f32> = (0..rows * W)
            .map(|i| ((i * 3 % 31) as f32 - 15.0) * 0.04)
            .collect();
        let reference: Vec<f64> = (0..W)
            .map(|channel| {
                (0..rows)
                    .map(|row| f64::from(coefficients[row]) * f64::from(values[row * W + channel]))
                    .sum()
            })
            .collect();
        let coefficients = DeviceBuffer::from_host(&stream, &coefficients)?;
        let values = DeviceBuffer::from_host(&stream, &values)?;
        let mut output = DeviceBuffer::<f32>::zeroed(&stream, W * 32)?;
        let config = LaunchConfig1D::new(32, 128, 0);
        let mut first: Option<Vec<f32>> = None;
        for kind in 0..4 {
            let start = Instant::now();
            match kind {
                0 => {
                    let launch = module.prepare_ordinary::<W>(config)?;
                    module.ordinary(&stream, &launch, &coefficients, &values, &mut output)?;
                }
                1 => {
                    let launch = module.prepare_device_default::<W>(config)?;
                    module.device_default(&stream, &launch, &coefficients, &values, &mut output)?;
                }
                2 => {
                    let launch = module.prepare_device_indexed::<W>(config)?;
                    module.device_indexed(&stream, &launch, &coefficients, &values, &mut output)?;
                }
                3 => {
                    let launch = module.prepare_device_iterator::<W>(config)?;
                    module.device_iterator(
                        &stream,
                        &launch,
                        &coefficients,
                        &values,
                        &mut output,
                    )?;
                }
                _ => unreachable!(),
            }
            let elapsed = start.elapsed();
            let result = output.to_host_vec(&stream)?;
            for (i, &actual) in result.iter().enumerate() {
                let expected = reference[i % W];
                assert!(
                    (f64::from(actual) - expected).abs() < 3e-5 * (1.0 + expected.abs()),
                    "rows={rows} variant={kind} channel={}: actual={actual} expected={expected}",
                    i % W
                );
            }
            if let Some(first) = &first {
                if kind < 3 {
                    assert!(
                        first
                            .iter()
                            .zip(&result)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "indexed helper changed rounding: rows={rows} variant={kind}"
                    );
                } else {
                    let max_delta = first
                        .iter()
                        .zip(&result)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    println!("rows={rows} iterator_max_delta={max_delta:e}");
                }
            } else {
                first = Some(result);
            }
            println!(
                "rows={rows} variant={kind} launch_us={:.3}: PASS",
                elapsed.as_secs_f64() * 1e6
            );
        }
    }
    Ok(())
}
