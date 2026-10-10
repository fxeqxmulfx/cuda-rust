/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests that all float ElementTypes can:
//! 1. Create a 1D tensor of ones
//! 2. Add two such tensors in a kernel
//! 3. Convert the result to f32
//! 4. Verify correctness on the host

use cuda_core::{f8e4m3fn, f8e5m2, DType};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{CompileOptions, DeviceOp, TileKernel, ToHostVecOp};
use half::{bf16, f16};
use std::sync::Arc;

mod common;

#[cutile::module]
mod float_add_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add_f16<const B: i32>(
        out: &mut Tensor<f16, { [B] }>,
        a: &Tensor<f16, { [-1] }>,
        b: &Tensor<f16, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_b = b.load_tile(shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    #[cutile::entry()]
    fn add_bf16<const B: i32>(
        out: &mut Tensor<bf16, { [B] }>,
        a: &Tensor<bf16, { [-1] }>,
        b: &Tensor<bf16, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_b = b.load_tile(shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    #[cutile::entry()]
    fn add_f32<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_b = b.load_tile(shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    #[cutile::entry()]
    fn add_f64<const B: i32>(
        out: &mut Tensor<f64, { [B] }>,
        a: &Tensor<f64, { [-1] }>,
        b: &Tensor<f64, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_b = b.load_tile(shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    #[cutile::entry()]
    fn increment_fp8<T: ElementType, const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        input: &Tensor<T, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let mut current: Tile<T, { [B] }> = input.load_tile(shape![B], [pid]);
        for _i in 0..2i32 {
            let expanded: Tile<f32, { [B] }> = convert_tile(current);
            let narrowed: Tile<T, { [B] }> = convert_tile(expanded + constant(1.0f32, shape![B]));
            current = narrowed;
        }
        let result: Tile<f32, { [B] }> = convert_tile(current);
        out.store(result);
    }

    // Note: FP8 types (f8e4m3fn, f8e5m2) don't support direct arithmetic
    // on GPU — they're storage/quantization formats. FP8 add kernels are
    // not included here. Use convert to f16/f32 for compute.
}

use float_add_module::{add_bf16, add_f16, add_f32, add_f64};

#[test]
fn add_ones_f16_and_convert_to_f32() {
    common::with_test_stack(|| {
        let a = cutile::api::ones::<f16>(&[128]).sync().expect("alloc a");
        let b = cutile::api::ones::<f16>(&[128]).sync().expect("alloc b");
        let mut out = cutile::api::zeros::<f16>(&[128]).sync().expect("alloc out");

        add_f16((&mut out).partition([128]), &a, &b)
            .sync()
            .expect("add_f16 failed");

        let result_f32: Vec<f32> = cutile::api::convert::<f16, f32>(Arc::new(out))
            .sync()
            .expect("convert failed")
            .dup()
            .to_host_vec()
            .sync()
            .expect("to_host");

        assert_eq!(result_f32.len(), 128);
        for (i, &v) in result_f32.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-3,
                "f16: element {i} = {v}, expected 2.0"
            );
        }
    });
}

#[test]
fn add_ones_bf16_and_convert_to_f32() {
    common::with_test_stack(|| {
        let a = cutile::api::ones::<bf16>(&[128]).sync().expect("alloc a");
        let b = cutile::api::ones::<bf16>(&[128]).sync().expect("alloc b");
        let mut out = cutile::api::zeros::<bf16>(&[128])
            .sync()
            .expect("alloc out");

        add_bf16((&mut out).partition([128]), &a, &b)
            .sync()
            .expect("add_bf16 failed");

        let result_f32: Vec<f32> = cutile::api::convert::<bf16, f32>(Arc::new(out))
            .sync()
            .expect("convert failed")
            .dup()
            .to_host_vec()
            .sync()
            .expect("to_host");

        assert_eq!(result_f32.len(), 128);
        for (i, &v) in result_f32.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-2,
                "bf16: element {i} = {v}, expected 2.0"
            );
        }
    });
}

#[test]
fn add_ones_f32_direct() {
    common::with_test_stack(|| {
        let a = cutile::api::ones::<f32>(&[128]).sync().expect("alloc a");
        let b = cutile::api::ones::<f32>(&[128]).sync().expect("alloc b");
        let mut out = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc out");

        add_f32((&mut out).partition([128]), &a, &b)
            .sync()
            .expect("add_f32 failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 128);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "f32: element {i} = {v}, expected 2.0"
            );
        }
    });
}

#[test]
fn add_ones_f64_and_convert_to_f32() {
    common::with_test_stack(|| {
        let a = cutile::api::ones::<f64>(&[128]).sync().expect("alloc a");
        let b = cutile::api::ones::<f64>(&[128]).sync().expect("alloc b");
        let mut out = cutile::api::zeros::<f64>(&[128]).sync().expect("alloc out");

        add_f64((&mut out).partition([128]), &a, &b)
            .sync()
            .expect("add_f64 failed");

        let result_f32: Vec<f32> = cutile::api::convert::<f64, f32>(Arc::new(out))
            .sync()
            .expect("convert failed")
            .dup()
            .to_host_vec()
            .sync()
            .expect("to_host");

        assert_eq!(result_f32.len(), 128);
        for (i, &v) in result_f32.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "f64: element {i} = {v}, expected 2.0"
            );
        }
    });
}

// FP8 types don't support direct arithmetic (addf) on GPU — they're
// storage formats. Test the create → convert → check pipeline instead.

#[test]
fn create_ones_f8e4m3fn_and_convert_to_f32() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::Fp8]) {
            return;
        }
        let ones = cutile::api::ones::<f8e4m3fn>(&[128]).sync().expect("alloc");

        let result_f32: Vec<f32> = cutile::api::convert::<f8e4m3fn, f32>(Arc::new(ones))
            .sync()
            .expect("convert failed")
            .dup()
            .to_host_vec()
            .sync()
            .expect("to_host");

        assert_eq!(result_f32.len(), 128);
        for (i, &v) in result_f32.iter().enumerate() {
            assert_eq!(v, 1.0, "f8e4m3fn: element {i}");
        }
    });
}

#[test]
fn create_ones_f8e5m2_and_convert_to_f32() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::Fp8]) {
            return;
        }
        let ones = cutile::api::ones::<f8e5m2>(&[128]).sync().expect("alloc");

        let result_f32: Vec<f32> = cutile::api::convert::<f8e5m2, f32>(Arc::new(ones))
            .sync()
            .expect("convert failed")
            .dup()
            .to_host_vec()
            .sync()
            .expect("to_host");

        assert_eq!(result_f32.len(), 128);
        for (i, &v) in result_f32.iter().enumerate() {
            assert_eq!(v, 1.0, "f8e5m2: element {i}");
        }
    });
}

// The tests below select the FP8 emulation. It has an effect only on `sm_80`
// through `sm_89`.
fn fp8_options() -> CompileOptions {
    CompileOptions::new().fp8_emulation(true)
}

fn zeros<T: DType>(len: usize) -> Tensor<T> {
    cutile::api::copy_host_vec_to_device(&Arc::new(vec![T::zero(); len]))
        .sync()
        .expect("copy zeros")
}

fn full<T: DType>(value: T, len: usize) -> Tensor<T> {
    let mut tensor = zeros(len);
    cutile::kernels::creation::full(
        cutile::prelude::value(value),
        (&mut tensor).partition([128]),
    )
    .compile_options(fp8_options())
    .sync()
    .expect("fill");
    tensor
}

fn arange<T: DType>(len: usize) -> Tensor<T> {
    let mut tensor = zeros(len);
    cutile::kernels::creation::arange((&mut tensor).partition([128]))
        .compile_options(fp8_options())
        .sync()
        .expect("arange");
    tensor
}

fn convert<From: DType, To: DType>(src: &Tensor<From>) -> Vec<To> {
    let mut dst = zeros(src.size());
    cutile::kernels::conversion::convert(src, (&mut dst).partition([128]))
        .compile_options(fp8_options())
        .sync()
        .expect("convert");
    dst.dup().to_host_vec().sync().expect("copy to host")
}

// Independent numerical reference: enumerate representable values instead of
// reproducing the compiler's integer rounding algorithm.
fn fp8_value(byte: u8, fraction_bits: i32, bias: i32, finite_only: bool) -> f64 {
    let magnitude = byte & 0x7f;
    let fraction = i32::from(magnitude) & ((1 << fraction_bits) - 1);
    let exponent = i32::from(magnitude) >> fraction_bits;
    let value = if finite_only && magnitude == 0x7f {
        f64::NAN
    } else if !finite_only && exponent == 31 {
        if fraction == 0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    } else if exponent == 0 {
        f64::from(fraction) * 2f64.powi(1 - bias - fraction_bits)
    } else {
        (1.0 + f64::from(fraction) / f64::from(1 << fraction_bits)) * 2f64.powi(exponent - bias)
    };
    if byte & 0x80 != 0 {
        -value
    } else {
        value
    }
}

fn nearest_fp8(value: f64, fraction: i32, bias: i32, finite_only: bool) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    if value.is_nan() {
        return sign | 0x7f;
    }
    let maximum = if finite_only { 0x7e } else { 0x7b };
    if value.abs() >= fp8_value(maximum, fraction, bias, finite_only) {
        return sign | maximum;
    }
    let mut best = 0;
    let mut distance = value.abs();
    for byte in 1..=maximum {
        let candidate = (value.abs() - fp8_value(byte, fraction, bias, finite_only)).abs();
        if candidate < distance || (candidate == distance && byte & 1 == 0) {
            best = byte;
            distance = candidate;
        }
    }
    sign | best
}

fn check_fp8_decoding<T: cuda_core::DType>(
    wrap: fn(u8) -> T,
    fraction: i32,
    bias: i32,
    finite_only: bool,
) {
    let bytes = Arc::new((0..=255).map(wrap).collect::<Vec<_>>());
    let src = cutile::api::copy_host_vec_to_device(&bytes)
        .sync()
        .expect("copy FP8 bytes");
    let f32_values = convert::<T, f32>(&src);
    let f64_values = convert::<T, f64>(&src);
    let f16_values = convert::<T, f16>(&src);
    let bf16_values = convert::<T, bf16>(&src);
    for byte in 0..=255u8 {
        let expected = fp8_value(byte, fraction, bias, finite_only);
        if expected.is_nan() {
            assert!(f32_values[usize::from(byte)].is_nan(), "FP8 byte {byte:#x}");
            assert!(f64_values[usize::from(byte)].is_nan(), "FP8 byte {byte:#x}");
            assert!(f16_values[usize::from(byte)].is_nan(), "FP8 byte {byte:#x}");
            assert!(
                bf16_values[usize::from(byte)].is_nan(),
                "FP8 byte {byte:#x}"
            );
        } else {
            assert_eq!(
                f32_values[usize::from(byte)].to_bits(),
                (expected as f32).to_bits(),
                "FP8 byte {byte:#x}"
            );
            assert_eq!(
                f64_values[usize::from(byte)].to_bits(),
                expected.to_bits(),
                "FP8 byte {byte:#x}"
            );
            assert_eq!(
                f16_values[usize::from(byte)].to_bits(),
                f16::from_f64(expected).to_bits(),
                "FP8 byte {byte:#x}"
            );
            assert_eq!(
                bf16_values[usize::from(byte)].to_bits(),
                bf16::from_f64(expected).to_bits(),
                "FP8 byte {byte:#x}"
            );
        }
    }
}

#[test]
fn decode_every_fp8_bit_pattern() {
    common::with_test_stack(|| {
        check_fp8_decoding(f8e4m3fn, 3, 7, true);
        check_fp8_decoding(f8e5m2, 2, 15, false);
    });
}

fn check_fp8_quantization<T: cuda_core::DType>(
    bits: fn(T) -> u8,
    fraction: i32,
    bias: i32,
    finite_only: bool,
) {
    let maximum = if finite_only { 0x7e } else { 0x7b };
    let mut doubles = vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::MAX,
        -f64::MAX,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
    ];
    let mut floats = vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];
    for byte in 0..maximum {
        let midpoint = (fp8_value(byte, fraction, bias, finite_only)
            + fp8_value(byte + 1, fraction, bias, finite_only))
            * 0.5;
        for raw in [
            midpoint.to_bits() - 1,
            midpoint.to_bits(),
            midpoint.to_bits() + 1,
        ] {
            let x = f64::from_bits(raw);
            doubles.extend([x, -x]);
        }
        for raw in [
            (midpoint as f32).to_bits() - 1,
            (midpoint as f32).to_bits(),
            (midpoint as f32).to_bits() + 1,
        ] {
            let x = f32::from_bits(raw);
            floats.extend([x, -x]);
        }
    }
    let host = Arc::new(doubles.clone());
    let src = cutile::api::copy_host_vec_to_device(&host)
        .sync()
        .expect("copy f64");
    let encoded = convert::<f64, T>(&src);
    for (&x, &actual) in doubles.iter().zip(&encoded) {
        let actual = bits(actual);
        if x.is_nan() {
            assert!(fp8_value(actual, fraction, bias, finite_only).is_nan());
        } else {
            assert_eq!(
                actual,
                nearest_fp8(x, fraction, bias, finite_only),
                "f64 input {x:?}"
            );
        }
    }
    let host = Arc::new(floats.clone());
    let src = cutile::api::copy_host_vec_to_device(&host)
        .sync()
        .expect("copy f32");
    let encoded = convert::<f32, T>(&src);
    for (&x, &actual) in floats.iter().zip(&encoded) {
        let actual = bits(actual);
        if x.is_nan() {
            assert!(fp8_value(actual, fraction, bias, finite_only).is_nan());
        } else {
            assert_eq!(
                actual,
                nearest_fp8(f64::from(x), fraction, bias, finite_only),
                "f32 input {x:?}"
            );
        }
    }
}

#[test]
fn fp8_quantization_rounds_ties_even_and_saturates() {
    common::with_test_stack(|| {
        check_fp8_quantization(|x: f8e4m3fn| x.0, 3, 7, true);
        check_fp8_quantization(|x: f8e5m2| x.0, 2, 15, false);
    });
}

#[test]
fn fp8_fill_preserves_bytes_and_partial_partitions() {
    common::with_test_stack(|| {
        for byte in [0x00, 0x80, 0x81, 0x7b, 0x7e, 0x7f, 0xff] {
            let e4 = full(f8e4m3fn(byte), 129);
            let host = e4.dup().to_host_vec().sync().expect("copy E4M3FN");
            assert_eq!(host, vec![f8e4m3fn(byte); 129]);
            let e5 = full(f8e5m2(byte), 129);
            let host = e5.dup().to_host_vec().sync().expect("copy E5M2");
            assert_eq!(host, vec![f8e5m2(byte); 129]);
        }
    });
}

#[test]
fn fp8_conversions_work_with_loop_carried_values() {
    common::with_test_stack(|| {
        let input = full(f8e4m3fn::one(), 129);
        let mut output = zeros::<f32>(129);
        float_add_module::increment_fp8((&mut output).partition([128]), &input)
            .compile_options(fp8_options())
            .sync()
            .expect("FP8 loop");
        assert_eq!(
            output.dup().to_host_vec().sync().expect("copy result"),
            vec![3.0f32; 129]
        );
    });
}

#[test]
fn fp8_converts_to_and_from_integer_tensors() {
    common::with_test_stack(|| {
        let e4_integer = convert::<f8e4m3fn, i32>(&arange(129));
        let e5_integer = convert::<f8e5m2, i32>(&arange(129));
        for i in 0..129 {
            let e4 = nearest_fp8(i as f64, 3, 7, true);
            let e5 = nearest_fp8(i as f64, 2, 15, false);
            assert_eq!(e4_integer[i], fp8_value(e4, 3, 7, true) as i32);
            assert_eq!(e5_integer[i], fp8_value(e5, 2, 15, false) as i32);
        }
    });
}
