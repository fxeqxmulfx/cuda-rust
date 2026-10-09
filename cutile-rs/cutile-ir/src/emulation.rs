/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Target-specific legalization of FP8 storage and conversions.
//!
//! On pre-Hopper targets the byte in memory is unchanged, while the emitted
//! element type is i8. Numerical conversions use ordinary integer/float ops.

use crate::bytecode::{encoding::EncodingWriter, Opcode};
use crate::capabilities::{CapabilityError, TargetCapabilities};
use crate::ir::*;
use crate::OpBuilder;

fn fp8(s: ScalarType) -> bool {
    matches!(s, ScalarType::F8E4M3FN | ScalarType::F8E5M2)
}

fn contains_fp8(ty: &Type) -> bool {
    match ty {
        Type::Scalar(s) => fp8(*s),
        Type::Pointer(p) => fp8(p.pointee),
        Type::Tile(t) => match &t.element_type {
            TileElementType::Scalar(s) => fp8(*s),
            TileElementType::Pointer(p) => fp8(p.pointee),
        },
        Type::TensorView(t) => fp8(t.element_type),
        Type::PartitionView(t) => fp8(t.tensor_view.element_type),
        Type::GatherScatterView(t) => fp8(t.tensor_view.element_type),
        Type::StridedView(t) => fp8(t.tensor_view.element_type),
        Type::Func(f) => f.inputs.iter().chain(&f.results).any(contains_fp8),
        Type::WithPointerAttribute(base, _) => contains_fp8(base),
        Type::Token => false,
    }
}

fn attr_contains_fp8(attr: &Attribute) -> bool {
    match attr {
        Attribute::Type(t) | Attribute::Integer(_, t) | Attribute::Float(_, t) => contains_fp8(t),
        Attribute::DenseElements(d) => contains_fp8(&d.element_type),
        Attribute::Array(attrs) => attrs.iter().any(attr_contains_fp8),
        Attribute::Dictionary(attrs) => attrs.iter().any(|(_, a)| attr_contains_fp8(a)),
        _ => false,
    }
}

/// Replace FP8 storage types recursively, keeping shapes, strides and pointers.
fn map_type(ty: &mut Type) {
    let scalar = |s: &mut ScalarType| {
        if fp8(*s) {
            *s = ScalarType::I8;
        }
    };
    match ty {
        Type::Scalar(s) => scalar(s),
        Type::Pointer(p) => scalar(&mut p.pointee),
        Type::Tile(t) => match &mut t.element_type {
            TileElementType::Scalar(s) => scalar(s),
            TileElementType::Pointer(p) => scalar(&mut p.pointee),
        },
        Type::TensorView(t) => scalar(&mut t.element_type),
        Type::PartitionView(t) => scalar(&mut t.tensor_view.element_type),
        Type::GatherScatterView(t) => scalar(&mut t.tensor_view.element_type),
        Type::StridedView(t) => scalar(&mut t.tensor_view.element_type),
        Type::Func(f) => {
            for t in f.inputs.iter_mut().chain(&mut f.results) {
                map_type(t);
            }
        }
        Type::WithPointerAttribute(base, _) => map_type(base),
        Type::Token => {}
    }
}

fn map_attr(attr: &mut Attribute) {
    match attr {
        Attribute::Float(value, ty) if matches!(&*ty, Type::Scalar(s) if fp8(*s)) => {
            // Reuse the constant writer, including E5M2 infinity encodings.
            let mut writer = EncodingWriter::new();
            writer.write_ap_float(*value, ty);
            *attr = Attribute::int(i64::from(writer.as_bytes()[0]), ScalarType::I8);
        }
        Attribute::Type(t) | Attribute::Integer(_, t) | Attribute::Float(_, t) => map_type(t),
        Attribute::DenseElements(d) => map_type(&mut d.element_type),
        Attribute::Array(attrs) => {
            for a in attrs {
                map_attr(a);
            }
        }
        Attribute::Dictionary(attrs) => {
            for (_, a) in attrs {
                map_attr(a);
            }
        }
        _ => {}
    }
}

fn scalar_type(ty: &Type) -> Option<ScalarType> {
    match ty {
        Type::Tile(TileType {
            element_type: TileElementType::Scalar(s),
            ..
        }) => Some(*s),
        _ => None,
    }
}

fn error(op: &Operation, caps: &TargetCapabilities, why: &str) -> CapabilityError {
    CapabilityError {
        message: format!(
            "software FP8 on {}: cuda_tile.{}: {why}",
            caps.architecture,
            op.opcode.name()
        ),
        location: Box::new(op.location.clone()),
    }
}

fn check_padding(ty: &Type) -> bool {
    match ty {
        Type::PartitionView(t) if fp8(t.tensor_view.element_type) => {
            t.padding_value.is_none_or(|p| p == PaddingValue::Zero)
        }
        Type::GatherScatterView(t) if fp8(t.tensor_view.element_type) => {
            t.padding_value.is_none_or(|p| p == PaddingValue::Zero)
        }
        Type::StridedView(t) if fp8(t.tensor_view.element_type) => {
            t.padding_value.is_none_or(|p| p == PaddingValue::Zero)
        }
        Type::Func(f) => f.inputs.iter().chain(&f.results).all(check_padding),
        Type::WithPointerAttribute(base, _) => check_padding(base),
        _ => true,
    }
}

fn check_attr_padding(attr: &Attribute) -> bool {
    match attr {
        Attribute::Type(t) | Attribute::Integer(_, t) | Attribute::Float(_, t) => check_padding(t),
        Attribute::DenseElements(d) => check_padding(&d.element_type),
        Attribute::Array(attrs) => attrs.iter().all(check_attr_padding),
        Attribute::Dictionary(attrs) => attrs.iter().all(|(_, a)| check_attr_padding(a)),
        _ => true,
    }
}

/// Lower signed FP8 storage and conversions for sm_80..sm_89. Native FP8
/// targets are untouched. Unsupported FP8 computation is diagnosed before any
/// mutation; callers must still validate the lowered module against its target.
pub fn lower_fp8_storage(
    module: &mut Module,
    caps: &TargetCapabilities,
) -> Result<(), CapabilityError> {
    if !caps.sm().is_some_and(|sm| (80..90).contains(&sm)) {
        return Ok(());
    }
    if !module.values.iter().any(|v| contains_fp8(&v.ty))
        && !module
            .globals
            .iter()
            .any(|g| contains_fp8(&g.value.element_type))
    {
        return Ok(());
    }

    // Do not turn an unsupported floating operation into an integer operation
    // by erasing its type. Only conversions and bit-preserving carriers qualify.
    for op in &module.operations {
        let types: Vec<_> = op
            .result_types
            .iter()
            .chain(op.operands.iter().map(|v| module.value_type(*v)))
            .collect();
        let touches_fp8 = types.iter().any(|t| contains_fp8(t))
            || op
                .attributes
                .iter()
                .any(|(_, attr)| attr_contains_fp8(attr));
        if !touches_fp8 {
            continue;
        }
        if types.iter().any(|t| !check_padding(t))
            || op.attributes.iter().any(|(_, a)| !check_attr_padding(a))
        {
            return Err(error(
                op,
                caps,
                "only zero padding is supported for emulated FP8 views",
            ));
        }
        use Opcode::*;
        match op.opcode {
            FToF | FToI | IToF => {
                if op.operands.len() != 1 || op.result_types.len() != 1 {
                    return Err(error(
                        op,
                        caps,
                        "conversion must have one operand and one result",
                    ));
                }
                let (Type::Tile(src), Type::Tile(dst)) =
                    (module.value_type(op.operands[0]), &op.result_types[0])
                else {
                    return Err(error(op, caps, "conversion requires scalar-element tiles"));
                };
                if src.shape != dst.shape
                    || scalar_type(&op.result_types[0]).is_none()
                    || scalar_type(module.value_type(op.operands[0])).is_none()
                {
                    return Err(error(
                        op,
                        caps,
                        "conversion tiles must have matching shapes and scalar elements",
                    ));
                }
                let from = scalar_type(module.value_type(op.operands[0])).unwrap();
                let to = scalar_type(&op.result_types[0]).unwrap();
                let valid = match op.opcode {
                    FToF => from.is_float() && to.is_float(),
                    FToI => fp8(from) && to.is_integer(),
                    IToF => from.is_integer() && fp8(to),
                    _ => unreachable!(),
                };
                if !valid {
                    return Err(error(
                        op,
                        caps,
                        "invalid conversion source or destination kind",
                    ));
                }
                crate::verify_target::verify_operation(module, op, caps)?;
                if op.opcode == IToF
                    && !op
                        .attributes
                        .iter()
                        .any(|(k, a)| k == "rounding_mode" && matches!(a, Attribute::Integer(0, _)))
                {
                    return Err(error(
                        op,
                        caps,
                        "conversion to FP8 requires nearest_even rounding",
                    ));
                }
                let supported = |s| {
                    fp8(s)
                        || matches!(
                            s,
                            ScalarType::F16
                                | ScalarType::BF16
                                | ScalarType::F32
                                | ScalarType::F64
                                | ScalarType::TF32
                        )
                        || s.is_integer()
                };
                if !supported(from) || !supported(to) {
                    return Err(error(
                        op,
                        caps,
                        "conversion of this alternative float format is not supported",
                    ));
                }
            }
            Assume => {
                if op
                    .operands
                    .iter()
                    .any(|v| scalar_type(module.value_type(*v)).is_some_and(fp8))
                {
                    return Err(error(
                        op,
                        caps,
                        "numerical assumptions on FP8 values are not supported",
                    ));
                }
                crate::verify_target::verify_operation(module, op, caps)?;
            }
            Entry
            | Return
            | Constant
            | Broadcast
            | Reshape
            | Permute
            | Select
            | Cat
            | Extract
            | Insert
            | Offset
            | IntToPtr
            | PtrToPtr
            | PtrToInt
            | GetGlobal
            | MakeTensorView
            | MakePartitionView
            | MakeGatherScatterView
            | MakeStridedView
            | GetTensorShape
            | GetIndexSpaceShape
            | LoadPtrTko
            | StorePtrTko
            | LoadViewTko
            | StoreViewTko
            | Bitcast
            | If
            | Yield
            | For
            | Continue
            | Break
            | Loop
            | Alloca => {
                crate::verify_target::verify_operation(module, op, caps)?;
            }
            _ => {
                return Err(error(
                    op,
                    caps,
                    "emulation supports storage and conversions; convert FP8 operands to f16/f32 before computation",
                ));
            }
        }
    }

    // Keep the original result values and their producers. Insert the software
    // conversion before each old op, then use reshape as a typed identity.
    for block_index in 0..module.blocks.len() {
        let old_ops = module.blocks[block_index].ops.clone();
        let mut new_ops = Vec::new();
        for id in old_ops {
            let op = module.op(id).clone();
            if matches!(op.opcode, Opcode::FToF | Opcode::FToI | Opcode::IToF)
                && (scalar_type(&op.result_types[0]).is_some_and(fp8)
                    || scalar_type(module.value_type(op.operands[0])).is_some_and(fp8))
            {
                let mut emit = Emitter::new(module, &op);
                let result = emit.convert(&op);
                new_ops.extend(emit.ops);
                *module.op_mut(id) = Operation {
                    opcode: Opcode::Reshape,
                    operands: vec![result],
                    result_types: op.result_types,
                    attributes: vec![],
                    regions: vec![],
                    location: op.location,
                };
            }
            new_ops.push(id);
        }
        module.blocks[block_index].ops = new_ops;
    }
    for value in &mut module.values {
        map_type(&mut value.ty);
    }
    for block in &mut module.blocks {
        for (_, ty) in &mut block.args {
            map_type(ty);
        }
    }
    for op in &mut module.operations {
        for ty in &mut op.result_types {
            map_type(ty);
        }
        for (_, attr) in &mut op.attributes {
            map_attr(attr);
        }
    }
    for global in &mut module.globals {
        map_type(&mut global.value.element_type);
    }
    Ok(())
}

struct Emitter<'m> {
    module: &'m mut Module,
    location: Location,
    shape: Vec<i64>,
    ops: Vec<OpId>,
}

impl<'m> Emitter<'m> {
    fn new(module: &'m mut Module, op: &Operation) -> Self {
        let Type::Tile(t) = &op.result_types[0] else {
            unreachable!()
        };
        Self {
            module,
            location: op.location.clone(),
            shape: t.shape.clone(),
            ops: vec![],
        }
    }

    fn op(
        &mut self,
        opcode: Opcode,
        operands: &[Value],
        scalar: ScalarType,
        attrs: Vec<(String, Attribute)>,
    ) -> Value {
        let (id, results) = OpBuilder::new(opcode, self.location.clone())
            .operands(operands.iter().copied())
            .result(Type::Tile(TileType {
                shape: self.shape.clone(),
                element_type: TileElementType::Scalar(scalar),
            }))
            .attrs(attrs)
            .build(self.module);
        self.ops.push(id);
        results[0]
    }

    fn constant(&mut self, scalar: ScalarType, bits: i64) -> Value {
        let data = bits.to_le_bytes()[..scalar.byte_width()].to_vec();
        self.op(
            Opcode::Constant,
            &[],
            scalar,
            vec![(
                "value".into(),
                Attribute::DenseElements(DenseElements {
                    element_type: Type::Scalar(scalar),
                    shape: self.shape.clone(),
                    data,
                }),
            )],
        )
    }

    fn binary(&mut self, opcode: Opcode, lhs: Value, rhs: Value) -> Value {
        let scalar = scalar_type(self.module.value_type(lhs)).unwrap();
        let attrs = match opcode {
            Opcode::ShRI | Opcode::MinI | Opcode::MaxI => {
                vec![("signedness".into(), Attribute::i32(0))]
            }
            Opcode::AddI | Opcode::SubI | Opcode::ShLI => {
                vec![("overflow".into(), Attribute::i32(0))]
            }
            Opcode::MulF => vec![("rounding_mode".into(), Attribute::i32(0))],
            _ => vec![],
        };
        self.op(opcode, &[lhs, rhs], scalar, attrs)
    }

    fn binary_const(&mut self, opcode: Opcode, lhs: Value, rhs: i64) -> Value {
        let scalar = scalar_type(self.module.value_type(lhs)).unwrap();
        let rhs = self.constant(scalar, rhs);
        self.binary(opcode, lhs, rhs)
    }

    fn cmp_const(&mut self, value: Value, rhs: i64, predicate: ComparisonPredicate) -> Value {
        let scalar = scalar_type(self.module.value_type(value)).unwrap();
        let rhs = self.constant(scalar, rhs);
        self.op(
            Opcode::CmpI,
            &[value, rhs],
            ScalarType::I1,
            vec![
                ("signedness".into(), Attribute::i32(0)),
                (
                    "comparison_predicate".into(),
                    Attribute::i32(predicate as i64),
                ),
            ],
        )
    }

    fn select(&mut self, condition: Value, yes: Value, no: Value) -> Value {
        self.op(
            Opcode::Select,
            &[condition, yes, no],
            scalar_type(self.module.value_type(yes)).unwrap(),
            vec![],
        )
    }

    fn nearest(&mut self, opcode: Opcode, input: Value, scalar: ScalarType) -> Value {
        let mut attrs = vec![("rounding_mode".into(), Attribute::i32(0))];
        if opcode == Opcode::IToF {
            attrs.push(("signedness".into(), Attribute::i32(0)));
        }
        self.op(opcode, &[input], scalar, attrs)
    }

    fn convert(&mut self, op: &Operation) -> Value {
        let src = scalar_type(self.module.value_type(op.operands[0])).unwrap();
        let dst = scalar_type(&op.result_types[0]).unwrap();
        let mut input = op.operands[0];
        if fp8(src) {
            input = self.decode(input, src);
        }
        if fp8(dst) {
            if op.opcode == Opcode::IToF {
                input = self.op(
                    Opcode::IToF,
                    &[input],
                    ScalarType::F32,
                    op.attributes.clone(),
                );
            } else if !matches!(src, ScalarType::F32 | ScalarType::F64) && !fp8(src) {
                input = self.nearest(Opcode::FToF, input, ScalarType::F32);
            }
            self.encode(input, dst)
        } else if op.opcode == Opcode::FToI {
            self.op(Opcode::FToI, &[input], dst, op.attributes.clone())
        } else if dst == ScalarType::F32 {
            input
        } else {
            self.op(Opcode::FToF, &[input], dst, op.attributes.clone())
        }
    }

    fn format(s: ScalarType) -> (i64, i64, i64) {
        if s == ScalarType::F8E4M3FN {
            (3, 7, 0x7e)
        } else {
            (2, 15, 0x7b)
        }
    }

    fn decode(&mut self, input: Value, format: ScalarType) -> Value {
        use ComparisonPredicate::*;
        use Opcode::*;
        use ScalarType::*;
        let (fraction, bias, _) = Self::format(format);
        let raw = self.op(
            ExtI,
            &[input],
            I32,
            vec![("signedness".into(), Attribute::i32(0))],
        );
        let abs = self.binary_const(AndI, raw, 0x7f);
        let sign = self.binary_const(AndI, raw, 0x80);
        let sign = self.binary_const(ShLI, sign, 24);
        let exponent = self.binary_const(ShRI, abs, fraction);
        let mantissa = self.binary_const(AndI, abs, (1 << fraction) - 1);
        let normal = self.binary_const(ShLI, abs, 23 - fraction);
        let normal = self.binary_const(AddI, normal, (127 - bias) << 23);
        let subnormal = self.nearest(IToF, mantissa, F32);
        let scale = self.constant(
            F32,
            i64::from(2f32.powi((1 - bias - fraction) as i32).to_bits()),
        );
        let subnormal = self.binary(MulF, subnormal, scale);
        let subnormal = self.op(Bitcast, &[subnormal], I32, vec![]);
        let is_subnormal = self.cmp_const(exponent, 0, Equal);
        let finite = self.select(is_subnormal, subnormal, normal);
        let nan = self.constant(I32, 0x7fc00000);
        let (is_special, special) = if format == F8E4M3FN {
            (self.cmp_const(abs, 0x7f, Equal), nan)
        } else {
            let inf = self.constant(I32, 0x7f800000);
            let is_nan = self.cmp_const(mantissa, 0, NotEqual);
            let special = self.select(is_nan, nan, inf);
            (self.cmp_const(exponent, 31, Equal), special)
        };
        let magnitude = self.select(is_special, special, finite);
        let bits = self.binary(OrI, magnitude, sign);
        self.op(Bitcast, &[bits], F32, vec![])
    }

    fn encode(&mut self, input: Value, format: ScalarType) -> Value {
        use ComparisonPredicate::*;
        use Opcode::*;
        use ScalarType::*;
        let double = scalar_type(self.module.value_type(input)) == Some(F64);
        let (integer, precision, source_bias, exponent_bits) = if double {
            (I64, 52, 1023, 11)
        } else {
            (I32, 23, 127, 8)
        };
        let (fraction, bias, max_code) = Self::format(format);
        let raw = self.op(Bitcast, &[input], integer, vec![]);
        let sign = self.binary_const(ShRI, raw, precision + exponent_bits);
        let sign = self.binary_const(ShLI, sign, 7);
        let abs = self.binary_const(AndI, raw, if double { i64::MAX } else { 0x7fffffff });
        let inf_bits = ((1 << exponent_bits) - 1) << precision;
        let is_nan = self.cmp_const(abs, inf_bits, GreaterThan);
        let max = if format == F8E4M3FN {
            448.0f64
        } else {
            57344.0f64
        };
        let max_bits = if double {
            max.to_bits() as i64
        } else {
            i64::from((max as f32).to_bits())
        };
        let bounded = self.binary_const(MinI, abs, max_bits);
        let shift = precision - fraction;
        let normal_lsb = self.binary_const(ShRI, bounded, shift);
        let normal_lsb = self.binary_const(AndI, normal_lsb, 1);
        let normal = self.binary_const(AddI, bounded, (1 << (shift - 1)) - 1);
        let normal = self.binary(AddI, normal, normal_lsb);
        let normal = self.binary_const(ShRI, normal, shift);
        let normal = self.binary_const(SubI, normal, (source_bias - bias) << fraction);

        // Subnormal rounding uses the implicit significand and a bounded shift.
        // Every speculative shift stays within the integer width, even for NaN,
        // huge values, source subnormals, and exact signed zero.
        let exponent = self.binary_const(ShRI, bounded, precision);
        let half_min_exp = source_bias - bias - fraction;
        let normal_min_exp = source_bias + 1 - bias;
        let bounded_exp = self.binary_const(MaxI, exponent, half_min_exp);
        let bounded_exp = self.binary_const(MinI, bounded_exp, normal_min_exp);
        let shift_base = self.constant(integer, source_bias + precision + 1 - bias - fraction);
        let sub_shift = self.binary(SubI, shift_base, bounded_exp);
        let significand = self.binary_const(AndI, bounded, (1 << precision) - 1);
        let significand = self.binary_const(OrI, significand, 1 << precision);
        let one = self.constant(integer, 1);
        let rounding_shift = self.binary_const(SubI, sub_shift, 1);
        let round_bias = self.binary(ShLI, one, rounding_shift);
        let round_bias = self.binary_const(SubI, round_bias, 1);
        let sub_lsb = self.binary(ShRI, significand, sub_shift);
        let sub_lsb = self.binary_const(AndI, sub_lsb, 1);
        let subnormal = self.binary(AddI, significand, round_bias);
        let subnormal = self.binary(AddI, subnormal, sub_lsb);
        let subnormal = self.binary(ShRI, subnormal, sub_shift);
        let zero = self.constant(integer, 0);
        let underflow = self.cmp_const(exponent, half_min_exp, LessThan);
        let subnormal = self.select(underflow, zero, subnormal);
        let is_subnormal = self.cmp_const(bounded, normal_min_exp << precision, LessThan);
        let finite = self.select(is_subnormal, subnormal, normal);
        let finite = self.binary_const(MinI, finite, max_code);
        let nan = self.constant(integer, 0x7f);
        let magnitude = self.select(is_nan, nan, finite);
        let bits = self.binary(OrI, magnitude, sign);
        self.op(
            TruncI,
            &[bits],
            I8,
            vec![("overflow".into(), Attribute::i32(0))],
        )
    }
}
