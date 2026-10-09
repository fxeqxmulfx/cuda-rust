/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile_ir::builder::{append_op, build_single_block_region};
use cutile_ir::bytecode::{write_bytecode_version, BytecodeVersion, Opcode};
use cutile_ir::capabilities::TargetCapabilities;
use cutile_ir::emulation::lower_fp8_storage;
use cutile_ir::ir::*;
use cutile_ir::OpBuilder;

fn tile(scalar: ScalarType) -> Type {
    Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(scalar),
    })
}

fn conversion(src: ScalarType, dst: ScalarType, opcode: Opcode) -> Module {
    let mut module = Module::new("fp8_conversion");
    let (region, block, args) = build_single_block_region(&mut module, &[tile(src)]);
    let rounding = if opcode == Opcode::FToI { 6 } else { 0 };
    let (op, _) = OpBuilder::new(opcode, Location::Unknown)
        .operand(args[0])
        .result(tile(dst))
        .attr("rounding_mode", Attribute::i32(rounding))
        .attr("signedness", Attribute::i32(1))
        .build(&mut module);
    append_op(&mut module, block, op);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("convert".into()))
        .attr(
            "function_type",
            Attribute::Type(Type::Func(FuncType {
                inputs: vec![tile(src)],
                results: vec![],
            })),
        )
        .region(region)
        .build(&mut module);
    module.functions.push(entry);
    module
}

#[test]
fn software_conversions_validate_and_serialize_on_pre_hopper_targets() {
    use ScalarType::*;
    for version in BytecodeVersion::SUPPORTED {
        for sm in [80, 86, 87, 88, 89] {
            let caps = TargetCapabilities::new(version, format!("sm_{sm}"));
            for fp8 in [F8E4M3FN, F8E5M2] {
                for other in [F16, BF16, F32, F64, TF32, F8E4M3FN, F8E5M2] {
                    if fp8 == other {
                        continue;
                    }
                    for (from, to) in [(fp8, other), (other, fp8)] {
                        let mut module = conversion(from, to, Opcode::FToF);
                        lower_fp8_storage(&mut module, &caps).unwrap();
                        caps.validate_module(&module).unwrap();
                        module.verify_dominance().unwrap();
                        module.verify_bytecode_indices().unwrap();
                        let bytes = write_bytecode_version(&module, version).unwrap();
                        cutile_ir::decode_bytecode(&bytes).unwrap();
                        let text = module.to_mlir_text();
                        assert!(!text.contains("f8e4m3fn") && !text.contains("f8e5m2"));
                    }
                }
                for integer in [I8, I16, I32, I64] {
                    for (from, to, op) in
                        [(fp8, integer, Opcode::FToI), (integer, fp8, Opcode::IToF)]
                    {
                        let mut module = conversion(from, to, op);
                        lower_fp8_storage(&mut module, &caps).unwrap();
                        caps.validate_module(&module).unwrap();
                        module.verify_dominance().unwrap();
                        module.verify_bytecode_indices().unwrap();
                        write_bytecode_version(&module, version).unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn native_targets_keep_the_original_fp8_ir_and_bytecode() {
    for sm in ["sm_90", "sm_90a", "sm_100", "sm_120"] {
        let caps = TargetCapabilities::new(BytecodeVersion::V13_4, sm);
        for fp8 in [ScalarType::F8E4M3FN, ScalarType::F8E5M2] {
            let mut module = conversion(fp8, ScalarType::F32, Opcode::FToF);
            let original = write_bytecode_version(&module, caps.bytecode_version).unwrap();
            lower_fp8_storage(&mut module, &caps).unwrap();
            caps.validate_module(&module).unwrap();
            assert_eq!(
                write_bytecode_version(&module, caps.bytecode_version).unwrap(),
                original
            );
        }
    }
}

#[test]
fn unsupported_fp8_computation_fails_before_rewriting_types() {
    let caps = TargetCapabilities::new(BytecodeVersion::V13_4, "sm_86");
    let mut module = conversion(ScalarType::F8E4M3FN, ScalarType::F32, Opcode::MmaF);
    let original = module.to_mlir_text();
    let err = lower_fp8_storage(&mut module, &caps).unwrap_err();
    assert!(err.message.contains("cuda_tile.mmaf"));
    assert!(err.message.contains("convert FP8 operands to f16/f32"));
    assert_eq!(module.to_mlir_text(), original);
}

#[test]
fn unsupported_rounding_is_not_hidden_by_type_erasure() {
    let caps = TargetCapabilities::new(BytecodeVersion::V13_4, "sm_86");
    let mut module = conversion(ScalarType::F32, ScalarType::F8E4M3FN, Opcode::FToF);
    let entry = module.functions[0];
    let block = module.region(module.op(entry).regions[0]).blocks[0];
    let convert = module.block(block).ops[0];
    module.op_mut(convert).attributes[0].1 = Attribute::i32(1);
    let original = module.to_mlir_text();
    let err = lower_fp8_storage(&mut module, &caps).unwrap_err();
    assert!(err.message.contains("rounding_mode"));
    assert_eq!(module.to_mlir_text(), original);
}

#[test]
fn fp8_global_initializers_keep_their_bytes() {
    let caps = TargetCapabilities::new(BytecodeVersion::V13_4, "sm_86");
    let mut module = conversion(ScalarType::F8E5M2, ScalarType::F32, Opcode::FToF);
    let data = vec![0x00, 0x80, 0x01, 0x7c, 0x7e, 0xff];
    module.globals.push(Global {
        sym_name: "bytes".into(),
        alignment: 1,
        constant: true,
        symbol_visibility: SymbolVisibility::Private,
        value: DenseElements {
            element_type: Type::Scalar(ScalarType::F8E5M2),
            shape: vec![6],
            data: data.clone(),
        },
    });
    lower_fp8_storage(&mut module, &caps).unwrap();
    caps.validate_module(&module).unwrap();
    assert_eq!(
        module.globals[0].value.element_type,
        Type::Scalar(ScalarType::I8)
    );
    assert_eq!(module.globals[0].value.data, data);
    write_bytecode_version(&module, caps.bytecode_version).unwrap();
}

#[test]
fn nonzero_padding_in_function_signatures_is_rejected() {
    let caps = TargetCapabilities::new(BytecodeVersion::V13_4, "sm_86");
    let ty = Type::PartitionView(PartitionViewType {
        tile_shape: vec![128],
        dim_map: vec![0],
        padding_value: Some(PaddingValue::NegZero),
        tensor_view: TensorViewType {
            element_type: ScalarType::F8E4M3FN,
            shape: vec![DYNAMIC],
            strides: vec![1],
        },
    });
    let mut module = Module::new("padding");
    let (region, block, _) = build_single_block_region(&mut module, std::slice::from_ref(&ty));
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("padding".into()))
        .attr(
            "function_type",
            Attribute::Type(Type::Func(FuncType {
                inputs: vec![ty],
                results: vec![],
            })),
        )
        .region(region)
        .build(&mut module);
    module.functions.push(entry);
    let original = module.to_mlir_text();
    let err = lower_fp8_storage(&mut module, &caps).unwrap_err();
    assert!(err.message.contains("only zero padding"));
    assert_eq!(module.to_mlir_text(), original);
}
