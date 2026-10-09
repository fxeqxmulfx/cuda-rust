/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops::{MirAssertOp, MirCondBranchOp, MirFuncOp, MirGotoOp, MirReturnOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{OperandSegmentSizesAttr, TypeAttr},
        op_interfaces::{BranchOpInterface, OperandSegmentInterface},
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
};

#[test]
fn test_mir_control_flow_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    // 1. MirGotoOp
    let target_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let src_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let arg_val = src_block.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![arg_val],
        vec![target_block],
        0,
    );
    let goto_op = MirGotoOp::new(op);
    assert!(goto_op.verify(&ctx).is_ok(), "Valid Goto");

    let op_bad = Operation::new(
        &mut ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![target_block],
        0,
    );
    assert!(
        MirGotoOp::new(op_bad).verify(&ctx).is_err(),
        "Goto missing operand"
    );

    // 2. MirCondBranchOp
    let true_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let false_block = BasicBlock::new(&mut ctx, None, vec![]);
    let cond_block = BasicBlock::new(&mut ctx, None, vec![i1_ty.into(), i32_ty.into()]);
    let cond_val = cond_block.deref(&ctx).get_argument(0);
    let val = cond_block.deref(&ctx).get_argument(1);

    let (cond_flat, cond_sizes) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cond_val], vec![val], vec![]]);
    let op_cond = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        cond_flat,
        vec![true_block, false_block],
        0,
    );
    MirCondBranchOp::new(op_cond).set_operand_segment_sizes(&ctx, cond_sizes);
    let cond_br = MirCondBranchOp::new(op_cond);
    assert!(cond_br.verify(&ctx).is_ok(), "Valid CondBranch");

    let (cond_bad_flat, cond_bad_sizes) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cond_val], vec![], vec![]]);
    let op_cond_bad = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        cond_bad_flat,
        vec![true_block, false_block],
        0,
    );
    MirCondBranchOp::new(op_cond_bad).set_operand_segment_sizes(&ctx, cond_bad_sizes);
    assert!(
        MirCondBranchOp::new(op_cond_bad).verify(&ctx).is_err(),
        "CondBranch missing operand"
    );

    // 3. MirReturnOp
    let func_ty = FunctionType::get(&ctx, vec![], vec![i32_ty.into()]);
    let func_ty_attr = TypeAttr::new(func_ty.into());

    let func_op_ptr = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func = MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    let entry_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let ret_val = entry_block.deref(&ctx).get_argument(0);

    let region = mir_func.get_operation().deref(&ctx).get_region(0);
    entry_block.insert_at_front(region, &ctx);

    let ret_op = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![ret_val],
        vec![],
        0,
    );
    ret_op.insert_at_back(entry_block, &ctx);

    let mir_ret = MirReturnOp::new(ret_op);
    assert!(mir_ret.verify(&ctx).is_ok(), "Valid Return");

    let f32_ty = FP32Type::get(&ctx);
    let f32_block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into()]);
    let f32_val = f32_block.deref(&ctx).get_argument(0);

    let ret_op_bad = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![f32_val],
        vec![],
        0,
    );
    ret_op_bad.insert_at_back(entry_block, &ctx);

    let mir_ret_bad = MirReturnOp::new(ret_op_bad);
    assert!(mir_ret_bad.verify(&ctx).is_err(), "Return type mismatch");

    // 4. MirAssertOp
    let assert_op = MirAssertOp::new(&mut ctx, cond_val);
    assert!(assert_op.verify(&ctx).is_ok(), "Valid Assert");
    assert!(
        MirAssertOp::new(&mut ctx, val).verify(&ctx).is_err(),
        "Assert cond type mismatch"
    );

    let assert_succ = BasicBlock::new(&mut ctx, None, vec![]);
    for (operands, successors) in [
        (vec![], vec![]),
        (vec![cond_val, cond_val], vec![]),
        (vec![cond_val], vec![assert_succ]),
    ] {
        let malformed = Operation::new(
            &mut ctx,
            MirAssertOp::get_concrete_op_info(),
            vec![],
            operands,
            successors,
            0,
        );
        assert!(
            Operation::get_op::<MirAssertOp>(malformed, &ctx)
                .unwrap()
                .verify(&ctx)
                .is_err(),
            "Assertions must have exactly one condition and no successors"
        );
    }
}

#[test]
fn branch_operand_layout_rejects_malformed_ir_and_preserves_edge_ranges() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let source = BasicBlock::new(&mut ctx, None, vec![i1_ty.into(), i32_ty.into()]);
    let condition = source.deref(&ctx).get_argument(0);
    let value = source.deref(&ctx).get_argument(1);
    let target = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);

    let goto = Operation::new(
        &mut ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![value],
        vec![target],
        0,
    );
    let goto = MirGotoOp::new(goto);
    assert!(goto.verify_successor_operand_layout(&ctx).is_ok());
    assert_eq!(goto.successor_operand_range(&ctx, 0), 0..1);
    assert_eq!(goto.successor_operands(&ctx, 0), vec![value]);

    for successors in [vec![], vec![target, target]] {
        let malformed = Operation::new(
            &mut ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![value],
            successors,
            0,
        );
        assert!(
            MirGotoOp::new(malformed)
                .verify_successor_operand_layout(&ctx)
                .is_err()
        );
    }

    let branch = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        vec![condition, value, value],
        vec![target, target],
        0,
    );
    let branch = MirCondBranchOp::new(branch);
    // Missing and malformed segment attributes must return errors before
    // the interface tries to index them.
    assert!(branch.verify_successor_operand_layout(&ctx).is_err());
    for sizes in [vec![1, 2], vec![1, 1, 0], vec![0, 2, 1], vec![2, 0, 1]] {
        branch.set_operand_segment_sizes(&ctx, OperandSegmentSizesAttr(sizes));
        assert!(branch.verify_successor_operand_layout(&ctx).is_err());
    }
    branch.set_operand_segment_sizes(&ctx, OperandSegmentSizesAttr(vec![1, 1, 1]));
    assert!(branch.verify_successor_operand_layout(&ctx).is_ok());
    assert_eq!(branch.successor_operand_range(&ctx, 0), 1..2);
    assert_eq!(branch.successor_operand_range(&ctx, 1), 2..3);
    assert_eq!(branch.successor_operands(&ctx, 0), vec![value]);
    assert_eq!(branch.successor_operands(&ctx, 1), vec![value]);

    assert_eq!(branch.add_successor_operand(&mut ctx, 0, value), 1);
    assert_eq!(branch.successor_operand_range(&ctx, 1), 3..4);
    assert_eq!(branch.remove_successor_operand(&mut ctx, 0, 1), value);
    assert_eq!(branch.successor_operand_range(&ctx, 1), 2..3);
    assert_eq!(branch.get_operation().deref(&ctx).get_operand(0), condition);
}
