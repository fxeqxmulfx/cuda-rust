// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kernel::{codegen::generic_kernel_no_instantiation_tokens, shared::take_block_shared};
use syn::parse_quote;

#[test]
fn shared_staging_lives_only_in_the_unsafe_entry() {
    let input = parse_quote! {
        #[launch_contract(domain = 1, block = (128, 1, 1))]
        pub fn tile<const N: usize>(#[block_shared(128)] input: &[f32]) { if N == 0 { return; } let _ = input[0]; }
    };
    let expanded = generic_kernel_no_instantiation_tokens(input, None);
    let file: syn::File = syn::parse2(expanded).unwrap();
    let functions: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(f) => Some(f),
            _ => None,
        })
        .collect();
    let implementation = functions.iter().find(|f| f.sig.ident == "tile").unwrap();
    assert!(matches!(implementation.sig.safety, syn::Safety::Default));
    assert!(
        !quote::quote!(#implementation)
            .to_string()
            .contains("sync_threads")
    );
    let entry = functions
        .iter()
        .find(|f| f.sig.ident.to_string().contains("cuda_oxide_kernel_"))
        .unwrap();
    assert!(matches!(entry.sig.safety, syn::Safety::Unsafe(_)));
    let body = quote::quote!(#entry).to_string();
    assert_eq!(body.matches("sync_threads").count(), 1);
    assert!(body.find("sync_threads").unwrap() < body.find("as_slice").unwrap());
    assert!(!body.contains("return"));
}

#[test]
fn shared_tiles_reject_mutability_lifetimes_and_invalid_capacities() {
    let cases: [syn::ItemFn; 8] = [
        parse_quote! { #[launch_contract(domain = 1)] fn bad(#[block_shared(0)] input: &[f32]) {} },
        parse_quote! { #[launch_contract(domain = 1)] fn bad(#[block_shared(4097)] input: &[f32]) {} },
        parse_quote! { #[launch_contract(domain = 1)] fn bad(#[block_shared(32)] input: &mut [f32]) {} },
        parse_quote! { #[launch_contract(domain = 1)] fn bad(#[block_shared(32)] input: &'static [f32]) {} },
        parse_quote! { #[launch_contract(domain = 1)] fn bad(#[block_shared(32)] input: &[bool]) {} },
        parse_quote! { #[launch_contract(domain = 2)] fn bad(#[block_shared(32)] input: &[f32]) {} },
        parse_quote! { fn bad(#[block_shared(32)] input: &[f32]) {} },
        parse_quote! { #[launch_contract(domain = 1)] fn bad(#[block_shared(32)] a: &[f32], #[block_shared(32)] b: &[f32]) {} },
    ];
    for mut input in cases {
        assert!(take_block_shared(&mut input).is_err());
    }
}

#[test]
fn shared_offsets_remap_uniform_scalars_and_reject_user_control_flow() {
    let mut input = parse_quote! {
        #[launch_contract(domain = 1, block = (128, 1, 1))]
        fn tile<const W: usize>(stride: usize, #[block_shared(128, offset = (block / stride) * W)] data: &[f32]) {}
    };
    let tile = take_block_shared(&mut input).unwrap().unwrap();
    let args: Vec<syn::Ident> = vec![parse_quote!(forwarded_stride), parse_quote!(forwarded_data)];
    let (prefix, _) = tile.prologue(&args[1], Some(&args));
    let prefix = prefix.to_string();
    assert!(prefix.contains("forwarded_stride"));
    assert!(prefix.contains("blockIdx_x"));
    assert!(prefix.contains("W"));
    for offset in [
        quote::quote!(thread::threadIdx_x()),
        quote::quote!({
            return;
        }),
        quote::quote!(unknown),
        quote::quote!(data.len()),
    ] {
        let mut input: syn::ItemFn = syn::parse2(quote::quote! {
            #[launch_contract(domain = 1, block = (128, 1, 1))]
            fn bad(#[block_shared(128, offset = #offset)] data: &[f32]) {}
        })
        .unwrap();
        assert!(take_block_shared(&mut input).is_err());
    }
}

#[test]
fn shared_staging_uses_checked_block_sizes_and_only_initializes_live_elements() {
    for (contract, constant_stride) in [
        (quote::quote!(domain = 1, block = (128, 1, 1)), true),
        (quote::quote!(domain = 1), false),
    ] {
        let mut input = syn::parse2(quote::quote! {
            #[launch_contract(#contract)]
            fn tile(#[block_shared(512)] input: &[f32]) {}
        })
        .unwrap();
        let tile = take_block_shared(&mut input).unwrap().unwrap();
        let source = parse_quote!(input);
        let (prefix, _) = tile.prologue(&source, None);
        let block: syn::Block = syn::parse2(quote::quote! { { #prefix } }).unwrap();
        let loop_ = block
            .stmts
            .iter()
            .find_map(|stmt| match stmt {
                syn::Stmt::Expr(syn::Expr::While(expr), _) => Some(expr),
                _ => None,
            })
            .unwrap();
        let condition = &loop_.cond;
        assert!(
            quote::quote!(#condition)
                .to_string()
                .contains("shared_length")
        );
        assert!(!quote::quote!(#loop_).to_string().contains("else"));
        let prefix = prefix.to_string();
        assert_eq!(!prefix.contains("blockDim_x"), constant_stride);
        assert_eq!(prefix.matches("sync_threads").count(), 1);
    }
}
