// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use crate::kernel::{codegen::generic_kernel_no_instantiation_tokens, pipeline::prepare_pipeline};
use syn::parse_quote;

fn valid() -> syn::ItemFn {
    parse_quote! {
        #[launch_contract(domain = 1, block = (128, 1, 1))]
        #[block_pipeline(rows = 4, iterations = block % count + 1)]
        fn tiled<const D: usize>(count: usize, #[tile_source(columns = D, stride = D, offset = 0)] input: &[f32]) {
            let init = || 0.0;
            let step = |state: &mut f32, _iteration: usize, tile: cuda_device::shared::BlockTile<'_, f32>| { if count == 0 { return; } *state += tile.get(0, 0).copied().unwrap_or(0.0); };
            let finish = |_state: f32| {};
        }
    }
}
#[test]
fn repeated_tiles_are_entry_only_and_complete_before_borrowing() {
    let expanded = generic_kernel_no_instantiation_tokens(valid(), None);
    let file: syn::File = syn::parse2(expanded).unwrap();
    let functions: Vec<_> = file
        .items
        .iter()
        .filter_map(|i| match i {
            syn::Item::Fn(f) => Some(f),
            _ => None,
        })
        .collect();
    let helper = functions.iter().find(|f| f.sig.ident == "tiled").unwrap();
    assert!(matches!(helper.sig.safety, syn::Safety::Default));
    let helper = quote::quote!(#helper).to_string();
    assert!(!helper.contains("sync_threads"));
    assert!(!helper.contains("cp_async"));
    let entry = functions
        .iter()
        .find(|f| f.sig.ident.to_string().contains("cuda_oxide_kernel_"))
        .unwrap();
    assert!(matches!(entry.sig.safety, syn::Safety::Unsafe(_)));
    let body = quote::quote!(#entry).to_string();
    assert!(body.find("cp_async_wait_all").unwrap() < body.find("from_initialized").unwrap());
    assert_eq!(body.matches("barrier_cta_sync_aligned").count(), 3);
}
#[test]
fn callbacks_cannot_bypass_the_collective_schedule() {
    for body in [
        quote::quote!({
            return;
        }),
        quote::quote!({
            let init = make_init();
            let step = |_: &mut ()| {};
            let finish = |_| {};
        }),
        quote::quote!({
            let init = || ();
            let step = |_: &mut ()| {};
            let finish = |_| {};
            thread::sync_threads();
        }),
    ] {
        let mut input = valid();
        input.block = Box::new(syn::parse2(body).unwrap());
        assert!(prepare_pipeline(&mut input).is_err());
    }
    for iterations in [
        quote::quote!(thread::threadIdx_x()),
        quote::quote!({
            return;
        }),
        quote::quote!(input.len()),
    ] {
        let mut input = valid();
        input.attrs[1] = syn::parse_quote!(#[block_pipeline(rows = 4, iterations = #iterations)]);
        assert!(prepare_pipeline(&mut input).is_err());
    }
    for source in [
        quote::quote!(#[tile_source(columns = count, stride = D, offset = 0)]),
        quote::quote!(#[tile_source(columns = D, stride = D, offset = input[0])]),
    ] {
        let mut input = valid();
        let syn::FnArg::Typed(input_arg) = &mut input.sig.inputs[1] else {
            unreachable!()
        };
        input_arg.attrs = vec![syn::parse_quote!(#source)];
        assert!(prepare_pipeline(&mut input).is_err());
    }
}
