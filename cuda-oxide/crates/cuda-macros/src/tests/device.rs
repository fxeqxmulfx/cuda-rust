// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::device::device_function_tokens;
use quote::quote;
use syn::parse_quote;

#[test]
fn generic_device_respects_explicit_and_conditional_inline_policies() {
    for policy in [
        quote! { #[inline(always)] },
        quote! { #[inline(never)] },
        quote! { #[inline] },
        quote! { #[cfg_attr(any(), inline(always))] },
        quote! { #[cfg_attr(all(), cfg_attr(all(), inline(always)))] },
    ] {
        let input = syn::parse2(quote! {
            #policy
            fn helper<const N: usize>(value: f32) -> f32 { value + N as f32 }
        })
        .unwrap();
        let tokens = device_function_tokens(input).to_string().replace(' ', "");
        let expected = policy.to_string().replace(' ', "");
        assert!(tokens.contains(&expected), "{tokens}");
        assert_eq!(
            tokens.matches("inline(never)").count(),
            usize::from(expected.contains("inline(never)")),
            "{tokens}"
        );
    }
}

#[test]
fn generic_device_defaults_remain_visible_to_the_collector() {
    let tokens = device_function_tokens(parse_quote! {
        fn helper<const N: usize>(value: f32) -> f32 { value + N as f32 }
    })
    .to_string()
    .replace(' ', "");
    assert_eq!(tokens.matches("inline(never)").count(), 1, "{tokens}");
    assert_eq!(tokens.matches("inline(always)").count(), 1, "{tokens}");
}
