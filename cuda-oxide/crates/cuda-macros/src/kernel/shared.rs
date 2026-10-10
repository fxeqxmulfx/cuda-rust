// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry-only, immutable shared tiles with no user-controlled barrier placement.
use crate::common::{attr_path_ends_with, internal_ident};
use crate::cuda_module::contract::LaunchContractArgs;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Expr, FnArg, GenericParam, Ident, ItemFn, LitInt, Pat, Token, Type,
    parse::{Parse, ParseStream},
};

pub(crate) struct BlockShared {
    pub(crate) index: usize,
    pub(crate) name: Ident,
    capacity: usize,
    block_threads: Option<u32>,
    element: Type,
    offset: Option<Expr>,
    offset_parameters: Vec<(String, usize)>,
}

struct TileArgs {
    capacity: LitInt,
    offset: Option<Expr>,
}
impl Parse for TileArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let capacity = input.parse()?;
        let offset = if input.is_empty() {
            None
        } else {
            input.parse::<Token![,]>()?;
            let key: Ident = input.parse()?;
            if key != "offset" {
                return Err(syn::Error::new_spanned(key, "expected offset = expression"));
            }
            input.parse::<Token![=]>()?;
            Some(input.parse()?)
        };
        Ok(Self { capacity, offset })
    }
}

fn validate_offset(expr: &Expr, names: &[String]) -> syn::Result<()> {
    match expr {
        Expr::Lit(expr) if matches!(expr.lit, syn::Lit::Int(_)) => Ok(()),
        Expr::Path(path)
            if path.qself.is_none()
                && path
                    .path
                    .get_ident()
                    .is_some_and(|name| names.contains(&name.to_string())) =>
        {
            Ok(())
        }
        Expr::Paren(expr) => validate_offset(&expr.expr, names),
        Expr::Group(expr) => validate_offset(&expr.expr, names),
        Expr::Binary(expr)
            if matches!(
                expr.op,
                syn::BinOp::Add(_)
                    | syn::BinOp::Sub(_)
                    | syn::BinOp::Mul(_)
                    | syn::BinOp::Div(_)
                    | syn::BinOp::Rem(_)
            ) =>
        {
            validate_offset(&expr.left, names)?;
            validate_offset(&expr.right, names)
        }
        _ => Err(syn::Error::new_spanned(
            expr,
            "shared offsets allow integer arithmetic on `block`, usize kernel scalars, and usize const parameters",
        )),
    }
}

/// Consume one annotation while the original launch signature is available.
pub(crate) fn take_block_shared(input: &mut ItemFn) -> syn::Result<Option<BlockShared>> {
    let mut tile = None;
    for (index, argument) in input.sig.inputs.iter_mut().enumerate() {
        let FnArg::Typed(argument) = argument else {
            continue;
        };
        for attr in &argument.attrs {
            if !attr_path_ends_with(attr, "block_shared") {
                continue;
            }
            if tile.is_some() {
                return Err(syn::Error::new_spanned(
                    attr,
                    "only one #[block_shared(N)] parameter is supported",
                ));
            }
            let TileArgs {
                capacity: literal,
                offset,
            } = attr.parse_args()?;
            let capacity: usize = literal.base10_parse()?;
            if !(1..=4096).contains(&capacity) {
                return Err(syn::Error::new_spanned(
                    literal,
                    "shared tile capacity must be a literal in 1..=4096",
                ));
            }
            let Type::Reference(reference) = argument.ty.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &argument.ty,
                    "#[block_shared] requires an immutable slice with an elided lifetime",
                ));
            };
            let Type::Slice(slice) = reference.elem.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &argument.ty,
                    "#[block_shared] requires an immutable slice with an elided lifetime",
                ));
            };
            if reference.mutability.is_some() || reference.lifetime.is_some() {
                return Err(syn::Error::new_spanned(
                    &argument.ty,
                    "#[block_shared] requires an immutable slice with an elided lifetime",
                ));
            }
            let Type::Path(element) = slice.elem.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &slice.elem,
                    "shared tiles support primitive numeric elements",
                ));
            };
            if element.qself.is_some()
                || !["f32", "f64", "i32", "u32", "i64", "u64"]
                    .iter()
                    .any(|name| element.path.is_ident(name))
            {
                return Err(syn::Error::new_spanned(
                    &slice.elem,
                    "shared tiles support f32, f64, i32, u32, i64, or u64",
                ));
            }
            let Pat::Ident(pattern) = argument.pat.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &argument.pat,
                    "shared tile parameters require an identifier",
                ));
            };
            tile = Some(BlockShared {
                index,
                name: pattern.ident.clone(),
                capacity,
                block_threads: None,
                element: (*slice.elem).clone(),
                offset,
                offset_parameters: Vec::new(),
            });
        }
        argument
            .attrs
            .retain(|attr| !attr_path_ends_with(attr, "block_shared"));
    }
    if let Some(tile) = &mut tile {
        if let Some(offset) = &tile.offset {
            for (index, arg) in input.sig.inputs.iter().enumerate() {
                if let FnArg::Typed(arg) = arg
                    && let Pat::Ident(pattern) = arg.pat.as_ref()
                    && let Type::Path(ty) = arg.ty.as_ref()
                    && ty.qself.is_none()
                    && ty.path.is_ident("usize")
                {
                    tile.offset_parameters
                        .push((pattern.ident.to_string(), index));
                }
            }
            let mut names: Vec<_> = tile
                .offset_parameters
                .iter()
                .map(|(name, _)| name.clone())
                .collect();
            names.extend(input.sig.generics.params.iter().filter_map(|param| {
                if let GenericParam::Const(param) = param
                    && let Type::Path(ty) = &param.ty
                    && ty.qself.is_none()
                    && ty.path.is_ident("usize")
                {
                    Some(param.ident.to_string())
                } else {
                    None
                }
            }));
            if names.iter().any(|name| name == "block") {
                return Err(syn::Error::new_spanned(
                    &input.sig,
                    "`block` is reserved for the block index in shared offset expressions",
                ));
            }
            names.push("block".into());
            validate_offset(offset, &names)?;
            // Parameter annotations count as uses even in the ordinary helper.
            for (_, index) in &tile.offset_parameters {
                if let FnArg::Typed(arg) = &input.sig.inputs[*index]
                    && let Pat::Ident(pattern) = arg.pat.as_ref()
                {
                    let name = &pattern.ident;
                    input
                        .block
                        .stmts
                        .insert(0, syn::parse_quote! { let _ = #name; });
                }
            }
        }

        let contract = input
            .attrs
            .iter()
            .find(|attr| attr_path_ends_with(attr, "launch_contract"))
            .ok_or_else(|| {
                syn::Error::new_spanned(
                    &input.sig,
                    "#[block_shared] requires #[launch_contract(domain = 1, ...)] below #[kernel]",
                )
            })?;
        let contract_args = contract.parse_args::<LaunchContractArgs>()?;
        if contract_args.domain != 1 {
            return Err(syn::Error::new_spanned(
                contract,
                "#[block_shared] requires a one-dimensional launch contract",
            ));
        }
        tile.block_threads = contract_args
            .exact_block
            .and_then(|(x, y, z)| (y == 1 && z == 1).then_some(x));
        if input.sig.constness.is_some() {
            return Err(syn::Error::new_spanned(
                &input.sig,
                "shared tile kernel entries cannot be const",
            ));
        }
    }
    Ok(tile)
}

impl BlockShared {
    /// The generated entry is unsafe to call as an ordinary device function:
    /// only the CUDA launch boundary guarantees uniform block participation.
    pub(crate) fn prologue(
        &self,
        source: &Ident,
        forwarded: Option<&[Ident]>,
    ) -> (TokenStream, TokenStream) {
        let capacity = self.capacity;
        let element = &self.element;
        let storage = internal_ident("__cuda_oxide_shared_storage");
        let pointer = internal_ident("__cuda_oxide_shared_pointer");
        let guard = internal_ident("__cuda_oxide_shared_guard");
        let base = internal_ident("__cuda_oxide_shared_base");
        let length = internal_ident("__cuda_oxide_shared_length");
        let lane = internal_ident("__cuda_oxide_shared_lane");
        let step = internal_ident("__cuda_oxide_shared_step");
        let block_threads = match self.block_threads {
            Some(threads) => quote! { #threads as ::core::primitive::usize },
            None => quote! { ::cuda_device::thread::blockDim_x() as ::core::primitive::usize },
        };
        let mut offset = self
            .offset
            .clone()
            .unwrap_or_else(|| syn::parse_quote! { block * #capacity });
        struct Remap<'a> {
            parameters: &'a [(String, usize)],
            forwarded: Option<&'a [Ident]>,
        }
        impl syn::visit_mut::VisitMut for Remap<'_> {
            fn visit_expr_mut(&mut self, expr: &mut Expr) {
                if let Expr::Path(path) = expr
                    && let Some(name) = path.path.get_ident()
                {
                    if name == "block" {
                        *expr = syn::parse_quote! { (::cuda_device::thread::blockIdx_x() as ::core::primitive::usize) };
                        return;
                    }
                    if let Some(forwarded) = self.forwarded
                        && let Some((_, index)) = self
                            .parameters
                            .iter()
                            .find(|(source, _)| source == &name.to_string())
                    {
                        let argument = &forwarded[*index];
                        *expr = syn::parse_quote! { (#argument as ::core::primitive::usize) };
                        return;
                    }
                    // Built-in casts prevent shadowed primitive names or
                    // associated types from introducing overloaded operators
                    // with user code before the collective barrier.
                    *expr = syn::parse_quote! { (#name as ::core::primitive::usize) };
                    return;
                }
                syn::visit_mut::visit_expr_mut(self, expr);
            }
        }
        syn::visit_mut::VisitMut::visit_expr_mut(
            &mut Remap {
                parameters: &self.offset_parameters,
                forwarded,
            },
            &mut offset,
        );
        let prologue = quote! {
            static mut #storage: ::cuda_device::SharedArray<#element, #capacity> = ::cuda_device::SharedArray::UNINIT;
            // SAFETY: a 1D launch assigns unique threadIdx_x values. No Rust
            // reference to the complete writable allocation is constructed.
            let #pointer = unsafe { ::cuda_device::SharedArray::as_raw_mut_ptr(&raw mut #storage) };
            let #base: ::core::primitive::usize = #offset;
            let #length = #source.len().saturating_sub(#base).min(#capacity);
            // Exact launch contracts make the stride a compile-time constant.
            let #step = #block_threads;
            let mut #lane = ::cuda_device::thread::threadIdx_x() as ::core::primitive::usize;
            while #lane < #length {
                // SAFETY: each lane owns a disjoint sequence in 0..length.
                // Unused capacity is never exposed through the borrowed slice.
                unsafe { #pointer.add(#lane).write(#source[#base + #lane]); }
                #lane += #step;
            }
            // Every launched thread reaches this barrier before any user code.
            ::cuda_device::thread::sync_threads();
            // SAFETY: the readable extent was initialized before the barrier,
            // remains immutable, and is borrowed only through this local guard.
            let #guard = unsafe { ::cuda_device::shared::__ReadOnlyTile::from_initialized(#pointer, #length) };
        };
        (prologue, quote! { #guard.as_slice() })
    }
}
