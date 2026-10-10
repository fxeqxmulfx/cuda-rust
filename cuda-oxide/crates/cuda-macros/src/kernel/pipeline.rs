// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Collective schedules live only in generated CUDA entries. Ordinary helpers
//! use global views, so calling one under divergent control flow remains safe.
use crate::common::{attr_path_ends_with, internal_ident};
use crate::cuda_module::contract::LaunchContractArgs;
use crate::kernel::shared::validate_offset;
use quote::quote;
use syn::{
    Expr, FnArg, GenericParam, Ident, ItemFn, Pat, Stmt, Token, Type,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

struct Args {
    rows: Expr,
    iterations: Expr,
}
impl Parse for Args {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let rows: Ident = input.parse()?;
        if rows != "rows" {
            return Err(syn::Error::new_spanned(rows, "expected rows"));
        }
        input.parse::<Token![=]>()?;
        let rows: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let key: Ident = input.parse()?;
        if key != "iterations" {
            return Err(syn::Error::new_spanned(key, "expected iterations"));
        }
        input.parse::<Token![=]>()?;
        Ok(Self {
            rows,
            iterations: input.parse()?,
        })
    }
}
struct SourceArgs {
    columns: Expr,
    stride: Expr,
    offset: Expr,
    swizzle: bool,
}
impl Parse for SourceArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut columns = None;
        let mut stride = None;
        let mut offset = None;
        let mut swizzle = None;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            if key == "layout" {
                if swizzle.is_some() {
                    return Err(syn::Error::new_spanned(key, "duplicate tile layout"));
                }
                let layout: Ident = input.parse()?;
                if layout != "xor32" {
                    return Err(syn::Error::new_spanned(
                        layout,
                        "supported tile layout is xor32",
                    ));
                }
                swizzle = Some(true);
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            let slot = match key.to_string().as_str() {
                "columns" => &mut columns,
                "stride" => &mut stride,
                "offset" => &mut offset,
                _ => {
                    return Err(syn::Error::new_spanned(
                        key,
                        "expected columns, stride, or offset",
                    ));
                }
            };
            if slot.is_some() {
                return Err(syn::Error::new_spanned(key, "duplicate tile field"));
            }
            *slot = Some(input.parse()?);
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        let missing = || {
            syn::Error::new(
                input.span(),
                "tile_source needs columns, stride, and offset",
            )
        };
        Ok(Self {
            columns: columns.ok_or_else(missing)?,
            stride: stride.ok_or_else(missing)?,
            offset: offset.ok_or_else(missing)?,
            swizzle: swizzle.unwrap_or(false),
        })
    }
}
struct Source {
    name: Ident,
    columns: Expr,
    stride: Expr,
    offset: Expr,
    swizzle: bool,
}
struct CopyGroup {
    key: String,
    columns: Expr,
    stride: Expr,
    offset: Ident,
    types: Vec<proc_macro2::TokenStream>,
    stores: Vec<proc_macro2::TokenStream>,
}
pub(crate) struct BlockPipeline {
    rows: Expr,
    iterations: Expr,
    sources: Vec<Source>,
    element: Type,
    threads: u32,
    callbacks: syn::Block,
}

fn const_names(input: &ItemFn) -> Vec<String> {
    input.sig.generics.params.iter().filter_map(|param| match param {
        GenericParam::Const(param) if matches!(&param.ty, Type::Path(p) if p.qself.is_none() && p.path.is_ident("usize")) => Some(param.ident.to_string()),
        _ => None,
    }).collect()
}
fn uniform_names(input: &ItemFn) -> Vec<String> {
    let mut names = const_names(input);
    names.extend(input.sig.inputs.iter().filter_map(|arg| match arg {
        FnArg::Typed(arg) if matches!(arg.ty.as_ref(), Type::Path(p) if p.qself.is_none() && p.path.is_ident("usize")) => match arg.pat.as_ref() { Pat::Ident(p) => Some(p.ident.to_string()), _ => None },
        _ => None,
    }));
    names
}

impl BlockPipeline {
    fn parse(input: &ItemFn) -> syn::Result<Option<Self>> {
        let Some(attr) = input
            .attrs
            .iter()
            .find(|a| attr_path_ends_with(a, "block_pipeline"))
        else {
            if input.sig.inputs.iter().any(|arg| matches!(arg, FnArg::Typed(a) if a.attrs.iter().any(|a| attr_path_ends_with(a, "tile_source")))) {
                return Err(syn::Error::new_spanned(&input.sig, "tile_source requires block_pipeline"));
            }
            return Ok(None);
        };
        if input
            .attrs
            .iter()
            .filter(|a| attr_path_ends_with(a, "block_pipeline"))
            .count()
            != 1
        {
            return Err(syn::Error::new_spanned(
                attr,
                "only one block_pipeline is supported",
            ));
        }
        let Args { rows, iterations } = attr.parse_args()?;
        validate_offset(&rows, &const_names(input))?;
        if let Expr::Lit(lit) = &rows
            && let syn::Lit::Int(lit) = &lit.lit
            && !(1..=64).contains(&lit.base10_parse::<usize>()?)
        {
            return Err(syn::Error::new_spanned(
                attr,
                "pipeline rows must be in 1..=64",
            ));
        }
        if input.sig.constness.is_some()
            || input.sig.asyncness.is_some()
            || !matches!(input.sig.output, syn::ReturnType::Default)
        {
            return Err(syn::Error::new_spanned(
                &input.sig,
                "pipeline kernels must be non-const, synchronous, and return unit",
            ));
        }
        let contract_attr = input
            .attrs
            .iter()
            .find(|a| attr_path_ends_with(a, "launch_contract"))
            .ok_or_else(|| {
                syn::Error::new_spanned(attr, "block_pipeline requires launch_contract")
            })?;
        let contract: LaunchContractArgs = contract_attr.parse_args()?;
        let Some((threads, 1, 1)) = contract.exact_block else {
            return Err(syn::Error::new_spanned(
                contract_attr,
                "pipeline requires an exact one-dimensional block",
            ));
        };
        if threads == 0 || !threads.is_multiple_of(32) {
            return Err(syn::Error::new_spanned(
                contract_attr,
                "pipeline block size must contain complete warps",
            ));
        }
        if contract.domain != 1 {
            return Err(syn::Error::new_spanned(
                contract_attr,
                "pipeline requires domain = 1",
            ));
        }
        let mut names = uniform_names(input);
        if names.iter().any(|s| s == "block") {
            return Err(syn::Error::new_spanned(
                &input.sig,
                "block is reserved in pipeline expressions",
            ));
        }
        names.push("block".into());
        validate_offset(&iterations, &names)?;
        let constants = const_names(input);
        let mut sources = Vec::new();
        let mut element = None;
        for arg in &input.sig.inputs {
            let FnArg::Typed(arg) = arg else { continue };
            let Pat::Ident(pat) = arg.pat.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &arg.pat,
                    "pipeline parameters need identifiers",
                ));
            };
            if arg
                .attrs
                .iter()
                .any(|a| attr_path_ends_with(a, "block_shared"))
            {
                return Err(syn::Error::new_spanned(
                    arg,
                    "pipeline and block_shared cannot be combined",
                ));
            }
            let attrs: Vec<_> = arg
                .attrs
                .iter()
                .filter(|a| attr_path_ends_with(a, "tile_source"))
                .collect();
            if attrs.len() > 1 {
                return Err(syn::Error::new_spanned(
                    arg,
                    "only one tile_source per parameter",
                ));
            }
            let Some(attr) = attrs.first() else { continue };
            let Type::Reference(reference) = arg.ty.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &arg.ty,
                    "tile_source requires an immutable slice",
                ));
            };
            let Type::Slice(slice) = reference.elem.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &arg.ty,
                    "tile_source requires an immutable slice",
                ));
            };
            if reference.mutability.is_some() || reference.lifetime.is_some() {
                return Err(syn::Error::new_spanned(
                    reference,
                    "tile_source needs an immutable slice with an elided lifetime",
                ));
            }
            let Type::Path(ty) = slice.elem.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &slice.elem,
                    "tile_source requires primitive numeric elements",
                ));
            };
            if ty.qself.is_some()
                || !["f32", "f64", "i32", "u32", "i64", "u64"]
                    .iter()
                    .any(|s| ty.path.is_ident(s))
            {
                return Err(syn::Error::new_spanned(
                    ty,
                    "tile_source requires f32, f64, i32, u32, i64, or u64",
                ));
            }
            if let Some(previous) = &element
                && quote!(#previous).to_string() != quote!(#ty).to_string()
            {
                return Err(syn::Error::new_spanned(
                    ty,
                    "pipeline sources must have the same element type",
                ));
            }
            element = Some((*slice.elem).clone());
            let SourceArgs {
                columns,
                stride,
                offset,
                swizzle,
            } = attr.parse_args()?;
            validate_offset(&columns, &constants)?;
            validate_offset(&stride, &constants)?;
            validate_offset(&offset, &names)?;
            sources.push(Source {
                name: pat.ident.clone(),
                columns,
                stride,
                offset,
                swizzle,
            });
        }
        if sources.is_empty() {
            return Err(syn::Error::new_spanned(
                attr,
                "pipeline requires at least one tile_source",
            ));
        }
        if input.block.stmts.len() != 3 {
            return Err(syn::Error::new_spanned(
                &input.block,
                "pipeline body must contain exactly init, step, and finish closure bindings",
            ));
        }
        for (stmt, expected) in input.block.stmts.iter().zip(["init", "step", "finish"]) {
            let Stmt::Local(local) = stmt else {
                return Err(syn::Error::new_spanned(
                    stmt,
                    "pipeline body only accepts closure bindings",
                ));
            };
            let Pat::Ident(pat) = &local.pat else {
                return Err(syn::Error::new_spanned(
                    local,
                    "pipeline callback needs an identifier",
                ));
            };
            if pat.ident != expected
                || pat.by_ref.is_some()
                || !matches!(local.init.as_ref(), Some(init) if init.diverge.is_none() && matches!(init.expr.as_ref(), Expr::Closure(c) if c.asyncness.is_none()))
            {
                return Err(syn::Error::new_spanned(
                    local,
                    "expected init, step, and finish synchronous closures",
                ));
            }
        }
        Ok(Some(Self {
            rows,
            iterations,
            sources,
            element: element.unwrap(),
            threads,
            callbacks: (*input.block).clone(),
        }))
    }

    pub(crate) fn bytes(&self) -> Expr {
        let rows = remap(&self.rows);
        let element = &self.element;
        let mut total = quote! { 0usize };
        for source in &self.sources {
            let columns = remap(&source.columns);
            if source.swizzle {
                total = quote! { { assert!((#columns as ::core::primitive::usize).is_multiple_of(32), "xor32 columns must be divisible by 32"); #total } };
            }
            total = quote! { match (#total).checked_add(#columns as ::core::primitive::usize) { Some(n) => n, None => panic!("pipeline columns overflow") } };
        }
        syn::parse_quote! { { let rows: ::core::primitive::usize = #rows; assert!(rows.wrapping_sub(1) < 64, "pipeline rows must be in 1..=64"); match (#total).checked_mul(2 * rows * ::core::mem::size_of::<#element>()) { Some(n) => n, None => panic!("pipeline shared memory overflow") } } }
    }

    pub(crate) fn body(&self, staged: bool) -> syn::Block {
        let callbacks = &self.callbacks.stmts;
        let rows = remap(&self.rows);
        let count = remap(&self.iterations);
        let iterations = internal_ident("__cuda_pipeline_iterations");
        let iteration = internal_ident("__cuda_pipeline_iteration");
        let state = internal_ident("__cuda_pipeline_state");
        let mut names = Vec::new();
        let mut views = Vec::new();
        let mut copy_groups: Vec<CopyGroup> = Vec::new();
        let mut offsets = Vec::new();
        let mut total = quote! { 0usize };
        let pointer = internal_ident("__cuda_pipeline_storage");
        let copy_iteration = internal_ident("__cuda_pipeline_copy_iteration");
        let slot = internal_ident("__cuda_pipeline_slot");
        let guards = self
            .sources
            .iter()
            .enumerate()
            .map(|(i, _)| internal_ident(&format!("__cuda_pipeline_guard_{i}")))
            .collect::<Vec<_>>();
        let element = &self.element;
        let copy = if matches!(element, Type::Path(p) if p.path.is_ident("f32") || p.path.is_ident("u32") || p.path.is_ident("i32"))
        {
            quote! { ::cuda_device::async_copy::cp_async_ca_4 }
        } else {
            quote! { ::cuda_device::async_copy::cp_async_ca_8 }
        };
        for (i, source) in self.sources.iter().enumerate() {
            let name = &source.name;
            let columns = remap(&source.columns);
            let stride = remap(&source.stride);
            let base = remap(&source.offset);
            let offset = internal_ident(&format!("__cuda_pipeline_offset_{i}"));
            offsets.push(quote! { let #offset: ::core::primitive::usize = if #iterations > 0 { #base } else { 0 }; });
            let guard = &guards[i];
            let view = internal_ident(&format!("__cuda_pipeline_view_{i}"));
            names.push(view.clone());
            let layout = if source.swizzle {
                quote! { .with_xor32() }
            } else {
                quote! {}
            };
            let destination = if source.swizzle {
                quote! { (__cuda_pipeline_row * #columns + (__cuda_pipeline_col & !31usize) + ((__cuda_pipeline_col ^ __cuda_pipeline_row) & 31usize)) }
            } else {
                quote! { __cuda_pipeline_lane }
            };
            if staged {
                views.push(quote! {
                    // SAFETY: this slot was fully initialized, its copies completed,
                    // and all block threads synchronized before any borrowed view.
                    let #guard = unsafe { ::cuda_device::shared::__ReadOnlyTile::from_initialized(#pointer.wrapping_add(#slot * __cuda_pipeline_extent + #total), #rows * #columns) };
                    let #view = ::cuda_device::shared::BlockTile::new(#guard.as_slice(), 0, #rows, #columns, #columns)#layout;
                });
                let key = quote! { #columns; #stride; #base }.to_string();
                let group = if let Some(group) = copy_groups.iter_mut().find(|g| g.key == key) {
                    group
                } else {
                    copy_groups.push(CopyGroup {
                        key,
                        columns: columns.clone(),
                        stride: stride.clone(),
                        offset: offset.clone(),
                        types: Vec::new(),
                        stores: Vec::new(),
                    });
                    copy_groups.last_mut().unwrap()
                };
                // Check each source's primitive type even when no copy executes.
                group.types.push(quote! { let #name: &[#element] = #name; });
                group.stores.push(quote! {
                    // SAFETY: lanes partition this source's writable region. Other
                    // sources occupy disjoint regions. Equal layouts share address
                    // arithmetic, but each source retains its own bounds check.
                    // The scheduler completes all copies before publication/reuse.
                    unsafe {
                        let dst = #pointer.add(__cuda_pipeline_copy_slot * __cuda_pipeline_extent + #total + #destination);
                        if let Some(src) = __cuda_pipeline_src.filter(|&n| n < #name.len()) {
                            #copy(dst.cast::<u32>(), #name.as_ptr().add(src).cast::<u32>());
                        } else { dst.write(0 as #element); }
                    }
                });
            } else {
                views.push(quote! {
                    let __cuda_pipeline_base = (#iteration).checked_mul(#rows).and_then(|n| n.checked_mul(#stride)).and_then(|n| n.checked_add(#offset)).unwrap_or(::core::primitive::usize::MAX);
                    let #view = ::cuda_device::shared::BlockTile::new(#name, __cuda_pipeline_base, #rows, #columns, #stride);
                });
            }
            total = quote! { (#total + #rows * #columns) };
        }
        let copy_threads = self.threads as usize;
        let copies: Vec<_> = copy_groups.into_iter().map(|CopyGroup { columns, stride, offset, types, stores, .. }| quote! {
            #(#types)*
            let mut __cuda_pipeline_lane = ::cuda_device::thread::threadIdx_x() as ::core::primitive::usize;
            while __cuda_pipeline_lane < #rows * #columns {
                let __cuda_pipeline_row = __cuda_pipeline_lane / #columns;
                let __cuda_pipeline_col = __cuda_pipeline_lane % #columns;
                let __cuda_pipeline_src = (#copy_iteration).checked_mul(#rows)
                    .and_then(|n| n.checked_add(__cuda_pipeline_row))
                    .and_then(|n| n.checked_mul(#stride))
                    .and_then(|n| n.checked_add(#offset))
                    .and_then(|n| n.checked_add(__cuda_pipeline_col));
                #(#stores)*
                __cuda_pipeline_lane += #copy_threads;
            }
        }).collect();
        if !staged {
            return syn::parse_quote! {{
                #(#callbacks)*
                let #iterations: ::core::primitive::usize = #count;
                let mut #state = init();
                #(#offsets)*
                for #iteration in 0..#iterations { #(#views)* step(&mut #state, #iteration, #(#names),*); }
                finish(#state);
            }};
        }
        let bytes = self.bytes();
        let threads = self.threads;
        syn::parse_quote! {{
            #(#callbacks)*
            let #iterations: ::core::primitive::usize = #count;
            #(#offsets)*
            let __cuda_pipeline_extent = (#bytes) / (2 * ::core::mem::size_of::<#element>());
            // SAFETY: the typed launch contract reserves exactly two aligned slots.
            let #pointer = ::cuda_device::DynamicSharedArray::<#element, 16>::get();
            if #iterations > 0 {
                let #copy_iteration: ::core::primitive::usize = 0;
                let __cuda_pipeline_copy_slot: ::core::primitive::usize = 0;
                #(#copies)*
                // SAFETY: the scheduler owns all copy lifetimes and completion.
                unsafe { ::cuda_device::async_copy::cp_async_commit_group(); }
            }
            let mut #state = init();
            if #iterations > 0 {
                // SAFETY: every issuing thread completes its pending first tile.
                unsafe { ::cuda_device::async_copy::cp_async_wait_all(); }
                // SAFETY: every launched thread follows the generated schedule.
                // Barrier 15 is isolated from safe user barriers on slot 0.
                unsafe { ::cuda_device::barrier::barrier_cta_sync_aligned(15, #threads); }
            }
            for #iteration in 0..#iterations {
                let #slot = #iteration % 2;
                if #iteration + 1 < #iterations {
                    let #copy_iteration: ::core::primitive::usize = #iteration + 1;
                    let __cuda_pipeline_copy_slot: ::core::primitive::usize = 1 - #slot;
                    #(#copies)*
                    // SAFETY: readers use the other slot; completion is below.
                    unsafe { ::cuda_device::async_copy::cp_async_commit_group(); }
                }
                { #(#views)* step(&mut #state, #iteration, #(#names),*); }
                if #iteration + 1 < #iterations {
                    // SAFETY: the issuing thread waits before any reader gets a view.
                    unsafe { ::cuda_device::async_copy::cp_async_wait_all(); }
                    // SAFETY: this publication barrier also waits for all previous
                    // readers before the next iteration can overwrite their slot.
                    // No tile borrow can escape the callback. Every launched thread
                    // follows this schedule, independently of user barriers on slot 0.
                    unsafe { ::cuda_device::barrier::barrier_cta_sync_aligned(15, #threads); }
                }
            }
            finish(#state);
        }}
    }
}

fn remap(expr: &Expr) -> Expr {
    struct Cast;
    impl syn::visit_mut::VisitMut for Cast {
        fn visit_expr_mut(&mut self, expr: &mut Expr) {
            if let Expr::Path(path) = expr
                && let Some(name) = path.path.get_ident()
            {
                *expr = if name == "block" {
                    syn::parse_quote! { (::cuda_device::thread::blockIdx_x() as ::core::primitive::usize) }
                } else {
                    syn::parse_quote! { (#name as ::core::primitive::usize) }
                };
            } else {
                syn::visit_mut::visit_expr_mut(self, expr);
            }
        }
    }
    let mut result = expr.clone();
    syn::visit_mut::VisitMut::visit_expr_mut(&mut Cast, &mut result);
    syn::parse_quote! { (#result) }
}

/// Derive both the host's exact resource proof and the device alignment marker.
pub(crate) fn prepare_pipeline(input: &mut ItemFn) -> syn::Result<Option<BlockPipeline>> {
    let Some(pipeline) = BlockPipeline::parse(input)? else {
        return Ok(None);
    };
    let attr = input
        .attrs
        .iter_mut()
        .find(|a| attr_path_ends_with(a, "launch_contract"))
        .unwrap();
    let contract: LaunchContractArgs = attr.parse_args()?;
    let capability = contract.min_compute_capability.max((8, 0));
    let (major, minor) = capability;
    let fields = attr.parse_args_with(Punctuated::<syn::Meta, Token![,]>::parse_terminated)?;
    let fields = fields.iter().filter(|m| {
        ![
            "dynamic_shared",
            "dynamic_shared_range",
            "dynamic_shared_alignment",
            "min_compute_capability",
        ]
        .iter()
        .any(|s| m.path().is_ident(s))
    });
    // The device attribute only needs an alignment declaration. cuda_module
    // replaces this envelope with the exact, const-dependent allocation proof.
    *attr = syn::parse_quote! { #[launch_contract(#(#fields,)* dynamic_shared_range = (0, 131072), dynamic_shared_alignment = 16, min_compute_capability = (#major, #minor))] };
    Ok(Some(pipeline))
}

pub(crate) fn take_pipeline(input: &mut ItemFn) -> syn::Result<Option<BlockPipeline>> {
    let pipeline = prepare_pipeline(input)?;
    if let Some(pipeline) = &pipeline {
        input
            .attrs
            .retain(|a| !attr_path_ends_with(a, "block_pipeline"));
        for arg in &mut input.sig.inputs {
            if let FnArg::Typed(arg) = arg {
                arg.attrs.retain(|a| !attr_path_ends_with(a, "tile_source"));
            }
        }
        *input.block = pipeline.body(false);
    }
    Ok(pipeline)
}
