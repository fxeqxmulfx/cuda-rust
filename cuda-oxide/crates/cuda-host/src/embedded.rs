/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side loading for embedded device artifact bundles.

use crate::ltoir;
pub use cuda_core::embedded::{
    ArtifactCompileOptions, ArtifactPayloadKind, EmbeddedModule, OwnedArtifactBundle,
    artifact_bundles_from_binary_path, artifact_bundles_from_current_exe,
    embedded_modules_from_current_exe,
};
use cuda_core::{CudaContext, CudaModule, DriverError};
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use thiserror::Error;

#[cfg(any(target_os = "linux", target_os = "android"))]
mod mapped_image;

/// Errors while discovering, building, or loading an embedded CUDA module.
#[derive(Debug, Error)]
pub enum EmbeddedModuleError {
    /// The artifact anchor could not be associated with its original mapped file.
    #[error("cannot read the binary containing the CUDA artifact anchor: {0}")]
    MappedImage(#[source] std::io::Error),

    /// Reading the embedded artifact section failed.
    #[error(transparent)]
    Core(#[from] cuda_core::EmbeddedModuleError),

    /// The named bundle was not present in the binary that was read.
    #[error("embedded CUDA module '{name}' was not found")]
    ModuleNotFound { name: String },

    /// The compiler did not retain an anchor for this module's mapped image.
    #[error("no compiler-owned CUDA artifact anchor is available for this module")]
    NoArtifactAnchor,

    /// No embedded bundles with loadable payloads were found.
    #[error("no embedded CUDA modules were found")]
    NoModules,

    /// A bundle existed, but it contained no supported payload.
    #[error("embedded CUDA module '{name}' has no supported payload")]
    UnsupportedPayload { name: String },

    /// PTX source could not be parsed or edited safely while merging bundles.
    #[error("embedded CUDA module '{name}' contains invalid PTX: {reason}")]
    InvalidPtx { name: String, reason: String },

    /// No PTX target is valid for the code of all merged bundles.
    #[error(
        "embedded CUDA module '{name}' targets `{target}`, which is not compatible with `{merged}`"
    )]
    IncompatibleTarget {
        name: String,
        target: String,
        merged: String,
    },

    /// More than one bundle defines the same kernel entry.
    #[error("kernel entry '{entry}' is defined by both embedded modules '{first}' and '{second}'")]
    ConflictingEntry {
        entry: String,
        first: String,
        second: String,
    },

    /// NVVM IR or LTOIR payload compilation failed.
    #[error("failed to build embedded CUDA module: {0}")]
    Ltoir(#[from] ltoir::LtoirError),

    /// The CUDA driver rejected the selected module image.
    #[error("failed to load embedded CUDA module: {0}")]
    Driver(#[from] DriverError),
}

/// Load a named embedded artifact bundle from the current executable.
///
/// Cubin and PTX payloads are loaded directly with the CUDA driver. NVVM IR and
/// LTOIR payloads are linked to an in-memory cubin for their original target.
/// A payload built for a standard pre-Blackwell target, such as `sm_86`, may
/// instead be converted to PTX and JIT-compiled by the driver on Blackwell.
pub fn load_embedded_module(
    ctx: &Arc<CudaContext>,
    name: &str,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_from_current_exe()?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

/// Load a named artifact bundle from the binary containing `anchor`.
///
/// Generated non-generic module loaders borrow their artifact anchor, which
/// the linker places in the same image as the bundle. This works for both an
/// executable and a shared library, including a library opened by a relative
/// path before the working directory changes.
///
/// On Linux and Android, this requires readable `/proc/self/maps` and
/// `/proc/self/map_files`. The original mapped file must still be accessible.
/// A missing, replaced, deleted or unreadable image is an error; it never
/// causes a search in another binary. Other operating systems are unsupported.
/// Payload selection and compilation are identical to [`load_embedded_module`].
pub fn load_embedded_module_from_anchor(
    ctx: &Arc<CudaContext>,
    name: &str,
    anchor: &u8,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_containing(anchor)?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

fn artifact_bundles_containing(
    anchor: &u8,
) -> Result<Vec<OwnedArtifactBundle>, EmbeddedModuleError> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let bytes = mapped_image::read(anchor).map_err(EmbeddedModuleError::MappedImage)?;
        oxide_artifacts::read_artifact_bundles_from_object_bytes(&bytes)
            .map_err(cuda_core::EmbeddedModuleError::Artifacts)
            .map_err(Into::into)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = anchor;
        Err(EmbeddedModuleError::MappedImage(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "artifact anchor discovery requires Linux or Android procfs",
        )))
    }
}

/// Merge all PTX bundles from the current executable into a single CUDA module.
///
/// When a generic kernel is monomorphized in a consuming crate, its PTX ends
/// up in that crate's bundle rather than the defining crate's bundle. This
/// function gathers every PTX bundle in the binary, merges them with
/// [`merge_ptx_bundles`], and loads the result as one CUDA module.
/// All kernel symbols are therefore available regardless of which crate bundle
/// they were compiled into.
///
/// Bundles with non-PTX payloads (NVVM IR, LTOIR, cubin) are skipped; use
/// `load_embedded_module` for those.
pub fn load_all_ptx_bundles_merged(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundles = artifact_bundles_from_current_exe()?;
    load_merged_ptx_bundles(ctx, &bundles)
}

/// Merge compiler-embedded PTX from the original image containing `anchor`.
///
/// Typed generic module loaders use their compiler-owned anchor, retaining
/// library specializations and checking the mapped file's identity before
/// reading any payload. No caller-selected path or replacement image is used.
/// All required specializations must be linked into the same image. This
/// requires readable Linux or Android procfs; unsupported or unavailable
/// provenance fails loading instead of falling back to another binary.
#[doc(hidden)]
pub fn load_all_ptx_bundles_merged_from_anchor(
    ctx: &Arc<CudaContext>,
    anchor: &u8,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundles = artifact_bundles_containing(anchor)?;
    load_merged_ptx_bundles(ctx, &bundles)
}

fn load_merged_ptx_bundles(
    ctx: &Arc<CudaContext>,
    bundles: &[OwnedArtifactBundle],
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let merged = merge_ptx_bundles(bundles)?;

    let module = ctx.load_module_from_image(merged.as_bytes())?;
    // Retain the merged module's `.entry` names (a few dozen bytes per
    // kernel). If a later `_TID_` generic-kernel lookup misses while a
    // same-base entry exists under a different hash, the launch paths can
    // then report a host/device type-identity naming divergence instead of an
    // opaque "named symbol not found". See `crate::entry_registry`.
    crate::entry_registry::register_merged_module_entries(&module, &merged);
    Ok(module)
}

/// Merge the PTX payloads of `bundles`, in iteration order, into one PTX
/// module string. The module starts with the highest `.version` of all
/// bundles and a `.target` that covers the targets of all bundles. The first
/// PTX bundle keeps its `.address_size` header, and later bundles contribute
/// bodies without it. Bundles without a PTX payload are skipped.
///
/// This is the pure half of [`load_all_ptx_bundles_merged`], exposed so tests
/// and examples can check order-dependent merge properties: module-scope
/// symbol uniqueness and extern alignment must hold for every bundle order,
/// not just the one the current executable happens to embed (#1277).
/// Module-scope variables without a linkage directive and non-extern device
/// helpers are renamed within their bundle; explicit device extern link names
/// and kernel identities remain.
/// Duplicate kernel definitions, including identical definitions, are rejected
/// before invoking the driver; typed loaders require an unambiguous entry.
pub fn merge_ptx_bundles<'a>(
    bundles: impl IntoIterator<Item = &'a OwnedArtifactBundle>,
) -> Result<String, EmbeddedModuleError> {
    let mut documents = Vec::new();
    for bundle in bundles {
        if let Some(bytes) = bundle.payload(ArtifactPayloadKind::Ptx) {
            let source = std::str::from_utf8(bytes)
                .map_err(|_| EmbeddedModuleError::UnsupportedPayload {
                    name: bundle.name.clone(),
                })?
                .trim_end_matches('\0');
            let document = ptx_parse::Document::parse(source).map_err(|error| {
                EmbeddedModuleError::InvalidPtx {
                    name: bundle.name.clone(),
                    reason: error.to_string(),
                }
            })?;
            documents.push((bundle.name.as_str(), document));
        }
    }
    if documents.is_empty() {
        return Err(EmbeddedModuleError::NoModules);
    }

    let state = PtxMergeState::new(&documents)?;
    // Later bundles may use instructions introduced after the ISA version or
    // target of the first bundle. One header covers all bundles in every order.
    let mut merged = String::new();
    if let Some((major, minor)) = state.version {
        merged.push_str(&format!(".version {major}.{minor}\n"));
    }
    if let Some(target) = &state.target {
        merged.push_str(&format!(".target {target}\n"));
    }
    let mut file_index_offset = 0;
    for (index, (name, document)) in documents.iter().enumerate() {
        let (body, max_file_index) =
            prepare_bundle_body(document, name, index, file_index_offset, &state)?;
        file_index_offset = file_index_offset.max(max_file_index);
        merged.push_str(&body);
        if !body.ends_with('\n') {
            merged.push('\n');
        }
    }
    Ok(merged)
}

/// PTX ISA version from a `.version` directive, as (major, minor).
type PtxVersion = (u32, u32);

/// PTX target from a `.target` directive: the architecture, its `a` or `f`
/// suffix, and options such as `debug`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PtxTarget {
    capability: u32,
    suffix: Option<char>,
    options: Vec<String>,
}

impl PtxTarget {
    fn parse(arguments: &str) -> Option<Self> {
        let mut parts = arguments.split(',').map(str::trim);
        let arch = parts.next()?.strip_prefix("sm_")?;
        let digits = arch
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(arch.len());
        let suffix = match &arch[digits..] {
            "" => None,
            "a" => Some('a'),
            "f" => Some('f'),
            _ => return None,
        };
        Some(Self {
            capability: arch[..digits].parse().ok()?,
            suffix,
            options: parts.map(str::to_string).collect(),
        })
    }

    /// Whether the code of `other` is valid for this target, and each device
    /// for this target can run it. An `a` or `f` target covers only itself.
    fn covers(&self, other: &Self) -> bool {
        (self.capability, self.suffix) == (other.capability, other.suffix)
            || (other.suffix.is_none() && self.capability >= other.capability)
    }

    /// The target of `self` or `other` that covers both, with the options of
    /// both. `None` if neither covers the other.
    fn join(&self, other: &Self) -> Option<Self> {
        let (base, rest) = if self.covers(other) {
            (self, other)
        } else if other.covers(self) {
            (other, self)
        } else {
            return None;
        };
        let mut joined = base.clone();
        for option in &rest.options {
            if !joined.options.contains(option) {
                joined.options.push(option.clone());
            }
        }
        Some(joined)
    }
}

impl std::fmt::Display for PtxTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sm_{}", self.capability)?;
        if let Some(suffix) = self.suffix {
            write!(f, "{suffix}")?;
        }
        for option in &self.options {
            write!(f, ", {option}")?;
        }
        Ok(())
    }
}

/// The facts about all bundles that the merge of each bundle uses.
#[derive(Default)]
struct PtxMergeState<'source> {
    /// The highest `.version` of all bundles.
    version: Option<PtxVersion>,
    /// The `.target` that covers the targets of all bundles.
    target: Option<PtxTarget>,
    /// The functions that a bundle declares `.extern`.
    extern_functions: HashSet<&'source str>,
}

impl<'source> PtxMergeState<'source> {
    fn new(
        documents: &[(&str, ptx_parse::Document<'source>)],
    ) -> Result<Self, EmbeddedModuleError> {
        let mut state = Self::default();
        let mut entries = HashMap::new();
        for (index, (name, document)) in documents.iter().enumerate() {
            let invalid_ptx = |reason: String| EmbeddedModuleError::InvalidPtx {
                name: name.to_string(),
                reason,
            };
            for directive in document.directives() {
                let arguments = directive.arguments();
                match directive.name() {
                    ".version" => {
                        let version = arguments
                            .split_whitespace()
                            .next()
                            .and_then(|value| value.split_once('.'))
                            .and_then(|(major, minor)| {
                                Some((major.parse().ok()?, minor.parse().ok()?))
                            })
                            .ok_or_else(|| {
                                invalid_ptx(format!("invalid PTX version `{arguments}`"))
                            })?;
                        state.version = state.version.max(Some(version));
                    }
                    ".target" => {
                        let target = PtxTarget::parse(arguments).ok_or_else(|| {
                            invalid_ptx(format!("invalid PTX target `{arguments}`"))
                        })?;
                        let joined = match &state.target {
                            None => target,
                            Some(merged) => merged.join(&target).ok_or_else(|| {
                                EmbeddedModuleError::IncompatibleTarget {
                                    name: name.to_string(),
                                    target: target.to_string(),
                                    merged: merged.to_string(),
                                }
                            })?,
                        };
                        state.target = Some(joined);
                    }
                    _ => {}
                }
            }
            for callable in document.callables() {
                let defined = callable.definition_scope().is_some();
                match callable.kind() {
                    // Explicit device externs keep their link names across bundles.
                    // Every other helper belongs to the artifact that compiled it.
                    ptx_parse::CallableKind::Function if callable.is_extern() => {
                        state.extern_functions.insert(callable.name());
                    }
                    ptx_parse::CallableKind::Entry if defined => {
                        if let Some(first) = entries.insert(callable.name(), index) {
                            return Err(EmbeddedModuleError::ConflictingEntry {
                                entry: callable.name().to_string(),
                                first: documents[first].0.to_string(),
                                second: name.to_string(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(state)
    }
}

/// Strip the headers and shift debug indices past the previous bundles.
/// Returns the rewritten text and its highest debug file index.
fn prepare_bundle_body(
    document: &ptx_parse::Document,
    bundle_name: &str,
    index: usize,
    file_index_offset: u64,
    state: &PtxMergeState,
) -> Result<(String, u64), EmbeddedModuleError> {
    let ptx = document.source();
    let invalid_ptx = |reason: String| EmbeddedModuleError::InvalidPtx {
        name: bundle_name.to_string(),
        reason,
    };
    let mut edits = ptx_parse::EditScript::new();
    let mut max_file_index = 0u64;

    // Independently compiled artifacts can contain the same device helper and
    // NVVM reflection string. Keep definitions and all references artifact-local
    // without changing kernel entry identities or explicit extern link names.
    let mut local_symbols = HashSet::new();
    for directive in document.directives() {
        match directive.name() {
            ".version" | ".target" => {
                edits
                    .delete(directive.line_span())
                    .map_err(|error| invalid_ptx(error.to_string()))?;
            }
            ".address_size" if index != 0 => {
                edits
                    .delete(directive.line_span())
                    .map_err(|error| invalid_ptx(error.to_string()))?;
            }
            // A module-scope variable without a linkage directive.
            ".global" | ".const" | ".shared" if directive.scope() == ptx_parse::ScopeId::ROOT => {
                let arguments = span_tokens(document, directive.arguments_span());
                local_symbols.extend(declared_variables(ptx, arguments));
            }
            ".file" | ".loc" => {
                let arguments: Vec<_> = span_tokens(document, directive.arguments_span()).collect();
                let index = shift_file_index(
                    &mut edits,
                    ptx,
                    arguments.first().copied(),
                    file_index_offset,
                )
                .map_err(invalid_ptx)?;
                max_file_index = max_file_index.max(index);
                if directive.name() == ".loc" {
                    for (position, attribute) in arguments.windows(2).enumerate() {
                        if attribute[0].text(ptx) == "," && attribute[1].text(ptx) == "inlined_at" {
                            let index = shift_file_index(
                                &mut edits,
                                ptx,
                                arguments.get(position + 2).copied(),
                                file_index_offset,
                            )
                            .map_err(invalid_ptx)?;
                            max_file_index = max_file_index.max(index);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    for callable in document.callables() {
        if callable.kind() == ptx_parse::CallableKind::Function
            && callable.definition_scope().is_some()
            && !state.extern_functions.contains(callable.name())
        {
            local_symbols.insert(callable.name());
        }
    }
    let prefix = format!("cuda_oxide_merge_{index}_");
    for token in document.tokens() {
        let text = token.text(ptx);
        if token.kind() == ptx_parse::TokenKind::Word && local_symbols.contains(text) {
            edits
                .replace(token.span(), format!("{prefix}{text}"))
                .map_err(|error| invalid_ptx(error.to_string()))?;
        }
    }

    let body = edits
        .apply(ptx)
        .map_err(|error| invalid_ptx(error.to_string()))?;
    Ok((body, max_file_index))
}

/// The non-trivia tokens in `span`.
fn span_tokens<'document>(
    document: &'document ptx_parse::Document,
    span: Range<usize>,
) -> impl Iterator<Item = &'document ptx_parse::Token> {
    let tokens = document.tokens();
    let first = tokens.partition_point(|token| token.span().end <= span.start);
    tokens[first..]
        .iter()
        .take_while(move |token| token.span().start < span.end)
        .filter(|token| !token.kind().is_trivia())
}

/// The names that the arguments of a variable directive declare, for example
/// `a` and `b` in `.global .align 4 .b32 a, b[2] = {1, 2};`.
fn declared_variables<'source, 'document>(
    ptx: &'source str,
    arguments: impl Iterator<Item = &'document ptx_parse::Token>,
) -> Vec<&'source str> {
    let mut names = Vec::new();
    let mut depth = 0usize;
    let mut initializer = false;
    for token in arguments {
        let text = token.text(ptx);
        match text {
            "[" | "(" | "{" => depth += 1,
            "]" | ")" | "}" => depth = depth.saturating_sub(1),
            "=" if depth == 0 => initializer = true,
            "," if depth == 0 => initializer = false,
            _ if depth == 0
                && !initializer
                && token.kind() == ptx_parse::TokenKind::Word
                && !text.starts_with(|c: char| c == '.' || c.is_ascii_digit()) =>
            {
                names.push(text);
            }
            _ => {}
        }
    }
    names
}

/// Shift one complete decimal index token, preserving surrounding PTX.
fn shift_file_index(
    edits: &mut ptx_parse::EditScript,
    ptx: &str,
    token: Option<&ptx_parse::Token>,
    offset: u64,
) -> Result<u64, String> {
    let token = token.ok_or("malformed debug directive: missing file index")?;
    let text = token.text(ptx);
    if token.kind() != ptx_parse::TokenKind::Word || !text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(format!("malformed debug file index: {text:?}"));
    }
    let index: u64 = text
        .parse()
        .map_err(|_| format!("debug file index out of range: {text:?}"))?;
    let shifted = index
        .checked_add(offset)
        .ok_or("debug file index overflow")?;
    if offset != 0 {
        edits
            .replace(token.span(), shifted.to_string())
            .map_err(|error| error.to_string())?;
    }
    Ok(shifted)
}

/// Load the first embedded artifact bundle with a supported payload.
pub fn load_first_embedded_module(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    for bundle in artifact_bundles_from_current_exe()? {
        match load_bundle(ctx, &bundle) {
            Ok(module) => return Ok(module),
            Err(EmbeddedModuleError::UnsupportedPayload { .. }) => continue,
            Err(error) => return Err(error),
        }
    }
    Err(EmbeddedModuleError::NoModules)
}

fn load_bundle(
    ctx: &Arc<CudaContext>,
    bundle: &OwnedArtifactBundle,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    if let Some(cubin) = bundle.payload(ArtifactPayloadKind::Cubin) {
        return Ok(ctx.load_module_from_image(cubin)?);
    }

    if let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) {
        return Ok(ctx.load_module_from_image(ptx)?);
    }

    if let Some(nvvm_ir) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
        let emitted = target_arch_for_bundle(bundle)?;
        let execution = ltoir::execution_arch_for_context(ctx)?;
        let image = match ltoir::execution_route(&emitted, &execution)? {
            ltoir::ExecutionRoute::Cubin => ltoir::build_cubin_from_nvvm_ir_with_compile_options(
                nvvm_ir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
            ltoir::ExecutionRoute::PtxBridge => ltoir::build_ptx_from_nvvm_ir_with_compile_options(
                nvvm_ir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
        };
        return Ok(ctx.load_module_from_image(&image)?);
    }

    if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
        let emitted = target_arch_for_bundle(bundle)?;
        let execution = ltoir::execution_arch_for_context(ctx)?;
        let image = match ltoir::execution_route(&emitted, &execution)? {
            ltoir::ExecutionRoute::Cubin => ltoir::link_ltoir_to_cubin_with_compile_options(
                ltoir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
            ltoir::ExecutionRoute::PtxBridge => ltoir::link_ltoir_to_ptx_with_compile_options(
                ltoir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
        };
        return Ok(ctx.load_module_from_image(&image)?);
    }

    Err(EmbeddedModuleError::UnsupportedPayload {
        name: bundle.name.clone(),
    })
}

fn target_arch_for_bundle(
    bundle: &OwnedArtifactBundle,
) -> Result<cuda_artifact_finalizer::CudaArch, ltoir::LtoirError> {
    let explicit = std::env::var("CUDA_OXIDE_TARGET").ok();
    target_arch_for_bundle_with_explicit(bundle, explicit.as_deref())
}

fn target_arch_for_bundle_with_explicit(
    bundle: &OwnedArtifactBundle,
    explicit_target: Option<&str>,
) -> Result<cuda_artifact_finalizer::CudaArch, ltoir::LtoirError> {
    ltoir::resolve_source_target(concrete_bundle_target(&bundle.target)?, explicit_target)
}

fn concrete_bundle_target(
    target: &str,
) -> Result<Option<cuda_artifact_finalizer::CudaArch>, ltoir::LtoirError> {
    match target {
        // Compatibility for artifacts emitted before concrete NVVM targets
        // were recorded. New bundles must never use these sentinels.
        "libdevice" | "nvvm-ir" => Ok(None),
        target => Ok(Some(target.parse()?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle_with_target(target: &str) -> OwnedArtifactBundle {
        OwnedArtifactBundle {
            name: "demo".to_string(),
            target: target.to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: Vec::new(),
            entries: Vec::new(),
        }
    }

    #[test]
    fn target_arch_uses_bundle_sm_target() {
        assert_eq!(
            concrete_bundle_target("sm_90")
                .unwrap()
                .map(|target| target.sm()),
            Some("sm_90".to_string())
        );
    }

    fn ptx_bundle(name: &str, ptx: &str) -> OwnedArtifactBundle {
        use oxide_artifacts::OwnedArtifactPayload;
        OwnedArtifactBundle {
            name: name.to_string(),
            target: "sm_80".to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: vec![OwnedArtifactPayload {
                kind: ArtifactPayloadKind::Ptx,
                name: name.to_string(),
                bytes: ptx.as_bytes().to_vec(),
            }],
            entries: Vec::new(),
        }
    }

    #[test]
    fn retains_the_highest_isa_version_in_either_bundle_order() {
        let first = ptx_bundle(
            "first",
            ".version 7.1 // old header\n.target sm_86\n.address_size 64\n.visible .entry first() { ret; }\n",
        );
        let second = ptx_bundle(
            "second",
            ".version 7.3\n.target sm_86\n.address_size 64\n.visible .entry second() { ret; }\n",
        );
        for bundles in [[&first, &second], [&second, &first]] {
            let merged = merge_ptx_bundles(bundles).unwrap();
            assert_eq!(merged.matches(".version").count(), 1);
            assert!(merged.contains(".version 7.3"));
        }
    }

    #[test]
    fn keeps_helper_definitions_and_reflection_strings_local_to_each_bundle() {
        let source = |entry: &str| {
            format!(
                ".version 8.9\n.target sm_80\n.address_size 64\n.global .align 1 .b8 _$_str[2] = {{ 1, 0 }};\n// helper _$_str is a comment\n.file 1 \"helper _$_str\"\n.visible .func helper() {{ ret; }}\n.visible .entry {entry}() {{ call helper, (); mov.u64 %rd1, _$_str; ret; }}\n"
            )
        };
        let first = ptx_bundle("library", &source("first"));
        let second = ptx_bundle("binary", &source("second"));
        for (a, b) in [(&first, &second), (&second, &first)] {
            let merged = merge_ptx_bundles([a, b]).unwrap();
            for index in 0..2 {
                assert!(merged.contains(&format!(".func cuda_oxide_merge_{index}_helper()")));
                assert!(merged.contains(&format!("call cuda_oxide_merge_{index}_helper, ()")));
                assert!(merged.contains(&format!(".b8 cuda_oxide_merge_{index}__$_str[2]")));
                assert!(merged.contains(&format!("mov.u64 %rd1, cuda_oxide_merge_{index}__$_str")));
            }
            assert!(merged.contains(".entry first()"));
            assert!(merged.contains(".entry second()"));
            assert_eq!(merged.matches("// helper _$_str is a comment").count(), 2);
            assert_eq!(merged.matches("\"helper _$_str\"").count(), 2);
        }
    }

    #[test]
    fn preserves_explicit_device_extern_link_names_across_bundles() {
        let first = ptx_bundle(
            "caller",
            ".version 8.9\n.target sm_80\n.address_size 64\n.extern .func exported();\n.visible .entry caller() { call exported, (); ret; }\n",
        );
        let second = ptx_bundle(
            "provider",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .func exported() { ret; }\n.visible .entry provider() { ret; }\n",
        );
        for bundles in [[&first, &second], [&second, &first]] {
            let merged = merge_ptx_bundles(bundles).unwrap();
            assert!(merged.contains(".extern .func exported()"));
            assert!(merged.contains(".visible .func exported()"));
            assert!(merged.contains("call exported, ()"));
            assert!(!merged.contains("_exported"));
        }
    }

    #[test]
    fn rejects_conflicting_entry_definitions_in_either_bundle_order() {
        let name = "copy_TID_0123456789abcdef0123456789abcdef";
        let first = ptx_bundle(
            "library",
            &format!(
                ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry {name}() {{ ret; }}\n"
            ),
        );
        let second = ptx_bundle(
            "binary",
            &format!(
                ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry {name}(.param .u64 pointer) {{ ret; }}\n"
            ),
        );
        for (a, b) in [(&first, &second), (&second, &first)] {
            match merge_ptx_bundles([a, b]) {
                Err(EmbeddedModuleError::ConflictingEntry {
                    entry,
                    first,
                    second,
                }) => {
                    assert_eq!(entry, name);
                    assert_eq!(first, a.name);
                    assert_eq!(second, b.name);
                }
                result => panic!("expected conflicting entry: {result:?}"),
            }
        }
        assert!(matches!(
            merge_ptx_bundles([&first, &first]),
            Err(EmbeddedModuleError::ConflictingEntry { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_entries_inside_one_bundle() {
        let bundle = ptx_bundle(
            "duplicate",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry copy() { ret; }\n.visible .entry copy() { ret; }\n",
        );
        assert!(matches!(
            merge_ptx_bundles([&bundle]),
            Err(EmbeddedModuleError::ConflictingEntry { .. })
        ));
    }

    #[test]
    fn permits_distinct_specializations_and_repeated_extern_declarations() {
        let first = ptx_bundle(
            "first",
            ".version 8.9\n.target sm_80\n.address_size 64\n.extern .func helper();\n.visible .entry copy_TID_11111111111111111111111111111111() { ret; }\n",
        );
        let second = ptx_bundle(
            "second",
            ".version 8.9\n.target sm_80\n.address_size 64\n.extern .func helper();\n.visible .entry copy_TID_22222222222222222222222222222222() { ret; }\n",
        );
        let merged = merge_ptx_bundles([&first, &second]).unwrap();
        assert_eq!(merged.matches(".entry copy_TID_").count(), 2);
        assert_eq!(merged.matches(".extern .func helper()").count(), 2);
    }

    /// The merge is order-explicit: exactly one header set (the highest
    /// `.version`, a `.target` that covers all bundles, the first PTX bundle's
    /// `.address_size`), every body present in iteration order, non-PTX
    /// bundles skipped. #1277's collision-free-namespace example checks the
    /// same function under both bundle orders at runtime.
    #[test]
    fn merges_bundles_in_iteration_order_with_one_header_set() {
        let first = ptx_bundle(
            "first",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry a() { ret; }\n",
        );
        let second = ptx_bundle(
            "second",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry b() { ret; }\n",
        );
        let skipped = bundle_with_target("sm_80");

        let forward = merge_ptx_bundles([&first, &skipped, &second]).unwrap();
        assert_eq!(forward.matches(".version").count(), 1);
        assert_eq!(forward.matches(".target sm_80").count(), 1);
        assert!(forward.find(".entry a()").unwrap() < forward.find(".entry b()").unwrap());

        let reversed = merge_ptx_bundles([&second, &first]).unwrap();
        assert_eq!(reversed.matches(".version").count(), 1);
        assert!(reversed.find(".entry b()").unwrap() < reversed.find(".entry a()").unwrap());

        assert!(matches!(
            merge_ptx_bundles([&skipped]),
            Err(EmbeddedModuleError::NoModules)
        ));
    }

    fn target_bundle(name: &str, target: &str) -> OwnedArtifactBundle {
        ptx_bundle(
            name,
            &format!(
                ".version 8.9\n.target {target}\n.address_size 64\n.visible .entry {name}() {{ ret; }}\n"
            ),
        )
    }

    #[test]
    fn merge_uses_a_target_that_covers_all_bundles() {
        for (first, second, expected) in [
            ("sm_75", "sm_80", "sm_80"),
            ("sm_80", "sm_90a", "sm_90a"),
            ("sm_100a", "sm_100a", "sm_100a"),
            ("sm_80, debug", "sm_86", "sm_86, debug"),
        ] {
            let first = target_bundle("first", first);
            let second = target_bundle("second", second);
            for bundles in [[&first, &second], [&second, &first]] {
                let merged = merge_ptx_bundles(bundles).unwrap();
                let header = format!(".version 8.9\n.target {expected}\n");
                assert!(merged.starts_with(&header), "{merged}");
                assert_eq!(merged.matches(".target").count(), 1);
            }
        }
    }

    #[test]
    fn merge_rejects_targets_that_no_target_covers() {
        for (first, second) in [("sm_90a", "sm_100"), ("sm_90a", "sm_100a")] {
            let first = target_bundle("first", first);
            let second = target_bundle("second", second);
            for bundles in [[&first, &second], [&second, &first]] {
                assert!(matches!(
                    merge_ptx_bundles(bundles),
                    Err(EmbeddedModuleError::IncompatibleTarget { .. })
                ));
            }
        }
        let error = merge_ptx_bundles([
            &target_bundle("first", "sm_90a"),
            &target_bundle("second", "sm_100"),
        ])
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "embedded CUDA module 'second' targets `sm_100`, which is not compatible with `sm_90a`"
        );
    }

    /// Prepare `ptx` as the bundle at `index`, without other bundles.
    fn prepare(
        ptx: &str,
        index: usize,
        file_index_offset: u64,
    ) -> Result<(String, u64), EmbeddedModuleError> {
        let document =
            ptx_parse::Document::parse(ptx).map_err(|error| EmbeddedModuleError::InvalidPtx {
                name: "test".to_string(),
                reason: error.to_string(),
            })?;
        prepare_bundle_body(
            &document,
            "test",
            index,
            file_index_offset,
            &PtxMergeState::default(),
        )
    }

    #[test]
    fn private_variables_are_all_names_of_a_module_scope_declaration() {
        let ptx = "\
.global .align 4 .b32 a, b[2] = {1, 2}, c = 3;
.global .align 8 .u64 table[2] = {generic(a), helper};
.visible .entry kernel()
{
\t.shared .align 4 .b8 inner[4];
\tld.global.u32 %r1, [c];
\tret;
}
.func helper() { ret; }
";
        let (body, _) = prepare(ptx, 0, 0).unwrap();
        assert_eq!(
            body,
            "\
.global .align 4 .b32 cuda_oxide_merge_0_a, cuda_oxide_merge_0_b[2] = {1, 2}, cuda_oxide_merge_0_c = 3;
.global .align 8 .u64 cuda_oxide_merge_0_table[2] = {generic(cuda_oxide_merge_0_a), cuda_oxide_merge_0_helper};
.visible .entry kernel()
{
\t.shared .align 4 .b8 inner[4];
\tld.global.u32 %r1, [cuda_oxide_merge_0_c];
\tret;
}
.func cuda_oxide_merge_0_helper() { ret; }
"
        );
    }

    #[test]
    fn strips_only_structural_ptx_module_headers() {
        let ptx = "\
// .target sm_1
.version 8.9
.target sm_120a, debug
.address_size 64
.visible .entry kernel() { ret; }
";
        let (body, max_file_index) = prepare(ptx, 1, 0).unwrap();
        assert_eq!(body, "// .target sm_1\n.visible .entry kernel() { ret; }\n");
        assert_eq!(max_file_index, 0);
    }

    #[test]
    fn debug_indices_ignore_comments_paths_and_label_substrings() {
        let ptx = ".file /* 90 */ 1 \"source12.rs\"\n.loc /* 80 */ 1 2 3, function_name $inlined_at7, inlined_at /* 70 */ 1 4 5 // inlined_at 60\n";
        let (body, max_index) = prepare(ptx, 0, 3).unwrap();
        assert_eq!(
            body,
            ".file /* 90 */ 4 \"source12.rs\"\n.loc /* 80 */ 4 2 3, function_name $inlined_at7, inlined_at /* 70 */ 4 4 5 // inlined_at 60\n"
        );
        assert_eq!(max_index, 4);
    }

    #[test]
    fn malformed_debug_indices_and_overflow_return_errors() {
        for ptx in [
            ".file \"source12.rs\"\n",
            ".file /* 12 */\n",
            ".file -1 \"source.rs\"\n",
            ".file 1foo \"source.rs\"\n",
            ".loc 1 2 3, inlined_at\n",
            ".loc 1 2 3, inlined_at -1 4 5\n",
            ".file 18446744073709551616 \"source.rs\"\n",
        ] {
            assert!(prepare(ptx, 0, 1).is_err(), "{ptx}");
        }
        assert!(
            prepare(".file 18446744073709551615 \"source.rs\"\n", 0, 1)
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
    }

    #[test]
    fn merged_bundles_renumber_debug_file_indices_per_bundle() {
        let first = ptx_bundle(
            "first",
            "\
.version 8.9
.target sm_80
.address_size 64
.file 1 \"a/lib.rs\"
.file 2 \"shared/thread.rs\"
.visible .entry a()
{
\t.loc\t1 10 5
\tret;
}
",
        );
        let second = ptx_bundle(
            "second",
            "\
.version 8.9
.target sm_80
.address_size 64
.file 1 \"b/main.rs\"
.visible .entry b()
{
\t.loc\t1 3 1, function_name $f, inlined_at 1 7 0
\tret;
}
",
        );
        let plain = ptx_bundle(
            "plain",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry c() { ret; }\n",
        );

        let merged = merge_ptx_bundles([&first, &plain, &second]).unwrap();

        // First bundle's table and references are untouched.
        assert!(merged.contains(".file 1 \"a/lib.rs\""));
        assert!(merged.contains(".file 2 \"shared/thread.rs\""));
        assert!(merged.contains(".loc\t1 10 5"));

        // The debug-free middle bundle does not advance the offset; the
        // second debug bundle continues after the first's maximum of 2 —
        // declaration, `.loc` reference, and `inlined_at` reference alike.
        assert!(merged.contains(".file 3 \"b/main.rs\""));
        assert!(merged.contains(".loc\t3 3 1, function_name $f, inlined_at 3 7 0"));

        // No index is declared twice across the merged module.
        let declared: Vec<&str> = merged
            .lines()
            .filter(|line| line.trim_start().starts_with(".file"))
            .map(|line| line.split_whitespace().nth(1).unwrap())
            .collect();
        let mut deduped = declared.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(
            declared.len(),
            deduped.len(),
            "duplicate .file index: {declared:?}"
        );
    }

    #[test]
    fn target_arch_uses_bundle_compute_target() {
        assert_eq!(
            concrete_bundle_target("compute_90")
                .unwrap()
                .map(|target| target.sm()),
            Some("sm_90".to_string())
        );
    }

    #[test]
    fn target_arch_falls_back_for_non_arch_target() {
        // Older bundles used these names instead of recording a concrete
        // architecture. New bundles always record a validated `sm_*` target.
        for legacy in ["libdevice", "nvvm-ir"] {
            assert_eq!(concrete_bundle_target(legacy).unwrap(), None);
        }
    }

    #[test]
    fn legacy_bundle_without_recorded_target_requires_explicit_target() {
        for sentinel in ["libdevice", "nvvm-ir"] {
            let bundle = bundle_with_target(sentinel);
            let error = target_arch_for_bundle_with_explicit(&bundle, None)
                .expect_err("the original build target is required");
            assert!(matches!(error, ltoir::LtoirError::TargetNotFound));

            let asserted =
                target_arch_for_bundle_with_explicit(&bundle, Some("compute_86")).unwrap();
            assert_eq!(asserted.sm(), "sm_86");
        }
    }

    #[test]
    fn recorded_bundle_target_overrides_explicit_environment_target() {
        let bundle = bundle_with_target("sm_90");
        let selected = target_arch_for_bundle_with_explicit(&bundle, Some("sm_120")).unwrap();
        assert_eq!(selected.sm(), "sm_90");
    }

    #[test]
    fn target_arch_rejects_malformed_bundle_target() {
        assert!(concrete_bundle_target("sm_90x").is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn bundles_of_an_executable_anchor_match_the_current_exe() {
        static ANCHOR: u8 = 0;
        let by_anchor = artifact_bundles_containing(&ANCHOR).unwrap();
        let by_current_exe = artifact_bundles_from_current_exe().unwrap();
        assert_eq!(by_anchor, by_current_exe);
    }
}
