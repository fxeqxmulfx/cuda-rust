/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Keep the artifact objects of upstream crates that instantiate generic
//! kernels.
//!
//! A crate puts the generic kernel specializations that it instantiates in
//! its own artifact object. In a library, this object is an rlib member, and
//! the linker extracts it only when a symbol refers to its anchor. The
//! `#[cuda_module]` loader refers only to the anchor of the crate that
//! defines the module. Thus, without this module, the linker does not keep
//! the specializations of a library that does not define the kernel.
//!
//! Before rustc links an executable or a dynamic library, find the
//! statically linked artifact objects with generic kernel specializations.
//! rustc then refers to their anchors from `symbols.o`.

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use object::read::archive::ArchiveFile;
use object::{Object, ObjectSymbol};
use oxide_artifacts::ArtifactEntryKind;
use rustc_codegen_ssa::CrateInfo;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_middle::middle::exported_symbols::SymbolExportKind;
use rustc_session::Session;
use rustc_session::config::Lto;
use rustc_structures::CrateType;

/// Add the anchors of the statically linked artifact objects with generic
/// kernel specializations to the symbols that rustc keeps.
pub(crate) fn retain_specialization_artifacts(sess: &Session, crate_info: &mut CrateInfo) {
    // With these LTO modes, rustc does not link the Rust objects of the
    // rlibs, and the artifact objects are Rust objects.
    let rlib_objects_removed = match sess.lto() {
        Lto::Fat => true,
        Lto::Thin => !sess.opts.cg.linker_plugin_lto.enabled(),
        Lto::No | Lto::ThinLocal => false,
    };
    if rlib_objects_removed {
        return;
    }

    let mut anchors = HashMap::new();
    for crate_type in crate_info.crate_types.clone() {
        if !matches!(
            crate_type,
            CrateType::Executable | CrateType::Dylib | CrateType::Cdylib | CrateType::Sdylib
        ) {
            continue;
        }
        let Some(linkage) = crate_info.dependency_formats.get(&crate_type) else {
            continue;
        };
        let mut symbols = Vec::new();
        for &cnum in &crate_info.used_crates {
            if linkage.get(cnum) != Some(&Linkage::Static) {
                continue;
            }
            let Some(rlib) = crate_info
                .used_crate_source
                .get(&cnum)
                .and_then(|source| source.rlib.as_deref())
            else {
                continue;
            };
            let crate_anchors = anchors.entry(cnum).or_insert_with(|| {
                specialization_anchors(rlib).unwrap_or_else(|error| {
                    sess.dcx().fatal(format!(
                        "[rustc_codegen_cuda] Failed to read the artifact objects of `{}`: {error}",
                        rlib.display()
                    ))
                })
            });
            symbols.extend(
                crate_anchors
                    .iter()
                    .map(|anchor| (anchor.clone(), SymbolExportKind::Data)),
            );
        }
        crate_info
            .linked_symbols
            .entry(crate_type)
            .or_default()
            .extend(symbols);
    }
}

/// The anchors of the artifact objects in `rlib` that have a generic kernel
/// specialization.
fn specialization_anchors(rlib: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let data = object::ReadCache::new(std::fs::File::open(rlib)?);
    let archive = ArchiveFile::parse(&data)?;
    let mut anchors = Vec::new();
    for member in archive.members() {
        let member = member?;
        // `write_artifact_object` gives this suffix to the artifact objects.
        if member.name().ends_with(b".embed.o") {
            anchors.extend(specialization_anchor(member.data(&data)?)?);
        }
    }
    Ok(anchors)
}

/// The anchor of an artifact object, if the object has a generic kernel
/// specialization.
fn specialization_anchor(object: &[u8]) -> Result<Option<String>, Box<dyn Error>> {
    let bundles = oxide_artifacts::read_artifact_bundles_from_object_bytes(object)?;
    // The collector reserves `_TID_` for generic kernel specializations.
    let specializes = bundles
        .iter()
        .flat_map(|bundle| &bundle.entries)
        .any(|entry| entry.kind == ArtifactEntryKind::Kernel && entry.symbol.contains("_TID_"));
    if !specializes {
        return Ok(None);
    }
    // The object also defines a weak legacy alias of the anchor.
    let file = object::File::parse(object)?;
    for symbol in file.symbols() {
        let name = symbol.name()?;
        if symbol.is_definition()
            && symbol.is_global()
            && !symbol.is_weak()
            && name.starts_with(reserved_oxide_symbols::ARTIFACT_ANCHOR_PREFIX)
        {
            return Ok(Some(name.to_string()));
        }
    }
    Err("an artifact object does not define an anchor".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact_object(entry: &str, anchor: &str) -> Vec<u8> {
        let spec = oxide_artifacts::ArtifactBundleSpec::new("library", "sm_80")
            .with_payload(oxide_artifacts::ArtifactPayloadSpec::new(
                oxide_artifacts::ArtifactPayloadKind::Ptx,
                "library.ptx",
                b".version 8.0\n",
            ))
            .with_entry(oxide_artifacts::ArtifactEntrySpec::new(
                entry,
                ArtifactEntryKind::Kernel,
            ));
        let blob = oxide_artifacts::build_artifact_blob(&spec).unwrap();
        oxide_artifacts::build_host_object_for_target_with_legacy_anchor(
            &blob,
            "x86_64-unknown-linux-gnu",
            anchor,
            &format!("{anchor}_legacy"),
        )
        .unwrap()
    }

    /// A GNU archive with members that have names of at most 15 bytes.
    fn archive(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut archive = b"!<arch>\n".to_vec();
        for (name, data) in members {
            let header = format!(
                "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
                format!("{name}/"),
                0,
                0,
                0,
                644,
                data.len()
            );
            archive.extend_from_slice(header.as_bytes());
            archive.extend_from_slice(data);
            if data.len() % 2 == 1 {
                archive.push(b'\n');
            }
        }
        archive
    }

    #[test]
    fn finds_the_anchors_of_artifact_objects_with_specializations() {
        let anchor = format!(
            "{}v2_library_0_1_0_library_nonbin",
            reserved_oxide_symbols::ARTIFACT_ANCHOR_PREFIX
        );
        let generic = artifact_object("scale_TID_0123456789abcdef0123456789abcdef", &anchor);
        let other_anchor = format!(
            "{}v2_other_0_1_0_other_nonbin",
            reserved_oxide_symbols::ARTIFACT_ANCHOR_PREFIX
        );
        let plain = artifact_object("vecadd", &other_anchor);
        let rlib = archive(&[
            ("lib.rmeta", b"metadata"),
            ("a.0.0.embed.o", &plain),
            ("a.0.1.embed.o", &generic),
            // Only the artifact objects are read.
            ("a.rcgu.o", &generic),
        ]);
        let directory = std::env::temp_dir().join(format!(
            "rustc-codegen-cuda-upstream-artifacts-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("liblibrary.rlib");
        std::fs::write(&path, rlib).unwrap();

        let anchors = specialization_anchors(&path).unwrap();
        std::fs::remove_dir_all(&directory).unwrap();
        assert_eq!(anchors, [anchor]);
    }
}
