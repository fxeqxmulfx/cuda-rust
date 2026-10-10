/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! In a cuda-oxide checkout, `cargo oxide` runs the `cargo-oxide` of the
//! checkout.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Makes a checkout at `root/simt`. Its `cargo-oxide` prints its arguments
/// and exits with 7. With `combined`, `root` is the combined repository root.
fn checkout(name: &str, combined: bool) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "cargo_oxide_checkout_{}_{}",
        name,
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    let simt = if combined {
        root.join("cuda-oxide")
    } else {
        root.clone()
    };
    let files = [
        (
            simt.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/cargo-oxide\"]\nresolver = \"3\"\n",
        ),
        (simt.join("crates/rustc-codegen-cuda/Cargo.toml"), ""),
        (
            simt.join("crates/cargo-oxide/Cargo.toml"),
            "[package]\nname = \"cargo-oxide\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
        ),
        (
            simt.join("crates/cargo-oxide/src/main.rs"),
            "fn main() {\n    let args: Vec<String> = std::env::args().skip(1).collect();\n    println!(\"checkout {args:?}\");\n    std::process::exit(7);\n}\n",
        ),
    ];
    for (path, text) in files {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    if combined {
        fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    }
    (root, simt)
}

fn cargo_oxide(dir: &Path, target: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cargo-oxide"))
        .args(args)
        .current_dir(dir)
        // The `cargo-oxide` of the checkout must not replace this one.
        .env("CARGO_TARGET_DIR", target)
        .output()
        .unwrap()
}

#[test]
fn cargo_subcommand_runs_the_checkout_command() {
    let (root, simt) = checkout("subcommand", false);
    let output = cargo_oxide(
        &simt.join("crates"),
        &root.join("target"),
        &["oxide", "list", "--json"],
    );
    let _ = fs::remove_dir_all(&root);
    assert_eq!(output.status.code(), Some(7), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "checkout [\"list\", \"--json\"]\n"
    );
}

#[test]
fn cargo_run_runs_its_own_command() {
    let (root, simt) = checkout("run", false);
    let output = cargo_oxide(&simt, &root.join("target"), &["--version"]);
    let _ = fs::remove_dir_all(&root);
    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("cargo-oxide "));
}

#[test]
fn combined_repository_root_runs_its_own_command() {
    let (root, _) = checkout("combined", true);
    let output = cargo_oxide(&root, &root.join("target"), &["oxide", "--version"]);
    let _ = fs::remove_dir_all(&root);
    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("cargo-oxide "));
}
