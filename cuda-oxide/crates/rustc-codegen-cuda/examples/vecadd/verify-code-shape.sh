#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# `DisjointSlice::get_mut` returns `Option<&mut T>`, and `None` is a null
# pointer. After the bounds check, the kernel must not compare the slice
# pointer with null.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/vecadd.ptx"
test -s "${ptx}"

python3 - "${ptx}" <<'PY'
import re
import sys

ptx_path = sys.argv[1]
ptx = open(ptx_path, encoding="utf-8").read()
# Strip comments so instructions mentioned in comments cannot satisfy a check.
ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
header = re.search(r"\.entry\s+vecadd\s*\([^)]*\)\s*\{", ptx)
if header is None:
    sys.exit(f"error: missing PTX entry definition vecadd in {ptx_path}")
depth = 1
end = header.end()
while depth and end < len(ptx):
    depth += (ptx[end] == "{") - (ptx[end] == "}")
    end += 1
if depth:
    sys.exit("error: unterminated PTX entry vecadd")
body = ptx[header.end():end - 1]

if not re.search(r"\bsetp\.\w+\.u64\b", body):
    sys.exit(f"error: vecadd lost its bounds check in {ptx_path}")

# Registers that hold a null pointer: a zero, or a zero in another state space.
null = {"0"}
for instruction in re.sub(r"[\w$]+:", " ", body).split(";"):
    words = [word for word in instruction.replace(",", " ").split() if word[0] != "@"]
    if len(words) == 3 and re.fullmatch(r"(mov|cvta(\.to)?\.\w+)\.[bsu]64", words[0]):
        if words[2] in null:
            null.add(words[1])
    if len(words) == 4 and re.fullmatch(r"setp\.(eq|ne)\.[bsu]64", words[0]):
        if words[2] in null or words[3] in null:
            sys.exit(f"error: vecadd compares a pointer with null in {ptx_path}")
print("vecadd: checked PTX PASS")
PY
