#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The runtime check passes also when the staged copy is slow. The copy loop
# must check the overflow one time for each tile, not for each element. The
# copy fills the elements out of bounds with zero, so the pipeline kernels have
# no shared stores.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/block_pipeline.ptx"

test -s "${ptx}"

# Print "entry loop" for each innermost loop of a pipeline entry that matches
# both regular expressions. LLVM writes the loop of each basic block as a
# comment.
loops_with() {
    awk -v first="$1" -v second="$2" '
        /^\.visible \.entry (sum|swizzled)_TID_/ {
            entry = $3
            sub(/\(.*/, "", entry)
        }
        /^\.visible \.entry / && !/(sum|swizzled)_TID_/ { entry = "" }
        entry == "" { next }
        /^\$L__BB[0-9_]+:/ {
            label = $1
            sub(/^\$L__/, "", label)
            sub(/:$/, "", label)
            loop = "none"
        }
        /in Loop: Header=/ {
            match($0, /Header=BB[0-9_]+/)
            loop = substr($0, RSTART + 7, RLENGTH - 7)
        }
        /This (Inner )?Loop Header/ { loop = label }
        $0 ~ first { has_first[entry " " loop] = 1 }
        $0 ~ second { has_second[entry " " loop] = 1 }
        END {
            for (key in has_first) {
                if (key in has_second) print key
            }
        }
    ' "${ptx}"
}

copy='cp[.]async[.]c[ag][.]shared'
if [[ -z "$(loops_with "${copy}" "${copy}")" ]]; then
    echo "error: no copy loop in ${ptx}" >&2
    exit 1
fi
checked="$(loops_with "${copy}" 'mul[.]hi[.]u64')"
if [[ -n "${checked}" ]]; then
    echo "error: overflow check in the copy loop of ${ptx}: ${checked}" >&2
    exit 1
fi
if grep -E '[[:space:]]st\.shared\.' "${ptx}" >/dev/null; then
    echo "error: shared store in ${ptx}" >&2
    exit 1
fi

echo "block_pipeline code shape: PASS"
