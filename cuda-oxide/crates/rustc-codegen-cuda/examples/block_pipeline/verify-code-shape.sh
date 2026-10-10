#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The runtime check passes also when the staged copy is slow. The copy loop
# must check the overflow one time for each tile, not for each element. The
# copy fills the elements out of bounds with zero, so the pipeline kernels have
# no shared stores. Row-major tiles with 16-byte rows and strides must also
# have a 16-byte copy. A `BlockTile` with a runtime offset and stride must also
# check the overflow one time, not for each element.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/block_pipeline.ptx"

test -s "${ptx}"

# Print "entry loop" for each innermost loop of an entry of the kernels in the
# first regular expression that matches the other two regular expressions.
# LLVM writes the loop of each basic block as a comment.
loops_with() {
    awk -v kernels="$1" -v first="$2" -v second="$3" '
        /^\.visible \.entry / {
            entry = $3
            sub(/\(.*/, "", entry)
            if (entry !~ "^(" kernels ")_TID_") entry = ""
            loop = "none"
        }
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
        loop != "none" && $0 ~ first { has_first[entry " " loop] = 1 }
        loop != "none" && $0 ~ second { has_second[entry " " loop] = 1 }
        END {
            for (key in has_first) {
                if (key in has_second) print key
            }
        }
    ' "${ptx}"
}

copy='cp[.]async[.]c[ag][.]shared'
if [[ -z "$(loops_with 'sum|swizzled' "${copy}" "${copy}")" ]]; then
    echo "error: no copy loop in ${ptx}" >&2
    exit 1
fi
checked="$(loops_with 'sum|swizzled' "${copy}" 'mul[.]hi[.]u64')"
if [[ -n "${checked}" ]]; then
    echo "error: overflow check in the copy loop of ${ptx}: ${checked}" >&2
    exit 1
fi
if grep -E '[[:space:]]st\.shared\.' "${ptx}" >/dev/null; then
    echo "error: shared store in ${ptx}" >&2
    exit 1
fi

# Of the `sum` kernels, D = 32 and D = 128 have 16-byte rows. The XOR layout
# of `swizzled` moves single elements.
vector_entries() {
    awk -v kernel="$1" '
        /^\.visible \.entry / { entry = $3 }
        index(entry, kernel "_TID_") == 1 && /cp[.]async[.]cg[.]shared[.]global .*, 16,/ {
            found[entry] = 1
        }
        END {
            count = 0
            for (key in found) count++
            print count
        }
    ' "${ptx}"
}
if [[ "$(vector_entries sum)" != 2 ]]; then
    echo "error: expected a 16-byte copy in 2 sum kernels of ${ptx}" >&2
    exit 1
fi
if [[ "$(vector_entries swizzled)" != 0 ]]; then
    echo "error: 16-byte copy in a swizzled kernel of ${ptx}" >&2
    exit 1
fi

load='ld[.]global[.]'
if [[ -z "$(loops_with view "${load}" "${load}")" ]]; then
    echo "error: no load loop in the view kernel of ${ptx}" >&2
    exit 1
fi
checked="$(loops_with view "${load}" 'mul[.]hi[.]u64')"
if [[ -n "${checked}" ]]; then
    echo "error: overflow check in the load loop of ${ptx}: ${checked}" >&2
    exit 1
fi
if awk '/^\.visible \.entry view_TID_/,/^}/' "${ptx}" | grep -E 'trap' >/dev/null; then
    echo "error: trap in the view kernel of ${ptx}" >&2
    exit 1
fi

echo "block_pipeline code shape: PASS"
