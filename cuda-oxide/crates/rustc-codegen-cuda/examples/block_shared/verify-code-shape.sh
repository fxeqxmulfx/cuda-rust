#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The runtime check passes also when the staged tiles are slow. The kernels
# must read a staged tile with `ld.shared`. A generic `ld` shows that the tile
# pointer is not known to be in shared memory.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/block_shared.ptx"

test -s "${ptx}"

# Print the body of the first PTX entry whose name starts with the marker.
entry_body() {
    local marker="$1"
    awk -v marker=".entry ${marker}" '
        !emit && !candidate && index($0, marker) {
            candidate = 1
        }
        candidate && $0 ~ /^[[:space:]]*;[[:space:]]*$/ {
            candidate = 0
            next
        }
        candidate && $0 ~ /^[[:space:]]*\{[[:space:]]*$/ {
            emit = 1
            candidate = 0
        }
        emit { print }
        emit && index($0, "End function") != 0 { exit }
    ' "${ptx}"
}

require_entry_shape() {
    local marker="$1"
    local description="$2"
    local pattern="$3"
    if ! entry_body "${marker}" | grep -E "${pattern}" >/dev/null; then
        echo "error: missing ${description} in ${ptx}:${marker}" >&2
        exit 1
    fi
}

reject_entry_shape() {
    local marker="$1"
    local description="$2"
    local pattern="$3"
    if entry_body "${marker}" | grep -E "${pattern}" >/dev/null; then
        echo "error: found ${description} in ${ptx}:${marker}" >&2
        exit 1
    fi
}

shared_load='[[:space:]]ld\.shared\.'
generic_load='[[:space:]]ld\.(v[248]\.)?[bfsu][0-9]+[[:space:]]'

for kernel in 'neighbor_u32(' 'shared_TID_'; do
    require_entry_shape "${kernel}" "shared tile load" "${shared_load}"
    reject_entry_shape "${kernel}" "generic tile load" "${generic_load}"
done

echo "block_shared code shape: PASS"
