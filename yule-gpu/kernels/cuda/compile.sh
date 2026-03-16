#!/usr/bin/env bash
# Compile CUDA C kernel sources (.cu) to PTX assembly (.ptx) using nvcc.
# Requires CUDA toolkit (nvcc) to be installed and on PATH.
#
# Usage: ./compile.sh [compute_capability]
#   compute_capability: target GPU arch, e.g. 70 (Volta), 80 (Ampere), 89 (Ada Lovelace)
#                       default: 70 (broad compatibility)
#
# Output: compiled/*.ptx files ready to be loaded by the CUDA backend at runtime.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE="${1:-70}"
OUT_DIR="$SCRIPT_DIR/compiled"

mkdir -p "$OUT_DIR"

KERNELS=(
    add
    silu_mul
    rms_norm
    softmax
    rope
    attn_score
    attn_value
    embed_lookup
    matmul
    qmv_q4_0
    qmv_q8_0
    qmv_q4_k
    qmv_q6_k
)

echo "Compiling CUDA kernels for sm_${COMPUTE}..."

for kernel in "${KERNELS[@]}"; do
    echo "  ${kernel}.cu -> ${kernel}.ptx"
    nvcc -ptx \
        -arch="sm_${COMPUTE}" \
        -o "$OUT_DIR/${kernel}.ptx" \
        "$SCRIPT_DIR/${kernel}.cu"
done

echo "Done. PTX files written to $OUT_DIR/"
