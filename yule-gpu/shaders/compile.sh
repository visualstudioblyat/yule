#!/bin/bash
# Compile GLSL compute shaders to SPIR-V. Requires glslc (Vulkan SDK).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/compiled"
mkdir -p "$OUT"

SHADERS=(add silu_mul rms_norm rope softmax embed_lookup attn_score attn_value qmv_q4_0 qmv_q4_k qmv_q8_0)

for s in "${SHADERS[@]}"; do
    glslc "$DIR/$s.comp" -o "$OUT/$s.spv"
    echo "$s.comp → $s.spv ($(wc -c < "$OUT/$s.spv") bytes)"
done
