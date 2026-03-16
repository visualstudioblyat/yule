#include <metal_stdlib>
using namespace metal;

// ---- Helper: unpack f16 from raw ushort bits ----
inline float unpack_f16(ushort bits) {
    return float(as_type<half>(bits));
}

// ---- Helper: read byte from byte-addressable buffer ----
inline uchar read_byte(device const uchar* data, uint byte_addr) {
    return data[byte_addr];
}

inline ushort read_u16(device const uchar* data, uint byte_addr) {
    return ushort(data[byte_addr]) | (ushort(data[byte_addr + 1]) << 8);
}

inline int read_i8(device const uchar* data, uint byte_addr) {
    uchar b = data[byte_addr];
    return int(b) - (int(b & 0x80) << 1);
}

// ============================================================
// 1. add_kernel
// ============================================================

struct AddParams {
    uint size;
};

kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant AddParams& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.size) return;
    out[idx] = a[idx] + b[idx];
}

// ============================================================
// 2. silu_mul_kernel
// ============================================================

struct SiluMulParams {
    uint size;
};

kernel void silu_mul_kernel(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant SiluMulParams& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.size) return;
    float x = gate[idx];
    float silu_x = x / (1.0f + exp(-x));
    out[idx] = silu_x * up[idx];
}

// ============================================================
// 3. rms_norm_kernel
// ============================================================

struct RmsNormParams {
    uint size;
    float eps;
};

kernel void rms_norm_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    threadgroup float partial_sums[256];

    float sum = 0.0f;
    for (uint i = tid; i < params.size; i += tpg) {
        float v = input[i];
        sum += v * v;
    }
    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = 1.0f / sqrt(partial_sums[0] / float(params.size) + params.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < params.size; i += tpg) {
        out[i] = input[i] * inv_rms * weight[i];
    }
}

// ============================================================
// 4. softmax_kernel
// ============================================================

struct SoftmaxParams {
    uint size;
};

kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    threadgroup float shared_data[256];

    // Max reduction
    float local_max = -1.0e30f;
    for (uint i = tid; i < params.size; i += tpg) {
        local_max = max(local_max, input[i]);
    }
    shared_data[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exp + sum reduction
    float local_sum = 0.0f;
    for (uint i = tid; i < params.size; i += tpg) {
        float e = exp(input[i] - max_val);
        out[i] = e;
        local_sum += e;
    }
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) shared_data[tid] += shared_data[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < params.size; i += tpg) {
        out[i] *= inv_sum;
    }
}

// ============================================================
// 5. rope_kernel
// ============================================================

struct RopeParams {
    uint pos;
    uint head_dim;
    uint freq_base_bits;
    uint n_heads_q;
    uint n_heads_k;
};

kernel void rope_kernel(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant RopeParams& params [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    uint half_dim = params.head_dim / 2;
    uint total_q = params.n_heads_q * half_dim;
    uint total_k = params.n_heads_k * half_dim;

    if (idx >= total_q + total_k) return;

    float freq_base = as_type<float>(params.freq_base_bits);

    if (idx < total_q) {
        uint pair_idx = idx % half_dim;
        uint head = idx / half_dim;
        uint base_off = head * params.head_dim;

        float freq = 1.0f / pow(freq_base, float(2 * pair_idx) / float(params.head_dim));
        float angle = float(params.pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float q0 = q[base_off + 2 * pair_idx];
        float q1 = q[base_off + 2 * pair_idx + 1];
        q[base_off + 2 * pair_idx]     = q0 * cos_a - q1 * sin_a;
        q[base_off + 2 * pair_idx + 1] = q0 * sin_a + q1 * cos_a;
    } else {
        uint k_idx = idx - total_q;
        uint pair_idx = k_idx % half_dim;
        uint head = k_idx / half_dim;
        uint base_off = head * params.head_dim;

        float freq = 1.0f / pow(freq_base, float(2 * pair_idx) / float(params.head_dim));
        float angle = float(params.pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float k0 = k[base_off + 2 * pair_idx];
        float k1 = k[base_off + 2 * pair_idx + 1];
        k[base_off + 2 * pair_idx]     = k0 * cos_a - k1 * sin_a;
        k[base_off + 2 * pair_idx + 1] = k0 * sin_a + k1 * cos_a;
    }
}

// ============================================================
// 6. embed_lookup_kernel
// ============================================================

struct EmbedLookupParams {
    uint embed_dim;
    uint token_idx;
};

kernel void embed_lookup_kernel(
    device const float* embed_table [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant EmbedLookupParams& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.embed_dim) return;
    uint tok = token_ids[params.token_idx];
    out[idx] = embed_table[tok * params.embed_dim + idx];
}

// ============================================================
// 7. attn_score_kernel
// ============================================================

struct AttnScoreParams {
    uint head_dim;
    uint seq_len;
    uint head_offset;
    uint kv_offset;
    uint kv_stride;
};

kernel void attn_score_kernel(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant AttnScoreParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint pos [[threadgroup_position_in_grid]]
) {
    if (pos >= params.seq_len) return;

    threadgroup float partial_dots[256];

    float dot = 0.0f;
    for (uint i = tid; i < params.head_dim; i += tpg) {
        dot += query[params.head_offset + i] * key_cache[pos * params.kv_stride + params.kv_offset + i];
    }
    partial_dots[tid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) partial_dots[tid] += partial_dots[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        scores[pos] = partial_dots[0] / sqrt(float(params.head_dim));
    }
}

// ============================================================
// 8. attn_value_kernel
// ============================================================

struct AttnValueParams {
    uint head_dim;
    uint seq_len;
    uint kv_offset;
    uint kv_stride;
    uint out_offset;
};

kernel void attn_value_kernel(
    device const float* attn_weights [[buffer(0)]],
    device const float* value_cache [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant AttnValueParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint dim_idx [[threadgroup_position_in_grid]]
) {
    if (dim_idx >= params.head_dim) return;

    threadgroup float partial_sums[256];

    float sum = 0.0f;
    for (uint pos = tid; pos < params.seq_len; pos += tpg) {
        sum += attn_weights[pos] * value_cache[pos * params.kv_stride + params.kv_offset + dim_idx];
    }

    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[params.out_offset + dim_idx] = partial_sums[0];
    }
}

// ============================================================
// 9. qmv_q4_0_kernel  (Q4_0: 18 bytes/block, 32 weights)
// ============================================================

struct QmvParams {
    uint n_rows;
    uint n_cols;
    uint blocks_per_row;
};

kernel void qmv_q4_0_kernel(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant QmvParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    if (row >= params.n_rows) return;

    threadgroup float partial_sums[256];

    uint row_byte_offset = row * params.blocks_per_row * 18u;
    float sum = 0.0f;

    for (uint block = tid; block < params.blocks_per_row; block += tpg) {
        uint boff = row_byte_offset + block * 18u;

        // Read f16 scale from first 2 bytes
        float scale = unpack_f16(read_u16(weights, boff));

        uint qoff = boff + 2u;
        uint col_base = block * 32u;

        for (uint j = 0u; j < 16u; j++) {
            uchar byte_val = read_byte(weights, qoff + j);

            float q0 = float(int(byte_val & 0xFu) - 8);
            float q1 = float(int((byte_val >> 4u) & 0xFu) - 8);

            uint c0 = col_base + j * 2u;
            if (c0 < params.n_cols)      sum += scale * q0 * input[c0];
            if (c0 + 1u < params.n_cols)  sum += scale * q1 * input[c0 + 1u];
        }
    }

    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[row] = partial_sums[0];
}

// ============================================================
// 10. qmv_q8_0_kernel  (Q8_0: 34 bytes/block, 32 weights)
// ============================================================

kernel void qmv_q8_0_kernel(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant QmvParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    if (row >= params.n_rows) return;

    threadgroup float partial_sums[256];

    uint row_byte_offset = row * params.blocks_per_row * 34u;
    float sum = 0.0f;

    for (uint block = tid; block < params.blocks_per_row; block += tpg) {
        uint boff = row_byte_offset + block * 34u;

        float scale = unpack_f16(read_u16(weights, boff));

        uint qoff = boff + 2u;
        uint col_base = block * 32u;

        for (uint j = 0u; j < 32u; j++) {
            uchar byte_val = read_byte(weights, qoff + j);
            int q = int(byte_val);
            if (q >= 128) q -= 256;

            uint c = col_base + j;
            if (c < params.n_cols) sum += scale * float(q) * input[c];
        }
    }

    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[row] = partial_sums[0];
}

// ============================================================
// 11. qmv_q4_k_kernel  (Q4_K: 144 bytes/super-block, 256 weights)
// ============================================================

kernel void qmv_q4_k_kernel(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant QmvParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    if (row >= params.n_rows) return;

    threadgroup float partial_sums[256];

    uint row_byte_offset = row * params.blocks_per_row * 144u;
    float sum = 0.0f;

    for (uint blk = tid; blk < params.blocks_per_row; blk += tpg) {
        uint boff = row_byte_offset + blk * 144u;

        float d    = unpack_f16(read_u16(weights, boff));
        float dmin = unpack_f16(read_u16(weights, boff + 2u));

        // Read 12 raw scale bytes
        uint q[12];
        for (uint i = 0u; i < 12u; i++) {
            q[i] = uint(read_byte(weights, boff + 4u + i));
        }

        // Extract 6-bit scales and mins (ggml get_scale_min_k4)
        uint sc8[8];
        uint mn8[8];
        for (uint j = 0u; j < 4u; j++) {
            sc8[j] = q[j] & 63u;
            mn8[j] = q[j + 4u] & 63u;
        }
        for (uint j = 4u; j < 8u; j++) {
            sc8[j] = (q[j + 4u] & 0xFu) | ((q[j - 4u] >> 6u) << 4u);
            mn8[j] = (q[j + 4u] >> 4u)  | ((q[j] >> 6u) << 4u);
        }

        uint quant_off = boff + 16u;
        uint col_base = blk * 256u;

        for (uint grp = 0u; grp < 4u; grp++) {
            uint qs_off = quant_off + grp * 32u;

            // Sub-block 2*grp: low nibbles (first 32 weights)
            uint sb0 = 2u * grp;
            float scale0 = d * float(sc8[sb0]);
            float min0   = dmin * float(mn8[sb0]);
            for (uint l = 0u; l < 32u; l++) {
                uint c = col_base + 64u * grp + l;
                if (c < params.n_cols) {
                    float qval = float(read_byte(weights, qs_off + l) & 0xFu);
                    sum += (qval * scale0 - min0) * input[c];
                }
            }

            // Sub-block 2*grp+1: high nibbles (next 32 weights)
            uint sb1 = 2u * grp + 1u;
            float scale1 = d * float(sc8[sb1]);
            float min1   = dmin * float(mn8[sb1]);
            for (uint l = 0u; l < 32u; l++) {
                uint c = col_base + 64u * grp + 32u + l;
                if (c < params.n_cols) {
                    float qval = float((read_byte(weights, qs_off + l) >> 4u) & 0xFu);
                    sum += (qval * scale1 - min1) * input[c];
                }
            }
        }
    }

    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2u; s > 0u; s >>= 1u) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[row] = partial_sums[0];
}

// ============================================================
// 12. qmv_q6_k_kernel  (Q6_K: 210 bytes/super-block, 256 weights)
// ============================================================

kernel void qmv_q6_k_kernel(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant QmvParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    if (row >= params.n_rows) return;

    threadgroup float partial_sums[256];

    uint row_byte_offset = row * params.blocks_per_row * 210u;
    float sum = 0.0f;

    for (uint block = tid; block < params.blocks_per_row; block += tpg) {
        uint boff = row_byte_offset + block * 210u;

        // d is at offset 208 within the block
        float d = unpack_f16(read_u16(weights, boff + 208u));

        uint col_base = block * 256u;

        // Process two halves of 128 weights each
        for (uint h = 0u; h < 2u; h++) {
            uint ql_off = boff + h * 64u;
            uint qh_off = boff + 128u + h * 32u;
            uint sc_off = boff + 192u;
            uint sc_base = h * 8u;
            uint out_base = col_base + h * 128u;

            for (uint l = 0u; l < 32u; l++) {
                uint is_idx = l / 16u;

                uint ql_lo = uint(read_byte(weights, ql_off + l));
                uint ql_hi = uint(read_byte(weights, ql_off + l + 32u));
                uint qh_val = uint(read_byte(weights, qh_off + l));

                int q1 = int((ql_lo & 0xFu) | (((qh_val >> 0u) & 0x3u) << 4u)) - 32;
                int q2 = int((ql_hi & 0xFu) | (((qh_val >> 2u) & 0x3u) << 4u)) - 32;
                int q3 = int((ql_lo >> 4u) | (((qh_val >> 4u) & 0x3u) << 4u)) - 32;
                int q4 = int((ql_hi >> 4u) | (((qh_val >> 6u) & 0x3u) << 4u)) - 32;

                float s1 = float(read_i8(weights, sc_off + sc_base + is_idx + 0u));
                float s2 = float(read_i8(weights, sc_off + sc_base + is_idx + 2u));
                float s3 = float(read_i8(weights, sc_off + sc_base + is_idx + 4u));
                float s4 = float(read_i8(weights, sc_off + sc_base + is_idx + 6u));

                uint c1 = out_base + l;
                uint c2 = out_base + l + 32u;
                uint c3 = out_base + l + 64u;
                uint c4 = out_base + l + 96u;

                if (c1 < params.n_cols) sum += d * s1 * float(q1) * input[c1];
                if (c2 < params.n_cols) sum += d * s2 * float(q2) * input[c2];
                if (c3 < params.n_cols) sum += d * s3 * float(q3) * input[c3];
                if (c4 < params.n_cols) sum += d * s4 * float(q4) * input[c4];
            }
        }
    }

    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[row] = partial_sums[0];
}

// ── Dense f32 GEMV: out[row] = A[row,:] · b[:] ──────────────────────────

struct MatmulParams {
    uint m;
    uint n;
    uint k;
};

kernel void matmul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    constexpr uint tpg = 256;
    threadgroup float partial_sums[tpg];

    if (row >= params.m) return;

    float sum = 0.0f;
    // GEMV path (n=1 or general): C[row,j] = sum_p A[row,p] * B[p,j]
    // For single-token decode n=1, this reduces to dot product
    for (uint j = 0; j < params.n; j++) {
        float dot = 0.0f;
        for (uint p = tid; p < params.k; p += tpg) {
            dot += a[row * params.k + p] * b[p * params.n + j];
        }
        partial_sums[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tpg / 2; s > 0; s >>= 1) {
            if (tid < s) partial_sums[tid] += partial_sums[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) out[row * params.n + j] = partial_sums[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
