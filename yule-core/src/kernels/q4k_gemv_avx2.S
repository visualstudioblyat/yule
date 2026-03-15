# yule_q4k_gemv_avx2.S — Hand-tuned Q4_K fused dequant+dot product kernel
#
# float yule_q4k_gemv_avx2(const uint8_t *weights, const float *activations)
#
# Computes: dot(dequantize_q4k(weights), activations) for one 144-byte super-block.
#
# System V AMD64 ABI:
#   rdi = weights pointer (144-byte Q4_K super-block)
#   rsi = activations pointer (256 × f32)
#   return: xmm0 = f32 dot product
#
# Q4_K super-block layout (144 bytes):
#   [0:2)    d      — f16 super-block scale
#   [2:4)    dmin   — f16 super-block minimum
#   [4:16)   scales — 12 bytes packed 6-bit sub-block scales + mins
#   [16:144) qs     — 128 bytes of packed 4-bit values (2 per byte)
#
# Processing: 4 groups of 64 weights. Each group = 32 bytes of qs.
#   Sub-block 2j:   low nibbles  of qs[32j..32j+32] → weights[64j..64j+32]
#   Sub-block 2j+1: high nibbles of qs[32j..32j+32] → weights[64j+32..64j+64]
#
# Fused dot per sub-block:
#   sum += (d * sc[k]) * dot(q4_values, act) - (dmin * mn[k]) * sum(act)
#
# Register allocation at peak (sub-block inner loop):
#   ymm0:  nibble mask constant (0x0F × 32)
#   ymm1:  scratch / widening
#   ymm2:  loaded packed quant bytes (16 bytes in xmm2)
#   ymm3:  scratch / widening
#   ymm4:  scratch / widening
#   ymm5:  scratch / widening
#   ymm6:  dot accumulator for current sub-block
#   ymm7:  activation-sum accumulator for current sub-block
#   ymm8-ymm11: activations (4 × 8 floats = 32 floats)
#   ymm12: broadcasted scale (d * sc[k])
#   ymm13: broadcasted min (dmin * mn[k])
#   ymm14: d (broadcasted)
#   ymm15: dmin (broadcasted)
#
# Callee-saved GPRs used: rbx (for scale extraction scratch)
# Stack frame: 64 bytes for extracted scales/mins (8+8 bytes as f32 = 64 bytes)

#if defined(__APPLE__)
#define CSYM(name) _##name
#else
#define CSYM(name) name
#endif

#if defined(_WIN32)
.section .rdata,"dr"
#else
.section .rodata
#endif
.align 32
.Lmask_0f:
    .byte 0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F
    .byte 0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F
    .byte 0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F
    .byte 0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F

.text
.globl CSYM(yule_q4k_gemv_avx2)
#if !defined(__APPLE__) && !defined(_WIN32)
.type yule_q4k_gemv_avx2, @function
#endif

CSYM(yule_q4k_gemv_avx2):
    .cfi_startproc

    # ── Prologue ──────────────────────────────────────────────────────
    pushq   %rbp
    .cfi_adjust_cfa_offset 8
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register %rbp

    pushq   %rbx
    .cfi_offset %rbx, -24

#if defined(_WIN32)
    # Windows x64 ABI: rcx=weights, rdx=activations
    # Normalize to SysV registers so the rest of the kernel is platform-agnostic
    pushq   %rdi
    pushq   %rsi
    movq    %rcx, %rdi                      # rdi = weights
    movq    %rdx, %rsi                      # rsi = activations
#endif

    # Reserve 64 bytes on stack for 16 f32 scale/min values (8 scales + 8 mins)
    subq    $64, %rsp

    # ── Load constants ────────────────────────────────────────────────
    vmovdqa .Lmask_0f(%rip), %ymm0         # ymm0 = [0x0F × 32] — nibble mask, constant for entire kernel

    # ── Load d and dmin ───────────────────────────────────────────────
    # d is f16 at [rdi+0], dmin is f16 at [rdi+2]
    # vcvtph2ps converts up to 4 f16 → 4 f32. We load 4 bytes = 2 f16 values.
    vmovd   (%rdi), %xmm1                  # xmm1 = [d_f16, dmin_f16] as raw bits in low 32 bits
    # Actually need to load as two separate f16 values for vcvtph2ps
    vpinsrw $0, (%rdi), %xmm14, %xmm14    # xmm14[15:0] = d (f16)
    vpinsrw $1, 2(%rdi), %xmm14, %xmm14   # xmm14[31:16] = dmin (f16)
    vcvtph2ps %xmm14, %xmm14              # xmm14 = [d_f32, dmin_f32, 0, 0]
    vbroadcastss %xmm14, %ymm14           # ymm14 = [d, d, d, d, d, d, d, d]
    vpshufd $0x55, %xmm14, %xmm1          # xmm1 = [dmin, dmin, dmin, dmin]
    # Wait — xmm14 was overwritten by vbroadcastss. Need to extract dmin first.
    # Let me redo this more carefully:
    vpinsrw $0, (%rdi), %xmm1, %xmm1      # xmm1[15:0] = d (f16)
    vpinsrw $1, 2(%rdi), %xmm1, %xmm1     # xmm1[31:16] = dmin (f16)
    vcvtph2ps %xmm1, %xmm1                # xmm1 = [d_f32, dmin_f32, 0, 0]
    vshufps $0x00, %xmm1, %xmm1, %xmm14   # xmm14 = [d, d, d, d]
    vshufps $0x55, %xmm1, %xmm1, %xmm15   # xmm15 = [dmin, dmin, dmin, dmin]
    vinsertf128 $1, %xmm14, %ymm14, %ymm14 # ymm14 = [d × 8]
    vinsertf128 $1, %xmm15, %ymm15, %ymm15 # ymm15 = [dmin × 8]

    # ── Extract all 8 scales and 8 mins from scales[12] at [rdi+4] ───
    # Scale 0: byte[rdi+4] & 0x3F
    movzbl  4(%rdi), %eax
    andl    $0x3F, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1          # xmm1 = (float)sc0
    vmulss  %xmm14, %xmm1, %xmm1          # xmm1 = d * sc0
    vmovss  %xmm1, (%rsp)                  # stack[0] = d * sc0

    # Scale 1: byte[rdi+5] & 0x3F
    movzbl  5(%rdi), %eax
    andl    $0x3F, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm14, %xmm1, %xmm1
    vmovss  %xmm1, 4(%rsp)                 # stack[4] = d * sc1

    # Scale 2: byte[rdi+6] & 0x3F
    movzbl  6(%rdi), %eax
    andl    $0x3F, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm14, %xmm1, %xmm1
    vmovss  %xmm1, 8(%rsp)                 # stack[8] = d * sc2

    # Scale 3: byte[rdi+7] & 0x3F
    movzbl  7(%rdi), %eax
    andl    $0x3F, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm14, %xmm1, %xmm1
    vmovss  %xmm1, 12(%rsp)                # stack[12] = d * sc3

    # Scale 4: (byte[rdi+12] & 0xF) | ((byte[rdi+4] >> 6) << 4)
    movzbl  12(%rdi), %eax
    andl    $0x0F, %eax                     # low 4 bits from scales[8]
    movzbl  4(%rdi), %ebx
    shrl    $6, %ebx                        # high 2 bits from scales[0]
    shll    $4, %ebx
    orl     %ebx, %eax                      # eax = complete 6-bit scale 4
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm14, %xmm1, %xmm1
    vmovss  %xmm1, 16(%rsp)                # stack[16] = d * sc4

    # Scale 5: (byte[rdi+13] & 0xF) | ((byte[rdi+5] >> 6) << 4)
    movzbl  13(%rdi), %eax
    andl    $0x0F, %eax
    movzbl  5(%rdi), %ebx
    shrl    $6, %ebx
    shll    $4, %ebx
    orl     %ebx, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm14, %xmm1, %xmm1
    vmovss  %xmm1, 20(%rsp)                # stack[20] = d * sc5

    # Scale 6: (byte[rdi+14] & 0xF) | ((byte[rdi+6] >> 6) << 4)
    movzbl  14(%rdi), %eax
    andl    $0x0F, %eax
    movzbl  6(%rdi), %ebx
    shrl    $6, %ebx
    shll    $4, %ebx
    orl     %ebx, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm14, %xmm1, %xmm1
    vmovss  %xmm1, 24(%rsp)                # stack[24] = d * sc6

    # Scale 7: (byte[rdi+15] & 0xF) | ((byte[rdi+7] >> 6) << 4)
    movzbl  15(%rdi), %eax
    andl    $0x0F, %eax
    movzbl  7(%rdi), %ebx
    shrl    $6, %ebx
    shll    $4, %ebx
    orl     %ebx, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm14, %xmm1, %xmm1
    vmovss  %xmm1, 28(%rsp)                # stack[28] = d * sc7

    # Min 0: byte[rdi+8] & 0x3F
    movzbl  8(%rdi), %eax
    andl    $0x3F, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm15, %xmm1, %xmm1          # xmm1 = dmin * mn0
    vmovss  %xmm1, 32(%rsp)                # stack[32] = dmin * mn0

    # Min 1: byte[rdi+9] & 0x3F
    movzbl  9(%rdi), %eax
    andl    $0x3F, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm15, %xmm1, %xmm1
    vmovss  %xmm1, 36(%rsp)                # stack[36] = dmin * mn1

    # Min 2: byte[rdi+10] & 0x3F
    movzbl  10(%rdi), %eax
    andl    $0x3F, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm15, %xmm1, %xmm1
    vmovss  %xmm1, 40(%rsp)                # stack[40] = dmin * mn2

    # Min 3: byte[rdi+11] & 0x3F
    movzbl  11(%rdi), %eax
    andl    $0x3F, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm15, %xmm1, %xmm1
    vmovss  %xmm1, 44(%rsp)                # stack[44] = dmin * mn3

    # Min 4: (byte[rdi+12] >> 4) | ((byte[rdi+8] >> 6) << 4)
    movzbl  12(%rdi), %eax
    shrl    $4, %eax                        # high nibble of scales[8]
    movzbl  8(%rdi), %ebx
    shrl    $6, %ebx
    shll    $4, %ebx
    orl     %ebx, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm15, %xmm1, %xmm1
    vmovss  %xmm1, 48(%rsp)                # stack[48] = dmin * mn4

    # Min 5: (byte[rdi+13] >> 4) | ((byte[rdi+9] >> 6) << 4)
    movzbl  13(%rdi), %eax
    shrl    $4, %eax
    movzbl  9(%rdi), %ebx
    shrl    $6, %ebx
    shll    $4, %ebx
    orl     %ebx, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm15, %xmm1, %xmm1
    vmovss  %xmm1, 52(%rsp)                # stack[52] = dmin * mn5

    # Min 6: (byte[rdi+14] >> 4) | ((byte[rdi+10] >> 6) << 4)
    movzbl  14(%rdi), %eax
    shrl    $4, %eax
    movzbl  10(%rdi), %ebx
    shrl    $6, %ebx
    shll    $4, %ebx
    orl     %ebx, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm15, %xmm1, %xmm1
    vmovss  %xmm1, 56(%rsp)                # stack[56] = dmin * mn6

    # Min 7: (byte[rdi+15] >> 4) | ((byte[rdi+11] >> 6) << 4)
    movzbl  15(%rdi), %eax
    shrl    $4, %eax
    movzbl  11(%rdi), %ebx
    shrl    $6, %ebx
    shll    $4, %ebx
    orl     %ebx, %eax
    vcvtsi2ss %eax, %xmm1, %xmm1
    vmulss  %xmm15, %xmm1, %xmm1
    vmovss  %xmm1, 60(%rsp)                # stack[60] = dmin * mn7

    # Stack layout now:
    #   [rsp+0..31]  = 8 × f32: d*sc[0..7]  (precomputed sub-block scales)
    #   [rsp+32..63] = 8 × f32: dmin*mn[0..7] (precomputed sub-block mins)

    # ── Initialize total accumulator ──────────────────────────────────
    vxorps  %xmm6, %xmm6, %xmm6           # xmm6 = running total (scalar f32)

    # ═══════════════════════════════════════════════════════════════════
    # GROUP 0: qs[16..48], sub-blocks 0 (low nibbles) and 1 (high nibbles)
    # ═══════════════════════════════════════════════════════════════════

    # ── Sub-block 0: low nibbles → weights[0..31], activations[0..31] ─
    vbroadcastss (%rsp), %ymm12            # ymm12 = d * sc[0] (scale for sub-block 0)
    vbroadcastss 32(%rsp), %ymm13          # ymm13 = dmin * mn[0] (min for sub-block 0)

    # Load 32 packed quant bytes from qs offset = rdi+16
    vmovdqu 16(%rdi), %ymm2                # ymm2 = 32 bytes of packed nibbles

    # Extract low nibbles: q_lo[i] = qs[i] & 0x0F
    vpand   %ymm0, %ymm2, %ymm3            # ymm3 = low nibbles as uint8 × 32

    # Widen low 8 bytes: uint8 → int32 → f32
    vpmovzxbd %xmm3, %ymm4                 # ymm4 = [q0, q1, q2, q3, q4, q5, q6, q7] as int32
    vcvtdq2ps %ymm4, %ymm4                 # ymm4 = same as f32

    # Load activations[0..7] and compute: dot += scale*q*act, asum += act
    vmovups (%rsi), %ymm8                   # ymm8 = act[0..7]
    vmulps  %ymm12, %ymm4, %ymm4           # ymm4 = scale * q[0..7]
    vfmsub231ps %ymm8, %ymm4, %ymm4        # Hmm, wrong approach. Let me use the formula directly.
    # Formula: sum += scale * dot(q, act) - min * sum(act)
    # Better: accumulate dot(q, act) and sum(act) separately, then combine at the end of sub-block.

    # Reset sub-block accumulators
    vxorps  %ymm4, %ymm4, %ymm4            # ymm4 = dot accumulator
    vxorps  %ymm5, %ymm5, %ymm5            # ymm5 = activation sum accumulator

    # Re-extract low nibbles
    vpand   %ymm0, %ymm2, %ymm3            # ymm3 = low nibbles as uint8 × 32

    # ── Chunk 0: bytes 0-7 → 8 weights, activations[0..7] ──
    vpmovzxbd %xmm3, %ymm1                 # ymm1 = q[0..7] as int32
    vcvtdq2ps %ymm1, %ymm1                 # ymm1 = q[0..7] as f32
    vmovups (%rsi), %ymm8                   # ymm8 = act[0..7]
    vfmadd231ps %ymm1, %ymm8, %ymm4        # dot_acc += q[0..7] * act[0..7]
    vaddps  %ymm8, %ymm5, %ymm5            # sum_acc += act[0..7]

    # ── Chunk 1: bytes 8-15 → 8 weights, activations[8..15] ──
    vextracti128 $1, %ymm3, %xmm1          # xmm1 = upper 16 bytes of ymm3 (nibbles 16..31)
    # Wait — ymm3 has 32 bytes of low nibbles. Bytes 0-15 are the first 16 nibbles.
    # We need bytes 8-15 for the next 8 values.
    # vpmovzxbd only reads 8 bytes from xmm source → produces 8 int32 in ymm.
    # For bytes 8-15, we need to shift xmm3 right by 8 bytes first.
    vpsrldq $8, %xmm3, %xmm1              # xmm1 = bytes 8-15 of low nibbles
    vpmovzxbd %xmm1, %ymm1                 # ymm1 = q[8..15] as int32
    vcvtdq2ps %ymm1, %ymm1                 # ymm1 = q[8..15] as f32
    vmovups 32(%rsi), %ymm8                 # ymm8 = act[8..15]
    vfmadd231ps %ymm1, %ymm8, %ymm4        # dot_acc += q[8..15] * act[8..15]
    vaddps  %ymm8, %ymm5, %ymm5            # sum_acc += act[8..15]

    # ── Chunk 2: bytes 16-23 → 8 weights, activations[16..23] ──
    vextracti128 $1, %ymm3, %xmm1          # xmm1 = bytes 16-31 of ymm3
    vpmovzxbd %xmm1, %ymm1                 # ymm1 = q[16..23] as int32
    vcvtdq2ps %ymm1, %ymm1                 # ymm1 = q[16..23] as f32
    vmovups 64(%rsi), %ymm8                 # ymm8 = act[16..23]
    vfmadd231ps %ymm1, %ymm8, %ymm4        # dot_acc += q[16..23] * act[16..23]
    vaddps  %ymm8, %ymm5, %ymm5            # sum_acc += act[16..23]

    # ── Chunk 3: bytes 24-31 → 8 weights, activations[24..31] ──
    vextracti128 $1, %ymm3, %xmm1
    vpsrldq $8, %xmm1, %xmm1              # xmm1 = bytes 24-31
    vpmovzxbd %xmm1, %ymm1                 # ymm1 = q[24..31] as int32
    vcvtdq2ps %ymm1, %ymm1                 # ymm1 = q[24..31] as f32
    vmovups 96(%rsi), %ymm8                 # ymm8 = act[24..31]
    vfmadd231ps %ymm1, %ymm8, %ymm4        # dot_acc += q[24..31] * act[24..31]
    vaddps  %ymm8, %ymm5, %ymm5            # sum_acc += act[24..31]

    # ── Combine sub-block 0 result ────────────────────────────────────
    # Horizontal sum of dot_acc (ymm4) and sum_acc (ymm5)
    vextractf128 $1, %ymm4, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4           # xmm4 = 4 partial sums
    vmovhlps %xmm4, %xmm1, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vshufps $0x55, %xmm4, %xmm4, %xmm1
    vaddss  %xmm1, %xmm4, %xmm4           # xmm4 = scalar dot(q, act)

    vextractf128 $1, %ymm5, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vmovhlps %xmm5, %xmm1, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vshufps $0x55, %xmm5, %xmm5, %xmm1
    vaddss  %xmm1, %xmm5, %xmm5           # xmm5 = scalar sum(act)

    # sum += scale * dot - min * asum
    vmulss  %xmm12, %xmm4, %xmm4          # xmm4 = scale * dot
    vmulss  %xmm13, %xmm5, %xmm5          # xmm5 = min * asum
    vsubss  %xmm5, %xmm4, %xmm4           # xmm4 = scale*dot - min*asum
    vaddss  %xmm4, %xmm6, %xmm6           # total += sub-block 0 contribution

    # ── Sub-block 1: high nibbles → weights[32..63], activations[32..63]
    vbroadcastss 4(%rsp), %ymm12           # ymm12 = d * sc[1]
    vbroadcastss 36(%rsp), %ymm13          # ymm13 = dmin * mn[1]

    # High nibbles: q_hi[i] = qs[i] >> 4
    vpsrlw  $4, %ymm2, %ymm3               # shift right by 4 (operates on 16-bit words)
    vpand   %ymm0, %ymm3, %ymm3            # mask to keep only low 4 bits = high nibbles

    vxorps  %ymm4, %ymm4, %ymm4            # dot_acc = 0
    vxorps  %ymm5, %ymm5, %ymm5            # sum_acc = 0

    # Chunk 0: q_hi[0..7], act[32..39]
    vpmovzxbd %xmm3, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 128(%rsi), %ymm8               # act[32..39] = rsi + 32*4
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    # Chunk 1: q_hi[8..15], act[40..47]
    vpsrldq $8, %xmm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 160(%rsi), %ymm8               # act[40..47]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    # Chunk 2: q_hi[16..23], act[48..55]
    vextracti128 $1, %ymm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 192(%rsi), %ymm8               # act[48..55]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    # Chunk 3: q_hi[24..31], act[56..63]
    vextracti128 $1, %ymm3, %xmm1
    vpsrldq $8, %xmm1, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 224(%rsi), %ymm8               # act[56..63]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    # Horizontal reduce and accumulate sub-block 1
    vextractf128 $1, %ymm4, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vmovhlps %xmm4, %xmm1, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vshufps $0x55, %xmm4, %xmm4, %xmm1
    vaddss  %xmm1, %xmm4, %xmm4

    vextractf128 $1, %ymm5, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vmovhlps %xmm5, %xmm1, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vshufps $0x55, %xmm5, %xmm5, %xmm1
    vaddss  %xmm1, %xmm5, %xmm5

    vmulss  %xmm12, %xmm4, %xmm4
    vmulss  %xmm13, %xmm5, %xmm5
    vsubss  %xmm5, %xmm4, %xmm4
    vaddss  %xmm4, %xmm6, %xmm6           # total += sub-block 1

    # ═══════════════════════════════════════════════════════════════════
    # GROUP 1: qs[48..80], sub-blocks 2 and 3
    # ═══════════════════════════════════════════════════════════════════

    vmovdqu 48(%rdi), %ymm2                # Load 32 bytes of packed nibbles (group 1)

    # ── Sub-block 2: low nibbles, act[64..95] ─────────────────────────
    vbroadcastss 8(%rsp), %ymm12           # d * sc[2]
    vbroadcastss 40(%rsp), %ymm13          # dmin * mn[2]
    vpand   %ymm0, %ymm2, %ymm3            # low nibbles

    vxorps  %ymm4, %ymm4, %ymm4
    vxorps  %ymm5, %ymm5, %ymm5

    vpmovzxbd %xmm3, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 256(%rsi), %ymm8               # act[64..71]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vpsrldq $8, %xmm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 288(%rsi), %ymm8               # act[72..79]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 320(%rsi), %ymm8               # act[80..87]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpsrldq $8, %xmm1, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 352(%rsi), %ymm8               # act[88..95]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    # Reduce sub-block 2
    vextractf128 $1, %ymm4, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vmovhlps %xmm4, %xmm1, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vshufps $0x55, %xmm4, %xmm4, %xmm1
    vaddss  %xmm1, %xmm4, %xmm4

    vextractf128 $1, %ymm5, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vmovhlps %xmm5, %xmm1, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vshufps $0x55, %xmm5, %xmm5, %xmm1
    vaddss  %xmm1, %xmm5, %xmm5

    vmulss  %xmm12, %xmm4, %xmm4
    vmulss  %xmm13, %xmm5, %xmm5
    vsubss  %xmm5, %xmm4, %xmm4
    vaddss  %xmm4, %xmm6, %xmm6

    # ── Sub-block 3: high nibbles, act[96..127] ──────────────────────
    vbroadcastss 12(%rsp), %ymm12          # d * sc[3]
    vbroadcastss 44(%rsp), %ymm13          # dmin * mn[3]
    vpsrlw  $4, %ymm2, %ymm3
    vpand   %ymm0, %ymm3, %ymm3

    vxorps  %ymm4, %ymm4, %ymm4
    vxorps  %ymm5, %ymm5, %ymm5

    vpmovzxbd %xmm3, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 384(%rsi), %ymm8               # act[96..103]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vpsrldq $8, %xmm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 416(%rsi), %ymm8               # act[104..111]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 448(%rsi), %ymm8               # act[112..119]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpsrldq $8, %xmm1, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 480(%rsi), %ymm8               # act[120..127]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    # Reduce sub-block 3
    vextractf128 $1, %ymm4, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vmovhlps %xmm4, %xmm1, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vshufps $0x55, %xmm4, %xmm4, %xmm1
    vaddss  %xmm1, %xmm4, %xmm4

    vextractf128 $1, %ymm5, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vmovhlps %xmm5, %xmm1, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vshufps $0x55, %xmm5, %xmm5, %xmm1
    vaddss  %xmm1, %xmm5, %xmm5

    vmulss  %xmm12, %xmm4, %xmm4
    vmulss  %xmm13, %xmm5, %xmm5
    vsubss  %xmm5, %xmm4, %xmm4
    vaddss  %xmm4, %xmm6, %xmm6

    # ═══════════════════════════════════════════════════════════════════
    # GROUP 2: qs[80..112], sub-blocks 4 and 5
    # ═══════════════════════════════════════════════════════════════════

    vmovdqu 80(%rdi), %ymm2                # Load 32 bytes (group 2)

    # ── Sub-block 4: low nibbles, act[128..159] ──────────────────────
    vbroadcastss 16(%rsp), %ymm12          # d * sc[4]
    vbroadcastss 48(%rsp), %ymm13          # dmin * mn[4]
    vpand   %ymm0, %ymm2, %ymm3

    vxorps  %ymm4, %ymm4, %ymm4
    vxorps  %ymm5, %ymm5, %ymm5

    vpmovzxbd %xmm3, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 512(%rsi), %ymm8               # act[128..135]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vpsrldq $8, %xmm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 544(%rsi), %ymm8               # act[136..143]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 576(%rsi), %ymm8               # act[144..151]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpsrldq $8, %xmm1, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 608(%rsi), %ymm8               # act[152..159]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextractf128 $1, %ymm4, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vmovhlps %xmm4, %xmm1, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vshufps $0x55, %xmm4, %xmm4, %xmm1
    vaddss  %xmm1, %xmm4, %xmm4

    vextractf128 $1, %ymm5, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vmovhlps %xmm5, %xmm1, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vshufps $0x55, %xmm5, %xmm5, %xmm1
    vaddss  %xmm1, %xmm5, %xmm5

    vmulss  %xmm12, %xmm4, %xmm4
    vmulss  %xmm13, %xmm5, %xmm5
    vsubss  %xmm5, %xmm4, %xmm4
    vaddss  %xmm4, %xmm6, %xmm6

    # ── Sub-block 5: high nibbles, act[160..191] ─────────────────────
    vbroadcastss 20(%rsp), %ymm12          # d * sc[5]
    vbroadcastss 52(%rsp), %ymm13          # dmin * mn[5]
    vpsrlw  $4, %ymm2, %ymm3
    vpand   %ymm0, %ymm3, %ymm3

    vxorps  %ymm4, %ymm4, %ymm4
    vxorps  %ymm5, %ymm5, %ymm5

    vpmovzxbd %xmm3, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 640(%rsi), %ymm8               # act[160..167]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vpsrldq $8, %xmm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 672(%rsi), %ymm8               # act[168..175]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 704(%rsi), %ymm8               # act[176..183]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpsrldq $8, %xmm1, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 736(%rsi), %ymm8               # act[184..191]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextractf128 $1, %ymm4, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vmovhlps %xmm4, %xmm1, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vshufps $0x55, %xmm4, %xmm4, %xmm1
    vaddss  %xmm1, %xmm4, %xmm4

    vextractf128 $1, %ymm5, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vmovhlps %xmm5, %xmm1, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vshufps $0x55, %xmm5, %xmm5, %xmm1
    vaddss  %xmm1, %xmm5, %xmm5

    vmulss  %xmm12, %xmm4, %xmm4
    vmulss  %xmm13, %xmm5, %xmm5
    vsubss  %xmm5, %xmm4, %xmm4
    vaddss  %xmm4, %xmm6, %xmm6

    # ═══════════════════════════════════════════════════════════════════
    # GROUP 3: qs[112..144], sub-blocks 6 and 7
    # ═══════════════════════════════════════════════════════════════════

    vmovdqu 112(%rdi), %ymm2               # Load 32 bytes (group 3)

    # ── Sub-block 6: low nibbles, act[192..223] ──────────────────────
    vbroadcastss 24(%rsp), %ymm12          # d * sc[6]
    vbroadcastss 56(%rsp), %ymm13          # dmin * mn[6]
    vpand   %ymm0, %ymm2, %ymm3

    vxorps  %ymm4, %ymm4, %ymm4
    vxorps  %ymm5, %ymm5, %ymm5

    vpmovzxbd %xmm3, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 768(%rsi), %ymm8               # act[192..199]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vpsrldq $8, %xmm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 800(%rsi), %ymm8               # act[200..207]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 832(%rsi), %ymm8               # act[208..215]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpsrldq $8, %xmm1, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 864(%rsi), %ymm8               # act[216..223]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextractf128 $1, %ymm4, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vmovhlps %xmm4, %xmm1, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vshufps $0x55, %xmm4, %xmm4, %xmm1
    vaddss  %xmm1, %xmm4, %xmm4

    vextractf128 $1, %ymm5, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vmovhlps %xmm5, %xmm1, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vshufps $0x55, %xmm5, %xmm5, %xmm1
    vaddss  %xmm1, %xmm5, %xmm5

    vmulss  %xmm12, %xmm4, %xmm4
    vmulss  %xmm13, %xmm5, %xmm5
    vsubss  %xmm5, %xmm4, %xmm4
    vaddss  %xmm4, %xmm6, %xmm6

    # ── Sub-block 7: high nibbles, act[224..255] ─────────────────────
    vbroadcastss 28(%rsp), %ymm12          # d * sc[7]
    vbroadcastss 60(%rsp), %ymm13          # dmin * mn[7]
    vpsrlw  $4, %ymm2, %ymm3
    vpand   %ymm0, %ymm3, %ymm3

    vxorps  %ymm4, %ymm4, %ymm4
    vxorps  %ymm5, %ymm5, %ymm5

    vpmovzxbd %xmm3, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 896(%rsi), %ymm8               # act[224..231]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vpsrldq $8, %xmm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 928(%rsi), %ymm8               # act[232..239]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 960(%rsi), %ymm8               # act[240..247]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextracti128 $1, %ymm3, %xmm1
    vpsrldq $8, %xmm1, %xmm1
    vpmovzxbd %xmm1, %ymm1
    vcvtdq2ps %ymm1, %ymm1
    vmovups 992(%rsi), %ymm8               # act[248..255]
    vfmadd231ps %ymm1, %ymm8, %ymm4
    vaddps  %ymm8, %ymm5, %ymm5

    vextractf128 $1, %ymm4, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vmovhlps %xmm4, %xmm1, %xmm1
    vaddps  %xmm1, %xmm4, %xmm4
    vshufps $0x55, %xmm4, %xmm4, %xmm1
    vaddss  %xmm1, %xmm4, %xmm4

    vextractf128 $1, %ymm5, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vmovhlps %xmm5, %xmm1, %xmm1
    vaddps  %xmm1, %xmm5, %xmm5
    vshufps $0x55, %xmm5, %xmm5, %xmm1
    vaddss  %xmm1, %xmm5, %xmm5

    vmulss  %xmm12, %xmm4, %xmm4
    vmulss  %xmm13, %xmm5, %xmm5
    vsubss  %xmm5, %xmm4, %xmm4
    vaddss  %xmm4, %xmm6, %xmm6

    # ── Return result ─────────────────────────────────────────────────
    vmovaps %xmm6, %xmm0                   # return value in xmm0

    # ── Epilogue ──────────────────────────────────────────────────────
    addq    $64, %rsp                       # free stack frame
    vzeroupper                              # clear upper YMM to avoid SSE transition penalty
#if defined(_WIN32)
    popq    %rsi
    popq    %rdi
#endif
    popq    %rbx
    popq    %rbp
    .cfi_def_cfa %rsp, 8
    ret

    .cfi_endproc

#if !defined(__APPLE__) && !defined(_WIN32)
.size yule_q4k_gemv_avx2, .-yule_q4k_gemv_avx2
#endif
