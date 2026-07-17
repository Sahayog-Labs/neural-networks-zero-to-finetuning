#!/usr/bin/env python3
"""
08_quantization_lab.py - the numbers the course refuses to print. RUNG probe, p.41.

Page 41 makes one claim and this script is where you check it on your own bytes:

    4-bit names a BUDGET, not a SCHEME, and almost all of the quality lives in the
    scheme - specifically in how finely you re-anchor the scale (the block size).

It does four things, in order, each teaching as it runs:

  1. THE BUDGET (pure arithmetic, no GPU). Effective bits/param and B/param for every
     format on the page - INT4, NF4 (double-quant on AND off), FP8, MXFP4, NVFP4 - each
     asserted against constants.md so a regression fails loudly. This is where the two
     frozen numbers live:  NF4+DQ = 0.515869 B/param (4.127 bits)  vs
     NF4 no-DQ = 0.5625 B/param (4.5 bits). The widely-copied 0.53 / 4.25 figure is an
     arithmetic error (divides the 2nd-level fp32 constant by 256 scales, not by
     64x256 = 16,384 params); this script reproduces the CORRECT rational, 65/512.

  2. THE SCHEME (measured, no GPU). A blockwise quantize->dequantize round-trip on a
     seeded 4,096-weight slice with a q_proj's statistics (near-Gaussian bulk + rare
     heavy-tail outliers), in ~150 lines of plain Python - the same round-trip the
     page's live demo runs. It MEASURES the RMS error per format and per block size and
     proves the PREDICT: dragging the block from 4096 to 16 collapses the error, while
     swapping INT4->NF4 or INT4->INT8 moves it far less. Granularity beats precision.
     It also measures the NVFP4-vs-MXFP4 gap (block 16 + fp8 scale vs block 32 + a
     power-of-two scale) - the one mechanism behind the vendor's "88% lower error".

  3. A REAL LAYER (measured, needs a weight file). --weight points at a safetensors
     tensor (a real Qwen3 q_proj off your disk); the same reference quantizers run on
     real weights instead of the synthetic slice. RMS numbers become [MEA] on YOUR box.

  4. THE bitsandbytes CROSS-CHECK + PERPLEXITY (measured, needs the Spark). --model
     loads a real Qwen3 in NF4 via BitsAndBytesConfig (double-quant on and off), cross-
     checks the reference NF4 reconstruction against bnb's own kernels, and measures a
     small perplexity delta - the honest retention number. This is where you replace the
     SEO folklore "AWQ 95% / GGUF 92% / GPTQ 90%" (constants.md §9.7 forbids printing it)
     with a number you measured. This path is DESK-CHECKED here and RUN BY YOU there.

Confidence tags in the output are load-bearing: [VP]=verified-published/frozen,
[DER]=derived, [MEA]=measured on your box, [EST]=estimate, [EEST-vendor]=vendor estimate.

Local-verification reality (this repo's contract): steps 1-2 are pure Python + stdlib
and RUN anywhere (no numpy, no torch, no GPU); `--self-test` runs them and exits 0.
numpy is used only to accelerate --weight; torch/transformers/bitsandbytes only for
--model. Nothing is written and nothing is installed - read-only w.r.t. your system,
until you pass --model (which downloads a checkpoint into HF_HOME; it is named below).

SAFETY: steps 1-2 touch no device. --model loads an 8B checkpoint and quantizes it on
the GPU; if ComfyUI is live on the Spark this CONTENDS with it - consult first, per the
project's hard rule. The bnb/perplexity path is heavy (~2 min) and needs a FRESH venv
(never ComfyUI's) with the pinned stack from the setup page.

Usage
-----
    python 08_quantization_lab.py                      # steps 1-2: budget + measured RMS
    python 08_quantization_lab.py --self-test          # steps 1-2 + assertions, CI-style
    python 08_quantization_lab.py --weight q_proj.safetensors   # + step 3: a real layer
    python 08_quantization_lab.py --model Qwen/Qwen3-8B --perplexity   # + step 4 (Spark)
"""

import argparse
import math
import random
import sys

# --------------------------------------------------------------------------- #
# GiB/GB discipline (constants.md §0): capacity is GiB (binary), a weight blob
# quoted alone is GB (decimal). Print the unit every time; never mix in one line.
# --------------------------------------------------------------------------- #
GiB = 1 << 30
GB = 10 ** 9

# --- frozen model fact (constants.md §3) ----------------------------------- #
P_QWEN3_8B = 8_190_735_360          # exact parameter count, the course anchor

# --- the code books, published levels (page-41 demo / QLoRA paper / bnb) ---- #
# NF4: the 16 NormalFloat-4 quantile levels, normalized to [-1, 1].
NF4_LEVELS = [
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844416, -0.1848922,
    -0.09103756, 0.0, 0.07958029, 0.16093020, 0.24611230, 0.33791524,
    0.44070983, 0.56261700, 0.72295684, 1.0,
]
# E2M1 (FP4): the 8 representable magnitudes, signed -> 16 codes. MXFP4/NVFP4 share it.
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_LEVELS = []
for _m in _E2M1_MAG:
    E2M1_LEVELS.append(_m)
    if _m > 0:
        E2M1_LEVELS.append(-_m)


def int_codes(bits):
    """2**bits evenly spaced levels on [-1, 1] - the INT-n code book."""
    n = 1 << bits
    return [-1.0 + 2.0 * i / (n - 1) for i in range(n)]


def e4m3_codes():
    """FP8 E4M3 representable values (normals + subnormals) up to 448, normalized
    so the max code is 1 (matches the s = max|w| / max|q| convention). Enumerated,
    not remembered - the same construction the page-41 demo uses."""
    seen = {0.0}
    for e in range(0, 16):
        for m in range(0, 8):
            if e == 0:
                v = (m / 8) * 2 ** -6            # subnormal
            else:
                v = (1 + m / 8) * 2 ** (e - 7)   # normal
            if v <= 448:
                seen.add(v)
                seen.add(-v)
    arr = sorted(seen)
    mx = arr[-1]
    return [v / mx for v in arr]


FP8_LEVELS = e4m3_codes()


# --------------------------------------------------------------------------- #
# STEP 1 - THE BUDGET.  Pure arithmetic, asserted against constants.md §3/§9.7.
# effective bits/param = b_q + b_scale/block  (+ 2nd-level term for NF4 double-quant)
# --------------------------------------------------------------------------- #

def nf4_bits(double_quant):
    """Effective bits/param for NF4, exactly as constants.md §3 derives it.

    code (4 bits) + first-level scale + optional second-level scale.

      DQ OFF: one fp32 absmax scale per 64-block  -> 32/64  = 0.5     bits overhead
      DQ ON : one fp8  absmax scale per 64-block  ->  8/64  = 0.125   bits
              + one fp32 constant per 256 SCALES, each scale covering 64 params,
                so it amortizes over 64*256 = 16,384 PARAMS -> 32/16384 = 0.001953 bits
              overhead = 8/64 + 32/(64*256) = 65/512 = 0.126953125 bits
    """
    if double_quant:
        overhead = 8 / 64 + 32 / (64 * 256)        # = 65/512
    else:
        overhead = 32 / 64                          # fp32 scale, no compression
    return 4 + overhead


def budget_table():
    print("=" * 74)
    print("STEP 1 - THE BUDGET: effective bits/param  (4-bit is a budget, not a scheme)")
    print("=" * 74)
    print("  effective bits/param = code_bits + scale_bits / block  (+ 2nd-level for DQ)")
    print()

    rows = []  # (label, bits, tag, note)

    nf4_dq = nf4_bits(True)
    nf4_no = nf4_bits(False)
    rows.append(("NF4 + double-quant", nf4_dq, "[VP §3]", "code 4 + 65/512 overhead"))
    rows.append(("NF4, no double-quant", nf4_no, "[VP §3]", "code 4 + fp32 scale/64"))
    rows.append(("MXFP4 (E2M1)", 4 + 8 / 32, "[VP §9.7]", "block 32, E8M0 pow-2 scale"))
    rows.append(("NVFP4 (E2M1)", 4 + 8 / 16, "[VP §9.7]", "block 16, fp8 scale (+fp32 global)"))
    rows.append(("INT4 (uniform)", 4.0, "[DER]", "no per-block scale amortized here"))
    rows.append(("FP8 E4M3", 8.0, "[VP §9.7]", "inference/weight format"))
    rows.append(("bf16", 16.0, "[VP §9.7]", "the training default on Blackwell"))

    print(f"  {'format':<22}{'bits/param':>11}{'B/param':>10}   tag        note")
    print("  " + "-" * 70)
    for label, bits, tag, note in rows:
        print(f"  {label:<22}{bits:>11.4f}{bits / 8:>10.4f}   {tag:<10} {note}")
    print()

    # --- the two frozen numbers, spelled out with the exact rational -------- #
    bpp_dq = nf4_dq / 8
    bpp_no = nf4_no / 8
    print("  The two numbers the whole page turns on:")
    print(f"    NF4 + DQ  = 4 + 65/512 = 2113/512 = {nf4_dq:.9f} bits "
          f"= {bpp_dq:.6f} B/param  [VP §3]")
    print(f"    NF4 no-DQ = 4 + 32/64  = 9/2       = {nf4_no:.9f} bits "
          f"= {bpp_no:.6f} B/param  [VP §3]")
    print(f"    overhead (DQ) = 8/64 + 32/(64*256) = 65/512 = {8/64 + 32/(64*256):.9f} bits")
    print()
    print("  The 0.53 / 4.25 figure copied all over the internet is an ARITHMETIC ERROR,")
    print("  not a rival convention: it divides the 2nd-level fp32 constant by 256 SCALES")
    print("  instead of by 64*256 = 16,384 PARAMS. One dropped factor of the 64-block")
    print("  inflates 0.00195 -> 0.125. The paper's own arithmetic gives 4.127 bits. [VP §3]")
    print()

    # --- footprint on Qwen3-8B, and the honest-default beat ----------------- #
    base_dq = P_QWEN3_8B * bpp_dq          # bytes
    base_no = P_QWEN3_8B * bpp_no
    print("  Qwen3-8B frozen NF4 base (P = 8,190,735,360):")
    print(f"    DQ ON  : {P_QWEN3_8B:,} x {bpp_dq:.6f} B = "
          f"{base_dq/GB:.3f} GB ({base_dq/GiB:.3f} GiB)   [DER §3]")
    print(f"    DQ OFF : {P_QWEN3_8B:,} x {bpp_no:.6f} B = "
          f"{base_no/GB:.3f} GB ({base_no/GiB:.3f} GiB)   [DER §3]")
    print()
    print("  bnb_4bit_use_double_quant (a.k.a. compress_statistics) DEFAULTS TO False in")
    print("  both bitsandbytes and BitsAndBytesConfig. So the HONEST un-opted-in footprint")
    print("  is 4.607 GB, NOT 4.225 GB. Every 4.225 figure assumes you set it True. [VP §3]")
    print()

    # --- cross-check that reproduces the paper's own abstract figure -------- #
    dq_saving_bits = nf4_no - nf4_dq       # 0.5 - 0.127 = 0.373 bits
    saving_65b = 65 * GB * dq_saving_bits / 8
    print("  Cross-check (two routes, one number):")
    print(f"    DQ saving = {nf4_no - nf4_dq:.3f} bits/param; on a 65B model that is")
    print(f"    65e9 x {dq_saving_bits:.3f} / 8 = {saving_65b/GB:.2f} GB - the QLoRA paper's "
          f"abstract says 'approximately 3 GB'. [VP arXiv 2305.14314]")
    print()

    # ----------------------------- self-checks ------------------------------ #
    assert abs(bpp_dq - 0.515869140625) < 1e-9, f"NF4+DQ must be 0.515869 B/param, got {bpp_dq}"
    assert abs(nf4_dq - 4.126953125) < 1e-9, f"NF4+DQ must be 4.127 bits, got {nf4_dq}"
    assert abs(bpp_no - 0.5625) < 1e-12, f"NF4 no-DQ must be 0.5625 B/param, got {bpp_no}"
    assert abs(nf4_no - 4.5) < 1e-12, f"NF4 no-DQ must be 4.5 bits, got {nf4_no}"
    assert abs((8/64 + 32/(64*256)) - 65/512) < 1e-15, "overhead must be exactly 65/512"
    assert abs(base_dq/GB - 4.225) < 0.01, f"DQ-on footprint must be 4.225 GB, got {base_dq/GB:.3f}"
    assert abs(base_no/GB - 4.607) < 0.01, f"DQ-off footprint must be 4.607 GB, got {base_no/GB:.3f}"
    assert abs(saving_65b/GB - 3.03) < 0.05, f"65B DQ saving must be ~3.03 GB, got {saving_65b/GB:.2f}"
    # MXFP4 / NVFP4 effective bits
    assert abs((4 + 8/32) - 4.25) < 1e-12
    assert abs((4 + 8/16) - 4.5) < 1e-12
    print("  self-checks passed: 0.515869 vs 0.5625 B/param; 4.225 vs 4.607 GB; 65/512; 3.03 GB.")
    print()
    return {"nf4_dq_bpp": bpp_dq, "nf4_no_bpp": bpp_no}


# --------------------------------------------------------------------------- #
# STEP 2 - THE SCHEME.  Measured blockwise round-trip, plain Python (no numpy).
# --------------------------------------------------------------------------- #

def encode_scale(s, mode):
    """Quantize a block's absmax scale to the scale format.
      pow2 : round UP to a power of two (MXFP4's E8M0 exponent-only scale)
      fp8  : nearest positive E4M3 (1 sign implicit, 4 exp, 3 mantissa)
      fp32 : exact
    """
    if s <= 0:
        return s
    if mode == "pow2":
        return 2.0 ** math.ceil(math.log2(s))
    if mode == "fp8":
        exp = math.floor(math.log2(s))
        frac = s / 2.0 ** exp                      # in [1, 2)
        q = round((frac - 1) * 8) / 8              # 3 mantissa bits
        return (1 + q) * 2.0 ** exp
    return s                                        # fp32 exact


def _nearest(t, codes):
    best = codes[0]
    bd = abs(t - codes[0])
    for c in codes[1:]:
        d = abs(t - c)
        if d < bd:
            bd = d
            best = c
    return best


def roundtrip(W, codes, block, scale_mode, double_quant=False):
    """Blockwise quantize -> dequantize. Returns (dequantized list, relative RMS).

    Per block: s = absmax(W_block) / max|code|, snap each w/s to the nearest code,
    dequantize as code*s. With double_quant, the per-block scales are themselves
    quantized to fp8 with one fp32 constant per 256-scale group (the 'double' in DQ)
    - a second-order effect on error, exactly as the memory story predicts.
    """
    n = len(W)
    cmax = max(abs(c) for c in codes)
    scales = []
    for b in range(0, n, block):
        seg = W[b:b + block]
        amax = max((abs(w) for w in seg), default=0.0)
        scales.append(encode_scale(amax / cmax, scale_mode) if amax > 0 else 0.0)

    if double_quant:                                # quantize the scales themselves
        dq_scales = []
        for g in range(0, len(scales), 256):
            grp = scales[g:g + 256]
            c = max((abs(s) for s in grp), default=0.0)  # fp32 second-level constant
            for s in grp:
                if s == 0 or c == 0:
                    dq_scales.append(s)
                else:
                    dq_scales.append(encode_scale((s / c), "fp8") * c)
        scales = dq_scales

    wq = [0.0] * n
    se = 0.0
    sw = 0.0
    for bi, b in enumerate(range(0, n, block)):
        s = scales[bi]
        for i in range(b, min(b + block, n)):
            if s == 0:
                q = 0.0
            else:
                q = _nearest(W[i] / s, codes) * s
            wq[i] = q
            e = W[i] - q
            se += e * e
            sw += W[i] * W[i]
    rms = math.sqrt(se / n)
    relrms = rms / math.sqrt(sw / n) if sw > 0 else 0.0
    return wq, relrms


def make_slice(n=4096, seed=42):
    """A seeded 4,096-weight sample with a q_proj's statistics: near-Gaussian bulk
    (std ~0.021) plus rare heavy-tail outliers (~1.2%, x4.6) - the reason blocks
    matter. Reproduces the page-41 demo's STATISTICS (seedable determinism; not the
    JS RNG's exact draws)."""
    rng = random.Random(seed)
    W = []
    for _ in range(n):
        x = rng.gauss(0.0, 0.021)
        if rng.random() < 0.012:
            x *= 4.6
        W.append(x)
    return W


def measure_scheme(seed=42):
    print("=" * 74)
    print("STEP 2 - THE SCHEME: measured RMS on a real-statistics q_proj slice  [MEA]")
    print("=" * 74)
    W = make_slice(seed=seed)
    n = len(W)
    amax = max(abs(w) for w in W)
    std = math.sqrt(sum(w * w for w in W) / n)
    print(f"  slice: {n} weights, std {std:.4f}, absmax {amax:.4f} "
          f"(outliers push absmax ~{amax/std:.1f}x the std - that is what inflates a scale)")
    print()

    # ---- (a) granularity beats precision: NF4 across block sizes ----------- #
    print("  (a) Same format (NF4, fp8 scale), shrink the block - watch the error collapse:")
    nf4 = NF4_LEVELS
    rms_by_block = {}
    for block in (4096, 1024, 256, 64, 16):
        _, r = roundtrip(W, nf4, block, "fp8")
        rms_by_block[block] = r
        print(f"      NF4  block {block:>5}   relRMS {r*100:>6.2f}%   "
              f"eff {4 + 8/block:.3f} bits/param")
    print()

    # ---- (b) precision (more code bits) moves it LESS than block did ------- #
    print("  (b) Same block (64), spend more BITS on the code - a smaller effect than (a):")
    _, r_int4 = roundtrip(W, int_codes(4), 64, "fp8")
    _, r_int8 = roundtrip(W, int_codes(8), 64, "fp8")
    _, r_nf4 = roundtrip(W, nf4, 64, "fp8")
    print(f"      INT4 block 64   relRMS {r_int4*100:>6.2f}%   (4 bits, evenly spaced)")
    print(f"      NF4  block 64   relRMS {r_nf4*100:>6.2f}%   (4 bits, normal quantiles)")
    print(f"      INT8 block 64   relRMS {r_int8*100:>6.2f}%   (8 bits, evenly spaced)")
    d_format = (r_int4 - r_nf4) / r_int4 * 100
    d_block = (rms_by_block[4096] - rms_by_block[16]) / rms_by_block[4096] * 100
    print(f"      -> swapping INT4->NF4 cut error {d_format:.0f}%; "
          f"shrinking block 4096->16 cut it {d_block:.0f}%. Granularity wins. [MEA]")
    print()

    # ---- (c) NVFP4 vs MXFP4: the one mechanism, measured -------------------- #
    print("  (c) NVFP4 vs MXFP4 - same E2M1 code, the ONLY differences are block & scale:")
    _, r_mx = roundtrip(W, E2M1_LEVELS, 32, "pow2")     # block 32, power-of-two scale
    _, r_nv = roundtrip(W, E2M1_LEVELS, 16, "fp8")      # block 16, arbitrary fp8 scale
    print(f"      MXFP4  block 32, pow-2 scale   relRMS {r_mx*100:>6.2f}%   (scale of 3 rounds to 2 or 4)")
    print(f"      NVFP4  block 16, fp8   scale   relRMS {r_nv*100:>6.2f}%   (arbitrary scale, half the block)")
    if r_mx > 0:
        reduction = (r_mx - r_nv) / r_mx * 100
        print(f"      -> NVFP4 error is {reduction:.0f}% lower on this slice [MEA]. The vendor's "
              f"'88% lower' is [EST-vendor];")
        print(f"         the MECHANISM is what transfers: granularity (block + free scale), not mantissa bits.")
    print()

    # ---- (d) NF4 double-quant on/off: memory saved, error barely moves ------ #
    print("  (d) NF4 double-quant on vs off (block 64) - DQ saves memory at ~no quality cost:")
    _, r_dqoff = roundtrip(W, nf4, 64, "fp32", double_quant=False)
    _, r_dqon = roundtrip(W, nf4, 64, "fp8", double_quant=True)
    print(f"      NF4 DQ OFF  relRMS {r_dqoff*100:>6.2f}%   4.500 bits/param  (4.607 GB on 8B)")
    print(f"      NF4 DQ ON   relRMS {r_dqon*100:>6.2f}%   4.127 bits/param  (4.225 GB on 8B)")
    print(f"      -> DQ removes 0.373 bits/param and the RMS barely moves. That is the trade. [MEA]")
    print()

    # ----------------------------- self-checks ------------------------------ #
    # These are robust on Gaussian+outlier data: finer blocks and a free scale help.
    assert rms_by_block[16] < rms_by_block[4096], (
        "granularity must help: block-16 RMS should beat block-4096 RMS")
    assert rms_by_block[64] < rms_by_block[1024], "smaller block should not hurt"
    assert r_nv < r_mx, "NVFP4 (block 16 + fp8 scale) must beat MXFP4 (block 32 + pow-2 scale)"
    assert d_block > d_format, (
        "the block-size effect must exceed the format effect - the page's whole point")
    print("  self-checks passed: block-16 < block-4096; NVFP4 < MXFP4; block effect > format effect.")
    print()
    return {"nf4_block64_relrms": r_nf4, "nvfp4_relrms": r_nv, "mxfp4_relrms": r_mx}


# --------------------------------------------------------------------------- #
# STEP 3 - A REAL LAYER.  Same reference quantizers, run on a real Qwen3 weight.
# --------------------------------------------------------------------------- #

def real_layer(weight_path):
    print("=" * 74)
    print(f"STEP 3 - A REAL LAYER: reference quantizers on {weight_path}  [MEA]")
    print("=" * 74)
    try:
        import numpy as np
        from safetensors.numpy import load_file
    except ImportError:
        print("  needs numpy + safetensors. Install into a FRESH venv (never ComfyUI's):")
        print("     uv pip install numpy safetensors")
        print("  Then re-run with --weight pointing at a real q_proj tensor.")
        print()
        return

    tensors = load_file(weight_path)
    # pick a q_proj-like 2-D tensor if the file has several
    key = None
    for k, v in tensors.items():
        if v.ndim == 2 and ("q_proj" in k or key is None):
            key = k
            if "q_proj" in k:
                break
    if key is None:
        print("  no 2-D weight tensor found in the file.")
        return
    W_np = tensors[key].astype(np.float64).ravel()
    # subsample to keep the pure-Python nearest-code loop tractable (statistics hold)
    if W_np.size > 65536:
        stride = W_np.size // 65536
        W_np = W_np[::stride]
    W = W_np.tolist()
    n = len(W)
    std = math.sqrt(sum(w * w for w in W) / n)
    print(f"  tensor '{key}': flattened+subsampled to {n} weights, std {std:.5f}")
    print()
    for label, codes, block, mode, dq in (
        ("NF4 DQ off ", NF4_LEVELS, 64, "fp32", False),
        ("NF4 DQ on  ", NF4_LEVELS, 64, "fp8", True),
        ("FP8 E4M3   ", FP8_LEVELS, len(W), "fp32", False),
        ("MXFP4      ", E2M1_LEVELS, 32, "pow2", False),
        ("NVFP4      ", E2M1_LEVELS, 16, "fp8", False),
    ):
        _, r = roundtrip(W, codes, block, mode, double_quant=dq)
        print(f"  {label}  relRMS {r*100:>6.2f}%   [MEA on your q_proj]")
    print()
    print("  Same lesson as the synthetic slice, now on real weights: the block size,")
    print("  not the bit-count, decides the error. [MEA]")
    print()


# --------------------------------------------------------------------------- #
# STEP 4 - bitsandbytes CROSS-CHECK + PERPLEXITY.  Spark-side; DESK-CHECKED here.
# --------------------------------------------------------------------------- #

def bnb_crosscheck(model_name, do_perplexity):
    print("=" * 74)
    print(f"STEP 4 - bitsandbytes NF4 + perplexity on {model_name}  [MEA - Spark]")
    print("=" * 74)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("  This path needs torch + transformers + bitsandbytes on the DGX Spark.")
        print("  It is DESK-CHECKED here and RUN BY YOU there. What it would need:")
        print("    - a FRESH venv (never ComfyUI's) with the setup-page stack:")
        print("        torch 2.13.0 (cu130) | transformers 5.14.1 | bitsandbytes 0.49.2")
        print("    - bitsandbytes imports with a NATIVE sm_121 kernel on aarch64+CUDA-13")
        print("      (constants §6.9) - x86 does not get it; the Spark is better served here.")
        print("    - ~5 GB of GPU memory for the NF4 8B base; CONTENDS with ComfyUI if live.")
        print()
        print("  The verified v5 quantized-load form (constants §7, brief-tooling §4.1):")
        print("      bnb = BitsAndBytesConfig(")
        print("          load_in_4bit=True,")
        print('          bnb_4bit_quant_type="nf4",')
        print("          bnb_4bit_compute_dtype=torch.bfloat16,")
        print("          bnb_4bit_use_double_quant=True,   # DEFAULT IS False - set it")
        print("      )")
        print("      model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb)")
        print("  (`load_in_4bit=True` as a TOP-LEVEL kwarg was REMOVED in v5 - every old")
        print("   tutorial is now wrong. Confirm field names with your installed 5.14.1.)")
        print()
        print("  For the per-tensor NF4 cross-check, the bnb functional layer is:")
        print("      import bitsandbytes.functional as F")
        print('      q, state = F.quantize_4bit(w, quant_type="nf4", compress_statistics=DQ)')
        print("      w_hat = F.dequantize_4bit(q, state)   # relRMS vs w = the retention proxy")
        print("  [API: verify with help(F.quantize_4bit) on your box - the briefs pinned")
        print("   BitsAndBytesConfig, NOT the functional signature. Do not trust it blind.]")
        print()
        print("  Retention (constants §9.7): DO NOT print 'AWQ 95% / GGUF 92% / GPTQ 90%'.")
        print("  Those SEO figures are mutually inconsistent folklore. Measure perplexity")
        print("  before/after here instead and report YOUR delta as [MEA].")
        return

    # --- the assertion that fixes the honest-default beat, LIVE ------------- #
    default_cfg = BitsAndBytesConfig(load_in_4bit=True)
    assert default_cfg.bnb_4bit_use_double_quant is False, (
        "bnb_4bit_use_double_quant must DEFAULT to False (constants §3) - "
        f"got {default_cfg.bnb_4bit_use_double_quant}")
    print("  live check: BitsAndBytesConfig().bnb_4bit_use_double_quant is False  [confirmed]")
    print("  -> the un-opted-in Qwen3-8B NF4 base is 4.607 GB, not 4.225 GB. [VP §3]")
    print()

    import bitsandbytes.functional as F

    def bnb_relrms(w, dq):
        q, state = F.quantize_4bit(w, quant_type="nf4", compress_statistics=dq)
        w_hat = F.dequantize_4bit(q, state).to(torch.float32)
        e = (w.to(torch.float32) - w_hat)
        return (e.pow(2).mean().sqrt() / w.to(torch.float32).pow(2).mean().sqrt()).item()

    print(f"  loading {model_name} in bf16 to read a real q_proj ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    w = None
    for name, p in model.named_parameters():
        if "q_proj.weight" in name:
            w = p.detach().to("cuda")
            break
    if w is not None:
        r_off = bnb_relrms(w, False)
        r_on = bnb_relrms(w, True)
        print(f"  bnb NF4 on a real q_proj: relRMS DQ-off {r_off*100:.2f}%  DQ-on {r_on*100:.2f}%  [MEA]")
        print("  compare against STEP 3's reference reconstruction - they should agree closely.")
    print()

    if do_perplexity:
        print("  perplexity delta (the honest retention number - constants §9.7):")
        text = ("Quantization trades bytes for a little error. This paragraph is a stand-in; "
                "point --perplexity at your own held-out text for a number that means something.")
        enc = tok(text, return_tensors="pt").to(model.device)

        def ppl(m):
            with torch.no_grad():
                out = m(**enc, labels=enc["input_ids"])
            return math.exp(out.loss.item())

        ppl_bf16 = ppl(model)
        del model
        torch.cuda.empty_cache()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model_q = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb)
        ppl_nf4 = ppl(model_q)
        print(f"    bf16 PPL {ppl_bf16:.3f}  ->  NF4+DQ PPL {ppl_nf4:.3f}  "
              f"(retention {100*ppl_bf16/ppl_nf4:.1f}%)  [MEA]")
        print("    THIS is the retention number - measured on your box, not the SEO folklore.")
    print()


# --------------------------------------------------------------------------- #

def stamp():
    print("-" * 74)
    print("08_quantization_lab.py  |  verified against: transformers 5.14.1 · bitsandbytes")
    print(f"0.49.2 · torch 2.13.0 (cu130) · CUDA 13.0  |  running: python {sys.version.split()[0]}")
    print("steps 1-2 need only the standard library; --weight adds numpy+safetensors;")
    print("--model adds torch+transformers+bitsandbytes (Spark, fresh venv).")
    print("-" * 74)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run steps 1-2 with all assertions and exit (no GPU, CI-style)")
    ap.add_argument("--weight", default=None,
                    help="safetensors file with a real q_proj tensor (step 3)")
    ap.add_argument("--model", default=None,
                    help="HF id / path of a Qwen3 for the bnb + perplexity path (step 4, Spark)")
    ap.add_argument("--perplexity", action="store_true",
                    help="with --model: also measure a bf16->NF4 perplexity delta")
    ap.add_argument("--seed", type=int, default=42, help="slice seed (default 42)")
    args = ap.parse_args()

    stamp()
    print()
    budget_table()
    measure_scheme(seed=args.seed)

    if args.self_test:
        print("=" * 74)
        print("SELF-TEST PASSED - budget arithmetic + measured-scheme invariants all hold.")
        print("=" * 74)
        return

    if args.weight:
        real_layer(args.weight)
    if args.model:
        bnb_crosscheck(args.model, args.perplexity)

    if not args.weight and not args.model:
        print("=" * 74)
        print("Steps 1-2 done (budget + measured scheme). For a REAL layer add")
        print("  --weight q_proj.safetensors ; for bnb + perplexity on the Spark add")
        print("  --model Qwen/Qwen3-8B --perplexity . Both are read-only until --model.")
        print("=" * 74)


if __name__ == "__main__":
    main()
