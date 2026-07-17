#!/usr/bin/env python3
"""
09_spark_capability_probe.py - settle the 62-vs-125 TF question on YOUR box.

Course artifact for p.51 (serving/eval/hardware), also cited by p.43's accumulate-precision
deep dive. Companion to `measure_your_box.py`: that script finds your roofline; this one
zooms in on the single unresolved number behind it - the dense BF16 compute ceiling.

THE QUESTION (constants §6.3, §10 exercise #2 - "the corpus's biggest gap, its best lab").
NVIDIA publishes no dense BF16 figure for GB10. Two inferred ceilings survive the audit and
they disagree by 2x:

    ~62.5 TFLOP/s   BF16 dense, FP32-accumulate   (what a plain PyTorch training matmul does)
    ~125  TFLOP/s   BF16 dense, FP16-accumulate   (full tensor-core rate; wrong for training)

Both are [INFERRED, not published]. The ridge - the FLOP/byte break-even that decides whether
a workload is memory- or compute-bound - rides on which is true:

    ridge = P_peak / BW = 62e12/273e9 = 227   (if ~62 TF)   OR   125e12/273e9 = 458   (if ~125 TF)

Community GEMM reports lean 62; an SDXL-BF16 datapoint leans 125. It is genuinely unresolved.
This script does NOT pick a side in code. It runs a big BF16 GEMM in BOTH accumulate modes,
measures YOUR box, prints both ceilings, and reports which one your measurement lands nearest.
The measurement is the answer.

Usage
-----
    python 09_spark_capability_probe.py                 # full sweep, ~30-60 s
    python 09_spark_capability_probe.py --quick         # smaller sweep, ~20 s
    python 09_spark_capability_probe.py --sizes 8192    # one size only
    python 09_spark_capability_probe.py --no-gpu        # the inferred arithmetic, no GPU
    python 09_spark_capability_probe.py --self-test     # arithmetic assertions only (CI / no torch)

SAFETY: the sweep allocates several GB of bf16 matrices and saturates the tensor cores for a
few seconds per size. If ComfyUI (or any training run) is live on the GPU, this CONTENDS with
it and - crucially - the contention drags your measured TFLOP/s DOWN, which can make a true
~125 TF box read like a false ~62 TF. Run it with the GPU otherwise idle, or trust `--no-gpu`.
Nothing is written and nothing is installed; this is read-only w.r.t. your system.
"""

import argparse
import sys

# --------------------------------------------------------------------------- #
# Unit discipline (constants §0). Compute in TFLOP/s (1e12), bandwidth in GB/s
# (1e9, vendor convention), memory compared to capacity in GiB (2**30).
# --------------------------------------------------------------------------- #
GiB = 1 << 30
GB = 10**9
TF = 10**12

# ----- FROZEN values from constants.md. Asserted before they are printed. ---- #
PUBLISHED_BW = 273.0            # GB/s, GB10 LPDDR5X spec (constants §6, §5.3)
CEIL_FP32ACC = 62.5            # TF, BF16 dense FP32-accumulate  [INF] constants §6.3
CEIL_FP16ACC = 125.0          # TF, BF16 dense FP16-accumulate  [INF] constants §6.3
CEIL_ROOFLINE = 62.0          # TF, the working training roofline used for the ridge (§6.3/§6.4)
RIDGE_LOW = 227               # FLOP/byte, if ~62 TF   [INF] constants §6.4
RIDGE_HIGH = 458              # FLOP/byte, if ~125 TF  [INF] constants §6.4
SM_COUNT = 48                 # GB10 SMs (constants §6.3)
CLOCK_GHZ = 2.6               # observed load clock (constants §6.3)
FP32ACC_FLOP_PER_SM_CLK = 512 # BF16 FP32-accumulate (constants §6.3)
FP16ACC_FLOP_PER_SM_CLK = 1024


# --------------------------------------------------------------------------- #
# Part 1 - the inferred arithmetic. No GPU. Asserted, then printed (§B.3).
# --------------------------------------------------------------------------- #

def inference_chain():
    """Reproduce and ASSERT the two inferred ceilings and the two candidate ridges."""
    print("=" * 70)
    print("THE INFERRED CEILINGS - [INF, NOT PUBLISHED]. NVIDIA prints no dense BF16.")
    print("=" * 70)

    # The marketing-number descent (constants §6.3). Assert each rung.
    fp4_sparse = 1000.0
    fp4_dense = fp4_sparse / 2          # 2:4 sparsity
    fp8_dense = fp4_dense / 2           # FP4 -> FP8
    bf16_fp16acc = fp8_dense / 2        # FP8 -> BF16, FP16-accumulate
    bf16_fp32acc = bf16_fp16acc / 2     # consumer-Blackwell FP32-accumulate penalty
    assert abs(bf16_fp16acc - CEIL_FP16ACC) < 1e-9, "125 TF chain broke"
    assert abs(bf16_fp32acc - CEIL_FP32ACC) < 1e-9, "62.5 TF chain broke"

    print("  1000 TFLOPS  FP4 sparse            (published, marketing)")
    print(f"  /2 (2:4 sparsity)   -> {fp4_dense:>6.1f}    FP4 dense")
    print(f"  /2 (FP4 -> FP8)     -> {fp8_dense:>6.1f}    FP8 dense")
    print(f"  /2 (FP8 -> BF16)    -> {bf16_fp16acc:>6.1f}    BF16 dense, FP16-accumulate   [INF]")
    print(f"  /2 (FP32-acc penalty)-> {bf16_fp32acc:>5.1f}   BF16 dense, FP32-accumulate   [INF]")
    print("                                     ^ what a plain PyTorch training matmul does")

    # Independent per-SM cross-check (constants §6.3).
    persm_fp32 = SM_COUNT * FP32ACC_FLOP_PER_SM_CLK * CLOCK_GHZ * 1e9 / TF
    persm_fp16 = SM_COUNT * FP16ACC_FLOP_PER_SM_CLK * CLOCK_GHZ * 1e9 / TF
    assert abs(persm_fp32 - 64.0) < 0.5, "per-SM FP32-acc cross-check broke"
    assert abs(persm_fp16 - 128.0) < 0.5, "per-SM FP16-acc cross-check broke"
    print()
    print("  Independent cross-check (per-SM arithmetic, constants sec 6.3):")
    print(f"    {SM_COUNT} SMs x {FP32ACC_FLOP_PER_SM_CLK} FP32-acc FLOP/SM/clk x {CLOCK_GHZ} GHz "
          f"= {persm_fp32:.1f} TF  (~64, matches 62.5)")
    print(f"    {SM_COUNT} SMs x {FP16ACC_FLOP_PER_SM_CLK} FP16-acc FLOP/SM/clk x {CLOCK_GHZ} GHz "
          f"= {persm_fp16:.1f} TF  (~128, matches 125)")

    # The two candidate ridges (constants §6.4). Ridge uses the 62 TF working roofline.
    ridge_low = CEIL_ROOFLINE * TF / (PUBLISHED_BW * GB)
    ridge_high = CEIL_FP16ACC * TF / (PUBLISHED_BW * GB)
    assert round(ridge_low) == RIDGE_LOW, f"ridge(62 TF) != {RIDGE_LOW}"
    assert round(ridge_high) == RIDGE_HIGH, f"ridge(125 TF) != {RIDGE_HIGH}"
    print()
    print(f"  Ridge  I* = P_peak / BW  (BW = {PUBLISHED_BW:.0f} GB/s published):")
    print(f"    if ~62 TF  -> {ridge_low:6.1f} FLOP/byte  ~ {RIDGE_LOW}   [INF]")
    print(f"    if ~125 TF -> {ridge_high:6.1f} FLOP/byte  ~ {RIDGE_HIGH}   [INF]")
    print("  The course prints BOTH and asserts NEITHER as fact. Your measurement settles it.")
    print()
    return ridge_low, ridge_high


# --------------------------------------------------------------------------- #
# Part 2 - the GPU measurement.
# --------------------------------------------------------------------------- #

def device_card():
    import torch
    print("=" * 70)
    print("DEVICE")
    print("=" * 70)
    cap = torch.cuda.get_device_capability(0)
    print(f"  name          {torch.cuda.get_device_name(0)}")
    print(f"  capability    sm_{cap[0]}{cap[1]}")
    print(f"  torch         {torch.__version__}   CUDA {torch.version.cuda}")
    print(f"  bf16 support  {torch.cuda.is_bf16_supported()}")
    print(f"  built for     {torch.cuda.get_arch_list()}")

    # The aarch64 / sm_121 binary-compat teaching point (hardware-ground-truth §3.1).
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        print(f"  note: no sm_{cap[0]}{cap[1]} cubin and no PTX - runs via CUDA minor-version binary")
        print(f"        compatibility (an sm_{cap[0]}0 kernel on an sm_{cap[0]}{cap[1]} device). Not a fault.")

    free, tot = torch.cuda.mem_get_info()
    print(f"  memory        {tot / GiB:.2f} GiB total | {free / GiB:.2f} GiB free  (unified)")
    contended = free < 0.5 * tot
    if contended:
        print()
        print("  !! Over half your memory is already in use - ComfyUI or a training run is live.")
        print("     Contention drags the measured TFLOP/s DOWN. A true ~125 TF box can read as")
        print("     a false ~62 TF under load. Treat a low number below as inconclusive, not proof.")
    print()
    return contended


def _time_matmul(n, iters, warmup):
    """One square bf16 GEMM timed with CUDA events. Returns TFLOP/s, or None on OOM."""
    import torch
    try:
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    except torch.cuda.OutOfMemoryError:
        return None
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        a @ b
    end.record()
    torch.cuda.synchronize()
    secs = start.elapsed_time(end) / 1000 / iters
    tf = (2 * n**3) / secs / TF          # multiply-add = 2 FLOP
    del a, b
    torch.cuda.empty_cache()
    return tf


def measure_gemm_both_modes(sizes, quick):
    """Dense bf16 GEMM in both accumulate modes. Print both; assert no winner."""
    import torch
    print("=" * 70)
    print("COMPUTE - dense BF16 GEMM in BOTH accumulate modes")
    print("=" * 70)

    # The one documented PyTorch knob that touches bf16 accumulation precision:
    # allow_bf16_reduced_precision_reduction permits a reduced-precision (fp16-style)
    # split-K reduction when True, and forces full fp32 reduction when False. It is the
    # closest user-selectable analogue of the FP32-acc vs FP16-acc fork. It may NOT flip
    # the tensor-core mode on every cuBLAS path - so if both columns come out equal, that
    # is itself the finding (cuBLAS picked the same kernel), and your rate still tells you
    # which physical ceiling you sit on.
    knob = "allow_bf16_reduced_precision_reduction"
    has_knob = hasattr(torch.backends.cuda.matmul, knob)
    if not has_knob:
        print(f"  note: torch.backends.cuda.matmul.{knob} not present on this build -")
        print("        measuring the default path only; report the single rate against both ceilings.")

    iters = 5 if quick else 20
    warmup = 3
    results = {"fp32_acc (reduction OFF)": {}, "fp16_acc (reduction ON)": {}}

    modes = [("fp32_acc (reduction OFF)", False), ("fp16_acc (reduction ON)", True)]
    if not has_knob:
        modes = [("default path", None)]
        results = {"default path": {}}

    for label, flag in modes:
        if has_knob and flag is not None:
            setattr(torch.backends.cuda.matmul, knob, flag)
        print(f"\n  -- {label} --")
        for n in sizes:
            tf = _time_matmul(n, iters, warmup)
            if tf is None:
                print(f"     {n:>6} x {n:<6}  OOM - skipped (free the GPU and retry)")
                continue
            results[label][n] = tf
            print(f"     {n:>6} x {n:<6}  {tf:>8.2f} TFLOP/s")

    # Restore the default so we leave no state behind.
    if has_knob:
        setattr(torch.backends.cuda.matmul, knob, True)

    best = {lbl: (max(v.values()) if v else 0.0) for lbl, v in results.items()}
    print()
    for lbl, b in best.items():
        if b:
            print(f"  peak  {lbl:<26} {b:8.2f} TFLOP/s")
    peak_overall = max(best.values()) if best else 0.0
    return peak_overall


def measure_bandwidth(quick):
    """Achieved bandwidth via a device-to-device copy. Published 273 GB/s is theoretical."""
    import torch
    print()
    print("=" * 70)
    print("BANDWIDTH - achieved, not theoretical")
    print("=" * 70)
    n = (256 if quick else 1024) * 1024 * 1024 // 4      # float32 elements
    try:
        src = torch.empty(n, device="cuda", dtype=torch.float32)
        dst = torch.empty(n, device="cuda", dtype=torch.float32)
    except torch.cuda.OutOfMemoryError:
        print("  OOM - not enough free memory to measure. Free the GPU and retry.")
        return 0.0
    src.fill_(1.0)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    iters = 5 if quick else 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    secs = start.elapsed_time(end) / 1000 / iters
    moved = 2 * src.numel() * 4                          # read + write
    bw = moved / secs / GB
    print(f"  copy {moved / GB:.2f} GB   {secs * 1000:>7.2f} ms   {bw:>7.1f} GB/s")
    print(f"  --> achieved {bw:.1f} of {PUBLISHED_BW:.0f} GB/s published = {100 * bw / PUBLISHED_BW:.0f}% of peak")
    del src, dst
    torch.cuda.empty_cache()
    return bw


def prefill_vs_decode(quick):
    """The two regimes side by side: a wide GEMM (prefill) vs a matrix-vector (decode)."""
    import torch
    print()
    print("=" * 70)
    print("PREFILL vs DECODE - same weights, opposite bottlenecks")
    print("=" * 70)
    d = 4096                                             # Qwen3-8B hidden size
    S = 2048                                             # a prefill sequence
    iters = 10 if quick else 40
    w = torch.randn(d, d, device="cuda", dtype=torch.bfloat16)

    def _run(x):
        for _ in range(3):
            x @ w
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            x @ w
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000 / iters

    x_prefill = torch.randn(S, d, device="cuda", dtype=torch.bfloat16)
    x_decode = torch.randn(1, d, device="cuda", dtype=torch.bfloat16)
    t_prefill = _run(x_prefill)
    t_decode = _run(x_decode)

    tf_prefill = (2 * S * d * d) / t_prefill / TF
    # Decode is bandwidth-bound: the cost is reading the weight matrix, not the flops.
    bytes_decode = d * d * 2                             # bf16 weights streamed once
    bw_decode = bytes_decode / t_decode / GB
    print(f"  prefill  x=[{S},{d}] @ W=[{d},{d}]   {t_prefill * 1e3:7.3f} ms   "
          f"{tf_prefill:7.1f} TFLOP/s   I = {S} FLOP/byte  -> compute-bound")
    print(f"  decode   x=[  1,{d}] @ W=[{d},{d}]   {t_decode * 1e3:7.3f} ms   "
          f"{bw_decode:7.1f} GB/s      I = 1 FLOP/byte  -> bandwidth-bound")
    print("  One weight matrix. The prefill saturates the tensor cores; the decode just")
    print("  reads memory. That gap, not the flop count, is the whole LLM-serving story.")
    del w, x_prefill, x_decode
    torch.cuda.empty_cache()
    return tf_prefill


def verdict(peak_tf, bw, ridge_low, ridge_high, contended):
    print()
    print("=" * 70)
    print("YOUR VERDICT - which ceiling, which ridge (measured, not asserted)")
    print("=" * 70)
    if not (peak_tf and bw):
        print("  Insufficient measurement (OOM). Re-run with the GPU idle.")
        return
    d62 = abs(peak_tf - CEIL_FP32ACC)
    d125 = abs(peak_tf - CEIL_FP16ACC)
    nearest = "~62.5 TF (FP32-accumulate)" if d62 <= d125 else "~125 TF (FP16-accumulate)"
    ridge_measured = peak_tf * TF / (bw * GB)
    print(f"  peak dense BF16   {peak_tf:8.2f} TFLOP/s  (measured)")
    print(f"  achieved BW       {bw:8.2f} GB/s     (measured)")
    print(f"  YOUR ridge        {ridge_measured:8.1f} FLOP/byte  (measured peak / measured BW)")
    print()
    print(f"    distance to ~62.5 TF ceiling: {d62:6.2f} TF")
    print(f"    distance to ~125  TF ceiling: {d125:6.2f} TF")
    print(f"    -> your measurement lands nearest {nearest}")
    print(f"       which points at the ridge ~{RIDGE_LOW if d62 <= d125 else RIDGE_HIGH} "
          f"(the {RIDGE_LOW}/{RIDGE_HIGH} fork, constants §6.4).")
    if contended:
        print()
        print("  !! You ran under GPU contention. A reading near ~62 TF here is NOT evidence")
        print("     for the 62 TF ceiling - contention alone produces it. Re-run idle to trust this.")
    print()
    print("  Either way the D-15 contrast holds: decode (I=1) sits hundreds-fold below BOTH")
    print(f"  {RIDGE_LOW} and {RIDGE_HIGH} -> bandwidth-bound; diffusion (I=4096) sits above both -> compute-bound.")
    print("  Only the printed ridge number moves. The story does not.")


def self_test():
    """No-GPU arithmetic assertions - runs in CI and on this Windows box (no torch)."""
    print("SELF-TEST - asserting the frozen arithmetic against constants.md")
    ridge_low, ridge_high = inference_chain()   # asserts ceilings, per-SM, ridges internally

    # Decode / prefill arithmetic intensity at bf16 (constants §6.5): I = tokens/forward.
    b_w = 2                                     # bytes per bf16 weight
    flop_per_weight = 2                         # one multiply-add
    I_decode = flop_per_weight / b_w * 1        # 1 token
    I_prefill = flop_per_weight / b_w * 2048    # 2048 tokens
    assert I_decode == 1, "decode intensity must be 1 FLOP/byte at bf16"
    assert I_prefill == 2048, "prefill intensity must equal tokens/forward"

    # The ridge fork the whole page rides on.
    assert round(ridge_low) == RIDGE_LOW and round(ridge_high) == RIDGE_HIGH
    # Decode is below both ridges; diffusion (I=4096) is above both.
    assert I_decode < RIDGE_LOW < RIDGE_HIGH
    assert 4096 > RIDGE_HIGH > RIDGE_LOW
    print("  decode  I = 1 FLOP/byte     < ridge 227 and 458  -> bandwidth-bound  [OK]")
    print("  diffusion I = 4096 FLOP/byte> ridge 227 and 458  -> compute-bound    [OK]")
    print("SELF-TEST PASSED - ceilings 62.5/125 TF, ridges 227/458, per-SM 64/128 TF all match.")


def main():
    ap = argparse.ArgumentParser(description="Settle the 62-vs-125 TF dense BF16 ceiling on YOUR GB10.")
    ap.add_argument("--sizes", default="4096,8192,12288,16384",
                    help="comma-separated square GEMM sizes (default 4096,8192,12288,16384)")
    ap.add_argument("--quick", action="store_true", help="smaller sweep / fewer iters, ~20 s")
    ap.add_argument("--no-gpu", action="store_true", help="inferred arithmetic only, no GPU")
    ap.add_argument("--self-test", action="store_true", help="arithmetic assertions only (no torch)")
    args = ap.parse_args()

    print("#" * 70)
    print("# 09_spark_capability_probe.py - dense BF16 ceiling, measured on YOUR box")
    print("#" * 70)
    print()

    if args.self_test:
        self_test()
        return

    ridge_low, ridge_high = inference_chain()

    if args.no_gpu:
        print("  (--no-gpu) Skipping the GPU sweep. On the Spark, run without --no-gpu to")
        print("  measure your dense BF16 GEMM in both accumulate modes and see which ceiling")
        print("  you land on. The arithmetic above is what the measurement will be judged against.")
        return

    try:
        import torch
    except ImportError:
        print("torch not importable here. On the Spark, use a torch-bearing venv, e.g.:")
        print("  ~/course/.venv/bin/python 09_spark_capability_probe.py")
        print("  (or ComfyUI's venv for a read-only probe: ~/ComfyUI/.venv/bin/python ...)")
        print("This script installs nothing and writes nothing.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available - stopping. Run this on the Spark.")
        sys.exit(1)

    try:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    except ValueError:
        print(f"bad --sizes {args.sizes!r} - expected comma-separated integers")
        sys.exit(2)
    if args.quick:
        sizes = sizes[:2]

    contended = device_card()
    peak_tf = measure_gemm_both_modes(sizes, args.quick)
    bw = measure_bandwidth(args.quick)
    prefill_vs_decode(args.quick)
    verdict(peak_tf, bw, ridge_low, ridge_high, contended)


if __name__ == "__main__":
    main()
