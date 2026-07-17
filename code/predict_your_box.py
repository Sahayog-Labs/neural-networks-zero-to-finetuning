#!/usr/bin/env python3
"""
predict_your_box.py - the capstone. The three numbers the course refused to hand you.

Course artifact for p.63 ("Predict your box"). This is the intellectual destination of the
whole corpus: you can now PREDICT what your hardware will do before you run it, and this one
console proves it against your own silicon. It unifies the p.18 memory ledger and the p.43/p.51
roofline lab - but predictive. Three numbers come out, all yours:

    1. THE WALL     usable memory   (torch.cuda.mem_get_info, constants sec 6.8)
    2. THE CEILING  dense BF16 GEMM  TFLOP/s   (unpublished; you MEASURE it)
    3. THE RIDGE    ceiling / bandwidth  FLOP/byte   (which decides bound-by-what)

The course never asserts a memory budget, never prints a dense-BF16 ceiling, never fixes the
ridge - because the honest numbers are [MEA] (yours to measure) or [INF] (genuinely unresolved:
62 vs 125 TF). This script predicts each from frozen [DER] arithmetic, then hands you the tools
to measure it and grade the prediction. The predict-then-measure loop IS the payoff.

The three predictions, from constants.md:
    WALL     full AdamW fine-tune of Qwen3-8B needs 122.05 GiB of STATE; your measured
             MemTotal is 121.6875 GiB [MEA-DEV] -> slack -0.36 GiB -> OOM, before activations.
    CEILING  62.5 TF (FP32-accumulate) or 125 TF (FP16-accumulate), both [INF, not published].
    RIDGE    227 FLOP/byte (if 62 TF) or 458 (if 125 TF), both [INF]. Decode (I=1) is far below
             both -> bandwidth-bound; diffusion (I=4096) is far above both -> compute-bound.

Usage
-----
    python predict_your_box.py              # predict (no GPU) + measure (GPU), ~30-60 s
    python predict_your_box.py --quick      # one GEMM size, fewer iters, ~20 s
    python predict_your_box.py --sizes 8192 # pick the GEMM size(s)
    python predict_your_box.py --no-gpu     # the predictions only; how to measure on the Spark
    python predict_your_box.py --self-test  # frozen-arithmetic assertions only (no torch, this box)

SAFETY: the measurement allocates a few GB of bf16 matrices and saturates the tensor cores for
a few seconds per size. If ComfyUI (or a training run) is live on the GPU, this CONTENDS with it
and the contention drags your measured TFLOP/s DOWN - a true ~125 TF box can then read as a false
~62 TF. Run with the GPU otherwise idle, or trust the --no-gpu predictions. Nothing is written and
nothing is installed; this is read-only w.r.t. your system. Run it in a FRESH venv, NEVER
ComfyUI's (hardware-ground-truth.md sec 3).
"""

import argparse
import random
import sys

# --------------------------------------------------------------------------- #
# Unit discipline (constants sec 0). Capacity in GiB (2**30); weights/state quoted
# alone in GB (1e9); compute in TFLOP/s (1e12); bandwidth in GB/s (1e9).
# --------------------------------------------------------------------------- #
GiB = 1 << 30
GB = 10**9
TF = 10**12

# ----- FROZEN values from constants.md. Asserted before they are printed (sec B.3). ----- #
P = 8_190_735_360               # Qwen3-8B parameter count, exact (constants sec 1.2, p.36)
FULLFT_BYTES = P * 16           # AdamW mixed state: 16 B/param (constants sec 2.1/2.2)
NEED_GIB = 122.05               # full-FT STATE in GiB (constants sec 2.2 - "THE number")  [DER]
MEASURED_TOTAL_GIB = 121.6875   # his usable MemTotal from /proc/meminfo (constants sec 6.8)  [MEA-DEV]
PHYSICAL_GIB = 128.0            # LPDDR5X physical parts (marketing capacity)
CARVEOUT_GIB = 6.3125           # 128 - 121.6875, firmware/driver reserve (constants sec 6.8)
SLACK_GIB = -0.36               # 121.6875 - 122.05, before activations (constants sec 2.2/6.8)

N_LORA = 43_646_976             # LoRA r=16 all-linear trainable params (constants sec 3)
LORA_STATE_BYTES = 2 * P + 16 * N_LORA   # bf16 base + 16 B/trainable (constants sec 2.3)
LORA_STATE_GB = 17.08           # GB (constants sec 2.3)
LORA_STATE_GIB = 15.91          # GiB (constants sec 2.3)
PARAM_RATIO = 187.7             # 8,190,735,360 / 43,646,976 (constants sec 2.3, the 187x identity)

PUBLISHED_BW = 273.0            # GB/s, GB10 LPDDR5X spec, THEORETICAL peak (constants sec 6)   [VP]
CEIL_FP32ACC = 62.5             # TF, BF16 dense FP32-accumulate (constants sec 6.3)            [INF]
CEIL_FP16ACC = 125.0            # TF, BF16 dense FP16-accumulate (constants sec 6.3)            [INF]
CEIL_ROOFLINE = 62.0            # TF, working training roofline behind the ridge (constants sec 6.3/6.4)
RIDGE_LOW = 227                 # FLOP/byte, if ~62 TF (constants sec 6.4)                      [INF]
RIDGE_HIGH = 458               # FLOP/byte, if ~125 TF (constants sec 6.4)                     [INF]
SM_COUNT = 48                   # GB10 SMs (constants sec 6.3)
CLOCK_GHZ = 2.6                 # observed load clock (constants sec 6.3)
FP32ACC_FLOP_PER_SM_CLK = 512   # BF16 FP32-accumulate (constants sec 6.3)
FP16ACC_FLOP_PER_SM_CLK = 1024  # BF16 FP16-accumulate (constants sec 6.3)

I_DECODE = 1                    # LLM decode batch-1 arithmetic intensity, bf16 (constants sec 6.5)
I_DIFFUSION = 4096              # diffusion-sample intensity (constants sec 6.5, D-15)


def stamp():
    """Version line - a support request should begin with a paste of real versions (sec B.7)."""
    print("verified against: torch 2.13.0 | transformers 5.14.1 | peft 0.19.1 | "
          "trl 1.8.x | CUDA 13.0")
    try:
        import torch
        print(f"running with:     torch {torch.__version__} | CUDA {torch.version.cuda}")
    except ImportError:
        print("running with:     torch NOT importable here (fine for --self-test / --no-gpu)")


# --------------------------------------------------------------------------- #
# PREDICTION 1 - THE WALL. Pure arithmetic, no GPU. Asserted, then printed.
# --------------------------------------------------------------------------- #

def predict_wall():
    """Predict the memory verdict from frozen [DER] state and his [MEA-DEV] MemTotal."""
    print("=" * 70)
    print("PREDICTION 1 - THE WALL  (memory: does a full 8B fine-tune fit?)")
    print("=" * 70)

    # Assert the frozen ledger BEFORE printing it (never launder an estimate into a fact).
    need = FULLFT_BYTES / GiB
    assert abs(need - NEED_GIB) < 0.01, f"full-FT need {need} != {NEED_GIB} GiB"
    slack = MEASURED_TOTAL_GIB - NEED_GIB
    assert abs(slack - SLACK_GIB) < 0.01, f"slack {slack} != {SLACK_GIB} GiB"
    assert abs((PHYSICAL_GIB - MEASURED_TOTAL_GIB) - CARVEOUT_GIB) < 1e-6, "carveout mismatch"

    print(f"  Qwen3-8B params            P = {P:>15,}   (exact, p.36)")
    print(f"  full AdamW state           16 B/param x P")
    print(f"                           = {FULLFT_BYTES:>15,} B")
    print(f"                           = {FULLFT_BYTES/GB:>8.2f} GB   (decimal)")
    print(f"                           = {need:>8.2f} GiB  (binary)   [DER]  <- STATE only")
    print()
    print(f"  Sold as   {PHYSICAL_GIB:>8.2f} GiB physical (marketing)")
    print(f"  Carveout -{CARVEOUT_GIB:>8.4f} GiB  firmware/driver reserve "
          f"({100*CARVEOUT_GIB/PHYSICAL_GIB:.1f}% gone before you start)")
    print(f"  You have  {MEASURED_TOTAL_GIB:>8.4f} GiB  usable MemTotal   [MEA-DEV, his box]")
    print()
    print(f"  PREDICT:  need {need:.2f} - have {MEASURED_TOTAL_GIB:.4f} = slack {slack:+.2f} GiB")
    print(f"            -> DOES NOT FIT, and that is BEFORE activations (+2-6 GB [EST]).")
    print(f"            The wall is real, and it is razor-thin: 0.3%. Step 3 confirms it live.")
    print()
    # The escape ladder payoff (memory-ledger thread FINAL beat, p.63).
    assert abs(LORA_STATE_BYTES / GB - LORA_STATE_GB) < 0.01, "LoRA state GB mismatch"
    assert abs(LORA_STATE_BYTES / GiB - LORA_STATE_GIB) < 0.01, "LoRA state GiB mismatch"
    assert abs(P / N_LORA - PARAM_RATIO) < 0.15, "187x parameter-ratio identity broke"
    print(f"  The escape: LoRA r=16 all-linear needs {LORA_STATE_GB:.2f} GB / "
          f"{LORA_STATE_GIB:.2f} GiB -> FITS.")
    print(f"  LoRA does not shrink the base or the activations - it deletes the optimizer")
    print(f"  state: trainable params drop 187x ({P:,} / {N_LORA:,} = {P/N_LORA:.1f}).")
    print()


# --------------------------------------------------------------------------- #
# PREDICTIONS 2 & 3 - THE CEILING and THE RIDGE. Inferred arithmetic, no GPU.
# --------------------------------------------------------------------------- #

def predict_ceiling_and_ridge():
    """Assert and print BOTH inferred ceilings and BOTH candidate ridges. Pick no winner."""
    print("=" * 70)
    print("PREDICTIONS 2 & 3 - THE CEILING and THE RIDGE  [INF, NOT PUBLISHED]")
    print("=" * 70)

    # The marketing-number descent (constants sec 6.3). Assert each rung, then print.
    fp4_sparse = 1000.0
    fp4_dense = fp4_sparse / 2          # 2:4 sparsity
    fp8_dense = fp4_dense / 2           # FP4 -> FP8
    bf16_fp16acc = fp8_dense / 2        # FP8 -> BF16, FP16-accumulate
    bf16_fp32acc = bf16_fp16acc / 2     # consumer-Blackwell FP32-accumulate penalty
    assert abs(bf16_fp16acc - CEIL_FP16ACC) < 1e-9, "125 TF descent broke"
    assert abs(bf16_fp32acc - CEIL_FP32ACC) < 1e-9, "62.5 TF descent broke"
    print("  NVIDIA publishes 1000 TFLOPS FP4 *with sparsity* and no dense BF16 figure.")
    print(f"    1000 FP4 sparse  /2 -> {fp4_dense:.0f} FP4 dense  /2 -> {fp8_dense:.0f} FP8 "
          f"/2 -> {bf16_fp16acc:.0f} BF16 FP16-acc  /2 -> {bf16_fp32acc:.1f} BF16 FP32-acc")
    print(f"    CEILING A: {CEIL_FP32ACC:>5.1f} TF  BF16 FP32-accumulate  [INF] "
          f"<- what a PyTorch training matmul does")
    print(f"    CEILING B: {CEIL_FP16ACC:>5.1f} TF  BF16 FP16-accumulate  [INF] "
          f"<- full tensor-core rate")

    # Independent per-SM cross-check (constants sec 6.3).
    persm_fp32 = SM_COUNT * FP32ACC_FLOP_PER_SM_CLK * CLOCK_GHZ * 1e9 / TF
    persm_fp16 = SM_COUNT * FP16ACC_FLOP_PER_SM_CLK * CLOCK_GHZ * 1e9 / TF
    assert abs(persm_fp32 - 64.0) < 0.5, "per-SM FP32-acc cross-check broke"
    assert abs(persm_fp16 - 128.0) < 0.5, "per-SM FP16-acc cross-check broke"
    print(f"    cross-check: {SM_COUNT} SMs x {FP32ACC_FLOP_PER_SM_CLK} FLOP/SM/clk x "
          f"{CLOCK_GHZ} GHz = {persm_fp32:.0f} TF (~62.5); x{FP16ACC_FLOP_PER_SM_CLK} "
          f"= {persm_fp16:.0f} TF (~125)")
    print()

    # The two candidate ridges (constants sec 6.4). Ridge uses the 62 TF working roofline.
    ridge_low = CEIL_ROOFLINE * TF / (PUBLISHED_BW * GB)
    ridge_high = CEIL_FP16ACC * TF / (PUBLISHED_BW * GB)
    assert round(ridge_low) == RIDGE_LOW, f"ridge(62 TF) rounds to {round(ridge_low)}, not {RIDGE_LOW}"
    assert round(ridge_high) == RIDGE_HIGH, f"ridge(125 TF) rounds to {round(ridge_high)}, not {RIDGE_HIGH}"
    print(f"  RIDGE  I* = ceiling / BW   (BW = {PUBLISHED_BW:.0f} GB/s published, theoretical):")
    print(f"    if ~62 TF  -> {ridge_low:6.1f}  ~ {RIDGE_LOW}   FLOP/byte  [INF]")
    print(f"    if ~125 TF -> {ridge_high:6.1f}  ~ {RIDGE_HIGH}   FLOP/byte  [INF]")
    print(f"    PREDICT:  decode (I={I_DECODE}) << both -> bandwidth-bound (slow);")
    print(f"              diffusion (I={I_DIFFUSION}) >> both -> compute-bound (fast).")
    print(f"    Both verdicts hold under EITHER ridge. Only the ridge number moves.")
    print("  The course asserts NEITHER ceiling. Step 3 measures yours and settles it.")
    print()
    return ridge_low, ridge_high


# --------------------------------------------------------------------------- #
# MEASUREMENT - Step 1 (the live wall) + Step 3 (the live ceiling & ridge).
# This is the p.63 console verbatim, wrapped in narration and a verdict.
# --------------------------------------------------------------------------- #

def measure_wall_live():
    """Step 1 of the page console: the real wall, on his box, this second."""
    import torch
    print("=" * 70)
    print("STEP 1 (measured) - THE WALL, live via torch.cuda.mem_get_info()")
    print("=" * 70)
    cap = torch.cuda.get_device_capability(0)
    print(f"  device        {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        print(f"  note: no sm_{cap[0]}{cap[1]} cubin and no PTX - runs via CUDA minor-version binary")
        print(f"        compatibility (an sm_{cap[0]}0 kernel on an sm_{cap[0]}{cap[1]} device). Not a fault.")

    free, total = torch.cuda.mem_get_info()          # bytes  (constants sec 6.8 verbatim)
    print(f"  usable now:   {free/GiB:.2f} GiB of {total/GiB:.2f} GiB")

    slack = total / GiB - NEED_GIB
    verdict = "FITS" if slack > 0 else "OOM, before activations"
    print(f"  full FT needs {NEED_GIB:.2f} GiB -> slack {slack:+.2f} GiB  ({verdict})")
    if abs(total / GiB - MEASURED_TOTAL_GIB) > 0.01 * MEASURED_TOTAL_GIB:
        print(f"  note: your total {total/GiB:.2f} GiB differs from the recorded {MEASURED_TOTAL_GIB:.4f} GiB")
        print(f"        [MEA-DEV] - a different carveout or DGX-OS build. Your number wins; use it.")
    else:
        print(f"  matches the predicted {MEASURED_TOTAL_GIB:.4f} GiB [MEA-DEV]: the wall is confirmed.")
    contended = free < 0.5 * total
    if contended:
        print()
        print("  !! Over half your memory is already in use - ComfyUI or a run is live.")
        print("     The ceiling in Step 3 will read pessimistically. Free the GPU to trust it.")
    print()
    return contended


def _time_gemm(n, iters, warmup):
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
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        c = a @ b                                    # noqa: F841 - timed, not used
    t1.record()
    torch.cuda.synchronize()
    secs = t0.elapsed_time(t1) / 1e3 / iters
    tf = 2 * n**3 / secs / TF                        # multiply-add = 2 FLOP
    del a, b
    torch.cuda.empty_cache()
    return tf


def measure_ceiling_and_ridge(sizes, quick, ridge_low, ridge_high, contended):
    """Step 3 of the page console: a big bf16 GEMM in BOTH accumulate modes -> ceiling, ridge."""
    import torch
    print("=" * 70)
    print("STEP 3 (measured) - THE CEILING and THE RIDGE, both accumulate modes")
    print("=" * 70)

    # allow_bf16_reduced_precision_reduction is the one documented knob that touches bf16
    # accumulation precision: OFF forces full fp32 reduction, ON permits an fp16-style
    # split-K reduction. It is the closest user-selectable analogue of the FP32/FP16-acc
    # fork. If both columns come out equal, cuBLAS picked the same kernel - that is itself
    # the finding; your rate still tells you which physical ceiling you sit on.
    knob = "allow_bf16_reduced_precision_reduction"
    has_knob = hasattr(torch.backends.cuda.matmul, knob)
    modes = [("fp32-acc", False), ("fp16-acc", True)] if has_knob else [("default", None)]
    if not has_knob:
        print(f"  note: torch.backends.cuda.matmul.{knob} absent on this build - default path only.")

    iters = 10 if quick else 30
    warmup = 5
    peak = 0.0
    for label, flag in modes:
        if has_knob and flag is not None:
            setattr(torch.backends.cuda.matmul, knob, flag)
        for n in sizes:
            tf = _time_gemm(n, iters, warmup)
            if tf is None:
                print(f"  {label}: {n:>6} x {n:<6}  OOM - skipped (free the GPU and retry)")
                continue
            ridge = tf * TF / (PUBLISHED_BW * GB)    # YOUR ridge: 227, or 458?
            peak = max(peak, tf)
            print(f"  {label}: {n:>6} x {n:<6}  {tf:7.1f} TF/s  ->  ridge {ridge:5.0f} FLOP/byte")
    if has_knob:
        setattr(torch.backends.cuda.matmul, knob, True)   # leave no state behind (the default)
    print()

    # Report which inferred ceiling the measurement lands nearest. MEASURE, never assert (sec D.7).
    if peak > 0:
        d62 = abs(peak - CEIL_FP32ACC)
        d125 = abs(peak - CEIL_FP16ACC)
        near_low = d62 <= d125
        print(f"  YOUR peak dense BF16 ceiling: {peak:.1f} TF/s (measured, not asserted)")
        print(f"    distance to 62.5 TF [INF]: {d62:5.1f}   to 125 TF [INF]: {d125:5.1f}")
        print(f"    -> nearest {'~62.5 TF (FP32-acc) -> ridge ~227' if near_low else '~125 TF (FP16-acc) -> ridge ~458'} "
              f"[INF]")
        if contended:
            print("    !! measured under contention - a low reading is not evidence for 62 TF. Re-run idle.")
    else:
        print("  No GEMM completed (OOM). Re-run with the GPU idle.")
    print()


def final_verdict():
    print("=" * 70)
    print("THE CAPSTONE - three numbers, once refused, now yours")
    print("=" * 70)
    print("  THE WALL     you measured your usable memory and watched 122.05 GiB miss it.")
    print("  THE CEILING  you measured the dense BF16 rate NVIDIA won't print.")
    print("  THE RIDGE    you divided the two and learned which side of it your work sits.")
    print()
    print("  Same box, three answers, one console: full FT OOMs; decode (I=1) is far below")
    print(f"  the ridge -> bandwidth-bound, slow; diffusion (I={I_DIFFUSION}) is far above it ->")
    print("  compute-bound, fast. You predicted all three before you ran a thing.")
    print("  The course's real destination was never 'operate the tools.' It was this.")


# --------------------------------------------------------------------------- #
# --self-test : frozen-arithmetic assertions only. No torch. Runs on this box.
# --------------------------------------------------------------------------- #

def self_test():
    print("SELF-TEST - asserting every frozen number against constants.md (no GPU)\n")
    predict_wall()                                   # asserts 122.05, -0.36, carveout, LoRA, 187x
    ridge_low, ridge_high = predict_ceiling_and_ridge()  # asserts 62.5/125, per-SM, 227/458

    # The roofline fork the whole page rides on, restated as bare assertions.
    assert I_DECODE == 1, "decode intensity must be 1 FLOP/byte at bf16"
    assert I_DECODE < RIDGE_LOW < RIDGE_HIGH, "decode must sit below both ridges"
    assert I_DIFFUSION > RIDGE_HIGH > RIDGE_LOW, "diffusion must sit above both ridges"
    assert round(ridge_low) == RIDGE_LOW and round(ridge_high) == RIDGE_HIGH

    print("  wall:    122.05 GiB need vs 121.6875 GiB have -> -0.36 GiB, OOM        [OK]")
    print("  ceiling: 62.5 / 125 TF [INF]; per-SM 64 / 128 TF cross-check          [OK]")
    print("  ridge:   227 / 458 FLOP/byte [INF]; decode<both<diffusion             [OK]")
    print("\nSELF-TEST PASSED - the three numbers and their arithmetic all match constants.md.")


def main():
    ap = argparse.ArgumentParser(
        description="Predict, then measure, the three numbers your GB10 refused to publish.")
    ap.add_argument("--sizes", default="8192,12288,16384",
                    help="comma-separated square GEMM sizes (default 8192,12288,16384)")
    ap.add_argument("--quick", action="store_true", help="one size, fewer iters, ~20 s")
    ap.add_argument("--no-gpu", action="store_true", help="predictions only; how to measure on the Spark")
    ap.add_argument("--self-test", action="store_true", help="frozen-arithmetic assertions only (no torch)")
    args = ap.parse_args()

    print("#" * 70)
    print("# predict_your_box.py - the capstone. Predict it, then measure it. (p.63)")
    print("#" * 70)
    print()

    if args.self_test:
        self_test()
        return

    stamp()
    print()
    predict_wall()
    ridge_low, ridge_high = predict_ceiling_and_ridge()

    if args.no_gpu:
        print("  (--no-gpu) Predictions printed. To MEASURE the wall and the ceiling, run this")
        print("  on the Spark WITHOUT --no-gpu, in a fresh venv (never ComfyUI's):")
        print("      ~/course/.venv/bin/python predict_your_box.py")
        print("  Step 1 calls mem_get_info() live; Step 3 times a bf16 GEMM in both accumulate")
        print("  modes. The arithmetic above is what those measurements get graded against.")
        return

    try:
        import torch
    except ImportError:
        print("torch not importable here. On the Spark, use a torch-bearing venv, e.g.:")
        print("  ~/course/.venv/bin/python predict_your_box.py")
        print("This script installs nothing and writes nothing (read-only w.r.t. your system).")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available - stopping. Run this on the Spark.")
        sys.exit(1)

    # Seedable determinism (sec B.7) - matrix content does not change GEMM timing, but the
    # house style seeds everything a script touches.
    random.seed(42)
    torch.manual_seed(42)

    try:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    except ValueError:
        print(f"bad --sizes {args.sizes!r} - expected comma-separated integers")
        sys.exit(2)
    if args.quick:
        sizes = sizes[:1]

    contended = measure_wall_live()
    measure_ceiling_and_ridge(sizes, args.quick, ridge_low, ridge_high, contended)
    final_verdict()


if __name__ == "__main__":
    main()
