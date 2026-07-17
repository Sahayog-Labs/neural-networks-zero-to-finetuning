#!/usr/bin/env python3
"""
bench_spark.py - the diffusion side of the roofline, measured on YOUR GB10.

Course artifact for p.60 (Latent diffusion & the roofline). This is the diffusion-track
twin of `02_kv_cache_and_roofline.py` (the LLM decode side) and a close cousin of
`09_spark_capability_probe.py` (which settles the dense-BF16 ceiling). Together they close
the D-15 contrast: same machine, same ridge, opposite verdicts.

THE ONE THING THIS PAGE PROVES (constants sec 6.5 / D-15): a FLUX denoising step processes
~4096 latent tokens per forward, so its arithmetic intensity is I ~ 4096 FLOP/byte. That sits
FAR ABOVE the Spark's ridge (227 if the BF16 ceiling is ~62 TF, 458 if ~125 TF), which makes
diffusion COMPUTE-bound - the exact opposite of LLM batch-1 decode (I = 1, hundreds-fold
bandwidth-bound). The Spark - memory-rich, bandwidth-poor - is the machine that is STRONG for
diffusion and WEAK for single-stream chat, and it is one formula read at two values of S_fwd.

THE DELETED SHOWPIECE (constants sec 9.6 / D-16). An earlier draft "predicted" the famous
"2.6 s / 1K image at FP4" benchmark from FLUX.1-dev/28-step arithmetic. That is a category
error and this script does NOT reproduce it: the 2.6 s figure is
        "FLUX.1-Schnell, 4 steps, 1024^2, batch 1, FP4"   [VP]
NOT FLUX.1-dev at 28 steps. Read as dev/28 it implies 1.06 PFLOP/s = 106% of the sparse-FP4
marketing peak - physically impossible. You cannot honestly predict a wall-clock when the FP4
dense ceiling is itself an unpublished +/-2x inference. So instead: you MEASURE your own box,
and check YOUR prediction against BOTH candidate ceilings.

Usage
-----
    python bench_spark.py --self-test        # frozen arithmetic only, no torch (runs anywhere)
    python bench_spark.py --no-gpu           # the prediction arithmetic, printed, no GPU
    python bench_spark.py                     # on the Spark: micro-bench + ridge GEMM + generate
    python bench_spark.py --no-generate       # skip the real FLUX generation (micro-bench only)
    python bench_spark.py --steps 28 --precision bf16,fp4
    python bench_spark.py --model black-forest-labs/FLUX.1-dev
    python bench_spark.py --single-file ~/ComfyUI/models/diffusion_models/flux1-dev.safetensors

SAFETY: the GEMM/micro-bench allocate several GB of bf16 tensors and saturate the tensor cores;
`--generate` loads a ~24 GB transformer and runs 28 denoising steps. YOUR ComfyUI is usually
live on this GPU. This script CONTENDS with it: contention drags every measured number down and
can make a compute-bound job look bandwidth-bound. It frees ITS OWN cache before each sweep and
refuses the heavy generation if too little memory is free (override with --force). It NEVER
touches ComfyUI's process and writes/installs nothing - read-only w.r.t. your system, except the
optional output image under --outdir. Stop ComfyUI, or run the light paths (--no-generate).
"""

import argparse
import sys

# --------------------------------------------------------------------------- #
# Unit discipline (constants sec 0): compute in TFLOP/s (1e12), bandwidth in
# GB/s (1e9, vendor convention), memory vs capacity in GiB (2**30).
# --------------------------------------------------------------------------- #
GiB = 1 << 30
GB = 10**9
TF = 10**12

# ---- FROZEN values from constants.md. Asserted before they are printed. ----- #
PUBLISHED_BW = 273.0             # GB/s, GB10 LPDDR5X (constants sec 6.1)
CEIL_FP32ACC = 62.0              # TF, BF16 dense FP32-accumulate, working roofline [INF] sec 6.3
CEIL_FP16ACC = 125.0            # TF, BF16 dense FP16-accumulate                    [INF] sec 6.3
CEIL_FP4_DENSE = 500.0         # TF, FP4 dense (~500-511 measured)                [INF]+[VP] sec 6.3
RIDGE_LOW = 227                 # FLOP/byte if ~62 TF   [INF] sec 6.4
RIDGE_HIGH = 458                # FLOP/byte if ~125 TF  [INF] sec 6.4
MARKETING_FP4_SPARSE = 1000.0  # TF, the "1 PFLOP" headline (FP4 + 2:4 sparsity) sec 6.2

# FLUX.1-dev geometry (constants sec 1-anchor for diffusion / sec 9.6, brief-diffusion sec 8.4).
FLUX1_N_LEDGER = 12.0e9         # params, for the 16 B/param ledger and the 2NS_img proof
FLUX1_N_FLOP = 11.9e9          # analytic param count used for the honest FLOP tally (sec 9.6)
FLUX1_D = 3072                 # d_model (constants sec 9.6)
S_IMG = 4096                   # latent tokens after 2x2 patchify at 1024^2 (sec 9.6)
S_TXT = 512                    # text tokens (constants sec 9.6, Schnell count)
S_TOTAL = S_IMG + S_TXT        # 4608
ATTN_LAYER_FACTOR = 57         # 19 double + 38 single blocks (constants sec 9.6)

# Frozen results the script reproduces & asserts (constants sec 9.6 / 6.3).
PERSTEP_LINEAR = 1.097e14      # FLOP, 2 N S at 1024^2, all 4608 tokens
PERSTEP_ATTN = 1.487e13       # FLOP, 4 S^2 d * 57
PERSTEP_TOTAL = 1.245e14      # FLOP/step
SCHNELL_4STEP = 4.98e14       # FLOP over 4 steps
SCHNELL_SECONDS = 2.609        # s, the measured Schnell/4 wall-clock
SCHNELL_TFPS = 191.0          # TF/s implied (4.98e14 / 2.609)
DEV28_IMPOSSIBLE_FLOP = 2.75e15  # 28 * 2 * 12e9 * 4096, the WRONG dev/28 reading
DEV28_SECONDS = 2.6           # the mislabeled wall-clock
COMPUTE_MS_PER_STEP = 1585.0  # ms, 2 * 12e9 * 4096 / 62e12  -> D-15 compute-bound proof
BANDWIDTH_MS_FLOOR = 88.0     # ms, 24.0 GB / 273 GB/s  -> a floor nobody is near
SDXL_TFLOP_FWD = 5.977         # TF/forward, SDXL UNet @1024^2 b1 (torch.FlopCounterMode) sec 6.3
SDXL_SECONDS = 8.571          # s/img, SDXL 1.0 bf16, 50 steps, batch 2, TensorRT [VP]
SDXL_STEPS = 50
SDXL_CFG_TFPS = 69.7          # TF/s sustained if CFG on -> leans toward 125 TF ceiling

# The label the course uses for the NVIDIA number EVERYWHERE (constants sec 9.6 / D-16).
NVIDIA_LABEL = "FLUX.1-Schnell, 4 steps, 1024^2, batch 1, FP4"


# --------------------------------------------------------------------------- #
# Part 1 - the frozen arithmetic. No GPU. Asserted, then printed (house rule B.3).
# --------------------------------------------------------------------------- #

def flop_model():
    """Reproduce and ASSERT the FLUX.1-dev per-step FLOP model and both ceilings.

    Returns the per-step FLOP count and the two candidate ridges. Everything here is
    checked against constants.md before it is printed - a regression fails loudly.
    """
    print("=" * 72)
    print("THE FLOP MODEL - one FLUX.1-dev denoising step at 1024^2  [DER, constants sec 9.6]")
    print("=" * 72)

    linear = 2 * FLUX1_N_FLOP * S_TOTAL              # 2 N S over all 4608 tokens
    attn = 4 * S_TOTAL**2 * FLUX1_D * ATTN_LAYER_FACTOR
    total = linear + attn
    assert abs(linear - PERSTEP_LINEAR) / PERSTEP_LINEAR < 0.01, "2NS term drifted from sec 9.6"
    assert abs(attn - PERSTEP_ATTN) / PERSTEP_ATTN < 0.02, "attention term drifted from sec 9.6"
    assert abs(total - PERSTEP_TOTAL) / PERSTEP_TOTAL < 0.01, "per-step total drifted"
    attn_share = 100 * attn / total

    print(f"  linear / matmul   2 N S      = {linear:.3e} FLOP   (N=11.9e9, S={S_TOTAL} tokens)")
    print(f"  attention         4 S^2 d*57 = {attn:.3e} FLOP   ({attn_share:.1f}% of the step)")
    print(f"  ------------------------------------------------")
    print(f"  per denoising step           = {total:.3e} FLOP")
    print(f"  the image-token 2NS lower bound: 2 x 12e9 x {S_IMG} = {2*FLUX1_N_LEDGER*S_IMG:.3e} FLOP")
    print()

    # D-15: is bf16 FLUX compute-bound or bandwidth-bound? Compute wins by ~18x.
    compute_ms = (2 * FLUX1_N_LEDGER * S_IMG) / (CEIL_FP32ACC * TF) * 1e3
    assert abs(compute_ms - COMPUTE_MS_PER_STEP) < 5, "1585 ms/step compute proof drifted"
    stream_ms = (FLUX1_N_LEDGER * 2 / GB) / PUBLISHED_BW * 1e3   # 24.0 GB / 273 GB/s
    assert abs(stream_ms - BANDWIDTH_MS_FLOOR) < 2, "88 ms bandwidth floor drifted"
    print("  IS IT COMPUTE- OR BANDWIDTH-BOUND?  (constants D-15 / sec 8.4 correction)")
    print(f"    compute:  2 x 12e9 x {S_IMG} / {CEIL_FP32ACC:.0f}e12  = {compute_ms:.0f} ms/step")
    print(f"    stream:   24.0 GB / {PUBLISHED_BW:.0f} GB/s          = {stream_ms:.0f} ms/step  (a FLOOR, not the bound)")
    print(f"    -> compute is {compute_ms/stream_ms:.0f}x the streaming floor: FLUX at bf16 is COMPUTE-BOUND.")
    print("       The Spark's weak bandwidth does NOT hurt here. Opposite of LLM decode.")
    print()

    # The ridge (constants sec 6.4) - unresolved between two ceilings.
    ridge_low = CEIL_FP32ACC * TF / (PUBLISHED_BW * GB)
    ridge_high = CEIL_FP16ACC * TF / (PUBLISHED_BW * GB)
    assert round(ridge_low) == RIDGE_LOW, f"ridge(62 TF) != {RIDGE_LOW}"
    assert round(ridge_high) == RIDGE_HIGH, f"ridge(125 TF) != {RIDGE_HIGH}"
    print("  THE RIDGE  I* = P_peak / BW  - [INF], and unsettled (constants sec 6.3/6.4):")
    print(f"    if ~62 TF (FP32-acc)  -> {ridge_low:6.1f} FLOP/byte ~ {RIDGE_LOW}   [INF]")
    print(f"    if ~125 TF (FP16-acc) -> {ridge_high:6.1f} FLOP/byte ~ {RIDGE_HIGH}   [INF]")

    # Where the two workloads land. I = tokens per forward at bf16 (constants sec 6.5).
    i_decode = 1
    i_diffusion = S_IMG
    assert i_decode < RIDGE_LOW < RIDGE_HIGH < i_diffusion, "the D-15 contrast broke"
    print(f"    LLM decode  I = {i_decode}      -> {RIDGE_LOW}x / {RIDGE_HIGH}x BELOW ridge  -> bandwidth-bound (Spark weak)")
    print(f"    diffusion   I = {i_diffusion}   -> {i_diffusion/RIDGE_LOW:.0f}x / {i_diffusion/RIDGE_HIGH:.0f}x ABOVE ridge  -> compute-bound (Spark strong)")
    print("    Same machine, same ridge, opposite verdicts - robust to WHICH ceiling is true.")
    print()
    return total, ridge_low, ridge_high


def honesty_exhibit():
    """The Schnell/4-vs-dev/28 label bug, and the SDXL cross-check. All asserted."""
    print("=" * 72)
    print("THE HONESTY EXHIBIT - what '2.6 s / 1K image' can and cannot mean")
    print("=" * 72)

    # The correct reading: Schnell, 4 steps. Assert the honest FLOP count and throughput.
    schnell_tfps = SCHNELL_4STEP / SCHNELL_SECONDS / TF
    assert abs(schnell_tfps - SCHNELL_TFPS) < 3, "Schnell/4 throughput drifted"
    mfu = 100 * schnell_tfps / CEIL_FP4_DENSE
    print(f'  CORRECT label (use it everywhere):  "{NVIDIA_LABEL} = 23 img/min = 2.6 s/img"  [VP]')
    print(f"    honest count: {SCHNELL_4STEP:.2e} FLOP / {SCHNELL_SECONDS:.3f} s = {schnell_tfps:.0f} TF/s")
    print(f"    = {mfu:.0f}% of the inferred ~{CEIL_FP4_DENSE:.0f} TF dense-FP4 ceiling -> physically plausible.")
    print()

    # The WRONG reading: dev/28. Assert it is impossible. We do NOT present it as OUR prediction.
    dev28_flop = 28 * 2 * FLUX1_N_LEDGER * S_IMG
    assert abs(dev28_flop - DEV28_IMPOSSIBLE_FLOP) / DEV28_IMPOSSIBLE_FLOP < 0.01, "dev/28 FLOP drifted"
    dev28_pflops = dev28_flop / DEV28_SECONDS / 1e15
    pct_of_peak = 100 * (dev28_flop / DEV28_SECONDS) / (MARKETING_FP4_SPARSE * TF)
    assert 105 < pct_of_peak < 107, "dev/28 impossibility check drifted from 106%"
    print(f'  WRONG label (the DELETED showpiece, shown only to refute it):')
    print(f"    read as FLUX.1-dev/28: {dev28_flop:.2e} FLOP / {DEV28_SECONDS:.1f} s = {dev28_pflops:.2f} PFLOP/s")
    print(f"    = {pct_of_peak:.0f}% of the {MARKETING_FP4_SPARSE:.0f} TF sparse-FP4 MARKETING peak - IMPOSSIBLE.")
    print("    dev/28 (numerator) and Schnell/4 (measurement) are unrelated; the 'agreement' was")
    print("    coincidence. This script does NOT reconstruct a 2.6 s prediction. You measure instead.")
    print()

    # SDXL BF16 cross-check on the ceiling (constants sec 6.3 / 9.6).
    sdxl_tfps = SDXL_STEPS * 2 * SDXL_TFLOP_FWD * TF / SDXL_SECONDS / TF   # CFG-on = 2 fwd/step
    assert abs(sdxl_tfps - SDXL_CFG_TFPS) < 1.5, "SDXL CFG-on cross-check drifted"
    print("  SDXL cross-check on the BF16 ceiling (constants sec 6.3):")
    print(f"    SDXL UNet = {SDXL_TFLOP_FWD} TFLOP/forward @1024^2 b1 [DER, torch.FlopCounterMode]")
    print(f"    SDXL 1.0 bf16 = {SDXL_SECONDS:.3f} s/img, {SDXL_STEPS} steps, TensorRT [VP]")
    print(f"    if CFG was ON (2 fwd/step): {sdxl_tfps:.1f} TF/s sustained > 62.5 TF FP32-acc ceiling")
    print(f"    -> would RULE OUT ~62 TF and lean toward ~125 TF. Suggestive, not conclusive (CFG unstated).")
    print()


def predict_dev28(perstep_flop):
    """His own prediction: FLUX.1-dev/28 wall-clock at each candidate ceiling. [INF]."""
    print("=" * 72)
    print("YOUR PREDICTION - FLUX.1-dev, 28 steps, 1024^2, bf16, before you run it  [INF]")
    print("=" * 72)
    print("  These are predictions, NOT frozen facts - they ride on the unresolved 62-vs-125 TF")
    print("  ceiling (+/- a factor of ~2). That is the whole point: you measure, then judge them.")
    print()
    for name, ceil in (("~62 TF (FP32-acc)", CEIL_FP32ACC), ("~125 TF (FP16-acc)", CEIL_FP16ACC)):
        s_step = perstep_flop / (ceil * TF)
        s_img = s_step * 28
        print(f"    at {name:<20} -> {s_step*1e3:7.0f} ms/step   {s_img:6.1f} s / 28-step image  [INF]")
    print()
    print(f"  FP4: no honest wall-clock prediction. The dense-FP4 ceiling (~{CEIL_FP4_DENSE:.0f} TF) is")
    print("  itself an unpublished inference; multiplying two error bars is not a prediction. Measure it.")
    print()


# --------------------------------------------------------------------------- #
# Part 2 - the GPU measurement.
# --------------------------------------------------------------------------- #

def device_card():
    """Print the device and flag ComfyUI contention. Returns (contended, free_gib)."""
    import torch
    print("=" * 72)
    print("DEVICE")
    print("=" * 72)
    cap = torch.cuda.get_device_capability(0)
    print(f"  name          {torch.cuda.get_device_name(0)}")
    print(f"  capability    sm_{cap[0]}{cap[1]}")
    print(f"  torch         {torch.__version__}   CUDA {torch.version.cuda}")
    print(f"  bf16 support  {torch.cuda.is_bf16_supported()}")
    print(f"  built for     {torch.cuda.get_arch_list()}")
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        print(f"  note: no sm_{cap[0]}{cap[1]} cubin and no PTX - runs via CUDA minor-version binary")
        print(f"        compatibility (an sm_{cap[0]}0 kernel on an sm_{cap[0]}{cap[1]} device). Not a fault.")
    free, tot = torch.cuda.mem_get_info()
    print(f"  memory        {tot/GiB:.2f} GiB total | {free/GiB:.2f} GiB free  (unified)")
    contended = free < 0.5 * tot
    if contended:
        print()
        print("  !! Over half your memory is already in use - ComfyUI or a training run is live.")
        print("     Contention drags every number below DOWN and can make a compute-bound job")
        print("     read as bandwidth-bound. Treat low numbers as inconclusive; re-run idle to trust.")
    print()
    return contended, free / GiB


def measure_bandwidth(quick):
    """Achieved bandwidth via a device-to-device copy. Published 273 GB/s is theoretical."""
    import torch
    print("=" * 72)
    print("BANDWIDTH - achieved, not theoretical")
    print("=" * 72)
    n = (256 if quick else 1024) * 1024 * 1024 // 4          # float32 elements
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
    moved = 2 * src.numel() * 4                              # read + write
    bw = moved / secs / GB
    print(f"  copy {moved/GB:.2f} GB   {secs*1e3:7.2f} ms   {bw:7.1f} GB/s")
    print(f"  --> achieved {bw:.1f} of {PUBLISHED_BW:.0f} GB/s published = {100*bw/PUBLISHED_BW:.0f}% of peak")
    print()
    del src, dst
    torch.cuda.empty_cache()
    return bw


def measure_ridge_gemm(quick):
    """Big dense bf16 GEMM in both accumulate modes - settle the ridge yourself.

    Mirrors 09_spark_capability_probe.py: the one documented PyTorch knob that touches bf16
    accumulation precision is allow_bf16_reduced_precision_reduction (ON = fp16-style split-K
    reduction, OFF = full fp32 reduction). If both columns match, cuBLAS picked one kernel and
    that is itself the finding. Returns peak measured TFLOP/s.
    """
    import torch
    print("=" * 72)
    print("RIDGE GEMM - dense bf16, both accumulate modes (settle 62 vs 125 TF)")
    print("=" * 72)
    sizes = [4096, 8192] if quick else [4096, 8192, 12288]
    knob = "allow_bf16_reduced_precision_reduction"
    has_knob = hasattr(torch.backends.cuda.matmul, knob)
    modes = [("fp32_acc (reduction OFF)", False), ("fp16_acc (reduction ON)", True)]
    if not has_knob:
        print(f"  note: torch.backends.cuda.matmul.{knob} absent on this build - default path only.")
        modes = [("default path", None)]
    iters = 5 if quick else 20
    peak = 0.0
    for label, flag in modes:
        if has_knob and flag is not None:
            setattr(torch.backends.cuda.matmul, knob, flag)
        print(f"\n  -- {label} --")
        for n in sizes:
            try:
                a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            except torch.cuda.OutOfMemoryError:
                print(f"     {n:>6} x {n:<6}  OOM - skipped")
                continue
            for _ in range(3):
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
            tf = (2 * n**3) / secs / TF
            peak = max(peak, tf)
            print(f"     {n:>6} x {n:<6}  {tf:8.2f} TFLOP/s")
            del a, b
            torch.cuda.empty_cache()
    if has_knob:
        setattr(torch.backends.cuda.matmul, knob, True)         # restore default
    d62, d125 = abs(peak - CEIL_FP32ACC), abs(peak - CEIL_FP16ACC)
    nearest = "~62 TF (FP32-acc)" if d62 <= d125 else "~125 TF (FP16-acc)"
    print(f"\n  peak dense bf16 {peak:8.2f} TFLOP/s  -> nearest ceiling {nearest}")
    print(f"     which points at the ridge ~{RIDGE_LOW if d62 <= d125 else RIDGE_HIGH} "
          f"(the {RIDGE_LOW}/{RIDGE_HIGH} fork, constants sec 6.4).")
    print()
    return peak


def microbench_dit_step(perstep_flop, quick):
    """Time the DOMINANT compute of one FLUX DiT step (the 2NS linear term), fwd and fwd+bwd.

    We do not load the model here - we build a stack of matmuls sized to FLUX.1-dev's real
    per-step linear FLOP over S=4608 tokens at width d=3072, so the FLOP count is honest and
    the timing is a faithful proxy for one denoising step's arithmetic. bf16 is measured;
    fp8 is measured only if this torch build exposes float8 scaled matmul; fp4 is PREDICTED
    from the ceiling and clearly labelled [INF] - torch has no native fp4 GEMM and faking a
    measurement would launder an estimate into a fact.
    """
    import torch
    print("=" * 72)
    print("DiT-STEP MICRO-BENCH - one step's linear compute, fwd and fwd+bwd")
    print("=" * 72)
    d = FLUX1_D
    S = S_TOTAL
    # A single (S x d) @ (d x 4d) @ (4d x d) MLP-shaped pass carries ~2*S*d*8d = 16 S d^2 FLOP.
    # We size a stack of such blocks so the measured FLOP matches the model's 2NS per step.
    block_flop = 2 * S * d * (4 * d) + 2 * S * (4 * d) * d       # up then down projection
    nblocks = max(1, round(PERSTEP_LINEAR / block_flop))
    measured_flop = nblocks * block_flop
    print(f"  proxy: {nblocks} MLP blocks of (S={S} x d={d}), total {measured_flop:.2e} FLOP")
    print(f"         (matches the model's 2NS linear term {PERSTEP_LINEAR:.2e} within "
          f"{100*abs(measured_flop-PERSTEP_LINEAR)/PERSTEP_LINEAR:.0f}%)")
    try:
        x = torch.randn(S, d, device="cuda", dtype=torch.bfloat16)
        w_up = [torch.randn(d, 4 * d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                for _ in range(nblocks)]
        w_dn = [torch.randn(4 * d, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                for _ in range(nblocks)]
    except torch.cuda.OutOfMemoryError:
        print("  OOM building the proxy - free the GPU (stop ComfyUI) and retry.")
        return

    def _forward():
        h = x
        for wu, wd in zip(w_up, w_dn):
            h = torch.nn.functional.gelu(h @ wu) @ wd
        return h

    def _time(fn, iters):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000 / iters

    iters = 3 if quick else 8
    t_fwd = _time(lambda: _forward().sum(), iters)

    def _fwd_bwd():
        out = _forward().sum()
        out.backward()
        for wu, wd in zip(w_up, w_dn):
            wu.grad = None
            wd.grad = None

    t_fb = _time(_fwd_bwd, iters)
    tf_fwd = measured_flop / t_fwd / TF
    print(f"  bf16 forward       {t_fwd*1e3:8.1f} ms/step   {tf_fwd:7.1f} TFLOP/s achieved")
    print(f"  bf16 fwd+bwd       {t_fb*1e3:8.1f} ms/step   ({t_fb/t_fwd:.1f}x forward - the 6ND vs 2ND training factor)")
    mfu62 = 100 * tf_fwd / CEIL_FP32ACC
    mfu125 = 100 * tf_fwd / CEIL_FP16ACC
    print(f"  -> forward MFU: {mfu62:.0f}% of 62 TF  /  {mfu125:.0f}% of 125 TF")

    # fp4 prediction only - honest, labelled, never a fake measurement.
    pred_fp4_ms = perstep_flop / (CEIL_FP4_DENSE * TF) * 1e3
    print(f"  fp4 forward        [INF prediction, not measured]: ~{pred_fp4_ms:.0f} ms/step at the")
    print(f"                     ~{CEIL_FP4_DENSE:.0f} TF dense-FP4 ceiling (+/- a factor of 2 - unpublished).")
    print()
    del x, w_up, w_dn
    torch.cuda.empty_cache()


def _load_flux(model_id, single_file):
    """Load a FLUX.1 pipeline via diffusers (v0.39.0 API). Returns pipe or raises."""
    import torch
    from diffusers import FluxPipeline                       # verified: diffusers 0.39.0 (constants sec 7)
    if single_file:
        pipe = FluxPipeline.from_single_file(single_file, torch_dtype=torch.bfloat16)
    else:
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    # Unified+coherent memory: offload is nearly free on the Spark (no PCIe copy). constants/brief sec 8.3.
    pipe.enable_model_cpu_offload()
    return pipe


def generate_flux(model_id, single_file, steps, perstep_flop, free_gib, force, outdir):
    """Actually generate one FLUX.1-dev image at `steps` and time each denoising step."""
    import time
    import torch
    print("=" * 72)
    print(f"GENERATE - FLUX.1-dev, {steps} steps, 1024^2, bf16 (times each step)")
    print("=" * 72)
    need_gib = FLUX1_N_LEDGER * 2 / GiB + 8                  # ~24 GB transformer + headroom
    if free_gib < need_gib and not force:
        print(f"  SKIPPED: only {free_gib:.1f} GiB free, need ~{need_gib:.0f} GiB for the transformer.")
        print("  ComfyUI is almost certainly holding the GPU. Stop it and re-run, or pass --force")
        print("  to try anyway (it will contend and the numbers will be pessimistic).")
        print()
        return
    try:
        pipe = _load_flux(model_id, single_file)
    except Exception as e:                                    # gated repo, missing weights, API drift
        print(f"  Could not load a FLUX pipeline: {type(e).__name__}: {e}")
        print("  On the Spark, either accept the black-forest-labs/FLUX.1-dev license on HF and")
        print("  `huggingface-cli login`, or point --single-file at your ComfyUI checkpoint, e.g.")
        print("    --single-file ~/ComfyUI/models/diffusion_models/flux1-dev.safetensors")
        print("  Your PREDICTION above still stands; this step just fills in the measured number.")
        print()
        return

    step_times = []
    last = {"t": None}

    def _cb(pipe_, step, timestep, cb_kwargs):
        now = time.perf_counter()
        if last["t"] is not None:
            step_times.append(now - last["t"])
        last["t"] = now
        return cb_kwargs

    gen = torch.Generator(device="cuda").manual_seed(42)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    image = pipe(
        "a lighthouse on a cliff at golden hour, photorealistic",
        num_inference_steps=steps,
        height=1024,
        width=1024,
        guidance_scale=3.5,                                  # FLUX.1-dev's distilled guidance (NOT CFG)
        generator=gen,
        callback_on_step_end=_cb,
    ).images[0]
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    if step_times:
        s_step = sum(step_times) / len(step_times)
    else:
        s_step = wall / steps
    implied_tfps = perstep_flop / s_step / TF
    peak_gib = torch.cuda.max_memory_allocated() / GiB
    print(f"  wall clock         {wall:8.2f} s / image ({steps} steps)")
    print(f"  per step (median-ish) {s_step*1e3:7.0f} ms/step")
    print(f"  implied throughput {implied_tfps:8.1f} TFLOP/s  (model {perstep_flop:.2e} FLOP/step / measured s)")
    print(f"  peak GPU memory    {peak_gib:8.2f} GiB")
    print()
    print(f"  vs your prediction: {100*implied_tfps/CEIL_FP32ACC:.0f}% of the 62 TF ceiling, "
          f"{100*implied_tfps/CEIL_FP16ACC:.0f}% of the 125 TF ceiling.")
    print("  A believable MFU (~30-60%) that lands well ABOVE the ridge confirms the D-15 verdict:")
    print("  diffusion is compute-bound on your box. (If it reads as bandwidth-bound, ComfyUI is up.)")
    if outdir:
        import os
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "bench_spark_flux.png")
        image.save(path)
        print(f"  wrote {path}")
    print()


# --------------------------------------------------------------------------- #
# Self-test - no GPU, no torch. Runs on this Windows box and in CI.
# --------------------------------------------------------------------------- #

def self_test():
    print("SELF-TEST - asserting the frozen diffusion arithmetic against constants.md")
    perstep, ridge_low, ridge_high = flop_model()   # asserts the FLOP model, ceilings, ridges
    honesty_exhibit()                               # asserts Schnell/4, dev/28-impossible, SDXL
    predict_dev28(perstep)                          # prints predictions (labelled [INF])

    # A few explicit end-to-end invariants the page rides on.
    assert round(ridge_low) == RIDGE_LOW and round(ridge_high) == RIDGE_HIGH
    assert S_IMG > RIDGE_HIGH > RIDGE_LOW > 1, "diffusion must sit above both ridges, decode below"
    # The honest Schnell/4 reading is plausible; the dev/28 reading is impossible.
    assert SCHNELL_TFPS < CEIL_FP4_DENSE, "Schnell/4 must be below the FP4 ceiling"
    assert (DEV28_IMPOSSIBLE_FLOP / DEV28_SECONDS) > MARKETING_FP4_SPARSE * TF, \
        "the dev/28 reading must exceed the marketing peak (that's why it's impossible)"
    # NVIDIA number is labelled correctly and we never claim a dev/28 wall-clock.
    assert "Schnell" in NVIDIA_LABEL and "4 steps" in NVIDIA_LABEL
    print("SELF-TEST PASSED - FLOP model, 62/125/500 TF ceilings, ridges 227/458, Schnell/4 label,")
    print("                   dev/28 impossibility, SDXL 5.977 TFLOP cross-check, I=4096 verdict.")


def main():
    ap = argparse.ArgumentParser(
        description="Diffusion-side roofline, measured on YOUR GB10 (D-15 / constants sec 9.6).")
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev",
                    help="HF repo id for the FLUX.1 pipeline (gated; accept the license first)")
    ap.add_argument("--single-file", default=None,
                    help="path to a local FLUX.1 checkpoint (e.g. your ComfyUI flux1-dev.safetensors)")
    ap.add_argument("--steps", type=int, default=28, help="denoising steps (FLUX.1-dev default 28)")
    ap.add_argument("--precision", default="bf16,fp4",
                    help="informational: which precisions the page discusses (bf16 measured, fp4 predicted)")
    ap.add_argument("--outdir", default=None, help="if set, save the generated image here")
    ap.add_argument("--quick", action="store_true", help="smaller sweeps / fewer iters")
    ap.add_argument("--no-generate", action="store_true", help="skip the real FLUX generation")
    ap.add_argument("--force", action="store_true", help="run generation even if memory looks tight")
    ap.add_argument("--no-gpu", action="store_true", help="prediction arithmetic only, no GPU")
    ap.add_argument("--self-test", action="store_true", help="frozen arithmetic assertions only (no torch)")
    args = ap.parse_args()

    print("#" * 72)
    print("# bench_spark.py - the diffusion side of the roofline, measured on YOUR box")
    print("#" * 72)
    print()

    if args.self_test:
        self_test()
        return

    perstep, _, _ = flop_model()
    honesty_exhibit()
    predict_dev28(perstep)

    if args.no_gpu:
        print("  (--no-gpu) Skipping the measurement. On the Spark, run without --no-gpu to time")
        print("  the DiT step, settle the ridge with a bf16 GEMM, and generate a real image -")
        print("  then check the predictions above against your own numbers.")
        return

    try:
        import torch
    except ImportError:
        print("torch not importable here. On the Spark, use a torch-bearing venv, e.g.:")
        print("  ~/course/.venv/bin/python bench_spark.py")
        print("  (or ComfyUI's venv for a read-only probe: ~/ComfyUI/.venv/bin/python bench_spark.py)")
        print("This script installs nothing and writes nothing (except --outdir).")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available - stopping. Run this on the Spark.")
        sys.exit(1)

    torch.manual_seed(42)                                    # seeded (house rule B.7)
    contended, free_gib = device_card()
    measure_bandwidth(args.quick)
    measure_ridge_gemm(args.quick)
    microbench_dit_step(perstep, args.quick)
    if not args.no_generate:
        generate_flux(args.model, args.single_file, args.steps, perstep,
                      free_gib, args.force, args.outdir)
    if contended:
        print("REMINDER: ComfyUI was live during this run. Every number above is pessimistic;")
        print("re-run with the GPU idle before trusting the ridge verdict.")


if __name__ == "__main__":
    main()
