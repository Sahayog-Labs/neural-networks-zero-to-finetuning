#!/usr/bin/env python3
"""
measure_your_box.py — find YOUR machine's roofline. Don't trust the spec sheet.

Course artifact for p.43 ("Your box: 273 GB/s, ridge ~227"). The published numbers for a
GB10 are unhelpful on purpose: NVIDIA quotes "1 PFLOP" of FP4 *with sparsity*, which is not
a number you can ever hit with a bf16 training step. The dense bf16 peak is unpublished.
So we measure it.

Two numbers come out, and the whole LLM-vs-diffusion story hangs off their ratio:

    ridge point  I* = peak_FLOPs / peak_bandwidth     [FLOP/byte]

Below I*, you are memory-bound: the machine is starving, waiting on DRAM.
Above I*, you are compute-bound: the machine is actually working.
LLM decode sits far below. Diffusion sits far above. Same box, opposite verdicts.

Usage
-----
    python measure_your_box.py              # everything
    python measure_your_box.py --quick      # smaller sweep, ~20s
    python measure_your_box.py --no-gpu     # memory facts only, no GPU contention

SAFETY: allocates a few GB of VRAM and saturates the GPU for a few seconds per size.
If something else is using the GPU (ComfyUI, a training run), this will contend with it.
Nothing is written and nothing is installed; it is read-only w.r.t. your system.
"""

import argparse
import sys

# --------------------------------------------------------------------------- #
# Part 1 — memory facts. No GPU needed. This is the p.18 ledger's reality check.
# --------------------------------------------------------------------------- #

GiB = 1 << 30
GB = 10**9


def memory_facts():
    print("=" * 68)
    print("MEMORY — what you actually have, versus what the box was sold as")
    print("=" * 68)

    total_b = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k == "MemTotal":
                    total_b = int(v.split()[0]) * 1024
                elif k == "MemAvailable":
                    avail_b = int(v.split()[0]) * 1024
    except FileNotFoundError:
        print("  (no /proc/meminfo — not Linux? skipping)")
        return

    print(f"  MemTotal      {total_b:>18,} B")
    print(f"                {total_b/GB:>18.2f} GB  (decimal — how it's marketed)")
    print(f"                {total_b/GiB:>18.4f} GiB (binary  — how DRAM actually works)")
    print(f"  MemAvailable  {avail_b/GiB:>18.2f} GiB (what you can have RIGHT NOW)")

    # "128 GB LPDDR5X" means 128 GiB of physical parts. Compare like with like.
    physical = 128 * GiB
    reserved = physical - total_b
    print()
    print(f"  Sold as       {physical/GiB:>18.2f} GiB physical")
    print(f"  You see       {total_b/GiB:>18.4f} GiB")
    print(f"  MISSING       {reserved/GiB:>18.4f} GiB  <-- firmware/driver carveout")
    print(f"                {100*reserved/physical:>18.2f} %   of the box, gone before you start")

    # The Qwen3-8B full-fine-tune ledger from constants.md §163.
    need_gib = 122.05
    print()
    print("  The p.18 question: does a full AdamW fine-tune of Qwen3-8B fit?")
    print(f"    need   {need_gib:>8.2f} GiB   (16 B/param x 8.19e9 params)")
    print(f"    have   {total_b/GiB:>8.2f} GiB")
    slack = total_b / GiB - need_gib
    verdict = "FITS" if slack > 0 else "DOES NOT FIT"
    print(f"    slack  {slack:>+8.2f} GiB   --> {verdict}")
    if slack < 0:
        print()
        print(f"    ...and that is BEFORE activations (+2-6 GB). It is not close.")
        print(f"    This is why LoRA exists. Not to compress the model —")
        print(f"    to delete the optimizer state.")
    print()


# --------------------------------------------------------------------------- #
# Part 2 — the two roofline numbers.
# --------------------------------------------------------------------------- #

def gpu_facts():
    import torch

    print("=" * 68)
    print("DEVICE")
    print("=" * 68)
    cap = torch.cuda.get_device_capability(0)
    print(f"  name          {torch.cuda.get_device_name(0)}")
    print(f"  capability    sm_{cap[0]}{cap[1]}")
    print(f"  torch         {torch.__version__}")
    print(f"  built for     {torch.cuda.get_arch_list()}")

    # The aarch64 trap, made visible.
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        print()
        print(f"  !! Your torch ships NO binary for sm_{cap[0]}{cap[1]} and no PTX.")
        print(f"     It runs via CUDA minor-version binary compatibility "
              f"(an sm_{cap[0]}0 cubin on an sm_{cap[0]}{cap[1]} device).")
        print(f"     It works — but nothing here was compiled for your GPU.")

    free, tot = torch.cuda.mem_get_info()
    print()
    print(f"  mem_get_info  total {tot/GiB:.2f} GiB | free {free/GiB:.2f} GiB")
    if free < 0.5 * tot:
        print(f"  !! Over half your memory is already spoken for. Something else is running.")
        print(f"     The numbers below will be pessimistic and noisy.")
    print()


def measure_flops(quick=False):
    """Dense bf16 matmul peak. NVIDIA does not publish this for GB10."""
    import torch

    print("=" * 68)
    print("COMPUTE — dense bf16 matmul (the number NVIDIA won't print)")
    print("=" * 68)

    sizes = [4096, 8192] if quick else [4096, 8192, 12288, 16384]
    best = 0.0
    for n in sizes:
        try:
            a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        except torch.cuda.OutOfMemoryError:
            print(f"  {n:>6} : OOM — skipping (something else is holding memory)")
            continue

        for _ in range(3):                      # warm up: let clocks ramp
            (a @ b)
        torch.cuda.synchronize()

        iters = 5 if quick else 20
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            (a @ b)
        end.record()
        torch.cuda.synchronize()

        secs = start.elapsed_time(end) / 1000 / iters
        flops = 2 * n**3                        # multiply-add = 2 FLOP
        tf = flops / secs / 1e12
        best = max(best, tf)
        print(f"  {n:>6} x {n:<6} {secs*1000:>8.2f} ms   {tf:>8.2f} TFLOP/s")
        del a, b
        torch.cuda.empty_cache()

    print(f"\n  --> peak measured: {best:.1f} TFLOP/s dense bf16")
    print(f"      compare: the marketing number is 1000 TFLOP/s (FP4, *with sparsity*).")
    print(f"      you are {1000/best:.0f}x below it, and that is expected, not a fault.")
    return best


def measure_bandwidth(quick=False):
    """Achieved bandwidth. Published 273 GB/s is a theoretical peak; nobody hits it."""
    import torch

    print()
    print("=" * 68)
    print("BANDWIDTH — achieved, not theoretical")
    print("=" * 68)

    n = (256 if quick else 1024) * 1024 * 1024 // 4      # float32 elements
    try:
        src = torch.empty(n, device="cuda", dtype=torch.float32)
        dst = torch.empty(n, device="cuda", dtype=torch.float32)
    except torch.cuda.OutOfMemoryError:
        print("  OOM — not enough free memory to measure. Free the GPU and retry.")
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
    moved = 2 * src.numel() * 4                 # read + write
    bw = moved / secs / 1e9
    print(f"  copy {moved/GB:.2f} GB   {secs*1000:>7.2f} ms   {bw:>7.1f} GB/s")

    published = 273.0
    print(f"\n  --> achieved {bw:.1f} GB/s of {published:.0f} GB/s published "
          f"= {100*bw/published:.0f}% of peak")
    print(f"      This fraction is the honest coefficient. Any heuristic of the form")
    print(f"      'tok/s ~= k x BW / model_GB' must use k ~= {bw/published:.2f}, not 1.0,")
    print(f"      and only in the batch-1 decode regime.")
    del src, dst
    torch.cuda.empty_cache()
    return bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sweep")
    ap.add_argument("--no-gpu", action="store_true", help="memory facts only")
    args = ap.parse_args()

    memory_facts()
    if args.no_gpu:
        return

    try:
        import torch
    except ImportError:
        print("torch not importable. On the Spark, ComfyUI's venv has one:")
        print("  ~/ComfyUI/.venv/bin/python measure_your_box.py")
        print("(read-only — this script installs nothing and writes nothing)")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available — stopping.")
        sys.exit(1)

    gpu_facts()
    tf = measure_flops(args.quick)
    bw = measure_bandwidth(args.quick)

    if tf and bw:
        ridge = tf * 1e12 / (bw * 1e9)
        print()
        print("=" * 68)
        print("YOUR ROOFLINE")
        print("=" * 68)
        print(f"  peak compute    {tf:>8.1f} TFLOP/s  (bf16, dense, measured)")
        print(f"  peak bandwidth  {bw:>8.1f} GB/s     (achieved, measured)")
        print(f"  RIDGE  I*     = {ridge:>8.1f} FLOP/byte")
        print()
        print(f"  An LLM decoding one token at a time has I = 1 FLOP/byte at bf16")
        print(f"  (2 FLOP per weight / 2 bytes per weight).")
        print(f"  That is {ridge:.0f}x below your ridge: memory-bound, badly. The GPU idles")
        print(f"  while DRAM struggles. Batching is the only lever that moves it.")
        print()
        print(f"  A diffusion step at 1024x1024 has I in the hundreds to thousands.")
        print(f"  That is above your ridge: compute-bound. The GPU is genuinely working.")
        print()
        print(f"  Same machine. Same formula. Opposite verdicts.")
        print(f"  That is why this box feels fast for images and slow for chat —")
        print(f"  and it was predictable from one division.")


if __name__ == "__main__":
    main()
