#!/usr/bin/env python3
"""
05_memory_ledger.py -- the memory calculator, as a program. Course RUNG 9 (p.49).

You have seen this ledger three times now: as a table on p.18, as a live bar on p.49, and
as the sentence "131 GB does not fit." This is that arithmetic made runnable -- point it at
ANY HuggingFace model id (or a local config.json) and it computes the four-bucket training
ledger straight from the config, reproduces the Qwen3-8B escape ladder to the byte, checks
its own numbers against constants.md, and then -- if you dare -- ACTUALLY ALLOCATES the 8B
full-fine-tune state and watches it OOM on your box. Making yourself run that OOM is the
emotional core of the LLM track: full fine-tuning an 8B does not fit on a Spark, and you
will have proved it with your own hands, not been told.

The four buckets (constants.md sec2.1), never conflated:
  W  weights     bf16, what the GPU computes with
  G  gradients   bf16, one per trainable weight
  O  optimizer   fp32 master + Adam m + Adam v  (this is the bucket LoRA deletes)
  A  activations [EST] 2-6 GB -- a SEPARATE, ESTIMATED line you MEASURE, never a fact,
                 and one LoRA does NOT shrink (constants.md sec2.4)

All arithmetic lives in utils/ledger.py, imported verbatim by the p.18 companion too, so
the page and this program can never drift.

Usage
-----
    python 05_memory_ledger.py                              # Qwen3-8B / full / adamw_mixed
    python 05_memory_ledger.py --model Qwen/Qwen3-8B --method full --optimizer adamw_mixed \
                               --batch 1 --seq 2048         # the canonical invocation
    python 05_memory_ledger.py --method lora --rank 16      # watch 4 of 5 segments collapse
    python 05_memory_ledger.py --method qlora               # NF4 base + adapter
    python 05_memory_ledger.py --all-methods                # the whole escape ladder
    python 05_memory_ledger.py --config ./config.json       # any local HF config
    python 05_memory_ledger.py --self-test                  # arithmetic + assertions, no GPU
    python 05_memory_ledger.py --attempt-oom                # allocate the 8B state until it dies

Before you run this on the Spark, run the setup checks the p.49 page prescribes:
    python 00_verify_env.py        # asserts torch, CUDA 13, sm_121 (12,1), bf16, bitsandbytes
    python measure_your_box.py     # your real MemTotal and roofline (p.43)
This script's box-touching paths (--measure-budget, --attempt-oom) read the same
mem_get_info() those do.

SAFETY
------
The default run, --self-test and --all-methods are PURE INTEGER ARITHMETIC: no torch, no
GPU, no network, nothing written or installed. They run anywhere, including this laptop.

--attempt-oom is different and deliberately dangerous: it allocates GPU memory in a loop
UNTIL the allocator raises OutOfMemoryError. On the Spark's UNIFIED memory that pressure is
shared with everything else -- if ComfyUI is up it can thrash or be killed. Only run
--attempt-oom on a box you control, with ComfyUI down, in the FRESH course venv
(NEVER ComfyUI's venv -- hardware-ground-truth sec3). This file never sshes anywhere and
never runs the GPU path for you; you run it, on your box, having read this paragraph.
"""

import argparse
import json
import os
import sys

# Make utils/ importable no matter the working directory (agent/CI safe).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.ledger import (  # noqa: E402
    GiB, GB, KiB, MB,
    OPT_STATE_BYTES, BASE_WEIGHT_BYTES, ALL_LINEAR,
    count_params, lora_params, bytes_per_param, full_ft_state,
    full_state_buckets, lora_state, kv_bytes, trainable_state_bytes, normalize,
)

# --------------------------------------------------------------------------- #
# Frozen configs so the default run and self-test work OFFLINE (no HF fetch on this box).
# Qwen3-8B: constants.md sec1.1 [VP, fetched from HF 2026-07-16].
# Llama-3.3-70B: constants.md sec3 [VP, fetched] -- the 70B QLoRA capability demo.
# --------------------------------------------------------------------------- #

QWEN3_8B = dict(
    model_type="qwen3", hidden_size=4096, num_hidden_layers=36,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    intermediate_size=12288, vocab_size=151936, tie_word_embeddings=False,
)
LLAMA_33_70B = dict(
    model_type="llama", hidden_size=8192, num_hidden_layers=80,
    num_attention_heads=64, num_key_value_heads=8, head_dim=128,
    intermediate_size=28672, vocab_size=128256, tie_word_embeddings=False,
)
FROZEN = {
    "qwen/qwen3-8b": QWEN3_8B, "qwen3-8b": QWEN3_8B, "qwen3": QWEN3_8B,
    "meta-llama/llama-3.3-70b-instruct": LLAMA_33_70B,
    "llama-3.3-70b": LLAMA_33_70B, "llama3.3-70b": LLAMA_33_70B,
}

# Frozen values every path asserts against (constants.md, cited inline).
FULL_STATE_B = 131_051_765_760       # 8,190,735,360 * 16          (sec2.2)
FULL_STATE_GB = 131.05
FULL_STATE_GIB = 122.05
LORA_STATE_B = 17_079_822_336        # 16.38 GB base + 0.70 GB adapter (sec2.3)
LORA_STATE_GB = 17.08
LORA_STATE_GIB = 15.91
RATIO_187 = 187.7                    # P / lora_params = trainable-state shrink (sec2.3)
QWEN3_P = 8_190_735_360             # (sec1.2)
QWEN3_LORA = 43_646_976            # r=16 all-linear (sec3)
LLAMA70_LORA = 207_093_760        # r=16 all-linear (sec3, the 70B demo)
MEASURED_MEMTOTAL_GIB = 121.6875   # his box, [MEA-DEV] hardware-ground-truth sec2 -- NOT 128
BUDGET_SLACK = -0.36               # 121.69 - 122.05, before activations (sec6.8)
ACT_EST_LOW_GB, ACT_EST_HIGH_GB = 2.0, 6.0   # [EST] B=1,S=2048, grad-checkpointing (sec2.2)

# The optimizer bytes/param table, in the escape-ladder order (constants.md sec2.3).
LADDER = [
    ("full", "adamw_mixed", "Full, AdamW mixed"),
    ("full", "adam8bit", "Full, 8-bit Adam"),
    ("full", "adam8bit_no_master", "Full, 8-bit Adam, no fp32 master"),
    ("full", "sgd_momentum", "SGD + momentum"),
    ("lora", "adamw_mixed", "LoRA r=16, all-linear"),
    ("qlora", "adamw_mixed", "QLoRA r=16 (NF4 base, DQ on)"),
]


def bar(width_frac, n=48):
    """A tiny text bar so the terminal echoes the p.49 live bar."""
    filled = max(0, min(n, round(width_frac * n)))
    return "#" * filled + "." * (n - filled)


# --------------------------------------------------------------------------- #
# Config resolution.
# --------------------------------------------------------------------------- #

def resolve_config(model, config_path):
    """Return (cfg_dict, label, source). Prefer an explicit --config; else a frozen model;
    else try transformers AutoConfig (works only on a box with the model cached / online)."""
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg, os.path.basename(config_path), f"local config.json ({config_path})"
    key = (model or "").strip().lower()
    if key in FROZEN:
        return FROZEN[key], model, "frozen in-course config [VP, constants.md]"
    # Last resort: ask transformers. Not available on this offline laptop.
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model).to_dict()
        return cfg, model, "transformers AutoConfig (fetched/cached)"
    except Exception as e:  # noqa: BLE001 -- report and guide, never crash the arithmetic
        raise SystemExit(
            f"\nCould not resolve config for {model!r}: {type(e).__name__}: {e}\n"
            f"  This laptop has no network/transformers. Options:\n"
            f"    - use a frozen model: Qwen/Qwen3-8B  or  Llama-3.3-70B\n"
            f"    - pass a local file:  --config path/to/config.json\n"
            f"    - or run on the Spark where the model is cached.\n"
        )


# --------------------------------------------------------------------------- #
# The four-bucket ledger for one (method, optimizer) choice.
# --------------------------------------------------------------------------- #

def compute_ledger(cfg, method, optimizer, rank, base="nf4_dq"):
    """Return an ordered list of (bucket_name, bytes) for W, G, O (state only) plus the
    total, trainable-param count, and P. Activations are handled separately (estimate)."""
    P = count_params(cfg)["total"]
    if method == "full":
        b = full_state_buckets(P, optimizer)
        buckets = [("W  weights   (bf16)", b["weight_bytes"]),
                   ("G  gradients (bf16)", b["grad_bytes"]),
                   (f"O  optimizer ({optimizer})", b["opt_bytes"])]
        return buckets, b["total"], P, P
    # LoRA / QLoRA: frozen base + trainable adapter.
    base_prec = "bf16" if method == "lora" else base
    s = lora_state(cfg, r=rank, optimizer=optimizer, base=base_prec)
    label = "bf16" if method == "lora" else "NF4+DQ"
    buckets = [
        (f"W  base weights  ({label}, frozen)", s["base_weight_bytes"]),
        ("W  adapter wts   (bf16, trainable)", s["adapter_weight_bytes"]),
        ("G  adapter grads (bf16)", s["grad_bytes"]),
        (f"O  adapter opt   ({optimizer})", s["opt_bytes"]),
    ]
    return buckets, s["total"], s["trainable_params"], s["P"]


def print_ledger(cfg, label, source, method, optimizer, rank, batch, seq, budget_gib):
    buckets, total, trainable, P = compute_ledger(cfg, method, optimizer, rank)

    print("=" * 74)
    print(f"THE FOUR-BUCKET MEMORY LEDGER -- {label}  ({method}, {optimizer})")
    print("=" * 74)
    print(f"  config source : {source}")
    print(f"  parameters    : {P:,}   (trainable this run: {trainable:,}"
          f" = {100 * trainable / P:.3f}% of P)")
    print(f"  regime        : batch={batch}  seq={seq}")
    print()
    print("  STATE (what must be resident before a single activation) -- W / G / O:")
    for name, b in buckets:
        frac = b / total if total else 0
        print(f"    {name:<38} {b / GB:8.2f} GB  {b / GiB:8.2f} GiB  {bar(frac, 30)}")
    print(f"    {'-' * 38} {'-' * 8}     {'-' * 8}")
    print(f"    {'STATE TOTAL (W+G+O)':<38} {total / GB:8.2f} GB  {total / GiB:8.2f} GiB")
    print()
    print(f"  A  activations                         "
          f"[EST] {ACT_EST_LOW_GB:.0f}-{ACT_EST_HIGH_GB:.0f} GB   <-- NOT a fact.")
    print(f"     A separate, ESTIMATED line (B={batch}, S={seq}, grad-checkpointing on).")
    print(f"     LoRA does NOT shrink it -- you still backprop the full graph (sec2.4).")
    print(f"     MEASURE it yourself:  torch.cuda.max_memory_allocated() / 2**30")
    print()

    # The verdict against HIS measured budget -- never a page-asserted number.
    slack = budget_gib - total / GiB
    verdict = "FITS" if slack > 0 else "DOES NOT FIT"
    print(f"  VERDICT against your measured budget:")
    print(f"    state need     {total / GiB:8.2f} GiB")
    print(f"    usable budget  {budget_gib:8.4f} GiB")
    print(f"    slack          {slack:+8.2f} GiB   -->  {verdict}")
    if slack < 0:
        real_low = -slack + ACT_EST_LOW_GB * GB / GiB
        real_high = -slack + ACT_EST_HIGH_GB * GB / GiB
        print(f"    ...and that is BEFORE activations. Real shortfall: "
              f"{real_low:.1f}-{real_high:.1f} GiB.")
    print()
    return total, trainable, P


# --------------------------------------------------------------------------- #
# The full escape ladder -- reproduce the p.49 table programmatically.
# --------------------------------------------------------------------------- #

def print_escape_ladder(cfg, rank, budget_gib):
    print("=" * 74)
    print("THE ESCAPE LADDER -- every rung computed from the config (p.49 table)")
    print("=" * 74)
    print(f"  {'Method':<34}{'State GB':>10}{'State GiB':>11}{'vs budget':>14}")
    print(f"  {'-' * 34}{'-' * 10}{'-' * 11}{'-' * 14}")
    for method, optimizer, name in LADDER:
        _, total, _, _ = compute_ledger(cfg, method, optimizer, rank)
        slack = budget_gib - total / GiB
        tag = "fits" if slack > 0 else f"{slack:+.2f} -- no"
        mult = ""
        # multiplier vs the full adamw_mixed rung, for the LoRA/QLoRA rows
        if method in ("lora", "qlora"):
            full = full_ft_state(count_params(cfg)["total"], "adamw_mixed")
            mult = f"  ({full / total:.1f}x)"
        print(f"  {name:<34}{total / GB:>10.2f}{total / GiB:>11.2f}{tag:>14}{mult}")
    print()
    # The better sentence: the TRAINABLE-state shrink, which is the parameter ratio.
    P = count_params(cfg)["total"]
    A = lora_params(cfg, r=rank)
    full_tr = trainable_state_bytes(P)
    lora_tr = trainable_state_bytes(A)
    print(f"  The sentence that matters -- trainable state alone (G+O, the part LoRA deletes):")
    print(f"    full : 14 B/param * {P:,} = {full_tr / GB:.2f} GB")
    print(f"    LoRA : 14 B/param * {A:,} = {lora_tr / GB:.2f} GB")
    print(f"    ratio = {full_tr / lora_tr:.1f}x  ==  P / LoRA-params = {P / A:.1f}x  (identical, sec2.3)")
    print(f"  LoRA does not compress the model -- it DELETES the optimizer. That is the 187x.")
    print()


# --------------------------------------------------------------------------- #
# LR principle -- D-09, this page owns it (constants.md sec9.4).
# --------------------------------------------------------------------------- #

def print_lr_principle():
    print("=" * 74)
    print("WHY THE LR CHANGES -- the principle this page owns (D-09, sec9.4)")
    print("=" * 74)
    print("  Fewer trainable parameters  ==>  higher learning rate.")
    print("  Full FT moves 8.19e9 params that already encode everything the model knows;")
    print("  it must tiptoe. LoRA moves 4.4e7 and can stride ~10x harder.")
    print(f"    LLM full FT (8.19e9 params)   LR  1e-5 - 2e-5")
    print(f"    LLM LoRA     (4.4e7 params)   LR  1e-4 - 3e-4   (~10x full-FT)")
    print(f"  Why LoRA tolerates it: B is zero-init, so the adapter starts as an exact")
    print(f"  no-op and the effective update is scaled by alpha/r (sec9.4).")
    print()


# --------------------------------------------------------------------------- #
# The budget: measure it if we can; otherwise use HIS frozen [MEA-DEV] number, labelled.
# --------------------------------------------------------------------------- #

def resolve_budget(override_gib):
    """Return (budget_gib, source_str). Priority: --mem-gib > live mem_get_info() >
    /proc/meminfo MemTotal > his frozen measured 121.6875 GiB [MEA-DEV]."""
    if override_gib is not None:
        return override_gib, f"--mem-gib {override_gib} (you told me)"
    # live GPU pool (the real thing, on the box)
    try:
        import torch
        if torch.cuda.is_available():
            _, total = torch.cuda.mem_get_info()
            return total / GiB, "torch.cuda.mem_get_info() -- MEASURED LIVE on this GPU"
    except Exception:  # noqa: BLE001
        pass
    # Linux host memory (unified pool ceiling), like measure_your_box.py
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    total_b = int(line.split()[1]) * 1024
                    return total_b / GiB, "/proc/meminfo MemTotal -- MEASURED on this host"
    except FileNotFoundError:
        pass
    return (MEASURED_MEMTOTAL_GIB,
            f"HIS box, frozen [MEA-DEV] {MEASURED_MEMTOTAL_GIB} GiB -- "
            f"run on your Spark to measure yours")


# --------------------------------------------------------------------------- #
# --self-test: the no-GPU CI path. Asserts every frozen number against constants.md.
# --------------------------------------------------------------------------- #

def self_test():
    print("=" * 74)
    print("SELF-TEST -- config-driven arithmetic vs constants.md [DER/VP]. No GPU.")
    print("=" * 74)
    checks = []

    def chk(desc, got, want, fmt=str):
        ok = got == want
        checks.append(ok)
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] {desc:<52} {fmt(got)}")
        if not ok:
            print(f"         expected {fmt(want)}")

    # 1. Parameter counts reproduced from config, to the byte.
    chk("Qwen3-8B total params (sec1.2)", count_params(QWEN3_8B)["total"], QWEN3_P,
        lambda v: f"{v:,}")
    chk("Qwen3-8B non-embedding (sec1.1)", count_params(QWEN3_8B)["non_embed"],
        6_946_071_552, lambda v: f"{v:,}")
    chk("Qwen3-8B LoRA r=16 all-linear (sec3)", lora_params(QWEN3_8B), QWEN3_LORA,
        lambda v: f"{v:,}")
    chk("Llama-3.3-70B LoRA r=16 all-linear (sec3)", lora_params(LLAMA_33_70B),
        LLAMA70_LORA, lambda v: f"{v:,}")

    # 2. The full-FT state -- THE number, all three presentations.
    full_b = full_ft_state(QWEN3_P, "adamw_mixed")
    chk("full-FT state bytes (sec2.2)", full_b, FULL_STATE_B, lambda v: f"{v:,} B")
    chk("full-FT state GB", round(full_b / GB, 2), FULL_STATE_GB, lambda v: f"{v} GB")
    chk("full-FT state GiB", round(full_b / GiB, 2), FULL_STATE_GIB, lambda v: f"{v} GiB")

    # 3. LoRA state -- 17.08 GB / 15.91 GiB.
    lora_b = lora_state(QWEN3_8B, r=16, base="bf16")["total"]
    chk("LoRA r=16 state bytes (sec2.3)", lora_b, LORA_STATE_B, lambda v: f"{v:,} B")
    chk("LoRA r=16 state GB", round(lora_b / GB, 2), LORA_STATE_GB, lambda v: f"{v} GB")
    chk("LoRA r=16 state GiB", round(lora_b / GiB, 2), LORA_STATE_GIB, lambda v: f"{v} GiB")

    # 4. The 187x identity: trainable-state shrink == parameter ratio.
    ratio_params = round(QWEN3_P / QWEN3_LORA, 1)
    ratio_state = round(trainable_state_bytes(QWEN3_P) / trainable_state_bytes(QWEN3_LORA), 1)
    chk("187x = P / LoRA-params (sec2.3)", ratio_params, RATIO_187, lambda v: f"{v}x")
    chk("187x = trainable-state ratio (sec2.3)", ratio_state, RATIO_187, lambda v: f"{v}x")
    chk("trainable state full = 114.67 GB", round(trainable_state_bytes(QWEN3_P) / GB, 2),
        114.67, lambda v: f"{v} GB")
    chk("trainable state LoRA = 0.61 GB", round(trainable_state_bytes(QWEN3_LORA) / GB, 2),
        0.61, lambda v: f"{v} GB")

    # 5. The budget verdict -- his measured number, DOES NOT FIT by 0.36 GiB.
    slack = round(MEASURED_MEMTOTAL_GIB - full_b / GiB, 2)
    chk("budget slack 121.69 - 122.05 (sec6.8)", slack, BUDGET_SLACK, lambda v: f"{v:+} GiB")
    chk("verdict", "DOES NOT FIT" if slack < 0 else "FITS", "DOES NOT FIT")

    # 6. KV cache cross-check (sec4) -- same config, same file.
    chk("KV cache full context bytes (sec4)", kv_bytes(QWEN3_8B), 6_039_797_760,
        lambda v: f"{v:,} B")

    # 7. Escape-ladder rungs (sec2.3).
    for method, opt, name, gib in [
        ("full", "adam8bit", "8-bit Adam", 76.28),
        ("full", "adam8bit_no_master", "8-bit Adam no master", 45.77),
        ("full", "sgd_momentum", "SGD+momentum", 91.54),
    ]:
        _, tot, _, _ = compute_ledger(QWEN3_8B, method, opt, 16)
        chk(f"ladder: {name} GiB (sec2.3)", round(tot / GiB, 2), gib, lambda v: f"{v} GiB")

    print()
    passed, total = sum(checks), len(checks)
    print(f"  {passed}/{total} checks passed.")
    if passed != total:
        print("  SELF-TEST FAILED.")
        return 1
    print("  SELF-TEST PASSED -- every frozen number reproduced from the config alone.")
    return 0


# --------------------------------------------------------------------------- #
# --attempt-oom: the emotional core. Allocate the 8B full-FT state until it dies.
# --------------------------------------------------------------------------- #

def attempt_oom(cfg, budget_gib, budget_source):
    P = count_params(cfg)["total"]
    target_b = full_ft_state(P, "adamw_mixed")
    print("=" * 74)
    print("--attempt-oom -- ALLOCATE THE 8B FULL-FT STATE UNTIL IT DIES")
    print("=" * 74)
    print("  SAFETY: this fills GPU memory in a loop until OutOfMemoryError. On the Spark's")
    print("  UNIFIED pool that pressure is shared with everything -- if ComfyUI is up it can")
    print("  thrash or be OOM-killed. Run this only on a box YOU control, ComfyUI DOWN, in the")
    print("  FRESH course venv (NEVER ComfyUI's -- hardware-ground-truth sec3). Ctrl-C is safe.")
    print()
    print(f"  target to reproduce : {target_b / GiB:.2f} GiB "
          f"(the full AdamW-mixed state for {P:,} params)")
    print(f"  your budget         : {budget_gib:.2f} GiB  ({budget_source})")
    print(f"  prediction          : allocation crosses the budget and OOMs BEFORE reaching")
    print(f"                        {target_b / GiB:.2f} GiB -- the state does not fit, by design.")
    print()

    try:
        import torch
    except ImportError:
        print("  torch not importable here (this laptop has none).")
        _how_to_run_on_spark()
        return 0
    if not torch.cuda.is_available():
        print(f"  torch {torch.__version__} present but CUDA not available -- no GPU to OOM here.")
        _how_to_run_on_spark()
        return 0

    cap = torch.cuda.get_device_capability(0)
    free0, total0 = torch.cuda.mem_get_info()
    print(f"  device {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
    print(f"  mem_get_info: {free0 / GiB:.2f} GiB free of {total0 / GiB:.2f} GiB")
    if free0 < 0.6 * total0:
        print("  !! Most memory is already spoken for -- ComfyUI is probably up. Bring it down")
        print("     for a clean demonstration, or you will OOM far below the state size.")
    print()

    chunk = 1 << 30  # 1 GiB of bf16 = 2^29 elements
    held = []
    grabbed = 0
    print("  allocating 1 GiB bf16 blocks ...")
    try:
        while True:
            held.append(torch.empty(chunk // 2, dtype=torch.bfloat16, device="cuda"))
            torch.cuda.synchronize()
            grabbed += chunk
            if grabbed % (8 * chunk) == 0:
                print(f"    held {grabbed / GiB:6.1f} GiB "
                      f"({100 * grabbed / target_b:.0f}% of the {target_b / GiB:.0f} GiB state)")
            if grabbed >= target_b:
                print(f"    reached {grabbed / GiB:.1f} GiB WITHOUT OOM -- this box is bigger than")
                print(f"    the state. On a 121.69 GiB Spark this line is never printed.")
                break
    except torch.cuda.OutOfMemoryError:
        print()
        print(f"  *** OutOfMemoryError at {grabbed / GiB:.1f} GiB held ***")
        print(f"  The 8B full-FT state is {target_b / GiB:.1f} GiB. You could not even hold it as")
        print(f"  raw tensors, let alone train through it. This is the wall -- measured, by you.")
        print(f"  Not a cliffhanger, not marketing: a number your own GPU just refused.")
    finally:
        del held
        torch.cuda.empty_cache()
    print()
    print("  This is why LoRA exists. Not to compress the model -- to DELETE the optimizer")
    print("  state (the 187x, sec2.3). Re-run with --method lora to watch the ledger collapse.")
    return 0


def _how_to_run_on_spark():
    print("  To reproduce the OOM, run this ON the Spark, in the fresh course venv:")
    print("      source ~/llm-ft/bin/activate      # NOT ~/ComfyUI/.venv")
    print("      python 00_verify_env.py            # confirm CUDA 13 / sm_121 / bf16 / bnb")
    print("      python 05_memory_ledger.py --attempt-oom")
    print("  It needs a CUDA GPU and will allocate until the driver refuses. ComfyUI DOWN.")
    print()


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="HF model id (frozen: Qwen/Qwen3-8B, Llama-3.3-70B) or use --config")
    ap.add_argument("--config", default=None, help="path to a local HF config.json")
    ap.add_argument("--method", choices=["full", "lora", "qlora"], default="full")
    ap.add_argument("--optimizer", choices=sorted(OPT_STATE_BYTES), default="adamw_mixed")
    ap.add_argument("--rank", "--lora-rank", type=int, default=16, dest="rank")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--mem-gib", type=float, default=None,
                    help="override the usable-memory budget (GiB). Default: MEASURE it.")
    ap.add_argument("--all-methods", action="store_true", help="print the whole escape ladder")
    ap.add_argument("--self-test", action="store_true",
                    help="arithmetic + assertions vs constants.md, no GPU (the CI path)")
    ap.add_argument("--attempt-oom", action="store_true",
                    help="ALLOCATE the 8B full-FT state until OOM (Spark-touching, dangerous)")
    args = ap.parse_args()

    print("05_memory_ledger.py -- the memory calculator as a program (course p.49, RUNG 9)")
    print(f"  python {sys.version.split()[0]}  |  arithmetic path is pure-Python, no torch")
    print()

    if args.self_test:
        raise SystemExit(self_test())

    cfg, label, source = resolve_config(args.model, args.config)
    budget_gib, budget_source = resolve_budget(args.mem_gib)

    if args.attempt_oom:
        # Show the ledger first so the number the OOM is chasing is on screen.
        print_ledger(cfg, label, source, "full", "adamw_mixed",
                     args.rank, args.batch, args.seq, budget_gib)
        raise SystemExit(attempt_oom(cfg, budget_gib, budget_source))

    print(f"  budget: {budget_source}")
    print()

    if args.all_methods:
        print_escape_ladder(cfg, args.rank, budget_gib)
        print_lr_principle()
    else:
        print_ledger(cfg, label, source, args.method, args.optimizer,
                     args.rank, args.batch, args.seq, budget_gib)
        print_lr_principle()
        print("  Next: --all-methods for the full ladder; --method lora to collapse the")
        print("  optimizer bucket; --attempt-oom to make your own GPU refuse the 8B state.")


if __name__ == "__main__":
    main()
