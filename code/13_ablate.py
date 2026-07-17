#!/usr/bin/env python3
"""
13_ablate.py - the rank x target_modules ablation. RUNG 6 companion, p.50.

Page 50 hands you the sentence "rank is nearly free in memory; the dataset sets
the rank, and all-linear beats attention-only at matched params" (D-06, D-11).
This script is where you STOP taking that on faith and reproduce it on your own
data. It sweeps

    r in {8, 32, 128, 256}   x   target in {attn-only, all-linear}

= 8 LoRA SFT runs on Qwen3-8B, ~11.5 min each (constants.md sec6.7: measured
6,969.59 tok/s, seq 2048, batch 4), and reports for each cell the trainable-param
count, the memory footprint, and - on the Spark - the held-out eval loss.

Two things the sweep makes concrete, both from constants.md sec3:

  1. RANK IS NEARLY FREE. Going r=8 -> r=256 all-linear multiplies the trainable
     parameters by 32x (21,823,488 -> 698,351,616) but total memory moves only
     16.73 GB -> 27.55 GB (1.65x), because the frozen bf16 base (16.38 GB) is the
     same in every cell and dominates. Every one of the 8 cells fits your measured
     121.6875 GiB box with >90 GiB to spare. That is what "rank is cheap" MEANS.

  2. ALL-LINEAR BEATS ATTENTION-ONLY. Attention-only (q,k,v,o) reaches only ~35%
     of the per-r trainable surface that all-linear does, because it freezes the
     three MLP matrices - 78.26% of every block (constants.md sec1.3, D-06). To
     match all-linear's parameter count, attention-only needs ~2.85x the rank; and
     even matched on parameters, all-linear wins because it can touch the MLP. The
     final eval-loss column is where you watch that happen on YOUR data [MEA].

The memory arithmetic is shared VERBATIM with the rest of the course via
utils/ledger.py (the same module p.49's 05_memory_ledger.py uses), so this script
and the page can never drift.

SAFETY
------
The --run path trains 8 real LoRA adapters on Qwen3-8B: heavy, GPU-saturating,
~1.5 hr wall-clock for the full grid (~1.2 hr if you trim to 6 cells). It WILL
contend with ComfyUI if that is live on the GPU - do not start it blind. It writes
one adapter dir per cell under --output-dir (each 87 MB at r=16 up to ~1.4 GB at
r=256, bf16); nothing else on your system is touched. The default path (no --run)
is the pure-arithmetic --self-test: it allocates nothing, needs no GPU, and runs
in milliseconds - run that first, on any machine, to see the whole prediction.

Usage
-----
    python 13_ablate.py                 # --self-test: the full prediction, no GPU
    python 13_ablate.py --self-test     #   (same thing, explicit)
    python 13_ablate.py --run           # the real 8-cell sweep on the Spark
    python 13_ablate.py --run --quick   # a 2-cell smoke run (r=8 & r=256, all-linear)
"""

import argparse
import os
import sys

# Share the course's ONE memory model. utils/ledger.py is a valid package module
# (unlike the numeric-prefixed scripts), so it imports cleanly once the script's own
# directory is on the path - regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ledger  # noqa: E402  (path insert must precede this import)

GiB = 1 << 30
GB = 10 ** 9

# --------------------------------------------------------------------------- #
# Frozen facts, transcribed from constants.md - never remembered.
# --------------------------------------------------------------------------- #
# Qwen3-8B config (constants.md sec1.1/sec1.2). This is the dict utils/ledger.py
# normalizes; count_params(QWEN3_8B)["total"] == 8,190,735,360, to the byte.
QWEN3_8B = dict(
    hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
    num_key_value_heads=8, head_dim=128, intermediate_size=12288,
    vocab_size=151936, tie_word_embeddings=False, model_type="qwen3",
)
P_TOTAL = 8_190_735_360                 # constants.md sec1.2/sec3, exact
BOX_GIB = 121.6875                      # measured MemTotal, hardware-ground-truth sec2 [MEA-DEV]
TOK_PER_S = 6969.59                     # Llama/Qwen 8B LoRA, seq 2048 batch 4, constants.md sec6.7 [VP]

ATTN_ONLY = ("q_proj", "k_proj", "v_proj", "o_proj")
ALL_LINEAR = ledger.ALL_LINEAR         # the 7 linears; the course default (D-06)
RANKS = (8, 32, 128, 256)
TARGETS = (("attn-only", ATTN_ONLY), ("all-linear", ALL_LINEAR))

# The frozen numbers this sweep must reproduce (constants.md sec3). If any of these
# drift, the ablation is lying about memory and the assertions below fire.
FROZEN = {
    ("all-linear", 16): 43_646_976,     # the anchor: r=16 all-linear = 0.533% of P
    ("all-linear", 256): 698_351_616,   # r=256 all-linear = 8.53% of P -> 27.55 GB total
    ("attn-only", 16): 15_335_424,      # attention-only r=16 (q,k,v,o only)
}
R256_ALL_LINEAR_GB = 27.55              # constants.md sec3, the headline "still trivially fits"


# --------------------------------------------------------------------------- #
# One cell of the grid: trainable params + the four-bucket memory ledger + the
# wall-clock, computed from arithmetic the course already taught. No torch here.
# --------------------------------------------------------------------------- #

def cell(target_name, targets, r, num_examples, epochs, avg_tokens):
    """Everything predictable about one (target, r) run, before it starts."""
    nt = ledger.lora_params(QWEN3_8B, r=r, targets=targets)

    # State memory: frozen bf16 base (P x 2, unchanged every cell) + trainable
    # adapter at 16 B/param (adamw_mixed: w2 + g2 + master4 + m4 + v4). This is the
    # ledger convention the whole course sizes against (constants.md sec2.1/sec2.3);
    # the --run path below uses optim="adamw_torch" so the MEASURED peak validates
    # exactly this predicted total, not a smaller 8-bit-optimizer number.
    st = ledger.lora_state(QWEN3_8B, r=r, targets=targets,
                           optimizer="adamw_mixed", base="bf16")
    total_gb = st["total"] / GB
    total_gib = st["total"] / GiB

    tokens = num_examples * epochs * avg_tokens
    secs = tokens / TOK_PER_S                      # LoRA throughput, batch-4 seq-2048 regime
    return {
        "target": target_name, "r": r, "nt": nt,
        "pct_P": 100 * nt / P_TOTAL,
        "total_gb": total_gb, "total_gib": total_gib,
        "fits": total_gib < BOX_GIB,
        "headroom_gib": BOX_GIB - total_gib,
        "secs": secs,
    }


def build_grid(num_examples, epochs, avg_tokens, ranks=RANKS):
    grid = []
    for tname, targets in TARGETS:
        for r in ranks:
            grid.append(cell(tname, targets, r, num_examples, epochs, avg_tokens))
    return grid


def fmt_time(secs):
    m = secs / 60
    return f"{m:.1f} min" if m < 60 else f"{m/60:.2f} hr"


# --------------------------------------------------------------------------- #
# The self-narrating prediction table + the assertions against constants.md sec3.
# --------------------------------------------------------------------------- #

def print_grid(grid, num_examples, epochs, avg_tokens):
    tokens = num_examples * epochs * avg_tokens
    print("=" * 78)
    print("THE ABLATION - predicted before a single step runs (constants.md sec3)")
    print("=" * 78)
    print(f"  base model      Qwen/Qwen3-8B  (P = {P_TOTAL:,} params, bf16 = 16.38 GB)")
    print(f"  your box        {BOX_GIB:.4f} GiB measured [MEA-DEV, hardware-ground-truth sec2]")
    print(f"  dataset         {num_examples:,} ex x {epochs} epochs x {avg_tokens} tok "
          f"= {tokens:,} training tokens")
    print(f"  throughput      {TOK_PER_S:,.2f} tok/s  [VP, constants.md sec6.7]")
    print()
    hdr = f"  {'target':<11} {'r':>4} {'trainable':>13} {'%P':>7} " \
          f"{'total GB':>10} {'GiB':>8} {'headroom':>10} {'~wall':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in grid:
        flag = "" if c["fits"] else "  <-- OVER BOX"
        print(f"  {c['target']:<11} {c['r']:>4} {c['nt']:>13,} {c['pct_P']:>6.3f}% "
              f"{c['total_gb']:>9.2f} {c['total_gib']:>8.2f} "
              f"{c['headroom_gib']:>8.1f}Gi {fmt_time(c['secs']):>9}{flag}")
    print()


def self_test(num_examples, epochs, avg_tokens):
    grid = build_grid(num_examples, epochs, avg_tokens)
    print_grid(grid, num_examples, epochs, avg_tokens)

    idx = {(c["target"], c["r"]): c for c in grid}

    # -- 1. the frozen param anchors (constants.md sec3). r=16 is not in the sweep
    #       grid, so recompute it directly - it is THE number the page reconciles. --
    got = ledger.lora_params(QWEN3_8B, r=16, targets=ALL_LINEAR)
    assert got == FROZEN[("all-linear", 16)], \
        f"r=16 all-linear must be 43,646,976 (constants sec3), got {got:,}"
    got_attn = ledger.lora_params(QWEN3_8B, r=16, targets=ATTN_ONLY)
    assert got_attn == FROZEN[("attn-only", 16)], \
        f"r=16 attn-only must be 15,335,424, got {got_attn:,}"
    assert idx[("all-linear", 256)]["nt"] == FROZEN[("all-linear", 256)], \
        f"r=256 all-linear must be 698,351,616 (constants sec3)"
    print("  [OK] param counts match constants.md sec3 "
          "(43,646,976 @ r=16; 698,351,616 @ r=256; 15,335,424 attn-only @ r=16).")

    # -- 2. the headline memory number: r=256 all-linear = 27.55 GB total. --
    r256 = idx[("all-linear", 256)]
    assert abs(r256["total_gb"] - R256_ALL_LINEAR_GB) < 0.06, \
        f"r=256 all-linear total must be {R256_ALL_LINEAR_GB} GB, got {r256['total_gb']:.3f}"
    # and the r=16 anchor closes to the course's 17.08 GB / 15.91 GiB.
    anchor = ledger.lora_state(QWEN3_8B, r=16, targets=ALL_LINEAR,
                               optimizer="adamw_mixed", base="bf16")
    assert abs(anchor["total"] / GB - 17.08) < 0.02, "r=16 all-linear must be 17.08 GB (constants sec2.3)"
    assert abs(anchor["total"] / GiB - 15.91) < 0.02, "r=16 all-linear must be 15.91 GiB (constants sec2.3)"
    print(f"  [OK] r=256 all-linear = {r256['total_gb']:.2f} GB total (constants sec3 = 27.55); "
          f"r=16 anchor = 17.08 GB / 15.91 GiB.")

    # -- 3. RANK IS NEARLY FREE: 32x the params, <2x the memory, every cell fits. --
    lo = idx[("all-linear", 8)]["total_gb"]
    hi = idx[("all-linear", 256)]["total_gb"]
    nt_lo = idx[("all-linear", 8)]["nt"]
    nt_hi = idx[("all-linear", 256)]["nt"]
    assert nt_hi / nt_lo == 32.0, "r=8->256 must be a 32x parameter jump"
    mem_ratio = hi / lo
    assert mem_ratio < 2.0, \
        f"memory must barely move r=8->256 (< 2x); got {mem_ratio:.2f}x for a 32x param jump"
    assert all(c["fits"] for c in grid), "every ablation cell must fit the measured box"
    assert r256["total_gib"] < 0.25 * BOX_GIB, \
        "even r=256 all-linear must sit under a quarter of the box (the 'nearly free' point)"
    worst_headroom = min(c["headroom_gib"] for c in grid)
    print(f"  [OK] rank is nearly free: params x{nt_hi//nt_lo} (r=8->256) but memory only "
          f"x{mem_ratio:.2f} ({lo:.2f}->{hi:.2f} GB); all 8 cells fit, >={worst_headroom:.0f} GiB spare.")

    # -- 4. ALL-LINEAR vs ATTN-ONLY: the matched-params arithmetic (D-06). --
    per_r_all = ledger.lora_params(QWEN3_8B, r=1, targets=ALL_LINEAR)   # 2,727,936
    per_r_attn = ledger.lora_params(QWEN3_8B, r=1, targets=ATTN_ONLY)   # 958,464
    ratio = per_r_all / per_r_attn
    assert abs(ratio - 2.84615) < 1e-4, \
        f"all-linear reaches {ratio:.3f}x the per-r surface of attn-only; expected 2.846"
    # attention-only at ANY rank in the grid trains fewer params than all-linear at
    # the same rank - it froze the MLP. Confirm across the whole sweep.
    for r in RANKS:
        a = idx[("all-linear", r)]["nt"]
        b = idx[("attn-only", r)]["nt"]
        assert b < a, f"attn-only r={r} must train fewer params than all-linear r={r}"
    print(f"  [OK] all-linear reaches {ratio:.3f}x the per-r trainable surface of attn-only "
          "(the MLP is 78.26% of the block, D-06).")
    print("       -> to MATCH all-linear's params, attn-only needs ~2.85x the rank; and even")
    print("          matched, all-linear wins because it can touch the MLP. That comparison is")
    print("          the eval-loss column of --run, measured on your data [MEA] - not a slide.")

    print()
    print("=" * 78)
    print("SELF-TEST PASSED - every predicted number closes against constants.md sec3.")
    print("Run `python 13_ablate.py --run` on the Spark to land these on your own data.")
    print("=" * 78)
    return grid


# --------------------------------------------------------------------------- #
# The real sweep. Trains one LoRA adapter per cell and measures eval loss + peak
# memory, so nothing above is asserted - it is confirmed. Spark-only; if the stack
# is not importable it prints exactly what to install (a FRESH venv, never ComfyUI's).
# --------------------------------------------------------------------------- #

def _stamp():
    """Version card - a support request should start with a paste of real versions."""
    import torch, transformers, peft, trl
    print(f"  torch {torch.__version__} | transformers {transformers.__version__} | "
          f"peft {peft.__version__} | trl {trl.__version__} | CUDA {torch.version.cuda}")


def run_sweep(args):
    # Verified July-2026 stack (constants.md sec7; brief-llm-finetuning sec13). If any
    # piece is missing, say so and stop - do NOT limp along on a half-installed stack.
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"[stop] the LLM stack is not importable ({e}).")
        print("  This runs on the Spark in a FRESH venv (NEVER ComfyUI's - you will break its")
        print("  working cu130 torch). See the setup page / spec-code sec A:")
        print("    uv venv --python 3.12 ~/course/.venv && source ~/course/.venv/bin/activate")
        print('    uv pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130')
        print('    uv pip install "transformers==5.14.1" "peft==0.19.1" "trl>=1.8,<2" datasets')
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[stop] CUDA not available. This sweep needs the GPU; run it on the Spark.")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"[stop] --data {args.data!r} not found. Build it first with 10_build_dataset.py (p.48).")
        sys.exit(1)

    print("=" * 78)
    print("REAL SWEEP - 8 LoRA SFT runs on Qwen3-8B (heavy; contends with ComfyUI if up)")
    print("=" * 78)
    _stamp()
    free, tot = torch.cuda.mem_get_info()
    print(f"  GPU free {free/GiB:.1f} GiB of {tot/GiB:.1f} GiB. "
          + ("" if free > 0.7 * tot else "!! something else is holding memory - results will be noisy."))
    print()

    ranks = (8, 256) if args.quick else RANKS
    targets_sweep = (("all-linear", "all-linear"),) if args.quick else \
        (("attn-only", list(ATTN_ONLY)), ("all-linear", "all-linear"))

    # One eval split, shared by every cell, so eval losses are comparable (seed 42).
    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.train_test_split(test_size=0.05, seed=42)

    results = []
    for tname, tmods in targets_sweep:
        for r in ranks:
            print("-" * 78)
            print(f"CELL  target={tname}  r={r}")
            # LoRA config. alpha = 2r holds the effective scale (alpha/r = 2) constant
            # across ranks, so this isolates CAPACITY (page 50: alpha is scale, not a
            # second capacity knob). use_rslora only bites at r>=64 (brief sec15) - try it.
            peft_config = LoraConfig(
                r=r,
                lora_alpha=2 * r,
                lora_dropout=0.0,
                target_modules=tmods,          # "all-linear" (str) OR the explicit q,k,v,o list
                use_rslora=(r >= 64),
                bias="none",
                task_type="CAUSAL_LM",
            )
            out = os.path.join(args.output_dir, f"{tname}-r{r}")
            training_args = SFTConfig(
                output_dir=out,
                # THE two-libraries-one-script fp32 trap: TRL defaults model dtype to
                # fp32 here even on a bf16 checkpoint (constants sec7.3). Pin it.
                model_init_kwargs={"dtype": torch.bfloat16},
                max_length=args.max_length,           # 2048 = the throughput-measured regime
                assistant_only_loss=True,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,        # effective batch 16 (< 32, page 50 warn)
                learning_rate=1e-4,                   # LLM LoRA ~10x full-FT (constants sec9.4)
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                # adamw_torch = fp32 master+m+v = 16 B/trainable-param, matching the ledger's
                # predicted total so the MEASURED peak validates the 17.08/27.55 GB prediction.
                optim="adamw_torch",
                max_grad_norm=1.0,
                logging_steps=10,
                eval_strategy="epoch",
                bf16=True,
                report_to="none",                     # v5 hazard: don't spawn a tracker per cell
                seed=42,
            )
            torch.manual_seed(42)
            torch.cuda.reset_peak_memory_stats()
            # ONE peft path: hand peft_config to SFTTrainer; do NOT also get_peft_model
            # (double-wrap = trl issue #3926). Pick one; this is it.
            trainer = SFTTrainer(
                model="Qwen/Qwen3-8B",
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=ds["test"],
                peft_config=peft_config,
            )
            # Confirm the trainable count off the REAL model equals the prediction.
            nt = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            predicted = ledger.lora_params(
                QWEN3_8B, r=r, targets=ALL_LINEAR if tname == "all-linear" else ATTN_ONLY)
            assert nt == predicted, \
                f"real trainable {nt:,} != predicted {predicted:,} - target/config mismatch"

            trainer.train()
            metrics = trainer.evaluate()
            peak_gib = torch.cuda.max_memory_allocated() / GiB
            pred_gib = ledger.lora_state(
                QWEN3_8B, r=r,
                targets=ALL_LINEAR if tname == "all-linear" else ATTN_ONLY,
                optimizer="adamw_mixed", base="bf16")["total"] / GiB
            row = {"target": tname, "r": r, "nt": nt,
                   "eval_loss": metrics.get("eval_loss", float("nan")),
                   "peak_gib": peak_gib, "pred_gib": pred_gib}
            results.append(row)
            print(f"  trainable {nt:,} (matches prediction) | eval_loss "
                  f"{row['eval_loss']:.4f} | peak {peak_gib:.2f} GiB "
                  f"(predicted state {pred_gib:.2f} GiB + activations)")
            del trainer
            torch.cuda.empty_cache()

    # The verdicts, measured. Memory: barely moved. Capability: all-linear wins.
    print("=" * 78)
    print("MEASURED VERDICTS")
    print("=" * 78)
    print(f"  {'target':<11}{'r':>5}{'trainable':>14}{'eval_loss':>12}{'peak GiB':>11}")
    for row in results:
        print(f"  {row['target']:<11}{row['r']:>5}{row['nt']:>14,}"
              f"{row['eval_loss']:>12.4f}{row['peak_gib']:>11.2f}")
    peaks = [r["peak_gib"] for r in results]
    print(f"\n  memory moved {min(peaks):.1f} -> {max(peaks):.1f} GiB across the whole sweep - "
          "rank is nearly free [MEA].")
    all_rows = [r for r in results if r["target"] == "all-linear"]
    attn_rows = [r for r in results if r["target"] == "attn-only"]
    if all_rows and attn_rows:
        best_all = min(r["eval_loss"] for r in all_rows)
        best_attn = min(r["eval_loss"] for r in attn_rows)
        verdict = "all-linear wins" if best_all < best_attn else "inspect - attn-only tied/won on YOUR data"
        print(f"  best all-linear eval_loss {best_all:.4f} vs best attn-only {best_attn:.4f}"
              f"  -> {verdict} [MEA]. (loss down = better at predicting your set; your set is not your goal.)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="pure-arithmetic prediction + assertions, no GPU (the default)")
    ap.add_argument("--run", action="store_true",
                    help="the real 8-cell LoRA sweep on the Spark (heavy, ~1.5 hr)")
    ap.add_argument("--quick", action="store_true",
                    help="with --run: a 2-cell smoke test (r=8 & r=256, all-linear)")
    ap.add_argument("--data", default="data/train.jsonl",
                    help="JSONL from 10_build_dataset.py (p.48) - --run only")
    ap.add_argument("--output-dir", default="./ablation-out",
                    help="where per-cell adapters are written - --run only")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="SFT max_length (2048 = the throughput-measured regime)")
    ap.add_argument("--num-examples", type=int, default=2000,
                    help="dataset size for the wall-clock prediction")
    ap.add_argument("--epochs", type=int, default=3, help="epochs (prediction + run)")
    ap.add_argument("--avg-tokens", type=int, default=800,
                    help="avg tokens/example for the wall-clock prediction")
    args = ap.parse_args()

    if args.run:
        # Always show the prediction first; the run is what validates it.
        self_test(args.num_examples, args.epochs, args.avg_tokens)
        print()
        run_sweep(args)
    else:
        self_test(args.num_examples, args.epochs, args.avg_tokens)


if __name__ == "__main__":
    main()
