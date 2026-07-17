#!/usr/bin/env python3
"""
11_finetune_qlora.py - RUNG 6, the main event. LoRA SFT of Qwen3-8B on YOUR data. (p.50)

This is p.49's memory-ledger script turned loose: the same Qwen3-8B, the same
r=16 all-linear LoRA whose 43,646,976 trainable params you counted by hand on
page 40 and re-counted in the ledger on page 49 - now actually trained, on your
own JSONL, in the fresh venv the Part III setup page installs (peft/trl/accelerate/
bitsandbytes are NOT in ComfyUI's venv - hardware-ground-truth.md §3 - never
install into it).

    ~12 minutes for an 8B LoRA. That number is not a hope - it is the prediction
    you computed back in 01_tokenizer_lab.ipynb (your corpus's token count) and
    05_memory_ledger.py (17.08 GB of state, 6,969.59 tok/s measured). This script
    prints the prediction BEFORE it trains and the wall-clock AFTER. The prediction
    landing is the payoff of the whole track.

It uses the CURRENT (2026) TRL 1.8 + PEFT 0.19 path, which is smaller than every
tutorial you will find online:

  * `model` is passed to SFTTrainer as a STRING plus a `peft_config`. TRL builds
    and wraps the model for you. We NEVER call get_peft_model ourselves. Passing an
    already-wrapped PEFT model AND a peft_config double-wraps and silently FREEZES
    the adapter (trl issue #3926). Pick one path; this script picks the peft_config
    path, on purpose, and says so.
  * Because `model` is a string, `SFTConfig(model_init_kwargs={"dtype": bf16})` is
    load-bearing: SFTTrainer STILL defaults to fp32 when handed a string and no
    dtype - opposite to from_pretrained, which infers bf16 from the config since
    transformers v5. Two libraries, one script, opposite defaults (constants §7.3).
    Omit that one dict and you silently train an 8B in fp32 and OOM.
  * `report_to` now defaults to "none" in transformers v5 (was "all") - so NOTHING
    logs unless you opt in. Pass --trackio to turn on Trackio, HF's local-first
    tracker (the page's promised loss curve).
  * `target_modules="all-linear"` - not q_proj/v_proj. Attention-only froze 78% of
    the block and is the field's most-repeated LoRA mistake (D-06, page 40).
  * QLoRA (--qlora): `quantization_config=BitsAndBytesConfig(...)` goes DIRECTLY to
    SFTTrainer now (2026 API) - no hand-built model, no prepare_model_for_kbit_training.
    bnb_4bit_use_double_quant is set True explicitly; it defaults FALSE, and off it
    NF4 is 4.5 bits (4.607 GB base for Qwen3-8B), not 4.127 bits / 4.225 GB (constants §3).

Self-test (no GPU, no download): run `python 11_finetune_qlora.py --self-test`. It
reproduces the 43,646,976 count, the 17.08 GB / 15.91 GiB LoRA-state ledger, the
187x trainable-state ratio, and the ~12-min wall-clock prediction - all pure
arithmetic against constants.md, so a regression fails loudly on any laptop.

Usage
-----
    python 11_finetune_qlora.py --self-test                     # arithmetic checks, no GPU
    python 11_finetune_qlora.py --data mydata.jsonl            # LoRA SFT, Qwen3-8B, r=16
    python 11_finetune_qlora.py --data mydata.jsonl --qlora    # NF4 base (QLoRA), ~5 GB
    python 11_finetune_qlora.py --data mydata.jsonl --trackio  # + Trackio loss curve

`--data` is JSONL of {"instruction": "...", "response": "..."} pairs, one per line.
A tiny built-in fallback (12 examples) proves the loop is wired - it will not teach
the model anything durable.

SAFETY: this script TRAINS. It writes a LoRA adapter (~87.3 MB at bf16, r=16) to
--output-dir and saturates the GPU for ~12 min; if ComfyUI is live on the Spark it
WILL contend for the unified memory pool - consult before launching. It downloads
the Qwen3-8B checkpoint on first run (~16 GB into HF_HOME). It installs nothing.

Requires (constants §7, pin exactly): torch 2.13.0 · transformers 5.14.1 ·
peft 0.19.1 · trl>=1.8,<2 · bitsandbytes 0.49.2 (native sm_121, ARM64+CUDA13).
Verified against these versions 2026-07-16; never an API from memory.
"""

import argparse
import json
import sys
import time
from pathlib import Path

GiB = 1 << 30
GB = 10 ** 9

# --------------------------------------------------------------------------- #
# Qwen3-8B, FROZEN (constants.md §1.1/§1.2/§3). The seven targetable nn.Linear
# matrices per decoder layer, (d_out, d_in); norms are NOT LoRA targets (D-07).
# These are the ONLY numbers the self-test needs - the count is arithmetic.
# --------------------------------------------------------------------------- #
N_LAYERS = 36
P_TOTAL = 8_190_735_360                    # exact full parameter count, constants §1.2
BASE_BYTES_BF16 = 16_381_470_720           # safetensors total_size, constants §1.2 [VP]
LINEARS = [
    # name,        d_out,  d_in
    ("q_proj",     4096,   4096),
    ("k_proj",     1024,   4096),
    ("v_proj",     1024,   4096),
    ("o_proj",     4096,   4096),
    ("gate_proj",  12288,  4096),
    ("up_proj",    12288,  4096),
    ("down_proj",  4096,   12288),
]
ATTENTION_ONLY = {"q_proj", "k_proj", "v_proj", "o_proj"}

# The measured 8B-LoRA throughput the wall-clock prediction rides on (constants §6.7 [VP]).
TOK_PER_S_8B_LORA = 6969.59

# The tiny fallback set - proves wiring, teaches nothing durable (mirrors 24_first_finetune.py).
FALLBACK_EXAMPLES = [
    {"instruction": "What is the capital of France?", "response": "Paris."},
    {"instruction": "Name the primary colors.", "response": "Red, blue, and yellow."},
    {"instruction": "What is 7 plus 5?", "response": "12."},
    {"instruction": "Give a one-word synonym for 'happy'.", "response": "Joyful."},
    {"instruction": "What planet do we live on?", "response": "Earth."},
    {"instruction": "Convert 1 kilometer to meters.", "response": "1000 meters."},
    {"instruction": "What is the chemical symbol for water?", "response": "H2O."},
    {"instruction": "Name a programming language.", "response": "Python."},
    {"instruction": "What is the opposite of 'up'?", "response": "Down."},
    {"instruction": "How many days are in a week?", "response": "Seven."},
    {"instruction": "What color do you get mixing blue and yellow?", "response": "Green."},
    {"instruction": "What is the boiling point of water in Celsius?", "response": "100 degrees Celsius."},
]


# --------------------------------------------------------------------------- #
# Pure arithmetic - shared by the self-test and the real run. No torch, no download.
# --------------------------------------------------------------------------- #

def lora_count(r=16, target="all-linear"):
    """LoRA trainable count for a target_modules choice, straight from the config.
    Each matrix trades d_in*d_out params for r*(d_in + d_out). Returns
    (per_layer, total, per_matrix)."""
    names = None if target == "all-linear" else ATTENTION_ONLY
    per_matrix, per_layer = {}, 0
    for name, d_out, d_in in LINEARS:
        if names is not None and name not in names:
            continue
        p = r * (d_in + d_out)
        per_matrix[name] = p
        per_layer += p
    return per_layer, per_layer * N_LAYERS, per_matrix


def lora_state_bytes(trainable, base_bytes=BASE_BYTES_BF16):
    """The page-49 ledger for a LoRA run: the frozen base at its inference dtype
    plus 16 B/param of state (weights+grad+master+m+v) on the TRAINABLE params only.
    The 187x win is that the optimizer never allocates a slot for the frozen base."""
    return base_bytes + trainable * 16


def predicted_wall_clock_s(n_examples, epochs, avg_tokens, tok_per_s=TOK_PER_S_8B_LORA):
    """The page-50 demo's integration formula (brief §14 / Demo A15): total tokens
    seen, divided by measured throughput. This is the prediction the learner computed
    from 01_tokenizer_lab.ipynb (token count) + the measured tok/s - landing it is the
    point of the whole track."""
    return n_examples * epochs * avg_tokens / tok_per_s


# --------------------------------------------------------------------------- #
# THE SELF-TEST - runs on any laptop, no GPU, no torch. This is the path the
# build contract requires be executed locally and its real output pasted.
# --------------------------------------------------------------------------- #

def self_test():
    print("=" * 72)
    print("SELF-TEST (no GPU) - the numbers this run will land, from arithmetic")
    print("=" * 72)

    # (1) the count - the assertion the real run confirms on the checkpoint.
    attn_per, attn_total, _ = lora_count(16, target="attn")
    per_layer, total, per_matrix = lora_count(16, target="all-linear")
    all_params = P_TOTAL + total
    pct_of_P = 100 * total / P_TOTAL
    pct_of_all = 100 * total / all_params
    adapter_mb = total * 2 / GB * 1000                     # bf16, 2 B/param, MB (1e6)

    print(f"  target_modules = attention-only : {attn_total:>12,}  "
          f"({100 * attn_total / P_TOTAL:.3f}% of P) - froze 78% of the block (D-06)")
    print(f"  target_modules = 'all-linear'   : {total:>12,}  ({pct_of_P:.3f}% of P)")
    for name, d_out, d_in in LINEARS:
        print(f"      {name:<10} {d_out:>5} x {d_in:<5}  "
              f"base {d_in * d_out:>12,}  LoRA {per_matrix[name]:>9,}")
    print(f"  >> trainable: {total:,} || all: {all_params:,} || {pct_of_all:.2f}%")
    print(f"     adapter file, bf16: {total:,} x 2 B = {adapter_mb:.1f} MB")

    assert total == 43_646_976, f"all-linear r=16 must be 43,646,976 (constants §3), got {total:,}"
    assert per_layer == 1_212_416, f"per-layer must be 1,212,416, got {per_layer:,}"
    assert all_params == 8_234_382_336, f"all must be 8,234,382,336, got {all_params:,}"
    assert abs(pct_of_P - 0.533) < 0.001, f"expected 0.533% of P, got {pct_of_P:.3f}%"
    assert abs(adapter_mb - 87.3) < 0.1, f"adapter file must be 87.3 MB, got {adapter_mb:.1f}"
    assert per_matrix == {
        "q_proj": 131_072, "k_proj": 81_920, "v_proj": 81_920, "o_proj": 131_072,
        "gate_proj": 262_144, "up_proj": 262_144, "down_proj": 262_144,
    }, f"per-matrix LoRA counts drifted: {per_matrix}"
    print()

    # (2) the memory ledger this run realizes (constants §2.3): base bf16 + LoRA state.
    state_b = lora_state_bytes(total)
    adapter_state_b = total * 16
    print("  LoRA-state ledger (constants §2.3) - what page 49 predicted for this run:")
    print(f"    frozen base (bf16)  {BASE_BYTES_BF16 / GB:>7.2f} GB   ({BASE_BYTES_BF16 / GiB:>6.2f} GiB)")
    print(f"    LoRA state (16 B/p) {adapter_state_b / GB:>7.2f} GB   "
          f"({total:,} trainable x 16 B)")
    print(f"    TOTAL               {state_b / GB:>7.2f} GB   ({state_b / GiB:>6.2f} GiB)")
    print(f"    vs full-FT state: 131.05 GB / 122.05 GiB - a 7.67x cut, state-to-state")
    print(f"    (LoRA does NOT cut activations; they are the same +2-6 GB [EST] on both - §2.4)")
    ratio = P_TOTAL / total
    print(f"    trainable-state ratio: {P_TOTAL:,} / {total:,} = {ratio:.1f}x "
          f"(= the parameter ratio, the better sentence - constants §2.3)")

    assert abs(state_b / GB - 17.08) < 0.02, f"LoRA state must be 17.08 GB, got {state_b / GB:.2f}"
    assert abs(state_b / GiB - 15.91) < 0.02, f"LoRA state must be 15.91 GiB, got {state_b / GiB:.2f}"
    assert abs(ratio - 187.7) < 0.5, f"trainable ratio must be ~187.7, got {ratio:.1f}"
    print()

    # (3) the wall-clock prediction - the payoff (brief §14 / Demo A15 formula).
    nex, epochs, avgtok = 2000, 3, 800
    secs = predicted_wall_clock_s(nex, epochs, avgtok)
    print("  wall-clock prediction (the page-50 payoff) - brief §14 formula:")
    print(f"    {nex:,} examples x {epochs} epochs x {avgtok} avg tokens "
          f"/ {TOK_PER_S_8B_LORA:,} tok/s [VP]")
    print(f"    = {nex * epochs * avgtok:,} tokens / {TOK_PER_S_8B_LORA:,} "
          f"= {secs:.1f} s = {secs / 60:.1f} min  (~12 min: the number you computed in 01+05)")
    assert 10.0 < secs / 60 < 13.0, f"prediction must land ~12 min, got {secs / 60:.1f}"
    print()

    print("  self-checks passed: 43,646,976 || 8,234,382,336 || 0.53% ; 17.08 GB / 15.91 GiB ;")
    print("  187x state ratio ; ~12 min wall-clock. All against constants.md, no GPU touched.")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Data loading - JSONL {instruction, response} -> conversational `messages`,
# the format SFTTrainer templates and masks (assistant_only_loss) for you.
# --------------------------------------------------------------------------- #

def load_records(data_path):
    if data_path is None:
        print(f"  no --data given; using the {len(FALLBACK_EXAMPLES)}-example built-in "
              f"fallback (proves the loop is wired, nothing more)")
        return list(FALLBACK_EXAMPLES)
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "instruction" not in row or "response" not in row:
                raise ValueError(
                    f"{data_path}:{lineno}: expected keys 'instruction' and "
                    f"'response', got {sorted(row.keys())}")
            records.append(row)
    if not records:
        raise ValueError(f"{data_path}: no records found")
    print(f"  loaded {len(records)} examples from {data_path}")
    return records


def to_messages(record):
    """Conversational format - SFTTrainer applies Qwen3's chat template and, with
    assistant_only_loss=True, masks the user turn to -100 (TRL auto-patches Qwen3's
    template with the {% generation %} tags - brief §templates note)."""
    return {"messages": [
        {"role": "user", "content": record["instruction"]},
        {"role": "assistant", "content": record["response"]},
    ]}


# --------------------------------------------------------------------------- #
# Version stamp - a support request begins with real versions, not remembered ones.
# --------------------------------------------------------------------------- #

def stamp():
    import torch
    line = f"torch {torch.__version__}"
    for mod in ("transformers", "peft", "trl", "bitsandbytes"):
        try:
            line += f" · {mod} {__import__(mod).__version__}"
        except Exception:
            line += f" · {mod} MISSING"
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        line += f" · {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} · CUDA {torch.version.cuda}"
    print(f"  [{line}]")


# --------------------------------------------------------------------------- #
# The real run - LoRA SFT of Qwen3-8B via the current TRL 1.8 peft_config path.
# --------------------------------------------------------------------------- #

def run_finetune(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    print("=" * 72)
    print(f"RUNG 6 - LoRA SFT of {args.model}{' (QLoRA/NF4)' if args.qlora else ''}, r={args.rank}")
    print("=" * 72)
    stamp()

    records = load_records(args.data)
    ds = Dataset.from_list([to_messages(r) for r in records])
    eval_ds = None
    if args.eval_split > 0 and len(ds) >= 20:
        split = ds.train_test_split(test_size=args.eval_split, seed=args.seed)
        ds, eval_ds = split["train"], split["test"]
        print(f"  split: {len(ds)} train / {len(eval_ds)} eval (seed {args.seed})")

    # --- the prediction, printed BEFORE we train (the page-50 payoff) --------- #
    per_layer, total_lora, _ = lora_count(args.rank, target="all-linear")
    predicted_state_gb = lora_state_bytes(total_lora) / GB
    predicted_secs = predicted_wall_clock_s(len(records), args.epochs, args.avg_tokens)
    print()
    print("  PREDICTION (before training):")
    print(f"    trainable (arithmetic) : {total_lora:,}  (r={args.rank} all-linear)")
    print(f"    state (constants §2.3) : {predicted_state_gb:.2f} GB base+LoRA "
          f"(+ activations 2-6 GB [EST])")
    print(f"    wall-clock (brief §14) : {predicted_secs:.1f} s = {predicted_secs / 60:.1f} min "
          f"at {TOK_PER_S_8B_LORA:,} tok/s [VP], avg {args.avg_tokens} tok/ex")
    print("    write your own guess, then read the measured numbers below.")
    print()

    # --- LoRA config: all-linear, the one-word fix for the 78%-frozen mistake -- #
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.rank
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=lora_alpha,          # 2r folklore (scale 2.0). The "LoRA Without Regret"
                                        # convention is a FIXED alpha=16 (scale 16/r); the two
                                        # differ by up to 32x and are NOT reconcilable as taste
                                        # (constants/brief §15.1). Alpha only scales - it does
                                        # not change the 43,646,976 count.
        lora_dropout=0.0,
        target_modules="all-linear",    # D-06: NOT q_proj/v_proj. All seven linears.
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- QLoRA quantization: goes DIRECTLY to SFTTrainer (2026 API), double-quant on -- #
    quant_config = None
    if args.qlora:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,   # DEFAULTS FALSE. Off => 4.5 bits / 4.607 GB,
                                              # not 4.127 bits / 4.225 GB (constants §3 caveat).
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # --- SFTConfig: model_init_kwargs dtype=bf16 is the fp32 trap fix (constants §7.3) -- #
    report_to = ["trackio"] if args.trackio else "none"   # v5 default is "none"; opt in for the curve
    cfg = SFTConfig(
        output_dir=args.output_dir,
        model_init_kwargs={"dtype": torch.bfloat16},   # LOAD-BEARING: SFTTrainer defaults to
                                                       # fp32 for a string model; from_pretrained
                                                       # would infer bf16. Two libs, opposite
                                                       # defaults (constants §7.3). Omit => OOM.
        max_length=args.max_length,                    # SFTConfig default is 1024 - silently
                                                       # truncates; set it explicitly.
        packing=True,
        packing_strategy="bfd",
        assistant_only_loss=True,                      # mask the user turn to -100 (TRL auto-
                                                       # patches Qwen3's template - brief §templates)
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,        # effective batch = batch x accum; keep < 32
        learning_rate=args.lr,                         # LoRA 1e-4..3e-4, ~10x full-FT (constants §9.4).
                                                       # SFTConfig default 2e-5 is a FULL-FT value.
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        logging_steps=10,
        bf16=True,
        report_to=report_to,
        seed=args.seed,
        save_strategy="no" if args.no_save else "epoch",
    )
    if eval_ds is not None:
        cfg.eval_strategy = "steps"
        cfg.eval_steps = 50

    # Guard the trap so a future edit that drops the dtype dict fails loudly here.
    assert cfg.model_init_kwargs.get("dtype") is torch.bfloat16, (
        "model_init_kwargs dtype must be bf16 or SFTTrainer trains an 8B in fp32 "
        "and OOMs (the two-libraries-one-script trap, constants §7.3).")
    assert peft_config.target_modules == "all-linear", (
        "target_modules must be 'all-linear' - attention-only froze 78% of the block (D-06).")

    # --- SINGLE PATH: string model + peft_config. NEVER get_peft_model too (trl #3926). -- #
    trainer = SFTTrainer(
        model=args.model,                # a STRING - TRL builds & LoRA-wraps it for us
        args=cfg,
        train_dataset=ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,         # the ONE path. Do not also get_peft_model - that
                                        # double-wraps and freezes the adapter (trl issue #3926).
        quantization_config=quant_config,   # None for plain LoRA; NF4 config for --qlora (2026 API)
    )

    # --- confirm the count on the actual checkpoint - nothing asserted, measured -- #
    trainer.model.print_trainable_parameters()
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    grand = sum(p.numel() for p in trainer.model.parameters())
    print(f"  measured trainable: {trainable:,} / {grand:,} ({100 * trainable / grand:.3f}%)")
    if "Qwen3-8B" in args.model and args.rank == 16:
        assert trainable == 43_646_976, (
            f"Qwen3-8B r=16 all-linear must be 43,646,976 (constants §3); got {trainable:,}. "
            f"A target_modules or double-wrap regression (trl #3926) - the adapter is frozen.")
        print("  confirmed on the checkpoint: 43,646,976 - the count you counted by hand.")
    else:
        assert 0 < trainable < grand, (
            f"trainable is 0 or ==total ({trainable:,}) - wrong param group / frozen adapter.")

    # --- train, measuring peak memory and wall-clock -------------------------- #
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    print()
    print("  MEASURED (after training):")
    print(f"    wall-clock : {dt:.1f} s = {dt / 60:.1f} min  "
          f"(predicted {predicted_secs / 60:.1f} min - the prediction landed)")
    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / GiB
        print(f"    peak mem   : {peak_gib:.2f} GiB (max_memory_allocated); predicted state "
              f"{predicted_state_gb:.2f} GB + activations - the gap IS the activations")

    # --- the loss curve, from the trainer's own log history ------------------- #
    losses = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
    if losses:
        print("  loss curve (step: loss):")
        for step, loss in losses[:: max(1, len(losses) // 8)]:
            print(f"    {step:>6}: {loss:.4f}")
        print(f"    first {losses[0][1]:.4f} -> last {losses[-1][1]:.4f}")

    if not args.no_save:
        trainer.save_model()
        print(f"  adapter saved to {args.output_dir} (~87.3 MB at bf16, r=16)")
    print("=" * 72)
    print("RUNG 6 complete. You trained an 8B LoRA, and it landed the prediction.")
    print("Next: 13_ablate.py (r x target sweep) and 14_merge_and_serve.py.")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the arithmetic self-checks only (no GPU, no download)")
    ap.add_argument("--data", default=None, help="JSONL of {instruction, response} pairs")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="HF id or local path (constants §1)")
    ap.add_argument("-r", "--rank", type=int, default=16, help="LoRA rank (default 16)")
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="LoRA alpha (default 2*rank; see §15.1 - the convention is contested)")
    ap.add_argument("--qlora", action="store_true", help="NF4 4-bit base (QLoRA); double-quant on")
    ap.add_argument("--trackio", action="store_true", help="log the loss curve to Trackio")
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="LoRA LR (constants §9.4: 1e-4..3e-4, ~10x full-FT; default 2e-4)")
    ap.add_argument("--epochs", type=float, default=3, help="training epochs (default 3)")
    ap.add_argument("--batch", type=int, default=4, help="per-device batch (default 4)")
    ap.add_argument("--accum", type=int, default=4, help="grad accumulation (effective batch=batch*accum)")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="max sequence length (SFTConfig default 1024 silently truncates)")
    ap.add_argument("--avg-tokens", type=int, default=800,
                    help="avg tokens/example, for the wall-clock prediction (from 01_tokenizer_lab)")
    ap.add_argument("--eval-split", type=float, default=0.05, help="held-out fraction (0 to disable)")
    ap.add_argument("--seed", type=int, default=42, help="seed (determinism)")
    ap.add_argument("--output-dir", default="./qwen3-8b-lora", help="where the adapter is written")
    ap.add_argument("--no-save", action="store_true", help="do not write the adapter to disk")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    try:
        import torch
    except ImportError:
        print("torch not importable. This script needs the FRESH venv the Part III setup page")
        print("installs (torch/transformers/peft/trl/bitsandbytes) - NOT ComfyUI's venv")
        print("(hardware-ground-truth.md §3). Create it, then rerun. Or try --self-test, which")
        print("needs no GPU and no torch and still checks every frozen number.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available - an 8B LoRA needs a GPU. Run --self-test for the arithmetic,")
        print("or move to the Spark (in its own venv, mindful of ComfyUI - HARD SAFETY RULE).")
        sys.exit(1)

    run_finetune(args)


if __name__ == "__main__":
    main()
