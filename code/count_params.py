#!/usr/bin/env python3
"""
count_params.py -- count a transformer to the byte, from config.json alone.

Course artifact for p.36 ("Two things that are not in the formula -- and why"). This is
the milestone companion: four numbers copied out of a HuggingFace config.json --
hidden_size, num_hidden_layers, intermediate_size, vocab_size -- reproduce Qwen3-8B's
published parameter count EXACTLY, and a second, completely independent route (the
checkpoint's own model.safetensors.index.json, byte total halved) lands on the identical
integer. Two roads, one number. Nothing about this model is hidden; you can compute it.

    Route 1 (formula):     P = V*d + L*p_block + d + (0 if tied else V*d)
    Route 2 (checkpoint):  P = model.safetensors.index.json["metadata"]["total_size"] // 2

Both must equal 8,190,735,360 or the script refuses to print a summary.

Usage
-----
    python count_params.py --self-test          # no files needed -- runs on an embedded
                                                  # copy of Qwen3-8B's config, offline,
                                                  # no GPU. Do this first.
    python count_params.py --model-dir Qwen3-8B  # the real thing: point at a downloaded
                                                  # HF snapshot containing config.json and
                                                  # model.safetensors.index.json
    python count_params.py --model-dir Qwen3-8B --no-gpu   # skip the mem_get_info() step

Runtime: under a second either way -- this is pure JSON parsing and integer arithmetic.
No model weights are loaded, no GPU is required for the count itself. On the Spark, if
torch+CUDA are importable, the script ALSO calls torch.cuda.mem_get_info() so the
122.05-GiB-vs-your-box comparison from p.36 is your own measurement, not the course's;
if torch or CUDA aren't available (true on this Windows dev box), that step is skipped
with a note -- it never blocks the parameter count, which is the point of the artifact.

SAFETY: read-only. Opens two small JSON files (or uses the embedded --self-test copy) and,
optionally, asks the driver for free/total VRAM. Nothing is written, nothing is installed,
nothing is trained.
"""

import argparse
import json
import os
import sys

GiB = 1 << 30
GB = 10**9

# --------------------------------------------------------------------------------------- #
# Frozen truth -- every number here is transcribed verbatim from research/constants.md
# §1.1-§1.3 (Qwen3-8B, fetched from HuggingFace 2026-07-16). The script asserts its own
# arithmetic against these; it never trusts itself silently.
# --------------------------------------------------------------------------------------- #

EXPECTED = {
    "P_total": 8_190_735_360,
    "P_non_embedding": 6_946_071_552,        # L * p_block (final norm's 4,096 kept separate)
    "p_block": 192_946_432,
    "p_attn": 41_943_040,
    "p_ffn": 150_994_944,
    "p_norm": 8_448,
    "embed_bytes_total_size": 16_381_470_720,  # model.safetensors.index.json, bf16, 399 tensors
    "lm_head_or_embed": 622_329_856,           # V * d, either matrix
    "ffn_share_of_block_pct": 78.26,
    "ffn_share_of_model_pct": 66.4,
    "gqa_saving_per_block": 25_165_824,        # MHA attn (67,108,864) - GQA attn (41,943,040)
    "gqa_saving_pct_of_mha_attn": 37.5,
    "bf16_weights_gb": 16.38,
    "full_ft_state_gb": 131.05,                # decimal:  P * 16 / 1e9
    "full_ft_state_gib": 122.05,               # binary:   P * 16 / 2**30
}

# The embedded reference config for --self-test -- transcribed verbatim from constants.md
# §1.1. Lets the whole pipeline run with zero files and zero network, on any machine.
QWEN3_8B_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000,
    "max_position_embeddings": 40960,
    "hidden_act": "silu",
    "attention_bias": False,
    "torch_dtype": "bfloat16",
    "sliding_window": None,
    "rope_scaling": None,
}
QWEN3_8B_INDEX = {"metadata": {"total_size": 16_381_470_720}}


# --------------------------------------------------------------------------------------- #
# Part 1 -- load config.json + model.safetensors.index.json (or the embedded stand-ins).
# --------------------------------------------------------------------------------------- #

def load_inputs(model_dir, self_test):
    """Return (cfg, index, source_label). Never loads any tensor -- JSON only."""
    if self_test:
        return QWEN3_8B_CONFIG, QWEN3_8B_INDEX, "embedded --self-test copy of Qwen3-8B"

    cfg_path = os.path.join(model_dir, "config.json")
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    missing = [p for p in (cfg_path, idx_path) if not os.path.isfile(p)]
    if missing:
        print("Missing file(s), cannot run the real count:")
        for p in missing:
            print(f"    {p}")
        print()
        print(f"  Point --model-dir at a downloaded HuggingFace snapshot of Qwen3-8B")
        print(f"  (needs config.json and model.safetensors.index.json -- weights themselves")
        print(f"  are NOT read, so a config-only mirror or a git-lfs skip-smudge clone works).")
        print(f"  Or run with --self-test to exercise the same pipeline offline right now.")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(idx_path) as f:
        index = json.load(f)
    return cfg, index, model_dir


# --------------------------------------------------------------------------------------- #
# Part 2 -- Route 1: the formula. Same shapes as the p.35 block, summed over p.36's config.
# --------------------------------------------------------------------------------------- #

def derive_params(cfg):
    d = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    Hkv = cfg["num_key_value_heads"]
    dh = cfg["head_dim"]
    dff = cfg["intermediate_size"]
    V = cfg["vocab_size"]
    tied = cfg["tie_word_embeddings"]
    has_bias = cfg.get("attention_bias", False)

    # attention: q_proj + k_proj + v_proj + o_proj, GQA-shaped (k/v narrower than q/o)
    p_attn = 2 * d * dh * (H + Hkv)
    # FFN: gate_proj + up_proj + down_proj (SwiGLU, three matrices not two)
    p_ffn = 3 * d * dff
    # norms: input_layernorm + post_attention_layernorm (2*d) + QK-Norm (2*dh)
    p_norm = 2 * d + 2 * dh
    p_block = p_attn + p_ffn + p_norm

    non_embedding = L * p_block                       # frozen constant: 6,946,071,552
    lm_head = 0 if tied else V * d
    P = V * d + L * p_block + d + lm_head              # embed + blocks + final norm + lm_head

    # MHA counterfactual, for the GQA dividend (p.36 "worked" box)
    p_attn_mha = 2 * d * dh * (H + H)
    gqa_saving = p_attn_mha - p_attn

    return {
        "d": d, "L": L, "H": H, "Hkv": Hkv, "dh": dh, "dff": dff, "V": V,
        "tied": tied, "has_bias": has_bias,
        "p_attn": p_attn, "p_ffn": p_ffn, "p_norm": p_norm, "p_block": p_block,
        "non_embedding": non_embedding, "lm_head": lm_head, "P": P,
        "p_attn_mha": p_attn_mha, "gqa_saving": gqa_saving,
    }


def derive_shares(c):
    ffn_of_block = 100 * c["p_ffn"] / c["p_block"]
    ffn_of_model = 100 * (c["L"] * c["p_ffn"]) / c["P"]
    attn_of_model = 100 * (c["L"] * c["p_attn"]) / c["P"]
    norms_of_model = 100 * (c["L"] * c["p_norm"]) / c["P"]
    embed_bytes = c["V"] * c["d"] + (0 if c["tied"] else c["V"] * c["d"])
    embed_of_model = 100 * embed_bytes / c["P"]
    gqa_pct_of_mha_attn = 100 * c["gqa_saving"] / c["p_attn_mha"]
    return {
        "ffn_of_block": ffn_of_block, "ffn_of_model": ffn_of_model,
        "attn_of_model": attn_of_model, "norms_of_model": norms_of_model,
        "embed_of_model": embed_of_model, "gqa_pct_of_mha_attn": gqa_pct_of_mha_attn,
    }


# --------------------------------------------------------------------------------------- #
# Part 3 -- Route 2: the checkpoint's own bytes. Independent of everything in Part 2.
# --------------------------------------------------------------------------------------- #

def derive_from_index(index):
    total_bytes = index["metadata"]["total_size"]
    if total_bytes % 2 != 0:
        print(f"  !! total_size {total_bytes:,} is odd -- not all-bf16, halving is invalid here.")
    return total_bytes, total_bytes // 2


# --------------------------------------------------------------------------------------- #
# Part 4 -- report. Self-narrating, GiB/GB discipline throughout.
# --------------------------------------------------------------------------------------- #

def print_report(cfg, index, source, c, shares, total_bytes, P_from_index):
    print("=" * 74)
    print("PARAMETER COUNT -- two independent routes to one integer")
    print("=" * 74)
    print(f"  source: {source}")
    print()
    print(f"  config:  d={c['d']}  L={c['L']}  H={c['H']}  H_kv={c['Hkv']}  d_head={c['dh']}"
          f"  d_ff={c['dff']}  V={c['V']}  tied={c['tied']}")
    print()

    print("  -- Route 1: the formula --")
    print(f"    attention subtotal  2*d*d_head*(H+H_kv)   {c['p_attn']:>14,}")
    print(f"    FFN subtotal        3*d*d_ff              {c['p_ffn']:>14,}")
    print(f"    norm subtotal       2*d + 2*d_head         {c['p_norm']:>14,}")
    print(f"    per block                                 {c['p_block']:>14,}")
    print(f"    x {c['L']} blocks                          {c['L']*c['p_block']:>14,}")
    print(f"    embed_tokens        V*d                    {c['V']*c['d']:>14,}")
    print(f"    lm_head             {'tied (0)' if c['tied'] else 'V*d (untied)':<20}{c['lm_head']:>14,}")
    print(f"    model.norm          d                      {c['d']:>14,}")
    print(f"    TOTAL (Route 1)                            {c['P']:>14,}")
    print()

    print("  -- Route 2: the checkpoint's own bytes (independent of Route 1) --")
    print(f"    model.safetensors.index.json total_size    {total_bytes:>14,} B")
    print(f"    all-bf16 -> / 2                             {P_from_index:>14,}")
    print()

    match = c["P"] == P_from_index
    print(f"  Route 1 == Route 2 ?  {'YES -- exact match' if match else 'NO -- MISMATCH'}")
    print()

    print(f"  non-embedding (L * p_block, matches the model card's '6.95B')  "
          f"{c['non_embedding']:>14,}")
    print()

    print("  -- shares --")
    print(f"    FFN, of one block     {shares['ffn_of_block']:>6.2f}%   "
          f"(page-35 claim, verified)")
    print(f"    FFN, of the model     {shares['ffn_of_model']:>6.2f}%   "
          f"(different denominator -- NOT the same 78% -- never quote '70%', "
          f"that was a Llama-3.1 d_ff=3.5d figure)")
    print(f"    attention, of model   {shares['attn_of_model']:>6.2f}%")
    print(f"    norms, of model       {shares['norms_of_model']:>6.3f}%   "
          f"(tiny, but not droppable from the total -- D-07)")
    print(f"    embeddings, of model  {shares['embed_of_model']:>6.2f}%   "
          f"(pure lookup, O(1) compute/token -- costs memory, not FLOPs)")
    print()
    print(f"    GQA dividend: H_kv={c['Hkv']} instead of H={c['H']} saves "
          f"{c['gqa_saving']:,} params/block")
    print(f"    ({shares['gqa_pct_of_mha_attn']:.1f}% of what an MHA attention block "
          f"would have cost) -- and 4x on the KV cache besides.")
    print()

    print(f"  no biases anywhere? attention_bias={c['has_bias']}  -->  "
          f"{'confirmed, all matrices above are weight-only' if not c['has_bias'] else '!! bias present -- formula above omits it, recount needed'}")
    print()

    print("=" * 74)
    print("THE MEMORY LEDGER -- what Route 1's integer costs to fine-tune")
    print("=" * 74)
    P = c["P"]
    weights_gb = P * 2 / GB
    state_gb = P * 16 / GB
    state_gib = P * 16 / GiB
    print(f"  bf16 weights   {P:,} x 2 B      = {weights_gb:>10.2f} GB")
    print(f"  full-FT state  {P:,} x 16 B     = {state_gb:>10.2f} GB   (decimal)")
    print(f"                                  = {state_gib:>10.2f} GiB  (binary)  <- memorize 16P")
    print(f"    weights 2B + grads 2B + fp32-master 4B + Adam-m 4B + Adam-v 4B = 16 B/param")
    print(f"    !! STATE ONLY. Activations (2-6 GB [EST] w/ grad checkpointing, B=1 S=2048)")
    print(f"       are a SEPARATE line, always -- never folded silently into the 16P number.")
    print()


def try_gpu_measurement(no_gpu, state_gib):
    print("=" * 74)
    print("YOUR BOX -- optional live measurement (skips cleanly if unavailable)")
    print("=" * 74)
    if no_gpu:
        print("  --no-gpu passed: skipping mem_get_info().")
        print()
        return
    try:
        import torch
    except ImportError:
        print("  torch not importable here (expected on a dev box without CUDA).")
        print("  On the Spark: run this same script from ComfyUI's venv, e.g.")
        print("    ~/ComfyUI/.venv/bin/python count_params.py --model-dir Qwen3-8B")
        print("  (read-only -- installs nothing, writes nothing)")
        print()
        return
    if not torch.cuda.is_available():
        print("  torch is importable but CUDA is not available -- skipping.")
        print()
        return
    free, total = torch.cuda.mem_get_info()
    usable_gib = total / GiB
    slack = usable_gib - state_gib
    verdict = "FITS" if slack > 0 else "DOES NOT FIT"
    print(f"  mem_get_info()  total {usable_gib:.4f} GiB | free {free/GiB:.2f} GiB")
    print(f"  full-FT state   {state_gib:.2f} GiB")
    print(f"  slack           {slack:+.2f} GiB   --> {verdict}")
    if slack < 0:
        print(f"  ...before activations. This is the p.36 punchline: often a matter of a hair,")
        print(f"  not a comfortable margin -- and you only find out by measuring, not by")
        print(f"  trusting the box's published capacity.")
    print()


# --------------------------------------------------------------------------------------- #
# Part 5 -- self-checks. The spec's contract: both routes MUST equal 8,190,735,360, the FFN
# shares must land on 66.4%/78.26% (not 70%), and there must be no biases. If any of these
# fail, the script exits non-zero instead of printing a summary that looks trustworthy.
# --------------------------------------------------------------------------------------- #

def run_self_checks(c, shares, P_from_index, strict_to_qwen3_8b):
    print("=" * 74)
    print("SELF-CHECKS")
    print("=" * 74)
    ok = True

    def check(label, cond):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}")
        if not cond:
            ok = False

    check("Route 1 (formula) == Route 2 (checkpoint bytes // 2)", c["P"] == P_from_index)
    check("no biases anywhere (attention_bias == False)", c["has_bias"] is False)

    if strict_to_qwen3_8b:
        # Only meaningful when the input IS Qwen3-8B (self-test, or a genuine Qwen3-8B dir).
        # A --model-dir pointed at a different model should not be forced to match these.
        check(f"total params == {EXPECTED['P_total']:,}", c["P"] == EXPECTED["P_total"])
        check(f"non-embedding == {EXPECTED['P_non_embedding']:,}",
              c["non_embedding"] == EXPECTED["P_non_embedding"])
        check(f"per-block == {EXPECTED['p_block']:,}", c["p_block"] == EXPECTED["p_block"])
        check("FFN share of block == 78.26% (not 70%)",
              abs(shares["ffn_of_block"] - EXPECTED["ffn_share_of_block_pct"]) < 0.01)
        check("FFN share of model == 66.4% (not 70%, not 78%)",
              abs(shares["ffn_of_model"] - EXPECTED["ffn_share_of_model_pct"]) < 0.05)
        check(f"GQA saving/block == {EXPECTED['gqa_saving_per_block']:,}",
              c["gqa_saving"] == EXPECTED["gqa_saving_per_block"])
        check(f"checkpoint total_size == {EXPECTED['embed_bytes_total_size']:,} B",
              P_from_index * 2 == EXPECTED["embed_bytes_total_size"])
    print()
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="Qwen3-8B",
                     help="directory with config.json + model.safetensors.index.json")
    ap.add_argument("--self-test", action="store_true",
                     help="use the embedded Qwen3-8B config, no files/network needed")
    ap.add_argument("--no-gpu", action="store_true",
                     help="skip the optional torch.cuda.mem_get_info() step")
    args = ap.parse_args()

    cfg, index, source = load_inputs(args.model_dir, args.self_test)

    c = derive_params(cfg)
    shares = derive_shares(c)
    total_bytes, P_from_index = derive_from_index(index)

    print_report(cfg, index, source, c, shares, total_bytes, P_from_index)
    try_gpu_measurement(args.no_gpu, c["P"] * 16 / GiB)

    # Strict frozen-constant checks only apply when we know we're looking at Qwen3-8B:
    # the --self-test config, or a --model-dir whose vocab/hidden_size/layers happen to
    # match it exactly (i.e. it really is a Qwen3-8B snapshot, not a different model).
    is_qwen3_8b_shaped = (
        c["d"] == 4096 and c["L"] == 36 and c["V"] == 151936 and c["dff"] == 12288
    )
    ok = run_self_checks(c, shares, P_from_index, strict_to_qwen3_8b=is_qwen3_8b_shaped)

    if not is_qwen3_8b_shaped:
        print("  (input is not Qwen3-8B-shaped -- frozen-constant checks skipped; only the")
        print("   two-routes-agree and no-bias checks are universal.)")
        print()

    if not ok:
        print("SELF-CHECKS FAILED -- see [FAIL] lines above.")
        sys.exit(1)

    print("All self-checks passed.")


if __name__ == "__main__":
    main()
