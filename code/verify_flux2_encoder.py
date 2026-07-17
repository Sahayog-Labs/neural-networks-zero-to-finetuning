#!/usr/bin/env python3
"""
verify_flux2_encoder.py — the branches grew back together, verified on YOUR disk.

Course artifact for p.64 ("The tracks rejoin"). Capstone, Part VII. The whole course
split into two tracks — the LLM you fine-tuned (Qwen3-8B) and the image model you sampled
(FLUX.2-dev) — and p.64 is the reveal that they were never two things. Your image model's
prompt reader is a Mistral-3: a 40-layer transformer with the same GQA and the same RoPE
you already own from the LLM track. This script opens the FLUX.2-dev text_encoder config
already sitting in your HuggingFace cache and reads that back to you from the metal.

It prints five numbers off your disk —

    hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, rope_theta

— then computes the encoder's KV cache and stands it BESIDE Qwen3-8B's, so the one fact the
course refused to hand you is now yours to check:

    Mistral-3 encoder  200 KiB/token   (40 layers, H_kv=8, d_head=160)
    Qwen3-8B           144 KiB/token   (36 layers, H_kv=8, d_head=128)   [constants.md §4]

Your image model's text encoder carries a HEAVIER per-token KV than the LLM you fine-tuned —
not because it has more heads (both are H_kv=8), but because it is deeper and wider per head.

The two shell one-liners from the page, for cross-checking without Python at all:

    D=~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots
    cat $D/*/model_index.json | python3 -c "import json,sys; print(json.load(sys.stdin)['text_encoder'])"
    # -> ['transformers', 'Mistral3ForConditionalGeneration']
    cat $D/*/text_encoder/config.json | python3 -c "import json,sys; c=json.load(sys.stdin); t=c.get('text_config',c); print(t['hidden_size'], t['num_hidden_layers'], t['num_attention_heads'], t['num_key_value_heads'])"
    # -> 5120 40 32 8

Usage
-----
    python3 verify_flux2_encoder.py                 # read YOUR disk (run on the Spark)
    python3 verify_flux2_encoder.py --config PATH   # point at a specific config.json
    python3 verify_flux2_encoder.py --self-test     # no disk, no GPU — arithmetic check

SAFETY: read-only. No GPU, no torch, no network, no download — pure stdlib json on a file
already in your cache. Nothing is written and nothing is installed. Runs in ~2 s.
This targets the Spark's HF cache; it is desk-checked on the build machine via --self-test.
"""

import argparse
import glob
import json
import os
import sys

# --------------------------------------------------------------------------- #
# Frozen numbers. Every one of these comes from the course's authority files;
# the script asserts them, it does not invent them.
# --------------------------------------------------------------------------- #

KIB = 1024
BF16_BYTES = 2  # dtype: bf16, p.41 number formats

# Qwen3-8B — THE anchor. constants.md §1.1 (config) and §4 (KV cache), [VP]/[DER].
QWEN = {
    "name": "Qwen3-8B",
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "d_head": 128,
}
QWEN_KV_KIB = 144       # constants.md §4: 147,456 B = 144 KiB/token  [DER]

# FLUX.2-dev Mistral-3 text encoder — the [VP] on-disk values the page reads
# (p.64 ledger). Used as the reference in --self-test when no real config is present,
# and as the expectation the live disk read is checked against.
FLUX2_REF = {
    "name": "FLUX.2-dev Mistral-3 encoder",
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rope_theta": 1e9,
}
FLUX2_KV_KIB = 200      # p.64 [DER]: 2*40*8*160*2 = 204,800 B = 200 KiB/token

DEFAULT_ROOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots")


# --------------------------------------------------------------------------- #
# The one formula, shared by both tracks. Count H_kv, NEVER H — notation §3.2,
# constants.md §4: "the single most common error in the field."
# --------------------------------------------------------------------------- #

def kv_kib_per_token(num_hidden_layers, num_key_value_heads, d_head, b=BF16_BYTES):
    """KV bytes per token, all layers = 2 * L * H_kv * d_head * b, expressed in KiB.
    (S=1, B=1.) The 2 is one K plus one V. This is the constants.md §4 equation."""
    kv_bytes = 2 * num_hidden_layers * num_key_value_heads * d_head * b
    return kv_bytes / KIB


# --------------------------------------------------------------------------- #
# Disk read.
# --------------------------------------------------------------------------- #

def find_config(explicit):
    """Return a path to text_encoder/config.json, or None. Never touches the network."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    hits = glob.glob(os.path.join(DEFAULT_ROOT, "*", "text_encoder", "config.json"))
    return hits[0] if hits else None


def load_text_config(path):
    """Load the config and return (raw_config, text_fields).

    Mistral3ForConditionalGeneration nests the LM fields under `text_config`, alongside
    `vision_config` (it is multimodal — it can read images as prompt too). Older / flattened
    layouts put them at the top level. The `c.get("text_config", c)` fallback handles both,
    exactly as the page's snippet does."""
    with open(path) as f:
        c = json.load(f)
    return c, c.get("text_config", c)


def derive(t):
    """Pull the five reported fields plus the derived d_head out of a text-config dict."""
    hidden = t["hidden_size"]
    heads = t["num_attention_heads"]
    return {
        "hidden_size": hidden,
        "num_hidden_layers": t["num_hidden_layers"],
        "num_attention_heads": heads,
        "num_key_value_heads": t["num_key_value_heads"],
        "rope_theta": t.get("rope_theta"),
        "d_head": hidden // heads,
    }


# --------------------------------------------------------------------------- #
# Narration.
# --------------------------------------------------------------------------- #

def print_architecture(raw, f):
    print("=" * 70)
    print("WHAT YOUR IMAGE MODEL READS PROMPTS WITH — off your disk")
    print("=" * 70)

    arch = None
    if isinstance(raw.get("architectures"), list) and raw["architectures"]:
        arch = raw["architectures"][0]
    is_mistral3 = bool(arch and "Mistral3" in arch)
    print(f"  architecture   {arch or '(not stated in config)'}")
    if is_mistral3:
        print("                 ^ a full transformer LM. Not CLIP, not T5, not a bag of words.")
    if "vision_config" in raw:
        print("  vision_config  present  -> multimodal: it can read IMAGES as prompt too")
    print()

    print("  Five fields, read straight from text_encoder/config.json:")
    print(f"    {'hidden_size':<24s} {f['hidden_size']}   (d — p.2 vectors & shapes)")
    print(f"    {'num_hidden_layers':<24s} {f['num_hidden_layers']}     (L — p.35 the block, stacked)")
    print(f"    {'num_attention_heads':<24s} {f['num_attention_heads']}     (H — p.30 heads)")
    print(f"    {'num_key_value_heads':<24s} {f['num_key_value_heads']}      (H_kv — p.30/37 GQA)")
    rope = f["rope_theta"]
    rope_str = f"{rope:.0e}" if isinstance(rope, (int, float)) else str(rope)
    print(f"    {'rope_theta':<24s} {rope_str}   (b_rope — p.33 RoPE)")
    print(f"    {'d_head = d / H':<24s} {f['d_head']}    ({f['hidden_size']} / {f['num_attention_heads']}, derived — p.7 shape ribbon)")
    print()
    return is_mistral3


def print_gqa_reveal(f):
    print("=" * 70)
    print("THE REVEAL — the branches grew back together (D-17)")
    print("=" * 70)
    H, Hkv = f["num_attention_heads"], f["num_key_value_heads"]
    is_gqa = Hkv < H
    ratio = H / Hkv if Hkv else float("nan")
    has_rope = isinstance(f["rope_theta"], (int, float)) and f["rope_theta"] > 0

    print(f"  GQA?   H={H} query heads share H_kv={Hkv} key/value heads "
          f"-> {ratio:.0f}x  {'YES' if is_gqa else 'NO'}")
    print(f"         Qwen3-8B is 32 / 8 = 4x. Mistral-3 is {H} / {Hkv} = {ratio:.0f}x. "
          "The SAME GQA gift.")
    print(f"  RoPE?  rope_theta set and positive -> {'YES' if has_rope else 'NO'}")
    print()
    print("  Mistral-3 = GQA + RoPE — the exact two mechanisms you derived on the LLM")
    print("  track (p.30/37 GQA, p.33 RoPE), now found unchanged inside your image model.")
    print("  The machine didn't change. The target did. That is the whole reveal.")
    print()
    return is_gqa and has_rope


def print_kv_side_by_side(f):
    print("=" * 70)
    print("THE ENCODER'S OWN KV CACHE — beside the LLM you fine-tuned")
    print("=" * 70)
    print("  KV bytes/token = 2 * L * H_kv * d_head * b     (count H_kv, NEVER H)")
    print()

    mistral = kv_kib_per_token(
        f["num_hidden_layers"], f["num_key_value_heads"], f["d_head"])
    qwen = kv_kib_per_token(
        QWEN["num_hidden_layers"], QWEN["num_key_value_heads"], QWEN["d_head"])

    print(f"  {FLUX2_REF['name']:<30s}  "
          f"2*{f['num_hidden_layers']}*{f['num_key_value_heads']}*{f['d_head']}*{BF16_BYTES}"
          f" = {int(mistral*KIB):>7,d} B = {mistral:>3.0f} KiB/token")
    print(f"  {QWEN['name']:<30s}  "
          f"2*{QWEN['num_hidden_layers']}*{QWEN['num_key_value_heads']}*{QWEN['d_head']}*{BF16_BYTES}"
          f" = {int(qwen*KIB):>7,d} B = {qwen:>3.0f} KiB/token")
    print()
    heavier = mistral > qwen
    print(f"  Your image model's text encoder carries a "
          f"{'HEAVIER' if heavier else 'lighter'} per-token KV "
          f"({mistral:.0f} vs {qwen:.0f} KiB)")
    print(f"  than the LLM you fine-tuned — not more heads (both H_kv=8), but deeper")
    print(f"  ({f['num_hidden_layers']} vs {QWEN['num_hidden_layers']} layers) and wider "
          f"per head ({f['d_head']} vs {QWEN['d_head']}).")
    print()
    return mistral, qwen


# --------------------------------------------------------------------------- #
# Self-checks. Frozen numbers are ASSERTED. The disk is READ (and compared,
# not asserted — the disk is ground truth; a mismatch is a signal, not a crash).
# --------------------------------------------------------------------------- #

def assert_anchor_arithmetic():
    """The frozen numbers must reproduce. These are constants.md/p.64, verbatim."""
    qwen = kv_kib_per_token(
        QWEN["num_hidden_layers"], QWEN["num_key_value_heads"], QWEN["d_head"])
    assert int(qwen * KIB) == 147_456, f"Qwen KV bytes {int(qwen*KIB)} != 147,456"
    assert qwen == QWEN_KV_KIB, f"Qwen KV {qwen} != {QWEN_KV_KIB} KiB (constants.md §4)"

    ref = derive(FLUX2_REF)
    mistral = kv_kib_per_token(
        ref["num_hidden_layers"], ref["num_key_value_heads"], ref["d_head"])
    assert ref["d_head"] == 160, f"d_head {ref['d_head']} != 160 (5120/32)"
    assert int(mistral * KIB) == 204_800, f"Mistral KV bytes {int(mistral*KIB)} != 204,800"
    assert mistral == FLUX2_KV_KIB, f"Mistral KV {mistral} != {FLUX2_KV_KIB} KiB (p.64)"

    # The reveal, on the reference values: GQA (H_kv<H, 4x) and RoPE present.
    assert ref["num_key_value_heads"] < ref["num_attention_heads"], "not GQA"
    assert ref["num_attention_heads"] // ref["num_key_value_heads"] == 4, "GQA ratio != 4x"
    assert ref["rope_theta"] and ref["rope_theta"] > 0, "no RoPE"
    return qwen, mistral


def run_self_test():
    print("SELF-TEST — no disk, no GPU. Asserting the frozen arithmetic of p.64.\n")
    qwen, mistral = assert_anchor_arithmetic()
    print(f"  [OK] Qwen3-8B KV        = {qwen:.0f} KiB/token  "
          f"(2*36*8*128*2 = 147,456 B)   [constants.md §4]")
    print(f"  [OK] Mistral-3 enc. KV  = {mistral:.0f} KiB/token  "
          f"(2*40*8*160*2 = 204,800 B)   [p.64 DER]")
    print(f"  [OK] d_head             = {5120 // 32} = 5120/32")
    print(f"  [OK] GQA ratio          = 32/8 = 4x   (same as Qwen3-8B — the D-17 reveal)")
    print(f"  [OK] RoPE               = present (rope_theta = 1e9)")
    print()
    print("  All frozen numbers reproduce. On the Spark, run without --self-test to")
    print("  read these same numbers off your own FLUX.2-dev config.json.")
    return 0


def run_live(config_path):
    path = find_config(config_path)
    if path is None:
        print("Could not find the FLUX.2-dev text_encoder config.", file=sys.stderr)
        print(f"  looked in: {DEFAULT_ROOT}/*/text_encoder/config.json", file=sys.stderr)
        print("  If it lives elsewhere, pass --config /path/to/config.json", file=sys.stderr)
        print("  This is the read-only, no-GPU step — it needs only the JSON on disk.",
              file=sys.stderr)
        print("  To desk-check the arithmetic with no file at all: --self-test",
              file=sys.stderr)
        return 2

    print(f"Reading: {path}\n")
    raw, t = load_text_config(path)
    try:
        f = derive(t)
    except KeyError as e:
        print(f"Key {e} missing from the config. Printing it whole so you can see the",
              file=sys.stderr)
        print("layout (some transformers versions flatten it):\n", file=sys.stderr)
        print(json.dumps(t, indent=2)[:2000], file=sys.stderr)
        return 3

    is_mistral3 = print_architecture(raw, f)
    reveal_ok = print_gqa_reveal(f)
    mistral, qwen = print_kv_side_by_side(f)

    # Assert the frozen anchor arithmetic ALWAYS (it cannot depend on disk).
    assert_anchor_arithmetic()

    # Compare the disk read to the frozen expectation — flag, don't crash.
    print("=" * 70)
    print("CHECKS")
    print("=" * 70)
    print(f"  [{'OK ' if is_mistral3 else '?? '}] architecture is Mistral3*  "
          f"(GQA + RoPE transformer, not CLIP/T5)")
    print(f"  [{'OK ' if reveal_ok else '?? '}] the reveal holds: GQA and RoPE both present")
    print(f"  [OK ] Qwen3-8B anchor KV asserts to {QWEN_KV_KIB} KiB/token  [constants.md §4]")

    if mistral == FLUX2_KV_KIB:
        print(f"  [OK ] your on-disk encoder KV = {mistral:.0f} KiB/token — "
              f"matches the p.64 ledger exactly")
    else:
        print(f"  [?? ] your on-disk encoder KV = {mistral:.0f} KiB/token — "
              f"p.64 records {FLUX2_KV_KIB}.")
        print(f"        The disk is ground truth. If FLUX.2-dev's config changed, the page's")
        print(f"        200 is what needs updating — recompute from the fields printed above.")
    print()
    print("  The three tracks — LLM, diffusion, and the math under both — meet here,")
    print("  on your disk. The dot product you met on p.3 is running in all 40 layers.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Verify the FLUX.2-dev Mistral-3 text encoder on your disk (p.64).")
    ap.add_argument("--config", metavar="PATH",
                    help="explicit path to text_encoder/config.json")
    ap.add_argument("--self-test", action="store_true",
                    help="no disk, no GPU — assert the frozen p.64 arithmetic and exit")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()
    return run_live(args.config)


if __name__ == "__main__":
    sys.exit(main())
