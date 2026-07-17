#!/usr/bin/env python3
"""
kv_cache_ledger.py -- what the KV cache actually costs, in bytes, at Qwen3-8B scale.

Course artifact for p.37 ("Attention at scale"). Page 32 (causal_mask.py) proved the KV
cache is *sound* -- earlier positions never change once written. This page answers the
question that one deferred: what does keeping those K,V around actually COST you? And it
settles the field's most repeated half-truth -- "attention is O(S^2), that's why long
context is hard" -- by showing exactly where the S^2 term stops being the bottleneck.

Three costs, from one config dict (constants.md sec1.1, all [VP]):

  L=36  H=32  H_kv=8  d_head=128  d_ff=12288  b=2 (bf16)  S=40960 (full context)

  (1) KV cache. The bytes you keep per token so you never recompute an earlier K or V.
      Per token per layer:  2 * H_kv * d_head * b = 2*8*128*2 = 4,096 B = 4 KiB.
      Per token all layers: * L = 147,456 B = 144 KiB/token.
      At full context:      * S = 6,039,797,760 B = 5.625 GiB.
      Note the H_kv, NOT H. This is the single most common error in the field: size the
      cache off the 32 query heads instead of the 8 key/value heads and you 4x your answer.
      The MHA counterfactual (H_kv=32) is 576 KiB/token -> 22.50 GiB, which would EXCEED
      the model's own 15.26 GiB of weights at full context. GQA's 4x is why 8 KV heads
      shared by 32 query heads is universal in 2026.

  (2) The FLOP crossover. Attention's S^2 term overtakes the FFN's linear-in-S term only
      when 4*S^2*d > 6*S*d*d_ff  <=>  S > 1.5*d_ff = 18,432 tokens. The d cancels; the
      crossover depends ONLY on the FFN expansion ratio. Below 18,432 tokens -- i.e. every
      ordinary chat -- the FFN dominates FLOPs and "it's O(S^2)" is the wrong bottleneck.

  (3) The score matrix that never gets built. The naive S x S attention scores at S=40,960
      in fp32 are 214.7 GB PER LAYER (constants sec5) -- 200.0 GiB, larger than the whole box
      -- and yet the model runs, because FlashAttention tiles the softmax and never
      materializes that matrix.

Then, on the box (opt-in, --measure): load Qwen3-8B, decode a handful of tokens with
use_cache=True, and watch torch.cuda.memory_allocated() climb by ~144 KiB per token --
the arithmetic above, measured. Ties into measure_your_box.py (page 43).

Runtime: arithmetic + self-test <1 s, CPU only, no torch needed. --measure ~3 min on his box.

SAFETY: the default run and --self-test are pure Python integer arithmetic -- no GPU, no
network, nothing installed or written. ONLY --measure touches the GPU: it loads Qwen3-8B
(~16.38 GB bf16) into unified memory and will CONTEND with ComfyUI if it is up on the Spark.
It reads and computes only -- installs nothing, writes nothing, trains nothing. Run it in the
FRESH course venv, NEVER ComfyUI's venv (hardware-ground-truth sec3). This file never sshes to
the Spark and never runs the GPU path for you; --measure is run by you, on the box.
"""

import argparse
import sys

# --------------------------------------------------------------------------- #
# Frozen Qwen3-8B numbers -- constants.md sec1.1 [VP], fetched from HF 2026-07-16.
# --------------------------------------------------------------------------- #

GiB = 1 << 30
GB = 10 ** 9
KiB = 1 << 10

CFG = dict(L=36, H=32, H_kv=8, d_head=128, d_ff=12288, b=2, S=40960)  # bf16 KV

# Frozen expected values the script asserts before it prints them (constants.md sec4, sec5).
KV_PER_TOKEN_LAYER = 4_096          # B   = 4 KiB           [DER, sec4]
KV_PER_TOKEN_ALL = 147_456          # B   = 144 KiB/token   [DER, sec4]
KV_FULL_CTX = 6_039_797_760         # B   = 5.625 GiB       [DER, sec4]
KV_MHA_FULL_CTX = 24_159_191_040    # B   = 22.50 GiB       [DER, sec4]
CROSSOVER_TOKENS = 18_432           # tokens                [DER, sec5]
SCORE_MATRIX_FP32 = 214_748_364_800  # B  = 214.7 GB/layer  [DER, sec5]
WEIGHTS_BYTES = 16_381_470_720      # B   = 16.38 GB / 15.26 GiB (constants sec1.2, sec4)


def kv_bytes(L, H_kv, d_head, b, S):
    """KV-cache bytes. Signature and body identical to the p.37 code block, so the page
    and this script can never drift.  2 = K and V.  H_kv, NOT H -- that is the whole point."""
    return 2 * S * L * H_kv * d_head * b


# --------------------------------------------------------------------------- #
# (1) The KV-cache ledger.
# --------------------------------------------------------------------------- #

def ledger():
    print("=" * 68)
    print("(1) KV CACHE -- what you keep per token so you never recompute a K or V")
    print("=" * 68)

    L, H, H_kv, d_head, b, S = (CFG[k] for k in ("L", "H", "H_kv", "d_head", "b", "S"))

    per_tok_layer = kv_bytes(1, H_kv, d_head, b, 1)      # one layer, one token
    per_tok_all = kv_bytes(L, H_kv, d_head, b, 1)         # all layers, one token
    full = kv_bytes(L, H_kv, d_head, b, S)                # GQA, full context
    mha = kv_bytes(L, H, d_head, b, S)                    # counterfactual: H_kv = H

    # Assert the frozen arithmetic BEFORE printing it (constants sec4; the spec's self-check).
    assert per_tok_layer == KV_PER_TOKEN_LAYER, f"{per_tok_layer} != {KV_PER_TOKEN_LAYER}"
    assert per_tok_all == KV_PER_TOKEN_ALL, f"{per_tok_all} != {KV_PER_TOKEN_ALL}"
    assert full == KV_FULL_CTX, f"{full} != {KV_FULL_CTX}"
    assert mha == KV_MHA_FULL_CTX, f"{mha} != {KV_MHA_FULL_CTX}"

    print(f"  config: L={L}  H={H}  H_kv={H_kv}  d_head={d_head}  b={b} (bf16)  S={S}")
    print()
    print(f"  per token, per layer   2 * H_kv * d_head * b = 2*{H_kv}*{d_head}*{b}")
    print(f"                       = {per_tok_layer:>13,} B  = {per_tok_layer // KiB} KiB")
    print(f"  per token, all {L} layers  * L")
    print(f"                       = {per_tok_all:>13,} B  = {per_tok_all // KiB} KiB/token  [DER, sec4]")
    print(f"  at full context S={S:,}   * S")
    print(f"                       = {full:>13,} B  = {full / GiB:.3f} GiB  [DER, sec4]")
    print()
    mha_per_token = kv_bytes(L, H, d_head, b, 1)     # all layers, one token, H_kv = H
    print(f"  The counterfactual -- if this were plain MHA (H_kv = H = {H}):")
    print(f"    per token = {mha_per_token:>7,} B = {mha_per_token // KiB} KiB/token"
          f"  ->  {mha / GiB:.2f} GiB at full context")
    saved = mha - full
    print(f"    GQA ratio H/H_kv = {H}/{H_kv} = {H // H_kv}  saves exactly {H // H_kv}x,"
          f" recovers {saved / GiB:.2f} GiB")
    print()
    print(f"  In GiB against the box (constants sec0 -- never mix GiB and GB when sizing):")
    print(f"    Qwen3-8B bf16 weights   {WEIGHTS_BYTES / GiB:>6.2f} GiB "
          f"({WEIGHTS_BYTES / GB:.2f} GB)")
    print(f"    GQA cache @ full ctx    {full / GiB:>6.3f} GiB  -- a third of the weights")
    print(f"    MHA cache @ full ctx    {mha / GiB:>6.2f} GiB  -- would OUTWEIGH the model")
    print(f"  That gap is why GQA is universal: without it, the thing you keep around to")
    print(f"  avoid recompute would outweigh the thing you are running.")
    print("  (1) KV ledger: 4 KiB/layer -> 144 KiB/token -> 5.625 GiB (GQA) vs 22.50 GiB (MHA) -- OK")
    print()
    return full


# --------------------------------------------------------------------------- #
# (2) The attention/FFN FLOP crossover.
# --------------------------------------------------------------------------- #

def crossover():
    print("=" * 68)
    print("(2) WHERE O(S^2) FINALLY BITES -- the attention/FFN FLOP crossover")
    print("=" * 68)

    d = CFG["H"] * CFG["d_head"]          # H * d_head = 4096 = d_model
    d_ff = CFG["d_ff"]
    assert d == 4096, f"H*d_head must be d_model=4096, got {d}"

    # Per layer, processing S tokens:
    #   attention (the S^2 part, QK^T + A@V) = 4 * S^2 * d      FLOP
    #   FFN (gate, up, down; linear in S)     = 6 * S * d * d_ff FLOP
    # Crossover: 4*S^2*d > 6*S*d*d_ff  <=>  S > (6/4)*d_ff = 1.5*d_ff.  The d cancels.
    crossover = (3 * d_ff) // 2
    assert crossover == CROSSOVER_TOKENS, f"{crossover} != {CROSSOVER_TOKENS}"

    print(f"  attention S^2 term   4 * S^2 * d          (d = H*d_head = {d:,})")
    print(f"  FFN linear term      6 * S * d * d_ff     (d_ff = {d_ff:,})")
    print(f"  4*S^2*d > 6*S*d*d_ff  <=>  S > 1.5 * d_ff = 1.5 * {d_ff:,}")
    print(f"                           = {crossover:,} tokens  [DER, sec5]")
    print()
    print(f"  The d cancels -- the crossover depends ONLY on the FFN expansion ratio,")
    print(f"  which is why it is exactly (3/2)*d_ff and nothing else.")
    print(f"  Below {crossover:,} tokens -- i.e. every ordinary chat -- the FFN dominates")
    print(f"  FLOPs. \"It's O(S^2)\" is the wrong bottleneck at chat lengths; at decode the")
    print(f"  binding constraint is KV-cache memory and bandwidth, not attention FLOPs.")
    print("  (2) crossover: attention overtakes FFN only above 18,432 tokens -- OK")
    print()


# --------------------------------------------------------------------------- #
# (3) The score matrix that never gets built.
# --------------------------------------------------------------------------- #

def score_matrix():
    print("=" * 68)
    print("(3) THE S x S SCORE MATRIX -- bigger than the box, and never materialized")
    print("=" * 68)

    H, S = CFG["H"], CFG["S"]
    # H query heads, S x S scores each, fp32 (4 B) -- the naive, pre-FlashAttention cost.
    scores = H * S * S * 4
    assert scores == SCORE_MATRIX_FP32, f"{scores} != {SCORE_MATRIX_FP32}"

    print(f"  naive scores, S={S:,}, fp32:  H * S^2 * 4 = {H} * {S:,}^2 * 4")
    print(f"                              = {scores:,} B")
    print(f"                              = {scores / GB:.1f} GB = {scores / GiB:.1f} GiB"
          f"  PER LAYER  [DER, sec5]")
    print(f"  One layer's score matrix ({scores / GiB:.1f} GiB) alone exceeds the entire box")
    print(f"  (121.69 GiB usable, constants sec6.8). And yet the model runs -- because")
    print(f"  FlashAttention tiles the softmax and never builds that matrix. The S^2 term")
    print(f"  is real in FLOPs but not in bytes; the memory wall you'd expect never appears.")
    print("  (3) score matrix: 214.7 GB/layer, correctly never allocated -- OK")
    print()


# --------------------------------------------------------------------------- #
# (4) Measure a real KV allocation on the box. OPT-IN, Spark-touching.
# --------------------------------------------------------------------------- #

def measure_on_box(model_id, prompt_len, new_tokens, seed):
    """Load Qwen3-8B, decode `new_tokens` tokens one at a time with use_cache=True, and
    watch torch.cuda.memory_allocated() climb by ~144 KiB per decoded token. The per-step
    memory DELTA isolates the KV growth: weights and the single-token logits are constant
    offsets, transient activations are freed on each forward's return, so the slope is the
    cache. This is the arithmetic of (1), measured. Ties into measure_your_box.py (p.43)."""
    print("=" * 68)
    print("(4) MEASURED -- the 144 KiB/token, on your box")
    print("=" * 68)
    print("  SAFETY: this loads Qwen3-8B (~16.38 GB bf16) into unified memory and will")
    print("  CONTEND with ComfyUI if it is up. It installs nothing, writes nothing, trains")
    print("  nothing. Run it in the FRESH course venv, NEVER ComfyUI's (hardware-ground-truth sec3).")
    print()

    try:
        import torch
    except ImportError:
        print("  torch is not importable in this environment.")
        _print_how_to_run_on_spark(model_id, prompt_len, new_tokens)
        return

    print(f"  torch {torch.__version__}")
    if not torch.cuda.is_available():
        print("  CUDA not available here -- this path only produces a real number on the Spark.")
        _print_how_to_run_on_spark(model_id, prompt_len, new_tokens)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    torch.manual_seed(seed)
    cap = torch.cuda.get_device_capability(0)
    print(f"  device {torch.cuda.get_device_name(0)}  capability sm_{cap[0]}{cap[1]}")
    free0, total0 = torch.cuda.mem_get_info()
    print(f"  mem_get_info: {free0 / GiB:.2f} GiB free of {total0 / GiB:.2f} GiB "
          f"(usable, NOT 128 -- see p.18/p.49)")
    if free0 < 0.5 * total0:
        print("  !! Over half your memory is already spoken for -- ComfyUI is probably up.")
        print("     The measurement still works; the box is just busier than ideal.")
    print()

    print(f"  loading {model_id} in bf16 ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    model.eval()
    torch.cuda.synchronize()

    weights_mem = torch.cuda.memory_allocated()
    print(f"  weights resident: {weights_mem / GB:.2f} GB "
          f"(predicted 16.38 GB, constants sec1.2)")
    print()

    # A prompt of the requested length (content is irrelevant to KV *size*).
    ids = torch.randint(0, tok.vocab_size, (1, prompt_len), device="cuda")

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)          # prefill
        cache = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)             # greedy first token
        torch.cuda.synchronize()

        mems = [torch.cuda.memory_allocated()]
        for _ in range(new_tokens):
            out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            mems.append(torch.cuda.memory_allocated())

    # Per-step deltas. Median is robust to the caching allocator's block rounding.
    deltas = sorted(mems[i + 1] - mems[i] for i in range(len(mems) - 1))
    measured = deltas[len(deltas) // 2]
    predicted = KV_PER_TOKEN_ALL
    print(f"  decoded {new_tokens} tokens; per-step memory_allocated() delta (median):")
    print(f"    measured   {measured:>10,} B/token  ({measured / KiB:.1f} KiB)")
    print(f"    predicted  {predicted:>10,} B/token  (144 KiB, constants sec4)")
    ratio = measured / predicted if predicted else 0.0
    print(f"    ratio      {ratio:>10.2f}x")
    print()
    if 0.9 <= ratio <= 1.6:
        print("  Within allocator-rounding tolerance: the 144 KiB/token is real, measured, yours.")
    else:
        print("  Off the prediction -- likely allocator block rounding, a resized cache, or")
        print("  contention. Re-run with ComfyUI down; inspect the raw deltas below.")
    print(f"  raw deltas (B): {[m2 - m1 for m1, m2 in zip(mems, mems[1:])]}")
    print("  (4) measured KV growth reconciled against 144 KiB/token -- see ratio above")
    print()


def _print_how_to_run_on_spark(model_id, prompt_len, new_tokens):
    print("  To get a real number, run this ON the Spark, in the fresh course venv:")
    print("      source ~/course/.venv/bin/activate      # NOT ~/ComfyUI/.venv")
    print(f"      python kv_cache_ledger.py --measure --model {model_id} \\")
    print(f"             --prompt-len {prompt_len} --new-tokens {new_tokens}")
    print("  It needs: torch (cu130 wheel) + transformers, a CUDA GPU, and ~17 GB free.")
    print("  It will load Qwen3-8B and decode tokens while watching memory_allocated().")
    print("  (4) skipped -- no GPU here; the arithmetic above stands on its own.")
    print()


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true",
                    help="arithmetic + assertions only, no GPU (the CI / no-box path). "
                         "This is what the default run already does before --measure.")
    ap.add_argument("--measure", action="store_true",
                    help="ALSO measure a real KV allocation on the box (Spark-touching, "
                         "~3 min, contends with ComfyUI). Run this on the Spark, not here.")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model id for --measure")
    ap.add_argument("--prompt-len", type=int, default=512, help="prefill length for --measure")
    ap.add_argument("--new-tokens", type=int, default=64, help="tokens to decode for --measure")
    ap.add_argument("--seed", type=int, default=37, help="RNG seed (default 37, the page number)")
    args = ap.parse_args()

    print("kv_cache_ledger.py -- what the KV cache costs at Qwen3-8B scale (course p.37)")
    print(f"  python {sys.version.split()[0]}  (arithmetic path is pure-Python, no torch)")
    print()

    ledger()
    crossover()
    score_matrix()

    if args.measure:
        measure_on_box(args.model, args.prompt_len, args.new_tokens, args.seed)
    else:
        print("=" * 68)
        print("ARITHMETIC + SELF-TEST PASSED (3 checks). To measure on the box, add --measure.")
        print("=" * 68)
        print("  1. KV ledger: 4 KiB/layer -> 144 KiB/token -> 5.625 GiB GQA vs 22.50 GiB MHA")
        print("  2. FLOP crossover: attention overtakes FFN only above 18,432 tokens")
        print("  3. score matrix: 214.7 GB/layer, correctly never materialized")


if __name__ == "__main__":
    main()
