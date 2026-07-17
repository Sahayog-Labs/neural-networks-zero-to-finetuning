#!/usr/bin/env python3
"""
02_kv_cache_and_roofline.py -- decoding is the most bandwidth-starved thing your Spark does.

Course artifact for p.46 ("Decoding, the KV cache & the roofline"), RUNG 10 of the code
ladder, decode side. Page 43 measured your box's roofline; page 46 puts LLM decode on it and
finds it 227-458x below the ridge -- brutally bandwidth-bound. This script is the lab that
turns that claim into a number you produce yourself.

Three things live here, and each is one division:

  (1) THE KV CACHE, per token. Every decoded token keeps one key and one value vector per
      layer so earlier positions never get recomputed. Qwen3-8B, bf16:
        2 * L * H_kv * d_head * b = 2*36*8*128*2 = 147,456 B = 144 KiB/token.
      The trap is H_kv=8 (key/value heads), NOT H=32 (query heads). Grouped-query attention
      shares 8 K/V heads across 32 query heads; using H over-sizes the cache by exactly 4x.

  (2) THE DECODE ROOFLINE. Arithmetic intensity I = tokens/forward (p.43). Decode does one
      token per forward, so I = 1 -- 227x to 458x below the Spark's ridge (I* = 227 if the
      BF16 ceiling is ~62 TF, 458 if ~125 TF; both [INF], both unresolved, see constants
      sec6.3-6.4). Decode is memory-bound: throughput is set by how fast weights stream once
      per token. Heuristic: decode tok/s ~= 0.65 * BW / (weight bytes read per token).

  (3) THE FAILURE MODES, which are the lesson. Four real dense batch-1 measurements land at
      60-75% MBU under the physically-correct byte convention (checkpoint MINUS the embedding
      table -- the embedding is gathered, not streamed). Two "impossible" numbers teach the
      domain: if the heuristic predicts >100% of peak bandwidth, the model is an MoE; if it
      still looks impossible, someone batched.

Default run: pure-Python arithmetic + assertions against constants.md (no torch, no GPU, <1 s).
--measure (opt-in, ON THE SPARK): load Qwen3-8B, watch the KV cache grow ~144 KiB/token, and
benchmark real decode tok/s against the 16.67 ceiling / ~10.8 expectation, ~20 min.

SAFETY: the default run and --self-test are pure integer/float arithmetic -- no GPU, no
network, nothing installed or written. ONLY --measure touches the GPU: it loads Qwen3-8B
(~16.38 GB bf16) into unified memory and decodes tokens, and it WILL CONTEND with ComfyUI if
it is up on the Spark -- shut ComfyUI down first or expect an OOM that is yours, not the
model's. It reads and computes only: installs nothing, writes nothing, trains nothing. Run it
in the FRESH course venv, NEVER ComfyUI's (hardware-ground-truth sec3). This file never sshes
to the Spark and never runs the GPU path for you; --measure is run by you, on the box.
"""

import argparse
import sys

# --------------------------------------------------------------------------- #
# Units (constants.md sec0). GiB for anything sized against the box's capacity;
# GB (decimal) for weights quoted alone and for bandwidth. Never mix in one line.
# --------------------------------------------------------------------------- #

GiB = 1 << 30
GB = 10 ** 9
KiB = 1 << 10

# --------------------------------------------------------------------------- #
# Frozen Qwen3-8B numbers -- constants.md sec1.1/sec1.2 [VP], the single arithmetic anchor.
# --------------------------------------------------------------------------- #

CFG = dict(L=36, H=32, H_kv=8, d_head=128, b=2)      # b = 2 bytes, bf16 KV
FULL_CTX = 40_960                                     # max context (notation sec4.2)

WEIGHTS_GB = 16.38                                    # Qwen3-8B bf16 weights [VP, sec1.2]
BW_GBPS = 273.0                                       # Spark memory bandwidth, GB/s [VP, sec6.7]
MBU_K = 0.65                                          # round MBU coefficient (mean 0.69) [EST, sec6.6]

# Frozen expected values asserted BEFORE they are printed (the spec's self-checks, sec4/sec6.7).
KV_PER_TOKEN = 147_456                                # B = 144 KiB/token, GQA        [DER, sec4]
KV_PER_TOKEN_MHA = 589_824                            # B = 576 KiB/token, H_kv=H=32  [DER, sec4]
BF16_CEILING = 16.67                                  # tok/s, 273/16.38, 100% MBU    [DER, sec6.7]
BF16_EXPECT = 10.8                                    # tok/s, x0.65                  [EST, sec6.7]
RIDGE_62 = 227                                        # FLOP/byte if ceiling ~62 TF   [INF, sec6.4]
RIDGE_125 = 458                                       # FLOP/byte if ceiling ~125 TF  [INF, sec6.4]

# Four measured dense batch-1 decode points -- constants.md sec6.7 [VP].
# Byte column is decode-traffic (checkpoint minus the embedding table), the physically
# correct convention (sec6.6). MBU is the frozen [VP] value; the arithmetic recovers it.
DOTS = [
    # name,                 decode-traffic GB, decode tok/s, frozen MBU
    ("Llama-3.1-8B  FP8",   8.00,  20.5,  0.603),
    ("Llama-3.1-70B FP8",   70.00, 2.7,   0.698),
    ("Llama-3.1-8B  NVFP4", 4.98,  38.65, 0.705),
    ("Qwen3-14B     NVFP4", 8.98,  22.71, 0.747),
]

# Qwen3-8B decode predictions across precisions -- constants.md sec6.7 [DER].
PRECISIONS = [
    # label,  weight GB, frozen ceiling, frozen expect
    ("bf16",  16.38, 16.67, 10.8),
    ("fp8",    8.19, 33.33, 21.7),
    ("NVFP4",  4.10, 66.6,  43.0),
]


def kv_bytes_per_token(L, H_kv, d_head, b):
    """KV bytes kept per decoded token, all layers. 2 = K and V. H_kv, NOT H -- the whole
    point. Signature-identical to the p.46 formula so the page and this script cannot drift."""
    return 2 * L * H_kv * d_head * b


# --------------------------------------------------------------------------- #
# (1) The KV cache: 144 KiB/token, and why it is H_kv not H.
# --------------------------------------------------------------------------- #

def kv_cache_section():
    print("=" * 70)
    print("(1) THE KV CACHE -- 144 KiB per token, and the 4x trap in one letter")
    print("=" * 70)

    L, H, H_kv, d_head, b = (CFG[k] for k in ("L", "H", "H_kv", "d_head", "b"))

    per_tok = kv_bytes_per_token(L, H_kv, d_head, b)     # GQA, all layers, one token
    per_tok_mha = kv_bytes_per_token(L, H, d_head, b)    # counterfactual: H_kv = H = 32

    # Assert the frozen arithmetic before printing it (constants sec4).
    assert per_tok == KV_PER_TOKEN, f"{per_tok} != {KV_PER_TOKEN}"
    assert per_tok_mha == KV_PER_TOKEN_MHA, f"{per_tok_mha} != {KV_PER_TOKEN_MHA}"

    per_layer = kv_bytes_per_token(1, H_kv, d_head, b)
    print(f"  config: L={L}  H={H} (query heads)  H_kv={H_kv} (key/value heads)  "
          f"d_head={d_head}  b={b} (bf16)")
    print()
    print(f"  per token, per layer   2 * H_kv * d_head * b = 2*{H_kv}*{d_head}*{b} "
          f"= {per_layer:,} B = {per_layer // KiB} KiB")
    print(f"  per token, all {L} layers  * L = {per_tok:,} B "
          f"= {per_tok // KiB} KiB/token   [DER, constants sec4]")
    print()
    print("  Scale by context (still B=1, bf16). The cache grows LINEARLY with S:")
    for S in (1_024, 4_096, 32_768, FULL_CTX, 131_072):
        gib = per_tok * S / GiB
        tag = ""
        if S == FULL_CTX:
            tag = "  <- full native context"
        if gib > WEIGHTS_GB * GB / GiB:
            tag = "  <- LARGER than the 16.38 GB of weights"
        print(f"    S = {S:>7,} tokens   {gib:>7.3f} GiB{tag}")
    print()
    print(f"  The trap: write H={H} (query heads) where the formula wants H_kv={H_kv} "
          f"(K/V heads) and")
    print(f"  you over-size the cache by exactly H/H_kv = {H}/{H_kv} = {H // H_kv}x. "
          f"Grouped-query")
    print(f"  attention shares {H_kv} K/V heads across {H} query heads; that {H // H_kv}x "
          f"is the entire")
    print(f"  reason GQA exists. MHA would cost {per_tok_mha // KiB} KiB/token "
          f"({per_tok_mha * FULL_CTX / GiB:.2f} GiB at full")
    print(f"  context) vs GQA's {per_tok // KiB} KiB/token "
          f"({per_tok * FULL_CTX / GiB:.3f} GiB). You cache K and V, never Q:")
    print(f"  q_i attends once from position i then is never re-read; K,V are re-read by "
          f"every future token.")
    print("  (1) KV cache: 144 KiB/token [DER, sec4], H_kv not H, GQA saves 4x -- OK")
    print()
    return per_tok


# --------------------------------------------------------------------------- #
# (2) The decode roofline: I = 1, the ridge, and the tok/s predictions.
# --------------------------------------------------------------------------- #

def roofline_section():
    print("=" * 70)
    print("(2) THE DECODE ROOFLINE -- I = 1, and the tok/s you can predict from one division")
    print("=" * 70)

    # Arithmetic intensity I = tokens per forward pass (constants sec6.5).
    print("  Arithmetic intensity  I = FLOPs/bytes = 2*N*S_fwd / 2*N = S_fwd  [constants sec6.5]")
    print("  Decode does ONE token per forward, so S_fwd = 1  =>  I = 1 FLOP/byte.")
    print(f"  The Spark's ridge I* = P_peak / BW is INFERRED, not published (sec6.3-6.4):")
    print(f"    I* = {RIDGE_62} FLOP/byte  if the BF16 ceiling is ~62 TF  [INF]")
    print(f"    I* = {RIDGE_125} FLOP/byte  if the ceiling is ~125 TF (the SDXL datapoint "
          f"leans here)  [INF]")
    print(f"  Either way decode's I = 1 sits {RIDGE_62}x to {RIDGE_125}x BELOW the ridge: "
          f"bandwidth-bound,")
    print("  badly. The GPU idles while DRAM streams weights. Prefill (S_fwd=2048) sits "
          "ABOVE the ridge")
    print("  -- compute-bound. Same weights, same machine, opposite verdicts. That is the "
          "whole story.")
    print()

    # The heuristic and the frozen Qwen3-8B predictions (constants sec6.6-6.7).
    print(f"  Heuristic [EST, +/-10-15%]:  decode tok/s ~= {MBU_K} * BW / (weight bytes/token)")
    print(f"  BW = {BW_GBPS:.0f} GB/s [VP].  Ceiling = BW/weights (100% MBU); "
          f"expectation = x{MBU_K}.")
    print()
    print(f"  {'precision':<9}{'weights':>10}{'ceiling tok/s':>16}{'expect x0.65':>16}")
    for label, wgb, fceil, fexp in PRECISIONS:
        ceiling = BW_GBPS / wgb
        expect = MBU_K * ceiling
        # Assert the derived predictions match the frozen table before printing (sec6.7).
        assert abs(ceiling - fceil) < 0.05, f"{label} ceiling {ceiling:.3f} != {fceil}"
        assert abs(expect - fexp) < 0.4, f"{label} expect {expect:.3f} != {fexp}"
        print(f"  {label:<9}{wgb:>8.2f} GB{ceiling:>14.2f}  {expect:>14.1f}")
    print()
    # The headline pair, asserted to the frozen values (sec6.7).
    bf16_ceiling = BW_GBPS / 16.38
    bf16_expect = MBU_K * bf16_ceiling
    assert abs(bf16_ceiling - BF16_CEILING) < 0.01, bf16_ceiling
    assert abs(bf16_expect - BF16_EXPECT) < 0.05, bf16_expect
    print(f"  So bf16 Qwen3-8B on the 273 GB/s bus:  CEILING {bf16_ceiling:.2f} tok/s "
          f"(100% MBU, unreachable),")
    print(f"  EXPECTATION {bf16_expect:.1f} tok/s (x{MBU_K}). Say WHICH is which -- they are "
          f"~6 tok/s apart, and")
    print("  confusing the ceiling for the expectation is the trap the page warns against.")
    print("  (2) roofline: I=1, ridge 227/458 [INF], bf16 16.67 ceiling / ~10.8 expect "
          "[DER, sec6.7] -- OK")
    print()
    return bf16_ceiling, bf16_expect


# --------------------------------------------------------------------------- #
# (3) The four measured MBU dots + the decode-traffic byte convention.
# --------------------------------------------------------------------------- #

def mbu_section():
    print("=" * 70)
    print("(3) FOUR MEASURED DOTS -- 60-75% MBU, once you count the bytes correctly")
    print("=" * 70)
    print("  The physically correct byte count is DECODE-TRAFFIC = checkpoint file MINUS the")
    print("  embedding table. The input embedding is GATHERED (one row looked up per token),")
    print("  not streamed -- so it is not in the per-token weight read. For a quantized model")
    print("  this is neither the file size nor params*bits; getting it wrong moves a point")
    print("  15-25 percentage points (constants sec6.6). Under this convention (sec6.7 [VP]):")
    print()
    print(f"  {'model':<22}{'traffic GB':>12}{'ceiling':>10}{'decode':>9}{'MBU':>9}")
    for name, gb, tps, fmbu in DOTS:
        ceiling = BW_GBPS / gb
        mbu = tps / ceiling
        # MBU is [DER] from measured tok/s and ceiling; the byte figures are approximate
        # (e.g. "~70 GB"), so print the FROZEN [VP] MBU and only assert the arithmetic
        # recovers it within rounding -- never launder an approximation past constants.md.
        assert abs(mbu - fmbu) < 0.01, f"{name}: {mbu:.4f} != {fmbu}"
        print(f"  {name:<22}{gb:>10.2f}  {ceiling:>8.2f}{tps:>9.1f}{fmbu * 100:>8.1f}%")
    print()
    print(f"  All four land 60-75% MBU (mean k=0.688). A round x{MBU_K} is within ~13% "
          f"worst-case.")
    print("  The dominant uncertainty is the BYTE COUNT, not the coefficient -- whole-file vs")
    print("  decode-traffic moves a single point 15-25 pp. The heuristic is only as good as")
    print("  your bytes-per-token number, and for a quantized model that is subtler than the file size.")
    print("  (3) MBU: four dots reconciled to 60.3-74.7% under decode-traffic bytes [DER, sec6.7] -- OK")
    print()


# --------------------------------------------------------------------------- #
# (4) The two failure modes -- the punchline, not refutations.
# --------------------------------------------------------------------------- #

def failure_modes_section():
    print("=" * 70)
    print("(4) THE FAILURE MODES ARE THE LESSON -- MoE and the hidden batch")
    print("=" * 70)

    # gpt-oss-20B: the impossible number that diagnoses an MoE.
    moe_tps, moe_gb = 49.7, 12.8       # constants sec6.6 [VP]
    implied_bw = moe_tps * moe_gb      # GB/s the heuristic would need
    pct = implied_bw / BW_GBPS
    # The teaching point is physical impossibility: implied BW exceeds the real bus.
    assert implied_bw > BW_GBPS, f"{implied_bw:.1f} should exceed the {BW_GBPS} GB/s bus"
    print(f"  gpt-oss-20B decodes at {moe_tps} tok/s on its {moe_gb} GB checkpoint. If every")
    print(f"  weight were read per token, that IMPLIES {implied_bw:.0f} GB/s "
          f"= {pct * 100:.0f}% of the {BW_GBPS:.0f} GB/s")
    print(f"  bus -- physically impossible. [VP: ~637 GB/s, 234%, constants sec6.6.]")
    print()
    print('  >> "If your heuristic predicts more than 100% of peak bandwidth, your model is')
    print('      an MoE."  gpt-oss-20B is mixture-of-experts: ~3.6B of ~20.9B params active')
    print("      per token, so only the active weights are read. On ~1.9 GB active -> ~35% MBU.")
    print()
    # DeepSeek-R1-14B: the other trap -- a hidden batch.
    print("  A second trap: DeepSeek-R1-14B's 83.5 tok/s looks impossible too, until you")
    print("  notice it was measured at BATCH 8, not 1 -- the weight read amortizes across 8")
    print("  streams (per-stream MBU 56.5%). Not a batch-1 datapoint.")
    print()

    # The batch-1 -> batch-32 measurement: the roofline, measured on his class of box.
    d1, d32 = 20.5, 368.0              # Llama-3.1-8B decode, LMSYS [VP, sec6.7]
    pf1, pf32 = 7991, 7949             # prefill, same runs [VP]
    gain = d32 / d1
    assert abs(gain - 18) < 0.2, f"decode gain {gain:.2f} should be ~18x"
    print(f"  The single most instructive measurement (Llama-3.1-8B, constants sec6.7 [VP]):")
    print(f"    decode   batch 1 -> batch 32:  {d1} -> {d32:.0f} tok/s  = {gain:.0f}x GAIN")
    print(f"    prefill  batch 1 -> batch 32:  {pf1:,} -> {pf32:,} tok/s  = FLAT")
    print("  Decode was starving for bandwidth and batching fed it (S_fwd: 1 -> 32, I climbs")
    print("  toward the ridge). Prefill was already compute-bound and gained nothing. That is")
    print("  the roofline of section (2), measured -- one formula, two verdicts.")
    print("  (4) failure modes: MoE >100%-BW punchline + 18x batch gain [VP, sec6.6-6.7] -- OK")
    print()


# --------------------------------------------------------------------------- #
# (5) Measure real decode + KV growth on the box. OPT-IN, Spark-touching.
# --------------------------------------------------------------------------- #

def measure_on_box(model_id, prompt_len, new_tokens, seed):
    """Load Qwen3-8B and produce TWO measured numbers the sections above only predicted:
       (a) KV growth: memory_allocated() climbs ~144 KiB per decoded token;
       (b) decode throughput: real tok/s, compared to the 16.67 ceiling / ~10.8 expectation,
           and the achieved MBU = measured / ceiling. Then place it on an ASCII roofline."""
    print("=" * 70)
    print("(5) MEASURED -- KV growth and decode tok/s, on YOUR box")
    print("=" * 70)
    print("  SAFETY: loads Qwen3-8B (~16.38 GB bf16) into unified memory and decodes tokens.")
    print("  It WILL CONTEND with ComfyUI if it is up -- shut ComfyUI down first or expect an")
    print("  OOM that is yours, not the model's. Installs nothing, writes nothing, trains")
    print("  nothing. Run in the FRESH course venv, NEVER ComfyUI's (hardware-ground-truth sec3).")
    print()

    try:
        import torch
    except ImportError:
        print("  torch is not importable in this environment.")
        _how_to_run_on_spark(model_id, prompt_len, new_tokens)
        return

    print(f"  torch {torch.__version__}")
    if not torch.cuda.is_available():
        print("  CUDA not available here -- this path only produces a real number on the Spark.")
        _how_to_run_on_spark(model_id, prompt_len, new_tokens)
        return

    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    torch.manual_seed(seed)
    cap = torch.cuda.get_device_capability(0)
    print(f"  device {torch.cuda.get_device_name(0)}  capability sm_{cap[0]}{cap[1]}")
    free0, total0 = torch.cuda.mem_get_info()
    print(f"  mem_get_info: {free0 / GiB:.2f} GiB free of {total0 / GiB:.2f} GiB "
          f"(usable, NOT 128 -- see p.18/p.49)")
    if free0 < 0.5 * total0:
        print("  !! Over half your memory is already spoken for -- ComfyUI is probably up.")
        print("     The tok/s number will be pessimistic and noisy. Re-run with ComfyUI down.")
    print()

    print(f"  loading {model_id} in bf16 ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    model.eval()
    torch.cuda.synchronize()
    weights_mem = torch.cuda.memory_allocated()
    print(f"  weights resident: {weights_mem / GB:.2f} GB (predicted {WEIGHTS_GB} GB, sec1.2)")
    print()

    ids = torch.randint(0, tok.vocab_size, (1, prompt_len), device="cuda")

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)          # prefill
        cache = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()

        # (a) KV growth, per-step memory_allocated() deltas.  (b) decode wall time.
        mems = [torch.cuda.memory_allocated()]
        t0 = time.perf_counter()
        for _ in range(new_tokens):
            out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            mems.append(torch.cuda.memory_allocated())
        secs = time.perf_counter() - t0

    # (a) KV growth: median delta is robust to the allocator's block rounding.
    deltas = sorted(mems[i + 1] - mems[i] for i in range(len(mems) - 1))
    kv_measured = deltas[len(deltas) // 2]
    kv_ratio = kv_measured / KV_PER_TOKEN
    print("  (a) KV cache growth")
    print(f"      measured   {kv_measured:>10,} B/token  ({kv_measured / KiB:.1f} KiB), "
          f"median per-step delta")
    print(f"      predicted  {KV_PER_TOKEN:>10,} B/token  (144 KiB, constants sec4)")
    print(f"      ratio      {kv_ratio:>10.2f}x  "
          f"{'-- within allocator rounding' if 0.9 <= kv_ratio <= 1.6 else '-- see raw deltas'}")
    print()

    # (b) Decode throughput vs the frozen ceiling / expectation.
    tps = new_tokens / secs
    ceiling = BW_GBPS / WEIGHTS_GB
    achieved_mbu = tps / ceiling
    implied_bw = tps * WEIGHTS_GB
    print("  (b) decode throughput")
    print(f"      decoded {new_tokens} tokens in {secs:.2f} s  =>  {tps:.2f} tok/s (measured)")
    print(f"      ceiling      {ceiling:.2f} tok/s  (273/16.38, 100% MBU)")
    print(f"      expectation  {MBU_K * ceiling:.1f} tok/s  (x{MBU_K})")
    print(f"      achieved MBU {achieved_mbu * 100:.1f}%   "
          f"(implied bandwidth {implied_bw:.0f} GB/s of {BW_GBPS:.0f})")
    if 0.55 <= achieved_mbu <= 0.80:
        print("      -> in the 60-75% band the four reference dots live in. The heuristic holds, on your box.")
    else:
        print("      -> outside the 60-75% band: likely ComfyUI contention, thermal throttle, or a")
        print("         short run. Re-run with ComfyUI down and more --new-tokens.")
    print()
    _ascii_roofline(WEIGHTS_GB, tps, ceiling)
    print("  (5) measured KV growth + decode tok/s placed on the roofline -- see ratios above")
    print()


def _ascii_roofline(weight_gb, tps, ceiling):
    """A tiny log-scale roofline: the BW/bytes ceiling line and where the measured point lands."""
    import math
    print("  roofline (log tok/s vs the 273 GB/s ceiling; * = you, . = ceiling):")
    lo, hi = 1.0, max(ceiling, tps) * 1.2
    width = 46

    def col(v):
        v = max(lo, min(hi, v))
        return int(width * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))

    row = [" "] * (width + 1)
    row[col(ceiling)] = "."
    row[col(tps)] = "*"
    print(f"    {tps:5.1f} you |{''.join(row)}")
    print(f"    {ceiling:5.1f} ceil|{' ' * width}.")
    print(f"    achieved {tps / ceiling * 100:.0f}% of the ceiling (batch-1 decode; batching is "
          f"the only lever that moves it right).")


def _how_to_run_on_spark(model_id, prompt_len, new_tokens):
    print("  To get real numbers, run this ON the Spark, in the fresh course venv:")
    print("      source ~/course/.venv/bin/activate      # NOT ~/ComfyUI/.venv")
    print(f"      python 02_kv_cache_and_roofline.py --measure --model {model_id} \\")
    print(f"             --prompt-len {prompt_len} --new-tokens {new_tokens}")
    print("  It needs: torch (cu130 wheel) + transformers, a CUDA GPU, and ~18 GB free.")
    print("  It will load Qwen3-8B, watch the KV cache grow, and time real decode tok/s.")
    print("  (5) skipped -- no GPU here; the arithmetic above stands on its own.")
    print()


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true",
                    help="arithmetic + assertions only, no GPU (the CI / no-box path). "
                         "This is what the default run already does before --measure.")
    ap.add_argument("--measure", action="store_true",
                    help="ALSO measure real KV growth + decode tok/s on the box (Spark-touching, "
                         "~20 min, contends with ComfyUI). Run this on the Spark, not here.")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model id for --measure")
    ap.add_argument("--prompt-len", type=int, default=512, help="prefill length for --measure")
    ap.add_argument("--new-tokens", type=int, default=128, help="tokens to decode for --measure")
    ap.add_argument("--seed", type=int, default=46, help="RNG seed (default 46, the page number)")
    args = ap.parse_args()

    print("02_kv_cache_and_roofline.py -- decode, the KV cache & the roofline (course p.46)")
    print(f"  python {sys.version.split()[0]}  (arithmetic path is pure-Python, no torch)")
    print("  verified against: transformers 5.14.1 · torch 2.13.0 · CUDA 13.0 (constants sec7)")
    print()

    kv_cache_section()
    roofline_section()
    mbu_section()
    failure_modes_section()

    if args.measure:
        measure_on_box(args.model, args.prompt_len, args.new_tokens, args.seed)
    else:
        print("=" * 70)
        print("ARITHMETIC + SELF-TEST PASSED (4 sections). To measure on the box, add --measure.")
        print("=" * 70)
        print("  1. KV cache      144 KiB/token [DER, sec4], H_kv=8 not H=32, GQA saves 4x")
        print("  2. roofline      decode I=1, ridge 227/458 [INF], bf16 16.67 ceil / ~10.8 expect")
        print("  3. MBU           four dots 60.3-74.7% under decode-traffic bytes [DER, sec6.7]")
        print("  4. failure modes MoE >100%-BW punchline + 18x batch-1->batch-32 decode gain")


if __name__ == "__main__":
    main()
