#!/usr/bin/env python3
"""
14_merge_and_serve.py - RUNG 11, closing the loop. Merge -> GGUF -> Ollama -> measure. (p.51)

Your p.50 adapter is 87.3 MB of dW sitting beside a frozen 16.38 GB base. To ship it
you fold the two together with peft's `merge_and_unload()`, convert the merged model to
GGUF, quantize to Q4_K_M for llama.cpp/Ollama (the default local serving path, D-10),
serve it, and MEASURE your own decode tok/s against the p.46 roofline. The number that
started at 16 B/param on p.18 and hit a wall at 122.05 GiB on p.49 ends here as a merged
bf16 checkpoint that is bandwidth-, not memory-, constrained at serve time.

    THERE IS EXACTLY ONE WAY TO GET THE MERGE WRONG, AND IT IS SILENT.
    If you trained with QLoRA, the base you loaded during training was already lossy 4-bit
    NF4. Calling merge_and_unload() on THAT folds a full-precision adapter update into an
    already-quantized representation, compounding the quantization loss a SECOND time
    (constants §9.7 - "NF4 is a training format, not a serving format"). The fix: reload
    the ORIGINAL bf16 checkpoint, attach the trained adapter to that, and merge there.
    This script ships an `assert` on the merge-base dtype so it cannot happen by accident.
    That assert is the whole point of the rung; everything else is plumbing.

The pipeline (the page's three commands, wired end to end):

  1. merge   - AutoModelForCausalLM.from_pretrained(base, dtype=bf16) -> assert bf16, not
               4-bit -> PeftModel.from_pretrained(base, adapter) -> merge_and_unload() ->
               save_pretrained. Merging adds ZERO parameters: the 87.3 MB adapter vanishes
               into the 16.38 GB base (constants §1.2/§3).
  2. convert - llama.cpp `convert_hf_to_gguf.py merged/ --outtype f16`, then
               `llama-quantize merged.gguf merged-Q4_K_M.gguf Q4_K_M`. Q4_K_M is a k-quant:
               different bit-widths for different tensors (brief-llm-finetuning §13, ~4.5 GB).
  3. serve   - write a Modelfile (FROM the GGUF), `ollama create`, `ollama run`, then hit
               ollama's /api/generate and read eval_count / eval_duration -> YOUR decode
               tok/s. Compare it to p.46's roofline band (bf16 16.67 ceiling / ~10.8 expect;
               Q4 reads fewer bytes/token, so it lands HIGHER - bandwidth again).

Serving recommendation (D-10): llama.cpp/Ollama is the default local path (the officially-
documented Arm DGX Spark learning path). vLLM is the THROUGHPUT option, via the
`vllm/vllm-openai:cu130-nightly` container pinned by digest - the PyPI aarch64 wheel is
built against CUDA 12 and fails to import on the Spark's CUDA-13-only stack (constants §6.9).

Self-test (no GPU, no download, no llama.cpp): `python 14_merge_and_serve.py --self-test`.
It reproduces the 43,646,976 count / 87.3 MB adapter, the 16.38 GB merged-bf16 size, the
p.46 decode band (16.67 tok/s ceiling, ~10.8 expected, constants §6.7), and - the load-
bearing check - proves the merge-base guard REJECTS an nf4/4-bit base and ACCEPTS bf16.
All pure arithmetic + logic against constants.md, so a regression fails loudly on any laptop.

Usage
-----
    python 14_merge_and_serve.py --self-test                            # checks only, no GPU
    python 14_merge_and_serve.py --adapter ./qwen3-8b-lora --base Qwen/Qwen3-8B
    python 14_merge_and_serve.py --adapter ./qwen3-8b-lora --merged-dir ./merged --skip-serve
    python 14_merge_and_serve.py --serve-only --ollama-name qwen3-8b-mine  # just measure decode

SAFETY: the merge step loads the full 16.38 GB bf16 base into the unified-memory pool and
writes a ~16.4 GB merged checkpoint + a ~4.5 GB GGUF to --merged-dir; ensure the disk has
room (~25 GB) and, if ComfyUI is live on the Spark, that the merge's memory spike won't
contend - consult before launching (HARD SAFETY RULE: never run on the Spark unprompted).
It downloads the Qwen3-8B base on first run if not cached. It installs nothing. It calls
external binaries (llama.cpp's convert/quantize, `ollama`) via subprocess and PRINTS every
command before running it - nothing is hidden.

Requires (constants §7, pin exactly): torch 2.13.0 · transformers 5.14.1 · peft 0.19.1 ·
trl>=1.8,<2. Serving also needs a built llama.cpp (convert_hf_to_gguf.py + llama-quantize)
and Ollama on the box. Verified against these versions 2026-07-16; never an API from memory.
transformers v5 note: `from_pretrained(dtype=...)`, NOT `torch_dtype=` (deprecated); safetensors
only (`safe_serialization=False` raises) - so we save with the default safetensors path.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

GiB = 1 << 30
GB = 10 ** 9
MB = 10 ** 6

# --------------------------------------------------------------------------- #
# Qwen3-8B, FROZEN (constants.md §1.2/§3/§6.7). The merge changes none of these:
# merge_and_unload() folds dW in place and adds zero parameters.
# --------------------------------------------------------------------------- #
P_TOTAL = 8_190_735_360             # exact full parameter count, constants §1.2
LORA_PARAMS = 43_646_976            # r=16 all-linear trainable count, constants §3
BASE_BYTES_BF16 = 16_381_470_720    # safetensors total_size, constants §1.2 [VP] = merged bf16 size

# The p.46 roofline the served number is checked against (constants §6.5/§6.7).
BW_GBPS = 273.0                     # achieved-bandwidth basis, constants §6.5 [VP]
MBU_K = 0.65                        # honest batch-1 decode coefficient (~0.65), constants §6.5/§6.7
QWEN3_8B_BF16_WEIGHTS_GB = BASE_BYTES_BF16 / GB          # 16.38 GB
GGUF_Q4KM_WEIGHTS_GB = 4.5         # ~4.5 GB, brief-llm-finetuning §13 [EST] - the MEASURED file wins

# The two frozen bf16 decode numbers this script must land against (constants §6.7).
BF16_CEILING_TOKS = 16.67          # 273 / 16.38, batch-1 bf16 ceiling [DER]
BF16_EXPECT_TOKS = 10.8            # ceiling x 0.65, the honest expectation [DER]


# --------------------------------------------------------------------------- #
# Pure arithmetic + logic - shared by the self-test and the real run. No torch.
# --------------------------------------------------------------------------- #

def adapter_mb(params=LORA_PARAMS):
    """Adapter file size at bf16 (2 B/param), in MB (1e6). r=16 all-linear -> 87.3 MB."""
    return params * 2 / MB


def decode_band(weight_gb, bw_gbps=BW_GBPS, k=MBU_K):
    """The p.46 batch-1 decode roofline for a dense model whose per-token weight traffic
    is `weight_gb` gigabytes: (ceiling, expectation). ceiling = BW / weight_gb; the honest
    expectation multiplies by the measured MBU coefficient k (~0.65). constants §6.5/§6.7.
    NOTE: decode-traffic bytes are file - embedding table (§6.6); using whole-file weight_gb
    makes the ceiling a hair pessimistic, which is the safe direction to be wrong."""
    ceiling = bw_gbps / weight_gb
    return ceiling, ceiling * k


def is_valid_merge_base(dtype_str, is_4bit, has_quant_config):
    """THE RULE, as pure logic so the self-test can prove it without a GPU: a legal merge
    target is the ORIGINAL bf16 base - never a quantized/NF4 one (constants §9.7). Returns
    True only for a non-quantized bf16 base. The torch-side assert (assert_bf16_base) checks
    exactly this predicate on a live model."""
    return (dtype_str == "torch.bfloat16") and (not is_4bit) and (not has_quant_config)


# --------------------------------------------------------------------------- #
# THE SELF-TEST - runs on any laptop, no GPU, no torch, no llama.cpp.
# --------------------------------------------------------------------------- #

def self_test():
    print("=" * 72)
    print("SELF-TEST (no GPU) - the numbers and the guard this rung must land")
    print("=" * 72)

    # (1) the adapter that gets merged away - the p.50 artifact.
    amb = adapter_mb()
    merged_gb = BASE_BYTES_BF16 / GB
    merged_gib = BASE_BYTES_BF16 / GiB
    print("  (1) what merges into what (constants §1.2/§3):")
    print(f"      LoRA adapter (r=16 all-linear) : {LORA_PARAMS:>13,} params  "
          f"= {amb:.1f} MB bf16")
    print(f"      frozen bf16 base               : {P_TOTAL:>13,} params  "
          f"= {merged_gb:.2f} GB ({merged_gib:.2f} GiB)")
    print(f"      merged bf16 checkpoint         : {P_TOTAL:>13,} params  "
          f"= {merged_gb:.2f} GB  <- merge adds ZERO params; the adapter vanishes into the base")
    assert abs(amb - 87.3) < 0.1, f"adapter must be 87.3 MB, got {amb:.1f}"
    assert abs(merged_gb - 16.38) < 0.01, f"merged bf16 must be 16.38 GB, got {merged_gb:.2f}"
    print()

    # (2) the merge-base guard - the load-bearing check of the whole rung (constants §9.7).
    print("  (2) the merge-base guard (constants §9.7 - the ONE silent mistake):")
    cases = [
        ("bf16 base (the ONLY legal target)",       "torch.bfloat16", False, False, True),
        ("NF4 QLoRA base (loaded 4-bit)",           "torch.uint8",    True,  True,  False),
        ("4-bit base kept in bf16 compute dtype",   "torch.bfloat16", True,  True,  False),
        ("base carrying a quantization_config",     "torch.bfloat16", False, True,  False),
    ]
    for label, dtype_str, is_4bit, has_qc, expected in cases:
        got = is_valid_merge_base(dtype_str, is_4bit, has_qc)
        verdict = "ACCEPT" if got else "REJECT"
        print(f"      {verdict:>6}  {label}")
        assert got == expected, (
            f"merge guard wrong for {label!r}: expected {expected}, got {got}. "
            f"Merging into a quantized base compounds NF4 loss twice (constants §9.7).")
    print("      -> the guard accepts bf16 and rejects every quantized target. Correct.")
    print()

    # (3) the served decode band - p.46's roofline, the number he MEASURES against.
    bf16_ceil, bf16_exp = decode_band(QWEN3_8B_BF16_WEIGHTS_GB)
    q4_ceil, q4_exp = decode_band(GGUF_Q4KM_WEIGHTS_GB)
    print("  (3) decode roofline to check the served tok/s against (constants §6.5/§6.7):")
    print(f"      bf16   {QWEN3_8B_BF16_WEIGHTS_GB:>5.2f} GB/tok -> "
          f"ceiling {bf16_ceil:5.2f} tok/s | expect x{MBU_K} = {bf16_exp:4.1f} tok/s   [DER]")
    print(f"      Q4_K_M {GGUF_Q4KM_WEIGHTS_GB:>5.2f} GB/tok -> "
          f"ceiling {q4_ceil:5.2f} tok/s | expect x{MBU_K} = {q4_exp:4.1f} tok/s   [EST base ~4.5 GB]")
    print(f"      served Q4 should land ABOVE bf16's {bf16_exp:.1f} tok/s - it reads fewer")
    print(f"      bytes/token, and decode is bandwidth-bound (the p.46 lesson, made concrete).")
    assert abs(bf16_ceil - BF16_CEILING_TOKS) < 0.05, (
        f"bf16 ceiling must be {BF16_CEILING_TOKS} tok/s (constants §6.7), got {bf16_ceil:.2f}")
    assert abs(bf16_exp - BF16_EXPECT_TOKS) < 0.1, (
        f"bf16 expectation must be ~{BF16_EXPECT_TOKS} tok/s (constants §6.7), got {bf16_exp:.2f}")
    assert q4_ceil > bf16_ceil, "Q4 reads fewer bytes/token; its ceiling must exceed bf16's"
    print()

    print("  self-checks passed: 43,646,976 / 87.3 MB ; 16.38 GB merged bf16 ; guard REJECTS")
    print("  nf4/4-bit & ACCEPTS bf16 ; decode band 16.67 ceiling / ~10.8 expect. No GPU touched.")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Version stamp - a support request begins with real versions, not remembered ones.
# --------------------------------------------------------------------------- #

def stamp():
    import torch
    line = f"torch {torch.__version__}"
    for mod in ("transformers", "peft"):
        try:
            line += f" · {mod} {__import__(mod).__version__}"
        except Exception:
            line += f" · {mod} MISSING"
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        line += f" · {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} · CUDA {torch.version.cuda}"
    print(f"  [{line}]")


# --------------------------------------------------------------------------- #
# STEP 1 - merge. The assert on the base dtype is the whole point of the rung.
# --------------------------------------------------------------------------- #

def assert_bf16_base(model):
    """THE assert (constants §9.7): the merge target must be the ORIGINAL bf16 base, never
    the NF4/4-bit QLoRA base. Reuses is_valid_merge_base() so page prose, self-test, and the
    live check cannot drift. Raises AssertionError - by design loud, not a warning."""
    import torch
    cfg = getattr(model, "config", None)
    has_qc = getattr(cfg, "quantization_config", None) is not None
    is_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    dtype_str = str(next(model.parameters()).dtype)      # e.g. "torch.bfloat16"
    assert is_valid_merge_base(dtype_str, is_4bit, has_qc), (
        f"MERGE TARGET IS QUANTIZED (dtype={dtype_str}, is_4bit={is_4bit}, "
        f"quantization_config={'set' if has_qc else 'none'}). merge_and_unload() into an "
        f"NF4/4-bit base compounds quantization loss a SECOND time (constants §9.7). Reload "
        f"the ORIGINAL bf16 checkpoint (dtype=torch.bfloat16, no quantization_config) and "
        f"attach the adapter to THAT.")
    assert dtype_str == "torch.bfloat16", f"merge base must be bf16, got {dtype_str}"


def merge(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("=" * 72)
    print(f"STEP 1 - merge adapter {args.adapter} into bf16 base {args.base}")
    print("=" * 72)
    stamp()

    print(f"  loading base {args.base} in bf16 (dtype=torch.bfloat16, transformers v5 arg)...")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)

    # --- THE assert. If you trained QLoRA, do NOT pass the NF4 base here. --------- #
    assert_bf16_base(base)
    print(f"  base dtype confirmed bf16, not quantized - safe to merge (constants §9.7).")

    print(f"  attaching adapter {args.adapter} and calling merge_and_unload()...")
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()      # folds (alpha/r) B A into W0 in place; unwraps PEFT

    # merge adds zero params - confirm the merged model is the plain base size again.
    n_params = sum(p.numel() for p in merged.parameters())
    print(f"  merged params: {n_params:,}  (adapter's {LORA_PARAMS:,} folded in, ZERO added)")
    if "Qwen3-8B" in args.base:
        assert n_params == P_TOTAL, (
            f"merged model must have {P_TOTAL:,} params (merge adds none); got {n_params:,}")

    out = Path(args.merged_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"  saving merged bf16 checkpoint to {out} (safetensors; ~{BASE_BYTES_BF16/GB:.2f} GB)...")
    merged.save_pretrained(out)            # safetensors by default (v5: safe_serialization=False raises)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)   # GGUF conversion needs it
    print(f"  merged checkpoint + tokenizer written to {out}")
    print("=" * 72)
    return out


# --------------------------------------------------------------------------- #
# STEP 2 - convert to GGUF and quantize Q4_K_M via llama.cpp (subprocess, printed).
# --------------------------------------------------------------------------- #

def _run(cmd):
    """Print a command, then run it. Never hide a subprocess (§B: no elided steps)."""
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def convert_gguf(args, merged_dir):
    print("=" * 72)
    print("STEP 2 - convert merged bf16 -> GGUF -> quantize Q4_K_M (llama.cpp)")
    print("=" * 72)
    llama = Path(args.llama_cpp)
    convert = llama / "convert_hf_to_gguf.py"        # modern name (older: convert-hf-to-gguf.py)
    quantize = llama / "llama-quantize"
    if not convert.exists():
        print(f"  !! {convert} not found. Build llama.cpp first, or pass --llama-cpp <dir>.")
        print(f"     (the page's commands: convert_hf_to_gguf.py merged/ --outtype f16 ;")
        print(f"      llama-quantize merged.gguf merged-Q4_K_M.gguf Q4_K_M)")
        sys.exit(1)

    f16_gguf = Path(merged_dir) / "merged-f16.gguf"
    q4_gguf = Path(merged_dir) / "merged-Q4_K_M.gguf"
    _run([sys.executable, str(convert), str(merged_dir), "--outtype", "f16", "--outfile", str(f16_gguf)])
    _run([str(quantize), str(f16_gguf), str(q4_gguf), "Q4_K_M"])

    size_gb = q4_gguf.stat().st_size / GB
    print(f"  Q4_K_M GGUF: {q4_gguf}  ({size_gb:.2f} GB)  "
          f"[MEA - the ~4.5 GB [EST] from brief §13, now real]")
    print("=" * 72)
    return q4_gguf


# --------------------------------------------------------------------------- #
# STEP 3 - serve with Ollama and MEASURE decode tok/s (ollama /api/generate).
# --------------------------------------------------------------------------- #

def ollama_create(args, gguf_path):
    print("=" * 72)
    print(f"STEP 3 - ollama create {args.ollama_name} (from the GGUF) + serve")
    print("=" * 72)
    modelfile = Path(args.merged_dir) / "Modelfile"
    modelfile.write_text(f"FROM {Path(gguf_path).resolve()}\n", encoding="utf-8")
    print(f"  wrote {modelfile}:  FROM {Path(gguf_path).resolve()}")
    _run(["ollama", "create", args.ollama_name, "-f", str(modelfile)])


def measure_decode(args):
    """Hit ollama's REST API and read the real decode rate. Ollama returns eval_count
    (tokens generated) and eval_duration (ns); tok/s = eval_count / eval_duration * 1e9.
    This is the MEASURED number - no stand-in - checked against p.46's band."""
    import urllib.request

    print("  measuring decode tok/s via ollama /api/generate (his own number)...")
    payload = json.dumps({
        "model": args.ollama_name,
        "prompt": args.prompt,
        "stream": False,
        "options": {"num_predict": args.num_predict, "temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{args.ollama_url}/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        result = json.loads(resp.read())
    wall = time.time() - t0

    eval_count = result.get("eval_count")
    eval_ns = result.get("eval_duration")
    if not eval_count or not eval_ns:
        print(f"  !! ollama did not return eval_count/eval_duration; got keys {sorted(result)}")
        sys.exit(1)
    toks = eval_count / eval_ns * 1e9

    bf16_ceil, bf16_exp = decode_band(QWEN3_8B_BF16_WEIGHTS_GB)
    q4_ceil, q4_exp = decode_band(GGUF_Q4KM_WEIGHTS_GB)
    print()
    print("  MEASURED (your box, Q4_K_M served by Ollama):")
    print(f"    decode: {toks:.2f} tok/s over {eval_count} generated tokens "
          f"(wall {wall:.1f} s incl. prompt eval)")
    print("  vs p.46's roofline band (constants §6.7):")
    print(f"    bf16   ceiling {bf16_ceil:5.2f} | expect {bf16_exp:4.1f} tok/s   [DER]")
    print(f"    Q4_K_M ceiling {q4_ceil:5.2f} | expect {q4_exp:4.1f} tok/s   [EST base ~4.5 GB]")
    if toks > bf16_exp:
        print(f"    -> {toks:.1f} > bf16's {bf16_exp:.1f}: served Q4 decodes faster because it")
        print(f"       reads fewer bytes/token. Same box, same bus - fewer bytes, more tokens.")
    else:
        print(f"    -> below the bf16 expectation - check for ComfyUI contention on the bus,")
        print(f"       or a cold model load. Re-run; decode is bandwidth-bound (p.46).")
    print(f"    (cross-check with 02_kv_cache_and_roofline.py --model {args.ollama_name} "
          f"--measure-decode)")
    print("=" * 72)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the arithmetic + guard checks only (no GPU, no llama.cpp)")
    ap.add_argument("--adapter", default=None, help="trained LoRA adapter dir (from p.50)")
    ap.add_argument("--base", default="Qwen/Qwen3-8B",
                    help="ORIGINAL bf16 base - NOT the NF4 QLoRA base (constants §9.7)")
    ap.add_argument("--merged-dir", default="./merged", help="where merged model + GGUF are written")
    ap.add_argument("--llama-cpp", default="./llama.cpp",
                    help="llama.cpp dir (convert_hf_to_gguf.py + llama-quantize)")
    ap.add_argument("--ollama-name", default="qwen3-8b-mine", help="Ollama model name to create/serve")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama REST endpoint")
    ap.add_argument("--prompt", default="Explain what a KV cache is in two sentences.",
                    help="prompt for the decode measurement")
    ap.add_argument("--num-predict", type=int, default=256, help="tokens to generate when measuring")
    ap.add_argument("--timeout", type=int, default=300, help="ollama request timeout (s)")
    ap.add_argument("--skip-serve", action="store_true", help="merge + convert only; no Ollama")
    ap.add_argument("--serve-only", action="store_true",
                    help="skip merge/convert; just measure decode of an existing Ollama model")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.serve_only:
        # No merge/convert - just the decode measurement against p.46's band.
        measure_decode(args)
        return

    if not args.adapter:
        print("no --adapter given. Pass the LoRA adapter dir from p.50 (e.g. ./qwen3-8b-lora),")
        print("or run --self-test for the arithmetic + merge-guard checks (no GPU needed).")
        sys.exit(1)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not importable. This script needs the FRESH venv the Part III setup page")
        print("installs (torch/transformers/peft) - NOT ComfyUI's venv (hardware-ground-truth §3).")
        print("Or run --self-test: it checks every frozen number and the merge guard with no GPU.")
        sys.exit(1)

    merged_dir = merge(args)
    if args.skip_serve:
        print("merge complete; --skip-serve set, stopping before GGUF/Ollama.")
        return
    gguf = convert_gguf(args, merged_dir)
    ollama_create(args, gguf)
    measure_decode(args)
    print("RUNG 11 complete. You merged, quantized, served, and measured your own decode rate.")
    print("Next: 12_evaluate.py (loss != quality) and 09_spark_capability_probe.py (the TF question).")


if __name__ == "__main__":
    main()
