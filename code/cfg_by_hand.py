#!/usr/bin/env python3
"""
cfg_by_hand.py — classifier-free guidance is ONE vector subtraction, in the model you run.

Course artifact for p.61 ("Conditioning: Text Encoders, Cross-Attention & CFG"),
diffusion manifest D9. The page derives CFG in five lines of Bayes and lands on:

    ~ε = ε_∅ + w_g · (ε_c − ε_∅)                       # the boxed formula on p.61

That is it. Two model predictions — prompt-on (ε_c) and prompt-off (ε_∅) — and one
`lerp` you are ALLOWED to push past 1. This script makes that formula concrete and
self-checks it, then runs YOUR real FLUX off disk to produce the w_g image strip.

WHAT IS ACTUALLY PROVABLE, AND WHERE
------------------------------------
The MATH is exact and needs no GPU. `--self-test` proves, in pure Python:
  1. the by-hand combine `ε_∅ + w_g(ε_c − ε_∅)` is BYTE-IDENTICAL to the canonical
     CFG combine every diffusion pipeline computes internally,
         noise_pred = uncond + w_g * (text − uncond)
     — it is literally the same expression, term for term (spec-code D9 self-check:
     "hand CFG matches the pipeline's at matched w_g", discharged at the formula level);
  2. at w_g = 1 it collapses to ε_c EXACTLY — "CFG off" is the honest conditional;
  3. ‖~ε − ε_∅‖ = w_g · ‖ε_c − ε_∅‖ — the norm grows LINEARLY in w_g (this IS the
     frying: the "noise" you subtract gets several times too big, §9.3 mechanism #1);
  4. the class-space analogy on the FROZEN canonical logits (constants §9.2):
     sharpening ŷ by a power w_g and renormalizing pushes mass onto the top class —
     the same move as CFG, on 3 classes instead of a latent (this is the p.61 aside);
  5. the COST the UI hides: real CFG is 2 passes/step. 20 steps @ cfg 7.5 = 40 evals.

WHY YOUR flux1-dev-kontext_fp8 IS THE INTERESTING CASE (the p.61 §"guidance ≠ cfg")
----------------------------------------------------------------------------------
Your kontext model is a FLUX.1-dev derivative, and FLUX.1-dev is GUIDANCE-DISTILLED:
the CFG behaviour was baked into the weights, and `guidance` is a NUMBER fed in like the
timestep t — ONE forward pass, no separate unconditional branch. So on THIS model,
"real 2-pass CFG by hand" is exactly the mistake the page warns against: you would be
double-guiding an already-guided model — slower and worse (constants §9.4). The GPU path
here therefore sweeps the DISTILLED `guidance_scale` (the knob that actually moves your
kontext model) across 1..8 and shows all N images cost the SAME one-pass-per-step compute
— `guidance` is an embedding, not a second pass. It also PRINTS exactly what a true
2-pass by-hand CFG loop WOULD require, and why you should not run it on this model.

    (The un-distilled variant — two transformer calls per step, combined by hand — is
     desk-checked but NOT shipped as live code: this repo's briefs do not pin a verified
     FluxPipeline true-CFG / transformer-forward signature, and the house rule is to never
     ship a diffusers API from memory. Verify `FluxPipeline.__call__`'s `true_cfg_scale` /
     `negative_prompt` against YOUR installed diffusers before wiring that path.)

Usage
-----
    python cfg_by_hand.py --self-test                 # NO GPU, NO torch: the math + assertions
    python cfg_by_hand.py --prompt "a red bicycle"    # sweep guidance 1..8 on your FLUX, save strip
    python cfg_by_hand.py --prompt "..." --wg 1 3 5 7  # custom guidance values
    python cfg_by_hand.py --prompt "..." --model ~/ComfyUI/models/diffusion_models/flux1-dev-kontext_fp8.safetensors

SAFETY. The generation path loads a multi-billion-parameter FLUX transformer and saturates
the GPU for seconds per image. If ComfyUI (or a training run) is live on your box it will
CONTEND for the same unified memory pool — prefer the fp8 kontext model, or run when idle.
Nothing is installed. The only writes are the PNG(s) in --outdir (default ./cfg_by_hand_out/).
The `--self-test` path touches no GPU and no disk.
"""

import argparse
import glob
import math
import os
import sys

GiB = 1 << 30

# --------------------------------------------------------------------------- #
# FROZEN truth. Pure arithmetic — this is what --self-test proves.
# constants.md §9.2 (canonical logits) + notation §4.4 (w_g range) + §9.3/§9.4.
# --------------------------------------------------------------------------- #

WG_MIN, WG_MAX = 1.0, 8.0                 # notation §4.4: CFG guidance scale, 1.0 – 8.0
WG_SWEET_SD15 = 7.5                        # the SD1.5 sweet spot (page 61 / §9.3)

# The canonical logits — FROZEN, [DER]. constants.md §9.2. Used everywhere the course
# needs a softmax: temperature demos, top-k, and the CFG "sharpening" aside on page 61.
CANON_LOGITS = [2.0, 1.0, 0.1]
CANON_PROBS = [0.659001, 0.242433, 0.098566]   # softmax(CANON_LOGITS)
CANON_CE_NATS = 0.417030                        # −ln(0.659001)


def cfg_combine(eps_null, eps_cond, wg):
    """CFG BY HAND — the boxed formula on page 61.

        ~ε = ε_∅ + w_g · (ε_c − ε_∅)

    This is exactly what a diffusion pipeline computes internally as
        noise_pred = noise_pred_uncond + w_g * (noise_pred_text − noise_pred_uncond).
    Same expression, term for term. Works on any vector; the model only supplies ε_∅, ε_c.
    """
    return [n + wg * (c - n) for n, c in zip(eps_null, eps_cond)]


def lerp(a, b, t):
    """Linear interpolation. CFG is lerp(uncond, cond, w_g) — but you may set t > 1."""
    return [(1.0 - t) * ai + t * bi for ai, bi in zip(a, b)]


def vnorm(v):
    return math.sqrt(sum(x * x for x in v))


def vsub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]


def softmax(z):
    m = max(z)
    ex = [math.exp(zi - m) for zi in z]
    s = sum(ex)
    return [e / s for e in ex]


def sharpen(probs, w):
    """Raise a distribution to a power w and renormalize — the CFG move in class space.
    p_i^w / Σ_j p_j^w. w=1 leaves it unchanged; w→∞ collapses onto the top class."""
    pw = [p ** w for p in probs]
    s = sum(pw)
    return [p / s for p in pw]


def real_cfg_passes(steps):
    """Real CFG needs BOTH ε_c and ε_∅ per step → 2 forward passes/step (§9.3)."""
    return 2 * steps


def distilled_passes(steps):
    """A guidance-distilled model (FLUX.1-dev/kontext/schnell) runs ONE pass/step:
    `guidance` is an embedded number, not a second branch (§9.4)."""
    return steps


# --------------------------------------------------------------------------- #
# --self-test : everything that runs WITHOUT torch or a GPU (this Windows box).
# --------------------------------------------------------------------------- #

def _check_formula():
    """The frozen assertions. A regression fails this loudly."""
    # A tiny toy pair (ε_∅, ε_c) — the model's two predictions, stand-ins for latents.
    e0 = [-0.35, 0.25, 0.10, -0.40]        # ε_∅  (prompt off)
    ec = [0.85, 0.65, -0.20, 0.30]          # ε_c  (prompt on)
    dir_vec = vsub(ec, e0)                   # the CFG direction — "what the prompt adds"
    dlen = vnorm(dir_vec)

    # (1) hand combine == canonical pipeline combine == lerp, for the whole w_g range.
    for wg in (0.0, 1.0, 2.0, 4.0, WG_SWEET_SD15, WG_MAX):
        hand = cfg_combine(e0, ec, wg)
        canonical = [u + wg * (t - u) for u, t in zip(e0, ec)]   # diffusers' own expression
        lp = lerp(e0, ec, wg)
        for a, b, c in zip(hand, canonical, lp):
            assert abs(a - b) < 1e-12, (wg, a, b)               # hand == pipeline combine
            assert abs(a - c) < 1e-12, (wg, a, c)               # == lerp(uncond, cond, w_g)

    # (2) w_g = 1 lands on ε_c (the honest conditional, "CFG off") to float precision;
    #     w_g = 0 lands on ε_∅ (prompt ignored) — exactly, since n + 0*(c-n) = n.
    assert all(abs(a - b) < 1e-12 for a, b in zip(cfg_combine(e0, ec, 1.0), ec)), \
        cfg_combine(e0, ec, 1.0)
    assert cfg_combine(e0, ec, 0.0) == e0, cfg_combine(e0, ec, 0.0)

    # (3) ‖~ε − ε_∅‖ = w_g · ‖ε_c − ε_∅‖ — norm grows LINEARLY in w_g (the frying).
    prev = -1.0
    for wg in (0.5, 1.0, 2.0, 4.0, 7.5, 8.0):
        tilde = cfg_combine(e0, ec, wg)
        step_len = vnorm(vsub(tilde, e0))
        assert abs(step_len - wg * dlen) < 1e-12, (wg, step_len, wg * dlen)
        assert step_len > prev                       # strictly increasing
        prev = step_len

    # (4) frozen canonical logits + the class-space CFG analogy (constants §9.2).
    y = softmax(CANON_LOGITS)
    for a, b in zip(y, CANON_PROBS):
        assert abs(a - b) < 1e-6, (a, b)
    assert abs(-math.log(y[0]) - CANON_CE_NATS) < 1e-5
    assert abs(sum(y) - 1.0) < 1e-12
    # sharpening by a power pushes mass onto the top class — monotone in w.
    top_prev = -1.0
    for w in (1.0, 2.0, 4.0, 8.0):
        sh = sharpen(y, w)
        assert abs(sum(sh) - 1.0) < 1e-12
        assert sh[0] > top_prev                      # top class gains mass as w rises
        top_prev = sh[0]
    assert sharpen(y, 1.0) == y or all(abs(a - b) < 1e-12 for a, b in zip(sharpen(y, 1.0), y))

    # (5) the cost the UI hides (§9.3) — exact pass-count arithmetic, uncontested.
    assert real_cfg_passes(20) == 40                 # 20 steps @ cfg 7.5 = 40 evals
    assert real_cfg_passes(28) == 56                 # FLUX.1-dev 28 steps, IF it needed real CFG
    assert distilled_passes(28) == 28                # it doesn't — distilled: 28 passes
    assert real_cfg_passes(28) == 2 * distilled_passes(28)

    # (6) w_g range is 1.0 – 8.0 (notation §4.4).
    assert (WG_MIN, WG_MAX) == (1.0, 8.0)

    return e0, ec, dir_vec, dlen


def self_test():
    print("=" * 74)
    print("SELF-TEST — CFG by hand: the formula, asserted. No GPU, no model loaded.")
    print("=" * 74)
    print("  Proving spec-code D9 against constants §9.2 / notation §4.4 / brief §9.3–9.4.")
    print("  Every number below is asserted; a regression fails this script loudly.\n")

    e0, ec, dir_vec, dlen = _check_formula()

    print("  THE FORMULA (page 61's boxed result)")
    print("    ~ε = ε_∅ + w_g · (ε_c − ε_∅)")
    print(f"    ε_∅ (prompt off) = {e0}")
    print(f"    ε_c (prompt on)  = {ec}")
    print(f"    CFG direction    = ε_c − ε_∅ = {[round(x, 3) for x in dir_vec]}   "
          f"‖·‖ = {dlen:.4f}")
    print("    This direction is the atom again — the same object you scale in a neuron's")
    print("    weighted sum, attention's QKᵀ, and LoRA's BA.  [THREAD: the dot product]\n")

    print("  IDENTITY: hand combine == the pipeline's own combine == lerp(ε_∅, ε_c, w_g)")
    print("    diffusers computes  noise_pred = uncond + w_g*(text − uncond)  — same expression.")
    print("    w_g |        ~ε (by hand)                | ‖~ε − ε_∅‖  | past ε_c by")
    print("    ----+--------------------------------------+-------------+------------")
    for wg in (0.0, 1.0, 2.0, 4.0, WG_SWEET_SD15, WG_MAX):
        tilde = cfg_combine(e0, ec, wg)
        step_len = vnorm(vsub(tilde, e0))
        past = max(0.0, wg - 1.0)
        flag = "  = ε_c (CFG off)" if wg == 1.0 else ("  = ε_∅ (prompt ignored)" if wg == 0.0 else "")
        print(f"    {wg:>3} | [{', '.join(f'{x:+.3f}' for x in tilde)}] | "
              f"{step_len:>9.4f}   | {past:>4.1f} dir-len{flag}")
    print(f"    ‖~ε − ε_∅‖ = w_g · {dlen:.4f}  — LINEAR in w_g.  THAT linear growth is the")
    print("    frying (§9.3 #1): the 'noise' you subtract gets several times too big.\n")

    print("  CFG IS EXTRAPOLATION, NOT INTERPOLATION")
    print("    w_g ≤ 1  → on the segment [ε_∅, ε_c]: the model's honest range.")
    print("    w_g > 1  → OFF the segment, past ε_c: a distribution no real image belongs to")
    print(f"              (~p_wg ∝ p(x)·p(c|x)^wg). The SD1.5 sweet spot {WG_SWEET_SD15} is OUTSIDE the data.\n")

    print("  THE SAME MOVE, IN CLASS SPACE (constants §9.2 canonical logits)")
    y = softmax(CANON_LOGITS)
    print(f"    z = {CANON_LOGITS}  →  ŷ = [{', '.join(f'{p:.6f}' for p in y)}]   "
          f"(CE = {CANON_CE_NATS} nats)")
    print("    raise ŷ to a power w_g and renormalize — mass concentrates on the top class:")
    for w in (1.0, 2.0, 4.0, 8.0):
        sh = sharpen(y, w)
        print(f"      w_g={w:>3}:  [{', '.join(f'{p:.4f}' for p in sh)}]   top = {sh[0]:.4f}")
    print("    Same as CFG sharpening a latent — like lowering a softmax temperature, on pixels.\n")

    print("  THE COST THE UI HIDES (§9.3) — real CFG is 2 forward passes per step")
    print(f"    20 steps @ cfg {WG_SWEET_SD15}:  20 × 2 = {real_cfg_passes(20)} model evaluations "
          "(the 'steps' number is half the truth).")
    print(f"    FLUX.1-dev, 28 steps:  IF it needed real CFG → {real_cfg_passes(28)} passes; "
          f"it's guidance-DISTILLED → {distilled_passes(28)} passes.")
    print("      [EST] wall-clock at ~88 ms/pass ≈ 4.9 s vs 2.5 s — but 88 ms is a bandwidth")
    print("      FLOOR, not the true step time (brief D-15): printed as illustration, NOT asserted.")
    print("    A negative prompt costs nothing extra: it just swaps ∅ → c⁻ in the pass you")
    print("    already run. On FLUX.1-dev (one pass) a negative prompt does NOTHING (§9.4).\n")

    print("  All assertions passed. ✓")
    print("  Run without --self-test, on your box, to sweep guidance 1..8 on your real FLUX")
    print("  and see the strip — with the guidance-distillation caveat for kontext printed live.")
    return 0


# --------------------------------------------------------------------------- #
# model discovery + defensive loader — transcribed from the VERIFIED sibling
# sample_flux.py (manifest D8). Same house pattern; no unverified diffusers API.
# --------------------------------------------------------------------------- #

def _expand(p):
    return os.path.abspath(os.path.expanduser(p)) if p else p


def find_flux_model(override):
    """Return (path_or_repo, is_single_file). Preference: the page's fast fp8 kontext
    FLUX.1, then any FLUX.1-dev, then FLUX.2-dev in the HF cache."""
    if override:
        ov = _expand(override) if (os.path.sep in override or override.startswith("~")) else override
        if os.path.isfile(ov) and ov.endswith(".safetensors"):
            return ov, True
        return ov, False

    for d in ("~/ComfyUI/models/diffusion_models", "~/ComfyUI/models/checkpoints",
              "~/comfyui/models/diffusion_models"):
        for pat in ("*flux1*dev*kontext*fp8*.safetensors", "*flux1*dev*fp8*.safetensors",
                    "*flux1*dev*.safetensors"):
            hits = sorted(glob.glob(os.path.join(_expand(d), pat)))
            if hits:
                return hits[0], True

    hub = _expand("~/.cache/huggingface/hub")
    for pat in ("*FLUX.1-dev*", "*FLUX.2-dev*", "*FLUX*2*dev*"):
        for h in sorted(glob.glob(os.path.join(hub, pat))):
            snaps = sorted(glob.glob(os.path.join(h, "snapshots", "*")))
            if snaps and os.path.exists(os.path.join(snaps[-1], "model_index.json")):
                return snaps[-1], False
    return None, False


def _load_pipeline(model, is_single_file, dtype):
    """Load a FLUX pipeline defensively (verified pattern from sample_flux.py). We do NOT
    hard-name a class we cannot verify: DiffusionPipeline reads model_index.json itself;
    for a single .safetensors we try the Kontext then the plain Flux loader, guarded by
    hasattr so an absent class is skipped, not an ImportError."""
    import diffusers
    from diffusers import DiffusionPipeline

    if not is_single_file:
        print(f"  loading (diffusers repo): {model}")
        return DiffusionPipeline.from_pretrained(model, torch_dtype=dtype)

    print(f"  loading (single file): {model}")
    errs = []
    for cls_name in ("FluxKontextPipeline", "FluxPipeline"):
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            continue
        try:
            return cls.from_single_file(model, torch_dtype=dtype)
        except Exception as e:                        # noqa: BLE001 — desk-checked path
            errs.append(f"{cls_name}: {e}")
    raise RuntimeError(
        "could not load the single-file checkpoint with any known FLUX pipeline.\n"
        "    A ComfyUI single-file fp8 model sometimes needs its text encoders/VAE\n"
        "    supplied separately. The frictionless path is a diffusers-format repo\n"
        "    (a cached FLUX.1-dev folder, or black-forest-labs/FLUX.2-dev).\n"
        "    tried:\n      " + "\n      ".join(errs))


def _report_family(pipe):
    """Read the live VAE config and report the FLUX family (verified surface only).
    kontext_fp8 is FLUX.1 → latent_channels 16, VAE f=8."""
    vae = getattr(pipe, "vae", None)
    if vae is None or not hasattr(vae, "config"):
        print("  [note] pipeline exposes no .vae to introspect — skipping family check.")
        return None
    boc = list(vae.config.block_out_channels)
    f = 2 ** (len(boc) - 1)
    lat_ch = int(vae.config.latent_channels)
    fam = "FLUX.1 (kontext/dev — guidance-DISTILLED)" if lat_ch == 16 else \
          ("FLUX.2-dev" if lat_ch == 32 else f"unknown (latent_channels={lat_ch})")
    print(f"  VAE config      f = {f}  latent_channels = {lat_ch}  →  {fam}")
    assert f == 8, f"expected FLUX VAE f=8, got {f}"
    return lat_ch


# --------------------------------------------------------------------------- #
# the real run — sweep the DISTILLED guidance on YOUR kontext model.
# --------------------------------------------------------------------------- #

def generate(args):
    try:
        import torch
    except ImportError:
        sys.exit(
            "This path needs torch + diffusers. On the Spark, ComfyUI's venv has them:\n"
            "  ~/ComfyUI/.venv/bin/python cfg_by_hand.py --prompt '...'\n"
            "Or, with no GPU at all, run the math:  python cfg_by_hand.py --self-test")

    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    wgs = args.wg
    for wg in wgs:
        if not (WG_MIN - 1e-9 <= wg <= WG_MAX + 1e-9):
            print(f"  [warn] w_g={wg} is outside the course range {WG_MIN}–{WG_MAX} "
                  f"(notation §4.4). Sweeping it anyway.")

    model, is_single = find_flux_model(args.model)
    if model is None:
        sys.exit("No FLUX model found. Point --model at your flux1-dev-kontext_fp8.safetensors "
                 "or a diffusers FLUX repo.")

    print("=" * 74)
    print("VERSIONS & SAFETY")
    print("=" * 74)
    import diffusers
    print(f"  torch {torch.__version__} | diffusers {diffusers.__version__} | "
          f"device {device} | dtype {args.dtype}")
    if device == "cpu":
        print("  [warn] no CUDA — a 12–32B FLUX on CPU is very slow. Consider --self-test.")
    if device == "cuda":
        free, tot = torch.cuda.mem_get_info()
        print(f"  mem_get_info    total {tot/GiB:.2f} GiB | free {free/GiB:.2f} GiB")
        if free < 0.5 * tot:
            print("  [warn] over half your unified memory is already spoken for — ComfyUI is")
            print("         probably live. This will contend; prefer the fp8 kontext model.")
    print()

    print("=" * 74)
    print("MODEL")
    print("=" * 74)
    pipe = _load_pipeline(model, is_single, dtype)
    pipe = pipe.to(device)
    lat_ch = _report_family(pipe)
    distilled = (lat_ch == 16)      # FLUX.1 family (your kontext) is guidance-distilled
    print()

    print("=" * 74)
    print("WHAT YOU ARE ABOUT TO SWEEP  — read this, it is the whole point of page 61")
    print("=" * 74)
    if distilled:
        print("  Your kontext model is FLUX.1-dev-derived → GUIDANCE-DISTILLED (§9.4).")
        print("  It has NO separate unconditional branch. The `guidance` you sweep below is")
        print("  a NUMBER embedded like the timestep t — ONE forward pass per step. It is")
        print("  NOT the 2-pass real CFG the formula describes. Doing real CFG by hand ON TOP")
        print("  would DOUBLE-guide an already-guided model: slower AND worse. So we sweep the")
        print("  knob that actually moves YOUR model, and watch the cost stay flat.")
    else:
        print("  This looks like a non-distilled FLUX. `guidance_scale` here is closer to a real")
        print("  guidance knob; the by-hand 2-pass combine would apply. (Verify your pipeline's")
        print("  true-CFG surface before trusting a numeric hand-vs-pipeline match — not pinned.)")
    print(f"  Real CFG WOULD cost: {args.steps} steps × 2 = {real_cfg_passes(args.steps)} passes.")
    print(f"  Distilled cost:      {args.steps} steps × 1 = {distilled_passes(args.steps)} passes.")
    print("  [EST] at ~88 ms/pass that is ≈ "
          f"{real_cfg_passes(args.steps)*0.088:.1f} s vs {distilled_passes(args.steps)*0.088:.1f} s "
          "— illustration only; 88 ms is a bandwidth floor, not the step time (brief D-15).")
    print()

    os.makedirs(_expand(args.outdir), exist_ok=True)
    gen_seed = args.seed

    print("=" * 74)
    print(f"SWEEP  — prompt: \"{args.prompt}\"  | seed {gen_seed} | steps {args.steps}")
    print("=" * 74)
    print("  w_g  | wall (s) | passes/step | file")
    print("  -----+----------+-------------+------------------------------")
    for wg in wgs:
        gen = torch.Generator(device=device).manual_seed(gen_seed)   # same seed → isolate w_g
        call = dict(prompt=args.prompt, num_inference_steps=args.steps,
                    guidance_scale=float(wg), height=args.height, width=args.width, generator=gen)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = _now()
        out = pipe(**call)
        if device == "cuda":
            torch.cuda.synchronize()
        wall = _now() - t0
        image = out.images[0]
        stem = f"cfg_wg{str(wg).replace('.', 'p')}_seed{gen_seed}"
        path = os.path.join(_expand(args.outdir), stem + ".png")
        image.save(path)
        pps = 1   # this kwarg-surface call is always ONE pipeline pass/step (distilled or not)
        peak = f"  peak {torch.cuda.max_memory_allocated()/GiB:.1f} GiB" if device == "cuda" else ""
        print(f"  {wg:>4} | {wall:>8.2f} | {pps:>11} | {os.path.basename(path)}{peak}")

    print()
    print("-" * 74)
    print("You have the strip. Line the PNGs up 1→8 and read it against page 61:")
    if distilled:
        print("  • the wall-times are ~FLAT across w_g — `guidance` is an embedding, not a 2nd")
        print("    pass. That flatness IS the proof it is not real CFG (§9.4).")
        print("  • the TRUE guidance vector ε_c − ε_∅ never gets computed on this model: there")
        print("    is no ε_∅ branch to subtract. The formula lives in --self-test; the weights")
        print("    already contain it here.")
    else:
        print("  • rising wall-time with w_g would signal a real 2-pass CFG pipeline.")
    print("  The math that all of this rests on: python cfg_by_hand.py --self-test")
    print("-" * 74)
    return 0


def _now():
    import time
    return time.perf_counter()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="CFG by hand on YOUR FLUX: sweep guidance 1..8 and print the "
                    "guidance-distillation caveat. Use --self-test for the GPU-free math.")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the CFG formula + frozen constants (no torch, no GPU)")
    ap.add_argument("--prompt", default="a red bicycle leaning on a white wall, soft morning light",
                    help="text prompt")
    ap.add_argument("--wg", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8],
                    help="guidance scales to sweep (course range 1.0–8.0, notation §4.4)")
    ap.add_argument("--model", default=None,
                    help="repo id, diffusers folder, or single .safetensors "
                         "(default: auto-find flux1-dev-kontext_fp8, then FLUX.1-dev)")
    ap.add_argument("--steps", type=int, default=20, help="num_inference_steps")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed — shared across the sweep")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--outdir", default="cfg_by_hand_out", help="where the PNGs go")
    # emit UTF-8 regardless of the host console codepage (his Spark is UTF-8; a Windows
    # cp1252 console would otherwise choke on §, ε, and the check-mark glyphs).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                 # noqa: BLE001
        pass

    args = ap.parse_args()

    if args.self_test:
        return self_test()
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
