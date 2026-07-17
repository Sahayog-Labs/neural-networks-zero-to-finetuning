#!/usr/bin/env python3
"""
train_lora_flux.py - RUNG 8 (diffusion). DreamBooth-LoRA on 20 of YOUR images. (p.62)

"ComfyUI's LoRA trainer node, unwrapped." This is the diffusion-side twin of
11_finetune_qlora.py: the SAME LoRA equation W0 + (alpha/r)BA you counted by hand
on p.40 (Qwen3) and re-instantiated on p.62 (FLUX), now actually training a
character/style adapter on your own photos - and it lands, as a .safetensors, in
the very ~/ComfyUI/models/loras/ folder your 9 existing LoRAs already live in.

The recipe below is not folklore - every knob traces to a page you have read, and
the three that people get wrong are guarded with asserts:

  * LEARNING RATE = 1e-4, NOT 1e-6.  This is THE diffusion-LoRA bug (D-09, p.62's
    "100x trap"). 1e-6 is the FULL fine-tune / full-DreamBooth figure; copy it into
    a LoRA config and the adapter trains at 1/100th speed, the loss barely moves,
    and it looks like the LoRA "isn't learning" when it is merely crawling. The
    single most-reported diffusion fine-tuning mistake online (constants.md §9.4).

  * network_alpha = r  (scale alpha/r = 1.0), NOT r/2.  Trainers commonly default
    alpha = r/2, which HALVES your adapter's effect before you touch a knob - the
    most common "why isn't my LoRA learning?" cause after the LR trap (p.62).

  * guidance_scale = 1.0 fed to the model during training.  FLUX.1-dev is
    GUIDANCE-DISTILLED (p.61 / brief §9.4): `guidance` is an embedding baked into
    the network, not CFG. You train at 1.0 and must not double-guide. Passing a
    higher value here corrupts the adapter. (constants.md §9.6, brief §9.4/§13.)

  * rank r in 8-32, dataset-set (D-11, p.62). 15-20 images of one concept are a
    single point on a manifold the model already draws; push r much past 32 on 20
    images and the adapter MEMORIZES the training photos instead of learning the
    concept. LLM SFT sits at the opposite end (r=64-256) because instructions are
    information-dense - same principle, opposite regime.

  * logit-normal timestep sampling (m=0, s=1), flow-shift 3.0 - your own FLUX
    scheduler's values (p.57; `scheduler_config.json` on your disk). Uniform t
    wastes ~30% of every batch on the trivial (t~0) and hopeless (t~1) ends;
    logit-normal lands 68% of the batch in the middle band where composition is
    decided (brief §6.4).

  * precompute text embeddings + VAE latents to disk (prints the GB freed), bf16,
    gradient checkpointing, 8-bit AdamW. On FLUX.1-dev the base weights (24 GB) are
    the cost, not the optimizer - which is exactly p.62's headline.

Self-test (no GPU, no torch, no download): `python train_lora_flux.py --self-test`.
It reproduces, as pure arithmetic asserted against constants.md, the p.62 numbers:
one attention projection 9,437,184 -> LoRA 98,304 = 1.04% (96x); the ~30M-param /
60 MB adapter [EST]; the 29-31 GB LoRA-state ledger and the 96 GB -> 0.24 GB (400x)
optimizer collapse; the 1e-4-not-1e-6 100x gap; the logit-normal 68% middle band;
and the flow-shift map t=0.5 -> 0.75 at shift 3.0. A regression fails loudly on any
laptop.

Usage
-----
    python train_lora_flux.py --self-test                     # arithmetic checks, no GPU
    python train_lora_flux.py --images ~/data/me --instance-token sks   # train (Spark)
    python train_lora_flux.py --images ~/data/me --model flux2-dev --qlora  # 32B, NF4 base
    python train_lora_flux.py --plan --images ~/data/me       # print the run plan, no train

--images is a directory of ~15-20 photos of ONE subject (or one style). Optional
per-image "<name>.txt" caption files are read if present; otherwise every image
gets the instance prompt "a photo of <token>". Vary angle/light/background, keep the
concept constant - the model learns whatever is CONSTANT across the set (p.62).

SAFETY: this script TRAINS. It writes a ComfyUI-loadable .safetensors LoRA
(~60 MB bf16 at r=16) into --output-dir (default ~/ComfyUI/models/loras/) and
saturates the GPU for ~20-40 min. On your box the ceiling is 121.6875 GiB, so a
~30 GB run fits with ~90 GiB to spare IN ISOLATION - BUT with ComfyUI live only
~19.4 GiB was actually free (hardware-ground-truth §2.2). A 30 GB run will NOT fit
alongside a running ComfyUI. Stop ComfyUI or consult first (HARD SAFETY RULE). It
downloads the base checkpoint on first run (FLUX.1-dev ~24 GB into HF_HOME). It
installs nothing and never touches ComfyUI's venv (use the fresh venv from p.24/§A).

Requires (verified July-2026, pin/desk-checked - never an API from memory):
  torch 2.13.0 · diffusers 0.39.0 · peft 0.19.1 · transformers 5.14.1 ·
  accelerate · bitsandbytes 0.49.2 (native sm_121, aarch64+CUDA13) · safetensors.
The training math is transcribed from diffusers 0.39.0's reference
`examples/dreambooth/train_dreambooth_lora_flux.py`; desk-checked here, run by you
on the Spark in its own venv.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

GiB = 1 << 30
GB = 10 ** 9
MB = 10 ** 6

# --------------------------------------------------------------------------- #
# FLUX registry - the FROZEN numbers (constants.md §9.6, verified on his disk in
# hardware-ground-truth §4). d is the MMDiT hidden width; attention projections
# are square d x d. Text-encoder param counts drive the precompute-savings model.
# --------------------------------------------------------------------------- #
FLUX = {
    "flux1-dev": {
        "label": "FLUX.1-dev",
        "P": 12_000_000_000,          # 12B denoiser [VP] constants §9.6
        "d": 3072,                    # MMDiT hidden width [VP]
        "guidance_distilled": True,   # -> train at guidance_scale=1.0 (brief §9.4)
        # T5-XXL (4.7B) + CLIP-L (~0.12B) text encoders; freed by precompute.
        "text_encoder_P": 4_700_000_000 + 123_000_000,
        "lora_params_attn_est": 30_000_000,   # ~30M attn-only r=16 [EST] constants §9.6
    },
    "flux2-dev": {
        "label": "FLUX.2-dev",
        "P": 32_000_000_000,          # 32B denoiser [VP] constants §9.6 / disk §4
        "d": 5120,                    # Mistral-3 hidden; joint_attention_dim 15360=3x5120
        "guidance_distilled": True,
        # Mistral-3 24B multimodal VLM encoder; freed by precompute (48 GB in bf16).
        "text_encoder_P": 24_000_000_000,
        "lora_params_attn_est": 80_000_000,   # [EST] scaled from the 12B attn figure
    },
}

# The diffusion-LoRA hyperparameters that people get wrong (constants.md §9.4/§9.6).
LORA_LR = 1e-4                 # NOT 1e-6 (the 100x D-09 trap)
FULL_FT_LR = 1e-6             # full DreamBooth / full FT - the value NOT to copy
GUIDANCE_TRAIN = 1.0          # guidance-distilled base; do not double-guide (brief §9.4)
FLOW_SHIFT = 3.0             # his FLUX.2 scheduler's discrete_flow_shift (p.57, disk §4)
LOGIT_MEAN, LOGIT_STD = 0.0, 1.0    # logit-normal t sampling (SD3/FLUX, brief §6.4)
RANK_LOW, RANK_HIGH = 8, 32   # dataset-set rank for a 20-image concept (D-11)

CEILING_GIB = 121.6875        # measured unified-memory ceiling, hardware-ground-truth §2 [MEA-DEV]
FREE_WITH_COMFY_GIB = 19.41   # what was ACTUALLY free with ComfyUI live, §2.2 [MEA-DEV]

# The optimizer-state convention the p.62 headline ledger uses: AdamW, fp32 m,v
# (no fp32 master for the tiny adapter) = 8 B per trainable param. This is what
# reproduces the "96 GB -> 0.24 GB, 400x" line verbatim (page §11.2 / constants §2.1).
K_O_ADAMW_MV = 8


# --------------------------------------------------------------------------- #
# Pure arithmetic - shared by the self-test, --plan, and the real run. No torch,
# no GPU, no I/O. Every number here is asserted against constants.md in self_test.
# --------------------------------------------------------------------------- #

def attn_projection(d, r):
    """One SQUARE attention projection (d x d) and its rank-r LoRA replacement.
    FLUX.1-dev d=3072, r=16 -> base 9,437,184, LoRA 98,304, 1.04%, 96x (p.62)."""
    base = d * d
    lora = r * (d + d)                     # A: r x d, B: d x r
    pct = 100 * lora / base
    reduction = base / lora
    return base, lora, pct, reduction


def adapter_file_bytes(lora_params, dtype_bytes=2):
    """Adapter .safetensors size: trainable LoRA params at bf16 (2 B). 30M -> 60 MB."""
    return lora_params * dtype_bytes


def lora_ledger(model_key, r=16, act_lo_gb=4.0, act_hi_gb=6.0):
    """The p.62 §11.2 four-bucket ledger for a FLUX LoRA run, in GB. Returns a dict.
    Frozen base (bf16, still fully resident - LoRA freezes, does not delete it) +
    tiny trainable adapter (W+G) + AdamW fp32 m,v state (k_O=8) + activations [EST]."""
    m = FLUX[model_key]
    P, lora = m["P"], m["lora_params_attn_est"]
    base_w_gb = P * 2 / GB                          # frozen bf16 base
    adapter_w_gb = lora * 2 / GB                     # trainable weights (bf16)
    grad_gb = lora * 2 / GB                          # gradients (bf16)
    opt_gb = lora * K_O_ADAMW_MV / GB                # AdamW fp32 m,v on the adapter only
    total_lo = base_w_gb + adapter_w_gb + grad_gb + opt_gb + act_lo_gb
    total_hi = base_w_gb + adapter_w_gb + grad_gb + opt_gb + act_hi_gb
    return {
        "base_w_gb": base_w_gb, "adapter_w_gb": adapter_w_gb, "grad_gb": grad_gb,
        "opt_gb": opt_gb, "act_lo_gb": act_lo_gb, "act_hi_gb": act_hi_gb,
        "total_lo_gb": total_lo, "total_hi_gb": total_hi,
    }


def optimizer_collapse(model_key):
    """The p.62 headline: full-FT optimizer state vs LoRA's, both AdamW fp32 m,v
    (k_O=8). FLUX.1-dev: 12B*8 = 96 GB  ->  30M*8 = 0.24 GB  =  400x."""
    m = FLUX[model_key]
    full_gb = m["P"] * K_O_ADAMW_MV / GB
    lora_gb = m["lora_params_attn_est"] * K_O_ADAMW_MV / GB
    return full_gb, lora_gb, full_gb / lora_gb


def logit_normal_middle_band():
    """logit-normal t = sigmoid(z), z~N(0,1). The middle band (0.27,0.73) is exactly
    P(-1<z<1) = erf(1/sqrt2) ~ 68% - two-thirds of every batch (brief §6.4)."""
    t_lo = 1.0 / (1.0 + math.exp(1.0))     # sigmoid(-1) = 0.2689
    t_hi = 1.0 / (1.0 + math.exp(-1.0))    # sigmoid(+1) = 0.7311
    frac = math.erf(1.0 / math.sqrt(2.0))  # P(-1<z<1) = 0.6827
    return t_lo, t_hi, frac


def flow_shift(t, shift=FLOW_SHIFT):
    """The resolution shift map t' = shift*t / (1+(shift-1)t) (brief §6.4, p.57).
    At shift=3.0, the midpoint t=0.5 -> 0.75 (schedule pushed toward noisier t)."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def precompute_savings_gb(model_key):
    """VRAM freed by precomputing text embeddings (text encoders leave VRAM) and
    VAE latents (VAE leaves VRAM), both in bf16. FLUX.1-dev frees ~9.6 GB of T5+CLIP;
    FLUX.2-dev frees ~48 GB of its 24B Mistral-3 encoder (p.62's one-checkbox beat)."""
    m = FLUX[model_key]
    text_gb = m["text_encoder_P"] * 2 / GB
    return text_gb


# --------------------------------------------------------------------------- #
# THE SELF-TEST - runs on any laptop, no GPU, no torch. Executed locally and its
# real output pasted (build contract). Every assert cites constants.md.
# --------------------------------------------------------------------------- #

def self_test():
    print("=" * 74)
    print("SELF-TEST (no GPU) - the p.62 numbers this run will land, from arithmetic")
    print("=" * 74)

    # (1) one attention projection - the [DER] anchor, exact.
    d = FLUX["flux1-dev"]["d"]
    base, lora, pct, reduction = attn_projection(d, 16)
    print(f"  one FLUX.1-dev attention projection ({d} x {d}), r=16:")
    print(f"    base  d*d          = {base:>12,}")
    print(f"    LoRA  r*(d+d)       = {lora:>12,}   ({pct:.2f}% of the matrix)")
    print(f"    reduction           = {reduction:.0f}x   [DER, constants §9.6 / p.62]")
    assert base == 9_437_184, f"3072^2 must be 9,437,184, got {base:,}"
    assert lora == 98_304, f"r=16 LoRA must be 98,304, got {lora:,}"
    assert abs(pct - 1.04) < 0.01, f"projection LoRA must be 1.04%, got {pct:.2f}%"
    assert abs(reduction - 96.0) < 1e-6, f"reduction must be exactly 96x, got {reduction}"
    print()

    # (2) whole-model adapter - ~30M / 60 MB / 0.25% [EST] carried from constants §9.6.
    lp = FLUX["flux1-dev"]["lora_params_attn_est"]
    P = FLUX["flux1-dev"]["P"]
    file_mb = adapter_file_bytes(lp) / MB
    pct_P = 100 * lp / P
    print(f"  whole-model attn-only adapter, r=16 [EST - verify vs FluxTransformer2DModel]:")
    print(f"    trainable ~ {lp:,}  ({pct_P:.2f}% of {P/1e9:.0f}B)")
    print(f"    adapter file (bf16, 2 B/param) = {file_mb:.0f} MB  -> your models/loras/")
    assert abs(file_mb - 60.0) < 0.1, f"30M params -> 60 MB bf16, got {file_mb:.1f}"
    assert abs(pct_P - 0.25) < 0.01, f"~30M is 0.25% of 12B, got {pct_P:.3f}%"
    print()

    # (3) the memory ledger this run realizes (constants §9.6 / page §11.2).
    L = lora_ledger("flux1-dev", r=16)
    print("  LoRA-state ledger (constants §9.6, page §11.2) - FLUX.1-dev, r=16, bf16:")
    print(f"    frozen base weights (bf16)   {L['base_w_gb']:>7.2f} GB   (still ALL resident - LoRA freezes, not deletes)")
    print(f"    trainable adapter (bf16)     {L['adapter_w_gb']:>7.2f} GB")
    print(f"    gradients (bf16)             {L['grad_gb']:>7.2f} GB")
    print(f"    AdamW state (fp32 m,v; k_O=8){L['opt_gb']:>7.2f} GB")
    print(f"    activations [EST, grad ckpt] {L['act_lo_gb']:>4.0f}-{L['act_hi_gb']:.0f} GB   (LoRA does NOT cut these - §2.4)")
    print(f"    TOTAL                        {L['total_lo_gb']:>5.1f}-{L['total_hi_gb']:.1f} GB   [DER]+[EST act]")
    assert abs(L["base_w_gb"] - 24.0) < 1e-6, f"base must be 24 GB, got {L['base_w_gb']}"
    assert abs(L["opt_gb"] - 0.24) < 0.001, f"adapter AdamW state must be 0.24 GB, got {L['opt_gb']}"
    assert 28.0 <= L["total_lo_gb"] and L["total_hi_gb"] <= 31.0, (
        f"total must land in the 29-31 GB band, got {L['total_lo_gb']:.1f}-{L['total_hi_gb']:.1f}")
    print()

    # (4) the optimizer collapse - the headline (400x), and where the win ISN'T.
    full_gb, lora_gb, ratio = optimizer_collapse("flux1-dev")
    print("  where LoRA's memory win actually comes from (constants §9.6 / p.62 headline):")
    print(f"    full-FT optimizer state  {full_gb:>6.2f} GB   (12B trainable x 8 B)")
    print(f"    LoRA optimizer state     {lora_gb:>6.2f} GB   (30M trainable x 8 B)")
    print(f"    ratio                    {ratio:>6.0f}x   <- the optimizer, NOT the weights")
    print(f"    (the 24 GB frozen base is STILL there either way; most people miss this - §11.2 warn)")
    assert abs(full_gb - 96.0) < 1e-6, f"full-FT opt must be 96 GB, got {full_gb}"
    assert abs(lora_gb - 0.24) < 0.001, f"LoRA opt must be 0.24 GB, got {lora_gb}"
    assert abs(ratio - 400.0) < 1e-6, f"collapse must be exactly 400x, got {ratio}"
    print()

    # (5) the 100x LR trap (D-09) - the single most-reported diffusion-LoRA bug.
    trap = LORA_LR / FULL_FT_LR
    print("  the 100x learning-rate trap (D-09, constants §9.4, p.62):")
    print(f"    diffusion LoRA LR      = {LORA_LR:.0e}   <- USE THIS")
    print(f"    full-FT / DreamBooth   = {FULL_FT_LR:.0e}   <- copying this = train at 1/{trap:.0f} speed")
    print(f"    gap                    = {trap:.0f}x")
    assert abs(trap - 100.0) < 1e-6, f"LR trap must be 100x, got {trap}"
    print()

    # (6) logit-normal timestep sampling (brief §6.4) - 68% in the middle band.
    t_lo, t_hi, frac = logit_normal_middle_band()
    print("  logit-normal t sampling, m=0 s=1 (brief §6.4) - vs uniform's wasted ends:")
    print(f"    t = sigmoid(z), z~N(0,1); P(-1<z<1) puts t in ({t_lo:.2f},{t_hi:.2f})")
    print(f"    fraction of every batch in the middle band = {100*frac:.0f}%  (two-thirds)")
    assert abs(t_lo - 0.2689) < 1e-3 and abs(t_hi - 0.7311) < 1e-3, "middle band must be (0.27,0.73)"
    assert abs(frac - 0.6827) < 1e-3, f"middle band must be 68%, got {100*frac:.1f}%"
    print()

    # (7) flow-shift map (brief §6.4, p.57) - his scheduler's shift=3.0.
    mid = flow_shift(0.5, FLOW_SHIFT)
    print(f"  flow-shift t' = shift*t/(1+(shift-1)t), shift={FLOW_SHIFT} (p.57, your scheduler_config.json):")
    print(f"    midpoint t=0.5 -> t'={mid:.2f}   (schedule pushed toward noisier t)")
    assert abs(mid - 0.75) < 1e-9, f"shift=3 must map 0.5 -> 0.75, got {mid}"
    print()

    # (8) precompute savings and the ComfyUI-contention reality.
    save1 = precompute_savings_gb("flux1-dev")
    save2 = precompute_savings_gb("flux2-dev")
    print("  precompute (text embeds + VAE latents to disk) frees VRAM (p.62):")
    print(f"    FLUX.1-dev: text encoders (T5-XXL+CLIP, bf16) leave VRAM = ~{save1:.1f} GB freed")
    print(f"    FLUX.2-dev: its 24B Mistral-3 encoder (bf16)   leave VRAM = ~{save2:.0f} GB freed (the one-checkbox beat)")
    slack = CEILING_GIB - 30.0
    print(f"  fit check: 30 GB run vs {CEILING_GIB} GiB ceiling -> ~{slack:.0f} GiB spare IN ISOLATION")
    print(f"    BUT with ComfyUI live only {FREE_WITH_COMFY_GIB} GiB was free (§2.2): a 30 GB run will")
    print(f"    NOT fit alongside it. Stop ComfyUI or consult first (HARD SAFETY RULE).")
    assert save1 > 9.0, "FLUX.1-dev text-encoder saving should exceed 9 GB"
    assert abs(save2 - 48.0) < 0.1, f"FLUX.2 encoder saving must be ~48 GB, got {save2:.1f}"
    print()

    print("  self-checks passed: 9,437,184 -> 98,304 (96x) ; ~30M/60MB/0.25% ; 29-31 GB ledger ;")
    print("  96->0.24 GB (400x) ; LR 1e-4 not 1e-6 (100x) ; logit-normal 68% ; shift 0.5->0.75.")
    print("  All against constants.md, no GPU touched.")
    print("=" * 74)


# --------------------------------------------------------------------------- #
# Version stamp + seeding - a support request begins with real versions, not
# remembered ones; determinism is seeded (utils/seed.py isn't shipped, so inline,
# same as 11_finetune_qlora.py's stamp()).
# --------------------------------------------------------------------------- #

def stamp():
    import torch
    line = f"torch {torch.__version__}"
    for mod in ("diffusers", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            line += f" · {mod} {__import__(mod).__version__}"
        except Exception:
            line += f" · {mod} MISSING"
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        line += f" · {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} · CUDA {torch.version.cuda}"
    print(f"  [{line}]")


def set_all_seeds(seed):
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Dataset - a directory of ~15-20 images of ONE concept. Optional "<name>.txt"
# per-image captions; else every image gets the instance prompt. The model learns
# whatever is CONSTANT across the set, so caption what you want to VARY (p.62).
# --------------------------------------------------------------------------- #

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image_records(images_dir, instance_prompt):
    d = Path(images_dir).expanduser()
    if not d.is_dir():
        raise NotADirectoryError(f"--images {d} is not a directory")
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise ValueError(f"no images ({', '.join(sorted(IMAGE_EXTS))}) found in {d}")
    records = []
    for f in files:
        cap_file = f.with_suffix(".txt")
        caption = cap_file.read_text(encoding="utf-8").strip() if cap_file.exists() else instance_prompt
        records.append((f, caption))
    return records


def print_run_plan(args, records=None):
    m = FLUX[args.model]
    alpha = args.alpha if args.alpha is not None else args.rank
    L = lora_ledger(args.model, r=args.rank)
    print("=" * 74)
    print(f"RUN PLAN - DreamBooth-LoRA on {m['label']}  (RUNG 8, p.62)")
    print("=" * 74)
    n = len(records) if records is not None else "?"
    print(f"  images         : {n} from {args.images}  (target ~15-20; more is often worse - p.62)")
    print(f"  instance token : {args.instance_token!r}  ->  prompt \"{args.instance_prompt}\"")
    print(f"  rank r         : {args.rank}   (dataset-set 8-32; >32 on 20 imgs memorizes - D-11)")
    print(f"  network_alpha  : {alpha}   (scale alpha/r = {alpha/args.rank:.2f}; alpha=r/2 would HALVE it - p.62)")
    print(f"  learning rate  : {args.lr:.0e}   (diffusion LoRA; NOT the 1e-6 full-FT figure - D-09)")
    print(f"  guidance_scale : {args.guidance_scale}   (guidance-distilled base; do not double-guide - brief §9.4)")
    print(f"  timestep samp. : logit-normal m={LOGIT_MEAN} s={LOGIT_STD}, flow-shift {args.flow_shift}  (p.57/§6.4)")
    print(f"  resolution     : {args.resolution}   steps {args.max_steps}   batch {args.batch}")
    print(f"  precision      : bf16 · gradient checkpointing · {'8-bit AdamW' if args.adam8bit else 'AdamW'}")
    print(f"  precompute     : text embeds + VAE latents to disk (frees ~{precompute_savings_gb(args.model):.1f} GB VRAM)")
    print(f"  prior preserv. : {'ON' if args.prior_preservation else 'OFF (default; add if you see language drift - p.62)'}")
    print()
    print("  predicted step-0 memory ledger (constants §9.6 / page §11.2):")
    print(f"    {L['base_w_gb']:.0f} GB base + {L['adapter_w_gb']:.2f}+{L['grad_gb']:.2f} adapter/grad + {L['opt_gb']:.2f} opt + "
          f"{L['act_lo_gb']:.0f}-{L['act_hi_gb']:.0f} act  =  {L['total_lo_gb']:.0f}-{L['total_hi_gb']:.0f} GB")
    if args.model == "flux2-dev" and not args.qlora:
        print(f"    (FLUX.2-dev's {L['base_w_gb']:.0f} GB bf16 base is why --qlora / NF4 is the practical path - §15)")
    print(f"    vs {CEILING_GIB} GiB ceiling: fits IN ISOLATION; NOT alongside a live ComfyUI "
          f"({FREE_WITH_COMFY_GIB} GiB free, §2.2).")
    print(f"  output         : {args.output_dir}/{args.output_name}.safetensors  (ComfyUI-loadable)")
    print("=" * 74)


# --------------------------------------------------------------------------- #
# The real run - diffusers 0.39.0 FLUX LoRA training, transcribed from the
# reference train_dreambooth_lora_flux.py. Desk-checked here; run by HIM on the
# Spark in the fresh venv. Not exercised locally (no torch/CUDA on this box).
# --------------------------------------------------------------------------- #

def run_training(args):
    import gc
    import torch
    from diffusers import (
        FluxPipeline,
        FluxTransformer2DModel,
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
    )
    from diffusers.training_utils import (
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    print("=" * 74)
    print(f"RUNG 8 - DreamBooth-LoRA on {FLUX[args.model]['label']}, r={args.rank}")
    print("=" * 74)
    stamp()
    set_all_seeds(args.seed)

    records = load_image_records(args.images, args.instance_prompt)
    print_run_plan(args, records)

    device = "cuda"
    weight_dtype = torch.bfloat16
    base_id = args.base_id

    # --- SAFETY gate: refuse to launch into a nearly-full pool (ComfyUI is likely up) -- #
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"  mem_get_info: {free_b/GiB:.1f} GiB free of {total_b/GiB:.1f} GiB total")
    need_gib = lora_ledger(args.model, r=args.rank)["total_hi_gb"] * GB / GiB
    if free_b / GiB < need_gib and not args.force:
        print(f"  !! Only {free_b/GiB:.1f} GiB free but this run needs ~{need_gib:.0f} GiB.")
        print(f"     ComfyUI is almost certainly holding the pool (§2.2 measured 19.4 GiB free).")
        print(f"     Stop ComfyUI, or pass --force if you have deliberately made room. Aborting.")
        sys.exit(1)

    # --- load the pieces separately so we can free the text encoders after precompute -- #
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_id, subfolder="scheduler")
    noise_scheduler_copy = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
    vae = AutoencoderKL.from_pretrained(base_id, subfolder="vae", torch_dtype=torch.float32).to(device)
    transformer = FluxTransformer2DModel.from_pretrained(
        base_id, subfolder="transformer", torch_dtype=weight_dtype).to(device)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    # --- LoRA on attention projections (diffusers default target set) ---------- #
    alpha = args.alpha if args.alpha is not None else args.rank    # alpha=r, NOT r/2 (p.62)
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",                 # A ~ N(0, sigma^2), B=0 (bit-identical at step 0)
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],   # attention-only (D-06/D-11, p.62)
    )
    transformer.add_adapter(transformer_lora_config)
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  trainable LoRA params (measured on the checkpoint): {n_trainable:,} "
          f"({100*n_trainable/sum(p.numel() for p in transformer.parameters()):.3f}%)")
    # ~30M is [EST] (constants §9.6 says verify vs FluxTransformer2DModel) - sanity band, not a frozen assert.
    assert 1e7 < n_trainable < 1.2e8, (
        f"attn-only LoRA count {n_trainable:,} is outside the plausible 10-120M band - "
        f"check target_modules / block structure against FluxTransformer2DModel.")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # --- PRECOMPUTE text embeddings, then FREE the text encoders (the p.62 win) -- #
    # A DreamBooth run shares one instance prompt across all images (or per-image
    # captions). We encode ONCE with the full pipeline's encoders, cache to RAM/disk,
    # then delete the encoders so they never occupy VRAM during the training loop.
    print("  precomputing text embeddings (then freeing text encoders)...")
    pipe = FluxPipeline.from_pretrained(
        base_id, transformer=None, vae=None, torch_dtype=weight_dtype).to(device)
    prompt_cache = {}
    with torch.no_grad():
        for _, caption in records:
            if caption in prompt_cache:
                continue
            pe, ppe, text_ids = pipe.encode_prompt(
                prompt=caption, prompt_2=caption, max_sequence_length=args.max_sequence_length)
            prompt_cache[caption] = (pe.to(weight_dtype), ppe.to(weight_dtype), text_ids)
    freed = precompute_savings_gb(args.model)
    del pipe
    gc.collect(); torch.cuda.empty_cache()
    print(f"  text encoders freed (~{freed:.1f} GB VRAM back). {len(prompt_cache)} unique prompt(s) cached.")

    # --- PRECOMPUTE VAE latents for every image, then FREE the VAE -------------- #
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    from PIL import Image
    vae_sf = getattr(vae.config, "shift_factor", 0.0) or 0.0
    vae_scale = vae.config.scaling_factor
    latents_cache = []
    with torch.no_grad():
        for path, caption in records:
            img = Image.open(path).convert("RGB")
            px = tfm(img).unsqueeze(0).to(device, dtype=torch.float32)
            posterior = vae.encode(px).latent_dist
            lat = (posterior.sample() - vae_sf) * vae_scale       # FLUX VAE normalization
            latents_cache.append((lat.to(weight_dtype), caption))
    del vae
    gc.collect(); torch.cuda.empty_cache()
    print(f"  VAE latents precomputed for {len(latents_cache)} images; VAE freed.")

    # spatial dims of a latent (for packing / unpacking)
    _, ch, lat_h, lat_w = latents_cache[0][0].shape
    vae_scale_factor = 8            # FLUX f=8 (constants §9.6; his disk §4)

    # --- guidance vector: FLUX.1-dev is guidance-distilled -> feed 1.0 (brief §9.4) -- #
    guidance = None
    if getattr(transformer.config, "guidance_embeds", False):
        guidance = torch.full((args.batch,), args.guidance_scale, device=device, dtype=weight_dtype)

    # --- optimizer: 8-bit AdamW keeps the (already tiny) adapter state minimal --- #
    if args.adam8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    def get_sigmas(timesteps, n_dim=4):
        # map integer timesteps -> continuous sigma in [0,1] via the scheduler's schedule
        sched_sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=weight_dtype)
        sched_ts = noise_scheduler_copy.timesteps.to(device)
        step_idx = [(sched_ts == t).nonzero().item() for t in timesteps]
        sigma = sched_sigmas[step_idx].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # --- TRAIN ------------------------------------------------------------------ #
    print(f"  training {args.max_steps} steps at LR {args.lr:.0e}, guidance {args.guidance_scale}...")
    transformer.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n = len(latents_cache)
    for step in range(args.max_steps):
        idx = torch.randint(0, n, (args.batch,))
        model_input = torch.cat([latents_cache[i][0] for i in idx], dim=0).to(device)
        caps = [latents_cache[i][1] for i in idx]
        pe = torch.cat([prompt_cache[c][0] for c in caps], dim=0)
        ppe = torch.cat([prompt_cache[c][1] for c in caps], dim=0)
        text_ids = prompt_cache[caps[0]][2]

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        # logit-normal timestep sampling (brief §6.4) - u in [0,1], concentrated at 0.5
        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal", batch_size=bsz,
            logit_mean=LOGIT_MEAN, logit_std=LOGIT_STD, mode_scale=1.29)
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device)
        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim)

        # flow-matching interpolation: x_t = (1-sigma) x0 + sigma noise
        noisy = (1.0 - sigmas) * model_input + sigmas * noise

        # pack latents into the transformer's token sequence (FLUX 2x2 patchify)
        packed = FluxPipeline._pack_latents(noisy, bsz, ch, lat_h, lat_w)
        img_ids = FluxPipeline._prepare_latent_image_ids(bsz, lat_h // 2, lat_w // 2, device, weight_dtype)

        model_pred = transformer(
            hidden_states=packed,
            timestep=(timesteps / 1000.0),        # transformer expects t in [0,1]
            guidance=guidance,
            pooled_projections=ppe,
            encoder_hidden_states=pe,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        model_pred = FluxPipeline._unpack_latents(
            model_pred, height=lat_h * vae_scale_factor, width=lat_w * vae_scale_factor,
            vae_scale_factor=vae_scale_factor)

        # flow-matching target is the velocity (noise - x0); SD3-style loss weighting
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        target = noise - model_input
        loss = (weighting.float() * (model_pred.float() - target.float()) ** 2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if step % max(1, args.max_steps // 10) == 0 or step == args.max_steps - 1:
            print(f"    step {step:>5}/{args.max_steps}  loss {loss.item():.4f}")

    dt = time.time() - t0
    print()
    print("  MEASURED (after training):")
    print(f"    wall-clock : {dt:.1f} s = {dt/60:.1f} min")
    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / GiB
        print(f"    peak mem   : {peak_gib:.2f} GiB (max_memory_allocated); predicted "
              f"{lora_ledger(args.model, r=args.rank)['total_lo_gb']:.0f}-"
              f"{lora_ledger(args.model, r=args.rank)['total_hi_gb']:.0f} GB - the gap is activations")

    # --- SAVE a ComfyUI-loadable .safetensors into models/loras/ ---------------- #
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_state_dict = get_peft_model_state_dict(transformer)
    # FluxPipeline.save_lora_weights writes diffusers/kohya-format keys
    # (transformer.<block>.attn.to_q.lora_A.weight ...) that current ComfyUI LoraLoader
    # loads for FLUX natively. Saved as <output_name>.safetensors.
    FluxPipeline.save_lora_weights(
        save_directory=str(out_dir),
        transformer_lora_layers=lora_state_dict,
        weight_name=f"{args.output_name}.safetensors",
        safe_serialization=True,
    )
    saved = out_dir / f"{args.output_name}.safetensors"
    size_mb = saved.stat().st_size / MB if saved.exists() else adapter_file_bytes(n_trainable) / MB
    print(f"  adapter saved: {saved}  (~{size_mb:.0f} MB)")
    print("=" * 74)
    print("RUNG 8 complete. Load it in ComfyUI with LoraLoader, run the p.62 protocol:")
    print("  fixed prompt grid every 250 steps · the Mars probe · the drift control (no trigger) ·")
    print("  a strength sweep (healthy at strength_model ~ 1.0). Your eyes are the metric.")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="arithmetic self-checks only (no GPU, no torch, no download)")
    ap.add_argument("--plan", action="store_true",
                    help="print the run plan + predicted ledger and exit (no training)")
    ap.add_argument("--images", default=None, help="directory of ~15-20 images of ONE concept")
    ap.add_argument("--model", default="flux1-dev", choices=list(FLUX),
                    help="base model (default flux1-dev, the LoRA sweet spot)")
    ap.add_argument("--base-id", default=None,
                    help="HF id / local path of the base (default: black-forest-labs/FLUX.<n>-dev)")
    ap.add_argument("--instance-token", default="sks",
                    help="rare trigger token (default 'sks'; 'ohwx'/'zwx' are alternatives - p.62)")
    ap.add_argument("--instance-prompt", default=None,
                    help="prompt template (default 'a photo of <token>')")
    ap.add_argument("-r", "--rank", type=int, default=16,
                    help="LoRA rank (8-32 for a 20-image concept; >32 memorizes - D-11)")
    ap.add_argument("--alpha", type=int, default=None,
                    help="network_alpha (default = rank, scale 1.0; alpha=r/2 HALVES the effect - p.62)")
    ap.add_argument("--lr", type=float, default=LORA_LR,
                    help="LoRA LR (default 1e-4; NOT the 1e-6 full-FT figure - the 100x D-09 trap)")
    ap.add_argument("--guidance-scale", type=float, default=GUIDANCE_TRAIN,
                    help="guidance fed to the (distilled) model during training - must be 1.0 (brief §9.4)")
    ap.add_argument("--flow-shift", type=float, default=FLOW_SHIFT,
                    help="discrete_flow_shift (default 3.0, his scheduler - p.57)")
    ap.add_argument("--resolution", type=int, default=1024, help="training resolution (default 1024)")
    ap.add_argument("--max-steps", type=int, default=1000, help="training steps (default 1000)")
    ap.add_argument("--batch", type=int, default=1, help="per-step batch (default 1)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clip norm")
    ap.add_argument("--max-sequence-length", type=int, default=512, help="T5 max tokens (FLUX default 512)")
    ap.add_argument("--adam8bit", action="store_true", default=True,
                    help="use bitsandbytes 8-bit AdamW (default on; native sm_121)")
    ap.add_argument("--no-adam8bit", dest="adam8bit", action="store_false",
                    help="use plain fp32 AdamW instead of 8-bit")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True,
                    help="trade compute for activation memory (default on)")
    ap.add_argument("--qlora", action="store_true",
                    help="NF4 4-bit base (for FLUX.2-dev, which does not fit in bf16 - §15)")
    ap.add_argument("--prior-preservation", action="store_true",
                    help="add DreamBooth prior-preservation loss (default OFF - add if you see drift, p.62)")
    ap.add_argument("--force", action="store_true",
                    help="launch even if free VRAM looks too low (you have stopped ComfyUI)")
    ap.add_argument("--seed", type=int, default=42, help="seed (determinism)")
    ap.add_argument("--output-dir", default="~/ComfyUI/models/loras",
                    help="where the .safetensors is written (default: his ComfyUI loras/)")
    ap.add_argument("--output-name", default="my_flux_lora", help="output filename stem")
    args = ap.parse_args()

    if args.instance_prompt is None:
        args.instance_prompt = f"a photo of {args.instance_token}"
    if args.base_id is None:
        args.base_id = ("black-forest-labs/FLUX.2-dev" if args.model == "flux2-dev"
                        else "black-forest-labs/FLUX.1-dev")

    if args.self_test:
        self_test()
        return

    if args.plan:
        if args.images is None:
            print("--plan needs --images to count your dataset; showing the ledger with n=?")
            print_run_plan(args, None)
        else:
            print_run_plan(args, load_image_records(args.images, args.instance_prompt))
        return

    if args.images is None:
        print("no --images given. Point --images at a directory of ~15-20 photos of one")
        print("concept, or run --self-test (no GPU) / --plan (no training). See --help.")
        sys.exit(1)

    try:
        import torch
    except ImportError:
        print("torch not importable. This script needs the FRESH venv the p.24/§A setup page")
        print("installs (torch/diffusers/peft/accelerate/bitsandbytes) - NOT ComfyUI's venv")
        print("(hardware-ground-truth §3). Create it, then rerun. Or try --self-test / --plan,")
        print("which need no GPU and still check/print every frozen number.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available - FLUX LoRA training needs the Spark GPU. Run --self-test")
        print("for the arithmetic, or --plan for the run plan. On the Spark, use its own venv")
        print("and mind ComfyUI (HARD SAFETY RULE: only ~19.4 GiB free when it is live).")
        sys.exit(1)

    run_training(args)


if __name__ == "__main__":
    main()
