#!/usr/bin/env python3
"""
sample_flux.py — real generation with YOUR FLUX off disk; the DiT/MMDiT made concrete.

Course artifact for p.59 ("The Denoiser's Body II: DiT, MMDiT & the Samplers"),
diffusion manifest D8. This is the script the page's "Try it" box defers to. It runs a
real FLUX generation on a model already on your disk, exposes seed / steps / guidance /
scheduler, and — the teaching payoff — prints the three shapes the whole page is about:

    the latent grid          [1, C, 128, 128]         (from p.53: 1024/f=8 = 128 per side)
    the token sequence       [1, 4096, C*2*2]         (after the DiT's 2x2 patchify)
    peak GPU memory          measured on YOUR box      (verifying p.60 on your machine)

Then you `cat transformer/config.json` and match every row of the page's config table.

WHY IT WORKS FOR ANY FLUX (the reconciliation this file bakes in)
-----------------------------------------------------------------
The page's box points at your `flux1-dev-kontext_fp8` (fast, ~3 s/image). The build spec
(spec-code.md D8) wants the FLUX.2-dev numbers asserted. These are the SAME script: the
latent-channel count is the only thing that changes between the two, and the token count is
identical. So this file reads the channel count off whatever model you give it and checks
the frozen set that applies:

    FLUX.1  (your kontext_fp8)   latent [1,16,128,128] = 262,144 elems -> 12.0x compression
                                 tokens 4096 of width  16*2*2 =  64   in_channels  64 = 16*2^2
                                 transformer: 24 heads x 128, joint_attention_dim 4096
    FLUX.2-dev  (your HF cache)  latent [1,32,128,128] = 524,288 elems ->  6.0x compression
                                 tokens 4096 of width  32*2*2 = 128   in_channels 128 = 32*2^2
                                 transformer: 48 heads x 128, 8 dual + 48 single blocks,
                                 mlp_ratio 3.0, rope_theta 2000, joint_attention_dim 15360 = 3*5120

The "4096 tokens are the same 4096" key box on the page is literally true: doubling the VAE
channels (FLUX.1 16 -> FLUX.2 32) doubles the per-token WIDTH (64 -> 128), not the token COUNT.

Frozen numbers checked here (constants.md §9.6 + hardware-ground-truth.md §4 [VP, his disk]):
    FLUX.2 latent [1,32,128,128]  524,288 elems   3,145,728 / 524,288 = 6.0x    4096 tokens
    FLUX.1 latent [1,16,128,128]  262,144 elems   3,145,728 / 262,144 = 12.0x   4096 tokens
    FLUX.2 in_channels 128 = 32*2^2 ;  15360 = 3*5120 ;  48 heads x 128 = 6144

Usage
-----
    python sample_flux.py --self-test                  # NO GPU, NO torch: the arithmetic only
    python sample_flux.py --prompt "a red bicycle"     # generate; auto-find a FLUX on your disk
    python sample_flux.py --prompt "..." --model ~/ComfyUI/models/diffusion_models/flux1-dev-kontext_fp8.safetensors
    python sample_flux.py --prompt "..." --model black-forest-labs/FLUX.2-dev --steps 28 --guidance 4.0
    python sample_flux.py --prompt "..." --seed 0 --scheduler flowmatch --dtype bf16

SAFETY. Generation loads a multi-billion-parameter transformer and saturates the GPU for
seconds to minutes. If ComfyUI (or a training run) is live on your box it will CONTEND for
the same unified memory pool — FLUX.2-dev is 32B and heavy; prefer the fp8 FLUX.1 model, or
run when the box is idle. Nothing is installed. The only writes are the PNG(s) in --outdir
(default ./sample_flux_out/). The `--self-test` path touches no GPU and no disk.
"""

import argparse
import glob
import os
import sys

GiB = 1 << 30
GB = 10 ** 9


# --------------------------------------------------------------------------- #
# The frozen truth. Pure arithmetic — this block is what --self-test proves,
# and what the generation path cross-checks the loaded config against.
# constants.md §9.6 ; hardware-ground-truth.md §4 (measured on his disk, [VP]).
# --------------------------------------------------------------------------- #

# per-family VAE + transformer facts. Values tagged below by confidence.
FLUX_FAMILIES = {
    "flux1": {
        "label": "FLUX.1-dev family (incl. your flux1-dev-kontext_fp8)",
        "f": 8,                    # 2^(len(block_out_channels)-1); [VP]
        "latent_channels": 16,     # [VP]
        # transformer/config.json fields:
        "num_attention_heads": 24,       # [VP, constants §9.6]
        "attention_head_dim": 128,       # [VP]
        "in_channels": 64,               # = 16 * 2^2  [DER]
        "joint_attention_dim": 4096,     # [VP]
        "mlp_ratio": 4.0,                # [VP]
        "rope_theta": 10000,             # [VP]
        # the 19 double + 38 single split is [EST] (medium) in constants — NOT asserted.
        "blocks_note": "19 double + 38 single  [EST — verify against FluxTransformer2DModel]",
        "text_hidden": None,             # T5-XXL path; joint_attention_dim is not 3*h here
    },
    "flux2": {
        "label": "FLUX.2-dev (your HF cache; every value read off your disk)",
        "f": 8,                    # exactly 3 downsamplers in ae.safetensors -> 2^3  [VP]
        "latent_channels": 32,     # decoder.conv_in.weight [512,32,3,3]  [VP]
        "num_attention_heads": 48,       # [VP, hardware-ground-truth §4]
        "attention_head_dim": 128,       # [VP]
        "in_channels": 128,              # = 32 * 2^2 packing  [VP]
        "joint_attention_dim": 15360,    # = 3 * 5120  [VP]
        "mlp_ratio": 3.0,                # [VP]
        "rope_theta": 2000,              # [VP]
        "num_dual_blocks": 8,            # [VP]
        "num_single_blocks": 48,         # [VP]
        "blocks_note": "8 dual + 48 single  [VP, his transformer/config.json]",
        "text_hidden": 5120,             # Mistral-3 hidden; 15360 = 3 * 5120  [VP]
    },
}


def latent_geometry(height, width, f, latent_channels, patch=2):
    """The p.53 -> p.59 pipeline, as arithmetic. No tensors, no GPU.

    pixels -> VAE (factor f, C channels) -> latent grid -> 2x2 patchify -> token sequence.
    Returns every number the page prints, so both --self-test and the real run agree.
    """
    assert height % (f * patch) == 0 and width % (f * patch) == 0, (
        f"{height}x{width} not divisible by f*patch = {f*patch}")
    h_lat, w_lat = height // f, width // f
    elements = latent_channels * h_lat * w_lat
    pixels = 3 * height * width
    compression = pixels / elements
    tokens_h, tokens_w = h_lat // patch, w_lat // patch
    tokens = tokens_h * tokens_w
    token_width = latent_channels * patch * patch      # packed 2x2 patch of C channels
    return {
        "height": height, "width": width, "f": f, "latent_channels": latent_channels,
        "latent_shape": [1, latent_channels, h_lat, w_lat],
        "elements": elements, "pixels": pixels, "compression": compression,
        "tokens": tokens, "tokens_grid": (tokens_h, tokens_w),
        "token_width": token_width,
        "packed_shape": [1, tokens, token_width],
    }


def _check_flux2_1024(fam):
    """FLUX.2-dev @ 1024^2 — the build spec's mandated assertions (constants §9.6)."""
    g = latent_geometry(1024, 1024, fam["f"], fam["latent_channels"])
    assert g["latent_shape"] == [1, 32, 128, 128], g["latent_shape"]
    assert g["elements"] == 524_288, g["elements"]
    assert g["pixels"] == 3_145_728, g["pixels"]
    assert abs(g["compression"] - 6.0) < 1e-9, g["compression"]
    assert g["tokens"] == 4096, g["tokens"]
    assert g["token_width"] == 128, g["token_width"]
    # transformer identities
    assert fam["in_channels"] == fam["latent_channels"] * 2 * 2 == 128, fam["in_channels"]
    assert fam["joint_attention_dim"] == 3 * fam["text_hidden"] == 15360, fam["joint_attention_dim"]
    assert fam["num_attention_heads"] * fam["attention_head_dim"] == 48 * 128 == 6144
    assert fam["num_dual_blocks"] == 8 and fam["num_single_blocks"] == 48
    return g


def _check_flux1_1024(fam):
    """FLUX.1 @ 1024^2 — the page-59 'Try it' promise for flux1-dev-kontext_fp8."""
    g = latent_geometry(1024, 1024, fam["f"], fam["latent_channels"])
    assert g["latent_shape"] == [1, 16, 128, 128], g["latent_shape"]
    assert g["elements"] == 262_144, g["elements"]
    assert g["pixels"] == 3_145_728, g["pixels"]
    assert abs(g["compression"] - 12.0) < 1e-9, g["compression"]
    assert g["tokens"] == 4096, g["tokens"]            # the SAME 4096 as FLUX.2
    assert g["token_width"] == 64, g["token_width"]
    assert fam["in_channels"] == fam["latent_channels"] * 2 * 2 == 64, fam["in_channels"]
    assert fam["joint_attention_dim"] == 4096, fam["joint_attention_dim"]
    assert fam["num_attention_heads"] * fam["attention_head_dim"] == 24 * 128 == 3072
    return g


# --------------------------------------------------------------------------- #
# --self-test : the whole thing that runs WITHOUT torch or a GPU (this box).
# --------------------------------------------------------------------------- #

def self_test():
    print("=" * 74)
    print("SELF-TEST — the DiT shape arithmetic, no GPU, no model loaded")
    print("=" * 74)
    print("  Reproducing constants.md §9.6 and hardware-ground-truth.md §4 [VP, his disk].")
    print("  Every number below is asserted; a regression fails this script loudly.\n")

    f2 = _check_flux2_1024(FLUX_FAMILIES["flux2"])
    f1 = _check_flux1_1024(FLUX_FAMILIES["flux1"])

    def show(name, fam, g):
        print(f"  {name}")
        print(f"    VAE            f = {fam['f']}   latent_channels = {fam['latent_channels']}")
        print(f"    latent grid    {g['latent_shape']}   "
              f"= {g['elements']:,} elements")
        print(f"    compression    {g['pixels']:,} px / {g['elements']:,} = "
              f"{g['compression']:.1f}x")
        print(f"    2x2 patchify   -> tokens {g['packed_shape']}  "
              f"({g['tokens_grid'][0]}x{g['tokens_grid'][1]} = {g['tokens']:,} tokens, "
              f"width {g['token_width']})")
        print(f"    transformer    {fam['num_attention_heads']} heads x "
              f"{fam['attention_head_dim']} = "
              f"{fam['num_attention_heads']*fam['attention_head_dim']} ; "
              f"in_channels {fam['in_channels']} = {fam['latent_channels']}*2^2 ; "
              f"joint_attention_dim {fam['joint_attention_dim']}")
        print(f"    blocks         {fam['blocks_note']}")
        print()

    show("FLUX.1-dev  (page-59 'Try it' target: your flux1-dev-kontext_fp8)",
         FLUX_FAMILIES["flux1"], f1)
    show("FLUX.2-dev  (build-spec D8 target: your HF cache, read off disk)",
         FLUX_FAMILIES["flux2"], f2)

    print("  KEY (page 59): the 4096 tokens are the SAME 4096.")
    print(f"    FLUX.1: {f1['tokens']:,} tokens x width {f1['token_width']}   "
          f"FLUX.2: {f2['tokens']:,} tokens x width {f2['token_width']}")
    print("    Doubling VAE channels doubles per-token WIDTH (64->128), not token COUNT.")
    print(f"    joint_attention_dim 15360 = 3 x 5120  (5120 = FLUX.2's Mistral-3 encoder hidden).")
    print()
    print("  All assertions passed. ✓")
    print("  (Run without --self-test, on your box, to generate a real image and read the")
    print("   same shapes off the live pipeline + measured peak GPU memory.)")
    return 0


# --------------------------------------------------------------------------- #
# model discovery (mirrors vae_ceiling.py's find_* helpers)
# --------------------------------------------------------------------------- #

def _expand(p):
    return os.path.abspath(os.path.expanduser(p)) if p else p


def find_flux_model(override):
    """Return (path_or_repo, is_single_file). Preference: the page's fast fp8 FLUX.1,
    then FLUX.2-dev in the HF cache. Override with --model (a repo id, a diffusers
    folder, or a single .safetensors file)."""
    if override:
        ov = _expand(override) if os.path.sep in override or override.startswith("~") else override
        if os.path.isfile(ov) and ov.endswith(".safetensors"):
            return ov, True
        return ov, False

    # 1) the page's target: a ComfyUI single-file fp8 FLUX.1 (fast, ~3 s/image)
    for d in ("~/ComfyUI/models/diffusion_models", "~/ComfyUI/models/checkpoints",
              "~/comfyui/models/diffusion_models"):
        for pat in ("*flux1*dev*kontext*fp8*.safetensors", "*flux1*dev*fp8*.safetensors",
                    "*flux1*dev*.safetensors"):
            hits = sorted(glob.glob(os.path.join(_expand(d), pat)))
            if hits:
                return hits[0], True

    # 2) FLUX.2-dev in the HF cache (a diffusers repo — heavier, 32B)
    hub = _expand("~/.cache/huggingface/hub")
    for pat in ("*FLUX.2-dev*", "*FLUX*2*dev*", "*FLUX.1-dev*"):
        for h in sorted(glob.glob(os.path.join(hub, pat))):
            snaps = sorted(glob.glob(os.path.join(h, "snapshots", "*")))
            if snaps and os.path.exists(os.path.join(snaps[-1], "model_index.json")):
                return snaps[-1], False
    return None, False


def _load_pipeline(model, is_single_file, dtype):
    """Load a FLUX pipeline defensively. `DiffusionPipeline.from_pretrained` reads
    model_index.json and instantiates the correct class itself (Flux2Pipeline /
    FluxPipeline / FluxKontextPipeline) — so we do not hard-name a class we cannot
    verify. For a single .safetensors we try the Kontext then the plain Flux loader,
    guarded by hasattr so an absent class is skipped, not an ImportError."""
    import torch
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
        except Exception as e:                          # noqa: BLE001 — desk-checked path
            errs.append(f"{cls_name}: {e}")
    raise RuntimeError(
        "could not load the single-file checkpoint with any known FLUX pipeline.\n"
        "    A ComfyUI single-file fp8 model sometimes needs its text encoders/VAE\n"
        "    supplied separately. The frictionless path is a diffusers-format repo:\n"
        "    --model black-forest-labs/FLUX.2-dev   (or your cached FLUX.1-dev folder).\n"
        "    tried:\n      " + "\n      ".join(errs))


def _family_of(latent_channels):
    for key, fam in FLUX_FAMILIES.items():
        if fam["latent_channels"] == latent_channels:
            return key, fam
    return None, None


def _cross_check_config(pipe):
    """Read the live transformer + VAE config and assert the frozen table for the
    detected family. This is the 'cat transformer/config.json and match every row'
    step from the page, done in code."""
    import torch  # noqa: F401

    vae = getattr(pipe, "vae", None)
    tr = getattr(pipe, "transformer", None)
    if vae is None or tr is None:
        print("  [note] pipeline exposes no .vae/.transformer to introspect — skipping "
              "config cross-check (generation still ran).")
        return None

    boc = list(vae.config.block_out_channels)
    f = 2 ** (len(boc) - 1)
    lat_ch = int(vae.config.latent_channels)
    key, fam = _family_of(lat_ch)

    print(f"  VAE config      block_out_channels len {len(boc)} -> f = {f} ; "
          f"latent_channels = {lat_ch}  ({fam['label'] if fam else 'unknown FLUX variant'})")
    assert f == 8, f"expected FLUX VAE f=8, got {f}"

    # transformer fields — read defensively (names differ slightly across variants).
    def cfg(*names, default=None):
        for n in names:
            if hasattr(tr.config, n):
                return getattr(tr.config, n)
        return default

    heads = cfg("num_attention_heads")
    head_dim = cfg("attention_head_dim")
    in_ch = cfg("in_channels")
    jad = cfg("joint_attention_dim")
    print(f"  transformer     class {type(tr).__name__} ; "
          f"{heads} heads x {head_dim} ; in_channels {in_ch} ; joint_attention_dim {jad}")

    if fam is None:
        print("  [note] latent_channels does not match a frozen family — printing only.")
        return key

    # in_channels is the 2x2 packing of the latent channels — a hard identity.
    assert in_ch == lat_ch * 4, f"in_channels {in_ch} != latent_channels*4 = {lat_ch*4}"
    # per-family frozen checks (only assert what is [VP]/[DER], never the [EST] block split)
    if key == "flux2":
        assert heads == 48 and head_dim == 128, (heads, head_dim)
        assert in_ch == 128, in_ch
        assert jad == 15360 == 3 * 5120, jad
        print("  self-check      FLUX.2: 48x128, in_channels 128 = 32*2^2, "
              "joint_attention_dim 15360 = 3*5120  ✓  (constants §9.6 / hw-ground-truth §4)")
    elif key == "flux1":
        assert heads == 24 and head_dim == 128, (heads, head_dim)
        assert in_ch == 64, in_ch
        assert jad == 4096, jad
        print("  self-check      FLUX.1: 24x128, in_channels 64 = 16*2^2, "
              "joint_attention_dim 4096  ✓  (constants §9.6)")
    return key


def _maybe_set_scheduler(pipe, name):
    """FLUX ships FlowMatchEulerDiscreteScheduler; keep it by default. Only swap if the
    user asks and the requested class exists — and refuse to silently hand FLUX a
    non-flow-match scheduler that would corrupt the sample."""
    have = type(pipe.scheduler).__name__
    if name in (None, "keep", "default"):
        print(f"  scheduler       {have}  (FLUX's own flow-match; kept)")
        return
    import diffusers
    mapping = {
        "flowmatch": "FlowMatchEulerDiscreteScheduler",
        "flow_match_euler": "FlowMatchEulerDiscreteScheduler",
        "flowmatch_heun": "FlowMatchHeunDiscreteScheduler",
    }
    cls_name = mapping.get(name)
    if cls_name is None:
        print(f"  scheduler       [warn] '{name}' is not a flow-match scheduler; FLUX "
              f"expects flow matching (page 57/59). Keeping {have}.")
        return
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        print(f"  scheduler       [warn] {cls_name} not in this diffusers build; keeping {have}.")
        return
    pipe.scheduler = cls.from_config(pipe.scheduler.config)
    print(f"  scheduler       {have} -> {type(pipe.scheduler).__name__}")


# --------------------------------------------------------------------------- #
# the real run
# --------------------------------------------------------------------------- #

def generate(args):
    try:
        import torch
    except ImportError:
        sys.exit(
            "This path needs torch + diffusers. On the Spark, ComfyUI's venv has them:\n"
            "  ~/ComfyUI/.venv/bin/python sample_flux.py --prompt '...'\n"
            "Or, with no GPU at all, run the arithmetic:  python sample_flux.py --self-test")

    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device == "cpu":
        print("  [warn] no CUDA — running on CPU. A 12–32B FLUX on CPU is very slow;")
        print("         this is really only useful to watch the shapes flow. Consider --self-test.")

    model, is_single = find_flux_model(args.model)
    if model is None:
        sys.exit("No FLUX model found. Point --model at your flux1-dev-kontext_fp8.safetensors "
                 "or a diffusers FLUX repo (e.g. black-forest-labs/FLUX.2-dev).")

    print("=" * 74)
    print("VERSIONS")
    print("=" * 74)
    import diffusers
    print(f"  torch {torch.__version__} | diffusers {diffusers.__version__} | "
          f"device {device} | dtype {args.dtype}")
    if device == "cuda":
        free, tot = torch.cuda.mem_get_info()
        print(f"  mem_get_info    total {tot/GiB:.2f} GiB | free {free/GiB:.2f} GiB")
        if free < 0.5 * tot:
            print("  [warn] over half your unified memory is already spoken for — ComfyUI is")
            print("         probably live. FLUX.2-dev (32B) may OOM; peak numbers will be noisy.")
    print()

    print("=" * 74)
    print("MODEL")
    print("=" * 74)
    pipe = _load_pipeline(model, is_single, dtype)
    pipe = pipe.to(device)
    key = _cross_check_config(pipe)
    _maybe_set_scheduler(pipe, args.scheduler)
    print()

    # geometry we EXPECT (arithmetic; asserted), independent of the live run.
    fam = FLUX_FAMILIES.get(key) if key else None
    if fam is not None:
        expect = latent_geometry(args.height, args.width, fam["f"], fam["latent_channels"])
        print("=" * 74)
        print("EXPECTED SHAPES (from the arithmetic, before we run)")
        print("=" * 74)
        print(f"  {args.height}x{args.width}: latent {expect['latent_shape']} "
              f"= {expect['elements']:,} elems -> {expect['compression']:.1f}x ; "
              f"tokens {expect['packed_shape']} ({expect['tokens']:,})")
        print()

    # capture the packed latent the transformer actually sees, via the stable callback API.
    seen = {}

    def _cb(pipe_, step, timestep, cbk):
        lat = cbk.get("latents")
        if lat is not None and "packed" not in seen:
            seen["packed"] = tuple(lat.shape)
        return cbk

    gen = torch.Generator(device=device).manual_seed(args.seed)

    print("=" * 74)
    print(f"GENERATE  — seed {args.seed} | steps {args.steps} | guidance {args.guidance}")
    print("=" * 74)
    print(f'  prompt: "{args.prompt}"')

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    call = dict(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        generator=gen,
    )
    # callback_on_step_end is the stable diffusers step hook; guard in case a given
    # pipeline class does not accept it.
    try:
        out = pipe(callback_on_step_end=_cb,
                   callback_on_step_end_tensor_inputs=["latents"], **call)
    except TypeError:
        print("  [note] this pipeline does not take callback_on_step_end; "
              "generating without the in-loop shape capture.")
        out = pipe(**call)

    image = out.images[0]
    os.makedirs(_expand(args.outdir), exist_ok=True)
    stem = f"flux_{args.seed}_{args.steps}steps"
    path = os.path.join(_expand(args.outdir), stem + ".png")
    image.save(path)

    print()
    print("=" * 74)
    print("WHAT THE TRANSFORMER SAW")
    print("=" * 74)
    if "packed" in seen:
        ps = seen["packed"]
        print(f"  in-loop latent  {list(ps)}   <- the TOKEN SEQUENCE, live off the pipeline")
        if len(ps) == 3:
            print(f"                  {ps[1]:,} tokens x width {ps[2]}  "
                  f"(this is the 4096 the page is about)")
            if fam is not None:
                assert ps[1] == expect["tokens"], (ps[1], expect["tokens"])
                print(f"  self-check      live token count {ps[1]:,} == "
                      f"arithmetic {expect['tokens']:,}  ✓")
    else:
        print("  (no packed latent captured — see note above; the arithmetic card still holds.)")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated()
        print(f"  peak GPU memory {peak/GiB:.2f} GiB  (torch.cuda.max_memory_allocated, measured)")
        print(f"                  of {torch.cuda.mem_get_info()[1]/GiB:.2f} GiB total on YOUR box")
    else:
        print("  peak GPU memory (n/a on CPU)")
    print(f"  wrote           {path}")

    print()
    print("-" * 74)
    print("You generated a real image, and the token sequence the DiT ran on is the same")
    print("4096 from page 53. Now `cat` your transformer/config.json and match the table.")
    print("-" * 74)
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Real FLUX generation on your disk; prints the DiT latent/token shapes "
                    "and peak memory. Use --self-test for the GPU-free arithmetic.")
    ap.add_argument("--self-test", action="store_true",
                    help="run the shape arithmetic only (no torch, no GPU) and assert every frozen number")
    ap.add_argument("--prompt", default="a red bicycle leaning on a white wall, soft morning light",
                    help="text prompt")
    ap.add_argument("--model", default=None,
                    help="repo id, diffusers folder, or single .safetensors "
                         "(default: auto-find flux1-dev-kontext_fp8, then FLUX.2-dev)")
    ap.add_argument("--steps", type=int, default=28, help="num_inference_steps")
    ap.add_argument("--guidance", type=float, default=3.5,
                    help="guidance_scale (FLUX dev is guidance-DISTILLED: this is the distilled "
                         "guidance embedding, not true CFG)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (determinism)")
    ap.add_argument("--scheduler", default="keep",
                    help="keep (default) | flowmatch | flowmatch_heun  (FLUX needs flow matching)")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--outdir", default="sample_flux_out", help="where the PNG goes")
    args = ap.parse_args()

    # emit UTF-8 regardless of the host console codepage (his Spark is UTF-8 already;
    # a Windows cp1252 console would otherwise choke on the check-mark glyphs).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                   # noqa: BLE001
        pass

    if args.self_test:
        return self_test()
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
