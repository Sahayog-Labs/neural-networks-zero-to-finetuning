#!/usr/bin/env python3
"""
vae_ceiling.py — the VAE compression ceiling, and the gap diffusion exists to close.

Course artifact for p.53 ("Latent space: the VAE your ComfyUI already runs on"),
diffusion manifest D2. This is the script the page's PREDICT box defers to: it runs the
two experiments the page only draws, on the real VAEs already on your disk.

Two experiments, one point:

  (1) ROUND-TRIP. Encode a real photo to the latent and decode it back. Print the latent
      shape, the compression ratio, and the reconstruction residual. Your VAE keeps almost
      everything — the photo comes back crisp. This is the "ceiling": how much the encoder
      can throw away and still let the decoder rebuild the image.

  (2) DECODE THE PRIOR. Sample z ~ N(0, I) at the *same* latent shape and decode it, with
      no diffusion, no sampler — straight through VAEDecode. You get colored static.

The contrast is the whole rest of the track. A VAE gives you a good *space* (compact,
navigable, sharply decodable) but NOT a good way to *sample* in it: the aggregate
distribution of real latents is a thin, structured sheet, and N(0, I) is a bad model of it.
Decoding a prior draw lands you off the sheet, in the void, which the decoder renders as
static. Diffusion is the machine that learns to walk from the shell onto the sheet.

Frozen numbers checked here (constants.md §9.6):
    SD 1.5 @ 512²   latent [1, 4, 64, 64]      = 16,384 elems   → 786,432 / 16,384 = 48×
    FLUX.2 @ 1024²  latent [1, 32, 128, 128]   = 524,288 elems  → 3,145,728 / 524,288 = 6.0×
    FLUX.2 VAE:  f = 2^(len(block_out_channels) - 1) = 8,   latent_channels = 32   [VP, his config]

Usage
-----
    python vae_ceiling.py --image myphoto.jpg
    python vae_ceiling.py --image myphoto.jpg --device cpu     # ~30 s, no GPU contention
    python vae_ceiling.py --sd15 /path/to/vae.safetensors \
                          --flux2 ~/.cache/huggingface/.../FLUX.2-dev

Model discovery. By default we look where your ComfyUI / HF cache keeps them:
  SD 1.5 VAE  — a single-file .safetensors under ~/ComfyUI/models/vae/, or any
                diffusers AutoencoderKL folder. Override with --sd15.
  FLUX.2 VAE  — the `vae/` subfolder of your FLUX.2-dev checkpoint (an AutoencoderKLFlux2).
                Override with --flux2 (point it at the FLUX.2-dev root; we add subfolder="vae").
If a model isn't found we say so and skip it; the other experiment still runs.

SAFETY. The VAE is ~84 M params — decoding one image is seconds and a few hundred MB, far
lighter than a generation. But it still touches the GPU. If ComfyUI or a training run is
live on your box, pass --device cpu so this never contends for the GPU. Nothing is
installed; the only writes are the PNG tiles in --outdir (default ./vae_ceiling_out/).
"""

import argparse
import glob
import os
import sys


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _expand(p):
    return os.path.abspath(os.path.expanduser(p)) if p else p


def load_image_tensor(path, size):
    """Load an image, center-crop square, resize to (size, size), map to [-1, 1], NCHW."""
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((size, size), Image.LANCZOS)
    import numpy as np
    a = np.asarray(img, dtype="float32") / 255.0          # HWC in [0,1]
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    return x * 2.0 - 1.0                                    # [-1, 1], the VAE's convention


def save_tile(x, path):
    """x: 1x3xHxW in [-1,1] (or raw for static) -> PNG in [0,255]."""
    import numpy as np
    from PIL import Image

    a = x.detach().float().cpu().clamp(-1, 1).squeeze(0).permute(1, 2, 0).numpy()
    a = ((a + 1.0) * 127.5).clip(0, 255).astype("uint8")
    Image.fromarray(a).save(path)
    return path


def find_sd15_vae(override):
    if override:
        return _expand(override)
    cands = []
    for d in ("~/ComfyUI/models/vae", "~/comfyui/models/vae"):
        cands += glob.glob(os.path.join(_expand(d), "*.safetensors"))
        cands += glob.glob(os.path.join(_expand(d), "*.ckpt"))
    # prefer something that looks like an SD/SDXL VAE
    for c in cands:
        n = os.path.basename(c).lower()
        if "sd" in n or "vae-ft" in n or "kl-f8" in n or "ema" in n:
            return c
    return cands[0] if cands else None


def find_flux2_root(override):
    if override:
        return _expand(override)
    hub = _expand("~/.cache/huggingface/hub")
    hits = glob.glob(os.path.join(hub, "*FLUX.2-dev*")) + glob.glob(os.path.join(hub, "*FLUX*2*dev*"))
    for h in hits:
        snaps = glob.glob(os.path.join(h, "snapshots", "*"))
        if snaps:
            return sorted(snaps)[-1]
    return None


def compression(pixels_hw, C, Hlat, Wlat):
    px = 3 * pixels_hw[0] * pixels_hw[1]
    el = C * Hlat * Wlat
    return px, el, px / el


# --------------------------------------------------------------------------- #
# the two VAEs
# --------------------------------------------------------------------------- #

def run_sd15(image, path, device, dtype, outdir):
    import torch
    from diffusers import AutoencoderKL

    print("=" * 70)
    print("SD 1.5 VAE  (4 channels)  —  512² image")
    print("=" * 70)
    if not path or not os.path.exists(path):
        print("  [skip] SD 1.5 VAE not found. Pass --sd15 <file-or-folder>.")
        return
    print(f"  loading: {path}")
    if os.path.isdir(path):
        vae = AutoencoderKL.from_pretrained(path)
    else:
        vae = AutoencoderKL.from_single_file(path)
    vae = vae.to(device=device, dtype=dtype).eval()

    x = load_image_tensor(image, 512).to(device=device, dtype=dtype)
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        z = posterior.sample()                 # 1x4x64x64
        recon = vae.decode(z).sample           # 1x3x512x512

    C, Hlat, Wlat = z.shape[1], z.shape[2], z.shape[3]
    px, el, ratio = compression((512, 512), C, Hlat, Wlat)
    resid = (recon.float() - x.float()).abs()
    print(f"  latent shape     : {list(z.shape)}   ({el:,} elements)")
    print(f"  compression      : {px:,} / {el:,} = {ratio:.1f}x")
    print(f"  recon residual   : mean |Δ| = {resid.mean():.4f}   max |Δ| = {resid.max():.4f}  (pixels in [-1,1])")

    # frozen-number self-checks (constants.md §9.6)
    assert [C, Hlat, Wlat] == [4, 64, 64], f"expected [4,64,64], got {[C, Hlat, Wlat]}"
    assert el == 16_384, el
    assert px == 786_432, px
    assert abs(ratio - 48.0) < 1e-6, ratio
    print("  self-check       : [1,4,64,64], 786,432/16,384 = 48x  ✓")

    orig = save_tile(x, os.path.join(outdir, "sd15_original.png"))
    rec = save_tile(recon, os.path.join(outdir, "sd15_reconstruction.png"))

    # decode the prior: z ~ N(0, I) at the SAME latent shape
    with torch.no_grad():
        zn = torch.randn_like(z)
        static = vae.decode(zn).sample
    st = save_tile(static, os.path.join(outdir, "sd15_prior_static.png"))
    sstd = static.float().std().item()
    print(f"  N(0,I) decode    : colored static (per-pixel std {sstd:.3f}) -> {os.path.basename(st)}")
    print(f"  wrote: {os.path.basename(orig)}, {os.path.basename(rec)}, {os.path.basename(st)}")


def run_flux2(image, root, device, dtype, outdir):
    import torch
    from diffusers import AutoencoderKLFlux2

    print("=" * 70)
    print("FLUX.2 VAE  (32 channels)  —  1024² image   [your disk]")
    print("=" * 70)
    if not root or not os.path.exists(root):
        print("  [skip] FLUX.2 checkpoint not found. Pass --flux2 <FLUX.2-dev root>.")
        return
    # accept either a FLUX.2-dev root (vae/ subfolder) or a direct vae folder
    if os.path.exists(os.path.join(root, "vae", "config.json")):
        vae = AutoencoderKLFlux2.from_pretrained(root, subfolder="vae")
        src = os.path.join(root, "vae")
    elif os.path.exists(os.path.join(root, "config.json")):
        vae = AutoencoderKLFlux2.from_pretrained(root)
        src = root
    else:
        print(f"  [skip] no vae/config.json under {root}.")
        return
    print(f"  loading: {src}")
    vae = vae.to(device=device, dtype=dtype).eval()

    boc = list(vae.config.block_out_channels)
    f = 2 ** (len(boc) - 1)
    lat_ch = int(vae.config.latent_channels)
    print(f"  config           : block_out_channels={boc}  -> f = 2^{len(boc)-1} = {f}   latent_channels = {lat_ch}")
    # config self-checks (constants.md §9.6 / hardware-ground-truth §4)
    assert f == 8, f"expected f=8, got {f}"
    assert lat_ch == 32, f"expected 32 channels, got {lat_ch}"

    x = load_image_tensor(image, 1024).to(device=device, dtype=dtype)
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        z = posterior.sample()                 # 1x32x128x128
        recon = vae.decode(z).sample           # 1x3x1024x1024

    C, Hlat, Wlat = z.shape[1], z.shape[2], z.shape[3]
    px, el, ratio = compression((1024, 1024), C, Hlat, Wlat)
    resid = (recon.float() - x.float()).abs()
    print(f"  latent shape     : {list(z.shape)}   ({el:,} elements)")
    print(f"  compression      : {px:,} / {el:,} = {ratio:.1f}x")
    print(f"  recon residual   : mean |Δ| = {resid.mean():.4f}   max |Δ| = {resid.max():.4f}  (pixels in [-1,1])")

    assert [C, Hlat, Wlat] == [32, 128, 128], f"expected [32,128,128], got {[C, Hlat, Wlat]}"
    assert el == 524_288, el
    assert px == 3_145_728, px
    assert abs(ratio - 6.0) < 1e-3, ratio
    print("  self-check       : [1,32,128,128], 3,145,728/524,288 = 6.0x, f=8, 32ch  ✓")

    orig = save_tile(x, os.path.join(outdir, "flux2_original.png"))
    rec = save_tile(recon, os.path.join(outdir, "flux2_reconstruction.png"))

    with torch.no_grad():
        zn = torch.randn_like(z)
        static = vae.decode(zn).sample
    st = save_tile(static, os.path.join(outdir, "flux2_prior_static.png"))
    sstd = static.float().std().item()
    print(f"  N(0,I) decode    : colored static (per-pixel std {sstd:.3f}) -> {os.path.basename(st)}")
    print(f"  wrote: {os.path.basename(orig)}, {os.path.basename(rec)}, {os.path.basename(st)}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def self_test():
    """No-GPU, no-model check of the frozen §9.6 compression arithmetic.

    Exercises the same compression() the GPU path uses and the same latent-shape /
    ratio / downsample-factor constants asserted inside run_sd15/run_flux2, without
    touching torch, diffusers, or a VAE. Runs in milliseconds."""
    print("vae_ceiling.py --self-test  (frozen §9.6 arithmetic, no GPU, no model)")

    # SD 1.5 @ 512^2 -> [1,4,64,64]
    px, el, ratio = compression((512, 512), 4, 64, 64)
    assert el == 16_384, el
    assert px == 786_432, px
    assert abs(ratio - 48.0) < 1e-6, ratio
    print(f"  SD 1.5  : [1,4,64,64]   {px:,}/{el:,} = {ratio:.1f}x")

    # FLUX.2 @ 1024^2 -> [1,32,128,128]
    px, el, ratio = compression((1024, 1024), 32, 128, 128)
    assert el == 524_288, el
    assert px == 3_145_728, px
    assert abs(ratio - 6.0) < 1e-3, ratio
    print(f"  FLUX.2  : [1,32,128,128] {px:,}/{el:,} = {ratio:.1f}x")

    # FLUX.2 config invariants (constants §9.6 / hardware-ground-truth §4)
    block_out_channels = [128, 256, 512, 512]  # his config; len-1 = 3 downsamples
    f = 2 ** (len(block_out_channels) - 1)
    assert f == 8, f
    assert 32 == 32  # latent_channels, checked against the live config in run_flux2
    print(f"  FLUX.2  : f = 2^{len(block_out_channels)-1} = {f}, latent_channels = 32")
    print("  self-check       : §9.6 compression ratios and f=8 exact  ✓")


def main():
    ap = argparse.ArgumentParser(description="VAE round-trip vs decode-the-prior, on your own VAEs.")
    ap.add_argument("--self-test", action="store_true",
                    help="check the frozen §9.6 compression arithmetic; no GPU, no model")
    ap.add_argument("--image", help="a real photo (jpg/png); center-cropped square")
    ap.add_argument("--sd15", default=None, help="SD 1.5 VAE .safetensors file or diffusers folder")
    ap.add_argument("--flux2", default=None, help="FLUX.2-dev checkpoint root (has a vae/ subfolder)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available). Use cpu to avoid GPU contention.")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--outdir", default="vae_ceiling_out", help="where the PNG tiles go")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.image:
        ap.error("the following arguments are required: --image "
                 "(or pass --self-test for the no-GPU arithmetic check)")

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("This script needs torch + diffusers. In ComfyUI's venv they're already there.")
    import torch

    if not os.path.exists(args.image):
        sys.exit(f"image not found: {args.image}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        dtype = torch.float32  # half on CPU is slow/unsupported for these ops
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    os.makedirs(_expand(args.outdir), exist_ok=True)
    outdir = _expand(args.outdir)
    print(f"device = {device}   dtype = {dtype}   outdir = {outdir}\n")

    sd15 = find_sd15_vae(args.sd15)
    flux2 = find_flux2_root(args.flux2)

    run_sd15(args.image, sd15, device, dtype, outdir)
    print()
    run_flux2(args.image, flux2, device, dtype, outdir)

    print("\n" + "-" * 70)
    print("Your VAE round-trips your photo (crisp). Your VAE's own prior gives static.")
    print("That gap — a good space with no good sampler — is the whole rest of the track.")
    print("-" * 70)


if __name__ == "__main__":
    main()
