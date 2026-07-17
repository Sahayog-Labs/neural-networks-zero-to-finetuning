#!/usr/bin/env python3
"""
ddpm_mnist.py -- RUNG 3, diffusion form: a ~1.5M-parameter U-Net that learns to
turn noise into MNIST digits. The trunk's CNN, made generative.

Course artifact for p.58 ("The Denoiser's Body I: the U-Net, in Depth") and
p.59 ("From U-Net to DiT/MMDiT"), diffusion manifest D4.

This is the promise page 58 makes literal: on p.53-57 you gave epsilon_theta its
JOB (predict the noise on a forward-corrupted sample); here it gets a BODY -- a
real convolutional encoder/decoder with skip connections, trained end to end, on
your Spark, in ~10 minutes. It ships the skip-severing flag page 58 advertises:
run --cut-skips and regenerate to reproduce the ablation the page demos in the
browser, but on a network that actually learned.

Why a U-Net and not a DiT (the p.59 hinge): at 1.5M parameters the convolutional
spatial hierarchy is an ASSET, not a cage. Below ~1B params / 256^2 the U-Net's
built-in spatial prior wins; DiT quality tracks GFLOPs and needs scale to beat it
(p.59, D-15). This script is the small-scale end of that rule, run by hand.

The frozen schedule is the same one every diffusion page uses (constants.md 9.6):
  linear beta_t: 1e-4 -> 0.02 over T=1000, running cumulative product abar_t,
  abar_1000 ~= 4e-5,  sqrt(abar_T) = 0.0063 (nonzero terminal SNR),
  eps-pred error amplification at t=999:  1/0.0063 = 159x.

--------------------------------------------------------------------------------
RUN IT
  python ddpm_mnist.py --self-test          # NO GPU, NO torch: math + arch check, <1 s
  python ddpm_mnist.py --epochs 20          # train on your Spark (~10-15 min), then sample
  python ddpm_mnist.py --epochs 20 --cut-skips   # the skip-severed ablation (p.58)
  python ddpm_mnist.py --sample-only --ckpt ddpm_mnist_out/model.pt   # sample from a checkpoint

SAFETY
  --self-test is CPU/numpy-only and touches nothing on disk or network.
  Training (default path) will:
    * DOWNLOAD MNIST (~11 MB) into ./ddpm_mnist_out/data the first time (torchvision).
    * WRITE a checkpoint (~6 MB, 1.5M params x 4 B fp32) and a samples PNG grid
      into ./ddpm_mnist_out/ .
    * allocate GPU memory (this model + batch is well under 1 GiB) and saturate the
      GPU for the training window -- it will CONTEND with ComfyUI if ComfyUI is up.
  It writes nothing outside ./ddpm_mnist_out/ and installs nothing.
"""

import argparse
import math
import sys

# torch / torchvision are imported LAZILY inside the functions that need them, so
# --self-test runs on a box that has only numpy (this Windows machine). See the
# builder contract: every script gets a dry-run path that runs WITHOUT a GPU.
import numpy as np

# --------------------------------------------------------------------------- #
# GiB/GB discipline (constants.md 0). This model is tiny, but we keep the habit.
# --------------------------------------------------------------------------- #
GiB = 1 << 30
GB = 10 ** 9

# --------------------------------------------------------------------------- #
# FROZEN architecture spec. base_channels=40 gives 1,495,521 params ~= 1.50M.
# The channel multipliers give feature-map channels [40, 80, 160] at the three
# resolutions [28, 14, 7]. GroupNorm uses 8 groups; 40/80/160 are all divisible
# by 8. temb_dim = 2 * base = 80. Changing any of these changes PARAMS_EXACT,
# which is asserted against the real nn.Module below -- page and script cannot
# drift (the same predict-then-measure discipline as utils/ledger.py).
# --------------------------------------------------------------------------- #
BASE_CH = 40
CH_MULT = (1, 2, 4)          # -> [40, 80, 160]
GROUPS = 8
TEMB_MUL = 2                 # temb_dim = TEMB_MUL * BASE_CH = 80
PARAMS_EXACT = 1_495_521     # asserted against sum(p.numel()) when torch is present
PARAMS_TARGET = 1_500_000    # the page's "~1.5M"

# --------------------------------------------------------------------------- #
# FROZEN DDPM schedule (constants.md 9.6). Kept as a pure-numpy function so the
# self-test can check it with no torch. The training/sampling code rebuilds the
# identical schedule in torch and asserts it matches this one.
# --------------------------------------------------------------------------- #
T_STEPS = 1000
BETA_MIN = 1e-4
BETA_MAX = 2e-2

FROZEN_SQRT_ABAR_T = 0.0063          # constants.md 9.6 [DER]
FROZEN_AMP = round(1.0 / FROZEN_SQRT_ABAR_T)   # 159 [DER]


def build_abar_np(T=T_STEPS, bmin=BETA_MIN, bmax=BETA_MAX):
    """abar[t] = prod_{s<=t}(1 - beta_s), abar[0] = 1. Linear beta. float64."""
    betas = np.linspace(bmin, bmax, T, dtype=np.float64)       # betas[0..T-1] -> steps 1..T
    abar = np.empty(T + 1, dtype=np.float64)
    abar[0] = 1.0
    abar[1:] = np.cumprod(1.0 - betas)
    return abar


# --------------------------------------------------------------------------- #
# Pure-python parameter counter. This mirrors the nn.Module built below, term
# for term, so the self-test can verify the ~1.5M figure with NO torch, and the
# real module asserts sum(numel) == this. If you edit the architecture, edit
# BOTH and the assertion will catch any mismatch.
# --------------------------------------------------------------------------- #
def _conv3(cin, cout):
    return 9 * cin * cout + cout            # 3x3 conv + bias


def _resblock(cin, cout, temb):
    p = 2 * cin                             # GroupNorm1 (weight, bias)
    p += _conv3(cin, cout)                  # conv1 3x3
    p += temb * cout + cout                 # time-embedding Linear -> cout
    p += 2 * cout                           # GroupNorm2
    p += _conv3(cout, cout)                 # conv2 3x3
    if cin != cout:
        p += cin * cout + cout              # 1x1 skip projection
    return p


def predict_param_count(base=BASE_CH):
    """Analytic param count of the U-Net below. Must equal sum(p.numel())."""
    C = base
    temb = TEMB_MUL * C
    p = 0
    # sinusoidal time embedding -> MLP: Linear(C, temb) + Linear(temb, temb)
    p += C * temb + temb
    p += temb * temb + temb
    # stem: Conv3x3(1 -> C)                                  res 28, ch C   [skip s0]
    p += _conv3(1, C)
    # encoder
    p += _resblock(C, C, temb)              # enc1                res 28, ch C   [skip s1]
    p += _conv3(C, C)                       # down1 (stride 2)    res 14, ch C
    p += _resblock(C, 2 * C, temb)          # enc2                res 14, ch 2C  [skip s2]
    p += _conv3(2 * C, 2 * C)               # down2 (stride 2)    res 7,  ch 2C
    # middle
    p += _resblock(2 * C, 4 * C, temb)      # mid1                res 7,  ch 4C
    p += _resblock(4 * C, 4 * C, temb)      # mid2                res 7,  ch 4C
    # decoder (upsample + concat skip)
    p += _conv3(4 * C, 2 * C)               # up2 conv            res 14, ch 2C
    p += _resblock(4 * C, 2 * C, temb)      # dec2 (concat s2)    res 14, ch 2C
    p += _conv3(2 * C, C)                   # up1 conv            res 28, ch C
    p += _resblock(2 * C, C, temb)          # dec1 (concat s1)    res 28, ch C
    p += _resblock(2 * C, C, temb)          # dec0 (concat s0)    res 28, ch C
    # head: GroupNorm + Conv3x3(C -> 1)
    p += 2 * C
    p += _conv3(C, 1)
    return p


# --------------------------------------------------------------------------- #
# The model. Defined inside a factory so `import torch` is lazy. The class is a
# faithful small DDPM U-Net: sinusoidal time embedding injected into every
# ResBlock via FiLM-style bias add (p.33 sinusoidal, p.34 residuals -- straight
# from the trunk). Three explicit skips (s0, s1, s2) so --cut-skips can zero the
# ones that jump the bottleneck, exactly the p.58 ablation.
# --------------------------------------------------------------------------- #
def build_unet(cut_skips=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    C = BASE_CH
    TEMB = TEMB_MUL * C

    def sinusoidal(t, dim):
        # t: (B,) integer timesteps -> (B, dim). Half sin, half cos. No params.
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        ang = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    class ResBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.norm1 = nn.GroupNorm(GROUPS, cin)
            self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
            self.temb = nn.Linear(TEMB, cout)
            self.norm2 = nn.GroupNorm(GROUPS, cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x, temb):
            h = self.conv1(F.silu(self.norm1(x)))
            h = h + self.temb(temb)[:, :, None, None]     # FiLM-style additive time bias
            h = self.conv2(F.silu(self.norm2(h)))
            return h + self.skip(x)

    class UNet(nn.Module):
        def __init__(self, cut_skips):
            super().__init__()
            self.cut_skips = cut_skips
            self.temb_mlp = nn.Sequential(nn.Linear(C, TEMB), nn.SiLU(), nn.Linear(TEMB, TEMB))
            self.stem = nn.Conv2d(1, C, 3, padding=1)
            self.enc1 = ResBlock(C, C)
            self.down1 = nn.Conv2d(C, C, 3, stride=2, padding=1)
            self.enc2 = ResBlock(C, 2 * C)
            self.down2 = nn.Conv2d(2 * C, 2 * C, 3, stride=2, padding=1)
            self.mid1 = ResBlock(2 * C, 4 * C)
            self.mid2 = ResBlock(4 * C, 4 * C)
            self.up2_conv = nn.Conv2d(4 * C, 2 * C, 3, padding=1)
            self.dec2 = ResBlock(4 * C, 2 * C)            # in = up(2C) concat s2(2C)
            self.up1_conv = nn.Conv2d(2 * C, C, 3, padding=1)
            self.dec1 = ResBlock(2 * C, C)                # in = up(C) concat s1(C)
            self.dec0 = ResBlock(2 * C, C)                # in = dec1(C) concat s0(C)
            self.head_norm = nn.GroupNorm(GROUPS, C)
            self.head = nn.Conv2d(C, 1, 3, padding=1)

        def _maybe_cut(self, skip):
            # The p.58 ablation: a severed skip carries NOTHING to the up-path.
            return torch.zeros_like(skip) if self.cut_skips else skip

        def forward(self, x, t):
            temb = self.temb_mlp(sinusoidal(t, C))
            s0 = self.stem(x)                              # 28, C
            s1 = self.enc1(s0, temb)                       # 28, C
            h = self.down1(s1)                             # 14, C
            s2 = self.enc2(h, temb)                        # 14, 2C
            h = self.down2(s2)                             # 7,  2C
            h = self.mid2(self.mid1(h, temb), temb)        # 7,  4C
            # up to 14
            h = self.up2_conv(F.interpolate(h, scale_factor=2, mode="nearest"))   # 14, 2C
            h = self.dec2(torch.cat([h, self._maybe_cut(s2)], dim=1), temb)       # 14, 2C
            # up to 28
            h = self.up1_conv(F.interpolate(h, scale_factor=2, mode="nearest"))   # 28, C
            h = self.dec1(torch.cat([h, self._maybe_cut(s1)], dim=1), temb)       # 28, C
            h = self.dec0(torch.cat([h, self._maybe_cut(s0)], dim=1), temb)       # 28, C
            return self.head(F.silu(self.head_norm(h)))

    return UNet(cut_skips)


# --------------------------------------------------------------------------- #
# --self-test: NO GPU, NO torch. Verifies the frozen schedule and the ~1.5M
# architecture with pure numpy + python arithmetic. This is the path the build
# machine (torch-less Windows) actually runs.
# --------------------------------------------------------------------------- #
def self_test():
    print("=" * 70)
    print("ddpm_mnist.py --self-test  (numpy/CPU only; no torch, no GPU, no disk)")
    print("=" * 70)

    # 1. The frozen schedule (constants.md 9.6).
    abar = build_abar_np()
    print("\n-- forward-noising schedule: linear beta 1e-4 -> 0.02, T=1000 --")
    print(f"   abar[0]     = {abar[0]:.6f}   (t=0: no noise yet)")
    print(f"   abar[1000]  = {abar[1000]:.3e}   (~= 4e-5, nonzero terminal SNR)  [DER, constants 9.6]")
    sqrt_abar_T = math.sqrt(abar[1000])
    print(f"   sqrt(abar_T)= {FROZEN_SQRT_ABAR_T}   [DER, constants 9.6]   (raw schedule: {sqrt_abar_T:.6f})")
    print(f"   eps-pred error amplification 1/sqrt(abar_T) = {FROZEN_AMP}x  [DER, constants 9.6]")
    assert abar[0] == 1.0, "abar[0] must be exactly 1"
    assert np.all(np.diff(abar) < 0), "abar must be strictly decreasing"
    assert abs(abar[1000] - 4e-5) < 1e-5, "abar_1000 must be ~4e-5 (constants 9.6)"
    assert abs(sqrt_abar_T - FROZEN_SQRT_ABAR_T) < 1e-4, "sqrt(abar_T) must round to frozen 0.0063"
    assert FROZEN_AMP == 159, "frozen eps-pred amplification must be 159x (constants 9.6)"

    # 2. The forward process preserves unit variance: x_t = sqrt(abar) x0 + sqrt(1-abar) eps
    #    Var(x_t) = abar*Var(x0) + (1-abar)*Var(eps) = 1 when both are N(0,1). Check empirically.
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal(200_000)
    eps = rng.standard_normal(200_000)
    for t in (1, 250, 500, 999):
        a = abar[t]
        xt = math.sqrt(a) * x0 + math.sqrt(1 - a) * eps
        v = xt.var()
        print(f"   t={t:>4}: Var(x_t) = {v:.4f}  (target 1.000 -- forward keeps unit variance)")
        assert abs(v - 1.0) < 0.02, f"forward variance off at t={t}"

    # 3. The ~1.5M architecture, counted analytically (matches the nn.Module).
    p = predict_param_count()
    print(f"\n-- U-Net parameter count (analytic; asserted == nn.Module when torch present) --")
    print(f"   base_channels = {BASE_CH}, channels [{BASE_CH}, {2*BASE_CH}, {4*BASE_CH}], temb_dim = {TEMB_MUL*BASE_CH}")
    print(f"   params = {p:,}  (~= {p/1e6:.2f} M)   target ~1.5M (page 58)")
    assert p == PARAMS_EXACT, f"param count {p:,} != frozen {PARAMS_EXACT:,} (edit predict + module together)"
    assert 1.40e6 <= p <= 1.60e6, "param count must be ~1.5M (page 58 promise)"

    # 4. GroupNorm divisibility sanity (a real bug-catcher: 8 must divide every channel count).
    for ch in (BASE_CH, 2 * BASE_CH, 4 * BASE_CH):
        assert ch % GROUPS == 0, f"GroupNorm({GROUPS}) cannot split {ch} channels"
    print(f"\n   GroupNorm({GROUPS}) divides all of {{{BASE_CH}, {2*BASE_CH}, {4*BASE_CH}}}  [OK]")

    print("\n[OK] schedule frozen (159x pole), forward keeps unit variance, arch = 1.50M params.")
    print("     On the Spark, run without --self-test to train and sample real digits.")


# --------------------------------------------------------------------------- #
# Version stamp -- so a support request starts with real versions (contract 7).
# --------------------------------------------------------------------------- #
def stamp():
    import torch
    line = f"torch {torch.__version__}"
    try:
        import torchvision
        line += f" | torchvision {torchvision.__version__}"
    except ImportError:
        line += " | torchvision MISSING (needed for MNIST download)"
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        line += f" | {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} | CUDA {torch.version.cuda}"
    else:
        line += " | CPU only (no CUDA)"
    print(line)


# --------------------------------------------------------------------------- #
# Training + sampling (the Spark path). Rebuilds the frozen schedule in torch and
# asserts it matches the numpy one, then trains eps-prediction and samples via the
# ancestral DDPM reverse process.
# --------------------------------------------------------------------------- #
def train_and_sample(args):
    import os
    import torch
    import torch.nn.functional as F

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print("ddpm_mnist.py -- U-Net DDPM on 28x28 MNIST  (RUNG 3, diffusion form)")
    print("=" * 70)
    stamp()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("\n!! No CUDA. This will train, but slowly. On the Spark you get ~10-15 min;")
        print("   on CPU expect far longer. Use --self-test for the no-GPU verification path.")
    else:
        free, tot = torch.cuda.mem_get_info()
        print(f"\n  GPU memory: {free/GiB:.2f} GiB free of {tot/GiB:.2f} GiB total")
        if free < 0.5 * tot:
            print("  !! Over half the GPU is already in use -- ComfyUI up? Training will contend.")

    # seeds (contract 7): reproducible
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- frozen schedule, in torch, cross-checked against numpy --
    abar_np = build_abar_np()
    betas = torch.linspace(BETA_MIN, BETA_MAX, T_STEPS, dtype=torch.float32, device=dev)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)                     # abar[i] for step i+1 (1..T)
    assert abs(abar[-1].item() - abar_np[T_STEPS]) < 1e-6, "torch schedule must match numpy"
    sqrt_abar = torch.sqrt(abar)
    sqrt_one_minus_abar = torch.sqrt(1.0 - abar)
    print(f"  schedule: sqrt(abar_T) = {sqrt_abar[-1].item():.6f} (frozen {FROZEN_SQRT_ABAR_T}), "
          f"amplification {FROZEN_AMP}x  [constants 9.6]")

    # -- model --
    model = build_unet(cut_skips=args.cut_skips).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\n  U-Net params: {nparams:,}  (~= {nparams/1e6:.2f} M, target ~1.5M)")
    assert nparams == PARAMS_EXACT, \
        f"real module has {nparams:,} params, analytic predict says {PARAMS_EXACT:,} -- they must agree"
    if args.cut_skips:
        print("  ** --cut-skips ACTIVE: all three skip connections are zeroed (p.58 ablation).")
        print("     Expect blurry, low-frequency digits -- the bottleneck cannot pass detail.")

    ckpt_mb = nparams * 4 / (1 << 20)
    print(f"  checkpoint on save: ~{ckpt_mb:.1f} MB (fp32). Output dir: {os.path.abspath(outdir)}")

    if args.sample_only:
        if not args.ckpt or not os.path.exists(args.ckpt):
            print("\n!! --sample-only needs --ckpt <path to model.pt>. Aborting.")
            sys.exit(1)
        model.load_state_dict(torch.load(args.ckpt, map_location=dev))
        print(f"\n  loaded {args.ckpt}; sampling {args.n_samples} digits ...")
        _sample_grid(model, abar, betas, alphas, dev, args, outdir)
        return

    # -- data (downloads MNIST ~11 MB on first run) --
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),              # [0,1] -> [-1,1], the diffusion range
    ])
    print("\n  loading MNIST (downloads ~11 MB into the output dir on first run) ...")
    ds = datasets.MNIST(os.path.join(outdir, "data"), train=True, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    print(f"  AdamW lr={args.lr} (diffusion beta=(0.9,0.999), constants 9.4); "
          f"epochs={args.epochs}, batch={args.batch}\n")

    import time
    model.train()
    for ep in range(args.epochs):
        t0 = time.time()
        running = 0.0
        for i, (x, _) in enumerate(loader):
            x = x.to(dev)                                  # (B,1,28,28) in [-1,1]
            b = x.size(0)
            t = torch.randint(0, T_STEPS, (b,), device=dev)   # 0..T-1 indexes abar[t]
            eps = torch.randn_like(x)
            xt = sqrt_abar[t][:, None, None, None] * x + sqrt_one_minus_abar[t][:, None, None, None] * eps
            eps_pred = model(xt, t + 1)                     # feed 1..T to the time embedding
            loss = F.mse_loss(eps_pred, eps)               # epsilon-prediction (p.55 target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            if i % 100 == 0:
                print(f"  epoch {ep+1:>2}/{args.epochs}  step {i:>4}  loss {loss.item():.4f}")
        dt = time.time() - t0
        print(f"  -- epoch {ep+1} done: mean loss {running/len(loader):.4f}  ({dt:.1f}s)")

    ckpt_path = os.path.join(outdir, "model_cutskips.pt" if args.cut_skips else "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  saved checkpoint: {ckpt_path}  (~{ckpt_mb:.1f} MB)")

    _sample_grid(model, abar, betas, alphas, dev, args, outdir)


def _sample_grid(model, abar, betas, alphas, dev, args, outdir):
    """Ancestral DDPM reverse process, then save an NxN PNG grid of digits."""
    import os
    import torch

    model.eval()
    n = args.n_samples
    sqrt_recip_alpha = torch.rsqrt(alphas)
    one_minus_abar = 1.0 - abar
    print(f"\n  sampling {n} digits via the ancestral reverse process ({T_STEPS} steps) ...")
    with torch.no_grad():
        x = torch.randn(n, 1, 28, 28, device=dev)          # start from pure noise, t=T
        for i in reversed(range(T_STEPS)):                 # i = T-1 .. 0 (indexes step i+1)
            t = torch.full((n,), i + 1, device=dev, dtype=torch.long)
            eps_pred = model(x, t)
            beta_i = betas[i]
            coef = beta_i / torch.sqrt(one_minus_abar[i])
            mean = sqrt_recip_alpha[i] * (x - coef * eps_pred)
            if i > 0:
                x = mean + torch.sqrt(beta_i) * torch.randn_like(x)   # add noise except last step
            else:
                x = mean
    imgs = (x.clamp(-1, 1) + 1) / 2                          # [-1,1] -> [0,1]

    # proxy quality readout (recognizability is a HUMAN check -- open the PNG).
    fg = (imgs > 0.5).float().mean().item()
    print(f"  foreground (pixels > 0.5) fraction: {fg*100:.1f}%  "
          f"(trained MNIST digits sit ~10-20%; near 0% or ~50% = didn't learn / noise)")

    grid = _make_grid_np(imgs.cpu().numpy())
    png = os.path.join(outdir, "samples_cutskips.png" if args.cut_skips else "samples.png")
    _save_png(grid, png)
    print(f"  saved samples: {png}")
    print("\n  OPEN THE PNG. Do they read as digits? That is RUNG 3, diffusion form: a")
    print("  1.5M-param conv net that learned to walk noise back into MNIST. With --cut-skips")
    print("  they should be visibly blurrier -- the p.58 ablation, on a net that actually trained.")


def _make_grid_np(imgs):
    """imgs: (N,1,28,28) in [0,1] -> single (H,W) grid, near-square layout."""
    n = imgs.shape[0]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pad = 2
    cell = 28 + pad
    canvas = np.zeros((rows * cell + pad, cols * cell + pad), dtype=np.float32)
    for k in range(n):
        r, c = divmod(k, cols)
        y, xx = r * cell + pad, c * cell + pad
        canvas[y:y + 28, xx:xx + 28] = imgs[k, 0]
    return canvas


def _save_png(gray, path):
    """Write a grayscale (H,W) float [0,1] array to PNG. Uses PIL if present, else
    a tiny hand-rolled PNG encoder so the script has no hard Pillow dependency."""
    arr = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(arr, mode="L").save(path)
        return
    except ImportError:
        pass
    # minimal zlib+PNG encoder (grayscale, 8-bit) -- no external deps.
    import struct
    import zlib
    h, w = arr.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)                                      # filter type 0 (none)
        raw.extend(arr[y].tobytes())
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)    # 8-bit grayscale
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def main():
    ap = argparse.ArgumentParser(description="DDPM U-Net on MNIST (course RUNG 3, diffusion form)")
    ap.add_argument("--self-test", action="store_true",
                    help="numpy/CPU-only: verify schedule + ~1.5M arch, no torch/GPU/disk")
    ap.add_argument("--epochs", type=int, default=20, help="training epochs (~10-15 min on the Spark)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--cut-skips", action="store_true",
                    help="sever all U-Net skip connections (the p.58 ablation)")
    ap.add_argument("--sample-only", action="store_true", help="skip training; sample from --ckpt")
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint for --sample-only")
    ap.add_argument("--n-samples", type=int, default=64, help="digits to sample into the grid")
    ap.add_argument("--out", type=str, default="ddpm_mnist_out", help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not importable here. On the Spark, use a FRESH course venv (NEVER ComfyUI's):")
        print("  ~/course/.venv/bin/python ddpm_mnist.py --epochs 20")
        print("For a no-GPU sanity check on any box (numpy only):")
        print("  python ddpm_mnist.py --self-test")
        sys.exit(1)

    train_and_sample(args)


if __name__ == "__main__":
    main()
