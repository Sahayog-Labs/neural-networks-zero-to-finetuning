#!/usr/bin/env python3
"""
flow_matching_2d.py -- DDPM is the arc, rectified flow is the chord. One file.

Course artifact for p.57 ("Flow Matching & Rectified Flow: your FLUX scheduler,
decoded"), manifest D7. The page's whole claim is "no third theory" -- flow
matching is diffusion_2d.py with THREE lines changed. This file IS that claim, in
code: it carries BOTH routes side by side on the same 8-Gaussians toy, so the
`diff` between them is literally the path, the target, and the sampler -- nothing
else. (diffusion_2d.py, p.54/manifest D3, is the ancestor; this file is the
minimal patch and reproduces the DDPM branch too, so it stands alone.)

What it delivers, matching the p.57 "Try it" box:
  1. The three-line reconciliation, PRINTED as a diff:
        path    xt = sqrt(abar_t) x0 + sqrt(1-abar_t) eps   ->   xt = (1-t) x0 + t eps
        target  target = eps                                 ->   target = eps - x0
        sampler DDIM curved step (arc)                        ->   Euler straight ODE step (chord)
  2. The geometry that names them: DDPM (a_t,b_t) lies on the CIRCLE a^2+b^2=1;
     rectified flow lies on the CHORD a_t+b_t=1, with a_1 = 0 EXACTLY -- so the
     terminal-SNR leak of p.54 (sqrt(abar_T) = 0.0063, a 159x eps->x0 blowup) is
     gone STRUCTURALLY, not patched. Both asserted against constants.md 9.6.
  3. A sweep of the sampler step count N in {1,2,4,8,16,32,64} for BOTH routes,
     with a sliced-Wasserstein-2 distance to the true distribution at each N --
     the chord's advantage at low N (and why base FM still is not one step: the
     network learns the CURVED marginal average of crossing straight lines).
  4. Your OWN scheduler_config.json, read off disk and matched against the table:
     FlowMatchEulerDiscreteScheduler, shift 3.0, use_dynamic_shifting, and the
     VESTIGIAL num_train_timesteps = 1000 (an indexing relic; FM is continuous in
     t in [0,1]). Plus the logit-normal shift warp and its frozen 68% band.

The t convention here is course-wide (notation.md 6): t = 0 is DATA, t = 1 is
NOISE, sampling runs t downward. DDPM ALSO has data at t=0 -- flow matching is
NOT "opposite to DDPM"; that retired brief-diffusion 6.2 claim is corrected here.

Usage
-----
    python flow_matching_2d.py --self-test        # pure-stdlib arithmetic, no torch/GPU
    python flow_matching_2d.py                    # full demo: train both, sweep N (~40 s CPU)
    python flow_matching_2d.py --quick            # fewer train steps / projections (~15 s)
    python flow_matching_2d.py --config path/to/scheduler_config.json   # match your FLUX
    python flow_matching_2d.py --plot out.png     # also write the W2-vs-N figure

SAFETY: CPU only, no GPU, no network. The full demo trains two tiny MLPs in
torch and (with --plot) writes ONE png where you name it; otherwise it writes
nothing and installs nothing. --self-test needs only the standard library.
"""

import argparse
import math
import sys

# --------------------------------------------------------------------------- #
# FROZEN CONSTANTS -- every one is asserted, none is invented. constants.md 9.6.
# --------------------------------------------------------------------------- #
T_DDPM = 1000                 # DDPM discrete steps
BETA_MIN, BETA_MAX = 1e-4, 2e-2
SQRT_ABAR_T = 0.0063          # sqrt(abar_1000), nonzero terminal SNR [DER, constants 9.6]
EPS_AMP = 159                 # 1/0.0063 rounded -> eps->x0 error blowup at t=999 [DER]
ABAR_T = 4e-5                 # abar_1000 ~= 4e-5 [DER]

# Your FLUX.2-dev scheduler (hardware-ground-truth.md 4, read off scheduler_config.json).
FLUX_SCHEDULER = "FlowMatchEulerDiscreteScheduler"
FLUX_SHIFT = 3.0
FLUX_NUM_TRAIN_TIMESTEPS = 1000     # VESTIGIAL -- FM is continuous; this is a DDPM-era relic
FLUX_USE_DYNAMIC_SHIFTING = True

# logit-normal timestep sampling: t = logistic(z), z ~ N(0,1). The band t in
# (0.27,0.73) is the image of z in (-1,1); P(-1<z<1) = 2*Phi(1)-1 = 0.6827 -> 68%.
BAND_LO, BAND_HI = 0.27, 0.73
BAND_FRAC = 0.6827            # the frozen worked number [DER, page 57]

SEED = 57057


# --------------------------------------------------------------------------- #
# PURE-STDLIB MATH -- the schedule and the two routes, no numpy, no torch.
# Used by --self-test so it runs on any Python (this Windows box has no torch).
# --------------------------------------------------------------------------- #

def build_abar(T=T_DDPM, bmin=BETA_MIN, bmax=BETA_MAX):
    """abar[t] = prod_{s<=t} (1 - beta_s), linear beta schedule. abar[0] = 1.

    IDENTICAL running cumulative product to prediction_targets.py / diffusion_2d.py
    -- NOT a Math.pow shortcut, which would give the wrong terminal value.
    """
    abar = [1.0] * (T + 1)
    ab = 1.0
    for t in range(1, T + 1):
        beta = bmin + (bmax - bmin) * (t - 1) / (T - 1)
        ab *= (1.0 - beta)
        abar[t] = ab
    return abar


def ddpm_ab(abar_t):
    """DDPM / VP-SDE coefficients: (a_t, b_t) = (sqrt(abar_t), sqrt(1-abar_t)).
    The ARC: they live on the circle a^2 + b^2 = 1."""
    return math.sqrt(abar_t), math.sqrt(1.0 - abar_t)


def fm_ab(t):
    """Rectified-flow coefficients: (a_t, b_t) = (1 - t, t).
    The CHORD: they live on the line a + b = 1. Velocity u = eps - x0 is constant."""
    return 1.0 - t, t


def logistic(z):
    return 1.0 / (1.0 + math.exp(-z))


def shift_warp(t, shift):
    """The resolution warp FLUX applies to the sampled timestep (page-57 JS, verbatim):
        t' = shift * t / (1 + (shift - 1) * t).
    shift = 1 is identity; shift > 1 slides mass toward the noisy (t->1) end."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# THE RECONCILIATION -- the three lines that turn DDPM into flow matching.
# --------------------------------------------------------------------------- #

def print_the_diff():
    print("=" * 72)
    print("THE RECONCILIATION -- flow matching is diffusion_2d.py, three lines changed")
    print("=" * 72)
    print("  There is no third theory. `diff diffusion_2d.py flow_matching_2d.py` is:")
    print()
    print("  (1) THE PATH        a_t x0 + b_t eps, with (a_t, b_t) =")
    print("      DDPM  (arc)     - xt = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps   # circle a^2+b^2=1")
    print("      FM    (chord)   + xt = (1 - t)*x0 + t*eps                     # line   a + b  =1")
    print()
    print("  (2) THE TARGET      what the network regresses to")
    print("      DDPM            - target = eps                               # the noise")
    print("      FM              + target = eps - x0                          # the CONSTANT velocity")
    print()
    print("  (3) THE SAMPLER     one step of the reverse process")
    print("      DDPM  (arc)     - DDIM: x0_hat then re-noise to the next t   # curved route")
    print("      FM    (chord)   + Euler: x <- x - v * dt, integrate t: 1->0  # straight ODE")
    print()
    print("  Everything else -- the net, the data, the loss -- is byte-identical.")
    print("  That is the whole 'no third theory' claim, as a three-line patch.")
    print()


# --------------------------------------------------------------------------- #
# SELF-TEST -- no torch, no numpy. Runs on this Windows box and in CI.
# --------------------------------------------------------------------------- #

def self_test():
    print("SELF-TEST -- asserting the frozen arithmetic (stdlib only, no torch)")
    print("-" * 72)

    abar = build_abar()

    # --- the DDPM terminal-SNR facts (constants 9.6) ---
    raw_sqrt_abar_T = math.sqrt(abar[T_DDPM])
    assert abs(abar[T_DDPM] - ABAR_T) < 1e-5, \
        f"abar_1000 must be ~4e-5, got {abar[T_DDPM]:.3e}"
    assert abs(raw_sqrt_abar_T - SQRT_ABAR_T) < 1e-4, \
        f"raw sqrt(abar_T) must round to the frozen 0.0063, got {raw_sqrt_abar_T:.6f}"
    assert round(1.0 / SQRT_ABAR_T) == EPS_AMP, "eps->x0 amplification must be 159x"
    print(f"  DDPM abar_1000 = {abar[T_DDPM]:.3e} (~4e-5)   sqrt(abar_T) = {SQRT_ABAR_T}"
          f"  -> eps->x0 blowup 1/{SQRT_ABAR_T} = {EPS_AMP}x  [DER, constants 9.6]  [OK]")

    # --- DDPM lives on the CIRCLE at every t; a_T is LEAKY (nonzero) ---
    max_circle_err = 0.0
    for t in range(T_DDPM + 1):
        a, b = ddpm_ab(abar[t])
        max_circle_err = max(max_circle_err, abs(a * a + b * b - 1.0))
    assert max_circle_err < 1e-9, f"DDPM must satisfy a^2+b^2=1, max err {max_circle_err:.2e}"
    a_T, _ = ddpm_ab(abar[T_DDPM])
    assert a_T > 0.0, "DDPM terminal a_T is LEAKY (nonzero) -- that is the bug"
    print(f"  DDPM  (a_t,b_t) on the CIRCLE a^2+b^2=1  (max err {max_circle_err:.1e})"
          f"   a_T = {a_T:.4f} > 0  (leaky)  [OK]")

    # --- FM lives on the CHORD a+b=1; a_1 = 0 EXACTLY; velocity is constant ---
    max_chord_err = 0.0
    for i in range(1001):
        t = i / 1000.0
        a, b = fm_ab(t)
        max_chord_err = max(max_chord_err, abs(a + b - 1.0))
    assert max_chord_err < 1e-12, f"FM must satisfy a+b=1, max err {max_chord_err:.2e}"
    a1, b1 = fm_ab(1.0)
    a0, b0 = fm_ab(0.0)
    assert a1 == 0.0 and b1 == 1.0, "FM at t=1 must be pure noise (a_1=0 exactly)"
    assert a0 == 1.0 and b0 == 0.0, "FM at t=0 must be pure data (b_0=0)"
    print(f"  FM    (a_t,b_t) on the CHORD  a+b=1     (max err {max_chord_err:.1e})"
          f"   a_1 = {a1:.1f} EXACTLY (no leak)  [OK]")

    # FM velocity u_t = d/dt[(1-t)x0 + t eps] = eps - x0, INDEPENDENT of t.
    # Check on scalars at several t: the finite-difference derivative equals eps - x0.
    x0, eps = 0.7, -1.3
    dt = 1e-6
    for t in (0.0, 0.25, 0.5, 0.75, 1.0 - dt):
        xt = (1 - t) * x0 + t * eps
        xt2 = (1 - (t + dt)) * x0 + (t + dt) * eps
        u_fd = (xt2 - xt) / dt
        assert abs(u_fd - (eps - x0)) < 1e-4, "FM velocity must be the constant eps - x0"
    print(f"  FM    velocity u_t = eps - x0 = {eps - x0:+.2f}  (constant in t, "
          f"finite-diff verified)  [OK]")

    # --- t convention: BOTH have data at t=0 (notation 6 correction) ---
    a_ddpm_0, b_ddpm_0 = ddpm_ab(abar[0])
    assert a_ddpm_0 == 1.0 and b_ddpm_0 == 0.0, "DDPM at t=0 is pure data too"
    assert a0 == 1.0 and b0 == 0.0
    print(f"  t=0 is DATA for BOTH: DDPM (a,b)=({a_ddpm_0:.0f},{b_ddpm_0:.0f}), "
          f"FM (a,b)=({a0:.0f},{b0:.0f})  -- FM is NOT 'opposite to DDPM'  [OK]")

    # --- the logit-normal shift warp and its 68% band (page 57) ---
    # The band t in (0.27,0.73) IS the image of z in (-1,1) under t = logistic(z),
    # so its mass is exactly P(-1<z<1) = 2*Phi(1) - 1 = 0.6827. The (0.27,0.73)
    # endpoints are logistic(-+1) rounded to 2 d.p. (0.2689 -> 0.27, 0.7311 -> 0.73).
    band = _norm_cdf(1.0) - _norm_cdf(-1.0)
    assert abs(band - BAND_FRAC) < 1e-3, f"logit-normal band must be ~68%, got {band:.4f}"
    assert round(logistic(-1.0), 2) == BAND_LO and round(logistic(1.0), 2) == BAND_HI, \
        "band endpoints must be logistic(-+1) rounded to 0.27/0.73"
    assert abs(shift_warp(0.5, 1.0) - 0.5) < 1e-12, "shift=1 must be identity"
    # shift>1 pushes every interior t UP (toward the noisy t=1 end).
    t_mid = shift_warp(0.5, FLUX_SHIFT)
    assert t_mid > 0.5, "shift=3 must slide mass toward the noisy end"
    print(f"  logit-normal band t in ({BAND_LO},{BAND_HI}) at shift=1 = {100*band:.1f}%"
          f"  (frozen 68%)  [OK]")
    print(f"  shift warp: t=0.5 at shift {FLUX_SHIFT} -> {t_mid:.3f}  (mass -> noisy end)  [OK]")

    # --- the FLUX config values the demo will match ---
    assert FLUX_NUM_TRAIN_TIMESTEPS == 1000, "vestigial num_train_timesteps is 1000"
    print(f"  FLUX config to match: {FLUX_SCHEDULER}, shift {FLUX_SHIFT}, "
          f"num_train_timesteps {FLUX_NUM_TRAIN_TIMESTEPS} (VESTIGIAL)  [OK]")

    print("-" * 72)
    print("SELF-TEST PASSED -- circle vs chord, a_1=0 exact, constant velocity,")
    print("                    159x terminal blowup, 68% band, shift warp: all frozen.")


# --------------------------------------------------------------------------- #
# THE FLUX CONFIG -- read your own scheduler_config.json and match the table.
# --------------------------------------------------------------------------- #

def match_flux_config(path):
    import json
    import os

    print("=" * 72)
    print("YOUR SCHEDULER -- cat scheduler_config.json, match it against the table")
    print("=" * 72)

    if not path or not os.path.exists(path):
        print("  No --config given (or file not found). On your box:")
        print("    cat ~/ComfyUI/models/.../FLUX.2-dev/scheduler/scheduler_config.json")
        print("  Then re-run with --config <that path>. Expected (hardware-ground-truth.md 4):")
        print(f"    _class_name           = {FLUX_SCHEDULER}")
        print(f"    shift                 = {FLUX_SHIFT}")
        print(f"    use_dynamic_shifting  = {FLUX_USE_DYNAMIC_SHIFTING}")
        print(f"    num_train_timesteps   = {FLUX_NUM_TRAIN_TIMESTEPS}   <-- VESTIGIAL")
        print("  num_train_timesteps is a DDPM-era indexing relic: flow matching is")
        print("  continuous in t in [0,1]. It is a real field in the config, not used")
        print("  as 1000 discrete steps. [VP: real default in FlowMatchEulerDiscreteScheduler]")
        print()
        return

    with open(path) as f:
        cfg = json.load(f)
    cls = cfg.get("_class_name", "?")
    shift = cfg.get("shift", None)
    dyn = cfg.get("use_dynamic_shifting", None)
    nts = cfg.get("num_train_timesteps", None)

    print(f"  _class_name           = {cls}")
    print(f"  shift                 = {shift}")
    print(f"  use_dynamic_shifting  = {dyn}")
    print(f"  num_train_timesteps   = {nts}   <-- VESTIGIAL (FM is continuous in [0,1])")
    print()
    if cls == FLUX_SCHEDULER:
        print(f"  [OK] scheduler is {FLUX_SCHEDULER} -- the chord sampler, plain Euler.")
    else:
        print(f"  [note] scheduler is {cls}, not the expected {FLUX_SCHEDULER}.")
    if shift is not None and abs(float(shift) - FLUX_SHIFT) < 1e-6:
        print(f"  [OK] shift = {FLUX_SHIFT} matches your FLUX.2-dev (the resolution warp).")
    elif shift is not None:
        print(f"  [note] shift = {shift} (course-frozen FLUX.2-dev value is {FLUX_SHIFT}).")
    if nts == FLUX_NUM_TRAIN_TIMESTEPS:
        print(f"  [OK] num_train_timesteps = {FLUX_NUM_TRAIN_TIMESTEPS}, and it is VESTIGIAL --")
        print("       do NOT read it as 1000 real steps. plain `euler` at ~28 steps is all")
        print("       your FLUX needs, because the chord has no curvature to miss.")
    print()


# --------------------------------------------------------------------------- #
# THE DEMO -- train both routes on 8-Gaussians, sweep N, sliced-W2 vs truth.
# torch/numpy imported HERE so --self-test never touches them.
# --------------------------------------------------------------------------- #

def sample_8gaussians(n, rng, radius=2.0, std=0.08):
    """n points from a ring of 8 isotropic Gaussians (the classic toy)."""
    import numpy as np
    angles = np.arange(8) * (2 * np.pi / 8)
    centers = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (8,2)
    idx = rng.integers(0, 8, size=n)
    return centers[idx] + std * rng.standard_normal((n, 2))


def sliced_w2(X, Y, n_proj=128, rng=None):
    """Sliced-Wasserstein-2 distance between two equal-size 2-D point clouds.
    W2 in 1-D is the L2 distance of sorted samples; average over random directions.
    Pure numpy -- no POT/scipy dependency."""
    import numpy as np
    if rng is None:
        rng = np.random.default_rng(0)
    d = X.shape[1]
    acc = 0.0
    for _ in range(n_proj):
        theta = rng.standard_normal(d)
        theta /= (np.linalg.norm(theta) + 1e-12)
        xp = np.sort(X @ theta)
        yp = np.sort(Y @ theta)
        acc += float(np.mean((xp - yp) ** 2))
    return math.sqrt(acc / n_proj)


def _build_net(torch):
    """A small time-conditioned MLP: (x, y, t) -> (2,). Rock-stable torch API only."""
    import torch.nn as nn

    class Field(nn.Module):
        def __init__(self, hidden=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 2),
            )

        def forward(self, x, t):
            # x (B,2), t (B,1) -> (B,2)
            return self.net(torch.cat([x, t], dim=1))

    return Field()


def train_field(torch, kind, data, abar, steps, batch, lr, seed):
    """Train one field. kind='ddpm' regresses eps; kind='fm' regresses eps - x0.

    --- THE THREE LINES, live ---   (see print_the_diff)
      PATH:   ddpm  xt = sqrt(abar_t) x0 + sqrt(1-abar_t) eps
              fm    xt = (1 - t) x0 + t eps
      TARGET: ddpm  target = eps
              fm    target = eps - x0
    """
    import numpy as np
    torch.manual_seed(seed)
    net = _build_net(torch)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    data_t = torch.tensor(data, dtype=torch.float32)
    n = data_t.shape[0]
    abar_t = torch.tensor(abar, dtype=torch.float32)  # (T+1,)

    for _ in range(steps):
        idx = torch.randint(0, n, (batch,))
        x0 = data_t[idx]                                  # (B,2)
        eps = torch.randn(batch, 2)
        if kind == "ddpm":
            ti = torch.randint(1, T_DDPM + 1, (batch, 1))          # discrete t
            ab = abar_t[ti.squeeze(1)].unsqueeze(1)                # (B,1)
            a, b = torch.sqrt(ab), torch.sqrt(1.0 - ab)
            xt = a * x0 + b * eps                                  # PATH (arc)
            target = eps                                          # TARGET
            tcond = ti.float() / T_DDPM                            # net sees t in [0,1]
        else:  # fm
            t = torch.rand(batch, 1)                               # continuous t in [0,1)
            xt = (1.0 - t) * x0 + t * eps                          # PATH (chord)
            target = eps - x0                                     # TARGET (constant velocity)
            tcond = t
        pred = net(xt, tcond)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def sample_ddpm(torch, net, abar, n_steps, n_samples, seed):
    """DDIM deterministic (eta=0) sampler over n_steps, t from T down to 0. Curved."""
    torch.manual_seed(seed)
    abar_t = torch.tensor(abar, dtype=torch.float32)
    x = torch.randn(n_samples, 2)                              # noise at t=T
    # a decreasing schedule of integer timesteps, e.g. [1000, ..., 1]
    ts = torch.linspace(T_DDPM, 1, n_steps).round().long().clamp(1, T_DDPM)
    with torch.no_grad():
        for i in range(n_steps):
            ti = ts[i]
            ab_t = abar_t[ti]
            a_t, b_t = torch.sqrt(ab_t), torch.sqrt(1.0 - ab_t)
            tcond = torch.full((n_samples, 1), float(ti) / T_DDPM)
            eps = net(x, tcond)
            x0_hat = (x - b_t * eps) / a_t                    # eps->x0 (the pole path)
            if i == n_steps - 1:
                x = x0_hat                                    # land on data
            else:
                ab_s = abar_t[ts[i + 1]]
                a_s, b_s = torch.sqrt(ab_s), torch.sqrt(1.0 - ab_s)
                x = a_s * x0_hat + b_s * eps                  # re-noise to next t (arc)
    return x.numpy()


def sample_fm(torch, net, n_steps, n_samples, seed):
    """Euler ODE on the velocity field, t from 1 (noise) down to 0 (data). Straight step."""
    torch.manual_seed(seed)
    x = torch.randn(n_samples, 2)                             # noise at t=1
    dt = 1.0 / n_steps
    with torch.no_grad():
        for i in range(n_steps):
            t = 1.0 - i * dt
            tcond = torch.full((n_samples, 1), t)
            v = net(x, tcond)                                # dx/dt = v = eps - x0
            x = x - v * dt                                   # SAMPLER: integrate t downward
    return x.numpy()


def run_demo(args):
    try:
        import numpy as np
        import torch
    except ImportError as e:
        print(f"  Need numpy + torch for the demo ({e}). On the Spark, use a torch venv:")
        print("    ~/course/.venv/bin/python flow_matching_2d.py")
        print("    (or ~/ComfyUI/.venv/bin/python for a read-only run -- CPU is fine here)")
        print("  Locally, --self-test needs neither and verifies all the arithmetic.")
        sys.exit(1)

    print(f"  torch {torch.__version__} | numpy {np.__version__} | seed {SEED} | CPU only")
    print()
    print_the_diff()

    steps = 800 if args.quick else 2500
    n_proj = 48 if args.quick else 128
    n_eval = 2000 if args.quick else 4000
    batch = 256
    lr = 2e-3

    rng = np.random.default_rng(SEED)
    abar = build_abar()
    data = sample_8gaussians(4000, rng)
    truth = sample_8gaussians(n_eval, np.random.default_rng(SEED + 1))

    print("=" * 72)
    print(f"TRAINING both fields on 8-Gaussians (steps={steps}, CPU) ...")
    print("=" * 72)
    print("  (same net, same data, same MSE loss -- only PATH + TARGET differ)")
    net_ddpm = train_field(torch, "ddpm", data, abar, steps, batch, lr, SEED)
    net_fm = train_field(torch, "fm", data, abar, steps, batch, lr, SEED)
    print("  trained.\n")

    Ns = [1, 2, 4, 8, 16, 32, 64]
    print("=" * 72)
    print("SWEEP -- sampler steps N vs sliced-Wasserstein-2 distance to the truth")
    print("=" * 72)
    print(f"  {'N':>4} | {'DDPM (arc, DDIM)':>18} | {'FM (chord, Euler)':>18}")
    print(f"  {'-'*4}-+-{'-'*18}-+-{'-'*18}")
    w2_rng = np.random.default_rng(SEED + 2)
    ddpm_w2, fm_w2 = [], []
    for N in Ns:
        xd = sample_ddpm(torch, net_ddpm, abar, N, n_eval, SEED + 10 + N)
        xf = sample_fm(torch, net_fm, N, n_eval, SEED + 10 + N)
        wd = sliced_w2(xd, truth, n_proj, w2_rng)
        wf = sliced_w2(xf, truth, n_proj, w2_rng)
        ddpm_w2.append(wd)
        fm_w2.append(wf)
        print(f"  {N:>4} | {wd:>18.4f} | {wf:>18.4f}")
    print()
    print("  Read the top rows: at N=1-2 the CHORD (FM) is already close while the ARC")
    print("  (DDPM/DDIM) is not -- a straight route has no curvature to miss in one step.")
    print("  But FM is NOT one-step-perfect either: the network learns the conditional")
    print("  average E[eps - x0 | x_t] of MANY crossing straight lines, which is CURVED.")
    print("  That curvature is why base FM still wants a handful of Euler steps.")
    print()

    match_flux_config(args.config)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(Ns, ddpm_w2, "o-", label="DDPM (arc, DDIM)")
            plt.plot(Ns, fm_w2, "s-", label="FM (chord, Euler)")
            plt.xscale("log", base=2)
            plt.xlabel("sampler steps N")
            plt.ylabel("sliced-W2 to truth")
            plt.title("Arc vs chord: distance to the 8-Gaussians vs step budget")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot, dpi=110)
            print(f"  wrote figure -> {args.plot}")
        except ImportError:
            print("  (matplotlib not available; skipping --plot. The table above is the result.)")


def main():
    ap = argparse.ArgumentParser(
        description="DDPM vs rectified flow on 8-Gaussians -- the arc and the chord, one file.")
    ap.add_argument("--self-test", action="store_true",
                    help="frozen arithmetic assertions only (stdlib, no torch/GPU)")
    ap.add_argument("--quick", action="store_true",
                    help="fewer train steps / projections (~15 s CPU)")
    ap.add_argument("--config", default=None,
                    help="path to your FLUX scheduler_config.json to match against the table")
    ap.add_argument("--plot", default=None,
                    help="also write the W2-vs-N figure to this png path")
    args = ap.parse_args()

    print("#" * 72)
    print("# flow_matching_2d.py -- p.57 (manifest D7): the arc, the chord, no third theory")
    print("#" * 72)
    print()

    if args.self_test:
        print_the_diff()
        self_test()
        return

    run_demo(args)


if __name__ == "__main__":
    main()
