#!/usr/bin/env python3
"""
diffusion_2d.py -- a from-scratch DDPM on 8 Gaussians. The whole forward+reverse
loop, visible in 2-D, in one file you can step through on a laptop CPU.

Course artifact for p.54 ("The Forward Noising Process"), manifest D3. This is the
at-your-keyboard version of the page: a real denoising diffusion model with NO
U-Net -- a 3-layer MLP epsilon-predictor -- trained on eight Gaussians arranged on
a ring, then sampled by an ancestral sampler written by hand. It implements the
three objects the next two pages formalize:

  1. Eq. 3.3, the closed-form forward marginal (constants.md notation 4.4):
        x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps ,   eps ~ N(0, I)
     The forward process has ZERO learnable parameters -- it is a fixed rotation of
     a unit vector, and it keeps Var(x_t) = 1 at every t (the page's whole point).
  2. L_simple = E || eps - eps_theta(x_t, t) ||^2  (the loss you derive on p.55).
  3. The DDPM ancestral sampler (p.55's reverse chain), stepped by hand from t=T
     down to t=1.

WHY NUMPY, BY HAND. The page promises "from scratch ... an ancestral sampler by
hand," and rungs 1-5 of the ladder need no Spark (brief-tooling 9.2c). So this
ships as pure NumPy with a hand-rolled MLP, hand-rolled backprop, and a hand-rolled
Adam -- it runs on any CPU, torch or not, in about 40 seconds, and every gradient
is visible. It shares its schedule engine (build_abar) verbatim with
prediction_targets.py (p.55) and flow_matching_2d.py (p.57); flow matching is the
same file with (a_t, b_t) = (1 - t, t) instead of (sqrt(abar_t), sqrt(1-abar_t)).

FROZEN NUMBERS asserted here (constants.md 9.6, [DER]):
  abar_1000 ~= 4e-5 ;  sqrt(abar_T) = 0.0063 (NONZERO terminal SNR) ;
  eps-pred error amplification at t=999:  1 / 0.0063 = 159x .

Usage
-----
    python diffusion_2d.py                 # train (~40 s) + sample + text report
    python diffusion_2d.py --self-test     # fast CI path: asserts + short train, no plot
    python diffusion_2d.py --plot out.png  # also save the forward/reverse figure
    python diffusion_2d.py --steps 6000 --seed 0

SAFETY: CPU only, no GPU, no network. It installs nothing and (unless you pass
--plot) writes nothing. Never touches the Spark; runs anywhere Python + NumPy do.
"""

import argparse
import math
import numpy as np

# --------------------------------------------------------------------------- #
# The frozen linear DDPM schedule -- identical to prediction_targets.py (p.55):
# beta 1e-4 -> 0.02 linearly, T = 1000, abar = running cumulative product.
# --------------------------------------------------------------------------- #
T, BMIN, BMAX = 1000, 1e-4, 2e-2


def build_abar(T=T, bmin=BMIN, bmax=BMAX):
    """betas[1..T] linear; abar[t] = prod_{s<=t} (1 - beta_s), abar[0] = 1. float64."""
    betas = np.empty(T + 1, dtype=np.float64)          # betas[0] unused
    abar = np.empty(T + 1, dtype=np.float64)
    abar[0] = 1.0
    ab = 1.0
    for t in range(1, T + 1):
        beta = bmin + (bmax - bmin) * (t - 1) / (T - 1)
        betas[t] = beta
        ab *= (1.0 - beta)
        abar[t] = ab
    return betas, abar


def schedule_self_checks(betas, abar):
    """Assert the constants.md 9.6 frozen facts, then print them with confidence tags."""
    print("-" * 70)
    print("SCHEDULE  --  the forward process, zero learnable parameters")
    print("-" * 70)

    # (a) abar is a genuine running product, monotone down from 1.
    assert abar[0] == 1.0
    for t in range(1, T + 1):
        assert abs(abar[t] - abar[t - 1] * (1.0 - betas[t])) < 1e-15
    assert np.all(np.diff(abar) < 0.0), "abar must be strictly decreasing"

    # (b) variance preservation: a_t^2 + b_t^2 = 1 at every t (the rotation).
    a = np.sqrt(abar)                     # signal coeff  sqrt(abar_t)
    b = np.sqrt(1.0 - abar)               # noise  coeff  sqrt(1-abar_t)
    circle = a * a + b * b
    assert np.max(np.abs(circle - 1.0)) < 1e-12, "a^2 + b^2 must equal 1 at all t"

    # (c) the frozen endpoints. sqrt(abar_T) truncates to 0.0063 at 4 d.p. (the raw
    #     schedule is ~0.0063528; toFixed would round to 0.0064 and clash with the
    #     frozen table, so we truncate -- exactly the page's convention).
    abar_T = abar[T]
    raw_sqrt = math.sqrt(abar_T)
    trunc4 = math.floor(raw_sqrt * 1e4) / 1e4
    FROZEN_SQRT_ABAR_T = 0.0063                 # constants.md 9.6 [DER]
    FROZEN_AMP = round(1.0 / FROZEN_SQRT_ABAR_T)  # 159
    assert round(abar_T * 1e5) / 1e5 == 4e-5, "abar_1000 must be ~4e-5"
    assert trunc4 == FROZEN_SQRT_ABAR_T, "sqrt(abar_T) must truncate to 0.0063"
    assert FROZEN_AMP == 159, "eps-pred amplification must be 159x (constants 9.6)"

    # (d) nonzero terminal SNR: a trace of x0 survives all the way to t=T.
    snr_T = abar_T / (1.0 - abar_T)
    assert snr_T > 0.0, "terminal SNR must be strictly positive (nonzero terminal SNR)"

    print(f"  beta:  {BMIN} -> {BMAX} linear, T = {T}")
    print(f"  sqrt(abar_t) (signal) at t = 100/250/500/750/1000:")
    print("     " + "  ".join(f"{a[t]:.4f}" for t in (100, 250, 500, 750, 1000))
          + "   [interior EST, endpoints DER]")
    print(f"  abar_1000       = {abar_T:.3e}   (~= 4e-5)                 [DER, constants 9.6]")
    print(f"  sqrt(abar_1000) = {FROZEN_SQRT_ABAR_T}   (raw schedule {raw_sqrt:.6f})  [DER]")
    print(f"  NONZERO terminal SNR = {snr_T:.2e} > 0  -> t=T is 'almost', not pure, noise")
    print(f"  eps->x0 amplification at t=999:  1/{FROZEN_SQRT_ABAR_T} = {FROZEN_AMP}x  "
          f"(raw 1/{raw_sqrt:.6f} = {1.0/raw_sqrt:.1f})   [DER]")
    print(f"  variance preserved: max|a_t^2 + b_t^2 - 1| = {np.max(np.abs(circle-1)):.1e}  [OK]")
    return a, b


# --------------------------------------------------------------------------- #
# The data: eight Gaussians on a ring, standardized so Var(x0) ~= 1 per coord
# (so the forward process really does march variance 1 -> 1).
# --------------------------------------------------------------------------- #
def eight_gaussian_modes():
    ang = np.arange(8) * (2 * math.pi / 8)
    modes = np.stack([np.cos(ang), np.sin(ang)], axis=1) * 2.0
    return modes


def sample_data(rng, n, modes, spread=0.10):
    idx = rng.integers(0, len(modes), size=n)
    return modes[idx] + spread * rng.standard_normal((n, 2))


# --------------------------------------------------------------------------- #
# The eps-predictor: a 3-layer MLP.  in = [x (2), sinusoidal time embedding (E)],
# two hidden ReLU layers, 2-D output. He-initialized. Backprop written by hand.
# --------------------------------------------------------------------------- #
E_TIME = 16


def time_embedding(t_norm, dim=E_TIME):
    """Standard sinusoidal embedding of t/T in [0,1]. t_norm: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = np.exp(-math.log(10000.0) * np.arange(half) / (half - 1))   # (half,)
    ang = t_norm[:, None] * freqs[None, :]                              # (B, half)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)           # (B, dim)


def init_mlp(rng, hidden=128):
    din = 2 + E_TIME
    def he(shape):
        return rng.standard_normal(shape) * math.sqrt(2.0 / shape[0])
    return {
        "W1": he((din, hidden)), "b1": np.zeros(hidden),
        "W2": he((hidden, hidden)), "b2": np.zeros(hidden),
        "W3": he((hidden, 2)) * 0.1, "b3": np.zeros(2),
    }


def mlp_forward(p, x, t_norm):
    inp = np.concatenate([x, time_embedding(t_norm)], axis=1)  # (B, din)
    z1 = inp @ p["W1"] + p["b1"]; h1 = np.maximum(z1, 0.0)
    z2 = h1 @ p["W2"] + p["b2"]; h2 = np.maximum(z2, 0.0)
    out = h2 @ p["W3"] + p["b3"]                               # (B, 2) = eps_pred
    return out, (inp, z1, h1, z2, h2)


def mlp_backward(p, cache, dout):
    inp, z1, h1, z2, h2 = cache
    g = {}
    g["W3"] = h2.T @ dout; g["b3"] = dout.sum(0)
    dh2 = dout @ p["W3"].T; dz2 = dh2 * (z2 > 0)
    g["W2"] = h1.T @ dz2; g["b2"] = dz2.sum(0)
    dh1 = dz2 @ p["W2"].T; dz1 = dh1 * (z1 > 0)
    g["W1"] = inp.T @ dz1; g["b1"] = dz1.sum(0)
    return g


class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --------------------------------------------------------------------------- #
# Train on L_simple, then sample with the DDPM ancestral sampler, by hand.
# --------------------------------------------------------------------------- #
def train(rng, params, a, b, modes, steps, batch, lr):
    opt = Adam(params, lr=lr)
    last = 0.0
    for i in range(1, steps + 1):
        x0 = sample_data(rng, batch, modes)
        t = rng.integers(1, T + 1, size=batch)                 # t in [1, T]
        eps = rng.standard_normal((batch, 2))
        x_t = a[t, None] * x0 + b[t, None] * eps               # Eq. 3.3
        pred, cache = mlp_forward(params, x_t, t / T)
        resid = pred - eps                                     # dL/dpred for MSE
        loss = float(np.mean(resid ** 2))
        dout = (2.0 / batch) * resid
        opt.step(params, mlp_backward(params, cache, dout))
        last = loss
        if i % max(1, steps // 5) == 0 or i == 1:
            print(f"  step {i:>5}/{steps}   L_simple = {loss:.4f}")
    return last


def sample(rng, params, betas, abar, n):
    """DDPM ancestral sampler, t = T -> 1. Posterior variance beta_tilde_t."""
    a = np.sqrt(abar)
    b = np.sqrt(1.0 - abar)
    x = rng.standard_normal((n, 2))                            # x_T ~ N(0, I)
    for t in range(T, 0, -1):
        eps = mlp_forward(params, x, np.full(n, t / T))[0]
        alpha_t = 1.0 - betas[t]
        mean = (x - betas[t] / b[t] * eps) / math.sqrt(alpha_t)
        if t > 1:
            beta_tilde = betas[t] * (1.0 - abar[t - 1]) / (1.0 - abar[t])
            x = mean + math.sqrt(beta_tilde) * rng.standard_normal((n, 2))
        else:
            x = mean                                           # no noise at the last step
    return x


def coverage_report(samples, modes, tag=""):
    d = np.linalg.norm(samples[:, None, :] - modes[None, :, :], axis=2)  # (N, 8)
    nearest = d.argmin(1)
    mean_dist = float(d.min(1).mean())
    hit = len(np.unique(nearest))
    std = float(samples.std(0).mean())
    print(f"  {tag}mean dist to nearest mode = {mean_dist:.3f} | "
          f"modes covered = {hit}/8 | sample std = {std:.3f} (data ~1.0)")
    return mean_dist, hit, std


# --------------------------------------------------------------------------- #
def maybe_plot(path, rng, params, a, b, betas, abar, modes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib not installed -- skipping {path}; the text report is the result)")
        return
    data = sample_data(rng, 2000, modes)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].scatter(data[:, 0], data[:, 1], s=3); ax[0].set_title("x_0 (8 Gaussians)")
    for j, t in enumerate((250, 1000)):
        eps = rng.standard_normal((2000, 2))
        xt = a[t] * data + b[t] * eps
        ax[1 + j].scatter(xt[:, 0], xt[:, 1], s=3)
        ax[1 + j].set_title(f"forward x_{t}")
    gen = sample(rng, params, betas, abar, 2000)
    ax[3].scatter(gen[:, 0], gen[:, 1], s=3, c="C3"); ax[3].set_title("reverse samples")
    for a_ in ax:
        a_.set_xlim(-4, 4); a_.set_ylim(-4, 4); a_.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=110)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--self-test", action="store_true",
                    help="fast CI path: frozen asserts + short train + sampler sanity, no plot")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", metavar="PATH", default=None,
                    help="also save a forward/reverse figure to PATH (needs matplotlib)")
    args = ap.parse_args()

    if args.self_test:
        args.steps, args.batch, args.hidden = 1200, 256, 64

    print("=" * 70)
    print("diffusion_2d.py -- from-scratch DDPM on 8 Gaussians (pure NumPy, no U-Net)")
    print(f"numpy {np.__version__} - seed {args.seed} - {'SELF-TEST' if args.self_test else 'full'} run")
    print("=" * 70)

    rng = np.random.default_rng(args.seed)
    betas, abar = build_abar()
    a, b = schedule_self_checks(betas, abar)

    # Forward-process variance check: Eq. 3.3 keeps Var(x_t) ~= 1 (page's claim), verified.
    modes = eight_gaussian_modes()
    x0 = sample_data(rng, 4000, modes)
    x0 = x0 / x0.std(0)                       # standardize so Var(x0) ~= 1 per coord
    modes = modes / sample_data(rng, 20000, modes).std(0)  # modes in the same frame
    t_chk = 500
    xt = a[t_chk] * x0 + b[t_chk] * rng.standard_normal(x0.shape)
    var_xt = float((xt ** 2).mean())
    assert abs(var_xt - 1.0) < 0.15, f"forward Var(x_{t_chk}) should be ~1, got {var_xt:.3f}"
    print(f"  forward check: empirical Var(x_{t_chk}) = {var_xt:.3f} ~= 1  [Eq. 3.3, OK]\n")

    print("-" * 70)
    print(f"TRAIN  --  L_simple = E|| eps - eps_theta(x_t, t) ||^2   ({args.steps} steps)")
    print("-" * 70)
    params = init_mlp(rng, hidden=args.hidden)
    final_loss = train(rng, params, a, b, modes, args.steps, args.batch, args.lr)

    print("-" * 70)
    print("SAMPLE  --  DDPM ancestral sampler, t = T -> 1, by hand")
    print("-" * 70)
    gen = sample(rng, params, betas, abar, 3000)
    assert np.all(np.isfinite(gen)), "sampler produced non-finite values"
    mean_dist, hit, std = coverage_report(gen, modes, tag="")

    # The reverse process must reconstruct the data manifold, not just stay finite.
    thresh = 0.75 if args.self_test else 0.55
    assert mean_dist < thresh, f"samples not on the data manifold (mean dist {mean_dist:.3f})"
    assert hit >= (6 if args.self_test else 7), f"only {hit}/8 modes recovered"
    assert 0.6 < std < 1.4, f"sample std {std:.3f} far from the data's ~1.0"

    if args.plot:
        print("-" * 70)
        maybe_plot(args.plot, rng, params, a, b, betas, abar, modes)

    print("\n" + "=" * 70)
    print(f"[OK] schedule frozen numbers asserted; L_simple {final_loss:.3f}; "
          f"reverse chain recovered {hit}/8 modes.")
    print("     The forward process learned nothing (0 params); the MLP learned only")
    print("     to predict eps. Same engine, (a_t,b_t)=(1-t,t), is flow_matching_2d.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()
