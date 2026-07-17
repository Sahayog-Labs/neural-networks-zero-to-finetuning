#!/usr/bin/env python3
"""
23_double_descent.py -- model-wise double descent, reproduced on a small
synthetic set. Sweep hidden-layer WIDTH (= the only trained parameter count,
P) across the interpolation threshold P ~= N_train and watch test error do
something classical statistics never predicted: rise to a peak right at the
threshold, then come back DOWN as P keeps growing.

Course artifact for p.23 ("Generalization II: Bias-Variance and Double
Descent"). Same construction as the page's browser demo, on purpose:

  - a FIXED (untrained) one-hidden-layer tanh network supplies the "features"
    phi(x) = tanh(x @ W1 + b1). It is never updated -- so it cannot quietly
    be doing the regularizing itself, and P below counts ONLY what is
    actually fit to data: the h read-out weights beta.
  - the read-out layer is solved EXACTLY, in closed form, by ridge regression:

        beta = argmin_beta  ||Phi_train @ beta - y_train||^2 + lambda ||beta||^2
             = (Phi_train^T Phi_train + lambda I)^-1 Phi_train^T y_train

    lambda = 0 is the classical minimum-norm least-squares solution (what a
    rank-deficient system falls back to when h > N_train). This is the
    textbook construction (Belkin et al. 2019; Hastie et al. 2019) that first
    pinned the phenomenon down cleanly -- solved exactly, so a spike in the
    curve is a mathematical fact, never an artifact of an unlucky SGD run.

Two things happen, in order:

  1. THE SPIKE: sweep width across the threshold at lambda=0. Test error
     falls, SPIKES hard right at width ~= N_train (an interpolating fit has
     essentially one solution there, and it is a brittle, high-variance one),
     then descends again past it -- the classical U followed by a second
     descent. Self-check: the spike must actually be there (this page's
     [EST] claim, a reproduction -- not a frozen number).
  2. THE HONEST PART: re-run the same sweep at a nonzero weight decay and
     show the peak flatten. The page's box warn is explicit that a real
     chunk of the headline spike is an artifact of training an
     under-regularized model right at its most brittle capacity -- watch it
     happen, live, on your own machine, not a claim to take on faith.

A note on --seed: the default (42) is verified below to show a clean spike and
a clean flattening under weight decay. This is a genuinely small-sample (N=60)
experiment, though, and a handful of --seed values draw a dataset/feature-bank
pair where the second descent doesn't land cleanly inside the 24 swept
widths -- the self-check assertions then fail LOUDLY rather than being
fudged to always pass. That variance is not a bug to paper over; it is the
same bias-variance-at-small-N lesson the page teaches, one level up.

Then, separately, the "which regime are you in" question the page's table
poses is answered with real numbers: his LoRA fine-tune sits at
P = 8,190,735,360 (Qwen3-8B total params, constants.md section 1.2) against
N = 500 (his hand-written dataset, decisions.md D-22 / brief-tooling-hardware
p.239 -- NOT a constants.md-frozen figure, printed as the course's stated
scale, not laundered into a fact). P/N is in the tens of millions: nowhere
NEAR the interpolation threshold this script's demo spikes at. Regularization
(LoRA's rank, early stopping, a small LR) is doing the work, not double
descent (page 23's "P >> N -- this is him" row).

Two run modes, same arithmetic, two different libraries:

    python 23_double_descent.py               # torch path -- the page's own
                                                # shown code (torch.randn +
                                                # torch.linalg.lstsq)
    python 23_double_descent.py --self-test    # pure-numpy path -- no torch,
                                                # no GPU -- proves the same
                                                # two assertions (spike exists,
                                                # spike flattens under decay)
                                                # on any machine with numpy

SAFETY: CPU only, seconds to tens of seconds. Largest matrix solved is
roughly (6 x N_train) square -- a few hundred rows at most, dense direct
solve, no GPU involved. Writes and installs nothing.
"""

import argparse
import math
import sys

# --------------------------------------------------------------------------- #
# Frozen / documented numbers this script reports, never invents.
# --------------------------------------------------------------------------- #
# Qwen3-8B total parameter count -- constants.md section 1.2, [VP], confirmed
# two independent ways (formula AND checkpoint's model.safetensors.index.json).
QWEN3_8B_TOTAL_PARAMS = 8_190_735_360
# LoRA r=16, all-linear, trainable count -- constants.md section 3 / line 242.
LORA_R16_TRAINABLE_PARAMS = 43_646_976
# His dataset scale -- NOT in constants.md's frozen tables. Documented in
# decisions.md D-22 ("written for 500 examples") and brief-tooling-hardware.md
# p.239 ("~500 hand-written examples"). Printed as the course's stated scale,
# tagged [EST]-of-a-decision, never asserted as a constants.md fact.
HIS_DATASET_EXAMPLES = 500

# --------------------------------------------------------------------------- #
# Synthetic-set defaults. Small on purpose ("a small synthetic set" per spec):
# closed-form solves stay instant even swept across ~24 widths.
# --------------------------------------------------------------------------- #
N_TOTAL_DEFAULT = 60            # total points -- matches the browser demo's own default
                                 # (assets/nn.js d-dd-ctrls "dataset size N" slider, value 60)
TRAIN_FRAC = 0.7                # -> N_train ~= 42, same as the browser demo's split
NOISE_DEFAULT = 0.15            # moons x/y input noise -- fixed in the browser demo too
                                 # (sweep() hardcodes {noise: 0.15, ...}, not a slider)
LABEL_NOISE_DEFAULT = 0.15      # fraction of labels flipped -- matches the browser demo's
                                 # "label noise" slider default. The demo's own comment
                                 # explains why this matters: "this is what makes
                                 # overfitting visible -- a model with enough capacity will
                                 # memorize the flipped points, and the val curve turns up
                                 # while the train curve keeps falling." Without label
                                 # noise the moons boundary is smooth enough that no width
                                 # is forced to memorize noise, and the threshold spike
                                 # never gets large enough to be unmissable.
SEED = 42
N_WIDTHS = 24                   # log-spaced sweep points, matches the browser demo
RIDGE_FLOOR = 1e-3              # numerical floor at "lambda=0" -- large enough that a
                                 # near-singular Phi^T Phi doesn't blow up on screen,
                                 # small enough the spike stays the dominant signal
                                 # (identical choice to the page's JS demo).
WD_COMPARE = 0.08               # the "turn lambda up" comparison run -- within the browser
                                 # demo's own lambda slider range (0 to 0.10, "the star control")


# --------------------------------------------------------------------------- #
# Dataset: two interleaving half-moons, identical construction to the page's
# browser demo's makeDataset("moons", ...) (assets/nn.js). Treated as a
# regression target y in {0.0, 1.0} so ridge regression -- not classification
# -- is what is being interpolated; this is exactly what the JS demo does too
# (ridgeFit + MSE test_loss).
# --------------------------------------------------------------------------- #
def moons_xy(n, noise, seed, label_noise=0.0):
    """Returns (X, y) as plain Python lists: X is n x 2, y is n floats in
    {0.0, 1.0}. Pure-Python/math so this ONE function backs both the numpy
    self-test and the torch default path -- the dataset itself can never
    drift between the two run modes.

    label_noise: fraction of labels independently flipped (0<->1) -- see the
    LABEL_NOISE_DEFAULT comment. This is what turns the interpolation
    threshold into a genuine variance blow-up: a width-P model at P~=N_train
    has just enough capacity to fit the flipped points too, and does, at the
    cost of generalizing badly."""
    rng = _Rng(seed)
    half = n // 2
    X, y = [], []
    for i in range(n):
        c = 0 if i < half else 1
        t = math.pi * (i / half if i < half else (i - half) / (n - half))
        if c == 0:
            px, py = math.cos(t), math.sin(t)
        else:
            px, py = 1 - math.cos(t), 0.5 - math.sin(t)
        x1 = (px - 0.5) * 1.6 + rng.normal(0, noise)
        x2 = (py - 0.25) * 1.6 + rng.normal(0, noise)
        X.append([x1, x2])
        y.append(float(c))
    if label_noise > 0.0:
        for i in range(n):
            if rng.uniform01() < label_noise:
                y[i] = 1.0 - y[i]
    return X, y


def to_pm1(y):
    """0.0/1.0 class labels -> +-1.0 regression targets -- exactly what the
    browser demo does (`ds.y.map(v => v === 1 ? 1 : -1)`) before fitting.
    Predicting the mean of a balanced +-1 set scores test MSE ~= 1.0, which
    is why the demo's readout treats 1.0 as the 'you learned nothing' line."""
    return [1.0 if v == 1.0 else -1.0 for v in y]


class _Rng:
    """Deterministic, dependency-free Box-Muller RNG so moons_xy() needs
    neither numpy nor torch -- the same dataset points feed both run modes,
    byte-identical, given the same seed."""

    def __init__(self, seed):
        self._state = seed & 0xFFFFFFFF
        if self._state == 0:
            self._state = 1
        self._spare = None

    def _next_u32(self):
        # xorshift32 -- fast, deterministic, good enough for a demo dataset.
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self._state = x & 0xFFFFFFFF
        return self._state

    def uniform01(self):
        return self._next_u32() / 4294967296.0

    def normal(self, mu=0.0, sigma=1.0):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return mu + sigma * v
        u1 = max(self.uniform01(), 1e-12)
        u2 = self.uniform01()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        self._spare = z1
        return mu + sigma * z0


def train_test_split(X, y, frac, seed):
    rng = _Rng(seed + 1)
    idx = list(range(len(X)))
    # Fisher-Yates using the same dependency-free RNG.
    for i in range(len(idx) - 1, 0, -1):
        j = int(rng.uniform01() * (i + 1))
        if j > i:
            j = i
        idx[i], idx[j] = idx[j], idx[i]
    n_train = int(round(len(X) * frac))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    Xtr = [X[i] for i in train_idx]
    ytr = [y[i] for i in train_idx]
    Xte = [X[i] for i in test_idx]
    yte = [y[i] for i in test_idx]
    return Xtr, ytr, Xte, yte


def log_widths(n_train, count=N_WIDTHS, override=None):
    """Log-spaced integer widths from 1 up to 6x N_train -- the same range
    the browser demo sweeps, so the threshold sits well inside the range on
    both sides. Pass `override` (an iterable of ints, e.g. from --widths) to
    sweep an exact custom list instead."""
    if override is not None:
        return sorted(set(int(w) for w in override))
    lo, hi = 1.0, max(8, n_train * 6)
    if count <= 1:
        return [int(round(hi))]
    out = []
    for k in range(count):
        frac = k / (count - 1)
        w = lo * (hi / lo) ** frac
        out.append(max(1, int(round(w))))
    # de-duplicate while preserving order (small widths can collide when rounded)
    seen, uniq = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


# --------------------------------------------------------------------------- #
# --self-test path: numpy only, no torch, no GPU.
# --------------------------------------------------------------------------- #
def ridge_fit_random_features_numpy(X_train, y_train, X_test, y_test, width, weight_decay, seed=0):
    """Same primal/dual switch as the browser demo's ridgeFit() (assets/nn.js):
    width <= N_train solves the (width x width) primal system directly; width
    > N_train solves the (N_train x N_train) DUAL (kernel) system instead and
    maps back -- at the ridge floor this dual form IS the minimum-norm
    least-squares interpolant (Belkin/Hastie's construction). Solving the huge
    near-singular primal system directly past the threshold does NOT recover
    this minimum-norm solution numerically; the dual branch is what makes the
    second descent a reproducible fact instead of solver noise."""
    import numpy as np

    rng = np.random.default_rng(seed)
    Xtr = np.asarray(X_train, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64)
    Xte = np.asarray(X_test, dtype=np.float64)
    yte = np.asarray(y_test, dtype=np.float64)
    n_train = Xtr.shape[0]

    # P = width = the ONLY trained parameters: the random hidden layer is
    # fixed, so it can't be quietly doing the regularizing itself.
    W1 = rng.standard_normal((2, width))
    b1 = rng.standard_normal(width)
    phi = lambda X: np.tanh(X @ W1 + b1)
    Phi_train, Phi_test = phi(Xtr), phi(Xte)

    ridge = weight_decay if weight_decay > 0 else RIDGE_FLOOR
    if width <= n_train:
        A = Phi_train.T @ Phi_train + ridge * np.eye(width)
        rhs = Phi_train.T @ ytr
        beta, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        preds = Phi_test @ beta
    else:
        K = Phi_train @ Phi_train.T
        alpha, *_ = np.linalg.lstsq(K + ridge * np.eye(n_train), ytr, rcond=None)
        beta = Phi_train.T @ alpha
        preds = Phi_test @ beta

    test_loss = float(np.mean((preds - yte) ** 2))
    return width, test_loss


def sweep_numpy(n_total, noise, weight_decay, seed=SEED, widths_override=None, label_noise=LABEL_NOISE_DEFAULT):
    X, y = moons_xy(n_total, noise, seed, label_noise=label_noise)
    Xtr, ytr, Xte, yte = train_test_split(X, to_pm1(y), TRAIN_FRAC, seed)
    n_train = len(Xtr)
    widths = log_widths(n_train, override=widths_override)
    results = [ridge_fit_random_features_numpy(Xtr, ytr, Xte, yte, w, weight_decay, seed=seed) for w in widths]
    return results, n_train


def _spike_diagnostics(results, n_train):
    """Same peak/minima logic as the browser demo's draw() (23-*.html): the
    THRESHOLD is the swept width closest to n_train. The PEAK is the highest
    test loss within a window around the threshold (0.7x-1.4x of it) -- NOT
    the single closest-width point and NOT a global max. The page's own
    comment explains why: 'a fixed random hidden layer this small ... can
    develop a SEPARATE, unrelated conditioning blow-up out at 2x-6x N_train
    -- a real number, but not the interpolation-threshold phenomenon this
    page teaches' -- so a global max would sometimes mislabel that unrelated
    blow-up as the threshold spike. The two comparison points are the
    CLASSICAL-side minimum (best test loss at or before the threshold) and
    the OVERPARAMETERIZED-side minimum (best test loss after it) -- 'the
    better minimum' the page's readout divides the peak by."""
    widths = [w for w, _ in results]
    losses = [l for _, l in results]
    thresh_i = min(range(len(widths)), key=lambda i: abs(widths[i] - n_train))
    thresh_w = widths[thresh_i]

    peak_i, peak_loss = thresh_i, -math.inf
    for i, w in enumerate(widths):
        if w < 0.7 * thresh_w or w > 1.4 * thresh_w:
            continue
        if losses[i] > peak_loss:
            peak_i, peak_loss = i, losses[i]

    # Same before/after split as the browser demo (P<=threshold vs
    # P>threshold), excluding the peak's own point so a minimum that happens
    # to coincide with the peak can't make "peak > minimum" fail by
    # construction rather than by the phenomenon being genuinely absent.
    before_idxs = [i for i, w in enumerate(widths) if w <= thresh_w and i != peak_i]
    first_min_i = min(before_idxs, key=lambda i: losses[i]) if before_idxs else thresh_i

    after_idxs = [i for i, w in enumerate(widths) if w > thresh_w and i != peak_i]
    second_min_i = min(after_idxs, key=lambda i: losses[i]) if after_idxs else thresh_i

    return {
        "thresh_i": thresh_i, "thresh_w": thresh_w,
        "peak_i": peak_i, "peak_w": widths[peak_i], "peak_loss": peak_loss,
        "first_min_i": first_min_i, "first_min_w": widths[first_min_i], "first_min_loss": losses[first_min_i],
        "second_min_i": second_min_i, "second_min_w": widths[second_min_i], "second_min_loss": losses[second_min_i],
    }


def _print_diagnostics(d, n_train):
    print(f"\n  threshold P={d['thresh_w']} (closest to N_train={n_train})")
    print(f"  peak (max MSE in 0.7x-1.4x of threshold): P={d['peak_w']} ({d['peak_w']/n_train:.2f}x), "
          f"MSE={d['peak_loss']:.5f}")
    print(f"  classical-side minimum (P<=threshold):    P={d['first_min_w']} ({d['first_min_w']/n_train:.2f}x), "
          f"MSE={d['first_min_loss']:.5f}")
    print(f"  overparameterized-side minimum (P>threshold): P={d['second_min_w']} "
          f"({d['second_min_w']/n_train:.2f}x), MSE={d['second_min_loss']:.5f}")
    better = min(d["first_min_loss"], d["second_min_loss"])
    print(f"  peak is {d['peak_loss']/better:.2f}x the better minimum")


def _assert_peak_flattens(d0, d1):
    """d0 = lambda=0 run, d1 = lambda=WD_COMPARE run. Both diagnostics dicts
    from _spike_diagnostics(). Asserts the peak-to-better-minimum RATIO (the
    page's own readout quantity) shrinks under weight decay, and returns the
    percent shrink for the caller to print."""
    ratio0 = d0["peak_loss"] / min(d0["first_min_loss"], d0["second_min_loss"])
    ratio1 = d1["peak_loss"] / min(d1["first_min_loss"], d1["second_min_loss"])
    assert ratio1 < ratio0, (
        f"[EST reproduction] weight decay should flatten the peak: peak/better-minimum ratio at "
        f"lambda={WD_COMPARE} ({ratio1:.2f}x) should be smaller than at lambda=0 ({ratio0:.2f}x), "
        f"got the opposite"
    )
    return 100 * (1 - (ratio1 - 1) / (ratio0 - 1)) if ratio0 > 1 else 0.0


def print_table(results, n_train, title):
    print(f"\n  {title}")
    print(f"  {'width P':>10}  {'P/N_train':>10}  {'test MSE':>12}")
    print("  " + "-" * 36)
    for w, loss in results:
        marker = "  <-- P~=N_train (threshold)" if abs(w - n_train) == min(abs(ww - n_train) for ww, _ in results) else ""
        print(f"  {w:>10}  {w / n_train:>9.2f}x  {loss:>12.5f}{marker}")


def regime_test():
    print("\n" + "=" * 72)
    print("WHICH REGIME IS *HIS* FINE-TUNE IN? (p.23's table, real numbers)")
    print("=" * 72)
    P = QWEN3_8B_TOTAL_PARAMS
    P_lora = LORA_R16_TRAINABLE_PARAMS
    N = HIS_DATASET_EXAMPLES
    assert P == 8_190_735_360, "Qwen3-8B total params must match constants.md section 1.2"
    assert P_lora == 43_646_976, "LoRA r=16 all-linear trainable count must match constants.md section 3"
    print(f"  P (Qwen3-8B, total)        = {P:>15,}   [VP, constants.md sec 1.2]")
    print(f"  P (LoRA r=16 all-linear)   = {P_lora:>15,}   [DER, constants.md sec 3]")
    print(f"  N (his dataset)            = {N:>15,}   [documented scale, decisions.md D-22 /")
    print(f"                                              brief-tooling-hardware.md p.239 --")
    print(f"                                              NOT a constants.md-frozen figure]")
    print(f"\n  P / N (full model)  = {P / N:>18,.1f}x")
    print(f"  P / N (LoRA-trained)= {P_lora / N:>18,.1f}x")
    print()
    print("  Both ratios are in the tens of thousands to tens of millions -- this demo's")
    print("  interpolation-threshold spike sits at P/N_train ~= 1.0x, right where the sweep")
    print("  above peaked. His fine-tune is nowhere near that peak; it is deep in the second")
    print("  descent's territory, P >> N (p.23's third table row -- 'this is him'). Double")
    print("  descent is not what is saving him from overfitting on 500 examples --")
    print("  regularization (LoRA's low rank, early stopping, a small LR, few epochs) is.")
    print("=" * 72)


def self_test(n_total, noise, seed, widths_override=None, label_noise=LABEL_NOISE_DEFAULT) -> None:
    import numpy as np

    print("=" * 72)
    print("23_double_descent.py -- self-test (pure numpy, no torch, no GPU)")
    print(f"numpy {np.__version__}   seed {seed}   N_total {n_total}   noise {noise}   "
          f"label_noise {label_noise}")
    print("=" * 72)

    # --- Run 1: lambda ~= 0 (the numerical floor) -- the spike must be there.
    results0, n_train = sweep_numpy(n_total, noise, weight_decay=0.0, seed=seed,
                                     widths_override=widths_override, label_noise=label_noise)
    print_table(results0, n_train, f"lambda=0 (ridge floor {RIDGE_FLOOR}) -- N_train={n_train}")
    d0 = _spike_diagnostics(results0, n_train)
    _print_diagnostics(d0, n_train)

    assert d0["peak_loss"] > d0["first_min_loss"], (
        f"[EST reproduction] expected a spike: peak MSE {d0['peak_loss']:.5f} at P={d0['peak_w']} should "
        f"exceed the classical-side minimum {d0['first_min_loss']:.5f} at P={d0['first_min_w']}, got the opposite"
    )
    assert d0["peak_loss"] > d0["second_min_loss"], (
        f"[EST reproduction] expected a SECOND DESCENT: peak MSE {d0['peak_loss']:.5f} at P={d0['peak_w']} "
        f"should exceed the overparameterized-side minimum {d0['second_min_loss']:.5f} at "
        f"P={d0['second_min_w']}, got the opposite -- no second descent"
    )
    print("\n  [OK] spike confirmed: test error rises to a peak near P~=N_train, then descends")
    print("       again past it -- the classical U followed by a second descent. [EST] --")
    print("       a reproduction on a synthetic set, not a frozen constants.md number.")

    # --- Run 2: weight_decay turned up -- the SAME spike should flatten.
    results1, _ = sweep_numpy(n_total, noise, weight_decay=WD_COMPARE, seed=seed,
                               widths_override=widths_override, label_noise=label_noise)
    print_table(results1, n_train, f"lambda={WD_COMPARE} -- same N_train={n_train}, peak should flatten")
    d1 = _spike_diagnostics(results1, n_train)
    _print_diagnostics(d1, n_train)

    shrink_pct = _assert_peak_flattens(d0, d1)
    print(f"\n  [OK] peak-to-better-minimum ratio shrank {shrink_pct:.0f}% under weight decay -- ")
    print("       'the honest part': a real chunk of the headline spike is a regularization")
    print("       artifact, not a separate law of nature. (p.23's box warn, watched")
    print("       happening on real numbers, not asserted.)")

    regime_test()

    print("\n" + "=" * 72)
    print("All self-test assertions PASS (numpy-only, no torch, no GPU required).")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Default path: torch -- the page's own shown code
# (ridge_fit_random_features), plus the same two assertions.
# --------------------------------------------------------------------------- #
def ridge_fit_random_features(X_train, y_train, X_test, y_test, width, weight_decay, seed=0):
    """The function shown on p.23's '.box try' codeblock (torch.randn +
    torch.linalg.lstsq), extended with the same primal/dual switch the
    browser demo's ridgeFit() uses (assets/nn.js): width <= N_train solves
    the (width x width) primal system directly, exactly as shown on the page;
    width > N_train solves the (N_train x N_train) DUAL system instead and
    maps back -- at the ridge floor this dual form IS the minimum-norm
    least-squares interpolant (Belkin/Hastie's construction). The page's
    inline snippet shows the primal branch only, for readability; both
    branches are needed to actually reproduce the second descent."""
    import torch

    g = torch.Generator().manual_seed(seed)
    n_train = X_train.shape[0]
    W1 = torch.randn(2, width, generator=g)
    b1 = torch.randn(width, generator=g)
    phi = lambda X: torch.tanh(X @ W1 + b1)
    Phi_train, Phi_test = phi(X_train), phi(X_test)
    ridge = weight_decay if weight_decay > 0 else RIDGE_FLOOR

    if width <= n_train:
        beta = torch.linalg.lstsq(
            Phi_train.T @ Phi_train + ridge * torch.eye(width), Phi_train.T @ y_train
        ).solution
        preds = Phi_test @ beta
    else:
        K = Phi_train @ Phi_train.T
        alpha = torch.linalg.lstsq(K + ridge * torch.eye(n_train), y_train).solution
        beta = Phi_train.T @ alpha
        preds = Phi_test @ beta

    test_loss = ((preds - y_test) ** 2).mean().item()
    return width, test_loss


def sweep_torch(n_total, noise, weight_decay, seed=SEED, widths_override=None, label_noise=LABEL_NOISE_DEFAULT):
    import torch

    X, y = moons_xy(n_total, noise, seed, label_noise=label_noise)
    Xtr, ytr, Xte, yte = train_test_split(X, to_pm1(y), TRAIN_FRAC, seed)
    n_train = len(Xtr)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float64)
    ytr_t = torch.tensor(ytr, dtype=torch.float64)
    Xte_t = torch.tensor(Xte, dtype=torch.float64)
    yte_t = torch.tensor(yte, dtype=torch.float64)
    widths = log_widths(n_train, override=widths_override)
    results = [
        ridge_fit_random_features(Xtr_t, ytr_t, Xte_t, yte_t, w, weight_decay, seed=seed)
        for w in widths
    ]
    return results, n_train


def run_torch(n_total, noise, seed, widths_override=None, label_noise=LABEL_NOISE_DEFAULT) -> None:
    import torch

    print("=" * 72)
    print("23_double_descent.py -- model-wise double descent, real closed-form ridge fits")
    print(f"torch {torch.__version__}   seed {seed}   N_total {n_total}   noise {noise}   "
          f"label_noise {label_noise}")
    print("=" * 72)

    results0, n_train = sweep_torch(n_total, noise, weight_decay=0.0, seed=seed,
                                     widths_override=widths_override, label_noise=label_noise)
    print_table(results0, n_train, f"lambda=0 (ridge floor {RIDGE_FLOOR}) -- N_train={n_train}")
    d0 = _spike_diagnostics(results0, n_train)
    _print_diagnostics(d0, n_train)

    assert d0["peak_loss"] > d0["first_min_loss"], "[EST reproduction] expected a spike before the threshold"
    assert d0["peak_loss"] > d0["second_min_loss"], "[EST reproduction] expected a second descent past the threshold"
    print("\n  [OK] spike confirmed: peak near P~=N_train, second descent past it. [EST] --")
    print("       a reproduction on a synthetic set, not a frozen constants.md number.")

    results1, _ = sweep_torch(n_total, noise, weight_decay=WD_COMPARE, seed=seed,
                               widths_override=widths_override, label_noise=label_noise)
    print_table(results1, n_train, f"lambda={WD_COMPARE} -- same N_train={n_train}, peak should flatten")
    d1 = _spike_diagnostics(results1, n_train)
    _print_diagnostics(d1, n_train)

    shrink_pct = _assert_peak_flattens(d0, d1)
    print(f"\n  [OK] peak-to-better-minimum ratio shrank {shrink_pct:.0f}% under weight decay -- the")
    print("       honest part, watched happening on real numbers, not asserted.")

    regime_test()

    print("\n" + "=" * 72)
    print("A U followed by a second descent -- and most of the U's sharp point was")
    print("regularization doing its job badly, not a new law bolted onto bias-variance.")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Model-wise double descent on a small synthetic set -- p.23 companion script."
    )
    ap.add_argument("--n-total", type=int, default=N_TOTAL_DEFAULT,
                     help=f"total synthetic points generated (default {N_TOTAL_DEFAULT})")
    ap.add_argument("--noise", type=float, default=NOISE_DEFAULT,
                     help=f"moons dataset input noise std (default {NOISE_DEFAULT})")
    ap.add_argument("--label-noise", type=float, default=LABEL_NOISE_DEFAULT,
                     help=f"fraction of labels flipped -- what makes the threshold spike "
                          f"unmissable (default {LABEL_NOISE_DEFAULT})")
    ap.add_argument("--seed", type=int, default=SEED,
                     help=f"RNG seed (default {SEED}, verified to show the spike). At N=60 "
                          f"this is a genuinely small-sample experiment -- not every seed "
                          f"draws a dataset/feature-bank pair where the second descent is "
                          f"visible in only 24 swept widths; that variance IS the lesson, "
                          f"not a bug, but the self-check can fail loudly on an unlucky seed.")
    ap.add_argument("--widths", type=str, default=None,
                     help="comma-separated widths to sweep instead of the default "
                          "log-spaced sweep, e.g. --widths 50,150,210,300,1000")
    ap.add_argument("--self-test", action="store_true",
                     help="pure-numpy path: no torch, no GPU required")
    args = ap.parse_args()

    widths_override = None
    if args.widths:
        try:
            widths_override = [int(w) for w in args.widths.split(",")]
        except ValueError:
            print(f"--widths must be a comma-separated list of ints, got: {args.widths!r}")
            sys.exit(2)

    if args.self_test:
        self_test(args.n_total, args.noise, args.seed, widths_override=widths_override,
                   label_noise=args.label_noise)
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not importable. On the Spark, ComfyUI's venv has one:")
        print("  ~/ComfyUI/.venv/bin/python 23_double_descent.py")
        print("Or run the dependency-free version right here:")
        print("  python 23_double_descent.py --self-test")
        sys.exit(1)

    run_torch(args.n_total, args.noise, args.seed, widths_override=widths_override,
              label_noise=args.label_noise)


if __name__ == "__main__":
    main()
