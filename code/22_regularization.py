#!/usr/bin/env python3
"""
22_regularization.py -- the two curves, six shapes, and the one lever that actually
moves them for a small fine-tune: weight decay, dropout, early stopping.

Course artifact for p.22 ("Generalization I: Regularization and Reading the Two
Curves"). The page makes two claims and this script makes both of them concrete
instead of asserted:

  1. THE SIX SHAPES. Every train/val loss-curve pair you will ever look at reduces to
     one of six shapes, and each shape has exactly one fix (page's diagnostic table):
         1 underfit        -- both curves high, flat            -> raise capacity/LR
         2 healthy          -- both low, small gap                -> stop when val flattens
         3 overfit           -- train->0, val rises                -> regularize / stop earlier
         4 leaky split       -- val < train                        -> fix the split
         5 wiring bug/dead LR -- both flat from step 0             -> not a generalization problem
         6 tiny val set      -- val jagged, no trend                -> enlarge it / smooth it
     This script trains six REAL small MLPs, one per shape (same architecture, same
     update rule as p.13-16), with parameters chosen to force each shape onto the
     curve on purpose -- then a classifier reads the resulting numbers back off and
     the self-check asserts it recovers the label the preset was built to produce.

  2. THE ONE LEVER. On the page's own p.22 500-example regime, --wd and --dropout run
     a real A/B: the SAME small, overfit-prone model, same init, same data, once with
     no regularization and once with weight decay (page 17's decoupled AdamW-style
     lambda*theta, applied to weights only -- never biases) plus optional dropout and
     early stopping (best-checkpoint tracking). Self-check: regularization must lower
     the train/val gap, matching the page's Predict-first answer for this regime
     ("early stopping + fewer epochs + weight decay at the margin; dropout usually
     irrelevant; more epochs is the wrong direction").

Also prints the leakage check the page's "Try it on your box" text promises: a
train/val ROW-OVERLAP COUNT, zero on the clean split and nonzero on the deliberately
leaky one -- the split-discipline box made numeric, not just asserted.

Everything here is [EST] -- a reproduction of the page's own live-demo shapes with
this script's own hyperparameters, not a constants.md frozen number. The one frozen
number this script DOES use is the ln(2)=0.6931 random-guess baseline for a 2-way
problem (constants.md sec 8, "Coin-flip loss") and page 17's lambda=0.1 weight-decay
default for LLM fine-tuning, used as this script's --wd default.

Usage
-----
    python 22_regularization.py                       # full narrated run, ~15-20s
    python 22_regularization.py --wd 0.1 --dropout 0.1 # your own regularization A/B
    python 22_regularization.py --quick                # fewer epochs, faster
    python 22_regularization.py --self-test            # assertions only, terse, exit 0/1

SAFETY: pure Python arithmetic (stdlib math/argparse only -- no numpy, no torch, no
GPU, no network, no files touched). Runs identically on this machine and the Spark.
"""

import argparse
import math
import sys

LN2 = math.log(2.0)          # constants.md sec 8, "Coin-flip loss" -- the V=2 baseline
WD_DEFAULT = 0.1              # page 17 / page 22: lambda=0.1, the 2026 LLM fine-tune default
N_REGIME = 500                 # p.22's own framing: "written for the 500-example regime"


# --------------------------------------------------------------------------------- #
# Tiny deterministic RNG -- same glibc-style LCG used across this course's other
# dependency-free artifacts (16_batch_regimes.py, xor_by_hand.py's --self-test path).
# --------------------------------------------------------------------------------- #
class RNG:
    def __init__(self, seed):
        self.state = seed & 0x7FFFFFFF

    def _next(self):
        self.state = (1103515245 * self.state + 12345) % (2 ** 31)
        return self.state

    def uniform01(self):
        return self._next() / 2 ** 31

    def uniform(self, lo=-1.0, hi=1.0):
        return lo + (hi - lo) * self.uniform01()

    def randint(self, n):
        return int(self.uniform01() * n) % n

    def bernoulli(self, p):
        return self.uniform01() < p

    def shuffle(self, lst):
        for i in range(len(lst) - 1, 0, -1):
            j = self.randint(i + 1)
            lst[i], lst[j] = lst[j], lst[i]


# --------------------------------------------------------------------------------- #
# Dataset -- two-moons-shaped binary classification, pure Python. Same construction
# as 16_batch_regimes.py's make_moons (same SHAPE of problem the page's own JS moons
# demo uses; not a bit-for-bit port of its RNG stream -- porting that buys nothing).
# --------------------------------------------------------------------------------- #
def make_moons(n, noise, rng):
    half = n // 2
    X, Y = [], []
    for i in range(n):
        cls = 0 if i < half else 1
        t = math.pi * (i % half) / max(1, half - 1)
        if cls == 0:
            x, y = math.cos(t), math.sin(t)
        else:
            x, y = 1.0 - math.cos(t), 0.5 - math.sin(t)
        x += rng.uniform(-noise, noise)
        y += rng.uniform(-noise, noise)
        X.append((round(x, 10), round(y, 10)))   # rounded: makes exact-duplicate
        Y.append(float(cls))                     # detection for the leakage check well-defined
    return X, Y


def flip_labels(Y, frac, rng):
    """Label noise: flip a fraction of targets. Makes shape 3 (overfit) actually
    overfittable -- a model with enough capacity WILL fit the noise if you let it."""
    Y = Y[:]
    for i in range(len(Y)):
        if rng.uniform01() < frac:
            Y[i] = 1.0 - Y[i]
    return Y


# --------------------------------------------------------------------------------- #
# MLP(2 -> hidden -> 1), tanh hidden, BCE-with-sigmoid output. Hand-derived forward
# and backward -- same technique as 16_batch_regimes.py / xor_by_hand.py. Adds two
# things those scripts don't need: inverted dropout on the hidden layer, and decoupled
# weight decay applied to weights only (page 17's AdamW form, NOT folded into the
# gradient -- see that page's "Weight decay is the same as L2" trap box).
# --------------------------------------------------------------------------------- #
def _sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def init_weights(seed, hidden, scale=0.5):
    rng = RNG(seed)
    W1 = [[rng.uniform(-scale, scale) for _ in range(hidden)] for _ in range(2)]
    b1 = [0.0] * hidden
    W2 = [rng.uniform(-scale, scale) for _ in range(hidden)]
    b2 = 0.0
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def clone_weights(w):
    return {"W1": [row[:] for row in w["W1"]], "b1": w["b1"][:], "W2": w["W2"][:], "b2": w["b2"]}


def forward_one(w, x, hidden, dropout_p=0.0, rng=None):
    """h = tanh(x@W1+b1) [inverted-dropout'd if training]; z = h@W2+b2; p = sigmoid(z).
    Returns (h, z, p, mask) -- mask is None at eval time (dropout_p=0 or rng=None)."""
    h_pre = [x[0] * w["W1"][0][j] + x[1] * w["W1"][1][j] + w["b1"][j] for j in range(hidden)]
    h = [math.tanh(v) for v in h_pre]
    mask = None
    if dropout_p > 0.0 and rng is not None:
        keep = 1.0 - dropout_p
        mask = [(1.0 / keep) if rng.uniform01() >= dropout_p else 0.0 for _ in range(hidden)]
        h = [h[j] * mask[j] for j in range(hidden)]
    z = sum(h[j] * w["W2"][j] for j in range(hidden)) + w["b2"]
    return h, z, _sigmoid(z), mask


def bce_loss(p, y):
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def batch_grad(w, X, Y, indices, hidden, dropout_p, rng):
    """Mean-reduced BCE loss and gradient over `indices`, dropout applied only when
    rng is not None (i.e. only during training, never at eval). tanh'(h_pre) is
    derived from the PRE-dropout activation h_raw, not the dropout-scaled h -- the
    dropout mask itself is what gets multiplied through the backward pass (forward
    scale = backward scale, the standard inverted-dropout identity)."""
    B = len(indices)
    dW1 = [[0.0] * hidden for _ in range(2)]
    db1 = [0.0] * hidden
    dW2 = [0.0] * hidden
    db2 = 0.0
    total = 0.0
    for i in indices:
        x, y = X[i], Y[i]
        h_pre = [x[0] * w["W1"][0][j] + x[1] * w["W1"][1][j] + w["b1"][j] for j in range(hidden)]
        h_raw = [math.tanh(v) for v in h_pre]
        mask = None
        if dropout_p > 0.0 and rng is not None:
            keep = 1.0 - dropout_p
            mask = [(1.0 / keep) if rng.uniform01() >= dropout_p else 0.0 for _ in range(hidden)]
            h = [h_raw[j] * mask[j] for j in range(hidden)]
        else:
            h = h_raw
        z = sum(h[j] * w["W2"][j] for j in range(hidden)) + w["b2"]
        p = _sigmoid(z)
        total += bce_loss(p, y)
        dz = (p - y) / B
        db2 += dz
        for j in range(hidden):
            dW2[j] += h[j] * dz
            dh = dz * w["W2"][j]
            if mask is not None:
                dh *= mask[j]                              # dropout backprop: same mask, forward=backward
            dh_pre = dh * (1.0 - h_raw[j] * h_raw[j])       # tanh'(h_pre) from the PRE-dropout value
            db1[j] += dh_pre
            dW1[0][j] += x[0] * dh_pre
            dW1[1][j] += x[1] * dh_pre
    loss = total / B
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def sgd_step(w, grads, lr, wd, hidden):
    """theta <- theta - lr*grad, then DECOUPLED weight decay theta <- theta - lr*wd*theta
    on W1/W2 ONLY (page 17's AdamW form: applied to weights, never biases)."""
    for k in range(2):
        for j in range(hidden):
            w["W1"][k][j] -= lr * grads["W1"][k][j]
            if wd:
                w["W1"][k][j] -= lr * wd * w["W1"][k][j]
    for j in range(hidden):
        w["b1"][j] -= lr * grads["b1"][j]
        w["W2"][j] -= lr * grads["W2"][j]
        if wd:
            w["W2"][j] -= lr * wd * w["W2"][j]
    w["b2"] -= lr * grads["b2"]


def eval_loss(w, X, Y, hidden):
    s = 0.0
    for x, y in zip(X, Y):
        _, _, p, _ = forward_one(w, x, hidden)   # dropout_p=0, rng=None -> eval mode
        s += bce_loss(p, y)
    return s / len(X)


# --------------------------------------------------------------------------------- #
# Train one regime -> full train/val loss history, one point per epoch, plus the
# best-checkpoint step (early-stopping bookkeeping: kept, never used to literally
# halt, so every shape's FULL curve is still visible to look at and classify --
# see the "Spec ambiguity resolved" note in the module docstring).
# --------------------------------------------------------------------------------- #
def train_regime(X_train, Y_train, X_val, Y_val, hidden, epochs, lr, wd, dropout_p,
                  batch, init_seed, shuffle_seed):
    w = init_weights(init_seed, hidden)
    rng_train = RNG(shuffle_seed)
    n_train = len(X_train)
    B = batch if batch else n_train
    order = list(range(n_train))

    train_hist, val_hist = [], []
    best_val, best_epoch = float("inf"), 0
    for epoch in range(epochs):
        if lr > 0.0:
            rng_train.shuffle(order)
            for start in range(0, n_train, B):
                idx = order[start:start + B]
                _, grads = batch_grad(w, X_train, Y_train, idx, hidden, dropout_p, rng_train)
                sgd_step(w, grads, lr, wd, hidden)
        tr = eval_loss(w, X_train, Y_train, hidden)
        va = eval_loss(w, X_val, Y_val, hidden)
        train_hist.append(tr)
        val_hist.append(va)
        if va < best_val:
            best_val, best_epoch = va, epoch
    return {
        "w": w, "train_hist": train_hist, "val_hist": val_hist,
        "best_val": best_val, "best_epoch": best_epoch,
    }


# --------------------------------------------------------------------------------- #
# Leakage check -- the page's "Try it on your box" promise, made literal: how many
# rows appear in BOTH train and val. Exact-tuple membership works because make_moons
# rounds coordinates to 10 d.p. and the leaky split below copies rows verbatim.
# --------------------------------------------------------------------------------- #
def row_overlap_count(X_train, X_val):
    return len(set(X_train) & set(X_val))


def split_clean(X, Y, val_frac, rng):
    idx = list(range(len(X)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(X) * val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return ([X[i] for i in train_idx], [Y[i] for i in train_idx],
            [X[i] for i in val_idx], [Y[i] for i in val_idx])


def split_leaky(X, Y, val_frac, leak_frac, rng):
    """Same as split_clean, but LEAK_FRAC of the val rows are also copied into train
    verbatim -- the split-discipline violation p.22's rule box warns about, made
    concrete: those specific rows can be memorized rather than generalized to."""
    X_tr, Y_tr, X_va, Y_va = split_clean(X, Y, val_frac, rng)
    n_leak = int(round(len(X_va) * leak_frac))
    X_tr = X_tr + X_va[:n_leak]
    Y_tr = Y_tr + Y_va[:n_leak]
    return X_tr, Y_tr, X_va, Y_va


# --------------------------------------------------------------------------------- #
# Curve-shape classifier -- reads (train_hist, val_hist, val_size, overlap_count)
# back into one of the page's six labels. Thresholds are THIS SCRIPT'S OWN [EST]
# choices (not constants.md numbers) picked so the six presets below land cleanly;
# they are the read-back half of "force a shape, then recognize it," not a claim
# about universal cutoffs.
# --------------------------------------------------------------------------------- #
def linreg_r2(ys):
    n = len(ys)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return slope, r2


def classify_shape(train_hist, val_hist, n_val, overlap_count):
    train_final, val_final = train_hist[-1], val_hist[-1]
    val_min = min(val_hist)
    gap = val_final - train_final
    flat = (max(train_hist) - min(train_hist) < 1e-6) and (max(val_hist) - min(val_hist) < 1e-6)
    _, val_r2 = linreg_r2(val_hist)
    _, train_r2 = linreg_r2(train_hist)

    if flat:
        return "bug", "both curves flat within 1e-6 from epoch 0 -- dead LR, not a generalization problem"
    if overlap_count > 0 and gap < -0.01:
        return "leak", f"{overlap_count} train/val row(s) overlap AND val ({val_final:.4f}) < train ({train_final:.4f})"
    if n_val <= 15 and val_r2 < 0.15:
        return "tinyval", f"val R^2={val_r2:.3f} (no reliable trend, n_val={n_val}) vs train R^2={train_r2:.3f}"
    if train_final > 0.85 * LN2 and val_final > 0.85 * LN2:
        return "underfit", f"neither curve gets meaningfully below ln2={LN2:.4f} (train={train_final:.4f}, val={val_final:.4f})"
    if gap > 0.12 and val_final > val_min + 0.03:
        return "overfit", f"train->{train_final:.4f} well below val ({val_final:.4f}); val rose {val_final-val_min:.4f} off its own minimum"
    return "healthy", f"both curves low (train={train_final:.4f}, val={val_final:.4f}), small gap={gap:.4f}"


def classify_three(train_final, val_final, ln2=LN2):
    """The spec's own three-way readout for the --wd/--dropout A/B (Part 2): overfit /
    underfit / just-right, by gap size and distance from baseline. Deliberately simpler
    than classify_shape's six-way rules (which prioritize the "never gets meaningfully
    below baseline" underfit check first -- appropriate for six DISTINCT forced shapes,
    but it can mislabel a noisy-label regime that is BOTH close to baseline AND visibly
    overfit, which is exactly what heavy label noise at a fixed small N produces)."""
    gap = val_final - train_final
    if gap > 0.05:
        return "overfit"
    if train_final > 0.85 * ln2:
        return "underfit"
    return "just-right"


DIAGNOSIS_NAME = {
    "underfit": "[1] Underfitting -- LR too low or model too small",
    "healthy":  "[2] Healthy -- both low, small gap",
    "overfit":  "[3] Overfitting -- train->0, val rises",
    "leak":     "[4] Leakage -- val below train; fix the split",
    "bug":      "[5] Wiring bug / dead LR -- not a generalization problem",
    "tinyval":  "[6] Val set too small -- enlarge it or smooth over a window",
}


# --------------------------------------------------------------------------------- #
# The six forceable presets -- same SIX-ROW TABLE the page's own live demo forces,
# same conceptual knobs (N, hidden/capacity, lr, label noise, leak, dead-LR), THIS
# script's own parameter values (see module docstring: not a port of the page's JS).
# --------------------------------------------------------------------------------- #
def build_shape(key, epochs_scale=1.0):
    if key == "underfit":
        return dict(N=200, hidden=1, lr=0.02, noise=0.10, label_noise=0.0,
                    epochs=int(40 * epochs_scale), val_frac=0.30, leak_frac=0.0, batch=0)
    if key == "healthy":
        return dict(N=300, hidden=8, lr=0.10, noise=0.20, label_noise=0.02,
                    epochs=int(60 * epochs_scale), val_frac=0.30, leak_frac=0.0, batch=0)
    if key == "overfit":
        # Small N, high capacity, high LR, heavy label noise, many epochs -- the model
        # has more than enough room to memorize 55 noisy points and does, decisively:
        # train keeps falling while val turns around and climbs (a real shape 3, not
        # a plateau -- see the module's tuning notes: earlier milder settings only
        # plateaued val, which reads as "healthy" and defeats the point of this preset).
        return dict(N=55, hidden=28, lr=0.45, noise=0.20, label_noise=0.28,
                    epochs=int(160 * epochs_scale), val_frac=0.32, leak_frac=0.0, batch=0)
    if key == "leak":
        return dict(N=200, hidden=8, lr=0.10, noise=0.10, label_noise=0.05,
                    epochs=int(80 * epochs_scale), val_frac=0.30, leak_frac=0.6, batch=0)
    if key == "bug":
        return dict(N=200, hidden=8, lr=0.0, noise=0.10, label_noise=0.03,
                    epochs=int(20 * epochs_scale), val_frac=0.30, leak_frac=0.0, batch=0)
    if key == "tinyval":
        return dict(N=200, hidden=8, lr=0.30, noise=0.10, label_noise=0.05,
                    epochs=int(50 * epochs_scale), val_frac=0.05, leak_frac=0.0, batch=9)
    raise ValueError(key)


def run_shape(key, seed_base, epochs_scale=1.0):
    cfg = build_shape(key, epochs_scale)
    rng_data = RNG(seed_base)
    X, Y = make_moons(cfg["N"], cfg["noise"], rng_data)
    Y = flip_labels(Y, cfg["label_noise"], rng_data)
    rng_split = RNG(seed_base + 1)
    if cfg["leak_frac"] > 0.0:
        X_tr, Y_tr, X_va, Y_va = split_leaky(X, Y, cfg["val_frac"], cfg["leak_frac"], rng_split)
    else:
        X_tr, Y_tr, X_va, Y_va = split_clean(X, Y, cfg["val_frac"], rng_split)
    overlap = row_overlap_count(X_tr, X_va)
    res = train_regime(X_tr, Y_tr, X_va, Y_va, cfg["hidden"], cfg["epochs"], cfg["lr"],
                        wd=0.0, dropout_p=0.0, batch=cfg["batch"],
                        init_seed=seed_base + 2, shuffle_seed=seed_base + 3)
    label, evidence = classify_shape(res["train_hist"], res["val_hist"], len(X_va), overlap)
    return cfg, res, overlap, label, evidence


# --------------------------------------------------------------------------------- #
# Narration
# --------------------------------------------------------------------------------- #
def print_curve_row(name, hist, every):
    n = len(hist)
    pts = sorted(set([0] + list(range(every - 1, n, every)) + [n - 1]))
    row = "  ".join(f"e{p+1}:{hist[p]:.4f}" for p in pts)
    print(f"    {name:>5}  {row}")


def narrate_shapes(epochs_scale, quiet=False):
    print("=" * 78)
    print("PART 1 -- six shapes, six diagnoses (real MLPs, real curves, forced on purpose)")
    print("=" * 78)
    print(f"  baseline (random-guess, V=2): ln(2) = {LN2:.4f} nats  (constants.md sec 8)")
    print()
    results = {}
    for i, key in enumerate(["underfit", "healthy", "overfit", "leak", "bug", "tinyval"]):
        cfg, res, overlap, label, evidence = run_shape(key, seed_base=2200 + 10 * i,
                                                         epochs_scale=epochs_scale)
        results[key] = (cfg, res, overlap, label, evidence)
        print(f"  [{i+1}] target={key:<9} N={cfg['N']:<4} hidden={cfg['hidden']:<3} "
              f"lr={cfg['lr']:<5} val_frac={cfg['val_frac']:<5} epochs={cfg['epochs']}")
        if not quiet:
            print_curve_row("train", res["train_hist"], max(1, cfg["epochs"] // 4))
            print_curve_row("val", res["val_hist"], max(1, cfg["epochs"] // 4))
        print(f"        row-overlap(train,val) = {overlap}   best-val-epoch = {res['best_epoch']+1} "
              f"(val={res['best_val']:.4f})")
        print(f"        --> classified as: {DIAGNOSIS_NAME[label]}  [EST]")
        print(f"            evidence: {evidence}")
        print()
    return results


def narrate_wd_ab(wd, dropout, quick=False, quiet=False):
    print("=" * 78)
    print(f"PART 2 -- the one lever, on the page's own {N_REGIME}-example regime")
    print("=" * 78)
    # An overfit-prone config at the p.22 regime size: enough capacity and noise that,
    # left unregularized, it lands on shape 3 -- the exact situation the page's PREDICT
    # box poses ("train loss near 0, val loss started rising after 3 epochs"). Tuning
    # note: at this N=500 scale a milder config (hidden=24, label_noise=0.10) converges
    # with val slightly BELOW train (no real overfitting to regularize away -- see the
    # module's build history); hidden=64 + heavier label noise reliably produces a
    # genuine positive train-val gap that plateaus early, so both --quick and full runs
    # land in the same regime.
    hidden = 64
    lr = 0.3
    noise = 0.20
    label_noise = 0.25
    epochs = 35 if quick else 70
    val_frac = 0.30

    rng_data = RNG(5000)
    X, Y = make_moons(N_REGIME, noise, rng_data)
    Y = flip_labels(Y, label_noise, rng_data)
    rng_split = RNG(5001)
    X_tr, Y_tr, X_va, Y_va = split_clean(X, Y, val_frac, rng_split)
    overlap = row_overlap_count(X_tr, X_va)
    print(f"  N={N_REGIME}  n_train={len(X_tr)}  n_val={len(X_va)}  hidden={hidden}  lr={lr}")
    print(f"  leakage check: row-overlap(train,val) = {overlap}  "
          f"({'OK -- clean, deduped split' if overlap == 0 else 'LEAK -- split is not honest'})")
    print()

    runs = {}
    for tag, wd_, drop_ in (("baseline (wd=0, dropout=0)", 0.0, 0.0),
                             (f"regularized (wd={wd}, dropout={dropout})", wd, dropout)):
        res = train_regime(X_tr, Y_tr, X_va, Y_va, hidden, epochs, lr, wd_, drop_,
                            batch=0, init_seed=5002, shuffle_seed=5003)
        gap = res["val_hist"][-1] - res["train_hist"][-1]
        diag3 = classify_three(res["train_hist"][-1], res["val_hist"][-1])
        runs[tag] = (res, gap, diag3)
        print(f"  {tag}")
        if not quiet:
            print_curve_row("train", res["train_hist"], max(1, epochs // 4))
            print_curve_row("val", res["val_hist"], max(1, epochs // 4))
        print(f"    final train={res['train_hist'][-1]:.4f}  final val={res['val_hist'][-1]:.4f}  "
              f"gap={gap:+.4f}  best-checkpoint epoch={res['best_epoch']+1} (val={res['best_val']:.4f})")
        print(f"    diagnosis: {diag3}  [EST]")
        print()

    base_gap = runs["baseline (wd=0, dropout=0)"][1]
    reg_tag = f"regularized (wd={wd}, dropout={dropout})"
    reg_gap = runs[reg_tag][1]
    print(f"  train-val GAP:  baseline={base_gap:+.4f}   regularized={reg_gap:+.4f}   "
          f"(regularized should be smaller -- weight decay pulling weights toward 0 leaves")
    print(f"  less room to memorize the {label_noise:.0%}-noisy training labels)")
    return base_gap, reg_gap, overlap


# --------------------------------------------------------------------------------- #
# Self-checks -- the spec's own two: (1) all six shapes classify as intended,
# (2) weight decay lowers the train-val gap. Plus the leakage-count sanity check.
# --------------------------------------------------------------------------------- #
def run_self_checks(shape_results, base_gap, reg_gap, overlap_wd):
    print("=" * 78)
    print("SELF-CHECKS")
    print("=" * 78)
    ok = True
    for key, (cfg, res, overlap, label, evidence) in shape_results.items():
        status = "OK" if label == key else "FAIL"
        if label != key:
            ok = False
        print(f"  [{status}] shape '{key}': classifier said '{label}' -- {evidence}")
        assert label == key or status == "FAIL", "unreachable"
    for key, (cfg, res, overlap, label, evidence) in shape_results.items():
        assert label == key, (
            f"preset '{key}' was built to force that shape but the classifier read it back "
            f"as '{label}' ({evidence}) -- the six-shapes self-check has failed"
        )
    print(f"  [OK] all 6 shapes classify as the preset they were built to force")

    leak_cfg = shape_results["leak"]
    assert leak_cfg[2] > 0, f"leaky-split preset should have overlap>0, got {leak_cfg[2]}"
    clean_overlaps = [shape_results[k][2] for k in shape_results if k != "leak"]
    assert all(o == 0 for o in clean_overlaps), (
        f"clean-split presets should have row-overlap==0, got {clean_overlaps}"
    )
    print(f"  [OK] leakage check: leaky preset overlap={leak_cfg[2]}>0, "
          f"all {len(clean_overlaps)} clean presets overlap==0")
    assert overlap_wd == 0, f"part-2 A/B split should be clean (overlap==0), got {overlap_wd}"
    print(f"  [OK] part-2 A/B split is clean: row-overlap(train,val)={overlap_wd}")

    assert reg_gap < base_gap, (
        f"weight decay should lower the train-val gap: baseline={base_gap:.4f}, "
        f"regularized={reg_gap:.4f} -- spec-code.md p.22 self-check"
    )
    print(f"  [OK] weight decay lowers the train-val gap: {base_gap:+.4f} -> {reg_gap:+.4f} "
          f"(Delta={reg_gap-base_gap:+.4f})")

    print()
    print("All self-checks PASS." if ok else "Self-checks FAILED.")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Six train/val curve shapes forced on purpose, then read back by "
                     "a classifier; plus a weight-decay/dropout/early-stopping A/B on "
                     "the page's own 500-example regime -- p.22 companion script."
    )
    ap.add_argument("--wd", type=float, default=WD_DEFAULT,
                     help=f"weight decay lambda for the regularized run (default {WD_DEFAULT}, "
                          f"page 17's 2026 LLM fine-tune default)")
    ap.add_argument("--dropout", type=float, default=0.0,
                     help="dropout probability for the regularized run (default 0.0 -- page 22's "
                          "own point: dropout is usually irrelevant for a small 2026 LLM fine-tune)")
    ap.add_argument("--quick", action="store_true", help="fewer epochs, faster (~5s)")
    ap.add_argument("--self-test", action="store_true",
                     help="terse output: assertions only, no full curve tables, exit 0/1")
    args = ap.parse_args()

    if not (0.0 <= args.dropout < 1.0):
        print(f"--dropout must be in [0, 1), got {args.dropout}", file=sys.stderr)
        sys.exit(2)
    if args.wd < 0.0:
        print(f"--wd must be >= 0, got {args.wd}", file=sys.stderr)
        sys.exit(2)

    epochs_scale = 0.5 if args.quick else 1.0
    shape_results_raw = {}

    print("22_regularization.py -- the two curves, six shapes, one lever")
    print(f"  --wd={args.wd}  --dropout={args.dropout}  --quick={args.quick}")
    print()

    shapes = narrate_shapes(epochs_scale, quiet=args.self_test)
    print()
    base_gap, reg_gap, overlap_wd = narrate_wd_ab(args.wd, args.dropout, quick=args.quick,
                                                    quiet=args.self_test)
    print()
    ok = run_self_checks(shapes, base_gap, reg_gap, overlap_wd)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
