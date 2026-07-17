#!/usr/bin/env python3
"""
16_batch_regimes.py -- one MLP, one update rule, four batch sizes.

Course artifact for p.16 (Batch, Mini-Batch, Stochastic -- and Why Noise Is a Feature).
The page's live demo trains a real MLP([2,16,1]) on `moons` (256 points) with NN.Trainer
and Adam, dragging a batch-size slider through B in {1, 8, 32, 256(=full)} -- see the
page's JS: `const N = 256; const BATCH_VALS = [1, 8, 32, 256]; const EPOCH_BUDGET = 10;
const THRESHOLD = 0.4;`. This script is that same experiment, ported to dependency-free
Python so it runs on any machine with python3 -- no torch, no numpy, no GPU:

  - The SAME update rule as page 13/14/15: theta <- theta - eta*g_k. Only what g_k is
    averaged over changes (page's own framing, "Three regimes, one update rule").
  - The SAME MLP architecture (2 -> 16 tanh -> 1, BCE-with-logits), hand-derived forward
    and backward -- see xor_by_hand.py for the same technique at hidden=8; this is the
    hidden=16 / tanh version the page's demo actually uses.
  - The SAME training budget: EPOCH_BUDGET=10 epochs of data, so B=1 gets far more
    (noisier) updates than B=full gets exact ones -- reproduced verbatim from the page.
  - The page's ".box try" text promises three things and this script delivers all three:
      1. steps-to-a-target-loss and final validation loss, per B  (section "REGIMES")
      2. the mean-vs-sum effective-learning-rate effect from the worked example
         (section "MEAN VS SUM")
      3. [spec-code.md addition, not in the page prose] a direct 1/B gradient-variance
         readout, empirically confirming Var[g_B] ~ Var[g_1]/B (section "1/B VARIANCE")

Spec ambiguity resolved: the page's ".box try" text promises a "final validation loss"
but the live demo has no held-out split (it trains and reads loss on all 256 points).
This script adds a genuine held-out split -- N_TRAIN=224 / N_VAL=32 -- so "final
validation loss" means something a held-out set actually earns. "B=full" here means
"the full TRAINING set" (224), not the page's literal N=256.

Usage
-----
    python 16_batch_regimes.py               # narrated run, all four batch regimes
    python 16_batch_regimes.py --quick        # fewer bootstrap resamples, same training
    python 16_batch_regimes.py --self-test    # assertions only, no narration, exit 0/1

SAFETY: pure Python arithmetic (no imports beyond stdlib math/argparse). No GPU, no
network, no files touched. Runtime ~2s (this artifact's spec-code.md budget).
"""

import argparse
import math

# --------------------------------------------------------------------------------- #
# Page-demo constants, reproduced verbatim from 16-minibatch-and-noise.html's JS
# (`const N = 256; const BATCH_VALS = [1, 8, 32, 256]; const EPOCH_BUDGET = 10;
# const THRESHOLD = 0.4;`) -- these are the page's OWN numbers, not constants.md
# frozen values (this is a toy training demo, not a hardware/model-size fact).
# --------------------------------------------------------------------------------- #
N = 256                       # page: const N = 256 (moons dataset size)
N_VAL = 32                    # this script's addition -- a real held-out split
N_TRAIN = N - N_VAL           # = 224
BATCH_LABELS = ["1", "8", "32", "full"]
BATCH_VALS = [1, 8, 32, N_TRAIN]     # page: [1, 8, 32, 256] -- "full" here is full-TRAIN
EPOCH_BUDGET = 10             # page: const EPOCH_BUDGET = 10
THRESHOLD = 0.4               # page: const THRESHOLD = 0.4 (below ln(2)=0.693 baseline)
LN2 = math.log(2.0)

HIDDEN = 16                   # page: MLP([2, 16, 1]) -- tanh hidden, BCE-with-logits out
INIT_SCALE = 0.5              # matches xor_by_hand.py's init convention (next(rng)*0.5)
LR = 0.5                      # this script's own choice: plain SGD, tuned so full-batch
                               # (only 10 steps -- EPOCH_BUDGET * ceil(224/224)) is a clean
                               # monotone descent (small-step descent-lemma regime) while
                               # still moving the smaller-B regimes well below THRESHOLD.
SEED_DATA = 1601               # p.16, "01" -- dataset + initial-weights seed
SEED_SHUFFLE_BASE = 70000      # per-regime shuffle-order seeds start here
N_RESAMPLES = 800              # bootstrap draws per B in the 1/B variance readout


# --------------------------------------------------------------------------------- #
# Tiny deterministic RNG -- same glibc-style LCG as xor_by_hand.py's `_lcg`, wrapped in
# a class so dataset generation, weight init, batch shuffling, and bootstrap resampling
# all share one dependency-free source of randomness. No numpy, no `random` module
# state shared with anything else -- bit-reproducible on any python3.
# --------------------------------------------------------------------------------- #
class RNG:
    def __init__(self, seed):
        self.state = seed & 0x7FFFFFFF

    def _next(self):
        self.state = (1103515245 * self.state + 12345) % (2 ** 31)
        return self.state

    def uniform01(self):
        """Uniform in [0, 1)."""
        return self._next() / 2 ** 31

    def uniform(self, lo=-1.0, hi=1.0):
        return lo + (hi - lo) * self.uniform01()

    def randint(self, n):
        """Uniform integer in [0, n)."""
        return int(self.uniform01() * n) % n

    def shuffle(self, lst):
        """In-place Fisher-Yates, using this RNG."""
        for i in range(len(lst) - 1, 0, -1):
            j = self.randint(i + 1)
            lst[i], lst[j] = lst[j], lst[i]


# --------------------------------------------------------------------------------- #
# Dataset -- a two-moons-shaped binary classification set, pure Python. Not the page's
# literal JS RNG stream (porting that bit-for-bit buys nothing the spec asks for); same
# SHAPE of problem (two interleaved arcs, additive noise, N=256), same architecture
# trained on it, which is what "the same MLP at B in {1,8,32,full}" requires.
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
        X.append((x, y))
        Y.append(float(cls))
    return X, Y


# --------------------------------------------------------------------------------- #
# MLP([2, 16, 1]) -- tanh hidden, BCE-with-logits output. Hand-derived forward/backward,
# same technique as xor_by_hand.py's --self-test path, at the page's actual hidden=16.
# --------------------------------------------------------------------------------- #
def _sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def init_weights(seed):
    rng = RNG(seed)
    W1 = [[rng.uniform(-INIT_SCALE, INIT_SCALE) for _ in range(HIDDEN)] for _ in range(2)]
    b1 = [0.0] * HIDDEN
    W2 = [rng.uniform(-INIT_SCALE, INIT_SCALE) for _ in range(HIDDEN)]
    b2 = 0.0
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def clone_weights(w):
    return {"W1": [row[:] for row in w["W1"]], "b1": w["b1"][:], "W2": w["W2"][:], "b2": w["b2"]}


def forward_one(w, x):
    """h = tanh(x @ W1 + b1);  z = h @ W2 + b2;  p = sigmoid(z)."""
    h_pre = [x[0] * w["W1"][0][j] + x[1] * w["W1"][1][j] + w["b1"][j] for j in range(HIDDEN)]
    h = [math.tanh(v) for v in h_pre]
    z = sum(h[j] * w["W2"][j] for j in range(HIDDEN)) + w["b2"]
    return h, z, _sigmoid(z)


def bce_loss(p, y):
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def batch_loss_and_grad(w, X, Y, indices, reduction="mean"):
    """J (or S) and its gradient over `indices`. reduction='mean' is the CORRECT,
    page-13-consistent choice (J = (1/B) sum L_b); reduction='sum' is the page's
    "summed-loss trap" (S = sum L_b) -- kept as a real, independently-computed second
    code path (not `mean_result * B` by algebra) so the MEAN-VS-SUM check below is a
    genuine cross-check, not a tautology."""
    B = len(indices)
    denom = float(B) if reduction == "mean" else 1.0
    dW1 = [[0.0] * HIDDEN for _ in range(2)]
    db1 = [0.0] * HIDDEN
    dW2 = [0.0] * HIDDEN
    db2 = 0.0
    total = 0.0
    for i in indices:
        x, y = X[i], Y[i]
        h, z, p = forward_one(w, x)
        total += bce_loss(p, y)
        dz = (p - y) / denom              # dL/dz for one example, scaled by the reduction
        db2 += dz
        for j in range(HIDDEN):
            dW2[j] += h[j] * dz
            dh_pre = dz * w["W2"][j] * (1.0 - h[j] * h[j])   # tanh'(h_pre) = 1 - tanh^2
            db1[j] += dh_pre
            dW1[0][j] += x[0] * dh_pre
            dW1[1][j] += x[1] * dh_pre
    loss = total / denom
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def sgd_step(w, grads, lr):
    for k in range(2):
        for j in range(HIDDEN):
            w["W1"][k][j] -= lr * grads["W1"][k][j]
    for j in range(HIDDEN):
        w["b1"][j] -= lr * grads["b1"][j]
        w["W2"][j] -= lr * grads["W2"][j]
    w["b2"] -= lr * grads["b2"]


def eval_loss(w, X, Y):
    s = 0.0
    for x, y in zip(X, Y):
        _, _, p = forward_one(w, x)
        s += bce_loss(p, y)
    return s / len(X)


# --------------------------------------------------------------------------------- #
# Train one batch-size regime, from the SAME shared initial weights (page's own
# framing: "the model always starts from the same initial weights, so the comparison
# is fair"), for EPOCH_BUDGET epochs of data.
# --------------------------------------------------------------------------------- #
def train_regime(B, init_w, X_train, Y_train, X_val, Y_val, epoch_budget, lr, shuffle_seed):
    w = clone_weights(init_w)
    rng = RNG(shuffle_seed)
    steps_per_epoch = math.ceil(N_TRAIN / B)
    total_steps = epoch_budget * steps_per_epoch
    eval_every = max(1, steps_per_epoch // 4)

    order = list(range(N_TRAIN))
    history = []                # mini-batch J, every step -- the noisy signal on the page
    threshold_step = None
    last_full_loss = eval_loss(w, X_train, Y_train)

    for step in range(total_steps):
        pos = step % steps_per_epoch
        if pos == 0:
            rng.shuffle(order)
        batch_idx = order[pos * B: pos * B + B]
        J, grads = batch_loss_and_grad(w, X_train, Y_train, batch_idx, reduction="mean")
        sgd_step(w, grads, lr)
        history.append(J)

        if (step + 1) % eval_every == 0 or step == total_steps - 1:
            last_full_loss = eval_loss(w, X_train, Y_train)
            if threshold_step is None and last_full_loss < THRESHOLD:
                threshold_step = step + 1

    val_loss = eval_loss(w, X_val, Y_val)
    return {
        "B": B, "history": history, "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps, "threshold_step": threshold_step,
        "final_train_loss": last_full_loss, "val_loss": val_loss,
    }


# --------------------------------------------------------------------------------- #
# 1/B gradient-variance readout. Fix the weights (the SHARED initial weights -- before
# any regime has trained), so this is a pure sampling-noise measurement, independent of
# training dynamics. For one example, dJ/db2 = (p - y) exactly (batch B=1's own
# gradient contribution to the output bias -- see batch_loss_and_grad above); for a
# batch of B examples averaged, it is the MEAN of B such residuals. Bootstrap-resample
# batches of size B from the fixed residual population and measure Var[mean-of-B]: for
# B iid draws, Var[mean] = Var[single]/B -- textbook, and this is the empirical check.
# --------------------------------------------------------------------------------- #
def residuals_at_weights(w, X, Y):
    return [forward_one(w, x)[2] - y for x, y in zip(X, Y)]


def bootstrap_batch_variance(residuals, B, resamples, rng):
    n = len(residuals)
    means = []
    for _ in range(resamples):
        s = 0.0
        for _ in range(B):
            s += residuals[rng.randint(n)]
        means.append(s / B)
    mu = sum(means) / len(means)
    var = sum((m - mu) ** 2 for m in means) / (len(means) - 1)
    return var


# --------------------------------------------------------------------------------- #
# Narration
# --------------------------------------------------------------------------------- #
def narrate(epoch_budget, lr, resamples, quiet_asserts=False):
    print("=" * 74)
    print("16_batch_regimes.py -- same MLP([2,16,1]), same update rule, four batch sizes")
    print("=" * 74)
    print(f"  N={N}  (N_train={N_TRAIN}, N_val={N_VAL})   B in {BATCH_LABELS} -> {BATCH_VALS}")
    print(f"  epoch_budget={epoch_budget}   lr={lr}   (page's own constants: N=256, "
          f"BATCH_VALS=[1,8,32,256], EPOCH_BUDGET=10, THRESHOLD={THRESHOLD})")
    print()

    rng_data = RNG(SEED_DATA)
    X_all, Y_all = make_moons(N, noise=0.2, rng=rng_data)
    idx_all = list(range(N))
    rng_data.shuffle(idx_all)   # moons is class-sorted by construction; shuffle before splitting
    val_idx = idx_all[:N_VAL]
    train_idx = idx_all[N_VAL:]
    X_train = [X_all[i] for i in train_idx]
    Y_train = [Y_all[i] for i in train_idx]
    X_val = [X_all[i] for i in val_idx]
    Y_val = [Y_all[i] for i in val_idx]

    init_w = init_weights(SEED_DATA + 1)   # shared starting point for every regime -- fair comparison
    init_loss = eval_loss(init_w, X_train, Y_train)
    print(f"  shared initial weights -> train loss at step 0 = {init_loss:.4f}  "
          f"(ln(2)={LN2:.4f} is the 50/50-guess baseline)")

    # ---- REGIMES: train B in {1, 8, 32, full}, same init, same budget --------------
    print()
    print("-" * 74)
    print("REGIMES -- B in {1, 8, 32, full}, EPOCH_BUDGET=%d epochs of data each" % epoch_budget)
    print("-" * 74)
    results = {}
    for label, B in zip(BATCH_LABELS, BATCH_VALS):
        r = train_regime(B, init_w, X_train, Y_train, X_val, Y_val, epoch_budget, lr,
                          shuffle_seed=SEED_SHUFFLE_BASE + B)
        results[label] = r
        th = f"step {r['threshold_step']}" if r["threshold_step"] is not None else "never (in budget)"
        print(f"  B={label:>4}  steps/epoch={r['steps_per_epoch']:>4}  total_steps={r['total_steps']:>5}  "
              f"first full-train-loss<{THRESHOLD}: {th:>16}  "
              f"final_train={r['final_train_loss']:.4f}  final_val={r['val_loss']:.4f}")
    print()
    print("  Read this the way the page frames it: B=1 gets 224x more updates than B=full")
    print("  (2240 vs 10 steps) for the SAME pass over the data -- noisier each, but far more")
    print("  of them. 'More, cheaper, slightly-wrong steps' -- throughput mechanism #3.")

    # ---- 1/B VARIANCE: the gradient-noise law, measured -----------------------------
    print()
    print("-" * 74)
    print("1/B VARIANCE -- Var[batch-mean gradient] should scale as 1/B (fixed weights)")
    print("-" * 74)
    residuals = residuals_at_weights(init_w, X_train, Y_train)
    rng_boot = RNG(SEED_DATA + 999)
    var_table = {}
    for B in (1, 8, 32):
        v = bootstrap_batch_variance(residuals, B, resamples, rng_boot)
        var_table[B] = v
        print(f"  B={B:>3}   Var[mean-of-{B} residuals], {resamples} bootstrap draws = {v:.6f}   "
              f"B*Var = {B*v:.6f}")
    print(f"  B=full  Var = 0.000000 exactly (no sampling randomness -- it's the same all-"
          f"{N_TRAIN} examples every single step; full batch has NO gradient noise, by definition)")
    print(f"  1/B law check: B*Var(B) should hold roughly constant across B -- it does, up to")
    print(f"  bootstrap sampling noise (see self-test tolerance below).")

    # ---- MEAN VS SUM: the effective-learning-rate bug, reproduced -------------------
    print()
    print("-" * 74)
    print("MEAN VS SUM -- the page's worked example, recomputed via two independent reductions")
    print("-" * 74)
    demo_idx = list(range(32))   # a real B=32 slice of the training set
    J_mean, g_mean = batch_loss_and_grad(init_w, X_train, Y_train, demo_idx, reduction="mean")
    S_sum, g_sum = batch_loss_and_grad(init_w, X_train, Y_train, demo_idx, reduction="sum")
    eta = 0.001
    eta_eff = eta * len(demo_idx)
    print(f"  B=32 slice: J (mean-reduced) = {J_mean:.6f}    S (sum-reduced) = {S_sum:.6f}")
    print(f"  S / J = {S_sum / J_mean:.4f}  (should be exactly B=32)")
    print(f"  grad_S[b2] / grad_J[b2] = {g_sum['b2'] / g_mean['b2']:.4f}  (should be exactly B=32)")
    print(f"  intend eta=0.001, sum instead of mean -> eta_eff = 0.001 x 32 = {eta_eff:.3f}")
    print(f"  ({eta_eff/eta:.0f}x too large -- exactly the page's worked-example number)")

    return results, var_table


# --------------------------------------------------------------------------------- #
# Self-test -- the two spec-mandated assertions (full-batch monotone, B=1 noisy) plus
# the 1/B variance law and the mean-vs-sum identity. Pure Python, no GPU, runs anywhere.
# --------------------------------------------------------------------------------- #
def self_test(epoch_budget=EPOCH_BUDGET, lr=LR, resamples=N_RESAMPLES):
    print("Running self-checks (no GPU, no display, pure Python arithmetic)...")

    rng_data = RNG(SEED_DATA)
    X_all, Y_all = make_moons(N, noise=0.2, rng=rng_data)
    idx_all = list(range(N))
    rng_data.shuffle(idx_all)
    val_idx = idx_all[:N_VAL]
    train_idx = idx_all[N_VAL:]
    X_train = [X_all[i] for i in train_idx]
    Y_train = [Y_all[i] for i in train_idx]
    X_val = [X_all[i] for i in val_idx]
    Y_val = [Y_all[i] for i in val_idx]
    assert len(X_train) == N_TRAIN and len(X_val) == N_VAL
    assert N_TRAIN == 224 and N_VAL == 32 and N_TRAIN + N_VAL == N == 256

    init_w = init_weights(SEED_DATA + 1)

    # --- spec check #1: full-batch loss trajectory is MONOTONE (deterministic GD) ---
    full_B = N_TRAIN
    r_full = train_regime(full_B, init_w, X_train, Y_train, X_val, Y_val, epoch_budget, lr,
                           shuffle_seed=SEED_SHUFFLE_BASE + full_B)
    hist = r_full["history"]
    assert len(hist) == epoch_budget, (
        f"full batch should take exactly epoch_budget={epoch_budget} steps "
        f"(steps_per_epoch=ceil({N_TRAIN}/{full_B})=1), got {len(hist)}"
    )
    non_increasing = all(hist[k + 1] <= hist[k] + 1e-9 for k in range(len(hist) - 1))
    assert non_increasing, (
        f"full-batch (deterministic) loss trajectory should be monotone non-increasing, "
        f"got {['%.5f' % v for v in hist]}"
    )
    print(f"  [OK] B=full ({full_B}): {epoch_budget}-step loss trajectory is monotone "
          f"non-increasing -- {hist[0]:.4f} -> {hist[-1]:.4f} (exact gradient, zero noise)")

    # --- spec check #2: B=1 loss trajectory is NOISY (non-monotone, real up-steps) ---
    r_b1 = train_regime(1, init_w, X_train, Y_train, X_val, Y_val, epoch_budget, lr,
                         shuffle_seed=SEED_SHUFFLE_BASE + 1)
    hist1 = r_b1["history"]
    assert len(hist1) == epoch_budget * N_TRAIN, "B=1 should take epoch_budget*N_train steps"
    ups = sum(1 for k in range(len(hist1) - 1) if hist1[k + 1] > hist1[k])
    frac_up = ups / (len(hist1) - 1)
    assert frac_up > 0.20, (
        f"B=1 should show real step-to-step noise (a meaningful fraction of up-steps), "
        f"got only {frac_up:.1%} of {len(hist1)-1} steps increasing -- too smooth to call noisy"
    )
    print(f"  [OK] B=1: {frac_up:.1%} of {len(hist1)-1} steps increase step-to-step -- "
          f"noisy, not monotone (single-example estimate of the true gradient)")

    # --- spec-code.md addition: 1/B gradient-variance readout ------------------------
    residuals = residuals_at_weights(init_w, X_train, Y_train)
    assert len(residuals) == N_TRAIN
    rng_boot = RNG(SEED_DATA + 999)
    var1 = bootstrap_batch_variance(residuals, 1, resamples, rng_boot)
    var8 = bootstrap_batch_variance(residuals, 8, resamples, rng_boot)
    var32 = bootstrap_batch_variance(residuals, 32, resamples, rng_boot)
    assert var1 > 0.0, "B=1 residual variance should be strictly positive (real data spread)"
    # B*Var(B) should be roughly constant (Var[mean of B iid draws] = Var[single]/B).
    # Bootstrap estimates carry their own sampling noise, so the tolerance is generous
    # (0.5x-2x band) -- a real violation of the 1/B law would be off by a much larger
    # factor (e.g. Var flat in B, or scaling as 1/B^2), not a bootstrap wobble.
    for B, varB in ((8, var8), (32, var32)):
        ratio = (B * varB) / var1
        assert 0.5 < ratio < 2.0, (
            f"1/B law check failed at B={B}: B*Var(B)/Var(1) = {ratio:.3f}, expected ~1.0 "
            f"(within bootstrap noise) if Var scales as 1/B"
        )
    print(f"  [OK] 1/B variance law: Var(1)={var1:.5f}  8*Var(8)={8*var8:.5f} "
          f"(ratio {8*var8/var1:.2f})  32*Var(32)={32*var32:.5f} (ratio {32*var32/var1:.2f}) "
          f"-- all within [0.5, 2.0]x of Var(1)")

    # --- page's mean-vs-sum worked example: eta=0.001, B=32 -> eta_eff=0.032 --------
    demo_idx = list(range(32))
    J_mean, g_mean = batch_loss_and_grad(init_w, X_train, Y_train, demo_idx, reduction="mean")
    S_sum, g_sum = batch_loss_and_grad(init_w, X_train, Y_train, demo_idx, reduction="sum")
    assert abs(S_sum - 32.0 * J_mean) < 1e-9, (
        f"summed loss should equal exactly B*mean loss: S={S_sum}, 32*J={32*J_mean}"
    )
    assert abs(g_sum["b2"] - 32.0 * g_mean["b2"]) < 1e-9, (
        f"summed gradient should equal exactly B*mean gradient (b2): "
        f"{g_sum['b2']} vs {32*g_mean['b2']}"
    )
    eta, B = 0.001, 32
    eta_eff = eta * B
    assert abs(eta_eff - 0.032) < 1e-12, f"eta_eff should be exactly 0.032, got {eta_eff}"
    print(f"  [OK] mean-vs-sum: S == 32*J exactly (two independent reductions agree); "
          f"eta=0.001, B=32 -> eta_eff={eta_eff:.3f} == the page's worked-example 32x bug")

    print("All self-checks PASS.")


def main():
    ap = argparse.ArgumentParser(
        description="Same MLP([2,16,1]) at B in {1,8,32,full}; 1/B gradient-variance "
                     "readout; asserts full-batch monotone, B=1 noisy -- p.16 companion script."
    )
    ap.add_argument("--epochs", type=int, default=EPOCH_BUDGET,
                     help=f"epoch budget per regime (default {EPOCH_BUDGET}, the page's own)")
    ap.add_argument("--lr", type=float, default=LR, help=f"SGD learning rate (default {LR})")
    ap.add_argument("--resamples", type=int, default=N_RESAMPLES,
                     help=f"bootstrap draws per B in the 1/B variance readout (default {N_RESAMPLES})")
    ap.add_argument("--quick", action="store_true",
                     help="fewer bootstrap resamples (150) for a faster narrated run")
    ap.add_argument("--self-test", action="store_true",
                     help="run assertions only, no narration, exit 0 on success")
    args = ap.parse_args()

    resamples = 150 if args.quick else args.resamples

    self_test(epoch_budget=args.epochs, lr=args.lr, resamples=max(resamples, 200))
    if args.self_test:
        return
    print()
    narrate(args.epochs, args.lr, resamples)


if __name__ == "__main__":
    main()
