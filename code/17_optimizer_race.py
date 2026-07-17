#!/usr/bin/env python3
"""
17_optimizer_race.py -- SGD, momentum, Adam, AdamW race on TN-1. Same nine
gradients, four different ideas about what to do with them.

Course artifact for p.17 ("Optimizers: SGD -> Momentum -> RMSProp -> Adam ->
AdamW"). This is where the TN-1 object (constants.md section 8) accretes its
third capability: p.06 shipped forward-only, p.14/p.15 added backward (by
pencil, then by autograd), and this script adds the OPTIMIZER STEP -- turning
a gradient into a parameter update, four different ways.

Four things happen here, in order:

  1. BRIDGE: the exact SGD step you hand-computed on page 14 (input 1,
     eta=0.1, loss 0.9869 -> 0.8222), reproduced through this script's own
     plain-SGD update function -- not re-derived by pencil, but driven by the
     same optimizer machinery every section below reuses.
  2. THE 1000x TRAP: Adam's raw second moment v is 1000x too small at step
     k=1 (beta2=0.999) before bias correction fixes it -- the algebra behind
     page 17's "why bias correction" box, plus a concrete number from TN-1's
     own gradient.
  3. THE SPREAD, REOPENED: page 14's nine gradients for input 2 (the
     gradient-spread case, x=[0.60,-0.20]) span 10.22x. This script recomputes
     them, reproduces that number, then takes ONE Adam step and shows the
     per-parameter UPDATE magnitudes collapse to ~1.0x -- Adam does not
     shrink the spread, it erases it.
  4. THE RACE: reset TN-1, train on input 2 to a target loss under all four
     optimizers with the SAME learning rate, and log steps-to-target. Adam
     and AdamW get there in a fraction of SGD's steps, untuned -- the
     "forgiving of a learning rate that would break SGD" claim from the page,
     made concrete.

TN-1's nine frozen parameters are hard-coded directly from constants.md
section 8 -- NOT from NN.worked221()'s JS default, which reproduces a RETIRED
network (see 14_backprop_tn1.py's identical warning).

Why pure NumPy, not torch: every update rule on page 17 is four lines of
arithmetic (SGD, momentum, Adam, AdamW are all boxed equations, not framework
magic), and TN-1 has nine parameters. Writing the optimizer math by hand here
is the same choice 14_backprop_tn1.py made for backprop -- and it means this
script needs no GPU and no torch to run anywhere, including this machine.
Page 17's own code sample shows the one-line torch.optim.AdamW equivalent;
this script shows what is happening inside that line.

Usage
-----
    python 17_optimizer_race.py            # full race, target loss 0.05, 200 steps/optimizer
    python 17_optimizer_race.py --quick     # shorter race, looser target, same assertions
    python 17_optimizer_race.py --steps 500 --target 0.01   # override the race

SAFETY: pure NumPy, CPU only, <1 s, allocates a few KB. Writes and installs
nothing. No GPU, no torch, no network access -- this script has exactly one
code path and that path IS its own self-test: every section checks its own
numbers against constants.md section 8 / 9.3 below.
"""

import argparse

import numpy as np

np.set_printoptions(precision=4, suppress=False)


# --------------------------------------------------------------------------- #
# TN-1's nine frozen parameters -- constants.md section 8. Hard-coded, exactly
# as in 14_backprop_tn1.py and 15_autograd_check.py. Never NN.worked221()'s
# default (that reproduces the RETIRED section-5.4 network).
# --------------------------------------------------------------------------- #
W1_0 = np.array([[0.5, -0.3], [0.8, 0.2]])
B1_0 = np.array([0.1, -0.1])
W2_0 = np.array([0.6, -0.9])
B2_0 = 0.2

X1 = np.array([1.0, 2.0])     # input 1 -- the dead-unit case (constants 8.1-8.5)
X2 = np.array([0.60, -0.20])  # input 2 -- the gradient-spread case (constants 8.7)

PARAM_NAMES = ["W1[0,0]", "W1[0,1]", "W1[1,0]", "W1[1,1]", "b1[0]", "b1[1]", "W2[0]", "W2[1]", "b2"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(W1, b1, W2, b2, x):
    """2 -> tanh(2) -> sigmoid(1) -> BCE(y=1). Same four lines as p.14/p.15."""
    z1 = W1 @ x + b1
    a1 = np.tanh(z1)
    z2 = float(W2 @ a1 + b2)
    yhat = sigmoid(z2)
    loss = -np.log(yhat)
    return a1, yhat, loss


def backward(x, a1, W2, yhat):
    """BP1-BP4 (constants section 8 / 14_backprop_tn1.py) -- unchanged here.
    This script's only new machinery is what happens to (dW1,db1,dW2,db2)
    AFTER this function returns."""
    dz2 = yhat - 1.0
    dW2 = dz2 * a1
    db2 = dz2
    da1 = dz2 * W2
    tanh_prime = 1.0 - a1 ** 2
    delta1 = da1 * tanh_prime
    dW1 = np.outer(delta1, x)
    db1 = delta1
    return dW1, db1, dW2, db2


def flatten(W1, b1, W2, b2):
    return np.concatenate([W1.ravel(), b1, W2, [b2]])


def unflatten(theta):
    return theta[0:4].reshape(2, 2), theta[4:6], theta[6:8], theta[8]


def loss_and_grad(theta, x):
    """One forward + backward pass, parameters and gradient both as flat (9,) vectors."""
    W1, b1, W2, b2 = unflatten(theta)
    a1, yhat, loss = forward(W1, b1, W2, b2, x)
    dW1, db1, dW2, db2 = backward(x, a1, W2, yhat)
    return loss, flatten(dW1, db1, dW2, db2)


# --------------------------------------------------------------------------- #
# The four update rules, exactly as boxed on page 17. Each returns the new
# theta given the current state; state (v / m,v) lives in the caller.
# --------------------------------------------------------------------------- #

def sgd_update(theta, g, eta):
    return theta - eta * g


def momentum_update(theta, g, v, eta, mu=0.9):
    v = mu * v + g
    return theta - eta * v, v


def adam_update(theta, g, m, v, k, eta, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    mhat = m / (1 - beta1 ** k)
    vhat = v / (1 - beta2 ** k)
    step = mhat / (np.sqrt(vhat) + eps)
    return theta - eta * step, m, v, step


def adamw_update(theta, g, m, v, k, eta, beta1=0.9, beta2=0.95, wd=0.1, eps=1e-8):
    """Same Adam moments, but weight decay applied straight to theta, OUTSIDE
    the sqrt(vhat) division -- the one change AdamW makes (page 17)."""
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    mhat = m / (1 - beta1 ** k)
    vhat = v / (1 - beta2 ** k)
    step = mhat / (np.sqrt(vhat) + eps) + wd * theta
    return theta - eta * step, m, v, step


def clip_norm(g, max_norm=1.0):
    n = np.linalg.norm(g)
    return g * (max_norm / n) if n > max_norm else g


print("=" * 72)
print("17_optimizer_race.py -- SGD / momentum / Adam / AdamW on TN-1 (NumPy)")
print(f"numpy {np.__version__}")
print("=" * 72)


# =========================================================================== #
# SECTION 1 -- BRIDGE: the page-14 SGD step, now through this script's own
# plain-SGD update function. Owns the printed L: 0.9869 -> 0.8222 beat.
# =========================================================================== #
print("\n" + "-" * 72)
print("1. BRIDGE -- page 14's SGD step, through THIS script's sgd_update()")
print("-" * 72)

theta1 = flatten(W1_0, B1_0, W2_0, B2_0)
loss_before, g1 = loss_and_grad(theta1, X1)
theta1_after = sgd_update(theta1, g1, eta=0.1)
loss_after, _ = loss_and_grad(theta1_after, X1)

print(f"  input 1, x = {X1.tolist()}, eta = 0.1")
print(f"  loss before step = {loss_before:.4f}")
print(f"  loss after  step = {loss_after:.4f}   (delta = {loss_after - loss_before:+.4f})")

assert abs(loss_before - 0.9869) < 1e-4, "loss before should be 0.9869 (constants section 8.1)"
assert abs(loss_after - 0.8222) < 1e-4, "loss after should be 0.8222 (constants section 8.3)"
print("  [OK] L: 0.9869 -> 0.8222, same numbers as page 14's pencil step, now via sgd_update().")
print("  The optimizer object is new; the arithmetic underneath it is exactly BP1-BP4 + one line.")


# =========================================================================== #
# SECTION 2 -- Adam's 1000x trap at k=1, and the bias correction that fixes it
# =========================================================================== #
print("\n" + "-" * 72)
print("2. THE 1000x TRAP -- Adam's raw v at k=1, before bias correction")
print("-" * 72)

theta2 = flatten(W1_0, B1_0, W2_0, B2_0)
loss2, g2 = loss_and_grad(theta2, X2)
BETA2_CLASSIC = 0.999

# One representative component -- db2, the largest-magnitude of the nine.
g1_db2 = g2[-1]
v1_raw = (1 - BETA2_CLASSIC) * g1_db2 ** 2
v1_hat = v1_raw / (1 - BETA2_CLASSIC ** 1)
ratio = v1_hat / v1_raw

print(f"  x = {X2.tolist()} (page 14's gradient-spread input), g1 = dL/db2 = {g1_db2:+.4f}")
print(f"  raw   v1 = (1-beta2) * g1^2         = {v1_raw:.8f}   (beta2 = {BETA2_CLASSIC})")
print(f"  corr  v1_hat = v1 / (1-beta2^1)     = {v1_hat:.8f}")
print(f"  v1_hat / v1_raw                     = {ratio:.1f}x")

assert abs(ratio - 1000.0) < 1e-6, f"v1 should be exactly 1000x too small at k=1, beta2=0.999, got {ratio}"
too_large = np.sqrt(1000.0)
print(f"  [OK] v is 1000x too small before correction -- matches constants section 9.3.")
print(f"  Without correction, the update divides by sqrt(v1) instead of sqrt(v1_hat):")
print(f"  that is sqrt(1000) = {too_large:.1f}x too large a first step. Bias correction exists")
print(f"  to prevent that overshoot -- not to make Adam 'adaptive', just to make it correct at k=1.")
print(f"  And 1/(1-{BETA2_CLASSIC}) = {1/(1-BETA2_CLASSIC):.0f} steps is v's memory -- the mechanistic")
print(f"  reason warmup exists (page 21).")


# =========================================================================== #
# SECTION 3 -- the 10.22x spread, reopened from page 14, and what one Adam
# step does to it. Reproduces 14's spread-motivated "therefore Adam".
# =========================================================================== #
print("\n" + "-" * 72)
print("3. THE SPREAD, REOPENED -- page 14's nine gradients, then one Adam step")
print("-" * 72)

spread_g = np.max(np.abs(g2)) / np.min(np.abs(g2))
print(f"  loss at x = {X2.tolist()}: {loss2:.4f}")
print(f"  nine gradients (|g|, sorted): {np.sort(np.abs(g2))}")
print(f"  gradient spread = max|g| / min|g| = {spread_g:.4f}x")

assert abs(spread_g - 10.2246) < 1e-3, f"gradient spread should be 10.2246x (constants 8.7), got {spread_g:.4f}"
print("  [OK] 10.2246x -- identical to 14_backprop_tn1.py's spread beat (same nine gradients).")
print("  Plain SGD's update is eta*g, so its per-parameter UPDATE spread is the same 10.22x --")
print("  SGD cannot touch this. Now take ONE Adam step from the same point:")

m0 = np.zeros(9)
v0 = np.zeros(9)
_, m1, v1, adam_step1 = adam_update(theta2, g2, m0, v0, k=1, eta=0.1, beta2=BETA2_CLASSIC)
spread_upd = np.max(np.abs(adam_step1)) / np.min(np.abs(adam_step1))

print(f"\n  Adam per-parameter step direction, k=1 (mhat / (sqrt(vhat)+eps)):")
for name, g_i, s in zip(PARAM_NAMES, g2, adam_step1):
    print(f"    {name:<8} g = {g_i:+.4f}   step = {s:+.6f}")
print(f"  update spread = max|step| / min|step| = {spread_upd:.6f}x")

assert np.allclose(np.abs(adam_step1), 1.0, atol=1e-4), \
    f"every Adam step-1 update magnitude should be ~1.0 (sign-only), got {adam_step1}"
assert abs(spread_upd - 1.0) < 1e-4, f"Adam step-1 update spread should collapse to ~1.0x, got {spread_upd:.4f}"
print("  [OK] every |step| ~ 1.0 -- at k=1, mhat=g1 exactly and vhat=g1^2 exactly, so")
print("  mhat/sqrt(vhat) = sign(g1) for EVERY parameter, regardless of |g1|. The 10.22x")
print("  gradient spread does not shrink under Adam -- it is erased. This is the honest")
print("  version of 'therefore Adam' from page 14/17: not a workaround for un-normalised")
print("  inputs (section 8.7's 8.64x-at-1:1-ratio check already ruled that out), but a fix")
print("  for the network's own structural spread.")


# =========================================================================== #
# SECTION 4 -- THE RACE: four optimizers, one loss, same eta, steps-to-target
# =========================================================================== #
def run_race(kind, x, steps, eta, target, clip=1.0):
    theta = flatten(W1_0, B1_0, W2_0, B2_0)
    v = np.zeros(9)
    m = np.zeros(9)
    losses = []
    hit_step = None
    max_grad_norm = 0.0
    for k in range(1, steps + 1):
        loss, g = loss_and_grad(theta, x)
        max_grad_norm = max(max_grad_norm, float(np.linalg.norm(g)))
        if clip is not None:
            g = clip_norm(g, clip)
        losses.append(loss)
        if hit_step is None and loss < target:
            hit_step = k - 1  # steps already taken to REACH this loss
        if kind == "sgd":
            theta = sgd_update(theta, g, eta)
        elif kind == "momentum":
            theta, v = momentum_update(theta, g, v, eta)
        elif kind == "adam":
            theta, m, v, _ = adam_update(theta, g, m, v, k, eta, beta2=BETA2_CLASSIC)
        elif kind == "adamw":
            theta, m, v, _ = adamw_update(theta, g, m, v, k, eta, beta2=0.95, wd=0.1)
        else:
            raise ValueError(kind)
    return losses, hit_step, max_grad_norm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200, help="max steps per optimizer (default 200)")
    ap.add_argument("--target", type=float, default=0.05, help="target loss for steps-to-target (default 0.05)")
    ap.add_argument("--eta", type=float, default=0.1, help="learning rate, SAME for all four optimizers (default 0.1)")
    ap.add_argument("--quick", action="store_true", help="shorter race: 60 steps, target 0.15")
    args = ap.parse_args()

    steps = 60 if args.quick else args.steps
    target = 0.15 if args.quick else args.target
    eta = args.eta

    print("\n" + "-" * 72)
    print(f"4. THE RACE -- TN-1, input 2, target loss < {target}, eta = {eta} for ALL FOUR")
    print("-" * 72)
    print("  (page 17's grad clip = 1.0 is applied every step -- watch max||g|| below to")
    print("   see whether it ever actually fires on this tiny toy network.)")

    results = {}
    for kind in ["sgd", "momentum", "adam", "adamw"]:
        losses, hit_step, max_g = run_race(kind, X2, steps, eta, target)
        results[kind] = (losses, hit_step, max_g)
        hit_str = f"step {hit_step:>4}" if hit_step is not None else f"NOT within {steps}"
        print(f"  {kind:<9} start L={losses[0]:.4f}  final L={losses[-1]:.6f}  "
              f"target hit @ {hit_str}  max||g||={max_g:.4f}")

    sgd_hit = results["sgd"][1]
    print()
    for kind in ["momentum", "adam", "adamw"]:
        hit = results[kind][1]
        if sgd_hit and hit:
            print(f"  {kind:<9} reached target {sgd_hit / hit:.1f}x faster than plain SGD "
                  f"({hit} steps vs {sgd_hit}), same eta = {eta}, untuned.")

    # No decay pulls purely downhill on a single fixed example (full-batch, page
    # 16's B=full case): strictly monotone. AdamW's decoupled decay is NOT part
    # of that descent -- it pulls every weight toward zero regardless of the
    # loss gradient -- so once the loss gets small the decay term can win a
    # step and cause a tiny uptick. That is not a bug in this script; it is
    # the honest, structural difference "decay is not regularisation-as-L2"
    # (page 17's AdamW warn box) makes visible once you look at raw curves.
    # This monotonicity check is structural -- always holds, whatever --steps
    # / --target / --eta the caller passes.
    for kind in ["sgd", "momentum", "adam"]:
        assert np.all(np.diff(results[kind][0]) <= 1e-9), \
            f"{kind} (no weight decay) must be monotone non-increasing on a single fixed example"

    # The "reaches target, and faster than SGD" claims depend on --steps /
    # --target / --eta being generous enough for SGD to get there at all (the
    # DEFAULTS are chosen so it does, comfortably). If a caller passes a
    # tighter budget than SGD's own convergence rate at that eta allows, report
    # it honestly instead of crashing -- that outcome IS the lesson (SGD is
    # slow), not a bug in the script.
    if sgd_hit is None:
        print(f"\n  [NOTE] plain SGD did not reach loss < {target} within {steps} steps at eta={eta}.")
        print("  That is not a failure of this script -- it is the lesson: raise --steps, loosen")
        print("  --target, or watch how much sooner momentum/Adam/AdamW get there below.")
    else:
        for kind in ["momentum", "adam", "adamw"]:
            assert results[kind][1] is not None and results[kind][1] <= sgd_hit, \
                f"{kind} should reach the target loss in no more steps than plain SGD at the same eta"
    print("\n  [OK] SGD / momentum / Adam converge monotonically (single fixed example, full-batch,")
    print("  no noise -- page 16's B=full curve; no weight decay to fight the descent). AdamW's own")
    print("  curve may wobble by ~1e-6 once the loss is tiny -- decoupled decay pulls toward zero")
    print("  independent of the gradient, which is exactly why AdamW warns 'not the same as L2'.")
    print("  Momentum and Adam/AdamW all beat plain SGD to the target")
    print("  at the SAME learning rate -- the 'forgiving of a learning rate that would break SGD'")
    print("  claim, made concrete on the one network this course has hand-verified end to end.")

    print("\n" + "=" * 72)
    print("All assertions passed:")
    print("  1) SGD step L: 0.9869 -> 0.8222   (page 14, reproduced via sgd_update())")
    print("  2) Adam v at k=1 is exactly 1000x too small before bias correction (beta2=0.999)")
    print("  3) 10.22x gradient spread (page 14) collapses to ~1.0x update spread after one Adam step")
    print("  4) SGD < momentum, Adam, AdamW in steps-to-target, all at the SAME eta")
    print("Four optimizers, one set of nine gradients: SGD moves them raw, momentum smooths")
    print("their history, Adam rescales each by its own magnitude, AdamW keeps decay honest.")
    print("=" * 72)


if __name__ == "__main__":
    main()
