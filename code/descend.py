#!/usr/bin/env python3
"""
descend.py -- gradient descent on a quartic, watched digit by digit. Page 04.

Course artifact for p.04 ("Derivatives and the theta-vs-x flip"). The page's Slope
Explorer / LR Playground, in runnable code: the same quartic

    f(x)  = x^4 - 3x^2 + x + 2
    f'(x) = 4x^3 - 6x + 1

and the ONE update rule the whole course is built on:

    x_(k+1) = x_k - eta * f'(x_k)

Run it at the page's own eta values and you get the page's own three verdicts:
  --lr 0.01   crawls smoothly downhill
  --lr 0.1    converges fast                          (this script's default)
  --lr 0.35   oscillates, then blows up to Inf -- the NaN in your real training
              logs later is this, at scale. Page 13 names the exact boundary,
              eta_crit = 2/lambda_max, on a quadratic.

Why torch tensors, not the page's inline plain-Python lines: plain Python's `**`
on a float RAISES OverflowError once a value is large enough to overflow a double
(try `10.0 ** 400` in a REPL) -- it does not quietly become inf. On this exact
quartic that crash arrives at eta=0.3, one step before the page's own eta=0.35 --
it would kill the divergence demo with a traceback instead of showing the Inf the
page's canvas draws. A torch tensor follows IEEE-754 float semantics and overflows
to inf silently, so `torch.isfinite` catches it cleanly, the way the canvas does.

No GPU anywhere in this script -- it IS the no-GPU self-test path (local
verification reality: CPU only, <1 s, a handful of scalars). Writes and installs
nothing.
"""

import argparse
import torch

SEED = 42
torch.manual_seed(SEED)          # no randomness used below; stamped for house style


# --------------------------------------------------------------------------
# The quartic and its derivative -- exactly the page's canvas equation.
# --------------------------------------------------------------------------
def f(x):
    return x**4 - 3 * x**2 + x + 2


def df(x):
    return 4 * x**3 - 6 * x + 1


# The two minima this quartic actually has -- the real roots of f'(x)=0 with
# f''(x)>0 -- found once via a root solve and frozen here so a regression fails
# loudly instead of drifting silently. This quartic is page-04-local (not a
# course-wide number), so the freeze lives in the script that owns it, not in
# constants.md.
X_MIN_LEFT,  F_MIN_LEFT  = -1.3008395659415773, -1.5139050389347888   # reached from x0=-2
X_MIN_RIGHT, F_MIN_RIGHT = 1.1309011226299868, 0.9297698182238459     # reached from x0=+2, small eta


def descend(lr, x0, steps=200, label=""):
    """Run the update rule; stop the instant x stops being finite and say so."""
    x = torch.tensor(float(x0), dtype=torch.float64)
    for k in range(steps):
        x = x - lr * df(x)
        if not torch.isfinite(x):
            print(f"  {label}diverged at step {k}: x -> {x.item()}")
            return x, k + 1, False
    print(f"  {label}x={x.item(): .6f}  f(x)={f(x).item(): .6f}  (ran the full {steps} steps)")
    return x, steps, True


def main():
    ap = argparse.ArgumentParser(description="Gradient descent on x^4-3x^2+x+2 -- page 04's playground, runnable.")
    ap.add_argument("--lr", type=float, default=0.1, help="learning rate eta (page default 0.1)")
    ap.add_argument("--x0", type=float, default=-2.0, help="starting point (page default -2.0)")
    ap.add_argument("--steps", type=int, default=200, help="max update steps")
    args = ap.parse_args()

    print("=" * 68)
    print("descend.py -- one update rule, three verdicts")
    print(f"torch {torch.__version__} - seed {SEED}")
    print("=" * 68)
    print("  f(x)  = x^4 - 3x^2 + x + 2")
    print("  f'(x) = 4x^3 - 6x + 1")
    print("  update: x_(k+1) = x_k - eta * f'(x_k)")

    print(f"\n-- your run: eta={args.lr}, x0={args.x0}, steps<={args.steps} --")
    x, k, finite = descend(args.lr, args.x0, args.steps)
    if finite:
        nearest = "left" if abs(x.item() - X_MIN_LEFT) < abs(x.item() - X_MIN_RIGHT) else "right"
        print(f"  landed near the {nearest} minimum")
    else:
        print("  this is the Inf you will see in a real loss curve when eta is too big for the")
        print("  curvature it is stepping on -- page 13 names the exact boundary, eta_crit = 2/lambda_max.")

    # ------------------------------------------------------------------
    # Self-check -- the page's own verdicts, asserted, not eyeballed.
    # Runs regardless of what --lr/--x0 were passed above.
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("SELF-CHECK -- the demo's own verdicts, asserted")
    print("=" * 68)

    print("\n1. small eta (0.05) from x0=-2.0 converges to the LEFT minimum:")
    x1, _, ok1 = descend(0.05, -2.0, 200, label="   ")
    assert ok1, "small-eta descent from x0=-2.0 should stay finite for 200 steps"
    assert abs(x1.item() - X_MIN_LEFT) < 1e-6, f"expected x -> {X_MIN_LEFT}, got {x1.item()}"
    assert abs(f(x1).item() - F_MIN_LEFT) < 1e-6, f"expected f(x) -> {F_MIN_LEFT}, got {f(x1).item()}"
    print(f"   PASS: x -> {x1.item():.10f}  f(x) -> {f(x1).item():.10f}")

    print("\n2. SAME small eta (0.05), x0=+2.0 -- same function, same optimizer, a DIFFERENT minimum:")
    x2, _, ok2 = descend(0.05, 2.0, 200, label="   ")
    assert ok2, "small-eta descent from x0=+2.0 should stay finite for 200 steps"
    assert abs(x2.item() - X_MIN_RIGHT) < 1e-6, f"expected x -> {X_MIN_RIGHT}, got {x2.item()}"
    assert abs(f(x2).item() - F_MIN_RIGHT) < 1e-6, f"expected f(x) -> {F_MIN_RIGHT}, got {f(x2).item()}"
    print(f"   PASS: x -> {x2.item():.10f}  f(x) -> {f(x2).item():.10f}  (not #1's minimum)")

    print("\n3. large eta (0.35, the page's own number) from x0=-2.0 -- diverges to Inf:")
    x3, k3, ok3 = descend(0.35, -2.0, 200, label="   ")
    assert not ok3, "eta=0.35 should diverge to Inf well before 200 steps, but it ran the full budget"
    assert k3 < 200, "divergence should be fast, not a slow drift"
    print(f"   PASS: not finite by step {k3 - 1}, exactly like the canvas at eta=0.35")

    print("\n[OK] small eta converges -- to a start-dependent minimum, same optimizer;")
    print("     large eta diverges to Inf. Same update rule, opposite fates, decided by")
    print("     one number. Page 13 makes that number precise: eta_crit = 2/lambda_max.")


if __name__ == "__main__":
    main()
