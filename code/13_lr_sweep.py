#!/usr/bin/env python3
"""
13_lr_sweep.py -- the ravine's stiff axis, collapsed to 1-D.

Course artifact for p.13 (Gradient Descent & the Loss Landscape). The page's live demo
runs real 2-D gradient descent on the anisotropic bowl

    f(x, y) = 0.5 * (A*x^2 + B*y^2),   A = 1, B = lambda_max = 20

starting at (x0, y0) = (-4, 3) -- see the page's `Surface3D` demo JS (`const A = 1, B = 20;
const START = [-4, 3];`). Because the bowl is separable, x and y never interact through the
update rule -- each axis runs its OWN 1-D gradient descent: x with curvature A=1 (mild,
converges almost regardless of eta), y with curvature B=20 (stiff -- this is the axis that
decides stability). Collapsing to that one stiff axis IS the "1-D quadratic" this script
sweeps:

    f(z)  = 0.5 * lambda * z^2         lambda = 20 = lambda_max   (constants.md Sec 9.3)
    f'(z) = lambda * z
    z_{k+1} = z_k - eta * lambda * z_k = (1 - eta*lambda) * z_k

Exact solution: z_k = (1 - eta*lambda)^k * z0 -- a pure geometric sequence. Stable
(|z_k| -> 0) iff the per-step multiplier m = |1 - eta*lambda| < 1, i.e. 0 < eta <
2/lambda = eta_crit. z0 = 3.0 is literally the page demo's y0 -- so this script's
trajectory IS the page demo's steep-axis trajectory, not a fresh toy example; the two
must (and do) agree number for number.

Usage
-----
    python 13_lr_sweep.py               # narrated sweep, prints all three trajectories
    python 13_lr_sweep.py --steps 20    # more steps per eta in the narration
    python 13_lr_sweep.py --self-test   # assertions only, no narration, exit 0/1

SAFETY: pure Python arithmetic. No GPU, no network, no files touched. Runtime <1s.
"""

import argparse

# ---- frozen numbers, constants.md Sec 9.3 [DER] ("ridge / eta_crit") --------
LAMBDA_MAX = 20.0                  # demo ravine's stiff-axis curvature (B in the page JS)
ETA_CRIT = 2.0 / LAMBDA_MAX        # = 0.1 exactly
Z0 = 3.0                           # page Surface3D START=(-4, 3): this is the y (stiff-axis) coord
ETAS = [0.05, 0.1, 0.15]           # the sweep this page's ".box try" promises


def multiplier(eta: float) -> float:
    """Per-step error multiplier in the stiff direction: m = |1 - eta*lambda_max|.
    Matches the page's worked box (`m = |1 - 20*eta|`) verbatim."""
    return abs(1.0 - eta * LAMBDA_MAX)


def trajectory(eta: float, steps: int):
    """Iterate z_{k+1} = z_k - eta*(lambda*z_k), the SAME update rule as the page's
    descend(): gradient first, then subtract eta*gradient. Not the closed form --
    this genuinely re-runs the loop, so agreement with the closed form below is a
    real check, not a tautology."""
    z = Z0
    pts = [z]
    for _ in range(steps):
        g = LAMBDA_MAX * z          # gradient of f(z) = 0.5*lambda*z^2
        z = z - eta * g
        pts.append(z)
        if abs(z) > 1e6:            # mirror the page demo's divergence guard
            break
    return pts


def classify(pts, eta: float) -> str:
    """Classify by the multiplier m=|1-eta*lambda|, not by comparing the last two
    points -- z_k can land exactly on 0 (eta=0.05) where two zeros in a row would
    otherwise look "constant amplitude" and be mislabelled marginal."""
    if abs(pts[-1]) > 1e6:
        return "DIVERGED"
    m = multiplier(eta)
    if m < 1.0 - 1e-9:
        return "converged (shrinking)"
    if m > 1.0 + 1e-9:
        return "diverging (growing)"
    return "marginal (constant amplitude, never settles)"


def narrate(steps: int) -> None:
    print("=" * 68)
    print("LEARNING-RATE SWEEP -- 1-D quadratic, stiff axis of the p.13 ravine")
    print("=" * 68)
    print(f"  lambda_max = {LAMBDA_MAX:.1f}                    (constants.md Sec 9.3)")
    print(f"  eta_crit   = 2/lambda_max = {ETA_CRIT:.3f}        (constants.md Sec 9.3, [DER])")
    print(f"  z0         = {Z0:.1f}   (page Surface3D START=(-4,3), the y-coordinate)")
    print()

    for eta in ETAS:
        pts = trajectory(eta, steps)
        m = multiplier(eta)
        verdict = classify(pts, eta)
        tag = "< eta_crit" if eta < ETA_CRIT - 1e-12 else ("= eta_crit" if abs(eta - ETA_CRIT) < 1e-12 else "> eta_crit")
        print(f"  eta = {eta:.2f}  ({tag})   multiplier m = |1 - {LAMBDA_MAX:.0f}*{eta:.2f}| = {m:.2f}   -> {verdict}")
        shown = pts[:6]
        trace = "  ".join(f"{v:+.4f}" for v in shown)
        print(f"    z_0..z_{len(shown) - 1}: {trace}" + (" ..." if len(pts) > 6 else ""))
        if len(pts) > 6:
            print(f"    z_{len(pts) - 1} (after {len(pts) - 1} steps): {pts[-1]:+.6g}")
        print()

    print("-" * 68)
    print("  eta=0.05 < eta_crit : multiplier 0.00 -> collapses to the axis in ONE step")
    print("  eta=0.10 = eta_crit : multiplier 1.00 -> the knife-edge, oscillates forever")
    print("  eta=0.15 > eta_crit : multiplier 2.00 -> error DOUBLES every step, diverges")
    print("-" * 68)
    print("  Same numbers as dragging the page's eta slider to 0.05 / 0.10 / 0.15 and")
    print("  reading off the y-coordinate of the dot -- this script IS that axis.")


def self_test() -> None:
    print("Running self-checks (no GPU, no display, pure arithmetic)...")

    # --- frozen constants agree with constants.md Sec 9.3 -------------------
    assert LAMBDA_MAX == 20.0, "lambda_max must match constants.md Sec 9.3 (demo ravine)"
    assert abs(ETA_CRIT - 0.1) < 1e-15, "eta_crit = 2/lambda_max must be exactly 0.1"

    # --- eta = 0.05: multiplier exactly 0 -> exact one-step collapse to zero ---
    m005 = multiplier(0.05)
    assert m005 == 0.0, f"eta=0.05 multiplier should be exactly 0.0, got {m005}"
    pts005 = trajectory(0.05, steps=5)
    assert pts005[1] == 0.0, f"eta=0.05 must land exactly on 0 after step 1, got {pts005[1]}"
    assert all(v == 0.0 for v in pts005[1:]), "once at 0 it stays at 0 (fixed point)"
    print(f"  eta=0.05: z_1 = {pts005[1]:.1f} exactly, stays there -- PASS "
          f"(page table: multiplier 0.00, 'collapses to the axis in one step')")

    # --- eta = 0.10 = eta_crit: multiplier exactly 1 -> the stability boundary ---
    # This is the transition the page marks with "the knife-edge": for eta just
    # below 0.1 the stiff-axis sequence is convergent (m<1, shrinking envelope);
    # for eta just above it is divergent (m>1, growing envelope). At eta=0.1
    # EXACTLY it sits ON that boundary -- constant amplitude, sign flips every
    # step, never decays and never grows. That is the monotone(shrink)-vs-
    # runaway-oscillation transition the spec calls out, and it must land here
    # to the last bit, not approximately.
    m010 = multiplier(0.1)
    assert m010 == 1.0, f"eta=0.1 multiplier must land exactly on the boundary m=1.0, got {m010}"
    pts010 = trajectory(0.1, steps=10)
    for k, v in enumerate(pts010):
        assert abs(abs(v) - Z0) < 1e-9, f"eta=0.1 step {k}: |z_k| should stay {Z0}, got {abs(v)}"
        want_positive = (k % 2 == 0)
        assert (v > 0) == want_positive, f"eta=0.1 step {k}: sign should flip every step, got z={v}"
    print(f"  eta=0.10: multiplier m={m010:.1f} exactly, |z_k|={Z0} constant and sign "
          f"flips for {len(pts010) - 1} steps -- PASS (the boundary, 'oscillates forever')")

    # --- eta = 0.15: multiplier exactly 2 -> magnitude doubles every step, diverges ---
    m015 = multiplier(0.15)
    assert m015 == 2.0, f"eta=0.15 multiplier should be exactly 2.0, got {m015}"
    pts015 = trajectory(0.15, steps=8)
    for k in range(1, len(pts015)):
        ratio = abs(pts015[k]) / abs(pts015[k - 1])
        assert abs(ratio - 2.0) < 1e-9, f"eta=0.15 step {k}: magnitude should double, ratio={ratio}"
    assert abs(pts015[-1]) > abs(Z0) * 2 ** (len(pts015) - 2), "eta=0.15 trajectory must be growing"
    print(f"  eta=0.15: multiplier m={m015:.1f} exactly, |z_k| doubles every step "
          f"({len(pts015) - 1} steps) -- PASS (diverges, matches page's 'error doubles')")

    # --- iterated update rule must match the closed form for ALL three etas ---
    for eta in ETAS:
        pts = trajectory(eta, steps=6)
        for k, v in enumerate(pts):
            closed = (1.0 - eta * LAMBDA_MAX) ** k * Z0
            assert abs(v - closed) < 1e-9, (
                f"eta={eta} step {k}: iterated {v} vs closed-form {closed} disagree"
            )
    print("  iterated update rule matches closed form z_k=(1-eta*lambda)^k * z0 "
          f"for all etas in {ETAS} -- PASS")

    # --- reproduce the page's own worked table (Sec "eta_crit, exactly") verbatim ---
    worked = {0.05: 0.00, 0.09: 0.80, 0.10: 1.00, 0.15: 2.00}
    for eta, m_expected in worked.items():
        m = multiplier(eta)
        assert abs(m - m_expected) < 1e-9, f"worked-table mismatch at eta={eta}: {m} vs {m_expected}"
    print(f"  multiplier formula reproduces the page's worked table {worked} -- PASS")

    print("All self-checks PASS.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="1-D quadratic LR sweep (eta in {0.05, 0.1, 0.15}); divergence at "
                     "eta_crit = 2/lambda_max = 0.1 -- p.13 companion script."
    )
    ap.add_argument("--steps", type=int, default=10,
                     help="steps per eta in the narrated sweep (default 10)")
    ap.add_argument("--self-test", action="store_true",
                     help="run assertions only, no narration, exit 0 on success")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return
    print()
    narrate(args.steps)


if __name__ == "__main__":
    main()
