#!/usr/bin/env python3
"""
score_check.py -- the score is a finite-difference FACT, not a claim you take on faith.

Course artifact for p.56 ("Score Matching & the SDE View"), manifest D6. The page asserts
an identity: the score of the forward-noised Gaussian equals the (negated, rescaled) noise
you added --

    s_theta(x_t, t) := grad_{x_t} log q(x_t | x0)  ==  -eps / sqrt(1 - abar_t)

This script does not assert that identity -- it MEASURES both sides and shows they agree.
Side A (numeric): central finite-difference the log-density q(x_t|x0) component-by-component
in x_t. Side B (analytic): the closed-form -eps/sqrt(1-abar_t) from Eq. 3.3's algebra. No
network, no training -- log q(x_t|x0) is an exact quadratic in x_t (a Gaussian log-density),
so a central difference has ZERO truncation error; whatever gap remains is float round-off.

  1. Build the same linear DDPM schedule as p.54/p.55/prediction_targets.py: beta 1e-4 -> 2e-2
     over T=1000, cumulative product (NOT a closed-form power -- the schedule is a product of
     1000 distinct betas).
  2. Fix one (x0, eps) pair (D=4 components, float64, seeded) -- mirrors the page's live JS
     demo (D56.2) so the script and the browser reproduce the same story.
  3. At a handful of noise levels t, build x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps, then
     for each component i: fd[i] = (logq(x_t + h e_i) - logq(x_t - h e_i)) / (2h).
  4. Compare fd against the closed form -eps/sqrt(1-abar_t): print both, the max absolute
     error, and the L2 relative error. Assert torch.allclose AND rel error < 1e-5 -- in
     float64, the same gradient-check discipline as code/14_backprop_tn1.py and
     code/15_autograd_check.py, just applied to grad-w.r.t.-the-input instead of
     grad-w.r.t.-the-weights (notation 4.4: s_theta = grad_x log p_t(x), NOT grad_theta L).

SAFETY: CPU only, float64, a handful of 4-vectors -- no GPU, no network, no training.
Writes and installs nothing. Runtime well under 5 seconds.
"""

import torch

SEED = 56056
torch.manual_seed(SEED)
torch.set_printoptions(precision=6)

DTYPE = torch.float64

# ------------------------------------------------------------------------------------
# The frozen linear DDPM schedule -- identical to p.54 / p.55 / prediction_targets.py:
# beta 1e-4 -> 2e-2 over T=1000, running cumulative product (not Math.pow). constants.md
# 9.6: abar_1000 ~= 4e-5, sqrt(abar_T) = 0.0063 (nonzero terminal SNR).
# ------------------------------------------------------------------------------------
T, BMIN, BMAX = 1000, 1e-4, 2e-2


def build_abar(T, bmin, bmax):
    """abar[t] = prod_{s<=t} (1 - beta_s). abar[0] = 1."""
    abar = torch.empty(T + 1, dtype=DTYPE)
    abar[0] = 1.0
    ab = 1.0
    for t in range(1, T + 1):
        beta = bmin + (bmax - bmin) * (t - 1) / (T - 1)
        ab *= (1.0 - beta)
        abar[t] = ab
    return abar


ABAR = build_abar(T, BMIN, BMAX)

print("=" * 74)
print("score_check.py -- the score identity, verified not asserted (p.56, manifest D6)")
print(f"torch {torch.__version__} - seed {SEED} - dtype {DTYPE} - linear DDPM, beta {BMIN}->{BMAX}, T={T}")
print("=" * 74)
print("\nidentity under test:  s_theta(x_t,t) = grad_{x_t} log q(x_t|x0)  ==  -eps / sqrt(1-abar_t)")
print("(notation 4.4: the score differentiates w.r.t. the INPUT x, never the weights theta)\n")

# ------------------------------------------------------------------------------------
# Fixed data vector + fixed noise draw, D=4 components -- mirrors the page's live demo
# (D56.2: x0 ~ 0.8*N(0,1), eps ~ N(0,1), dim 4).
# ------------------------------------------------------------------------------------
D = 4
x0 = torch.randn(D, dtype=DTYPE) * 0.8
eps = torch.randn(D, dtype=DTYPE)

print(f"x0  = {x0.tolist()}")
print(f"eps = {eps.tolist()}")


def logq(xt, ca, om):
    """log q(x_t | x0) up to an additive constant that does not depend on x_t.
    q(x_t|x0) = N(x_t; ca*x0, om*I)  =>  log q = -||x_t - ca*x0||^2 / (2*om) + const."""
    d = xt - ca * x0
    return -(d @ d) / (2.0 * om)


def finite_diff_score(xt, ca, om, h=1e-4):
    """Central finite difference of grad_{x_t} logq, one component at a time."""
    fd = torch.empty(D, dtype=DTYPE)
    for i in range(D):
        xp, xm = xt.clone(), xt.clone()
        xp[i] += h
        xm[i] -= h
        fd[i] = (logq(xp, ca, om) - logq(xm, ca, om)) / (2.0 * h)
    return fd


# ------------------------------------------------------------------------------------
# Sweep a handful of noise levels -- early (little noise), middle, and the near-terminal
# t=999 where prediction_targets.py's 159x eps->x0 amplification lives. "A few noise
# levels", per the page's box-try text.
# ------------------------------------------------------------------------------------
SWEEP_T = [1, 100, 500, 900, 999]
REL_TOL = 1e-5  # constants: same gradient-check discipline as 14/15, applied to grad_x

print("\n" + "-" * 74)
print(f"{'t':>5}  {'abar_t':>12}  {'sqrt(1-abar_t)':>16}  {'max|fd-analytic|':>18}  {'rel L2 err':>12}")
print("-" * 74)

worst_rel = 0.0
for t in SWEEP_T:
    ca = ABAR[t].sqrt()               # sqrt(abar_t)
    om = 1.0 - ABAR[t]                # 1 - abar_t
    so = om.sqrt()                    # sqrt(1 - abar_t)

    xt = ca * x0 + so * eps           # Eq. 3.3 forward sample

    analytic = -eps / so              # closed form: -eps / sqrt(1-abar_t)
    fd = finite_diff_score(xt, ca, om)

    abs_err = (fd - analytic).abs()
    max_abs_err = abs_err.max().item()
    rel_err = (fd - analytic).norm().item() / analytic.norm().item()
    worst_rel = max(worst_rel, rel_err)

    # box-try text, verbatim pattern: assert torch.allclose(fd, -eps / (1-abar_t).sqrt())
    assert torch.allclose(fd, analytic, atol=1e-8, rtol=1e-5), \
        f"FD score must match the closed form -eps/sqrt(1-abar_t) at t={t}"
    assert rel_err < REL_TOL, f"relative error {rel_err:.2e} exceeds {REL_TOL:.0e} at t={t}"

    print(f"{t:>5}  {ABAR[t].item():>12.3e}  {so.item():>16.6f}  {max_abs_err:>18.2e}  {rel_err:>12.2e}")

print("-" * 74)
print(f"\n[OK] all {len(SWEEP_T)} noise levels: torch.allclose AND rel L2 error < {REL_TOL:.0e} in float64.")
print(f"     worst relative error across the sweep: {worst_rel:.2e}  (machine round-off, not truncation --")
print("     log q(x_t|x0) is an exact quadratic in x_t, so central FD has zero truncation term.)")
print("\nThe score is not a network's opinion about the data. It is the gradient of a Gaussian")
print("log-density, and a finite difference finds it to machine precision. 'The score identity,")
print("verified, not asserted' (p.56 box-try). Same discipline as 14_backprop_tn1.py's nine hand")
print("gradients and 15_autograd_check.py's autograd match -- just grad_x instead of grad_theta.")
