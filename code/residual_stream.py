#!/usr/bin/env python3
"""
residual_stream.py -- the gradient highway, p.34's worked number, made real.

Course artifact for p.34 (Part III, "Residuals and normalization"). Spec: D.6. NOTE ON
ATTACHMENT: as of this build, p.34's ".box try" is a live JS-only demo ("Drive the
highway yourself") and does not reference a code/ file -- this script is a standalone
Part III companion (spec-code.md D.6 calls it "ADDED BY THE ASSEMBLER", lower priority,
"note-only unless parent wants it + a page reference added"). It reproduces the page's
worked example exactly, in case a page reference gets added later.

THE STORY (p.34's "number that sells it"):
  A healthy per-layer Jacobian norm gamma=0.9, compounded through Qwen3-8B's L=36 layers
  (num_hidden_layers=36 [VP], constants.md Sec 1.1) with NO residual connection:

      gamma^L = 0.9^36 = 0.0225   -- only 2.25% of the layer-36 gradient reaches layer 0.

  Switch the residual on. Each per-layer multiplier becomes I + dF_l/dx (D-14, the
  residual WHY handed off from p.19). With a small, mean-zero learned update
  (||dF/dx|| ~= 0.1), the ACTUAL matrix product stays O(1) ~= 1.0:

      1 / 0.9^36 = 44.4x more gradient reaching layer 0, at the SAME depth.

  Push to GLM-5's reported depth, L=78 [V-sec]: 0.9^78 = 2.70e-4 vs ~1 -- about 3,700x.
  For calibration, constants.md Sec 9.3's own L=100 amplifier number: 0.9^100 = 2.7e-5 [DER].

THE ASYMMETRY THE PAGE'S LIVE DEMO USES (and this script mirrors exactly):
  - Plain (no residual) trace: a literal scalar power, gamma^l. There is nothing to
    "compute" beyond arithmetic -- that IS the no-residual story: every layer just
    shrinks the signal by the same healthy-looking factor, and healthy-looking factors
    compound into a near-zero one.
  - Residual trace: NOT a scalar. It is the actual matrix product
        prod_{l=1}^{L} (I + dF_l/dx),
    with each dF_l/dx a real d x d matrix, independently random, rescaled so its
    operator (spectral) norm is exactly ||dF/dx|| (the "small learned update" slider on
    the page, default 0.1, mean-zero). This is deliberately NOT a toy scalar: it is the
    thing the "+I term is untouched by any of the above" claim is actually claiming, and
    it is what makes "stays O(1) no matter what F does" a checkable statement rather
    than an assertion.

Usage
-----
    python residual_stream.py                    # narrated run at L=36, 78, 100
    python residual_stream.py --layers 60         # narrate one custom depth too
    python residual_stream.py --branch-norm 0.3   # the p.34 slider, 0 - 0.5 per spec
    python residual_stream.py --self-test         # assertions only, no narration, exit 0/1

SAFETY: pure NumPy, float64, CPU only. No GPU, no network, no files touched, nothing
installed. Runtime ~1s even at L=200.
"""

import argparse

import numpy as np

# --------------------------------------------------------------------------------- #
# Frozen numbers -- constants.md verbatim. [VP] = vendor-published config, [V-sec] =
# verified secondary source (brief-architectures.md), [DER] = derived from those.
# --------------------------------------------------------------------------------- #

GAMMA = 0.9                      # "healthy" per-layer Jacobian norm, p.34 worked example
L_QWEN3 = 36                     # num_hidden_layers, constants.md Sec 1.1 [VP]
L_GLM5 = 78                      # GLM-5 depth (cut from GLM-4.7's 92), brief-architectures.md [V-sec]
L_CALIBRATION = 100              # constants.md Sec 9.3's own calibration depth

AMP_0_9_36 = GAMMA ** L_QWEN3    # 0.0225 -- p.34's "2.25% of the gradient survives"
AMP_0_9_78 = GAMMA ** L_GLM5     # 2.70e-4 -- p.34's GLM-5 depth
AMP_0_9_100 = GAMMA ** L_CALIBRATION   # 2.7e-5 -- constants.md Sec 9.3 [DER], exact tag

FOLD_36 = 1.0 / AMP_0_9_36       # 44.4x -- p.34's headline fold-improvement
FOLD_78 = 1.0 / AMP_0_9_78       # ~3,700x -- p.34's GLM-5 headline

BRANCH_NORM_DEFAULT = 0.1        # ||dF/dx|| ~= 0.1, mean ~= 0 -- p.34's worked example
D_TOY = 64                       # toy bus width (free choice -- the real d=4096 lives in
                                  # 20_norm_axes.py / heads_are_a_reshape.py, not here)
SEED = 0


# --------------------------------------------------------------------------------- #
# The two traces the page's live demo plots.
# --------------------------------------------------------------------------------- #

def plain_decay(L: int, gamma: float = GAMMA) -> list:
    """No residual: the surviving-gradient ratio at depth l is the literal scalar power
    gamma^l. Index 0 is depth 0 (ratio 1.0, nothing shrunk yet); index L is the full
    stack. This is deliberately just arithmetic -- see module docstring."""
    return [gamma ** l for l in range(L + 1)]


def _spectral_rescale(M: np.ndarray, target_norm: float) -> np.ndarray:
    """Rescale M so its operator (spectral, 2-) norm is exactly target_norm. This is
    what makes ||dF/dx|| ~= 0.1 a literal, checkable property of the matrix, not a hope."""
    s = np.linalg.norm(M, 2)
    return M * (target_norm / s)


def residual_product(L: int, branch_norm: float = BRANCH_NORM_DEFAULT,
                      d: int = D_TOY, seed: int = SEED) -> list:
    """Residual: build L independent random d x d Jacobians dF_l/dx, each rescaled to
    operator norm exactly branch_norm (mean-zero: entries are N(0,1) before rescaling,
    so the rescaled matrix is mean-zero too). Track the ACTUAL matrix product
        g_l = (I + dF_l/dx)^T g_{l-1},  g_0 = ones(d)
    applied to a fixed unit-scale start vector, l = 1..L -- exactly the page's
    "prod(1 + dF_l)" computed live. Returns ||g_l|| / ||g_0|| at each l (index 0 = 1.0,
    matching plain_decay's convention so the two lists are directly comparable)."""
    rng = np.random.default_rng(seed)
    g = np.ones(d, dtype=np.float64)
    base = float(np.linalg.norm(g))
    ratios = [1.0]
    for _ in range(L):
        M = rng.normal(0.0, 1.0, size=(d, d))
        M = _spectral_rescale(M, branch_norm)     # dF_l/dx, ||.||_2 == branch_norm exactly
        g = g + M.T @ g                            # (I + dF_l/dx)^T applied -- the +I term
        ratios.append(float(np.linalg.norm(g)) / base)
    return ratios


def fold_improvement(L: int, branch_norm: float = BRANCH_NORM_DEFAULT,
                      d: int = D_TOY, seed: int = SEED, gamma: float = GAMMA) -> tuple:
    """Convenience: (plain_ratio_at_L, residual_ratio_at_L, fold) for one depth."""
    plain_L = gamma ** L
    res_L = residual_product(L, branch_norm, d, seed)[-1]
    return plain_L, res_L, res_L / plain_L


# --------------------------------------------------------------------------------- #
# Narration
# --------------------------------------------------------------------------------- #

def narrate(extra_layers: int, branch_norm: float) -> None:
    print("=" * 76)
    print("THE GRADIENT HIGHWAY -- p.34's worked number, live")
    print("=" * 76)
    print(f"  gamma (healthy per-layer Jacobian norm)   = {GAMMA}")
    print(f"  ||dF/dx|| (small learned update, mean~=0) = {branch_norm}  (p.34 slider, 0-0.5)")
    print(f"  toy bus width d = {D_TOY}  (the real bus is d=4096 -- see 20_norm_axes.py)")
    print()

    print("-" * 76)
    print(f"  {'depth L':>10}  {'what':<14}  {'no-residual (gamma^L)':>22}  "
          f"{'residual (actual prod)':>23}  {'fold':>10}")
    print("-" * 76)
    for L, tag in [(L_QWEN3, "Qwen3-8B"), (L_GLM5, "GLM-5"), (L_CALIBRATION, "calibration")]:
        plain_L, res_L, fold = fold_improvement(L, branch_norm)
        print(f"  {L:>10}  {tag:<14}  {plain_L:>22.3e}  {res_L:>23.4f}  {fold:>9,.1f}x")
    print()

    print(f"  Read the L=36 row against the page: {AMP_0_9_36:.4f} == 0.0225 (2.25% of the")
    print(f"  gradient survives, no residual). Switch the residual on: the row reads ~1.0.")
    print(f"  Fold-improvement at the SAME depth: {FOLD_36:.1f}x -- 'the number that sells it.'")
    print()
    print(f"  Push to GLM-5's depth (L={L_GLM5}, cut from GLM-4.7's 92 [V-sec]): the")
    print(f"  no-residual trace craters to {AMP_0_9_78:.2e}; the residual trace barely moves.")
    print(f"  Fold-improvement: ~{FOLD_78:,.0f}x. 'The deeper you go, the more the residual is")
    print(f"  doing' -- this is that claim, computed, not asserted.")
    print()

    if extra_layers not in (L_QWEN3, L_GLM5, L_CALIBRATION):
        plain_L, res_L, fold = fold_improvement(extra_layers, branch_norm)
        print(f"  --layers {extra_layers}: no-residual = {plain_L:.3e}, "
              f"residual = {res_L:.4f}, fold = {fold:,.1f}x")
        print()

    print("=" * 76)
    print("WHY THIS IS THE BACKWARD-PASS STORY, NOT THE FORWARD ONE (p.34 misconception #1)")
    print("=" * 76)
    print("  'Residuals help the network remember the input' is real but secondary. The")
    print("  load-bearing mechanism is backward: y = x + F(x) => dy/dx = I + dF/dx, and the")
    print("  I-term is untouched by anything F does -- it is not attenuated, not learned")
    print("  away, not optimizer-dependent. That is what the residual-column table above")
    print("  just measured: whatever the random dF_l/dx matrices did, the +I term kept the")
    print("  product near 1.0 at every depth tried.")
    print()
    print("  And p.34 misconception #2: this makes depth TRAINABLE, not automatically")
    print(f"  better. GLM-5 itself cut depth 92->78 while widening the bus (d: 5120->6144)")
    print("  [V-sec] -- a 2026 lab deciding a previous generation had overshot on depth.")


# --------------------------------------------------------------------------------- #
# Self-test -- no GPU needed, pure NumPy on CPU.
# --------------------------------------------------------------------------------- #

def self_test() -> None:
    print("Running self-checks (NumPy, CPU only, no GPU needed)...")

    # --- frozen constants match constants.md / the page, verbatim -------------------
    assert L_QWEN3 == 36, "Qwen3-8B num_hidden_layers must be 36 (constants.md Sec 1.1 [VP])"
    assert L_GLM5 == 78, "GLM-5 depth must be 78 (brief-architectures.md [V-sec])"
    assert abs(AMP_0_9_36 - 0.0225) < 1e-3, f"0.9^36 should be ~0.0225, got {AMP_0_9_36:.5f}"
    assert abs(AMP_0_9_78 - 2.70e-4) < 1e-5, f"0.9^78 should be ~2.70e-4, got {AMP_0_9_78:.3e}"
    assert abs(AMP_0_9_100 - 2.7e-5) < 1e-6, f"0.9^100 should be ~2.7e-5, got {AMP_0_9_100:e}"
    assert abs(FOLD_36 - 44.4) < 0.1, f"fold-improvement at L=36 should be ~44.4x, got {FOLD_36:.2f}"
    assert abs(FOLD_78 - 3707.5) < 5.0, f"fold-improvement at L=78 should be ~3,707x, got {FOLD_78:.1f}"
    print(f"  frozen constants (0.9^36={AMP_0_9_36:.4f}, 0.9^78={AMP_0_9_78:.3e}, "
          f"0.9^100={AMP_0_9_100:.2e}, folds 44.4x/3,700x) match constants.md / p.34 -- PASS")

    # --- plain (no-residual) trace: exact scalar power, monotonically decaying --------
    plain = plain_decay(L_QWEN3)
    assert len(plain) == L_QWEN3 + 1, "plain_decay must return one ratio per depth 0..L inclusive"
    assert plain[0] == 1.0, "depth-0 ratio must be exactly 1.0 (nothing shrunk yet)"
    assert abs(plain[-1] - AMP_0_9_36) < 1e-12, "plain_decay(36)[-1] must equal gamma^36 exactly"
    assert all(plain[i] > plain[i + 1] for i in range(len(plain) - 1)), (
        "no-residual trace must be strictly decreasing every layer (gamma < 1)"
    )
    print(f"  no-residual trace: exact gamma^l, strictly decreasing, "
          f"plain[36]={plain[-1]:.4f} -- PASS")

    # --- residual trace: THE spec's core self-check -- grad-norm stays O(1) -----------
    for L, tag in [(L_QWEN3, "L=36 (Qwen3)"), (L_GLM5, "L=78 (GLM-5)"),
                   (L_CALIBRATION, "L=100 (calibration)")]:
        ratios = residual_product(L, BRANCH_NORM_DEFAULT)
        assert len(ratios) == L + 1, f"residual_product({L}) must return L+1 ratios"
        assert ratios[0] == 1.0, "depth-0 ratio must be exactly 1.0"
        lo, hi = min(ratios), max(ratios)
        # O(1): stays within a generous 2x band of the start, at every depth tried --
        # same band convention as 19_grad_flow.py's residual self-check.
        assert 0.5 <= lo and hi <= 2.0, (
            f"residual grad-norm ratio should stay O(1) (within [0.5, 2.0]) at {tag}, "
            f"got range [{lo:.4f}, {hi:.4f}]"
        )
        assert all(np.isfinite(r) for r in ratios), f"residual ratios must never be inf/nan at {tag}"
        print(f"  residual trace at {tag}: ratio stays in [{lo:.4f}, {hi:.4f}] "
              f"(O(1), the +I term / D-14) -- PASS")

    # --- each dF_l/dx really does have operator norm == branch_norm, not approximately -
    rng = np.random.default_rng(SEED)
    M = rng.normal(0.0, 1.0, size=(D_TOY, D_TOY))
    M_scaled = _spectral_rescale(M, BRANCH_NORM_DEFAULT)
    measured = float(np.linalg.norm(M_scaled, 2))
    assert abs(measured - BRANCH_NORM_DEFAULT) < 1e-9, (
        f"rescaled Jacobian operator norm should be exactly {BRANCH_NORM_DEFAULT}, got {measured}"
    )
    print(f"  _spectral_rescale: operator norm of a rescaled random Jacobian == "
          f"{BRANCH_NORM_DEFAULT} to 1e-9 -- PASS")

    # --- residual dramatically beats plain at the SAME depth (the whole point) --------
    for L in (L_QWEN3, L_GLM5, L_CALIBRATION):
        plain_L, res_L, fold = fold_improvement(L, BRANCH_NORM_DEFAULT)
        assert res_L > plain_L, f"residual must beat plain at L={L} (got {res_L} <= {plain_L})"
        assert fold > 1.0, f"fold-improvement must exceed 1x at L={L}, got {fold}"
    print(f"  residual beats plain at every depth tried (36, 78, 100) -- PASS")

    # --- determinism: same seed, same branch_norm -> bit-identical residual trace -----
    r1 = residual_product(L_QWEN3, BRANCH_NORM_DEFAULT, seed=SEED)
    r2 = residual_product(L_QWEN3, BRANCH_NORM_DEFAULT, seed=SEED)
    assert r1 == r2, "same seed must reproduce the identical residual trace (seedable determinism)"
    print("  determinism: same seed -> bit-identical residual trace -- PASS")

    print("All self-checks PASS.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="The gradient highway: no-residual gamma^L vs the actual residual "
                     "matrix product -- p.34 companion script (spec-code.md D.6)."
    )
    ap.add_argument("--layers", type=int, default=L_QWEN3,
                     help=f"an extra custom depth to narrate alongside 36/78/100 "
                          f"(default {L_QWEN3}, i.e. no extra row)")
    ap.add_argument("--branch-norm", type=float, default=BRANCH_NORM_DEFAULT,
                     help=f"||dF/dx||, the p.34 slider, 0-0.5 (default {BRANCH_NORM_DEFAULT})")
    ap.add_argument("--self-test", action="store_true",
                     help="run assertions only, no narration, exit 0 on success")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return
    print()
    narrate(args.layers, args.branch_norm)


if __name__ == "__main__":
    main()
