#!/usr/bin/env python3
"""
19_grad_flow.py -- the SAME geometric product as init (page 19 Part A), now on the
backward pass.

Course artifact for p.19, second ".box try" (Part B: vanishing/exploding gradients;
Part C: gradient clipping). Part A of the page showed a 100-stage forward amplifier:
g=1.1 -> 1.1^100 = 13,781 (clips); g=0.9 -> 0.9^100 = 2.7e-5 (vanishes). Backprop
multiplies the SAME kind of per-layer factor, just going the other way:

    delta^(1) = ( prod_l  W^(l+1)^T diag(phi'(z^(l))) )  delta^(L)

a product of L Jacobian factors. Sigmoid's derivative is bounded by 0.25 (hit only at
z=0), so a deep sigmoid stack multiplies by <=0.25 every layer:

    0.25^10 = 9.54e-7          0.25^32 = 5.4e-20   (constants.md Sec 9.3, [DER])

The quiz on this page is explicit that 5.4e-20 is "tiny, not zero" -- it is STILL a
representable normal fp32 number (fp32's smallest normal is ~1.18e-38). That is the
textbook bound alone (max derivative, ignoring the weight matrices). This script goes
one step further than the textbook number: it runs a REAL 48-layer MLP with real
random weight matrices, whose spectral norm ALSO shrinks the signal on top of the
derivative bound, and watches the gradient norm hit an actual float32 0.0 -- not
"tiny," genuinely dead -- at just the layer the page predicts, "layer ~32."

Three stacks, one shared random input, same depth:
  sigmoid   -- too-small-scale init (the p.19 "too small" failure mode, now on
               gradients instead of activations) -> dies by layer ~32.
  relu      -- He/Kaiming init (std = sqrt(2/d), constants.md Sec 9.3's He formula)
               -> stays bounded, never underflows, but still drifts (correct init
               alone is not a guarantee at depth -- this is the setup for page 20's
               normalization and the residual fix below).
  residual  -- y = x + 0.1*F(x), Jacobian I + dF/dx (D-14 "residual WHY") -> the +I
               term is untouched by any of the above; gradient norm stays O(1) for
               all 48 layers, no matter what F does.

Then a Part C companion: a hand-rolled global-norm gradient clip, numerically
identical to `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` (constants.md
Sec 9.3: grad clip = 1.0, near-universal, [VP]) -- prints the pre-clip and post-clip
norm for an exploding case and a case already under the limit (clipping must be a
no-op there, or it is silently doing the wrong thing -- p.19's own misconception #4).
If torch happens to be importable, the SAME numbers are recomputed with the real
`clip_grad_norm_` and checked to agree, as a cross-check on the hand-rolled version.

Usage
-----
    python 19_grad_flow.py                # narrated run, all three stacks + clip demo
    python 19_grad_flow.py --layers 80    # deeper stack (still int-64 dims, ~instant)
    python 19_grad_flow.py --self-test    # assertions only, no narration, exit 0/1

SAFETY: pure NumPy, float32, CPU only. No GPU, no network, no files touched, nothing
installed. Runtime <2s. The optional torch cross-check only runs if torch is already
importable (e.g. inside ComfyUI's venv on the Spark) -- it is read-only there too.
"""

import argparse

import numpy as np

# --------------------------------------------------------------------------------- #
# Frozen numbers -- constants.md Sec 9.3, transcribed verbatim, [DER] unless noted.
# --------------------------------------------------------------------------------- #

SIGMOID_MAX_DERIV = 0.25                 # sigma'(0) = 0.25, the peak; saturates below it
AMP_0_25_10 = 0.25 ** 10                 # 9.54e-7
AMP_0_25_32 = 0.25 ** 32                 # 5.4e-20 -- "below" nothing; still representable
AMP_1_1_100 = 1.1 ** 100                 # 13,781 -- Part A's forward "clips" case
AMP_0_9_100 = 0.9 ** 100                 # 2.7e-5 -- Part A's forward "vanishes" case
AMP_0_9_36 = 0.9 ** 36                   # 0.0225 -- the residual-WHY worked number (D-14)
GRAD_CLIP_MAX_NORM = 1.0                 # constants.md Sec 9.3: "near-universal", [VP]
FP32_MIN_NORMAL = 2.0 ** -126            # 1.175e-38, for the "still representable" contrast

# --------------------------------------------------------------------------------- #
# Toy-stack config. d=64 is a free choice (this script's job is the GRADIENT geometric
# product, not the He-std=0.0221-at-d=4096 number -- that one belongs to
# 19_init_variance.py, the page's OTHER ".box try"). Depth 48 gives headroom past the
# ~32-layer death point so the "flat zero from here on" tail is visible.
# --------------------------------------------------------------------------------- #

D = 64
L_LAYERS = 48
SEED = 0
SIGMOID_INIT_SCALE = 0.1                 # deliberately too-small for sigmoid (p.19 Part A)
RELU_INIT_SCALE = (2.0 / D) ** 0.5       # He/Kaiming: std = sqrt(2/n_in), constants Sec 9.3
RESIDUAL_INIT_SCALE = 0.1
RESIDUAL_BRANCH_SCALE = 0.1              # F(x) is a SMALL learned update, spec-part3's ~0.1


# --------------------------------------------------------------------------------- #
# Activations and their derivatives (in terms of the values already computed forward,
# same shortcut backprop always uses: sigmoid' = y*(1-y), relu' = (z>0)).
# --------------------------------------------------------------------------------- #

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_layers(scale, d, layers, seed):
    """Independent random weights per layer, float32 throughout -- this is what lets a
    real underflow happen; float64 would just push the death layer much deeper."""
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0.0, scale, size=(d, d)).astype(np.float32) for _ in range(layers)]
    x0 = rng.normal(0.0, 1.0, size=d).astype(np.float32)
    x0 = (x0 / np.linalg.norm(x0)).astype(np.float32)
    return Ws, x0


def forward_backward_plain(Ws, x0, activation):
    """Plain deep MLP x_l = act(W_l @ x_{l-1}), no bias (b=0 is fine -- the random W
    already breaks symmetry, p.19's own misconception about all-zeros). Returns the
    gradient norm indexed by STEPS BACK FROM THE OUTPUT (not the forward layer index):
    norms[0] = dL/dx_L = ones (the undecayed start), norms[k] = dL/dx_{L-k} after k
    layers of backprop. This is the natural axis for '0.25^32' / 'underflows by layer
    ~32' -- a FIXED distance from the loss, independent of how deep the whole stack
    is (constants.md Sec 9.3's 0.25^10 / 0.25^32 are themselves indexed this way: 10
    or 32 factors multiplied in from the output, not from some absolute layer number)."""
    L = len(Ws)
    xs = [x0]
    zs = []
    for W in Ws:
        z = W @ xs[-1]
        zs.append(z)
        y = sigmoid(z) if activation == "sigmoid" else np.maximum(z, 0.0)
        xs.append(y.astype(np.float32))

    d = x0.shape[0]
    g = np.ones(d, dtype=np.float32)
    norms = [float(np.linalg.norm(g))]          # norms[0]: 0 steps back, at the output
    for l in range(L - 1, -1, -1):
        if activation == "sigmoid":
            y = xs[l + 1]
            deriv = (y * (1.0 - y)).astype(np.float32)
        else:
            deriv = (zs[l] > 0).astype(np.float32)
        g = (Ws[l].T @ (g * deriv)).astype(np.float32)
        norms.append(float(np.linalg.norm(g)))   # norms[k]: k steps back
    return norms


def forward_backward_residual(Ws, x0, activation, branch_scale):
    """y = x + branch_scale * F(x), F = act(W @ x). Backward:
    dL/dx_{l-1} = dL/dx_l @ (I + d F_l/dx_{l-1})
                = dL/dx_l  +  dL/dx_l @ dF_l/dx_{l-1}       -- the +I term (D-14).
    Same 'steps back from the output' indexing as forward_backward_plain, so the two
    are directly comparable index-for-index."""
    L = len(Ws)
    xs = [x0]
    zs = []
    for W in Ws:
        z = W @ xs[-1]
        zs.append(z)
        f = sigmoid(z) if activation == "sigmoid" else np.maximum(z, 0.0)
        f = (f * branch_scale).astype(np.float32)
        xs.append((xs[-1] + f).astype(np.float32))

    d = x0.shape[0]
    g = np.ones(d, dtype=np.float32)
    norms = [float(np.linalg.norm(g))]
    for l in range(L - 1, -1, -1):
        z = zs[l]
        if activation == "sigmoid":
            y = sigmoid(z)
            deriv = (y * (1.0 - y) * branch_scale).astype(np.float32)
        else:
            deriv = ((z > 0).astype(np.float32) * branch_scale)
        g = (g + Ws[l].T @ (g * deriv)).astype(np.float32)   # the "+g" IS the +I term
        norms.append(float(np.linalg.norm(g)))
    return norms


def first_underflow_layer(norms):
    """First 'steps back from the output' index where the gradient norm is an exact
    float32 0.0 -- a genuine underflow, not "small". None if it never happens."""
    for k, n in enumerate(norms):
        if n == 0.0:
            return k
    return None


# --------------------------------------------------------------------------------- #
# Part C: gradient clipping. Hand-rolled, matching torch.nn.utils.clip_grad_norm_'s
# global-norm algorithm: compute one norm across ALL parameters, rescale everything by
# the SAME factor if (and only if) that norm exceeds max_norm.
# --------------------------------------------------------------------------------- #

def clip_grad_norm_(grads, max_norm, eps=1e-6):
    """grads: list of np.ndarray (mirrors a list of .grad tensors). Mutates in place.
    Returns the PRE-clip total norm (what torch's version also returns)."""
    total_norm = float(np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads)))
    clip_coef = max_norm / (total_norm + eps)
    clip_coef_clamped = min(clip_coef, 1.0)      # never SCALE UP an under-limit gradient
    if clip_coef_clamped < 1.0:
        for g in grads:
            g *= clip_coef_clamped
    return total_norm


def torch_clip_cross_check(pre_norm_target, max_norm):
    """Optional: if torch is importable (e.g. ComfyUI's venv on the Spark), rebuild the
    SAME exploding-gradient case as real nn.Parameter .grad tensors and clip with the
    real torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) -- cross-checking the
    hand-rolled version above against the real API this page teaches. Returns None if
    torch is not importable (graceful skip, same pattern as 05_lora_from_scratch.py)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    torch.manual_seed(SEED)
    # Two parameters whose combined grad norm reproduces pre_norm_target exactly.
    p1 = nn.Parameter(torch.zeros(3))
    p2 = nn.Parameter(torch.zeros(5))
    g = torch.randn(8)
    g = g / g.norm() * pre_norm_target
    p1.grad = g[:3].clone()
    p2.grad = g[3:].clone()

    pre = float(torch.sqrt(p1.grad.pow(2).sum() + p2.grad.pow(2).sum()))
    torch.nn.utils.clip_grad_norm_([p1, p2], max_norm)
    post = float(torch.sqrt(p1.grad.pow(2).sum() + p2.grad.pow(2).sum()))
    return pre, post


# --------------------------------------------------------------------------------- #
# Narration
# --------------------------------------------------------------------------------- #

def narrate(layers: int) -> None:
    print("=" * 74)
    print("GRADIENT FLOW -- the backward-pass version of p.19's forward amplifier")
    print("=" * 74)
    print(f"  Part A (forward, on the page): g=1.1 -> 1.1^100 = {AMP_1_1_100:,.0f}  (clips)")
    print(f"                                  g=0.9 -> 0.9^100 = {AMP_0_9_100:.2e}  (vanishes)")
    print(f"  Part B (backward, THIS script): same product, delta^(1) = prod_l W^T diag(phi').")
    print(f"    max sigmoid derivative sigma'(0) = {SIGMOID_MAX_DERIV}")
    print(f"    0.25^10 = {AMP_0_25_10:.3e}   (constants.md Sec 9.3)")
    print(f"    0.25^32 = {AMP_0_25_32:.2e}   (constants.md Sec 9.3) -- 'tiny, not zero':")
    print(f"      fp32 smallest normal = {FP32_MIN_NORMAL:.2e}, and {AMP_0_25_32:.1e} is still")
    print(f"      {AMP_0_25_32 / FP32_MIN_NORMAL:.1e}x ABOVE that floor -- representable, dead in practice.")
    print(f"    D-14 residual WHY, worked number: 0.9^36 = {AMP_0_9_36:.4f} vs residual's ~1")
    print()

    Ws_sig, x0 = build_layers(SIGMOID_INIT_SCALE, D, layers, SEED)
    Ws_relu, _ = build_layers(RELU_INIT_SCALE, D, layers, SEED)
    Ws_res, _ = build_layers(RESIDUAL_INIT_SCALE, D, layers, SEED)

    norms_sig = forward_backward_plain(Ws_sig, x0, "sigmoid")
    norms_relu = forward_backward_plain(Ws_relu, x0, "relu")
    norms_res = forward_backward_residual(Ws_res, x0, "sigmoid", RESIDUAL_BRANCH_SCALE)

    dead_layer = first_underflow_layer(norms_sig)

    print("-" * 74)
    print(f"  {layers}-layer toy MLP, d={D}, float32 throughout, same random input to all three.")
    print(f"  sigmoid : too-small init (scale={SIGMOID_INIT_SCALE}) -- p.19's 'too small' failure,")
    print(f"            now on gradients instead of activations.")
    print(f"  relu    : He init (scale=sqrt(2/d)={RELU_INIT_SCALE:.4f}) -- the 'correct' fix alone.")
    print(f"  residual: y=x+{RESIDUAL_BRANCH_SCALE}*F(x), same sigmoid F -- the +I term (D-14).")
    print("-" * 74)
    header = f"  {'steps back':>10}  {'sigmoid':>12}  {'relu (He)':>12}  {'residual':>12}"
    print(header)
    print(f"  {'(from output)':>10}")
    shown_rows = list(range(0, layers + 1, max(1, layers // 12)))
    if shown_rows[-1] != layers:
        shown_rows.append(layers)
    for l in shown_rows:
        s = "0.0 (DEAD)" if norms_sig[l] == 0.0 else f"{norms_sig[l]:.3e}"
        tail = "   (all the way back, at the input)" if l == layers else ""
        print(f"  {l:>10}  {s:>12}  {norms_relu[l]:>12.3e}  {norms_res[l]:>12.4f}{tail}")
    print()

    if dead_layer is not None:
        print(f"  sigmoid gradient hits an EXACT float32 0.0 after {dead_layer} steps back from the")
        print(f"  output -- 'underflows by layer ~32' (p.19). From there on toward the input, every")
        print(f"  remaining layer gets ZERO update: a dead network, not a slow one. The textbook")
        print(f"  bound alone (0.25^32={AMP_0_25_32:.1e}) is still representable; the REAL weight")
        print(f"  matrices push it the rest of the way to exact 0.")
    else:
        print(f"  sigmoid gradient never hit exact 0.0 in {layers} layers (try --layers {layers*2}).")

    print(f"  relu (He) gradient: never underflows, stays within "
          f"[{min(norms_relu):.2e}, {max(norms_relu):.2e}] -- bounded, but NOT flat: correct init")
    print(f"  alone still drifts at depth (this is exactly what page 20's normalization and the")
    print(f"  residual below are for).")
    print(f"  residual gradient: stays within "
          f"[{min(norms_res):.4f}, {max(norms_res):.4f}] across all {layers} layers -- O(1),")
    print(f"  full stop. The +I term in dy/dx = I + dF/dx does not care what F does.")
    print()

    print("=" * 74)
    print("GRADIENT CLIPPING -- torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
    print("=" * 74)
    print(f"  constants.md Sec 9.3: global-norm clip at {GRAD_CLIP_MAX_NORM} -- 'near-universal', [VP]")
    print(f"  Order matters (p.19 misconception #4): clip AFTER backward(), BEFORE step().")
    print()

    exploding = [np.array([3.0, 4.0], dtype=np.float32), np.array([0.0, 12.0], dtype=np.float32)]
    pre_exp = clip_grad_norm_(exploding, GRAD_CLIP_MAX_NORM)
    post_exp = float(np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in exploding)))
    print(f"  case 1 (exploding): pre-clip norm = {pre_exp:.4f}  ->  post-clip norm = {post_exp:.6f}")
    print(f"    (target {GRAD_CLIP_MAX_NORM} -- direction preserved, magnitude capped)")

    under = [np.array([0.1, 0.2], dtype=np.float32), np.array([0.0, 0.05], dtype=np.float32)]
    pre_under = clip_grad_norm_(under, GRAD_CLIP_MAX_NORM)
    post_under = float(np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in under)))
    print(f"  case 2 (already under limit): pre-clip norm = {pre_under:.4f}  "
          f"->  post-clip norm = {post_under:.6f}")
    print(f"    (unchanged -- clipping a healthy gradient must be a no-op, or it's silently wrong)")
    print()

    cross = torch_clip_cross_check(pre_exp, GRAD_CLIP_MAX_NORM)
    if cross is None:
        print("  torch not importable here -- skipping the real clip_grad_norm_ cross-check.")
        print("  On the Spark (torch IS importable in ComfyUI's venv), this script re-derives the")
        print("  same pre/post numbers with the actual torch.nn.utils.clip_grad_norm_ call.")
    else:
        t_pre, t_post = cross
        print(f"  torch cross-check: torch pre={t_pre:.4f} post={t_post:.6f}  "
              f"vs hand-rolled pre={pre_exp:.4f} post={post_exp:.6f}")
        print(f"  {'AGREE' if abs(t_post - post_exp) < 1e-4 else 'DISAGREE -- investigate'}")


# --------------------------------------------------------------------------------- #
# Self-test -- no GPU needed, pure NumPy on CPU. Deterministic given SEED=0, but the
# sigmoid death-layer assertion uses a band, not exact equality: matrix-multiply
# reduction order (and hence the last-bit rounding that decides WHICH layer first
# rounds to exact 0.0) can differ by a layer or two across BLAS backends/platforms.
# The self-check is "underflows by layer ~32", not "underflows at literally 32".
# --------------------------------------------------------------------------------- #

def self_test() -> None:
    print("Running self-checks (NumPy, CPU only, no GPU needed)...")

    # --- frozen constants match constants.md Sec 9.3 verbatim -----------------------
    assert abs(AMP_0_25_10 - 9.54e-7) < 1e-8, f"0.25^10 should be ~9.54e-7, got {AMP_0_25_10:e}"
    assert abs(AMP_0_25_32 - 5.4e-20) / 5.4e-20 < 0.02, f"0.25^32 should be ~5.4e-20, got {AMP_0_25_32:e}"
    assert abs(AMP_1_1_100 - 13781) < 1.0, f"1.1^100 should be ~13,781, got {AMP_1_1_100:.1f}"
    assert abs(AMP_0_9_100 - 2.7e-5) < 1e-6, f"0.9^100 should be ~2.7e-5, got {AMP_0_9_100:e}"
    assert abs(AMP_0_9_36 - 0.0225) < 1e-3, f"0.9^36 should be ~0.0225, got {AMP_0_9_36:.5f}"
    # the "still representable" claim the quiz insists on (distractor: "it's zero" is wrong)
    assert AMP_0_25_32 > FP32_MIN_NORMAL, "0.25^32 must be ABOVE fp32 min-normal (representable, not literally 0)"
    print("  frozen constants (0.25^10, 0.25^32, 1.1^100, 0.9^100, 0.9^36) match constants.md "
          "Sec 9.3 -- PASS")

    # --- sigmoid stack: must actually underflow to exact 0.0, and land near layer 32 ---
    Ws_sig, x0 = build_layers(SIGMOID_INIT_SCALE, D, L_LAYERS, SEED)
    norms_sig = forward_backward_plain(Ws_sig, x0, "sigmoid")
    dead = first_underflow_layer(norms_sig)
    assert dead is not None, "sigmoid stack should hit an exact float32 0.0 gradient somewhere"
    assert 24 <= dead <= 40, f"sigmoid should underflow 'by layer ~32', got {dead} steps back"
    assert all(n == 0.0 for n in norms_sig[dead:]), "once dead, every later step-back must stay exactly 0.0"
    assert norms_sig[0] > 0.0, "0 steps back (the ones-vector start itself) must still be nonzero"
    print(f"  sigmoid stack: gradient underflows to exact 0.0 after {dead} steps back from the output "
          f"(band [24,40] for 'layer ~32') -- PASS")

    # --- relu (He init) stack: must NOT underflow, must stay in a sane bounded range ---
    Ws_relu, _ = build_layers(RELU_INIT_SCALE, D, L_LAYERS, SEED)
    norms_relu = forward_backward_plain(Ws_relu, x0, "relu")
    assert all(n > 0.0 for n in norms_relu), "He-init relu stack must never hit exact 0.0"
    assert all(np.isfinite(n) for n in norms_relu), "He-init relu stack must never hit inf/nan"
    assert min(norms_relu) > 1e-6, f"relu min grad norm {min(norms_relu):.2e} suspiciously close to underflow"
    assert max(norms_relu) < 1e6, f"relu max grad norm {max(norms_relu):.2e} suspiciously close to overflow"
    print(f"  relu (He init) stack: no underflow, no overflow, bounded in "
          f"[{min(norms_relu):.2e}, {max(norms_relu):.2e}] for all {L_LAYERS} layers -- PASS")

    # --- residual stack: grad norm must stay O(1) -- within a generous 2x band --------
    Ws_res, _ = build_layers(RESIDUAL_INIT_SCALE, D, L_LAYERS, SEED)
    norms_res = forward_backward_residual(Ws_res, x0, "sigmoid", RESIDUAL_BRANCH_SCALE)
    base = norms_res[0]                                   # 0 steps back = the ones-vector norm, exactly
    assert abs(base - float(np.linalg.norm(np.ones(D, dtype=np.float32)))) < 1e-4
    lo, hi = min(norms_res), max(norms_res)
    assert 0.5 * base <= lo and hi <= 2.0 * base, (
        f"residual grad norm should stay O(1) (within 2x of the start), got range [{lo:.4f}, {hi:.4f}] "
        f"vs start {base:.4f}"
    )
    print(f"  residual stack: grad norm stays in [{lo:.4f}, {hi:.4f}] across all {L_LAYERS} layers "
          f"(within 2x of the ~{base:.2f} start) -- PASS (O(1), the D-14 residual WHY)")

    # --- residual is dramatically healthier than plain sigmoid at the same layer -------
    assert norms_res[dead] > 0.0 and norms_sig[dead] == 0.0, (
        "at sigmoid's death layer, the residual stack's gradient must still be alive"
    )
    print(f"  at {dead} steps back (where sigmoid died): sigmoid grad = 0.0, residual grad = "
          f"{norms_res[dead]:.4f} -- PASS (the +I path is untouched)")

    # --- hand-rolled clip_grad_norm_: exploding case gets capped to exactly max_norm ---
    exploding = [np.array([3.0, 4.0], dtype=np.float32), np.array([0.0, 12.0], dtype=np.float32)]
    pre = clip_grad_norm_(exploding, GRAD_CLIP_MAX_NORM)
    assert abs(pre - 13.0) < 1e-4, f"pre-clip norm of [3,4,0,12] should be 13.0, got {pre}"
    post = float(np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in exploding)))
    assert abs(post - GRAD_CLIP_MAX_NORM) < 1e-4, f"post-clip norm should be ~{GRAD_CLIP_MAX_NORM}, got {post}"
    print(f"  clip (exploding case): pre={pre:.4f} -> post={post:.6f} ~= max_norm={GRAD_CLIP_MAX_NORM} -- PASS")

    # --- hand-rolled clip_grad_norm_: under-limit case must be a strict no-op ----------
    under = [np.array([0.1, 0.2], dtype=np.float32), np.array([0.0, 0.05], dtype=np.float32)]
    under_copy = [g.copy() for g in under]
    pre_u = clip_grad_norm_(under, GRAD_CLIP_MAX_NORM)
    assert pre_u < GRAD_CLIP_MAX_NORM, f"test case should already be under the limit, pre={pre_u}"
    for g, g0 in zip(under, under_copy):
        assert np.array_equal(g, g0), "clipping a gradient already under max_norm must not change it"
    print(f"  clip (under-limit case): pre={pre_u:.4f} < max_norm, values UNCHANGED -- PASS "
          f"(no-op, per p.19 misconception #4)")

    # --- optional torch cross-check, only if torch happens to be importable here ------
    cross = torch_clip_cross_check(pre, GRAD_CLIP_MAX_NORM)
    if cross is None:
        print("  torch not importable here -- skipping real clip_grad_norm_ cross-check "
              "(runs automatically wherever torch IS available, e.g. the Spark's ComfyUI venv).")
    else:
        t_pre, t_post = cross
        assert abs(t_pre - pre) < 1e-3, f"torch pre-clip norm {t_pre} should match hand-rolled {pre}"
        assert abs(t_post - post) < 1e-4, f"torch post-clip norm {t_post} should match hand-rolled {post}"
        print(f"  torch cross-check: real clip_grad_norm_ matches hand-rolled version exactly -- PASS")

    print("All self-checks PASS.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deep-MLP gradient flow: sigmoid vs relu (He) vs residual, plus a "
                     "torch.nn.utils.clip_grad_norm_ cross-check -- p.19 companion script."
    )
    ap.add_argument("--layers", type=int, default=L_LAYERS,
                     help=f"depth of the toy stack for narration (default {L_LAYERS})")
    ap.add_argument("--self-test", action="store_true",
                     help="run assertions only, no narration, exit 0 on success")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return
    print()
    narrate(args.layers)


if __name__ == "__main__":
    main()
