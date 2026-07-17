#!/usr/bin/env python3
"""
15_autograd_check.py -- TN-1's nine gradients again, this time from torch
autograd, reconciled with page 14's pencil numbers to the last decimal --
and the one place they disagree on purpose.

Course artifact for p.15 ("Autograd + the float that isn't zero"). This is
RUNG 2 of the backprop ladder -- RUNG 1 was p.14's 14_backprop_tn1.py, pure
NumPy, no autograd. This script rebuilds the identical TN-1 network as an
nn.Sequential, sets the same frozen weights, calls loss.backward(), and
asserts the result against RUNG 1's hand gradients (constants.md section 8).

TN-1 (constants.md section 8; brief-training's rival 2-2-1 network is
RETIRED -- D-02). Architecture: 2 -> 2 (tanh) -> 1 (sigmoid) -> BCE.

    W1 = [[ 0.5, -0.3],   b1 = [ 0.1]     W2 = [0.6, -0.9]   b2 = 0.2
          [ 0.8,  0.2]]        [-0.1]

Hard-coded directly from constants.md, exactly as p.14 does -- never from
the page's "NN.worked221()" JS default (that reproduces the RETIRED network).

Two canonical inputs, both y = 1, eta = 0.1:
    input 1: x = [1.0, 2.0]     -- the DEAD-UNIT case. Owns the mandated
                                    "1.4020191230201817e-08 float that isn't
                                    zero" beat (constants.md section 8.4) and
                                    the atol=1e-4 / -0.7527 assertion (8.5).
    input 2: x = [0.60, -0.20]  -- the GRADIENT-SPREAD case, 10.22x, all nine
                                    live. Asserted at 5 d.p., atol=1e-5 (8.7)
                                    -- reusing input 1's looser 1e-4 here would
                                    repeat the same tolerance-coincidence trap
                                    section 8.5 warns against.

RUNG 2's whole point (page 15's PREDICT box): on paper, input 1's
dL/dW2[0] is exactly 0 (the dead unit, a1[0]=tanh(0)=0). Torch prints
1.4e-08. Both are right -- 0.5 - 0.6 + 0.1 is not zero in binary floating
point, and this network hits that non-associativity residual for free.

SAFETY: this network has 9 parameters and both inputs are length-2 vectors --
forward+backward is microseconds. No GPU needed, no CUDA calls anywhere in
this script; the real path below uses torch on CPU only. Writes and installs
nothing.

Usage
-----
    python 15_autograd_check.py             # needs torch (CPU is enough, no GPU)
    python 15_autograd_check.py --self-test # no torch needed -- numpy float32
                                             # independently reproduces the
                                             # mandated 1.4020191230201817e-08
                                             # and both assertion margins
"""

import argparse
import sys


# --------------------------------------------------------------------------- #
# TN-1's nine frozen parameters -- constants.md section 8. Hard-coded, no
# defaults, no JS demo object. Do not recompute, do not round differently.
# --------------------------------------------------------------------------- #
W1_LIST = [[0.5, -0.3], [0.8, 0.2]]
B1_LIST = [0.1, -0.1]
W2_LIST = [0.6, -0.9]
B2_VAL = 0.2
ETA = 0.1

X1 = [1.0, 2.0]      # dead-unit case
X2 = [0.60, -0.20]   # gradient-spread case

D13_FRAMING = """\
  By hand you got exactly 0. Torch prints 1.4e-08. Both are right. 0.5 - 0.6
  + 0.1 is not zero in binary floating point -- it is the textbook
  non-associativity demo, and you just hit it for free. This is the lesson
  of the logsumexp page come back around: the math on paper and the math in
  the machine are not the same math. The gradient is zero to seven decimals,
  the unit is dead, and the lesson stands -- but the number on your screen
  will not be 0.0, and a course that told you it would be is a course that
  has never run its own code.
"""


# --------------------------------------------------------------------------- #
# Self-test -- no torch required. Independently reproduces, in plain NumPy
# at float32, the exact bit values torch prints (matmul on a 2-vector is the
# same IEEE-754 operations either way), plus the assertion-margin arithmetic
# constants.md sections 8.5 and 8.7 document. Run this on any box, no GPU.
# --------------------------------------------------------------------------- #
def self_test():
    import numpy as np

    print("=" * 72)
    print("15_autograd_check.py --self-test  (numpy float32 stand-in, no torch)")
    print(f"numpy {np.__version__}")
    print("=" * 72)

    W1 = np.array(W1_LIST, dtype=np.float32)
    b1 = np.array(B1_LIST, dtype=np.float32)
    W2 = np.array(W2_LIST, dtype=np.float32)
    b2 = np.float32(B2_VAL)

    def forward32(x):
        z1 = W1 @ x + b1
        a1 = np.tanh(z1)
        z2 = np.float32(W2 @ a1 + b2)
        yhat = np.float32(1.0 / (1.0 + np.exp(-z2)))
        return z1, a1, z2, yhat

    def backward32(x, a1, yhat):
        dz2 = np.float32(yhat - np.float32(1.0))
        dW2 = dz2 * a1
        db2 = dz2
        da1 = dz2 * W2
        tanhp = np.float32(1.0) - a1 ** 2
        delta1 = da1 * tanhp
        dW1 = np.outer(delta1, x)
        db1 = delta1
        return dz2, dW2, db2, dW1, db1

    # --- 1. the float64-vs-float32 non-associativity demo, constants sec 8.4 ---
    print("\n-- 1. 0.5 - 0.6 + 0.1: on paper it's 0, on a machine it isn't --")
    z11_f64 = 0.5 * 1.0 + (-0.3) * 2.0 + 0.1              # Python default = float64
    z11_f32 = np.float32(0.5) * np.float32(1.0) + np.float32(-0.3) * np.float32(2.0) + np.float32(0.1)
    print(f"  z_11 = 0.5*1.0 + (-0.3)*2.0 + 0.1")
    print(f"    float64 -> {z11_f64!r}")
    print(f"    float32 -> {float(z11_f32)!r}")
    assert z11_f64 == 2.7755575615628914e-17, f"float64 residual drifted: {z11_f64!r}"
    assert float(z11_f32) == -2.2351741790771484e-08, f"float32 residual drifted: {float(z11_f32)!r}"
    print("  [OK] matches constants.md section 8.4's mandated residuals.")

    # --- 2. full float32 forward+backward, input 1 -- reproduces torch's own ---
    #     dW2 bit-for-bit, because matmul on a length-2 vector is the same
    #     IEEE-754 sequence of ops whichever framework performs it.
    print("\n-- 2. propagate that residual through the whole network (input 1) --")
    x1 = np.array(X1, dtype=np.float32)
    z1_1, a1_1, z2_1, yhat_1 = forward32(x1)
    dz2_1, dW2_1, db2_1, dW1_1, db1_1 = backward32(x1, a1_1, yhat_1)
    print(f"  dW2 (float32) = [{float(dW2_1[0])!r}, {float(dW2_1[1])!r}]")
    mandated_dw2 = (1.4020191230201817e-08, -0.5021151900291443)
    assert float(dW2_1[0]) == mandated_dw2[0], (
        f"dW2[0] should reproduce torch's mandated {mandated_dw2[0]!r}, got {float(dW2_1[0])!r}")
    assert abs(float(dW2_1[1]) - mandated_dw2[1]) < 1e-7, (
        f"dW2[1] should be near {mandated_dw2[1]!r}, got {float(dW2_1[1])!r}")
    print(f"  [OK] bit-for-bit reproduction of the mandated torch float32 dW2 =")
    print(f"       {list(mandated_dw2)}")
    print(D13_FRAMING)

    # --- 3. the assertion-margin arithmetic, constants section 8.5 ---
    print("-- 3. why input 1 asserts at atol=1e-4, and why -0.7527 not -0.7528 --")
    exact_w1_01 = -0.7527033  # constants.md section 8.2, 7-s.f. exact
    err_correct = abs(-0.7527 - exact_w1_01)
    err_stale = abs(-0.7528 - exact_w1_01)
    budget = 1e-4 * (2 ** 0.5)  # atol as used by torch.allclose is per-element; report the raw atol
    print(f"  |exact - (-0.7527)| = {err_correct:.3e}   (corrected rounding)")
    print(f"  |exact - (-0.7528)| = {err_stale:.3e}   (brief's stale rounding)")
    assert err_correct < 1e-5, "corrected rounding should carry a wide margin under atol=1e-4"
    assert 9e-5 < err_stale < 1e-4, "stale rounding should just barely survive atol=1e-4 (~10% margin)"
    print("  [OK] -0.7528 survives atol=1e-4 only by a ~10% margin (a tolerance")
    print("       coincidence); -0.7527 survives by ~30x. Fixing the rounding is")
    print("       what makes this assertion mean something.")

    # --- 4. full float32 forward+backward, input 2 -- the 10.22x spread beat ---
    print("\n-- 4. input 2 (no dead unit) -- why it asserts at 5 d.p., atol=1e-5 --")
    x2 = np.array(X2, dtype=np.float32)
    z1_2, a1_2, z2_2, yhat_2 = forward32(x2)
    dz2_2, dW2_2, db2_2, dW1_2, db1_2 = backward32(x2, a1_2, yhat_2)
    print(f"  dL/dW1 (float32) = {dW1_2}")
    mandated_dw1_2 = [[-0.13475, 0.04492], [0.22140, -0.07380]]
    err_4dp = max(abs(float(dW1_2[0, 1]) - 0.0449), 0)
    print(f"  smallest-magnitude entry dW1[0][1] = {float(dW1_2[0,1]):.6f}")
    assert abs(float(dW1_2[0, 1]) - 0.0449) < 5e-5, "4-d.p. margin claim depends on this value"
    print(f"    at 4 d.p. (0.0449) margin to the float value is only ~{err_4dp:.1e} --")
    print(f"    ~2x headroom on a small-magnitude gradient (constants section 8.7's")
    print(f"    warning). At 5 d.p., atol=1e-5 restores real headroom:")
    assert all(
        abs(float(dW1_2[i, j]) - mandated_dw1_2[i][j]) < 1e-5
        for i in range(2) for j in range(2)
    ), f"dW1 (input 2) should match {mandated_dw1_2} at atol=1e-5, got {dW1_2}"
    print(f"  [OK] {mandated_dw1_2} matches at atol=1e-5.")

    # --- 5. argparse sanity (this flag is the one that got us here) ---
    print("\n-- 5. argument parsing --")
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    parsed = ap.parse_args(["--self-test"])
    assert parsed.self_test is True
    print("  [OK] --self-test parses.")

    print("\n" + "=" * 72)
    print("All self-test assertions passed -- no torch needed. The real script")
    print("(no flags, torch installed) rebuilds this in nn.Sequential and prints")
    print("the identical numbers from actual autograd, not a numpy stand-in.")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# The real thing -- torch nn.Sequential + autograd. CPU is enough; no CUDA
# call anywhere below. This is what the page's "Try it" box promises.
# --------------------------------------------------------------------------- #
def run_torch():
    import torch
    import torch.nn as nn

    torch.manual_seed(0)  # irrelevant to the math (all weights are set explicitly) but keeps the run deterministic

    print("=" * 72)
    print("15_autograd_check.py -- TN-1's nine gradients, via torch autograd")
    print(f"torch {torch.__version__}")
    print("=" * 72)

    # model[0] = Linear (W1,b1) -> model[1] = Tanh -> model[2] = Linear (W2,b2)
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Tanh(),
        nn.Linear(2, 1),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(W1_LIST, dtype=torch.float32))
        model[0].bias.copy_(torch.tensor(B1_LIST, dtype=torch.float32))
        model[2].weight.copy_(torch.tensor([W2_LIST], dtype=torch.float32))
        model[2].bias.copy_(torch.tensor([B2_VAL], dtype=torch.float32))

    print("\nTN-1 as nn.Sequential(Linear(2,2), Tanh(), Linear(2,1)); sigmoid+BCE")
    print("fused via -log(sigmoid(z2)) for y=1 (BP1 identity dL/dz2 = yhat - y).")
    print(f"  Linear[0].weight = {model[0].weight.tolist()}   .bias = {model[0].bias.tolist()}")
    print(f"  Linear[2].weight = {model[2].weight.tolist()}   .bias = {model[2].bias.tolist()}")

    def step(x_list, label):
        model.zero_grad(set_to_none=True)  # page 15's own zero_grad lesson, applied to itself
        x = torch.tensor(x_list, dtype=torch.float32).unsqueeze(0)  # (1,2)
        z2 = model(x)                          # (1,1), pre-sigmoid logit
        yhat = torch.sigmoid(z2)
        loss = -torch.log(yhat).squeeze()      # BCE, y = 1
        loss.backward()
        print(f"\n  {label}")
        print(f"    yhat = {yhat.item():.4f}   loss = {loss.item():.4f}")
        return yhat.item(), loss.item()

    # ===================================================================== #
    # INPUT 1 -- x = [1.0, 2.0], the dead-unit case
    # ===================================================================== #
    print("\n" + "-" * 72)
    print("INPUT 1: x = [1.0, 2.0], y = 1  --  the DEAD-UNIT case")
    print("-" * 72)
    yhat_1, loss_1 = step(X1, "torch forward")
    assert abs(yhat_1 - 0.3727) < 1e-4, "yhat should match constants section 8.1"
    assert abs(loss_1 - 0.9869) < 1e-4, "loss should match constants section 8.1"

    grad_w1_1 = model[0].weight.grad
    grad_w2_1 = model[2].weight.grad
    print(f"\n  dL/dW1 (torch) =\n{grad_w1_1}")
    print(f"  dL/dW2 (torch) = {grad_w2_1}")

    # ---- the mandated assertion, input 1: atol=1e-4, note -0.7527 not -0.7528 ----
    assert torch.allclose(
        model[0].weight.grad,
        torch.tensor([[-0.3764, -0.7527], [0.2028, 0.4056]]), atol=1e-4
    ), f"input-1 dL/dW1 should match p.14's pencil grads at atol=1e-4, got {grad_w1_1}"
    print("  [OK] torch dL/dW1 matches page 14's pencil grads at atol=1e-4")
    print("       (note -0.7527, not the brief's stale -0.7528 -- constants sec 8.5).")

    # ---- THE mandated beat: the "exactly zero" print, and why it isn't ----
    dw2_first = grad_w2_1.flatten()[0].item()
    dw2_second = grad_w2_1.flatten()[1].item()
    print(f"\n  torch float32 dW2 -> [{dw2_first!r}, {dw2_second!r}]")
    print("  On paper (page 14) this first entry is exactly 0 -- the dead unit,")
    print("  a1[0] = tanh(0) = 0. Torch just printed something else. Both are right:")
    print(D13_FRAMING)
    assert dw2_first != 0.0, "the whole point of this beat is that it is NOT literal 0.0"
    assert abs(dw2_first - 1.4020191230201817e-08) < 1e-9, (
        f"mandated float32 dW2[0] should be ~1.4020191230201817e-08, got {dw2_first!r}")
    print("  [OK] matches the mandated constants.md section 8.4 value to <1e-9.")

    # ===================================================================== #
    # INPUT 2 -- x = [0.60, -0.20], the gradient-spread case
    # ===================================================================== #
    print("\n" + "-" * 72)
    print("INPUT 2: x = [0.60, -0.20], y = 1  --  the GRADIENT-SPREAD case")
    print("-" * 72)
    yhat_2, loss_2 = step(X2, "torch forward")
    assert abs(yhat_2 - 0.5407) < 1e-4, "yhat should match constants section 8.7"
    assert abs(loss_2 - 0.6148) < 1e-4, "loss should match constants section 8.7"

    grad_w1_2 = model[0].weight.grad
    print(f"\n  dL/dW1 (torch) =\n{grad_w1_2}")

    # ---- the mandated assertion, input 2: 5 d.p., atol=1e-5 -- NOT 1e-4 ----
    assert torch.allclose(
        model[0].weight.grad,
        torch.tensor([[-0.13475, 0.04492], [0.22140, -0.07380]]), atol=1e-5
    ), f"input-2 dL/dW1 should match p.14's pencil grads at atol=1e-5, got {grad_w1_2}"
    print("  [OK] torch dL/dW1 matches page 14's pencil grads at atol=1e-5, 5 d.p.")
    print("       (input 1's looser atol=1e-4 would pass here too, by only a ~2x")
    print("       margin on the smallest gradient -- the same coincidence trap")
    print("       section 8.5 warns about. This input asserts tighter on purpose.)")

    print("\n" + "=" * 72)
    print("Both inputs verified against page 14's pencil gradients. RUNG 1 (numpy")
    print("by hand) and RUNG 2 (torch autograd) agree to the assertion's last digit")
    print("-- except for the one float that isn't zero, which is the whole page.")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                     help="run the numpy-only stand-in (no torch, no GPU needed)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not importable here. This script needs torch (CPU is enough --")
        print("no CUDA call in it), e.g. the Spark's ComfyUI venv:")
        print("  ~/ComfyUI/.venv/bin/python 15_autograd_check.py")
        print()
        print("To verify the math right now without torch, run:")
        print("  python 15_autograd_check.py --self-test")
        sys.exit(1)

    run_torch()


if __name__ == "__main__":
    main()
