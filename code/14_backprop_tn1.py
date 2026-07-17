#!/usr/bin/env python3
"""
14_backprop_tn1.py -- TN-1's nine gradients, by hand, in plain NumPy. No autograd.

Course artifact for p.14 ("Backprop by Pencil: TN-1's Nine Gradients"). This is
RUNG 1 of the backprop ladder (numpy by hand) -- RUNG 2 is p.15's
15_autograd_check.py, which rebuilds this exact network in torch and asserts
its autograd grad agrees with the hand grads computed here to the last digit.

TN-1 is the course's single frozen 9-parameter network (constants.md section 8;
brief-training's rival 2-2-1 network is retired -- D-02). Architecture:

    2 -> 2 (tanh) -> 1 (sigmoid) -> BCE

    W1 = [[ 0.5, -0.3],   b1 = [ 0.1]     W2 = [0.6, -0.9]   b2 = 0.2
          [ 0.8,  0.2]]        [-0.1]

This script hard-codes that config directly from constants.md -- it does not
call any JS demo default (the page's "NN.worked221()" reproduces a RETIRED
network and must never be treated as this script's source of truth).

Two canonical inputs, both with true label y = 1, eta = 0.1 (constants section 8):

    input 1: x = [1.0, 2.0]     -- the DEAD-UNIT case (a1[0] saturates to ~0)
    input 2: x = [0.60, -0.20]  -- the GRADIENT-SPREAD case (10.22x, all nine live)

Backprop here is four equations, applied twice (once per layer), nothing more:
    BP1  dL/dz2        = yhat - y                     (sigmoid+BCE fuse to this)
    BP2  dL/dW2, dL/db2 = dL/dz2 * a1,  dL/dz2
    BP3  delta1         = (dL/dz2 * W2) * tanh'(z1)    (the chain rule through tanh)
    BP4  dL/dW1, dL/db1 = outer(delta1, x),  delta1

WARNING (constants.md section 8.4): input 1's dL/dW2[0] is the "dead unit"
gradient. It rounds to 0.0000 at four decimals, but it is NOT exactly zero in
float -- this script prints it at full precision so you can see that for
yourself, and does not claim otherwise. The float story (why, and what torch
prints instead) is page 15's beat, not this one.

SAFETY: pure NumPy, CPU only, <0.1 s, allocates a few KB. Writes and installs
nothing. No GPU, no torch, no network access -- this script has exactly one
code path and that path IS its own self-test: every run checks its own
numbers against constants.md section 8 / 8.7 below.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=False)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(W1, b1, W2, b2, x):
    """2 -> tanh(2) -> sigmoid(1) -> BCE(y=1). Returns every intermediate."""
    z1 = W1 @ x + b1
    a1 = np.tanh(z1)
    z2 = float(W2 @ a1 + b2)
    yhat = sigmoid(z2)
    loss = -np.log(yhat)          # BCE with y = 1 collapses to -log(yhat)
    return z1, a1, z2, yhat, loss


def backward(x, a1, W2, yhat):
    """The four backprop equations (BP1-BP4), applied by hand, no autograd."""
    dz2 = yhat - 1.0                       # BP1: sigmoid+BCE fuse to (yhat - y)
    dW2 = dz2 * a1                         # BP2
    db2 = dz2                              # BP2
    da1 = dz2 * W2                         # dL/da1 = dz2 * W2  (chain through the linear layer)
    tanh_prime = 1.0 - a1 ** 2             # tanh'(z) = 1 - tanh(z)^2 = 1 - a1^2
    delta1 = da1 * tanh_prime              # BP3
    dW1 = np.outer(delta1, x)              # BP4
    db1 = delta1                           # BP4
    return dz2, dW2, db2, da1, tanh_prime, delta1, dW1, db1


def sgd_step(W1, b1, W2, b2, dW1, db1, dW2, db2, eta=0.1):
    return W1 - eta * dW1, b1 - eta * db1, W2 - eta * dW2, b2 - eta * db2


# --------------------------------------------------------------------------- #
# TN-1's nine frozen parameters -- constants.md section 8. Hard-coded, no
# defaults, no JS demo object. Do not recompute, do not round differently.
# --------------------------------------------------------------------------- #
W1 = np.array([[0.5, -0.3], [0.8, 0.2]])
b1 = np.array([0.1, -0.1])
W2 = np.array([0.6, -0.9])
b2 = 0.2
ETA = 0.1

print("=" * 72)
print("14_backprop_tn1.py -- TN-1's nine gradients by hand (NumPy, no autograd)")
print(f"numpy {np.__version__}")
print("=" * 72)
print("\nTN-1: 2 -> tanh(2) -> sigmoid(1) -> BCE, 9 parameters, frozen (constants sec 8).")
print(f"  W1 = {W1.tolist()}   b1 = {b1.tolist()}")
print(f"  W2 = {W2.tolist()}   b2 = {b2}")


# =========================================================================== #
# INPUT 1 -- x = [1.0, 2.0], the dead-unit case (constants section 8.1-8.5)
# =========================================================================== #
print("\n" + "-" * 72)
print("INPUT 1: x = [1.0, 2.0], y = 1  --  the DEAD-UNIT case")
print("-" * 72)

x1 = np.array([1.0, 2.0])
z1_1, a1_1, z2_1, yhat_1, loss_1 = forward(W1, b1, W2, b2, x1)

print(f"\n  z1   = {z1_1}")
print(f"  a1   = {a1_1}")
print(f"  z2   = {z2_1:.4f}")
print(f"  yhat = {yhat_1:.4f}")
print(f"  loss = {loss_1:.4f}")

# forward self-checks -- constants.md section 8.1
assert abs(z1_1[1] - 1.1) < 1e-9, "z1[1] should be 1.1"
assert abs(a1_1[1] - 0.8005) < 1e-4, "a1[1] should be 0.8005"
assert abs(z2_1 - (-0.5204)) < 1e-4, f"z2 should be -0.5204 (section 8.1, the corrected rounding), got {z2_1:.4f}"
assert abs(yhat_1 - 0.3727) < 1e-4, "yhat should be 0.3727"
assert abs(loss_1 - 0.9869) < 1e-4, "loss should be 0.9869 (section 8.1, SIX corrections apply -- 0.9870 is stale)"
print("  [OK] forward matches constants section 8.1 (loss = 0.9869, not the stale 0.9870).")

# --- z1[0] is where the "exactly zero" trap lives. Show it honestly. ---
print(f"\n  z1[0] at full precision = {z1_1[0]!r}")
print("  0.5*1.0 - 0.3*2.0 + 0.1 rounds to 0.0000 at four decimals, but the value")
print("  above is NOT the literal float 0.0 -- binary non-associativity (constants")
print("  section 8.4). This script does not claim otherwise; the full float story")
print("  ('the math on paper and the math in the machine are not the same math')")
print("  is page 15's beat, not this one.")

dz2_1, dW2_1, db2_1, da1_1, tanhp_1, delta1_1, dW1_1, db1_1 = backward(x1, a1_1, W2, yhat_1)

print(f"\n  dL/dz2      = {dz2_1:.4f}")
print(f"  dL/dW2      = {dW2_1}")
print(f"  dL/db2      = {db2_1:.4f}")
print(f"  dL/da1      = {da1_1}")
print(f"  tanh'(z1)   = {tanhp_1}")
print(f"  delta1      = {delta1_1}")
print(f"  dL/dW1      = {dW1_1}")
print(f"  dL/db1      = {db1_1}")

# WARNING (spec-code.md p.14 entry): do NOT print dL/dW2[0] as "exactly 0".
print(f"\n  dL/dW2[0] at full precision = {dW2_1[0]!r}  <- rounds to 0.0000, is NOT")
print("  the literal float 0.0 (this is the dead-unit gradient: a1[0] ~ tanh(z1[0])")
print("  is tiny but not zero, and tanh'(0) = 1, so the row above it still moves).")

# backward self-checks -- constants.md section 8.2 (the six-corrected roundings)
assert abs(dz2_1 - (-0.6273)) < 1e-4, "dL/dz2 should be -0.6273"
assert abs(dW2_1[1] - (-0.5021)) < 1e-4, "dL/dW2[1] should be -0.5021 (corrected from -0.5022)"
assert abs(da1_1[1] - 0.5645) < 1e-4, "dL/da1[1] should be 0.5645 (corrected from 0.5646)"
assert abs(tanhp_1[1] - 0.3592) < 1e-4, "tanh'(1.1) should be 0.3592"
assert abs(delta1_1[0] - (-0.3764)) < 1e-4 and abs(delta1_1[1] - 0.2028) < 1e-4, "delta1 should be [-0.3764, 0.2028]"
assert np.allclose(dW1_1, np.array([[-0.3764, -0.7527], [0.2028, 0.4056]]), atol=1e-4), \
    f"dL/dW1 should be [[-0.3764,-0.7527],[0.2028,0.4056]] (corrected from -0.7528), got {dW1_1}"
assert np.allclose(db1_1, delta1_1), "dL/db1 == delta1 (BP4)"
print("  [OK] backward matches constants section 8.2 to the six corrected roundings")
print("       (W1[0][1] = -0.7527, not the brief's stale -0.7528).")

# --- one SGD step, eta = 0.1 (constants section 8.3) ---
W1p, b1p, W2p, b2p = sgd_step(W1, b1, W2, b2, dW1_1, db1_1, dW2_1, db2_1, ETA)
_, _, z2_1p, yhat_1p, loss_1p = forward(W1p, b1p, W2p, b2p, x1)
delta_loss_1 = loss_1p - loss_1

print(f"\n  after one SGD step (eta = {ETA}):")
print(f"    W1' = {W1p.tolist()}")
print(f"    b1' = {b1p.tolist()}")
print(f"    W2' = {W2p.tolist()}   b2' = {b2p:.4f}")
print(f"    z2 (after)   = {z2_1p:.4f}")
print(f"    yhat (after) = {yhat_1p:.4f}")
print(f"    loss (after) = {loss_1p:.4f}   (delta = {delta_loss_1:+.4f})")

assert abs(z2_1p - (-0.2434)) < 1e-4, "z2 after step should be -0.2434 (corrected from -0.2433)"
assert abs(yhat_1p - 0.4395) < 1e-4, "yhat after step should be 0.4395"
assert abs(loss_1p - 0.8222) < 1e-4, "loss after step should be 0.8222"
assert abs(delta_loss_1 - (-0.1646)) < 1e-4, "delta loss should be -0.1646 (corrected from -0.1648)"
print("  [OK] one SGD step: loss 0.9869 -> 0.8222 (delta -0.1646). It learned, by pencil.")


# =========================================================================== #
# INPUT 2 -- x = [0.60, -0.20], the gradient-spread case (constants section 8.7)
# =========================================================================== #
print("\n" + "-" * 72)
print("INPUT 2: x = [0.60, -0.20], y = 1  --  the GRADIENT-SPREAD case (no dead unit)")
print("-" * 72)

x2 = np.array([0.60, -0.20])
z1_2, a1_2, z2_2, yhat_2, loss_2 = forward(W1, b1, W2, b2, x2)

print(f"\n  z1   = {z1_2}")
print(f"  a1   = {a1_2}")
print(f"  z2   = {z2_2:.4f}")
print(f"  yhat = {yhat_2:.4f}")
print(f"  loss = {loss_2:.4f}")

assert np.allclose(z1_2, [0.46, 0.34], atol=1e-9), "z1 should be [0.46, 0.34]"
assert np.allclose(a1_2, [0.4301, 0.3275], atol=1e-4), "a1 should be [0.4301, 0.3275]"
assert abs(z2_2 - 0.1633) < 1e-4, "z2 should be 0.1633"
assert abs(yhat_2 - 0.5407) < 1e-4, "yhat should be 0.5407"
assert abs(loss_2 - 0.6148) < 1e-4, "loss should be 0.6148"
print("  [OK] forward matches constants section 8.7. No dead unit -- both tanh' are healthy.")

dz2_2, dW2_2, db2_2, da1_2, tanhp_2, delta1_2, dW1_2, db1_2 = backward(x2, a1_2, W2, yhat_2)

print(f"\n  dL/dz2      = {dz2_2:.4f}")
print(f"  dL/da1      = {da1_2}")
print(f"  tanh'(z1)   = {tanhp_2}")
print(f"  delta1      = {delta1_2}")
print(f"  dL/dW1      = {dW1_2}")
print(f"  dL/db1      = {db1_2}")
print(f"  dL/dW2      = {dW2_2}")
print(f"  dL/db2      = {db2_2:.4f}")

assert abs(dz2_2 - (-0.4593)) < 1e-4, "dL/dz2 should be -0.4593"
assert np.allclose(da1_2, [-0.2756, 0.4133], atol=1e-4), "dL/da1 should be [-0.2756, 0.4133]"
assert np.allclose(tanhp_2, [0.8150, 0.8928], atol=1e-4), "tanh' should be [0.8150, 0.8928]"
assert np.allclose(delta1_2, [-0.2246, 0.3690], atol=1e-4), "delta1 should be [-0.2246, 0.3690]"
assert np.allclose(dW1_2, [[-0.1348, 0.0449], [0.2214, -0.0738]], atol=1e-4), \
    f"dL/dW1 should be [[-0.1348,0.0449],[0.2214,-0.0738]], got {dW1_2}"
assert np.allclose(db1_2, [-0.2246, 0.3690], atol=1e-4), "dL/db1 should be [-0.2246, 0.3690]"
assert np.allclose(dW2_2, [-0.1975, -0.1504], atol=1e-4), "dL/dW2 should be [-0.1975, -0.1504]"
assert abs(db2_2 - (-0.4593)) < 1e-4, "dL/db2 should be -0.4593"
print("  [OK] backward matches constants section 8.7's nine FROZEN gradients (all live).")

# --- THE SPREAD BEAT: max|g| / min|g| over all nine live gradients = 10.22x ---
nine_grads = np.concatenate([dW1_2.ravel(), db1_2, dW2_2, [db2_2]])
assert nine_grads.size == 9, f"expected nine gradients, got {nine_grads.size}"
spread = np.max(np.abs(nine_grads)) / np.min(np.abs(nine_grads))
argmax_name = "dL/db2"
argmin_name = "dL/dW1[0][1]"

print(f"\n  nine gradients (|g|, sorted): {np.sort(np.abs(nine_grads))}")
print(f"  spread = max|g| / min|g| = {np.max(np.abs(nine_grads)):.6f} / {np.min(np.abs(nine_grads)):.6f}"
      f" = {spread:.4f}x")
print(f"    largest  = {argmax_name}  = {db2_2:+.4f}")
print(f"    smallest = {argmin_name}  = {dW1_2[0, 1]:+.4f}")
print("  The largest gradient is ten times the smallest -- one learning rate cannot")
print("  serve all nine well. This is the honest replacement for the retired 12.7x")
print("  beat (constants section 8.6): structural, survives normalization, and it")
print("  is why page 17 reaches for Adam.")

assert abs(spread - 10.2246) < 1e-3, f"spread should be 10.2246x (constants section 8.7), got {spread:.4f}x"
print("  [OK] spread = 10.2246x, matching constants section 8.7 -- the honest gradient-spread beat.")

# --- one SGD step, eta = 0.1 (constants section 8.7) ---
W1p2, b1p2, W2p2, b2p2 = sgd_step(W1, b1, W2, b2, dW1_2, db1_2, dW2_2, db2_2, ETA)
_, _, z2_2p, yhat_2p, loss_2p = forward(W1p2, b1p2, W2p2, b2p2, x2)
delta_loss_2 = loss_2p - loss_2

print(f"\n  after one SGD step (eta = {ETA}):")
print(f"    yhat (after) = {yhat_2p:.4f}")
print(f"    loss (after) = {loss_2p:.4f}   (delta = {delta_loss_2:+.4f})")

assert abs(yhat_2p - 0.5695) < 1e-4, "yhat after step should be 0.5695"
assert abs(loss_2p - 0.5630) < 1e-4, "loss after step should be 0.5630"
assert abs(delta_loss_2 - (-0.0518)) < 1e-4, "delta loss should be -0.0518"
print("  [OK] one SGD step: loss 0.6148 -> 0.5630 (delta -0.0518).")

print("\n" + "=" * 72)
print("All assertions passed. Same nine equations (BP1-BP4), two inputs:")
print("input 1 has a dead unit and a float that isn't quite zero; input 2 has no")
print("dead unit and a 10.22x gradient spread that survives normalization.")
print("Page 15 rebuilds this exact network in torch and checks autograd against")
print("every hand gradient printed above, to the last digit.")
print("=" * 72)
