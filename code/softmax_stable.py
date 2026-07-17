#!/usr/bin/env python3
"""
softmax_stable.py -- softmax two ways, and why only one survives a big logit.

Course artifact for p.09 ("Logs, Softmax & Cross-Entropy"). This is the at-your-
keyboard version of the page's Demo 1 softmax strip: the naive definition and the
max-subtracted "stable" one are the SAME algebra, but on a large logit the naive
one overflows to nan and the stable one is fine.

  1. Both softmaxes on the canonical logits [2.0, 1.0, 0.1] (constants.md
     section 9.2). They AGREE: p_0 = 0.659001, L = -ln(p_0) = 0.417030 nats,
     and the softmax+CE gradient p - y = [-0.340999, +0.242433, +0.098566]
     sums to exactly 0.
  2. Both softmaxes on [800., 1., 1.]. Naive exp/sum(exp) does exp(800) = inf,
     then inf/inf = nan. The stable one subtracts the max first (an EXACT shift,
     not an approximation) so the biggest exponent is e^0 = 1 -- correct answer.
  3. F.cross_entropy(logits, target) reproduces L = 0.417030 straight from the
     LOGITS. It fuses softmax + log + NLL with the log-sum-exp trick inside, so
     you never hand it probabilities -- that split is the real bug this page warns
     about.

SAFETY: CPU only, <1 s, allocates a few KB. Writes and installs nothing.
"""

import torch
import torch.nn.functional as F


def softmax_naive(z):
    e = torch.exp(z)              # exp(800) -> inf
    return e / e.sum()           # inf / inf -> nan


def softmax_stable(z):
    z = z - z.max()              # exact shift; biggest exponent is e^0 = 1
    e = torch.exp(z)
    return e / e.sum()


print("=" * 68)
print("softmax_stable.py -- same math, one survives a big logit")
print(f"torch {torch.__version__}")
print("=" * 68)

# --------------------------------------------------------------------------
# 1. Canonical logits: the two softmaxes agree (constants.md section 9.2)
# --------------------------------------------------------------------------
z = torch.tensor([2.0, 1.0, 0.1])
p_naive = softmax_naive(z)
p = softmax_stable(z)
print("\n-- canonical logits z = [2.0, 1.0, 0.1] --")
print(f"naive : {p_naive.tolist()}")
print(f"stable: {p.tolist()}")           # [0.659001, 0.242433, 0.098566]

L = (-torch.log(p[0])).item()            # true class is 0
print(f"p_0 = {p[0].item():.6f}   L = -ln(p_0) = {L:.6f} nats")

y = torch.tensor([1., 0., 0.])           # one-hot true class 0
grad = p - y                             # the softmax+CE gradient, "predicted - actual"
print(f"gradient p - y = {grad.tolist()}  (sums to {grad.sum().item():.6f})")

assert torch.allclose(p, p_naive, atol=1e-6), "on friendly logits both agree"
assert abs(p[0].item() - 0.659001) < 1e-5, f"p_0 should be 0.659001, got {p[0]}"
assert abs(L - 0.417030) < 1e-5, f"L should be 0.417030, got {L}"
assert abs(grad.sum().item()) < 1e-6, "p - y sums to 0 (softmax and one-hot both sum to 1)"

# --------------------------------------------------------------------------
# 2. A big logit: naive overflows to nan, stable is fine
# --------------------------------------------------------------------------
big = torch.tensor([800., 1., 1.])
p_big_naive = softmax_naive(big)         # exp(800) = inf -> inf/inf = nan
p_big_stable = softmax_stable(big)       # max-subtracted -> [1., 0., 0.]
print("\n-- big logit z = [800., 1., 1.] --")
print(f"naive : {p_big_naive.tolist()}  <- exp(800) overflowed")
print(f"stable: {p_big_stable.tolist()}")

assert torch.isnan(p_big_naive).any(), "naive path should return nan on a big logit"
assert torch.isfinite(p_big_stable).all(), "stable path stays finite"

# --------------------------------------------------------------------------
# 3. The right way: hand cross_entropy the LOGITS, never probabilities.
# --------------------------------------------------------------------------
logits = z.unsqueeze(0)                  # shape (1, 3)
target = torch.tensor([0])               # true class index
L_ce = F.cross_entropy(logits, target).item()
print("\n-- F.cross_entropy reproduces the loss from LOGITS --")
print(f"F.cross_entropy(logits, target) = {L_ce:.6f}  (fused, stable)")

assert abs(L_ce - 0.417030) < 1e-5, f"F.cross_entropy should reproduce 0.417030, got {L_ce}"

print("\n[OK] the stable and unstable softmaxes are the same math; only one survives a big logit.")
