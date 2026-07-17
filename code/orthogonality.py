#!/usr/bin/env python3
"""
orthogonality.py -- your first measurement of superposition, verified in torch.

Course artifact for p.03 ("The Dot Product -- the Atom"). This is the at-your-
keyboard version of the page's right-hand histogram panel: it measures, rather
than asserts, that two random directions in a high-dimensional space are almost
always perpendicular.

  1. Samples ONE pair of random unit vectors in R^4096 -- Qwen3-8B's residual-
     stream width (constants.md section 1.1) -- and prints their cosine. You
     will see roughly 0.01-0.02: near-orthogonal on a single draw.
  2. Loops 10,000 times and prints the empirical standard deviation of that
     cosine. It lands on 1/sqrt(4096) = 1/64 = 0.0156 -- the theoretical std of
     the cosine of two random points on a high-dimensional sphere. That the
     spread is this tight is WHY a 4096-wide model isn't limited to 4096 ideas:
     it can pack far more than 4096 near-orthogonal "concepts" into the space
     (superposition).

SAFETY: CPU only, <1 s, allocates a few MB. Writes and installs nothing.
"""

import math
import torch

SEED = 42
torch.manual_seed(SEED)

D = 4096          # Qwen3-8B residual-stream width (constants.md section 1.1)
N_TRIALS = 10_000

print("=" * 68)
print("orthogonality.py -- random vectors in R^4096 are almost always perpendicular")
print(f"torch {torch.__version__} - seed {SEED}")
print("=" * 68)


def random_unit(d):
    """One random unit vector in R^d: Gaussian, then normalized to length 1."""
    v = torch.randn(d)
    return v / v.norm()


# --------------------------------------------------------------------------
# 1. A single pair: sample two random unit vectors and read their cosine.
#    For unit vectors, cosine similarity IS the dot product (denominator = 1).
# --------------------------------------------------------------------------
a = random_unit(D)
b = random_unit(D)
one_cos = torch.dot(a, b).item()
print(f"\n-- one pair in R^{D} --")
print(f"cos(a, b) = {one_cos:+.4f}   (near 0: the two directions are nearly orthogonal)")

# --------------------------------------------------------------------------
# 2. Ten thousand pairs: the empirical std of the cosine tracks 1/sqrt(d).
# --------------------------------------------------------------------------
cosines = torch.empty(N_TRIALS)
for t in range(N_TRIALS):
    cosines[t] = torch.dot(random_unit(D), random_unit(D))

emp_mean = cosines.mean().item()
emp_std = cosines.std().item()
predicted = 1.0 / math.sqrt(D)     # = 1/64 = 0.015625

print(f"\n-- {N_TRIALS:,} pairs in R^{D} --")
print(f"empirical mean = {emp_mean:+.4f}   (hovers at 0)")
print(f"empirical std  = {emp_std:.4f}")
print(f"theory 1/sqrt(d) = 1/sqrt({D}) = 1/64 = {predicted:.4f}")

# Finite-sample Monte Carlo, not an exact equality: check it tracks the theory.
assert abs(emp_mean) < 0.01, f"mean cosine should hover near 0, got {emp_mean}"
assert abs(emp_std - predicted) / predicted < 0.10, (
    f"empirical std {emp_std} should track 1/sqrt(d)={predicted}")

print("\n[OK] std tracks 1/sqrt(d): random directions in R^4096 sit near 90 degrees.")
print("     You just measured why a 4096-wide model isn't limited to 4096 ideas.")
