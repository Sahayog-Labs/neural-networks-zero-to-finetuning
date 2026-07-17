#!/usr/bin/env python3
"""
dot_product_geometry.py -- the dot product as a similarity meter, verified in torch.

Course artifact for p.27 ("Embeddings and the Dot Product"). Two things, in order:

  1. Reproduces the page's worked king/queen/bicycle numbers bit-for-bit, including
     the "scale king by 10x" demonstration that the RAW dot product is bilinear in
     magnitude while cosine similarity is invariant to it -- the exact conflation
     that motivates QK-norm (Qwen3 ships q_norm/k_norm for this reason, p.30/34).
  2. Samples random unit vectors at d in {2, 64, 4096} -- the last is Qwen3-8B's
     actual residual-stream width (constants.md section 1.1) -- and measures the
     mean/std of 5,000 pairwise cosines. The prediction, std ~= 1/sqrt(d), is a
     standard result for random points on a high-dimensional sphere: as d grows,
     two random directions are overwhelmingly close to orthogonal. This is the
     live, at-scale extension of the page's Panel 2 histogram (which is capped
     at d=512 in-browser for interactivity).

SAFETY: CPU only, <1 s, allocates a few MB at d=4096. Writes and installs nothing.
"""

import math
import torch

SEED = 42
torch.manual_seed(SEED)

print("=" * 68)
print("dot_product_geometry.py -- the dot product as a similarity meter")
print(f"torch {torch.__version__} - seed {SEED}")
print("=" * 68)

# --------------------------------------------------------------------------
# 1. The worked example: king / queen / bicycle (page 27's frozen numbers)
# --------------------------------------------------------------------------
king    = torch.tensor([0.9, 0.8, 0.1])
queen   = torch.tensor([0.85, -0.7, 0.15])
bicycle = torch.tensor([-0.4, 0.05, 0.9])
cos = torch.nn.CosineSimilarity(dim=0)

kq_dot = (king @ queen).item()
kq_cos = cos(king, queen).item()
kb_dot = (king @ bicycle).item()
kb_cos = cos(king, bicycle).item()

print("\n-- worked example: king / queen / bicycle --")
print(f"king . queen   = {kq_dot:.4f}   cos(theta) = {kq_cos:.4f}   theta = {math.degrees(math.acos(kq_cos)):.1f} deg")
print(f"king . bicycle = {kb_dot:.4f}   cos(theta) = {kb_cos:.4f}   theta = {math.degrees(math.acos(kb_cos)):.1f} deg")

assert abs(kq_dot - 0.220) < 0.001, f"king.queen should be 0.220, got {kq_dot}"
assert abs(kq_cos - 0.164) < 0.001, f"cos(king,queen) should be 0.164, got {kq_cos}"
assert abs(kb_dot - (-0.230)) < 0.001, f"king.bicycle should be -0.230, got {kb_dot}"
# The dot-product thread: king/queen sit CLOSER (higher cosine) than king/bicycle --
# similarity as a ranking survives even though these are unrelated toy numbers.
assert kq_cos > kb_cos, "king should be more similar to queen than to bicycle -- similarity ordering broke"
print("[OK] worked numbers match the page; king is closer to queen than to bicycle.")

# Scale king by 10: raw dot scales linearly, cosine is invariant. This is the
# exact reason 2026-era models QK-norm before scoring attention (p.30/34).
king_scaled = king * 10.0
scaled_dot = (king_scaled @ queen).item()
scaled_cos = cos(king_scaled, queen).item()
print(f"\nscale king by 10x: dot = {scaled_dot:.4f} (was {kq_dot:.4f}, ratio {scaled_dot/kq_dot:.2f}x)"
      f"   cos(theta) = {scaled_cos:.4f} (unchanged)")
assert abs(scaled_dot / kq_dot - 10.0) < 1e-4, "raw dot must scale exactly 10x"
assert abs(scaled_cos - kq_cos) < 1e-5, "cosine must be invariant to rescaling one vector"
print("[OK] raw dot scaled 10x; cosine held fixed -- direction and magnitude, decoupled.")

# --------------------------------------------------------------------------
# 2. Near-orthogonality at scale: d in {2, 64, 4096}
# --------------------------------------------------------------------------
print("\n-- near-orthogonality: random unit vectors, 5000 pairs each --")
N_PAIRS = 5000
for d in (2, 64, 4096):
    U = torch.randn(N_PAIRS, d)
    V = torch.randn(N_PAIRS, d)
    U = U / U.norm(dim=1, keepdim=True)
    V = V / V.norm(dim=1, keepdim=True)
    cosines = (U * V).sum(dim=1)                 # (N_PAIRS,) -- one dot product per pair, rows already unit length
    mean, std = cosines.mean().item(), cosines.std().item()
    predicted = 1.0 / math.sqrt(d)
    print(f"d={d:5d}:  mean={mean:+.4f}  std={std:.4f}  predicted 1/sqrt(d)={predicted:.4f}")
    # A loose tolerance: this is a finite-sample Monte Carlo estimate, not an exact
    # equality, so we check the ORDER OF MAGNITUDE the page's claim depends on.
    assert abs(mean) < 0.05, f"mean cosine should hover near 0 at d={d}, got {mean}"
    assert abs(std - predicted) / predicted < 0.25, f"sampled std should track 1/sqrt(d) at d={d}: {std} vs {predicted}"

print("\n[OK] std tracks 1/sqrt(d) at every d checked -- the higher the dimension,")
print("     the more overwhelmingly two random directions sit near 90 degrees.")
print(f"     At Qwen3-8B's real d=4096: predicted std = 1/sqrt(4096) = 1/64 = {1/64:.4f}.")
