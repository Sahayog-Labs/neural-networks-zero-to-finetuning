#!/usr/bin/env python3
"""
heads_are_a_reshape.py -- multi-head attention is a reshape, not H matrices,
and the sqrt(d_head) scale is a thermostat you can watch with one Jacobian.

Course artifact for p.30 ("Attention II: the sqrt(d_head) Thermostat and Many
Heads"). Four things, in order, each with a self-check so the page's frozen
numbers are proven on your machine, not asserted at you:

  1. The reshape. One real Qwen3-8B q_proj -- a single (4096, 4096) matrix --
     becomes H=32 independent heads of d_head=128 via .view(B,S,H,d_head)
     .transpose(1,2). No second matmul, no extra params. And the inverse
     (transpose back, flatten) returns the ORIGINAL tensor bit-for-bit: that
     round-trip is the proof the "heads" were a view all along.
  2. The thermostat's setpoint. At d_head=128, iid unit-variance q,k make the
     raw score q.k have std sqrt(128) = 11.3137 (constants.md section 5). We
     both assert the analytic value and MEASURE it over 200k sampled pairs.
  3. The partition (misconception (d): "more heads = more capacity"). The
     attention parameter count 2*d*d_head*(H + H_kv) depends only on the two
     projection WIDTHS (query width H*d_head = d, kv width H_kv*d_head), never
     on how finely you slice them into heads. Re-splitting Qwen3-8B's
     (H=32, d_head=128) into (H=8, d_head=512) -- keeping both widths, so the
     GQA group H_kv goes 8 -> 2 -- lands on the IDENTICAL 41,943,040. It is a
     partition of a fixed width, not an addition of capacity.
  4. The Jacobian cliff. softmax's diagonal Jacobian A0*(1-A0) on the frozen
     logits [2.0, 1.0, 0.1]: healthy 0.2247 unscaled, ~1.2e-5 once multiplied
     by sqrt(128). Same relative preference, a gradient ~18,000x smaller --
     which in bf16 rounds to zero. That is why the scale exists.

SAFETY: CPU only, torch on CPU, < 1 s, allocates a few MB. Writes/installs nothing.
"""

import math
import torch
import torch.nn.functional as F

SEED = 30
torch.manual_seed(SEED)

print("=" * 70)
print("heads_are_a_reshape.py -- multi-head attention is a reshape, not H matrices")
print(f"torch {torch.__version__} - seed {SEED}")
print("=" * 70)

# Qwen3-8B attention config (constants.md section 1.1).
B, S, d, H, d_head, H_kv = 1, 10, 4096, 32, 128, 8
assert H * d_head == d, "Qwen3-8B convention: H*d_head == d (32*128 == 4096)"

# --------------------------------------------------------------------------
# 1. The reshape: one (4096, 4096) q_proj -> 32 heads of 128, and back again.
# --------------------------------------------------------------------------
q_proj = torch.nn.Linear(d, H * d_head, bias=False)     # ONE matrix
assert q_proj.weight.shape == (H * d_head, d), "weight is (out, in) -- ALWAYS"

x = torch.randn(B, S, d)
q = q_proj(x)                                           # (1, 10, 4096)
q_heads = q.view(B, S, H, d_head).transpose(1, 2)       # (1, 32, 10, 128) -- no new params

print("\n-- 1. the reshape --")
print(f"q_proj.weight : {tuple(q_proj.weight.shape)}  (one matrix, {q_proj.weight.numel():,} params)")
print(f"q            : {tuple(q.shape)}  ->  view+transpose  ->  q_heads {tuple(q_heads.shape)}")

# Concatenating the heads back is the INVERSE reshape: transpose (1,2) back and
# flatten the last two axes must reconstruct the original q bit-for-bit. If the
# heads were ever "separate matrices" this round-trip could not be exact.
q_concat = q_heads.transpose(1, 2).reshape(B, S, H * d_head)
assert torch.equal(q_concat, q), "concat(heads) must be the exact inverse of the reshape"
print("[OK] concat(split(q)) == q, bit-for-bit -- the heads were a view, not a split.")

# --------------------------------------------------------------------------
# 2. The thermostat setpoint: std(q.k) = sqrt(d_head) = 11.3137.
# --------------------------------------------------------------------------
print("\n-- 2. std(q.k) at d_head=128 --")
analytic_std = math.sqrt(d_head)
assert abs(analytic_std - 11.3137) < 1e-4, f"sqrt(128) should be 11.3137, got {analytic_std}"

N = 200_000
qs = torch.randn(N, d_head)                             # iid mean-0 var-1 (true at init)
ks = torch.randn(N, d_head)
scores = (qs * ks).sum(dim=1)                           # one dot product per row
measured_std = scores.std().item()
print(f"analytic std(q.k) = sqrt({d_head}) = {analytic_std:.4f}")
print(f"measured std over {N:,} pairs = {measured_std:.4f}")
# Finite-sample Monte Carlo, so a loose band -- the point is it lands on 11.31.
assert abs(measured_std - analytic_std) / analytic_std < 0.02, \
    f"measured std should track sqrt(d_head): {measured_std} vs {analytic_std}"
print("[OK] raw scores really do have std ~= 11.31 -- the exact number we divide by.")

# --------------------------------------------------------------------------
# 3. The partition: attention params are independent of the H/d_head split.
# --------------------------------------------------------------------------
def attn_params(d, H, d_head, H_kv):
    # q_proj + o_proj use the full query width H*d_head; k_proj + v_proj use the
    # (GQA) kv width H_kv*d_head. So the count is 2*d*d_head*(H + H_kv), which
    # depends only on the two WIDTHS, not on the head count itself.
    return 2 * d * d_head * (H + H_kv)

split_a = attn_params(4096, 32, 128, 8)                 # Qwen3-8B: 32 heads of 128, GQA H_kv=8
split_b = attn_params(4096,  8, 512, 2)                 # same widths, sliced coarser: 8 heads of 512

print("\n-- 3. more heads != more capacity (misconception (d)) --")
print(f"(H=32, d_head=128, H_kv=8):  query width {32*128}, kv width {8*128}  ->  P_attn = {split_a:,}")
print(f"(H=8,  d_head=512, H_kv=2):  query width {8*512}, kv width {2*512}  ->  P_attn = {split_b:,}")
assert split_a == split_b, "re-splitting the same widths must not change the param count"
assert split_a == 41_943_040, "Qwen3-8B per-block attention subtotal is frozen at 41,943,040 (constants section 1.2)"
print(f"[OK] both splits == {split_a:,} -- a partition of a fixed width, not an addition.")

# --------------------------------------------------------------------------
# 4. The Jacobian cliff: the sqrt(d_head) thermostat, as a gradient.
# --------------------------------------------------------------------------
print("\n-- 4. softmax Jacobian A0*(1-A0) on the frozen logits [2.0, 1.0, 0.1] --")
z = torch.tensor([2.0, 1.0, 0.1])
jac = {}
for scale in (1.0, d_head ** 0.5):
    a = torch.softmax(z * scale, dim=-1)
    j = (a[0] * (1 - a[0])).item()
    jac[scale] = (a[0].item(), j)
    tag = "unscaled" if scale == 1.0 else f"x sqrt(128)={scale:.4f}"
    print(f"scale {scale:8.4f} ({tag:>16}):  A0 = {a[0].item():.6f}   A0*(1-A0) = {j:.3e}")

assert abs(jac[1.0][1] - 0.2247) < 1e-3, "unscaled Jacobian should be 0.2247"
assert jac[d_head ** 0.5][1] < 1e-4, "scaled Jacobian should collapse below 1e-4"
ratio = jac[1.0][1] / jac[d_head ** 0.5][1]
print(f"[OK] gradient shrank {ratio:,.0f}x -- ~1.2e-5 against activations of order 1")
print("     rounds to zero in bf16 (8 mantissa bits, resolution 2^-8 ~= 0.004).")

print("\n" + "=" * 70)
print("All self-checks passed: the heads are a reshape, the scale is sqrt(128),")
print("the split is a partition, and dropping it kills the gradient.")
print("=" * 70)
