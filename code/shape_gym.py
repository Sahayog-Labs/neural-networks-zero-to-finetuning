#!/usr/bin/env python3
"""
shape_gym.py — the (64,) vs (64,1) broadcast trap, live.

Course artifact for p.02 ("Vectors, Matrices, Tensors — Shape Is the Skill"). The
single highest-value warning in that chapter: subtracting a (64,) prediction from a
(64,1) target does NOT error. PyTorch broadcasts them to (64,64) and silently
computes garbage. This script makes the garbage visible, then fixes it with one
.squeeze() and shows the two losses disagree.

SAFETY: CPU only, <1 s, allocates a few KB. Writes and installs nothing.
"""

import torch

SEED = 42
torch.manual_seed(SEED)

print("=" * 68)
print("shape_gym.py — the silent broadcast trap")
print(f"torch {torch.__version__} · seed {SEED}")
print("=" * 68)

# Same 64 numbers underneath both versions of the "loss" below, so the ONLY
# variable is the shape of `target`.
pred = torch.randn(64)          # shape (64,)   — the model's raw output
target = torch.randn(64)        # shape (64,)   — the true targets, as generated

print(f"\npred.shape    = {tuple(pred.shape)}")
print(f"target.shape  = {tuple(target.shape)}")

# --- The bug: someone upstream returns targets with a trailing singleton axis
#     (a very common shape out of a DataLoader that forgot .squeeze()). ---
target_buggy = target.unsqueeze(1)                     # (64,) -> (64, 1)
print(f"target_buggy.shape = {tuple(target_buggy.shape)}  <-- the trap")

diff_buggy = pred - target_buggy                        # broadcasts to (64, 64)!
print(f"\n(pred - target_buggy).shape = {tuple(diff_buggy.shape)}   NO ERROR RAISED")
print("Right-align the shapes: (64,) vs (64, 1) -> both dims 'compatible' by")
print("broadcasting, so PyTorch builds a (64, 64) matrix of every pred[j] minus")
print("every target[i] instead of the 64 elementwise differences you meant.")

loss_buggy = (diff_buggy ** 2).mean()
print(f"\nloss_buggy  (from the (64,64) broadcast) = {loss_buggy.item():.6f}")

# --- The fix: squeeze the trailing axis back to (64,) before subtracting. ---
target_fixed = target_buggy.squeeze()                   # (64, 1) -> (64,)
assert target_fixed.shape == pred.shape, "squeeze() should restore (64,)"
diff_fixed = pred - target_fixed
print(f"\n(pred - target_fixed).shape = {tuple(diff_fixed.shape)}   the shape you meant")

loss_fixed = (diff_fixed ** 2).mean()
print(f"loss_fixed  (elementwise, correct)        = {loss_fixed.item():.6f}")

# --- The whole point: they disagree, and only one of them means what the code
#     author intended. Nothing about the buggy run raised an error or a warning. ---
assert loss_buggy.item() != loss_fixed.item(), \
    "expected the (64,64) broadcast loss to differ from the elementwise loss"

print(f"\nloss_buggy != loss_fixed: {loss_buggy.item():.6f} != {loss_fixed.item():.6f}  [OK]")
print("\nThe habit that catches this before it trains for three epochs on garbage:")
print("    assert pred.shape == target.shape   # BEFORE every loss, no exceptions")
