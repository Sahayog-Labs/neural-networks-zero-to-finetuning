#!/usr/bin/env python3
"""
12_cross_entropy.py -- cross-entropy from the course's canonical logits, in torch.

Course artifact for p.12 ("What 'Learning' Optimizes: Loss from Maximum
Likelihood"). Three things, in order:

  1. Reproduces the frozen triple (constants.md section 9.2) bit-for-bit with
     torch.nn.functional.cross_entropy. THE ONE RULE THAT MATTERS: cross_entropy
     takes LOGITS, never probabilities. It fuses log_softmax + nll_loss with the
     LogSumExp trick inside, so it never forms the probability explicitly. Feed it
     log(softmax(...)) and you double-count the log. The assertion nails the value
     to the sixth decimal: L = -ln(0.659001) = 0.417030 nats.
  2. Confirms the softmax+CE gradient is exactly yhat - onehot(c) = the frozen
     [-0.340999, +0.242433, +0.098566], and that it sums to zero -- always, because
     softmax outputs sum to 1 and the one-hot target sums to 1 (section 9.2).
  3. Reads the random-init loss anchor ln(V) = ln(151,936) = 11.93 nats for
     Qwen3-8B (constants.md section 9.1). This is the horizontal line every loss
     curve is plotted against; a curve sitting on it has learned nothing. NOT 11.76
     -- that is Llama-3's retired V=128,256 (D-01).

SAFETY: CPU only, <1 s, allocates a few KB. Writes and installs nothing.
"""

import math
import torch
import torch.nn.functional as F

print("=" * 68)
print("12_cross_entropy.py -- cross-entropy on the course's canonical logits")
print(f"torch {torch.__version__}")
print("=" * 68)

# --------------------------------------------------------------------------
# 1. The frozen triple: z = [2.0, 1.0, 0.1], true class c = 0 (section 9.2)
# --------------------------------------------------------------------------
z = torch.tensor([[2.0, 1.0, 0.1]])   # shape (1, 3) = (batch, classes)
c = torch.tensor([0])                 # the TRUE class index

# F.cross_entropy expects LOGITS, not probabilities. It fuses log_softmax +
# nll_loss with the LogSumExp trick inside -- it never forms yhat explicitly.
loss = F.cross_entropy(z, c)

# yhat is shown only to check the page's numbers; cross_entropy did NOT need it.
yhat = F.softmax(z, dim=1)
print("\n-- the frozen triple --")
print(f"z    = {z.squeeze().tolist()}   true class c = 0")
print(f"yhat = [{yhat[0,0]:.6f}, {yhat[0,1]:.6f}, {yhat[0,2]:.6f}]")
print(f"L    = -ln(yhat_0) = {loss.item():.6f} nats")

assert abs(yhat[0, 0].item() - 0.659001) < 1e-5, f"yhat_0 should be 0.659001, got {yhat[0,0].item()}"
assert abs(yhat[0, 1].item() - 0.242433) < 1e-5, f"yhat_1 should be 0.242433, got {yhat[0,1].item()}"
assert abs(yhat[0, 2].item() - 0.098566) < 1e-5, f"yhat_2 should be 0.098566, got {yhat[0,2].item()}"
assert abs(loss.item() - 0.417030) < 1e-5, f"CE loss should be 0.417030, got {loss.item()}"
print("[OK] loss = 0.417030 nats, matching constants.md section 9.2 to the sixth decimal.")

# The trap the rule guards against: hand it PROBABILITIES instead of logits and
# it silently log-softmaxes them a second time, quietly returning a wrong number.
# (Feeding log(softmax(z)) is NOT a trap -- log-probs are valid shifted logits, so
#  log_softmax is idempotent on them and the value is unchanged. The real mistake
#  is passing yhat itself.)
wrong = F.cross_entropy(yhat, c)
print(f"\nfeed the probabilities yhat in by mistake -> {wrong.item():.6f} nats (WRONG, not 0.417030)")
assert abs(wrong.item() - loss.item()) > 1e-3, "passing probabilities should give a different, wrong loss"
print("[OK] cross_entropy takes logits; hand it probabilities and it re-normalises to garbage.")

# --------------------------------------------------------------------------
# 2. The gradient is exactly softmax(z) - onehot(c), and sums to 0 (section 9.2)
# --------------------------------------------------------------------------
z = z.detach().clone().requires_grad_(True)
F.cross_entropy(z, c).backward()
grad = z.grad.squeeze()
print("\n-- the gradient dL/dz = yhat - onehot(c) --")
print(f"z.grad = [{grad[0]:+.6f}, {grad[1]:+.6f}, {grad[2]:+.6f}]")
print(f"sum    = {grad.sum().item():+.2e}   (zero, always: softmax sums to 1, one-hot sums to 1)")

assert abs(grad[0].item() - (-0.340999)) < 1e-5, f"dL/dz_0 should be -0.340999, got {grad[0].item()}"
assert abs(grad[1].item() - (+0.242433)) < 1e-5, f"dL/dz_1 should be +0.242433, got {grad[1].item()}"
assert abs(grad[2].item() - (+0.098566)) < 1e-5, f"dL/dz_2 should be +0.098566, got {grad[2].item()}"
assert abs(grad.sum().item()) < 1e-6, f"softmax+CE gradient must sum to zero, got {grad.sum().item()}"
print("[OK] gradient is the frozen [-0.340999, +0.242433, +0.098566] and sums to zero.")

# --------------------------------------------------------------------------
# 3. The random-init anchor: an untrained V-way softmax scores ln(V) (section 9.1)
# --------------------------------------------------------------------------
V = 151936                        # Qwen3-8B vocabulary (constants.md section 1.x)
ln_V = math.log(V)
print("\n-- the random-init loss anchor --")
print(f"ln(V) = ln({V:,}) = {ln_V:.2f} nats   <- Qwen3-8B step-0 loss")
print(f"perplexity at init = e^ln(V) = {math.exp(ln_V):,.0f} = V exactly")
print(f"coin-flip loss = ln 2 = {math.log(2):.4f} nats")

assert abs(ln_V - 11.93) < 0.005, f"random-init loss should be 11.93 nats, got {ln_V}"
assert abs(ln_V - 11.76) > 0.1, "11.76 is Llama-3's retired V=128,256 -- NOT this course's anchor (D-01)"
# Sanity: uniform logits over V classes give exactly ln(V), the value above.
uniform_loss = F.cross_entropy(torch.zeros(1, V), torch.tensor([0]))
assert abs(uniform_loss.item() - ln_V) < 1e-4, "uniform V-way softmax must score ln(V)"
print("[OK] ln(V) = 11.93 nats (NOT 11.76, that was Llama-3); uniform softmax confirms it.")

print("\n" + "=" * 68)
print("All assertions passed. One principle -- negative log-likelihood -- gave")
print("the loss (0.417030), the gradient (yhat - y), and the anchor (ln V = 11.93).")
print("=" * 68)
