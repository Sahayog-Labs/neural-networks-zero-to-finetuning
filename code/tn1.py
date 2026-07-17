#!/usr/bin/env python3
"""
tn1.py -- TN-1, your first real network, run forward in PyTorch.

Course artifact for p.06 ("TN-1: Your First Real Network, by Pencil and by
PyTorch"). This is the exact ~15-line forward script printed on that page: it
builds the two linear layers, loads TN-1's nine frozen parameters (constants.md
section 8), runs the forward pass on x = [1.0, 2.0], and asserts the output
matches the pencil to four decimals. It prints:

    yhat = 0.3727
    loss = 0.9869

FORWARD ONLY -- there is deliberately no loss.backward() here. This file
ACCRETES across the course, always the same nine numbers:
    p.14  adds .backward() and the nine-gradient check,
    p.15  adds the autograd assertion and the 1.4e-08 "exactly zero is false"
          float surprise,
    p.17  adds TN-1's first SGD step (loss 0.9869 -> 0.8222).
Keep this file; you will grow it.

RUN IT WITH THE RIGHT PYTHON: on the DGX Spark, torch lives inside ComfyUI's
venv (~/ComfyUI/.venv). Run this with that Python, or a fresh separate venv --
but never pip-install the course's later training stack into ComfyUI's venv
(hardware-ground-truth.md section 3). For this page, plain torch alone suffices.

SAFETY: CPU only, <0.1 s, allocates a few KB. Writes and installs nothing.
"""

import torch
import torch.nn as nn

torch.set_grad_enabled(False)            # forward only -- no gradients on this page

lin1 = nn.Linear(2, 2)                   # weight shape (2, 2) = (out, in)
lin2 = nn.Linear(2, 1)                   # weight shape (1, 2)

# TN-1's nine frozen parameters (constants.md section 8). W is (out, in) always.
lin1.weight[:] = torch.tensor([[0.5, -0.3], [0.8, 0.2]])
lin1.bias[:]   = torch.tensor([0.1, -0.1])
lin2.weight[:] = torch.tensor([[0.6, -0.9]])
lin2.bias[:]   = torch.tensor([0.2])

x    = torch.tensor([1.0, 2.0])
a1   = torch.tanh(lin1(x))               # two hidden units
yhat = torch.sigmoid(lin2(a1))           # one output probability
loss = -torch.log(yhat)                  # BCE for the positive label y = 1

print(f"yhat = {yhat.item():.4f}")       # 0.3727
print(f"loss = {loss.item():.4f}")       # 0.9869
assert abs(yhat.item() - 0.3727) < 1e-4
