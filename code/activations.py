#!/usr/bin/env python3
"""
activations.py -- the real reason ReLU won, measured, not asserted.

Course artifact for p.11 ("Activations, Why ReLU Won, and the MLP's Honest
Limits"). Three things, none of them looked up:

  1. Tabulates phi'_max for sigmoid/tanh/ReLU/GELU by autograd -- not a table
     you memorized -- by sampling each function densely over [-6, 6] and
     reading off the largest gradient torch itself computes.
  2. Prints 0.25**32 = 5.4e-20 (constants.md section 9.3 [DER]) against BOTH
     fp32 floors: torch.finfo(torch.float32).tiny (the representable floor --
     the ceiling sits comfortably above it) and .eps (the precision floor --
     the ceiling sits far below it, which is the one that actually kills a
     weight update).
  3. Builds a REAL 32-layer sigmoid stack and a real 32-layer ReLU stack,
     backpropagates through each once, and prints the layer-1 gradient norm.
     Sigmoid's underflows to a practical zero; ReLU's survives at ~1.

"The myth said 'ReLU is faster.' Your own terminal says 'ReLU is the one
whose gradient isn't zero.'"

SAFETY: CPU only, <1 s, allocates a few KB. Writes and installs nothing.
"""

import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)

print("=" * 68)
print("activations.py -- gradient flow: sigmoid vs. ReLU at depth 32")
print(f"torch {torch.__version__} - seed {SEED}")
print("=" * 68)


# --------------------------------------------------------------------------
# 1. phi'_max for each activation, measured by autograd, not memorized.
# --------------------------------------------------------------------------
def phi_max(name):
    x = torch.linspace(-6, 6, 200_001, requires_grad=True)
    if name == "sigmoid":
        y = torch.sigmoid(x)
    elif name == "tanh":
        y = torch.tanh(x)
    elif name == "relu":
        y = torch.relu(x)
    elif name == "gelu":
        y = nn.functional.gelu(x)
    else:
        raise ValueError(name)
    g = torch.autograd.grad(y.sum(), x)[0]
    return g.max().item()


print("\n-- 1. phi'_max, measured live by autograd over [-6, 6] --")
measured = {}
for name in ["sigmoid", "tanh", "relu", "gelu"]:
    measured[name] = phi_max(name)
    print(f"{name:8s} phi'_max = {measured[name]:.4f}")

# The two frozen ceilings this page names (constants.md section 9.3 [DER]):
assert abs(measured["sigmoid"] - 0.25) < 1e-3, "sigmoid's ceiling should be 0.25"
assert abs(measured["tanh"] - 1.0) < 1e-3, "tanh's ceiling should be 1.0"

# --------------------------------------------------------------------------
# 2. The idealized ceiling, named against BOTH fp32 floors.
# --------------------------------------------------------------------------
print("\n-- 2. 0.25**32 against fp32's two different floors --")
ceiling = 0.25 ** 32
tiny = torch.finfo(torch.float32).tiny  # smallest NORMAL, 1.1754944e-38
eps = torch.finfo(torch.float32).eps    # precision floor, 1.1920929e-07
print(f"0.25**32        = {ceiling:.2e}   (constants.md section 9.3 [DER])")
print(f"fp32 min-normal = {tiny:.2e}   (ceiling is ABOVE this -- still representable)")
print(f"fp32 epsilon    = {eps:.2e}   (ceiling is far BELOW this -- an update this")
print("                              size vanishes on addition to any O(1) weight)")

assert abs(ceiling - 5.4e-20) / 5.4e-20 < 0.02, "0.25**32 should read ~5.4e-20"
assert ceiling > tiny, "the ceiling is representable in fp32..."
assert ceiling < eps, "...but invisible to any O(1) weight update"
print("[OK] representable, but practically zero -- 'practically', not 'representably'.")

# --------------------------------------------------------------------------
# 3. A REAL 32-layer stack, one backward pass each -- not the idealized
#    ceiling, the actual product PyTorch computes through 32 real layers.
# --------------------------------------------------------------------------
def deep_grad(act):
    x = torch.tensor(1.0, requires_grad=True)  # float32 by default
    h = x
    for _ in range(32):
        h = act(h)
    h.backward()
    return x.grad.item()


print("\n-- 3. a real 32-deep stack, one backward() each --")
g_sigmoid = deep_grad(torch.sigmoid)
g_relu = deep_grad(torch.relu)
print(f"32-deep sigmoid stack, layer-1 grad = {g_sigmoid:.3e}")
print(f"32-deep ReLU stack,    layer-1 grad = {g_relu:.3e}")

assert abs(g_sigmoid) < 1e-15, "sigmoid should be practically zero"
assert g_relu > 0.5, "ReLU's gradient should survive at ~1"

print("\n[OK] sigmoid's layer-1 gradient underflows to a practical zero;")
print("     ReLU's does not.")
print("\nThe myth said 'ReLU is faster.' Your own terminal says 'ReLU is the")
print("one whose gradient isn't zero.'")
