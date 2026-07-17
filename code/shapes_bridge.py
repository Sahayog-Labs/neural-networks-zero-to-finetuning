#!/usr/bin/env python3
"""
shapes_bridge.py -- the row/column shape bridge, verified in torch.

Course artifact for p.07 ("Row or Column? The Shape Bridge"). nn.Linear(2, 3)
is the smallest layer that can prove the (out, in) convention -- a 2->2 layer
is ambiguous (notation.md section 9, anti-pattern #25). This script prints
weight.shape, then computes the SAME layer's output two ways -- column form
(W @ x) and row-batched form (x @ W.T) -- and asserts they agree bit-for-bit.
Only the batch axis moves; W never does.

SAFETY: CPU only, <1 s, allocates a few KB. Writes and installs nothing.
"""

import torch

SEED = 42
torch.manual_seed(SEED)

print("=" * 68)
print("shapes_bridge.py -- same W, two conventions")
print(f"torch {torch.__version__} - seed {SEED}")
print("=" * 68)

# nn.Linear(in_features=2, out_features=3): weight is ALWAYS (out, in).
layer = torch.nn.Linear(2, 3)
print(f"\nnn.Linear(2, 3).weight.shape = {tuple(layer.weight.shape)}   # (out, in) = (3, 2)")
assert tuple(layer.weight.shape) == (3, 2), "weight must be (out, in), never (in, out)"

W = layer.weight.detach().clone()      # (3, 2) -- the SAME matrix both forms use below
b = layer.bias.detach().clone()        # (3,)
x = torch.tensor([1.0, 2.0])           # one example, column-form shape (2,)

# --- Column form: single-example maths, y = W x + b. ---
y_col = W @ x + b                      # (3,2) @ (2,) -> (3,)
print(f"\ncolumn form   y = W @ x + b       -> {tuple(y_col.shape)}  {y_col.tolist()}")

# --- Row-batched form: code's convention, Y = X W^T + b. Batch of 1, same x. ---
X = x.unsqueeze(0)                     # (1, 2) -- x promoted to a one-row batch
y_row = X @ W.T + b                    # (1,2) @ (2,3) -> (1,3)
print(f"row form      Y = X @ W.T + b     -> {tuple(y_row.shape)}  {y_row.squeeze(0).tolist()}")

# The whole lesson: same W, same numbers -- only the batch axis moved.
agree = torch.allclose(y_col, y_row.squeeze(0))
assert agree, "column and row forms must agree exactly"
print(f"\ny_col == y_row.squeeze(0): {agree}  [OK]")
print("Only the batch axis moved. (out, in) never budged.")
