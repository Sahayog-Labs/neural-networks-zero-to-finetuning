#!/usr/bin/env python3
"""
inductive_bias.py -- the FC-vs-conv parameter blow-up, in a real nn.Module.

Course artifact for p.25 ("Architecture Is Inductive Bias"). Builds the exact
two layers the page's killer worked example counts by hand, on a real
224x224x3 image:

  1. nn.Linear(150528, 1000) -- a dense layer taking the flattened image
     straight to 1,000 hidden units. One weight per (input, output) pair.
  2. nn.Conv2d(3, 64, kernel_size=3) -- 64 filters of 3x3x3, shared across
     every spatial position, plus one bias per filter.

sum(p.numel() for p in m.parameters()) is asked of each real nn.Module and
checked against the page's frozen worked-example numbers: 150,528,000 and
1,792, an 84,000x gap in one layer. The point isn't that conv is smaller --
it's that the FC layer *could* learn the same edge detector at all 50,176
pixel positions independently, given enough data; the conv layer is *handed*
translation-equivariance by construction, never learned into place.

SAFETY: CPU only, <1 s, allocates a few MB (the two layers' weight tensors).
Writes and installs nothing.
"""

import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)

print("=" * 68)
print("inductive_bias.py -- FC vs conv parameter count, real nn.Module")
print(f"torch {torch.__version__} - seed {SEED}")
print("=" * 68)

# --------------------------------------------------------------------------
# The page's worked example, on a real 224x224 RGB image (page 25 defaults)
# --------------------------------------------------------------------------
SIDE, C, HIDDEN = 224, 3, 1000
FILTERS, K = 64, 3

in_features = SIDE * SIDE * C  # 224*224*3 = 150,528

fc = nn.Linear(in_features, HIDDEN)          # bias=True by default -- PyTorch's real module
conv = nn.Conv2d(C, FILTERS, kernel_size=K)  # bias=True by default

# The page's own formulas are asymmetric on purpose (ledger table above the demo):
#   params_FC   = side^2 * C * hidden           -- WEIGHT only, "one weight per (input,output) pair"
#   params_conv = C_out * (C*k^2 + 1)            -- includes the +1 bias per filter
# So the fair comparison to the frozen 150,528,000 / 1,792 figures reads weight-only
# off the real nn.Linear and weight+bias off the real nn.Conv2d -- matching the page,
# not silently dropping PyTorch's default bias=True (shown separately below).
fc_weight_only = fc.weight.numel()
fc_params_total = sum(p.numel() for p in fc.parameters())
conv_params = sum(p.numel() for p in conv.parameters())

print("\n-- the two layers, same 224x224x3 image --")
print(f"nn.Linear({in_features}, {HIDDEN}) "
      f"-> {fc_weight_only:,} weight params (page's figure)"
      f" + {fc.bias.numel():,} bias = {fc_params_total:,} total (PyTorch's real default)")
print(f"nn.Conv2d({C}, {FILTERS}, kernel_size={K}) "
      f"-> {conv.weight.numel():,} weight + {conv.bias.numel():,} bias = {conv_params:,} total")

assert fc_weight_only == 150_528_000, f"FC weight count should be 150,528,000, got {fc_weight_only}"
assert conv_params == 1_792, f"conv param count should be 1,792, got {conv_params}"
print("[OK] FC and conv param counts match the page's worked example exactly:")
print("     150,528,000 and 1,792 -- printed from a real nn.Module, not a slide.")

# --------------------------------------------------------------------------
# The 84,000x gap the page derives by hand, confirmed from the real modules
# --------------------------------------------------------------------------
ratio = fc_weight_only / conv_params
print(f"\nratio = {fc_weight_only:,} / {conv_params:,} = {ratio:,.0f}x")

assert abs(ratio - 84_000) < 1e-6, f"ratio should be exactly 84,000x, got {ratio}"
print("[OK] 150,528,000 / 1,792 = 84,000x exactly -- a real nn.Module, not a slide.")

print("\n" + "=" * 68)
print("Same image, two layers: the conv layer is handed translation-equivariance")
print("by construction (one 27-weight stencil, shared everywhere); the FC layer")
print("would have to rediscover it from data, 84,000x more parameters to search.")
print("=" * 68)
