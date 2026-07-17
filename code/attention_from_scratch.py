#!/usr/bin/env python3
"""
attention_from_scratch.py -- the boxed formula, by hand, on real Q/K/V shapes.

Course artifact for p.29 ("Attention I: The Soft Lookup"). Three parts, each
proving one fact the page asserts rather than just illustrating it:

  PART 0 -- the worked example, reproduced exactly. Two 2-D tokens ("it",
     "animal"), fixed W_Q / W_K, hand-computed on the page to S_it,animal =
     0.2520 and S_animal,it = 1.0836. Same numbers, computed in torch, proving
     the score matrix is ASYMMETRIC whenever W_Q != W_K (page's directed-
     relevance argument, "it" needs "animal" more than "animal" needs "it").

  PART 1 -- the boxed formula, unbatched, exactly the page's own code block
     (S=4 queries, S_kv=6 keys/values, d_head=8). The two asserts ARE the
     lesson: rows of A sum to 1 (columns do not, notation section 7), and the
     hand-written version matches F.scaled_dot_product_attention to float
     tolerance.

  PART 2 -- the full ribbon, batched, with heads, at Qwen3-8B's real
     d_head=128 (constants.md section 1.1), and S != S_kv on purpose -- the
     page's "the key/value set S_kv can be any length, independent of S" fact,
     the one that makes cross-attention (page 31) obvious. Every tensor's
     shape is asserted against the ribbon table, not just printed.

SAFETY: CPU only, torch only, < 1 s, allocates a few MB. Writes/installs
  nothing. No GPU, no download -- this is Part III trunk, runs on a laptop.
"""

import math
import torch
import torch.nn.functional as F

SEED = 0
torch.manual_seed(SEED)

print("=" * 70)
print("attention_from_scratch.py -- the boxed formula, by hand, on real shapes")
print(f"torch {torch.__version__} - seed {SEED}")
print("=" * 70)

# ==========================================================================
# PART 0 -- the worked example (p.29 "one score, both directions"), reproduced.
# ==========================================================================
print("\n[PART 0] the worked example: d = d_head = 2, W_Q != W_K, by hand")

WQ = torch.tensor([[1.0, 0.3], [-0.2, 1.0]])     # (out, in) -- ALWAYS
WK = torch.tensor([[0.8, -0.4], [0.5, 0.9]])

labels = ["it", "animal"]
X = torch.tensor([[0.9, 0.3],       # x_it
                   [0.6, 0.9]])     # x_animal

Q0 = X @ WQ.T          # row-batched q = x W_Q^T, equals the column form W_Q x
K0 = X @ WK.T
S0 = Q0 @ K0.T          # (2,2): S0[i,j] = q_i . k_j

s_it_animal = S0[0, 1].item()      # "it" queries, "animal" key
s_animal_it = S0[1, 0].item()      # "animal" queries, "it" key
print(f"  q_it     = ({Q0[0,0]:.2f}, {Q0[0,1]:.2f})   k_animal = ({K0[1,0]:.2f}, {K0[1,1]:.2f})")
print(f"  S[it,animal]   = q_it . k_animal   = {s_it_animal:.4f}")
print(f"  q_animal = ({Q0[1,0]:.2f}, {Q0[1,1]:.2f})   k_it     = ({K0[0,0]:.2f}, {K0[0,1]:.2f})")
print(f"  S[animal,it]   = q_animal . k_it   = {s_animal_it:.4f}")

assert abs(s_it_animal - 0.2520) < 1e-4, f"S[it,animal] should be 0.2520, got {s_it_animal}"
assert abs(s_animal_it - 1.0836) < 1e-4, f"S[animal,it] should be 1.0836, got {s_animal_it}"
assert abs(s_it_animal - s_animal_it) > 0.5, "the score matrix should be visibly asymmetric"
print(f"  [OK] {s_it_animal:.4f} != {s_animal_it:.4f} -- asymmetric, exactly because W_Q != W_K.")

# ==========================================================================
# PART 1 -- the boxed formula, unbatched (the page's own code block, verbatim
#           shapes): S=4 queries, S_kv=6 keys/values, d_head=8.
# ==========================================================================
print("\n[PART 1] the boxed formula, unbatched: S=4, S_kv=6, d_head=8")

torch.manual_seed(SEED)
S, S_kv, d_head = 4, 6, 8
Q = torch.randn(S, d_head)
K = torch.randn(S_kv, d_head)
V = torch.randn(S_kv, d_head)


def attention(Q, K, V):
    """The boxed formula: softmax(QK^T / sqrt(d_head)) V, softmax over the
    LAST axis (the keys). Works for any leading (batch/head) dims."""
    d_head = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_head)          # (..., S, S_kv)
    scores = scores - scores.max(dim=-1, keepdim=True).values      # online-safe softmax
    A = scores.exp()
    A = A / A.sum(dim=-1, keepdim=True)                            # softmax over keys
    return A @ V, A                                                 # (..., S, d_head), (..., S, S_kv)


O, A = attention(Q, K, V)
print(f"  Q {tuple(Q.shape)}  K {tuple(K.shape)}  V {tuple(V.shape)}  ->  A {tuple(A.shape)}  O {tuple(O.shape)}")

row_sums = A.sum(dim=-1)
col_sums = A.sum(dim=0)
print(f"  row sums (queries)  min={row_sums.min():.6f} max={row_sums.max():.6f}  (should be 1.000000)")
print(f"  col sums (keys)     min={col_sums.min():.4f} max={col_sums.max():.4f}  (should NOT be 1)")

# The two asserts ARE the lesson (spec-code.md D.6 p.29).
assert torch.allclose(row_sums, torch.ones(S), atol=1e-6), "every row of A must sum to 1"
assert not torch.allclose(col_sums, torch.ones(S_kv), atol=1e-3), "columns must NOT sum to 1 in general"
print("  [OK] rows of A sum to 1; columns do not (notation section 7).")

O_ref = F.scaled_dot_product_attention(Q, K, V)
assert torch.allclose(O, O_ref, atol=1e-5), "hand-written attention must match the fused kernel"
print("  [OK] hand-written attention == F.scaled_dot_product_attention (atol=1e-5).")

# ==========================================================================
# PART 2 -- the full ribbon: batched, with heads, at Qwen3-8B's real d_head,
#           and S != S_kv on purpose (constants.md section 1.1).
# ==========================================================================
print("\n[PART 2] the full ribbon: batched, H heads, Qwen3-8B's real d_head=128")

B, H, d_head2 = 2, 32, 128          # Qwen3-8B: H=32 query heads, d_head=128
S2, S2_kv = 10, 17                  # DELIBERATELY different -- S_kv is free (p.29's claim)

sqrt_d_head = math.sqrt(d_head2)
print(f"  sqrt(d_head) = sqrt({d_head2}) = {sqrt_d_head:.4f}  (Qwen3, constants.md section 1.1)")
assert abs(sqrt_d_head - 11.3137) < 1e-4, f"sqrt(128) should be 11.3137, got {sqrt_d_head}"

Q2 = torch.randn(B, H, S2, d_head2)
K2 = torch.randn(B, H, S2_kv, d_head2)
V2 = torch.randn(B, H, S2_kv, d_head2)

# Shapes obey the ribbon (p.29 "shape ribbon, batched, with heads") -- assert
# every tensor against the table, not just print it.
assert Q2.shape == (B, H, S2, d_head2)
assert K2.shape == (B, H, S2_kv, d_head2)
assert V2.shape == (B, H, S2_kv, d_head2)

O2, A2 = attention(Q2, K2, V2)

assert A2.shape == (B, H, S2, S2_kv), f"A must be (B,H,S,S_kv), got {tuple(A2.shape)}"
assert O2.shape == (B, H, S2, d_head2), f"O must be (B,H,S,d_head), got {tuple(O2.shape)}"

print(f"  Q  {tuple(Q2.shape)}")
print(f"  K  {tuple(K2.shape)}   <-- S_kv={S2_kv} != S={S2}, and nothing breaks")
print(f"  V  {tuple(V2.shape)}")
print(f"  A  {tuple(A2.shape)}   scores/weights, last axis = keys")
print(f"  O  {tuple(O2.shape)}   <-- O has S={S2} rows (one per QUERY), never S_kv")

row_sums2 = A2.sum(dim=-1)
assert torch.allclose(row_sums2, torch.ones(B, H, S2), atol=1e-6), "every row of A must sum to 1"
print("  [OK] every row of A sums to 1, for every (batch, head) slice, at real Qwen3-8B width.")

O2_ref = F.scaled_dot_product_attention(Q2, K2, V2)
assert torch.allclose(O2, O2_ref, atol=1e-5), "batched hand-written attention must match the fused kernel"
print("  [OK] batched hand-written attention == F.scaled_dot_product_attention (atol=1e-5).")

print("\n" + "=" * 70)
print("All self-checks passed. Same formula, three scales: a 2x2 worked example,")
print("an 8-wide toy, and Qwen3-8B's real d_head=128 with S_kv free of S.")
print("Page 30 asks why sqrt(d_head) and not something else.")
print("=" * 70)
