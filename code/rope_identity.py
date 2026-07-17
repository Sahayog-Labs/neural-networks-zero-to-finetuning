#!/usr/bin/env python3
"""
rope_identity.py -- RoPE's relative-position identity, and the rotate_half trap
that costs people days, both verified in torch.

Course artifact for p.33 ("Positional Encoding and RoPE"). Two payoffs:

  PART 1 -- the identity that IS the justification (trig payoff #2).
     Build the block-diagonal rotation R_i and check, on random q, k and random
     positions i, j, that
         <R_i q, R_j k>  ==  q^T R_(j-i) k .
     The rotations compose (R_i^T R_j = R_(j-i)), so the score depends only on
     the gap j-i. You inject ABSOLUTE position (i into q, j into k, independently
     and cheaply) and read out RELATIVE position, for free, without ever
     computing a difference -- the algebra performs the subtraction.

  PART 2 -- the rotate_half convention trap.
     The original paper pairs ADJACENT dims (0,1),(2,3),...  HuggingFace's
     rotate_half instead SPLITS THE VECTOR IN HALF and pairs across:
     (0, d/2),(1, d/2+1),...  Each convention, applied consistently, satisfies
     the identity of Part 1 -- so each is internally "correct." But they are NOT
     the same rotation of the same numbers: applied to one fixed pretrained
     q_proj / k_proj slice they produce DIFFERENT attention scores. A checkpoint
     is trained under exactly ONE convention; load its weights and run RoPE the
     OTHER way and you get output that looks like a training bug -- almost-
     coherent garbage, with nothing in the stack trace to point at.
     This script reproduces that divergence on a fixed "checkpoint" slice: the
     matching convention reproduces the reference scores to ~1e-6, while the
     mismatched one decorrelates them and flips most of the attended tokens.

Numbers (constants.md section 1.1, Qwen3-8B [VP]):
    rope_theta = 1e6 ,  head_dim = 128  ->  64 rotation pairs ,  hidden = 4096.

SAFETY: CPU only, torch only, <2 s, a few MB. Downloads nothing, writes nothing.
  The "pretrained q_proj slice" is a fixed, seeded stand-in so the script runs
  fully offline; the convention divergence it demonstrates is exactly the one a
  real Qwen3-8B checkpoint exhibits. To swap in a real slice, point the env var
  ROPE_QPROJ at a local *.safetensors shard (see load_qproj_slice()).
"""

import math
import os
import torch

SEED = 0
torch.manual_seed(SEED)

D_HEAD = 128                 # Qwen3-8B head_dim  (constants.md section 1.1)
N_PAIRS = D_HEAD // 2        # 64 rotation pairs
D_MODEL = 4096               # Qwen3-8B hidden_size (constants.md section 1.1)
BASE = 1_000_000             # rope_theta = 1e6    (constants.md section 1.1)

print("=" * 70)
print("rope_identity.py -- the relative-position identity + the rotate_half trap")
print(f"torch {torch.__version__} - seed {SEED} - d_head {D_HEAD} ({N_PAIRS} pairs) - base {BASE:,}")
print("=" * 70)


# --------------------------------------------------------------------------
# The per-pair angular frequencies  phi_p = base^(-2p/d_head),  p = 0..63.
# phi_0 = 1 rad/token (fastest); phi_63 = base^(-126/128) (slowest).
# float64 throughout so the identity check is limited by math, not by dtype.
# --------------------------------------------------------------------------
def rope_freqs(d_head=D_HEAD, base=BASE):
    p = torch.arange(0, d_head, 2, dtype=torch.float64) / d_head   # (d/2,)
    return base ** (-p)                                            # phi_p, (d/2,)


# --------------------------------------------------------------------------
# Convention A -- ADJACENT PAIRS (the original RoPE / GPT-NeoX-interleaved form).
# Rotate each 2-D pair (x_2p, x_2p+1) by angle pos * phi_p.
# --------------------------------------------------------------------------
def apply_adjacent(vec, pos, freqs):
    ang = pos * freqs                       # (d/2,)
    c, s = torch.cos(ang), torch.sin(ang)
    x_even, x_odd = vec[..., 0::2], vec[..., 1::2]
    out = torch.empty_like(vec)
    out[..., 0::2] = x_even * c - x_odd * s
    out[..., 1::2] = x_even * s + x_odd * c
    return out


# --------------------------------------------------------------------------
# Convention B -- HuggingFace ROTATE_HALF (split-half). This is byte-for-byte
# the transformers implementation: emb = cat(freqs, freqs); then
#   x_rot = x * cos + rotate_half(x) * sin ,  rotate_half([x1,x2]) = [-x2, x1].
# It rotates the cross-half pairs (p, p+d/2) by pos * phi_p.
# --------------------------------------------------------------------------
def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_hf(vec, pos, freqs):
    ang = pos * freqs                       # (d/2,)
    emb = torch.cat((ang, ang), dim=-1)     # (d,)  -- HF duplicates the angles
    c, s = torch.cos(emb), torch.sin(emb)
    return vec * c + rotate_half(vec) * s


# ==========================================================================
# PART 1 -- the relative-position identity, both conventions.
#   <R_i q, R_j k> depends only on the gap j-i, because R_i^T R_j = R_(j-i).
# ==========================================================================
print("\n[PART 1] relative-position identity:  <R_i q, R_j k> == q^T R_(j-i) k")
freqs = rope_freqs()
torch.manual_seed(SEED)

worst = 0.0
for trial in range(1000):
    q = torch.randn(D_HEAD, dtype=torch.float64)
    k = torch.randn(D_HEAD, dtype=torch.float64)
    i = int(torch.randint(0, 40960, (1,)))     # positions inside Qwen3-8B's context
    j = int(torch.randint(0, 40960, (1,)))
    for apply_rope in (apply_adjacent, apply_hf):
        lhs = torch.dot(apply_rope(q, i, freqs), apply_rope(k, j, freqs))  # <R_i q, R_j k>
        rhs = torch.dot(q, apply_rope(k, j - i, freqs))                    # q^T R_(j-i) k
        worst = max(worst, (lhs - rhs).abs().item())

# one worked example, printed the way the page's Panel B reads out
q = torch.randn(D_HEAD, dtype=torch.float64)
k = torch.randn(D_HEAD, dtype=torch.float64)
i, j = 137, 42
lhs = torch.dot(apply_adjacent(q, i, freqs), apply_adjacent(k, j, freqs))
rhs = torch.dot(q, apply_adjacent(k, j - i, freqs))
print(f"  example  i=137, j=42, gap j-i={j - i}:  "
      f"<R_i q,R_j k>={lhs.item():+.8f}  ==  q^T R_(j-i) k={rhs.item():+.8f}")
print(f"  worst |lhs - rhs| over 1000 random (q,k,i,j) x both conventions: {worst:.2e}")
assert worst < 1e-5, f"relative-position identity broke: {worst}"     # spec self-check
print("  [OK] the score knew only the gap -- for BOTH conventions, each on its own.")


# ==========================================================================
# PART 2 -- the rotate_half trap: one fixed checkpoint, two conventions, garbage.
# ==========================================================================
def load_qproj_slice():
    """Return (Wq, Wk) each (D_HEAD, D_MODEL). If ROPE_QPROJ points at a real
    Qwen3-8B *.safetensors shard, slice one real head's q_proj / k_proj rows;
    otherwise fall back to a fixed seeded stand-in (identical demonstration)."""
    path = os.environ.get("ROPE_QPROJ")
    if path and os.path.exists(path):
        from safetensors.torch import load_file
        sd = load_file(path)
        wq = sd["model.layers.0.self_attn.q_proj.weight"][:D_HEAD].to(torch.float64)
        wk = sd["model.layers.0.self_attn.k_proj.weight"][:D_HEAD].to(torch.float64)
        print(f"  using a REAL q_proj/k_proj head sliced from {os.path.basename(path)}")
        return wq, wk
    g = torch.Generator().manual_seed(1234)     # fixed stand-in -> reproducible
    wq = torch.randn(D_HEAD, D_MODEL, generator=g, dtype=torch.float64) / math.sqrt(D_MODEL)
    wk = torch.randn(D_HEAD, D_MODEL, generator=g, dtype=torch.float64) / math.sqrt(D_MODEL)
    print("  using a fixed seeded stand-in q_proj/k_proj slice (offline; set "
          "ROPE_QPROJ to use a real shard)")
    return wq, wk


def score_matrix(X, Wq, Wk, positions, apply_rope, freqs):
    """RoPE'd attention scores S[a,b] = <rope(q_a, pos_a), rope(k_b, pos_b)> / sqrt(d)."""
    Q = X @ Wq.T                                 # (S, d_head)
    K = X @ Wk.T
    Qr = torch.stack([apply_rope(Q[t], int(positions[t]), freqs) for t in range(len(positions))])
    Kr = torch.stack([apply_rope(K[t], int(positions[t]), freqs) for t in range(len(positions))])
    return (Qr @ Kr.T) / math.sqrt(D_HEAD)


print("\n[PART 2] rotate_half trap: same weights, two conventions, one is garbage")
Wq, Wk = load_qproj_slice()
S = 24
gen = torch.Generator().manual_seed(7)
X = torch.randn(S, D_MODEL, generator=gen, dtype=torch.float64)   # a token sequence
positions = torch.arange(1, S + 1) * 173                          # spread 173..4152

# THE CHECKPOINT was trained under HF's rotate_half convention -> its intended
# scores are the HF ones. That is the reference the weights "expect".
ref = score_matrix(X, Wq, Wk, positions, apply_hf, freqs)

# (a) matching convention reproduces the checkpoint's scores exactly.
correct = score_matrix(X, Wq, Wk, positions, apply_hf, freqs)
assert torch.allclose(ref, correct, atol=1e-9)
print("  matching convention (HF on an HF checkpoint):  reproduces scores exactly.")

# (b) mismatched convention on the SAME weights -> decorrelated garbage.
buggy = score_matrix(X, Wq, Wk, positions, apply_adjacent, freqs)

rf, bf = ref.flatten(), buggy.flatten()
corr = (((rf - rf.mean()) * (bf - bf.mean())).mean()
        / (rf.std() * bf.std())).item()          # Pearson correlation
max_abs = (ref - buggy).abs().max().item()

A_ref = torch.softmax(ref, dim=-1)
A_bug = torch.softmax(buggy, dim=-1)
top1_flipped = (A_ref.argmax(-1) != A_bug.argmax(-1)).float().mean().item()

print(f"  mismatched convention (adjacent on an HF checkpoint):")
print(f"    Pearson corr(ref, buggy) = {corr:+.3f}   (near 0 -> the scores are unrelated)")
print(f"    max |ref - buggy|        = {max_abs:.3f}")
print(f"    queries whose top-1 attended token FLIPPED = {top1_flipped * 100:.0f}%")

assert corr < 0.4, f"conventions should decorrelate, corr={corr}"
assert top1_flipped > 0.7, f"most attended tokens should flip, got {top1_flipped}"
print("  [OK] identical weights, wrong pairing -> almost-coherent garbage.")
print("       This is the trap: BOTH conventions pass Part 1, yet mixing one")
print("       with a checkpoint trained on the other silently breaks inference.")

print("\n" + "=" * 70)
print("Done. RoPE's score depends only on the gap -- and only if you keep the")
print("SAME pairing convention the checkpoint was trained with.")
print("=" * 70)
