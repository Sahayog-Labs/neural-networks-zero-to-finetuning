#!/usr/bin/env python3
"""
self_and_cross.py -- self- and cross-attention are ONE function, called two
ways. The only thing that changes is where K,V come from.

Course artifact for p.31 ("Attention III: Self vs Cross -- The Diffusion
Hinge"). This is the single most important paragraph in the trunk for the
diffusion track: when you type "a red car" into ComfyUI and the car comes out
red, the mechanism is cross-attention -- the image QUERIES the text. Nothing
else is new. Same F.scaled_dot_product_attention call as self-attention;
only the K,V *source* (and therefore the column count of A) changes.

Two call sites, same function:

    self  -> attn(Q=x,   K=x,   V=x)      square  A: (S_img, S_img)
    cross -> attn(Q=x,   K=y,   V=y)      rect.   A: (S_img, S_txt)

x is an "image-patch" query strip (S_img=8 patches). y is a "text" key/value
strip (S_txt=5 tokens) at a DIFFERENT native width than x -- exactly like a
T5 (width 4096) or CLIP (width 2048) text encoder bolted onto a U-Net. K and
V must match Q in d_head, never in native width; W_K,W_V absorb whatever
width the text encoder ships. That is why you can swap text encoders without
touching the image tower.

We also hand-derive the attention matrix A = softmax(QK^T / sqrt(d_head))
once and assert it agrees with what the fused F.scaled_dot_product_attention
call produced -- so "one function" is not a black box, it's this formula.

SAFETY: CPU only, torch on CPU, < 1 s, allocates a few KB. Writes/installs
nothing.
"""

import math
import torch
import torch.nn.functional as F

SEED = 31
torch.manual_seed(SEED)

print("=" * 70)
print("self_and_cross.py -- self vs cross-attention: one function, two K/V sources")
print(f"torch {torch.__version__} - seed {SEED}")
print("=" * 70)

# --------------------------------------------------------------------------
# Setup: an image-patch query strip and a text key/value strip, at DIFFERENT
# native widths -- the "you can bolt a T5/CLIP onto the same U-Net" point.
# --------------------------------------------------------------------------
B = 1                     # batch
H = 1                     # single head -- the H-way split is page 30's story, not this page's
d_head = 128               # shared attention width Q,K,V all land in (constants.md sec 5)
S_img = 8                  # image-patch query strip length (page-31 quiz numeric: A has 8 rows)
S_txt = 5                  # text key/value strip length (page-31 quiz: A's columns track this)
d_img = 4096                # image-tower native width (Qwen3-scale, constants.md sec 1.1)
d_txt = 2048                # text-encoder native width (CLIP-scale) -- deliberately != d_img

x = torch.randn(B, S_img, d_img)     # the image patches ("the query, always")
y = torch.randn(B, S_txt, d_txt)     # the text tokens ("the key/value, in cross-attn")

# ONE query projection. It only ever looks at image patches, in BOTH call
# sites below -- self-attention and cross-attention share it verbatim.
W_Q = torch.nn.Linear(d_img, H * d_head, bias=False)

# Self-attention's own K,V: project the SAME image patches (width d_img).
W_K_self = torch.nn.Linear(d_img, H * d_head, bias=False)
W_V_self = torch.nn.Linear(d_img, H * d_head, bias=False)

# Cross-attention's own K,V: project the TEXT tokens (width d_txt != d_img).
# This is the whole trick: K must match Q in d_head (both land at 128), but
# the SOURCE width (2048) is free -- W_K,W_V absorb it. Swap in a T5 encoder
# (width 4096) tomorrow and only these two matrices' input dim changes.
W_K_cross = torch.nn.Linear(d_txt, H * d_head, bias=False)
W_V_cross = torch.nn.Linear(d_txt, H * d_head, bias=False)


def heads(t, seq_len):
    """(B, S, H*d_head) -> (B, H, S, d_head), the shape F.sdpa expects."""
    return t.view(B, seq_len, H, d_head).transpose(1, 2)


def attn(q, k, v):
    """The ONE function. No mask, no causality -- image self-attention is
    bidirectional (images are not time-ordered; that's page 32's story)."""
    return F.scaled_dot_product_attention(q, k, v)


def manual_attn(q, k, v):
    """Hand-derive A = softmax(QK^T / sqrt(d_head)) @ V, to show attn() above
    is exactly this formula, not a black box."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)   # (B,H,Sq,Sk)
    A = torch.softmax(scores, dim=-1)
    return A, A @ v


# --------------------------------------------------------------------------
# Call site 1: SELF-attention. Q, K, V all read the image strip.
# --------------------------------------------------------------------------
q_self = heads(W_Q(x), S_img)
k_self = heads(W_K_self(x), S_img)
v_self = heads(W_V_self(x), S_img)

out_self = attn(q_self, k_self, v_self)
A_self, out_self_manual = manual_attn(q_self, k_self, v_self)

print("\n-- self-attention: attn(Q=x, K=x, V=x) --")
print(f"  x (image patches) : {tuple(x.shape)}")
print(f"  q,k,v             : {tuple(q_self.shape)}  (all from x)")
print(f"  A = softmax(QK^T) : {tuple(A_self.shape)}   <- SQUARE, S_img x S_img")
print(f"  out               : {tuple(out_self.shape)}")

assert torch.allclose(out_self, out_self_manual, atol=1e-5), \
    "F.scaled_dot_product_attention must equal the hand softmax(QK^T/sqrt(d))@V formula"
print("  [OK] F.scaled_dot_product_attention == the hand-derived formula.")

# --------------------------------------------------------------------------
# Call site 2: CROSS-attention. Q still reads the image strip (SAME W_Q);
# K, V now read the text strip -- a different sequence, a different width.
# --------------------------------------------------------------------------
q_cross = heads(W_Q(x), S_img)              # identical call to q_self above
k_cross = heads(W_K_cross(y), S_txt)         # from y (text), not x
v_cross = heads(W_V_cross(y), S_txt)         # from y (text), not x

out_cross = attn(q_cross, k_cross, v_cross)   # the SAME attn() function
A_cross, out_cross_manual = manual_attn(q_cross, k_cross, v_cross)

print("\n-- cross-attention: attn(Q=x, K=y, V=y) --")
print(f"  y (text tokens)   : {tuple(y.shape)}   <- native width {d_txt}, NOT {d_img}")
print(f"  q                 : {tuple(q_cross.shape)}  (from x, same W_Q as self-attn)")
print(f"  k,v               : {tuple(k_cross.shape)}  (from y, via W_K_cross/W_V_cross)")
print(f"  A = softmax(QK^T) : {tuple(A_cross.shape)}   <- RECTANGULAR, S_img x S_txt")
print(f"  out               : {tuple(out_cross.shape)}")

assert torch.allclose(out_cross, out_cross_manual, atol=1e-5), \
    "F.scaled_dot_product_attention must equal the hand softmax(QK^T/sqrt(d))@V formula"
print("  [OK] F.scaled_dot_product_attention == the hand-derived formula.")

# --------------------------------------------------------------------------
# The self-check: self and cross differ ONLY in the K/V tensor and the
# column count of A. The Q projection, the rows of A, and the output shape
# are all untouched.
# --------------------------------------------------------------------------
print("\n-- the self-check: only K/V changed --")

assert torch.equal(q_self, q_cross), \
    "Q must be bit-identical: both call sites project the SAME x through the SAME W_Q"
print("  [OK] q_self == q_cross, bit-for-bit -- the query never noticed anything changed.")

assert k_self.shape[-2] != k_cross.shape[-2], "K's sequence length must differ (image vs text source)"
assert v_self.shape[-2] != v_cross.shape[-2], "V's sequence length must differ (image vs text source)"
print(f"  [OK] K,V source differs: self reads x ({S_img} tokens), cross reads y ({S_txt} tokens).")

assert A_self.shape[-2] == A_cross.shape[-2] == S_img, \
    "A's ROW count must be S_img in BOTH cases -- one row per query, and the query is always the image"
print(f"  [OK] rows(A_self) == rows(A_cross) == {S_img}  (page 29 fact #1: one output row per query).")

assert A_self.shape[-1] != A_cross.shape[-1], "A's COLUMN count must differ -- it tracks the K/V source length"
assert A_self.shape[-1] == S_img and A_cross.shape[-1] == S_txt, \
    f"A_self should be square ({S_img}x{S_img}), A_cross should be ({S_img}x{S_txt})"
print(f"  [OK] cols(A_self)={A_self.shape[-1]} (square)  vs  cols(A_cross)={A_cross.shape[-1]} (rectangular).")
print(f"       Quiz numeric: cross-attention A has {A_cross.shape[-2]} rows "
      f"(one per image patch), {A_cross.shape[-1]} columns (tracks the text length).")

assert out_self.shape == out_cross.shape == (B, H, S_img, d_head), \
    "the OUTPUT shape must be identical regardless of K/V source -- it always has S_img rows"
print(f"  [OK] out_self.shape == out_cross.shape == {tuple(out_self.shape)} "
      f"-- output width never depends on where K,V came from.")

# --------------------------------------------------------------------------
# The "aha": drop a word from the prompt -> a COLUMN vanishes, rows untouched,
# and the image patches re-weight their attention over the survivors.
# --------------------------------------------------------------------------
print("\n-- the aha: drop a text token, a column vanishes, rows are untouched --")
S_txt_dropped = S_txt - 1
y_dropped = y[:, :S_txt_dropped, :]                        # drop the last "word"
k_dropped = heads(W_K_cross(y_dropped), S_txt_dropped)
v_dropped = heads(W_V_cross(y_dropped), S_txt_dropped)
A_dropped, out_dropped = manual_attn(q_cross, k_dropped, v_dropped)

print(f"  text strip {S_txt} -> {S_txt_dropped} tokens")
print(f"  A_cross         : {tuple(A_cross.shape)}")
print(f"  A_dropped       : {tuple(A_dropped.shape)}")
assert A_dropped.shape[-2] == A_cross.shape[-2] == S_img, "dropping a text token must not change the row count"
assert A_dropped.shape[-1] == S_txt_dropped == A_cross.shape[-1] - 1, "dropping one text token removes one column"
assert out_dropped.shape == out_cross.shape, "output shape is unchanged -- still S_img rows, still d_head wide"
print(f"  [OK] rows stay at {S_img}; columns go {A_cross.shape[-1]} -> {A_dropped.shape[-1]}.")
print("       The image is still 8 patches querying. It just has one fewer key to look at.")
print("       The prompt is a SET of keys the image queries; add or remove keys and the")
print("       image just re-asks. That is why prompts are unreliable: you don't control the query.")

print("\n" + "=" * 70)
print("All self-checks passed: self- and cross-attention are the SAME function.")
print("Self:  attn(Q=x, K=x,   V=x)    -> square A, K/V from the image itself.")
print("Cross: attn(Q=x, K=y,   V=y)    -> rect.  A, K/V from a DIFFERENT sequence/width.")
print("The image is always the query. The text is always the key/value. That's the hinge.")
print("=" * 70)
