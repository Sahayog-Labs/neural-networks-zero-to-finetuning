#!/usr/bin/env python3
"""
build_block.py -- assemble ONE Qwen3-8B transformer block from scratch and let
PyTorch's own sum(p.numel()) confirm the two numbers page 35 derived by hand:
the block weighs 192,946,432 parameters, and the FFN is 78.26% of it.

Course artifact for p.35 ("The Transformer Block, Assembled -- and Why
Decoder-Only Won"). The block is the 2026 skeleton every open frontier model
repeats: pre-norm RMSNorm, GQA (8 KV heads tiled to 32), QK-Norm, RoPE on q,k
only, causal scaled-dot-product attention, then a SwiGLU FFN -- no biases
anywhere (attention_bias: false), RoPE with zero parameters. Nothing is elided;
the forward pass actually runs on a random input so the shapes are real.

Two self-checks, so the page's frozen numbers are proven on your machine, not
asserted at you:
  1. sum(p.numel()) over the whole block == 192,946,432 (constants.md 1.2).
  2. FFN params / block params == 78.26% (constants.md 1.3) -- and the attention
     and norm subtotals land on 41,943,040 and 8,448 on the way.

SAFETY: CPU only, torch on CPU, < 1 s, allocates a few MB. Writes/installs nothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Qwen3-8B config (constants.md 1.1) -- every param count below follows from these.
D, H, H_KV, D_HEAD, D_FF = 4096, 32, 8, 128, 12288
assert H * D_HEAD == D                          # query width == model width
EPS = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))  # learned gain -- the only params
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * self.g


def build_rope(seq, d_head, theta=1_000_000.0):
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    ang = torch.outer(torch.arange(seq).float(), inv)           # (S, d_head/2)
    emb = torch.cat([ang, ang], dim=-1)                         # (S, d_head)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):                                    # x: (B, Hn, S, d_head)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    rot = torch.cat([-x2, x1], dim=-1)                          # rotate_half
    return x * cos + rot * sin


def repeat_kv(t, n):                                            # (B, H_kv, S, d) -> (B, H_kv*n, S, d)
    b, h, s, d = t.shape
    return t[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms_1 = RMSNorm(D)                                 # pre-norm, attention (p.34)
        self.rms_2 = RMSNorm(D)                                 # pre-norm, FFN
        self.q_proj = nn.Linear(D, H * D_HEAD, bias=False)      # (4096, 4096) -> 32 heads
        self.k_proj = nn.Linear(D, H_KV * D_HEAD, bias=False)   # (1024, 4096) -> 8 KV heads: GQA (p.37)
        self.v_proj = nn.Linear(D, H_KV * D_HEAD, bias=False)   # (1024, 4096)
        self.o_proj = nn.Linear(H * D_HEAD, D, bias=False)      # (4096, 4096): the write port (p.30)
        self.q_norm = RMSNorm(D_HEAD)                           # QK-Norm (p.30)
        self.k_norm = RMSNorm(D_HEAD)
        self.gate = nn.Linear(D, D_FF, bias=False)              # SwiGLU: THREE matrices, no bias
        self.up = nn.Linear(D, D_FF, bias=False)
        self.down = nn.Linear(D_FF, D, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        h = self.rms_1(x)                                       # normalize a COPY, keep +x clean
        q = self.q_norm(self.q_proj(h).view(B, S, H, D_HEAD)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(B, S, H_KV, D_HEAD)).transpose(1, 2)
        v = self.v_proj(h).view(B, S, H_KV, D_HEAD).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # RoPE on q,k only -- never v (p.33)
        k, v = repeat_kv(k, H // H_KV), repeat_kv(v, H // H_KV)  # tile 8 KV heads -> 32
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # causal mask = the LLM hinge (p.32)
        x = x + self.o_proj(a.transpose(1, 2).reshape(B, S, D))      # highway #1 (p.34)
        h = self.rms_2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))         # SwiGLU FFN + highway #2
        return x


def main():
    print("=" * 70)
    print("build_block.py -- one Qwen3-8B block, counted by PyTorch itself")
    print(f"torch {torch.__version__} - CPU")
    print("=" * 70)

    torch.manual_seed(35)
    block = TransformerBlock()

    # It actually runs -- a real forward on a random residual stream, S=16 tokens.
    B, S = 1, 16
    cos, sin = build_rope(S, D_HEAD)
    cos, sin = cos[None, None], sin[None, None]                 # broadcast over (B, heads)
    x = torch.randn(B, S, D)
    y = block(x, cos, sin)
    assert y.shape == x.shape, "a block maps the residual stream to itself"
    print(f"\nforward: x {tuple(x.shape)} -> y {tuple(y.shape)}  (block preserves the stream shape)")

    # Subtotals, straight from the module tree -- no hand-typed counts.
    def count(*mods):
        return sum(p.numel() for m in mods for p in m.parameters())

    attn = count(block.q_proj, block.k_proj, block.v_proj, block.o_proj,
                 block.q_norm, block.k_norm)
    attn_lin = count(block.q_proj, block.k_proj, block.v_proj, block.o_proj)
    ffn = count(block.gate, block.up, block.down)
    norm = count(block.rms_1, block.rms_2, block.q_norm, block.k_norm)
    total = sum(p.numel() for p in block.parameters())

    print("\n-- subtotals (PyTorch's own numel) --")
    print(f"attention (q,k,v,o proj)  : {attn_lin:>13,}")
    print(f"FFN (gate,up,down)        : {ffn:>13,}")
    print(f"norms (2 RMSNorm + QK-Norm): {norm:>13,}")
    print(f"block total               : {total:>13,}")

    # 1. The whole block is exactly the frozen integer.
    assert attn_lin == 41_943_040, f"attention subtotal {attn_lin:,} != 41,943,040"
    assert ffn == 150_994_944, f"FFN subtotal {ffn:,} != 150,994,944"
    assert norm == 8_448, f"norm subtotal {norm:,} != 8,448"
    assert total == 192_946_432, f"block total {total:,} != 192,946,432"
    print(f"\n[OK] sum(p.numel()) == 192_946_432 -- matches the page, to the parameter.")

    # 2. The FFN is 78.26% of the block.
    share = ffn / total * 100
    print(f"[OK] FFN share = {ffn:,} / {total:,} = {share:.2f}%")
    assert abs(share - 78.26) < 0.01, f"FFN share {share:.2f}% != 78.26%"
    print("     attention -- the famous part -- is barely a fifth "
          f"({attn_lin / total * 100:.2f}%). No biases; RoPE is 0 params.")

    print("\n" + "=" * 70)
    print("Both self-checks passed: the block runs, weighs 192,946,432 params,")
    print("and the FFN holds 78.26% of them.")
    print("=" * 70)


if __name__ == "__main__":
    main()
