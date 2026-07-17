"""
ledger.py -- the pure-arithmetic memory model for the ANN course.

ONE source of truth for the four-bucket training-memory ledger, shared *verbatim* by:
  - code/05_memory_ledger.py   (the p.49 CLI -- "RUNG 9", the calculator as a program)
  - the page-18 trunk companion (the same numbers, live in the browser)

so the page and the script can never drift. Every function here is integer/exact-rational
arithmetic: no torch, no GPU, no network, no I/O. It runs in microseconds on a CPU and is
the thing the self-tests assert against constants.md. Import it; do not reimplement it.

The four buckets (constants.md sec2.1):
  W  weights      -- bf16 resident copy the GPU computes with          2 B/trainable param
  G  gradients    -- one per trainable weight                          2 B/trainable param
  O  optimizer    -- fp32 master + Adam m + Adam v (adamw_mixed)      12 B/trainable param
  A  activations  -- the forward tensors kept for backward   [EST], NOT modelled here
                     (a separate, separately-labelled line -- never folded into W/G/O)

W + G + O = 16 B/param for a full AdamW-mixed fine-tune. That 16 is "THE number"
(constants.md sec2.2). A is deliberately excluded from every function below: it is an
estimate the learner MEASURES, never a frozen fact, and LoRA does NOT reduce it
(constants.md sec2.4).

Units discipline (constants.md sec0): 1 GB = 1e9 B (decimal, how memory is marketed);
1 GiB = 2**30 B (binary, how DRAM actually addresses). Never mix them when sizing.
"""

GiB = 1 << 30          # 1,073,741,824  -- binary gibibyte
GB = 10 ** 9           # 1,000,000,000  -- decimal gigabyte
MiB = 1 << 20
KiB = 1 << 10
MB = 10 ** 6

# --------------------------------------------------------------------------- #
# Per-parameter byte costs -- constants.md sec2.1, "the course's most reused table".
# --------------------------------------------------------------------------- #

WEIGHT_BYTES = 2       # bf16 resident weight, per trainable param
GRAD_BYTES = 2         # bf16 gradient,        per trainable param

# k_O = optimizer-state bytes per TRAINABLE param (state only; weights+grads are separate).
#   adamw_mixed       fp32 master 4 + Adam m 4 + Adam v 4                       = 12
#   adam8bit          fp32 master 4 + 8-bit m 1 + 8-bit v 1                     =  6
#   adam8bit_no_master             8-bit m 1 + 8-bit v 1  (stalled-update risk) =  2
#   sgd_momentum      fp32 master 4 + fp32 momentum buffer 4                    =  8
#   sgd               fp32 master 4                                            =  4
OPT_STATE_BYTES = {
    "adamw_mixed": 12,
    "adam8bit": 6,
    "adam8bit_no_master": 2,
    "sgd_momentum": 8,
    "sgd": 4,
}

# Frozen base-weight cost per param, for the FROZEN base under LoRA/QLoRA (constants.md sec3):
#   bf16    2 bytes                  = 16 bits
#   nf4     4.5 bits (DQ *off*)      = 0.5625 B/param   (the un-opted-in default -- sec3 caveat)
#   nf4_dq  4 + 65/512 bits (DQ on)  = 2113/4096 B/param = 0.515869 B/param  (arXiv 2305.14314)
#   int8    8 bits                   = 1.0 B/param
BASE_WEIGHT_BYTES = {
    "bf16": 2.0,
    "nf4": 0.5625,
    "nf4_dq": 2113 / 4096,      # 4.126953125 bits/param, verbatim from constants.md sec3
    "int8": 1.0,
}

# Default LoRA target set: "all-linear" = the 7 linear projections per block (D-06:
# attention-only froze 78% of the block, so the course default is all-linear).
ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# --------------------------------------------------------------------------- #
# Config normalisation. Accepts a raw HuggingFace config.json dict (any Qwen3-/Llama-
# style decoder) and fills the fields the arithmetic needs, with the same defaults
# transformers uses. QK-Norm (Qwen3 adds a per-head RMSNorm on q and k) is inferred
# from model_type/architectures unless the config states it outright.
# --------------------------------------------------------------------------- #

def _has_qk_norm(cfg):
    if "use_qk_norm" in cfg:
        return bool(cfg["use_qk_norm"])
    mt = str(cfg.get("model_type", "")).lower()
    arch = " ".join(cfg.get("architectures", []) or []).lower()
    return mt.startswith("qwen3") or "qwen3" in arch


def normalize(cfg):
    """Pull the ledger-relevant dimensions out of a HF config dict, applying HF's own
    defaults. Returns a plain dict; does not mutate the input."""
    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg.get("head_dim", hidden // heads))
    kv_heads = int(cfg.get("num_key_value_heads", heads))   # HF default: = heads (i.e. MHA)
    return {
        "hidden": hidden,
        "layers": int(cfg["num_hidden_layers"]),
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate": int(cfg["intermediate_size"]),
        "vocab": int(cfg["vocab_size"]),
        "tie": bool(cfg.get("tie_word_embeddings", False)),
        "qk_norm": _has_qk_norm(cfg),
    }


# --------------------------------------------------------------------------- #
# Parameter counting. Reproduces constants.md sec1.2 exactly (8,190,735,360 for Qwen3-8B),
# to the byte, from the config alone.
# --------------------------------------------------------------------------- #

def count_params(cfg):
    """Exact parameter count for a decoder-only transformer with SwiGLU FFN (gate+up+down)
    and RMSNorm, from a HF config dict. Reproduces the checkpoint's own integer -- no
    'closes to three sig figs'; it closes to the byte (constants.md sec3)."""
    c = normalize(cfg)
    d, L, H, Hkv, hd = c["hidden"], c["layers"], c["heads"], c["kv_heads"], c["head_dim"]
    dff, V, tie, qk = c["intermediate"], c["vocab"], c["tie"], c["qk_norm"]

    # Attention: q_proj + o_proj use H heads; k_proj + v_proj use Hkv heads (GQA). No bias.
    attn = 2 * d * hd * (H + Hkv)          # (q+o) + (k+v), factored
    ffn = 3 * d * dff                       # gate + up + down (SwiGLU)
    norms = 2 * d                           # input_layernorm + post_attention_layernorm
    if qk:
        norms += 2 * hd                     # q_norm + k_norm (QK-Norm, Qwen3)
    per_block = attn + ffn + norms
    blocks = per_block * L

    embed = V * d
    lm_head = 0 if tie else V * d
    final_norm = d
    total = blocks + embed + lm_head + final_norm

    return {
        "total": total,
        "non_embed": blocks,               # the 6.95B "non-embedding" figure (excl. final norm)
        "per_block": per_block,
        "attn_per_block": attn,
        "ffn_per_block": ffn,
        "embeddings": embed + lm_head,
        "final_norm": final_norm,
    }


# --------------------------------------------------------------------------- #
# LoRA parameter counting. Per matrix a LoRA adapter adds r*(d_in + d_out) params
# (two factors A: r x d_in and B: d_out x r), vs the frozen d_in*d_out (constants.md sec3).
# --------------------------------------------------------------------------- #

def _linear_shapes(c):
    """(d_out, d_in) for each of the 7 linear projections, from normalized dims."""
    d, H, Hkv, hd, dff = c["hidden"], c["heads"], c["kv_heads"], c["head_dim"], c["intermediate"]
    q_out = H * hd
    kv_out = Hkv * hd
    return {
        "q_proj": (q_out, d),
        "k_proj": (kv_out, d),
        "v_proj": (kv_out, d),
        "o_proj": (d, q_out),
        "gate_proj": (dff, d),
        "up_proj": (dff, d),
        "down_proj": (d, dff),
    }


def lora_params(cfg, r=16, targets=ALL_LINEAR):
    """Trainable LoRA parameter count for the given targets, all L blocks.
    Qwen3-8B r=16 all-linear -> 43,646,976 (constants.md sec3)."""
    c = normalize(cfg)
    shapes = _linear_shapes(c)
    per_block = 0
    for name in targets:
        if name not in shapes:
            raise KeyError(f"unknown LoRA target {name!r}; known: {sorted(shapes)}")
        d_out, d_in = shapes[name]
        per_block += r * (d_in + d_out)
    return per_block * c["layers"]


# --------------------------------------------------------------------------- #
# The per-param cost tuple, and the state totals. THE four-bucket arithmetic.
# --------------------------------------------------------------------------- #

def bytes_per_param(optimizer="adamw_mixed"):
    """(W, G, O) bytes for ONE trainable parameter under bf16 mixed precision.
    Sum is the famous 16 for adamw_mixed (constants.md sec2.1). Activations (A) are
    NOT here -- they are estimated and measured, never a per-param constant."""
    if optimizer not in OPT_STATE_BYTES:
        raise KeyError(f"unknown optimizer {optimizer!r}; known: {sorted(OPT_STATE_BYTES)}")
    return (WEIGHT_BYTES, GRAD_BYTES, OPT_STATE_BYTES[optimizer])


def full_ft_state(P, optimizer="adamw_mixed"):
    """State bytes (W+G+O, NO activations) for a FULL fine-tune of P parameters.
    Qwen3-8B, adamw_mixed -> 131,051,765,760 B = 131.05 GB = 122.05 GiB (constants.md sec2.2)."""
    w, g, o = bytes_per_param(optimizer)
    return P * (w + g + o)


def lora_state(cfg, r=16, targets=ALL_LINEAR, optimizer="adamw_mixed", base="bf16"):
    """Four-bucket state bytes for a LoRA fine-tune: the FROZEN base (P params at `base`
    precision, weights only -- no grads, no optimizer) plus the trainable adapter
    (16 B/param under adamw_mixed). Qwen3-8B r=16 all-linear, bf16 base
    -> 17,079,822,336 B = 17.08 GB / 15.91 GiB (constants.md sec2.3)."""
    P = count_params(cfg)["total"]
    A = lora_params(cfg, r=r, targets=targets)
    w, g, o = bytes_per_param(optimizer)
    base_bytes = round(P * BASE_WEIGHT_BYTES[base])          # frozen weights (whole bytes)
    adapter_bytes = A * (w + g + o)                          # trainable: full 16 B/param
    # Bucketed so the CLI can print W / G / O separately and have them sum exactly.
    return {
        "base_weight_bytes": base_bytes,                    # -> W (frozen)
        "adapter_weight_bytes": A * w,                      # -> W (trainable)
        "grad_bytes": A * g,                                # -> G
        "opt_bytes": A * o,                                 # -> O
        "total": base_bytes + adapter_bytes,
        "trainable_params": A,
        "P": P,
    }


def full_state_buckets(P, optimizer="adamw_mixed"):
    """W / G / O byte buckets for a full fine-tune, summing to full_ft_state(P, optimizer)."""
    w, g, o = bytes_per_param(optimizer)
    return {"weight_bytes": P * w, "grad_bytes": P * g, "opt_bytes": P * o,
            "total": P * (w + g + o)}


# --------------------------------------------------------------------------- #
# KV cache -- the inference-time ledger (constants.md sec4). Not part of the training
# state buckets, but lives here because it is the same config-driven arithmetic and the
# p.18 companion references it. Note H_kv, NOT H -- the field's single most common error.
# --------------------------------------------------------------------------- #

def kv_bytes(cfg, b=2, S=40960):
    """KV-cache bytes for S tokens (batch 1): 2 * L * H_kv * d_head * b * S.
    Qwen3-8B, bf16, full context -> 6,039,797,760 B = 5.625 GiB (constants.md sec4)."""
    c = normalize(cfg)
    return 2 * c["layers"] * c["kv_heads"] * c["head_dim"] * b * S


# --------------------------------------------------------------------------- #
# The trainable-state shrink -- the "187x" identity (constants.md sec2.3), the better
# sentence than "7.7x total". LoRA does not shrink weights; it deletes the optimizer.
# --------------------------------------------------------------------------- #

def trainable_state_bytes(n_trainable, optimizer="adamw_mixed"):
    """G + O bytes for n_trainable params -- the part LoRA actually collapses.
    Excludes the resident weights (which do not shrink). Full Qwen3-8B: 14 B/param
    * 8.19e9 = 114.67 GB; LoRA r=16: 14 * 43.6e6 = 0.61 GB; ratio = 187.7 = P/lora."""
    _, g, o = bytes_per_param(optimizer)
    return n_trainable * (g + o)


if __name__ == "__main__":
    # A quick self-check when run directly (the CLI's --self-test does the full sweep).
    QWEN3_8B = dict(hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
                    num_key_value_heads=8, head_dim=128, intermediate_size=12288,
                    vocab_size=151936, tie_word_embeddings=False, model_type="qwen3")
    assert count_params(QWEN3_8B)["total"] == 8_190_735_360
    assert lora_params(QWEN3_8B) == 43_646_976
    assert full_ft_state(8_190_735_360) == 131_051_765_760
    assert kv_bytes(QWEN3_8B) == 6_039_797_760
    print("ledger.py self-check OK:",
          f"P=8,190,735,360  LoRA=43,646,976  full=131.05 GB  KV=5.625 GiB")
