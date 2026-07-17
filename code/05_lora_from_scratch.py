#!/usr/bin/env python3
"""
05_lora_from_scratch.py - LoRA in ~40 lines of PyTorch, no `peft`. RUNG 5, p.40.

The page derived the trick once, to the byte. This script is the same trick in
runnable code: implement a LoRA linear by hand (the LoRALinear class below is the
~40 load-bearing lines), wrap Qwen3-8B's `q_proj` with it, and confirm the exact
count the worked table claims -

    trainable: 43,646,976 || all: 8,234,382,336 || 0.53%

- appears on your own box, `all-linear`, r=16. Not a toy: 43,646,976 is
constants.md §3, the same integer page 40's ledger reconciles to 8,190,735,360.

It does four things, in order:

  1. The LoRALinear class: h = W0 x + (alpha/r) B A x, with A ~ N(0, sigma^2) and
     B init ZEROS. Freezing W0 is a requires_grad_(False) - the .detach() beat.
  2. The B-first proof, on a real autograd graph: at step 0, BA == 0 so the wrapped
     layer is bit-identical to the base, AND grad(A) is *identically* zero while
     grad(B) is not - because dL/dA passes through B (=0) but dL/dB passes through
     A (!=0). The adapter unfreezes itself, B first (page 40's chain-rule beat).
  3. The count, from Qwen3-8B's config arithmetic: attention-only vs `all-linear`.
     Flip target_modules to `all-linear` and the count lands on 43,646,976 = the
     worked table. This part is pure Python and self-checks WITHOUT downloading 8B.
  4. Optional (--model): load the real Qwen3-8B in bf16, wrap every linear with the
     hand-written LoRALinear, and print the same trainable||all||pct line off the
     actual checkpoint, so nothing here is asserted - it is confirmed.

Steps 1-3 run on a laptop CPU in seconds (spec-code §rung-5: rungs 1-5 need no
Spark). Step 4 needs the 8B checkpoint and ~1 min. Requires: torch (for 1-2, 4);
transformers (for 4 only). The diffusion track ships a SEPARATE lora_from_scratch.py
(FLUX.2 host, same equation) - see spec-code §F for why both exist.

Usage
-----
    python 05_lora_from_scratch.py                 # steps 1-3, no model download
    python 05_lora_from_scratch.py --model Qwen/Qwen3-8B   # + step 4, the real thing
"""

import argparse
import sys

GB = 10 ** 9

# --------------------------------------------------------------------------- #
# Qwen3-8B linear inventory - constants.md §1.2/§3, (out x in). These are the
# seven nn.Linear matrices per decoder layer; norms are NOT LoRA targets (D-07),
# which is exactly why the per-linear subtotal (6,945,767,424) is not the model's
# parameter count until the norms + embeddings are added back.
# --------------------------------------------------------------------------- #
N_LAYERS = 36
P_TOTAL = 8_190_735_360          # constants.md §3 - the full parameter count, exact
LINEARS = [
    # name,        d_out,  d_in
    ("q_proj",     4096,   4096),
    ("k_proj",     1024,   4096),
    ("v_proj",     1024,   4096),
    ("o_proj",     4096,   4096),
    ("gate_proj",  12288,  4096),
    ("up_proj",    12288,  4096),
    ("down_proj",  4096,   12288),
]
ATTENTION_ONLY = {"q_proj", "k_proj", "v_proj", "o_proj"}


def lora_params(d_out, d_in, r):
    """LoRA trades d_in*d_out parameters for r*(d_in + d_out) - the A (r,d_in)
    and B (d_out,r) matrices, nothing else. This is the whole arithmetic."""
    return r * (d_in + d_out)


def count_trainable(r=16, target="all-linear"):
    """Per-layer and whole-model LoRA trainable count for a target_modules choice,
    computed straight from the config. Returns (per_layer, total, per_matrix)."""
    names = None if target == "all-linear" else ATTENTION_ONLY
    per_matrix = {}
    per_layer = 0
    for name, d_out, d_in in LINEARS:
        if names is not None and name not in names:
            continue
        p = lora_params(d_out, d_in, r)
        per_matrix[name] = p
        per_layer += p
    return per_layer, per_layer * N_LAYERS, per_matrix


# --------------------------------------------------------------------------- #
# Step 3 - the count, from arithmetic (no torch, no download).
# --------------------------------------------------------------------------- #

def report_counts(r=16):
    print("=" * 72)
    print(f"THE COUNT - Qwen3-8B, LoRA r={r}, from config arithmetic (constants.md §3)")
    print("=" * 72)

    # The default reflex many people reach for first: attention-only.
    attn_per_layer, attn_total, _ = count_trainable(r, target="q,k,v,o")
    print(f"  target_modules = attention-only (q,k,v,o):")
    print(f"    per layer {attn_per_layer:>12,}  x{N_LAYERS} = {attn_total:>13,} "
          f"= {100 * attn_total / P_TOTAL:.3f}% of P")
    print(f"    (this is the stale advice page 40 warns about - it froze 78% of the model)")
    print()

    # Flip to all-linear - the worked table.
    per_layer, total, per_matrix = count_trainable(r, target="all-linear")
    print(f"  target_modules = 'all-linear' (the 7 linears x {N_LAYERS} layers):")
    for name, d_out, d_in in LINEARS:
        base = d_in * d_out
        print(f"    {name:<10} {d_out:>5} x {d_in:<5}  base {base:>12,}  "
              f"LoRA {per_matrix[name]:>9,}")
    print(f"    {'per layer':<10} {'':>13} {'':>17}  {'':>12}  "
          f"       {per_layer:>9,}")
    print(f"    {'x' + str(N_LAYERS) + ' layers':<10} {'':>13} {'':>17}  {'':>12}  "
          f"       {total:>9,}")
    print()

    all_params = P_TOTAL + total
    pct_of_all = 100 * total / all_params      # peft's print_trainable_parameters() ratio
    pct_of_P = 100 * total / P_TOTAL           # the worked table's 0.533%
    print(f"  >> trainable: {total:,} || all: {all_params:,} || {pct_of_all:.2f}%")
    print(f"     ({pct_of_P:.3f}% of the base P = {P_TOTAL:,} - 'one parameter in 188')")

    adapter_mb = total * 2 / GB * 1000         # bf16, 2 B/param, in MB (1e6 base)
    print(f"     adapter file, bf16: {total:,} x 2 B = {adapter_mb:.1f} MB - you can email it")
    print()

    # ------------- self-checks against the frozen constants (§3) ------------- #
    assert total == 43_646_976, (
        f"all-linear r=16 must be 43,646,976 (constants.md §3), got {total:,}")
    assert per_layer == 1_212_416, (
        f"per-layer must be 1,212,416 (constants.md §3), got {per_layer:,}")
    assert all_params == 8_234_382_336, (
        f"all = P + trainable must be 8,234,382,336, got {all_params:,}")
    assert abs(pct_of_P - 0.533) < 0.001, f"expected 0.533% of P, got {pct_of_P:.3f}%"
    assert abs(adapter_mb - 87.3) < 0.1, f"adapter file must be 87.3 MB, got {adapter_mb:.1f}"
    # per-matrix values, verbatim from the constants table
    expected = {"q_proj": 131_072, "k_proj": 81_920, "v_proj": 81_920,
                "o_proj": 131_072, "gate_proj": 262_144, "up_proj": 262_144,
                "down_proj": 262_144}
    assert per_matrix == expected, f"per-matrix LoRA counts drifted: {per_matrix}"
    print("  self-checks passed: 43,646,976 || 8,234,382,336 || 0.53% (constants.md §3).")
    print()
    return total, all_params


# --------------------------------------------------------------------------- #
# Steps 1-2 - the LoRA linear itself (~40 lines) and the B-first proof. Needs torch.
# --------------------------------------------------------------------------- #

def torch_demo():
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        """A frozen base linear W0 with a trainable low-rank adapter:

            h = W0 x  +  (alpha / r) * B @ A @ x

        A is (r, d_in), init N(0, sigma^2); B is (d_out, r), init ZEROS -> B@A = 0,
        so the wrapped layer is bit-identical to the base at step 0. Freezing W0 is
        just requires_grad_(False): the optimizer never allocates a grad/master/m/v
        slot for it, and that single fact is the entire memory win (page 40)."""

        def __init__(self, base, r=16, alpha=16, sigma=None):
            super().__init__()
            d_out, d_in = base.weight.shape
            self.base = base
            self.base.weight.requires_grad_(False)                 # freeze W0 (.detach())
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)
            self.r, self.alpha, self.scaling = r, alpha, alpha / r
            sigma = (1.0 / d_in) if sigma is None else sigma
            self.A = nn.Parameter(torch.randn(r, d_in) * sigma)    # down-projection (r, d_in)
            self.B = nn.Parameter(torch.zeros(d_out, r))           # up-projection  (d_out, r), ZERO

        def forward(self, x):                                      # x: (..., d_in)
            ax = x @ self.A.t()                                    # (..., r)      -- squeeze to rank
            bax = ax @ self.B.t()                                  # (..., d_out)  -- lift back up
            return self.base(x) + self.scaling * bax               # W0 x + (alpha/r) B A x

        def trainable_numel(self):
            return self.A.numel() + self.B.numel()

    print("=" * 72)
    print("STEP 1-2 - the LoRALinear class, and the B-first proof (real autograd)")
    print("=" * 72)

    torch.manual_seed(0)
    d_in, d_out, r = 4096, 4096, 16
    base = nn.Linear(d_in, d_out, bias=False)
    lora = LoRALinear(base, r=r, alpha=16)

    # (a) bit-identical to the base at step 0, because B = 0 -> BA = 0 -> dW = 0.
    x = torch.randn(4, d_in)
    with torch.no_grad():
        h_base = base(x)
        h_lora = lora(x)
        max_diff = (h_base - h_lora).abs().max().item()
    print(f"  q_proj wrapped: {d_out}x{d_in}, r={r}. Trainable in adapter: "
          f"{lora.trainable_numel():,} (= 131,072, matches the table).")
    print(f"  step 0: max|W0 x  -  LoRA(x)| = {max_diff:.2e}  -> bit-identical to base (B=0).")
    assert max_diff == 0.0, "B=0 must make the wrapped layer EXACTLY the base at step 0"

    # (b) B moves first: one backward, then read the gradient norms.
    target = torch.randn(4, d_out)
    loss = ((lora(x) - target) ** 2).mean()
    loss.backward()
    gA = lora.A.grad.norm().item()
    gB = lora.B.grad.norm().item()
    print(f"  step 1: ||grad A|| = {gA:.3e}  (identically zero - A is frozen until B moves)")
    print(f"          ||grad B|| = {gB:.3e}  (nonzero - B moves FIRST)")
    print(f"  dL/dA passes through B (=0), so it vanishes; dL/dB passes through A (!=0),")
    print(f"  so B is what moves on step 1. The adapter unfreezes itself. (page 40, D-05.)")
    assert gA == 0.0, "grad(A) must be exactly 0 at step 0 (it is proportional to B=0)"
    assert gB > 0.0, "grad(B) must be nonzero at step 0 (it is proportional to A!=0)"
    # base weight got no gradient at all - it was frozen.
    assert base.weight.grad is None, "the frozen base must receive NO gradient"
    print("  self-checks passed: BA==0 at init, grad(A)==0, grad(B)!=0, W0 frozen.")
    print()
    return LoRALinear


# --------------------------------------------------------------------------- #
# Step 4 - optional: the real Qwen3-8B, wrapped with the hand-written LoRALinear.
# --------------------------------------------------------------------------- #

def real_model_demo(model_name, LoRALinear, r=16):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM

    print("=" * 72)
    print(f"STEP 4 - the real thing: {model_name} in bf16, every linear wrapped by hand")
    print("=" * 72)

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)                          # freeze the whole base first

    # Wrap every nn.Linear inside the decoder layers = target_modules="all-linear".
    wrapped = 0
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and any(
                    t in child_name for t, _, _ in [(n, 0, 0) for n, _, _ in LINEARS]):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=16))
                wrapped += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  wrapped {wrapped} linear modules with LoRALinear (all-linear).")
    print(f"  >> trainable: {trainable:,} || all: {total:,} || "
          f"{100 * trainable / total:.2f}%")
    # The count off the real checkpoint must match the arithmetic exactly.
    assert trainable == 43_646_976, (
        f"real-model trainable count {trainable:,} != 43,646,976 - a target_modules or "
        f"config mismatch (constants.md §3).")
    print("  confirmed on the checkpoint: 43,646,976. Nothing here was asserted - it was measured.")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="HF id / local path of Qwen3-8B for the optional real run (step 4)")
    ap.add_argument("-r", "--rank", type=int, default=16, help="LoRA rank (default 16)")
    args = ap.parse_args()

    # Steps 1-2 need torch; if it is missing, still run the pure-arithmetic count (step 3).
    LoRALinear = None
    try:
        LoRALinear = torch_demo()
    except ImportError:
        print("torch not importable - skipping the LoRALinear/autograd demo (steps 1-2).")
        print("The count below (step 3) is pure arithmetic and still self-checks.\n")

    report_counts(r=args.rank)

    if args.model:
        if LoRALinear is None:
            print("--model needs torch + transformers, which are not importable. "
                  "Install them into a FRESH venv (never ComfyUI's) and rerun.")
            sys.exit(1)
        real_model_demo(args.model, LoRALinear, r=args.rank)

    print("=" * 72)
    print("All sections passed their self-checks. LoRA, from scratch, to the byte.")
    print("=" * 72)


if __name__ == "__main__":
    main()
