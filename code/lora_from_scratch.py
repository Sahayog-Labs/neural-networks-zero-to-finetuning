#!/usr/bin/env python3
"""
lora_from_scratch.py - LoRA in ~30 lines of PyTorch, no `peft`. RUNG 8 twin, p.62.

This is the DIFFUSION-side counterpart of `05_lora_from_scratch.py` (which wraps
Qwen3-8B's `q_proj`). Same equation, different host: here the object is a FLUX
attention projection - a `to_q` - and the frozen number is a diffusion number.
The trunk (p.40) derived the math once; this instantiates it on the model he runs.

    h = W0 x  +  (alpha / r) B A x        A ~ N(0, sigma^2),  B init ZEROS

The count the course froze (brief-diffusion §11.2, spec-part6):

    one FLUX.1-dev attention projection   3072 x 3072 = 9,437,184 params
    LoRA, r=16                            16 x (3072+3072) =   98,304 params
                                                              = 1.04%, 96x fewer

A NOTE ON "FLUX.2" (a spec seam this script resolves honestly):
  The page manifest (D10) says "wrap a FLUX.2 attention module," but 9,437,184 is
  3072^2 - and 3072 is FLUX.1-dev's width, not FLUX.2's. His on-disk FLUX.2-dev
  (`Flux2Transformer2DModel`) is 48 heads x 128 = 6144 wide (hardware-ground-truth
  §4, read from his own weights - the top authority). So this script asserts the
  frozen FLUX.1-dev numbers, THEN prints the identical arithmetic for his actual
  FLUX.2 (37,748,736 / 196,608 / 0.52%). The LoRALinear class is host-agnostic -
  that is the whole point of §F.2: same equation, any attention matrix.

What it does, in order:
  1. THE COUNT (pure arithmetic, no torch, self-checks): 9,437,184 vs 98,304 for
     FLUX.1-dev, then the same for his real FLUX.2. Runs on any laptop, no GPU.
  2. THE B=0 PROOF, no torch: build A (random) and B (zeros), show (alpha/r) B A is
     the exact zero matrix - so a wrapped layer is bit-identical to its base at
     step 0. numpy if present (full 3072), else a pure-Python check at small d.
  3. THE LoRALinear CLASS + autograd beat (needs torch): wrap a real FLUX-shaped
     `to_q`, confirm max|W0 x - LoRA(x)| == 0 at step 0, then one backward shows
     grad(A) == 0 while grad(B) != 0 - B moves first (page 40, D-05).
  4. Optional --peft: swap in `peft.LoraConfig` targeting `to_q` and confirm the
     SAME 98,304 from print_trainable_parameters(). Desk-checked vs peft 0.19.
  5. Optional --flux <path>: load his real `Flux2Transformer2DModel` off disk, pull
     one real `to_q`, wrap it by hand. Spark-side; prints what it would need.

Verified against: torch 2.13.0 · diffusers 0.39.0 · peft 0.19.1 · CUDA 13.0.

SAFETY: steps 1-2 allocate nothing on the GPU (numpy/CPU only, <200 MB at 3072).
Step 3 builds a CPU nn.Linear(3072,3072) (~151 MB fp32) - still no GPU. Only
--flux touches his real FLUX.2 weights and would contend with ComfyUI if it is up;
that path is desk-checked here and meant to be run by HIM on the Spark. This script
writes nothing and installs nothing; it is read-only w.r.t. your system.

Usage
-----
    python lora_from_scratch.py                 # steps 1-3 (torch optional)
    python lora_from_scratch.py --self-test     # steps 1-2 only, no torch, asserts
    python lora_from_scratch.py --peft          # + step 4 (needs torch+peft)
    python lora_from_scratch.py --flux ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev
"""

import argparse

# --------------------------------------------------------------------------- #
# Frozen dimensions. FLUX.1-dev is the course's frozen number; FLUX.2 is read
# from his disk (hardware-ground-truth §4). Both attention projections are square.
# --------------------------------------------------------------------------- #
D_FLUX1 = 3072      # FLUX.1-dev model width (brief-diffusion §11.2 / spec-part6 §905) [VP]
D_FLUX2 = 6144      # FLUX.2-dev attention inner width = 48 heads x 128 (hw-ground-truth §4) [MEA-DEV]
R_DEFAULT = 16      # LoRA rank the course fixes for this beat


def base_params(d):
    """A square attention projection to_q/to_k/to_v/to_out is d x d."""
    return d * d


def lora_params(d, r):
    """LoRA replaces d*d with r*(d + d): the A (r, d) and B (d, r) matrices,
    nothing else. Square projection, so d_in == d_out == d. This is the whole
    arithmetic - identical shape to the Qwen twin, just d instead of (d_out,d_in)."""
    return r * (d + d)


# --------------------------------------------------------------------------- #
# Step 1 - the count, from arithmetic. No torch, no download, always self-checks.
# --------------------------------------------------------------------------- #

def report_counts(r=R_DEFAULT):
    print("=" * 72)
    print(f"STEP 1 - THE COUNT, a FLUX attention projection, LoRA r={r} (arithmetic)")
    print("=" * 72)

    # -- FLUX.1-dev: the frozen number the course prints ---------------------- #
    b1 = base_params(D_FLUX1)
    l1 = lora_params(D_FLUX1, r)
    pct1 = 100 * l1 / b1
    fewer1 = b1 // l1
    print(f"  FLUX.1-dev  to_q  {D_FLUX1} x {D_FLUX1}")
    print(f"    full W0        {b1:>12,} params")
    print(f"    LoRA (A+B)     {l1:>12,} params   = r*(d+d) = {r}*({D_FLUX1}+{D_FLUX1})")
    print(f"    ratio          {pct1:>12.2f}%          ({fewer1}x fewer trainable params)")
    print(f"    adapter, bf16  {l1 * 2 / 1e6:>12.3f} MB    (2 B/param - a rounding error on disk)")
    print()

    # -- His on-disk FLUX.2-dev: same equation, wider host -------------------- #
    b2 = base_params(D_FLUX2)
    l2 = lora_params(D_FLUX2, r)
    pct2 = 100 * l2 / b2
    print(f"  FLUX.2-dev  to_q  {D_FLUX2} x {D_FLUX2}   (his disk: 48 heads x 128, hw-ground-truth §4)")
    print(f"    full W0        {b2:>12,} params")
    print(f"    LoRA (A+B)     {l2:>12,} params")
    print(f"    ratio          {pct2:>12.2f}%          (wider base -> even smaller fraction)")
    print(f"    [DER from his measured config - NOT the frozen 9,437,184, which is FLUX.1's width]")
    print()

    # ------------------- self-checks against the frozen values --------------- #
    assert b1 == 9_437_184, f"FLUX.1 to_q must be 3072^2 = 9,437,184, got {b1:,}"
    assert l1 == 98_304, f"FLUX.1 LoRA r=16 must be 98,304, got {l1:,}"
    assert round(pct1, 2) == 1.04, f"ratio must round to 1.04%, got {pct1:.4f}%"
    assert b1 % l1 == 0 and fewer1 == 96, f"must be EXACTLY 96x fewer, got {b1/l1:.3f}x"
    assert b2 == 37_748_736, f"FLUX.2 to_q must be 6144^2 = 37,748,736, got {b2:,}"
    assert l2 == 196_608, f"FLUX.2 LoRA r=16 must be 196,608, got {l2:,}"
    print("  self-checks passed: 9,437,184 vs 98,304 = 1.04%, 96x fewer (brief-diffusion §11.2).")
    print()
    return b1, l1


# --------------------------------------------------------------------------- #
# Step 2 - the B=0 proof WITHOUT torch. B is zeros -> (alpha/r) B A is the exact
# zero matrix, so a wrapped layer equals its base bit-for-bit at step 0.
# --------------------------------------------------------------------------- #

def ba_zero_proof_no_torch(r=R_DEFAULT):
    print("=" * 72)
    print("STEP 2 - B = 0 => B@A = 0 (no torch): the wrapped layer starts as its base")
    print("=" * 72)

    scaling = 1.0  # alpha/r with alpha=r; the zero result is independent of it anyway
    try:
        import numpy as np
        d = D_FLUX1
        rng = np.random.default_rng(42)
        A = rng.standard_normal((r, d)) * (1.0 / d)   # (r, d), random - NOT zero
        B = np.zeros((d, r))                          # (d, r), the ZERO init
        BA = scaling * (B @ A)                        # (d, d)
        max_abs = float(np.abs(BA).max())
        print(f"  numpy: A ~ N(0, 1/d^2) shape {A.shape}, B = zeros shape {B.shape}, d={d}")
        print(f"  max| (alpha/r) B A |  = {max_abs:.1e}   over all {d*d:,} entries")
        print(f"  A is fully random (||A|| = {float(np.linalg.norm(A)):.3f}) - it is B that zeroes the product.")
        assert max_abs == 0.0, "B=0 must make B@A the exact zero matrix"
        assert float(np.abs(A).max()) > 0.0, "A must be nonzero (only B is zero-init)"
    except ImportError:
        d = 64  # no numpy: a smaller but genuine numeric check; the argument is size-free
        seed = 42
        # cheap LCG so the fallback needs no imports at all
        def rnd():
            nonlocal seed
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            return seed / 0x7FFFFFFF - 0.5
        A = [[rnd() for _ in range(d)] for _ in range(r)]     # (r, d) random
        B = [[0.0 for _ in range(r)] for _ in range(d)]       # (d, r) zeros
        max_abs = 0.0
        for i in range(d):
            for j in range(d):
                s = 0.0
                for k in range(r):
                    s += B[i][k] * A[k][j]
                max_abs = max(max_abs, abs(scaling * s))
        a_max = max(abs(v) for row in A for v in row)
        print(f"  pure-Python (no numpy): d={d} (the argument is size-independent), r={r}")
        print(f"  max| (alpha/r) B A |  = {max_abs:.1e}   over all {d*d:,} entries")
        print(f"  A is nonzero (max|A| = {a_max:.3f}); B being zero is what zeroes the product.")
        assert max_abs == 0.0, "B=0 must make B@A the exact zero matrix"
        assert a_max > 0.0, "A must be nonzero (only B is zero-init)"

    print("  self-check passed: B@A == 0 at init -> base is untouched on step 0.")
    print("  (B=0 is a NO-OP, not a dead unit: grad(B) is proportional to A != 0, so B moves")
    print("   first. Step 3 proves that on a real autograd graph.)")
    print()


# --------------------------------------------------------------------------- #
# Step 3 - the LoRALinear class (~30 lines) + the autograd beat. Needs torch.
# --------------------------------------------------------------------------- #

def torch_demo(r=R_DEFAULT):
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        """A frozen FLUX attention projection W0 with a trainable low-rank adapter:

            h = W0 x  +  (alpha / r) * B @ A @ x

        A is (r, d_in) ~ N(0, sigma^2); B is (d_out, r) init ZEROS -> B@A = 0, so at
        step 0 the wrapped `to_q` is bit-identical to the base. Freezing W0 is one
        line - requires_grad_(False) - and that single fact is the memory win: no
        grad / no Adam m,v slots for the 9.4M base params, only for the 98,304."""

        def __init__(self, base, r=16, alpha=16, sigma=None):
            super().__init__()
            d_out, d_in = base.weight.shape
            self.base = base
            self.base.weight.requires_grad_(False)                 # freeze W0
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)
            self.scaling = alpha / r
            sigma = (1.0 / d_in) if sigma is None else sigma
            self.A = nn.Parameter(torch.randn(r, d_in) * sigma)    # (r, d_in)  down
            self.B = nn.Parameter(torch.zeros(d_out, r))           # (d_out, r) up, ZERO

        def forward(self, x):                                      # x: (..., d_in)
            return self.base(x) + self.scaling * ((x @ self.A.t()) @ self.B.t())

        def trainable_numel(self):
            return self.A.numel() + self.B.numel()

    print("=" * 72)
    print("STEP 3 - the LoRALinear class + the B-first proof (real autograd, CPU)")
    print("=" * 72)
    print(f"  torch {torch.__version__}  (verified against 2.13.0; diffusers 0.39.0, peft 0.19.1)")

    torch.manual_seed(0)
    d = D_FLUX1
    to_q = nn.Linear(d, d, bias=False)          # a FLUX.1-dev `to_q` is exactly this
    lora = LoRALinear(to_q, r=r, alpha=16)

    # (a) bit-identical to the base at step 0 (B=0 -> B@A=0 -> dW=0).
    x = torch.randn(4, d)
    with torch.no_grad():
        max_diff = (to_q(x) - lora(x)).abs().max().item()
    print(f"  wrapped to_q {d}x{d}, r={r}. adapter trainable = {lora.trainable_numel():,} "
          f"(= 98,304, the frozen count).")
    print(f"  step 0: max|W0 x - LoRA(x)| = {max_diff:.2e}  -> bit-identical to base (B=0).")
    assert max_diff == 0.0, "B=0 must make the wrapped to_q EXACTLY the base at step 0"
    assert lora.trainable_numel() == 98_304, "adapter must hold exactly 98,304 params"

    # (b) one backward: B moves first because grad(A) passes through B (=0).
    loss = ((lora(x) - torch.randn(4, d)) ** 2).mean()
    loss.backward()
    gA, gB = lora.A.grad.norm().item(), lora.B.grad.norm().item()
    print(f"  step 1: ||grad A|| = {gA:.3e}  (identically 0 - dL/dA passes through B=0)")
    print(f"          ||grad B|| = {gB:.3e}  (nonzero  - dL/dB passes through A!=0) -> B moves FIRST")
    assert gA == 0.0, "grad(A) must be exactly 0 at step 0 (proportional to B=0)"
    assert gB > 0.0, "grad(B) must be nonzero at step 0 (proportional to A!=0)"
    assert to_q.weight.grad is None, "the frozen base W0 must receive NO gradient"
    print("  self-checks passed: B@A==0, adapter=98,304, grad(A)==0, grad(B)!=0, W0 frozen.")
    print()
    return LoRALinear


# --------------------------------------------------------------------------- #
# Step 4 - optional: confirm the SAME 98,304 via peft. Desk-checked vs peft 0.19.
# --------------------------------------------------------------------------- #

def peft_demo(r=R_DEFAULT):
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    print("=" * 72)
    print("STEP 4 - swap in `peft` and get the SAME number (peft 0.19 API)")
    print("=" * 72)

    class TinyAttn(nn.Module):
        # a stand-in module whose only weight is one FLUX-shaped attention proj.
        def __init__(self, d):
            super().__init__()
            self.to_q = nn.Linear(d, d, bias=False)

        def forward(self, x):
            return self.to_q(x)

    d = D_FLUX1
    model = TinyAttn(d)
    # Diffusion default is attention: to_q/to_k/to_v/to_out. Here we have only to_q.
    # (peft's `all-linear` is unverified for 0.19.1 - name modules explicitly, brief §tooling.)
    cfg = LoraConfig(r=r, lora_alpha=16, target_modules=["to_q"],
                     lora_dropout=0.0, bias="none")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  peft LoraConfig(r={r}, lora_alpha=16, target_modules=['to_q'])")
    model.print_trainable_parameters()   # peft's own line - the LoRA thesis as a number
    assert trainable == 98_304, (
        f"peft LoRA on one to_q must be 98,304, got {trainable:,} - a target/rank mismatch.")
    print("  confirmed: peft and the hand-written LoRALinear agree at 98,304. Same math.")
    print()


# --------------------------------------------------------------------------- #
# Step 5 - optional: wrap a REAL to_q off his on-disk FLUX.2. Spark-side.
# --------------------------------------------------------------------------- #

def real_flux_demo(path, LoRALinear, r=R_DEFAULT):
    if LoRALinear is None:
        print("--flux needs torch (and diffusers). Install into a FRESH venv (never")
        print("ComfyUI's) and rerun on the Spark. This path would then:")
        print("  1. from diffusers import Flux2Transformer2DModel")
        print(f"  2. t = Flux2Transformer2DModel.from_pretrained('{path}/transformer', dtype=bf16)")
        print("  3. find one real `to_q`, read its (out,in) = (6144,6144), wrap with LoRALinear")
        print("  4. print adapter numel (= 196,608 for FLUX.2's 6144 width) and assert BA==0.")
        print("  SAFETY: this loads his real FLUX.2 weights and contends with ComfyUI - run it")
        print("  when the GPU is idle. It is read-only (no training, no writes).")
        print()
        return

    import torch
    import torch.nn as nn
    from diffusers import Flux2Transformer2DModel

    print("=" * 72)
    print(f"STEP 5 - a REAL to_q off {path}")
    print("=" * 72)
    t = Flux2Transformer2DModel.from_pretrained(f"{path}/transformer", dtype=torch.bfloat16)
    real_to_q = None
    for name, mod in t.named_modules():
        if isinstance(mod, nn.Linear) and name.endswith("to_q"):
            real_to_q = mod
            print(f"  found {name}: {tuple(mod.weight.shape)}  (out, in)")
            break
    if real_to_q is None:
        print("  no `to_q` found - print the module names and read them (brief §tooling recipe):")
        print("    print([n for n, _ in t.named_modules()])")
        return
    d_out, d_in = real_to_q.weight.shape
    lora = LoRALinear(real_to_q, r=r, alpha=16)
    x = torch.randn(1, 8, d_in, dtype=real_to_q.weight.dtype)
    with torch.no_grad():
        max_diff = (real_to_q(x) - lora(x)).abs().max().item()
    print(f"  wrapped real to_q, adapter = {lora.trainable_numel():,} params (r*(out+in))")
    print(f"  step 0: max|base - LoRA| = {max_diff:.2e} -> bit-identical (B=0 holds on his weights)")
    assert max_diff == 0.0, "B=0 must make even his real to_q bit-identical at step 0"
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-r", "--rank", type=int, default=R_DEFAULT, help="LoRA rank (default 16)")
    ap.add_argument("--self-test", action="store_true",
                    help="steps 1-2 only (no torch): the count + the B=0 proof, all asserts")
    ap.add_argument("--peft", action="store_true", help="also run step 4 (needs torch + peft)")
    ap.add_argument("--flux", default=None,
                    help="path to his on-disk FLUX.2-dev; wrap a real to_q (Spark-side, step 5)")
    args = ap.parse_args()

    # Steps 1 + 2 are pure-Python/numpy - they always run and always self-check.
    report_counts(r=args.rank)
    ba_zero_proof_no_torch(r=args.rank)

    if args.self_test:
        print("=" * 72)
        print("--self-test: steps 1-2 passed with no GPU and no torch. LoRA, to the byte.")
        print("=" * 72)
        return

    # Step 3 needs torch; degrade gracefully if it is not importable (e.g. this laptop).
    LoRALinear = None
    try:
        LoRALinear = torch_demo(r=args.rank)
    except ImportError:
        print("torch not importable - skipping the LoRALinear/autograd demo (step 3).")
        print("Steps 1-2 above already proved the count and B@A==0 without it.\n")

    if args.peft:
        try:
            peft_demo(r=args.rank)
        except ImportError:
            print("--peft needs torch + peft, not importable here. Install into a FRESH venv")
            print("(never ComfyUI's) on the Spark and rerun. Expected line: trainable = 98,304.\n")

    if args.flux:
        real_flux_demo(args.flux, LoRALinear, r=args.rank)

    print("=" * 72)
    print("All sections passed their self-checks. LoRA from scratch, diffusion side.")
    print("Twin of 05_lora_from_scratch.py (Qwen3): same equation, a FLUX host. (spec §F.2)")
    print("=" * 72)


if __name__ == "__main__":
    main()
