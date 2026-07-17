#!/usr/bin/env python3
"""
receptive_field.py -- the receptive-field recurrence, r_l = r_(l-1) + (k_l-1)*jump,
run against the exact three presets on page 26's live "Receptive-Field Grower" demo.

Course artifact for p.26 ("Convolutions: What Your U-Net Is Made Of"). The page's canvas
demo drives this same recurrence in JavaScript to grow a glowing square over an editable
conv stack; this script is the from-scratch reference, so a learner can read one function
and know exactly what that square is computing -- no browser required.

    RF_0 = 1,    RF_l = RF_(l-1) + (k_l - 1) * prod_(j<l) s_j     (notation.md sec 4.2's canonical
                                                                     symbol is RF_l, "not r" --
                                                                     page 26's own math writes the
                                                                     same recurrence as r_l; this
                                                                     script uses RF_l throughout)

Three presets, transcribed verbatim from the page's PRESETS object
(26-convolutions-the-unet-atom.html, the "Try it" box's own demo):

  plain -- five 3x3, stride-1 convs                    the "~112 layers to reach 224px" beat
            (RF grows LINEARLY: 3,5,7,9,11 -- quiz's frozen answer for 5 layers)
  unet  -- twelve 3x3 convs, stride alternating 1,2,... the page's own worked numbers:
            RF = 13px after 4 layers, RF = 29px after 6 layers ("Flip one layer from s=1 to
            s=2 and the RF square jumps (13 -> 29 in the encoder preset)")
  ones  -- ten 1x1, stride-1 convs                      RF PINNED at 1px through all 10 layers
            -- a 1x1 conv mixes CHANNELS, never SPACE; this is exactly what proj_in/proj_out
            do around a U-Net's attention block, per the page's own "Try it" text.

Also computes the EFFECTIVE receptive field by path-counting (the page's deep-dive: "the
effective receptive field is smaller than the theoretical one, and Gaussian"). Starting one
unit of influence at the traced output pixel, walk backward through the stack: upsample by
this layer's stride (insert zeros), then box-convolve by this layer's kernel. The support of
the resulting distribution is exactly the theoretical RF (a second, independent derivation of
r_l, checked against the first); the >=5%-of-peak core (same threshold the page's canvas uses
to draw the "faint glow") is the effective RF that survives averaging along real paths.

Usage
-----
    python receptive_field.py                # arithmetic + a real nn.Conv2d cross-check
                                               # (needs torch; skipped gracefully if absent)
    python receptive_field.py --self-test     # pure-Python arithmetic only -- no torch, no
                                               # GPU, no numpy -- runs on any python3
    python receptive_field.py --preset unet   # print/check just one preset

SAFETY: CPU only, well under 1 s even with the torch cross-check (tiny tensors, 2 channels).
No GPU, no network, no files written or installed.
"""

import argparse
import sys

# --------------------------------------------------------------------------------- #
# The three presets, transcribed verbatim from page 26's JS `PRESETS` object.
# --------------------------------------------------------------------------------- #
PRESETS = {
    "plain": [(3, 1)] * 5,
    "unet": [(3, s) for s in [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]],
    "ones": [(1, 1)] * 10,
}


def receptive_field_growth(layers):
    """RF_0 = 1; for each layer, RF += (k-1)*jump, THEN jump *= s. (notation.md sec 4.2 names
    this RF_l; page 26's own math writes the identical recurrence as r_l -- same quantity.)

    Mirrors page 26's JS `rfStats()` exactly, including the order: the recurrence widens
    using the jump accumulated BEFORE this layer's own stride is folded in -- a layer's
    kernel sees the input at the spacing its PREDECESSORS created, not the spacing it itself
    creates. Returns [RF_0, RF_1, ..., RF_L] (length len(layers)+1).
    """
    r = 1
    jump = 1
    history = [r]
    for k, s in layers:
        r = r + (k - 1) * jump
        jump = jump * s
        history.append(r)
    return history


def path_count_distribution(layers):
    """Backward path count: start [1] at the traced output pixel; for each layer, LAST to
    FIRST, upsample by that layer's stride (insert s-1 zeros between samples) then
    box-convolve by that layer's kernel. Mirrors page 26's JS `pathDist()` exactly. The
    cumulative stride effect compounds naturally through the repeated upsampling, so this is
    an independent derivation of the same theoretical RF: len(result) == RF from the
    recurrence above (asserted below), even though neither function calls the other.
    """
    dist = [1]
    for k, s in reversed(layers):
        up = []
        for i, v in enumerate(dist):
            up.append(v)
            if i < len(dist) - 1:
                up.extend([0] * (s - 1))
        conv = [0] * (len(up) + k - 1)
        for i, v in enumerate(up):
            for j in range(k):
                conv[i + j] += v
        dist = conv
    return dist


def effective_rf(dist, threshold=0.05):
    """The page's own >=5%-of-peak threshold for the "faint glow" core (page 26 JS `render()`:
    `marg.findIndex(v => v >= 0.05)` from each end). Returns the width of that core in px."""
    peak = max(dist)
    above = [i for i, v in enumerate(dist) if v / peak >= threshold]
    if not above:
        return 1
    return above[-1] - above[0] + 1


def print_growth_table(name, layers):
    history = receptive_field_growth(layers)
    dist = path_count_distribution(layers)
    eff = effective_rf(dist)

    assert len(dist) == history[-1], (
        f"{name}: path-count distribution length ({len(dist)}) should equal the theoretical "
        f"RF from the recurrence ({history[-1]}) -- two independent derivations of the same "
        f"number must agree."
    )

    print(f"\n  preset '{name}': {len(layers)} layers  {layers}")
    print(f"    {'layer':>6} {'k':>3} {'s':>3} {'jump (before)':>14} {'RF_l':>6}")
    jump = 1
    for i, (k, s) in enumerate(layers, start=1):
        print(f"    {i:>6} {k:>3} {s:>3} {jump:>14} {history[i]:>6}")
        jump *= s
    print(f"    theoretical RF after all {len(layers)} layers:  {history[-1]}px")
    print(f"    effective RF (>=5% of peak path-weight):   {eff}px  "
          f"(path-count support = {len(dist)}px, matches theoretical RF)")
    return history, eff


# --------------------------------------------------------------------------------- #
# Optional bonus: cross-check the pure arithmetic against a REAL nn.Conv2d stack --
# feed a one-hot impulse gradient at a deep-interior output pixel and measure exactly
# which input pixels get nonzero gradient. Needs torch; skipped gracefully if absent.
# The --self-test path never calls this (it proves the recurrence needs no library at all).
# --------------------------------------------------------------------------------- #
def run_torch_crosscheck():
    import torch
    import torch.nn as nn

    print("\n" + "=" * 68)
    print("TORCH CROSS-CHECK -- a real nn.Conv2d stack, not just arithmetic")
    print("=" * 68)
    torch.manual_seed(42)

    # Small cases only (RF stays modest) so the input tensor stays tiny and fast, but with
    # generous margin (input = 4*RF) so the traced output pixel's true support cannot be
    # clipped by the 'same' zero-padding at the border -- otherwise this check would be
    # measuring padding artifacts, not the receptive field.
    cases = [("plain", PRESETS["plain"]), ("unet[:6]", PRESETS["unet"][:6])]
    for name, layers in cases:
        rf = receptive_field_growth(layers)[-1]
        input_side = 4 * rf + 5

        net = nn.Sequential(*[
            nn.Conv2d(2, 2, kernel_size=k, stride=s, padding=k // 2)
            for k, s in layers
        ])
        x = torch.randn(1, 2, input_side, input_side, requires_grad=True)
        y = net(x)
        oh, ow = y.shape[-2], y.shape[-1]
        cy, cx = oh // 2, ow // 2

        y[0, 0, cy, cx].backward()
        grad = x.grad[0, 0]
        nz = (grad.abs() > 0).nonzero()
        support_h = (nz[:, 0].max() - nz[:, 0].min() + 1).item()
        support_w = (nz[:, 1].max() - nz[:, 1].min() + 1).item()

        print(f"  {name}: theoretical RF = {rf}px  |  real Conv2d gradient support = "
              f"{support_h}x{support_w}px  (input {input_side}px, traced pixel at "
              f"output ({cy},{cx}) of {oh}x{ow})")
        assert support_h == rf and support_w == rf, (
            f"{name}: real Conv2d gradient support ({support_h}x{support_w}) should equal "
            f"theoretical RF ({rf}x{rf}) -- if it doesn't, either the traced pixel is too "
            f"close to the padded border (increase margin) or the recurrence is wrong."
        )
    print("  [OK] the recurrence's theoretical RF exactly matches the nonzero-gradient")
    print("       support of a real nn.Conv2d stack -- the same math either way.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true",
                     help="pure-Python arithmetic only -- skip the torch cross-check")
    ap.add_argument("--preset", choices=["plain", "unet", "ones", "all"], default="all",
                     help="which preset(s) to print/check (default: all three)")
    args = ap.parse_args()

    print("=" * 68)
    print("receptive_field.py -- the RF recurrence, RF_l = RF_(l-1) + (k_l-1)*jump")
    print("=" * 68)
    print("  RF_0 = 1,   RF_l = RF_(l-1) + (k_l - 1) * prod_(j<l) s_j   (notation.md sec 4.2;")
    print("  page 26's own math renders the identical recurrence as r_l -- same quantity)")

    names = list(PRESETS) if args.preset == "all" else [args.preset]
    results = {name: print_growth_table(name, PRESETS[name]) for name in names}

    print("\n" + "=" * 68)
    print("SELF-CHECKS -- against the page's own frozen hand values (notation.md sec 4.2, RF_l)")
    print("=" * 68)

    if "plain" in results:
        h, _ = results["plain"]
        assert h[-1] == 11, f"plain preset: RF after 5 layers should be 11, got {h[-1]}"
        print(f"  [OK] plain (5x 3x3 stride-1): RF = {h[-1]}px == 11 (3,5,7,9,11 -- linear")
        print(f"       growth; this is why plain stacks need ~112 layers to reach 224px)")

    if "unet" in results:
        h, _ = results["unet"]
        assert h[4] == 13, f"unet preset: RF after 4 layers should be 13, got {h[4]}"
        assert h[6] == 29, f"unet preset: RF after 6 layers should be 29, got {h[6]}"
        cum_stride_6 = 1
        for k, s in PRESETS["unet"][:6]:
            cum_stride_6 *= s
        assert cum_stride_6 == 8, f"cumulative stride after 6 unet layers should be 8, got {cum_stride_6}"
        print(f"  [OK] unet (3x3, stride 1,2,1,2,...): RF after 4 layers = {h[4]}px == 13,")
        print(f"       after 6 layers = {h[6]}px == 29 -- exactly the page's own \"13 -> 29\"")
        print(f"       example. Cumulative stride after 6 layers = {cum_stride_6}x == 8 (2^3).")

    if "ones" in results:
        h, eff = results["ones"]
        assert all(v == 1 for v in h), f"ones preset: RF should stay pinned at 1 throughout, got {h}"
        assert eff == 1, f"ones preset: effective RF should also be 1, got {eff}"
        print(f"  [OK] ones (10x 1x1 stride-1): RF stays pinned at {h[-1]}px through all "
              f"{len(PRESETS['ones'])} layers")
        print(f"       -- a 1x1 conv mixes CHANNELS, never SPACE (exactly what proj_in/proj_out")
        print(f"       do around a U-Net's attention block).")

    for name, (h, eff) in results.items():
        assert eff <= h[-1], f"{name}: effective RF ({eff}) should never exceed theoretical RF ({h[-1]})"
    print(f"\n  [OK] effective RF (path-count core) <= theoretical RF for every preset checked")
    print(f"       -- \"the glow is smaller than the box\" (deep-dive: Gaussian falloff).")

    if args.self_test:
        print("\n(--self-test: pure-Python arithmetic only, torch cross-check skipped)")
        return

    try:
        run_torch_crosscheck()
    except ImportError:
        print()
        print("torch not importable -- the arithmetic self-checks above already prove the")
        print("recurrence on their own (no library required for the theory).")
        print("On the Spark:  ~/ComfyUI/.venv/bin/python receptive_field.py")
        print("Or explicitly skip the torch step anywhere:  python receptive_field.py --self-test")
        sys.exit(0)

    print("\n" + "=" * 68)
    print("WHERE THIS LEAVES YOU")
    print("=" * 68)
    print("  One recurrence, three regimes: plain stacks grow the RF LINEARLY (slow -- ~112")
    print("  layers to cover 224px), strided/pooled stacks grow it EXPONENTIALLY (the U-Net's")
    print("  actual trick), and 1x1 convs never grow it at all (channels, not space). The")
    print("  glowing square on page 26 is this exact function, redrawn every time you edit")
    print("  a layer.")


if __name__ == "__main__":
    main()
