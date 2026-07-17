#!/usr/bin/env python3
"""
19_init_variance.py -- per-layer activation std under four init schemes: zeros,
too-small, too-large, He. One forward pass, no training, no gradients at all --
the signal either survives depth or it doesn't, and that is decided entirely by
one number: the variance of the weight you started with.

Course artifact for p.19 ("Initialization, Vanishing/Exploding Gradients, and Why
Residuals Exist"). The page's frozen number (constants.md section 9.3):

    He init at d = 4096:  Var(w) = 2 / 4096,  std = sqrt(2/4096) = 0.0221
                           (matches GPT-2's hand-picked 0.02, not a coincidence)

The mechanism, made concrete instead of asserted: stack DEPTH layers of a fixed
WIDTH=4096, no bias, ReLU between them. Each layer computes

    z = x @ W          W ~ N(0, std_w^2), fan_in = WIDTH
    a = ReLU(z)

For a zero-mean input with variance V, Var(z) = fan_in * std_w^2 * V, and because
ReLU zeroes (in expectation) half of a zero-mean Gaussian's mass, Var(a) is about
Var(z)/2. So each layer multiplies the running variance by the factor

    m = fan_in * std_w^2 / 2

m < 1  -> vanishing (too-small: geometric decay to the noise floor, invisible by
          layer ~4)
m > 1  -> exploding (too-large: geometric growth, many orders of magnitude by
          layer ~16)
m == 1 -> He's whole point. Solve fan_in*std_w^2/2 = 1 for std_w and you get
          std_w = sqrt(2/fan_in) -- variance is CONSERVED, layer after layer,
          however deep the stack goes. That is the 0.0221 above, and it is why
          He (not "just make it small") is the fix for ReLU nets.
zeros  -> the degenerate m: W=0 forces z=0 forces a=0 forever. Not a special
          case of "too small" -- it is IDENTICAL activations (all exactly 0),
          the symmetry-never-breaks trap the page's misconceptions box names.

Two run modes, same arithmetic, two different libraries:

    python 19_init_variance.py              # torch path -- needs torch, checks
                                             # the manual std against
                                             # torch.nn.init.kaiming_normal_
    python 19_init_variance.py --self-test  # pure-numpy path -- no torch, no
                                             # GPU -- proves the same four
                                             # assertions on any machine with
                                             # numpy installed

SAFETY: CPU only. Peak allocation is DEPTH x (WIDTH x WIDTH) float32 weight
matrices, default 16 x 4096 x 4096 x 4 bytes = ~1.05 GB, freed layer by layer
(only one W lives at a time). Runtime a few seconds either path. Writes and
installs nothing.
"""

import argparse
import math
import sys

# --------------------------------------------------------------------------- #
# Frozen numbers -- constants.md section 9.3.
# --------------------------------------------------------------------------- #
WIDTH = 4096                       # fan_in = fan_out = WIDTH throughout the stack
DEPTH = 16                         # deep enough that vanish/explode are unmissable
BATCH = 64
SEED = 42

HE_STD_D4096 = math.sqrt(2.0 / WIDTH)   # the frozen 0.0221 (constants.md sec 9.3)
TOO_SMALL_STD = 1.0e-3                  # ~22x smaller than He std -> visibly vanishes
TOO_LARGE_STD = 0.5                     # ~22x larger than He std  -> visibly explodes

SCHEMES = ["zeros", "too_small", "too_large", "he"]


def scheme_std(name: str) -> float:
    """The std of W used to build each scheme's weight matrix (0.0 for zeros)."""
    return {
        "zeros": 0.0,
        "too_small": TOO_SMALL_STD,
        "too_large": TOO_LARGE_STD,
        "he": HE_STD_D4096,
    }[name]


# --------------------------------------------------------------------------- #
# Pure-numpy forward pass -- used by BOTH modes' arithmetic (the torch path only
# adds the library-vs-hand-formula cross-check on top of this).
# --------------------------------------------------------------------------- #
def forward_pass_numpy(std_w: float, width=WIDTH, depth=DEPTH, batch=BATCH, seed=SEED):
    """Stack `depth` no-bias linear-then-ReLU layers, width x width, weight std
    `std_w`. Returns the list of per-layer post-ReLU activation std, length depth+1
    (index 0 is the input, matching the page's 'layer 0 = input' convention)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch, width)).astype(np.float64)   # input std = 1.0
    stds = [float(x.std())]
    for layer in range(depth):
        if std_w == 0.0:
            W = np.zeros((width, width), dtype=np.float64)
        else:
            W = rng.normal(0.0, std_w, size=(width, width)).astype(np.float64)
        z = x @ W
        a = np.maximum(z, 0.0)
        s = float(a.std())
        stds.append(s)
        x = a
        if not math.isfinite(s):
            # Overflow: pad the rest of the table with inf so callers see the
            # full-length list without re-computing on a blown-up tensor.
            stds.extend([float("inf")] * (depth - layer - 1))
            break
    return stds


def narrate_table(results: dict, depth: int) -> None:
    header = f"  {'layer':>5}  " + "  ".join(f"{s:>14}" for s in SCHEMES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for layer in range(depth + 1):
        row = f"  {layer:>5}  "
        cells = []
        for s in SCHEMES:
            v = results[s][layer]
            cells.append(f"{v:>14.3e}" if math.isfinite(v) else f"{'inf/overflow':>14}")
        print(row + "  ".join(cells))


# --------------------------------------------------------------------------- #
# --self-test path: numpy only, no torch, no GPU.
# --------------------------------------------------------------------------- #
def self_test() -> None:
    import numpy as np

    print("=" * 72)
    print("19_init_variance.py -- self-test (pure numpy, no torch, no GPU)")
    print(f"numpy {np.__version__}   seed {SEED}   width {WIDTH}   depth {DEPTH}")
    print("=" * 72)

    print(f"\nHe std at d={WIDTH}: sqrt(2/{WIDTH}) = {HE_STD_D4096:.6f}")
    assert abs(HE_STD_D4096 - 0.0221) < 1e-4, (
        f"He std at d=4096 should be 0.0221 (constants.md sec 9.3), got {HE_STD_D4096:.6f}"
    )
    print("[OK] matches constants.md section 9.3's frozen 0.0221 (~GPT-2's hand-picked 0.02).")

    results = {s: forward_pass_numpy(scheme_std(s)) for s in SCHEMES}
    print("\nper-layer activation std (layer 0 = input, std = 1.0 by construction):")
    narrate_table(results, DEPTH)

    # --- zeros: collapses to IDENTICAL (not merely small) activations, immediately
    zeros_acts = results["zeros"]
    assert all(v == 0.0 for v in zeros_acts[1:]), (
        f"zeros-init must collapse every post-layer std to exactly 0.0, got {zeros_acts[1:]}"
    )
    print("\n[OK] zeros-init: std == 0.0 exactly from layer 1 onward -- collapsed, not shrunk.")
    print("     (Weights are IDENTICAL across a layer, not necessarily zero-gradient --")
    print("     the misconception the page's box warns about; this script shows the")
    print("     activation side of that collapse, the gradient side is 19_grad_flow.py.)")

    # --- too-small: geometric vanishing, unmistakable well before layer DEPTH
    small_acts = results["too_small"]
    assert small_acts[1] > 0.0, "too-small init should still produce a nonzero first layer"
    assert small_acts[-1] < small_acts[1] * 1e-6, (
        f"too-small init should vanish by many orders of magnitude over {DEPTH} layers: "
        f"layer1={small_acts[1]:.3e}, layer{DEPTH}={small_acts[-1]:.3e}"
    )
    print(f"\n[OK] too-small (std={TOO_SMALL_STD}): layer-1 std {small_acts[1]:.3e} -> "
          f"layer-{DEPTH} std {small_acts[-1]:.3e} -- vanished by >1e6x. This is the same")
    print("     geometric-product mechanism as vanishing GRADIENTS (19_grad_flow.py),")
    print("     just run forward on activations instead of backward on gradients.")

    # --- too-large: geometric explosion, unmistakable well before layer DEPTH
    large_acts = results["too_large"]
    assert math.isfinite(large_acts[-1]) or large_acts[-1] == float("inf"), "sanity"
    finite_growth = math.isfinite(large_acts[-1]) and large_acts[-1] > large_acts[1] * 1e6
    overflowed = large_acts[-1] == float("inf")
    assert finite_growth or overflowed, (
        f"too-large init should explode by many orders of magnitude (or overflow) over "
        f"{DEPTH} layers: layer1={large_acts[1]:.3e}, layer{DEPTH}={large_acts[-1]}"
    )
    tail = "overflowed to inf" if overflowed else f"{large_acts[-1]:.3e}"
    print(f"\n[OK] too-large (std={TOO_LARGE_STD}): layer-1 std {large_acts[1]:.3e} -> "
          f"layer-{DEPTH} std {tail} -- exploded by >1e6x.")

    # --- He: variance CONSERVED -- std stays in a tight band across all DEPTH layers
    he_acts = results["he"]
    ratio = he_acts[-1] / he_acts[1]
    assert 0.5 < ratio < 2.0, (
        f"He init should hold activation std roughly flat across depth: "
        f"layer1={he_acts[1]:.4f}, layer{DEPTH}={he_acts[-1]:.4f}, ratio={ratio:.3f}"
    )
    print(f"\n[OK] He (std={HE_STD_D4096:.4f}): layer-1 std {he_acts[1]:.4f} -> "
          f"layer-{DEPTH} std {he_acts[-1]:.4f} (ratio {ratio:.2f}x) -- flat.")
    print("     fan_in * std_w^2 / 2 = "
          f"{WIDTH * HE_STD_D4096**2 / 2:.4f} (should be 1.0: variance neither grows nor")
    print("     shrinks per layer -- the entire content of 'He init' is this one equation.")

    assert abs(WIDTH * HE_STD_D4096**2 / 2.0 - 1.0) < 1e-9, (
        "He's defining identity fan_in*std_w^2/2 == 1 must hold exactly by construction"
    )

    print("\n" + "=" * 72)
    print("All self-test assertions PASS (numpy-only, no torch, no GPU required).")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Default path: torch, plus the library cross-check the page promises
# (torch.nn.init.kaiming_normal_).
# --------------------------------------------------------------------------- #
def run_torch() -> None:
    import torch

    torch.manual_seed(SEED)
    print("=" * 72)
    print("19_init_variance.py -- per-layer activation std, four init schemes")
    print(f"torch {torch.__version__}   seed {SEED}   width {WIDTH}   depth {DEPTH}")
    print("=" * 72)

    print(f"\nHe std at d={WIDTH}: sqrt(2/{WIDTH}) = {HE_STD_D4096:.6f}")
    assert abs(HE_STD_D4096 - 0.0221) < 1e-4, (
        f"He std at d=4096 should be 0.0221 (constants.md sec 9.3), got {HE_STD_D4096:.6f}"
    )
    print("[OK] matches constants.md section 9.3's frozen 0.0221.")

    # Cross-check against the library, not just the hand formula: build a real
    # nn.Linear, kaiming_normal_ its weight (mode='fan_in', ReLU gain), and
    # compare the EMPIRICAL std of the initialized weight to sqrt(2/fan_in).
    layer = torch.nn.Linear(WIDTH, WIDTH, bias=False)
    torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
    kaiming_std = layer.weight.detach().std().item()
    print(f"torch.nn.init.kaiming_normal_ empirical std: {kaiming_std:.6f} "
          f"(hand formula: {HE_STD_D4096:.6f})")
    assert abs(kaiming_std - HE_STD_D4096) < 5e-4, (
        f"kaiming_normal_'s empirical std should match sqrt(2/fan_in)={HE_STD_D4096:.6f}, "
        f"got {kaiming_std:.6f} -- library and hand formula must agree"
    )
    print("[OK] the hand formula IS what torch.nn.init.kaiming_normal_ computes.")

    def forward_pass_torch(std_w: float):
        x = torch.randn(BATCH, WIDTH, dtype=torch.float64)   # input std = 1.0
        stds = [x.std().item()]
        for layer_idx in range(DEPTH):
            if std_w == 0.0:
                W = torch.zeros(WIDTH, WIDTH, dtype=torch.float64)
            else:
                W = torch.randn(WIDTH, WIDTH, dtype=torch.float64) * std_w
            z = x @ W
            a = torch.relu(z)
            s = a.std().item()
            stds.append(s)
            x = a
            if not math.isfinite(s):
                stds.extend([float("inf")] * (DEPTH - layer_idx - 1))
                break
        return stds

    results = {s: forward_pass_torch(scheme_std(s)) for s in SCHEMES}
    print("\nper-layer activation std (layer 0 = input, std = 1.0 by construction):")
    narrate_table(results, DEPTH)

    zeros_acts = results["zeros"]
    assert all(v == 0.0 for v in zeros_acts[1:]), "zeros-init must collapse to exactly 0.0"
    print("\n[OK] zeros-init collapses every activation to exactly 0.0 -- identical, not")
    print("     merely small. Biases CAN start at zero (the random W already broke the")
    print("     tie); weights cannot, or every unit in a layer computes the same thing.")

    small_acts = results["too_small"]
    assert small_acts[-1] < small_acts[1] * 1e-6, "too-small init should vanish by >1e6x"
    print(f"\n[OK] too-small (std={TOO_SMALL_STD}): vanished from {small_acts[1]:.3e} to "
          f"{small_acts[-1]:.3e} over {DEPTH} layers.")

    large_acts = results["too_large"]
    overflowed = large_acts[-1] == float("inf")
    finite_growth = math.isfinite(large_acts[-1]) and large_acts[-1] > large_acts[1] * 1e6
    assert overflowed or finite_growth, "too-large init should explode by >1e6x (or overflow)"
    tail = "overflowed to inf" if overflowed else f"{large_acts[-1]:.3e}"
    print(f"\n[OK] too-large (std={TOO_LARGE_STD}): exploded from {large_acts[1]:.3e} to "
          f"{tail} over {DEPTH} layers.")

    he_acts = results["he"]
    ratio = he_acts[-1] / he_acts[1]
    assert 0.5 < ratio < 2.0, f"He init should hold std roughly flat, ratio={ratio:.3f}"
    print(f"\n[OK] He (std={HE_STD_D4096:.4f}): held flat, {he_acts[1]:.4f} -> "
          f"{he_acts[-1]:.4f} (ratio {ratio:.2f}x) across all {DEPTH} layers.")

    print("\n" + "=" * 72)
    print("Same signal, four variances. Zero collapses it, too-small starves it,")
    print("too-large drowns it, He (std = sqrt(2/fan_in)) is the ONE value that lets")
    print("a network go arbitrarily deep without either failure -- and it is the")
    print("first of three tools (with residuals and normalization, pages 19-20)")
    print("that make deep nets trainable at all.")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-layer activation std under zeros / too-small / too-large / "
                     "He init on a deep no-bias ReLU stack -- p.19 companion script."
    )
    ap.add_argument("--self-test", action="store_true",
                     help="pure-numpy path: no torch, no GPU required")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not importable. On the Spark, ComfyUI's venv has one:")
        print("  ~/ComfyUI/.venv/bin/python 19_init_variance.py")
        print("Or run the dependency-free version right here:")
        print("  python 19_init_variance.py --self-test")
        sys.exit(1)

    run_torch()


if __name__ == "__main__":
    main()
