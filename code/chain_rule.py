#!/usr/bin/env python3
"""
chain_rule.py -- the chain rule, worked by hand, then verified by autograd.

Course artifact for p.05 (The Chain Rule). Two things happen here:

  1. The EIGHT-LINE one-neuron verification. Forward through z = w*x + b,
     a = sigma(z), L = (a - y)**2, then ONE .backward() call. This is the
     ONE .backward() Part I permits -- not to teach the reverse pass (that
     is page 14's mechanized job), but to prove the pencil arithmetic on
     page 5 was right: pencil -0.1872, autograd -0.1872, same number.

  2. The deep-sigmoid underflow. N identical sigmoid units are stacked
     (w=1, b=0.1, h0=1) and we read off d(output)/d(h0) -- the gradient
     that reaches layer 1 -- for N = 1..12. Every sigmoid link multiplies
     in a local slope <= 0.25 (worse, usually 0.19-0.21 away from z=0), so
     the product collapses by seven-plus orders of magnitude over 12
     layers. This is the "add across paths, multiply along a path" rule
     watched dying in real time -- the reason sigmoid hidden units were
     abandoned (page 11 names it).

SAFETY: CPU only, <1 s. Writes and installs nothing.

Usage
-----
    python chain_rule.py              # full run (needs torch; CPU is enough)
    python chain_rule.py --self-test  # pure-Python/math dry run, no torch
"""

import argparse
import math

# --------------------------------------------------------------------------- #
# Part 0 -- pure-Python reference values (no torch anywhere below this line).
# These are the SAME two computations the torch section performs; --self-test
# runs only this half, so the arithmetic can be checked on a machine with no
# GPU and no torch install at all.
# --------------------------------------------------------------------------- #

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def pencil_one_neuron():
    """p.05's worked table, by hand: x=2, w=0.5, b=0.1, target y=1."""
    x, w, b, y = 2.0, 0.5, 0.1, 1.0
    z = w * x + b
    a = sigmoid(z)
    L = (a - y) ** 2
    dLda = 2 * (a - y)          # d L / d a
    dadz = a * (1 - a)          # d a / d z  (sigmoid')
    dzdw = x                    # d z / d w
    dzdb = 1.0                  # d z / d b
    dLdw = dLda * dadz * dzdw   # chain: multiply the three local slopes
    dLdb = dLda * dadz * dzdb
    return dict(z=z, a=a, L=L, dLda=dLda, dadz=dadz, dLdw=dLdw, dLdb=dLdb)


def pencil_deep_sigmoid(n, w=1.0, b=0.1, h0=1.0):
    """N stacked sigmoid units. Returns d(h_n)/d(h0) as the literal running
    product of local slopes -- exactly what a reverse pass accumulates, just
    computed here by hand instead of by a graph."""
    h = h0
    grad = 1.0
    for _ in range(n):
        z = w * h + b
        a = sigmoid(z)
        dadz = a * (1 - a)
        grad *= w * dadz        # chain rule: multiply in this layer's local slope
        h = a
    return grad


def self_test():
    print("=" * 68)
    print("chain_rule.py --self-test  (pure Python + math, no torch, no GPU)")
    print("=" * 68)

    print("\nPART 1 -- one-neuron pencil arithmetic")
    ref = pencil_one_neuron()
    print(f"  z={ref['z']:.5f}  a={ref['a']:.5f}  L={ref['L']:.5f}")
    print(f"  dL/da={ref['dLda']:.5f}  da/dz={ref['dadz']:.5f}")
    print(f"  dL/dw = ({ref['dLda']:.5f})({ref['dadz']:.5f})(2) = {ref['dLdw']:.4f}")
    print(f"  dL/db = ({ref['dLda']:.5f})({ref['dadz']:.5f})(1) = {ref['dLdb']:.4f}")
    assert abs(ref["dLdw"] - (-0.1872)) < 1e-3, "pencil dL/dw must round to -0.1872"
    assert abs(ref["dLdb"] - (-0.0936)) < 1e-3, "pencil dL/db must round to -0.0936"
    print("  [OK] matches the page's -0.1872 / -0.0936")

    print("\nPART 2 -- deep sigmoid chain, N=1..12 (w=1, b=0.1, h0=1)")
    print(f"  {'N':>3}  {'grad reaching layer 1':>24}   0.25^N (loose upper bound)")
    grads = []
    for n in range(1, 13):
        g = pencil_deep_sigmoid(n)
        grads.append(g)
        print(f"  {n:>3}  {g: .6e}   <= {0.25 ** n:.3e}")

    assert grads[0] > 0.1, "N=1 gradient should be near the single-link ~0.187 slope"
    assert grads[-1] < 1e-6, "N=12 gradient must have collapsed below 1e-6"
    assert grads[0] / grads[-1] > 1e6, "the N=1 -> N=12 drop must span 6+ orders of magnitude"
    for a, b in zip(grads, grads[1:]):
        assert b < a, "the product of factors <1 must shrink monotonically"
    print(f"\n  [OK] N=1 grad {grads[0]:.4f}  ->  N=12 grad {grads[-1]:.3e}"
          f"   ({grads[0] / grads[-1]:.1e}x collapse)")
    print("\nAll self-test assertions passed.")


# --------------------------------------------------------------------------- #
# Part 1 (torch) -- the 8-line one-neuron verification. The ONE .backward()
# Part I permits: proving the pencil, not teaching the reverse pass.
# --------------------------------------------------------------------------- #

def run_one_neuron(torch):
    print("=" * 68)
    print("PART 1 -- the 8-line one-neuron verification (the ONE .backward()")
    print("          Part I permits -- proving the pencil, not the reverse pass)")
    print("=" * 68)

    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.1, requires_grad=True)
    x = torch.tensor(2.0)
    y = torch.tensor(1.0)

    z = w * x + b
    a = torch.sigmoid(z)
    loss = (a - y) ** 2
    loss.backward()                       # <- the one .backward() call
    print("  z = w*x + b ; a = sigmoid(z) ; L = (a-y)**2 ; L.backward()")
    print(f"  forward:  z={z.item():.5f}  a={a.item():.5f}  L={loss.item():.5f}")
    print(f"  autograd: w.grad={w.grad.item():.4f}   b.grad={b.grad.item():.4f}")

    ref = pencil_one_neuron()
    print(f"  pencil:   dL/dw={ref['dLdw']:.4f}   dL/db={ref['dLdb']:.4f}")

    assert abs(w.grad.item() - (-0.1872)) < 1e-3, "autograd w.grad must match the pencil -0.1872"
    assert abs(b.grad.item() - (-0.0936)) < 1e-3, "autograd b.grad must match the pencil -0.0936"
    assert abs(w.grad.item() - ref["dLdw"]) < 1e-6, "autograd must agree with the pure-Python pencil calc"
    print("  [OK] pencil == autograd == -0.1872 (and -0.0936 for b)")
    print()


# --------------------------------------------------------------------------- #
# Part 2 (torch) -- deep sigmoid chains, N=1..12, watched underflow live.
# --------------------------------------------------------------------------- #

def run_deep_sigmoid(torch):
    print("=" * 68)
    print("PART 2 -- deep sigmoid chains: watch the gradient underflow")
    print("=" * 68)
    print("  N stacked sigmoid units, w=1, b=0.1, h0=1. Reading off")
    print("  d(output)/d(h0) -- the gradient that reaches layer 1.")
    print()
    print(f"  {'N':>3}  {'grad (autograd)':>18}  {'grad (pencil)':>16}")

    grads = []
    for n in range(1, 13):
        h0 = torch.tensor(1.0, requires_grad=True)
        h = h0
        wv, bv = 1.0, 0.1
        for _ in range(n):
            z = h * wv + bv
            h = torch.sigmoid(z)
        h.backward()
        g_auto = h0.grad.item()
        g_pencil = pencil_deep_sigmoid(n)
        grads.append(g_auto)
        print(f"  {n:>3}  {g_auto: .6e}  {g_pencil: .6e}")
        assert abs(g_auto - g_pencil) < max(1e-9, abs(g_pencil) * 1e-4), \
            f"autograd and pencil must agree at N={n}"

    assert grads[-1] < 1e-6, "by N=12 the gradient reaching layer 1 must have collapsed below 1e-6"
    assert grads[0] / grads[-1] > 1e6, "N=1 -> N=12 must span 6+ orders of magnitude"
    print(f"\n  --> {grads[0]:.4f} at N=1 collapses to {grads[-1]:.2e} at N=12"
          f"  ({grads[0] / grads[-1]:.1e}x smaller).")
    print("      Every sigmoid link costs a factor <= 0.25 (usually worse, away from z=0).")
    print("      This is the vanishing-gradient problem, felt, not just named. ReLU's fix")
    print("      (local slope exactly 1) is page 11; the residual +1 fix is page 34.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true",
                     help="pure-Python/math dry run -- no torch, no GPU needed")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    try:
        import torch
    except ImportError:
        print("torch not importable. On the Spark, ComfyUI's venv has one:")
        print("  ~/ComfyUI/.venv/bin/python chain_rule.py")
        print("(CPU is enough -- this script never touches the GPU)")
        print()
        print("To check the arithmetic without torch at all, run:")
        print("  python chain_rule.py --self-test")
        raise SystemExit(1)

    print(f"chain_rule.py -- torch {torch.__version__}, CPU")
    print()
    run_one_neuron(torch)
    run_deep_sigmoid(torch)


if __name__ == "__main__":
    main()
