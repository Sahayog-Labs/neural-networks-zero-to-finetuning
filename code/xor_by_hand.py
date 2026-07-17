#!/usr/bin/env python3
"""
xor_by_hand.py -- ReLU builds XOR with four lines of arithmetic; a linear net cannot,
at any depth or width.

Course artifact for p.10 ("A Neuron, XOR, and the Linear Collapse"). Taught here
BEFORE activations (p.11) by design -- D-19 rules "collapse first": establish that a
nonlinearity is MANDATORY, not decorative, before ReLU/GELU are presented as a menu
of choices.

Two acts, same lesson, same four points:

  1. HARD-CODE the two-ReLU-neuron weights the page derives by hand --
         h1 = ReLU(x1 + x2),    h2 = ReLU(x1 + x2 - 1),    y = h1 - 2*h2
     -- and run all four corners of the truth table. Zero training. Four lines of
     arithmetic solve a function no single straight line can separate.

  2. COLLAPSE it: swap those two ReLU neurons for two nn.Linear layers with NO
     activation, then TRAIN on the same four points. affine(affine(x)) is still just
     one affine map (a matrix and a bias, nothing more), so however long you train,
     the loss floors at ln(2) = 0.6931... -- the entropy of "always guess 50/50" --
     because XOR is not linearly separable. This is provable, not just observed: the
     collapsed loss is convex in the weights and its gradient is exactly zero at
     w1=w2=b=0, so "predict 0.5 for everything" IS the global optimum. Depth without
     a bend buys nothing but a slower way to compute the same straight line.

SAFETY: CPU only, well under 1 s (4 points, a couple thousand tiny gradient steps).
Writes and installs nothing.

Usage
-----
    python xor_by_hand.py              # the page's exact demo -- needs torch
    python xor_by_hand.py --self-test  # pure-Python reimplementation, no torch, no
                                        # GPU, no numpy -- proves the same two
                                        # assertions on any machine with python3
"""

import argparse
import math
import sys

SEED = 42
LN2 = math.log(2.0)  # entropy of "always guess 50/50" -- the provable floor, Act 2

# --------------------------------------------------------------------------------- #
# Shared ground truth -- pure Python, no tensor library required. Both the torch
# path and the --self-test path start from these same four points.
# --------------------------------------------------------------------------------- #
XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_TARGET = [0.0, 1.0, 1.0, 0.0]


def relu_xor_by_hand():
    """h1 = ReLU(x1+x2), h2 = ReLU(x1+x2-1), y = h1 - 2*h2 -- exact, no training."""
    out = []
    for x1, x2 in XOR_INPUTS:
        h1 = max(0.0, x1 + x2)
        h2 = max(0.0, x1 + x2 - 1.0)
        out.append(h1 - 2.0 * h2)
    return out


def run_act1():
    print("-" * 68)
    print("ACT 1 -- two ReLU neurons, hard-coded, zero training")
    print("-" * 68)
    print("  h1 = ReLU(x1+x2)      h2 = ReLU(x1+x2-1)      y = h1 - 2*h2")
    out = relu_xor_by_hand()
    for (x1, x2), y, t in zip(XOR_INPUTS, out, XOR_TARGET):
        mark = "OK" if abs(y - t) < 1e-9 else "MISS"
        print(f"    ({x1:.0f},{x2:.0f}) -> h1={max(0.,x1+x2):.1f}  "
              f"h2={max(0.,x1+x2-1):.1f}  y={y:.1f}   target={t:.0f}   [{mark}]")
    assert out == XOR_TARGET, f"ReLU-by-hand should be exactly {XOR_TARGET}, got {out}"
    print(f"  [OK] output == {XOR_TARGET} -- exact. No gradient step touched this.")
    print(f"       h1 and h2 measure the SAME direction (x1+x2); the only thing that")
    print(f"       tells them apart is the kink where ReLU clamps to zero. The bend")
    print(f"       did all of it.")
    return out


# --------------------------------------------------------------------------------- #
# Act 2, torch path -- verbatim the page's box, extended with self-narration and
# the assertion the spec requires.
# --------------------------------------------------------------------------------- #
def run_act2_torch(iters, lr):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    print()
    print("-" * 68)
    print("ACT 2 -- collapse it: two nn.Linear layers, NO activation, TRAINED")
    print("-" * 68)
    X = torch.tensor(XOR_INPUTS)
    y = torch.tensor(XOR_TARGET)

    net = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 1))  # affine o affine = affine
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    checkpoints = {0, iters // 20, iters // 4, iters - 1}
    loss = None
    for step in range(iters):
        loss = nn.functional.binary_cross_entropy_with_logits(net(X).squeeze(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in checkpoints:
            print(f"    step {step:>5}   loss={loss.item():.4f}")

    final_loss = loss.item()
    print(f"\n  linear-net loss floors at {final_loss:.4f}")
    print(f"  compare: ln(2) = {LN2:.4f}  ('always guess 50/50', i.e. random)")
    assert final_loss > 0.4, (
        f"linear-only net should floor near ln(2)={LN2:.4f}, well above 0; got "
        f"{final_loss:.4f} -- if this is near 0, something is wrong, because a "
        f"stack of nn.Linear layers with no activation is mathematically incapable "
        f"of separating XOR (see the numpy proof in --self-test)."
    )
    print(f"  [OK] loss floors near ln(2) after {iters} Adam steps -- it never learns")
    print(f"       XOR. affine(affine(x)) == one affine map, period.")
    return final_loss


# --------------------------------------------------------------------------------- #
# Act 2, --self-test path -- pure Python (no numpy, no torch, no GPU). Reimplements
# the same 2->8->1 linear-only network and its exact gradient by hand, so the proof
# runs on a machine that has nothing but python3 installed.
# --------------------------------------------------------------------------------- #
def _sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _matvec(mat, vec):
    return [sum(m * v for m, v in zip(row, vec)) for row in mat]


def _lcg(seed):
    """Tiny linear-congruential generator -- deterministic, no random module state
    shared with anything else, so --self-test is bit-reproducible without numpy."""
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield (state / 2**31) * 2.0 - 1.0  # uniform in [-1, 1)


def train_linear_collapse_pure_python(iters=4000, lr=0.5, hidden=8, seed=SEED):
    """Two Linear layers (2->hidden->1), NO activation, trained by hand-derived
    gradient descent on binary-cross-entropy-with-logits. Returns the loss history.

    Forward:  h = X @ W1 + b1      z = h @ W2 + b2      p = sigmoid(z)
    Backward: dz = (p - y) / N                                  (BCE-with-logits)
              dW2 = h^T . dz        db2 = sum(dz)
              dh  = dz outer W2     dW1 = X^T . dh      db1 = sum(dh, axis=0)
    """
    rng = _lcg(seed)
    W1 = [[next(rng) * 0.5 for _ in range(hidden)] for _ in range(2)]   # 2 x hidden
    b1 = [0.0] * hidden
    W2 = [next(rng) * 0.5 for _ in range(hidden)]                       # hidden x 1
    b2 = 0.0

    N = len(XOR_INPUTS)
    history = []
    for step in range(iters):
        # ---- forward ----
        hs, zs, ps = [], [], []
        for (x1, x2) in XOR_INPUTS:
            h = [x1 * W1[0][j] + x2 * W1[1][j] + b1[j] for j in range(hidden)]
            z = sum(h[j] * W2[j] for j in range(hidden)) + b2
            hs.append(h)
            zs.append(z)
            ps.append(_sigmoid(z))

        loss = -sum(
            t * math.log(max(p, 1e-12)) + (1 - t) * math.log(max(1 - p, 1e-12))
            for p, t in zip(ps, XOR_TARGET)
        ) / N
        history.append(loss)

        # ---- backward (exact, by hand) ----
        dz = [(p - t) / N for p, t in zip(ps, XOR_TARGET)]
        dW2 = [sum(hs[i][j] * dz[i] for i in range(N)) for j in range(hidden)]
        db2 = sum(dz)
        dh = [[dz[i] * W2[j] for j in range(hidden)] for i in range(N)]
        dW1 = [
            [sum(XOR_INPUTS[i][k] * dh[i][j] for i in range(N)) for j in range(hidden)]
            for k in range(2)
        ]
        db1 = [sum(dh[i][j] for i in range(N)) for j in range(hidden)]

        # ---- update (plain gradient descent -- the loss is convex, no momentum
        # needed to find the floor) ----
        for k in range(2):
            for j in range(hidden):
                W1[k][j] -= lr * dW1[k][j]
        for j in range(hidden):
            b1[j] -= lr * db1[j]
            W2[j] -= lr * dW2[j]
        b2 -= lr * db2

    return history


def run_act2_self_test(iters, lr):
    print()
    print("-" * 68)
    print("ACT 2 (--self-test, pure Python) -- same collapse, hand-derived gradient")
    print("-" * 68)
    history = train_linear_collapse_pure_python(iters=iters, lr=lr)
    checkpoints = [0, iters // 20, iters // 4, iters - 1]
    for step in sorted(set(c for c in checkpoints if 0 <= c < iters)):
        print(f"    step {step:>5}   loss={history[step]:.4f}")

    final_loss = history[-1]
    print(f"\n  linear-net loss floors at {final_loss:.4f}")
    print(f"  compare: ln(2) = {LN2:.4f}  ('always guess 50/50', i.e. random)")
    print(f"\n  Why this floor is PROVABLE, not just observed: the collapsed network")
    print(f"  is one affine map z = w.x + b, and its BCE loss is convex in (w, b).")
    print(f"  At w1=w2=b=0, sigmoid(0)=0.5 for every point, and the gradient works")
    print(f"  out to exactly zero (XOR's four points are symmetric under swapping")
    print(f"  x1<->x2). A convex function with a zero gradient at a point is AT its")
    print(f"  global minimum there -- so 'predict 0.5 for everything' provably IS")
    print(f"  the best a linear model can do on XOR. No amount of training escapes it.")

    assert abs(final_loss - LN2) < 0.02, (
        f"linear-only net should converge to ln(2)={LN2:.4f} (the provable global "
        f"minimum), got {final_loss:.4f} -- check the hand-derived gradient above."
    )
    print(f"  [OK] loss converges to ln(2) to within 0.02 -- the linear-collapse")
    print(f"       floor, confirmed by a hand-derived gradient, no library at all.")
    return final_loss


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--self-test", action="store_true",
                     help="pure-Python path: no torch, no GPU, no numpy")
    ap.add_argument("--iters", type=int, default=None,
                     help="training steps for Act 2 (default: 2000 torch / 4000 self-test)")
    ap.add_argument("--lr", type=float, default=None,
                     help="learning rate for Act 2 (default: 0.1 torch / 0.5 self-test)")
    args = ap.parse_args()

    print("=" * 68)
    print("xor_by_hand.py -- ReLU solves XOR in 4 lines; a linear net never can")
    print("=" * 68)

    relu_out = run_act1()
    assert relu_out == XOR_TARGET  # belt-and-braces: the spec's first self-check

    if args.self_test:
        iters = args.iters if args.iters is not None else 4000
        lr = args.lr if args.lr is not None else 0.5
        final_loss = run_act2_self_test(iters, lr)
    else:
        iters = args.iters if args.iters is not None else 2000
        lr = args.lr if args.lr is not None else 0.1
        try:
            import torch  # noqa: F401
        except ImportError:
            print()
            print("torch not importable. On the Spark, ComfyUI's venv has one:")
            print("  ~/ComfyUI/.venv/bin/python xor_by_hand.py")
            print("Or run the dependency-free version right here:")
            print("  python xor_by_hand.py --self-test")
            sys.exit(1)
        final_loss = run_act2_torch(iters, lr)

    print()
    print("=" * 68)
    print("WHERE THIS LEAVES YOU")
    print("=" * 68)
    print("  ReLU:   4 lines of arithmetic, exact XOR.        loss/error = 0")
    print(f"  Linear: {iters} training steps, provably stuck.   loss floors at "
          f"{final_loss:.3f} ~ ln(2)")
    print("  You now NEED a bend, and you have proof -- the collapse theorem on one")
    print("  side, XOR on the other. Page 11 asks which bend, and why ReLU won.")


if __name__ == "__main__":
    main()
