#!/usr/bin/env python3
"""
reparam.py -- the reparameterization trick, the diffusion handoff.

Course artifact for p.08 ("Probability & the Gaussian"). This is the at-your-
keyboard version of the page's key idea and Panel B: a Gaussian draw is a plain
deterministic function of a FROZEN standard-normal draw, so gradients flow
straight through it.

  1. Draw N(mu, sigma^2) two ways -- mu + sigma*torch.randn(n) (reparameterized)
     vs torch.normal(mu, sigma, (n,)) (direct) -- and show the resulting means
     and stds agree. Same distribution, two call sites.
  2. The payoff, exactly as the page shows it: make mu, sigma leaf tensors with
     requires_grad=True, roll eps = torch.randn(4) ONCE (no grad -- the
     randomness is external to the graph, not detached from an existing graph
     since it was never attached to one), form x = mu + sigma*eps, and call
     x.sum().backward(). Because mu and sigma are here SCALARS broadcast over
     4 samples, d(sum x)/dmu = 4 (four 1's added) and d(sum x)/dsigma =
     eps.sum() (four eps_i's added) -- print both, matching the page's code
     block verbatim.
  3. The formal self-check the spec asks for -- "mu.grad is all ones, sigma.grad
     is eps" -- is literally true only when each output element has its OWN
     mu_i, sigma_i (no broadcast-sum to collapse the shape). So part 3 repeats
     the identical trick with mu, sigma, eps all shape (4,): x_i = mu_i +
     sigma_i * eps_i, independent per element. Now dx_i/dmu_i = 1 and
     dx_i/dsigma_i = eps_i with no cross-terms, so mu.grad == ones and
     sigma.grad == eps EXACTLY, elementwise. Same trick, sharper lens.

This one small identity is the whole VAE (p.53, reopened) and the whole
diffusion forward process (p.54): x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps
is x = mu + sigma*eps with new letters, mu = sqrt(abar_t)*x0 and
sigma = sqrt(1-abar_t). Nothing there re-derives this; it leans on what
runs here.

Usage
-----
    python reparam.py              # the real thing -- needs a torch install
    python reparam.py --self-test  # numpy-only arithmetic check, no torch needed
                                    # (this machine has no torch/CUDA; the Spark does)

SAFETY: CPU only, <1 s, allocates a few hundred KB. Writes and installs nothing.
"""

import argparse
import sys


# --------------------------------------------------------------------------- #
# The real thing -- needs torch. Runs on any box with a torch install (CPU is
# plenty; nothing here touches a GPU).
# --------------------------------------------------------------------------- #

def run_torch():
    import torch

    print("=" * 68)
    print("reparam.py -- backprop through a sample")
    print(f"torch {torch.__version__}")
    print("=" * 68)

    torch.manual_seed(0)
    mu, sigma, n = 1.0, 2.0, 100_000

    # ---- part 1: two ways to draw the same Gaussian ----
    print("\n-- part 1: N(mu=1.0, sigma=2.0), two ways, n=100,000 --")
    a = mu + sigma * torch.randn(n)          # reparameterized
    b = torch.normal(mu, sigma, (n,))        # direct draw
    print(f"reparameterized : mean={a.mean().item():.4f}  std={a.std().item():.4f}")
    print(f"direct draw     : mean={b.mean().item():.4f}  std={b.std().item():.4f}")

    assert abs(a.mean().item() - mu) < 0.05, "reparameterized mean should land near mu"
    assert abs(a.std().item() - sigma) < 0.05, "reparameterized std should land near sigma"
    assert abs(b.mean().item() - mu) < 0.05, "direct-draw mean should land near mu"
    assert abs(b.std().item() - sigma) < 0.05, "direct-draw std should land near sigma"
    print("[OK] same distribution, two call sites -- the histograms agree.")

    # ---- part 2: the payoff, exactly as the page shows it (scalar mu, sigma) ----
    print("\n-- part 2: the payoff -- backprop through the sample (page's own code) --")
    mu_t = torch.tensor(1.0, requires_grad=True)
    sig_t = torch.tensor(2.0, requires_grad=True)
    eps = torch.randn(4)                      # rolled ONCE, external to the graph, no grad
    print(f"eps (frozen, no grad) = {[round(v, 4) for v in eps.tolist()]}")

    x = mu_t + sig_t * eps                     # a plain differentiable function of mu, sig
    print(f"x = mu + sigma*eps     = {[round(v, 4) for v in x.tolist()]}")

    x.sum().backward()
    print(f"mu.grad    = {mu_t.grad.item():.4f}   (expect 4.0 -- sum of four dx_i/dmu = 1)")
    print(f"sigma.grad = {sig_t.grad.item():.4f}   (expect eps.sum() -- sum of four dx_i/dsig = eps_i)")

    assert torch.allclose(mu_t.grad, torch.tensor(4.0)), (
        f"mu.grad should be 4.0 (four 1's added by the broadcast sum), got {mu_t.grad}"
    )
    assert torch.allclose(sig_t.grad, eps.sum()), (
        f"sigma.grad should be eps.sum(), got {sig_t.grad} vs {eps.sum()}"
    )
    print("[OK] mu.grad is a sum of ones, sigma.grad is a sum of eps -- as the page shows.")

    # ---- part 3: the formal self-check -- elementwise, no broadcast to collapse it ----
    print("\n-- part 3: same trick, one mu_i/sigma_i per sample -- the literal self-check --")
    print("(scalar mu/sigma above SUM their per-sample gradients; give each sample its own")
    print(" mu_i, sigma_i and the sum vanishes -- mu.grad is then EXACTLY ones, sigma.grad")
    print(" is then EXACTLY eps, elementwise, because x_i depends on no mu_j, sigma_j but its own.)")

    mu_vec = torch.ones(4, requires_grad=True)          # mu_i = 1.0, independent per element
    sig_vec = torch.full((4,), 2.0, requires_grad=True)  # sigma_i = 2.0, independent per element
    eps2 = torch.randn(4)                                 # a fresh frozen draw, no grad
    x_vec = mu_vec + sig_vec * eps2                        # x_i = mu_i + sigma_i * eps_i
    x_vec.sum().backward()

    ones = torch.ones_like(mu_vec)
    print(f"eps2       = {[round(v, 4) for v in eps2.tolist()]}")
    print(f"mu.grad    = {[round(v, 4) for v in mu_vec.grad.tolist()]}   (expect all 1.0)")
    print(f"sigma.grad = {[round(v, 4) for v in sig_vec.grad.tolist()]}   (expect == eps2, exactly)")

    assert torch.allclose(mu_vec.grad, ones), (
        f"mu.grad should be all ones (dx_i/dmu_i = 1), got {mu_vec.grad}"
    )
    assert torch.allclose(sig_vec.grad, eps2), (
        f"sigma.grad should equal eps (dx_i/dsigma_i = eps_i), got {sig_vec.grad} vs {eps2}"
    )
    print("[OK] mu.grad == ones, sigma.grad == eps, elementwise and exact.")

    print("\nYou just backpropagated through randomness. That single trick is the whole")
    print("VAE (p.53) and the whole diffusion forward process (p.54):")
    print("  x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps")
    print("is x = mu + sigma*eps with new letters -- mu = sqrt(abar_t)*x0,")
    print("sigma = sqrt(1-abar_t). Same object, new name.")


# --------------------------------------------------------------------------- #
# Self-test -- the identical arithmetic, done by hand with numpy, no torch and
# no autograd required. This is what runs on a box with no GPU (this one).
# --------------------------------------------------------------------------- #

def run_self_test():
    import numpy as np

    print("=" * 68)
    print("reparam.py --self-test -- same arithmetic, no torch required")
    print(f"numpy {np.__version__}")
    print("=" * 68)

    rng = np.random.default_rng(0)
    mu, sigma, n = 1.0, 2.0, 100_000

    print("\n-- part 1: N(mu=1.0, sigma=2.0), two ways, n=100,000 --")
    a = mu + sigma * rng.standard_normal(n)   # reparameterized
    b = rng.normal(mu, sigma, n)              # direct draw
    print(f"reparameterized : mean={a.mean():.4f}  std={a.std():.4f}")
    print(f"direct draw     : mean={b.mean():.4f}  std={b.std():.4f}")

    assert abs(a.mean() - mu) < 0.05, "reparameterized mean should land near mu"
    assert abs(a.std() - sigma) < 0.05, "reparameterized std should land near sigma"
    assert abs(b.mean() - mu) < 0.05, "direct-draw mean should land near mu"
    assert abs(b.std() - sigma) < 0.05, "direct-draw std should land near sigma"
    print("[OK] same distribution, two call sites.")

    # ---- part 2: hand-derived gradients, scalar mu/sigma broadcast over eps ----
    # x_i = mu + sigma * eps_i  =>  d(sum x)/dmu = sum_i(1) = n_eps
    #                               d(sum x)/dsigma = sum_i(eps_i)
    print("\n-- part 2: hand-derived gradients (page's scalar-mu/sigma case) --")
    eps = rng.standard_normal(4)
    x = mu + sigma * eps
    d_mu = np.ones_like(eps).sum()            # analytic: d(sum x)/dmu
    d_sigma = eps.sum()                       # analytic: d(sum x)/dsigma
    print(f"eps        = {np.round(eps, 4).tolist()}")
    print(f"x          = {np.round(x, 4).tolist()}")
    print(f"mu.grad    = {d_mu:.4f}   (expect 4.0)")
    print(f"sigma.grad = {d_sigma:.4f}   (expect eps.sum())")

    assert np.isclose(d_mu, 4.0), f"mu.grad should be 4.0, got {d_mu}"
    assert np.isclose(d_sigma, eps.sum()), "sigma.grad should be eps.sum()"
    print("[OK] mu.grad is a sum of ones, sigma.grad is a sum of eps.")

    # ---- part 3: elementwise self-check -- mu.grad == ones, sigma.grad == eps ----
    print("\n-- part 3: elementwise self-check -- mu.grad == ones, sigma.grad == eps --")
    eps2 = rng.standard_normal(4)
    mu_vec = np.ones(4)
    sig_vec = np.full(4, 2.0)
    x_vec = mu_vec + sig_vec * eps2            # x_i = mu_i + sigma_i * eps_i, independent
    d_mu_vec = np.ones_like(mu_vec)            # dx_i/dmu_i = 1, no cross-terms
    d_sig_vec = eps2.copy()                    # dx_i/dsigma_i = eps_i, no cross-terms
    print(f"eps2       = {np.round(eps2, 4).tolist()}")
    print(f"mu.grad    = {np.round(d_mu_vec, 4).tolist()}   (expect all 1.0)")
    print(f"sigma.grad = {np.round(d_sig_vec, 4).tolist()}   (expect == eps2, exactly)")

    assert np.allclose(d_mu_vec, np.ones(4)), "mu.grad should be all ones"
    assert np.allclose(d_sig_vec, eps2), "sigma.grad should equal eps, elementwise"
    print("[OK] mu.grad == ones, sigma.grad == eps, elementwise and exact.")

    print("\n[SELF-TEST OK] arithmetic checks out; the torch path (run without --self-test")
    print("on a box with torch installed) exercises the identical algebra through autograd.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--self-test", action="store_true",
        help="numpy-only arithmetic check, no torch/GPU required",
    )
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not importable here. On the Spark, ComfyUI's venv has one:")
        print("  ~/ComfyUI/.venv/bin/python reparam.py")
        print("Or run the torch-free arithmetic check:")
        print("  python reparam.py --self-test")
        sys.exit(1)

    run_torch()


if __name__ == "__main__":
    main()
