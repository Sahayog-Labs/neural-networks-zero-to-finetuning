"""
svd_rank_probe.py  ·  ANN course page 39 (Rank, SVD & Low-Rank Approximation)

The LoRA hypothesis, verified on the real matrices — not a 64x64 toy.

Claim (page 39, D-12): a pretrained weight W0 is *nearly full rank*, while the
*update* dW a fine-tune adds on top is *low-rank*. Their scree plots have
opposite shapes. This script measures both on the actual Qwen3-8B q_proj
(4096x4096) and prints the rank each needs to reach 90% of its energy.

Usage:
    python svd_rank_probe.py \
        --weight Qwen3-8B/model-00001-of-00005.safetensors \
        --delta  delta_q_proj.pt

`delta_q_proj.pt` is (merged_finetuned_q_proj - W0), same shape as W0. Produce it
once from any LoRA/full fine-tune you have merged back into the base weights.
"""
import argparse
import torch
# NOTE: `safetensors` is imported lazily inside main(), only on the real-weights path.
# Keeping it out of module scope means `--self-test` (and a clean argparse usage message
# on a bare invocation) works on a box that has torch but not safetensors installed.

Q_PROJ_KEY = "model.layers.0.self_attn.q_proj.weight"


def energy_rank(W, frac=0.90):
    """Rank needed to retain `frac` of the squared-Frobenius energy, plus the
    full descending singular-value spectrum (for the scree plot)."""
    s = torch.linalg.svdvals(W.float())          # singular values, descending
    energy = torch.cumsum(s ** 2, dim=0) / torch.sum(s ** 2)
    rank = int((energy < frac).sum()) + 1
    return rank, s


def self_test():
    """No-GPU, no-weights sanity check of the core claim and of energy_rank().

    Builds a *synthetic* pair with the shapes and rank-structure the real q_proj and
    its fine-tune update have: W0 near-full-rank (random 4096x4096), dW genuinely
    low-rank (an outer product of a handful of vectors). Verifies energy_rank reports
    W0 ~ full and dW tiny, and that the page-39 assertion (rd << r0//4) holds — the
    same assertion the real-weights path makes, exercised here on CPU in ~1 s.
    """
    print("svd_rank_probe.py --self-test  (synthetic 4096x4096, CPU, no safetensors)")
    torch.manual_seed(0)
    n = 4096
    W0 = torch.randn(n, n)                      # near-full-rank by construction
    true_rank = 8
    A = torch.randn(n, true_rank)
    B = torch.randn(true_rank, n)
    dW = A @ B                                   # exactly rank-8 -> 90%-energy rank <= 8

    assert W0.shape == (n, n), W0.shape
    assert dW.shape == W0.shape, (dW.shape, W0.shape)

    r0, s0 = energy_rank(W0)
    rd, sd = energy_rank(dW)
    print(f"  W0  90%-energy rank: {r0:4d} / {n}   (slow decay -> nearly full rank)")
    print(f"  dW  90%-energy rank: {rd:4d} / {n}   (rank-{true_rank} outer product)")

    assert rd <= true_rank, (rd, true_rank)      # a rank-r matrix has <= r nonzero sigmas
    assert r0 > n // 4, r0                        # random square matrix is nearly full rank
    assert rd < r0 // 4, "expected dW's 90%-energy rank to be far below W0's"
    # energy_rank arithmetic: cumulative energy at the reported rank must clear the fraction
    e = (torch.cumsum(s0 ** 2, dim=0) / torch.sum(s0 ** 2))
    assert e[r0 - 1] >= 0.90 and (r0 == 1 or e[r0 - 2] < 0.90), "energy_rank off-by-one"
    print("  self-check       : dW rank << W0 rank, energy_rank boundary exact  ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic CPU sanity check; no GPU, no weights, no safetensors")
    ap.add_argument("--weight", help="safetensors shard holding q_proj")
    ap.add_argument("--delta", help="dW = finetuned_q_proj - W0 (.pt)")
    ap.add_argument("--key", default=Q_PROJ_KEY)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.weight or not args.delta:
        ap.error("the following arguments are required: --weight, --delta "
                 "(or pass --self-test for the no-weights sanity check)")

    from safetensors.torch import load_file
    W0 = load_file(args.weight)[args.key]
    dW = torch.load(args.delta)

    assert W0.shape == (4096, 4096), W0.shape      # the real thing, not a toy
    assert dW.shape == W0.shape, (dW.shape, W0.shape)

    r0, s0 = energy_rank(W0)
    rd, sd = energy_rank(dW)

    print(f"W0  90%-energy rank: {r0:4d} / 4096   (slow decay -> nearly full rank)")
    print(f"dW  90%-energy rank: {rd:4d} / 4096   (falls off a cliff -> low rank)")
    print(f"top-8 sigma  W0: {[round(float(x), 3) for x in s0[:8]]}")
    print(f"top-8 sigma  dW: {[round(float(x), 3) for x in sd[:8]]}")

    # the whole point, as an assertion: the update is far lower-rank than the weight
    assert rd < r0 // 4, "expected dW's 90%-energy rank to be far below W0's"

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.semilogy(range(1, len(s0) + 1), s0.cpu(), label=f"W0  (90% @ rank {r0})")
        plt.semilogy(range(1, len(sd) + 1), sd.cpu(), label=f"dW  (90% @ rank {rd})")
        plt.xlabel("singular-value index i")
        plt.ylabel("sigma_i (log)")
        plt.title("Scree: pretrained weight vs fine-tune update")
        plt.legend()
        plt.tight_layout()
        plt.savefig("scree.png", dpi=120)
        print("wrote scree.png")
    except ImportError:
        print("(matplotlib not installed; skipped the scree plot)")


if __name__ == "__main__":
    main()
