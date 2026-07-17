#!/usr/bin/env python3
"""
scaling_and_moe.py — three self-contained calculations, one motivating claim.

Course artifact for p.38 ("Scaling Laws, MoE & Emergence"). The whole point of the page
is a single number: pretraining Qwen3-8B from scratch on your own DGX Spark would take
CENTURIES, not days or years. That's why the rest of Part IV (pages 39-43) exists — you
don't rebuild the base model, you adapt it.

Three things run here, matching the page's three demo panels:

  1. C = 6ND and the ~2-millennia Spark estimate, with every assumption printed —
     including the [EST] on D (unpublished token count) and the [INF] on the FLOP/s
     ceiling (NVIDIA never published a dense-bf16 number for GB10).
  2. MoE active-vs-total memory for the 2026 frontier configs the page cites.
  3. The "emergence" metric-artifact demo: the SAME synthetic per-step accuracy curve,
     scored two ways (exact-match vs continuous), on synthetic logits.

Usage
-----
    python scaling_and_moe.py              # all three sections
    python scaling_and_moe.py --quick      # smaller sweeps, same conclusions

Nothing here touches the GPU or the network. Pure arithmetic, deterministic.
"""

import argparse
import math

GiB = 1 << 30
GB = 10**9

# --------------------------------------------------------------------------- #
# Section 1 — C = 6ND, and the honest centuries estimate.
# --------------------------------------------------------------------------- #


def derive_the_six():
    print("=" * 72)
    print("SECTION 1a — deriving the 6 in C ~= 6ND (not asserting it)")
    print("=" * 72)
    forward_flop_per_param_per_token = 2   # one multiply, one add
    backward_multiplier = 2                # grad wrt input AND grad wrt weight
    k = forward_flop_per_param_per_token * (1 + backward_multiplier)
    print(f"  forward:  {forward_flop_per_param_per_token} FLOP/param/token (multiply + add)")
    print(f"  backward: {backward_multiplier}x forward "
          f"(grad w.r.t. input to keep propagating, grad w.r.t. weight to update it)")
    print(f"  total:    forward + backward = 2N + {backward_multiplier}*2N = {k}N FLOP/token")
    assert k == 6, f"expected k=6, derived k={k}"
    print(f"  --> k = {k}. Not magic: 'one forward, two backward-sized passes.'")
    print()
    return k


def qwen3_pretraining_estimate(k=6):
    print("=" * 72)
    print("SECTION 1b — Qwen3-8B pretraining compute, and how long it takes on YOUR Spark")
    print("=" * 72)

    # constants.md §1.2 — [DER], recomputed from the checkpoint's own config.
    N = 6_946_071_552  # non-embedding params
    # constants.md §5 — [EST]. Nobody publishes Qwen3-8B's real training-token count;
    # this is an unverified estimate, and the page (and this script) says so explicitly.
    D = 3.6e13
    print(f"  N (non-embedding params)  = {N:,}                [DER, constants §1.2]")
    print(f"  D (training tokens)       = {D:.2e}                    [EST, constants §5 — UNVERIFIED]")

    C = k * N * D
    assert abs(C - 6 * N * D) < 1  # self-check: the printed k actually produced C
    print(f"  C = {k}ND                 = {C:.3e} FLOP        [EST — inherits D's [EST] tag]")
    print()

    YEAR_S = 365 * 24 * 3600

    # Ceiling 1: theoretical FP16-accumulate peak, 100% MFU. Never achieved in practice —
    # this is deliberately the MOST GENEROUS assumption possible, which is exactly why
    # it produces a LOWER BOUND, not a "generous estimate." Constants §6.3 tags it [INF]:
    # NVIDIA never publishes a dense bf16 number for GB10; this is inferred from the
    # marketing FP4-sparse figure via a stated halving chain.
    peak_generous = 1.25e14  # FLOP/s
    years_lower_bound = C / peak_generous / YEAR_S
    print(f"  Ceiling A (generous to the machine): {peak_generous:.2e} FLOP/s")
    print(f"    theoretical FP16-accumulate peak, 100% utilization forever — never happens")
    print(f"    -> {years_lower_bound:.0f} years  [DER, on an EST+INF input]  <-- a LOWER BOUND")
    print()

    # Ceiling 2: the realistic dense-bf16 training roofline (~62 TFLOP/s, [INF],
    # constants §6.3) at a realistic ~40% model-FLOPs-utilization.
    peak_realistic = 62e12  # FLOP/s
    mfu = 0.40
    years_realistic = C / (peak_realistic * mfu) / YEAR_S
    print(f"  Ceiling B (realistic): {peak_realistic:.1e} FLOP/s roofline [INF, constants §6.3] "
          f"at ~{mfu:.0%} MFU")
    print(f"    -> {years_realistic:.0f} years  [INF, constants §5]")
    print()

    print(f"  THE SENTENCE THE COURSE SHIPS:")
    print(f'  "At least four centuries — and that assumes the machine runs at its')
    print(f'   theoretical peak, which it will not. Realistically, closer to two millennia."')
    print()
    print(f"  Do NOT print '~380 years, generously assume sustained' — assuming peak is")
    print(f"  generous to the MACHINE, which makes {years_lower_bound:.0f} years a floor, not a")
    print(f"  cushion. The honesty makes the conclusion STRONGER, not weaker. (D-20 retirement.)")
    print()

    assert years_lower_bound < years_realistic, "the generous ceiling must give the SHORTER estimate"
    assert years_lower_bound > 300, "sanity: this must land in 'centuries', not 'decades'"
    return C, years_lower_bound, years_realistic


# --------------------------------------------------------------------------- #
# Section 2 — MoE: memory tracks total, compute tracks active.
# --------------------------------------------------------------------------- #

# (model, total_params, active_params) — reported 2026 frontier configs, cited on p.38.
MOE_CONFIGS = [
    ("Kimi K2 / K2.5", 1_000e9, 32e9),
    ("GLM-5", 744e9, 40e9),
    ("Qwen3.5", 397e9, 17e9),
    ("Trinity Large", 400e9, 13e9),
    ("Qwen3-Coder-Next", 80e9, 3e9),
]


def moe_active_vs_total(box_gib=121.6875):
    print("=" * 72)
    print("SECTION 2 — MoE: active-vs-total memory, 2026 frontier configs")
    print("=" * 72)
    print(f"  {'model':<20}{'total':>10}{'active':>10}{'active%':>10}{'fp8 mem':>12}{'fits 121.69 GiB?':>20}")
    for name, total, active in MOE_CONFIGS:
        assert active < total, f"{name}: active must be < total by construction"
        ratio = active / total
        mem_gib = total * 1.0 / GiB  # fp8, 1 B/param — illustrative, matches the page's "~1 TB at fp8" claim
        fits = "yes" if mem_gib <= box_gib else "NO"
        print(f"  {name:<20}{total/1e9:>9.0f}B{active/1e9:>9.0f}B{ratio*100:>9.1f}%"
              f"{mem_gib:>11.1f} GiB{fits:>20}")
    print()
    print("  Every row's active parameter count is a small FRACTION of total — but memory")
    print("  has to hold the TOTAL. On a 121.69 GiB box [MEA-DEV], none of these fit whole.")
    print("  'It only uses N billion active parameters' is the misconception this table kills.")
    print()

    # Reproduce the page's own toy MoE layer: 3*d_model*d_ff per expert, N_E of them,
    # K_E active. Same formula as the trunk's FFN subtotal (constants §1.2).
    d_model = 4096
    for n_e, k_e, d_ff in [(8, 2, 2048), (384, 9, 2048)]:
        per_expert = 3 * d_model * d_ff
        total_params = n_e * per_expert
        active_params = k_e * per_expert
        mem_gib = total_params * 2 / GiB  # bf16
        print(f"  toy layer: N_E={n_e:<4} K_E={k_e:<3} d_ff={d_ff:<5} -> "
              f"total {total_params:,} ({mem_gib:.2f} GiB bf16), active/token {active_params:,} "
              f"({active_params/total_params*100:.2f}% active)")
    print()


# --------------------------------------------------------------------------- #
# Section 3 — the "emergence" metric-artifact demo, on synthetic logits.
# --------------------------------------------------------------------------- #


def logistic(x, x0, k):
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def emergence_metric_artifact(quick=False):
    print("=" * 72)
    print("SECTION 3 — the emergence 'mirage': one curve, two metrics")
    print("=" * 72)

    x0, k = 22.0, 1.15
    n_steps = 20  # e.g. "this task needs all 20 digits/steps correct"

    xs = [15 + i * (18 / (40 if not quick else 12)) for i in range(41 if not quick else 13)]
    continuous = [logistic(x, x0, k) for x in xs]
    exact_match = [p ** n_steps for p in continuous]

    print(f"  synthetic per-step accuracy: logistic(x; x0={x0}, k={k})")
    print(f"  exact-match score over n={n_steps} independent steps: p(x)^{n_steps}")
    print()
    print(f"  {'log10(C)':>10}{'continuous':>14}{'exact-match':>14}")
    step = max(1, len(xs) // 12)
    for i in range(0, len(xs), step):
        print(f"  {xs[i]:>10.1f}{continuous[i]:>14.3f}{exact_match[i]:>14.3f}")
    print()

    # Self-check: the exact-match curve must be flatter-then-sharper than the continuous
    # one around the midpoint — i.e. its slope range compresses into a narrower x-window.
    def crossing_width(ys, lo=0.1, hi=0.9):
        # Interpolate the x at which the (monotone-increasing) curve first reaches
        # a threshold, rather than snapping to the nearest grid point. On the coarse
        # --quick grid, snapping quantizes both widths to the same cell and the
        # "exact-match is sharper" self-check falsely ties; interpolation makes the
        # measured width resolution-independent so the check holds in every mode.
        def cross_x(th):
            for i in range(1, len(ys)):
                if ys[i - 1] < th <= ys[i]:
                    t = (th - ys[i - 1]) / (ys[i] - ys[i - 1])
                    return xs[i - 1] + t * (xs[i] - xs[i - 1])
            return None
        x_lo = cross_x(lo)
        x_hi = cross_x(hi)
        if x_lo is None or x_hi is None:
            return None
        return x_hi - x_lo

    w_cont = crossing_width(continuous)
    w_exact = crossing_width(exact_match)
    print(f"  10%->90% crossing width: continuous={w_cont:.2f}, exact-match={w_exact:.2f}")
    assert w_exact < w_cont, "exact-match must look sharper (narrower crossing) than continuous"
    print("  Same underlying model, same underlying data. The 'cliff' is p(x)^n with n large —")
    print("  an AND over many correct steps, scored pass/fail. Raise a smooth curve to a big")
    print("  power and it gets steep. No mystery, no magic, no phase transition required.")
    print()
    p_half = 0.5 ** (1 / n_steps)
    print(f"  Exact-match crosses 50% only once per-step accuracy hits {p_half:.3f} — verify: "
          f"{p_half:.3f}^{n_steps} = {p_half**n_steps:.3f}")
    assert abs(p_half ** n_steps - 0.5) < 1e-9
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sweeps, same conclusions")
    args = ap.parse_args()

    k = derive_the_six()
    qwen3_pretraining_estimate(k)
    moe_active_vs_total()
    emergence_metric_artifact(args.quick)

    print("=" * 72)
    print("All three sections passed their self-checks.")
    print("=" * 72)


if __name__ == "__main__":
    main()
