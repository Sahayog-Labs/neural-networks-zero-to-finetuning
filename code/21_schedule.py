#!/usr/bin/env python3
"""
21_schedule.py -- warmup-cosine, indexed by the OPTIMIZER STEP k, never the epoch.

Course artifact for p.21 (Schedules & Why Warmup Exists). Implements the page's boxed
formula verbatim:

    eta_k = eta_max * k / T_warm                                    k <= T_warm  (linear warmup)
    eta_k = eta_min + 0.5*(eta_max-eta_min)*(1+cos(pi*(k-T_warm)/(T_total-T_warm)))   k > T_warm  (cosine decay)

with eta_min = eta_max / 10 (constants.md Sec 9.4: "cosine to eta_max/10") and the page's own
production example T_warm=2000, T_total=100000, eta_max=2e-4, AdamW betas=(0.9, 0.95).

Two things this script exists to make impossible to miss, because they are the page's whole
point:

  1. THE PEAK IS EXACTLY AT k = T_warm.  Not "around there" -- exactly, by construction
     (the two pieces of the schedule meet at multiplier 1.0).
  2. k IS THE OPTIMIZER STEP, NOT THE EPOCH.  Calling scheduler.step() once per epoch instead
     of once per step is a real, common bug (page's "Schedule by epoch" warn box) -- and it is
     silent, because nothing crashes. This script reproduces that bug on purpose and shows the
     multiplier stuck near zero for the entire run when T_warm (in STEPS) is confused for a
     count of EPOCHS.

Reason 2 from the page -- the one you can verify yourself -- is Adam's beta2=0.999 second-
moment memory: 1/(1-0.999) = 1000 steps (constants.md Sec 9.3, [DER], computed on p.17). That
is *why* T_warm defaults cluster at "1000-ish, rounded up to a flat 2000" (constants.md Sec
9.4, deep-dive on the page): warmup has to outlast the window in which one unlucky early
gradient can still poison v_k.

Usage
-----
    python 21_schedule.py                       # narrated schedule + step-vs-epoch bug demo
    python 21_schedule.py --total 50000 --warmup 500
    python 21_schedule.py --self-test            # assertions only, no narration, exit 0/1

SAFETY: pure Python + math module. No GPU, no network, no files touched, runtime <1s. If a
torch install is importable (e.g. the Spark's ComfyUI venv, or a fresh venv per the hardware
brief) the script additionally builds the page's exact torch.optim.lr_scheduler.LambdaLR and
cross-checks it against the pure-math formula above -- but this is optional and skipped
cleanly when torch is absent, which is the case on this machine.
"""

import argparse
import math

# --------------------------------------------------------------------------------------- #
# Frozen numbers -- constants.md Sec 9.4 ("2026 LLM AdamW defaults") and Sec 9.3 (Adam
# beta2 memory, [DER], the mechanism Reason 2 on the page is built on).
# --------------------------------------------------------------------------------------- #

ETA_MIN_RATIO = 0.1                 # eta_min = eta_max / 10 -- constants.md Sec 9.4, [VP]
ADAM_BETA2 = 0.999                  # constants.md Sec 9.3 / page 17
ADAM_BETA2_MEMORY_STEPS = 1.0 / (1.0 - ADAM_BETA2)   # = 1000 exactly, [DER]

# The page's own worked code block (lines under "Try it" / code/21_schedule.py on p.21):
# T_warm=2000, T_total=100000, eta_max=2e-4, AdamW betas=(0.9, 0.95).
DEFAULT_T_WARM = 2000
DEFAULT_T_TOTAL = 100_000
DEFAULT_ETA_MAX = 2e-4


# --------------------------------------------------------------------------------------- #
# The schedule itself -- pure Python, matches the page's torch lr_lambda term for term.
# Returns the MULTIPLIER (0..1, floor eta_min_ratio), i.e. exactly what a torch LambdaLR
# lr_lambda is supposed to return; eta_k = eta_max * multiplier.
# --------------------------------------------------------------------------------------- #

def warmup_cosine(k: int, T_warm: int, T_total: int, eta_min_ratio: float = ETA_MIN_RATIO) -> float:
    """k is the OPTIMIZER STEP, not the epoch -- the bug this function exists to prevent."""
    if k <= T_warm:
        return k / max(T_warm, 1)
    progress = (k - T_warm) / max(T_total - T_warm, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return eta_min_ratio + (1.0 - eta_min_ratio) * cosine


def eta(k: int, T_warm: int, T_total: int, eta_max: float, eta_min_ratio: float = ETA_MIN_RATIO) -> float:
    return eta_max * warmup_cosine(k, T_warm, T_total, eta_min_ratio)


def try_torch_crosscheck(T_warm: int, T_total: int, eta_max: float) -> None:
    """Optional: build the page's EXACT torch.optim.lr_scheduler.LambdaLR and confirm it
    agrees with warmup_cosine() above at a handful of steps. Needs only CPU torch -- no CUDA,
    no GPU contention -- but this machine has no torch install at all, so this is expected to
    print the "not available" note here and actually run on the Spark or any fresh venv."""
    try:
        import torch
    except ImportError:
        print("  torch not importable here -- skipping the LambdaLR cross-check.")
        print("  (Runs on any CPU-only torch install; not GPU-dependent. On the Spark, use")
        print("   a fresh venv per hardware-ground-truth.md Sec 3 -- never ComfyUI's.)")
        return

    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=eta_max, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda k: warmup_cosine(k, T_warm=T_warm, T_total=T_total))

    check_steps = sorted({0, 1, T_warm // 2, T_warm, T_warm + 1, T_total // 2, T_total})
    print("  torch LambdaLR cross-check (page's exact scheduler construction):")
    worst = 0.0
    for k in check_steps:
        # LambdaLR's internal counter starts at 0 and multiplies base_lr by lr_lambda(epoch);
        # step() advances it, so read get_last_lr() after stepping to k.
        opt.param_groups[0]["lr"] = eta_max
        scheduler.last_epoch = k - 1
        scheduler.step()
        got = scheduler.get_last_lr()[0]
        want = eta(k, T_warm, T_total, eta_max)
        worst = max(worst, abs(got - want))
        print(f"    k={k:>7}  torch LambdaLR={got:.8f}  pure-math={want:.8f}")
    assert worst < 1e-9, f"torch LambdaLR disagrees with pure-math formula by {worst}"
    print(f"  agreement within {worst:.2e} -- PASS")


# --------------------------------------------------------------------------------------- #
# Narration
# --------------------------------------------------------------------------------------- #

def _bar(multiplier: float, width: int = 40) -> str:
    n = max(0, min(width, round(multiplier * width)))
    return "#" * n + "." * (width - n)


def narrate(T_warm: int, T_total: int, eta_max: float) -> None:
    print("=" * 78)
    print("WARMUP-COSINE SCHEDULE -- indexed by optimizer step k (p.21)")
    print("=" * 78)
    print(f"  T_warm     = {T_warm:>8,} steps   T_total = {T_total:>8,} steps  "
          f"({100*T_warm/T_total:.2f}% warmup)")
    print(f"  eta_max    = {eta_max:.2e}")
    print(f"  eta_min    = {eta_max * ETA_MIN_RATIO:.2e}   (= eta_max/10, constants.md Sec 9.4)")
    print()

    print("  Phase 1 -- linear warmup, k = 0 .. T_warm:")
    warm_pts = sorted({0, T_warm // 4, T_warm // 2, (3 * T_warm) // 4, T_warm})
    for k in warm_pts:
        m = warmup_cosine(k, T_warm, T_total)
        print(f"    k={k:>7}  eta_k={eta(k, T_warm, T_total, eta_max):.2e}  "
              f"[{_bar(m)}] {m:.3f}")
    print()

    print("  Phase 2 -- cosine decay, k = T_warm .. T_total:")
    n_pts = 10
    decay_pts = sorted({T_warm + round(i * (T_total - T_warm) / n_pts) for i in range(n_pts + 1)})
    for k in decay_pts:
        m = warmup_cosine(k, T_warm, T_total)
        print(f"    k={k:>7}  eta_k={eta(k, T_warm, T_total, eta_max):.2e}  "
              f"[{_bar(m)}] {m:.3f}")
    print()
    print(f"  Never reaches multiplier 0.0 -- floors at {ETA_MIN_RATIO:.2f} "
          f"(eta_max/10), by design (p.21 'Decay to eta_max/10, not to zero').")

    print()
    print("-" * 78)
    print("  WHY T_warm defaults to ~1000-2000, not 10 or 100,000:")
    print(f"    Adam's beta2={ADAM_BETA2} second moment has a memory of "
          f"1/(1-{ADAM_BETA2}) = {ADAM_BETA2_MEMORY_STEPS:.0f} steps (constants.md Sec 9.3, [DER]).")
    print(f"    One bad early gradient can poison v_k for that whole window. T_warm must be at")
    print(f"    least on that order -- the page's flat default of 2000 is that floor, rounded up.")
    print("-" * 78)

    print()
    print("=" * 78)
    print("THE BUG: scheduling by EPOCH instead of by STEP k (p.21 'Five ways to misread this')")
    print("=" * 78)
    demo_epochs = 10
    steps_per_epoch = T_total // demo_epochs
    print(f"  Same run, viewed two ways: {demo_epochs} epochs x {steps_per_epoch:,} steps/epoch "
          f"= {T_total:,} steps total.")
    print(f"  T_warm = {T_warm:,} steps was tuned for STEP-indexing.")
    print()
    print("  correct (k = global step, incremented every optimizer.step()):")
    correct_at_warm = warmup_cosine(T_warm, T_warm, T_total)
    print(f"    at k=T_warm={T_warm:,}  (epoch {T_warm / steps_per_epoch:.2f} of the run)  "
          f"multiplier={correct_at_warm:.3f}  <-- reaches full eta_max on schedule")
    print()
    print("  buggy (scheduler.step() called once per EPOCH -- k passed in is the epoch count):")
    for e in range(0, demo_epochs + 1, 2):
        m_bug = warmup_cosine(e, T_warm, T_total)
        print(f"    \"k\"=epoch {e:>2}  multiplier={m_bug:.5f}  "
              f"[{_bar(m_bug)}]  -- essentially stuck near zero")
    print()
    print(f"  With T_warm={T_warm:,} steps and only {demo_epochs} epoch-indexed calls, the schedule")
    print(f"  never gets past k={demo_epochs} out of a {T_warm:,}-step warmup window: the model trains at a")
    print(f"  learning rate pinned near zero for the ENTIRE run and nothing ever crashes to tell you.")


# --------------------------------------------------------------------------------------- #
# Self-test -- no GPU, no torch required. This is the path this course's local-verification
# rule runs: py_compile + this function, with real output pasted into the build record.
# --------------------------------------------------------------------------------------- #

def self_test() -> None:
    print("Running self-checks (no GPU, pure math)...")

    # --- frozen constants agree with constants.md Sec 9.4 / 9.3 ------------------------
    assert ETA_MIN_RATIO == 0.1, "eta_min must be exactly eta_max/10 per constants.md Sec 9.4"
    assert abs(ADAM_BETA2_MEMORY_STEPS - 1000.0) < 1e-9, (
        "Adam beta2=0.999 memory must be exactly 1000 steps (constants.md Sec 9.3, [DER])")
    print(f"  eta_min_ratio=0.1, Adam beta2 memory=1/(1-0.999)="
          f"{ADAM_BETA2_MEMORY_STEPS:.0f} steps -- PASS (constants.md Sec 9.3/9.4)")

    T_warm, T_total, eta_max = DEFAULT_T_WARM, DEFAULT_T_TOTAL, DEFAULT_ETA_MAX

    # --- 1. PEAK IS EXACTLY AT k = T_warm ------------------------------------------------
    m_at_warm = warmup_cosine(T_warm, T_warm, T_total)
    assert m_at_warm == 1.0, f"multiplier at k=T_warm must be exactly 1.0, got {m_at_warm}"
    # and it is the max over the whole run, not just locally
    sample_ks = list(range(0, T_warm, max(1, T_warm // 50))) + \
        list(range(T_warm, T_total, max(1, (T_total - T_warm) // 50))) + [T_total]
    peak_k = max(sample_ks, key=lambda k: warmup_cosine(k, T_warm, T_total))
    assert warmup_cosine(peak_k, T_warm, T_total) <= 1.0 + 1e-12
    assert abs(warmup_cosine(T_warm, T_warm, T_total) - max(
        warmup_cosine(k, T_warm, T_total) for k in sample_ks)) < 1e-12, (
        "k=T_warm must attain the global max multiplier over the sampled run")
    print(f"  peak-at-end-of-warmup: multiplier(k=T_warm={T_warm:,}) = {m_at_warm:.6f} exactly, "
          f"and is the run's maximum -- PASS")

    # --- 2. warmup is linear 0 -> 1 -------------------------------------------------------
    assert warmup_cosine(0, T_warm, T_total) == 0.0, "warmup must start at multiplier 0.0"
    m_half = warmup_cosine(T_warm // 2, T_warm, T_total)
    assert abs(m_half - 0.5) < 1e-9, f"warmup midpoint should be ~0.5, got {m_half}"
    print(f"  linear warmup: k=0 -> 0.000, k=T_warm/2 -> {m_half:.3f} -- PASS")

    # --- 3. cosine decays to the floor exactly at k = T_total -----------------------------
    m_at_end = warmup_cosine(T_total, T_warm, T_total)
    assert abs(m_at_end - ETA_MIN_RATIO) < 1e-9, (
        f"multiplier at k=T_total must equal eta_min_ratio={ETA_MIN_RATIO}, got {m_at_end}")
    assert m_at_end > 0.0, "schedule must never reach exactly 0 (p.21: floor, not decay-to-zero)"
    print(f"  decay floor: multiplier(k=T_total) = {m_at_end:.6f} = eta_min_ratio "
          f"(never reaches 0) -- PASS")

    # --- 4. monotonic: non-decreasing through warmup, non-increasing through decay -------
    warm_seq = [warmup_cosine(k, T_warm, T_total) for k in range(0, T_warm + 1, max(1, T_warm // 40))]
    assert all(b >= a - 1e-12 for a, b in zip(warm_seq, warm_seq[1:])), "warmup must be non-decreasing"
    decay_seq = [warmup_cosine(k, T_warm, T_total) for k in range(T_warm, T_total + 1, max(1, (T_total - T_warm) // 40))]
    assert all(b <= a + 1e-12 for a, b in zip(decay_seq, decay_seq[1:])), "cosine decay must be non-increasing"
    print(f"  monotonicity: {len(warm_seq)} warmup samples non-decreasing, "
          f"{len(decay_seq)} decay samples non-increasing -- PASS")

    # --- 5. THE STEP-VS-EPOCH BUG: schedule indexes by step, not epoch -------------------
    # If someone mistakenly calls scheduler.step() once per EPOCH instead of once per
    # optimizer step, the value passed in as "k" is really an epoch count (0..E), which for
    # any realistic E is far smaller than T_warm. The schedule must then stay pinned near
    # zero for the whole run -- reproducing the page's "can end up doing nothing at all" bug.
    demo_epochs = 10
    epoch_multipliers = [warmup_cosine(e, T_warm, T_total) for e in range(demo_epochs + 1)]
    assert all(m < 0.01 for m in epoch_multipliers), (
        f"epoch-indexed misuse should stay far below peak (<0.01) when T_warm={T_warm} steps "
        f">> {demo_epochs} epochs, got max={max(epoch_multipliers)}")
    assert m_at_warm - max(epoch_multipliers) > 0.99, (
        "the gap between correct step-indexed peak and buggy epoch-indexed misuse must be huge")
    print(f"  step-vs-epoch bug: epoch-indexed misuse over {demo_epochs} epochs stays below "
          f"{max(epoch_multipliers):.4f} (vs. true peak 1.0) -- PASS "
          f"(schedule genuinely indexes by step k, not epoch)")

    # --- 6. eta_k = eta_max * multiplier, at the two anchor points -----------------------
    assert eta(T_warm, T_warm, T_total, eta_max) == eta_max, "eta at k=T_warm must equal eta_max exactly"
    assert abs(eta(T_total, T_warm, T_total, eta_max) - eta_max * ETA_MIN_RATIO) < 1e-15, (
        "eta at k=T_total must equal eta_max/10 exactly")
    print(f"  eta_k anchors: eta(T_warm)={eta(T_warm, T_warm, T_total, eta_max):.2e}=eta_max, "
          f"eta(T_total)={eta(T_total, T_warm, T_total, eta_max):.2e}=eta_max/10 -- PASS")

    print("All self-checks PASS.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Warmup-cosine LR schedule as a function of optimizer step k -- p.21 companion script."
    )
    ap.add_argument("--total", type=int, default=DEFAULT_T_TOTAL,
                     help=f"T_total, total planned optimizer steps (default {DEFAULT_T_TOTAL:,})")
    ap.add_argument("--warmup", type=int, default=DEFAULT_T_WARM,
                     help=f"T_warm, number of warmup steps (default {DEFAULT_T_WARM:,})")
    ap.add_argument("--eta-max", type=float, default=DEFAULT_ETA_MAX,
                     help=f"peak learning rate eta_max (default {DEFAULT_ETA_MAX:.0e})")
    ap.add_argument("--self-test", action="store_true",
                     help="run assertions only, no narration, exit 0 on success")
    args = ap.parse_args()

    if args.warmup > args.total:
        raise SystemExit(f"--warmup ({args.warmup}) must be <= --total ({args.total})")

    self_test()
    if args.self_test:
        return
    print()
    narrate(args.warmup, args.total, args.eta_max)
    print()
    try_torch_crosscheck(args.warmup, args.total, args.eta_max)


if __name__ == "__main__":
    main()
