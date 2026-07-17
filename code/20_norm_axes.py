#!/usr/bin/env python3
"""
20_norm_axes.py -- LayerNorm, RMSNorm, BatchNorm are one operation, three axes.

Course artifact for p.20 (Normalization -- the axis question). The page's whole
claim: X has shape (B, ..., d) -- B examples, each a d-dim feature vector -- and
there are exactly two axes you can reduce over.

    BatchNorm    reduce over B   (down the batch)   one stat per FEATURE (column)
    LayerNorm    reduce over d   (across features)  one stat per EXAMPLE (row)
    RMSNorm      reduce over d, same as LayerNorm -- but skips the mean subtraction

    LN(x)_i = gamma_i * (x_i - mu) / sqrt(sigma^2 + eps) + beta_i
    RMS(x)_i = gamma_i * x_i / sqrt(mean(x_j^2) + eps)          <- no mu, no beta

This script does four things, in order:

  1. Reproduces the page's worked-by-hand triple x=[2.0,1.0,0.1]: LN -> [1.246,
     -0.043,-1.203], RMS -> [1.548,0.774,0.077] (the "drops the mean" claim, made
     a number: RMSNorm's output still averages to 0.800, not 0).
  2. Runs LayerNorm / RMSNorm / BatchNorm on one random (B=4, d=6) tensor and
     asserts each axis claim: LayerNorm's ROWS are mean~0/var~1, RMSNorm's ROWS
     have RMS~1 (mean left nonzero), BatchNorm's COLUMNS are mean~0/var~1.
  3. Proves "RMSNorm does strictly fewer ops" as a real count, not a vibe: a
     tiny op-counting wrapper runs both formulas and asserts RMSNorm calls
     subtraction ZERO times (LayerNorm calls it once, for the centring) and has
     a strictly smaller total call count.
  4. Reads Qwen3's rms_norm_eps (frozen at constants.md section 1.1, [VP],
     rms_norm_eps: 1e-06) -- optionally from a real downloaded config.json via
     --config, so the theory sits next to the config it predicts.

Backends
--------
Runs on **numpy alone** (no torch, no GPU needed) -- this is the self-test path
that verifies on any box, including one with no ML stack installed at all. If
torch IS importable, it additionally cross-checks the hand LayerNorm against
`torch.nn.functional.layer_norm` (the exact call the page names) and, on torch
>= 2.4 which ships one, `torch.nn.functional.rms_norm`.

Usage
-----
    python 20_norm_axes.py                       # full narration
    python 20_norm_axes.py --config path/to/Qwen3-8B/config.json
    python 20_norm_axes.py --self-test            # assertions only, exit 0/1

SAFETY: CPU only, <1 s, allocates a few KB. Writes nothing, installs nothing,
downloads nothing (no network calls -- --config reads a LOCAL file you already
have; without it, the frozen constants.md copy is used and labelled as such).
"""

import argparse
import json
import math
import sys

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# --------------------------------------------------------------------------
# Frozen numbers (constants.md section 1.1, [VP], Qwen3-8B config.json,
# fetched 2026-07-16 -- see the JSON block reproduced in FROZEN_QWEN3_CONFIG).
# --------------------------------------------------------------------------
RMS_NORM_EPS = 1e-6          # Qwen3's rms_norm_eps, constants.md section 1.1
LN_EPS = 1e-5                # page demo's LayerNorm/BatchNorm eps (torch default)
BN_EPS = 1e-5

# Verbatim from constants.md section 1.1 -- the anchor model's config, frozen
# here so the script runs with no network access. --config overrides this with
# a real file if you have one on disk (e.g. the HF cache on the Spark).
FROZEN_QWEN3_CONFIG = {
    "hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32,
    "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
    "vocab_size": 151936, "tie_word_embeddings": False, "rms_norm_eps": 1e-06,
    "rope_theta": 1000000, "max_position_embeddings": 40960, "hidden_act": "silu",
    "attention_bias": False, "torch_dtype": "bfloat16", "sliding_window": None,
    "rope_scaling": None,
}

# The course's canonical triple (used everywhere: p.09, p.12, p.20, ...)
CANONICAL_X = [2.0, 1.0, 0.1]
# Page's worked-by-hand results (3 d.p., "By hand, on the course's canonical
# triple") -- what section 2 below reproduces and asserts against.
WORKED_LN = [1.246, -0.043, -1.203]
WORKED_RMS = [1.548, 0.774, 0.077]
WORKED_RMS_MEAN = 0.800   # RMSNorm's output does NOT average to 0 -- the point


# ==========================================================================
# Part 1 -- plain-numpy hand formulas (the ground truth every backend must
# agree with; this is what runs when there is no torch on the box at all).
# ==========================================================================

def layer_norm_np(x, gamma=1.0, beta=0.0, eps=LN_EPS):
    """LN(x)_i = gamma_i*(x_i - mu)/sqrt(sigma^2 + eps) + beta_i, reduced over
    the LAST axis (per-row: one mu, one sigma^2 per example)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)     # biased, 1/d divisor
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def rms_norm_np(x, gamma=1.0, eps=RMS_NORM_EPS):
    """RMS(x)_i = gamma_i * x_i / sqrt(mean(x_j^2) + eps), reduced over the LAST
    axis. NO mu, NO beta -- that omission is the entire lesson of this page."""
    ms = (x ** 2).mean(axis=-1, keepdims=True)
    return gamma * x / np.sqrt(ms + eps)


def batch_norm_np(x, gamma=1.0, beta=0.0, eps=BN_EPS):
    """BatchNorm(x)_j = gamma_j*(x_j - mu_j)/sqrt(sigma_j^2 + eps) + beta_j,
    reduced over the FIRST axis (per-feature/column: one mu, one sigma^2 per
    column, shared across the B rows). The only axis flip versus LayerNorm."""
    mu = x.mean(axis=0, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=0, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


# ==========================================================================
# Part 2 -- an op-counting wrapper. Proves "RMSNorm does strictly fewer ops
# (no mean-subtract)" as a COUNT, not an assertion-by-assertion of the reader.
# Every arithmetic call site increments a shared counter; we run both
# formulas through it and compare.
# ==========================================================================

class OpCounter:
    def __init__(self):
        self.sub = 0
        self.add = 0
        self.mul = 0
        self.div = 0
        self.reduce = 0   # mean() calls
        self.other = 0    # sqrt(), **2

    @property
    def total(self):
        return self.sub + self.add + self.mul + self.div + self.reduce + self.other


class Counted:
    """Wraps a numpy array; every elementwise op it performs increments the
    shared OpCounter. This is NOT a FLOP counter (it counts CALL SITES, one
    per vectorised op) -- exactly the granularity the page's claim needs:
    "RMSNorm skips the mean-subtraction step", i.e. one fewer call to `-`."""

    __slots__ = ("val", "counter")

    def __init__(self, val, counter):
        self.val = np.asarray(val, dtype=np.float64)
        self.counter = counter

    def _wrap(self, val):
        return Counted(val, self.counter)

    @staticmethod
    def _unwrap(other):
        return other.val if isinstance(other, Counted) else other

    def __sub__(self, other):
        self.counter.sub += 1
        return self._wrap(self.val - self._unwrap(other))

    def __rsub__(self, other):
        self.counter.sub += 1
        return self._wrap(self._unwrap(other) - self.val)

    def __add__(self, other):
        self.counter.add += 1
        return self._wrap(self.val + self._unwrap(other))

    __radd__ = __add__

    def __mul__(self, other):
        self.counter.mul += 1
        return self._wrap(self.val * self._unwrap(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        self.counter.div += 1
        return self._wrap(self.val / self._unwrap(other))

    def __rtruediv__(self, other):
        self.counter.div += 1
        return self._wrap(self._unwrap(other) / self.val)

    def __pow__(self, p):
        self.counter.other += 1
        return self._wrap(self.val ** p)

    def sqrt(self):
        self.counter.other += 1
        return self._wrap(np.sqrt(self.val))

    def mean(self, axis=-1, keepdims=True):
        self.counter.reduce += 1
        return self._wrap(self.val.mean(axis=axis, keepdims=keepdims))


def layer_norm_counted(x_counted, gamma=1.0, beta=0.0, eps=LN_EPS):
    mu = x_counted.mean()
    centred = x_counted - mu                 # <-- the subtraction LayerNorm needs
    var = (centred ** 2).mean()
    normed = centred / (var + eps).sqrt()
    return normed * gamma + beta


def rms_norm_counted(x_counted, gamma=1.0, eps=RMS_NORM_EPS):
    ms = (x_counted ** 2).mean()
    normed = x_counted / (ms + eps).sqrt()    # <-- no subtraction anywhere
    return normed * gamma


def count_ops(d=6):
    """Run both formulas on a fresh d-vector through independent OpCounters
    and return (layer_counter, rms_counter)."""
    x = np.arange(1, d + 1, dtype=np.float64)   # any nonzero vector will do

    lc = OpCounter()
    layer_norm_counted(Counted(x, lc), gamma=1.0, beta=0.0)

    rc = OpCounter()
    rms_norm_counted(Counted(x, rc), gamma=1.0)

    return lc, rc


# ==========================================================================
# Part 3 -- Qwen3's rms_norm_eps: from a real config.json if given, else the
# frozen constants.md copy (labelled honestly either way -- notation section 9).
# ==========================================================================

def load_qwen3_eps(config_path):
    if config_path is None:
        return FROZEN_QWEN3_CONFIG["rms_norm_eps"], "frozen constants.md section 1.1 [VP] copy, no --config given"
    with open(config_path) as f:
        cfg = json.load(f)
    if "rms_norm_eps" not in cfg:
        raise KeyError(f"{config_path} has no 'rms_norm_eps' key -- is this really a Qwen3 config.json?")
    return cfg["rms_norm_eps"], f"read live from {config_path}"


# ==========================================================================
# Narration + self-test
# ==========================================================================

def self_test(config_path=None, quiet=False):
    p = (lambda *a, **k: None) if quiet else print

    if not HAVE_NUMPY:
        print("!! numpy is not importable -- this script needs numpy at minimum "
              "(it is a torch dependency, so any box with torch has it too).")
        print("   pip install numpy   # or, on the Spark, use a venv that has torch")
        sys.exit(1)

    p("Running self-checks (numpy only, no GPU, no torch required)...")

    # ---- 1. the worked-by-hand triple ------------------------------------
    x = np.array(CANONICAL_X)
    ln = layer_norm_np(x.reshape(1, -1), eps=LN_EPS)[0]
    rms = rms_norm_np(x.reshape(1, -1), eps=RMS_NORM_EPS)[0]

    for got, want, name in zip(ln, WORKED_LN, "012"):
        assert abs(got - want) < 5e-4, f"LN(x)_{name} should be {want}, got {got}"
    for got, want, name in zip(rms, WORKED_RMS, "012"):
        assert abs(got - want) < 5e-4, f"RMS(x)_{name} should be {want}, got {got}"
    assert abs(rms.mean() - WORKED_RMS_MEAN) < 5e-4, (
        f"RMSNorm output should still average to {WORKED_RMS_MEAN} (not 0) -- that IS the "
        f"'drops the mean' claim; got {rms.mean()}"
    )
    assert abs(ln.mean()) < 1e-9, "LayerNorm output must be centred on 0 by construction"
    p(f"  [OK] LN({CANONICAL_X}) = {[round(float(v), 3) for v in ln]}  (matches page's [1.246,-0.043,-1.203])")
    p(f"  [OK] RMS({CANONICAL_X}) = {[round(float(v), 3) for v in rms]}  (matches page's [1.548,0.774,0.077])")
    p(f"  [OK] RMSNorm output mean = {rms.mean():.3f} =/= 0  (LayerNorm output mean = {ln.mean():.1e})")

    # ---- 2. the random (B=4, d=6) tensor: axis claims ---------------------
    rng = np.random.default_rng(42)             # seeded -- deterministic, notation-clean
    X = rng.normal(loc=0.0, scale=3.0, size=(4, 6))   # off-centre/off-scale on purpose

    LN = layer_norm_np(X, eps=LN_EPS)
    RMS = rms_norm_np(X, eps=RMS_NORM_EPS)
    BN = batch_norm_np(X, eps=BN_EPS)

    row_mean_ln = LN.mean(axis=1)
    row_var_ln = LN.var(axis=1)
    assert np.allclose(row_mean_ln, 0.0, atol=1e-6), f"LayerNorm rows must be mean~0, got {row_mean_ln}"
    assert np.allclose(row_var_ln, 1.0, atol=1e-3), f"LayerNorm rows must be var~1, got {row_var_ln}"
    p(f"  [OK] LayerNorm: every ROW mean~0 ({row_mean_ln.max():.1e} max) / var~1 ({row_var_ln})")

    row_rms = np.sqrt((RMS ** 2).mean(axis=1))
    row_mean_rms = RMS.mean(axis=1)
    assert np.allclose(row_rms, 1.0, atol=1e-3), f"RMSNorm rows must be RMS~1, got {row_rms}"
    assert not np.allclose(row_mean_rms, 0.0, atol=1e-2), (
        "RMSNorm rows should NOT be mean~0 (no centring) -- got suspiciously-near-zero means"
    )
    p(f"  [OK] RMSNorm: every ROW RMS~1 ({row_rms}) but row mean left nonzero ({row_mean_rms})")

    col_mean_bn = BN.mean(axis=0)
    col_var_bn = BN.var(axis=0)
    assert np.allclose(col_mean_bn, 0.0, atol=1e-6), f"BatchNorm columns must be mean~0, got {col_mean_bn}"
    assert np.allclose(col_var_bn, 1.0, atol=1e-3), f"BatchNorm columns must be var~1, got {col_var_bn}"
    p(f"  [OK] BatchNorm: every COLUMN mean~0 / var~1 -- the ONE axis flip from LayerNorm")

    # cross-check: BatchNorm(X) == LayerNorm(X.T).T (same formula, axis swapped)
    assert np.allclose(BN, layer_norm_np(X.T, eps=BN_EPS).T, atol=1e-8), (
        "BatchNorm and LayerNorm must be the SAME formula with the axis transposed"
    )
    p(f"  [OK] BatchNorm(X) == LayerNorm(X.T).T exactly -- 'one operation, three call sites'")

    # ---- 3. op-count proof: RMSNorm strictly fewer ops, zero subtractions -
    lc, rc = count_ops(d=6)
    assert lc.sub == 1, f"LayerNorm's hand formula must call subtraction exactly once (centring), got {lc.sub}"
    assert rc.sub == 0, f"RMSNorm's hand formula must NEVER call subtraction, got {rc.sub}"
    assert rc.total < lc.total, (
        f"RMSNorm must do strictly fewer total ops than LayerNorm: RMSNorm={rc.total}, LayerNorm={lc.total}"
    )
    p(f"  [OK] op count -- LayerNorm calls sub={lc.sub} (the -mu centring), total={lc.total} ops")
    p(f"  [OK] op count -- RMSNorm  calls sub={rc.sub} (NEVER),           total={rc.total} ops "
      f"({lc.total - rc.total} fewer)")

    # ---- 4. Qwen3's rms_norm_eps -------------------------------------------
    eps, source = load_qwen3_eps(config_path)
    assert abs(eps - 1e-6) < 1e-12, f"Qwen3's rms_norm_eps should be 1e-6, got {eps} ({source})"
    assert abs(RMS_NORM_EPS - 1e-6) < 1e-12, "this script's RMS_NORM_EPS constant has drifted from 1e-6"
    p(f"  [OK] rms_norm_eps = {eps:.0e}  ({source})")

    # ---- 5. optional torch cross-check -------------------------------------
    if HAVE_TORCH:
        xt = torch.tensor(X)
        ln_t = F.layer_norm(xt, normalized_shape=(6,), weight=None, bias=None, eps=LN_EPS)
        assert np.allclose(ln_t.numpy(), LN, atol=1e-6), (
            "hand-written LayerNorm must match torch.nn.functional.layer_norm exactly"
        )
        p(f"  [OK] hand LayerNorm matches torch.nn.functional.layer_norm to 1e-6 "
          f"(torch {torch.__version__})")

        if hasattr(F, "rms_norm"):
            rms_t = F.rms_norm(xt, normalized_shape=(6,), weight=None, eps=RMS_NORM_EPS)
            assert np.allclose(rms_t.numpy(), RMS, atol=1e-6), (
                "hand-written RMSNorm must match torch.nn.functional.rms_norm exactly"
            )
            p(f"  [OK] hand RMSNorm matches torch.nn.functional.rms_norm to 1e-6")
        else:
            p(f"  (torch {torch.__version__} has no F.rms_norm -- skipping that one cross-check;"
              f" hand formula already verified against constants.md above)")
    else:
        p("  (torch not importable -- ran the numpy-only path. This IS the intended no-GPU,")
        p("   no-ML-stack self-test; every assertion above still holds without torch.)")

    p("All self-checks PASS.")


def narrate(config_path=None):
    print("=" * 72)
    print("NORMALIZATION -- one operation, three reduction axes")
    print(f"numpy {'yes' if HAVE_NUMPY else 'NO'}"
          + (f" | torch {torch.__version__}" if HAVE_TORCH else " | torch NO"))
    print("=" * 72)

    print("\n-- 1. the worked-by-hand triple x = [2.0, 1.0, 0.1] (d=3, gamma=1, beta=0) --")
    x = np.array(CANONICAL_X)
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    print(f"  LayerNorm: mu={mu:.4f}, sigma=sqrt(var)={math.sqrt(var):.4f}")
    ln = layer_norm_np(x.reshape(1, -1), eps=LN_EPS)[0]
    print(f"             LN(x) = {[round(float(v), 3) for v in ln]}  (page: {WORKED_LN})")
    rms_val = math.sqrt((x ** 2).mean())
    print(f"  RMSNorm:   RMS(x) = sqrt(mean(x^2)) = {rms_val:.4f}   <- no mu computed at all")
    rms = rms_norm_np(x.reshape(1, -1), eps=RMS_NORM_EPS)[0]
    print(f"             RMS(x) = {[round(float(v), 3) for v in rms]}  (page: {WORKED_RMS})")
    print(f"  RMSNorm output mean = {rms.mean():.3f}  <- NOT 0 (LayerNorm's is 0 by construction).")
    print(f"  That is the whole claim: RMSNorm drops the -mu centring and saves the arithmetic.")

    print("\n-- 2. one random (B=4, d=6) activation tensor, all three normalizers --")
    rng = np.random.default_rng(42)
    X = rng.normal(loc=0.0, scale=3.0, size=(4, 6))
    print(f"  X (seed=42, loc=0, scale=3):\n{np.array2string(X, precision=3, suppress_small=True)}")

    LN = layer_norm_np(X, eps=LN_EPS)
    RMS = rms_norm_np(X, eps=RMS_NORM_EPS)
    BN = batch_norm_np(X, eps=BN_EPS)

    print(f"\n  LayerNorm (reduce over d, per ROW):")
    print(f"    row means  = {np.array2string(LN.mean(axis=1), precision=2, suppress_small=True)}  (~0)")
    print(f"    row vars   = {np.array2string(LN.var(axis=1), precision=3)}  (~1)")

    print(f"\n  RMSNorm (reduce over d, per ROW, no mean):")
    print(f"    row RMS    = {np.array2string(np.sqrt((RMS**2).mean(axis=1)), precision=3)}  (~1)")
    print(f"    row means  = {np.array2string(RMS.mean(axis=1), precision=2)}  (NOT ~0 -- no centring)")

    print(f"\n  BatchNorm (reduce over B, per COLUMN):")
    print(f"    col means  = {np.array2string(BN.mean(axis=0), precision=2, suppress_small=True)}  (~0)")
    print(f"    col vars   = {np.array2string(BN.var(axis=0), precision=3)}  (~1)")
    print(f"  BatchNorm(X) == LayerNorm(X.T).T -- literally the same function, axis transposed.")

    print("\n-- 3. counting the ops: 'RMSNorm does strictly fewer' as a number --")
    lc, rc = count_ops(d=6)
    print(f"  LayerNorm call sites: mean x{lc.reduce}  sub x{lc.sub}  mul x{lc.mul}  "
          f"add x{lc.add}  div x{lc.div}  sqrt/pow x{lc.other}   = {lc.total} total")
    print(f"  RMSNorm  call sites: mean x{rc.reduce}  sub x{rc.sub}  mul x{rc.mul}  "
          f"add x{rc.add}  div x{rc.div}  sqrt/pow x{rc.other}   = {rc.total} total")
    print(f"  RMSNorm has {lc.total - rc.total} fewer call sites, and ZERO of them are a subtraction.")
    print(f"  ('sub' is exactly the -mu centring LayerNorm needs and RMSNorm skips.)")

    print("\n-- 4. Qwen3's stability floor, next to the theory it predicts --")
    eps, source = load_qwen3_eps(config_path)
    print(f"  rms_norm_eps = {eps:.0e}   ({source})")
    print(f"  This is the eps inside RMS(x)_i = gamma_i * x_i / sqrt(mean(x_j^2) + eps) --")
    print(f"  it only matters when mean(x_j^2) is near zero (a near-dead activation); it is")
    print(f"  smaller than LayerNorm's usual 1e-5 because RMS is never subtracting a mean that")
    print(f"  could itself be near zero, so it needs less of a stability floor.")

    print("\n" + "=" * 72)
    print("Same operation. Different axis. RMSNorm additionally skips the mean-subtract.")
    print("That is the entire taxonomy -- constants.md section 1.1, rms_norm_eps: 1e-06.")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(
        description="LayerNorm / RMSNorm / BatchNorm on one tensor -- the axis question, p.20."
    )
    ap.add_argument("--config", metavar="PATH", default=None,
                     help="path to a real Qwen3 config.json (e.g. from the HF cache on the Spark); "
                          "default uses the frozen constants.md copy, no network access")
    ap.add_argument("--self-test", action="store_true",
                     help="run assertions only, no narration, exit 0/1")
    args = ap.parse_args()

    self_test(config_path=args.config)
    if args.self_test:
        return
    print()
    narrate(config_path=args.config)


if __name__ == "__main__":
    main()
