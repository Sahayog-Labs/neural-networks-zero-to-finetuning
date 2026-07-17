"""
memory.py -- the ONE peak-VRAM stopwatch, shared verbatim by every training/inference
script in the course (spec-code.md sec B).

Every script that trains or generates prints a MEASURED peak-memory line next to the
PREDICTED number from utils/ledger.py or 05_memory_ledger.py -- landing that prediction
is the whole point of p.18/p.49 (constants.md sec2). Before this file, that measurement
was hand-rolled at every call site:

    torch.cuda.reset_peak_memory_stats()
    ... do the heavy thing ...
    peak_gib = torch.cuda.max_memory_allocated() / GiB

(see 11_finetune_qlora.py, 13_ablate.py, 15_grpo_rlvr.py, bench_spark.py, sample_flux.py,
train_lora_flux.py -- six copies of the same five lines). `peak_memory()` is that pattern
as a context manager, so new scripts import it instead of retyping it:

    from utils.memory import peak_memory

    with peak_memory(label="LoRA SFT") as pm:
        trainer.train()
    print(f"peak: {pm.peak_gib:.2f} GiB")   # None if no CUDA -- always safe to read

Units discipline (constants.md sec0): GiB = 2**30 B, binary, how DRAM actually addresses.
This module NEVER reports GB (decimal) -- peak VRAM is always a GiB number in this course.

GOTCHA, stated because it bites people: `reset_peak_memory_stats` resets a counter that is
GLOBAL to the device, not scoped to your `with` block. Two `peak_memory()` blocks back to
back on the same device is fine (each reset wipes the other's history, which is what you
want -- a fresh watermark per block). But a `peak_memory()` block nested INSIDE another one
resets the outer block's counter too, so the outer block's `peak_gib` will only reflect
memory allocated AFTER the inner block reset it, not the true peak of the whole outer span.
Do not nest these; run them sequentially.

No CUDA / no torch (this laptop; also `--no-gpu` runs on the Spark): the context manager
does not raise -- it becomes a plain wall-clock timer, `peak_bytes`/`peak_gib` stay `None`.
Every call site above already needed an `if torch.cuda.is_available():` guard before this
existed; the guard now lives here, once.
"""

import sys
import time

GiB = 1 << 30       # 1,073,741,824 -- binary gibibyte (constants.md sec0)


class peak_memory:
    """Context manager: measures GPU peak-allocated memory (and wall-clock) across the
    `with` block, wrapping torch.cuda.reset_peak_memory_stats() / max_memory_allocated().

    Attributes set on the instance after __exit__ runs:
        peak_bytes   int or None   raw torch.cuda.max_memory_allocated() result
        peak_gib     float or None peak_bytes / GiB (binary) -- what you print
        elapsed_s    float         wall-clock seconds spent inside the `with` block

    `peak_bytes`/`peak_gib` are None (not 0.0) whenever there is no usable CUDA device --
    "we didn't measure" must never print as "we measured zero".

    Args:
        device  passed straight through to reset_peak_memory_stats()/max_memory_allocated()
                (None = current device, matching torch's own default).
        label   if given, __exit__ prints one self-narrating line. Omit to stay silent and
                read pm.peak_gib / pm.elapsed_s yourself (e.g. to assert against a
                prediction, as 11_finetune_qlora.py does).

    Exceptions raised inside the `with` block are timed/measured (elapsed_s and, if CUDA
    was live, peak_gib both still get set) and then always re-raised -- this context
    manager never swallows an exception from the wrapped code.
    """

    def __init__(self, device=None, label=None):
        self.device = device
        self.label = label
        self.peak_bytes = None
        self.peak_gib = None
        self.elapsed_s = None
        self._cuda = False
        self._torch = None

    def __enter__(self):
        try:
            import torch
        except ImportError:
            self._cuda = False
        else:
            self._torch = torch
            self._cuda = bool(torch.cuda.is_available())
            if self._cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_s = time.monotonic() - self._t0
        if self._cuda:
            self.peak_bytes = self._torch.cuda.max_memory_allocated(self.device)
            self.peak_gib = self.peak_bytes / GiB
        if self.label:
            self._report()
        return False   # never swallow an exception from the wrapped block

    def _report(self):
        if self.peak_gib is None:
            print(f"  [{self.label}] {self.elapsed_s:.1f}s "
                  f"(no CUDA device -- peak memory not measured)")
        else:
            print(f"  [{self.label}] peak GPU memory {self.peak_gib:.2f} GiB "
                  f"(max_memory_allocated, device={self.device!r})  {self.elapsed_s:.1f}s")


# --------------------------------------------------------------------------- #
# SELF-TEST -- runs on any laptop, no GPU required (spec-code.md builder contract).
#
# Two branches are exercised:
#   (a) the REAL no-torch/no-CUDA fallback -- this machine has no torch, so calling
#       peak_memory() plain exercises the actual ImportError path, not a simulation.
#   (b) a STUBBED torch.cuda -- injected into sys.modules so the reset/measure call
#       sequence and the bytes->GiB arithmetic are checked line-for-line against a known
#       byte count, without needing real hardware.
# --------------------------------------------------------------------------- #

def self_test():
    print("=" * 72)
    print("SELF-TEST (no GPU) -- utils/memory.py")
    print("=" * 72)

    # (a) real environment: no torch installed here at all.
    real_torch_present = "torch" in sys.modules or _importable("torch")
    with peak_memory(label="no-op block") as pm:
        total = sum(range(1000))
    assert total == 499500
    if not real_torch_present:
        assert pm.peak_bytes is None, "no torch installed -- peak_bytes must stay None"
        assert pm.peak_gib is None, "no torch installed -- peak_gib must stay None"
        print("  (a) no torch on this box: peak_gib is None, not 0.0 -- confirmed")
    else:
        print("  (a) skipped (a real torch IS importable here; see branch (b) for the "
              "arithmetic check regardless)")
    assert pm.elapsed_s is not None and pm.elapsed_s >= 0.0
    print(f"      elapsed_s = {pm.elapsed_s:.6f}s (wall-clock still measured with no GPU)")

    # (b) stubbed torch.cuda: known byte count in, exact GiB out.
    fake_bytes = 3 * GiB + 512 * (1 << 20)          # 3.5 GiB exactly
    calls = {"reset": 0, "max": 0, "reset_device": None, "max_device": None}

    fake_torch = _make_fake_torch(fake_bytes, calls)
    sys.modules_backup = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        with peak_memory(device=0, label=None) as pm2:
            pass
        assert calls["reset"] == 1, "reset_peak_memory_stats must be called exactly once"
        assert calls["max"] == 1, "max_memory_allocated must be called exactly once"
        assert calls["reset_device"] == 0 and calls["max_device"] == 0, (
            "device argument must be threaded through to both torch.cuda calls")
        assert pm2.peak_bytes == fake_bytes
        assert pm2.peak_gib == 3.5, f"3 GiB + 512 MiB must be exactly 3.5 GiB, got {pm2.peak_gib}"
        print(f"  (b) stubbed CUDA: {fake_bytes:,} B -> {pm2.peak_gib:.4f} GiB "
              f"(expected 3.5000) -- confirmed")
        print(f"      reset/measure call count: {calls['reset']}/{calls['max']} "
              f"-- confirmed exactly-once each")

        # exception propagation: must NOT be swallowed, and elapsed_s must still be set.
        raised = False
        try:
            with peak_memory(device=0, label=None) as pm3:
                raise ValueError("boom")
        except ValueError:
            raised = True
        assert raised, "peak_memory() must re-raise exceptions from the wrapped block"
        assert pm3.elapsed_s is not None and pm3.peak_gib == 3.5, (
            "measurement must still be recorded even when the block raised")
        print("  (c) exception inside `with` block propagates, and is still measured "
              "-- confirmed")

        # label path: exercise _report()'s two branches without crashing.
        with peak_memory(device=None, label="labelled block, no CUDA device arg") as pm4:
            pass
        print(f"      (printed line above is pm4._report() with peak_gib={pm4.peak_gib})")
    finally:
        if sys.modules_backup is not None:
            sys.modules["torch"] = sys.modules_backup
        else:
            sys.modules.pop("torch", None)
        del sys.modules_backup

    print()
    print("ALL CHECKS PASSED")


def _importable(name):
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def _make_fake_torch(fake_bytes, calls):
    """A minimal stand-in for the `torch` module exposing only what peak_memory() touches:
    torch.cuda.is_available() / reset_peak_memory_stats(device) / max_memory_allocated(device).
    Not a real CUDA simulation -- just enough surface to prove the call sequence and the
    bytes -> GiB arithmetic are wired correctly."""
    import types

    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: True

    def reset_peak_memory_stats(device=None):
        calls["reset"] += 1
        calls["reset_device"] = device

    def max_memory_allocated(device=None):
        calls["max"] += 1
        calls["max_device"] = device
        return fake_bytes

    cuda_mod.reset_peak_memory_stats = reset_peak_memory_stats
    cuda_mod.max_memory_allocated = max_memory_allocated

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = cuda_mod
    return torch_mod


if __name__ == "__main__":
    self_test()
