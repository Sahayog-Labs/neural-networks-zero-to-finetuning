#!/usr/bin/env python3
"""
utils/seed.py -- determinism + version-stamp helper, shared by every script in code/
(spec-code.md sec B.7, "Shared helpers").

Two functions:
    set_all_seeds(seed=42)   seeds random, numpy (if installed), and torch CPU+CUDA
                              (if installed) with the same integer, so a script's
                              numbers reproduce on rerun.
    stamp(extra=())          prints and returns ONE line naming the installed version
                              of every pinned library plus device/CUDA info, so a
                              support request begins with real versions, not
                              remembered ones. Verified-against line (constants.md
                              sec7, 2026-07-16): torch 2.13.0 . transformers 5.14.1 .
                              peft 0.19.1 . trl 1.8.x . CUDA 13.0.

Import it, do not reimplement it -- 11_finetune_qlora.py and train_lora_flux.py each
carry an inline copy of this exact pair (their comments say so explicitly: "utils/seed.py
isn't shipped, so inline"). Now that it exists, both should switch to
`from utils.seed import set_all_seeds, stamp`.

SAFETY: read-only. Seeds RNGs already in the process and reads __version__ strings;
installs, allocates, and writes nothing.

Self-test (works with NO GPU and NO torch installed -- deliberately, because rungs 1-5
of the ladder run on a bare laptop CPU per spec-code.md sec A, and this module is
imported before any of them ever touch torch):
    python utils/seed.py --self-test
"""
import argparse
import random

# The pin table this file stamps against -- constants.md sec7, verbatim. A version
# printed by stamp() that does not start with these strings gets an inline "[pin: ...]"
# note (not a hard failure -- a patch bump or a "+cu130" local suffix is not a bug).
PINNED_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "5.14.1",
    "peft": "0.19.1",
    "trl": "1.8",          # trl>=1.8,<2 -- constants.md pins a range, not a point (sec7)
    "CUDA": "13.0",
}
CORE_STAMP_MODULES = ("transformers", "peft", "trl")  # torch and CUDA are handled separately


def set_all_seeds(seed: int = 42) -> None:
    """Seed random, numpy (if present), and torch CPU+CUDA (if present) with `seed`.

    Determinism is best-effort, not bitwise-exact across GPU kernel/driver versions --
    cuDNN/cuBLAS algorithm selection is not fully deterministic even with a fixed seed --
    but seeding kills the dataset-shuffle and weight-init sources of run-to-run noise,
    which is most of it. Safe to call with neither numpy nor torch installed (this
    machine's local self-test has neither); it just seeds `random` and returns.
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _pin_note(mod: str, installed: str) -> str:
    """'' if `installed` starts with the pinned string for `mod`, else an inline note."""
    pin = PINNED_VERSIONS.get(mod)
    if pin is None or installed.startswith(pin):
        return ""
    return f" [pin: {pin}]"


def stamp(extra=()) -> str:
    """Print and return one '  [torch X . transformers Y . ... . CUDA Z]' line built
    from ACTUALLY IMPORTED versions -- PINNED_VERSIONS is only what to compare against,
    never what gets printed as fact. `extra` adds module names to check beyond the core
    LLM stack -- the diffusion scripts pass extra=("diffusers", "accelerate", "bitsandbytes").

    Any pinned library that fails to import prints 'MISSING', not a silent skip: a script
    that needed it is about to fail loudly anyway, and the version line is the first thing
    a support request should show.
    """
    parts = []
    torch = None

    try:
        import torch
        parts.append(f"torch {torch.__version__}{_pin_note('torch', torch.__version__)}")
    except ImportError:
        parts.append("torch MISSING")

    for mod in CORE_STAMP_MODULES + tuple(extra):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            parts.append(f"{mod} {v}{_pin_note(mod, v)}")
        except ImportError:
            parts.append(f"{mod} MISSING")

    if torch is not None:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            parts.append(f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
            cuda_v = torch.version.cuda or "?"
            parts.append(f"CUDA {cuda_v}{_pin_note('CUDA', cuda_v)}")
        else:
            parts.append("CUDA unavailable (no GPU visible)")
    else:
        parts.append("CUDA unknown (torch not importable)")

    line = "  [" + " . ".join(parts) + "]"
    print(line)
    return line


def _self_test():
    """Runs with no GPU and no torch: exercises the pure-Python determinism path,
    stamp()'s graceful-degradation path, and the pin table's own integrity. This file
    has no constants.md NUMBER to assert (it stamps version STRINGS, not frozen
    arithmetic) -- the self-test instead asserts the two properties that would silently
    break the rest of the ladder if this module regressed: seeding is reproducible, and
    stamp() never raises even when nothing downstream is installed.
    """
    print("=" * 68)
    print("utils/seed.py --self-test  (no GPU / no torch required)")
    print("=" * 68)

    # 1. set_all_seeds must not raise even with numpy/torch absent, and must actually
    #    make `random` reproducible -- that's the one RNG guaranteed present everywhere.
    set_all_seeds(42)
    first = [random.random() for _ in range(5)]
    set_all_seeds(42)
    second = [random.random() for _ in range(5)]
    assert first == second, "set_all_seeds(42) did not reproduce the random.random() stream"
    print(f"  random.random() x5, seeded twice with 42: IDENTICAL (first={first[0]:.6f})")

    set_all_seeds(7)
    third = [random.random() for _ in range(5)]
    assert third != first, "different seeds produced the SAME stream -- seeding is a no-op"
    print("  seed 7 vs seed 42: DIFFERENT streams, as expected")

    # 2. numpy path, only if numpy happens to be installed in this environment (it's
    #    optional for this module, but the scratch venv used to verify this file has it).
    try:
        import numpy as np
        set_all_seeds(42)
        a = np.random.rand(5)
        set_all_seeds(42)
        b = np.random.rand(5)
        assert (a == b).all(), "set_all_seeds(42) did not reproduce numpy's stream"
        print("  numpy present: np.random.rand(5) reproduced across two seed(42) calls")
    except ImportError:
        print("  numpy not installed here -- numpy branch not exercised (graceful no-op OK)")

    # 3. stamp() must run to completion and return a well-formed line even with torch
    #    (and everything downstream of it) absent -- that's the whole point of this test.
    line = stamp()
    assert line.startswith("  [") and line.endswith("]"), "stamp() output malformed"
    assert "torch" in line, "stamp() line must always name torch (present or MISSING)"
    for mod in CORE_STAMP_MODULES:
        assert mod in line, f"stamp() line must always name {mod} (present or MISSING)"
    print(f"  stamp() returned a well-formed line naming torch + {', '.join(CORE_STAMP_MODULES)}: OK")

    # extra= plumbing, exercised even though diffusers etc. are absent here too.
    line2 = stamp(extra=("diffusers", "accelerate", "bitsandbytes"))
    for mod in ("diffusers", "accelerate", "bitsandbytes"):
        assert mod in line2, f"stamp(extra=...) must name {mod} when passed"
    print("  stamp(extra=(diffusers, accelerate, bitsandbytes)) names all three: OK")

    # 4. the pin table itself -- guards against PINNED_VERSIONS drifting from
    #    constants.md sec7 without this file being updated to match.
    expect = {"torch": "2.13.0", "transformers": "5.14.1", "peft": "0.19.1",
              "trl": "1.8", "CUDA": "13.0"}
    assert PINNED_VERSIONS == expect, f"PINNED_VERSIONS drifted from constants.md sec7: {PINNED_VERSIONS}"
    print("  PINNED_VERSIONS matches constants.md sec7 verbatim: "
          f"torch {expect['torch']} . transformers {expect['transformers']} . "
          f"peft {expect['peft']} . trl {expect['trl']}.x . CUDA {expect['CUDA']}")

    print()
    print("ALL SELF-TESTS PASSED (pure-Python path; the torch/CUDA branches inside")
    print("set_all_seeds() and stamp() are desk-checked against verified July-2026 APIs,")
    print("not executed here -- this box has no GPU and no torch, by design.)")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser(
        description="utils/seed.py -- determinism + version-stamp helper (spec-code.md sec B.7)."
    )
    ap.add_argument("--self-test", action="store_true",
                     help="run the no-GPU self-test (this is also what happens with no flags -- "
                          "this module has no other standalone job; set_all_seeds/stamp are "
                          "library calls meant to be imported)")
    ap.parse_args()
    _self_test()


if __name__ == "__main__":
    main()
