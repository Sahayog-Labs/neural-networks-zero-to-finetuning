#!/usr/bin/env python3
"""
00_verify_env.py - the FIRST thing you run on the Spark. p.49 + the setup page.

Every support question about this course starts with a paste of this script's
output. It reads your machine and either prints a clean card - or fails loudly at
the exact line that is wrong, naming the trap. It asserts nothing it cannot read
off the device; it installs nothing and allocates no meaningful memory (it reads
`mem_get_info`, checks bf16 support, and imports bitsandbytes - a small CUDA
context init, nothing that saturates the GPU). It is SAFE to run while ComfyUI is
live: it contends with nothing.

What it proves, on THIS box (hardware-ground-truth §3, constants §6.9), and why
each line matters:

    torch 2.13.0 | cuda 13.0 | device NVIDIA GB10 | capability (12,1)
    arch_list ['sm_80','sm_90','sm_100','sm_110','sm_120']   <- NO sm_121, no PTX
    bf16 supported: True | bnb import: OK (native sm_121)
    transformers 5.14.1 | peft 0.19.1 | trl 1.8.x
    unified memory: 121.69 GiB total (mem_get_info)   [note: NOT 128 - see p.18/p.49]

The load-bearing, non-obvious facts it checks:

  1. capability == (12, 1). Your GPU is sm_121.
  2. sm_121 is NOT in torch's arch_list, AND torch runs anyway. This is the whole
     aarch64 story: the wheel ships a sm_120 cubin and NO PTX, and it runs on
     sm_121 by CUDA minor-version *binary* compatibility - not a JIT compile.
     (The "first kernel JITs the PTX" folklore is FALSE here. There is no PTX.)
  3. `import bitsandbytes` succeeds - because bitsandbytes ships a NATIVE sm_121
     build for ARM64+CUDA-13 (x86 does not get it). Your QLoRA path depends on it.
  4. THE trap: any package pinning `libcudart.so.12` dies at *import* on the
     CUDA-13-only DGX OS, before kernel compatibility is even evaluated. If that
     is what is wrong, this script says so by name and exits non-zero.
  5. mem_get_info total ~= 121.69 GiB, NOT 128. The 6.31 GiB carveout is real and
     it is why the p.18/p.49 full-FT ledger misses by a hair.

This script TARGETS the Spark (needs torch+CUDA). This Windows dev box has neither,
so:
  * a normal run here prints how to run it on the Spark (in the FRESH venv, never
    ComfyUI's) and stops cleanly;
  * `--self-test` runs the pure decision logic - the arch-list rule, the memory
    tolerance arithmetic, the libcudart-trap classifier, the capability/version
    checks - against frozen constants, WITHOUT a GPU. Use it to desk-check the
    logic anywhere.

Usage
-----
    python 00_verify_env.py             # the real probe (Spark, fresh venv)
    python 00_verify_env.py --self-test # pure logic checks, no GPU, runs anywhere
"""

import argparse
import sys

# --------------------------------------------------------------------------- #
# GiB/GB discipline (constants §0). Capacity is GiB; weights alone are GB.
# --------------------------------------------------------------------------- #
GiB = 1 << 30
GB = 10 ** 9

# --------------------------------------------------------------------------- #
# Frozen ground truth - every number here is measured/verified, not remembered.
#   capability, arch_list, memory : hardware-ground-truth §1-3 [MEA-DEV 2026-07-16]
#   version pins                  : constants §7 [VP, PyPI 2026-07-16]
#   bitsandbytes native sm_121 /  : constants §6.9 [VP]
#   the libcudart.so.12 trap
# --------------------------------------------------------------------------- #
EXPECTED_CAPABILITY = (12, 1)                 # sm_121, read off the device
# torch's release wheel ships these cubins and NO PTX; sm_121 is absent on purpose.
EXPECTED_ARCH_LIST = ["sm_80", "sm_90", "sm_100", "sm_110", "sm_120"]

PHYSICAL_GIB = 128.0                           # "128 GB LPDDR5X" = 128 GiB of parts
MEM_TOTAL_BYTES = 130_662_936_576              # /proc/meminfo MemTotal x 1024, exact
# Derive GiB from the byte PRIMITIVE, not from a rounded label. The byte value
# (130,662,936,576) is the raw reading; dividing by 2^30 gives 121.6893 GiB, which
# the course reports to 2 d.p. as 121.69 (the mem_get_info card, spec-code D.0).
# NOTE (spec ambiguity resolved): hardware-ground-truth §2 also writes "121.6875
# GiB EXACTLY" - that is a rounded-to-1/16 display slip; 121.6875 != the byte value
# by ~2 MB. The byte primitive wins; 121.69 is the number every page shows.
MEM_TOTAL_GIB = MEM_TOTAL_BYTES / GiB          # = 121.6893...
MEM_TOTAL_DISPLAY = 121.69                      # the 2-d.p. figure the pages print
CARVEOUT_BYTES = 137_438_953_472 - MEM_TOTAL_BYTES  # 128 GiB - MemTotal = 6,776,016,896
CARVEOUT_GIB = CARVEOUT_BYTES / GiB            # = 6.3107... (not 6.3125)
MEM_TOLERANCE_PCT = 1.0                        # warn (not fail) outside +/-1%

# Recommended pins for the FRESH llm-ft venv (constants §7). These are targets, not
# assertions: ComfyUI's venv legitimately runs torch 2.12.1 / transformers 5.12.1
# (hardware-ground-truth §3), so we PRINT versions and soft-note drift - we never
# fail the env check on a minor-version difference.
RECOMMENDED = {
    "torch": "2.13.0",
    "transformers": "5.14.1",
    "peft": "0.19.1",
    "trl": "1.8.0",
    "bitsandbytes": "0.49.2",
    "diffusers": "0.39.0",
}
CUDA_MAJOR_EXPECTED = 13                        # anything cu12x is the libcudart trap


# --------------------------------------------------------------------------- #
# Pure decision logic - SHARED by the real probe and --self-test, so the two can
# never drift (the same §6.2-JS discipline the pages use).
# --------------------------------------------------------------------------- #

def arch_note(capability, arch_list):
    """The heart of the aarch64 teaching point.

    Returns (runs_via_binary_compat, message). If the device's own sm string is
    absent from the wheel's arch_list, torch runs it anyway via CUDA minor-version
    BINARY compatibility (a sm_x0 cubin on a sm_xN device, same major x) - NOT a
    PTX JIT, because release wheels ship no PTX. If the exact sm string IS present,
    there is a native cubin and no compatibility story to tell.
    """
    sm = f"sm_{capability[0]}{capability[1]}"
    if sm in arch_list:
        return False, (f"{sm} IS in the arch_list - a native cubin exists for your "
                       f"GPU; the binary-compat story does not apply.")
    fallback = f"sm_{capability[0]}0"          # same major, minor 0
    return True, (
        f"{sm} is NOT in the arch_list and there is no PTX, yet torch runs.\n"
        f"     It executes a {fallback} cubin on your {sm} device by CUDA "
        f"minor-version BINARY compatibility (shared major {capability[0]}).\n"
        f"     This is NOT a first-kernel PTX JIT - release wheels ship no PTX. "
        f"A stray TORCH_CUDA_ARCH_LIST or a source build can break what works.")


def within_tolerance(measured_gib, expected_gib, pct):
    """True if measured is within +/-pct% of expected. Used to WARN (not fail) if
    the learner's carveout differs from the reference 121.6875 GiB."""
    return abs(measured_gib - expected_gib) <= (pct / 100.0) * expected_gib


def carveout_gib(physical_gib, total_gib):
    """The GiB reserved by firmware/driver before you allocate a byte."""
    return physical_gib - total_gib


def classify_import_error(msg):
    """Classify a failed `import bitsandbytes` (or any CUDA pkg) so the script can
    name the trap instead of dumping a raw traceback. Returns one of:
        'libcudart12' - THE trap: a CUDA-12 pin dying on the CUDA-13-only DGX OS
        'missing'     - the package simply is not installed (fresh-venv reminder)
        'other'       - something else; surface it verbatim
    """
    low = msg.lower()
    if "libcudart.so.12" in low or "libcudart.so.12" in msg:
        return "libcudart12"
    # a generic cu12 shared-object miss is the same trap wearing a different name
    if "libcudart" in low and ".so.12" in low:
        return "libcudart12"
    if "no module named" in low:
        return "missing"
    return "other"


def cuda_major(cuda_version_str):
    """Major from a torch.version.cuda string like '13.0' -> 13. None if unknown."""
    if not cuda_version_str:
        return None
    try:
        return int(str(cuda_version_str).split(".")[0])
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------- #
# Version stamp helper - prints installed vs recommended, soft-notes drift.
# --------------------------------------------------------------------------- #

def _probe_version(module_name):
    """Return the installed __version__ string, or None if not importable."""
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def print_version_stamp():
    """§B rule 7: a support request begins with a paste of REAL versions."""
    print("-" * 72)
    print("VERSIONS - installed vs the constants §7 pin (drift is a note, not a fail)")
    print("-" * 72)
    all_present = True
    for name, want in RECOMMENDED.items():
        got = _probe_version(name)
        if got is None:
            all_present = False
            note = "NOT INSTALLED"
            if name in ("peft", "trl", "bitsandbytes"):
                note += " - LLM stack, install into the FRESH venv (never ComfyUI's)"
            print(f"  {name:<14} {'-':<12}  (want {want})  <- {note}")
        else:
            flag = "" if got == want else f"  <- pinned {want}; you have {got}"
            print(f"  {name:<14} {got:<12}  (want {want}){flag}")
    print()
    return all_present


# --------------------------------------------------------------------------- #
# The real probe - needs torch + CUDA. Runs on the Spark, in the fresh venv.
# --------------------------------------------------------------------------- #

def check_env():
    import torch

    print("=" * 72)
    print("00_verify_env - reading YOUR machine (the first thing you run)")
    print("=" * 72)

    # --- device identity -------------------------------------------------- #
    if not torch.cuda.is_available():
        print(f"  torch {torch.__version__} imported, but torch.cuda.is_available()")
        print("  is False - no GPU here. On the Spark that means the wrong venv or a")
        print("  broken install; on a CPU-only dev box it just means this isn't the")
        print("  Spark. Either way, the device facts below cannot be read. Stopping.")
        print("  To desk-check the decision logic anywhere: python 00_verify_env.py --self-test")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)          # -> a tuple (12, 1)
    arch_list = torch.cuda.get_arch_list()
    cuda_ver = torch.version.cuda
    print(f"  torch        {torch.__version__}")
    print(f"  cuda (torch) {cuda_ver}")
    print(f"  device       {name}")
    print(f"  capability   sm_{cap[0]}{cap[1]}   {tuple(cap)}")
    print(f"  arch_list    {arch_list}")
    print()

    # --- assert 1: capability == (12, 1) ---------------------------------- #
    assert tuple(cap) == EXPECTED_CAPABILITY, (
        f"device capability {tuple(cap)} != {EXPECTED_CAPABILITY} (sm_121). This is "
        f"not the GB10 the course is built against (hardware-ground-truth §1).")
    print(f"  [ok] capability is {EXPECTED_CAPABILITY} = sm_121.")

    # --- assert 2: sm_121 NOT in arch_list, AND torch runs (it did: we're here) #
    sm = f"sm_{cap[0]}{cap[1]}"
    runs_compat, note = arch_note(tuple(cap), arch_list)
    assert sm not in arch_list, (
        f"{sm} unexpectedly IS in arch_list {arch_list}. The course's whole "
        f"binary-compat teaching point (constants §6.9) assumes it is absent; your "
        f"wheel differs - investigate before trusting the rest.")
    print(f"  [ok] {sm} is absent from arch_list - and torch imported and saw the")
    print(f"       device, so it RAN anyway. That is the lesson:")
    print(f"       {note}")
    print()

    # --- assert 3: bf16 supported ----------------------------------------- #
    assert torch.cuda.is_bf16_supported(), (
        "bf16 is not supported on this device - the entire mixed-precision "
        "training path (p.49) assumes it. Something is very wrong.")
    print(f"  [ok] bf16 supported: True.")

    # --- assert 4: bitsandbytes imports (native sm_121), catch the libcudart trap #
    _check_bitsandbytes()

    # --- version stamp ---------------------------------------------------- #
    print()
    print_version_stamp()

    # cu12 pin sanity: torch itself must not be a CUDA-12 build on this OS
    maj = cuda_major(cuda_ver)
    if maj is not None and maj < CUDA_MAJOR_EXPECTED:
        print(f"  !! torch reports CUDA {cuda_ver} (major {maj}). The DGX OS is "
              f"CUDA-13-only; a cu12x build is the libcudart trap waiting to happen.")
        print(f"     Reinstall the cu130 wheel (constants §6.9).")
        print()

    # --- memory: the 121.69-not-128 beat, WARN on carveout drift ---------- #
    _check_memory(torch)

    print("=" * 72)
    print("Environment verified. Every assertion above read a fact off YOUR box.")
    print("Next: measure_your_box.py, then 05_memory_ledger.py (p.49).")
    print("=" * 72)


def _check_bitsandbytes():
    """Import bitsandbytes; on failure, NAME the trap (constants §6.9) and exit."""
    try:
        import bitsandbytes as bnb
        ver = getattr(bnb, "__version__", "unknown")
        print(f"  [ok] import bitsandbytes: OK (v{ver}) - native sm_121 build for")
        print(f"       ARM64+CUDA-13 (x86 does NOT get it). Your QLoRA path is safe.")
    except (ImportError, OSError) as e:
        kind = classify_import_error(str(e))
        if kind == "libcudart12":
            print()
            print("  " + "!" * 66)
            print("  THE libcudart.so.12 TRAP (constants §6.9).")
            print("  A package pinning libcudart.so.12 fails at IMPORT on the Spark's")
            print("  CUDA-13-only DGX OS - before kernel compatibility is even")
            print("  evaluated. This is the narrow, sharp trap, not 'aarch64 is hard'.")
            print("  Fix: install the CUDA-13 (cu130) build; never a cu12x wheel.")
            print(f"  raw error: {e}")
            print("  " + "!" * 66)
            sys.exit(2)
        elif kind == "missing":
            print("  !! bitsandbytes is NOT installed. It is not in ComfyUI's venv and")
            print("     not system-wide (hardware-ground-truth §3). Install it into the")
            print("     FRESH venv - never ComfyUI's:")
            print('       uv pip install "bitsandbytes==0.49.2"   # constants §7')
            print("     Your QLoRA path (p.50) cannot run until this imports.")
            sys.exit(3)
        else:
            print(f"  !! import bitsandbytes failed for an unexpected reason: {e}")
            sys.exit(4)


def _check_memory(torch):
    print()
    print("-" * 72)
    print("UNIFIED MEMORY - what you have, not what it was sold as")
    print("-" * 72)
    free, total = torch.cuda.mem_get_info()
    total_gib = total / GiB
    print(f"  mem_get_info total  {total:>18,} B")
    print(f"                      {total / GB:>18.2f} GB  (decimal - marketing)")
    print(f"                      {total_gib:>18.4f} GiB (binary  - what DRAM is)")
    print(f"  mem_get_info free   {free / GiB:>18.4f} GiB (right now, this instant)")
    print()
    print(f"  Sold as   {PHYSICAL_GIB:>10.4f} GiB physical")
    print(f"  You get   {total_gib:>10.4f} GiB  <- note: NOT 128 (see p.18/p.49)")
    print(f"  Carveout  {carveout_gib(PHYSICAL_GIB, total_gib):>10.4f} GiB gone to "
          f"firmware/driver before you start")
    print()

    if within_tolerance(total_gib, MEM_TOTAL_GIB, MEM_TOLERANCE_PCT):
        print(f"  [ok] total is within {MEM_TOLERANCE_PCT:.0f}% of the reference "
              f"{MEM_TOTAL_DISPLAY:.2f} GiB (hardware-ground-truth §2).")
    else:
        # WARN, do not fail - his carveout may legitimately differ.
        print(f"  !! total {total_gib:.4f} GiB differs from the reference "
              f"{MEM_TOTAL_DISPLAY:.2f} GiB by >{MEM_TOLERANCE_PCT:.0f}%.")
        print(f"     Not a failure - your firmware carveout simply differs. But the")
        print(f"     p.18/p.49 ledger numbers are keyed to 121.69 GiB; recompute the")
        print(f"     fit margin against YOUR total.")
    print()


# --------------------------------------------------------------------------- #
# --self-test - pure logic, no GPU. Exercises the SAME helpers the probe uses,
# against frozen constants and adversarial synthetic inputs. Runs anywhere.
# --------------------------------------------------------------------------- #

def self_test():
    print("=" * 72)
    print("00_verify_env --self-test : pure decision logic, NO GPU (constants §6.9)")
    print("=" * 72)
    failures = 0

    def check(label, cond, detail=""):
        nonlocal failures
        mark = "ok  " if cond else "FAIL"
        if not cond:
            failures += 1
        print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
        return cond

    # -- capability tuple comparison (assert 1's logic) -------------------- #
    print("\n  capability check:")
    check("(12,1) == expected sm_121", (12, 1) == EXPECTED_CAPABILITY)
    check("(12,0) is correctly rejected", (12, 0) != EXPECTED_CAPABILITY,
          "a sm_120 device would fail assert 1")

    # -- arch_note: the binary-compat teaching point (assert 2's logic) ---- #
    print("\n  arch_list / binary-compat rule:")
    runs, msg = arch_note((12, 1), EXPECTED_ARCH_LIST)
    check("sm_121 absent from frozen arch_list -> runs via binary compat", runs)
    check("  and the message says binary, not PTX",
          "BINARY compatibility" in msg and "no PTX" in msg)
    # flip it: a wheel that DID ship sm_121 must NOT tell the compat story
    runs2, msg2 = arch_note((12, 1), EXPECTED_ARCH_LIST + ["sm_121"])
    check("sm_121 present -> native cubin, no compat story", not runs2)

    # -- memory arithmetic (assert-free WARN path) ------------------------- #
    print("\n  memory arithmetic (from the byte primitive, not a rounded label):")
    check("MemTotal 130,662,936,576 B -> 121.6893 GiB (pages show 121.69)",
          abs(MEM_TOTAL_GIB - 121.6893) < 1e-4, f"got {MEM_TOTAL_GIB:.4f}")
    check("that rounds to the 121.69 the mem_get_info card prints",
          round(MEM_TOTAL_GIB, 2) == MEM_TOTAL_DISPLAY, f"got {MEM_TOTAL_GIB:.2f}")
    carve = carveout_gib(PHYSICAL_GIB, MEM_TOTAL_GIB)
    check("carveout 128 - 121.6893 -> 6.3107 GiB",
          abs(carve - 6.3107) < 1e-4, f"got {carve:.4f}")
    check("carveout matches the 6,776,016,896 B on record",
          abs(carve - CARVEOUT_GIB) < 1e-9, f"got {carve:.4f} vs {CARVEOUT_GIB:.4f}")
    check("the measured total is within 1% of its own reference -> no warn",
          within_tolerance(MEM_TOTAL_GIB, MEM_TOTAL_GIB, MEM_TOLERANCE_PCT))
    check("110.0 GiB is OUTSIDE 1% -> would warn",
          not within_tolerance(110.0, MEM_TOTAL_GIB, MEM_TOLERANCE_PCT))

    # -- the libcudart.so.12 trap classifier (assert 4's logic) ------------ #
    print("\n  libcudart.so.12 trap classifier:")
    trap_msg = "libcudart.so.12: cannot open shared object file: No such file or directory"
    check("a libcudart.so.12 error is caught as the trap",
          classify_import_error(trap_msg) == "libcudart12")
    check("a plain 'No module named bitsandbytes' -> 'missing'",
          classify_import_error("No module named 'bitsandbytes'") == "missing")
    check("an unrelated error -> 'other'",
          classify_import_error("some other failure") == "other")

    # -- cuda major parsing (the cu12-pin sanity note) --------------------- #
    print("\n  cuda-version parsing:")
    check("'13.0' -> major 13 (expected)", cuda_major("13.0") == CUDA_MAJOR_EXPECTED)
    check("'12.4' -> major 12 (would warn: cu12 on a CUDA-13 OS)",
          cuda_major("12.4") == 12)
    check("None cuda string -> None (handled, no crash)", cuda_major(None) is None)

    print()
    print("=" * 72)
    if failures == 0:
        print("--self-test PASSED: every decision the real probe makes is correct")
        print("against the frozen constants. Run WITHOUT --self-test on the Spark")
        print("(in the fresh venv) to read your actual device.")
        print("=" * 72)
        return 0
    print(f"--self-test FAILED: {failures} logic check(s) wrong. Fix before shipping.")
    print("=" * 72)
    return 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the pure decision logic against frozen constants, no GPU")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    # The real probe. Needs torch; this dev box (Windows, no torch/CUDA) will land
    # in the except and be told how to run on the Spark.
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch is not importable here.")
        print()
        print("This script TARGETS the Spark (GB10, sm_121, CUDA 13). Run it there,")
        print("in the FRESH llm-ft venv - never ComfyUI's:")
        print("    source ~/llm-ft/bin/activate")
        print("    python 00_verify_env.py")
        print()
        print("To desk-check the decision LOGIC anywhere without a GPU:")
        print("    python 00_verify_env.py --self-test")
        sys.exit(1)

    check_env()


if __name__ == "__main__":
    main()
