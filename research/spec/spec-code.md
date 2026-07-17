# BUILD SPECIFICATION — THE CODE LADDER · every artifact in `ANN-Course/code/`

**Author of record:** code-ladder spec writer. **Frozen-file authority order obeyed:**
`hardware-ground-truth.md` > `constants.md` > `decisions.md`/`notation.md` > briefs.
Primary source: `brief-tooling-hardware.md` §3 (the 10-rung ladder) + §14 (file list); secondary:
`brief-llm-finetuning.md` and `brief-diffusion.md` artifact lists, as instantiated by the already-written
page specs `spec-part2.md` (trunk 12–24), `spec-part4.md` (adaptation 38–43), `spec-part5.md` (LLM 44–52),
`spec-part6.md` (diffusion 53–62 + capstone).

**What this document is.** The single authoritative catalog of every runnable file the course ships in `code/`.
Each page spec names its `.box try` artifact by filename; **this file is where those filenames are reconciled,
their I/O and runtime fixed, and their self-check assertions frozen.** A builder implements any one script 1:1
from its entry here without opening a brief. `measure_your_box.py` **already exists** in `code/` and is the
reference implementation — every other script adopts its conventions (§B).

**The prime directive (constants.md preamble).** *Never launder an estimate into a fact.* Every script that
prints a number the course treats as [INF]/[EST]/[MEA] prints the confidence label too, and every script that
prints a [DER]/[VP] fact **asserts it against `constants.md` before printing it**. A script that measures
(memory, TFLOP/s, bandwidth, tok/s, retention) prints the method and the learner's own number — never a frozen
stand-in. The two anti-patterns this build fails most (notation §9 #22/#23) are a number contradicting
`constants.md` and a [INF]/[EST]/[MEA] number printed as fact. This learner owns the box and will check.

---

## §A — THE ENVIRONMENT · ONE recommended path

**The reality on his box (`hardware-ground-truth.md` §3, [MEA-DEV] 2026-07-16 — outranks the briefs):**
- GB10, **sm_121 (12.1)**, aarch64, DGX OS (Ubuntu 24.04.4), driver 580.159.03, **CUDA 13.0**, Python 3.12.3.
- **There is NO system PyTorch.** `python3 -c "import torch"` → `ModuleNotFoundError`. Torch exists only inside
  `~/ComfyUI/.venv/` (torch **2.12.1+cu130**, transformers 5.12.1, diffusers 0.39.0).
- **peft / trl / accelerate / bitsandbytes are NOT installed anywhere.** The LLM stack does not exist on the box.
- 2.5 TB free on NVMe; 435 GB HF cache; ~270 GB ComfyUI models incl. **12 GB of his own LoRAs (9 files)**.

**THE recommended path — a fresh venv, never ComfyUI's (`hardware-ground-truth.md` §3 mandate, D-21 setup page).**
The setup page (Part III, before the fork) ships this and page 24 / all Part IV–V `.box try` scripts assume it:

```bash
# 1. HF cache on the NVMe FIRST (it will hit hundreds of GB). constants §7 / tooling §9.3
export HF_HOME=/mnt/nvme/hf                       # or ~/hf on the 3.7 TB disk
# 2. a FRESH venv — do NOT pip-install into ~/ComfyUI/.venv (you will break a working cu130 stack)
uv venv --python 3.12 ~/course/.venv && source ~/course/.venv/bin/activate
# 3. torch: the PyPI aarch64 cu130 wheel RUNS on GB10 via sm_120→sm_121 BINARY compat (no PTX). constants §6.9
uv pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130
# 4. the HF layer, all pinned (constants §7 — trl is 1.8, NOT the briefs' stale 0.29.1)
uv pip install "transformers==5.14.1" "peft==0.19.1" "trl>=1.8,<2" \
               "bitsandbytes==0.49.2" "diffusers==0.39.0" "accelerate" "datasets"
```

**Four hard rules for this exact machine — each is a `.box warn` on the setup page:**
1. **Never `pip install torch` into ComfyUI's venv, and never set `TORCH_CUDA_ARCH_LIST`.** The stock wheel ships
   **SASS-only, no PTX** (`get_arch_list()` → `['sm_80','sm_90','sm_100','sm_110','sm_120']`, no sm_121); it runs
   on a compatibility guarantee, and a stray arch-list env var or a source build breaks what currently works
   (`hardware-ground-truth.md` §3.1). **The "first-kernel-JIT / PTX-compile" story is FALSE here — delete it** (D-10h).
2. **The real import-time trap is a `libcudart.so.12` pin.** Any package built against CUDA 12 dies at *import* on
   the CUDA-13-only DGX OS, *before* kernel compatibility is even evaluated (constants §6.9). This is narrower and
   sharper than "aarch64 is painful."
3. **bitsandbytes ships NATIVE sm_121** into the aarch64+CUDA-13 wheel (`build_capability="…;120;121"`); x86 does
   not get it (constants §6.9, D-10i). The 70B-QLoRA and QLoRA paths depend on it. The Spark is *better* served
   than an x86 box here — say so.
4. **vLLM is NOT a pip install.** The PyPI aarch64 vLLM wheel is CUDA-12 and fails at import. The validated path is
   the **`vllm/vllm-openai:cu130-nightly` container, pinned by digest** (constants §6.9, D-10j). Only two scripts
   need it: `15_grpo_rlvr.py` (generation-in-the-loop) and the optional throughput path of `14_merge_and_serve.py`.
   Default local serving is **llama.cpp / Ollama** (D-10 serving ruling).

**Rungs 1–5 need no Spark.** `01_backprop_numpy`-class scripts through `05_lora_from_scratch` run on a laptop CPU
(tooling §9.2c). Say so loudly; a learner away from his box is not blocked on the trunk.

**`Dockerfile` (shipped, optional):** `FROM nvcr.io/nvidia/pytorch:<cu130-tag>` + the HF layer above via
`uv pip install --system`, **without** re-installing torch (that would replace NVIDIA's tuned aarch64 build —
the single highest-value warning on the setup page, tooling §9.1). Verify the current NGC tag at build time; the
principle (match the container's CUDA to the host's 13.0) is what is stable.

---

## §B — SELF-CHECK DISCIPLINE · the pattern every script obeys (from `measure_your_box.py`)

`measure_your_box.py` already sets these conventions; every other script matches them:

1. **Self-narrating output.** Print a banner, the inputs, the computation, the verdict — the terminal transcript
   should read like the page. No bare tensors.
2. **GiB/GB discipline (constants §0).** `GiB = 1<<30`, `GB = 10**9`. Anything compared to hardware capacity is
   **GiB**; weights/state quoted alone are **GB**. Print the unit every time; never mix in one comparison.
3. **Assert the frozen arithmetic, THEN measure.** Every script that reproduces a `constants.md` [DER]/[VP] value
   asserts it (`assert abs(got - FROZEN) < tol`) so a regression fails loudly. Every [MEA] value is measured on
   his box and printed with the method — never a stand-in. The ledger script asserts **122.05 GiB** (the derived
   need) then measures his **121.6875 GiB** (`torch.cuda.mem_get_info()` / `/proc/meminfo`).
4. **Confidence tags in the output.** When a printed number is [INF]/[EST]/[MEA-DEV], the print statement says so
   (e.g. `"ridge I* = 227 FLOP/byte [INFERRED, not published — see constants §6.4]"`).
5. **SAFETY line** in the module docstring for anything that allocates GPU memory or saturates the device: state
   that it contends with ComfyUI if it is up, and that it writes/installs nothing (read-only w.r.t. the system)
   unless the script's whole job is to train (then name the output dir and its size).
6. **No elided imports, no `# ...`** (notation §9 #18). Scripts run as-is.
7. **Seeded** (`utils/seed.py`) and **version-stamped** (print `torch/transformers/peft/trl/CUDA` at the top) so a
   support request begins with a paste of real versions.

**Shared helpers (imported, not attached to a page):**
- `code/utils/seed.py` — `set_all_seeds(42)` (torch/np/random) + `stamp()` printing the verified-against version
  line: `torch 2.13.0 · transformers 5.14.1 · peft 0.19.1 · trl 1.8.x · CUDA 13.0`.
- `code/utils/memory.py` — a `peak_memory()` context manager wrapping `torch.cuda.reset_peak_memory_stats()` /
  `max_memory_allocated()`; returns GiB. Used by every training/inference script.
- `code/utils/ledger.py` — the pure-arithmetic memory model (`bytes_per_param`, `lora_params`, `kv_bytes`,
  `full_ft_state`) shared verbatim by `05_memory_ledger.py` (CLI) and asserted by the trunk page-18 companion, so
  page and script can never drift (mirrors the widget/§6.2-JS discipline).

---

## §C — THE CANONICAL LADDER · concept → shipped filename → page

The task's 11-rung ladder + the capability probe, mapped to the files the page specs actually attach. **All
filenames on disk are distinct**; the numeric prefix means *page number* in the trunk companions (12–24) and
*ladder rung* in the LLM track — see §F for the reconciliation policy.

| # | Rung (task) | Shipped file(s) | Page (D-21) | Anchor |
|---|---|---|---|---|
| 1 | numpy by hand | `14_backprop_tn1.py` | 14 | TN-1 to the digit |
| 2 | torch matching TN-1 | `15_autograd_check.py` | 15 | TN-1 autograd, float-reality |
| 3 | MNIST | `03_mlp_mnist.py` (**gap — see §F**) + `ddpm_mnist.py` | Part I/III · 58 | 101,770-param MLP; U-Net |
| 4 | tokenizer | `01_tokenizer_lab.ipynb` | 44 | Qwen3 tokenizer |
| 5 | LoRA from scratch | `05_lora_from_scratch.py` (Qwen3) · `lora_from_scratch.py` (FLUX.2) | 40 · 62 | 43,646,976 params |
| 6 | peft SFT (Qwen3-8B) | `11_finetune_qlora.py` + `10_build_dataset.py` | 50 · 48 | 8B LoRA, ~12 min |
| 7 | GRPO (Qwen3-0.6B) | `15_grpo_rlvr.py` | 52 | vLLM cu130 container |
| 8 | diffusion LoRA (his disk) | `train_lora_flux.py` + `caption_ab.py` | 62 | 20 of his images |
| 9 | memory CLI | `05_memory_ledger.py` (+ `measure_your_box.py`) | 49 · 18 | 122.05 GiB assert |
| 10 | throughput probe | `09_spark_capability_probe.py` · `02_kv_cache_and_roofline.py` · `bench_spark.py` | 51 · 46 · 60 | 62-vs-125 TF |
| 11 | merge / serve | `14_merge_and_serve.py` | 51 | merge into **bf16** base |

**Plus** the environment/probe backbone: `00_verify_env.py` (49/setup), `measure_your_box.py` (18/43, **exists**),
and the trunk page-companions and diffusion teaching scripts catalogued in full below.

---

## §D — FULL ARTIFACT SPECS

Format per entry: **filename · page(D-21) · purpose · inputs → outputs · runtime on his box · self-checks.**
Runtimes are for the DGX Spark GB10; the trunk companions also run on a laptop CPU in seconds.

### D.0 — Environment & probe backbone

---
**`code/00_verify_env.py`** · pages: setup (Part III) + 49 · build **S**
**Purpose:** the first thing he runs; every support question starts with its output. Prints and *asserts* the stack.
**Inputs → outputs:** none → a stamped card:
```
torch 2.13.0 | cuda 13.0 | device NVIDIA GB10 | capability (12,1) | arch_list ['sm_80'..'sm_120']
bf16 supported: True | bnb import: OK (native sm_121) | transformers 5.14.1 | peft 0.19.1 | trl 1.8.x
unified memory: 121.69 GiB total (mem_get_info)   [note: NOT 128 — see p.18/p.49]
```
**Runtime:** ~3 s.
**Self-checks:** `assert torch.cuda.get_device_capability(0) == (12,1)`; `assert torch.cuda.is_bf16_supported()`;
`assert "sm_121" not in torch.cuda.get_arch_list()` **and it still runs** (the binary-compat teaching point,
`hardware-ground-truth.md` §3.1 — print the callout); `import bitsandbytes` must succeed (native sm_121, constants
§6.9); warn (not fail) if `mem_get_info()` total is not within 1% of **121.69 GiB** ("your carveout differs").
Fails loudly if any package pins `libcudart.so.12` (catch the ImportError, name the trap, constants §6.9).

---
**`code/measure_your_box.py`** · pages: 18 (`--no-gpu`), 43 (centerpiece) · build **O** · **ALREADY EXISTS — do not rewrite; spec the pages around it.**
**Purpose:** find HIS roofline; refuse the marketing number. Part 1 = memory facts (page-18 ledger reality check);
Part 2 = dense bf16 matmul TFLOP/s + achieved bandwidth + ridge (page-43 lab).
**Inputs → outputs:** flags `--quick`, `--no-gpu` → memory card (MemTotal/MemAvailable/carveout, the 122.05-GiB
fit question) and, with GPU, the measured `peak TFLOP/s`, `achieved GB/s`, and `ridge I* = FLOP/byte`.
**Runtime:** `--no-gpu` <1 s; full sweep ~20–40 s (`--quick` ~20 s).
**Self-checks (as shipped):** asserts nothing it can't measure; hard-codes the frozen need **122.05 GiB** and the
frozen physical **128 GiB** to compute the carveout and the −0.36 GiB miss; prints the marketing 1000 TFLOP/s
(FP4-sparse) beside the measured dense bf16 and the ratio; states the achieved-bandwidth fraction as the honest
decode coefficient `k` (batch-1 only). **Page-18 seam:** page 18 runs it `--no-gpu` and must NOT assert the ~62 TF
ceiling or the ridge — those are page 43's [INF] predict-then-measure (spec-part2 appendix).

---
**`code/09_spark_capability_probe.py`** · page 51 (also referenced by page 43's deepdive) · build **O**
**Purpose:** **settle the 62-vs-125 TF question by measurement** (constants §6.3, §10 exercise #2 — "the corpus's
biggest gap, its best lab"). A one-page card of what his box actually does at BF16.
**Inputs → outputs:** `--sizes 4096,8192,12288,16384` → a table of a big BF16 GEMM run in **both accumulate modes**
(FP16-accumulate via default cuBLAS path vs FP32-accumulate via `torch.backends.cuda.matmul.allow_tf32`/highest-
precision reduction), plus prefill-vs-decode micro-benchmarks and achieved bandwidth. Prints which ceiling his
box lands at and the resulting ridge.
**Runtime:** ~30–60 s.
**Self-checks / framing:** print **both** inferred ceilings — **~62.5 TF FP32-accumulate** and **~125 TF FP16-
accumulate** — tagged **[INF, not published]**, and report which his measurement matches. Print the two candidate
ridges **227** (if ~62 TF) and **458** (if ~125 TF), tagged [INF]. Do **not** assert a winner in code comments —
the measurement is the answer (constants §6.3 flags this genuinely unresolved; the SDXL datapoint leans 125 TF,
community GEMM reports lean 62 TF). Cross-check the per-SM arithmetic in a comment: `48 SMs × 512 FP32-acc
FLOP/SM/clk × 2.6 GHz ≈ 64 TF`; `×1024 (FP16-acc) ≈ 128 TF` (constants §6.3). Warn if ComfyUI is up (contention
will pull the number down and read as a false "62 TF" result).

---

### D.1 — Trunk Part II companions (pages 12–24)

These are **page-companion live-math scripts**, fully specified inline in `spec-part2.md`; reproduced here as the
authoritative `code/` catalog with the 5 required fields. Each runs in seconds on CPU. Prefix = page number.

---
**`code/12_cross_entropy.py`** · p.12 · reproduce the frozen canonical logits.
Inputs → outputs: none → prints for `z=[2.0,1.0,0.1]`, true class 0: `ŷ=[0.659001,0.242433,0.098566]`,
`L=0.417030` nats, `∂L/∂z=[-0.340999,0.242433,0.098566]`. Runtime <1 s.
Self-checks: `assert` each against constants §9.2 to 6 d.p.; `assert abs(sum(∂L/∂z)) < 1e-9` ("sums to zero,
always"); also verify random-init `L = ln(151936) = 11.93` and `PPL = V` (§9.1).

---
**`code/13_lr_sweep.py`** · p.13 · 1-D quadratic, sweep `η∈{0.05,0.1,0.15}`.
Inputs → outputs: none → three trajectories; shows divergence at `η_crit = 2/λ_max = 0.1` for the ravine
`λ_max=20` (constants §9.3). Runtime <1 s. Self-check: `assert` the `η=0.1` step lands exactly on the boundary
(monotone-then-oscillate transition); reproduces the Surface3D demo's numbers so page and script agree.

---
**`code/14_backprop_tn1.py`** · p.14 · **RUNG 1 (numpy by hand).** TN-1's nine gradients, both inputs, explicit
NumPy chain rule — no autograd.
Inputs → outputs: none → for **input 1** `x=[1,2]`: forward `ŷ=0.3727`, `L=0.9869`; nine grads incl.
`∂L/∂W1=[[-0.3764,-0.7527],[0.2028,0.4056]]`; one SGD step η=0.1 → `L=0.8222`, `ΔL=-0.1646`. For **input 2**
`x=[0.60,-0.20]`: `ŷ=0.5407`, `L=0.6148`; nine grads; **spread 10.22×**; step → `L=0.5630`.
Runtime <1 s. Self-checks: `assert` every value against constants §8/§8.7 (input 1 the six-corrected roundings;
input 2 the nine frozen grads); `assert abs(max|g|/min|g| - 10.2246) < 1e-3` (the spread beat, all nine live).
**⚠️ Do NOT print the dead-unit `∂L/∂W2[0]` as "exactly 0"** — see `15`.

---
**`code/15_autograd_check.py`** · p.15 · **RUNG 2 (torch matches TN-1 to the digit) + "the float that isn't zero."**
Builds TN-1 as `nn.Sequential`, sets the frozen weights, runs autograd, asserts agreement with `14`.
Inputs → outputs: none → torch grads printed beside the hand grads; the mandated float beat:
`torch float32 dW2 → [1.4020191230201817e-08, -0.5021151900291443]` — **not `0.0`**.
Runtime <2 s. Self-checks: **input 1** `assert torch.allclose(grad, [[-0.3764,-0.7527],[0.2028,0.4056]], atol=1e-4)`
(note `-0.7527`, not `-0.7528`; constants §8.5 — at `atol=1e-5` the brief's value fails). **input 2**
`assert torch.allclose(grad, [[-0.13475,0.04492],[0.22140,-0.07380]], atol=1e-5)` (5 d.p., NOT 4 — constants §8.7).
Print the mandated D-13 framing verbatim: `0.5-0.6+0.1` is float non-associativity; "the math on paper and the math
in the machine are not the same math"; the gradient is zero to seven decimals, the unit is dead, the lesson stands.

---
**`code/16_batch_regimes.py`** · p.16 · same MLP at `B∈{1,8,32,full}`; loss-curve noise vs batch.
Inputs → outputs: none → four `Timeline` curves + the `1/B` gradient-variance readout. Runtime ~2 s.
Self-check: `assert` full-batch loss is monotone (deterministic) while B=1 is noisy; reproduces the page's numbers.

---
**`code/17_optimizer_race.py`** · p.17 · **RUNG (optimizer).** Trains the TN-1 MLP (input-2 config) to convergence
under SGD / SGD+momentum / Adam / AdamW.
Inputs → outputs: none → four convergence curves; the momentum `1/(1-μ)=10×` and Adam `β2=0.999 → 1000-step
memory` beats (constants §9.3). Runtime ~5 s. Self-checks: `assert` Adam's `v̂` at k=1 is 1000× too small before
bias correction; reproduce the spread-motivated "therefore Adam" from the 10.22× of `14`.

---
**`code/19_init_variance.py`** + **`code/19_grad_flow.py`** · p.19 (**init + vanishing/exploding/clipping + residuals-WHY**).
`19_init_variance.py`: per-layer activation std for zeros / too-small / too-large / He init; He std at d=4096 is
`sqrt(2/4096)=0.0221` (constants §9.3). Self-check: `assert abs(he_std - 0.0221) < 1e-4`; `assert` zeros-init
collapses all activations. `19_grad_flow.py`: deep MLP, per-layer grad norm for sigmoid vs relu vs residual; the
amplifier numbers `1.1^100=13781`, `0.9^100=2.7e-5`, `0.25^32=5.4e-20` (below fp32 min-normal), `0.9^36=0.0225` vs
residual ≈1 (constants §9.3, D-14 residual WHY). Runtime ~2 s. Self-checks: `assert` the sigmoid chain underflows by
layer ~32; `assert` residual keeps grad-norm O(1).

---
**`code/20_norm_axes.py`** · p.20 (**normalization — the axis question**). LayerNorm / RMSNorm / BatchNorm on one
random tensor — "the only difference is which axis you reduce over."
Inputs → outputs: none → the three outputs + the axis each reduces. Runtime <1 s. Self-checks: `assert
NN.rmsNorm` matches a hand `x/sqrt(mean(x²)+eps)*γ`; `assert` RMSNorm does strictly fewer ops (no mean-subtract);
uses `rms_norm_eps=1e-6` (Qwen3, constants §1.1).

---
**`code/21_schedule.py`** · p.21 (**schedules & why warmup exists**). Warmup-cosine LR schedule as a pure function of
the **optimizer step `k`** (not epoch).
Inputs → outputs: `--total`, `--warmup` → the schedule plotted; the 2026 LLM defaults (warmup 1–10%, cosine to
η_max/10, constants §9.4). Runtime <1 s. Self-check: `assert` peak at end-of-warmup; `assert` schedule indexes by
step, not epoch (the bug the `k`-not-`t` notation enforces, notation §3.1). Warmup framed as the Adam-pathology fix
(β₂'s memory, computed on p.17).

---
**`code/22_regularization.py`** · p.22 (**regularization & the two curves**). Trains a small model with/without weight
decay (and dropout/early-stopping), written for the 500-example regime; shows the six train/val-curve shapes and their
six diagnoses.
Inputs → outputs: `--wd`, `--dropout` → train/val curves + the overfit/underfit/just-right diagnosis. Runtime ~20 s.
Self-checks: `assert` weight decay lowers the train-val gap; label the curves [EST] (a reproduction, not a frozen
number).

---
**`code/23_double_descent.py`** · p.23 · model-wise double descent on a small synthetic set.
Inputs → outputs: `--widths` → test error vs capacity, the classical U then the second descent past the
interpolation threshold. Runtime ~30 s. Self-check: `assert` the interpolation-threshold spike exists; label the
result [EST] (a reproduction, not a frozen number). Foreshadows rank as the bias-variance knob (D-11).

---
**`code/24_first_finetune.py`** · p.24 · **MILESTONE (before the fork).** A *real* small LoRA fine-tune on his box —
the trunk's first end-to-end run, deliberately kept tiny.
Inputs → outputs: `--model Qwen3-0.6B --data <jsonl>` → a trained adapter + before/after generations on ~50
examples; prints `print_trainable_parameters()` and the wall-clock. Runtime ~3–8 min on Qwen3-0.6B.
Self-checks: `assert` the adapter changed the output; `assert` peak memory < his measured budget; requires the
fresh venv from §A (page-24 seam — depends on the setup page's peft/trl install, spec-part2 appendix).

---

### D.2 — Trunk Part IV adaptation probes (pages 38–43)

---
**`code/scaling_and_moe.py`** · p.38 · build **O** · **builder creates** (spec-part4 p.38 `.box try`).
Purpose: motivate the whole Part — a base model is centuries of compute, so you adapt, you don't rebuild (D-14).
Computes $C=6ND$ and the ~2-millennia Spark estimate (printing the assumptions, the [EST] on $D$, and the [INF] on
the FLOP/s ceiling explicitly, constants §5/§6); computes MoE active-vs-total memory for the 2026 config table;
reproduces the "emergence" metric-artifact demo (exact-match vs continuous score on the same synthetic logits).
Runtime ~5 s. Self-checks: `assert C == 6*N*D` for the printed $N,D$; `assert` the active-memory figure < total for
each MoE row; label the years estimate a **lower bound** (peak-MFU assumption is generous to the machine, constants
§5). **⚠️ Do NOT print the retired "~380 years, generously assume sustained" framing** — it is inverted (D-20).

> **⚠️ Re-homing note (2026-07-16, §D-21a):** page 38 is **scaling laws / MoE / emergence**, not a decision page.
> The old fan-out's `code/adapt_decision_probe.py` (the fine-tune-on-your-docs → hallucinate vs RAG inoculation) was
> tied to the deleted page-38 "adaptation decision" entry. That topic and its probe now live **only at p.47**, and
> p.47 already ships `06_rag_vs_finetune_showdown.ipynb` (§D.3), which performs the identical inoculation.
> `adapt_decision_probe.py` is therefore **retired as a duplicate** — no shipped page attaches it; do not build it.

---
**`code/svd_rank_probe.py`** · p.39 · build **O** · **the LoRA hypothesis, verified.**
Purpose: show `ΔW` is low-rank while `W0` is not (D-12). Load a real Qwen3 `q_proj` (4096×4096) and a saved `ΔW`
from an actual fine-tune, `torch.linalg.svd` both, plot both scree curves, print the rank for 90% energy.
Inputs → outputs: `--weight q_proj.safetensors --delta dW.pt` → two scree plots + the two ranks. Runtime ~10 s.
Self-checks: `assert` the full matrix is 4096×4096 (not a toy); `assert` `ΔW`'s 90%-energy rank ≪ `W0`'s ("random
and W0 decay slowly; ΔW falls off a cliff" — D-12). Ships the rank-slider demo's data.

---
**`code/05_lora_from_scratch.py`** · p.40 · build **O** · **RUNG 5 (LoRA from scratch, LLM side).**
Purpose: implement a LoRA linear in ~40 lines of PyTorch (no `peft`), wrap Qwen3-8B's `q_proj`, print the count.
Inputs → outputs: none → `LoRALinear` with `h = W0 x + (α/r) B A x`, `A ~ N(0,σ²)`, **`B` init zeros**; prints
`trainable: 43,646,976 || all: 8,234,382,336 || 0.53%`, then flips `target_modules="all-linear"` and shows the count.
Runtime ~1 min (loads 8B in bf16). Self-checks: `assert` LoRA params = **43,646,976** (constants §3, r=16
all-linear) and **0.533%** of P; `assert` at step 0 `BA==0` so the wrapped model is bit-identical to base; the
`.box warn` (B-init-zeros ≠ dead: `∂h/∂B ∝ (Ax)ᵀ ≠ 0` so B moves first). Adapter-file size check: `43,646,976×2 =
87.3 MB` bf16 (constants §3). **Note:** the diffusion track ships a *separate* `lora_from_scratch.py` (FLUX.2, §D.4)
— see §F for why both exist.

---
**`code/08_quantization_lab.py`** · p.41 · build **O** · **the numbers the course refuses to print.**
Purpose: quantize a real Qwen3 layer to NF4 (double-quant on/off), FP8, NVFP4; measure RMS error + a small
perplexity delta on his box (constants §10 exercise #6; §9.7 forbids printing retention as fact).
Inputs → outputs: none → a table of B/param and measured RMS error per format. Runtime ~2 min.
Self-checks: `assert` NF4+DQ = **0.515869 B/param (4.127 bits)** and NF4 without DQ = **0.5625 B/param (4.5 bits)**
(constants §3 — the 0.53 is refuted, do not reproduce it); `assert` `bnb_4bit_use_double_quant` defaults **False**
(so the honest un-opted-in figure is 4.607 GB for Qwen3-8B, not 4.225 GB); print AWQ/GGUF/GPTQ retention as
**measured [MEA]**, never the SEO "95/92/90%" (constants §9.7). Teach the scale-granularity lesson (NVFP4 vs MXFP4,
88% lower error, constants §9.7).

---
**`code/inspect_my_comfy_graph.py`** · p.42 · build **S** · **his node graph, typed.**
Purpose: make the D-18 taxonomy concrete on *his* files. Point at a saved ComfyUI workflow (`~/ComfyUI`), list each
node, tag it **θ-edit** (LoRA/DreamBooth/full-FT) vs **c-edit** (ControlNet/IP-Adapter/prompt).
Inputs → outputs: `--workflow <path.json>` → per-node tags + "your graph changes the base model in 1 place and the
conditioning in 4." Runtime ~2 s. Self-check: `assert` LoRA nodes → θ, ControlNet/IP-Adapter/prompt → c ("after
you're done, is the base model different?" D-18). Uses his real 12 GB LoRA collection (`hardware-ground-truth` §5).

*(Page 43 attaches `measure_your_box.py`, already specced in D.0.)*

---

### D.3 — LLM track Part V (pages 44–52)

---
**`code/01_tokenizer_lab.ipynb`** · p.44 · build **S** · **RUNG 4 (tokenizer).**
Purpose: de-mystify the LLM input side. Load the real Qwen3 tokenizer, reproduce the 3-token `strawberry`, measure
`ρ` (chars/token) on *his own* corpus, and **print his corpus's total token count** — this feeds p.49's wall-clock.
Inputs → outputs: `--corpus <dir>` → ids, shapes, `ρ` for English vs code vs his jargon, total tokens.
Runtime ~2 min. Self-checks: `assert vocab_size == 151936` (constants §1.1); `assert ρ_english ≈ 4` and print it
as [EST] (constants §9.5); persist the token count for p.49. Notebook (`.ipynb`) so he can poke interactively.

---
**`code/03_base_vs_instruct.py`** · p.45 · build **S**
Purpose: the same prompt into base and instruct; print token-level probabilities; watch the base not stop; then
wrap the base in a chat template. Inputs → outputs: `--model Qwen3-8B` → two generations + the template diff.
Runtime ~10 min (loads 8B twice / sequentially). Self-check: `assert` the base's EOS probability stays low
(it doesn't stop); the instruct model does. Label behavioural claims [EST].

---
**`code/02_kv_cache_and_roofline.py`** · p.46 · build **O** · **RUNG 10 (throughput, decode side).**
Purpose: **measure** the KV cache growing with context (verify 144 KiB/token empirically), benchmark decode tok/s,
plot measured-vs-273 GB/s roofline, reproduce 60–75% MBU on his box.
Inputs → outputs: `--model Qwen3-8B --precision bf16` → KV-growth curve, decode tok/s, the roofline plot with his
measured point.
Runtime ~20 min. **SAFETY:** contends with ComfyUI if up. Self-checks: `assert` per-token KV = `2·L·H_kv·d_head·2
= 147,456 B = 144 KiB` (constants §4, note **H_kv=8**, not H); predict bf16 decode ceiling **16.67 tok/s** and
expectation **~10.8** (×0.65, constants §6.7) and print measured-vs-predicted; reproduce the batch-1→batch-32
**18×** decode gain while prefill stays flat (constants §6.7); print the decode-traffic byte convention
(file − embedding table, constants §6.6) and the MoE punchline ("if your heuristic predicts >100% of peak
bandwidth, your model is an MoE"). Ridge tags **[INF]** (227/458).

---
**`code/06_rag_vs_finetune_showdown.ipynb`** + **`code/07_rag_pipeline.py`** · p.47 · build **O** · **★★ the inoculation.**
Purpose: (a) LoRA-SFT an 8B on 200 of his docs, ask a fact from doc #147 → hallucinates fluently; (b) RAG the same
corpus → correct + citation; (c) both. "One hour permanently prevents the field's most expensive mistake."
`07_rag_pipeline.py`: chunk→embed→FAISS + a NumPy baseline that beats it under 100k → BM25 → RRF → rerank, ablate
recall@5. Inputs → outputs: `--docs <dir>` → the three answers + a recall@5 table.
Runtime ~45 min (the LoRA train dominates). Self-checks: `assert` SFT answer ≠ ground truth, RAG answer == it;
`assert` recall@5 improves through the BM25→RRF→rerank stages; label all effects [EST]/[MEA].

---
**`code/10_build_dataset.py`** · p.48 · build **O** · **RUNG 6 partner (the data pipeline).**
Purpose: the JSONL → `datasets` → chat-template → tokenized-tensor pipeline on his own ~500 examples; contains the
`13_chat_template_and_mask` beat (print `apply_chat_template` output + the `-100` loss mask for one real example,
assert the assistant-only %). Inputs → outputs: `--in <jsonl>` → tokenized dataset + the masked example.
Runtime ~1–3 min. Self-checks: `assert` labels use **`-100`** on non-assistant positions (the `m_i` mask,
notation §4.5); `assert` a document-level train/test split (no leakage); print the assistant-token fraction.

---
**`code/05_memory_ledger.py`** · p.49 · build **O** · **RUNG 9 (memory CLI).**
Purpose: the calculator as a CLI; compute the four-bucket ledger for any HF model id from its `config.json`
(reproduce the Qwen3-8B table programmatically), then **actually allocate and watch the 8B full-FT attempt OOM**.
Inputs → outputs: `python 05_memory_ledger.py --model Qwen/Qwen3-8B --method full --optimizer adamw_mixed
--batch 1 --seq 2048` → the bucket table (W/G/O/A) in GiB + verdict; `--attempt-oom` allocates until it fails.
Runtime: table <5 s; `--attempt-oom` a few min then OOMs. **SAFETY:** `--attempt-oom` deliberately exhausts memory;
warn it can thrash unified memory if ComfyUI is up. Self-checks: `assert` full-FT state = **131.05 GB / 122.05 GiB**
(`8,190,735,360 × 16`, constants §2.2); `assert` LoRA r=16 all-linear = **17.08 GB / 15.91 GiB**; `assert` the
187× identity `8,190,735,360 / 43,646,976 = 187.7` (= 114.67/0.61, constants §2.3/D-05); **measure** his
`mem_get_info()` total (**121.6875 GiB** on his box, [MEA-DEV]) and compute 122.05 − 121.69 = **−0.36 GiB → DOES
NOT FIT** before activations (constants §6.8, `hardware-ground-truth` §2). Print activations as **[EST] 2–6 GB**,
never a fact, and have him measure with `torch.cuda.max_memory_allocated()`. Shares `utils/ledger.py` with the
page-18 companion so page and CLI cannot drift. **Also invokes `00_verify_env.py` and `measure_your_box.py`.**
State the LR-vs-trainable-params principle here (D-09, this page owns it): fewer trainable params ⇒ higher LR.

---
**`code/11_finetune_qlora.py`** + **`code/13_ablate.py`** · p.50 · build **O** · **RUNG 6 (peft SFT, the main event).**
`11_finetune_qlora.py`: the p.49 script parameterized on his data — `transformers` + `peft` + `trl` `SFTTrainer`,
LoRA SFT on Qwen3-8B, Trackio logging. **~12 min for 8B LoRA — matching the prediction he computed in `01`+`05`.
The prediction landing is the payoff of the whole track.**
Inputs → outputs: `--data <jsonl> --model Qwen/Qwen3-8B --rank 16` → a trained adapter + the loss curve + wall-clock.
Runtime **~12 min** (constants §2.3/§6.7: 8B LoRA = 17.08 GB, 6,969.59 tok/s). Self-checks: `assert` LoRA params ==
43,646,976 via `print_trainable_parameters()`; **`assert` `SFTConfig` passes `model_init_kwargs={"dtype":
torch.bfloat16}`** (the two-libraries-one-script fp32 trap, constants §7.3); `target_modules="all-linear"` default
(D-06 — attention-only froze 78% of the block); raise LR ~10× over full-FT (LoRA 1e-4, constants §9.4); use
`bnb_4bit_use_double_quant=True` if QLoRA path (constants §3 caveat). **v5 hazard warns:** `report_to` defaults
`"none"` now; model dtype loads as-saved (constants §7.1). Do NOT double-wrap PEFT (trl issue #3926 — pick one path).
`13_ablate.py`: sweep `r∈{8,32,128,256} × {attn-only, all-linear}`, 6–8 runs × ~12 min ≈ **1.2 hr** — he personally
reproduces "all-linear beats attention-only at matched params" and "rank is nearly free in memory, dataset sets
rank" (D-06, D-11) on his own data. Self-check: `assert` memory barely moves r=8→256 (constants §3: r=256 = 27.55 GB).

---
**`code/14_merge_and_serve.py`** + **`code/12_evaluate.py`** · p.51 · build **O** · **RUNG 11 (merge/serve) + eval.**
`14_merge_and_serve.py`: `merge_and_unload()` into the **original bf16 base, NOT the NF4 QLoRA base** → GGUF →
`Q4_K_M` → `ollama create` → serve → measure decode tok/s vs `02`'s roofline. Inputs → outputs:
`--adapter <dir> --base Qwen/Qwen3-8B` → a served GGUF + measured decode tok/s. Runtime ~5–10 min.
Self-checks: **`assert` the merge target is the bf16 base** (merging into NF4 is lossy — ship the assert,
spec-part5 p.51 / constants §9.7); measured decode tok/s within the p.46 roofline band (~10.8 expected bf16, higher
for Q4). Default path Ollama/llama.cpp; vLLM cu130-container is the throughput option (D-10 serving ruling, §A rule 4).
`12_evaluate.py`: held-out loss + his task metric + a before/after **catastrophic-forgetting probe** (10–20 general
prompts, diff by eye) + LLM-as-judge (average both orderings, report answer lengths, warn if judge family ==
data-generator family — the closed-loop trap). Runtime ~10 min. Self-check: `assert` the judge runs both orderings
(position-bias fix); print "loss ↓ means better at predicting your dataset; your dataset is not your goal" (key box).

*(Page 51 also attaches `09_spark_capability_probe.py`, specced in D.0.)*

---
**`code/15_grpo_rlvr.py`** · p.52 (`52-grpo-rlvr-capstone.html`) · build **O** · **RUNG 7 (GRPO), stretch ~2 hr.**
Purpose: GRPO with a real verifiable reward on **Qwen3-0.6B/1.7B** (generation-in-the-loop must finish fast; the
arithmetic transfers to 8B by substituting `d,L,H_kv,d_ff`). `--vllm_mode colocate` via the **cu130 container**;
a llama.cpp fallback if the container isn't up.
Inputs → outputs: `--model Qwen/Qwen3-0.6B --reward <fn>` → a group of `G` sampled completions, their rewards
`r_i`, the advantages `A_i`, and the policy update; a before/after task-accuracy delta.
Runtime **~2 hr**. **SAFETY:** heavy, generation-in-the-loop; needs the container running. Self-checks: `assert`
the group advantage `A_i = (r_i − mean)/std` — e.g. `r_i=1`, mean 0.4, std 0.5 → **A_i = 1.2** (the quiz numeric);
`assert` `GRPOConfig.vllm_mode` default is `"colocate"` at trl≥1.8 (constants §7.2 — the v0→v1 flip); the `.box warn`
that RLVR reweights checkable behaviours, does not install facts. Requires the cu130 container (§A rule 4;
constants §6.9, D-10j — the only gate that was holding this capstone).

---

### D.4 — Diffusion track Part VI (pages 53–65)

**Audience (`hardware-ground-truth` §5):** a heavy *video*-diffusion practitioner. Every `.box try` here targets a
model **already on his disk** (FLUX.2-dev, SD1.5, `flux1-dev-kontext_fp8`, `z_image_turbo`, his `models/loras/`) —
no SD1.5 toys taught at him. The diffusion spec references these by a **manifest ID (D2–D17)**; the manifest table
lives in §E. Prefix-free filenames (no numeric collision with the LLM ladder). The 2-D scripts run on CPU/seconds;
the FLUX-scale scripts use his real weights.

---
**`code/vae_ceiling.py`** · p.53 (D-21: 52) · manifest **D2** · build **O**
Purpose: the VAE compression ceiling. Encode/decode a real photo through his **SD1.5 VAE**; show the reconstruction
and the information kept vs discarded. Inputs → outputs: `--image <jpg>` → original / latent / reconstruction +
compression factor. Runtime ~10 s. Self-checks: `assert` SD1.5 latent `[1,4,64,64]` at 512² and compression
`786,432/16,384 = 48×` (constants §9.6); `assert` FLUX.2 (from his config) `f=8`, 32 ch (§E manifest reads his disk).

---
**`code/diffusion_2d.py`** · p.54 (D-21: 53) · manifest **D3** · build **O**
Purpose: DDPM from scratch on 8-Gaussians (~200 lines), the whole forward+reverse loop visible in 2-D.
Inputs → outputs: none → animated forward noising and reverse sampling; the `β_t: 1e-4→0.02`, `ᾱ_T≈4e-5`,
`√ᾱ_T=0.0063` schedule (constants §9.6). Runtime ~2–5 min to train. Self-checks: `assert ᾱ` cumulative-product and
the nonzero terminal SNR; the `ε`-error amplification `1/0.0063 = 159×` at t=999 (constants §9.6).

---
**`code/ddpm_mnist.py`** · p.58 (D-21: 57) · manifest **D4** · build **S** · **RUNG 3 (MNIST, diffusion form).**
Purpose: a ~1.5M-param **U-Net** on 28×28 MNIST — a real conv denoiser, the trunk's CNN made generative.
Inputs → outputs: `--epochs N` → trained U-Net + sampled digits. Runtime ~10–20 min. Self-check: `assert` param
count ≈1.5M; samples become recognizable digits. **Note:** this is the diffusion U-Net, *not* the classifier MLP —
see §F for the `03_mlp_mnist.py` gap.

---
**`code/prediction_targets.py`** · p.55 (D-21: 54) · manifest **D5** · build **O**
Purpose: the ε / x0 / v parameterizations are one identity. Take one `(x0, ε, t)`, compute all three targets and
show they reconstruct each other. Inputs → outputs: none → the three targets + the `(a_t,b_t)` route table
(constants notation §4.4). Runtime <5 s. Self-checks: `assert` the DDPM `(a_t,b_t)=(√ᾱ_t,√(1−ᾱ_t))` satisfies
`a²+b²=1` (the circle); reproduce the four-framework table (DDPM/rectified-flow/VE/cosine).

---
**`code/score_check.py`** · p.56 (D-21: 55) · manifest **D6** · build **S**
Purpose: the score is a finite-difference fact. FD the Gaussian log-density, match the analytic score.
Inputs → outputs: none → analytic vs FD score, rel error. Runtime <5 s. Self-check: `assert` rel error < 1e-5 in
float64 (the same gradient-check discipline as `14`); `s_θ = ∇_x log p_t(x)` (notation §4.4).

---
**`code/flow_matching_2d.py`** · p.57 (D-21: 56) · manifest **D7** · build **O**
Purpose: the **same** 2-D file as `diffusion_2d.py` with `(a_t,b_t)=(1−t, t)` — flow matching is a choice of path,
not a new theory. Inputs → outputs: none → the straight-line reverse ODE vs DDPM's curved SDE. Runtime ~2–5 min.
Self-checks: `assert a_t+b_t=1` (the chord); open his FLUX config and match `FlowMatchEulerDiscreteScheduler`,
1000 vestigial train steps, shift 3.0 (`hardware-ground-truth` §4). Correct the retired "FM t is opposite to DDPM"
claim — both have data at t=0 (notation §6).

---
**`code/sample_flux.py`** · p.59 (D-21: 58) · manifest **D8** · build **O**
Purpose: real generation with **his FLUX.2-dev** off disk — the DiT/MMDiT made concrete. Inputs → outputs:
`--prompt "..." --steps 28` → an image + the latent `[1,32,128,128]` and 4096 tokens after 2×2 patchify.
Runtime ~1–2 min/image (his weights, bf16/fp8). Self-checks: `assert` FLUX.2 latent 524,288 elements → **6.0×**
compression, **4096** tokens (constants §9.6, verified on his disk `hardware-ground-truth` §4); the DiT config
(48 heads × 128, 8 dual + 48 single layers, `joint_attention_dim=15360=3×5120`).

---
**`code/bench_spark.py`** · p.60 (D-21: 59) · manifest **D16** · build **O** · **RUNG 10 (throughput, diffusion side).**
Purpose: **replaces the deleted FLUX-benchmark showpiece** (D-16). He runs **his FLUX.1-dev at 28 steps** at bf16
and FP4 himself and checks his own prediction against *both* rooflines. Inputs → outputs:
`--model flux1-dev --steps 28 --precision bf16,fp4` → measured s/step, s/image, implied TF/s vs the 62/125 TF and
500 TF-FP4 ceilings. Runtime ~a few min. Self-checks: label the NVIDIA number **"FLUX.1-Schnell, 4 steps, 1024²,
batch 1, FP4"** everywhere (constants §9.6 / D-16 — NOT dev/28); the SDXL UNet cross-check **5.977 TFLOP/forward**;
the diffusion verdict I=4096 → **compute-bound**, opposite of decode (D-15). Do NOT reconstruct the invalid 2.6 s
prediction. **SAFETY:** contends with ComfyUI.

---
**`code/cfg_by_hand.py`** · p.61 (D-21: 60) · manifest **D9** · build **O**
Purpose: CFG is one vector subtraction, in the model he runs. Run **his `flux1-dev-kontext_fp8`** with CFG done by
hand: `ε = ε_∅ + w_g(ε_c − ε_∅)`. Inputs → outputs: `--prompt --wg 1..8` → images across guidance scales + the
guidance vector. Runtime ~1 min/image. Self-checks: `assert` the hand CFG matches the pipeline's at matched `w_g`;
`w_g` range 1.0–8.0 (notation §4.4); the dot-product thread (CFG direction = the atom again).

---
**Page 62 (D-21: 61) diffusion-LoRA artifacts** · build **O** · **RUNG 8 (diffusion LoRA on his data).**
- **`code/lora_from_scratch.py`** · manifest **D10** — `LoRALinear` in ~30 lines (no `peft`) wrapping a real
  **FLUX.2** attention block. Runtime ~1 min. Self-check: `assert` `BA==0` at init (base bit-identical); this is the
  **diffusion-side** counterpart of `05_lora_from_scratch.py` (Qwen3) — same equation, different host (§F).
- **`code/train_lora_flux.py`** · manifest **D11**, **the main artifact** — DreamBooth-LoRA on **20 of his own
  images**, `diffusers`, "ComfyUI's LoRA trainer node, unwrapped." Inputs → outputs: `--images <dir of 20>
  --instance-token <rare>` → a trained FLUX LoRA he can load in ComfyUI. Runtime ~30–60 min. Self-checks:
  diffusion **LoRA LR = 1e-4** (NOT full-FT 1e-6 — the 100× D-09 error; constants §9.4); `r=8–32` (dataset-set
  rank, D-11 — high r on 20 images memorizes); FLUX.1-dev LoRA state ≈ **29–31 GB** total, fits his measured budget
  (constants §9.6). Filename is `train_dreambooth_lora_flux.py`-style — "DreamBooth is the recipe, LoRA is the
  storage" (D-18).
- **`code/caption_ab.py`** · manifest **D17**, the experiment nobody ships — trains **two SD1.5 LoRAs on the
  identical 20 images**, one with detailed captions, one with a bare token; compares. Runtime ~20–40 min.
  Self-check: label the outcome [EST]/[MEA] (a real ablation, not a frozen claim).

*(Pages 63–65, D-21: 62–64, the capstone/rejoining zone, are another spec's; they attach no new `code/` artifact
beyond re-using `train_lora_flux.py` and the LLM track's scripts for the D-17 "the branches grew back together"
beat — the FLUX.2 text encoder is a 40-layer Mistral-3 with GQA + RoPE, verifiable by `cat` of his own config.)*

---

### D.5 — Trunk Part I companions (pages 01–11) · ADDED BY THE ASSEMBLER · RE-HOMED to §D-21a (2026-07-16)

The Part I spec (`spec-part1.md`) commits these `.box try` files; the original catalog above predated the D-21a
restoration and mis-homed several. **Re-homed here to the canonical Part I order** (04 = derivatives, 05 = chain
rule, 08 = probability/reparameterization, 09 = logs/softmax/CE, 10 = XOR/collapse, 11 = activations). All are CPU,
seconds, `torch`-only, and obey §B (self-narrating, seeded, version-stamped, no elided imports). Listed in page
order; the numeric-prefix scheme does not apply to Part I (these carry descriptive names, one per page).

---
**`code/shape_gym.py`** · p.02 · build **S**. Builds `(64,)` and `(64,1)` tensors, prints `((pred-target)**2).mean()`
for the buggy broadcast vs the `.squeeze()`-fixed version so the silent broadcast happens and an `assert` catches it.
Runtime <1 s. Self-check: `assert` the buggy MSE ≠ the fixed MSE (the `(64,)`-vs-`(64,1)` trap, notation §7).

---
**`code/orthogonality.py`** · p.03 · build **S**. Two random unit vectors in $\mathbb R^{4096}$; prints cosine
≈0.01–0.02; loops 10 000× and prints empirical std ≈ $1/\sqrt{4096}=0.0156$. Runtime <1 s. Self-check:
`assert abs(std - 0.0156) < 2e-3` (high-dim near-orthogonality; feeds the embeddings thread, p.27).

---
**`code/descend.py`** · p.04 · build **S**. A quartic and its derivative; a descent loop with `--lr`, printing the
trajectory and detecting `Inf`. `--lr 0.35` diverges exactly like the page-04 demo. Runtime <1 s. Self-check:
`assert` convergence at small `η` and `Inf` at large `η` ("the NaN in your real logs is this, at scale"; the exact
`η_crit`=2/λ_max is page 13's).

---
**`code/chain_rule.py`** · p.05 · build **S**. The 8-line one-neuron autograd verification (prints `-0.1872`), plus a
loop stacking $N$ sigmoid layers printing the layer-1 gradient for $N=1..12$ so he watches it underflow. Runtime <1 s.
Self-check: `assert` the autograd grad matches the pencil `-0.1872`; the deep-sigmoid gradient underflows by ~$N=12$.
(This is the ONE `.backward()` Part I permits — a verification of the pencil, not the reverse-pass lesson, which is
p.14.)

---
**`code/tn1.py`** · page **06** (Early Real Thing; **accretes into the Part II companions at 14 / 15 / 17**) · build
**O** · **the course-spine Python script.** Ships complete-and-runnable-**forward** at p.06 (`ŷ=0.3727`, `L=0.9869`)
— **NO backprop on this page** (Part I is forward-only, §D-21a). The accretion happens on the Part II pages, where
the object becomes: `.backward()` + the nine-gradient check on **p.14** (`14_backprop_tn1.py`); the autograd
assertion + the `1.4e-08` float beat on **p.15** (`15_autograd_check.py`, which **owns** the mandated framing); the
SGD step `L: 0.9869 -> 0.8222` on **p.17** (`17_optimizer_race.py`). Runtime <1 s. Self-checks: `assert` the forward
values against `constants.md` §8.1. **⚠️ Hard-code the TN-1 config — never rely on `NN.worked221()`'s default (it
reproduces the RETIRED §5.4 network).** The `atol=1e-4` `-0.7527` assertion and the printed
`1.4020191230201817e-08` live on the p.14/p.15 companions, not here.

---
**`code/shapes_bridge.py`** · p.07 · build **S**. `nn.Linear(2,3)`, print `weight.shape → (3,2)`; do `W@x` (column)
and `x@W.T` (row) and `assert torch.allclose(...)`. Runtime <1 s. Self-check: the two forms are bit-equal ("only the
batch axis moved" — notation §1.2). **Uses `nn.Linear(2,3)`, never `(2,2)`** (notation §9 #25).

---
**`code/reparam.py`** · p.08 · build **O** · **the reparameterization trick, the ★ diffusion handoff.** ~18 lines:
draw $\mathcal N(\mu,\sigma^2)$ two ways in `torch` — `mu + sigma*torch.randn(n)` vs `torch.normal(mu, sigma, (n,))`
— and show the histograms match. Then the payoff: make `mu`/`sigma` `requires_grad=True`, form `x = mu + sigma*eps`
with a **detached** `eps`, `x.sum().backward()`, and print `mu.grad` (all ones) and `sigma.grad` (equals `eps`) —
gradients flow through the sample. Runtime <1 s. Self-check: `assert torch.allclose(mu.grad, ones)` and
`assert torch.allclose(sigma.grad, eps)`. "You just backpropagated through randomness — the whole VAE and the whole
diffusion forward process." (Pages 53–57 reopen this; do not re-teach it there.)

---
**`code/softmax_stable.py`** · p.09 · build **S**. Compute softmax two ways on the canonical logits `[2.0,1.0,0.1]`
(they agree; print $p_0=0.659001$, $\mathcal L=0.417030$, gradient $p-y$ summing to 0) and on `[800.,1.,1.]` (naive
`exp/Σexp` → `nan`; stable → correct). One line shows `F.cross_entropy(logits, target)` reproduces $\mathcal L$ from
*logits* directly. Runtime <1 s. Self-checks: `assert abs(L - 0.417030) < 1e-5`; `assert` the naive path is `nan`
and the stable path finite (the underflow-felt-first beat; `constants.md` §9.2).

---
**`code/xor_by_hand.py`** · p.10 · build **S**. Hardcodes the ReLU XOR weights, runs all four inputs → `[0,1,1,0]`;
then stacks two `nn.Linear` with no activation and shows it **cannot** fit XOR (loss floors). Runtime <1 s.
Self-check: `assert` the ReLU net outputs `[0,1,1,0]` and the linear-only net cannot (the D-19 linear-collapse beat,
taught **before** activations per D-19).

---
**`code/activations.py`** · p.11 · build **S**. Tabulate $\phi'_{\max}$ for sigmoid/tanh/ReLU/GELU; print
$0.25^{32}=5.4\times10^{-20}$ vs `torch.finfo(torch.float32).tiny`; build a 32-layer sigmoid stack and a 32-layer
ReLU stack, one backward each, print layer-1 grad norms (sigmoid ≈ 0 in fp32, ReLU healthy). Runtime <1 s.
Self-check: `assert` the sigmoid-stack layer-1 grad underflows and the ReLU-stack does not ("ReLU is the one whose
gradient isn't zero"; feeds SwiGLU named for p.36).

---

### D.6 — Trunk Part III companions (pages 25–37) · ADDED BY THE ASSEMBLER (2026-07-16)

Committed by `spec-part3.md`. CPU/seconds unless a real Qwen3 shape is loaded; obey §B.

---
**`code/inductive_bias.py`** · p.25 · build **S**. `nn.Linear(150528,1000)` param blow-up vs a conv layer; the
fully-connected-vs-conv contrast (the 1,792-param conv). Runtime <1 s. Self-check: `assert` the FC param count and
the conv count differ by the expected factor (architecture = inductive bias).

---
**`code/receptive_field.py`** · p.26 · build **O**. Implements the receptive-field recurrence for a conv config list,
prints `RF_ℓ` growth. Runtime <1 s. Self-check: `assert` the recurrence against a hand value (notation §4.2 `RF_ℓ`).

---
**`code/dot_product_geometry.py`** · p.27 · build **O**. The king/queen/bicycle embedding cosine numbers; dot product
as similarity. Runtime <1 s. Self-check: `assert` the cosine ordering (the dot-product thread, atom → similarity).

---
**`code/rnn_bottleneck.py`** · p.28 · build **S**. Times a Python loop of 2048 matrix-vector products vs one batched
matmul — the sequential bottleneck that forced attention. Runtime ~seconds. Self-check: `assert` the loop is
wall-clock-slower; the point is *sequential dependency*, not FLOPs.

---
**`code/attention_from_scratch.py`** · p.29 · build **O**. Implements the boxed attention formula on real Q/K/V
shapes, softmax over the **last (key)** axis. Runtime ~seconds. Self-checks: `assert` each row of `A` sums to 1
(columns do not, notation §7); shapes obey the ribbon.

---
**`code/heads_are_a_reshape.py`** · p.30 · build **S**. Takes a real Qwen3 `q_proj` shape; shows multi-head = a
reshape+transpose, with the `√d_head = √128 = 11.3137` scaling. Runtime ~seconds. Self-check: `assert 11.3137`
(constants §5) and that concatenating heads is the inverse reshape.

---
**`code/self_and_cross.py`** · p.31 · build **S** · **self vs cross-attention, the diffusion hinge.** Calls **one**
`F.scaled_dot_product_attention` and produces self vs cross by swapping only the K/V source — attention is
mask-agnostic; cross-attention takes K,V from a *different* sequence (the non-square A). Runtime ~seconds.
Self-check: `assert` self and cross differ only in the K/V tensor and the column count of A. (Causal masking is the
*next* page's story — `causal_mask.py`, p.32.)

---
**`code/causal_mask.py`** · p.32 · build **S** · **causal masking, the LLM hinge.** (1) `F.scaled_dot_product_attention(..., is_causal=True)`
→ `assert` the weight matrix is strictly-lower-triangular and every row sums to 1; (2) reproduces the
**multiply-after-softmax bug** (early rows sum to <1); (3) from **one** forward pass over a length-$S$ sequence,
computes the full per-position CE vector — **$S$ loss terms, one pass** — and prints $S$ (4096 signals per pass, the
LLM hinge); (4) a caching demo: generate 3 tokens, `assert` the cached $K,V$ for earlier positions are bit-identical
across steps (causality ⇒ the cache is sound). Runtime ~seconds. (KV-cache *sizing* is p.37's `kv_cache_ledger.py`.)

---
**`code/rope_identity.py`** · p.33 · build **O**. Implements $R_i$ and verifies
`allclose(⟨R_i q, R_j k⟩, ⟨R_{i-j} q, k⟩)` — RoPE depends only on relative position; `rope_theta = 1e6` (constants §1.1).
Runtime <1 s. Self-check: `assert` the relative-position identity to 1e-5 (trig payoff #2).

---
**`code/residual_stream.py`** · p.34 · build **S**. A 36-block toy stack with and without skip connections; per-layer
grad norm `0.9^36 ≈ 0.0225` (no skips) vs `≈ 1` (residual). Runtime ~seconds. Self-checks: `assert` the residual
stack keeps grad-norm O(1) (constants §9.3; D-14 residual WHY handed off from p.19, WHERE is this page).

---
**`code/build_block.py`** · p.35 · build **S**. A ~40-line `TransformerBlock` (RMSNorm, GQA q/k/v, SwiGLU FFN),
forward on a real `(B,S,d)` shape; assembles what pages 20/29–33 taught. Runtime ~seconds. Self-check: `assert`
output shape == input shape (residual stream width preserved); pre-norm placement (D-14).

---
**`code/count_params.py`** · p.36 · build **S** · **milestone companion.** Loads Qwen3-8B's `config.json`, computes
**8,190,735,360** via the §1.2 formula **and** independently from `model.safetensors.index.json` `total_size ÷ 2`.
Runtime <1 s. Self-checks: `assert` **both routes == 8,190,735,360** (the strongest credibility beat, constants §1.2);
`assert` FFN share = 66.4% of model / 78.26% of block (not 70%); no biases (`attention_bias:false`).

---
**`code/kv_cache_ledger.py`** · p.37 · build **S** · **O(S²), FlashAttention, GQA, KV cache.** From a config dict
computes **144 KiB/token** (note `H_kv=8`, not `H`), the full-context **5.625 GiB** at S=40,960, and the MHA
counterfactual **22.50 GiB**; then measures a real KV-cache allocation on his box with `torch.cuda.memory_allocated()`
before/after a generate call (ties to `measure_your_box.py`). Runtime <1 s. Self-checks: `assert` per-token KV =
`2·L·H_kv·d_head·2 = 147,456 B` (constants §4); `assert` the attention/FFN FLOP crossover at `S > 1.5·d_ff = 18,432`.
**(Scaling laws / MoE moved to p.38's `scaling_and_moe.py`, §D.2 — not here.)**

---

### D.7 — Capstone Part VII companions (pages 63–65) · ADDED BY THE ASSEMBLER (2026-07-16)

Committed by `spec-part7.md`; both are "builder-creates" scripts that reuse earlier engines. Obey §B; install nothing
into ComfyUI's venv (`hardware-ground-truth.md` §3).

---
**`code/predict_your_box.py`** · p.63 (D-21: 62) · build **O**. Step 1 `free,total = torch.cuda.mem_get_info()`
(constants §6.8 verbatim); step 2 compute **122.05 GiB** against *his* `total` (**121.6875 GiB** [MEA-DEV] → −0.36 GiB);
step 3 a big bf16 GEMM timed against **both** 62/125 TF ceilings in both accumulate modes (this is the core of
`09_spark_capability_probe.py`, reused). Runtime ~30–60 s. Self-checks: ridge/ceiling printed **[INF]** (227/458);
`assert` the ledger need 122.05 GiB; measure, never assert, the ceiling. **The intellectual capstone — the three
numbers the course refused to hand him, now his.**

---
**`code/verify_flux2_encoder.py`** · p.64 (D-21: 63) · build **O**. Loads his on-disk
`…/FLUX.2-dev/…/text_encoder/config.json`, prints `hidden_size, num_hidden_layers, num_attention_heads,
num_key_value_heads, rope_theta`, computes the **200 KiB/token** KV, and prints it **side-by-side with Qwen3-8B's
144 KiB/token**. Two shell one-liners inline (no elided imports, notation §9 #18). Runtime ~2 s. Self-check: the
FLUX.2 text encoder is a Mistral-3 (GQA + RoPE) — the D-17 "the branches grew back together" reveal, verified on his disk.

---

## §E — PACKAGING & THE MANIFEST

**Shipped alongside the scripts (tooling §14):**
- `code/README.md` — the canonical ordering (§C), the "rungs 1–5 need no Spark" note, the version stamp, and the
  prefix-scheme explainer (§F).
- `code/pyproject.toml` + `code/uv.lock` — the pins from §A (`torch==2.13.0`, `transformers==5.14.1`,
  `peft==0.19.1`, `trl>=1.8,<2`, `bitsandbytes==0.49.2`, `diffusers==0.39.0`, Python 3.12). Ship one lockfile; the
  box on the page says "if this errors years later, the fix is almost never 'the math changed'."
- `code/Dockerfile` — NGC cu130 base + HF layer, **no torch reinstall** (§A).
- `code/utils/{seed,memory,ledger}.py` — §B.

**Diffusion manifest map (the D-codes the diffusion spec references → files):**

| D-code | File | Page (D-21) |
|---|---|---|
| D2 | `vae_ceiling.py` | 53 (52) |
| D3 | `diffusion_2d.py` | 54 (53) |
| D4 | `ddpm_mnist.py` | 58 (57) |
| D5 | `prediction_targets.py` | 55 (54) |
| D6 | `score_check.py` | 56 (55) |
| D7 | `flow_matching_2d.py` | 57 (56) |
| D8 | `sample_flux.py` | 59 (58) |
| D9 | `cfg_by_hand.py` | 61 (60) |
| D10 | `lora_from_scratch.py` (FLUX.2) | 62 (61) |
| D11 | `train_lora_flux.py` | 62 (61) |
| D16 | `bench_spark.py` | 60 (59) |
| D17 | `caption_ab.py` | 62 (61) |

*(D1, D12–D15 are unused by the shipped pages — the diffusion spec's numbering has gaps; no file is owed. Recorded
so no builder invents one.)*

---

## §F — CROSS-PART SEAMS (name the pages other writers must match)

1. **The numeric-prefix scheme is dual, and that is a documented wart, not a bug.** In the **trunk companions
   (pages 12–24)** the prefix is the **page number** (`12_cross_entropy.py` = page 12). In the **LLM track
   (44–52)** the prefix is the **ladder rung** (`01_tokenizer_lab`, `02_kv_cache`, `05_memory_ledger`,
   `11_finetune_qlora`, `14_merge_and_serve`, `15_grpo_rlvr`). This produces **repeated prefixes across different
   files** — `05_lora_from_scratch.py`(p.40) vs `05_memory_ledger.py`(p.49); `12_cross_entropy.py`(p.12) vs
   `12_evaluate.py`(p.51); `13_lr_sweep.py`(p.13) vs `13_ablate.py`(p.50); `14_backprop_tn1.py`(p.14) vs
   `14_merge_and_serve.py`(p.51); `15_autograd_check.py`(p.15) vs `15_grpo_rlvr.py`(p.52). **All full filenames are
   distinct, so there is no filesystem collision** — they coexist in one `code/` dir. The README (§E) documents the
   two families and the canonical §C ordering. **No renames are forced; do not "fix" the prefixes** — the page
   specs (parts 2/4/5) have committed to these exact `code/...` paths in their `.box try` lines.
2. **Two `lora_from_scratch` files exist by design.** `05_lora_from_scratch.py` (p.40, wraps **Qwen3-8B q_proj**,
   asserts 43,646,976 params) and `lora_from_scratch.py` (p.62, wraps **FLUX.2 attention**). Same equation
   (`W0 + (α/r)BA`, B-init-zeros), different host, different track. Keep both filenames as the specs wrote them.
3. **MNIST gap — RESOLVED by the assembler (2026-07-16).** The task's rung 3 (MNIST) is served in the diffusion
   track by `ddpm_mnist.py` (a U-Net denoiser, p.58). The tooling brief's classifier `03_mlp_mnist.py` (784→128→10,
   **101,770 params**) is **attached by no page 01–65** — a full scan of all seven part specs finds **no reference to
   `03_mlp_mnist.py` and no occurrence of "101,770" anywhere**. The recurring-number worry is therefore moot: no page
   owes the script. **Ruling:** `03_mlp_mnist.py` is **optional / unshipped**; do not invent a page for it. If a
   future revision wants a classifier-MNIST rung, it must add both the page and the script together.
4. **`00_verify_env.py` and `measure_your_box.py` are shared across parts** — `00` attaches to both the setup page
   (Part III) and p.49; `measure_your_box.py` (already in `code/`) attaches to p.18 (`--no-gpu`) and p.43
   (full). Do not duplicate; cross-reference. Page 18 must not assert the ~62 TF ceiling/ridge (that is p.43/p.51,
   [INF]) — already flagged in spec-part2's appendix.
5. **The setup page (Part III) owns the fresh-venv install (§A).** Every Part IV/V `.box try` and page 24's
   milestone depend on it. If the Part III author changes the venv path or the pins, this file's §A must be matched.
6. **`worked221()` trap (spec-part2 appendix, restated for completeness):** the JS `NN.worked221()` default
   reproduces the RETIRED §5.4 network; the **Python** `14_backprop_tn1.py` must hard-code the TN-1 configs from
   constants §8/§8.7 and never rely on a default. Any Part I/III page that ships a TN-1 script inherits this.
7. **★ TN-1 backprop/autograd home — RESOLVED by §D-21a: ONE home, pages 14–15; Part I is forward-only.** The
   earlier escalation (backprop appearing at both Part I 09/10 and Part II 14/15) is closed. Under the canonical
   table, **Part I teaches forward + the chain rule only**; the reverse pass over TN-1 is **page 14**
   (`14_backprop_tn1.py`, numpy-by-hand, the nine gradients), autograd + the "float that isn't zero" beat is **page
   15** (`15_autograd_check.py`, which **owns** the mandated `constants.md §8.4` `.box key`, the `1.4020191230201817e-08`
   value, and the input-1 `atol=1e-4`/`-0.7527` assertion), and the first SGD step is **page 17**
   (`17_optimizer_race.py`). `code/tn1.py` (p.06) ships **forward-only** and is the object those Part II scripts
   accrete — it does **not** carry a backprop/autograd/SGD stage of its own. **`constants.md §8.4`'s "page ~10"
   pointer is stale** (documented in spec-part1 p.10's spec-writer note and spec-part2's appendix): page 10 is
   XOR/linear-collapse and carries no autograd. No structural ambiguity remains; the distinct filenames were already
   collision-free.
