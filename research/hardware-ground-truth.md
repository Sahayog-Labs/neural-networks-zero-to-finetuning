# hardware-ground-truth.md — MEASURED ON A REAL DGX SPARK (public, redacted edition)

**Status: MEASURED 2026-07-16 on the learner's own NVIDIA DGX Spark.**
**This file OUTRANKS everything.** Where a brief, `constants.md`, `decisions.md`, or an
inference disagrees with a number here, this file wins — these values were read off a real
machine, not published specs.

> **Public note:** the original edition of this file included the specific machine's network
> identity and a personal model/software inventory; those sections are removed here. Every
> *product-level* measurement below is preserved verbatim — these are facts about any DGX
> Spark, and you can (and should) reproduce them on yours with `code/measure_your_box.py`.

Confidence label for everything below: **MEASURED-ON-DEVICE** — the only label that outranks
`verified-primary`.

---

## 1. The machine (product facts, all confirmed on-device)

| Fact | Value | How obtained |
|---|---|---|
| GPU | `NVIDIA GB10` | `nvidia-smi` |
| Compute capability | **12.1 (sm_121)** | `torch.cuda.get_device_capability(0)` → `(12, 1)` |
| CPU | 20-core Arm: 10× Cortex-X925 + 10× Cortex-A725 | `lscpu` |
| Architecture | `aarch64` | `uname -m` |
| OS | Ubuntu-based DGX OS (24.04 line), NVIDIA kernel | `/etc/os-release` |
| CUDA (driver) | 13.0 | `nvidia-smi` |

---

## 2. ⛔ MEMORY — THE BIG ONE. THE PUBLISHED "128 GB" IS NOT WHAT YOU GET.

```
/proc/meminfo MemTotal:  127,600,524 kB
             = 130,662,936,576 bytes
             = 130.66 GB (decimal)
             = 121.6875 GiB (binary)   ← EXACTLY 121.6875, not approximately

torch.cuda.mem_get_info() total = 121.69 GiB   ← the GPU sees the SAME pool (unified)
```

| Quantity | Value |
|---|---|
| 128 GiB physical (what "128 GB LPDDR5X" means) | 137,438,953,472 B = 128.0000 GiB |
| **What the OS actually reports** | 130,662,936,576 B = **121.6875 GiB** |
| **Reserved by firmware/driver — invisible to you** | **6,776,016,896 B = 6.3125 GiB** |

**~6.31 GiB — 4.9% of the box — is gone before you allocate a single byte.**

### 2.1 The consequence the course is built around

The full-fine-tune state for Qwen3-8B at 16 B/param is **122.05 GiB** (`constants.md` §2):

| Comparison | Verdict |
|---|---|
| Naive "131 GB needed vs 128 GB" | wrong reasoning — mixes decimal GB with binary GiB |
| "122.05 GiB vs 128 GiB physical → fits with ~6 GiB spare" | **refuted by measurement** — assumes the physical 128 GiB is usable |
| **MEASURED: 122.05 GiB needed vs 121.6875 GiB existing** | **−0.36 GiB. DOES NOT FIT** — by 0.3%, *before* activations (+2–6 GB) |

The course (p.18) deliberately never asserts this budget: the learner runs
`torch.cuda.mem_get_info()` on their own machine first and computes 122.05 GiB against
*their* number. The lesson generalizes: **published capacity is never usable capacity; the
spec sheet is off by ~4.9% before you start.** Also distinguish three numbers people
conflate: published (128 GB) vs `MemTotal` (121.69 GiB, the real ceiling) vs
`MemAvailable` / `mem_get_info` free (whatever is left *right now* under your workload).

---

## 3. ⛔ THE aarch64 / sm_121 SOFTWARE REALITY (proven on-device)

```python
torch.cuda.get_device_capability(0)  → (12, 1)      # the GPU is sm_121
torch.cuda.get_arch_list()           → ['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']
```

**The PyPI torch wheel contains NO sm_121 binary and NO PTX — and it runs anyway**, via
CUDA *minor-version binary compatibility* (an sm_120 cubin runs on sm_121 because they share
major version 12). Consequences the course teaches:

- The "PTX JIT fallback" story you may have read is **false** for release wheels — they ship
  SASS only. There is no PTX to JIT.
- Because the stack runs on a compatibility guarantee rather than a native build, a stray
  `TORCH_CUDA_ARCH_LIST` or an ill-configured source build can break what currently works.
- Practical traps verified for this platform: anything pinning `libcudart.so.12` dies at
  import on the CUDA-13-only DGX OS; vLLM needs the cu130 container (the PyPI aarch64 wheel
  targets CUDA 12); **bitsandbytes ships native sm_121 on ARM64** — the Spark is *better*
  served than x86 there.
- Recommended setup: a **fresh venv** for the course's training stack. Never install into an
  existing app's venv (e.g. ComfyUI's) — you will break a working tool.

---

## 4. ⛔ FLUX.2 — RESOLVED FROM THE ACTUAL WEIGHTS

Read from a local `black-forest-labs/FLUX.2-dev` checkout (reproduce with
`code/verify_flux2_encoder.py`):

| Question | Answer | Evidence |
|---|---|---|
| VAE spatial factor | **f = 8** | exactly 3 downsamplers in `ae.safetensors` → 2³ |
| Latent channels | **32** | `decoder.conv_in.weight` = `[512, 32, 3, 3]` |
| Encoder output | **64** = 32 μ + 32 logvar | `encoder.conv_out.weight` = `[64, 512, 3, 3]` |
| Tokens @ 1024² | **4096** | 128×128 latent grid, 2×2 packed → 64×64; confirmed by `scheduler_config.json` `max_image_seq_len: 4096` |
| VAE size | **84.0M params**, F32 | summed from the safetensors header |
| Scheduler | **FlowMatchEulerDiscrete**, 1000 train steps, shift 3.0, dynamic shifting | `scheduler/scheduler_config.json` |

### 4.1 The capstone fact (p.64)

`model_index.json` → `"text_encoder": ["transformers", "Mistral3ForConditionalGeneration"]`
— a 40-layer multimodal **Mistral-3** (hidden 5120, GQA 32/8 heads, RoPE θ=1e9, vocab
131,072, plus a vision tower and `PixtralProcessor`). The diffusion model's transformer has
`joint_attention_dim: 15360 = 3 × 5120`. **"The tracks grew back together" is a fact you can
verify with `cat` on files you already have** — every component of that encoder is something
the LLM track teaches.

---

## 5. Still unmeasured (deliberately taught as YOUR exercises)

| Open | Why it matters | How to close |
|---|---|---|
| **GB10 dense BF16 TFLOPS** | NVIDIA publishes only FP4-with-sparsity (1 PFLOP). The inferred ~125 TF (fp16-acc) / ~62.5 TF (fp32-acc) is **inference, not fact** — and it sets the roofline ridge (p.43): 227 vs 458 FLOP/byte. | `code/measure_your_box.py` / `code/09_spark_capability_probe.py` |
| **Achieved memory bandwidth** | Published 273 GB/s is a theoretical peak. The decode heuristic (p.46) needs the achieved fraction (~60–75% across verified measurements, batch-1 dense). | `code/measure_your_box.py` |
| Real decode tok/s for Qwen3-8B | p.43/46's predict-then-check lands on this. | `code/02_kv_cache_and_roofline.py` after setup |

**The course's rule stands: never let these be filled in by inference and shipped as fact.**
Measure them on your box — that's the point.
