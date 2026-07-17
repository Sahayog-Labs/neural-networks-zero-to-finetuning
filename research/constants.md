# constants.md — FROZEN. Single source of truth.

**Status: RATIFIED 2026-07-16.** Every page-builder agent MUST read this file and MUST cite these values
verbatim. **No agent may edit this file.** If a number you need is not here, do not invent it — flag it to the
architect. If a number here contradicts a brief, **this file wins** (the briefs predate two verification passes).

**Confidence legend — printed next to every value and, where the course prints the number, printed to the learner too:**

| Tag | Meaning | How the course may present it |
|---|---|---|
| **[VP]** | verified-primary — fetched from a primary source (config.json, safetensors index, NVIDIA docs, PyPI) this session | as fact |
| **[DER]** | derived — arithmetic from [VP] inputs, independently recomputed | as fact, with the derivation shown |
| **[INF]** | inferred — not published; deduced from published numbers by a stated chain | **must be labelled "inferred, not published" on the page**, chain shown |
| **[EST]** | unpublished-estimate / folklore / practitioner consensus | **must be labelled an estimate on the page.** Never printed as a fact. |
| **[MEA]** | must be measured by the learner on his own box; the course ships the script | the course states the method, not the number |
| **[MEA-DEV]** | **measured on the learner's actual DGX Spark this session** (`hardware-ground-truth.md`) | as fact — but where the pedagogy is predict-then-measure, the page still has him measure; the spec merely *knows* what he will see |

> **Authority chain:** `hardware-ground-truth.md` (measured on-device 2026-07-16) **outranks this file** for
> hardware facts; this file outranks `decisions.md` reasoning; both outrank the briefs. The measured facts that
> matter most: **usable memory is 121.6875 GiB, not 128** (firmware carveout eats 6.31 GiB = 4.9%); the installed
> torch really does ship no sm_121 binary and no PTX (§6.9 confirmed on-device); peft/trl/accelerate/bitsandbytes
> are NOT installed on the box (the setup page installs into a fresh venv, never into ComfyUI's).

> **The prime directive of this file:** *do not launder an estimate into a fact.* Three of the most quotable
> numbers in the source briefs (the "1 PFLOP", the "128 GB budget", the "2.6 s FLUX image") were laundered
> estimates or category errors. They are corrected below. If you feel a page needs a crisper number than this
> file gives, the honest page is the one that says "measure it."

---

## 0. UNIT CONVENTION — READ FIRST. This is a hard rule.

The single most consequential arithmetic error in the source corpus was a GB/GiB mismatch. Therefore:

| Quantity | Unit | Why |
|---|---|---|
| **Anything compared against hardware capacity** | **GiB** ($2^{30}$ B) | DRAM is binary. "128 GB of LPDDR5X" means 128 GiB = 137.44e9 bytes. |
| **Model weights, optimizer state, dataset size quoted on their own** | **GB** ($10^9$ B) | matches every model card and the `bytes/param × params` arithmetic |
| **Bandwidth** | **GB/s** ($10^9$ B/s) | vendor convention, universal |
| **KV cache** | **KiB/token, GiB total** | it is compared against capacity |

**Every page that states a memory figure MUST state the unit and MUST NOT mix the two in one comparison.**
When a page compares a model's footprint to the box, it converts to GiB first and says so.

`1 GiB = 1.0737 GB` · `1 GB = 0.9313 GiB`

---

## 1. THE ANCHOR MODEL — Qwen3-8B

**Ruling (see decisions.md D-01): Qwen3-8B is the course-wide arithmetic anchor. Llama-3.1-8B, Llama-2-7B,
and the generic "7B" are RETIRED and must not appear in any parameter, memory, or KV arithmetic.**

Chosen because: its `config.json` is ungated and was fetched this session; the derived parameter count matches
the checkpoint's own safetensors index **exactly** (not approximately); Apache-2.0, so no license digression
mid-lesson; and it is simultaneously the anchor *and* runnable on the learner's box (LoRA = 17.08 GB, ~12 min).

### 1.1 Config — [VP], fetched from HuggingFace 2026-07-16

```json
{"hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32,
 "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
 "vocab_size": 151936, "tie_word_embeddings": false, "rms_norm_eps": 1e-06,
 "rope_theta": 1000000, "max_position_embeddings": 40960, "hidden_act": "silu",
 "attention_bias": false, "torch_dtype": "bfloat16", "sliding_window": null,
 "rope_scaling": null}
```

**The fourteen numbers that must recur until they are reflexive** — every page touching a transformer uses these:

`4096` (d_model) · `36` (L) · `32` (H) · `8` (H_kv) · `128` (d_head) · `12288` (d_ff = 3d) ·
`151936` (V) · `40960` (context) · `1e6` (rope_theta) · `8,190,735,360` (P) · `6,946,071,552` (non-embed) ·
`16.38 GB` (bf16 weights) · `144 KiB/token` (KV) · `16 B/param` (full-FT ledger)

### 1.2 Parameter derivation — [DER], recomputed twice, confirmed against the checkpoint

| Tensor | Formula | Params |
|---|---|---|
| `q_proj` | 4096 × (32×128) | **16,777,216** |
| `k_proj` | 4096 × (8×128) | **4,194,304** |
| `v_proj` | 4096 × (8×128) | **4,194,304** |
| `o_proj` | (32×128) × 4096 | **16,777,216** |
| *attention subtotal* | $2\,d\,d_{head}(H + H_{kv})$ | **41,943,040** |
| `gate_proj` | 4096 × 12288 | **50,331,648** |
| `up_proj` | 4096 × 12288 | **50,331,648** |
| `down_proj` | 12288 × 4096 | **50,331,648** |
| *FFN subtotal* | $3\,d\,d_{ff}$ | **150,994,944** |
| `input_layernorm` + `post_attention_layernorm` | 2 × 4096 | **8,192** |
| `q_norm` + `k_norm` (QK-Norm) | 2 × 128 | **256** |
| *norm subtotal* | | **8,448** |
| **per block** | | **192,946,432** |
| **× 36 blocks** | | **6,946,071,552** |
| `embed_tokens` | 151936 × 4096 | **622,329,856** |
| `lm_head` (untied) | 151936 × 4096 | **622,329,856** |
| `model.norm` | 4096 | **4,096** |
| **TOTAL** | | **8,190,735,360** |

**No biases anywhere** (`attention_bias: false`). **RoPE has zero parameters.**

**The independent confirmation — the course MUST show this, it is the strongest credibility beat available:**
`model.safetensors.index.json` reports `total_size = 16,381,470,720` bytes. All 399 tensors are bf16 (2 B).
$16{,}381{,}470{,}720 / 2 = \mathbf{8{,}190{,}735{,}360}$ — **exact match, from the checkpoint itself, not from the
formula.** Two independent routes to the same integer. [VP]

### 1.3 Derived shares — [DER]

| Quantity | Value | Note |
|---|---|---|
| FFN share **of a block** | $150{,}994{,}944 / 192{,}946{,}432 = \mathbf{78.26\%}$ | "the FFN is 78% of the block" |
| FFN share **of the whole model** | $5{,}435{,}817{,}984 / 8{,}190{,}735{,}360 = \mathbf{66.4\%}$ | ⚠️ **not** 78% — different denominator. Both are true; say which. |
| Attention, all blocks | 1,509,949,440 | 18.4% of model |
| Norms, all | 308,224 | 0.004% — but see D-07: they are **not** droppable from the total |
| Embeddings | 1,244,659,712 = **15.20%** | pure lookup, $O(1)$ compute/token |
| GQA saving vs MHA | 25,165,824/block = **37.5% of the attention block** | *and* 4× on the KV cache — two wins |
| Model card cross-check | card says 8.2 B total / 6.95 B non-embedding | ✓ exact to every published figure |

> **⚠️ Retired number:** foundations' *"~70% of the model is FFN"* was a Llama-3.1-8B figure (d_ff = 14336 = 3.5d).
> Qwen3-8B is d_ff = 12288 = **3d**. Use 66.4% (whole model) or 78.26% (per block). Never 70%.

### 1.4 The secondary models — [VP] where marked

| Role | Model | When it may be used |
|---|---|---|
| **Anchor — all arithmetic** | **Qwen3-8B** | everywhere. Non-negotiable. |
| **Smoke test / 60-second runs / GRPO stretch** | Qwen3-0.6B, Qwen3-1.7B | only where a run must finish in <2 min |
| **Contrast, once, in the MoE section** | Kimi K2 (1T/32B-A) | to make "memory tracks total, compute tracks active" concrete |
| **70B QLoRA capability demo** | Llama-3.3-70B (d=8192, L=80, d_ff=28672, H_kv=8) [VP, fetched] | §LLM hardware page only |

**Llama-3.1-8B is retired from this course.** Its config is gated (401), so it cannot be verified live, and
mixing it with Qwen3 was the corpus's largest silent conflict.

---

## 2. THE MEMORY LEDGER

### 2.1 Bytes per parameter — [DER], the course's most reused table

Standard bf16 mixed precision + AdamW:

| Item | Precision | B/param | Why it exists |
|---|---|---|---|
| weights | bf16 | 2 | what the GPU computes with |
| gradients | bf16 | 2 | one per weight |
| **fp32 master weights** | fp32 | 4 | bf16 has 7 mantissa bits; a 2e-5 update to an O(1) weight rounds to **zero** |
| Adam $m$ | fp32 | 4 | optimizer state |
| Adam $v$ | fp32 | 4 | optimizer state |
| **TOTAL** | | **16** | **memorize `16P`** |

**Both framings are correct and the course must state both** (a learner who reads two blog posts will otherwise
hit an apparent contradiction): 16 B/param is **4× the fp32 weight size** *and* **8× the bf16 weight size**.
Same 16 bytes.

Optimizer variants, $k_O$ = bytes per **trainable** param for optimizer state alone:

| Optimizer | State | $k_O$ |
|---|---|---|
| SGD | — | 0 |
| SGD + momentum | 1 buffer fp32 | 4 |
| **AdamW, fp32 $m,v$** | 2 buffers | **8** |
| **AdamW mixed-precision (fp32 master + $m$ + $v$)** | 3× fp32 | **12** |
| AdamW 8-bit (bnb) | quantized $m,v$ | ~2 |
| Muon (2-D params only) | 1 buffer | ~4 |

### 2.2 Qwen3-8B full fine-tune — [DER]. **THE number.**

$$8{,}190{,}735{,}360 \times 16\ \text{B} = 131{,}051{,}765{,}760\ \text{B}$$

| Presentation | Value |
|---|---|
| **decimal** | **131.05 GB** |
| **binary** | **122.05 GiB** |
| per-component | weights 16.38 GB · grads 16.38 GB · master 32.76 GB · $m$ 32.76 GB · $v$ 32.76 GB |

**⚠️ THIS IS STATE ONLY. Activations are a separate, separately-labelled line, always.** [EST] Activations at
B=1, S=2048 with gradient checkpointing: **2–6 GB**. Without checkpointing: far more. The course states this as
an estimate and has the learner measure it.

> **The 128-vs-131 cliffhanger as *written* was a GB-vs-GiB artifact — but the wall is REAL. [MEA-DEV]**
> Measured on his box (`hardware-ground-truth.md` §2): usable `MemTotal` = **121.6875 GiB**, not 128 —
> firmware carveout eats 6.31 GiB. **122.05 GiB vs 121.69 GiB = −0.36 GiB. Does not fit**, by 0.3%, before
> activations. The course still must **not** assert the budget on the page; the learner **measures**
> (`torch.cuda.mem_get_info()`, `free -g`) and computes 122.05 GiB against *his* number. See §6.8 and D-04's
> addendum.

### 2.3 The escape ladder — [DER]

| Method | B/param | Qwen3-8B state | vs 121.69 GiB measured [MEA-DEV] |
|---|---|---|---|
| Full, AdamW mixed | 16 | **131.05 GB / 122.05 GiB** | **−0.36 GiB — does not fit** (learner re-measures live) |
| Full, 8-bit Adam | 10 | **81.91 GB / 76.28 GiB** | fits |
| Full, 8-bit Adam, no fp32 master | 6 | **49.14 GB / 45.77 GiB** | fits; ⚠️ stalled-update risk |
| SGD + momentum (bf16 w+g, fp32 m+master) | ~12 | **98.29 GB / 91.54 GiB** | fits; converges badly on transformers |
| **LoRA r=16, all-linear** | see §3 | **17.08 GB / 15.91 GiB** | fits, 7.7× |
| **QLoRA r=16 (NF4 base)** | see §3 | **~4.93 GB / 4.59 GiB** state | fits, ~26× |

**Reduction claims — use these exact words:**
- Full → LoRA: **131.05 GB → 17.08 GB = 7.67×** (state-to-state, both exclude activations)
- **Trainable state alone: 114.67 GB → 0.61 GB = 187×** — and **187 is exactly the parameter ratio**
  ($8{,}190{,}735{,}360 / 43{,}646{,}976 = 187.7$), which is the *better* sentence.
- Full → QLoRA: **131.05 → ~4.93–7 GB = 19–26×.** Say the range, not "~20×".

> **RETIRED, do not use:** ~~"224×"~~ (mismatched numerator/denominator), ~~"417×"~~ (correct, but for the
> retired Llama-2-7B/attn-only example), ~~"130 GB → 14.3 GB"~~ (compares state+activations to state-only —
> LoRA does **not** reduce activation memory; you still backprop through the full network), ~~"96 GB AdamW on 8B"~~
> (foundations §2 teaser — wrong, it is 131.05 GB), ~~"7B full FT ≈ 112 GB"~~ (retired anchor).

### 2.4 LoRA memory does **not** include an activation saving — [DER], warning box

LoRA freezes weights; it does **not** shorten the graph. Activations are ~unchanged between full FT and LoRA
at the same B and S. **Every LoRA-vs-full comparison in this course is state-to-state, with activations named
as a separate identical line on both sides.** A learner who measures LoRA's real footprint will find it above
17.08 GB by exactly the activation term, and the course must have told him so first.

---

## 3. LoRA ARITHMETIC — Qwen3-8B — [DER]

$$\text{LoRA params per matrix} = r\,(d_{in} + d_{out}) \qquad\text{vs}\qquad d_{in} \cdot d_{out}$$

| Matrix | Shape (out × in) | Base params | LoRA @ r=16 |
|---|---|---|---|
| `q_proj` | 4096 × 4096 | 16,777,216 | 131,072 |
| `k_proj` | 1024 × 4096 | 4,194,304 | 81,920 |
| `v_proj` | 1024 × 4096 | 4,194,304 | 81,920 |
| `o_proj` | 4096 × 4096 | 16,777,216 | 131,072 |
| `gate_proj` | 12288 × 4096 | 50,331,648 | 262,144 |
| `up_proj` | 12288 × 4096 | 50,331,648 | 262,144 |
| `down_proj` | 4096 × 12288 | 50,331,648 | 262,144 |
| **per layer (7 linears)** | | **192,937,984** | **1,212,416** |
| **× 36** | | **6,945,767,424** | **43,646,976** |

**⚠️ Base-count reconciliation — the fix for the corpus's third conflict (D-07).** The 6,945,767,424 above
**omits the norms** (they are not LoRA targets, correctly). It is **not** the model's non-embedding parameter
count. The reconciliation line the course prints is:

$$6{,}945{,}767{,}424 \;+\; \underbrace{36 \times 8{,}448}_{304{,}128\ \text{norms}} \;+\; \underbrace{4{,}096}_{\text{final norm}} \;+\; 2\times622{,}329{,}856 \;=\; \mathbf{8{,}190{,}735{,}360}$$

**Exact.** No page may say "closes to three significant figures." It closes to the byte, and saying so is the point.

| Quantity | Value |
|---|---|
| LoRA r=16 all-linear, trainable | **43,646,976 = 0.533% of P** |
| LoRA r=16, single 4096×4096 matrix | 131,072 vs 16,777,216 = **128× reduction** |
| LoRA r=256 all-linear | **698,351,616 = 8.53% of P** → **27.55 GB** total. Still trivially fits. |
| Adapter file, r=16, bf16 | 43,646,976 × 2 = **87.3 MB** |
| **Llama-3.3-70B, r=16, all-linear** [DER from fetched config: d=8192, L=80, d_ff=28672, H_kv=8] | **207,093,760 = 0.294%** → 3.31 GB state → **≈ 39.7 GB** total with NF4 base |

> **RETIRED:** ~~"70B adapters ≈ 0.4% ≈ 280M → 4.5 GB → ~41 GB"~~ — used the hand-wave percentage that the
> same brief's own demo spec explicitly forbids. Conclusion survives — ≈39.7 GB is comfortably inside the
> **measured 121.69 GiB** [MEA-DEV] — but the numbers do not.

**QLoRA:** NF4 with double-quant effective **0.516 B/param = 4.127 bits/param** [VP — arXiv 2305.14314 §3
*Double Quantization* + bitsandbytes `main` source, both fetched and recomputed this session]. **The 0.53 figure
is REFUTED — it was an arithmetic error, not a rival convention. There is no range.**

**Derivation (exact rationals, all four cross-checks close):**
- NF4 weight = 4 bits/param.
- First-level scale: fp8 absmax per 64-block (with DQ) → $8/64 = 0.125$ bits/param.
- Second-level: one fp32 constant per **256 scales**, and each scale covers **64 params**, so it amortizes over
  $64\times256 = 16{,}384$ params → $32/16384 = 0.001953125$ bits/param.
- **Overhead = $8/64 + 32/(64\cdot256) = 65/512 = 0.126953125 \to \mathbf{0.127}$ bits** — matches the paper verbatim.
- **Total = $4 + 65/512 = 2113/512 = \mathbf{4.126953125}$ bits $= \mathbf{0.515869}$ B/param.**
- Cross-checks that reproduce the paper's own published figures: reduction $0.5 - 0.127 = 0.373$ bits ✓;
  DQ saving on 65B $= 65\text{e}9 \times 0.373/8 = \mathbf{3.03\ GB}$ ✓ (paper abstract: "approximately 3 GB").

> **⚠️ The 0.53 error, named so no builder repeats it:** `brief-llm-finetuning.md:750` wrote
> `4 + 8/64 + 32/256 = 4.25`. It divided the second-level fp32 constant by 256 (the number of *scales* it covers)
> instead of by $64\times256 = 16{,}384$ (the *parameters* those scales cover) — a bits-per-scale term added to a
> bits-per-param budget. One missing factor of the 64-blocksize, inflating $0.00195 \to 0.125$. The brief flagged
> its own suspicion; the paper wins. `brief-tooling-hardware.md:650` already had 0.127 correct.

Qwen3-8B NF4 base (DQ on): $8{,}190{,}735{,}360 \times 0.515869 = \mathbf{4.225\ GB\ (3.935\ GiB)}$. [DER]

> **⚠️ Honest default caveat — worth one course beat.** `bnb_4bit_use_double_quant` / `compress_statistics`
> defaults to **False** in both `bitsandbytes` and `BitsAndBytesConfig`. **Un-opted-in NF4 is 4.5 bits =
> 0.5625 B/param → 4.607 GB for Qwen3-8B, not 4.225 GB.** Every QLoRA memory figure in this course assumes the
> learner explicitly sets `bnb_4bit_use_double_quant=True`. Say so once. [VP — bitsandbytes source]

---

## 4. KV CACHE — [DER]

$$\text{KV bytes} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot b \cdot S \cdot B$$

**Note the $H_{kv}$, not $H$.** This is the single most common error in the field and all three source briefs
get it right in all three places. Do not regress it.

| Quantity | Qwen3-8B, bf16 |
|---|---|
| per token per layer | $2\times8\times128\times2 = \mathbf{4{,}096\ B = 4\ KiB}$ |
| **per token, all 36 layers** | $147{,}456\ \text{B} = \mathbf{144\ KiB/token}$ |
| at S = 40,960 (full context) | 6,039,797,760 B = **5.625 GiB** |
| at S = 32,768 | 4,831,838,208 B = **4.50 GiB** |
| at S = 131,072 (hypothetical YaRN) | **18.0 GiB** |
| **if it were MHA** ($H_{kv}=32$) | 576 KiB/token → **22.50 GiB** at 40,960. GQA saves **4×**, recovers **16.88 GiB** |
| KV @ 32k vs bf16 weights | 4.50 GiB / 15.26 GiB = **29%** |
| KV @ 131k vs weights | 18.0 GiB > 15.26 GiB — **the cache exceeds the model** |

---

## 5. COMPUTE AND SCALING — [DER]

| Quantity | Value |
|---|---|
| matmul FLOPs | $2mnp$ for $(m,n)@(n,p)$ — count mul and add separately, as vendors do |
| forward FLOPs/token | $\approx 2N$ ($N$ = **non-embedding** params) |
| **training FLOPs** | $\mathbf{C \approx 6ND}$ — derive the 6: forward 2, backward 4 |
| attention vs FFN crossover | attention FLOPs exceed FFN when $S > 1.5\,d_{ff} = \mathbf{18{,}432}$ tokens |
| naive score matrix, S=40,960, fp32 | $32 \times 40960^2 \times 4 = \mathbf{214.7\ GB}$ **per layer** |
| $\text{std}(q\cdot k)$ at $d_{head}=128$ | $\sqrt{128} = \mathbf{11.3137}$ |
| Qwen3-8B pretraining compute | $6 \times 6.95\times10^9 \times 3.6\times10^{13} = \mathbf{1.5\times10^{24}}$ FLOP **[D is [EST] — ~36T tokens is unverified]** |

**Pretraining on the Spark — the corrected sentence.** $1.5\times10^{24} / 1.25\times10^{14} = 381$ years — but
$1.25\times10^{14}$ is the **theoretical FP16-accumulate peak at 100% MFU**, which no machine achieves. At the
realistic ~62 TF FP32-accumulate roofline (§6) and ~40% MFU it is **~1,900 years**. The course says:

> **"At least four centuries — and that assumes the machine runs at its theoretical peak, which it will not.
> Realistically, closer to two millennia."**

Never "generously assume ~sustained." Assuming peak is generous **to the machine**, which makes 380 years a
**lower bound**, not a generous one. The conclusion is strengthened, not weakened, by honesty.

---

## 6. THE DGX SPARK — [VP] except where marked

### 6.1 Verified specs — NVIDIA `docs.nvidia.com/dgx/dgx-spark/hardware.html` + product page

| Spec | Value | Tag |
|---|---|---|
| Superchip | **NVIDIA GB10** Grace-Blackwell | [VP] |
| CPU | **20-core Arm**: 10× Cortex-X925 + 10× Cortex-A725 | [VP] |
| GPU | Blackwell, **6,144 CUDA cores** (48 SMs × 128), 5th-gen Tensor Cores | [VP] |
| **Compute capability** | **sm_121** (12.1) | [VP] — PyTorch 2.13 ships an explicit `121` entry; bitsandbytes targets `121` for ARM64 |
| Architecture | **aarch64** | [VP] |
| **Memory** | **128 GB LPDDR5X unified**, 256-bit, coherent CPU+GPU over NVLink-C2C | [VP] |
| **Memory bandwidth** | **273 GB/s** | **[VP] — CONFIRMED. brief-architectures' `[M]` doubt is CLEARED; brief-tooling was right.** |
| Storage | 1 TB or 4 TB self-encrypting NVMe | [VP] |
| OS | **NVIDIA DGX OS** (Ubuntu-based), **CUDA 13.0** | [VP] |
| Power | 240 W PSU; GB10 SoC 140 W TDP | [VP] |
| Thermal | LMSYS report **no throttling under sustained load** | [VP] |
| Price | ~$3,000–4,700 by SKU | [EST] |

### 6.2 The headline number is marketing — say so, kindly, once

**"1 PFLOP / 1000 TOPS" is FP4 *with 2:4 structured sparsity*.** [VP — NVIDIA's own footnote reads
"Theoretical FP4 TOPS **using the sparsity feature**".] The learner's weights are not 2:4 sparse. **Never size
a training run off this number.**

### 6.3 Dense BF16 — UNPUBLISHED. The course's most important honesty beat.

**NVIDIA publishes no dense BF16 figure for GB10.** Confirmed absent. [VP — absence verified]

**The inference chain — [INF]. The course prints this chain, labelled "inferred, not published":**

```
1,000 TFLOPS  FP4 sparse            (published, marketing)
÷ 2  (2:4 sparsity)      →   500    FP4 dense
÷ 2  (FP4 → FP8)         →   250    FP8 dense
÷ 2  (FP8 → BF16)        →   125    BF16 dense, FP16-accumulate
÷ 2  (consumer-Blackwell FP32-accumulate penalty)
                         →  ~62.5   BF16 dense, FP32-accumulate  ← what PyTorch actually does
```

**Two independent checks validate the chain and the course should show both:**
1. **The ÷2 sparsity step is measured, not assumed.** An NVIDIA-forum microbenchmark with a reproducible
   open-source tool measured **~511 TFLOPS dense NVFP4** (102% of the implied 500) and ~1,014–1,022 sparse
   across multiple units. [VP]
2. **The per-SM rate checks out.** 48 SMs × 4,096 FP4 FLOP/SM/clk × 2.6 GHz (observed load clock) = **511 TF** —
   matching the measurement. The same 48 × 1,024 × 2.6 GHz → **128 TF** BF16 (FP16-accum); 48 × 512 × 2.6 GHz →
   **64 TF** (FP32-accum). [INF, arithmetically consistent]

| Ceiling | Value | Tag | Use |
|---|---|---|---|
| **Practical dense-BF16 roofline for training** | **~62 TFLOP/s** | **[INF]** | **the course's roofline. Label it inferred.** |
| BF16, FP16-accumulate | ~125 TFLOP/s | [INF] | the *wrong* ceiling for a real training matmul — say so |
| FP4 dense | ~500–511 TFLOP/s | [INF] + [VP measurement] | |
| ~~31 TFLOPS FP32~~ | **RETIRED** | | appears in brief-foundations and brief-pedagogy; unverified and unsourced. **Do not print.** |

> **This resolves a live community mystery and the course should say so.** Reports of "~60 TFLOPS measured BF16"
> on GB10 are widely blamed on missing sm_121-tuned kernels. **That is largely wrong: ~60 TF is ~96% of the
> 62.5 TF FP32-accumulate ceiling.** The hardware is performing correctly. 125 TF is the ceiling only for
> FP16-accumulate, which is *not* what a plain PyTorch training matmul uses. **Quoting 125 TF as "the roof" for
> an FP32-accumulate GEMM would make the learner's measurements look like a 50% failure when they are at ~96% of
> achievable.**

> **⚠️ 62 TF vs 125 TF is NOT settled — a NEW empirical datapoint pulls the other way. Flag it, do not launder
> 62 TF into a fact.** The NVIDIA blog's **SDXL 1.0 BF16** benchmark (7 img/min = 8.571 s/img, 50 steps, TensorRT)
> is a live cross-check on the BF16 ceiling. The SDXL UNet is **5.977 TFLOP/forward @1024², batch 1** (measured
> directly, `torch.FlopCounterMode`, 2.567 B-param config). If CFG was **on** (2 forwards/step, SDXL's default):
> $50\times2\times5.977\text{e}12 / 8.571 = \mathbf{69.7\ TF/s}$ sustained BF16 — a hard *lower bound* on usable
> BF16 throughput, and it **exceeds the 62.5 TF FP32-accumulate ceiling (111%), which would rule that ceiling
> out** and point to ~125 TF (FP16-accumulate). If CFG was **off**, it is 34.9 TF/s and says nothing. **The blog
> does not state CFG**, so this is suggestive, not conclusive — but it is a concrete reason to treat **~62 TF and
> ~125 TF as genuinely unresolved [INF]**. The SDXL datapoint leans toward 125 TF. **`09_spark_capability_probe.py`
> (exercise #2) settles it by measurement — which is exactly why the course measures rather than asserts.**

### 6.4 THE RIDGE POINT — [INF], working value 227, but flagged

$$I^\star = \frac{P_{\text{peak}}}{BW} = \frac{62\times10^{12}}{273\times10^{9}} = \mathbf{227\ \text{FLOP/byte}}\quad(\text{if }P_{\text{peak}}\approx62\text{ TF});\qquad \frac{125\times10^{12}}{273\times10^{9}} = 458\ \text{FLOP/byte}\quad(\text{if }\approx125\text{ TF})$$

> **⚠️ The ridge number is [INF] and depends on the unresolved 62-vs-125 TF ceiling (§6.3).** Working value:
> **227** (from ~62 TF). If the true usable BF16 ceiling is ~125 TF — which the SDXL datapoint leans toward —
> the ridge is **458**. **The course prints 227 with the inference label and names 458 as the live alternative;
> it does not print either as fact.** Measuring the ceiling (exercise #2) fixes the ridge.
>
> **The qualitative D-15 contrast is robust to this either way:** LLM decode ($I=1$) stays hundreds-fold
> bandwidth-bound below **both** 227 and 458; diffusion ($I=4096$) stays comfortably compute-bound above **both**.
> The "opposite verdicts" story does not depend on which ceiling is right — only the printed ridge number does.

> **RETIRED, all three:** ~~3,663 FLOP/byte~~ (brief-pedagogy §7.3 — uses the 1-PFLOP sparse-FP4 marketing
> number), ~~"Ferrari with a garden hose"~~ (rhetorically great, arithmetically unfair — it divides a sparse-FP4
> peak by a real bandwidth). **Note: brief-tooling's 458 is NOT retired outright — it is the correct ridge *if*
> the ceiling is 125 TF, which is now a live possibility.** The honest version is still damning either way:
> a bandwidth-starved machine balance of 227–458 FLOP/byte. Survives contact with his own measurement.

### 6.5 ★ The unifying formula — the deliberate LLM/diffusion contrast (D-15)

For a dense transformer at bf16, one forward pass:

$$I \;=\; \frac{\text{FLOPs}}{\text{bytes}} \;=\; \frac{2 N \cdot S_{\text{fwd}}}{2N} \;=\; \boxed{\,S_{\text{fwd}}\ \text{FLOP/byte}\,}$$

**Arithmetic intensity equals tokens per forward pass.** One formula, frozen, used in both tracks. **Ridge is
227 (working) or 458 (if the ceiling is ~125 TF) — see §6.4; the verdicts below hold under both.**

| Workload | $S_{\text{fwd}}$ | $I$ | vs ridge (227 / 458) | Verdict |
|---|---|---|---|---|
| **LLM decode, batch 1** | 1 | **1** | **227× / 458× below** | bandwidth-bound. **The Spark is weak.** |
| LLM decode, batch 32 | 32 | 32 | 7× / 14× below | still bandwidth-bound |
| LLM prefill, S=2048 | 2048 | 2048 | 9× above / 4.5× above | compute-bound |
| **Diffusion, 4096 latent tokens** | 4096 | **4096** | **18× / 9× above** | compute-bound. **The Spark is strong.** |

> **This is the engineered contrast the LLM brief asked for, and it is one formula, not two stories.** Same
> machine, same ridge, opposite verdicts, because the two tracks sit on opposite sides of $I^\star$. **General
> rule: at bf16, arithmetic intensity ≈ tokens per forward.** Fix the bytes/param and it generalizes:
> $I \approx S_{\text{fwd}} \times (2 / b_W)$.
>
> **Correction to brief-tooling §5.2:** it states decode has $I \approx 2$ FLOP/byte. It is **1** at bf16
> (2 FLOP per param ÷ 2 bytes per param). Off by 2×.

### 6.6 Decode roofline — the heuristic, honestly restated

$$\boxed{\ \text{decode tok/s} \;\approx\; 0.65 \times \frac{\text{BW (GB/s)}}{\textbf{weight bytes read per token (GB)}}\quad\text{— dense models, batch 1}\ }$$

**[EST] — the heuristic SURVIVES.** The earlier "REFUTED — only two of four measurements land in band" verdict
rested on a **strawman substitution**: it swapped in DeepSeek-R1-14B (batch 8) and gpt-oss-20B (MoE) for the
brief's two NVFP4 points and knocked *those* down. The brief's actual four dense batch-1 points are below, and
**all four land 60–75% MBU under a consistent decode-traffic byte count.** The DeepSeek and gpt-oss cases are the
two instructive **failure modes**, not failed supports.

**The real variable is the byte count, and it moves each point 15–25 pp.** The NVFP4 checkpoints were read from
their actual safetensors headers this session: they are **6.03 GB / 10.54 GB** (`nvidia/Llama-3.1-8B-Instruct-NVFP4`,
`nvidia/Qwen3-14B-NVFP4`), **not** the brief's assumed 4.5/7.5 GB — NVFP4 is not a uniform 4.5 bits/param
(`embed_tokens`+`lm_head` stay BF16, per-block FP8 scales add ~0.4–0.8 GB). Three defensible byte conventions
give three different MBUs for the *same* measurement:

| Convention | 8B FP8 | 70B FP8 | 8B NVFP4 | 14B NVFP4 | band | ×0.65 max err |
|---|---|---|---|---|---|---|
| params only | 60.3% | 69.8% | 63.9%* | 69.1%* | 60–70% | 7.8% |
| whole checkpoint file | 68.2% | 71.9% | 85.3% | 87.7% | 68–88% | 25.9% |
| **checkpoint − embedding table** | 60.3% | 69.8% | **70.5%** | **74.8%** | **60–75%** | **13.1%** |

\* using the brief's superseded 4.5/7.5 GB guess.

**The physically correct convention is the third:** decode reads every weight except the embedding lookup
(`embed_tokens` is *gathered* — one row per token, not streamed). Under it all four land **60–75%**, mean $k=0.688$,
and even a round **×0.65** is within **13%** worst-case. **The dominant uncertainty is the byte count, not the
coefficient** — whole-file vs decode-traffic moves a single point 15–25 pp. **That is the actual lesson:** the
heuristic is only as good as your bytes-per-token number, and for quantized models that is subtler than the file
size. Coefficient 0.65 is the round, slightly-conservative choice (mean 0.69); error bars **±10–15%**.

**The two failure modes — teach them as the punchline, not as refutations:**

| Case | Why it breaks the heuristic |
|---|---|
| **gpt-oss-20B (49.7 tok/s, Ollama)** | On its real 12.8 GB checkpoint the *implied* bandwidth is 637 GB/s = **234% of the 273 GB/s bus** — impossible. It is an **MoE** (~3.6B active of ~20.9B). On active weights (~1.9 GB) → 35% MBU. |
| **DeepSeek-R1-14B (83.5 tok/s)** | **batch 8, not 1.** Weight reads amortize over 8 streams; per-stream MBU 56.5%. Not a batch-1 datapoint. |

> **The MoE blow-up is a gift — ship it as the punchline:** *"If your heuristic predicts more than 100% of peak
> bandwidth, your model is an MoE."* One sentence teaches the heuristic, its domain, and MoE's memory/compute
> decoupling at once. And the byte-count subtlety is the deeper lesson: for a quantized model, weight-bytes-per-token
> is neither the file size nor params×bits — it is the file minus the embedding table.

### 6.7 Measured performance — [VP], use these verbatim

**Fine-tuning — NVIDIA developer blog, seq 2048, PyTorch:**

| Task | Throughput |
|---|---|
| Llama 3.2 **3B** full fine-tune | **13,519.54 tok/s** |
| Llama 3.1 **8B** LoRA (seq 2048, batch 4) | **6,969.59 tok/s** |
| Llama 3.3 **70B** QLoRA (seq 2048, batch 8) | **759.79 tok/s** |

> ⚠️ An aggregator circulates **"53,657 tok/s for Llama 3.1 8B LoRA."** NVIDIA's own blog says **6,969.59**.
> Trust the primary. brief-diffusion §15.3 quotes the inflated figure — **correct it.**

**Inference — LMSYS, SGLang, FP8 [VP]:**

| Model | Prefill | Decode b=1 | Decode b=32 |
|---|---|---|---|
| Llama 3.1 **8B** | 7,991 tok/s | **20.5** | **368** |
| Llama 3.1 **70B** | 803 tok/s | **2.7** | — |
| GPT-OSS 20B (MXFP4, Ollama) | 2,053 | 49.7 | — |

**Contrast [VP]:** RTX 5090 → 8,519 / 205. RTX Pro 6000 Blackwell → 10,108 / 215. **~4× faster decode.**

> **The batch-1 → batch-32 row is the single most instructive measurement available:** decode 20.5 → 368
> (**18×**) while prefill is **flat** (7,991 → 7,949). Prefill was already compute-bound and gained nothing;
> decode was bandwidth-bound and gained ~linearly. **That is §6.5's formula, measured, on his machine.**

**MBU check — [DER] from the above:**

**Byte column = decode-traffic (checkpoint − embedding table), the physically-correct convention (§6.6).**
NVFP4 checkpoint sizes are the **real safetensors totals** (6.03 / 10.54 GB), not the retired 4.5/7.5 GB guess.

| Model | Decode-traffic bytes | Ceiling 273/bytes | Measured | MBU |
|---|---|---|---|---|
| Llama 3.1 8B FP8 | ~8.0 GB (file ≈ params) | 34.12 | 20.5 | **60.3%** |
| Llama 3.1 70B FP8 | ~70 GB | 3.90 | 2.7 | **69.8%** |
| Llama 3.1 8B NVFP4 | 6.03 file − 1.05 embed = ~4.98 GB | 54.8 | 38.65 | **70.5%** |
| Qwen3 14B NVFP4 | 10.54 file − 1.56 embed = ~8.98 GB | 30.4 | 22.71 | **74.7%** |

**Qwen3-8B decode predictions — [DER]:**

| Precision | Weights | Ceiling | Expect (×0.65) |
|---|---|---|---|
| bf16 | 16.38 GB | **16.67 tok/s** | **~10.8 tok/s** |
| fp8 | 8.19 GB | 33.33 | ~21.7 |
| NVFP4 | ~4.10 GB | 66.6 | ~43 |

> **RETIRED:** brief-pedagogy's *"~19.5 tok/s"* — it is a 7B/14 GB ceiling, presented as a prediction, for a
> retired anchor. Use 16.67 (ceiling) / ~10.8 (expectation) for Qwen3-8B bf16, and **say which is which.**

### 6.8 ★ THE SPARK BUDGET — [MEA]. Not a constant. Do not fabricate one.

**The course does not assert a usable-memory budget.** It ships a script whose step 1 is:

```python
free, total = torch.cuda.mem_get_info()   # bytes
print(f"usable now: {free/2**30:.2f} GiB of {total/2**30:.2f} GiB")
```

and then computes **122.05 GiB** against *his* measured number.

**Why this is mandatory, not fastidious:** the source briefs assert three mutually inconsistent budgets —
128 GB, 119.2 GiB, "budget ~110 GB usable" — and **the course explicitly dares him to reproduce an OOM on
hardware he owns**. He will find the state allocates fine and OOMs for a different reason. A course that tells
a man with a DGX Spark what his DGX Spark has is a course that gets caught. **Measure. The number is his, the
beat is stronger, and it cannot be falsified by a unit convention.**

**RESOLVED BY MEASUREMENT — [MEA-DEV], 2026-07-16, on the learner's own box (`hardware-ground-truth.md` §2):**

```
/proc/meminfo MemTotal = 130,662,936,576 B = 121.6875 GiB (exactly)
torch.cuda.mem_get_info() total = 121.69 GiB   (unified — GPU sees the same pool)
```

| Budget | 122.05 GiB state | Verdict |
|---|---|---|
| ~~128 GiB physical~~ | ~~+5.95 — FITS~~ | **counterfactual — that memory does not exist for you** |
| **121.6875 GiB measured MemTotal** | **−0.36 GiB** | **DOES NOT FIT — misses by 0.3%, before activations** |
| MemAvailable at measurement time | 93.84 GiB | −28.2 GiB — and this is the realistic number with his ComfyUI up |

**The conclusion "full FT of an 8B does not fit on a Spark" is now MEASURED FACT, not inference.** The firmware/
driver carveout eats **6.3125 GiB (4.9%)** of the physical 128 GiB before a single byte is allocated; activations
add 2–6 GB on top of the 122.05 GiB state. The predict-then-measure page design STANDS — he still runs
`mem_get_info()` live and watches 122.05 miss 121.69 by a hair. The razor-thin −0.36 GiB margin is *more*
affecting than either fictional version (the 3 GB overshoot or the 6 GB of headroom), and the villain
generalizes: **published capacity is never usable capacity; the spec sheet is off by 4.9% before you start.**

### 6.9 aarch64 / sm_121 — the corrected story

> **[MEA-DEV] Confirmed on the learner's own box 2026-07-16:** `torch.cuda.get_device_capability(0)` → `(12,1)`;
> `torch.cuda.get_arch_list()` → `['sm_80','sm_90','sm_100','sm_110','sm_120']` — **no sm_121 entry, no PTX,
> and it runs anyway.** The installed stack is ComfyUI's venv only (torch 2.12.1+cu130, transformers 5.12.1,
> diffusers 0.39.0); **no system torch, no peft/trl/accelerate/bitsandbytes anywhere.** The setup page installs
> the LLM stack into a NEW venv — it must never touch ComfyUI's. See `hardware-ground-truth.md` §3.

| Claim | Verdict |
|---|---|
| PyPI `torch` aarch64 cu130 wheel runs on GB10 | ✅ — but via **sm_120→sm_121 binary compatibility only** |
| ~~"binary compat + PTX JIT"~~ | ❌ **REFUTED.** Release wheels ship **SASS-only, no PTX** (`_ptx_arches()` returns `set()` on release builds). |
| ~~"the first kernel JIT after a driver change is slow — that's the PTX compile"~~ | ❌ **Cannot happen on a stock PyPI wheel. There is no PTX to compile.** That story applies to nightlies and NGC containers only. **Delete it.** |
| PyTorch runtime accepts cc 12.1 | ✅ `DEVICE_REQUIREMENT[120] = _CompatInterval(start=120)`; `major==12 and minor>=0` passes, no warning |
| **bitsandbytes ships NATIVE sm_121** | ✅ **[VP]** — `build-cuda.sh`: ARM64 + CUDA 13 → `build_capability="75;80;90;100;110;120;121"`. x86 does **not** get 121. **This is what "Add release for DGX Spark" means, and it inverts the briefs' framing: the Spark is *better* served than an x86 box here.** Matters — the 70B QLoRA path depends on bitsandbytes. |
| **vLLM SM121** | ✅ **REFUTED as blocked.** Issue #31128 **closed** (PR #37700). Official vLLM DGX Spark blog (2026-06-01): runs the standard OpenAI-compatible server, **CUDA graphs on by default, no `--enforce-eager`.** ⚠️ **But the validated path is the container `vllm/vllm-openai:cu130-nightly`, pinned by digest. The PyPI aarch64 wheel is built against CUDA 12 and will fail on the Spark's CUDA-13-only stack.** |
| The real trap | **Any package pinning `libcudart.so.12` fails at import** on CUDA-13-only DGX OS — *before* kernel compatibility is even evaluated. **Narrower and sharper than "aarch64 is painful."** |

Wheels verified present for Linux aarch64: torch, transformers, peft, trl, diffusers, bitsandbytes, vllm. [VP]

---

## 7. LIBRARY VERSIONS — [VP], PyPI, 2026-07-16. Pin all of them.

| Package | Version | Released | Note |
|---|---|---|---|
| `torch` | **2.13.0** | 2026-07-08 | |
| `transformers` | **5.14.1** | **2026-07-16 (today)** | v5 — see below |
| `peft` | **0.19.1** | 2026-04-16 | `LoraConfig` **stable**; course code is safe |
| **`trl`** | **1.8.0** | 2026-07-09 | ⚠️ **brief-tooling's `0.29.1` is STALE.** Pin `trl>=1.8,<2`. |
| `diffusers` | **0.39.0** | 2026-07-03 | |
| `bitsandbytes` | **0.49.2** | **2026-02-16** | brief-tooling misread the year as 2025. Resolved. |
| Python | **3.12** | | transformers v5 needs ≥3.10; DGX OS ships 3.12 |
| CUDA | **13.0** | | |

### 7.1 transformers v5 — mostly SOFT, not the hard break the briefs assume

**⚠️ Do NOT trust the GitHub release-notes summary.** It claims `warmup_ratio` was removed; **it is still
present** in `training_args.py`. Verified against v5.14.1 source.

**Actually breaking:** TF/Flax/JAX **gone** · `encode_plus`/`batch_encode_plus` **removed** (call the tokenizer
directly) · legacy cache format, `BetterTransformer`, head masking/pruning, `torchscript`/`torch.fx` removed ·
`TrainingArguments` losses: `overwrite_output_dir`, `jit_mode_eval`, `tpu_num_cores`, `past_index`, `ray_scope`,
`mp_parameters` · safetensors only.

**The real hazard — silent default changes:** `report_to` now defaults to `"none"` (was `"all"`) · new
`use_cache` field defaults `False` · **model `dtype` loads as-saved** (was float32) · requires
`huggingface-hub>=1.5,<2`, `torch>=2.4`, Python ≥3.10.

**NOT breaking, contrary to brief-tooling:** `torch_dtype` is **soft-deprecated with a working BC shim**
(`logger.warning_once("torch_dtype is deprecated! Use dtype instead!")`) — old code runs · `batch_decode`
**still exists** · `warmup_ratio`, `warmup_steps`, `bf16`, `gradient_checkpointing`, `logging_dir` all intact.

### 7.2 trl v0→v1 is a small break

Per `MIGRATION.md`: *"if you're already on v0.29, the changes are minimal."* Only:
`GRPOConfig`/`RLOOConfig` `vllm_mode` default `"server"`→`"colocate"` · `SFTConfig` `packing="bfd-requeue"`→`"bfd_split"` ·
trainers no longer auto-strip `None` from datasets. **Bump the pin and move on.**

### 7.3 The trap the course MUST name — [VP]

`SFTTrainer` **still defaults to fp32** when `model` is a string and `model_init_kwargs` omits `dtype` —
*"This differs from `from_pretrained`, where (since Transformers v5) the dtype is inferred from the model
config."* **Two libraries, one script, opposite defaults.** Always pass `model_init_kwargs={"dtype": torch.bfloat16}`.

**`SFTConfig` defaults that differ from `TrainingArguments` [VP]:** `logging_steps=10` · **`gradient_checkpointing=True`**
(already on — do **not** present it as a discovery) · `bf16=True` · `learning_rate=2e-5` (**a full-FT default —
raise ~10× for LoRA**) · **`max_length=1024`** (silently truncates) · `loss_type="chunked_nll"`.

### 7.4 torchtune — [VP] the honest note

**Wound down, not actively maintained.** `torchforge` is **not** a drop-in successor (targets RL/agentic, explicitly
experimental). **Build nothing on torchtune.** But **do cite the pytorch.org DGX Spark blog** (full FT of
Llama-3.1-8B, seq 16,384, batch 16, bf16, ~80% of 128 GB, ~8 h/epoch) with the footnote: *the best-documented
DGX Spark fine-tuning result in existence runs on a library that is dying.* That is a genuine lesson.
**Its optimizer config is [MEA]/[EST] — the reconciliation (8-bit Adam + gradient checkpointing) is unverified.
Ship it as a puzzle for the learner to close by measurement, not as a resolved story.**

---

## 8. THE WORKED EXAMPLE — TN-1. FROZEN.

**Ruling D-02: TN-1 is the course's single canonical 9-parameter network. brief-training §5.4's rival
2-2-1 network is RETIRED.** Nine pages depend on these values. **Do not recompute, do not round differently,
do not re-derive.**

**Architecture:** 2 → 2 (tanh) → 1 (sigmoid) → BCE. **9 parameters.**

```
W₁ = [[ 0.5, -0.3],     b₁ = [ 0.1]      W₂ = [0.6, -0.9]     b₂ = 0.2
      [ 0.8,  0.2]]           [-0.1]

x = [1.0, 2.0]          y = 1            η = 0.1
```

### 8.1 Forward — [DER, mpmath 30 d.p.]

| Quantity | Value (4 d.p., as printed) | Exact |
|---|---|---|
| $z_{1,1}$ | **0.0** | see §8.4 — **it is not exactly 0 in float** |
| $z_{1,2}$ | **1.1** | 1.1 |
| $a_{1,1}$ | **0.0** | tanh(0) = 0 |
| $a_{1,2}$ | **0.8005** | 0.8004990218 |
| $z_2$ | **−0.5204** | −0.5204491196 (printed value corrected from −0.5205 — last-digit mis-rounding caught by the p.14 pilot builder 2026-07-16) |
| $\hat y$ | **0.3727** | 0.3727472205 |
| $\mathcal{L}$ | **0.9869** ⚠️ | 0.9868547821 |

### 8.2 Backward — [DER]

| Quantity | Value (as printed) | Exact |
|---|---|---|
| $\partial\mathcal{L}/\partial z_2 = \hat y - y$ | **−0.6273** | −0.6272527795 |
| $\partial\mathcal{L}/\partial W_2$ | **[0.0, −0.5021]** ⚠️ | [~0, −0.5021152364] |
| $\partial\mathcal{L}/\partial b_2$ | **−0.6273** | |
| $\partial\mathcal{L}/\partial \mathbf{a}_1$ | **[−0.3764, 0.5645]** ⚠️ | [−0.3763516677, 0.5645275015] |
| $\tanh'(0)$ | **1.0** | |
| $\tanh'(1.1)$ | **0.3592** | 0.3592013 |
| $\boldsymbol\delta_1$ | **[−0.3764, 0.2028]** | [−0.3763517, 0.2027790] |
| $\partial\mathcal{L}/\partial W_1$ | **[[−0.3764, −0.7527], [0.2028, 0.4056]]** ⚠️ | [[−0.3763517, −0.7527033], [0.2027790, 0.4055580]] |
| $\partial\mathcal{L}/\partial \mathbf{b}_1$ | **[−0.3764, 0.2028]** | |

### 8.3 One SGD step, η = 0.1, and the verify pass — [DER]

```
W₂' = [0.6, -0.8498]    b₂' = 0.2627
W₁' = [[0.5376, -0.2247], [0.7797, 0.1594]]    b₁' = [0.1376, -0.1203]
```

| Quantity | Value | Exact |
|---|---|---|
| $z_2$ (after) | **−0.2434** ⚠️ | −0.2433775221 |
| $\hat y$ (after) | **0.4395** | 0.4394541821 |
| $\mathcal{L}$ (after) | **0.8222** | 0.8222218173 |
| **$\Delta\mathcal{L}$** | **−0.1646** ⚠️ | −0.1646329648 |

**$\hat y$: 0.3727 → 0.4395. Loss: 0.9869 → 0.8222, down 0.1646. It learned. He did it with a pencil.**

> **⚠️ SIX CORRECTIONS to brief-pedagogy §9.2.** All are last-digit; all trace to the brief propagating 4-d.p.
> rounded intermediates. **The values above are correct. Use them.**
> `L 0.9870 → 0.9869` · `∂L/∂W₂ −0.5022 → −0.5021` · `∂L/∂a₁[1] 0.5646 → 0.5645` ·
> `∂L/∂W₁[0][1] −0.7528 → −0.7527` · `z₂(verify) −0.2433 → −0.2434` · `ΔL 0.1648 → 0.1646`

### 8.4 ★ THE "EXACTLY ZERO" IS FALSE. Mandated framing — every agent, no exceptions.

**brief-pedagogy §9.2 promises the learner will "see ∂L/∂W₂'s first entry come out *exactly zero*." He will not.**

```
z_{1,1} = 0.5*1.0 + (-0.3)*2.0 + 0.1        # the textbook 0.5 - 0.6 + 0.1 non-associativity demo
  float64 →  2.7755575615628914e-17         (not 0.0)
  float32 → -2.2351742e-08                  (not 0.0)
torch float32 dW2 → [1.4020191230201817e-08, -0.5021151900291443]
```

This beat's home is **page 15 (Autograd + the float that isn't zero — D-21a is authoritative;** this file's
earlier "page ~10" pointer predates the canonical table and is retired). It lands at the exact moment autograd
is supposed to stop being magic, under a course whose promise is **"three representations, one number."** Note
the ordering: the learner now FEELS underflow on page 9 (logsumexp) first, so this beat arrives as the *return*
of that lesson, not its preview. Left unpatched, ~9 fan-out agents each write "you will see 0.0" and the spine
visibly breaks on his own box.

**MANDATED PAGE TEXT — the spec ships this framing and agents may not regress it:**

> *"By hand you got exactly 0. Torch prints `1.4e-08`. **Both are right.** `0.5 − 0.6 + 0.1` is not zero in
> binary floating point — it is the textbook non-associativity demo, and you just hit it for free. This is the
> lesson of the logsumexp page come back around: **the math on paper and the math in the machine are not the
> same math.** The gradient is zero to seven decimals, the unit is dead, and the lesson
> stands — but the number on your screen will not be `0.0`, and a course that told you it would be is a course
> that has never run its own code."*

**Turn the bug into the asset. It is a better page than the one that was planned.**

### 8.5 The mandated autograd assertion

```python
assert torch.allclose(model[0].weight.grad,
    torch.tensor([[-0.3764, -0.7527], [0.2028, 0.4056]]), atol=1e-4)
```

**Note `−0.7527`, not `−0.7528`.** With the brief's `−0.7528` the error is **9.66e-05** against a budget of
**1.075e-04** — a **10% margin**, surviving only because `atol=1e-4` happens to be loose enough to absorb the
brief's own rounding error. At `atol=1e-5` it **fails**. With the corrected `−0.7527` the error drops to
**3.4e-06** — a 30× margin. **Fixing the rounding is what makes a nine-page thread stop depending on a tolerance
coincidence.**

### 8.6 THE GRADIENT-SPREAD BEAT — RESOLVED (remedy (a) adopted; D-02 §Z-1 closed)

**Remedy (a) is adopted and its nine gradients are now computed, frozen, and cross-checked (see §8.7).** TN-1
now has **one network, two inputs**: `x=[1.0,2.0]` (the dead-unit case, §8.1–8.5) and `x=[0.60,−0.20]` (the
gradient-spread case, §8.7). This is strictly better than either retired brief — same object, new guise.

**⚠️ A SEVENTH mis-rounding, not in the audit's six.** §8.6's old text (and D-02) claimed *"TN-1's first-input
spread is 0.6273 / 0.2028 = 3.09×."* **REFUTED.** That took the max over the *intermediates* ($|dz_2|=0.6273$),
not over the *nine parameter gradients*. Over the nine, $|\partial\mathcal{L}/\partial W_1[0][1]| = 0.752703$ is
the largest, so the **true first-input spread = 0.752703 / 0.202779 = 3.7119×** (min taken over the eight live
gradients; the ninth, $\partial\mathcal{L}/\partial W_2[0]$, is the dead-unit zero). But the first input is **not**
the beat: it has a dead unit, so its spread story is contaminated by the zero. **The gradient-spread beat uses
the second input (§8.7): 10.22×, all nine live.** `12.7×` is retired regardless — it belonged to the retired network.

### 8.7 SECOND CANONICAL INPUT — `x = [0.60, −0.20]`, TN-1's weights — FROZEN. [DER, three independent methods]

Same $W_1, b_1, W_2, b_2, y=1, \eta=0.1$ as §8. **No dead unit, no saturation** ($\tanh'=0.8150, 0.8928$ — both
healthy). This is the **retired §5.4's own input**, so §5.4's re-hosted prose changes zero numbers below the input.

**Forward — [DER]:**

| Quantity | Value (4 d.p., as printed) | Exact (12 s.f.) |
|---|---|---|
| $z_{1,1}$ | **0.46** | 0.46 (exact 2-d.p. decimal) |
| $z_{1,2}$ | **0.34** | 0.34 (exact 2-d.p. decimal) |
| $a_{1,1}$ | **0.4301** | 0.430084211402 |
| $a_{1,2}$ | **0.3275** | 0.327477394809 |
| $z_2$ | **0.1633** | 0.163320871513 |
| $\hat y$ | **0.5407** | 0.540739701539 |
| $\mathcal{L}$ | **0.6148** | 0.614817259104 |

**Backward — [DER]:**

| Quantity | Value (as printed) | Exact |
|---|---|---|
| $\partial\mathcal{L}/\partial z_2 = \hat y - y$ | **−0.4593** | −0.459260298461 |
| $\partial\mathcal{L}/\partial \mathbf{a}_1$ | **[−0.2756, 0.4133]** | [−0.275556179077, 0.413334268615] |
| $\tanh'(0.46), \tanh'(0.34)$ | **[0.8150, 0.8928]** | [0.815027571103, 0.892758555889] |
| $\boldsymbol\delta_1$ | **[−0.2246, 0.3690]** | [−0.224585883335, 0.369007704748] |

**The nine gradients — FROZEN. Verified three ways** (mpmath 50 d.p. analytic ↔ torch autograd f64, max Δ 5.3e-17
= f64 ulp ↔ central FD in mpmath at h=1e-20, max Δ 1.5e-31). Loss cross-check: torch `0.6148172591038918` vs
mpmath `0.61481725910389186`, Δ = 1.3e-17.

| param | 4 d.p. (as printed) | exact (10 d.p.) |
|---|---|---|
| $\partial\mathcal{L}/\partial W_1[0][0]$ | **−0.1348** | −0.1347515300 |
| $\partial\mathcal{L}/\partial W_1[0][1]$ | **+0.0449** | +0.0449171767 |
| $\partial\mathcal{L}/\partial W_1[1][0]$ | **+0.2214** | +0.2214046228 |
| $\partial\mathcal{L}/\partial W_1[1][1]$ | **−0.0738** | −0.0738015409 |
| $\partial\mathcal{L}/\partial b_1[0]$ | **−0.2246** | −0.2245858833 |
| $\partial\mathcal{L}/\partial b_1[1]$ | **+0.3690** | +0.3690077047 |
| $\partial\mathcal{L}/\partial W_2[0]$ | **−0.1975** | −0.1975206033 |
| $\partial\mathcal{L}/\partial W_2[1]$ | **−0.1504** | −0.1503973661 |
| $\partial\mathcal{L}/\partial b_2$ | **−0.4593** | −0.4592602985 |

**★ THE SPREAD BEAT — `10.22×`, all nine gradients live:**

$$\frac{\max|g|}{\min|g|} = \frac{0.459260}{0.044917} = \mathbf{10.2246\times}\qquad(\partial\mathcal{L}/\partial b_2 \text{ vs } \partial\mathcal{L}/\partial W_1[0][1])$$

**"The largest gradient is ten times the smallest"** — one order of magnitude, rounder and more quotable than the
retired 12.7×, with **no zero gradient**. This is the honest replacement for §5.4's "12.7× → therefore Adam" beat.

**Why this input and not a bigger-spread one (the honest reason — ship it):** since $\partial\mathcal{L}/\partial W_1[i][j]=\delta_1[i]\,x_j$,
spread is bought by widening the feature ratio $|x_0/x_1|$. An input like `[0.10,−2.00]` reaches 42× — but the
correct answer to *that* is **"normalize your inputs,"** not "use Adam." The decisive check: **even at a perfectly
balanced 1:1 feature ratio, the best achievable spread is still 8.64×** — the spread is *structural*, it survives
normalization, and that is what makes "therefore Adam" honest here. `[0.60,−0.20]` sits at a mild 3:1 ratio where
the multiplicative story (3.0× feature × 1.64× δ-ratio × layer term) is transparent rather than rigged.

**One SGD step, η = 0.1 — [DER]:** $\hat y$ 0.5407 → **0.5695**, $\mathcal{L}$ 0.6148 → **0.5630**, $\Delta\mathcal{L}$ = **−0.0518**.

**Mandated autograd assertion for the second input — assert at 5 d.p., NOT 4:**

```python
assert torch.allclose(model[0].weight.grad,
    torch.tensor([[-0.13475, 0.04492], [0.22140, -0.07380]]), atol=1e-5)
```

**⚠️ Do NOT reuse input 1's `atol=1e-4` here.** At 4 d.p. the assertion passes with max err **4.85e-05 — only a
2.1× margin** (because 0.0449 is small in magnitude), repeating exactly the tolerance-coincidence pattern §8.5
warns against. At 5 d.p. with `atol=1e-5` the margin is restored.

---

## 9. OTHER RECURRING VALUES

### 9.1 Loss anchors — [DER]

| Quantity | Value | Note |
|---|---|---|
| **Random-init loss, Qwen3-8B** | $\ln(151{,}936) = \mathbf{11.93}$ nats | **step-0 sanity check.** ⚠️ **NOT 11.76** — that is Llama-3's V=128,256. Retired. |
| Perplexity at init | **151,936** — exactly $V$, by construction | satisfying check |
| Coin-flip loss | $\ln 2 = \mathbf{0.6931}$ | |
| Random-guess, $V$ classes | $\ln V$ | **plot it as a horizontal line on every loss curve** |
| Well-trained 8B on English web text | $\mathcal{L} \approx 2.0$, PPL ≈ 7.4 | **[EST]** — corpus- and tokenizer-dependent |
| Narrow-domain LoRA | $\mathcal{L} \approx 0.5$–1.2 | **[EST]** |
| Memorization signal | train $\mathcal{L} < 0.1$ while eval rises | diagnostic |

### 9.2 The canonical logits — FROZEN. **[DER]**

$$z = [2.0,\ 1.0,\ 0.1],\quad \text{true class } c = 0$$

$e^z = [7.389056, 2.718282, 1.105171]$, $\sum = 11.212509$
$\hat y = [\mathbf{0.659001},\ 0.242433,\ 0.098566]$
$\mathcal{L} = -\ln(0.659001) = \mathbf{0.417030}$ nats
$\partial\mathcal{L}/\partial z = \hat y - y = [\mathbf{-0.340999},\ +0.242433,\ +0.098566]$ — **sums to zero, always.**

> **Two rivals RETIRED:** brief-foundations' 4-way `[2.0, 1.0, 0.1, −1.0]` → L = 0.4491, and brief-architectures'
> 3-way `[2, 1, 0]` → A = [0.665, 0.245, 0.090]. **Use `[2.0, 1.0, 0.1]` everywhere** — temperature demos, top-k,
> top-p, the √d_head demo, the CFG discussion.

### 9.3 Optimizer and training constants — [VP]/[EST] as marked

| Quantity | Value | Tag |
|---|---|---|
| max sigmoid derivative | $\sigma'(0) = \mathbf{0.25}$; $0.25^{10} = 9.54\times10^{-7}$; $0.25^{32} = 5.4\times10^{-20}$ (**below fp32 min-normal**) | [DER] |
| the amplifier numbers | $1.1^{100} = 13{,}781$; $0.9^{100} = 2.7\times10^{-5}$ | [DER] |
| He init at $d = 4096$ | $\text{Var}(w) = 2/4096$, **std = 0.0221** ≈ GPT-2's 0.02 | [DER] |
| momentum amplification | $\mu = 0.9 \Rightarrow 1/(1-\mu) = \mathbf{10\times}$ | [DER] |
| Adam bias correction at $k{=}1$ | $v$ is **1000×** too small ($\beta_2 = 0.999$) | [DER] |
| Adam $\beta_2$ memory | $1/(1-0.999) = \mathbf{1000}$ steps → **the mechanistic reason warmup exists** | [DER] |
| grad clip | **1.0** — near-universal | [VP] |
| healthy update ratio $\|\Delta\theta\|/\|\theta\|$ | $\approx 10^{-3}$ | **[EST]** — traces to Karpathy's CS231n notes; widely repeated, never formally validated. **Label it a rule of thumb.** |
| fp16 max | **65,504** → use bf16 | [VP] |
| float32 epsilon | $1.19\times10^{-7}$ → **gradient-check in float64** | [VP] |
| ridge / $\eta_{crit}$ (demo ravine $\lambda_{max}{=}20$) | $\eta_{crit} = 2/\lambda_{max} = \mathbf{0.1}$ exactly | [DER] |

### 9.4 The learning-rate table — [EST], with the principle that generates it

**The principle (state it; it is transferable and both tracks need it): fewer trainable parameters ⇒ higher LR.**
Full FT moves 8.19B parameters that already encode everything the model knows; it must tiptoe. Textual inversion
moves 768 and can stride.

| Method | LR | Trainable params |
|---|---|---|
| LLM full FT (8B) | **1e-5 – 2e-5** | 8.19e9 |
| Diffusion **full** FT / DreamBooth | **1e-6 – 5e-6** | 1.2e10 |
| **LLM LoRA** | **1e-4 – 3e-4** (~10× full-FT) | 4.4e7 |
| **Diffusion LoRA** | **1e-4** | 3e7 |
| Pretraining from scratch | 3e-4 (~1B) → 1e-4 (7-8B) | — |
| Textual inversion | **5e-4 – 5e-3** | **768** |

> **⚠️ Correction to brief-foundations §4:** it lists *"diffusion fine-tune η ≈ 1e-5 to 1e-6"* without qualifier.
> That is the **full** fine-tune figure. **Diffusion LoRA is 1e-4** — a 100× difference. Conflating them would
> make every diffusion LoRA in the course fail to learn. Fix.

**Why LoRA tolerates 10×:** $B$ is zero-init, so the adapter starts as an exact no-op and the effective update
is scaled by $\alpha/r$.

**2026 LLM AdamW defaults [VP]:** $\beta = (0.9, \mathbf{0.95})$ — not 0.999; text has huge gradient spikes from
rare tokens and a 1000-step memory poisons the denominator · $\varepsilon$ = 1e-8 · **wd = 0.1** · clip 1.0 ·
warmup 1–10% or 2000 steps · cosine to $\eta_{max}/10$.
**Diffusion:** $\beta = (0.9, 0.999)$ · wd 0.01 · **constant** LR is common practice for FLUX LoRA. **[EST]**
**No-decay list:** biases, norm gains, usually embeddings.

### 9.5 Tokenization — [VP]/[EST]

| Quantity | Value | Tag |
|---|---|---|
| Qwen3 vocab | **151,936** | [VP] |
| chars/token, English prose | $\rho \approx \mathbf{4}$ — the standing rule of thumb | [EST] |
| chars/token, code / non-Latin | $\rho \approx$ 1.5–2.5 | [EST] |
| logits tensor, B=1 S=2048 | $2048 \times 151936 \times 2 = \mathbf{622\ MB}$ — **for one tensor**; fp32 CE makes it 1.2 GB, and you need its gradient too | [DER] |

**That 622 MB is why TRL 1.8 defaults `loss_type="chunked_nll"`** — it drops `-100` positions *before* the
`lm_head` matmul. Real, current, and it explains an API default he will see. [VP]

### 9.6 Diffusion — see decisions.md D-16 for the FLUX benchmark correction

| Quantity | Value | Tag |
|---|---|---|
| SD1.5 / SDXL latent, 512² / 1024² | $[1,4,64,64]$ / $[1,4,128,128]$ | [VP] |
| SD1.5 compression | $786{,}432 / 16{,}384 = \mathbf{48\times}$ | [DER] |
| FLUX.1 latent, 1024² | $[1,16,128,128]$ = 262,144 → **12× compression** | [VP] |
| **FLUX.1 tokens after 2×2 patchify** | $64 \times 64 = \mathbf{4096}$ tokens of width 64 → projected to $d = 3072$ | [DER] |
| **FLUX.2 VAE spatial factor** | $f = 2^{4-1} = \mathbf{8}$ (`block_out_channels` length 4) | **[VP]** — shipped `vae/config.json`, `AutoencoderKLFlux2` |
| **FLUX.2 latent, 1024²** | $[1,\mathbf{32},128,128]$ = **524,288 elements** → **6.0× compression** | **[VP]** — `latent_channels: 32` |
| **FLUX.2 tokens after 2×2 patchify** | $[1,128,64,64]$ → $64\times64 = \mathbf{4096}$ tokens of **width 128** | [DER] — same token count as FLUX.1; doubles per-token width (64→128), not count |
| pixel-space attention, 1024² | $(1024^2)^2 = 1.10\times10^{12}$ entries = **2.2 TB** per head per layer per step, fp16 | [DER] |
| latent-space attention | $4096^2 = 1.68\times10^7$ = **33.5 MB** | [DER] |
| **the ratio** | $\mathbf{65{,}536\times} = 16^4$ | [DER] — **the justification for latent diffusion. Print it large.** |
| DDPM $\bar\alpha_{1000}$ | $\approx 4\times10^{-5}$, $\sqrt{\bar\alpha_T} = 0.0063$ — **nonzero terminal SNR** | [DER] |
| $\epsilon$-pred error amplification at $t{=}999$ | $1/0.0063 = \mathbf{159\times}$ | [DER] |
| FLUX.1-dev | 12B, $d=3072$, MMDiT, T5-XXL 4.7B + CLIP-L, **guidance-distilled** | [VP] |
| FLUX.2-dev | **32B** + **Mistral-3 24B VLM** encoder, **32-ch VAE, $f=8$** [VP, shipped config] | [VP] |
| FLUX.2-klein-4B | **Apache 2.0**, ~13 GB, **4 steps @ guidance_scale 1.0** | [VP, model card] |
| Z-Image-Turbo | **6B**, 8 NFE, Apache 2.0, **#1 open-weights** on AA Image Arena — **above FLUX.2-dev (32B)** | [VP] |
| FLUX.1-dev full FT | $12\times10^9 \times 16 = \mathbf{192\ GB}$ + ~4 activations ≈ **196 GB** | [DER] |
| FLUX.1-dev LoRA r=16 | ≈ **29–31 GB** (24 base + 0.36 adapter/state + 4–6 activations) | [DER] + [EST] activations |
| FLUX.1-dev LoRA params, r=16, attn only | **≈ 30M = 0.25% of 12B** → 60 MB bf16 | **[EST]** — medium confidence on the 19+38 block structure. Verify against `FluxTransformer2DModel`. |

**⚠️ FLUX.1 @ FP4 = "23 img/min / 2.6 s per 1K image" [VP] is for FLUX.1-**Schnell**, 4 steps, 1024², batch 1 —
NOT FLUX.1-dev at 28 steps.** [VP — confirmed against the NVIDIA developer blog this session.] See D-16.
brief-diffusion §8.4 and brief-tooling §3 both treat it as dev/28-step and build a "we predicted a published
benchmark" showpiece on it. **The showpiece is INVALID and must be deleted, not relabelled — you cannot honestly
predict a 2.6 s wall-clock when the FP4 dense ceiling is itself an unpublished inference (±factor 2).**

The arithmetic proves the dev/28 reading impossible: $28 \times 2\times12\text{e}9\times4096 = 2.75\text{e}15$ FLOP
$\div 2.6$ s = **1.06 PFLOP/s** — 106% of the *sparse-FP4 marketing peak*, on a workload using neither FP4 nor
sparsity. The numerator (dev/28) and the measurement (Schnell/4) are unrelated quantities; their "agreement" was
coincidence.

**The honest FLOP count for the correct variant (Schnell, 4 steps) — [DER]:** per denoising step @1024² (4096
img + 512 txt tokens): linear/matmul $2NS = 1.097\text{e}14$; attention $4S^2 d\cdot57 = 1.487\text{e}13$ (11.9%);
**total/step $= 1.245\text{e}14$ FLOP → 4 steps $= 4.98\text{e}14$ FLOP.** Implied sustained throughput
$= 4.98\text{e}14 / 2.609 = \mathbf{191\ TF/s}$ (or 150 TF/s using only the $2NS_{\text{img}}$ term). Against the
inferred ~500 TF dense-FP4 ceiling that is **30–38% MFU — physically plausible** (vs the dev/28 reading's demanded
1341 TF/s). Analytical param count for the reconstruction closes to 11.83 B vs the real 11.9 B. **Replace the
showpiece with `bench_spark.py`: the learner runs FLUX.1-dev at 28 steps himself (bf16 and FP4) and checks his
own prediction against both rooflines. Label the NVIDIA number "FLUX.1-Schnell, 4 steps, 1024², batch 1, FP4" everywhere.**

**SDXL 1.0 @ BF16 = 7 img/min = 8.571 s/img** [VP] — **50 steps, batch 2, TensorRT** (CFG not stated). The SDXL
UNet is **5.977 TFLOP/forward @1024², batch 1** [DER, `torch.FlopCounterMode`]. This is a live cross-check on the
BF16 ceiling (§6.3): CFG-on → 69.7 TF/s, which would rule out the 62.5 TF FP32-accum ceiling and lean toward 125 TF.

### 9.7 Number formats — [VP]/[EST]

| Format | Bits | Block | Scale | Quality vs FP16 | Tag |
|---|---|---|---|---|---|
| bf16 | 16 | — | — | baseline; **8 exp bits = fp32's range**, 7 mantissa | [VP] |
| fp16 | 16 | — | — | **5 exp bits, max 65,504** → needs GradScaler. **Never use it on Blackwell.** | [VP] |
| FP8 E4M3 | 8 | per-tensor/channel | fp32 | 0.5–2% degradation | [EST] |
| **MXFP4** | 4 (E2M1) | 32 | **E8M0 — powers of two only** | significant drop | [VP] |
| **NVFP4** | 4 (E2M1) | **16** | **FP8 micro-scale + fp32 global** | <1% on many tasks; **88% lower quantization error than MXFP4** | [EST — vendor] |
| NF4 | ~4.13–4.25 | 64 | fp32 → 8-bit (double quant) | **a training format, not a serving format** | [VP] |

> ⚠️ **The "AWQ 95% / GGUF 92% / GPTQ 90%" retention figures are from SEO-grade sources, are mutually
> inconsistent, and MUST NOT be printed.** The *ordering* is probably right. **Make measuring them an exercise.**
>
> **The NVFP4-vs-MXFP4 story is worth telling properly and it has a one-sentence explanation:** MXFP4 forces
> every block's scale to a power of two, so a block wanting a scale of 3 rounds to 2 or 4 and eats up to 33%
> error *before quantizing a single weight*. NVFP4 allows an arbitrary FP8 scale and halves the block to 16.
> **That's the whole difference, and it's worth 88% of the error.** General lesson: **quantization error is
> dominated by scale granularity, not mantissa bits.**

---

## 10. WHAT IS NOT A CONSTANT — the [MEA] list

The course ships a script for each. **No agent may print a number for these.**

| # | Quantity | Script | Why measured |
|---|---|---|---|
| 1 | **Usable memory on his box** | `00_verify_env.py`, `05_memory_ledger.py` | §6.8. The whole 122-vs-128 beat depends on it being his number. |
| 2 | **Dense BF16 TFLOP/s on GB10** | `09_spark_capability_probe.py` | Unpublished, and **genuinely unresolved: ~62 TF (FP32-accum) vs ~125 TF (FP16-accum) — the SDXL BF16 datapoint leans 125 TF, community GEMM reports lean 62 TF** (§6.3). **Have him run a big BF16 GEMM against both ceilings in both accumulate modes.** Whichever it lands at, understanding *why* teaches accumulate precision better than any paragraph — and it fixes the ridge (227 vs 458). The corpus's biggest gap, its best lab. |
| 3 | Achieved memory bandwidth | `09_...` | 273 GB/s is peak theoretical |
| 4 | Activation memory | `05_memory_ledger.py` | $c \approx 16$ is a rule of thumb, not a derivation |
| 5 | His decode tok/s vs the roofline | `02_kv_cache_and_roofline.py` | reproduces the 60–75% MBU on his own box; also lets him see decode-traffic bytes = file − embedding table (§6.6) |
| 6 | Quantization quality retention | `08_quantization_lab.py` | §9.7 |
| 7 | LoRA rank ablation | `13_ablate.py` | he can run 6× 12-min jobs |
| 8 | The torchtune 8B/16K reconciliation | `05_...` | §7.4 |
| 9 | FLUX step time at bf16 vs FP4 | `bench_spark.py` | §9.6's correction |

> **The meta-thread — the course's actual destination:** in artifact #1 he counts his tokens; in #5 he computes
> his memory; from §6.7's table he predicts his wall-clock; in #11 he runs it **and the prediction is right.**
> Then in #2/#14 he predicts his serving speed from the roofline **and that's right too.** *A learner who can
> predict his own hardware's behaviour from first principles before running anything* is a better destination
> than "can operate the tools." Build for it.

---

## 11. CHANGE LOG — corrections applied to the briefs by this file

| # | Was | Now | Source |
|---|---|---|---|
| 1 | Llama-3.1-8B / 7B anchors | **Qwen3-8B, everywhere** | D-01 |
| 2 | TN-1's six roundings | corrected §8 | arithmetic audit |
| 3 | "exactly zero" gradient | **false in every float** — mandated framing §8.4 | arithmetic audit |
| 4 | two rival 9-param networks | **TN-1 only** | D-02 |
| 5 | 224× / 417× optimizer shrink | **187×** (= the parameter ratio, on the Qwen3 anchor) | D-05 |
| 6 | 130 GB → 14.3 GB | **131.05 GB → 17.08 GB**, state-to-state | D-03 |
| 7 | 131 GB vs 128 GB "3 GB over" | **unit artifact.** 122.05 GiB vs a **measured** budget | D-04 |
| 8 | 70B: 280M / 4.5 GB / 41 GB | **207,093,760 / 3.31 GB / ≈39.7 GB** | arithmetic audit |
| 9 | 1 PFLOP / 125 TF / 31 TF roofline | **~62 TF [INF]**, chain shown | specs audit |
| 10 | ridge 3,663 / 458 FLOP/byte | **227 working [INF]; 458 live if ceiling is 125 TF** — §6.4 | D-14 + Task 3 |
| 11 | "60–69% MBU across four measurements" | **heuristic SURVIVES**: four dense points land 60–75% under decode-traffic bytes; MoE + batch-8 are the two failure modes; MoE case is the punchline | specs audit + Task 1 resolution |
| 12 | trl 0.29.1 | **trl 1.8.0** | specs audit |
| 13 | "torch runs via PTX JIT" | **SASS-only; no PTX in release wheels** | specs audit |
| 14 | vLLM blocked on SM121 | **supported — via the cu130 container** | specs audit |
| 15 | ln V = 11.76 | **11.93** (Qwen3) | D-01 |
| 16 | three rival canonical logit vectors | **[2.0, 1.0, 0.1]** | D-08 |
| 17 | "FLUX 2.6 s/image predicts our arithmetic" | **Schnell/4-step. Showpiece DELETED → `bench_spark.py`** | D-16 + Task 3 |
| 18 | "~70% of params are FFN" | **66.4% of model / 78.26% of block** | D-01 |
| 19 | diffusion FT η = 1e-5–1e-6 (unqualified) | **full FT 1e-6–5e-6; LoRA 1e-4** | D-09 |
| 20 | 53,657 tok/s Llama-8B LoRA | **6,969.59** (primary source) | specs audit |
| 21 | "12.7× gradient spread" | **retired**; §8.6/§8.7 resolved: TN-1 second input `[0.60,−0.20]` → **10.22×** | D-02 + Task 1 resolution |
| 22 | QLoRA NF4 "0.516 vs 0.53 unresolved" | **0.516 B/param (4.127 bits) [VP]. 0.53 REFUTED** — brief's arithmetic error (÷256 not ÷16384) | Task 2 resolution |
| 23 | TN-1 first-input spread "3.09×" | **3.7119×** over the nine gradients (7th mis-rounding); but the spread beat uses the 2nd input | Task 1 resolution |
| 24 | FLUX.2 VAE "$f{=}8$, 32ch — unverified" | **[VP]** from shipped `vae/config.json` + `AutoencoderKLFlux2`; 6.0× compression, 4096 tokens | Task 2 resolution |
| 25 | NVFP4 checkpoint sizes 4.5 / 7.5 GB | **real safetensors 6.03 / 10.54 GB**; decode-traffic = file − embed table | Task 1 resolution |
| 26 | BF16 ceiling "~62 TF, highest-value correction" | **62 vs 125 TF UNRESOLVED [INF]**; SDXL BF16 datapoint (69.7 TF/s if CFG-on) leans 125 TF; measure it | Task 3 resolution |
