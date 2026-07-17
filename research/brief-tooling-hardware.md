# Research Brief: Tooling, Code, and Hardware Reality

> **⚠️ THIS BRIEF PREDATES TWO VERIFICATION PASSES. `constants.md`, `decisions.md`, `notation.md` OVERRIDE IT.**
> Load-bearing corrections: **(1)** **trl 0.29.1 → 1.8.0**; the "prefer plain `Trainer`" advice is OVERRULED —
> use `SFTTrainer` (D-10f, §4.3). **(2)** the **PTX-JIT story is FALSE for release wheels** — they ship SASS-only,
> no PTX; torch runs via sm_120→sm_121 *binary* compat; the real trap is `libcudart.so.12` pins failing at import
> (D-10h, §4.4/5.x). **(3)** vLLM is **unblocked** (cu130 container, by digest); **bitsandbytes ships native
> sm_121** into the aarch64+CUDA-13 wheel (D-10i/j). **(4)** transformers v5 break is **mostly SOFT** — `torch_dtype`,
> `batch_decode`, `warmup_ratio` all survive; hazard is silent default changes (D-10g, §4.1). **(5)** ridge is
> **227 working / 458 if ceiling is 125 TF** ([INF], not settled); QLoRA overhead **0.127 bits** here was RIGHT
> (promote to [VP]). **(6)** anchor is **Qwen3-8B**. See `decisions.md` §D-20.

**For:** curriculum architect, ANN course (high-school algebra → LLM/diffusion fine-tuning)
**Date:** 2026-07-16
**Scope:** the real-code layer, the environment, the DGX Spark, the memory budget, failure modes
**Confidence key used throughout:** `[VERIFIED]` = fetched from primary source this session · `[REPORTED]` = from a secondary source, plausible, not primary-confirmed · `[UNVERIFIED]` = my prior, treat as a claim to check before shipping

---

## 0. The framing this brief argues for

The learner already runs ComfyUI on a DGX Spark. He does not need to be sold on "AI is cool." He needs the **mechanical chain of custody** from the math on the page to the bytes on his GPU. The single most valuable thing this course's code layer can do is make that chain *unbroken*: every number the course computes by hand should be reproducible by a script he runs, and every script should print numbers that match the page.

The organizing intuition for the whole tooling track:

> **A neural network training run is a memory allocation problem wearing a math costume.** The math tells you *what* to compute; the memory budget tells you *whether you can*. Every practical decision — batch size, LoRA rank, precision, checkpointing — is a trade in one currency: bytes.

Lead with that. It is the thread that makes the tooling track feel like one subject instead of a pile of library trivia.

### The second organizing intuition (the DGX Spark one)

> **The Spark has a huge warehouse and a narrow loading dock.** 128 GB of memory means enormous models *fit*. 273 GB/s of bandwidth means moving their weights is slow. Capacity is generous; bandwidth is the bottleneck. This single fact predicts almost everything the machine is good and bad at.

This should be stated on the hardware page and then *referred back to* every time a benchmark number surprises the learner.

---

## 1. Version reality, July 2026

Verify these again at build time. Library churn is the enemy of a course with real code.

| Package | Version | Date | Confidence | Note |
|---|---|---|---|---|
| `torch` | **2.13.0** | 2026-07-08 | [VERIFIED] PyPI + release notes | FlexAttention on MPS is the headline; nothing in the core teaching surface (tensors/autograd/nn.Module) changed |
| `transformers` | **5.14.1** | 2026-07-16 | [VP] PyPI | **v5 is a MOSTLY-SOFT break (D-10g) — `torch_dtype`/`batch_decode`/`warmup_ratio` all survive; the hazard is silent default changes. See §4.1.** |
| `trl` | ~~**0.29.1**~~ → **1.8.0** | 2026-07-09 | **[VP — STALE FIXED]** PyPI | **This brief's 0.29.1 is STALE. Pin `trl>=1.8,<2`.** The v0→v1 break is small (`MIGRATION.md`: "if you're already on v0.29, the changes are minimal"). See `constants.md` §7 / decisions D-10f. |
| `peft` | **0.19.1** | 2026-04-16 | [VERIFIED] PyPI | |
| `diffusers` | **0.39.0** | 2026 | [VERIFIED] PyPI (date not exposed) | |
| `bitsandbytes` | **0.49.2** | [REPORTED] date field looked wrong in the fetched page (said Feb 2025 for 0.49.2, which cannot be right given 0.49.0 added CUDA 13) — **re-verify** | | aarch64 + sm_121 story in §5.4 |
| `accelerate`, `datasets` | not separately verified | | [UNVERIFIED] | pin at build time |
| `torchtune` | **winding down / not actively maintained** | 2025→ | [REPORTED], strong signal | See §4.6 — this is a trap, NVIDIA's own DGX Spark fine-tuning blog uses it |

**Python:** transformers v5 requires **Python ≥ 3.10** and **torch ≥ 2.4** [VERIFIED via migration guide summary]. DGX Spark ships Python 3.12 [REPORTED]. Target **3.12** for the course.

### 1.1 The pinning policy the course must adopt

Do **not** write `pip install transformers`. Write a lockfile. The course should ship one `pyproject.toml` + `uv.lock` and state, in a box:

> These pins are from July 2026. If you are reading this later and something errors, the fix is almost never "the math changed." Check `uv lock --upgrade` last, not first.

And every code page should carry a one-line **"verified against"** stamp: `torch 2.13.0 · transformers 5.14.1 · peft 0.19.1 · trl 1.8.0 · CUDA 13.0` (**trl bumped from the stale 0.29.1**).

---

## 2. Teaching the *code*, not just the math

The learner is "rusty but once trained." He can read a for-loop. What he will *not* have intuition for — and what every PyTorch tutorial assumes and never says:

### 2.1 The five things that actually confuse people about PyTorch

**(a) A tensor is a struct, not an array.**
Intuition first: *A tensor is a flat 1-D block of numbers plus a set of instructions for how to pretend it has a shape.*

Concretely, `x = torch.zeros(2, 3, 4)` is:
- `storage`: 24 contiguous float32 values = 96 bytes
- `shape`: `(2, 3, 4)`
- `stride`: `(12, 4, 1)` — how many elements to skip per index step
- `dtype`, `device`, `requires_grad`

The element at `x[i, j, k]` lives at storage offset $i\cdot 12 + j\cdot 4 + k\cdot 1$. In general for stride $s = (s_0, \dots, s_{n-1})$ and index $(i_0,\dots,i_{n-1})$:

$$\text{offset} = \sum_{d=0}^{n-1} i_d \, s_d$$

*Units:* offset in **elements** (multiply by `dtype.itemsize` bytes for the byte offset).

**Why this earns its place:** it explains, in one stroke, why `.T` and `.view()` and slicing are free, why `.contiguous()` sometimes costs a copy, and why `.reshape()` sometimes silently copies and sometimes doesn't. Demo it: `x.T.stride()` is `(1, 12, 4)`-ish — same storage, different instructions.

*Misconception:* "`.view()` and `.reshape()` are the same." Correction: `view` **requires** compatible strides and errors otherwise; `reshape` falls back to a copy. `.T.view(-1)` errors; `.T.reshape(-1)` works and costs a memcpy. Show both, show the error text.

**(b) Broadcasting is not magic, it's a rule.**
Intuition: *Align shapes from the right. A dimension of size 1 gets stretched for free (stride 0). Anything else must match.*

$$(B, 1, D) + (1, T, D) \rightarrow (B, T, D)$$

*Misconception:* the classic silent bug — `pred` is `(N, 1)`, `target` is `(N,)`, so `pred - target` broadcasts to `(N, N)` and your MSE is computed over $N^2$ pairs. It trains. Badly. It never errors. **This deserves a warning box with a worked demo**: N=4, show the (4,4) matrix appearing.

**(c) `requires_grad` is a property of *leaves*, and the graph is rebuilt every forward.**
Intuition: *Autograd is a tape recorder that starts blank each forward pass and is erased when you call `.backward()`.*

*Misconceptions, all real:*
- "The graph persists." No — `.backward()` frees it. A second `.backward()` errors with *"Trying to backward through the graph a second time."* Fix: usually you didn't mean to; occasionally `retain_graph=True`.
- "`.grad` is overwritten each backward." **No — it ACCUMULATES.** This is why `optimizer.zero_grad()` exists and why forgetting it is the single most common training bug. Correction that fixes it: show `loss.backward(); loss.backward()` produces `2×` the gradient. Make them see the doubling.
- "`with torch.no_grad()` makes it faster to train." No — it makes it impossible to train. It's for inference and for in-place parameter updates.
- "`.detach()` and `.item()` are the same." `.detach()` returns a tensor sharing storage, no graph; `.item()` returns a Python float and syncs the GPU (a hidden performance cliff in a logging loop).

**(d) Device is not a setting, it's part of the type.**
Intuition: *A CPU tensor and a CUDA tensor are different types that refuse to talk.*
`RuntimeError: Expected all tensors to be on the same device` is the #1 first-week error. Teach the discipline: **one `DEVICE` constant at the top, `.to(DEVICE)` on the model once, `.to(DEVICE)` on each batch in the loop, never anywhere else.**

Note for the Spark: unified memory means `.to("cuda")` is *not* a PCIe copy — but PyTorch still tracks device and still errors. The check is logical, not physical. This is worth a sentence; it will otherwise confuse him.

**(e) `nn.Module` is a parameter registry with a `__call__`.**
Intuition: *A Module is a Python object that keeps a list of tensors it owns, so `.parameters()`, `.to()`, `.state_dict()`, and the optimizer all work without you tracking anything.*

*Misconception:* "I'll store my weights in a plain Python list." They won't move with `.to(device)`, won't appear in `.parameters()`, won't save. Correction: `nn.Parameter`, `nn.ModuleList`, `register_buffer` (for non-trained state like a running mean or a causal mask).

*Misconception:* "call `model.forward(x)`." Use `model(x)` — `forward` skips hooks. Minor, but it's the kind of thing that quietly breaks LoRA later, since PEFT works by hooks/wrapping.

### 2.2 The training loop, annotated

This exact block should appear once, early, be memorized, and then be *referred to by line* forever after:

```python
model.train()                              # 1. dropout/BN into training mode
for epoch in range(EPOCHS):
    for xb, yb in loader:                  # 2. DataLoader yields batched tensors
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)   # 3. gradients ACCUMULATE - clear them
        pred = model(xb)                   # 4. forward: builds the tape
        loss = criterion(pred, yb)         # 5. scalar
        loss.backward()                    # 6. walk the tape backward, fill .grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 7. optional
        optimizer.step()                   # 8. p -= lr * f(p.grad)
```

Eight lines. Every subsequent framework (Trainer, SFTTrainer, accelerate) is **this loop with the eight lines hidden**. Say that explicitly. The learner's anxiety about high-level libraries dissolves the moment he knows there's an eight-line loop under the hood and he could write it himself.

*`set_to_none=True` note:* it's the default since torch 2.0 [UNVERIFIED — check]; it sets `.grad = None` rather than zeroing, saving a kernel launch and a little memory. Mention once, don't dwell.

### 2.3 DataLoader — what to actually say

Intuition: *The GPU is a chef who cooks in 20 ms. The DataLoader is the kitchen porter. If the porter is slow, the chef stands idle no matter how fast he cooks.*

The parameters that matter and nothing else:
- `batch_size` — see §6, this is the memory dial
- `shuffle=True` for train, `False` for eval (the correction: *why* — SGD's convergence proof assumes i.i.d. sampling; sorted data gives you correlated gradients and the loss curve goes lumpy)
- `num_workers` — on the Spark, start at **8** (20 CPU cores; leave headroom). `num_workers=0` means loading blocks the training step.
- `pin_memory=True` on discrete GPUs. **On the Spark this is near-pointless** — unified memory, no PCIe hop. Honest note: it's not harmful, it's just not the win it is on an x86 + PCIe box. [UNVERIFIED whether it's actively harmful on GB10; do not overclaim.]
- `drop_last=True` when your loss/batchnorm can't handle a ragged final batch
- `collate_fn` — the place where padding happens. This is where the LLM track's "why is my sequence length 512 when I asked for 128" bug lives.

*Misconception:* "`num_workers=32` is faster." Each worker forks the process and copies the dataset object. More workers = more RAM and more contention. Measure it — this is a 90-second demo (`time` a fixed number of steps at `num_workers` ∈ {0, 2, 4, 8, 16}).

---

## 3. The code artifact ladder

This is the core recommendation of the brief. The progression is **numpy → PyTorch → real fine-tune**, and each rung must *prove something the previous rung asserted*.

### Rung 1 — `01_backprop_numpy.py`
**Purpose:** prove the math is real. No autograd. Hand-derived gradients for a 2-layer MLP on a 2-D toy problem.

**Why it must exist:** the learner will spend the rest of his life calling `.backward()`. He gets exactly one chance to know what it does. If the course skips this, `.backward()` is magic forever and every debugging session is superstition.

Concretely: 2-in / 8-hidden (tanh) / 1-out, binary cross-entropy, on the two-moons dataset (n=200). Parameter count: $2{\times}8 + 8 + 8{\times}1 + 1 = 33$. Small enough to print every gradient.

The forward:
$$z_1 = W_1 x + b_1,\quad a_1 = \tanh(z_1),\quad z_2 = W_2 a_1 + b_2,\quad \hat y = \sigma(z_2)$$

Shapes: $x \in \mathbb{R}^{B\times 2}$, $W_1 \in \mathbb{R}^{2\times 8}$, $b_1 \in \mathbb{R}^{8}$, $W_2 \in \mathbb{R}^{8\times 1}$, $b_2 \in \mathbb{R}^{1}$, $\hat y \in \mathbb{R}^{B \times 1}$.

The backward, written out because *writing it out is the point*:
$$\delta_2 = \hat y - y \quad (B\times1)$$
$$\frac{\partial L}{\partial W_2} = \frac{1}{B} a_1^\top \delta_2 \quad (8\times1), \qquad \frac{\partial L}{\partial b_2} = \frac{1}{B}\sum_b \delta_{2,b}$$
$$\delta_1 = (\delta_2 W_2^\top) \odot (1 - a_1^2) \quad (B\times8)$$
$$\frac{\partial L}{\partial W_1} = \frac{1}{B} x^\top \delta_1 \quad (2\times8), \qquad \frac{\partial L}{\partial b_1} = \frac{1}{B}\sum_b \delta_{1,b}$$

($\odot$ = elementwise; $1-a_1^2$ is $\tanh'$; the $\frac{1}{B}$ is the mean reduction.)

**The mandatory feature: a gradient check.** This is the moment the course earns trust.
$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \varepsilon e_i) - L(\theta - \varepsilon e_i)}{2\varepsilon}, \quad \varepsilon = 10^{-5}$$
Report the relative error $\frac{|g_{\text{analytic}} - g_{\text{numeric}}|}{\max(|g_a|,|g_n|) + 10^{-8}}$. **It should be < $10^{-7}$ in float64.**

Say the number out loud in the text: *"You will see `max rel err: 3.2e-09`. That number is the course's promise that the math on the page is the math in the machine."* Use float64 for the check — in float32 the central difference is limited to ~$10^{-3}$ relative and the learner will think he has a bug when he has floating point. **This is a real misconception worth a box:** *"A gradient check that fails at 1e-3 in float32 is not a failed gradient check."*

### Rung 2 — `02_backprop_torch.py`
**Purpose:** the *same net, same data, same seed, same loss curve* — with `.backward()` instead of 30 lines of chain rule.

The deliverable is a diff, not a file. The course page should show the two side by side and highlight: **~35 lines of hand-derived backward collapse to one line.** And then the killer move:

```python
# assert the two agree
assert np.allclose(grads_numpy["W1"], model.fc1.weight.grad.T.numpy(), atol=1e-6)
```

That assertion passing is *the* emotional beat of the whole tooling track. Autograd stops being magic and becomes "the thing that does what I did, but correctly, every time."

**What autograd buys, stated concretely:** for our 33-parameter net, hand-derivation is an afternoon. For a 4-billion-parameter transformer with 36 layers, RMSNorm, RoPE, grouped-query attention, and SwiGLU, hand-derivation is *not an afternoon*. Autograd's cost is roughly **2× the forward FLOPs and ~1 forward's worth of stored activations**. That's the deal: you pay ~2-3× compute and a pile of memory, you get correctness for free. This framing also sets up gradient checkpointing (§7.2) as "trade some of that back."

### Rung 3 — `03_mlp_mnist.py`
**Purpose:** the first `nn.Module` + `DataLoader` + real dataset. The eight-line loop in situ.

784→128→10, ReLU, cross-entropy, Adam lr=1e-3, batch 64. **Parameter count: $784{\times}128 + 128 + 128{\times}10 + 10 = 101{,}770$.** ~99% test accuracy... no: be honest, an MLP gets **~97.5–98%** on MNIST; 99%+ is a convnet. Don't inflate it. On the Spark this trains in well under a minute.

Reuse `101,770` as a recurring number. When the course later says "Qwen3-4B has ~4×10⁹ parameters," the learner should be able to say "40,000 of my MNIST nets."

### Rung 4 — `04_tokenizer_and_shapes.py`
**Purpose:** de-mystify the input side of an LLM before touching training. No training at all — just load a tokenizer, encode a sentence, print the ids, print the shapes at every layer boundary via a forward hook.

This artifact is disproportionately valuable and almost always skipped. Everything downstream (padding, attention masks, labels=-100, `max_seq_length`) is unintelligible without it.

### Rung 5 — `05_lora_from_scratch.py`
**Purpose:** implement LoRA as ~15 lines of `nn.Module` **before** `peft` is ever imported.

$$h = W_0 x + \frac{\alpha}{r} B A x$$

- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, **frozen**
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$, init $\mathcal{N}(0, \sigma^2)$ (Kaiming-uniform in the reference impl)
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$, **init zeros** — so $BA = 0$ at step 0 and the adapted model is *exactly* the base model
- $r$ = rank (unitless), $\alpha$ = scaling (unitless), $\frac{\alpha}{r}$ = the effective scale

**Intuition first:** *You're not changing the weight matrix. You're bolting a low-rank correction onto it — a thin bottleneck that can only express $r$ directions of change. Freezing the big matrix and training the thin one is 99% of the savings for most of the benefit.*

*Misconception #1:* "LoRA reduces the memory of the model." **No.** The frozen base weights are still fully resident. LoRA reduces the *optimizer state and gradient* memory, which for Adam is the 3× multiplier on trainable params. Show the arithmetic (§6.3) — it's the correction that actually fixes this.

*Misconception #2:* "B init zeros means it can't learn — the gradient is zero." Wrong, and it's a great teaching moment: $\frac{\partial h}{\partial A} \propto B^\top$ which *is* zero at step 0, but $\frac{\partial h}{\partial B} \propto (Ax)^\top$ which is **not** zero. So $B$ moves first, then $A$ has a nonzero gradient. Both are asymmetric-initialized for exactly this reason: if *both* were zero, nothing would ever move. **Warning box.**

*Misconception #3:* "higher rank is always better." The field genuinely disagrees here (see §11.1). Report the disagreement, don't resolve it.

Attach `05` to the LoRA page, and have it wrap the `04` model so the learner sees his own LoRA change the output of a real tokenizer-fed model.

### Rung 6 — `06_finetune_llm_peft.py`
**Purpose:** the real thing. `transformers` + `peft` + `trl`, LoRA SFT on a small instruct model, ~10–30 min on the Spark.

**Model recommendation:** a **4B-class** model. Candidates, all [REPORTED] as current and healthy in 2026: **Qwen3-4B** (Apache 2.0, best ecosystem/tooling story), **Gemma 3 4B** (128K ctx, multimodal), **SmolLM3-3B** (fully open — data + recipe, not just weights; pedagogically the most honest choice). Also mentioned as current 8-9B-class: **Qwen3.5-9B**, **Granite 4.1 8B**.

**Opinionated pick: Qwen3-4B for the main path, SmolLM3-3B as the "fully reproducible" alternate.** Reason: Apache 2.0 avoids a license digression, 4B fits comfortably, and Qwen3 has the deepest supply of quantizations and community fine-tunes for the learner to compare against. **[REPORTED — verify the exact current Qwen3 point-release name before build; the family has iterated.]**

**Dataset:** something small, inspectable, and *his*. The course's destination is "fine-tune for your own application/database," so the dataset page must teach the JSONL → `datasets` → chat-template → tokenized-tensor pipeline on ~500 hand-written examples. 500 examples is enough to visibly change behavior and small enough to read all of.

### Rung 7 — `07_finetune_diffusion_lora.py`
**Purpose:** the diffusion track's equal-depth counterpart. `diffusers` LoRA on a small SD-class model or a distilled Flux variant.

**Honest note the course must make:** the learner already does this in ComfyUI. The value here is *not* "learn to make a LoRA" — he can. The value is **"see the same LoRA equation you wrote in Rung 5, attached to a UNet/DiT cross-attention block, trained by the same eight-line loop."** Frame it explicitly as *"ComfyUI's LoRA trainer node, unwrapped."* That's the hook that makes this rung land for *this* learner rather than being a rerun.

**Verified anchor number:** DGX Spark does **Flux.1 12B at FP4 → 23 images/min (~2.6 s per 1K image)** and **SDXL 1.0 at BF16 → 7 images/min** [VERIFIED — NVIDIA developer blog]. Use these as the inference baselines the learner can check against his own ComfyUI numbers. If his ComfyUI numbers differ, that's a *teaching opportunity* about precision and scheduler steps, not an error.

### Rung 8 — `08_memory_budget.py`
**Purpose:** the calculator from §6, as a CLI. `python 08_memory_budget.py --params 8e9 --precision bf16 --method qlora --batch 4 --seq 2048` → a table.

This is the artifact that ties the course together, because it's the one he'll actually keep and reuse after the course ends.

### Rung 9 — `09_measure_throughput.py`
**Purpose:** teach him to measure rather than believe. Times N steps, reports tokens/sec and peak memory, with correct CUDA synchronization.

**The bug this artifact exists to teach:** CUDA is asynchronous. `t0 = time.time(); model(x); t1 = time.time()` measures **kernel launch time, not compute time** — you'll get an absurd number like 0.2 ms for a 4B forward. The fix is `torch.cuda.synchronize()` before both timestamps, plus discarding the first ~5 iterations (warmup, autotuning, allocator caching). **This is a warning box and a demo:** show the wrong number and the right number side by side. It's a genuinely universal mistake.

```python
for _ in range(5): step()                    # warmup
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20): step()
torch.cuda.synchronize(); t1 = time.perf_counter()
tok_per_s = 20 * batch * seq / (t1 - t0)
print(f"{tok_per_s:,.0f} tok/s | peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
```

### Rung 10 — `10_merge_and_serve.py`
**Purpose:** close the loop. `merge_and_unload()`, save, load in a fresh process, generate. The learner should end holding an artifact he made that does something.

**Total: 10 files.** Resist adding more. Each one must earn its page.

---

## 4. HuggingFace ecosystem — verified API surface

### 4.1 `transformers` v5 — the breaking changes that matter [VERIFIED via migration guide]

> **⚠️ CORRECTED — v5's break is NARROWER than this section claims (D-10g, verified against v5.14.1 source).**
> Two rows below are wrong: (1) **`torch_dtype` is NOT removed** — it is soft-deprecated with a working BC shim
> (`logger.warning_once("torch_dtype is deprecated! Use dtype instead!")`), so old code *runs*. (2) **`batch_decode`
> still exists**; **`warmup_ratio` was NEVER removed** (the GitHub changelog summary is wrong — do not trust it).
> **The real hazard is silent DEFAULT changes, not removals:** `report_to` now defaults `"none"` (was `"all"`);
> a new `use_cache` field defaults `False`; **model `dtype` loads as-saved** (was float32). *Actually* removed:
> TF/Flax/JAX, `encode_plus`/`batch_encode_plus`, legacy cache format, `BetterTransformer`, `torchscript`/`torch.fx`,
> and several `TrainingArguments` fields (`overwrite_output_dir`, `jit_mode_eval`, `tpu_num_cores`, `past_index`,
> `ray_scope`, `mp_parameters`). See `constants.md` §7.1.

This is the biggest tooling risk in the whole course. Any tutorial the learner Googles will be v4-era and **will not run**.

| v4 | v5 | Impact on the course |
|---|---|---|
| TF/JAX backends | **PyTorch only** | Simplifies teaching. Say it. |
| `torch_dtype=torch.bfloat16` | **`dtype=` ; default is `"auto"`** | *Big deal.* Models now load in whatever precision they were saved in. Code that assumed fp32 now silently runs bf16. **Warning box.** |
| `load_in_4bit=True` / `load_in_8bit=True` | **removed — use `BitsAndBytesConfig`** | Every QLoRA tutorial online is now wrong. Show the correct form. |
| `safe_serialization=False` | **removed** — safetensors always | Fine; mention safetensors is a *good* thing (no pickle → no arbitrary code execution on `from_pretrained`). Real security point, one sentence. |
| `requests` backend | **`httpx`** | `except requests.HTTPError` → `httpx.HTTPError` |
| `TRANSFORMERS_CACHE` | **`HF_HOME`** | Matters on the Spark: point `HF_HOME` at the NVMe, not at a small root partition. |
| `FeatureExtractor` | **`ImageProcessor`** | diffusion/multimodal track |
| `transformers chat` server | **`transformers serve`** | if the course demos serving |
| — | requires **Python ≥3.10, torch ≥2.4, huggingface_hub ≥1.3 <2.0** | pin it |
| — | one tokenizer file per model, Rust-backed path preferred | mostly invisible |

Correct v5 quantized load:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto"
)
```
[VERIFIED that `load_in_4bit` as a top-level kwarg is removed; the `BitsAndBytesConfig` field names above are [UNVERIFIED] against 5.14.1 specifically — **check before ship**, they were stable through v4 but this is exactly the kind of thing v5 renames.]

### 4.2 `peft` 0.19.1 — LoRA config

```python
from peft import LoraConfig, get_peft_model, TaskType

cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
# -> trainable params: X || all params: Y || trainable%: Z
```

**Teaching notes:**
- `target_modules` is **architecture-specific**. The names above are Llama/Qwen/Mistral-family. They will not match Gemma's or a DiT's. The recipe for finding them: `print([n for n, _ in model.named_modules()])` and read. **Teach the recipe, not the list** — the list rots, the recipe doesn't.
- `target_modules="all-linear"` is the lazy correct answer and worth mentioning as the escape hatch. [UNVERIFIED for 0.19.1 — was present in earlier versions.]
- `print_trainable_parameters()` is the single best pedagogical method in the entire HF ecosystem. It prints the LoRA thesis as a number. **Put its output on the page.**
- Common convention `lora_alpha = 2r`. It is a *convention*, not a law (§11.1).

*Misconception:* "`target_modules` should include everything for best results." Attention-only vs. attention+MLP is a real, live trade-off — MLP targeting roughly triples adapter params for typically-modest gain. Present as a trade, not a rule.

### 4.3 `trl` ~~0.29.1~~ **1.8.0** — `SFTTrainer` [VERIFIED that the churn is real]

> **⚠️ CORRECTED — pin `trl>=1.8,<2` (decisions D-10f / §7).** This section was written against the stale
> 0.29.1. The v0→v1 break is small (`MIGRATION.md`): `GRPOConfig`/`RLOOConfig` `vllm_mode` default
> `"server"`→`"colocate"`; `SFTConfig` `packing="bfd-requeue"`→`"bfd_split"`; trainers no longer auto-strip `None`
> from datasets. The `SFTConfig` snippet below is essentially correct at 1.8 (`max_length`, `processing_class`,
> `peft_config` all still valid). **The "prefer plain `Trainer`" recommendation at the end is OVERRULED** — it was
> based on the stale version; use `trl>=1.8` + `SFTTrainer`. **New trap [VP]:** `SFTTrainer` still defaults to
> **fp32** when `model` is a string and `model_init_kwargs` omits `dtype` — pass `model_init_kwargs={"dtype": torch.bfloat16}`.

The churn, chronologically:
- **v0.12.0:** `tokenizer=` deprecated in favour of `processing_class=` [VERIFIED]
- **v0.16.0:** `tokenizer=` **removed** [VERIFIED]
- Various: `dataset_num_proc`, `max_seq_length`, `packing` moved into `SFTConfig` rather than being trainer kwargs [REPORTED — the `SFTTrainer.__init__() got an unexpected keyword argument 'dataset_num_proc'` error is a well-attested symptom]

At 0.29.1, roughly:
```python
from trl import SFTTrainer, SFTConfig

cfg = SFTConfig(
    output_dir="out/",
    max_length=1024,              # NOTE: was max_seq_length in older trl
    packing=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
)
trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds,
    processing_class=tokenizer,   # NOT tokenizer=
    peft_config=cfg_lora,
)
trainer.train()
```
**[UNVERIFIED at 0.29.1: the exact `SFTConfig` field names — particularly `max_length` vs `max_seq_length`, and whether `peft_config` is still on the trainer. These moved. The architect MUST have someone run this against 0.29.1 before it ships.]** Given the demonstrated rate of churn, I'd go further:

> ~~**Recommendation: consider writing the LLM fine-tune with the plain `transformers.Trainer` + `peft`.**~~
> **OVERRULED (D-10f).** This rested on the stale 0.29.1 and its churn. On `trl>=1.8` the API is stable; **use
> `SFTTrainer`.** Chat templating + packing are real value-add and the v1 break is minimal. A `Trainer` fallback
> is optional, not the default.

Other trl trainers to name-drop but not teach: `DPOTrainer`, `GRPOTrainer`, `KTOTrainer`, `RewardTrainer` [VERIFIED present in 0.29.x]. One paragraph: *"SFT teaches the model to imitate. DPO/GRPO teach it to prefer. This course does SFT; here's the map of the rest."*

*Misconception:* "`peft_config` + an already-`get_peft_model`'d model." There's a known issue where passing an already-wrapped PEFT model to SFTTrainer freezes the adapter [REPORTED — trl issue #3926]. Pick **one** path: either `get_peft_model` yourself and don't pass `peft_config`, or pass `peft_config` and hand over a bare model. Don't do both. **Warning box.**

### 4.4 `datasets`, `accelerate`

- `datasets`: teach `load_dataset("json", data_files=...)`, `.map()`, `.train_test_split()`. That's it. The Arrow/memory-mapping story is one sentence: *"it's memory-mapped, so a 40 GB dataset doesn't need 40 GB of RAM."*
- `accelerate`: the learner has **one GPU**. `accelerate` is genuinely near-useless to him except that `Trainer` uses it internally and `accelerate config` / `accelerate launch` appear in every error message. Give it **half a page**: what it is, why it's in your stack trace, and `accelerate env` as a diagnostic. Do not teach multi-node. (Exception: if the course covers **connecting two Sparks** via the ConnectX-7 — NVIDIA publishes a playbook for exactly this [REPORTED]. That's an appendix, not a page.)

### 4.5 `bitsandbytes` — and the aarch64 caveat

Purpose in the course: NF4 quantization for QLoRA, and 8-bit optimizers. See §5.4 for the ARM story — **this is the single most likely thing to break on the Spark** and needs its own verification pass.

### 4.6 `torchtune` — the trap

**NVIDIA's and PyTorch's own DGX Spark fine-tuning showcase uses torchtune** [VERIFIED — the pytorch.org blog "Unlock Reasoning in Llama 3.1-8B via Full Fine-Tuning on NVIDIA DGX Spark" runs `tune run full_finetune_single_device --config fft-8b.yaml`]. Meanwhile **torchtune is no longer actively maintained; development wound down in 2025** [REPORTED, strong signal — repo carries a maintenance warning].

`torchforge` is **not** a drop-in successor — it targets RL post-training and agentic workflows and is explicitly experimental [REPORTED].

**Recommendation:** do **not** build the course's fine-tuning path on torchtune. Use `transformers`/`peft`. **But do cite the torchtune blog for its numbers** (§5.5) with a footnote that the tool is in wind-down. This is exactly the kind of honest note the brief was asked for: the best-documented DGX Spark fine-tuning result in existence is built on a library that is dying.

---

## 5. The NVIDIA DGX Spark

### 5.1 Verified specs [VERIFIED — docs.nvidia.com/dgx/dgx-spark/hardware.html]

| Spec | Value |
|---|---|
| Superchip | NVIDIA **GB10** Grace-Blackwell |
| CPU | **20-core Arm**: 10× Cortex-X925 + 10× Cortex-A725 |
| GPU | Blackwell, **6,144 CUDA cores**, 5th-gen Tensor Cores |
| GPU SMs | **48** [REPORTED, consistent across sources] |
| Compute capability | **sm_121** (12.1) [REPORTED, consistent] |
| Memory | **128 GB LPDDR5x unified**, CPU+GPU coherent |
| **Bandwidth** | **273 GB/s** |
| Storage | 1 TB or 4 TB self-encrypting NVMe M.2 |
| Network | 1× RJ-45 10 GbE, **ConnectX-7 SmartNIC**, Wi-Fi 7, BT 5.4 |
| Perf | up to **1,000 TOPS** inference; **1 PFLOP FP4 with sparsity** |
| Claimed capacity | up to **200B params** (single), **405B** (dual-Spark) |
| Physical | 150 × 150 × 50.5 mm, 1.2 kg, **240 W** external PSU |
| Thermal | 5–30 °C ideal |
| OS | **DGX OS 7.x** (Ubuntu 24.04-based); reported **7.5.0** [REPORTED] |
| CPU↔GPU link | NVLink-C2C [REPORTED] |
| Price | ~**$4,699** [REPORTED] |

### 5.2 The intuition that makes the specs make sense

> **1 PFLOP of compute served by 273 GB/s of memory is a Ferrari with a garden hose for a fuel line.**

Make this quantitative — it's the best single piece of arithmetic in the hardware chapter.

**Arithmetic intensity.** Define
$$I = \frac{\text{FLOPs performed}}{\text{bytes moved}} \quad [\text{FLOP/byte}]$$

The machine's **ridge point** (the roofline knee) is
$$I^* = \frac{P_{\text{peak}}}{BW}$$

with $P_{\text{peak}}$ in FLOP/s and $BW$ in bytes/s. Taking a dense-BF16 peak — **[UNVERIFIED: NVIDIA quotes 1 PFLOP at FP4 *with sparsity*; the honest dense-BF16 number is not something I could confirm. A common inference is ~125 TFLOP/s dense BF16 (1 PFLOP FP4-sparse ÷ 2 for sparsity ÷ 4 for the FP4→BF16 precision step), but NVIDIA has not, to my knowledge, published this cleanly. Flag it in the text rather than asserting it.]** — use $P \approx 1.25\times10^{14}$ FLOP/s:

$$I^* = \frac{1.25\times10^{14}}{2.73\times10^{11}} \approx \mathbf{458 \ \text{FLOP/byte}}$$

Now the punchline. **Batch-1 LLM decoding has $I \approx 2$ FLOP/byte** — you move each weight once (2 bytes in BF16) and do exactly one multiply-add (2 FLOPs) with it. $2 \ll 458$. You are **memory-bound by a factor of ~230×**. You are using well under 1% of the machine's arithmetic capability, and *no amount of compute would help.*

This one calculation explains:
- why the Spark decodes at ~20–50 tok/s while advertising a petaflop
- why batching helps *enormously* (batch B raises $I$ to $\approx 2B$)
- why quantization helps decoding (fewer bytes per weight → $I$ rises → and the byte count is what you're paying for)
- why **training** is a much better fit for this machine than **serving** — training runs at batch 4–16 with 2K–16K sequences, so $I$ is in the hundreds and you're actually near the ridge

**This is the hardware chapter's thesis. Put it on the page as a number.**

### 5.3 Verified performance numbers — use these verbatim, repeatedly

**Fine-tuning** [VERIFIED — NVIDIA developer blog]:

| Workload | Throughput |
|---|---|
| Llama 3.2 **3B**, **full** fine-tune | **13,519.5 tok/s** |
| Llama 3.1 **8B**, **LoRA** | **6,969.6 tok/s** |
| Llama 3.3 **70B**, **QLoRA** | **759.8 tok/s** |

*(Note: an aggregator claimed "53,657.6 tok/s for Llama 3.1 8B LoRA on DGX Spark." NVIDIA's own blog says **6,969.59**. Trust the primary source. Flagging because the inflated number is circulating and the architect may encounter it.)*

**Diffusion** [VERIFIED — same source]:

| Workload | Throughput |
|---|---|
| **Flux.1 12B @ FP4** | **23 img/min** (2.6 s per 1K image) |
| **SDXL 1.0 @ BF16** | **7 img/min** |

**Inference** [VERIFIED — LMSYS review, 2025-10]:

| Model / stack | Prefill | Decode |
|---|---|---|
| GPT-OSS 20B (MXFP4), Ollama | 2,053 tok/s | **49.7 tok/s** |
| Llama 3.1 8B (FP8), SGLang, **batch 1** | 7,991 tok/s | **20.5 tok/s** |
| Llama 3.1 8B (FP8), SGLang, **batch 32** | 7,949 tok/s | **368 tok/s** |
| Llama 3.1 70B (FP8), SGLang | 803 tok/s | **2.7 tok/s** |
| Qwen3 235B, **dual Spark** | 23,477 tok/s | **11.73 tok/s** [VERIFIED, NVIDIA] |

**Comparison points** [VERIFIED — LMSYS]:

| Machine | GPT-OSS 20B prefill / decode |
|---|---|
| **DGX Spark** | 2,053 / **49.7** |
| RTX 5090 | 8,519 / **205** |
| RTX Pro 6000 Blackwell | 10,108 / **215** |

**The batch-1 → batch-32 row is the single most instructive number in this brief.** Decode goes 20.5 → 368 tok/s (**18×**) while prefill is *flat* (7,991 → 7,949). Prefill was already compute-bound and gained nothing; decode was bandwidth-bound and gained almost linearly with batch. **This is the roofline model, measured, on the learner's own machine.** Build a demo around it (§8.1).

Also note: LMSYS report **no thermal throttling under sustained load** [VERIFIED] — worth saying, because the form factor invites suspicion.

### 5.4 The aarch64 trap — the practical heart of this section

**Intuition:** *Every Python wheel with compiled code was built for a specific CPU architecture. Almost the entire ML ecosystem was built for x86_64. The Spark is aarch64. When something doesn't install, it is nine times out of ten this, and never anything more interesting.*

What's verified:
- **The Spark is aarch64. x86-only images, wheels, and binaries will not run.** [REPORTED, unambiguous]
- **PyTorch publishes aarch64 wheels, and on Linux aarch64 the CUDA-enabled wheel is what `pip install torch` gets you by default** [VERIFIED — PyPI/pytorch.org]. So the base case is fine.
- **`bitsandbytes` publishes CUDA Linux aarch64 (sbsa) wheels** targeting sm75/sm80/sm90/sm100, and 0.49.0 added **CUDA 13.0 compatibility across Linux x86-64, Linux aarch64, and Windows x86-64**, plus a changelog line reading **"Add release for DGX Spark"** [VERIFIED via releases page]. **RESOLVED (D-10i): `build-cuda.sh` targets `build_capability="75;80;90;100;110;120;121"` for ARM64+CUDA-13 — bitsandbytes ships NATIVE sm_121, and x86 does not get it. The 70B QLoRA path depends on this.** [VP]

**The sm_121 subtlety — get this right, it's the most confusing thing on the machine:**
- The GB10 GPU is **sm_121**.
- PyTorch's shipped CUDA kernels compile **through sm_120**.
- **sm_120 binaries are binary-compatible with sm_121** [VP] — this is the real mechanism.
- ~~NGC/release wheels ship sm_120 + compute_120 PTX which JITs to sm_121~~ **← REFUTED for RELEASE wheels
  (D-10h / constants §6.9). Release PyPI wheels ship SASS-only, NO PTX** (`_ptx_arches()` returns `set()` on
  release builds). PTX shipping is a **nightly / NGC-container** thing, not the stock `pip install torch`.

> **⚠️ CORRECTED — the "PTX JIT" story is FALSE on a stock wheel (D-10h).** It runs via **sm_120→sm_121 binary
> compatibility only.** ~~"The first CUDA kernel JIT after a driver/toolkit change is slow — that's the PTX
> compile"~~ **cannot happen on a release PyPI wheel: there is no PTX to compile.** Delete that sentence; it
> applies to nightlies and NGC containers only. **The real trap is narrower and sharper:** any package pinning
> `libcudart.so.12` fails at *import* on the CUDA-13-only DGX OS, before kernel compatibility is even evaluated.
> **vLLM is NO LONGER blocked** — issue #31128 closed (PR #37700); it runs the standard OpenAI-compatible server
> with CUDA graphs on by default, **via the `vllm/vllm-openai:cu130-nightly` container pinned by digest** (the
> PyPI aarch64 wheel is CUDA-12 and fails). And **bitsandbytes compiles NATIVE sm_121** into the aarch64+CUDA-13
> wheel (x86 does not get it) — the Spark is *better* served than an x86 box here.

**The honest summary for the learner:**

> Your Spark is a first-class citizen for PyTorch, transformers, peft, and diffusers. It is a **second-class citizen for anything that ships a hand-compiled CUDA kernel**: flash-attention forks, some quantization kernels, some inference servers, most research repos. When you hit that wall, the error will mention `aarch64`, `sm_121`, or "no kernel image is available for execution on the device." That last one means: *this wheel was compiled for someone else's GPU.*

**Things likely to be painful on aarch64** [UNVERIFIED as a set — this is pattern-matching, not a tested list, and the architect should have someone actually try each]:
- `flash-attn` (source builds; long compiles) — but note **PyTorch's built-in `F.scaled_dot_product_attention` / FlexAttention needs no external package and is the right answer for the course anyway**
- `xformers`
- AutoGPTQ / AutoAWQ and similar kernel-shipping quantizers
- `triton` — reportedly needs recompilation in some Spark configurations [REPORTED]
- anything installed from a GitHub URL rather than PyPI

**Course guidance: prefer PyTorch-native.** `F.scaled_dot_product_attention` instead of flash-attn; `bitsandbytes` (which has aarch64 wheels) instead of GPTQ/AWQ. This isn't just aarch64 hygiene — it's better pedagogy, because SDPA is the function whose math the course derived.

### 5.5 What the Spark can actually do

**Verified ceiling:** **full fine-tune of Llama 3.1-8B at seq_len 16,384, batch 16, bf16, using ~80% of 128 GB, at ~8 h/epoch** (ToolACE, 11k pairs, 3 epochs ≈ 24 h) [VERIFIED — pytorch.org blog, via torchtune].

That number deserves emphasis: **a full fine-tune — not LoRA — of an 8B model at 16K context, on a 1.2 kg box drawing 240 W, overnight.** No consumer GPU does this; a 5090's 32 GB can't hold it at any batch size. This is exactly the trade the machine makes: it is *slower* than a 5090 and it does things a 5090 *cannot do at all*. Say both halves.

**Realistic capability map** — [VERIFIED] rows marked, rest [UNVERIFIED extrapolation from the memory math in §6, and should be labelled as such in the course]:

| Task | Verdict |
|---|---|
| Full FT, 3B | ✅ 13.5k tok/s [VERIFIED] |
| Full FT, 8B @ 16K ctx, batch 16 | ✅ ~8 h/epoch [VERIFIED] |
| LoRA, 8B | ✅ 7.0k tok/s [VERIFIED] |
| QLoRA, 70B | ✅ 760 tok/s [VERIFIED] — ~18× slower than 8B LoRA, so an epoch that took 1 h takes ~18 h |
| Full FT, 70B | ❌ ~1.1 TB of optimizer state in fp32 Adam. Not close. |
| Diffusion LoRA (SDXL / Flux-class) | ✅ [UNVERIFIED but inference numbers make it near-certain] |
| Batch-1 chat serving | ⚠️ works, ~20–50 tok/s — **4× slower than a 5090** [VERIFIED] |
| Serving a 70B to users | ❌ 2.7 tok/s [VERIFIED]. Unusable. |
| Pretraining from scratch | ❌ Not the machine. Not any single machine. |
| Running 100B+ models *at all* | ✅ — and this is the point. Slowly. |

**What it is bad at, stated plainly:**
1. **Latency.** ~4× slower than an RTX 5090 or RTX Pro 6000 on decode [VERIFIED]. If you want a fast local chatbot, you bought the wrong box.
2. **Anything bandwidth-bound.** 273 GB/s vs ~1,800 GB/s (RTX 5090's GDDR7) is a ~6.6× deficit. [5090 bandwidth: UNVERIFIED, ~1.79 TB/s from memory — check.]
3. **Bleeding-edge repos.** aarch64 + sm_121 means you are in a small minority of the CI matrix.
4. **Serving.** It is a development machine.

**What it is uniquely good at:**
1. **Capacity per dollar per watt.** 128 GB coherent at 240 W and ~$4,699.
2. **Full fine-tunes of 8B-class models at long context** — a 5090 simply cannot.
3. **Fitting the whole loop on one box.** No cloud, no data egress, no meter running.
4. **Being the same CUDA/Arm stack as GB200 racks** — what you build here lifts.

---

## 6. The memory budget — the recurring thread

This is the reusable tool. It should appear on the hardware page, be re-derived on the LoRA page, be used on every fine-tune page, and exist as both a widget (§8.2) and a CLI (`08_memory_budget.py`).

### 6.1 The intuition

> *Training memory is four buckets, and only one of them is the model. Weights, gradients, optimizer state, activations. Novices size their GPU by bucket one and get OOM'd by buckets three and four.*

### 6.2 The equation

$$M_{\text{total}} = \underbrace{M_W}_{\text{weights}} + \underbrace{M_G}_{\text{grads}} + \underbrace{M_O}_{\text{optimizer}} + \underbrace{M_A}_{\text{activations}} + \underbrace{M_{\text{frag}}}_{\text{overhead}}$$

Symbols:
| Symbol | Meaning | Units |
|---|---|---|
| $N$ | total parameters | count |
| $N_t$ | **trainable** parameters | count |
| $b_W$ | bytes/param for stored weights | B (fp32=4, bf16/fp16=2, fp8=1, NF4=0.5) |
| $b_G$ | bytes/param for gradients | B (matches training dtype, usually 2) |
| $k_O$ | optimizer bytes per **trainable** param | B |
| $B$ | batch size | count |
| $L$ | sequence length | tokens |
| $d$ | hidden size | count |
| $n_L$ | number of layers | count |
| $c$ | activation constant (empirical) | — |

$$M_W = N \cdot b_W \qquad M_G = N_t \cdot b_G \qquad M_O = N_t \cdot k_O$$

**$k_O$ by optimizer — the number people get wrong:**
| Optimizer | State | $k_O$ |
|---|---|---|
| SGD (no momentum) | — | **0** |
| SGD + momentum | 1 buffer | **4** (fp32) |
| **AdamW, fp32 states** | $m$, $v$ | **8** |
| **AdamW mixed-precision (fp32 master copy + $m$ + $v$)** | 3× fp32 | **12** |
| AdamW 8-bit (bitsandbytes) | quantized $m$, $v$ | **~2** |

**The AdamW-is-a-3×-tax intuition:** *Adam remembers two running averages per weight — a mean and a variance — each the same size as the weight. Plus, in mixed precision, a full-precision master copy so tiny updates aren't lost to rounding. You store your model four times to train it once.*

Activations (the hard one, the one that's empirical):
$$M_A \approx c \cdot B \cdot L \cdot d \cdot n_L \cdot b_{\text{act}}$$
with $c \approx 10\text{–}20$ for a modern transformer block without checkpointing. **State clearly that $c$ is a rule of thumb, not a derivation** — it depends on which intermediates the implementation chooses to save, on SDPA vs. naive attention, on fused kernels. **The honest move is: estimate with $c=16$, then *measure* with `torch.cuda.max_memory_allocated()`.** That's the lesson.

With **gradient checkpointing**, activation memory drops to roughly
$$M_A^{\text{ckpt}} \approx c \cdot B \cdot L \cdot d \cdot \sqrt{n_L} \cdot b_{\text{act}}$$
at a cost of ~**30–40%** extra compute (one extra forward per checkpointed segment). [The $\sqrt{n_L}$ is the classic optimal-checkpoint-placement result; PyTorch's `gradient_checkpointing_enable()` in practice checkpoints **per layer**, giving $M_A \approx c\cdot B\cdot L\cdot d\cdot 1 \cdot b_{\text{act}}$ + one layer's internals — i.e. the $n_L$ factor collapses to ~1, which is *better* than $\sqrt{n_L}$ for memory and worse for compute. **[UNVERIFIED — I'd present the per-layer version since that's what HF actually does, and mention $\sqrt{n}$ as the theory.]** This is a place the architect should not let me hand-wave.]

$M_{\text{frag}}$: allocator fragmentation + CUDA context + cuBLAS workspaces. **Budget 10%.** Non-negotiable in practice.

### 6.3 Worked examples — carry the arithmetic all the way

Use **1 GiB = $2^{30}$ = 1.074×10⁹ B**. Be consistent; GB vs GiB confusion is a real source of "why doesn't 128 GB fit 128 GB."

---

**Example A — Full fine-tune, Llama-3.2-3B, bf16 + fp32 AdamW master, B=4, L=2048, d=3072, n_L=28**

*(d and n_L for 3.2-3B: [UNVERIFIED] — check the config.)*

- $M_W = 3.0\times10^9 \times 2 = 6.0$ GB = **5.6 GiB**
- $M_G = 3.0\times10^9 \times 2 = 6.0$ GB = **5.6 GiB**
- $M_O = 3.0\times10^9 \times 12 = 36.0$ GB = **33.5 GiB**  ← *the bucket nobody budgets for*
- $M_A \approx 16 \times 4 \times 2048 \times 3072 \times 28 \times 2$, carried through:
  $16 \times 4 = 64$; $64 \times 2048 = 131{,}072$; $\times 3072 = 4.03\times10^{8}$; $\times 28 = 1.13\times10^{10}$; $\times 2 \text{ B} = 2.25\times10^{10}$ B = **21.0 GiB**
- Subtotal: $5.6 + 5.6 + 33.5 + 21.0 = 65.7$ GiB; **+10% → ~72 GiB**

**Verdict: fits in 128 GB with room. Matches reality** — NVIDIA measured full FT of Llama 3.2 3B at 13.5k tok/s [VERIFIED]. ✅ The model is 5.6 GiB and the training run is 72 GiB. **That ratio — 13× — is the lesson.**

---

**Example B — Full fine-tune, Llama-3.1-8B, the verified config: L=16,384, B=16, bf16**

- $M_W = 8.03\times10^9 \times 2 = 16.1$ GB = **15.0 GiB**
- $M_G = $ **15.0 GiB**
- $M_O = 8.03\times10^9 \times 12 = 96.4$ GB = **89.7 GiB**
- Running total already **119.7 GiB of 119.2 GiB usable** (128 GB = 119.2 GiB). **Before a single activation.**

**And yet NVIDIA/PyTorch report this configuration running at ~80% peak of 128 GB** [VERIFIED]. **The naive budget says it cannot fit. It does. Why?**

This is a *fantastic* teaching moment — the course should pose it as a puzzle before answering. Candidate resolutions:
- torchtune's recipe likely uses **`AdamW8bit`** or a fused/low-precision optimizer → $k_O \approx 2$ instead of 12, dropping $M_O$ from 89.7 → **15.0 GiB**. Total becomes $15+15+15 = 45$ GiB + activations.
- **Activation memory at B=16, L=16384 is enormous**: $16 \times 4 \times 16{,}384 \times 4096 \times 32 \times 2 / 2^{30}$ — with $c=16$: $16\times16 = 256$; $\times 16384 = 4.19\times10^6$; $\times 4096 = 1.72\times10^{10}$; $\times 32 = 5.5\times10^{11}$; $\times 2 = 1.1\times10^{12}$ B = **1,024 GiB**. Absurd. → **gradient checkpointing is mandatory**, and with per-layer checkpointing the $\times 32$ collapses, giving ~32 GiB.
- So a plausible reconciliation: $15 (W) + 15 (G) + 15 (O_{\text{8bit}}) + 32 (A_{\text{ckpt}}) = 77$ GiB, +10% ≈ **85 GiB ≈ 66% of 128 GB.** In the neighbourhood of the reported 80%. **[UNVERIFIED — I could not fetch torchtune's `fft-8b.yaml` to confirm the optimizer. The architect should fetch that config; it turns this from a plausible story into a verified one, and it is worth doing because this example is load-bearing.]**

**Either way, the pedagogy is right:** the naive estimate says "impossible," reality says "80%," and the gap is *entirely* explained by 8-bit optimizer + gradient checkpointing. **That gap IS the lesson of the whole chapter.** Lead the page with the contradiction.

---

**Example C — LoRA, Llama-3.1-8B, r=16, attention-only, bf16, B=4, L=2048**

Trainable params: for each of $q,k,v,o$ in each of 32 layers, with $d=4096$ (and GQA making $k,v$ smaller — ignore for the estimate, flag the simplification):
$$N_t \approx n_L \times 4 \times (r \cdot d_{\text{in}} + d_{\text{out}} \cdot r) = 32 \times 4 \times (16{\cdot}4096 + 4096{\cdot}16) = 32 \times 4 \times 131{,}072 = 16.8\times10^6$$

**≈ 16.8M trainable of 8.03B total = 0.21%.** *(This is roughly what `print_trainable_parameters()` will print. Put the real printed line on the page once you've run it.)*

- $M_W = 15.0$ GiB (frozen, but **still resident** — the misconception)
- $M_G = 16.8\times10^6 \times 2 = 33.6$ MB = **0.03 GiB**
- $M_O = 16.8\times10^6 \times 12 = 202$ MB = **0.19 GiB**
- $M_A \approx 16 \times 4 \times 2048 \times 4096 \times 32 \times 2 = 3.4\times10^{10}$ B = **32 GiB** (no ckpt)
- Total ≈ $15 + 0.03 + 0.19 + 32 = 47.2$ GiB, +10% ≈ **52 GiB**

**The headline:** optimizer+gradient memory went from **95 GiB → 0.22 GiB, a 430× reduction.** Weights didn't move at all. Activations didn't move at all.

**This is the single most important number in the LoRA chapter** and it directly kills the "LoRA shrinks the model" misconception. LoRA is *not* a model compression technique. It is an **optimizer-state compression technique.** And note the sting in the tail: at B=4/L=2048, **activations (32 GiB) now dominate everything else combined.** So after LoRA, your next lever is *not* a smaller adapter — it's gradient checkpointing or a smaller batch. **The bottleneck moved.** Teach that: optimizing is whack-a-mole, and you must re-measure after every change.

---

**Example D — QLoRA, Llama-3.3-70B, NF4 base, r=16, B=1, L=1024**

- $M_W = 70.6\times10^9 \times 0.5 = 35.3$ GB = **32.9 GiB** — plus NF4's quantization constants. With double-quant, overhead is ~**0.127 bits/param** **[VP — CONFIRMED, promoted from UNVERIFIED. This brief had it right; `brief-llm-finetuning`'s 4.25/0.53 was the arithmetic error. See `constants.md` §3 / decisions §Z-4.]**, so ~$70.6\times10^9 \times 0.127/8 = 1.12$ GB ≈ **1.0 GiB**. Call it **34 GiB**. (Total NF4 base = 4.127 bits = 0.516 B/param **only with double-quant on**; it defaults off — un-opted-in is 4.5 bits = 0.5625 B/param.)
- $N_t$: 80 layers × 4 proj × $2\cdot16\cdot8192$ = $80 \times 4 \times 262{,}144 = 83.9\times10^6$ → **0.12% of total**
- $M_G = 0.17$ GiB, $M_O = 1.0$ GiB
- $M_A$ (B=1, L=1024, d=8192, n_L=80, with checkpointing ≈ /n_L): $16 \times 1 \times 1024 \times 8192 \times 2 = 2.7\times10^{8}$ B ≈ **0.25 GiB** — plus the dequantization workspace, which is real and not in this formula.
- Total ≈ **36 GiB**, +10% ≈ **40 GiB**

**Verdict: fits with 88 GB to spare. And NVIDIA measured it: 759.8 tok/s** [VERIFIED]. ✅

**The honest note:** 760 tok/s vs 6,970 for 8B LoRA is **9.2× slower for 8.8× the parameters** — almost exactly linear, which tells you QLoRA on this box is *bandwidth*-bound, not compute-bound, and that the NF4 dequant is nearly free. That's a beautiful, checkable inference from two published numbers. Use it.

And: **you have 88 GB of headroom at B=1.** So raise the batch. This is where the learner discovers that *the memory calculator's job is not to tell you if it fits — it's to tell you how much you're leaving on the table.*

---

**Summary table for the course (all on the Spark's 128 GB / 119.2 GiB):**

| Model | Method | $N_t$ | %train | W | G+O | A | Total | Fits? |
|---|---|---|---|---|---|---|---|---|
| 3B | Full | 3.0B | 100% | 5.6 | 39.1 | 21.0 | ~72 GiB | ✅ |
| 8B | Full (naive AdamW) | 8.0B | 100% | 15.0 | 104.7 | 32+ | ~167 GiB | ❌ |
| 8B | Full (8-bit opt + ckpt) | 8.0B | 100% | 15.0 | 30.0 | ~32 | ~85 GiB | ✅ [matches ~80% report] |
| 8B | LoRA r=16 | 16.8M | 0.21% | 15.0 | 0.22 | 32.0 | ~52 GiB | ✅ |
| 70B | Full | 70.6B | 100% | 132 | 918 | — | >1 TB | ❌ |
| 70B | QLoRA r=16 | 83.9M | 0.12% | 34.0 | 1.2 | 0.25 | ~40 GiB | ✅ |

**Reuse this table.** It should appear at least three times in the course.

---

## 7. Failure modes

### 7.1 OOM — the decision tree

**Intuition:** *OOM is not a bug. It's the machine telling you your arithmetic was optimistic. Every fix is a trade of one resource for another; there are no free ones.*

The error you'll see:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 119.18 GiB of which 1.42 GiB is free.
```
**Read it. It tells you exactly how much it wanted and how much was left.** Learners panic and don't read it. Warning box.

**The ladder, in order, with the price of each rung:**

| Fix | Memory saved | Price |
|---|---|---|
| 1. Reduce `per_device_train_batch_size` | linear in $B$ | *effective* batch shrinks → noisier gradients → **use accumulation to compensate (7.3)** |
| 2. `gradient_checkpointing=True` | ~$n_L$× on activations | **+30–40% step time** |
| 3. Shorten `max_length` | linear in $L$ (quadratic if attention isn't SDPA/flash) | you truncate your data. Check what you're cutting. |
| 4. 8-bit optimizer (`adamw_bnb_8bit`) | $k_O$: 12 → ~2 | tiny/no quality loss, well-attested |
| 5. LoRA instead of full FT | $M_G + M_O$ → ~0 | quality gap that is genuinely contested (§11.2) |
| 6. QLoRA (NF4 base) | $M_W$: ×0.25 | slower per step; measurable quality cost |
| 7. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | recovers fragmentation | none, really. **Try this early — it's free.** |

**Anti-patterns worth a warning box:**
- **`torch.cuda.empty_cache()` in the training loop.** It doesn't fix OOM — the allocator's cache is *already* available for reuse. It just makes every step slower by forcing re-allocation. It's cargo cult. (It *is* legitimate between distinct phases, e.g. train → eval.)
- **Accumulating a tensor with a live graph into a list**: `losses.append(loss)` retains the entire computation graph for every step, and memory grows monotonically until death. `losses.append(loss.item())`. **This is the classic "it OOMs at step 400 but was fine at step 1" bug** — and "OOM'd *later*, not immediately" is the diagnostic fingerprint of a leak vs. a budget error. Teach the fingerprint.
- **Evaluating without `torch.no_grad()`** — builds a graph you never use.

**The Spark's unified-memory wrinkle [UNVERIFIED, and important]:** on a discrete GPU, OOM is clean — you hit 24 GB and stop. On unified memory, GPU and CPU share the same 128 GB pool, so the OS, your browser, and DGX OS itself are all in the same budget, and the failure mode may be swapping/thrashing rather than a clean OOM. **The architect should have this tested on real hardware.** Practical guidance: `free -h` and `nvidia-smi` tell you *different overlapping things* on this machine; budget ~110 GB usable, not 128. Do not let the course assert unified-memory OOM semantics without someone having actually pushed a Spark over the edge and written down what happened.

### 7.2 Gradient checkpointing

**Intuition:** *Instead of writing down every intermediate result on the way forward so you can use it on the way back, throw most of them away and re-derive them when you need them. You trade time for space — you're paying compute to avoid remembering.*

```python
model.gradient_checkpointing_enable()
model.config.use_cache = False   # MUST. KV cache and checkpointing conflict.
```

**The `use_cache=False` line is a top-3 gotcha.** You get a warning, not an error, and it's easy to miss. Warning box.

**Also:** with PEFT + checkpointing you often need `model.enable_input_require_grads()`, or the checkpointed segment has no grad-requiring input and silently produces no gradients [REPORTED — long-standing PEFT+checkpointing interaction; verify at 0.19.1]. This one is nasty because **it fails silently: loss doesn't move and nothing errors.** If the course teaches one debugging reflex it should be: *after step 1, assert some parameter's `.grad` is not None and not all-zero.*

### 7.3 Batch size vs. gradient accumulation

**Intuition:** *Accumulation is how you take a big step using a small stomach — chew four small bites before swallowing once.*

$$B_{\text{eff}} = B_{\text{device}} \times n_{\text{accum}} \times n_{\text{GPUs}}$$

On the Spark $n_{\text{GPUs}}=1$, so $B_{\text{eff}} = B_{\text{device}} \times n_{\text{accum}}$.

$B_{\text{device}}{=}4, n_{\text{accum}}{=}8 \Rightarrow B_{\text{eff}}{=}32$, using the memory of 4.

**Not exactly equivalent, and the course should say so honestly:**
- BatchNorm statistics are computed per *device* batch → not equivalent. (Mostly moot: transformers use LayerNorm/RMSNorm, which are per-example. Diffusion UNets sometimes have GroupNorm — also per-example. So in practice, for this course's models, this caveat rarely bites. Say that.)
- **Throughput is worse**: 8 forward/backward passes at B=4 are less efficient than 1 at B=32 (kernel launch overhead, lower arithmetic intensity, poorer tensor-core occupancy).
- Dropout masks differ. Irrelevant in expectation.

**The rule:** maximize $B_{\text{device}}$ until just before OOM, *then* use accumulation to reach your target $B_{\text{eff}}$. Not the reverse.

*Misconception:* "gradient accumulation saves memory." It doesn't save *versus* the same $B_{\text{device}}$ — it lets you *simulate* a large batch at small-batch memory. The memory is whatever $B_{\text{device}}$ costs. Subtle but the phrasing matters.

*Known trap [REPORTED]:* there was a well-publicized gradient-accumulation normalization bug in HF Trainer (loss averaged per-microbatch rather than per-token, biasing toward short sequences) — **fixed**, but it's a great "even the pros get this wrong" sidebar and a reason to trust `assert`s over vibes.

### 7.4 Precision — fp32 / fp16 / bf16 / fp8 / fp4

**Intuition:** *A float is a sign, an exponent (how big), and a mantissa (which one). Exponent bits buy you **range**; mantissa bits buy you **precision**. In deep learning, range matters more — you can tolerate a fuzzy number but not an infinite one.*

| dtype | bits | exp | mant | max | rel. precision |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~3.4e38 | ~1e-7 |
| **bf16** | 16 | **8** | **7** | **~3.4e38** | ~1e-2 |
| fp16 | 16 | **5** | **10** | **~65,504** | ~1e-3 |
| fp8 E4M3 | 8 | 4 | 3 | ~448 | ~1e-1 |
| fp8 E5M2 | 8 | 5 | 2 | ~57,344 | — |
| NF4 | 4 | — (nonlinear, 16 levels fitted to a normal) | | | |

**The key fact, stated as the one thing to remember:** *bf16 has **the same exponent range as fp32** and just fewer mantissa bits. fp16 has fp32's precision-ish and a **drastically smaller range**.* That's why fp16 needs a **GradScaler** (gradients underflow below ~6e-8 and silently become zero) and bf16 doesn't.

**Recommendation: use bf16 on the Spark. Always.** Blackwell supports it natively; there is no reason to reach for fp16 on this hardware. fp16 + GradScaler is a whole chapter of pain (`inf` checks, skipped steps, scale-factor dynamics) that this learner never needs to experience. **Mention GradScaler in one sentence, in the past tense.** This is a case where "the field moved on" is the honest teaching.

fp8: [UNVERIFIED for *training* on GB10 specifically] — Blackwell has FP8 tensor cores, and the LMSYS numbers use FP8 for *inference* [VERIFIED]. FP8 *training* recipes exist (`transformer-engine`) but are finicky and arch-sensitive; on aarch64 + sm_121 I'd expect friction. **Recommend: teach fp8 as an inference/weight format, not as a training precision, and say why.**

NVFP4/MXFP4: the Spark's petaflop number is FP4 [VERIFIED], GPT-OSS 20B was benchmarked at MXFP4, Flux at FP4 [VERIFIED]. FP4 is an *inference* format. Worth a paragraph in the diffusion track since Flux-at-FP4 is the headline diffusion number.

### 7.5 Throughput measurement

Covered in Rung 9 (§3). The two rules: **synchronize, and warm up.** And measure `tokens/sec`, not `steps/sec` — steps/sec is meaningless across batch/seq changes and makes optimizations look better or worse than they are.

Also: `nvidia-smi` on the Spark reports differently than on a discrete GPU (unified memory). `torch.cuda.max_memory_allocated()` is the number to trust for *your process*; `nvidia-smi` for *the machine*. They will disagree. Say so before it confuses him.

---

## 8. Interactive demo specifications

### 8.1 The Roofline / Arithmetic-Intensity demo ★ (highest value in this brief)

**Plotted:** log-log. x = arithmetic intensity $I$ (FLOP/byte), y = achieved throughput (FLOP/s). The roofline:
$$P(I) = \min(P_{\text{peak}},\ BW \times I)$$
Draw as two straight lines meeting at the ridge $I^* = P_{\text{peak}}/BW$. Overlay **actual measured points** from §5.3 as dots the learner can hover.

**Sliders:**
1. **Batch size** $B$: 1 → 64. Moves a marker along the roofline: $I \approx 2B$ for decode. The marker crawls up the diagonal and then flattens.
2. **Weight precision**: fp32 / bf16 / fp8 / nf4 → changes bytes/param → $I \approx \frac{2B}{b_W}$ → marker moves right.
3. **Machine**: DGX Spark (273 GB/s, ~125 TF) / RTX 5090 (~1,790 GB/s, ~210 TF dense bf16) / H100 (3,350 GB/s, ~990 TF bf16). Redraws the roof. **[Peak numbers for 5090/H100: UNVERIFIED — verify. The H100 bandwidth figure for SXM HBM3 is well-known-ish; the 5090's is from memory.]**

**Exact JS:**
```js
const I  = (2 * B) / bytesPerParam;            // FLOP/byte, decode regime
const P  = Math.min(P_peak, BW * I);           // FLOP/s
const tps = P / (2 * N_params);                // tokens/sec, since ~2N FLOP/token
```
Plot the roof by sweeping `I` over `[1e-2, 1e4]` logarithmically.

**The insight when he drags:** at B=1 the marker sits deep on the diagonal — *bandwidth-bound, 200× below the roof.* Drag B to 32 and the marker climbs and the tok/s readout jumps ~18×, matching the **verified 20.5 → 368 tok/s** measurement. **Then switch the machine to RTX 5090 and watch the roof rise 6× — but only the diagonal part moves the marker, because the bottleneck was never the roof.** That's the whole hardware chapter in one drag.

**Validation:** the demo must reproduce the LMSYS numbers within a factor of ~2. If it doesn't, the model is wrong and should be fixed or its assumptions stated. Print the measured point next to the predicted one. *Showing the model being imperfect is better teaching than hiding it.*

### 8.2 The Memory Budget calculator ★

**Plotted:** a stacked horizontal bar, 0 → 128 GB, with a hard red line at 119.2 GiB and a dashed amber line at ~110 GiB ("realistic usable"). Segments: Weights (blue) / Gradients (orange) / Optimizer (red) / Activations (green) / Overhead (grey).

**Controls:**
- Model: dropdown {0.5B, 3B, 8B, 14B, 32B, 70B, 120B, custom N}, which also sets $d$ and $n_L$
- Method: {Full, LoRA, QLoRA} + rank slider $r$ ∈ {4…256} + target-modules toggle {attn-only, attn+MLP}
- Precision: {fp32, bf16, fp8, nf4}
- Optimizer: {SGD, SGD+mom, AdamW fp32, AdamW mixed, AdamW 8-bit}
- Batch $B$: 1 → 64 (log)
- Seq $L$: 128 → 32768 (log)
- Gradient checkpointing: on/off
- Device: {DGX Spark 128 GB, RTX 5090 32 GB, A100 80 GB, H100 80 GB}

**Exact JS** (§6.2 verbatim, so page and widget cannot drift):
```js
const bW = {fp32:4, bf16:2, fp8:1, nf4:0.5}[precision];
const kO = {sgd:0, sgdm:4, adamw32:8, adamw_mixed:12, adamw8bit:2}[optimizer];

const Nt = method === "full" ? N
         : nLayers * nTargets * 2 * r * d;     // LoRA/QLoRA adapter params

const M_W = N  * bW;
const M_G = Nt * 2;                            // grads in training dtype
const M_O = Nt * kO;
const c   = 16;
const M_A = c * B * L * d * nLayers * 2 / (ckpt ? nLayers : 1);
const total = (M_W + M_G + M_O + M_A) * 1.10;  // +10% overhead
```
Readout in **GiB**, plus a verdict: ✅ fits / ⚠️ tight / ❌ OOM.

**The insight when he drags:**
1. Set 8B / Full / AdamW-mixed. **Red (optimizer) is the biggest segment.** Not blue. This alone is worth the widget.
2. Flip to LoRA. **Red and orange vanish.** Blue doesn't move an inch. *"LoRA didn't shrink the model."*
3. Now drag $r$ from 4 to 256. **Almost nothing happens.** The single most valuable insight in the widget: *rank is nearly free — you have been agonizing over a parameter that costs you 0.4 GiB at r=256.* This will genuinely surprise a ComfyUI user who has been treating rank as the big dial.
4. Now drag $L$ from 512 to 16384 with checkpointing off. **Green explodes and swallows the bar.** *Sequence length, not model size, is what kills you.*
5. Flip checkpointing on. Green collapses. Note the "+35% step time" badge that appears.

**Bake in the verified points** as preset buttons: "NVIDIA's 3B full FT," "The 8B/16K/batch-16 run," "70B QLoRA." Each preset shows the widget's estimate next to the *measured* number. **Where they disagree (they will — see Example B), say so on the widget.** A calculator that admits its error is a better teacher than one that doesn't.

### 8.3 Autograd tape visualizer

**Plotted:** a DAG for a tiny expression, e.g. $L = (\sigma(w x + b) - y)^2$. Nodes = tensors, edges = ops. Forward pass animates values flowing right; `.backward()` animates gradients flowing left, each edge labelled with its local derivative.

**Controls:** sliders for $w$, $b$, $x$, $y$; a "step" button walking one node at a time; a toggle for `requires_grad` on $w$ and on $x$.

**Exact JS:** hard-code the chain rule for this one expression. $z = wx+b$; $a = \sigma(z)$; $L = (a-y)^2$. Then $\frac{\partial L}{\partial a} = 2(a-y)$, $\frac{\partial L}{\partial z} = 2(a-y)\,a(1-a)$, $\frac{\partial L}{\partial w} = 2(a-y)\,a(1-a)\,x$, $\frac{\partial L}{\partial b} = 2(a-y)\,a(1-a)$, $\frac{\partial L}{\partial x} = 2(a-y)\,a(1-a)\,w$.

**Insight:** toggling `requires_grad=False` on $x$ **greys out exactly one edge** and changes nothing else. That's what freezing a weight *is*. It sets up LoRA (freeze $W_0$) with zero new machinery. **Dependency-wise this demo must come before LoRA.**

Bonus: a "backward twice" button that turns the graph red and prints the real error text. Learners remember errors they caused on purpose.

### 8.4 Precision explorer

**Plotted:** a number line, log scale, $10^{-45}$ → $10^{40}$, with representable-range bars for fp32/bf16/fp16/fp8-E4M3. Below: the actual bit pattern of a value the user types.

**Controls:** a text box for a decimal number; a dtype selector. Shows the exact stored value, the rounding error, and flags overflow→`inf` / underflow→`0`.

**Exact JS:** implement encode/decode properly. fp16: `Math.fround` won't do it — write it, it's 20 lines (sign, exponent bias 15, 10 mantissa bits, handle subnormals). bf16 is easier: truncate/round the fp32 bit pattern to the top 16 bits.

**Insight:** type `6e-8` — a plausible gradient magnitude. **bf16: fine. fp16: 0.** That single keystroke is the entire argument for bf16 and for why GradScaler had to exist. Then type `70000`: **fp16 → inf. bf16 → 70016.** Wrong, but *finite*, and finite errors train.

### 8.5 Broadcasting / shape puzzle

**Plotted:** two shape-editors and the resulting shape, with the alignment-from-the-right rule drawn as columns.

**Insight:** build `(N,1)` and `(N,)` and watch the result light up as `(N,N)` **in red**, with the caption *"this does not error. this trains. your loss is wrong."* Small demo, huge payoff.

---

## 9. Environment setup — one opinionated path

### 9.1 The recommendation

**On the DGX Spark: use NGC containers. Full stop.**

Rationale, and it's specific to this learner's hardware rather than a general dogma:
- NGC PyTorch containers ship aarch64 builds matched to the host CUDA (13.0) and NVRTC, with **sm_120 + compute_120 PTX that JITs to sm_121** [REPORTED]. The single hardest problem on this machine — arch/toolkit matching on ARM — is *pre-solved*.
- DGX OS ships with the container runtime configured.
- It is what NVIDIA tests. On aarch64 you want to be on the tested path.

```bash
docker run --gpus all -it --rm \
  -v $HOME/course:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:25.10-py3
```
**[UNVERIFIED: `25.10-py3` is the tag reported as matching CUDA 13.0. By July 2026 there will be newer tags (26.xx). Verify the current tag; the *principle* — match the container's CUDA to the host's — is what's stable.]**

Then inside, for the HF layer only:
```bash
uv pip install --system \
  "transformers==5.14.1" "peft==0.19.1" "trl>=1.8,<2" \
  "datasets" "accelerate" "bitsandbytes" "diffusers==0.39.0"
```
**Do not `pip install torch` inside an NGC container.** It will replace NVIDIA's tuned aarch64 build with a generic one and you will spend an evening wondering why things got slower or broke. **This is a warning box and it is the single highest-value sentence in the setup page.**

### 9.2 Fallbacks

**(a) uv, native, no container** — for the learner's laptop, or on the Spark if he dislikes containers:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init && uv venv --python 3.12
uv add torch --index https://download.pytorch.org/whl/cu130   # verify cu130 aarch64 exists
uv add transformers peft trl datasets accelerate bitsandbytes
```
**[UNVERIFIED: whether `download.pytorch.org/whl/cu130` publishes aarch64/sbsa wheels, and whether the default PyPI `torch` aarch64 wheel is CUDA-enabled *for sm_120/121*. One source says CUDA-enabled wheels are the pip default on Linux aarch64 [VERIFIED-ish]; another documents real aarch64 CUDA install pain. **The architect must test this on hardware.** It is a 10-minute test and it determines whether §9.2(a) is a fallback or a lie.]**

**uv vs conda, the honest 2026 answer** [VERIFIED as the consensus framing]: uv is 10–100× faster and handles packages+venvs+Python versions+lockfiles; conda/mamba still wins when you need *non-Python system libraries* (CUDA toolkit, GDAL, ffmpeg) that pip cannot reach. **For this course, the container supplies the system layer, so uv is the right tool for the Python layer and conda is unnecessary.** That's a clean, defensible position: *container for the C/CUDA world, uv for the Python world, conda for neither.*

**(b) conda/mamba** — mention in two sentences as "what you'll see in older tutorials." Don't teach it.

**(c) Rent a GPU** — see §12. For the numpy/PyTorch rungs (1–5), anything works, including a laptop CPU. **Say that loudly**: rungs 1–5 do not need the Spark, and a learner without one should not be blocked.

### 9.3 The non-negotiables

- `export HF_HOME=/mnt/nvme/hf` — the model cache will hit hundreds of GB. Default is `~/.cache/huggingface`. On a 1 TB Spark this matters within a week.
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — free fragmentation relief.
- `huggingface-cli login` for gated models (Llama). Qwen/SmolLM3 avoid this — **another point for Qwen3-4B as the course's default: no gate, no waiting on an approval email mid-lesson.**
- A `verify_env.py` that prints and asserts:
  ```
  torch 2.13.0 | cuda 13.0 | device NVIDIA GB10 | capability (12,1) | 128 GB unified
  bf16 supported: True | bnb import: OK | transformers 5.14.1
  ```
  **Ship this as artifact 00.** Every support question the course would otherwise generate starts with "paste the output of `00_verify_env.py`."

---

## 10. Tracking, checkpointing, reproducibility — proportionate

The learner has one machine and one experiment at a time. **Do not teach MLOps.** One page:

**Tracking:** TensorBoard is in the container and `Trainer` writes to it with `report_to="tensorboard"`. `tensorboard --logdir out/`. That's the whole lesson. Mention W&B exists in one sentence; don't make him make an account mid-course.

**What to log, and it's short:** train loss, eval loss, learning rate, grad-norm, tokens/sec, peak memory. **grad-norm is the underrated one** — it spikes before the loss diverges, so it's your early-warning light. That's a *why*, and it's the reason it's on the list.

**Checkpointing:** `save_strategy="steps"`, `save_steps=100`, `save_total_limit=2`. The arithmetic that motivates `save_total_limit`: a full-FT 8B checkpoint with fp32 AdamW state is $8\times10^9 \times (2 + 12) = 112$ GB **each**. Three of them fills your NVMe. LoRA adapters are **~34 MB** at r=16. **That 3,300× ratio is another place LoRA's real value shows up** — and it's the reason he can keep 50 experiments on disk. Connect it back.

**Reproducibility, honestly:**
```python
torch.manual_seed(42); np.random.seed(42); random.seed(42)
```
gets you *most* of the way. Bit-exact reproducibility additionally needs `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and **costs real speed**, and **still won't survive a driver or library version change**.

**Say the true thing:** *floating-point addition is not associative, GPU reductions are non-deterministically ordered, so two runs of the same code on the same machine can differ in the last bits, and those bits amplify over 10,000 steps.* The practical target is **statistical** reproducibility (same seed → same-shaped curve, same final eval within noise), not bit-exactness. Chasing bit-exactness is a beginner's trap that costs throughput for no benefit. Set the seed, log the versions, move on.

---

## 11. Genuine disagreements — flag, don't resolve

### 11.1 LoRA rank
No consensus. r=8–16 is the common default; the LoRA paper's own claim was that very low rank suffices; later work argues higher rank matters for *new knowledge* as opposed to *style/format adaptation*. And there's a live argument that $\alpha/r$ scaling makes rank comparisons confounded (rank-stabilized LoRA / `use_rslora` scales by $\alpha/\sqrt{r}$ instead, precisely because the standard scaling misbehaves at high rank). **Present the trade, give r=16 as a starting point, and point at §8.2's demo #3: memory is nearly rank-independent, so *just try both* — the cost of the experiment is an hour, not a GPU.** That's a better answer than a number.

### 11.2 LoRA vs. full fine-tune quality
Genuinely contested. LoRA is generally accepted as competitive for style/format/instruction-following adaptation and generally accepted as weaker for injecting substantial new knowledge. There's a well-known "LoRA learns less and forgets less" framing (the forgetting-less part is a *feature* if you care about not destroying base capabilities). **The Spark can do both for an 8B model** [VERIFIED] — so this course is in the unusual position of being able to *run the experiment* rather than cite it. **Recommend it as an optional capstone.** That's a strong course.

### 11.3 Optimal batch size / LR
There is no formula. The linear scaling heuristic ($\text{lr} \propto B$) and the square-root heuristic ($\text{lr} \propto \sqrt{B}$) both have support and both fail outside their regimes. Practical anchors for the course, which are conventions rather than derivations and should be labelled as such:
- **LoRA SFT: lr = 1e-4 to 2e-4**, cosine, warmup_ratio 0.03
- **Full FT: lr = 1e-5 to 2e-5** — *10× lower, and the learner should be asked why before being told.* (Because you're moving 8 billion parameters that already encode everything the model knows, versus 17 million that started at zero.)
Say plainly: **these are folklore that works, not theory.** The course is more trustworthy for admitting it.

### 11.4 Does the Spark make sense at all?
Real disagreement, and the course should host it rather than duck it. The bear case: $4,699 buys ~1,500 hours of RunPod H100 [VERIFIED: ~$1.99–2.69/hr], each H100 hour being ~4× a Spark hour → **~6,000 Spark-equivalent hours**, and an RTX 5090 is 4× faster at decode for less money [VERIFIED]. The bull case: 128 GB coherent at 240 W does things no 5090 can do at any price [VERIFIED via the 8B/16K full-FT result], no data leaves the building, no meter runs, no cold-start, and it's the same Arm+CUDA stack as GB200. **Both are true. Present both. The learner already owns one — the useful framing isn't "was this wise" but "what is this machine's comparative advantage, and how do I route work to it or away from it?"** That's the adult version of the question.

### 11.5 Numbers I could not verify and would not ship without checking
- **Dense BF16 peak FLOP/s for GB10.** NVIDIA publishes FP4-sparse (1 PFLOP) and TOPS (1,000). The dense-bf16 number is inferred, not published, as far as I could find. §5.2 and §8.1 depend on it. **Either find it or present the roofline with an explicitly-labelled estimated roof.** The latter is honest and still teaches the concept.
- The `bitsandbytes` 0.49.2 release date (the page returned something inconsistent).
- Whether the vLLM sm_121/aarch64 issue is resolved.
- Exact `SFTConfig`/`LoraConfig` field names at 0.29.1 / 0.19.1.
- `d`, `n_L` for the specific models used in §6.3.
- torchtune's `fft-8b.yaml` optimizer choice — needed to close Example B.
- Unified-memory OOM semantics on GB10.

---

## 12. Local vs. rented — honest cost framing

**Verified July 2026 rates:**

| GPU | $/hr |
|---|---|
| H100 PCIe (RunPod) | **$1.99** |
| H100 SXM (RunPod) | **$2.69** |
| H100 SXM (Lambda) | **$4.29** |
| H100 (Vast.ai range) | **$1.49–6.98** |
| A100 80GB (RunPod) | **$1.19–1.39** |
| A100 80GB (Spheron spot) | **$0.60** |
| B200 (Spheron spot) | **$2.12** |
| B200 (RunPod on-demand) | **$5.89** |

**Market note [VERIFIED]:** H100 rates fell **64–75%** from Q4 2024 to early 2026 — $8–10/hr became $2–3/hr. **This cuts against buying hardware, and the course should say so**, because it's the honest read: the rental market got much cheaper while the learner's Spark did not.

**The framing:**

$$t_{\text{breakeven}} = \frac{C_{\text{hardware}}}{r_{\text{rent}} \times s} \quad \text{where } s = \frac{\text{rented throughput}}{\text{local throughput}}$$

At $C = \$4{,}699$, $r = \$2.50$/hr, $s = 4$: $t = \frac{4699}{2.50 \times 4} \approx$ **470 hours of local compute** — ~3 weeks of 24/7 use. Add ~$0.05/hr electricity at 240 W and it barely moves.

**But breakeven is the wrong frame and the course should say so.** The right frames:

**Use local when:** the data can't leave; you're iterating (100 two-minute runs beats one two-hour run, and the cloud's friction tax on iteration is brutal); it's after midnight and you want to try something; the job runs for days and you don't want a meter; you're learning (the *worst* time to be paying by the second is while you're making mistakes — **and this course is exactly that time**).

**Rent when:** you need >128 GB (a real 70B full FT); you need it *fast* (a week of Spark = a day of 8×H100); it's one job, once; you need a GPU the Spark isn't (some x86-only stack).

**The honest sentence for this learner:** *"You own the machine. The question is not whether it was a good buy — it's which jobs to route to it. Route iteration, privacy, and overnight work to the Spark. Route 'I need 400 GB' and 'I need it by Friday' to the cloud. The skills are identical; only the `--device` line changes."*

---

## 13. Dependency order

```
00 verify_env
    ↓
[TENSOR MECHANICS]  storage/shape/stride → broadcasting → dtype
    ↓
[CALCULUS BY HAND]  chain rule → 01_backprop_numpy → GRADIENT CHECK PASSES
    ↓
[AUTOGRAD]  8.3 tape demo → 02_backprop_torch → the numpy/torch assert passes
    ↓                              ↑
    │                   ← must come AFTER 01, or autograd is magic
[nn.Module + LOOP]  03_mlp_mnist  (101,770 params)
    ↓
[PRECISION]  8.4 explorer → bf16 vs fp16   ─┐
[MEMORY BUDGET]  §6 + 8.2 calculator       ─┤ both feed everything below
[ROOFLINE]  §5.2 + 8.1 demo                ─┘
    ↓
[HARDWARE]  DGX Spark §5 — lands only after roofline
    ↓
[TOKENIZERS]  04_tokenizer_and_shapes
    ↓
[LoRA]  05_lora_from_scratch  ← REQUIRES: freezing (8.3), memory budget (8.2)
    ↓
    ├─────────────────────────┬──────────────────────────┐
[LLM TRACK]              [DIFFUSION TRACK]         [SHARED]
06_finetune_llm_peft     07_finetune_diffusion     08_memory_budget
                                                   09_measure_throughput
    └─────────────────────────┴──────────────────────────┘
    ↓
10_merge_and_serve → capstone (his own data)
```

**Hard ordering constraints and *why*:**

1. **`01_backprop_numpy` strictly before `02_backprop_torch`.** If autograd arrives first it is never demystified. The whole ladder's credibility rests on this.
2. **Precision before memory budget.** $b_W$ is a *term* in the memory equation. You cannot compute the budget without knowing what a bf16 costs.
3. **Memory budget before hardware.** "128 GB" is a meaningless number until he can spend it. Introduce the currency, then the wallet.
4. **Roofline before any benchmark number.** Otherwise "1 PFLOP but 50 tok/s" reads as a contradiction or a lie, and he'll (correctly) stop trusting the course. Give him the model *first*, and every subsequent number confirms it instead of confusing him.
5. **Freezing (8.3) before LoRA.** LoRA = frozen $W_0$ + trainable $BA$. If "frozen" isn't already concrete, LoRA is two mysteries.
6. **`05_lora_from_scratch` before `peft` is imported.** Same principle as #1: implement it, then let the library do it. `peft` becomes a labour-saver rather than an oracle.
7. **Tokenizers before LLM fine-tuning.** Non-negotiable. Every LLM training bug is a shape/mask/label bug wearing a costume.
8. **`09_measure_throughput` early enough to be used**, not as a postscript. Ideally right after `03`, so that every subsequent page can say "measure it" and mean something.

---

## 14. Recommended file list

Ships as `code/` with a `README.md`, one `pyproject.toml`, one `uv.lock`, one `Dockerfile`.

| File | Purpose | Attaches to |
|---|---|---|
| `00_verify_env.py` | Print/assert torch, CUDA, capability, bf16, memory. First thing he runs. | Setup |
| `01_backprop_numpy.py` | 2-8-1 MLP, hand-derived grads, **gradient check < 1e-7**. Proves the math. | Backprop |
| `02_backprop_torch.py` | Same net, autograd, asserts agreement with `01`. | Autograd |
| `03_mlp_mnist.py` | First `nn.Module` + `DataLoader`. 101,770 params, ~97.5% test. | Training loop |
| `04_tokenizer_and_shapes.py` | Tokenize, print ids/shapes/masks. No training. | Tokenization |
| `05_lora_from_scratch.py` | LoRA in ~15 lines of `nn.Module`. B=0 init demo. | LoRA |
| `06_finetune_llm_peft.py` | Qwen3-4B LoRA SFT, `transformers`+`peft` (+optional `trl`). | LLM fine-tune |
| `07_finetune_diffusion_lora.py` | `diffusers` LoRA. "ComfyUI, unwrapped." | Diffusion fine-tune |
| `08_memory_budget.py` | CLI memory calculator. Same math as the widget. | Memory (recurring) |
| `09_measure_throughput.py` | Correct timing: synchronize + warmup. Peak memory. | Performance |
| `10_merge_and_serve.py` | `merge_and_unload`, save, reload, generate. | Capstone |
| `utils/seed.py` | Seeding + version stamping. | (imported) |
| `utils/memory.py` | Peak-memory context manager. | (imported) |
| `pyproject.toml` + `uv.lock` | The pins. | Setup |
| `Dockerfile` | `FROM nvcr.io/nvidia/pytorch:<tag>` + the HF layer. | Setup |

**13 Python files. Hold the line.** Every artifact the architect is tempted to add should have to displace one of these.

---

## 15. The five things this brief most wants the architect to take

1. **The roofline calculation ($I^* \approx 458$ FLOP/byte vs. decode's $I \approx 2$) is the hardware chapter's thesis.** It is one line of arithmetic and it explains every benchmark number the learner will ever see on this machine. Build the chapter around it, not around a spec table.

2. **The memory budget is the course's spine, not a sidebar.** Weights / grads / optimizer / activations, with $k_O = 12$ for mixed-precision AdamW. It should recur so often it becomes reflex. The LoRA insight (**430× less optimizer memory, 0× less weight memory**) is the payoff.

3. **Example B is the best teaching moment in the brief and it's currently unfinished.** Naive math says the verified 8B/16K/batch-16 full FT cannot fit; NVIDIA reports it at 80% of 128 GB. The gap is 8-bit optimizer + gradient checkpointing. **Someone must fetch torchtune's `fft-8b.yaml` and close it.** A verified contradiction-and-resolution is worth ten pages of exposition.

4. **`01` → `02` → the passing assert is the emotional spine.** Do not let it get cut for length. Everything downstream is trust, and this is where trust is minted.

5. **transformers v5 + trl's churn means every tutorial he Googles is wrong.** Warn him explicitly and early: *"if you find a tutorial with `load_in_4bit=True` or `tokenizer=`, it's v4-era. The concepts transfer; the code doesn't."* This single warning box will save him more hours than any other paragraph in the course.

---

## Sources

- [DGX Spark Hardware Overview — NVIDIA Docs](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [How NVIDIA DGX Spark's Performance Enables Intensive AI Tasks — NVIDIA Technical Blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/)
- [NVIDIA DGX Spark In-Depth Review — LMSYS Org](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- [Unlock Reasoning in Llama 3.1-8B via Full Fine-Tuning on NVIDIA DGX Spark — PyTorch Blog](https://pytorch.org/blog/unlock-reasoning-in-llama-3-1-8b-via-full-fine-tuning-on-nvidia-dgx-spark/)
- [transformers MIGRATION_GUIDE_V5.md](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [torch on PyPI](https://pypi.org/project/torch/) · [transformers](https://pypi.org/project/transformers/) · [trl](https://pypi.org/project/trl/) · [peft](https://pypi.org/project/peft/) · [diffusers](https://pypi.org/project/diffusers/)
- [TRL SFT Trainer docs](https://huggingface.co/docs/trl/en/sft_trainer) · [trl releases](https://github.com/huggingface/trl/releases)
- [bitsandbytes releases](https://github.com/bitsandbytes-foundation/bitsandbytes/releases)
- [torchtune repo](https://github.com/meta-pytorch/torchtune) · [Introducing torchforge — PyTorch](https://pytorch.org/blog/introducing-torchforge/)
- [vLLM issue #36821 — No sm_121 support on aarch64 / DGX Spark](https://github.com/vllm-project/vllm/issues/36821)
- [DGX Spark GB10 CUDA 13.0 Python 3.12 SM_121 — PyTorch Forums](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744)
- [NVIDIA/dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- [Runpod pricing](https://www.runpod.io/pricing) · [Lambda pricing](https://lambda.ai/pricing) · [Vast.ai pricing](https://vast.ai/pricing)
- [H100 Rental Prices Compared — IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [The Best Open-Source Small Language Models in 2026 — BentoML](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
