# RESEARCH BRIEF — ARCHITECTURES: FROM MLP TO TRANSFORMER

**Audience for this brief:** the curriculum architect writing the build spec.
**Date of research:** 2026-07-16. All time-sensitive claims carry a confidence tag.
**Scope:** the bridge from "a net" to "the models we actually fine-tune." Ends where the
LLM track and the diffusion track fork.

**Confidence legend:**
- `[V]` verified this session against a primary source (HF config.json, paper, official blog)
- `[V-sec]` verified against a secondary source (reputable blog/survey), not primary
- `[M]` from model memory, not verified — architect should re-check before it ships
- `[D]` genuinely disputed in the field; the course must present it as disputed

---

## 0. THE SPINE — DEPENDENCY ORDER (read this first)

This is the order I recommend, and the *reason* each edge exists. Violating an edge produces
a specific, predictable confusion, named below.

```
[from prior brief: MLP, backprop, gradient descent, loss]
        |
  (A) Architecture = inductive bias        <- the organizing principle; everything hangs here
        |
        +---> (B) Weight sharing & locality  ->  (C) CNN: kernel/stride/pad/pool/RF
        |                                             |
        |                                             +--> feeds DIFFUSION TRACK (U-Net)
        |
        +---> (D) Embeddings & dot-product similarity
                    |
                    +---> (E) RNN + the long-range problem + WHY they lost
                    |          (E exists ONLY to motivate F. Do not over-teach it.)
                    |
                    +---> (F) ATTENTION  <-- the load-bearing chapter, ~35% of this unit
                                |
                                +--> (F1) Q/K/V as soft lookup
                                +--> (F2) scaled dot-product + why sqrt(d_k)
                                +--> (F3) multi-head
                                +--> (F4) self vs CROSS  --> DIFFUSION CONDITIONING (flag hard)
                                +--> (F5) causal masking --> LLM TRACK
                                +--> (F6) O(n^2) + 2026 mitigations
                                        |
        (G) Residual connections --------+
        (H) Normalization (LayerNorm/RMSNorm) --+
        (I) Positional encoding -> RoPE --------+
                                                |
                                    (J) THE TRANSFORMER BLOCK (assembly)
                                                |
                                    (K) Encoder / decoder / enc-dec; why decoder-only won
                                                |
                                    (L) Parameter counting on a real model card
                                                |
                                    (M) Scaling laws, MoE, "emergence"
                                                |
                        ===== FORK: LLM track | Diffusion track =====
```

**Why each edge, specifically:**

- **A before everything.** Without the inductive-bias frame, the learner experiences CNNs and
  transformers as arbitrary trivia ("why 3x3? why 8 heads?"). With it, every architectural
  choice is an answer to "what do I believe about my data?"
- **D before F.** Attention is *entirely* built on dot-product similarity between vectors. A
  learner who has not internalized "dot product = alignment = similarity" will read
  `softmax(QK^T/sqrt(d_k))V` as symbol soup. **This is the single most common failure point in
  every transformer tutorial ever written.** Budget real time on D.
- **E before F, but E is a *motivation chapter*, not a *skills chapter*.** The learner will
  never implement an LSTM. E exists to make the learner *feel* the pain that attention solves
  (sequential bottleneck + gradient decay over time), so that attention lands as relief rather
  than as one more formula. Do not teach LSTM gate equations in full. See §4.
- **G and H before J.** You cannot assemble the block without knowing why the `+ x` and the
  norm are there. But G/H *after* F, because the learner needs to have seen a stack of
  attention layers before "why does a 36-layer stack train at all?" is a question they care
  about.
- **I before J.** Position must be motivated by attention's permutation-equivariance, which
  only becomes visible once F is understood. Teaching RoPE before attention is teaching an
  answer to an unasked question.
- **L after J.** Parameter counting is the *payoff*: it converts "I read a diagram" into "I can
  look at any model card and know what every number means." Put it immediately after assembly
  while the dims are hot.
- **C feeds diffusion, F4 feeds diffusion, F5 feeds LLM.** These are the three explicit
  hand-offs to the downstream tracks. Mark them in the text with a persistent visual token so
  the learner sees the trunk branching.

**Recommended page budget** (of a ~48-page course, this unit is roughly pages 14–30):
A: 1 | B+C: 3.5 | D: 2.5 | E: 1.5 | F: 6 | G+H: 1.5 | I: 1.5 | J: 1.5 | K: 1 | L: 1.5 | M: 1.5

---

## 1. (A) ARCHITECTURE IS INDUCTIVE BIAS

### Intuition first (lead with this sentence)

> **An architecture is a hard-coded prejudice about your data. It's the set of beliefs you build
> into the wiring so the network doesn't have to learn them from scratch — and every belief you
> install is a shortcut if you're right and a wall if you're wrong.**

Second sentence, for the "rusty but trained" learner: *an MLP can in principle represent any
function (universal approximation), so architecture is never about what's* representable *—
it's about what's* **learnable from finite data in finite time**.

### The math worth writing down

The universal approximation theorem is the setup for the joke. State it, then puncture it:

$$\text{For any continuous } f:[0,1]^n \to \mathbb{R} \text{ and } \varepsilon>0, \;\exists\, N,\; w_i \in \mathbb{R},\; \mathbf{a}_i \in \mathbb{R}^n,\; b_i \in \mathbb{R} \text{ s.t. } \left|\sum_{i=1}^{N} w_i\,\sigma(\mathbf{a}_i^\top \mathbf{x} + b_i) - f(\mathbf{x})\right| < \varepsilon$$

- $n$ — input dimension (scalar, dimensionless)
- $N$ — number of hidden units; **the theorem says nothing about how large $N$ must be**
- $\sigma$ — any non-polynomial activation
- $\mathbf{a}_i \in \mathbb{R}^n$, $b_i \in \mathbb{R}$, $w_i \in \mathbb{R}$ — learned params

**The puncture:** the theorem is an *existence* claim about weights. It is silent on (i) how big
$N$ is, (ii) whether SGD finds those weights, (iii) how much data it takes. Architecture is
entirely about (i)-(iii).

### The killer worked example — make the learner feel it

A 224×224 RGB image. Fully-connected first layer to 1000 hidden units:

- Input dim: $224 \times 224 \times 3 = 150{,}528$
- Weights in layer 1: $150{,}528 \times 1000 = 150{,}528{,}000 \approx 1.5\times10^8$
- **In one layer.** In bf16: $\approx 301$ MB.

Now a conv layer, 64 filters of 3×3×3:

- Weights: $3 \times 3 \times 3 \times 64 = 1{,}728$, plus 64 biases $= 1{,}792$
- **84,000× fewer parameters**, and it sees every position.

The FC layer *could* learn to be translation-equivariant. It would need to independently
rediscover the same edge detector at all 50,176 spatial positions from data. The conv layer is
*handed* that fact. That is inductive bias, in one number: **1,792 vs 150,528,000.**

### The table the course should print once and refer back to forever

| Architecture | The belief it installs | Mechanism that installs it | Fails when |
|---|---|---|---|
| MLP | none (all inputs equally related) | dense matmul | data has any structure at all |
| CNN | nearby things relate; the same pattern matters everywhere | local kernels + weight sharing | relationships are long-range or non-spatial |
| RNN | the past matters through a running summary; order is causal | recurrent state | dependencies are long-range or you want parallelism |
| Transformer | **any token may relate to any other; which ones is data-dependent** | attention | $n$ is huge (cost) or data is tiny (no bias to lean on) |
| MoE | different inputs need different sub-functions | routed sparse FFN | routing collapses; you're memory-bound |

The transformer's bias is unusually *weak* — it barely assumes anything except "relationships
exist and should be learned per-input." **That is exactly why it needs so much data and why it
wins once you have it.** This sentence is the thesis of the whole unit. Say it in A, and again
in M when scaling laws make it quantitative.

### Misconceptions to box

- ❌ *"A bigger MLP can do anything a CNN can, so CNNs are just an optimization."*
  ✅ **Correction:** true about *representation*, false about *learning*. The MLP must learn
  translation-equivariance from data; a CNN gets it free. Point at 1,792 vs 150M and at the fact
  that a 150M-parameter first layer overfits CIFAR-scale data instantly. **What actually fixes
  this:** make the learner compute both parameter counts by hand. The abstract argument doesn't
  land; the arithmetic does.
- ❌ *"More inductive bias is better."*
  ✅ **Correction:** bias is a bet. At small data it pays; at large data it *caps* you. The
  transformer beating CNNs on vision (ViT) only above ~100M images `[M]` is the cleanest
  demonstration: same task, and which architecture wins *flips* as a function of dataset size.
- ❌ *"Architecture is where the intelligence lives."*
  ✅ **Correction:** in 2026 nearly every frontier open-weight model is the *same* block
  (RMSNorm → GQA/MLA attention → RMSNorm → SwiGLU FFN, residuals throughout, RoPE). What
  differs is scale, data, and training. Architecture is the *substrate*, not the secret. See §12.

---

## 2. (B),(C) CNNs — PROPORTIONATE, BECAUSE THE U-NET IS MADE OF THEM

**Framing note for the architect:** this learner already runs ComfyUI. He has *used* U-Nets.
Teach CNNs as "here is what the boxes in your ComfyUI workflow are made of," not as an image
classification course. **Do not spend pages on ImageNet history, AlexNet, VGG, ResNet-50
lineage.** The only things that must survive into the diffusion track are: kernel, stride,
padding, channels, pooling/downsample, transposed-conv/upsample, receptive field, and
equivariance. Roughly 3.5 pages. `[design judgment, not a fact claim]`

### Intuition first

> **A convolution is one small pattern-matching stencil that you slide over the whole image,
> asking at every position: "how much does the patch here look like my stencil?" The output is a
> map of where the pattern is. You learn the stencil; the sliding is free.**

Second sentence: *and because you slide the same stencil everywhere, an edge detector learned
from the top-left corner works in the bottom-right for free. That's the whole trick.*

### The math, with shapes

Discrete 2-D cross-correlation (which is what every framework calls "convolution" — flag the
naming lie, see misconceptions):

$$y[c_{\text{out}}, i, j] = b[c_{\text{out}}] + \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{u=0}^{k_h-1} \sum_{v=0}^{k_w-1} W[c_{\text{out}}, c_{\text{in}}, u, v]\; \cdot\; x[c_{\text{in}},\; s\cdot i + u - p,\; s\cdot j + v - p]$$

Symbol table (**print this; the course reuses these letters**):

| Symbol | Meaning | Type / shape / units |
|---|---|---|
| $x$ | input feature map | $(C_{\text{in}}, H, W)$ float tensor |
| $y$ | output feature map | $(C_{\text{out}}, H', W')$ |
| $W$ | kernel weights | $(C_{\text{out}}, C_{\text{in}}, k_h, k_w)$ |
| $b$ | bias | $(C_{\text{out}},)$ |
| $C_{\text{in}}, C_{\text{out}}$ | input/output channel counts | int |
| $k_h, k_w$ | kernel height/width | int, usually 3 |
| $s$ | stride | int, px per step |
| $p$ | zero-padding | int, px each side |
| $d$ | dilation | int, default 1 |

Output size:

$$H' = \left\lfloor \frac{H + 2p - d\,(k_h-1) - 1}{s} \right\rfloor + 1$$

Parameter count for one conv layer:

$$P_{\text{conv}} = C_{\text{out}}\,\big(C_{\text{in}} \cdot k_h \cdot k_w + 1\big)$$

FLOPs (multiply-accumulates ×2) for one conv layer:

$$\text{FLOPs} = 2 \cdot H' \cdot W' \cdot C_{\text{out}} \cdot C_{\text{in}} \cdot k_h \cdot k_w$$

**Note the asymmetry the course must call out:** conv parameters are independent of $H,W$;
conv *compute* is proportional to $H'W'$. This is why a U-Net at 1024×1024 is not bigger than at
512×512, but is ~4× slower. The learner running ComfyUI has felt exactly this and never had it
explained. **This is a high-value "oh, THAT's why" moment — do not skip it.**

### Worked example — carry it to a number

`nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1)` on a 64×64 feature map
(a plausible interior block of an SDXL-class U-Net at 512×512 latent resolution `[M]`, latents
being 64×64 for a 512px image at VAE downscale factor 8 `[V-sec]`):

- $P = 320 \times (320 \times 3 \times 3 + 1) = 320 \times 2881 = \mathbf{921{,}920}$ params
- $H'=W'=64$ (padding 1, stride 1, kernel 3 → size preserved: $\lfloor(64+2-2-1)/1\rfloor+1 = 64$ ✓)
- FLOPs $= 2 \times 64 \times 64 \times 320 \times 320 \times 9 = 2 \times 4096 \times 320 \times 2880 = \mathbf{7.55\times10^{9}}$ ≈ 7.5 GFLOP
- **Per single conv, per single denoising step.** At 30 sampling steps and ~40 such convs, that's
  order $10^{13}$ FLOPs before you've touched attention. This is why sampling is slow, and it's
  arithmetic the learner can verify.

### Receptive field — the concept that must survive

> **Intuition: the receptive field of a neuron is the patch of original image that can possibly
> influence it. Depth is how a network sees big things while only ever looking at small
> things.**

Recurrence for a stack of layers $\ell = 1 \ldots L$:

$$r_0 = 1, \qquad r_\ell = r_{\ell-1} + (k_\ell - 1)\cdot \prod_{j=1}^{\ell-1} s_j$$

- $r_\ell$ — receptive field size in input pixels after layer $\ell$
- $k_\ell$ — kernel size at layer $\ell$
- $s_j$ — stride at layer $j$; the product is the cumulative downsampling

**Worked, all 3×3 stride-1:** $r = 1, 3, 5, 7, 9, \ldots$ — grows *linearly*, $r_L = 2L+1$. To
see a 224-px object you need 111 layers. Unworkable.

**Worked, with stride-2 downsamples every 2 layers** (i.e. a U-Net encoder):
- L1 (k3,s1): $r = 1 + 2\cdot1 = 3$
- L2 (k3,s2): $r = 3 + 2\cdot1 = 5$, cumulative stride now 2
- L3 (k3,s1): $r = 5 + 2\cdot2 = 9$
- L4 (k3,s2): $r = 9 + 2\cdot2 = 13$, cumulative stride 4
- L5 (k3,s1): $r = 13 + 2\cdot4 = 21$
- L6 (k3,s2): $r = 21 + 2\cdot4 = 29$, cumulative stride 8
- L7: $r = 29 + 2\cdot8 = 45$; L8: $r = 45+2\cdot8=61$, cum. stride 16
- L9: $r=61+2\cdot16=93$; L10: $r=93+2\cdot16=125$, cum. stride 32
- L11: $r=125+2\cdot32=189$; L12: $r=189+2\cdot32=253$ → **exceeds 224 at 12 layers.**

**The insight to state explicitly:** 111 layers → 12 layers. Downsampling makes receptive field
grow *geometrically* instead of linearly. **This is the entire architectural reason the U-Net has
a downsampling encoder.** Not "efficiency" — *reach*. And it is precisely the reason a pure-conv
U-Net at 512×512 still can't easily make the top-left of an image agree with the bottom-right,
which is the reason SDXL bolts self-attention into the low-resolution middle blocks, and the
reason FLUX threw the U-Net away for a pure transformer. **This single worked example is the
bridge from CNNs to attention to the 2026 diffusion state of the art. Give it a whole page.**

### Translation equivariance vs invariance — the precision that matters

$$\text{Conv}(\text{Shift}_\delta(x)) = \text{Shift}_\delta(\text{Conv}(x)) \quad \textbf{equivariance}$$
$$\text{GlobalPool}(\text{Shift}_\delta(x)) = \text{GlobalPool}(x) \quad \textbf{invariance}$$

- Conv is **equivariant**: move the cat, the cat-features move with it.
- Global pooling is **invariant**: move the cat, the "is there a cat" score doesn't change.
- **Why this distinction is load-bearing for diffusion:** a U-Net must be *equivariant*, not
  invariant. It outputs an image-shaped thing; if you shift the noise, you want the predicted
  noise to shift. A classifier wants invariance. **Diffusion U-Nets therefore have no global
  pooling** — and the learner who wondered why generative CNNs look different from the
  classifiers in every tutorial now knows.

### Pooling / downsampling in 2026

Max-pooling is largely historical in generative models `[V-sec / M]`. Modern U-Nets downsample
with **strided convolution** (learned) and upsample with **transposed conv or
nearest-neighbour-upsample + conv** (the latter avoids checkerboard artifacts). Teach maxpool in
one paragraph as "the old way, and here is why it lost: it throws away *where*, and a generative
model needs *where*." Then move on. `[M]`

### Misconceptions to box

- ❌ *"A convolution kernel is 2-D."*
  ✅ **Correction:** it is **3-D per filter** — $(C_{\text{in}}, k_h, k_w)$ — and you have
  $C_{\text{out}}$ of them, so the weight tensor is 4-D. Each filter collapses *all* input
  channels to *one* output channel. **What fixes it:** make them compute
  $P = C_{\text{out}}(C_{\text{in}}k_hk_w+1)$ for a layer and check it against
  `sum(p.numel() for p in conv.parameters())` in PyTorch. Seeing 921,920 print out is the fix.
- ❌ *"Deep nets see the whole image at layer 1."*
  ✅ **Correction:** layer 1 sees 3×3 pixels. Full stop. Run the RF recurrence.
- ❌ *"Framework `Conv2d` computes a mathematical convolution."*
  ✅ **Correction:** it computes **cross-correlation** — no kernel flip. It doesn't matter,
  because the kernel is learned and a flipped kernel is just as learnable. But say it once, so a
  learner with a DSP background isn't quietly confused for the rest of the course.
- ❌ *"Convolution is translation-invariant."*
  ✅ **Correction:** equivariant. See above. And even equivariance is *broken* in practice by
  padding at the borders and by aliasing in strided downsampling `[M]` — worth one sentence of
  honesty; it's the reason diffusion models sometimes behave oddly at image edges.

### Demo (C): **The Receptive-Field Grower**

- **Plot:** left panel, a 128×128 input grid. Right panel, a schematic layer stack.
- **Controls:** an "add layer" button with per-layer dropdowns for $k \in \{1,3,5,7\}$ and
  $s \in \{1,2\}$; a slider selecting *which output neuron* (i,j) to trace.
- **What JS computes live:** the exact recurrence
  $r_\ell = r_{\ell-1} + (k_\ell-1)\prod_{j<\ell}s_j$ and cumulative stride
  $S_\ell = \prod_{j\le\ell}s_j$; then highlights the input square of side $r_L$ centered at
  $(S_L\cdot i, S_L\cdot j)$. Also renders a heat overlay of the *effective* receptive field by
  literally counting paths (a 2-D convolution of indicator kernels), which shows the Gaussian-ish
  falloff — the effective RF is smaller than the theoretical RF, and that's a real published
  result `[M]` worth showing rather than telling.
- **Numeric readout:** $r_L$, cumulative stride, total params, total FLOPs at a chosen input size.
- **The insight on drag:** flip one layer from s=1 to s=2 and watch $r_L$ jump from 13 to 29.
  *"Downsampling is not a compression trick. It's how the network sees far."*
- **Second insight:** stack ten k=1 convs and watch $r_L$ stay at 1. *"1×1 convs mix channels,
  never space."* — which is exactly what the learner needs to understand `proj_in`/`proj_out`
  around the attention blocks in a U-Net.

---

## 3. (D) EMBEDDINGS AND THE DOT PRODUCT

**This section is where the attention chapter is won or lost.** If the learner leaves D without
a physical feel for "dot product = alignment," F will not land. Do not rush it.

### Intuition first

> **An embedding is a learned address. You give every token its own point in a few-thousand-
> dimensional space, and you arrange the space so that "near" means "means something similar."
> Nothing about the token's identity matters after this — only its coordinates.**

For the dot product, the intuition sentence is:

> **The dot product asks "how much of this vector points along that one?" It is a similarity
> meter with a needle: big positive = same direction = similar; zero = orthogonal = unrelated;
> negative = opposite.**

### The math

An embedding table is a lookup, which is a matmul with a one-hot vector — say this, because it
explains why gradients flow to embeddings at all:

$$\mathbf{e}_t = E^\top \mathbf{o}_t, \qquad E \in \mathbb{R}^{V \times d},\; \mathbf{o}_t \in \{0,1\}^V \text{ one-hot}$$

- $V$ — vocabulary size (Qwen3: **151,936** `[V]`)
- $d$ — model width (Qwen3-8B: **4096** `[V]`)
- $E$ — the embedding matrix, $151{,}936 \times 4096 = \mathbf{622{,}329{,}856}$ params ≈ 622M
- $\mathbf{e}_t \in \mathbb{R}^{4096}$

**Print that number.** 622M — that's 7.6% of Qwen3-8B's 8.19B parameters sitting in a table
that does nothing but look things up. It recurs in §11.

Dot product and its geometry:

$$\mathbf{a}\cdot\mathbf{b} = \sum_{i=1}^{d} a_i b_i = \|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta$$

$$\cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|} \in [-1, 1]$$

**This is where high-school trig is genuinely load-bearing, and it is one of the few places to
drop down.** The learner knows $\cos\theta$. Connect it: the dot product *is* the cosine, scaled
by the two lengths. The whole of attention is trigonometry with 128 axes instead of 2.

Batched, the similarity of every query against every key is one matmul:

$$S = QK^\top, \qquad Q \in \mathbb{R}^{n \times d_k},\; K \in \mathbb{R}^{m \times d_k} \;\Rightarrow\; S \in \mathbb{R}^{n \times m}$$

$$S_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$$

**Say this out loud in the text:** "$QK^\top$ is not a mysterious operation. It is
$n \times m$ dot products, computed all at once, because that's what matrix multiplication *is*."
Many learners never notice that a matmul *is* a table of dot products. Once they see it, the
attention formula stops being scary.

### Worked example — do this arithmetic on the page

Toy 3-D "embeddings" (a course-owned running example; keep these vectors and reuse them in the
attention section so the numbers are familiar):

- $\mathbf{v}_{\text{king}} = (0.9,\; 0.8,\; 0.1)$
- $\mathbf{v}_{\text{queen}} = (0.85,\; -0.7,\; 0.15)$
- $\mathbf{v}_{\text{bicycle}} = (-0.4,\; 0.05,\; 0.9)$

Interpretation of axes (invented, for teaching): dim0 ≈ "royalty", dim1 ≈ "maleness",
dim2 ≈ "inanimate object".

- $\mathbf{v}_{\text{king}}\cdot\mathbf{v}_{\text{queen}} = 0.9(0.85) + 0.8(-0.7) + 0.1(0.15) = 0.765 - 0.560 + 0.015 = \mathbf{0.220}$
- $\|\mathbf{v}_{\text{king}}\| = \sqrt{0.81+0.64+0.01} = \sqrt{1.46} = 1.208$
- $\|\mathbf{v}_{\text{queen}}\| = \sqrt{0.7225+0.49+0.0225} = \sqrt{1.235} = 1.111$
- $\cos\theta = 0.220/(1.208 \times 1.111) = 0.220/1.342 = \mathbf{0.164}$ → $\theta = 80.6°$
- $\mathbf{v}_{\text{king}}\cdot\mathbf{v}_{\text{bicycle}} = -0.36 + 0.04 + 0.09 = \mathbf{-0.23}$
- $\cos\theta = -0.23/(1.208 \times 0.986) = \mathbf{-0.193}$ → $\theta = 101.1°$

**The lesson the arithmetic teaches:** king and queen are *nearly orthogonal* here (80.6°)
because they disagree hard on one axis. Similarity is not a single fact — it's a projection, and
*which* directions you care about changes the answer. This sets up, three sections later, the
single most important idea in attention: **Q and K are learned projections that decide which
directions "similar" is measured along.** Plant the seed here, harvest it in F1.

Then the norm/magnitude point:
- Scale $\mathbf{v}_{\text{king}}$ by 10: $\mathbf{v}\cdot\mathbf{v}_{\text{queen}} = 2.20$,
  10× bigger. $\cos\theta$: **unchanged at 0.164.**
- **Insight:** the dot product conflates *direction* (meaning) with *magnitude* (confidence /
  frequency / whatever the training happened to encode in length). Attention uses the raw dot
  product, *not* cosine — so magnitude matters, and that's part of why we have QK-Norm in 2026
  models (Qwen3 has `q_norm`/`k_norm` `[V]`, see §11).

### What embedding space actually looks like — be honest

The course should say what's true, not the popular story:

1. **Anisotropy `[M]`, and it's important.** Contextual embeddings from real LMs do **not**
   fill the sphere. They occupy a narrow cone; random pairs of tokens have *high* average cosine
   similarity (reported values around 0.5–0.9 in various studies, depending on model and layer).
   Consequence: **raw cosine similarity is a badly calibrated ruler.** This is not a footnote —
   it is *the* reason RAG systems use embedding models specifically fine-tuned with contrastive
   objectives rather than just pulling hidden states out of an LLM. Flag forward to the RAG
   brief.
2. **Linear analogies are real but oversold.** `king - man + woman ≈ queen` works in
   word2vec/GloVe-era static embeddings, and it works *partly* — the nearest neighbour is often
   `queen` only after you exclude the input words, which is a well-known and rarely-mentioned
   caveat `[M]`. **Say this.** The learner has seen the demo; tell them the demo has its thumb on
   the scale.
3. **Directions carry meaning; the linear representation hypothesis.** The genuinely important
   and *still live* claim is that high-level features are encoded roughly as linear directions in
   activation space. Evidence: steering vectors work, sparse autoencoders find interpretable
   directions, LoRA works at all. `[D]` — the field does not agree on how complete this is (see
   superposition, and the "features are directions but there are more features than dimensions"
   line of work). **This is a legitimate open question and the course should say so.** It matters
   because it is the *implicit justification* for why fine-tuning a low-rank update works.
4. **Dimensionality intuition is wrong and should be corrected.** In $d=4096$, essentially all
   random pairs are near-orthogonal ($\mathbb{E}[\cos\theta] = 0$, $\text{std} \approx 1/\sqrt{d}
   = 0.0156$). So you can pack **exponentially many** almost-orthogonal directions into 4096
   dims — the Johnson–Lindenstrauss regime. This is why 4096 numbers can hold far more than 4096
   concepts, and it is the quantitative core of "superposition." `[M]` for the specific framing;
   the JL math is `[V]`-solid.

### Misconceptions to box

- ❌ *"Each embedding dimension means something."*
  ✅ **Correction:** basis directions are arbitrary; meaning lives in *directions*, generally not
  axis-aligned, and features outnumber dimensions (superposition). **What fixes it:** the
  JL/near-orthogonality calculation above — once you show that $2^{100}$ near-orthogonal
  directions fit in 4096 dims, "one dim = one concept" is obviously wasteful and the learner
  drops it themselves.
- ❌ *"Embeddings are the model's dictionary — token X always has vector V."*
  ✅ **Correction:** true only of the *input* embedding table (layer 0). After one transformer
  block, the vector at position $t$ is *contextual* — "bank" in "river bank" and "bank account"
  have diverged. **The entire point of the stack is to move vectors.** The input table is the
  starting address, not the meaning. **What fixes it:** show layer-0 vs layer-N cosine
  similarity for a polysemous word.
- ❌ *"Higher cosine similarity = more related."*
  ✅ **Correction:** only within a calibrated embedding space. Anisotropy means 0.7 might be the
  *floor*. Always compare against the *distribution* of random-pair similarities in that specific
  model, not against an absolute threshold. **This misconception actively breaks people's RAG
  systems**, so it's worth real emphasis. Flag forward.

### Demo (D): **The Dot-Product Similarity Meter** — precursor to the Q/K/V widget

- **Plot:** a 2-D plane (deliberately 2-D — trig the learner can see) with two draggable vectors
  $\mathbf{a}$ and $\mathbf{b}$ from the origin. Draw the projection of $\mathbf{b}$ onto
  $\mathbf{a}$ as a dashed segment. A needle gauge on the side.
- **Controls:** drag either vector head. Toggle: "show raw dot product" / "show cosine."
- **JS math (exact):** `dot = a.x*b.x + a.y*b.y`; `cos = dot/(hypot(a)*hypot(b))`;
  projection point = `a * (dot / (a.x*a.x+a.y*a.y))`. Render $\theta = \text{atan2}$ difference,
  in degrees.
- **The insight on drag:** (1) rotate $\mathbf{b}$ through 90° and watch the dot hit exactly
  zero — *orthogonal = "these have nothing to say about each other."* (2) With "cosine" toggled,
  *lengthen* $\mathbf{a}$ and watch the raw dot balloon while cosine sits still. **The learner
  should conclude: the dot product mixes "how aligned" with "how big," and attention scores
  inherit that.**
- **Extension panel (this is the bit that earns its keep):** a second view with a $d$ slider
  (2 → 512). It samples 2000 pairs of random unit vectors in $\mathbb{R}^d$ and histograms their
  cosine similarity. **Drag $d$ from 2 to 512 and watch a broad flat distribution collapse into a
  spike at zero with width $1/\sqrt{d}$.** Insight: *"In high dimensions, everything is
  perpendicular to everything. That's not a bug — it's the storage capacity."* Cheap to
  implement (Box–Muller for Gaussians, normalize, dot), and it is genuinely the most
  counter-intuitive true fact in the unit.

---

## 4. (E) RNNs — A MOTIVATION CHAPTER, ~1.5 PAGES

**Architect's note: resist the urge to teach this properly.** The learner will never write an
LSTM. This chapter has exactly two jobs: (1) make the sequential-bottleneck pain *visceral*, so
attention arrives as relief; (2) install the vocabulary (hidden state, timestep, BPTT) that the
diffusion track will *also* need when we discuss the sampling loop. Give LSTM gates a diagram and
a paragraph, not a derivation.

### Intuition first

> **An RNN reads like you read: one word at a time, carrying a running summary in your head. The
> summary is the only thing that survives from word to word — which is both the elegance and the
> death sentence.**

### The math (state it, don't dwell)

$$\mathbf{h}_t = \tanh\!\big(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b}\big), \qquad \mathbf{y}_t = W_{hy}\mathbf{h}_t$$

- $\mathbf{x}_t \in \mathbb{R}^{d_{\text{in}}}$ — input at step $t$
- $\mathbf{h}_t \in \mathbb{R}^{d_h}$ — hidden state, *the entire memory of the past*
- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$, $W_{xh} \in \mathbb{R}^{d_h \times d_{\text{in}}}$ — **shared across all $t$** (this is the RNN's inductive bias: the update rule is time-invariant)
- $t \in \{1..n\}$ — timestep

### The two failures — and be precise that they are two, not one

**Failure 1 — vanishing/exploding gradients (a *learning* problem).** BPTT:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_n} \prod_{t=2}^{n} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}, \qquad \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \text{diag}\big(1-\mathbf{h}_t^2\big)\,W_{hh}^\top$$

The gradient is a product of $n{-}1$ Jacobians. Let $\gamma = \|\partial\mathbf{h}_t/\partial\mathbf{h}_{t-1}\|$.

**Worked arithmetic (put this on the page):**
- $\gamma = 0.9$, $n = 100$: $0.9^{99} = 2.95\times10^{-5}$. The gradient reaching step 1 is
  **30,000× weaker** than at step 100.
- $\gamma = 0.9$, $n = 1000$: $0.9^{999} \approx 1.7\times10^{-46}$. **Below bf16's smallest
  normal (~$1.2\times10^{-38}$). Literally zero.**
- $\gamma = 1.1$, $n = 100$: $1.1^{99} = 1.25\times10^{4}$. Explodes. (Fixable by clipping.
  Vanishing is not.)
- Note $\tanh'(h) = 1 - h^2 \le 1$, so the diag factor is $\le 1$ *always*, biasing the whole
  product toward decay. **Vanishing is the default, not the accident.**

LSTM/GRU fix *this one* — the cell state $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ has a nearly-additive path with Jacobian $\approx \text{diag}(\mathbf{f}_t)$, and $\mathbf{f}_t \to 1$ gives $\gamma \to 1$: **a gradient highway.** One paragraph. Flag it explicitly as *the same idea as a residual connection* (§8) — same problem, same fix, different decade. **This cross-reference is worth a lot: it turns two facts into one idea.**

**Failure 2 — the sequential dependency (a *hardware* problem). THIS is what actually killed RNNs, and the course must be unambiguous about it.**

$\mathbf{h}_t$ requires $\mathbf{h}_{t-1}$. For a sequence of $n$ tokens, training requires **$n$
sequential steps that cannot be overlapped.** Depth in time is not parallelizable.

**Worked comparison — make the number brutal:**
- Sequence $n = 2048$. RNN: **2048 dependent kernel launches** per layer per forward pass. Each
  launch is a small matmul — say $d_h = 4096$, so $\mathbf{h}_{t-1}W_{hh}$ is a
  $(1 \times 4096)\times(4096\times4096)$ matrix-*vector* product: ~33.5 MFLOP, but it must read
  $4096^2 \times 2 = 33.5$ MB of weights. **Arithmetic intensity ≈ 0.5 FLOP/byte.** A GB10-class
  GPU wants ~300+ FLOP/byte to saturate its tensor cores. You are running at well under 1% of
  peak, 2048 times in a row.
- Transformer: **one** $(2048 \times 4096) \times (4096 \times 4096)$ matmul. Same weight bytes
  read, 2048× the FLOPs. Arithmetic intensity ≈ 1000 FLOP/byte. **Tensor cores saturate.**

> **The sentence the course should print in bold:** *RNNs didn't lose because they couldn't
> represent language. They lost because a matrix-vector product wastes a GPU and a
> matrix-matrix product doesn't. The transformer's real invention was trading a sequential
> $O(n)$ dependency for a parallel $O(n^2)$ one — and paying $n^2$ in FLOPs you can actually
> spend is better than paying $n$ in FLOPs you can't.*

This framing also **pre-loads §7 (the $O(n^2)$ problem)**: the learner will understand that
$n^2$ was never an accident to be apologised for — it was the *deal*. And it pre-loads the
autoregressive decoding discussion in the LLM track, where the transformer *becomes* sequential
again at inference time and suddenly has RNN's exact problem (this is why decoding is
memory-bandwidth-bound; see §11 for the DGX Spark arithmetic). **Flag that irony — it's a
genuinely clarifying observation and it recurs.**

### Honest caveat (2026) — do not write RNNs off, the field didn't

`[V]` As of Jan–Feb 2026, recurrent/linear-attention mechanisms are **back in frontier
open-weight models**, hybridized rather than pure: **Qwen3-Coder-Next (80B-A3B) uses Gated
DeltaNet + gated attention in a 3:1 ratio**; **Qwen3.5 (397B-A17B)** uses the same hybrid family;
**Ant Group's Ling 2.5 1T** hybridizes Lightning Attention with MLA and reports ~3.5× throughput
over Kimi K2 at 32k context. `[V-sec, via Raschka's Jan–Feb 2026 architecture survey]`

**Why this matters pedagogically and why the course must include it:** the modern recurrent
layers (Mamba/SSM, DeltaNet, linear attention) fixed *Failure 2*, not just Failure 1 — they are
formulated so training can be parallelized (associative scan / chunked matmul form) while
inference stays $O(1)$ memory per token. **The lesson is not "RNNs were wrong." It's "the
sequential *training* dependency was the killer, and when someone removed it, recurrence came
back."** That's a much better lesson than a linear history, and it makes the parallelism point
unforgettable because it's the punchline of both halves.

`[D]` **Genuine uncertainty to flag:** whether hybrid linear/recurrent attention will displace
full attention at the frontier is **not settled in 2026.** Evidence for: the models above ship
and are competitive; long-context economics strongly favour them. Evidence against: MiniMax M2.5
(230B) is `[V-sec]` described as deliberately "plain GQA, no efficiency tweaks" and remains
competitive — a strong signal from a serious lab that the exotic stuff isn't yet a free win. The
honest 2026 statement: **full attention is still the default; hybrids are the fastest-moving
frontier; nobody has declared victory.** Do not let the course imply otherwise in either
direction.

### Misconceptions to box

- ❌ *"LSTMs solved the long-range problem."*
  ✅ **Correction:** they solved the *gradient* problem, substantially. They did not solve the
  *information bottleneck* — everything must still squeeze through a fixed-size $\mathbf{h}_t$ —
  and they did not solve parallelism at all. **Two of three problems remained.**
- ❌ *"Transformers replaced RNNs because attention is smarter."*
  ✅ **Correction:** attention is *dumber* (weaker inductive bias, §1) and *more expensive*
  ($O(n^2)$ vs $O(n)$). It won on **parallelism**, which converted into scale, which converted
  into data. **What fixes this:** the arithmetic-intensity comparison above. Once a learner sees
  0.5 vs 1000 FLOP/byte, "smarter" stops being the explanation they reach for. This misconception
  is *worth killing hard* because it's the seed of the "architecture is where intelligence lives"
  error from §1 and the "emergence is magic" error in §12.
- ❌ *"RNNs are obsolete."*
  ✅ **Correction:** see the 2026 hybrid models above. Recurrence returned once training-time
  parallelism was solved.

---

## 5. (F) ATTENTION — THE LOAD-BEARING CHAPTER

**Budget: ~6 pages. This is the center of the course.** Everything before it is setup;
everything after it is consequence. If the architect cuts elsewhere to fund this, that's correct.

### F1 — Q/K/V: derive the intuition BEFORE the formula

**Do not open with $\text{softmax}(QK^\top/\sqrt{d_k})V$.** Open with the lookup table. Build the
formula *as the answer to a question the learner is already asking.*

**Step 1 — the hard lookup the learner already knows.** A Python dict:

```python
d = {"cat": [0.2, 0.9], "dog": [0.7, 0.1]}
d["cat"]          # -> [0.2, 0.9]
d["kitten"]       # -> KeyError.
```

Three objects: a **query** (`"kitten"`), a set of **keys** (`"cat"`, `"dog"`), a set of
**values** (the lists). The lookup is: compare query to every key, find the exact match, return
that value.

**Step 2 — name the two problems.** (a) It's *brittle*: "kitten" should have gotten something
cat-ish. (b) It's *not differentiable*: the match is a hard equality test; there is no gradient,
so you can never learn what to look up.

**Step 3 — soften it, one step at a time.** Replace exact equality with a **graded similarity
score** (dot product — §3 already made this the natural move). Replace "return the one match"
with a **weighted average of all the values**, weighted by how well each key matched.

$$\text{output} = \sum_{j} \underbrace{\frac{\exp(\mathbf{q}\cdot\mathbf{k}_j)}{\sum_{j'}\exp(\mathbf{q}\cdot\mathbf{k}_{j'})}}_{\text{how much key } j \text{ matched, normalized}} \cdot\; \mathbf{v}_j$$

> **The intuition sentence for the whole chapter:**
> **Attention is a dictionary lookup where the keys don't have to match exactly, and instead of
> returning one value you get a blend of all of them, weighted by how well each key matched.
> Softening the lookup is what makes it differentiable — and differentiable means the network can
> learn *what to look for*.**

**Step 4 — and here's the actual point.** The tokens don't come with queries, keys, and values.
**Every token computes its own, with learned projections:**

$$\mathbf{q}_i = W_Q^\top \mathbf{x}_i, \qquad \mathbf{k}_j = W_K^\top \mathbf{x}_j, \qquad \mathbf{v}_j = W_V^\top \mathbf{x}_j$$

- $\mathbf{x}_i \in \mathbb{R}^{d}$ — the token's current representation
- $W_Q, W_K \in \mathbb{R}^{d \times d_k}$, $W_V \in \mathbb{R}^{d \times d_v}$ — **learned**

**Say this explicitly, it's the punchline of F1:**
- $\mathbf{q}_i$ = "**what am I looking for?**" — asked by token $i$, about the rest of the
  sequence.
- $\mathbf{k}_j$ = "**what do I advertise about myself?**" — token $j$'s billboard.
- $\mathbf{v}_j$ = "**what do I actually hand over if you pick me?**"

**Why three projections and not one** — this is the question nobody answers and every learner
silently has:
1. **$Q$ vs $K$ must be separate because relevance is asymmetric.** In "the animal didn't cross
   the street because *it* was too tired," `it` should attend to `animal`. But `animal` should not
   necessarily attend to `it` with the same weight. If $W_Q = W_K$ then $S_{ij} = S_{ji}$ — the
   score matrix is **forced symmetric**, and you have thrown away the ability to represent
   directed relations. **Callback to §3:** we showed that king·queen depends on *which directions
   you measure along*. $W_Q$ and $W_K$ are exactly the learned choice of those directions — and
   choosing *different* directions for the asker and the advertiser is what buys asymmetry.
2. **$V$ must be separate from $K$ because "how I'm found" ≠ "what I contribute."** The token
   `Paris` might advertise (via $\mathbf{k}$) "I am a capital city, ask me about France" while
   carrying (via $\mathbf{v}$) the actual geographic/cultural content. Merging them forces the
   index to be the payload. **Analogy that lands for this learner:** a database index vs the row
   it points at. You don't store the row in the B-tree.

**Misconception to box right here** (it is the #1 attention misconception):
- ❌ *"Q, K, V are three different things the model has, like three memories."*
  ✅ **Correction:** they are **three different projections of the same vector.** For self-
  attention, every token produces all three from its own $\mathbf{x}_i$. The same token is
  simultaneously asking, advertising, and offering. **What fixes it:** show the code —
  `q, k, v = x @ Wq, x @ Wk, x @ Wv` — *same `x`, three times.* One line kills it. This is also
  the precise hinge for understanding cross-attention (F4), where $Q$ comes from one sequence and
  $K,V$ from another — a learner who thinks Q/K/V are "three memories" cannot understand
  cross-attention *at all*, which means they cannot understand diffusion conditioning. **High
  stakes; box it prominently.**

### F2 — Scaled dot-product attention, every shape written out

$$\boxed{\;\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V\;}$$

**The full shape table. The course should print this and never let the learner guess a shape
again.** (Single head, single batch element, then batched.)

| Object | Shape | What it is | Notes |
|---|---|---|---|
| $X$ | $(n, d)$ | input token reps | $n$ = seq len, $d$ = model dim |
| $W_Q$ | $(d, d_k)$ | query projection | learned |
| $W_K$ | $(d, d_k)$ | key projection | learned; **must share $d_k$ with $W_Q$** |
| $W_V$ | $(d, d_v)$ | value projection | learned; $d_v$ *need not* equal $d_k$ (it usually does) |
| $Q = XW_Q$ | $(n, d_k)$ | queries | one row per token |
| $K = XW_K$ | $(m, d_k)$ | keys | $m$ = *source* length; $m = n$ for self-attn |
| $V = XW_V$ | $(m, d_v)$ | values | **$K$ and $V$ must share $m$, not $d$** |
| $S = QK^\top$ | $(n, m)$ | raw scores | $S_{ij} = \mathbf{q}_i\cdot\mathbf{k}_j$ |
| $S/\sqrt{d_k}$ | $(n, m)$ | scaled scores | |
| $M$ | $(n, m)$ | mask | additive; $0$ or $-\infty$ |
| $A = \text{softmax}_{\text{row}}(\cdot)$ | $(n, m)$ | attention weights | **each row sums to 1** |
| $AV$ | $(n, d_v)$ | output | one row per *query* token |

**Batched with heads:** $(B, H, n, d_k) \times (B, H, d_k, m) \to (B, H, n, m)$, then
$\times (B,H,m,d_v) \to (B,H,n,d_v)$.

**Three shape facts to hammer** (these are where learners actually get lost, and they're cheap to
fix):
1. **The output has $n$ rows, one per query.** Attention is a function *of the query set*. The
   key/value set can be any length. **This single fact makes cross-attention obvious later.**
2. **Softmax is over the *last* axis ($m$, the keys).** Row $i$ of $A$ is a probability
   distribution over "where token $i$ looked." **Columns do not sum to 1.** Learners
   constantly get this backwards when reading heatmaps. Box it.
3. **$d_k$ links Q and K. $m$ links K and V. $d$ never appears after the projections.** The whole
   operation is dimensionally rigid and there is exactly one legal way to wire it.

**Why softmax specifically** (worth two sentences, because "it's just a softmax" is a dodge):
- Enforces $\sum_j A_{ij} = 1$, $A_{ij} > 0$ → the output is a **convex combination** of value
  vectors, so it lives in the convex hull of $V$ and cannot blow up. Attention is stable by
  construction.
- It is smooth, so gradients flow to *every* key, including the ones that lost. A hard argmax
  gives gradient only to the winner and nothing learns. **This is the same "soften it to make it
  differentiable" move as F1 step 3 — same idea, now made concrete.**
- Its temperature is implicit in the scale of the logits — **which is exactly what $\sqrt{d_k}$
  is controlling.** Segue.

### F2b — The $\sqrt{d_k}$: the real reason, derived

**This is the most-hand-waved line in every transformer explainer. Do it properly — it takes
half a page and it is genuinely satisfying.**

> **Intuition first: as you add more dimensions to the dot product, the scores don't just get
> noisier — they get *bigger*, and a softmax fed big numbers stops being soft. It becomes a hard
> argmax, and a hard argmax has no gradient. The $\sqrt{d_k}$ is a thermostat that keeps the
> softmax at a workable temperature no matter how wide you make the head.**

**The derivation.** Model $\mathbf{q}, \mathbf{k}$ as having iid components with mean 0,
variance 1 (which is roughly what standard init + normalization gives you at the start of
training):

$$s = \mathbf{q}\cdot\mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$

$$\mathbb{E}[s] = \sum_i \mathbb{E}[q_i]\mathbb{E}[k_i] = 0 \quad (\text{independence})$$

$$\text{Var}(s) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k}\mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = \sum_{i=1}^{d_k} 1\cdot 1 = \boxed{d_k}$$

$$\Rightarrow \quad \text{std}(s) = \sqrt{d_k}$$

**Therefore:** dividing by $\sqrt{d_k}$ restores unit variance. That's the whole move. It is not
a fudge factor; it is the exact normalizing constant.

**With Qwen3-8B's real `head_dim = 128` `[V]`:** $\text{std}(s) = \sqrt{128} = \mathbf{11.31}$.
Logits routinely land in $[-34, +34]$ (±3σ) **before** scaling.

**Now show what that does to the softmax — carry the arithmetic:**

Take three keys with *unscaled* logits $[2, 1, 0]$ (a mild preference):
- $e^2 = 7.389,\; e^1 = 2.718,\; e^0 = 1.000$; sum $= 11.107$
- $A = [\mathbf{0.665},\; \mathbf{0.245},\; \mathbf{0.090}]$ — a real blend. Gradients flow to all
  three.

Now the *same relative preference* at $\sqrt{d_k}$ scale — multiply by 11.31 → $[22.6, 11.3, 0]$:
- Factor out $e^{22.6}$: $A \propto [1,\; e^{-11.3},\; e^{-22.6}] = [1,\; 1.24\times10^{-5},\; 1.53\times10^{-10}]$
- $A = [\mathbf{0.9999876},\; \mathbf{1.24\times10^{-5}},\; \mathbf{1.5\times10^{-10}}]$
- **This is a one-hot vector in a trenchcoat.**

**And now the part that actually matters — the gradient.** The softmax Jacobian diagonal is
$\partial A_i/\partial s_i = A_i(1 - A_i)$:
- Unscaled, key 1: $0.665 \times 0.335 = \mathbf{0.223}$. Healthy.
- Scaled-up, key 1: $0.9999876 \times 1.24\times10^{-5} = \mathbf{1.24\times10^{-5}}$.
- Scaled-up, key 2: $1.24\times10^{-5} \times (1 - 1.24\times10^{-5}) \approx \mathbf{1.24\times10^{-5}}$.
- **~18,000× smaller gradient.** In bf16 (which has ~8 bits of mantissa, relative resolution
  ~$2^{-8} = 0.004$), a gradient of $10^{-5}$ relative to activations of order 1 is **rounded to
  zero.**

> **The sentence to print:** *Without the $\sqrt{d_k}$, attention doesn't get worse — it gets
> **stuck**. The softmax saturates, the Jacobian collapses, and $W_Q$ and $W_K$ stop receiving
> gradient at all. The model can't learn what to look for, because looking has become a hard
> decision and hard decisions have no derivative.*

**Two honest caveats the course should include:**
- `[D]` The mean-0/var-1 assumption holds at initialization. **After training, learned $Q$/$K$
  have whatever statistics training gave them, and the $\sqrt{d_k}$ is no longer exactly the
  right constant.** This is precisely why 2026 models add **QK-Norm** — an explicit RMSNorm on
  $\mathbf{q}$ and $\mathbf{k}$ *before* the dot product — to re-impose the assumption at every
  step instead of hoping. **Qwen3-8B has `q_norm` and `k_norm`, each of size `head_dim=128`
  `[V]`** — the learner will literally see these in the state dict, and they should know why.
  This is a great "the textbook formula is 2017; here's what 2026 actually ships" moment. Trinity
  Large, Cohere Tiny Aya (which notably *dropped* QK-Norm), and others vary here `[V-sec]` —
  **so it's a real design choice, not settled.**
- The scaling is a *temperature*. $\text{softmax}(s/\tau)$ with $\tau = \sqrt{d_k}$. Learners who
  have set `temperature` in ComfyUI or an LLM sampler already own this concept — **connect it
  explicitly.** Same knob, different place in the pipeline.

**Misconceptions to box:**
- ❌ *"$\sqrt{d_k}$ stops the numbers from overflowing."*
  ✅ **Correction:** overflow is not the issue — softmax is implemented with the max-subtraction
  trick and never overflows. **The issue is gradient death via softmax saturation.** Numerically
  the forward pass is *fine* without the scale; the *backward* pass is dead. **What fixes it:**
  compute the Jacobian numbers above. This is the correction that actually sticks, because it
  reframes the constant from "numerical hygiene" to "learnability," which is the whole theme of
  §1.
- ❌ *"Why $\sqrt{d_k}$ and not $d_k$?"*
  ✅ **Correction:** because we're normalizing a **standard deviation**, not a variance.
  $\text{Var}(s) = d_k \Rightarrow \text{std}(s) = \sqrt{d_k}$. Dividing by $d_k$ would
  over-shrink by a factor of $\sqrt{d_k}$ (11.3× at $d_k{=}128$), giving logits with std 0.088 —
  a softmax so flat it's a uniform average, which is the *opposite* failure and equally useless.
  **Both failure modes are worth showing on the demo slider (see below) — the learner should see
  the two cliffs and the valley between them.**
- ❌ *"$d_k$ is the model dimension."*
  ✅ **Correction:** $d_k$ is the **per-head** dimension. Qwen3-8B: $d = 4096$ but
  $d_k = 128$ `[V]`. The scale uses **128**, so $\sqrt{d_k} = 11.31$, not $\sqrt{4096} = 64$.
  Getting this wrong is a common from-scratch-implementation bug.

### F3 — Multi-head attention

> **Intuition: one attention head can only measure similarity in one way — it has one $W_Q$, one
> $W_K$, so one notion of "relevant." Multi-head is running several relevance-detectors in
> parallel on different learned subspaces, so one head can track grammar while another tracks
> subject-verb agreement and another just copies the previous token.**

**Callback to §3, and it pays off here:** we showed that king·queen = 0.164 or something else
entirely depending on which directions you project onto. **A head is a choice of projection. Many
heads = many simultaneous, different opinions about what "similar" means.** This is the cleanest
way to motivate multi-head and it costs one sentence because §3 did the work.

$$\text{head}_h = \text{Attention}(XW_Q^{(h)},\; XW_K^{(h)},\; XW_V^{(h)}), \qquad h = 1..H$$
$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\,W_O$$

Shapes, with **Qwen3-8B's real numbers `[V]`**:

| Object | Shape | Qwen3-8B |
|---|---|---|
| $d$ | model dim | 4096 |
| $H$ | number of query heads | 32 |
| $d_k = d_v$ | head dim | **128** (note: $32 \times 128 = 4096 = d$, but this is a *convention*, not a law) |
| $W_Q^{(h)}$ | $(d, d_k)$ | $(4096, 128)$ |
| $W_Q$ (all heads, fused) | $(d, H d_k)$ | $(4096, 4096)$ |
| $\text{head}_h$ | $(n, d_v)$ | $(n, 128)$ |
| Concat | $(n, H d_v)$ | $(n, 4096)$ |
| $W_O$ | $(H d_v, d)$ | $(4096, 4096)$ |

**Implementation truth to state plainly** (it demystifies every codebase the learner will read):
you do **not** have $H$ separate matrices. You have one $(4096, 4096)$ `q_proj`, and you
`.view(B, n, 32, 128).transpose(1, 2)` to get $(B, 32, n, 128)$. **The heads are a reshape.** The
math is $H$ independent attentions; the code is one matmul and a view. Learners reading
`modeling_qwen3.py` for the first time hit this and stall — pre-empt it.

**Why $W_O$ exists** (usually unexplained, and the explanation is good): concatenation just
stacks $H$ subspace-outputs side by side; they've never interacted. $W_O$ is the layer that
**mixes the heads' findings back into one shared representation.** Without it, head $h$'s output
would be permanently stuck in dims $[128h, 128(h{+}1))$ of the residual stream. It is also,
usefully, the reason you can view MHA as $\sum_h \text{head}_h W_O^{(h)}$ — a *sum of per-head
contributions written into the residual stream* — which is the mental model the
interpretability literature uses and which makes §8's "residual stream as a shared bus" click.

**What heads actually learn — be honest, this is `[D]` territory:**
- Some heads are cleanly interpretable and reproducibly so: **previous-token heads**,
  **positional/offset heads**, **duplicate-token heads**, and **induction heads** (which
  implement "I saw `[A][B]` earlier, I'm now at `[A]`, so predict `[B]`" — the mechanistic
  substrate of in-context learning). `[M]` for the specific taxonomy; the induction-head result is
  well-replicated.
- **But:** most heads are *not* cleanly interpretable. Heads are **polysemantic** — one head does
  several unrelated things depending on context. And a large fraction of heads in a trained model
  can be **pruned with little loss** `[M]`, which is a strong hint that the clean "each head has a
  job" story is a story.
- **Attention sinks are real and worth mentioning:** many heads dump a large fraction of their
  attention mass onto the first token (or a BOS token) regardless of content — a learned
  "no-op," because softmax rows must sum to 1 and there's no way to say "nothing here is
  relevant." `[M]` This is a great fact: it explains a visual artifact the learner *will* see the
  moment they plot a real attention map, and it teaches that **softmax's normalization constraint
  has behavioral consequences.** Without this warning, the demo below will confuse them.

**Misconceptions to box:**
- ❌ *"More heads = more capacity."*
  ✅ **Correction:** with the standard $H \cdot d_k = d$ convention, total attention parameters
  are **identical** regardless of $H$. 32 heads × 128 dims and 8 heads × 512 dims have exactly
  the same parameter count. You are trading **number of distinct relevance-notions** against
  **expressiveness per notion**. It's a partition, not an addition. **What fixes it:** have them
  compute $P_{\text{attn}}$ for both and see the same number.
- ❌ *"Each head learns one linguistic function."*
  ✅ **Correction:** see above — polysemanticity and prunability. Some do; most don't. **The
  clean pictures in papers are cherry-picked, and saying so builds the learner's calibration.**
- ❌ *"Heads talk to each other inside the attention layer."*
  ✅ **Correction:** heads are **completely independent** until $W_O$. No head sees another head's
  scores. All cross-head interaction happens in $W_O$ and in later layers via the residual stream.

### F4 — Self vs Cross attention — **THE DIFFUSION HINGE. FLAG THIS HARD.**

**Architect: this subsection is the single most important paragraph in the entire trunk for the
diffusion track. The learner already uses ComfyUI. He has typed prompts and watched images
change and has never been told the mechanism. This is where you tell him. Make it a landmark —
its own page, its own visual treatment.**

The *only* difference:

| | Self-attention | Cross-attention |
|---|---|---|
| $Q$ from | sequence $X$ | sequence $X$ (the "receiver") |
| $K, V$ from | sequence $X$ | **sequence $Y$ (the "source")** |
| Shapes | $Q:(n,d_k)$, $K,V:(n,d_k)$ | $Q:(n,d_k)$, $K,V:(\mathbf{m},d_k)$, $m \ne n$ allowed |
| Score matrix | $(n, n)$ | $(n, \mathbf{m})$ |
| Output | $(n, d_v)$ | $(n, d_v)$ — **still one row per query** |
| Question it answers | "which other tokens *of mine* matter to me?" | "which parts of **that other thing** matter to me?" |

$$\text{CrossAttn}(X, Y) = \text{softmax}\!\left(\frac{(XW_Q)(YW_K)^\top}{\sqrt{d_k}}\right)(YW_V)$$

**That's it. That is the entire mechanism of text-to-image conditioning.** Write it exactly like
this:

> **In a text-to-image diffusion model, the image is the query and the text is the key/value.**
> Every image patch asks, "which words are about me?" The word embeddings answer. The patch
> pulls in a weighted blend of word-meanings and updates itself accordingly. **When you type
> "a red car" and the car comes out red, this matmul is why.**

**Concrete shapes for a real system — SDXL-class U-Net `[M`, architect should verify against
`diffusers` before shipping`]`:**
- $Y$ = CLIP text embeddings, shape $(77, 2048)$ — 77 tokens (the CLIP context length that every
  ComfyUI user has bumped into), 2048 dims (SDXL concatenates CLIP-L 768 + CLIP-G 1280).
- $X$ = flattened U-Net spatial features at some resolution, e.g. $(32\times32, 640) = (1024, 640)$.
- Cross-attention score matrix: $(1024, 77)$. **1024 image positions × 77 text tokens.**
- **Every entry $A_{ij}$ is literally "how much does image-patch $i$ care about word $j$."**
- This matrix *is* what a cross-attention-map visualization tool plots. This is what
  attention-coupling / prompt-editing / regional-prompting nodes manipulate. **The learner has
  used these nodes.** Now he knows they are writing into $A$.

**Flag forward, explicitly, in the text:**
- **→ Diffusion track, conditioning:** this is the whole of classifier-free-guidance's *target*.
  CFG runs the model twice — once with $Y$ = prompt, once with $Y$ = empty — and extrapolates.
  **CFG only makes sense once you know $Y$ enters through $K,V$.**
- **→ Diffusion track, IP-Adapter / ControlNet:** IP-Adapter adds a *second* cross-attention over
  image embeddings. Once the learner knows cross-attention takes an arbitrary $(m, \cdot)$
  source, "you can condition on anything you can embed" is *obvious* rather than magic.
- **→ 2026 update — and this one is important, because it's where the field moved `[V]`:**
  FLUX.1 (12B) and FLUX.2 (32B) **do not use a separate cross-attention layer at all.** They use
  **MMDiT / joint attention**: text and image tokens are *concatenated into one sequence* and run
  through **self**-attention, so image↔text, image↔image, and text↔text similarity are all
  computed in the same score matrix. `[V]` FLUX.1: **19 double-stream + 38 single-stream blocks,
  ~12B params, ~54% of params in the double-stream blocks**; FLUX.2: **32B params, 8
  double-stream + 48 single-stream, ~24% of params in double-stream and ~73% in single-stream,
  single text encoder (Mistral Small 3.1, max 512 tokens), new `AutoencoderKLFlux2` VAE, SwiGLU
  MLP, no bias params, shared time/guidance modulation, >80 GB VRAM unoptimized.**
  **The pedagogical payoff:** cross-attention is a *special case* of self-attention over a
  concatenated sequence with a mask that blocks some quadrants. Teaching cross-attention first
  and then revealing MMDiT as "just concatenate and let self-attention sort it out" is a genuinely
  elegant arc and it lands the learner exactly at the 2026 state of the art. **The 77-token CLIP
  bottleneck the learner has cursed at in ComfyUI is a direct consequence of the old design; FLUX
  removing it is a direct consequence of the new one.**
  **Note the tension worth flagging:** FLUX.2 needing >80 GB unoptimized is *directly relevant*
  to a learner with a 128 GB DGX Spark — it fits in unified memory where it wouldn't fit on a
  consumer 24 GB card. Good, concrete, motivating. `[V]`

**Misconceptions to box:**
- ❌ *"Cross-attention is a different mechanism from self-attention."*
  ✅ **Correction:** same function, different arguments. `attn(Q=x, K=y, V=y)` vs
  `attn(Q=x, K=x, V=x)`. **The code is identical.** Show both call sites side by side. **What
  fixes it:** literally one PyTorch function called twice.
- ❌ *"The prompt is 'injected' or 'mixed in' to the image."*
  ✅ **Correction:** the image queries the prompt. It is a *pull*, not a *push*, and the pull is
  content-addressed — the patch decides what's relevant to it. This reframing explains why prompts
  are unreliable: **you don't control the query.**
- ❌ *"$K$ and $V$ must have the same dimension as $Q$."*
  ✅ **Correction:** $K$ must match $Q$ in $d_k$ (they get dot-producted). $V$ must match $K$ in
  $m$ (sequence length). **$Y$'s native width (2048 for SDXL text) is free** — that's what $W_K$
  and $W_V$ are for; they project from $Y$'s width down to $d_k$. **This is exactly why you can
  bolt a T5 or a Mistral or a CLIP onto the same U-Net: the projection absorbs the mismatch.**

### F5 — Causal masking — **THE LLM HINGE**

> **Intuition: if you're training a model to predict the next word, and you let it look at the
> next word, it will learn to look at the next word. The mask is a blindfold that makes the
> training task honest — and it lets you get $n$ training examples out of one sequence instead of
> one.**

$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases} \qquad \Rightarrow \qquad \text{softmax}(s_{ij} + M_{ij})\big|_{j>i} = \frac{e^{-\infty}}{\cdot} = 0$$

- $M \in \mathbb{R}^{n\times n}$, added to the scaled scores **before** softmax.
- In practice: `-inf` is a large negative (e.g. `torch.finfo(dtype).min`); in bf16 that's
  about $-3.39\times10^{38}$.
- **Why additive and not multiplicative:** additive-then-softmax renormalizes the *surviving*
  entries so each row still sums to 1. Multiplying $A$ by a 0/1 mask *after* softmax leaves rows
  summing to less than 1 — the output shrinks toward zero for early tokens. **This is a real bug
  people write.** Box it.

**The efficiency insight that motivates the whole architecture — this is the good bit:**

With a causal mask, position $i$'s output depends only on positions $\le i$. So **one forward
pass over a length-$n$ sequence produces $n$ simultaneous next-token predictions**, each with a
valid loss term:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n} \log p_\theta(x_{i+1} \mid x_{\le i})$$

- $n = 4096$ → **4096 training signals from one forward pass.**
- **This is the deal that makes LLM pretraining economically possible.** An encoder-decoder or a
  masked-LM gets far fewer signals per token processed (BERT-style MLM masks ~15% of tokens →
  ~0.15 signals per token vs **1.0** for causal LM — **a 6.7× data-efficiency gap on the same
  compute**). `[M]` for the 15% figure.
- **This is a large part of the honest answer to "why did decoder-only win"** (§10). Not
  elegance. Signal density.

**Misconceptions to box:**
- ❌ *"Causal masking makes the model slower / it's a limitation we tolerate."*
  ✅ **Correction:** it is *the enabling trick* for parallel training. Without it you'd need one
  forward pass per prediction — you'd be back to RNN economics. The mask *buys* you the
  parallelism. **What fixes it:** the 4096-signals-per-pass number.
- ❌ *"The mask is applied to the output."*
  ✅ **Correction:** before the softmax, to the logits. See the renormalization argument.
- ❌ *"Causal = the model reads left to right."*
  ✅ **Correction:** it processes **all positions simultaneously** during training. Every
  position is computed in the same matmul. The mask constrains *what each position may see*, not
  *when it is computed*. **This is the exact distinction that makes transformers ≠ RNNs, so it is
  worth its own box.** (At *inference* it does become sequential — the irony flagged in §4.)
- ❌ *"Diffusion U-Nets use causal masks too."*
  ✅ **Correction:** no — image self-attention is **bidirectional/unmasked**. Every patch sees
  every patch. Causality is a property of *time-ordered generative factorization*, and images
  aren't time-ordered. **Good contrast to draw explicitly at the fork.**

### F6 — The $O(n^2)$ problem and 2026 mitigations

> **Intuition: attention's superpower is that every token can look at every token. Its curse is
> the same sentence — "every × every" is $n^2$, and $n^2$ grows faster than your patience.**

**The costs, stated separately — because they have different fixes and conflating them is the
main confusion here:**

| Cost | Scaling | Real number (Qwen3-8B, $n = 40{,}960$ = its `max_position_embeddings` `[V]`) |
|---|---|---|
| Score-matrix **memory**, naive | $O(H n^2)$ | $32 \times 40960^2 \times 4\text{B (fp32)} = \mathbf{214.7\ GB}$ — **per layer** |
| Attention **FLOPs** | $O(n^2 d)$ | $\approx 4 n^2 d = 4(40960)^2(4096) = \mathbf{2.75\times10^{13}}$ = 27.5 TFLOP per layer |
| FFN FLOPs | $O(n d^2)$ | $\approx 6 n d\, d_{\text{ff}} = 6(40960)(4096)(12288) = \mathbf{1.24\times10^{13}}$ = 12.4 TFLOP per layer |
| **KV cache** (decode) | $O(n\, L\, n_{kv} d_h)$ | see below |

**Print the 214.7 GB.** It is the number that makes FlashAttention obviously necessary rather
than a clever optimization. A single attention layer's score matrix, at the model's own advertised
context length, does not fit in any GPU made. **Yet the model runs. How?**

**The crossover point, worked (this is a genuinely useful piece of engineering intuition):**
attention FLOPs exceed FFN FLOPs when $4n^2 d > 6 n d\, d_{\text{ff}}$, i.e. when
$n > 1.5\, d_{\text{ff}} = 1.5 \times 12288 = \mathbf{18{,}432}$ tokens for Qwen3-8B.
**Below ~18k tokens, the FFN dominates compute, not attention.** This is a genuinely
counter-intuitive and *useful* fact: at typical chat lengths, the $O(n^2)$ term isn't your
bottleneck at all. **Say this, because the standard narrative badly overstates when $n^2$ bites.**

**Mitigation 1 — FlashAttention: fix the memory, not the FLOPs. `[V]`**

**The key insight, stated precisely, because everyone gets this wrong:** FlashAttention computes
**exactly the same** attention. It is not an approximation. It never materializes the $(n,n)$
matrix in HBM — it tiles the computation, keeps blocks in SRAM, and uses the **online softmax**
(running max + running normalizer, à la the classic streaming-logsumexp trick) so the
normalization can be corrected as new blocks arrive. Memory: $O(n^2) \to O(n)$. FLOPs: **the
same, or slightly more** (there's recomputation in the backward pass). It's faster anyway because
attention is **memory-bandwidth-bound**, not compute-bound, and you removed almost all the HBM
traffic.

> **The sentence:** *FlashAttention made attention faster by making it do more arithmetic. That's
> only paradoxical if you think the GPU's problem is arithmetic. Its problem is memory.*

**2026 status `[V]`:** **FlashAttention-4** — paper published **March 5, 2026** ("FlashAttention-4:
Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling," Tri Dao et al.), code
shipped on GitHub before that; preliminary results at Hot Chips Aug 2025. Available as
`flash-attn-4` on PyPI. Targets **NVIDIA Blackwell (B200/GB200)**. Key facts worth stating:
- FA3 targeted Hopper, reaching **~740 TFLOP/s at ~75% utilization** `[V]`.
- Blackwell is **asymmetric**: tensor-core throughput more than doubled (**~1 PFLOPS → ~2.25
  PFLOPS FP16/BF16**) while shared-memory bandwidth and the **SFU (special function unit, which
  computes `exp`)** did *not* scale with it. `[V]`
- **So FA4 computes `exp()` via a polynomial approximation on the FMA units instead of the SFU** —
  moving the exponential off the bottlenecked unit onto the compute Blackwell has in surplus.
  `[V]`
- FA4 is written entirely in **CuTe-DSL** (Python-embedded, part of NVIDIA CUTLASS); Tri Dao notes
  compile/install now takes **seconds instead of minutes or hours**. `[V]`

**Why this is worth teaching to *this* learner, and not just name-dropping:** he owns a
**Grace-Blackwell DGX Spark (GB10)**. FA4's entire design rationale — "the tensor cores got 2.25×
faster but the exp unit didn't, so we moved exp to the tensor-adjacent units" — is a *concrete,
legible instance of the co-design principle he can verify on his own hardware*. **This is the best
available demonstration that "architecture" in 2026 means algorithm+silicon together.** Give it a
paragraph, not a footnote.

**Mitigation 2 — GQA/MQA: fix the KV cache, not the score matrix. `[V]`**

**Different problem entirely — flag the distinction, because learners merge these two and then
can't reason about either.** FlashAttention fixes *training/prefill activation memory*. GQA fixes
*decode-time KV cache*. Orthogonal. Both ship together.

At decode, you cache $K$ and $V$ for all previous tokens:

$$\text{KV bytes} = 2 \cdot n \cdot L \cdot n_{kv} \cdot d_h \cdot \text{bytes}_{\text{dtype}}$$

- $2$ — one for $K$, one for $V$
- $n$ — tokens cached; $L$ — layers; $n_{kv}$ — **key/value heads**; $d_h$ — head dim

**Qwen3-8B, real config `[V]`:** $L=36$, $n_{kv}=8$, $d_h=128$, bf16 (2 bytes).

- **Per token, per layer:** $2 \times 8 \times 128 \times 2 = \mathbf{4{,}096}$ bytes = 4 KiB
- **Per token, all layers:** $4096 \times 36 = 147{,}456$ bytes = **144 KiB per token**
- **At $n = 40{,}960$ (full context):** $147{,}456 \times 40{,}960 = 6{,}039{,}797{,}760$ bytes =
  **5.63 GiB**
- **If it were MHA** ($n_{kv} = 32$, i.e. no GQA): **576 KiB/token → 22.5 GiB.** GQA ratio
  $32/8 = 4$ → **exactly 4× saved, 16.9 GiB recovered.**
- **Context:** the weights are $8.19\times10^9 \times 2 = 16.4$ GB. **Without GQA, the KV cache at
  full context would exceed the model itself.** *That* is why GQA is universal in 2026.

**GQA mechanically:** $H = 32$ query heads share $n_{kv} = 8$ KV heads — **4 query heads per KV
head.** MHA is $n_{kv} = H$; MQA is $n_{kv} = 1$; GQA interpolates. MQA is too lossy at scale;
GQA-8 is the empirical sweet spot and it is what almost everything ships. `[V]` for Qwen3;
`[V-sec]` for "MiniMax M2.5 (230B) uses plain GQA."

**Mitigation 3 — MLA (Multi-head Latent Attention). `[V-sec]`**
DeepSeek-lineage models compress KV into a **low-rank latent** and cache *that*, decompressing on
the fly. Used in **Kimi K2/K2.5 (1T total, 32B active, 384 experts/layer — 8 routed + 1 shared)**,
**GLM-5 (744B-A40B)**, **Ling 2.5 1T**, **Sarvam 105B**. `[V-sec, Raschka Jan-Feb 2026 survey]`
Compresses harder than GQA at the cost of extra matmuls. **Teach the *idea* (cache a compressed
representation, pay compute to save memory — a bandwidth/compute trade), not the equations.**

**Mitigation 4 — Sparsity / sliding window / hybrid attention. `[V-sec]`**
Don't attend to everything. **2026 concrete examples:**
- **Arcee Trinity Large (400B-A13B):** sliding window attention, **4096-token window, 3:1
  local:global layer ratio** — i.e. 3 cheap local layers per 1 expensive global layer. Plus
  QK-Norm, gated attention, depth-scaled RMSNorm, DeepSeek-style MoE.
- **Qwen3-Coder-Next (80B-A3B):** **Gated DeltaNet + gated attention, 3:1**, **262k native
  context**.
- **GLM-5:** "DeepSeek sparse attention" + MLA.
- **The pattern to name:** *most layers get a cheap local mechanism; a few get full global
  attention. The global layers do the long-range work; the local layers do the volume.* The
  local:global ratio (commonly 3:1) is the tuning knob. **This is the dominant 2026 long-context
  design and the course should name the pattern, since it generalizes across labs.**

**Mitigation 5 — FlexAttention (PyTorch). `[V]`**
`torch.nn.attention.flex_attention` (PyTorch 2.5+). You write `score_mod` / `mask_mod` as
**ordinary Python functions**, and `torch.compile` fuses them into block-sparse
FlashAttention-equivalent kernels. A `BlockMask` object carries the sparsity structure and can
change *without recompiling*. `[V]` **Caveats to state honestly `[V]`:** kernel options are
**not yet considered public API** and may change; there are documented **numerical discrepancies
vs `F.scaled_dot_product_attention`**; and you must **profile to confirm you got a `flex` /
`block_sparse` kernel and not a silent fallback** to dense SDPA.

**For the runnable-code sections, the recommendation is:** teach `F.scaled_dot_product_attention`
as the default (stable, public API, dispatches to a FlashAttention backend automatically), show
the naive implementation first *for understanding*, and present FlexAttention as the escape hatch
for custom masks. **Do not build the course's core code path on FlexAttention given the API
instability.** `[design judgment grounded in the verified caveats]`

**Misconceptions to box:**
- ❌ *"FlashAttention is an approximation / it's linear attention."*
  ✅ **Correction:** it is **numerically exact** (up to floating-point reassociation). It changes
  *where the data lives*, not *what is computed*. **What fixes it:** show that the online-softmax
  identity is an algebraic rearrangement, not a truncation.
- ❌ *"FlashAttention makes attention $O(n)$."*
  ✅ **Correction:** **memory** becomes $O(n)$. **FLOPs stay $O(n^2)$.** Two different
  complexities and only one of them improved. This distinction is the whole reason 128k context
  is *possible* but still *expensive*.
- ❌ *"GQA is an approximation that hurts quality."*
  ✅ **Correction:** it's an architectural choice made **before pretraining** — the model is
  *trained* with 8 KV heads and never had 32. There is nothing being approximated. (Uptraining an
  MHA model into GQA post-hoc is a different thing and *does* cost a little.) `[M]`
- ❌ *"$O(n^2)$ is why long context is hard."*
  ✅ **Correction:** partly, but at $n < 18{,}432$ for Qwen3-8B **the FFN dominates FLOPs** (worked
  above), and at decode time the binding constraint is **KV-cache memory and bandwidth**, not
  attention FLOPs. **"It's $n^2$" is the answer people give; it's usually the wrong bottleneck.**
  This correction is high-value — it's the difference between reciting and understanding.

### Demo (F-a): **The Attention Heatmap the Learner Drives**

**This is the flagship demo of the course. Budget accordingly.**

- **Plot:** an $n \times m$ heatmap of $A$ = the attention weight matrix. Rows = queries
  (labeled with tokens), columns = keys (labeled with tokens). Below it, a bar chart of the
  selected row.
- **Setup:** a short fixed sentence, ~10 tokens, e.g. `The animal didn't cross the street because
  it was too tired`. Ship **real pre-computed embeddings** (dump them from a small model to a
  JSON blob, ~10 tokens × 64 dims = trivial payload) rather than random vectors — the demo must
  compute real math on real data. `[implementation note]`
- **Controls:**
  1. **Head selector** (dropdown, heads 0..7) — swaps in a different pre-dumped $W_Q, W_K, W_V$.
  2. **Temperature / scale slider**: $\tau$ from $0.1\sqrt{d_k}$ to $10\sqrt{d_k}$, log scale,
     **with $\tau = \sqrt{d_k}$ marked with a detent.**
  3. **Causal mask toggle.**
  4. **Row selector** (click a row).
- **Exact JS to implement (no shortcuts, this is all of it):**
  ```
  Q = X @ Wq            // (n,dk)
  K = X @ Wk            // (m,dk)
  V = X @ Wv            // (m,dv)
  S = Q @ K.T           // (n,m)   S[i][j] = dot(Q[i], K[j])
  S = S / tau
  if (causal) for i,j: if (j > i) S[i][j] = -Infinity
  for each row i:
      mx = max_j S[i][j]                       // max-subtraction: numerically required
      e  = exp(S[i][j] - mx)
      A[i][j] = e / sum_j(e)
  O = A @ V             // (n,dv)
  ```
  Plus, for the gradient readout: `jac[i][j] = A[i][j] * (1 - A[i][j])`.
- **Live numeric readouts (these are what make it a lesson rather than a picture):**
  - **Row entropy** $H_i = -\sum_j A_{ij}\log A_{ij}$, displayed in nats, next to
    $\log m$ (the uniform maximum).
  - **Max softmax Jacobian** $\max_{ij} A_{ij}(1-A_{ij})$ — labeled **"gradient health."**
  - **Effective number of keys attended** $= \exp(H_i)$ — a beautiful, legible statistic.
- **The insights, in order, on drag:**
  1. **Drag $\tau$ far below the detent** (large logits): the heatmap **snaps to a hard diagonal
     stripe**. Entropy → 0. Effective-keys → 1. **Gradient health → ~$10^{-5}$, turns red.**
     *"I broke it by making it too confident."*
  2. **Drag $\tau$ far above the detent**: heatmap goes **uniformly grey.** Entropy → $\log m$.
     Effective-keys → $m$. Output = plain average of all values, **identical for every query
     row** — the layer has become a constant function. *"I broke it by making it too humble.
     Attention that attends to everything attends to nothing."*
  3. **Return to the detent:** structured, mid-entropy, healthy gradient. **The learner has now
     physically located the valley between two cliffs, and $\sqrt{d_k}$ is the thing that puts you
     in it.** *This is the single best moment in the unit.*
  4. **Toggle causal:** the upper triangle blacks out and **the surviving weights in each row
     visibly get bigger** (renormalization!). *"Masking isn't deletion; the row still sums to
     one. That's why the mask goes in before the softmax."* — this makes the additive-vs-
     multiplicative box *felt* instead of *told*.
  5. **Switch heads:** genuinely different patterns from the *same sentence*. One head diagonal,
     one head previous-token, one head dumping everything on token 0. **Explicitly call out the
     attention-sink head** — "this head found nothing relevant and is parking its probability
     mass on the first token, because softmax won't let it say 'nothing.'"

### Demo (F-b): **The Q/K/V Dot-Product Widget**

- **Plot:** three panels. Left: token vectors $\mathbf{x}_i$ as points in a 2-D projection.
  Middle: the *same* tokens after $W_Q$ (blue) and after $W_K$ (orange), **in the same 2-D
  space**. Right: the resulting score matrix.
- **Controls:** a 2×2 (or 3×3) matrix editor for $W_Q$ and one for $W_K$ — **four draggable
  numbers each.** Plus a "set $W_Q = W_K$" button.
- **JS math:** identical to F-a but with $d = d_k = 2$ so it's fully visualizable. Compute
  $Q = XW_Q$, $K = XW_K$, $S = QK^\top$, render arrows and the heatmap.
- **The insights:**
  1. **Press "$W_Q = W_K$" and watch the score matrix become visibly symmetric.**
     *"One projection means relevance is mutual. Two projections is what buys 'it' → 'animal'
     without 'animal' → 'it'."* **This is the demo that makes the "why three matrices" answer
     stick**, and it's the only way to make it stick, because the argument is abstract until you
     see the matrix go symmetric.
  2. **Rotate $W_Q$ by 90°** and watch which tokens become "similar" completely change. *"The
     model isn't finding pre-existing similarity. It's learning what similarity means."* —
     **direct payoff of the §3 king/queen calculation.** Close that loop in the text.
  3. **Collapse $W_K$ to rank 1** (make both rows parallel): all keys land on a line, and the
     score matrix's columns become nearly proportional — **every query gets nearly the same answer.
     Rank collapse, visible.** Sets up LoRA's rank intuition for the fine-tuning brief. **Flag
     forward.**

---

## 6. (I) POSITIONAL ENCODING — WHY, AND WHY RoPE WON

**Order note:** this must come *after* F, because the motivation is a property of attention that
the learner has to have seen.

### Intuition first — and open with the *problem*, not the solution

> **Attention has no idea what order anything is in. Shuffle the words, and the attention math
> gives you the same answers in a shuffled order — it's a set operation wearing a sequence
> costume. "Dog bites man" and "man bites dog" are the same input. Positional encoding is how we
> tell it.**

### Prove the problem — don't assert it

Let $P$ be an $n\times n$ permutation matrix. Then:

$$\text{Attention}(PX) = P\cdot\text{Attention}(X)$$

**Sketch the proof, it's three lines and it's convincing:** $Q' = PXW_Q = PQ$, $K' = PK$,
$V' = PV$. Then $S' = Q'K'^\top = PQK^\top P^\top = PSP^\top$. Softmax is row-wise and $P^\top$
permutes columns consistently, so $A' = PAP^\top$. Then $A'V' = PAP^\top PV = PAV$ (since
$P^\top P = I$). **∎**

**Attention is permutation-equivariant. Note that the FFN, being position-wise, is too. So the
entire transformer, minus positional encoding, cannot distinguish "dog bites man" from "man bites
dog."** Say this bluntly. It shocks people and it should.

**Contrast:** a CNN gets position for free from the kernel geometry — the stencil *has* a
left and a right. An RNN gets it free from the loop order. **The transformer threw away both
sources and has to buy position back.** That's the trade, and it's worth naming as a cost of the
weak-inductive-bias bet from §1. **Nice closure.**

### The three approaches

**1. Sinusoidal (Vaswani 2017) — absolute, added to embeddings.**

$$PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

- $pos \in \{0..n-1\}$; $i \in \{0..d/2-1\}$; result added to $\mathbf{x}_{pos}$.
- **Intuition:** it's a binary counter in continuous form — high-frequency dims flip fast (the
  ones digit), low-frequency dims flip slowly (the millions digit). Together they encode $pos$
  uniquely. **This analogy is good and underused; keep it.**
- **Fatal-ish flaw:** it's **absolute**. It tells the model "I am token 5," but what the model
  usually needs is "you are 3 tokens behind me." Relative position must be *inferred* from two
  absolute ones. And **added** to the embedding, it competes for the same vector space as
  meaning.

**2. Learned absolute (BERT, GPT-2) — a lookup table $(n_{\max}, d)$.**
- Simple. Works.
- **Fatal flaw: it cannot extrapolate at all.** Position 2049 in a model trained to 2048 has a
  literally untrained, random embedding. **Hard wall.** This is why GPT-2 could not be
  context-extended and Llama can.

**3. RoPE (Rotary Position Embedding) — 2026's default, essentially universally. `[V]`**

> **Intuition, and this is the sentence to lead with: RoPE doesn't add position to the vector —
> it *rotates* the vector by an angle proportional to its position. Two tokens' dot product then
> depends only on the *angle between them*, which is the *difference* in their positions. You get
> relative position for free, out of the geometry, without ever computing a difference.**

**The mechanism.** Split $\mathbf{q} \in \mathbb{R}^{d_h}$ into $d_h/2$ **2-D pairs**. Rotate pair
$i$ by angle $m\theta_i$ where $m$ = position:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}, \qquad \theta_i = b^{-2i/d_h}$$

- $m \in \mathbb{Z}_{\ge0}$ — absolute position of the token (in tokens)
- $i \in \{0, \ldots, d_h/2 - 1\}$ — which frequency pair
- $b$ — the **RoPE base**, `rope_theta` in the config. **Qwen3-8B: $b = 1{,}000{,}000$ `[V]`**
  (original paper / Llama-1/2: $b = 10{,}000$)
- $d_h$ — head dim. **Qwen3-8B: 128 `[V]`** → **64 rotation pairs.**
- Applied to $\mathbf{q}$ and $\mathbf{k}$ **only** — never to $\mathbf{v}$, never to the residual
  stream. **This is the elegant bit and it's worth stating loudly.**

**The property that makes it work — write this out, it's the entire justification:**

Let $R_m$ be the block-diagonal rotation for position $m$. Rotation matrices satisfy
$R_m^\top R_n = R_{n-m}$ (rotating by $-m$ then $+n$ = rotating by $n-m$). Therefore:

$$\langle R_m\mathbf{q},\; R_n\mathbf{k}\rangle = (R_m\mathbf{q})^\top(R_n\mathbf{k}) = \mathbf{q}^\top R_m^\top R_n \mathbf{k} = \mathbf{q}^\top R_{n-m}\,\mathbf{k}$$

$$\boxed{\;\text{The attention score depends on } (n - m) \text{ — the relative position — and on nothing else positional.}\;}$$

**This is the whole reason RoPE won, and it should be presented as the punchline it is:**
- You **inject absolute** position ($m$ and $n$ independently, cheaply, locally, at each layer's
  Q/K).
- You **get relative** position out ($n-m$), for free, because rotations compose.
- **You never compute $n-m$.** The algebra of rotation does it. *This is genuinely beautiful
  and the learner should be told it's beautiful.*
- Rotation is **norm-preserving** ($\|R_m\mathbf{q}\| = \|\mathbf{q}\|$), so position information
  **does not compete with semantic information for magnitude.** Contrast sinusoidal, which *adds*
  to the vector and *does* compete. This is a concrete, mechanical reason RoPE > additive.
- And this is where **high-school trig is genuinely, non-negotiably load-bearing.** The learner
  knows the 2×2 rotation matrix. Show it's the same one. **Drop to foundations here — this is one
  of the two or three places in the course where it's clearly warranted** (the other being
  §3's cosine).

**Frequencies — carry the arithmetic, it's the practical part:**

With $d_h = 128$, $b = 10{,}000$ (the classic):
- Fastest pair, $i=0$: $\theta_0 = 10000^0 = 1$ rad/token → **wavelength $2\pi \approx 6.28$
  tokens.**
- Slowest pair, $i=63$: $\theta_{63} = 10000^{-126/128} = e^{-0.984 \times 9.210} = 1.156\times10^{-4}$
  → **wavelength $2\pi/\theta \approx 54{,}350$ tokens.**

With $d_h = 128$, $b = 1{,}000{,}000$ (**Qwen3-8B's actual value `[V]`**):
- Fastest, $i=0$: $\theta_0 = 1$ → wavelength **6.28 tokens** (unchanged — the base doesn't move
  the fast end).
- Slowest, $i=63$: $\theta_{63} = 10^{6 \times (-126/128)} = 10^{-5.906} = 1.24\times10^{-6}$ →
  wavelength $\approx \mathbf{5.06\times10^{6}}$ tokens.

**The insight to state:** **raising the base stretches the low-frequency end.** The slowest pair
must not complete a full rotation within the context window, or positions alias — position 0 and
position $\lambda$ become indistinguishable in that pair. **$b=10{,}000$ gives $\lambda_{\max}
\approx 54$k, which was fine at 2k context and marginal at 128k. $b = 10^6$ gives 5M and is
comfortable.** Qwen3-8B's `max_position_embeddings = 40960` `[V]` sits far inside its 5.06M
slowest wavelength. **This directly explains what "increase rope_theta for long context" means —
a thing the learner has certainly seen in a config and never understood.** High-value.

**Long-context extension `[V-sec]`:** `rope_scaling` in the config (Qwen3-8B ships `null` `[V]` —
i.e. no scaling, it's natively 40,960). The family:
- **Position Interpolation (PI):** squash positions $m \to m \cdot L_{\text{train}}/L_{\text{new}}$.
  Crude; hurts local resolution.
- **NTK-aware:** raise the base instead of squashing positions — stretches low frequencies while
  leaving high frequencies alone.
- **YaRN:** per-frequency-band strategy — interpolate the low-frequency (long-wavelength) bands,
  leave high-frequency bands alone, plus a temperature correction. `[V-sec]` reported as needing
  **~10× fewer tokens and ~2.5× fewer training steps** than PI, enabling ~128k.
- **LongRoPE**, and 2026 work like **MrRoPE** (Mixed-radix RoPE, arXiv 2601.22181) which unifies
  the extension zoo as *radix-conversion strategies* and offers training-free variants. `[V-sec]`

**The unifying idea to give the learner:** *all of these are the same move — the model was trained
on a certain range of rotation angles; to run it longer, you rescale the frequency spectrum so
the angles it sees at position 100k look like angles it saw during training.* **One sentence,
covers the whole family.**

**Implementation gotcha worth one box `[M`, architect should verify against
`transformers/models/qwen3/modeling_qwen3.py``]`:** HuggingFace's `rotate_half` does **not** pair
adjacent dims $(0,1), (2,3), \ldots$ as the paper's notation implies. It pairs
$(0, d_h/2), (1, d_h/2+1), \ldots$ — i.e. it splits the vector in half and pairs across. **The two
are equivalent up to a permutation of dimensions (which the learned weights absorb), but a
from-scratch implementation that mixes conventions with pretrained weights produces garbage that
looks like a training bug.** This costs people days. Box it.

### Misconceptions to box

- ❌ *"Positional encoding tells the model the order."*
  ✅ **Correction:** it makes order *recoverable*. Nothing forces the model to use it. Empirically
  some heads ignore position entirely. **The transformer is given the option, not the
  instruction.**
- ❌ *"RoPE is added to the input embeddings."*
  ✅ **Correction:** sinusoidal/learned are added, **once, at the bottom.** RoPE is applied to
  **$Q$ and $K$ only, inside every attention layer, every layer, every time.** Never to $V$,
  never to the residual stream. **What fixes it:** show the code — `q, k = apply_rotary(q, k, ...)`
  sits *between* the projection and the matmul. This placement is the entire design.
- ❌ *"RoPE gives unlimited context length."*
  ✅ **Correction:** RoPE *extrapolates* better than learned embeddings but **degrades well before
  it fails outright.** Beyond the trained range, attention patterns fall apart — hence the
  entire YaRN/NTK/LongRoPE industry, which would not exist if RoPE extrapolated for free.
- ❌ *"Decoder-only models need positional encoding."*
  ✅ **Correction — and this one is a genuinely interesting `[D]`:** causal masking *by itself*
  breaks permutation-equivariance (token $i$ can see exactly $i$ predecessors, so the model can
  in principle count). **"NoPE" (no positional encoding) decoder-only models do learn position
  and can work.** `[M`, and the result is real but the practical conclusion is contested`]`
  **In practice everyone ships RoPE anyway** because NoPE is worse at long context in most
  reported comparisons. **Include this. It's the kind of "wait, the thing everyone says is
  necessary isn't strictly necessary" fact that builds real understanding, and it's honest about
  the field not having a clean story.**

### Demo (I): **The Positional-Encoding Visualizer**

- **Panel 1 — Sinusoidal:** a $(n{=}128) \times (d{=}64)$ heatmap of $PE_{pos,i}$, with a
  **base slider ($10^3$ to $10^7$, log)**. JS: the formula, directly.
  **Insight on drag:** the characteristic diagonal-stripe zebra pattern **stretches** as base
  rises. Left columns (high freq) oscillate visibly; right columns (low freq) are almost
  constant across the whole window. *"The right-hand dimensions barely change over 128 tokens.
  Those are the ones doing long-range work — and if your context is longer than their wavelength,
  they wrap around and lie to you."*
- **Panel 2 — RoPE, the money panel:** a **unit circle**. Draw $\mathbf{q}$ at position $m$ as an
  arrow (one selected frequency pair) and $\mathbf{k}$ at position $n$ as another.
  - **Controls:** slider $m \in [0, 200]$, slider $n \in [0, 200]$, dropdown for which pair
    $i \in \{0..63\}$, slider for base $b$.
  - **JS:** $\theta_i = b^{-2i/128}$; draw $\mathbf{q}$ at angle $\phi_q + m\theta_i$ and
    $\mathbf{k}$ at $\phi_k + n\theta_i$; **display the dot product live** and, crucially, the
    quantity $\mathbf{q}^\top R_{n-m}\mathbf{k}$ computed *separately* alongside it.
  - **THE insight — this is the one to engineer for:** **drag $m$ and $n$ together, keeping
    $n - m = 5$ fixed. Both arrows spin. The dot product does not move. At all.** Then release
    one and watch it change. *"The score only ever knew about the gap. It never knew where you
    were."* **Show the two readouts — the actual dot product and $\mathbf{q}^\top R_{n-m}\mathbf{k}$ —
    agreeing to 8 decimal places as you drag.** The identity, verified live, by the learner's own
    hand. **This is the best possible way to teach $R_m^\top R_n = R_{n-m}$ and it costs 40 lines
    of JS.**
- **Panel 3 — Aliasing:** plot $\cos(m\theta_{63})$ for $m \in [0, 200{,}000]$ at $b=10^4$ vs
  $b=10^6$, with a draggable vertical line at "your context length."
  **Insight:** at $b{=}10^4$ the curve completes multiple cycles inside 200k — **position 0 and
  position 54,350 have identical encodings in this dimension.** At $b{=}10^6$ it hasn't even
  finished a quarter turn. *"That's what `rope_theta: 1000000` in the config file is buying you.
  You've seen that line. Now you know what it does."*

---

## 7. (G) RESIDUAL CONNECTIONS

### Intuition first

> **A residual connection says: "here's the input; the layer's job is only to compute *what to
> change about it*." The default behavior becomes "do nothing," and doing nothing is a thing a
> 100-layer stack desperately needs to be able to do.**

Second intuition, the one that names the section:

> **The gradient highway: the `+ x` gives the gradient a road from the loss straight back to
> layer 1 that doesn't pass through a single weight matrix. Nothing on that road can shrink it.**

$$\mathbf{y} = \mathbf{x} + F(\mathbf{x};\theta)$$

- $\mathbf{x} \in \mathbb{R}^{d}$ — **must have the same shape as $\mathbf{y}$**; this is why $d$
  is constant through the whole transformer stack. (Say this — learners wonder why the width never
  changes. *This* is why.)
- $F$ — the sublayer (attention or FFN)

### The math that makes it obvious

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = I + \frac{\partial F}{\partial \mathbf{x}}$$

Through $L$ layers:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}\prod_{\ell=1}^{L}\left(I + \frac{\partial F_\ell}{\partial \mathbf{x}_{\ell-1}}\right)$$

**Expand the product and look at what's inside:** it contains the term $I \cdot I \cdots I = I$ —
**an unattenuated path from the loss to layer 0.** Compare §4's RNN, where the product was
$\prod \gamma$ with $\gamma \le 1$ and $0.9^{99} = 3\times10^{-5}$.

**Worked comparison — the number that sells it:**
- **No residual**, per-layer Jacobian norm $\gamma = 0.9$, $L = 36$ (Qwen3-8B's depth `[V]`):
  $0.9^{36} = \mathbf{0.0225}$. Gradient at layer 0 is **2.25%** of what it was at layer 36.
- **With residual**, and $\|\partial F/\partial x\| = 0.1$ (a *small* learned update — which is
  what residual blocks actually learn early in training): the multiplier per layer is
  $\approx 1 + \epsilon$ where $\epsilon$ has mean ~0 (updates aren't systematically aligned), so
  the product stays $O(1)$. $\mathbf{\approx 1.0}$.
- **44× more gradient at layer 0**, at $L{=}36$. At $L = 78$ (**GLM-5's layer count `[V-sec]`**):
  $0.9^{78} = 2.6\times10^{-4}$ vs $\approx 1$. **~3,800×.** **The deeper you go, the more the
  residual is doing.** *Residual connections are not an optimization — they are the thing that
  makes depth legal.*

### The residual stream — the mental model to install, because it pays off everywhere

Rewrite the stack as:

$$\mathbf{x}_L = \mathbf{x}_0 + \sum_{\ell=1}^{L} F_\ell(\mathbf{x}_{\ell-1})$$

> **The residual stream is a shared bus, $d = 4096$ wide, running the length of the model. Every
> layer *reads* from it, computes a contribution, and *adds* to it. Nothing is ever overwritten;
> everything is accumulated. The model isn't a pipeline that transforms — it's a whiteboard that
> 36 committees write on in sequence.**

**This model is worth real investment because it explains, downstream, a lot of things the learner
will meet:**
- **Why LoRA works:** you're adding a small extra writer to the bus. `→ fine-tuning brief`
- **Why layers can be pruned/skipped** with surprisingly little damage `[M]` — you removed one
  writer from a bus with 35 others.
- **Why activation steering works:** you can just *add a vector to the bus.* `→ links to §3's
  linear representation hypothesis`
- **Why $W_O$ matters** (§F3): it's the head's *write port* onto the bus.
- **Why $d$ is constant everywhere:** it's a bus. Buses have a width.

**Flag forward hard: this is the single most reusable mental model in the course.** It should be
introduced here and then *invoked by name* in the fine-tuning brief, the interpretability
discussion, and the conditioning discussion.

### Pre-norm vs post-norm — a real detail with a real consequence

$$\text{Post-LN (Vaswani 2017):}\quad \mathbf{y} = \text{LN}(\mathbf{x} + F(\mathbf{x}))$$
$$\text{Pre-LN (everything since ~2020):}\quad \mathbf{y} = \mathbf{x} + F(\text{LN}(\mathbf{x}))$$

- **Post-LN puts a normalization *on* the residual path.** The highway is no longer clean — the
  gradient must pass through LN's Jacobian at every layer. **Consequence: post-LN transformers
  need a learning-rate warmup to train at all, and get unstable past ~12-ish layers `[M]`.**
- **Pre-LN keeps the residual path completely clean** — `x + something`, no operations on `x`
  itself. **Consequence: trains stably at 36, 78, 100+ layers.**
- **Every 2026 model is pre-norm.** `[V-sec]` This is a **settled** question, one of the few.
- **RMSNorm, not LayerNorm:** drops the mean-centering, keeps only the scale:
  $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}$,
  $\text{RMSNorm}(\mathbf{x}) = \mathbf{g} \odot \mathbf{x}/\text{RMS}(\mathbf{x})$ with learned
  gain $\mathbf{g} \in \mathbb{R}^d$ and **no bias**. Cheaper, works as well. **Qwen3-8B:
  `rms_norm_eps = 1e-6` `[V]`** — that $\epsilon$ is right there in the config and the learner
  should be able to point at it in the formula.
- **2026 variants to mention in one line each `[V-sec]`:** GLM-5/Trinity use **depth-scaled
  RMSNorm**; QK-Norm (§F2b) is an RMSNorm applied to $\mathbf{q},\mathbf{k}$; Cohere Tiny Aya uses
  **parallel blocks** (attention and FFN both read $\mathbf{x}$ and both write to the stream,
  rather than sequentially) — **which the residual-stream model makes instantly comprehensible.**
  Nice payoff.

### Misconceptions to box

- ❌ *"Residuals help the network 'remember' the input."*
  ✅ **Correction:** the forward-pass story is secondary. **The mechanism is the backward pass** —
  $\partial y/\partial x = I + \partial F/\partial x$, and that $I$ is a term nothing can shrink.
  **What fixes it:** the $0.9^{36} = 0.0225$ vs $\approx 1$ arithmetic. Compute both.
- ❌ *"Residuals let you make networks arbitrarily deep."*
  ✅ **Correction:** they make deep networks *trainable*, not *better*. Returns diminish; there's
  an optimal depth/width ratio for a given parameter budget, and it's an active research question
  `[D]`. **Note GLM-5 *reduced* depth from 92 → 78 while *widening* embeddings 5,120 → 6,144
  relative to GLM-4.7 `[V-sec]` — a real 2026 lab deciding depth had overshot.** Concrete
  evidence that "deeper is better" is false, from this year.
- ❌ *"`x + F(x)` — doesn't that break if F outputs a different shape?"*
  ✅ **Correction:** yes, and that's *why* $d$ is constant through the stack. In CNNs, where
  channel counts change, you need a 1×1 conv on the skip path to match shapes. **In a transformer
  you never need that — a deliberate design choice, and now the learner knows the width isn't
  constant by accident.**

---

## 8. (J) THE TRANSFORMER BLOCK, ASSEMBLED

**Pedagogical instruction for the architect: assemble it on the page, one line at a time, with the
learner able to answer "why is this line here?" from what they've already learned. Do not show the
finished block and then explain it. Build it.**

### The 2026 block (pre-norm, RMSNorm, GQA, RoPE, SwiGLU) — as PyTorch pseudocode

```python
def block(x, freqs_cis):          # x: (B, n, d)
    # ---- Attention sublayer ----
    h = rmsnorm_1(x)                        # (B, n, d)      [§7: pre-norm keeps the highway clean]
    q = h @ Wq                              # (B, n, H*dh)   [§F1: "what am I looking for"]
    k = h @ Wk                              # (B, n, Hkv*dh) [§F6: Hkv < H  -> GQA, 4x less KV cache]
    v = h @ Wv                              # (B, n, Hkv*dh)
    q = q.view(B, n, H,   dh).transpose(1,2)    # (B, H,   n, dh)  [§F3: heads are a reshape]
    k = k.view(B, n, Hkv, dh).transpose(1,2)    # (B, Hkv, n, dh)
    v = v.view(B, n, Hkv, dh).transpose(1,2)
    q = q_norm(q); k = k_norm(k)            # [§F2b: QK-Norm - re-impose the var-1 assumption]
    q, k = apply_rope(q, k, freqs_cis)      # (B,H,n,dh)     [§I: rotate, don't add. Q,K only.]
    k = repeat_kv(k, H // Hkv)              # (B, H, n, dh)  [broadcast 8 KV heads to 32 Q heads]
    v = repeat_kv(v, H // Hkv)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # [§F2,F5,F6: exact, FlashAttn backend]
    a = a.transpose(1,2).reshape(B, n, H*dh)
    x = x + a @ Wo                          # (B, n, d)      [§F3: Wo mixes heads; §7: the "+x" IS the highway]

    # ---- FFN sublayer ----
    h = rmsnorm_2(x)
    x = x + (silu(h @ W_gate) * (h @ W_up)) @ W_down          # SwiGLU
    return x                                # (B, n, d)
```

**Six "why is this line here" answers the learner should now be able to give unprompted. Make this
an explicit self-check exercise — it's the best possible test of whether §§F, G, I landed:**
1. Why `rmsnorm_1` *before* and not after? → §7 pre-norm; keeps the residual path clean.
2. Why is `Wk` narrower than `Wq`? → §F6 GQA; 8 KV heads not 32; 4× KV cache saving.
3. Why is RoPE applied to `q,k` but not `v`? → §I; the rotation identity only buys you anything
   inside a dot product. $V$ is never dot-producted with anything.
4. Why `+ x` twice? → §7; two sublayers, two writes to the residual stream, two clean highways.
5. Why `Wo`? → §F3; heads are independent until this matmul.
6. Why `q_norm`/`k_norm`? → §F2b; $\sqrt{d_k}$ assumed unit-variance $q,k$, and training doesn't
   preserve that. Re-impose it.

### The FFN — and why it's bigger than you'd guess

> **Intuition: attention moves information between positions. The FFN thinks about it. Attention
> is the only place tokens talk to each other; the FFN is applied to each position completely
> independently — 4096 numbers in, 4096 numbers out, same weights at every position, no
> communication.**

**Classic (2017):** $\text{FFN}(\mathbf{x}) = W_2\,\text{ReLU}(W_1\mathbf{x} + b_1) + b_2$, with
$d_{\text{ff}} = 4d$. Two matrices.

**2026 (SwiGLU):** **three** matrices —

$$\text{FFN}(\mathbf{x}) = W_{\text{down}}\Big(\underbrace{\text{SiLU}(W_{\text{gate}}\mathbf{x})}_{\text{the gate}} \odot \underbrace{(W_{\text{up}}\mathbf{x})}_{\text{the content}}\Big), \qquad \text{SiLU}(z) = z\,\sigma(z) = \frac{z}{1+e^{-z}}$$

- $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$;
  $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$
- **No biases** — Qwen3-8B: `attention_bias: false` `[V]`; FLUX.2 `[V]` explicitly eliminates bias
  params throughout. **Modern models have essentially no bias terms anywhere. Worth a sentence:
  with normalization layers carrying a learned gain, biases turn out to be redundant.** `[M]`
- **Qwen3-8B `[V]`: $d = 4096$, $d_{\text{ff}} = 12{,}288 = 3d$** — *not* $4d$. **Say why:** SwiGLU
  uses 3 matrices instead of 2, so to keep parameters constant you shrink $d_{\text{ff}}$ by
  $2/3$: $\frac{2}{3}\times 4d = \frac{8}{3}d \approx 2.67d$. Qwen3 rounds up to exactly $3d$.
  **The learner who wonders "why 12288 and not 16384" — which is a real thing people wonder —
  now knows. It's not arbitrary; it's a parameter-budget correction for the third matrix.**
  `[V for the numbers; M for the 2/3 rationale, though it is the standard account]`
- `hidden_act: "silu"` `[V]` — right there in the config.

**The gating intuition (worth two sentences, because "SwiGLU works better empirically" is a
non-answer):** $W_{\text{up}}\mathbf{x}$ computes *content*; $\text{SiLU}(W_{\text{gate}}\mathbf{x})$
computes a *per-dimension volume knob* between roughly 0 and 1. The FFN can therefore say "compute
this feature, but only let it through if that other condition holds" — **multiplicative
interaction, which a ReLU MLP can only approximate.** `[M, this is the standard intuition; the
empirical superiority is well-established, the mechanistic story less so — say so]`

**The parameter shock — print it:** the FFN is $3 \times 4096 \times 12288 = 150{,}994{,}944$
params. The attention is $41{,}943{,}040$. **The FFN is 78% of the block.** Attention gets all the
attention; **the FFN holds most of the weights.** This is a great, slightly funny fact, and it
sets up MoE (§12) perfectly: *if you want to make a model sparse, you sparsify the part that's 78%
of it.* **Deliberate setup — use it.**

---

## 9. (K) ENCODER-ONLY / DECODER-ONLY / ENCODER-DECODER

| | Encoder-only | Decoder-only | Encoder-decoder |
|---|---|---|---|
| Example | BERT, embedding models | **every 2026 LLM** | T5, original NMT transformer |
| Attention | bidirectional | causal | enc: bidir; dec: causal + **cross** to enc |
| Training objective | masked LM (~15% of tokens) | next-token (**100%** of tokens) | denoising / seq2seq |
| Signals per token | **~0.15** | **1.0** | ~0.15–1 |
| Good at | representation, classification, **retrieval** | generation, everything else | fixed-input→output tasks |
| 2026 status | **alive and essential — in RAG** | **won** | mostly gone from LLMs; **alive in diffusion** |

### Why decoder-only won — the honest, multi-part answer

Not one reason. Give all four; the single-cause story is the misconception:

1. **Signal density (§F5).** Causal LM extracts a loss term from **every** token; MLM from
   ~15%. **~6.7× more learning signal per FLOP of forward pass.** `[M for 15%]` **When your
   bottleneck is compute and your data is the internet, this dominates everything else.** This is
   probably the biggest single factor and it's rarely stated first.
2. **Task universality.** Every task can be cast as text-completion. Classification, translation,
   QA, extraction — all next-token prediction with the right prompt. **An encoder-decoder needs to
   know what's "input" and what's "output" at architecture-design time. A decoder-only doesn't
   care.** In-context learning falls out of this for free — and *was not designed in*.
3. **Simplicity → scalability.** One stack, one objective, one attention pattern. Encoder-decoder
   has two stacks, cross-attention, and an input/output split to get right. **Fewer things to
   tune is a real advantage at 1000-GPU scale.** Boring but true.
4. **KV-cache economics.** Causal masking means a token's K/V never change once computed. **You
   can cache them forever.** Bidirectional attention means adding a token invalidates *every*
   previous token's representation — no incremental decoding at all. **This is what makes
   generation cheap and it is a direct gift of causality.**

**Be honest about what was lost:** bidirectional encoders are **still better at pure
representation**, because a token's representation can depend on what comes *after* it. **This is
exactly why 2026 RAG pipelines use dedicated (usually encoder-based, bidirectional, contrastively
trained) embedding models rather than pulling hidden states out of your decoder-only LLM.
Decoder-only won at *generation*; it did not win at *representation*.** `[M for the 2026 practice
claim, though it is standard]` **→ Flag forward to the RAG brief; this is the architectural reason
RAG needs a second model, and it's a question the learner will definitely ask.**

**And note the fork:** encoder-decoder isn't dead — **it moved to diffusion.** A text-to-image
model *is* an encoder-decoder: a text encoder (CLIP / T5-XXL / **Mistral Small 3.1 in FLUX.2
`[V]`**) produces $K,V$; the U-Net/DiT is the decoder that cross-attends to them. **The learner
should see that the architecture they were told "lost" is the one they've been running in ComfyUI
this whole time.** Genuinely satisfying, and it's the cleanest possible transition into the fork.
**Recommend ending the unit on this beat.**

---

## 10. (L) PARAMETER COUNTING — DERIVE IT, THEN VERIFY IT AGAINST A REAL MODEL CARD

**This is the payoff section. The learner should finish it able to open any HuggingFace
`config.json` and predict the parameter count to within a rounding error. That capability is
concrete, testable, and immediately useful to someone who fine-tunes models.**

### The general formula

Per block:

$$P_{\text{attn}} = \underbrace{d \cdot H d_h}_{W_Q} + \underbrace{d \cdot H_{kv} d_h}_{W_K} + \underbrace{d \cdot H_{kv} d_h}_{W_V} + \underbrace{H d_h \cdot d}_{W_O} = 2d\,d_h(H + H_{kv})$$

$$P_{\text{ffn}} = 3\, d \cdot d_{\text{ff}} \quad (\text{SwiGLU; use } 2\,d\,d_{\text{ff}} \text{ for classic 2-matrix FFN})$$

$$P_{\text{norm}} = 2d \;(\text{two RMSNorms}) \;+\; 2 d_h \;(\text{QK-Norm, if present})$$

$$P_{\text{block}} = 2d\,d_h(H + H_{kv}) + 3d\,d_{\text{ff}} + 2d + 2d_h$$

Whole model:

$$P_{\text{total}} = \underbrace{V d}_{\text{embed}} + \underbrace{L \cdot P_{\text{block}}}_{\text{blocks}} + \underbrace{d}_{\text{final norm}} + \underbrace{V d \cdot \mathbb{1}[\neg\text{tied}]}_{\text{lm\_head}}$$

**Note the absences and say why:** no positional-embedding table (**RoPE has zero parameters** —
it's pure geometry, computed from $b$ and the position; this is a real and underappreciated
advantage). No biases (`attention_bias: false` `[V]`).

### Worked, exactly, against Qwen3-8B's real `config.json` `[V, fetched from HuggingFace 2026-07-16]`

```json
{"hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32,
 "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
 "vocab_size": 151936, "tie_word_embeddings": false, "rms_norm_eps": 1e-06,
 "rope_theta": 1000000, "max_position_embeddings": 40960, "hidden_act": "silu",
 "attention_bias": false, "torch_dtype": "bfloat16", "sliding_window": null,
 "rope_scaling": null}
```

**Carry every multiplication. Do not skip steps — the arithmetic IS the lesson.**

**Attention, per block:**
- $W_Q$: $4096 \times (32 \times 128) = 4096 \times 4096 = 16{,}777{,}216$
- $W_K$: $4096 \times (8 \times 128) = 4096 \times 1024 = 4{,}194{,}304$
- $W_V$: $4096 \times 1024 = 4{,}194{,}304$
- $W_O$: $4096 \times 4096 = 16{,}777{,}216$
- **Subtotal: $41{,}943{,}040$**
- *(Check the GQA saving: with MHA, $W_K$ and $W_V$ would each be 16,777,216 → subtotal
  67,108,864. **GQA saves 25,165,824 params/block = 37.5% of the attention block**, on top of the
  4× KV-cache saving. Two wins, and people usually only mention one.)*

**FFN (SwiGLU), per block:**
- $W_{\text{gate}}$: $4096 \times 12288 = 50{,}331{,}648$
- $W_{\text{up}}$: $4096 \times 12288 = 50{,}331{,}648$
- $W_{\text{down}}$: $12288 \times 4096 = 50{,}331{,}648$
- **Subtotal: $150{,}994{,}944$**

**Norms, per block:**
- `input_layernorm`: 4096; `post_attention_layernorm`: 4096
- `q_norm`: 128; `k_norm`: 128 (QK-Norm — Qwen3 has these)
- **Subtotal: $8{,}448$**

**Per block total:** $41{,}943{,}040 + 150{,}994{,}944 + 8{,}448 = \mathbf{192{,}946{,}432}$

- FFN share: $150{,}994{,}944 / 192{,}946{,}432 = \mathbf{78.3\%}$ ← **the §8 claim, verified**

**× 36 layers:** $192{,}946{,}432 \times 36 = \mathbf{6{,}946{,}071{,}552}$

**Embeddings:**
- `embed_tokens`: $151{,}936 \times 4096 = 622{,}329{,}856$
- `lm_head` (untied, `tie_word_embeddings: false`): $622{,}329{,}856$
- `model.norm`: $4{,}096$

$$P_{\text{total}} = 6{,}946{,}071{,}552 + 622{,}329{,}856 + 622{,}329{,}856 + 4{,}096 = \mathbf{8{,}190{,}735{,}360}$$

### ✅ THE VERIFICATION — this is the moment

**The Qwen3-8B model card reports: total ≈ 8.2 B, non-embedding = 6.95 B. `[V]`**

- Our derived non-embedding: $6{,}946{,}071{,}552 = \mathbf{6.95\ B}$ ✓ **exact**
- Our derived total: $8{,}190{,}735{,}360 = \mathbf{8.19\ B} \to$ "8.2 B" ✓ **exact**

> **Print this in a box:** *We just derived, from four numbers in a config file and nothing else,
> a count that matches the official model card to every significant figure they publish. The model
> card is not telling you something you can't compute. You can compute it. Nothing about this
> model is hidden.*

**That is the single most empowering moment available in this unit.** It converts model cards from
authority to arithmetic. Build the whole section toward it.

**Also worth noting:** "Qwen3-8B" is an 8.19B-parameter model. **The name is honest** — but note
that 622M+622M = **1.24 B params (15.2% of the model) are pure lookup tables**, and that the
"8B" in the name includes them while the "non-embedding 6.95B" figure is what actually does
computation. **When you compare an "8B" model to another "8B" model with a smaller vocab, you are
not comparing equals.** Good calibration lesson; vocab sizes vary a lot (32k → 256k).

### The three numbers the learner should now be able to produce for any model

**1. Weights in memory:**
$$M_{\text{weights}} = P \times \text{bytes/param}$$
- bf16/fp16: 2 B → $8.19\times10^9 \times 2 = \mathbf{16.4\ GB}$
- fp8: 1 B → **8.2 GB**
- int4 (Q4): 0.5 B → **4.1 GB** (plus quantization scales, ~5-10% overhead)
- **fp32: 32.8 GB** — and nobody does this for inference.

**2. KV cache:** (from §F6) $2 n L H_{kv} d_h \cdot \text{bytes}$ = **144 KiB/token**, **5.63 GiB
at 40,960 tokens.**

**3. Full fine-tuning memory** — the number that decides what he can actually do on his box:

| Component | Formula | Qwen3-8B, bf16 + AdamW |
|---|---|---|
| Weights | $2P$ | 16.4 GB |
| Gradients | $2P$ | 16.4 GB |
| Adam $m$ (fp32) | $4P$ | 32.8 GB |
| Adam $v$ (fp32) | $4P$ | 32.8 GB |
| fp32 master weights | $4P$ | 32.8 GB |
| **Subtotal** | $\approx \mathbf{16P}$ | **131.2 GB** |
| Activations | depends on $B$, $n$, checkpointing | +GBs |

> **$\approx 16\times$ the parameter count in bytes, for full fine-tuning with AdamW.** Memorize
> **16P**. It is the most useful single heuristic in applied deep learning.

**The DGX Spark punchline — and this is the emotional core of the whole course, so land it
precisely:**
- **DGX Spark: 128 GB unified LPDDR5X, GB10 Grace-Blackwell, ~1 PFLOP FP4 (sparse), ~273 GB/s
  memory bandwidth.** `[M — architect MUST verify these specs against NVIDIA's page before
  shipping; the 273 GB/s figure in particular is load-bearing and I could not verify it this
  session]`
- **Full fine-tune of Qwen3-8B needs ~131 GB + activations. His 128 GB box misses. Barely.**
  *By about 3 GB.* **That near-miss is the most motivating fact available** — the learner is
  *right at the boundary*, and the next brief is about the technique that moves the boundary.
- **LoRA on the same model:** weights frozen (16.4 GB, no gradients, no optimizer state) + LoRA
  params (a rank-16 LoRA on all attention projections is ~$10^7$ params `[M]`, so ~20 MB weights
  + ~160 MB optimizer state) + activations. **~20 GB total. Fits with 100+ GB to spare.**
- **The framing:** *"Full fine-tuning your 8B model: 131 GB, doesn't fit. LoRA: 20 GB, fits six
  times over. That factor of six-and-a-half is the entire reason this course's destination is
  reachable on hardware you own. The next unit is about how."*
  **→ This is the hand-off to the fine-tuning brief. It should be the last paragraph the learner
  reads in this unit.**

**Bonus number, and it's a good one — the bandwidth ceiling:** at decode you must read **every
weight** for **every token**. $16.4\ \text{GB} / 273\ \text{GB/s} = 60\ \text{ms/token}$ →
**~16.6 tokens/s, hard arithmetic ceiling, independent of how fast the tensor cores are.** `[M,
contingent on the bandwidth figure]` **This is §4's irony made numeric: at decode, a transformer
has RNN economics — one token at a time, matrix-*vector* products, arithmetic intensity ~1
FLOP/byte, tensor cores idle.** It's also why batching and speculative decoding exist, and why
quantizing to int4 roughly quadruples decode speed (4.1 GB / 273 GB/s → 15 ms/token → ~66 tok/s)
**even though it does nothing for FLOPs.** *The learner has watched his tokens/sec change when he
switched quantization. Now he can predict it with division.* **Extremely high-value; include it.**

### Misconceptions to box

- ❌ *"A 7B model needs 7 GB of VRAM."*
  ✅ **Correction:** 7 GB is int8 weights *only*. bf16 inference: 14 GB + KV cache + activations.
  Full fine-tune: **~112 GB.** The multiplier between "the number in the name" and "the memory you
  need" ranges from **0.5× to 16×** depending on what you're doing. **What fixes it:** the table
  above. Make them fill it in for a second model.
- ❌ *"Parameter count tells you the compute cost."*
  ✅ **Correction:** for MoE, **catastrophically false** — Kimi K2 is 1T params but activates 32B
  `[V-sec]`. Compute tracks *active* params; memory tracks *total*. **These decoupled in 2026 and
  the model-name convention (`397B-A17B`) exists precisely because of it.** Teach the notation
  explicitly — `A` = active. The learner will see it on every model card and nobody defines it.
- ❌ *"Embedding parameters are just as important as the rest."*
  ✅ **Correction:** they're a lookup table — $O(1)$ compute per token (one row read), regardless
  of being 15% of the params. **They cost memory, not FLOPs.** This is why $C \approx 6ND$ uses
  **non-embedding** $N$ — and why model cards report "non-embedding params" as a separate line.
  **The learner has seen that line and wondered why. Now they know.**

---

## 11. (M) SCALING LAWS, MoE, AND "EMERGENCE"

### Scaling laws

> **Intuition: loss falls as a straight line on a log-log plot, across ten orders of magnitude of
> compute. It is the most reliable quantitative fact in machine learning, and nobody knows why it
> is true.**

Chinchilla-style form:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

- $L$ — cross-entropy loss, **nats per token**
- $N$ — non-embedding parameters
- $D$ — training tokens
- $E \approx 1.69$ — irreducible loss (the entropy of text itself)
- $A \approx 406.4$, $\alpha \approx 0.34$; $B \approx 410.7$, $\beta \approx 0.28$
- `[M` — these are the Chinchilla (Hoffmann et al. 2022) fitted constants from memory. **They are
  dataset-specific and the architect should verify or, better, present them as "one paper's fit"
  rather than constants of nature.** The *functional form* is the durable part.`]`

Compute: $C \approx 6ND$ FLOPs. **Derive the 6, don't assert it:** forward is ~2 FLOPs per
parameter per token (one multiply, one add), backward is ~2× forward (gradients w.r.t. inputs and
w.r.t. weights). $2 + 4 = 6$. **This is a 30-second derivation that makes the constant stop being
magic.**

**Chinchilla-optimal:** minimize $L$ subject to $C = 6ND$ → **$D \approx 20N$.**

**And then say what actually happened, because this is where the textbook is wrong:**
**Nobody trains Chinchilla-optimal anymore.** Chinchilla minimizes *training* compute. **Real
labs minimize training + a billion tokens of inference**, which pushes you to smaller models
trained far longer. Llama-3-8B: 15T tokens `[M]` = **~1,900 tokens/param, ~95× past
Chinchilla-optimal.** Qwen3 pretrained on ~36T tokens `[M — verify]`. **The correct 2026 statement
is: Chinchilla is the right answer to a question nobody is asking.** Present it as a landmark, not
a recipe.

**Worked — the number that justifies the entire course's existence:**
- Qwen3-8B: $N = 6.95\times10^9$ `[V]`, $D \approx 3.6\times10^{13}$ `[M]`
- $C \approx 6 \times 6.95\times10^9 \times 3.6\times10^{13} = \mathbf{1.5\times10^{24}}$ FLOPs
- **On a DGX Spark**, generously assume ~$1.25\times10^{14}$ FLOP/s sustained bf16 `[M`, verify`]`:
  $1.5\times10^{24} / 1.25\times10^{14} = 1.2\times10^{10}\ \text{s} = \mathbf{\approx 380\ years}$.

> **The sentence:** *Pretraining Qwen3-8B on your DGX Spark would take about four centuries. This
> is not a hardware problem you can shop your way out of. **This is why the destination of this
> course is fine-tuning.** You are not going to build the base model. You are going to take
> someone's four centuries of compute and spend four hours redirecting it at your problem.*

**Put this early in the course too** — it reframes fine-tuning from "the cheap option" to "the
only option," which is both true and motivating.

### Mixture of Experts

> **Intuition: the FFN is 78% of every block (§8, §10 — verified) and it's doing something
> different for a Python token than for a Portuguese token. So build 128 FFNs, and for each token
> let a tiny router pick 8. You get 128 FFNs' worth of stored knowledge for 8 FFNs' worth of
> compute.**

$$\mathbf{y} = \sum_{e \in \text{TopK}(g(\mathbf{x}))} g_e(\mathbf{x})\cdot \text{FFN}_e(\mathbf{x}), \qquad g(\mathbf{x}) = \text{softmax}(W_r\mathbf{x})$$

- $W_r \in \mathbb{R}^{d \times E}$ — the router; **$E$ = number of experts; $K$ = experts
  activated per token.** $W_r$ is *tiny* — for GLM-5, $6144 \times 256 = 1.6$M per layer.
- **The routing is per-token, per-layer.** Not per-sequence. **Say this — "expert" strongly
  implies "the French expert," and it is wrong.** Different tokens in the same sentence go to
  different experts, and the same token goes to different experts at different layers.

**Verified 2026 MoE configs `[V-sec, Raschka Jan–Feb 2026 survey]` — a table worth printing:**

| Model | Total | Active | Experts | Notes |
|---|---|---|---|---|
| Kimi K2 / K2.5 | **1 T** | **32 B** | **384/layer (8 routed + 1 shared)** | MLA, SwiGLU; K2.5 multimodal, ~15T mixed tokens |
| GLM-5 | **744 B** | **40 B** | **256 (8 + 1 shared)** | 78 layers (down from 92), $d$=6144 (up from 5120), $d_{\text{ff}}$=2048, MLA + sparse attn |
| Qwen3.5 | **397 B** | **17 B** | — | hybrid Gated-DeltaNet attn; also 27B, 35B-A3B, 122B-A10B |
| Arcee Trinity Large | **400 B** | **13 B** | — | SWA 4096, 3:1 local:global, DeepSeek-like MoE |
| StepFun Step 3.5 Flash | **196 B** | **11 B** | — | 128k ctx, MTP-3, gated attn, ~100 tok/s on Hopper |
| Qwen3-Coder-Next | **80 B** | **3 B** | high count + shared | 262k native ctx |
| Ling 2.5 | **1 T** | — | — | Lightning Attn + MLA; ~3.5× K2's throughput @32k |
| Llama 4 Maverick | — | — | **2 active, $d_{\text{ff}}$=8192** | *fewer, bigger* experts |
| DeepSeek V3 | — | — | **9 active, $d_{\text{ff}}$=2048** | *more, smaller* experts |

**The Maverick-vs-V3 contrast is the best thing in this table — use it.** Two serious labs made
**opposite** choices on the same axis (few-big vs many-small experts) at the same time. `[D]`
**This is a live design disagreement, and showing the learner that the field disagrees about a
first-order architectural parameter is worth more than any settled fact.**

**Sparsity ratios worth computing:** GLM-5 = 40/744 = **5.4% active**. Kimi K2 = 32/1000 =
**3.2%**. Qwen3-Coder-Next = 3/80 = **3.75%**. **The 2026 frontier converged on ~3-5% activation
without agreeing on how to get there.** That's a real observation.

**The shared expert** (in almost every 2026 design): one expert that **always** runs, for every
token. **Intuition: some processing is needed by everything — grammar, basic syntax. Making the
router rediscover that for every token wastes routing capacity and forces redundant copies into
the specialized experts.** Pin it. `[M for the rationale; V-sec for the prevalence]`

**Honest costs — the course must state these or it's selling:**
- **Memory is total, not active.** Kimi K2 = 1T params. **At fp8 that's 1 TB of weights.** You
  are not running it on a 128 GB box. **MoE trades *your* memory for *someone else's* compute
  savings — it is a data-center optimization that is actively hostile to local inference.**
  This is directly relevant to a DGX Spark owner and he should hear it plainly.
- **Load balancing is a real, unsolved-ish problem.** Routers collapse onto a few experts without
  auxiliary losses. `[M]`
- **Fine-tuning MoE is harder** — routing can shift; experts can be starved. `[M]` **→ flag
  forward to the fine-tuning brief; if the learner tries to LoRA a MoE model this will bite him.**

### "Emergence" — `[D]`, and the course must be honest

> **The claim: some capabilities appear *abruptly* at a scale threshold — flat, flat, flat, then
> suddenly the model can do 3-digit arithmetic. Nothing in the loss curve predicted it.**

**The 2026 state of the debate, verified this session:**

- **The mirage argument (Schaeffer et al., NeurIPS 2023):** the sharpness is an artifact of the
  **metric**, not the model. `[V]` **Discontinuous metrics** — exact-match, multiple-choice
  accuracy — *manufacture* discontinuities: getting 4 of 5 digits right scores exactly 0, same as
  getting 0 of 5. Swap to a **continuous** metric — token edit distance, Brier score, per-token
  log-likelihood — and **the same models on the same task produce smooth, predictable curves.**
  `[V]` **This is a strong and largely-accepted result and the course must lead with it, because
  it is the correction to the popular story.**
- **What is NOT resolved, as of 2026 `[V]`:**
  1. **Predictability in advance.** Even granting smoothness in hindsight, **nobody can reliably
     predict *which* capability appears at *what* scale before training.** Smooth ≠ forecastable.
     This is the live question and there's a research line on it (e.g. "Predicting Emergent
     Capabilities by Finetuning," "Random Scaling of Emergent Capabilities"). `[V]`
  2. **Whether continuous metrics fully dissolve it.** Recent work (≈Feb 2025 onward) argues
     **no** — reporting that **bimodality persists even under continuous metrics**: continuous
     loss distributions across seeds remain **visibly bimodal**, especially where runs split
     roughly evenly into successes and failures. `[V]` **A distribution with two modes is not
     explained by "you picked a bad metric."**
  3. There is 2025-26 work (e.g. "Why are LLMs' abilities emergent?", arXiv 2508.04401) arguing
     for emergence from other directions. `[V-sec]`

**The honest 2026 formulation for the course — I'd suggest close to this wording:**

> *"Emergence" bundles two claims. The first — **"capability curves have sharp discontinuities"** —
> is largely a metric artifact, and this is well-established: measure continuously and the cliff
> becomes a slope. The second — **"we cannot predict in advance which capabilities a bigger model
> will have"** — is **not** an artifact and **is still true in 2026**. That second claim is the
> one that matters for safety, for planning, and for anyone deciding what to build. **Be
> skeptical of "emergent" when it's used to mean "magic." Take it seriously when it's used to
> mean "we genuinely could not have called this in advance."** And note the field has not
> converged: recent results show bimodal outcome distributions that survive the continuous-metric
> fix, which nobody has fully explained.*

**Why this belongs in the course rather than being cut as too-meta:** it is a **calibration
lesson**. This learner is going to read a lot of hype. Giving him one worked example of "here is a
famous claim, here is the deflationary correction, **and here is the part of the original claim
that survived the correction, and here is what's still open**" is a transferable skill and worth
more than another architecture diagram. **Do not let it become a both-sides mush — the metric
critique is *substantially correct* and should be stated as such. The residue is real too.**

### Misconceptions to box

- ❌ *"Scaling laws guarantee that more compute = better model."*
  ✅ **Correction:** they predict **pretraining loss on the training distribution.** The map from
  loss to *capability* is empirical, noisy, and not part of the law. The law is about
  cross-entropy in nats. It is not about intelligence.
- ❌ *"MoE models are more efficient."*
  ✅ **Correction:** **more compute-efficient per parameter; less memory-efficient per
  capability.** They're a *data-center* optimization. **For local inference on a fixed-memory box,
  a dense model of the same memory footprint usually beats an MoE.** Directly relevant to him.
- ❌ *"Emergent abilities prove something deep is happening."*
  ✅ **Correction:** see above. Most of the *sharpness* is your ruler. **The unpredictability is
  real.** Two different claims; keep them apart.
- ❌ *"Chinchilla tells you how to train a model."*
  ✅ **Correction:** Chinchilla optimizes *training* FLOPs only. Add inference to the objective
  and the answer moves hard toward smaller-and-longer. **Every 2026 model is far past
  Chinchilla-optimal, on purpose.**

---

## 12. CROSS-BRIEF RECONCILIATION — THINGS THE ARCHITECT MUST RESOLVE

These are places where this brief **will** collide with others. Flagging rather than deciding:

1. **§10's `16P` fine-tuning memory table and the DGX Spark near-miss (131 GB vs 128 GB) belong
   to the FINE-TUNING brief's territory.** I included them because they are the *emotional
   payoff* of parameter counting and I think this unit should end on them. **Architect: decide
   who owns this arithmetic, and make sure both briefs use the same numbers.** If the fine-tuning
   brief has different figures for Qwen3-8B, one of us is wrong. **Recommend: this unit *derives*
   `16P` and states the near-miss as a cliffhanger; the fine-tuning brief *resolves* it.**

2. **The reference model.** I standardized on **Qwen3-8B** because I verified its `config.json`
   directly and the derivation matches its model card to the digit (§10). **Strong
   recommendation: make Qwen3-8B the course-wide reference model**, so 4096 / 36 / 32 / 8 / 128 /
   12288 / 151936 / 40960 become numbers the learner recognizes on sight. **If another brief has
   standardized on Llama or gpt-oss, reconcile — the recurrence is the whole point and it's worth
   more than any individual model choice.**

3. **Cross-attention (§F4) is the diffusion brief's foundation.** I've built the trunk to deliver
   the learner to the diffusion track already understanding that conditioning = cross-attention =
   `attn(Q=image, K=text, V=text)`. **The diffusion brief should not re-derive attention.** But it
   **must** handle the fact that **FLUX.1/FLUX.2 use MMDiT joint attention, not separate
   cross-attention** `[V]` — I set that up here as "cross-attention is the special case," and the
   diffusion brief needs to land it. **If the diffusion brief teaches SDXL's U-Net as the primary
   example, we need to agree on whether the course's diffusion destination is U-Net-era (SDXL) or
   DiT-era (FLUX.2). This learner runs ComfyUI *today*, so he plausibly uses both — but they are
   architecturally different and the CNN section (§2) is sized on the assumption that U-Nets
   matter. If the diffusion brief goes pure-DiT, §2 should shrink further.** **This is the single
   biggest open coordination question and it should be settled early.**

4. **Embedding geometry / anisotropy (§3) is a load-bearing dependency for the RAG brief.** I've
   flagged forward that raw cosine similarity is badly calibrated and that this is *why* RAG uses
   dedicated bidirectional embedding models (§9). **The RAG brief should pick this up by name
   rather than re-explaining embeddings.**

5. **The "residual stream as a shared bus" model (§7) is the mental model that makes LoRA
   intuitive** ("you're adding another writer to the bus"). **I introduce and name it here.
   Recommend the fine-tuning brief invoke it by name rather than building a new metaphor.** Also
   §F-b demo's rank-collapse panel deliberately pre-loads LoRA's rank intuition.

6. **The `[M]` DGX Spark specs (128 GB, ~273 GB/s, ~1 PFLOP FP4) are load-bearing in three
   places** (§10 fine-tuning near-miss, §10 bandwidth ceiling / 16.6 tok/s, §11 the 380-year
   pretraining figure) **and I could not verify them this session.** **Someone must verify these
   against NVIDIA's official spec page before anything ships.** If the bandwidth figure is wrong,
   the tokens/sec arithmetic — which is one of the best moments in the unit because the learner
   can check it against his own box — becomes actively harmful.

7. **Optimizer/training-dynamics overlap:** §4 (BPTT, vanishing gradients) and §7 (residuals,
   pre-norm, warmup) both touch gradient flow. **If a separate TRAINING brief covers vanishing
   gradients, §4 should defer to it and keep only the sequential-parallelism argument** (which is
   the part that's genuinely architectural). **Recommend: gradient *mechanics* live in the
   training brief; §4 keeps only "$0.9^{99}$, therefore LSTM, therefore the gradient-highway idea
   that §7 reuses."**

8. **Numbers I recommend making course-wide constants** (so they recur and become familiar):
   `4096` (d), `36` (layers), `32/8` (Q/KV heads → GQA ratio 4), `128` (head_dim →
   $\sqrt{128}=11.31$), `12288` ($d_{\text{ff}} = 3d$), `151936` (vocab), `40960` (ctx),
   `1e6` (rope_theta), `8.19B` (total), `6.95B` (non-embedding), `16.4 GB` (bf16 weights),
   `144 KiB/token` (KV), `16P` (full-FT memory), `6ND` (compute).
   **These fourteen numbers should appear in every brief that touches a transformer.**
