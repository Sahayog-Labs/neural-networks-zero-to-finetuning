<!--
REPAIR LOG — 2026-07-16, restored against decisions.md §D-21a (canonical page table; content allocation FROZEN).
Defect #3 (Part III): the fan-out folded causal masking into page 31 and pulled O(S²)/KV forward to page 32.
Moves made (existing sections salvaged wholesale — the defect was ALLOCATION, not craftsmanship):
  • Page 32 is now CAUSAL MASKING, its own page (the LLM hinge). Extracted the "F5 — causal masking" section,
    the additive −∞ mask box, the 4096-signals-per-pass box, and misconceptions (d)(e)(f)(g) OUT of the old
    page 31 and expanded them into a full page (objectives, PREDICT, sections, causal-mask-toggle demo, code
    artifact, 7-question quiz, threads, cross-refs). The efficiency insight (position i depends only on ≤i ⇒
    generation can cache) is developed here.
  • Page 31 now owns SELF vs CROSS-attention only (the diffusion hinge). Its causal content moved to 32; it
    now cross-refs 32 for the causal story. Its demo was re-centered on the cross-attention (non-square A)
    view; the causal-toggle demo moved to page 32.
  • Page 37 is now O(S²)/FlashAttention/GQA/KV cache — the entry that was at page 32 was moved here wholesale
    and its header/internals/cross-refs renumbered 32→37.
  • The old page 37 (Scaling laws / MoE / "emergence") was PHYSICALLY MOVED OUT of this file to spec-part4.md
    as page 38 (Part IV opener). It no longer appears here.
Cross-refs touched by the 31/32/37/38 moves were repaired throughout (KV-cache refs 32→37; causal refs 31→32;
scaling/MoE refs 37→38). Pages 33–36 keep their numbers unchanged.
-->

# BUILD SPEC — PART III: ARCHITECTURE (pages 25–37)

**Primary source:** `brief-architectures.md`. **Governing files (win over the brief in every conflict):**
`constants.md`, `notation.md`, `decisions.md`, `hardware-ground-truth.md`. Every number below carries a
`constants.md` confidence tag; nothing tagged [INF]/[EST]/[MEA] may be printed as fact (notation §9 #22–#23).

**What this Part does.** It is the bridge from "a net that trains" (Parts I–II) to "the models we actually
fine-tune." It is a stretch of the shared **trunk**: page 37 hands off to Part IV (Adaptation), which is still
shared — the LLM|Diffusion fork proper comes only after Part IV (page 43). Everything here is taught **once**,
mask-agnostically, so both tracks inherit it (D-14).

---

## GLOBAL CONVENTIONS FOR PART III (apply to every page 25–37)

**Build model legend (the "O/S" column).** The literal numbered D-21 outline with an O/S column is **not
present in `decisions.md`** (see the note at the end of this preamble). I use: **S = Standard build** — page
assembled entirely from stock primitives (`makeCtrl`, `Plot`, `renderQuiz`, `Heatmap`+`NN.attention`,
`TensorViz`, `NetGraph`, `eng`, `hl`); **O = Original build** — page needs bespoke canvas/JS beyond stock
primitives (a hand-drawn unit circle, a receptive-field grower, a positional-encoding zebra map). Builders on
an O page still load and reuse the stock primitives; they add one custom `<canvas>` draw routine.

**Notation — mandatory corrections to the brief (notation.md is normative; the brief violates it repeatedly):**
- **$W$ is ALWAYS $(d_{\text{out}}, d_{\text{in}})$** (notation §1, hard invariant #1). The brief writes
  $W_Q \in \mathbb{R}^{d\times d_k}$ and $\mathbf{q}=W_Q^\top\mathbf{x}$ — that is $(d_{\text{in}},d_{\text{out}})$.
  **Rewrite all attention/projection code in the row-batched form** (notation §1.3: *all attention is
  row-batched*): $Q = XW_Q^\top$, with $X\in\mathbb{R}^{(S,d)}$, $W_Q\in\mathbb{R}^{(d_{\text{head}},d)}$,
  $Q\in\mathbb{R}^{(S,d_{\text{head}})}$. Every projection weight in a shape ribbon reads `(out, in)`.
- **Sequence length is $S$, not $n$.** Query position $i\in 1..S$; key/value position $j\in 1..S$ (notation §3).
  Every attention page carries the Translation note: *"papers write $t$, $i$, $n$, or $j$ for position, and $T$,
  $L$, $N$, or `seq_len` for $S$."*
- **Per-head dim is $d_{\text{head}}$ (=128), not $d_k$.** Translation note: *"papers write $d_k$, $d_v$, $d_h$,
  `head_dim`."* Heads $H=32$; KV heads $H_{kv}=8$. Model width $d=d_{\text{model}}=4096$.
- **Geometric angle is $\vartheta$, never $\theta$** (notation §5.4, which names this brief explicitly:
  *"brief-foundations and brief-architectures both write $\theta$ for the angle — correct them."*). $\theta$ =
  parameters only.
- Attention weights matrix is $A$, shape $(S,S)$, **rows sum to 1, columns do not** (notation §4.2, §7). Additive
  mask $M$, $0$ or $-\infty$. On these pages $A$ is attention's (no LoRA $A/B$ here).
- Sigmoid $\sigma(\cdot)$ **with an explicit argument** = logistic sigmoid (trunk; SiLU $=z\,\sigma(z)$). No
  $\sigma_t$ appears in Part III, so the collision (notation §5) cannot fire here.
- Losses: per-example $\mathcal{L}$, batch-averaged $J$ (notation §2).

**Mechanical requirements every page must satisfy (notation §9, checkable):** every equation with >3 distinct
symbols carries a **Symbol Ledger** (the "From" column is mandatory); every multi-step tensor computation carries
a **shape ribbon**; every worked example terminates in an actual decimal; every demo computes the page's equation
live with values substituted in; every quiz has ≥1 numeric with an explicit tolerance and **no** vocabulary
questions (D-19); every distractor traces to a named misconception; every deviation from field convention gets a
Translation note on the page.

**Expertise-reversal discipline (brief-pedagogy §6, D-19).** Main column is written for the **low-assistance
(expert) reader** — this learner is a rusty ex-practitioner and a heavy ComfyUI user. Novice scaffolding lives in
**collapsed `.deepdive`** blocks, present but not in the flow. Never re-explain something he demonstrably owns
(the −0.428 tax): he knows trig, dot products, `temperature`, quantization tok/s changes, the 77-token CLIP limit,
U-Nets. Lead from those.

**The four recurring threads (notation §10; every beat MUST be tagged `[THREAD:… beat k/n]` and must (a) name it
as the same object, (b) say what's new, (c) point to the prior beat).** Beats live in this Part:
- **The dot product (the atom):** pages 27 (similarity meter), 29 (`QK^T` = a table of dot products), 30
  (per-head projections = choosing directions), 33 (RoPE score depends only on the angle between rotated q,k).
- **Qwen3-8B (the anchor):** every page uses the fourteen frozen numbers (constants §1.1); climaxes on page 36.
- **The chain rule (the verb):** pages 28 (BPTT product of Jacobians), 30 (softmax Jacobian death), 34 (the
  residual $I$-term nothing can shrink), 33 (RoPE never computes a difference).
- **The memory budget (the reality check):** pages 37 (KV cache 144 KiB/token), 36 (16P ledger; measured
  budget), 38 (Part IV — MoE memory = total not active).

**Track hand-off tokens.** Mark the three trunk→track branches visually (a `.track-banner` or an inline
`→ Diffusion` / `→ LLM` badge): **CNN → diffusion U-Net (pages 26–27)**, **cross-attention → diffusion
conditioning (page 31, the diffusion hinge)**, **causal masking → LLM (page 32, the LLM hinge)**.

**⚠️ Retired numbers that a faithful reading of the brief would reproduce — do NOT (D-20):** "misses by ~3 GB"
(→ measured −0.36 GiB, he measures); "~70% FFN" (→ 66.4% of model / 78.26% of block); canonical logits `[2,1,0]`
(→ `[2.0,1.0,0.1]`, D-08); "~380 years, generously assume sustained" (→ *lower bound* at 100% MFU; "at least four
centuries… realistically closer to two millennia", constants §5); "128 GB usable" (→ 121.6875 GiB measured,
hardware §2); "1 PFLOP" / "31 TFLOPS FP32" as a compute ceiling (→ FP4-sparse marketing; dense bf16 ~62 TF [INF],
unresolved vs ~125 TF); $\theta$ for the angle (→ $\vartheta$); $W$ as (in,out).

**⚠️ Spec-alignment note.** (1) **RECONCILED against `decisions.md` §D-21a — the canonical page table now exists
and this Part has been repaired to match it** (see the REPAIR LOG at the top of this file). The 25–37 allocation
is now frozen by D-21a: 29–32 = attention (soft lookup / scale+heads / self-vs-cross / **causal masking, its own
page**), 33 RoPE, 34 residuals, 35 block, 36 param-count, 37 O(S²)/FlashAttention/KV. Scaling laws/MoE/emergence
moved OUT to Part IV page 38. The earlier O/S column reconstruction stands (D-21a paraphrases titles only). (2)
Cross-references to Parts I/II/IV use anchors confirmed against D-21a (TN-1 → p.6 forward / p.14 backprop;
memory-ledger → p.18; roofline → p.43; diffusion track 53–62); Part IV pages are 38–43.

**D-21 numbering.** No renumber shift touches this Part — the GRPO/diffusion renumber only affects pages ≥52. So
for pages 25–37, final number = D-21 number. Each entry states `(D-21: NN)` = its own number.

---

# PART III page-by-page

---

## 25 — architecture-inductive-bias.html
**Title:** "Architecture Is Inductive Bias" · **Part III — Architecture** · (D-21: 25) · **Build: S**

**Learning objectives:**
1. State that architecture is not about what a net *can represent* (universal approximation) but about what it
   can *learn from finite data in finite time*.
2. Read every architectural choice as an installed prior — "a bet about your data" — that pays at small data and
   caps you at large data.
3. Compute, by hand, the parameter cost of a dense first layer vs a conv layer on the same image, and feel the
   gap.
4. Locate the transformer's *weak* inductive bias as the reason it needs scale and wins once it has it.

**PREDICT (opens the page; brief-pedagogy §5.2 — he has a hook: he's seen both MLPs and CNNs):**
> *"A 224×224 RGB image goes into a fully-connected layer with 1000 hidden units. A conv layer with 64 filters of
> 3×3×3 processes the same image. Which has more parameters, and by roughly what factor — 10×, 1,000×, or
> 100,000×?"* (≤3 options, commit before scrolling.) Resolve it with the arithmetic below; name the gap.

**Section outline:**
- *Intuition (lead sentence, verbatim from brief §1):* "An architecture is a hard-coded prejudice about your
  data… a shortcut if you're right and a wall if you're wrong." Second sentence for this learner: an MLP is a
  universal approximator, so architecture is never about *representability* — only about *learnability from
  finite data in finite time*.
- *The universal approximation theorem, stated then punctured.* Present $\big|\sum_{i=1}^{N} w_i\,\phi(\mathbf{a}_i^\top\mathbf{x}+b_i)-f(\mathbf{x})\big|<\varepsilon$
  (use $\phi$ = generic activation per notation §4.1; **not** $\sigma$ unless argument-explicit). Symbol Ledger
  required (>3 symbols). The puncture, inline: the theorem is an *existence* claim about weights — silent on how
  large $N$ is, whether SGD finds them, how much data it takes. Architecture is entirely about those three.
- *The killer worked example (carry to numbers) [DER, brief §1]:* input dim $224\times224\times3=150{,}528$;
  dense layer-1 weights $150{,}528\times1000=150{,}528{,}000\approx1.5\times10^8$ (≈301 MB bf16). Conv layer:
  $3\times3\times3\times64=1{,}728$ + 64 biases $=1{,}792$. **1,792 vs 150,528,000 — an 84,000× gap, in one
  layer.** State the lesson: the FC layer *could* rediscover one edge detector at all 50,176 positions from data;
  the conv is *handed* translation-equivariance.
- *The table to print once and cite forever (brief §1):* rows MLP / CNN / RNN / Transformer / MoE × columns
  {belief installed, mechanism, fails when}. The transformer row is the thesis: **"any token may relate to any
  other; which ones is data-dependent"**, installed by attention, fails when $S$ is huge (cost) or data is tiny
  (no bias to lean on). State the thesis sentence: the transformer's bias is unusually *weak*, which is exactly
  why it needs so much data and wins once you have it. Foreshadow page 38 (Part IV — scaling makes this quantitative).
- **Misconceptions (`.box warn`):**
  (a) *"A bigger MLP can do anything a CNN can, so CNNs are just an optimization."* → true about representation,
  false about learning; point at 1,792 vs 150M. **Fix by arithmetic, not argument.**
  (b) *"More inductive bias is better."* → bias is a bet; at large data it *caps* you (ViT beats CNNs only above
  ~100M images [M, label as such] — which architecture wins *flips* with dataset size).
  (c) *"Architecture is where the intelligence lives."* → in 2026 nearly every frontier open model is the *same*
  block (RMSNorm → GQA/MLA attn → RMSNorm → SwiGLU FFN, residual, RoPE); what differs is scale, data, training.
  Architecture is the substrate. (Forward-ref page 35; page 38 Part IV.)

**Interactive demo — "The inductive-bias parameter counter" (S; `makeCtrl` + `Plot` + readout):**
- Sliders: image side (32–512 px), channels (1/3), hidden units for the FC layer (100–4000), conv filters
  (8–256), kernel $k\in\{1,3,5,7\}$ (stepped).
- Live math (exact): FC params $=(\text{side}^2\cdot C)\cdot\text{hidden}$; conv params $=C_{\text{out}}(C\cdot k^2+1)$.
  Plot both as bars on a **log-y** axis; readout the ratio via `eng()`.
- The "aha": drag image side up — the FC bar explodes quadratically while the conv bar **doesn't move** (conv
  params are independent of $H,W$). *"The conv doesn't care how big the image is. That indifference is the bias."*
  Plant the compute-vs-params asymmetry to be paid off on page 26.

**Code artifact (`.box try`): `code/inductive_bias.py`** — builds `nn.Linear(150528,1000)` and
`nn.Conv2d(3,64,3)`, prints `sum(p.numel() …)` for each; the learner watches 150,528,000 and 1,792 print and
matches them to his hand arithmetic.

**Quiz (6):**
1. MCQ: universal approximation guarantees — *a set of weights exists* (correct) vs "SGD will find them" / "with
   few units" / "for any architecture equally" [distractors = misconception (a)].
2. Numeric: dense layer, 128×128×3 input, 512 units → params? (num 25,165,824, tol 100000, unit params).
3. Numeric: `Conv2d(3, 64, kernel=3)` params incl. bias → (num 1792, tol 0).
4. MCQ: ViT beats CNN on vision only when — *dataset is very large* (correct) [distractor: "always, it's newer" =
   misconception (b)].
5. MCQ: the transformer's inductive bias relative to a CNN's is — *weaker* (correct) [distractor: "stronger,
   that's why it wins"].
6. MCQ (diagnosis): you have 500 labeled images and pick a huge transformer; it overfits instantly. Best fix —
   *use an architecture with more inductive bias (CNN) or get more data* (correct).

**Recurring-thread touchpoints:** `[THREAD:Qwen3-8B beat 1/…]` — first appearance of the anchor family framing
(the block every 2026 model shares). Cross-refs: → page 26 (CNN mechanics), → page 38 (Part IV — scaling makes
"weak bias" quantitative), ← Parts I–II (MLP, backprop, loss).

---

## 26 — convolutions-the-unet-atom.html
**Title:** "Convolutions: What Your U-Net Is Made Of" · **Part III** · (D-21: 26) · **Build: O** (receptive-field grower)

**Framing (brief §2):** this learner runs ComfyUI and has *used* U-Nets. Teach CNNs as "here's what the boxes in
your workflow are made of," **not** an ImageNet history. No AlexNet/VGG/ResNet lineage. Only what survives into
the diffusion track: kernel, stride, padding, channels, downsample/upsample, **receptive field**, equivariance.
Mark this page `→ Diffusion U-Net`.

**Learning objectives:**
1. Read the 2-D cross-correlation as a slid stencil; compute output size, params, and FLOPs for one conv layer.
2. Explain the params-vs-compute asymmetry (params independent of $H,W$; compute $\propto H'W'$) and connect it to
   "1024² is not bigger than 512² but is ~4× slower" — a thing he's felt in ComfyUI.
3. Compute the receptive-field recurrence and see why downsampling makes reach grow *geometrically*, not linearly.
4. Distinguish equivariance (conv) from invariance (global pool) and explain why a generative U-Net has no global
   pooling.

**PREDICT (he has felt this):** *"You render at 1024×1024 instead of 512×512. Does the U-Net now have (a) ~4× more
parameters, (b) the same parameters but ~4× more compute, or (c) both?"* Resolve with the asymmetry below.

**Section outline:**
- *Intuition (brief §2):* a convolution is one small pattern-matching stencil slid over the image, asking at every
  position "how much does the patch here look like my stencil?"; you learn the stencil, the sliding is free — so
  an edge detector learned top-left works bottom-right for free.
- *The math, with shapes and a shape ribbon.* Cross-correlation $y[c_{\text{out}},i,j]=b[c_{\text{out}}]+\sum_{c_{\text{in}},u,v}W[c_{\text{out}},c_{\text{in}},u,v]\,x[c_{\text{in}},s i+u-p,\,s j+v-p]$
  with the full symbol table (Symbol Ledger). Output size $H'=\lfloor (H+2p-d(k_h-1)-1)/s\rfloor+1$. Params
  $P_{\text{conv}}=C_{\text{out}}(C_{\text{in}}k_hk_w+1)$. FLOPs $=2H'W'C_{\text{out}}C_{\text{in}}k_hk_w$.
- *The asymmetry (`.box key`):* conv **params** are independent of $H,W$; conv **compute** $\propto H'W'$. This is
  why a U-Net at 1024² is not bigger than at 512² but is ~4× slower. High-value "oh, THAT's why" (brief §2).
- *Worked example [DER, brief §2]:* `Conv2d(320, 320, 3, padding=1)` on a 64×64 map (a plausible SDXL-class
  interior block; latents 64×64 for a 512-px image at VAE factor 8 [V-sec]): $P=320\times(320\cdot9+1)=320\times2881=\mathbf{921{,}920}$;
  $H'=W'=64$; FLOPs $=2\cdot64\cdot64\cdot320\cdot320\cdot9=\mathbf{7.55\times10^{9}}$ ≈ 7.5 GFLOP — **per conv,
  per denoising step**; at ~40 such convs × ~30 steps, order $10^{13}$ FLOP before attention. This is why sampling
  is slow, and it's arithmetic he can verify.
- *Receptive field (give it real space — brief §2 calls it "the bridge from CNNs to attention to 2026 diffusion").*
  Recurrence $r_0=1$, $r_\ell=r_{\ell-1}+(k_\ell-1)\prod_{j<\ell}s_j$ (Symbol Ledger). Worked all-3×3-stride-1:
  $r=1,3,5,7,\dots=2\ell+1$ → **111 layers to see a 224-px object**. Worked with stride-2 every other layer (a
  U-Net encoder): carry the full ladder $3,5,9,13,21,29,45,61,93,125,189,\mathbf{253}$ → **exceeds 224 at 12
  layers**. State it: **111 → 12.** Downsampling makes reach grow *geometrically*. **This is the architectural
  reason the U-Net has a downsampling encoder — not efficiency, *reach*** — and the reason SDXL bolts
  self-attention into its low-res middle blocks, and the reason FLUX threw the U-Net away for a pure transformer
  (forward-ref page 31/35).
- *Equivariance vs invariance (brief §2):* $\text{Conv}(\text{Shift}_\delta x)=\text{Shift}_\delta(\text{Conv}\,x)$
  (equivariant) vs $\text{GlobalPool}(\text{Shift}_\delta x)=\text{GlobalPool}(x)$ (invariant). A U-Net must be
  equivariant (it outputs an image-shaped thing; shift the noise, the predicted noise should shift), so
  **diffusion U-Nets have no global pooling.** Downsample = strided conv (learned); upsample = transposed conv or
  NN-upsample+conv (the latter avoids checkerboard). Maxpool: one paragraph as "the old way; it throws away
  *where*, and generation needs *where*."
- **Misconceptions (`.box warn`, brief §2):** (a) "a kernel is 2-D" → it's 3-D per filter $(C_{\text{in}},k_h,k_w)$,
  4-D tensor overall; fix by computing 921,920 and matching PyTorch. (b) "deep nets see the whole image at layer
  1" → layer 1 sees 3×3; run the recurrence. (c) "`Conv2d` computes a convolution" → cross-correlation (no flip);
  harmless because the kernel is learned, but say it once for the DSP-literate. (d) "conv is translation-*invariant*"
  → equivariant; and even that is broken at borders by padding and by aliasing in strided downsampling — one
  sentence of honesty (it's why diffusion models sometimes misbehave at image edges).

**Interactive demo — "The Receptive-Field Grower" (O; custom canvas, brief Demo C):**
- Left panel: a 128×128 input grid. Right panel: a schematic layer stack.
- Controls: "add layer" with per-layer $k\in\{1,3,5,7\}$ and $s\in\{1,2\}$; a picker for which output neuron
  $(i,j)$ to trace.
- JS computes live: $r_\ell=r_{\ell-1}+(k_\ell-1)\prod_{j<\ell}s_j$ and cumulative stride $S_\ell=\prod_{j\le\ell}s_j$;
  highlights the input square of side $r_L$ centered at $(S_L i, S_L j)$; overlays the *effective* RF by counting
  paths (a 2-D convolution of indicator kernels — shows the Gaussian-ish falloff, effective < theoretical, a real
  result [M, label]). Readout: $r_L$, cumulative stride, total params, total FLOPs at chosen input size.
- The "aha" on drag: flip one layer s=1→s=2 and watch $r_L$ jump 13→29. *"Downsampling isn't compression; it's how
  the net sees far."* Second aha: stack ten $k{=}1$ convs, $r_L$ stays 1 — *"1×1 convs mix channels, never space"*
  (exactly what `proj_in`/`proj_out` around a U-Net's attention blocks do).

**Code artifact (`.box try`): `code/receptive_field.py`** — implements the recurrence for a config list, prints
$r_L$ and params, and asserts against `sum(p.numel())` of an `nn.Sequential` of `Conv2d`s built to match.

**Quiz (6):**
1. Numeric: `Conv2d(320,320,3,padding=1)` on 64×64 → params (num 921920, tol 0).
2. Numeric: same layer FLOPs (num 7.55e9, tol 0.1e9).
3. MCQ: rendering at 1024² vs 512² changes — *compute ~4×, params unchanged* (correct) [distractors = PREDICT
   options a/c].
4. Numeric: all-3×3-stride-1 stack, receptive field after 5 layers (num 11, tol 0).
5. MCQ: why diffusion U-Nets have no global pooling — *they need equivariance, not invariance* (correct).
6. MCQ (diagnosis): your generative CNN produces checkerboard artifacts on upsample — *use NN-upsample+conv
   instead of transposed conv* (correct).

**Recurring-thread touchpoints:** `[THREAD:Qwen3-8B beat 2/…]` implicit (compute-vs-memory asymmetry seeds the
memory-budget thread). Cross-refs: → page 31 (why SDXL bolts in attention; MMDiT), → diffusion track U-Net pages
(53–62), ← page 25 (the 1,792 number).

---

## 27 — embeddings-and-the-dot-product.html
**Title:** "Embeddings and the Dot Product" · **Part III** · (D-21: 27) · **Build: O** (draggable-vector meter)

**Framing (brief §3):** "This is where the attention chapter is won or lost. If the learner leaves without a
physical feel for *dot product = alignment*, attention will not land. Do not rush it." He knows trig — **drop to
foundations here** (one of the two or three warranted places, the other being RoPE on page 33).

**Learning objectives:**
1. Read an embedding as a learned address in a few-thousand-dim space where "near" means "similar."
2. Read the dot product as a similarity meter: positive = aligned, zero = orthogonal, negative = opposed; and
   $\mathbf{a}\cdot\mathbf{b}=\|\mathbf{a}\|\|\mathbf{b}\|\cos\vartheta$.
3. See that $QK^\top$ *is* a table of dot products — a matmul is $S\times S$ similarities computed at once.
4. Internalize high-dimensional near-orthogonality ($\mathbb{E}[\cos\vartheta]=0$, std $\approx 1/\sqrt{d}$) as
   the storage capacity that makes 4096 dims hold far more than 4096 concepts.

**PREDICT:** *"Two random unit vectors in 4096 dimensions. Is their typical angle closer to 0°, 45°, or 90°?"*
(Almost everyone says 45°; the answer is ~90°.) Resolve with the near-orthogonality demo.

**Section outline:**
- *Intuition (brief §3):* an embedding is a learned address; nothing about the token's identity matters after this
  — only its coordinates. The dot product asks "how much of this vector points along that one?" — a similarity
  meter with a needle.
- *The math.* Embedding lookup as a one-hot matmul: $\mathbf{e}_i=E^\top\mathbf{o}_i$, $E\in\mathbb{R}^{(V,d)}$
  (say this — it explains why gradients flow to embeddings). **Qwen3-8B [VP]:** $V=151{,}936$, $d=4096$, so
  $E$ is $151{,}936\times4096=\mathbf{622{,}329{,}856}$ params ≈ 622M — **print it**: 7.6% of the model in a pure
  lookup table (recurs on page 36). Dot-product geometry (Symbol Ledger, and note $\vartheta$ not $\theta$):
  $\mathbf{a}\cdot\mathbf{b}=\sum_i a_i b_i=\|\mathbf{a}\|\|\mathbf{b}\|\cos\vartheta$, $\cos\vartheta\in[-1,1]$.
  Then $S=QK^\top$, $S_{ij}=\mathbf{q}_i\cdot\mathbf{k}_j$ — say out loud: *"$QK^\top$ is not mysterious; it is
  $S\times S$ dot products at once, because that is what matmul is."*
- *Worked example (do the arithmetic on the page; these vectors recur in the attention widget) [DER, brief §3]:*
  toy 3-D embeddings $\mathbf{v}_{\text{king}}=(0.9,0.8,0.1)$, $\mathbf{v}_{\text{queen}}=(0.85,-0.7,0.15)$,
  $\mathbf{v}_{\text{bicycle}}=(-0.4,0.05,0.9)$. Carry: king·queen $=0.765-0.560+0.015=\mathbf{0.220}$;
  $\|\text{king}\|=\sqrt{1.46}=1.208$, $\|\text{queen}\|=\sqrt{1.235}=1.111$; $\cos\vartheta=0.220/1.342=\mathbf{0.164}\Rightarrow\vartheta=80.6°$.
  king·bicycle $=\mathbf{-0.23}$, $\cos\vartheta=\mathbf{-0.193}\Rightarrow\vartheta=101.1°$. The lesson: king and
  queen are *nearly orthogonal here* because they disagree hard on one axis — **similarity is a projection; which
  directions you care about changes the answer.** Plant the seed: *Q and K are learned projections that choose
  those directions* (harvested page 29–30). Then scale king by 10: raw dot $\to2.20$ (10×), $\cos\vartheta$
  **unchanged at 0.164** — the dot product conflates direction (meaning) with magnitude (confidence), which is
  why 2026 models add QK-Norm (Qwen3 has `q_norm`/`k_norm` [VP]; page 30/34).
- *What embedding space actually looks like — be honest (`.deepdive` for depth, but keep the headline inline):*
  (1) **anisotropy** [M, label] — contextual embeddings occupy a narrow cone; random-pair cosine is *high*
  (~0.5–0.9), so raw cosine is a badly calibrated ruler — *the* reason RAG uses contrastively-trained embedders,
  not raw LLM hidden states (forward-ref RAG, Part IV). (2) `king−man+woman≈queen` is real but oversold (works
  only after excluding the input words) [M]. (3) linear-representation hypothesis [D] — features as directions;
  the implicit justification for why a low-rank update (LoRA) works (forward-ref). (4) **near-orthogonality** —
  in $d=4096$, $\mathbb{E}[\cos\vartheta]=0$, std $\approx 1/\sqrt{d}=0.0156$; exponentially many
  almost-orthogonal directions fit (Johnson–Lindenstrauss) — the quantitative core of "superposition" (JL is
  [VP]-solid; the framing [M]).
- **Misconceptions (`.box warn`):** (a) "each embedding dimension means something" → meaning lives in directions,
  generally not axis-aligned; features outnumber dims; fix with the JL count. (b) "token X always has vector V" →
  only the layer-0 input table; after one block the vector is *contextual* ("river bank" vs "bank account"
  diverge). (c) "higher cosine = more related" → only within a calibrated space; anisotropy means 0.7 may be the
  *floor*; compare against the model's random-pair distribution. **This one actively breaks RAG systems** —
  emphasize, flag forward.

**Interactive demo — "The Dot-Product Similarity Meter" (O; custom canvas, brief Demo D; precursor to page 30's
Q/K/V widget):**
- Panel 1: a 2-D plane with two draggable vectors $\mathbf{a},\mathbf{b}$ from the origin; draw $\mathbf{b}$'s
  projection onto $\mathbf{a}$ as a dashed segment; a needle gauge. Toggle "raw dot" / "cosine". JS (exact):
  `dot=a.x*b.x+a.y*b.y`, `cos=dot/(hypot(a)*hypot(b))`, projection `= a*(dot/(a.x²+a.y²))`, $\vartheta$ from
  `atan2` difference in degrees. Aha: rotate $\mathbf{b}$ through 90° → dot hits exactly 0 (orthogonal); with
  "cosine" on, lengthen $\mathbf{a}$ → raw dot balloons, cosine sits still.
- Panel 2 (the one that earns its keep): a $d$ slider (2→512). Sample 2000 pairs of random unit vectors in
  $\mathbb{R}^d$ (Box–Muller → normalize → dot), histogram the cosines. Drag $d$ up and watch a broad flat
  distribution collapse to a spike at 0 of width $1/\sqrt{d}$. *"In high dimensions, everything is perpendicular
  to everything — and that's the storage capacity."* (Resolves the PREDICT.)

**Code artifact (`.box try`): `code/dot_product_geometry.py`** — computes the king/queen/bicycle numbers; then
samples random unit vectors in $d\in\{2,64,4096\}$ and prints mean/std of pairwise cosine (≈0 ± $1/\sqrt d$).

**Quiz (6):**
1. Numeric: king·queen for the frozen vectors (num 0.220, tol 0.005).
2. Numeric: $\cos\vartheta$ between king and queen (num 0.164, tol 0.005).
3. MCQ: scaling one vector by 10 changes — *raw dot (×10), not cosine* (correct).
4. Numeric: typical std of cosine between random unit vectors in $d=4096$ (num 0.0156, tol 0.003).
5. MCQ: `embed_tokens` for Qwen3-8B has how many params — *622,329,856* (correct) [distractors: 8.19B / 4096 /
   151,936].
6. MCQ (diagnosis): your RAG retrieves garbage; cosine sims are all ~0.8 — *anisotropy; the space isn't
   calibrated, use a contrastively-trained embedder* (correct).

**Recurring-thread touchpoints:** `[THREAD:dot-product beat 3/…]` (neuron → matmul → **now the similarity meter**;
next beat: page 29, `QK^T`). `[THREAD:Qwen3-8B]` (622M embedding table → page 36). Cross-refs: → page 29–30
(learned projections choose the directions), → RAG (Part IV).

---

## 28 — rnns-why-attention.html
**Title:** "RNNs and the Sequential Bottleneck: Why Attention Had to Exist" · **Part III** · (D-21: 28) · **Build: S**

**Framing (brief §4):** "Resist the urge to teach this properly. The learner will never write an LSTM." Two jobs
only: (1) make the sequential-bottleneck pain visceral so attention arrives as relief; (2) install vocabulary
(hidden state, timestep, BPTT) the diffusion sampling loop reuses. LSTM gates get a diagram and a paragraph, not
a derivation.

**Learning objectives:**
1. Read the RNN recurrence and see the hidden state as the *entire* memory of the past.
2. Distinguish the two failures cleanly: vanishing/exploding gradients (a *learning* problem, which LSTMs mostly
   fixed) vs the sequential dependency (a *hardware* problem, which killed RNNs).
3. Explain, with arithmetic intensity, why a matrix-vector product wastes a GPU and a matrix-matrix product
   doesn't — the real reason transformers won.
4. Know that recurrence returned in 2026 (hybrids) *once training-time parallelism was solved* — so the lesson is
   about the training dependency, not "RNNs were dumb."

**PREDICT:** *"Why did transformers beat RNNs — because attention is (a) smarter/more expressive, (b) cheaper per
token, or (c) parallelizable across the sequence?"* (Most say (a); the answer is (c), and it's *dumber* and *more
expensive* per the brief.) Resolve with the arithmetic-intensity comparison.

**Section outline:**
- *Intuition (brief §4):* an RNN reads like you read — one token at a time, carrying a running summary; the
  summary is the only thing that survives, which is both the elegance and the death sentence.
- *The math (state it, don't dwell).* $\mathbf{h}_i=\tanh(W_{hh}\mathbf{h}_{i-1}+W_{xh}\mathbf{x}_i+\mathbf{b})$,
  $\mathbf{y}_i=W_{hy}\mathbf{h}_i$ (Symbol Ledger; **use $i$ for the step, not $t$** — notation §3; Translation
  note: papers write $t$). Weights shared across all $i$ (the RNN's inductive bias: a time-invariant update).
- *Failure 1 — vanishing/exploding gradients (learning) `[THREAD:chain-rule]`.* BPTT:
  $\partial\mathcal{L}/\partial\mathbf{h}_1=(\partial\mathcal{L}/\partial\mathbf{h}_S)\prod_{i=2}^{S}\partial\mathbf{h}_i/\partial\mathbf{h}_{i-1}$,
  $\partial\mathbf{h}_i/\partial\mathbf{h}_{i-1}=\text{diag}(1-\mathbf{h}_i^2)W_{hh}^\top$ — a product of $S{-}1$
  Jacobians (this **is** the chain rule again — tag it as the same verb from Parts I–II). Worked [DER]: with
  $\gamma=0.9$, $S=100$: $0.9^{99}=2.95\times10^{-5}$ (30,000× weaker at step 1); $S=1000$: $0.9^{999}\approx1.7\times10^{-46}$,
  **below bf16's smallest normal $1.2\times10^{-38}$ — literally zero** [VP for the bf16 floor, constants §9.7];
  $\gamma=1.1$, $S=100$: $1.1^{99}=1.25\times10^{4}$ explodes (fixable by clipping; vanishing is not). Note
  $\tanh'(h)=1-h^2\le1$ always → the product is biased toward decay; **vanishing is the default, not the
  accident.** LSTM/GRU cell state has a near-additive path with Jacobian $\approx\text{diag}(\mathbf{f}_i)$,
  $\mathbf{f}_i\to1$ → a gradient highway. **One paragraph, and flag it as the *same idea as a residual
  connection* (page 34) — same problem, same fix, different decade.** (Turns two facts into one idea.)
- *Failure 2 — the sequential dependency (hardware). This is what killed RNNs.* $\mathbf{h}_i$ needs
  $\mathbf{h}_{i-1}$: $S$ sequential, non-overlappable steps. Worked comparison, make it brutal [DER, brief §4]:
  $S=2048$, $d_h=4096$. RNN: 2048 dependent matrix-*vector* products, each ~33.5 MFLOP but reading
  $4096^2\times2=33.5$ MB of weights → **arithmetic intensity ≈ 0.5 FLOP/byte**, run 2048× in a row at <1% of
  peak. Transformer: **one** $(2048\times4096)\times(4096\times4096)$ matmul, same weight bytes read, 2048× the
  FLOPs → **intensity ≈ 1000 FLOP/byte**, tensor cores saturate. Print the bold sentence (brief §4): *"RNNs didn't
  lose because they couldn't represent language. They lost because a matrix-vector product wastes a GPU and a
  matrix-matrix product doesn't. The transformer traded a sequential $O(S)$ dependency for a parallel $O(S^2)$
  one — and paying $S^2$ FLOPs you can spend beats paying $S$ FLOPs you can't."* Pre-loads page 37's $O(S^2)$ (it
  was the *deal*, not an accident) and the LLM-decode irony (at inference the transformer becomes sequential again
  and gets RNN economics — forward-ref page 37/36 and the roofline page 43).
- *Honest 2026 caveat (`.deepdive`; brief §4):* recurrent/linear-attention is back, **hybridized** — Qwen3-Coder-Next
  (80B-A3B, Gated DeltaNet + gated attn, 3:1), Qwen3.5 (397B-A17B), Ling 2.5 1T (Lightning Attn + MLA) [V-sec,
  Raschka survey — label]. They fixed *Failure 2* (parallelizable training via associative scan/chunked matmul)
  while keeping $O(1)$ inference memory. Lesson: **the sequential *training* dependency was the killer; remove it
  and recurrence returns.** But full attention is still the default; MiniMax M2.5 (230B) ships "plain GQA, no
  tricks" and stays competitive [V-sec] — nobody has declared victory [D]. Do not imply otherwise either way.
- **Misconceptions (`.box warn`):** (a) "LSTMs solved long-range" → solved the *gradient* problem, not the
  information bottleneck (fixed $\mathbf{h}$) and not parallelism — two of three remained. (b) "transformers won
  because attention is smarter" → attention is *dumber* (weaker bias) and *more expensive* ($O(S^2)$); it won on
  parallelism; fix with 0.5 vs 1000 FLOP/byte. (c) "RNNs are obsolete" → see the 2026 hybrids.

**Interactive demo — "Gradient decay vs the highway" (S; `Plot`):**
- Sliders: $\gamma$ (0.5–1.2), sequence length $S$ (2–1000, log). Plot $\gamma^{S}$ on a **log-y** axis vs $S$;
  overlay a horizontal line at the bf16 min-normal $1.2\times10^{-38}$ and a second trace for an LSTM-like
  "highway" ($\gamma_{\text{eff}}\approx1$, flat near 1). Readout: gradient-at-step-1 ratio via `eng()`, and
  whether it has underflowed bf16. Aha: drag $\gamma$ from 1.0 down to 0.9 and watch the curve dive below the
  underflow line well before $S=1000$; snap $\gamma$ to 1 (the highway) and it stays put — visual preview of page
  34's residual $I$-term.

**Code artifact (`.box try`): `code/rnn_bottleneck.py`** — times a Python loop of 2048 matrix-vector products vs
one matrix-matrix product of equal weight-bytes on his box; prints FLOPs, bytes, arithmetic intensity, and wall
time for each. (He *sees* the matmul finish faster despite doing 2048× the FLOPs.)

**Quiz (6):**
1. Numeric: $0.9^{99}$ (num 2.95e-5, tol 0.3e-5).
2. MCQ: what actually killed RNNs — *the sequential training dependency (a hardware problem)* (correct)
   [distractors = misconception (b) and "vanishing gradients, unfixably"].
3. Numeric: arithmetic intensity of a batch-1 matrix-vector product reading 33.5 MB for 33.5 MFLOP, in FLOP/byte
   (num 1.0, tol 0.2) — *note the ~1, and that this is exactly LLM decode's problem later.*
4. MCQ: the LSTM cell-state fix is the same idea as — *a residual connection* (correct).
5. MCQ: in 2026, recurrence returned because — *training-time parallelism was solved (chunked/scan form)* (correct).
6. MCQ (diagnosis): your from-scratch RNN's loss won't drop and its early-layer grads are ~1e-40 — *vanishing
   gradients; add a gated/residual path or clip won't help (clip fixes explosion, not vanishing)* (correct).

**Recurring-thread touchpoints:** `[THREAD:chain-rule beat …]` (derivative → backprop → **BPTT as a product of
Jacobians**; next: page 30 softmax Jacobian, page 34 the residual $I$). `[THREAD:memory-budget]` seed (arithmetic
intensity, paid off pages 37/36/43). Cross-refs: → page 37 ($O(S^2)$ was the deal), → page 34 (the highway),
→ roofline page 43 (decode = RNN economics).

---

## 29 — attention-soft-lookup.html
**Title:** "Attention I: The Soft Lookup" · **Part III** · (D-21: 29) · **Build: O** (Q/K/V widget)

**Attention spans four pages (29–32): soft lookup (29), the $\sqrt{d_{\text{head}}}$ scale + multi-head (30),
self vs cross (31), and causal masking (32) — taught mask-agnostically with cross-attention first-class (D-14 #1,
the highest-stakes ask in the trunk). Budget accordingly — this is the center of the course (brief §5).**

**Learning objectives:**
1. Derive attention as a *softened, differentiable* dictionary lookup — build the formula as the answer to a
   question, never open with `softmax(QK^T/√d)V`.
2. Read Q/K/V as three learned projections of the *same* vector: "what am I looking for / what do I advertise /
   what do I hand over."
3. Explain why three projections and not one: $Q\ne K$ buys asymmetric relevance; $V\ne K$ separates "how I'm
   found" from "what I contribute."
4. Write every shape in the attention formula and never guess a shape again.

**PREDICT (light — this is partly definitional, so keep it to the "why three" hook per brief-pedagogy §5.2
restriction):** *"Self-attention makes each token compute a query, a key, and a value. Are these three different
things the token stores, or three views of one thing?"* Resolve with the one-line code kill.

**Section outline:**
- *F1 — derive the intuition before the formula (brief §5 F1).* Step 1: the hard lookup he already knows — a
  Python dict, `d["kitten"]` → `KeyError`. Three objects: query, keys, values. Step 2: name the two problems —
  brittle (kitten should get something cat-ish) and non-differentiable (hard equality → no gradient → can't learn
  what to look up). Step 3: soften — replace equality with a graded similarity score (the dot product, page 27's
  atom) and "return one match" with a *weighted average of all values*:
  $\text{output}=\sum_j \frac{\exp(\mathbf{q}\cdot\mathbf{k}_j)}{\sum_{j'}\exp(\mathbf{q}\cdot\mathbf{k}_{j'})}\mathbf{v}_j$
  (Symbol Ledger). The chapter's intuition sentence (`.box key`, brief §5): *"Attention is a dictionary lookup
  where the keys don't have to match exactly, and instead of one value you get a blend of all of them, weighted
  by how well each key matched. Softening the lookup is what makes it differentiable — and differentiable means
  the network can learn what to look for."* Step 4 (the punchline of F1): tokens don't come with q/k/v; each
  computes its own with **learned projections**. Present in **frozen notation, row-batched, $W$=(out,in):**
  $Q=XW_Q^\top$, $K=XW_K^\top$, $V=XW_V^\top$, with $W_Q,W_K\in\mathbb{R}^{(d_{\text{head}},d)}$,
  $W_V\in\mathbb{R}^{(d_{\text{head}},d)}$, $X\in\mathbb{R}^{(S,d)}$ (shape ribbon mandatory; Translation note:
  *"papers write $\mathbf{q}=W_Q\mathbf{x}$ with column vectors; we use the row-batched form for all attention,
  notation §1.3."*). Say the three roles: $\mathbf{q}_i$ = "what am I looking for?"; $\mathbf{k}_j$ = "what do I
  advertise?"; $\mathbf{v}_j$ = "what do I hand over if picked?"
- *Why three and not one (brief §5 F1 — the question every learner silently has):* (1) $Q\ne K$ because relevance
  is **asymmetric** — in "the animal didn't cross the street because *it* was too tired," `it`→`animal` need not
  equal `animal`→`it`; if $W_Q=W_K$ then $S_{ij}=S_{ji}$, the score matrix is *forced symmetric* and directed
  relations are gone. Callback to page 27: king·queen depends on which directions you measure along; $W_Q,W_K$
  are exactly that learned choice, and choosing *different* directions for asker vs advertiser buys asymmetry. (2)
  $V\ne K$ because "how I'm found" ≠ "what I contribute" — `Paris` advertises "ask me about France" (via
  $\mathbf{k}$) while carrying the geographic content (via $\mathbf{v}$); analogy: a database index vs the row it
  points at — you don't store the row in the B-tree.
- *F2 — scaled dot-product attention, every shape (brief §5 F2).* The boxed formula in frozen notation:
  $$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}}+M\right)V.$$
  Print the **full shape table** (Symbol Ledger + shape ribbon): $X\,(S,d)$; $W_Q,W_K\,(d_{\text{head}},d)$;
  $W_V\,(d_{\text{head}},d)$; $Q\,(S,d_{\text{head}})$; $K\,(S_{\text{kv}},d_{\text{head}})$;
  $V\,(S_{\text{kv}},d_{\text{head}})$; $S=QK^\top\,(S,S_{\text{kv}})$; $M\,(S,S_{\text{kv}})$;
  $A=\text{softmax}_{\text{row}}\,(S,S_{\text{kv}})$; $AV\,(S,d_{\text{head}})$. Batched-with-heads ribbon:
  $(B,H,S,d_{\text{head}})\times(B,H,d_{\text{head}},S_{\text{kv}})\to(B,H,S,S_{\text{kv}})\to(B,H,S,d_{\text{head}})$.
  **Three shape facts to hammer (`.box rule`, notation §7):** (1) the output has $S$ rows, one per **query** — the
  key/value set can be any length (**this single fact makes cross-attention obvious on page 31**); (2) softmax is
  over the **last** axis (the keys) — **row $i$ of $A$ sums to 1, columns do not** (learners read heatmaps
  backwards); (3) $d_{\text{head}}$ links $Q,K$; $S_{\text{kv}}$ links $K,V$; $d$ never appears after the
  projections — the wiring is dimensionally rigid, exactly one legal way.
- *Why softmax specifically (two sentences, brief §5 F2):* enforces $\sum_j A_{ij}=1$, $A_{ij}>0$ → output is a
  **convex combination** of value vectors, so it lives in the convex hull of $V$ and can't blow up (stable by
  construction); and it's smooth, so gradient flows to *every* key including the losers (a hard argmax gives
  gradient only to the winner → nothing learns) — the same "soften to differentiate" move as F1, now concrete. Its
  temperature is implicit in the logit scale — which is exactly what $\sqrt{d_{\text{head}}}$ controls (segue to
  page 30).
- **Misconception (`.box warn`, the #1 attention misconception, brief §5 F1 — box it prominently):** *"Q, K, V are
  three different things the model has, like three memories."* → they are **three projections of the same
  vector**; for self-attention every token makes all three from its own $\mathbf{x}_i$. Fix with one line of
  code: `q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T` — *same `x`, three times*. **This is the exact hinge for
  cross-attention (page 31): a learner who thinks Q/K/V are "three memories" cannot understand cross-attention at
  all, hence cannot understand diffusion conditioning.**

**Interactive demo — "The Q/K/V Dot-Product Widget" (O; custom canvas, brief Demo F-b):**
- Three panels. Left: token vectors $\mathbf{x}_i$ as points in a 2-D projection. Middle: the *same* tokens after
  $W_Q$ (blue) and after $W_K$ (orange) in the same 2-D space. Right: the resulting score matrix $S=QK^\top$ as a
  small heatmap.
- Controls: a 2×2 matrix editor for $W_Q$ and one for $W_K$ (four draggable numbers each; **stored (out,in)** per
  notation — a `nn.Linear(2,3)`-style shape would be unambiguous but here $d=d_{\text{head}}=2$ for
  visualizability, so annotate the ambiguity explicitly per notation §1's spec trap); a "set $W_Q=W_K$" button.
- JS math: $Q=XW_Q^\top$, $K=XW_K^\top$, $S=QK^\top$; render arrows and the heatmap live. Ahas: (1) press
  "$W_Q=W_K$" and watch the score matrix become visibly **symmetric** — *"one projection means relevance is
  mutual; two is what buys 'it'→'animal' without 'animal'→'it'."* (2) Rotate $W_Q$ by 90° and watch which tokens
  are "similar" completely change — *"the model isn't finding pre-existing similarity; it's learning what
  similarity means"* (closes page 27's king/queen loop — say so). (3) Collapse $W_K$ to rank 1 (rows parallel):
  all keys land on a line, score-matrix columns go proportional, every query gets nearly the same answer — **rank
  collapse, visible** (flag forward to SVD/LoRA, Part IV).

**Code artifact (`.box try`): `code/attention_from_scratch.py`** — implements the boxed formula with the
online-safe softmax (max-subtraction) and asserts `torch.allclose` against
`F.scaled_dot_product_attention(q,k,v)` for random Q/K/V; prints the shape ribbon at each step.

**Quiz (6):**
1. MCQ: Q, K, V are — *three learned projections of the same vector* (correct) [distractor: "three memories" =
   the boxed misconception].
2. MCQ: setting $W_Q=W_K$ forces the score matrix to be — *symmetric* (correct).
3. Numeric/shape: $X$ is $(128,4096)$, $W_Q$ is $(128,4096)$ (out,in). Rows in $Q=XW_Q^\top$? (num 128, tol 0);
   and cols? (num 128, tol 0). [Diagnoses the (out,in)/row-batched convention.]
4. MCQ: softmax runs over — *the key axis (last), so each query-row sums to 1* (correct) [distractor: "the query
   axis / columns sum to 1"].
5. MCQ: why $V$ is separate from $K$ — *"how I'm found" ≠ "what I contribute"* (correct).
6. MCQ (diagnosis): your attention output has the wrong number of rows — *you softmaxed/indexed the query axis or
   swapped $K$/$V$ lengths; output rows = number of queries $S$* (correct).

**Recurring-thread touchpoints:** `[THREAD:dot-product beat 4/…]` ($QK^\top$ = a table of dot products; from page
27's meter). `[THREAD:Qwen3-8B]` ($d_{\text{head}}=128$, $H=32$). Cross-refs: → page 30 ($\sqrt{d_{\text{head}}}$
+ multi-head), → page 31 (self vs cross), → SVD/LoRA (Part IV, rank collapse).

---

## 30 — attention-scale-and-heads.html
**Title:** "Attention II: The √d̲ₕₑₐ𝒹 Thermostat and Many Heads" · **Part III** · (D-21: 30) · **Build: S** (flagship Heatmap)

**This carries the flagship demo of the course (brief Demo F-a). Budget accordingly.**

**Learning objectives:**
1. Derive $\text{std}(\mathbf{q}\cdot\mathbf{k})=\sqrt{d_{\text{head}}}$ and see the scale as a *thermostat* on the
   softmax temperature, not "numerical hygiene."
2. Show *quantitatively* that without the scale the softmax saturates and its Jacobian collapses — attention
   doesn't get worse, it gets **stuck** (no gradient to $W_Q,W_K$).
3. Read multi-head as running several relevance-detectors on different learned subspaces; know that heads are a
   *reshape*, that $W_O$ mixes them, and that total attention params are independent of $H$.
4. Recognize QK-Norm as re-imposing the unit-variance assumption that training destroys (Qwen3's `q_norm`/`k_norm`).

**PREDICT (strong hook — he's set `temperature` in ComfyUI/samplers, brief-pedagogy §7.3 style):** *"You feed a
softmax logits that are 10× bigger than usual. Does it become (a) sharper/more one-hot, (b) flatter, or (c)
unchanged — and what happens to the gradient flowing back through it?"* Resolve at the detent in the demo.

**Section outline:**
- *F2b — the √d̲ₕₑₐ𝒹, derived (brief §5 F2b — "the most hand-waved line in every explainer; do it properly").*
  Intuition first: more dimensions in the dot product → bigger scores → a softmax fed big numbers stops being soft,
  becomes a hard argmax, and a hard argmax has no gradient; $\sqrt{d_{\text{head}}}$ is a thermostat. Derivation
  (model $q_i,k_i$ iid mean-0 var-1, ~true at init): $s=\mathbf{q}\cdot\mathbf{k}=\sum_i q_ik_i$, $\mathbb{E}[s]=0$,
  $\text{Var}(s)=\sum_i \mathbb{E}[q_i^2]\mathbb{E}[k_i^2]=d_{\text{head}}$ → $\text{std}(s)=\sqrt{d_{\text{head}}}$
  (Symbol Ledger). Dividing by $\sqrt{d_{\text{head}}}$ restores unit variance — the exact normalizing constant,
  not a fudge. **Qwen3-8B [VP]: $d_{\text{head}}=128$ → $\text{std}(s)=\sqrt{128}=\mathbf{11.3137}$** [DER,
  constants §5]; logits routinely land in $[-34,+34]$ (±3σ) before scaling.
- *What that does to the softmax — carry the frozen arithmetic [DER, constants §9.2; use `[2.0,1.0,0.1]`, NOT
  `[2,1,0]` — D-08].* Canonical logits $z=[2.0,1.0,0.1]$: $e^z=[7.389056,2.718282,1.105171]$, $\sum=11.212509$,
  $\hat y=[\mathbf{0.659001},0.242433,0.098566]$ — a real blend, gradients to all three. Now the *same* relative
  preference at $\sqrt{d_{\text{head}}}$ scale — multiply by 11.31 → $[22.6,11.3,1.13]$: factor out the max →
  $A\approx[\mathbf{0.99999},\,\sim1.2\times10^{-5},\,\sim1.5\times10^{-10}]$ — **a one-hot vector in a trenchcoat.**
  Then the part that matters — the **gradient** `[THREAD:chain-rule]`: softmax Jacobian diagonal
  $\partial A_i/\partial s_i=A_i(1-A_i)$. Unscaled key 0: $0.659001\times0.340999=\mathbf{0.2247}$ (healthy);
  scaled-up: $\approx1.2\times10^{-5}$ — **~18,000× smaller**; in bf16 (≈8 mantissa bits, relative resolution
  $2^{-8}\approx0.004$) a gradient of $10^{-5}$ against activations of order 1 **rounds to zero**. Print the
  sentence (brief §5 F2b): *"Without the √d̲ₕₑₐ𝒹, attention doesn't get worse — it gets stuck. The softmax
  saturates, the Jacobian collapses, and $W_Q,W_K$ stop receiving gradient. The model can't learn what to look
  for, because looking has become a hard decision and hard decisions have no derivative."*
- *Two honest caveats (`.deepdive`):* (1) mean-0/var-1 holds only at init; after training $Q,K$ have whatever
  statistics training gave them, so $\sqrt{d_{\text{head}}}$ is no longer exactly right — which is why 2026 models
  add **QK-Norm** (an RMSNorm on $\mathbf{q},\mathbf{k}$ before the dot product) to re-impose the assumption every
  step. **Qwen3-8B has `q_norm`,`k_norm`, each size $d_{\text{head}}=128$ [VP]** — he'll see them in the state
  dict; a real design choice, not settled (Cohere Tiny Aya *dropped* QK-Norm) [V-sec]. (2) The scale is a
  **temperature** $\tau=\sqrt{d_{\text{head}}}$ — the same knob he sets in a sampler; connect explicitly.
- *F3 — multi-head (brief §5 F3).* Intuition: one head has one $W_Q,W_K$ → one notion of "relevant"; multi-head
  runs several relevance-detectors on different learned subspaces (one tracks grammar, one subject-verb agreement,
  one just copies the previous token). Callback to page 27: a head *is* a choice of projection; many heads = many
  simultaneous opinions about "similar." Math: $\text{head}_h=\text{Attention}(XW_Q^{(h)\top},XW_K^{(h)\top},XW_V^{(h)\top})$,
  $\text{MHA}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_H)\,W_O^\top$ (frozen notation, $W_O$ (out,in)).
  **Qwen3-8B shapes [VP]** (table, shape ribbon): $d=4096$, $H=32$, $d_{\text{head}}=128$ (note $32\times128=4096$
  is a *convention*, not a law), fused $W_Q\,(4096,4096)$, $\text{head}_h\,(S,128)$, concat $(S,4096)$,
  $W_O\,(4096,4096)$. **Implementation truth (`.box key`, brief §5 F3):** you do **not** have $H$ separate
  matrices — one $(4096,4096)$ `q_proj`, then `.view(B,S,32,128).transpose(1,2)`. **The heads are a reshape**;
  the math is $H$ independent attentions, the code is one matmul and a view — pre-empt the stall when he first
  reads `modeling_qwen3.py`. Why $W_O$ exists: concatenation stacks $H$ subspace-outputs that never interacted;
  $W_O$ mixes them back into the shared representation (and it's the head's *write port* onto the residual stream
  — seeds page 34).
- *What heads actually learn — be honest (`.deepdive`, [D] territory, brief §5 F3):* some heads are cleanly
  interpretable and reproducible (previous-token, positional, duplicate-token, **induction heads** — the
  mechanistic substrate of in-context learning); but most are **polysemantic**, many can be pruned with little
  loss, and **attention sinks are real** — many heads dump mass onto token 0/BOS regardless of content, a learned
  "no-op" because softmax rows must sum to 1 (this explains a visual artifact he *will* see in the demo — warn him
  first) [M for the taxonomy; induction-head result well-replicated].
- **Misconceptions (`.box warn`):** (a) *"√d̲ₕₑₐ𝒹 stops overflow"* → overflow isn't the issue (max-subtraction
  handles it); the issue is **gradient death via saturation** — the forward pass is fine without the scale, the
  *backward* pass is dead; fix with the Jacobian numbers. (b) *"why √ and not $d_{\text{head}}$?"* → we normalize a
  **std**, not a variance; dividing by 128 over-shrinks by 11.3× → logits std 0.088 → a uniform-average softmax,
  the *opposite* failure (both cliffs are on the demo). (c) *"$d_{\text{head}}$ is the model dimension"* → it's
  **per-head**: $d=4096$ but $d_{\text{head}}=128$, so the scale is $\sqrt{128}=11.31$, not $\sqrt{4096}=64$ (a
  common from-scratch bug). (d) *"more heads = more capacity"* → with $H\,d_{\text{head}}=d$, total attention
  params are **identical** regardless of $H$ (32×128 and 8×512 have the same count); it's a partition, not an
  addition — fix by computing both. (e) *"each head learns one linguistic function"* → polysemanticity/prunability;
  clean paper pictures are cherry-picked. (f) *"heads talk to each other inside the layer"* → completely
  independent until $W_O$.

**Interactive demo — "The Attention Heatmap You Drive" (S; `Heatmap` (viz.js) + `NN.attention`, brief Demo F-a —
the flagship):**
- Uses `NN.attention(Q,K,V,{temperature, causal})` which returns weights `A`, per-row `entropy`, and
  `gradient-health` (max softmax Jacobian) per the engine contract. Ship a fixed ~10-token sentence with **real
  pre-dumped embeddings** and per-head $W_Q,W_K,W_V$ (a small JSON blob) so the math is real, not random.
- Plot: an $S\times S$ heatmap of $A$ (rows = query tokens, cols = key tokens; sequential colormap; cell text;
  hover), and below it a bar chart of the clicked row.
- Controls: (1) head selector (dropdown 0..7, swaps pre-dumped $W$s); (2) **temperature/scale slider**
  $\tau\in[0.1\sqrt{d_{\text{head}}},\,10\sqrt{d_{\text{head}}}]$, **log**, with a **detent at
  $\tau=\sqrt{d_{\text{head}}}=11.31$**; (3) causal-mask toggle (its insight is *paid off on page 32* — keep the
  control here but point forward); (4) row selector.
- Live readouts (what make it a lesson): row **entropy** $H_i=-\sum_j A_{ij}\log A_{ij}$ in nats next to $\log S$
  (the uniform max); **gradient health** $=\max_{ij}A_{ij}(1-A_{ij})$ (turns red when tiny); **effective keys**
  $=\exp(H_i)$.
- Ahas in order (brief F-a): (1) drag $\tau$ far *below* the detent → heatmap snaps to a hard diagonal stripe,
  entropy→0, eff-keys→1, gradient health→~$10^{-5}$ **red** — *"I broke it by making it too confident."* (2) Drag
  far *above* → uniformly grey, entropy→$\log S$, eff-keys→$S$, every row identical, the layer became a constant
  function — *"I broke it by making it too humble; attention that attends to everything attends to nothing."* (3)
  Return to the detent → structured, mid-entropy, healthy — **the learner physically locates the valley between
  two cliffs, and $\sqrt{d_{\text{head}}}$ is the thing that puts him in it. The single best moment in the unit.**
  (4) Switch heads → genuinely different patterns from the same sentence; **call out the attention-sink head**
  parking mass on token 0.

**Code artifact (`.box try`): `code/heads_are_a_reshape.py`** — takes a real Qwen3 `q_proj` shape, shows that
`.view(B,S,H,d_head).transpose(1,2)` gives $H$ independent heads; computes $P_{\text{attn}}$ for (H=32,d̲=128) and
(H=8,d̲=512) and asserts they're equal; prints the softmax Jacobian for scaled vs unscaled logits.

**Quiz (7):**
1. Numeric: $\text{std}(\mathbf{q}\cdot\mathbf{k})$ at $d_{\text{head}}=128$ (num 11.3137, tol 0.01).
2. Numeric: softmax of the frozen logits $[2.0,1.0,0.1]$ — first component $\hat y_0$ (num 0.659001, tol 0.001).
3. Numeric: softmax Jacobian $A_0(1-A_0)$ for the *unscaled* frozen logits (num 0.2247, tol 0.005).
4. MCQ: removing $\sqrt{d_{\text{head}}}$ mainly hurts — *the backward pass (gradient death), the forward pass is
   fine* (correct) [distractor: "the forward pass overflows" = misconception (a)].
5. MCQ: attention params with $H=32,d_{\text{head}}=128$ vs $H=8,d_{\text{head}}=512$ — *identical* (correct)
   [distractor: "32 heads has 4× more" = misconception (d)].
6. MCQ: a head dumping most of its mass on token 0 is — *an attention sink (a learned no-op), not a bug* (correct).
7. MCQ: QK-Norm exists because — *training breaks the mean-0/var-1 assumption √d̲ₕₑₐ𝒹 relied on* (correct).

**Recurring-thread touchpoints:** `[THREAD:chain-rule beat …]` (softmax Jacobian collapse — same verb as page 28's
BPTT). `[THREAD:dot-product]` (heads = choosing directions, page 27). `[THREAD:Qwen3-8B]` ($\sqrt{128}$, `q_norm`/
`k_norm`). Cross-refs: → page 31 (self vs cross), → page 32 (causal toggle paid off), → page 34 (RMSNorm, $W_O$ as write port), → page 36
($P_{\text{attn}}=41{,}943{,}040$/block).

---

## 31 — self-vs-cross-attention.html
**Title:** "Attention III: Self vs Cross — The Diffusion Hinge" · **Part III** · (D-21: 31) · **Build: S** · **track-relevant: diffusion (both inherit)**

**⚠️ THE DIFFUSION HINGE. FLAG THIS HARD (brief §5 F4; D-14 #1).** "The single most important paragraph in the
entire trunk for the diffusion track." He uses ComfyUI, has typed prompts and watched images change, and has
never been told the mechanism. This page tells him. Mark `→ Diffusion` badges. **The LLM hinge — causal masking —
is the *next* page (32); this page stays bidirectional and hands the causal story to 32.**

**Learning objectives:**
1. State the *only* difference between self- and cross-attention: where $K,V$ come from.
2. Read text-to-image conditioning as one cross-attention matmul — "the image is the query, the text is the
   key/value."
3. Explain why $K,V$ can come from a *different-width* modality ($W_K,W_V$ absorb it) — the reason you can bolt a
   T5/Mistral/CLIP onto the same U-Net.
4. Locate CFG, IP-Adapter/ControlNet, MMDiT, and the 77-token CLIP bottleneck as consequences of where $K,V$
   enter; and know that the *other* $K,V$-routing story — causal masking, for time-ordered generation — is page 32.

**PREDICT (he has the deepest hook here):** *"When you type 'a red car' and the car comes out red, the prompt
reaches the image through attention. Is the image attending to the text, the text attending to the image, or are
they concatenated and attending jointly?"* (All three are 'right' for different model families — resolve with
cross-attn first, MMDiT as the 2026 twist.)

**Section outline:**
- *F4 — self vs cross (brief §5 F4).* The comparison table (self | cross): $Q$ from $X$ both times; $K,V$ from $X$
  (self) vs **from a second sequence $Y$** (cross); shapes $K,V:(S,d_{\text{head}})$ vs $(S_{\text{txt}},d_{\text{head}})$
  with $S_{\text{txt}}\ne S$ allowed; score matrix $(S,S)$ vs $(S,S_{\text{txt}})$; output $(S,d_{\text{head}})$
  **both** (still one row per query — page 29's fact #1). The formula (frozen notation, shape ribbon):
  $$\text{CrossAttn}(X,Y)=\text{softmax}\!\left(\frac{(XW_Q^\top)(YW_K^\top)^\top}{\sqrt{d_{\text{head}}}}\right)(YW_V^\top).$$
  Say it (`.box key`, brief §5 F4): *"In a text-to-image diffusion model, the image is the query and the text is
  the key/value. Every image patch asks 'which words are about me?' The word embeddings answer. When you type 'a
  red car' and the car comes out red, this matmul is why."*
- *Concrete shapes for a real system (brief §5 F4 — SDXL-class, [M], "verify against `diffusers` before
  shipping"):* $Y$ = CLIP text embeddings $(S_{\text{txt}}{=}77,\,2048)$ (77 = the CLIP context length every
  ComfyUI user has bumped into; 2048 = CLIP-L 768 + CLIP-G 1280); $X$ = flattened U-Net features e.g. $(1024,640)$;
  cross-attention score matrix $(1024,77)$ — **1024 image positions × 77 text tokens, each entry $A_{ij}$ = "how
  much does image-patch $i$ care about word $j$."** This matrix *is* what cross-attention-map / prompt-editing /
  regional-prompting nodes manipulate. **He has used those nodes; now he knows they write into $A$.**
- *Flag forward, explicitly (brief §5 F4, `→ Diffusion` badges):* CFG's target is this $Y$-path — run twice
  ($Y$=prompt, $Y$=∅) and extrapolate; **CFG only makes sense once you know $Y$ enters through $K,V$** (diffusion
  track, ~53–62). IP-Adapter adds a *second* cross-attention over image embeddings — "condition on anything you
  can embed" becomes obvious. **2026 update ([VP], the important one):** FLUX.1/FLUX.2 use **no separate
  cross-attention** — **MMDiT / joint attention** concatenates text + image tokens into one sequence and runs
  **self**-attention, so image↔text, image↔image, text↔text are all in one score matrix. FLUX.1: 19 double-stream
  + 38 single-stream blocks, ~12B params [VP]; FLUX.2: **32B, 8 double + 48 single, single Mistral-3 text encoder
  (≤512 tokens), 32-ch VAE, no bias, >80 GB unopt** [VP, constants §9.6 / hardware §4]. **Pedagogical payoff:
  cross-attention is a *special case* of self-attention over a concatenated sequence with a mask that blocks some
  quadrants.** Teach cross first, reveal MMDiT as "just concatenate and let self-attention sort it out." Note the
  tension worth flagging: FLUX.2 needing >80 GB *fits in his 128 GB unified memory where it wouldn't on a 24 GB
  card* — concrete and motivating (and the measured usable-memory caveat lives on page 36 / hardware §2). And the
  77-token CLIP bottleneck he's cursed at is a direct consequence of the old design; FLUX removing it, of the new.
- *The other $K,V$-routing story is causal masking (bridge to page 32, `→ LLM` badge, keep it to one paragraph).*
  Cross-attention routes $K,V$ from a *different sequence*; **causal masking routes attention within *one* sequence
  by time-order — position $i$ may see only $\le i$.** That is the LLM hinge and it gets its own page next (32):
  the additive $-\infty$ mask, "4096 signals per pass," and the generation-time caching payoff all live there. **The
  contrast to draw here at the diffusion side (`.box warn`, brief §5 F4/F5):** *diffusion image self-attention is
  **bidirectional/unmasked*** — causality is a property of time-ordered *generation*, and images aren't
  time-ordered. So the same `attn()` primitive is masked for an LLM and unmasked for a U-Net; the mask, not the
  mechanism, is what differs. **Draw this fork contrast explicitly** (it is the seam between the two tracks).
- **Misconceptions (`.box warn`, brief §5 F4):** (a) *"cross-attention is a different mechanism"* → same
  function, different arguments: `attn(Q=x,K=y,V=y)` vs `attn(Q=x,K=x,V=x)`; **the code is identical** — show both
  call sites. (b) *"the prompt is injected/mixed into the image"* → the image *queries* the prompt; it's a *pull*,
  content-addressed by the patch — which is why prompts are unreliable: *you don't control the query.* (c) *"$K,V$
  must match $Q$'s width"* → $K$ matches $Q$ in $d_{\text{head}}$; $V$ matches $K$ in length $S_{\text{txt}}$; $Y$'s
  native width (2048) is free — $W_K,W_V$ absorb it, **which is exactly why you can bolt a T5/Mistral/CLIP onto the
  same U-Net.** (d) *"diffusion U-Nets use causal masks"* → no; image self-attention is **bidirectional/unmasked**;
  causality is a property of time-ordered generation and images aren't time-ordered — the fork contrast above
  (causal masking itself is dissected on page 32).

**Interactive demo — the cross-attention (non-square) view (S; `Heatmap` + `NN.attention`, reusing the page-30
widget):**
- This page's insight: a **cross-attention view** with two token strips — an "image-patch" query strip of length 8
  and a "text" key strip of length 5, each with real small embeddings — rendering an $(8,5)$ **rectangular** $A$.
  The learner sees a **non-square** attention matrix and that the output still has 8 rows (one per query, page
  29's fact #1). Readout: the cross matrix is $(S_{\text{img}},S_{\text{txt}})$; changing $S_{\text{txt}}$ changes
  columns, never rows. Aha: drop a word from the "prompt" strip → a column vanishes, rows are unchanged, and the
  image patches re-weight over the survivors — *"the prompt is a set of keys the image queries; add or remove keys
  and the image just re-asks."* (The **causal-mask toggle** demo — triangle appears, per-row entropy drops — is on
  page 32, built from the same widget; keep the two insights on their own pages.)

**Code artifact (`.box try`): `code/self_and_cross.py`** — calls **one** function `F.scaled_dot_product_attention`
two ways: self (Q=K=V=x) and cross (Q=image, K=V=text of a *different* length); prints both shape ribbons to show
the code is identical and only the $K,V$ arguments (and thus the column count of $A$) change. (The causal call is
demonstrated on page 32's `code/causal_mask.py`.)

**Quiz (6):**
1. MCQ: the *only* difference between self- and cross-attention — *where $K,V$ come from* (correct).
2. MCQ: in text-to-image cross-attention, the query is — *the image; text is key/value* (correct) [distractor:
   "text is the query" = misconception (b)].
3. MCQ: you can attach a T5 (width 4096) or CLIP (width 2048) to the same U-Net because — *$W_K,W_V$ project $Y$'s
   width down to $d_{\text{head}}$* (correct) [distractor: "$K,V$ must match $Q$'s width" = misconception (c)].
4. Numeric/shape: image query strip length 8, text key strip length 5 → the cross-attention matrix $A$ has how
   many rows? (num 8, tol 0) — *one per query; the column count (5) tracks the text length.*
5. MCQ: FLUX's MMDiT relates to cross-attention as — *cross-attention is a special case of self-attention over a
   concatenated sequence* (correct).
6. MCQ (diagnosis): a colleague says "the diffusion U-Net must use a causal mask like the LLM does" — *false;
   image self-attention is bidirectional/unmasked, images aren't time-ordered* (correct) [distractor: "true, all
   transformers are causal" = misconception (d); the causal mechanism is page 32].

**Recurring-thread touchpoints:** `[THREAD:dot-product]` (the same score matrix, now rectangular).
`[THREAD:memory-budget]` seed (77-token limit; FLUX.2 >80 GB vs his 128 GB — paid off page 36). Cross-refs: → page
32 (causal masking — the LLM $K,V$-routing story), → page 35 (why decoder-only won; enc-dec moved to diffusion),
→ page 36 (his box fits FLUX.2), → diffusion track conditioning/CFG/IP-Adapter (53–62), → page 30 (the heatmap,
temperature).

---

## 32 — causal-masking.html
**Title:** "Causal Masking: 4096 Signals Per Pass — The LLM Hinge" · **Part III** · (D-21: 32) · **Build: S** · **track-relevant: LLM (the LLM hinge)**

**⚠️ THE LLM HINGE. FLAG THIS HARD (brief §5 F5; D-14).** Page 31 gave the diffusion hinge (where $K,V$ come from);
this page gives the LLM hinge: how *one* sequence is masked by time-order so that a single forward pass yields as
many training signals as it has tokens. This is the beat that makes LLM pretraining economically possible — give
it the landmark treatment. Mark `→ LLM` badges. **The demo is the causal-mask toggle** (the triangle appears; the
per-row entropy drops).

**Learning objectives:**
1. Read the causal mask as an *additive* $-\infty$ mask applied to the logits **before** softmax, and explain why
   additive-before-softmax (not multiplicative-after) is the only correct form.
2. Derive "one forward pass over length $S$ yields $S$ next-token training signals," compute the
   **4096-signals-per-pass** and the ~6.7× data-efficiency gap vs MLM — the economics that make LLM pretraining
   feasible (the LLM hinge).
3. Explain the generation payoff: position $i$'s output depends only on positions $\le i$, so past positions' $K,V$
   never change and generation can **cache** them (the KV cache, sized on page 37).
4. Distinguish *what each position may see* (the mask) from *when it is computed* (all at once, one matmul) — the
   exact distinction that makes a transformer ≠ an RNN at training, and the irony that it becomes sequential at
   decode (page 28).

**PREDICT (strong hook — he's trained/watched LLMs generate):** *"You have one 4,096-token document. Training a
next-token LLM on it, how many separate 'predict the next token' learning signals do you get from a **single**
forward pass — 1, about 15% of 4,096, or all 4,096?"* (All 4,096 — and that number is the whole reason LLM
pretraining is affordable. Resolve with the mask.)

**Section outline:**
- *Intuition (brief §5 F5, `→ LLM` badge).* If you train a model to predict the next token and let it *see* the
  next token, it learns the trivial cheat — copy the token to its right. The causal mask is a blindfold that keeps
  training honest: each position may attend only to itself and its predecessors. The non-obvious win is what that
  blindfold *buys*: because every position is now a legitimate "predict my successor" problem, **one sequence of
  length $S$ becomes $S$ training examples at once**, computed in a single parallel forward pass.
- *The mask (frozen notation; Symbol Ledger).* $M_{ij}=0$ if $j\le i$, $-\infty$ if $j>i$; added to the scaled
  scores **before** softmax: $A=\text{softmax}_{\text{row}}\!\big(QK^\top/\sqrt{d_{\text{head}}}+M\big)$. In
  practice $-\infty$ is `torch.finfo(dtype).min` ($\approx-3.39\times10^{38}$ in bf16), so $\exp(\cdot)\to0$ for the
  masked entries. $M$ is $(S,S)$, strictly upper-triangular in its $-\infty$ entries; row $i$ has $i{+}1$ live
  columns.
- **Why additive, not multiplicative (`.box warn`, the real bug people write).** Additive-then-softmax
  renormalizes the *surviving* entries so **each row still sums to 1**. Multiplying $A$ by a $0/1$ mask *after*
  softmax leaves early rows summing to $<1$ (row 0 keeps only its own weight, ~$1/S$ of the mass) and shrinks their
  output toward zero — a subtle, common, and destructive bug. The mask must go in **before** the softmax, as a
  logit, exactly because softmax's job is to renormalize over the survivors.
- **THE efficiency insight — 4096 signals per pass (`.box key`, brief §5 F5).** With a causal mask, position $i$'s
  output depends only on tokens $\le i$, so **one forward pass over length $S$ yields $S$ simultaneous next-token
  predictions**, each a valid loss term:
  $$J=-\frac1S\sum_{i=1}^{S}\log p_\theta\!\big(x_{i+1}\mid x_{\le i}\big)\qquad(\text{batch-averaged }J;\ \text{Symbol Ledger}).$$
  $S=4096$ → **4096 training signals from one pass** [DER]. This is the deal that makes LLM pretraining economical.
  Contrast MLM (BERT), which masks ~15% of tokens → ~0.15 signals/token vs **1.0** for causal LM: a **~6.7×
  data-efficiency gap per forward FLOP** [M for the 15%]. **This is the largest single piece of the honest "why
  decoder-only won" story (page 35).**
- **The generation payoff — why causality *is* the KV cache (`.box key`; expand — `[THREAD:memory-budget]`).**
  Turn the mask around: because position $i$ depends only on $\le i$, when you generate token-by-token the keys and
  values of all *earlier* positions are **frozen** — appending a new token never changes them. So you compute each
  position's $K,V$ **once** and **cache** them; every new token is a single query against the cached $K,V$, not a
  full recompute of the sequence. **The KV cache is not a bolt-on optimization; it is a direct consequence of
  causality** — caching is only *sound* because the mask guarantees the past can't change. (Page 37 sizes it:
  144 KiB/token for Qwen3-8B.) The flip side, honestly: at inference the transformer goes **sequential again**, one
  token at a time, and inherits **RNN economics** — bandwidth-bound decode (page 28's irony; roofline page 43).
- *Parallel training vs sequential inference (the distinction to nail).* At **training** all $S$ positions are
  computed in the **same matmul**; the mask constrains *what each position may see*, not *when it is computed*.
  That is the exact property that separates a transformer from an RNN (page 28): the RNN's sequential dependency is
  in the *computation*; the transformer's causality is only in the *visibility*. **At decode** — and only at decode
  — it does become sequential, because each generated token feeds the next.
- **Misconceptions (`.box warn`, brief §5 F5):** (a) *"causal masking makes the model slower / it's a limitation"*
  → it is the **enabling trick for parallel training**; without it you'd have RNN economics at *training* time too;
  fix with 4096-signals-per-pass. (b) *"the mask is applied to the output / to $A$ after softmax"* → **to the
  logits, before softmax, additively** (the row-sum bug above). (c) *"causal = the model reads left-to-right, one
  token at a time"* → at training **all positions are computed in one matmul**; the mask limits visibility, not
  timing; only at *inference* does it go sequential (the page-28 irony). (d) *"the KV cache is a separate
  optimization you add later"* → it is a **consequence** of causality; past $K,V$ are frozen by the mask, which is
  the only reason caching is correct. (e) *"diffusion U-Nets are causal too"* → no; image self-attention is
  bidirectional/unmasked (page 31's fork contrast) — causality is a property of time-ordered generation.

**Interactive demo — "The causal-mask toggle" (S; `Heatmap` (viz.js) + `NN.attention(Q,K,V,{causal})`, reusing the
page-30/31 widget — this is the page's centerpiece):**
- Same fixed ~10-token sentence and per-head pre-dumped $W$s as page 30, so the math is real. `NN.attention` takes
  `{causal}` and returns weights `A`, per-row `entropy`, and `gradient-health` per the engine contract.
- **The toggle (build the page around it):** flip **causal on** → the upper triangle of the $S\times S$ heatmap
  **blacks out** (the $-\infty$ entries → 0 after softmax) and **the surviving weights in each row visibly grow**
  as the row renormalizes over its $\le i$ survivors — *"masking isn't deletion; the row still sums to one; that is
  why the mask goes in before the softmax."* Makes the additive-vs-multiplicative box **felt.**
- **Per-row entropy readout (the quantitative beat):** show $H_i=-\sum_j A_{ij}\log A_{ij}$ per row and
  **effective keys** $=\exp(H_i)$. With causal on, **row 0 has entropy 0 / 1 effective key** (it can only see
  itself) and entropy climbs down the rows as each position gains more visible keys — *"the entropy per row drops
  under the mask, and it drops most at the top, because early tokens have almost nothing to look at."* Watch the
  entropy column fall the instant the triangle appears.
- **The signals counter (second readout):** a strip of the $S$ positions, each lit as a **valid next-token
  prediction**; a readout "**$S$ training signals from one pass**," with an MLM comparison line ($0.15S$). Slider
  $S$ (up to 4096) → the counter tracks it to 4096 while the MLM line lags at ~614.

**Code artifact (`.box try`): `code/causal_mask.py`** — (1) calls `F.scaled_dot_product_attention(..., is_causal=True)`
and asserts the weight matrix has **strictly-lower-triangular support and every row sums to 1**; (2) reproduces the
**multiply-after-softmax bug** and prints early rows summing to $<1$ so he sees the failure; (3) from **one** forward
pass over a length-$S$ sequence, computes the full length-$S$ per-position cross-entropy vector — **$S$ loss terms,
one pass** — and prints $S$; (4) a tiny caching demo: generate 3 tokens and assert the cached $K,V$ for earlier
positions are **bit-identical** across steps (causality → the cache is sound).

**Quiz (7):**
1. Numeric: a causal pass over $S=2048$ tokens yields how many next-token training signals? (num 2048, tol 0).
2. Numeric: $S=4096$ → training signals from one forward pass (num 4096, tol 0).
3. MCQ: the causal mask is applied — *to the logits, before softmax, additively ($-\infty$)* (correct) [distractors
   = misconception (b): "to $A$ after softmax" / "multiplicatively"].
4. MCQ (diagnosis): early tokens' outputs are near-zero and their attention rows sum to $<1$ — *you multiplied a
   $0/1$ mask after softmax instead of adding $-\infty$ before it* (correct).
5. MCQ: causal masking enables the KV cache because — *position $i$ depends only on $\le i$, so past $K,V$ never
   change and can be cached* (correct) [distractor: "it makes the attention matmul cheaper"].
6. MCQ: at *training* time, causal masking means — *all positions are computed in one matmul; the mask limits what
   each may see, not when it is computed* (correct) [distractor: "the model processes tokens strictly one at a
   time" = misconception (c)].
7. MCQ: MLM (BERT, ~15% masked) vs causal LM in signals per forward pass — *causal LM gets ~6.7× more loss signals*
   (correct) [distractor: "they're equal; both see every token"].

**Recurring-thread touchpoints:** `[THREAD:memory-budget beat …]` (causal ⇒ past $K,V$ frozen ⇒ **cacheable**; page
37 sizes the cache). `[THREAD:chain-rule]` implicit (the softmax renormalization the mask relies on — page 30's
Jacobian). `[THREAD:Qwen3-8B]` (a 4096-token training slice; decode goes bandwidth-bound). Cross-refs: → page 31
(self vs cross — the *other* $K,V$-routing story), → page 37 (KV cache — sizes what causality lets you cache),
→ page 35 (why decoder-only won; signal density leads), → page 28 (decode = RNN economics), → roofline page 43
(bandwidth-bound decode).

---

## 33 — positional-encoding-rope.html
**Title:** "Positional Encoding and RoPE" · **Part III** · (D-21: 33) · **Build: O** (unit-circle RoPE panel) · **trig payoff #2**

**Order note (brief §6):** must come *after* attention — the motivation is a property of attention he has now
seen. **This is the second place high-school trig is non-negotiably load-bearing (the first was page 27's cosine).
Drop to foundations here** (brief §6).

**Learning objectives:**
1. Prove (three lines) that attention + position-free FFN is **permutation-equivariant** — "dog bites man" and
   "man bites dog" are the same input.
2. Contrast the three encodings (sinusoidal absolute / learned absolute / RoPE) and why RoPE won.
3. Derive the RoPE identity $R_i^\top R_j = R_{j-i}$ and see that the attention score depends only on the
   *relative* position $(j-i)$ — relative position for free, from the geometry, without ever computing a
   difference.
4. Read `rope_theta` in a config: raising the base stretches the low-frequency end so the slowest pair doesn't
   alias inside the context window.

**PREDICT:** *"Shuffle the words in a sentence and feed them to a transformer with no positional encoding. Does the
output change, come out shuffled the same way, or stay identical per-token?"* (It's equivariant — same outputs,
shuffled the same way; "man bites dog" = "dog bites man".) Resolve with the permutation proof.

**Section outline:**
- *Intuition (open with the problem, brief §6):* attention has no idea what order anything is in — it's a set
  operation in a sequence costume; positional encoding is how we tell it order.
- *Prove the problem (don't assert it, brief §6).* Let $P$ be an $S\times S$ permutation matrix:
  $\text{Attention}(PX)=P\cdot\text{Attention}(X)$. Three-line sketch (Symbol Ledger): $Q'=PQ$, $K'=PK$, $V'=PV$;
  $S'=Q'K'^\top=PSP^\top$; softmax is row-wise and $P^\top$ permutes columns consistently so $A'=PAP^\top$; then
  $A'V'=PAP^\top PV=PAV$ (since $P^\top P=I$). ∎ **Attention is permutation-equivariant, and the position-wise FFN
  is too, so the whole transformer minus positional encoding cannot distinguish "dog bites man" from "man bites
  dog."** Contrast: a CNN gets position free from kernel geometry, an RNN free from loop order; **the transformer
  threw away both and must buy position back** — a cost of the weak-inductive-bias bet (page 25). Nice closure.
- *The three approaches (brief §6).* **(1) Sinusoidal absolute (2017):** $PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$,
  $PE_{(pos,2i+1)}=\cos(\cdot)$, added to $\mathbf{x}_{pos}$ — a continuous binary counter (fast dims = ones digit,
  slow dims = millions digit). Flaws: it's *absolute* (the model usually wants "you're 3 behind me," which must be
  inferred from two absolutes) and *added* (competes with meaning for vector space). **(2) Learned absolute
  (BERT/GPT-2):** a lookup table $(S_{\max},d)$ — simple, works, but **cannot extrapolate at all**: position 2049
  in a model trained to 2048 is a literally untrained random vector (why GPT-2 couldn't be context-extended and
  Llama can).
- **(3) RoPE — 2026's default, essentially universal [VP] `[THREAD:dot-product]`.** Intuition (`.box key`, brief
  §6): *"RoPE doesn't add position — it rotates the vector by an angle proportional to its position. Two tokens'
  dot product then depends only on the angle between them, which is the difference of their positions. You get
  relative position for free, from the geometry, without ever computing a difference."* Mechanism (frozen
  notation; **positions $i,j$ per notation §3, not $m,n$**; Translation note: *"the RoPE papers write $m,n$; these
  are sequence positions."*): split $\mathbf{q}\in\mathbb{R}^{d_{\text{head}}}$ into $d_{\text{head}}/2$ 2-D pairs,
  rotate pair $p$ by angle $i\,\varphi_p$ where $\varphi_p=b^{-2p/d_{\text{head}}}$:
  $$\begin{pmatrix}q'_{2p}\\q'_{2p+1}\end{pmatrix}=\begin{pmatrix}\cos i\varphi_p & -\sin i\varphi_p\\ \sin i\varphi_p & \cos i\varphi_p\end{pmatrix}\begin{pmatrix}q_{2p}\\q_{2p+1}\end{pmatrix}$$
  (Symbol Ledger; $b$ = `rope_theta` = **1,000,000 for Qwen3-8B [VP]**; $d_{\text{head}}=128$ → **64 rotation
  pairs**; applied to $\mathbf{q},\mathbf{k}$ **only**, never $\mathbf{v}$, never the residual stream — the elegant
  bit). **The property that makes it work (write it out — the whole justification):** with $R_i$ the block-diagonal
  rotation for position $i$, rotations satisfy $R_i^\top R_j=R_{j-i}$, so
  $$\langle R_i\mathbf{q},R_j\mathbf{k}\rangle=\mathbf{q}^\top R_i^\top R_j\mathbf{k}=\mathbf{q}^\top R_{j-i}\mathbf{k}.$$
  **The score depends on $(j-i)$ — the relative position — and nothing else positional.** You inject **absolute**
  position ($i,j$ independently, cheaply, at each layer's Q/K) and get **relative** ($j-i$) out for free because
  rotations compose; **you never compute $j-i$** — the algebra does it. Rotation is norm-preserving
  ($\|R_i\mathbf{q}\|=\|\mathbf{q}\|$), so position **doesn't compete with meaning for magnitude** (contrast
  sinusoidal, which adds and does compete). *This is beautiful; tell him it's beautiful.* And **this is the 2×2
  rotation matrix he already knows — show it's the same one.**
- *Frequencies — carry the arithmetic [DER, brief §6]:* with $d_{\text{head}}=128$, $b=10{,}000$ (classic):
  fastest pair $\varphi_0=1$ rad/token → wavelength $2\pi\approx6.28$ tokens; slowest $\varphi_{63}=10000^{-126/128}\approx1.156\times10^{-4}$
  → wavelength $\approx54{,}350$ tokens. With $b=1{,}000{,}000$ (**Qwen3-8B's value [VP]**): fastest still 6.28
  tokens; slowest $\varphi_{63}=10^{-5.906}\approx1.24\times10^{-6}$ → wavelength $\approx\mathbf{5.06\times10^{6}}$
  tokens. **Insight:** raising the base stretches the low-frequency end; the slowest pair must not complete a full
  rotation inside the context window or positions **alias** (position 0 and position $\lambda$ become
  indistinguishable in that pair). $b=10^4$ gives $\lambda_{\max}\approx54$k (fine at 2k, marginal at 128k); $b=10^6$
  gives 5M — comfortable. Qwen3-8B's `max_position_embeddings=40960` [VP] sits far inside 5.06M. **This is what
  "increase `rope_theta` for long context" means — a config line he's seen and never understood.**
- *Long-context extension (`.deepdive`, [V-sec], brief §6):* `rope_scaling` (Qwen3-8B ships `null` [VP] — natively
  40,960). PI (squash positions — hurts local resolution); NTK-aware (raise the base — stretch lows, leave highs);
  YaRN (per-band: interpolate low-freq, leave high-freq, + temperature — ~10× fewer tokens, ~2.5× fewer steps for
  ~128k); LongRoPE / MrRoPE (radix-conversion view). Unifying idea: *all are the same move — rescale the frequency
  spectrum so angles at position 100k look like angles seen in training.*
- *Implementation gotcha (`.box warn`, [M], brief §6):* HuggingFace's `rotate_half` pairs
  $(0,d_{\text{head}}/2),(1,d_{\text{head}}/2{+}1),\dots$ — **splits the vector in half and pairs across**, not
  adjacent $(0,1),(2,3)$ as the paper notation implies. Equivalent up to a dimension permutation the learned
  weights absorb, but mixing conventions with pretrained weights makes garbage that looks like a training bug —
  costs people days.
- **Misconceptions (`.box warn`, brief §6):** (a) *"positional encoding tells the model the order"* → it makes
  order *recoverable*; nothing forces the model to use it (some heads ignore position). (b) *"RoPE is added to the
  input embeddings"* → sinusoidal/learned are added once at the bottom; **RoPE is applied to $Q,K$ only, inside
  every attention layer, every time, never to $V$ or the residual stream** — show the code: `q,k = apply_rope(q,k,…)`
  sits *between* projection and matmul. (c) *"RoPE gives unlimited context"* → it *extrapolates* better than
  learned but **degrades before it fails**; hence the whole YaRN/NTK industry. (d) *"decoder-only models need
  positional encoding"* ([D], genuinely interesting) → causal masking alone breaks permutation-equivariance (token
  $i$ sees exactly $i$ predecessors, so it can count); **"NoPE" decoder-only models do learn position and can
  work** [M] — everyone ships RoPE anyway because NoPE is worse at long context in most reports. Include it — "the
  thing everyone says is necessary isn't strictly necessary" builds real understanding.

**Interactive demo — "The RoPE Unit Circle" (O; custom canvas, brief Demo I, panel 2 — the money panel):**
- Panel A (sinusoidal, S sub-widget via `Plot`/heatmap): a $128\times64$ heatmap of $PE_{pos,i}$ with a base
  slider ($10^3$–$10^7$, log). Aha: the diagonal zebra stretches as base rises; right columns (low freq) barely
  change over 128 tokens — *"those do the long-range work; if your context exceeds their wavelength they wrap and
  lie."*
- Panel B (RoPE, the money panel; custom canvas unit circle): draw $\mathbf{q}$ at position $i$ (one selected
  frequency pair) as an arrow at angle $\phi_q+i\varphi_p$ and $\mathbf{k}$ at position $j$ at $\phi_k+j\varphi_p$.
  Controls: sliders $i,j\in[0,200]$, pair picker $p\in\{0..63\}$, base $b$. JS: $\varphi_p=b^{-2p/128}$; **display
  the dot product live, and, separately, the quantity $\mathbf{q}^\top R_{j-i}\mathbf{k}$.** **THE insight (engineer
  for it):** drag $i$ and $j$ **together** keeping $j-i=5$ fixed — both arrows spin, **the dot product does not
  move at all**; release one and it changes. *"The score only ever knew about the gap. It never knew where you
  were."* Show the two readouts agreeing to ~8 decimals as he drags — the identity $R_i^\top R_j=R_{j-i}$ verified
  live, by his own hand. ~40 lines of JS.
- Panel C (aliasing, `Plot`): $\cos(i\,\varphi_{63})$ for $i\in[0,200{,}000]$ at $b=10^4$ vs $b=10^6$, with a
  draggable "your context length" vline. Aha: at $b=10^4$ the curve completes multiple cycles inside 200k
  (position 0 ≡ position 54,350 in this dim); at $b=10^6$ it hasn't finished a quarter turn — *"that's what
  `rope_theta: 1000000` in the config buys you."*

**Code artifact (`.box try`): `code/rope_identity.py`** — implements $R_i$, verifies `allclose(⟨R_i q, R_j k⟩,
qᵀ R_{j-i} k)` across random $i,j$; then applies HF `rotate_half` vs the adjacent-pair convention to the *same*
pretrained q_proj slice and shows one matches the checkpoint and the other produces garbage.

**Quiz (7):**
1. MCQ: a transformer with no positional encoding is — *permutation-equivariant (can't tell "dog bites man" from
   "man bites dog")* (correct).
2. MCQ: RoPE is applied to — *$Q$ and $K$ only, inside every attention layer* (correct) [distractors: "input
   embeddings once" / "Q,K,V" = misconception (b)].
3. Numeric: Qwen3-8B's RoPE base `rope_theta` (num 1000000, tol 0).
4. Numeric: number of RoPE rotation pairs for $d_{\text{head}}=128$ (num 64, tol 0).
5. MCQ: the RoPE score $\langle R_i q, R_j k\rangle$ depends on — *the relative position $j-i$ only* (correct).
6. MCQ: raising `rope_theta` from $10^4$ to $10^6$ — *stretches the low-frequency wavelengths so they don't alias
   inside a long context* (correct) [distractor: "makes the fast pairs faster"].
7. MCQ (diagnosis): your from-scratch RoPE with a HF checkpoint outputs garbage that looks like a training bug —
   *`rotate_half` pairing convention mismatch (split-half vs adjacent)* (correct).

**Recurring-thread touchpoints:** `[THREAD:dot-product beat …]` (the score is a rotated dot product depending only
on the angle between q and k). `[THREAD:chain-rule]` (RoPE never *computes* a difference — the algebra does it).
`[THREAD:Qwen3-8B]` (`rope_theta=1e6`, 40,960 context, 64 pairs). Cross-refs: → page 35 (RoPE has zero params →
param count), ← page 27 (cosine/trig), → diffusion track (t-embedding is a *different* sinusoidal use, D-14).

---

## 34 — residuals-and-normalization.html
**Title:** "Residual Connections and Normalization" · **Part III** · (D-21: 34) · **Build: S**

**Ownership note (D-14):** the *why* of gradient flow and the *why* of normalization's axis choice are trunk
mechanisms introduced earlier (Parts I–II ≈ the vanishing-gradient and normalization pages); **this page owns the
*where*** — the residual stream as a bus, where the `+x` sits in the block, pre-norm placement, RMSNorm's config
line. Reference the earlier mechanism pages rather than re-deriving from zero, but include the layer-count
arithmetic (it's the architecture payoff).

**Learning objectives:**
1. Read a residual connection as "the layer computes only *what to change*," with a default behavior of "do
   nothing" — and see the backward-pass mechanism: $\partial y/\partial x = I + \partial F/\partial x$, an
   $I$-term nothing can shrink.
2. Install the **residual stream** mental model (a shared $d=4096$-wide bus every layer reads from and adds to) —
   the single most reusable model in the course.
3. Explain pre-norm vs post-norm and why every 2026 model is pre-norm; read RMSNorm and its `rms_norm_eps` config
   line.
4. Compute the layer-count arithmetic ($0.9^{36}=0.0225$ vs $\approx1$) that makes depth legal.

**PREDICT (he saw the LSTM highway on page 28):** *"A 36-layer stack, each layer's Jacobian norm ≈0.9. Without a
skip connection, how much of the layer-36 gradient reaches layer 0 — about 90%, 22%, or 2%?"* (2.25%.) Resolve;
then show the residual restores it to ≈100%.

**Section outline:**
- *Intuition (brief §7):* a residual says "here's the input; your job is only *what to change about it*"; the
  default becomes "do nothing," which a 100-layer stack desperately needs to be able to do. Second, the one that
  names the section: *the gradient highway — the `+x` gives the gradient a road from the loss straight back to
  layer 1 that doesn't pass through a single weight matrix; nothing on that road can shrink it.*
- *The math (brief §7) `[THREAD:chain-rule]`.* $\mathbf{y}=\mathbf{x}+F(\mathbf{x};\theta)$ ($\mathbf{x}$ must have
  the same shape as $\mathbf{y}$ — **this is why $d$ is constant through the whole stack; say it, learners wonder**).
  $\partial\mathbf{y}/\partial\mathbf{x}=I+\partial F/\partial\mathbf{x}$; through $L$ layers
  $\partial\mathcal{L}/\partial\mathbf{x}_0=(\partial\mathcal{L}/\partial\mathbf{x}_L)\prod_{\ell=1}^{L}(I+\partial F_\ell/\partial\mathbf{x}_{\ell-1})$
  (Symbol Ledger). Expand and note the product contains $I\cdot I\cdots I=I$ — **an unattenuated path from loss to
  layer 0** — vs page-28's RNN product $\prod\gamma$ with $\gamma\le1$. **This is the same verb (the chain rule) and
  the same fix (a near-identity Jacobian) as the LSTM cell state on page 28 — tag it.**
- *Worked comparison (the number that sells it) [DER, constants §9.3]:* no residual, $\gamma=0.9$, $L=36$ (Qwen3-8B
  depth [VP]): $0.9^{36}=\mathbf{0.0225}$ — 2.25% of the layer-36 gradient reaches layer 0. With residual and a
  small learned update ($\|\partial F/\partial x\|\approx0.1$, mean ≈0): the per-layer multiplier is $\approx1+\epsilon$,
  the product stays $O(1)\approx\mathbf{1.0}$. **44× more gradient at $L=36$.** At $L=78$ (GLM-5's depth [V-sec]):
  $0.9^{78}=2.6\times10^{-4}$ vs ≈1 → **~3,800×.** *Residual connections aren't an optimization — they are the
  thing that makes depth legal.*
- *The residual stream (install it — brief §7, "the single most reusable mental model in the course").* Rewrite
  the stack: $\mathbf{x}_L=\mathbf{x}_0+\sum_{\ell=1}^{L}F_\ell(\mathbf{x}_{\ell-1})$ (Symbol Ledger). The model
  (`.box key`): *"The residual stream is a shared bus, $d=4096$ wide, running the length of the model. Every layer
  reads from it, computes a contribution, and adds to it. Nothing is overwritten; everything accumulates. The
  model isn't a pipeline that transforms — it's a whiteboard 36 committees write on in sequence."* Cash it out
  (flag forward): why LoRA works (a small extra writer on the bus → Part IV); why layers can be pruned/skipped [M];
  why activation steering works (add a vector to the bus → page 27's linear-representation hypothesis); why $W_O$
  matters (it's a head's *write port* — page 30); why $d$ is constant (it's a bus).
- *Pre-norm vs post-norm (brief §7).* Post-LN (2017): $\mathbf{y}=\text{LN}(\mathbf{x}+F(\mathbf{x}))$ — puts a
  norm *on* the residual path; the highway is no longer clean; needs LR warmup, unstable past ~12 layers [M].
  Pre-LN (since ~2020): $\mathbf{y}=\mathbf{x}+F(\text{LN}(\mathbf{x}))$ — residual path stays `x + something`,
  trains stably at 36/78/100+ layers. **Every 2026 model is pre-norm** [V-sec] — a *settled* question. **RMSNorm,
  not LayerNorm:** $\text{RMS}(\mathbf{x})=\sqrt{\frac1d\sum_i x_i^2+\varepsilon}$,
  $\text{RMSNorm}(\mathbf{x})=\mathbf{g}\odot\mathbf{x}/\text{RMS}(\mathbf{x})$, learned gain $\mathbf{g}\in\mathbb{R}^d$,
  **no bias** — drops mean-centering, keeps scale, cheaper, works as well. **Qwen3-8B: `rms_norm_eps = 1e-6` [VP]**
  — that $\varepsilon$ is right there in the config; he should point at it in the formula. (`.deepdive`, [V-sec]:
  depth-scaled RMSNorm in GLM-5/Trinity; QK-Norm (page 30) is an RMSNorm on $\mathbf{q},\mathbf{k}$; Cohere Tiny
  Aya uses **parallel blocks** — attention and FFN both read $\mathbf{x}$ and both write the stream — which the
  bus model makes instantly comprehensible.)
- **Misconceptions (`.box warn`, brief §7):** (a) *"residuals help the network remember the input"* → the
  forward-pass story is secondary; **the mechanism is the backward pass** — the $I$-term nothing can shrink; fix
  with $0.9^{36}=0.0225$ vs ≈1. (b) *"residuals let you make networks arbitrarily deep"* → they make deep networks
  *trainable*, not *better*; returns diminish (GLM-5 *cut* depth 92→78 while widening $d$ 5120→6144 vs GLM-4.7
  [V-sec] — a 2026 lab deciding depth overshot). (c) *"`x+F(x)` breaks if $F$ changes shape"* → yes, which is
  *why* $d$ is constant; CNNs need a 1×1 conv on the skip to match shapes, transformers never do.

**Interactive demo — "The gradient highway" (S; `Plot`, log-y; and optional `NetGraph` re-tint):**
- Sliders: per-layer Jacobian norm $\gamma$ (0.5–1.2), number of layers $L$ (1–120), residual on/off toggle, and
  (residual-on) the learned-update norm $\|\partial F/\partial x\|$ (0–0.5). Plot the gradient-reaching-layer-0
  ratio vs $L$ on log-y: no-residual trace $\gamma^L$; residual trace staying $O(1)$. Detents at $L=36$ (Qwen3) and
  $L=78$ (GLM-5). Readout: the ratio and the fold-improvement via `eng()`.
- Aha: toggle residual off at $L=36$ → the ratio dives to 0.0225 (red); on → snaps to ≈1; drag $L$ to 78 and watch
  the no-residual trace hit ~$10^{-4}$ while the residual trace barely moves. *"The deeper you go, the more the
  residual is doing."*

**Code artifact (`.box try`): `code/residual_stream.py`** — builds a 36-block toy stack with and without skips,
initializes each block's Jacobian to norm ~0.9, and prints the layer-0 gradient magnitude for both; then reads
Qwen3-8B's `rms_norm_eps` from a config and evaluates RMSNorm on a real hidden vector.

**Quiz (6):**
1. Numeric: $0.9^{36}$ (num 0.0225, tol 0.001).
2. MCQ: the primary mechanism of a residual connection is — *the backward pass: an $I$-term nothing can shrink*
   (correct) [distractor: "it remembers the input" = misconception (a)].
3. MCQ: why is $d$ constant through the whole transformer — *the residual bus requires matching shapes to add*
   (correct).
4. MCQ: every 2026 model uses — *pre-norm (norm inside the sublayer, residual path clean)* (correct).
5. Numeric: Qwen3-8B's `rms_norm_eps` (num 1e-6, tol 0).
6. MCQ (diagnosis): your 40-layer from-scratch transformer won't train without a warmup and diverges — *you used
   post-norm; switch to pre-norm* (correct).

**Recurring-thread touchpoints:** `[THREAD:chain-rule beat …]` (the $I$-term; same verb as page 28/30).
`[THREAD:Qwen3-8B]` ($L=36$, `rms_norm_eps`). Cross-refs: → page 35 (assemble the block), → LoRA (Part IV, "a small
writer on the bus"), ← page 28 (the LSTM highway).

---

## 35 — the-transformer-block.html
**Title:** "The Transformer Block, Assembled — and Why Decoder-Only Won" · **Part III** · (D-21: 35) · **Build: S** (TensorViz)

**Pedagogical instruction (brief §8):** **build the block on the page, one line at a time**, with the learner able
to answer "why is this line here?" from what he's already learned. **Do not show the finished block then explain
it. Build it.** This is the assembly page (J) plus encoder/decoder (K); D-14 keeps the transformer block in the
trunk (splitting it post-fork would duplicate ~6 pages and guarantee drift).

**Learning objectives:**
1. Assemble the 2026 block (pre-norm, RMSNorm, GQA, QK-Norm, RoPE, SwiGLU, residuals) and justify every line.
2. Read the SwiGLU FFN and explain why $d_{\text{ff}}=3d$ (not $4d$) — a parameter-budget correction for the third
   matrix — and why the FFN is 78% of the block.
3. Compare encoder-only / decoder-only / encoder-decoder and give the honest four-part answer for why decoder-only
   won.
4. See that encoder-decoder didn't die — it moved to diffusion (a text-to-image model *is* an encoder-decoder).

**PREDICT:** *"In one transformer block, where do most of the *parameters* live — in attention (the famous part)
or the feed-forward network?"* (The FFN, 78% — most people guess attention.) Resolve with the SwiGLU count.

**Section outline:**
- *The 2026 block, assembled as PyTorch pseudocode (brief §8; present in frozen row-batched form, $W$ (out,in), and
  annotate each line with the section that justifies it).* Ship the block with the six-line "why is this here"
  self-check (an explicit exercise — the best test of whether pages 30/33/34 landed):
  1. `rmsnorm_1` before, not after → page 34 pre-norm; keeps the highway clean.
  2. $W_K$ narrower than $W_Q$ → page 37 GQA; 8 KV heads not 32; 4× KV saving.
  3. RoPE on `q,k` not `v` → page 33; the rotation identity only pays off inside a dot product; $V$ is never
     dot-producted.
  4. `+ x` twice → page 34; two sublayers, two clean highways.
  5. `Wo` → page 30; heads are independent until this matmul (the write port).
  6. `q_norm`/`k_norm` → page 30 QK-Norm; re-impose the unit-variance $\sqrt{d_{\text{head}}}$ assumed.
  Shape ribbon on every line: `x (B,S,d=4096)` → `q (B,S,H*d̲)` → view `(B,H,S,d̲=128)` → SDPA `(B,H,S,d̲)` →
  `reshape (B,S,4096)` → `x + a@Wo.T (B,S,4096)`.
- *The FFN, and why it's bigger than you'd guess (brief §8).* Intuition: attention *moves* information between
  positions; the FFN *thinks* about it, applied to each position independently (4096 in, 4096 out, same weights
  everywhere, no communication). Classic (2017): $\text{FFN}=W_2\,\text{ReLU}(W_1\mathbf{x}+b_1)+b_2$,
  $d_{\text{ff}}=4d$, two matrices. **2026 SwiGLU (three matrices):**
  $\text{FFN}(\mathbf{x})=W_{\text{down}}(\text{SiLU}(W_{\text{gate}}\mathbf{x})\odot(W_{\text{up}}\mathbf{x}))$,
  $\text{SiLU}(z)=z\,\sigma(z)$ (Symbol Ledger; SiLU written by name — a diffusion-track page might reuse SiLU, and
  $\sigma$-with-argument is fine in the trunk, notation §5). **No biases** (Qwen3 `attention_bias: false` [VP];
  FLUX.2 eliminates bias throughout [VP]) — with norm layers carrying a learned gain, biases are redundant [M].
  **Qwen3-8B [VP]: $d=4096$, $d_{\text{ff}}=12288=3d$ — not $4d$.** Say why: SwiGLU uses 3 matrices instead of 2,
  so to keep params constant you shrink $d_{\text{ff}}$ by 2/3: $\frac23\times4d=\frac83d\approx2.67d$; Qwen3 rounds
  up to exactly $3d$. **The "why 12288 not 16384" question — real, and now answered: a parameter-budget correction
  for the third matrix** (⚠️ this is the corrected framing; the retired Llama-3.1 example used $d_{\text{ff}}=14336=3.5d$
  — D-01; do not use it). `hidden_act: "silu"` [VP], right there in the config. Gating intuition (`.deepdive`):
  $W_{\text{up}}\mathbf{x}$ = content, $\text{SiLU}(W_{\text{gate}}\mathbf{x})$ = a per-dimension volume knob ≈[0,1]
  → "compute this feature, but only let it through if that condition holds" — multiplicative interaction a ReLU MLP
  can only approximate [M]. **The parameter shock (`.box key`):** FFN $=3\times4096\times12288=150{,}994{,}944$;
  attention $=41{,}943{,}040$; **the FFN is 78% of the block** [DER, constants §1.3 — 78.26%; **not** the retired
  "~70%"]. *Attention gets all the attention; the FFN holds most of the weights* — sets up MoE (page 38, Part IV): sparsify
  the part that's 78% of it.
- *Encoder / decoder / encoder-decoder (brief §9).* The comparison table (encoder-only | decoder-only |
  enc-decoder): attention bidirectional | causal | enc bidir + dec causal+cross; objective MLM (~15%) | next-token
  (100%) | denoising; signals/token ~0.15 | **1.0** | ~0.15–1; 2026 status: **alive in RAG** | **won** | **alive in
  diffusion**. **Why decoder-only won — the honest four-part answer (brief §9; the single-cause story is the
  misconception):** (1) **signal density** (page 32) — causal LM gets a loss term from *every* token, MLM from
  ~15% → ~6.7× more signal per forward FLOP; when compute is the bottleneck and data is the internet, this
  dominates and is rarely stated first. (2) **task universality** — every task casts as text-completion;
  in-context learning falls out for free and *wasn't designed in*. (3) **simplicity → scalability** — one stack,
  one objective, one attention pattern; fewer things to tune at 1000-GPU scale. (4) **KV-cache economics** — causal
  masking means a token's K/V never change once computed, so you cache forever; bidirectional attention
  invalidates every previous token when you add one — no incremental decoding (page 37 KV cache; causal masking on
  page 32 is what makes the cache sound). **Be honest about
  what was lost:** bidirectional encoders are still better at *representation* (a token can depend on what comes
  after) — **which is exactly why 2026 RAG uses dedicated bidirectional contrastively-trained embedders, not
  decoder hidden states** (page 27 anisotropy; flag forward to RAG). Decoder-only won at *generation*, not
  *representation*.
- *The fork beat (brief §9, end the page here — D-17's convergence setup):* encoder-decoder isn't dead — **it moved
  to diffusion.** A text-to-image model *is* an encoder-decoder: a text encoder (CLIP / T5-XXL / **Mistral-3 in
  FLUX.2 [VP]**) produces $K,V$; the U-Net/DiT is the decoder that cross-attends to them (page 31). *He should see
  that the architecture he was told "lost" is the one he's been running in ComfyUI all along.* (This foreshadows the
  page-38/fork reveal (Part IV opener) and D-17's rejoining beat in the diffusion track.)
- **Misconceptions (`.box warn`):** (a) *"attention holds most of a block's parameters"* → the FFN does (78%); fix
  by computing both subtotals. (b) *"$d_{\text{ff}}=4d$"* → $3d$ for SwiGLU (budget correction for the third
  matrix); Qwen3's 12288 = 3×4096. (c) *"decoder-only won because it's more elegant / smarter"* → four concrete
  reasons, signal density first; not elegance. (d) *"encoder-decoder is obsolete"* → it's what he runs in ComfyUI.

**Interactive demo — "Build-a-block + the matmul shapes" (S; `TensorViz` for the matmuls + a stepping control):**
- A "reveal the block one line at a time" stepper: each click adds a line to the pseudocode and lights up the
  corresponding `TensorViz` matmul animation (e.g. `x @ Wq.T`, the head reshape, `q@k.T`, `a@Wo.T`, the SwiGLU
  three-matmul) with the shape ribbon updating live (TensorViz animates the matmuls per the engine contract). At
  each step, show the "why is this line here?" answer. Readout: running parameter subtotal for the block, landing
  on **192,946,432** (page 36's number).
- Aha: when the FFN's three matmuls appear, the parameter subtotal **jumps by 150,994,944** in one step — the FFN's
  78% made visible as a single leap.

**Code artifact (`.box try`): `code/build_block.py`** — a ~40-line `TransformerBlock` (RMSNorm, GQA q/k/v with
`repeat_kv`, `apply_rope`, `F.scaled_dot_product_attention(is_causal=True)`, `Wo`, SwiGLU), no elided imports;
`sum(p.numel())` asserts **192,946,432** and prints the FFN share **78.26%**.

**Quiz (7):**
1. Numeric: FFN parameters in one Qwen3-8B block (num 150994944, tol 0).
2. Numeric: FFN share of the block, in % (num 78.26, tol 0.1).
3. MCQ: $d_{\text{ff}}=3d$ (not $4d$) because — *SwiGLU has 3 matrices; shrink $d_{\text{ff}}$ by 2/3 to hold
   params constant* (correct).
4. MCQ: the biggest single reason decoder-only won — *signal density: a loss term from every token vs ~15%*
   (correct) [distractors = "it's more elegant" / "bidirectional is weaker" (misconception c)].
5. MCQ: `q_norm`/`k_norm` are in the block because — *QK-Norm re-imposes the variance $\sqrt{d_{\text{head}}}$
   assumed* (correct).
6. MCQ: encoder-decoder in 2026 is — *alive in diffusion (text encoder → K,V; U-Net/DiT decoder cross-attends)*
   (correct).
7. MCQ (diagnosis): you want to sparsify a dense model to cut compute — *sparsify the FFN (78% of the block) → MoE*
   (correct).

**Recurring-thread touchpoints:** `[THREAD:Qwen3-8B]` (the whole block, 192,946,432). `[THREAD:dot-product]`,
`[THREAD:chain-rule]` recur in the "why is this line here" answers. Cross-refs: → page 36 (× 36 blocks + embeddings
= 8.19B), → page 38 (Part IV — sparsify the FFN = MoE), → page 37 (KV cache), → diffusion track (enc-dec reveal, D-17), → RAG (Part IV).

---

## 36 — parameter-counting-qwen3.html
**Title:** "Parameter Counting on Qwen3-8B — Closed Twice" · **Part III** · (D-21: 36) · **Build: S** · **milestone**

**This is the payoff section (brief §10). The learner should finish able to open any HuggingFace `config.json` and
predict the parameter count to a rounding error.** Uses constants §1's exact derivation. **The prime directive: the
count closes to the byte, twice — from the formula AND from the checkpoint's own safetensors index — do not say
"closes to three significant figures" (D-07).**

**Learning objectives:**
1. Derive the general per-block and whole-model parameter formulas and apply them to Qwen3-8B's real config.
2. Reproduce **8,190,735,360** exactly, and verify it two independent ways (the formula; and
   16,381,470,720 bytes ÷ 2).
3. Produce the three downstream numbers for any model: weights in memory, KV cache, full-fine-tune memory (16P).
4. Reconcile the LoRA-target subtotal (norms omitted) back to the full count by adding the norms — it closes to the
   byte.

**PREDICT (he's now assembled the block):** *"Qwen3-8B's model card says '8.2B total, 6.95B non-embedding.' From
four numbers in the config — 4096, 36, 12288, 151936 — can you get that exactly, or only to a rough 5%?"* (Exactly,
to the byte.) Resolve with the derivation.

**Section outline:**
- *The general formula (brief §10; frozen notation).* Per block:
  $P_{\text{attn}}=2d\,d_{\text{head}}(H+H_{kv})$; $P_{\text{ffn}}=3d\,d_{\text{ff}}$ (SwiGLU);
  $P_{\text{norm}}=2d+2d_{\text{head}}$ (two RMSNorms + QK-Norm). Whole model:
  $P_{\text{total}}=Vd+L\cdot P_{\text{block}}+d+Vd\cdot\mathbb{1}[\neg\text{tied}]$ (Symbol Ledger). **Note the
  absences and say why:** no positional-embedding table (**RoPE has zero parameters** — pure geometry, page 33);
  no biases (`attention_bias: false` [VP]).
- *Worked, exactly, against Qwen3-8B's real config [VP, constants §1.1/§1.2 — carry every multiplication; the
  arithmetic IS the lesson].* Print the config JSON. Attention/block: $W_Q\,4096\times4096=16{,}777{,}216$;
  $W_K\,4096\times1024=4{,}194{,}304$; $W_V\,4{,}194{,}304$; $W_O\,16{,}777{,}216$; **subtotal 41,943,040**. (GQA
  check: MHA would make $W_K,W_V$ each 16,777,216 → 67,108,864, so **GQA saves 25,165,824/block = 37.5% of the
  attention block** [DER, constants §1.3] — on top of the 4× KV-cache saving; two wins.) FFN/block:
  $W_{\text{gate}},W_{\text{up}},W_{\text{down}}$ each $4096\times12288=50{,}331{,}648$ → **150,994,944**; FFN share
  $150{,}994{,}944/192{,}946{,}432=\mathbf{78.26\%}$ (the page-35 claim verified). Norms/block:
  `input_layernorm` 4096 + `post_attention_layernorm` 4096 + `q_norm` 128 + `k_norm` 128 = **8,448.** Per block:
  $41{,}943{,}040+150{,}994{,}944+8{,}448=\mathbf{192{,}946{,}432}$. **× 36 = 6,946,071,552.** Embeddings:
  `embed_tokens` $151{,}936\times4096=622{,}329{,}856$; `lm_head` (untied, `tie_word_embeddings: false`)
  $622{,}329{,}856$; `model.norm` 4096. **Total:**
  $6{,}946{,}071{,}552+622{,}329{,}856+622{,}329{,}856+4{,}096=\mathbf{8{,}190{,}735{,}360}$.
- *✅ THE VERIFICATION — closed twice (brief §10 + constants §1.2, the strongest credibility beat in the course).*
  **Route 1 (model card):** card says 8.2B total / 6.95B non-embedding [VP]; our non-embedding 6,946,071,552 =
  **6.95B exact**, total 8,190,735,360 = **8.19B → "8.2B" exact**. **Route 2 (the checkpoint's own index — the
  independent confirmation, MUST be shown):** `model.safetensors.index.json` reports `total_size = 16,381,470,720`
  bytes; all 399 tensors are bf16 (2 B): $16{,}381{,}470{,}720/2=\mathbf{8{,}190{,}735{,}360}$ — **exact match,
  from the checkpoint itself, not the formula. Two independent routes to the same integer** [VP, constants §1.2].
  Print the box (brief §10): *"We derived, from four numbers in a config and nothing else, a count matching the
  official card to every significant figure they publish — and the checkpoint's own byte total, halved, lands on
  the same integer. Nothing about this model is hidden. You can compute it."* **The single most empowering moment
  in this unit** — build toward it.
- *The LoRA-target reconciliation (constants §3 / D-07 — pre-empts a Part IV conflict; `.deepdive` or inline box).*
  The 7-linear-per-layer LoRA-target subtotal **omits the norms** (correctly — norms aren't LoRA targets):
  $6{,}945{,}767{,}424$. It is **not** the model's parameter count. It closes to the full count by adding them:
  $6{,}945{,}767{,}424+\underbrace{36\times8{,}448}_{304{,}128}+\underbrace{4{,}096}_{\text{final norm}}+2\times622{,}329{,}856=\mathbf{8{,}190{,}735{,}360}$.
  **Exact — to the byte. No page may say "closes to three significant figures" (D-07).**
- *The three numbers he can now produce for any model (brief §10):*
  **(1) Weights in memory** $=P\times b_W$: bf16 (2B) $=8.19\times10^9\times2=\mathbf{16.38\text{ GB}}$ [VP,
  constants §1.1]; fp8 8.19 GB; int4 ~4.1 GB (+ scales); fp32 32.8 GB (nobody does this for inference).
  **(2) KV cache** (page 37) $=$ **144 KiB/token, 5.625 GiB at 40,960** [DER, constants §4].
  **(3) Full-fine-tune memory — the memory-budget thread climax `[THREAD:memory-budget]`.** The 16 B/param ledger
  (constants §2.1 — reference the trunk memory-ledger page ≈ p.18 for the *why* of each row): weights $2P$ 16.38 GB
  + grads $2P$ 16.38 GB + fp32 master $4P$ 32.76 GB + Adam $m$ $4P$ 32.76 GB + Adam $v$ $4P$ 32.76 GB = **16P =
  131.05 GB (decimal) = 122.05 GiB (binary)** [DER, constants §2.2]. **Memorize 16P.** ⚠️ **This is STATE ONLY;
  activations are a separate, separately-labelled line, [EST] 2–6 GB with gradient checkpointing (constants §2.2).**
  **⚠️ THE DGX-SPARK BEAT — measured, not asserted (D-04, hardware §2; do NOT print "misses by ~3 GB").** The
  course asserts **no** budget. The learner runs, live: `free, total = torch.cuda.mem_get_info()` and computes
  122.05 GiB against **his** number. Frozen measured facts (state them as [MEA-DEV], and still have him measure):
  usable `MemTotal` = **121.6875 GiB** (firmware carveout eats 6.3125 GiB = 4.9% of the physical 128 GiB, before a
  single byte); **122.05 GiB state vs 121.6875 GiB = −0.36 GiB — does not fit, by 0.3%, before activations**
  (hardware §2, constants §6.8). The villain generalizes: **published capacity is never usable capacity; the spec
  sheet is off by 4.9% before you start.** Hand off to Part IV: *"Full fine-tuning your 8B: 122 GiB of state,
  misses your box by a hair, before activations. LoRA: ~16 GB, fits many times over. That gap is the entire reason
  this course's destination is reachable on hardware you own. The next Part is how"* (constants §2.3: 131.05 GB →
  17.08 GB = **7.67×**; trainable state alone 114.67 → 0.61 GB = **187×**, exactly the parameter ratio
  8,190,735,360 ÷ 43,646,976).
- *The decode bandwidth ceiling (brief §10 bonus; keep light — developed fully on the roofline page 43).* At decode
  you read **every weight for every token**: $16.38\text{ GB}/273\text{ GB/s}=60\text{ ms/token}\to$ **16.67 tok/s
  ceiling** [DER, constants §6.7; **273 GB/s is [VP]-confirmed**, not the retired [M]]; realistic expectation
  ~**10.8 tok/s** (×0.65 heuristic — say which is which: 16.67 is the ceiling, 10.8 is the expectation, constants
  §6.7). This is page-28's irony made numeric — at decode a transformer has RNN economics; it's why int4 (~4.10 GB
  → ~66 tok/s ceiling) roughly quadruples decode speed *even though it does nothing for FLOPs*; **he's watched his
  tok/s change with quantization — now he can predict it with division.** (Forward-ref page 43 for the measurement.)
- **Misconceptions (`.box warn`, brief §10):** (a) *"an 8B model needs 8 GB"* → that's ~int8 weights only; bf16
  inference 16.4 GB + KV + activations; full FT **131.05 GB**; the multiplier from "the name" to "the memory"
  ranges 0.5×–16×; fill in the table for a second model. (b) *"parameter count tells you compute cost"* → for MoE,
  catastrophically false (Kimi K2 is 1T params, 32B active [V-sec]); compute tracks *active*, memory tracks
  *total*; the `397B-A17B` notation exists because of it (page 38, Part IV). (c) *"embedding params matter as much
  as the rest"* → a lookup table, $O(1)$ compute/token; costs memory, not FLOPs — why $C\approx6ND$ uses
  **non-embedding** $N$ and cards report it separately (page 38, Part IV).

**Interactive demo — "The config-file parameter counter" (S; `makeCtrl` + readout, optional `TensorViz`):**
- Sliders/number inputs mirroring a `config.json`: $d$ (512–8192), $L$ (1–100), $H$ (per convention), $H_{kv}$
  (1–H), $d_{\text{head}}$, $d_{\text{ff}}$, $V$ (32k–256k), tied/untied toggle. Live math: the exact per-block and
  total formulas; readouts of total params, non-embedding, embedding %, FFN share, GQA saving/block, bf16 weight
  GB, KV KiB/token, and **16P** full-FT GB/GiB — all via `eng()`. Pre-load the Qwen3-8B values as the default and
  show the total lock onto **8,190,735,360**.
- Ahas: (1) set the config to Qwen3-8B and watch the total hit the exact integer and the FFN share hit 78.26%. (2)
  Flip tied/untied → total jumps by 622,329,856 (the lm_head). (3) Drag $V$ from 151,936 to 32,000 → the total
  drops but "non-embedding" is unchanged — *"two '8B' models with different vocabs aren't equals."* (4) Set a
  Llama-3.3-70B-shaped config ($d$=8192, $L$=80, $d_{\text{ff}}$=28672, $H_{kv}$=8 [VP, constants §1.4]) and read
  its full-FT 16P against his 121.69 GiB box.

**Code artifact (`.box try`): `code/count_params.py`** — loads Qwen3-8B's `config.json`, computes 8,190,735,360 via
the formula, reads `model.safetensors.index.json` `total_size` and divides by 2 to confirm, and (if the checkpoint
is present) asserts against `sum(p.numel())`. Prints all three downstream numbers and runs `mem_get_info()` so the
122.05-GiB-vs-his-box beat is his own measurement (ties to `measure_your_box.py`).

**Milestone:** `.milestone` "You can now open any model card and derive every number on it." (~end of Part III's
core; ~57% of the course.)

**Quiz (8):**
1. Numeric: Qwen3-8B total parameters (num 8190735360, tol 0).
2. Numeric: non-embedding parameters (num 6946071552, tol 0).
3. Numeric: the safetensors `total_size` (16,381,470,720 B) ÷ 2 (num 8190735360, tol 0) — *the independent route.*
4. Numeric: GQA saving per attention block (num 25165824, tol 0) — *37.5% of the attention block.*
5. Numeric: bf16 weights in GB (num 16.38, tol 0.05).
6. Numeric: full-fine-tune state (16P) in GB (num 131.05, tol 0.2) — *state only; activations separate.*
7. MCQ: does full fine-tuning of Qwen3-8B fit on his DGX Spark — *no; 122.05 GiB state vs 121.69 GiB measured, and
   you find out by measuring `mem_get_info()`* (correct) [distractors: "yes, 128 GB" / "misses by 3 GB" = retired].
8. MCQ: model cards report "non-embedding params" separately because — *compute (and $C=6ND$) tracks non-embedding;
   embeddings are $O(1)$ lookup* (correct).

**Recurring-thread touchpoints:** `[THREAD:Qwen3-8B beat …/… — the climax]` (8,190,735,360 closed twice).
`[THREAD:memory-budget beat …]` (16P; the measured 121.6875 GiB; 7.67× / 187× hand-off). Cross-refs: → Part IV LoRA
(187× = the parameter ratio; QLoRA), → memory-ledger page (~p.18) for the 16-byte row-by-row *why*, → roofline page
43 (decode ceiling measured), → page 37 (KV cache), → page 38 Part IV (MoE decouples memory from compute).

---

## 37 — attention-at-scale.html
**Title:** "Attention at Scale: O(S²), FlashAttention, and the KV Cache" · **Part III** · (D-21: 37) · **Build: S** · **closes Part III → hands off to Part IV**

**Framing (brief §5 F6):** separate the costs — they have *different fixes* and conflating them is the main
confusion. This page is the **final memory-budget thread beat of the trunk** (constants §4, §5): the 16P ledger
(page 36) is done; this adds the KV cache and the decode-economics story, then **hands off to Part IV (Adaptation,
38–43)** — the last page of Part III. It builds directly on causal masking (page 32): the cache it sizes exists
*because* causality freezes past $K,V$.

**Learning objectives:**
1. Separate the three attention costs: score-matrix memory ($O(HS^2)$), attention FLOPs ($O(S^2 d)$), and the KV
   cache ($O(S L H_{kv}d_{\text{head}})$) — each with its own fix.
2. Compute the naive score-matrix memory (214.7 GB/layer at full context) and see why FlashAttention is
   *necessary*, not clever — and that it's numerically *exact*, trading arithmetic for HBM traffic.
3. Compute the KV cache (144 KiB/token, 5.625 GiB at 40,960) and the 4× GQA saving, and see the cache would exceed
   the model without it — the cache that page 32's causality made possible.
4. Know the crossover: below $S=18{,}432$ tokens the **FFN**, not attention, dominates FLOPs — "it's $O(S^2)$" is
   usually the wrong bottleneck.

**PREDICT:** *"Qwen3-8B advertises a 40,960-token context. Its single-layer attention score matrix at that length,
in fp32, is about — 200 MB, 2 GB, or 200 GB?"* (It's 214.7 GB *per layer* — and yet the model runs. How?) Resolve
with FlashAttention.

**Section outline:**
- *The costs, stated separately (table, brief §5 F6; Qwen3-8B, $S=40{,}960$ = `max_position_embeddings` [VP]):*
  score-matrix **memory** naive $O(HS^2)$: $32\times40960^2\times4\text{B}=\mathbf{214.7\ GB}$ **per layer** [DER,
  constants §5]; attention **FLOPs** $\approx4S^2d=4(40960)^2(4096)=\mathbf{2.75\times10^{13}}=27.5$ TFLOP/layer;
  FFN FLOPs $\approx6Sd\,d_{\text{ff}}=6(40960)(4096)(12288)=\mathbf{1.24\times10^{13}}=12.4$ TFLOP/layer; KV cache
  (decode) — below. **Print the 214.7 GB** (brief §5 F6): a single layer's score matrix at the model's own
  advertised context doesn't fit in any GPU made, yet the model runs — motivates FlashAttention.
- *The crossover, worked (brief §5 F6 — genuinely counter-intuitive):* attention FLOPs exceed FFN FLOPs when
  $4S^2d>6Sd\,d_{\text{ff}}$, i.e. $S>1.5\,d_{\text{ff}}=1.5\times12288=\mathbf{18{,}432}$ tokens [DER, constants
  §5]. **Below ~18k tokens the FFN dominates compute, not attention.** Say it — the standard narrative overstates
  when $O(S^2)$ bites; at typical chat lengths it isn't your bottleneck.
- *Mitigation 1 — FlashAttention (fix the memory, not the FLOPs) [VP, brief §5 F6].* It computes **exactly the
  same** attention (not an approximation): never materializes the $(S,S)$ matrix in HBM, tiles into SRAM, uses the
  **online softmax** (running max + running normalizer — the streaming-logsumexp trick) so normalization corrects
  as blocks arrive. Memory $O(S^2)\to O(S)$; FLOPs the same or slightly more (backward recomputation). Faster
  anyway because attention is **memory-bandwidth-bound**, not compute-bound. The sentence (brief §5 F6):
  *"FlashAttention made attention faster by making it do more arithmetic. That's only paradoxical if you think the
  GPU's problem is arithmetic. Its problem is memory."* 2026 status (`.deepdive`): **FlashAttention-4** (paper Mar
  5 2026, `flash-attn-4` on PyPI), targets Blackwell; Blackwell is **asymmetric** (tensor cores ~2.25× but the SFU
  that computes `exp` didn't scale), so FA4 computes `exp()` by a polynomial on the FMA units — **a concrete
  instance of the co-design principle he can verify on his own GB10** (label the throughput figures [V]).
- *Mitigation 2 — GQA/MQA (fix the KV cache, not the score matrix) [VP] `[THREAD:memory-budget]`.* Flag the
  distinction: FlashAttention fixes *training/prefill activation memory*; GQA fixes *decode-time KV cache* —
  orthogonal, both ship together. (The KV cache is the thing page 32's causality lets you keep at all — here we
  count its bytes.) KV bytes $=2\,S\,L\,H_{kv}\,d_{\text{head}}\,b$ (**note $H_{kv}$, not $H$ —
  constants §4 calls this the single most common field error; do not regress**). Qwen3-8B [VP], bf16:
  per token per layer $2\times8\times128\times2=\mathbf{4096\text{ B}=4\text{ KiB}}$; **per token, all 36 layers
  $=147{,}456\text{ B}=144\text{ KiB/token}$**; at $S=40{,}960$: $\mathbf{5.625\text{ GiB}}$ [DER, constants §4].
  **If it were MHA** ($H_{kv}=32$): 576 KiB/token → **22.50 GiB**; GQA ratio $32/8=4$ → **exactly 4× saved, 16.88
  GiB recovered.** Context: bf16 weights are 16.38 GB; **without GQA the KV cache at full context would exceed the
  model itself** — *that* is why GQA is universal in 2026. Mechanically: $H=32$ query heads share $H_{kv}=8$ KV
  heads (4 query heads per KV head); MHA is $H_{kv}=H$, MQA is $H_{kv}=1$, GQA interpolates; GQA-8 is the sweet
  spot [VP for Qwen3].
- *Mitigations 3–5 (`.deepdive`, one line each):* **MLA** (DeepSeek-lineage: cache a low-rank latent, decompress
  on the fly — compute-for-memory trade; Kimi K2/K2.5, GLM-5, Ling 2.5 [V-sec]) — teach the *idea*, not the
  equations; **sparse/sliding-window/hybrid** (Trinity Large SWA-4096 3:1 local:global; Qwen3-Coder-Next Gated
  DeltaNet 3:1, 262k ctx — name the pattern: most layers cheap-local, a few global, ratio commonly 3:1 [V-sec]);
  **FlexAttention** (PyTorch 2.5+, `score_mod`/`mask_mod` as Python fused by `torch.compile`) — honest caveats:
  kernel options not yet public API, documented numerical discrepancies vs SDPA, profile to confirm you didn't get
  a silent dense fallback [V]. **Code recommendation:** teach `F.scaled_dot_product_attention` as the default
  (stable, dispatches to a FlashAttention backend), show the naive impl first for understanding, present
  FlexAttention as the escape hatch — **do not build the core code path on FlexAttention** (API instability).
- **Misconceptions (`.box warn`, brief §5 F6):** (a) *"FlashAttention is an approximation / linear attention"* →
  numerically exact (up to FP reassociation); changes *where data lives*, not *what's computed*; the online-softmax
  identity is algebraic, not truncation. (b) *"FlashAttention makes attention $O(S)$"* → **memory** becomes
  $O(S)$; **FLOPs stay $O(S^2)$** — why 128k context is *possible* but still *expensive*. (c) *"GQA is a
  quality-hurting approximation"* → it's an architectural choice made **before** pretraining — the model is trained
  with 8 KV heads and never had 32; nothing is approximated (uptraining MHA→GQA post-hoc is different and does cost
  a little) [M]. (d) *"$O(S^2)$ is why long context is hard"* → below 18,432 tokens the **FFN dominates FLOPs**,
  and at decode the binding constraint is **KV-cache memory/bandwidth**, not attention FLOPs — "it's $S^2$" is
  usually the wrong bottleneck.

**Interactive demo — "The three attention costs" (S; `Plot`, log-y):**
- Sliders: sequence length $S$ (128–131,072, log), precision for the score matrix (fp32/bf16), $H_{kv}$ (1/8/32 —
  MQA/GQA/MHA). Three live traces on a log-y axis vs $S$: score-matrix memory $HS^2 b$, attention FLOPs $4S^2d$,
  FFN FLOPs $6Sd\,d_{\text{ff}}$; plus a fourth readout: KV-cache size $2SLH_{kv}d_{\text{head}}b$ in GiB. Vertical
  detents at $S=18{,}432$ (the FFN/attention crossover) and $S=40{,}960$ (Qwen3's context). Readouts via `eng()`.
- Ahas: (1) at $S<18{,}432$ the FFN-FLOP trace sits *above* the attention-FLOP trace — *"below chat length,
  attention isn't your compute bottleneck."* (2) Flip $H_{kv}$ 8→32 and the KV readout jumps 5.625→22.50 GiB,
  crossing above the 16.38 GB weight line — *"without GQA the cache outgrows the model."* (3) Watch the
  score-matrix trace blow past any real GPU memory at full context — *"and yet it runs, because FlashAttention
  never builds that matrix."*

**Code artifact (`.box try`): `code/kv_cache_ledger.py`** — from a config dict computes 144 KiB/token, the
full-context 5.625 GiB, and the MHA counterfactual 22.50 GiB; then measures a real KV-cache allocation on his box
with `torch.cuda.memory_allocated()` before/after a generate call (ties to `measure_your_box.py`).

**Quiz (7):**
1. Numeric: Qwen3-8B KV cache per token, all layers, in KiB (num 144, tol 1).
2. Numeric: KV cache at $S=40{,}960$ in GiB (num 5.625, tol 0.05).
3. Numeric: the FFN/attention FLOP crossover $S$ for Qwen3-8B (num 18432, tol 0).
4. MCQ: FlashAttention changes — *memory to $O(S)$; FLOPs stay $O(S^2)$* (correct) [distractors = misconceptions
   (a)/(b)].
5. MCQ: GQA vs MHA saves — *4× on the KV cache (8 KV heads vs 32)* (correct) [distractor: "reduces attention
   FLOPs"].
6. MCQ: at decode time the binding constraint is usually — *KV-cache memory/bandwidth, not attention FLOPs*
   (correct).
7. MCQ (diagnosis): you OOM building an attention matrix at 64k context in a from-scratch impl — *use
   `F.scaled_dot_product_attention` (a FlashAttention backend) so the $(S,S)$ matrix is never materialized* (correct).

**Recurring-thread touchpoints:** `[THREAD:memory-budget beat …/final trunk beat]` (16 B/param → the 16P ledger
(page 36) → **now the KV cache** — the last memory beat before Part IV; paid off at the roofline, page 43).
`[THREAD:Qwen3-8B]` ($H_{kv}=8$, $L=36$). Cross-refs: → page 32 (causality is why the cache is sound), → page 36
(weights vs KV vs full-FT), → page 28 (decode = RNN economics), → Part IV (38–43, adaptation) and → roofline page
43 (bandwidth-bound decode).

---

## END OF PART III

**Seams the fork and downstream specs must match (named by page number):**
- **Page 31 (self vs cross, the diffusion hinge)** hands $K,V$-from-a-second-sequence, CFG-as-the-$Y$-path, and
  MMDiT-as-masked-self-attention to the **diffusion track (53–62)**. Cross-attention must be treated as *already
  taught*, not re-introduced.
- **Page 32 (causal masking, the LLM hinge)** hands the additive $-\infty$ mask, the 4096-signals-per-pass
  economics, and "causality ⇒ the KV cache is sound" to the **LLM track (44–52)** and to page 37 (which sizes the
  cache). Causal masking must be treated as *already taught*, not re-introduced by the LLM track.
- **Page 36 (16P ledger; measured 121.6875 GiB; 7.67× / 187×; decode ceiling)** hands the memory ledger and the
  LoRA hand-off to **Part IV (LoRA/QLoRA, pp.40–41)** and the decode ceiling to the **roofline page (43)**. The
  measured budget and the "closes to the byte" reconciliation must not be re-derived differently.
- **Page 37 (O(S²), FlashAttention, GQA, KV cache)** is the last page of Part III; it hands the KV-cache/decode
  economics to the **roofline page (43)** and closes the trunk into **Part IV (Adaptation, 38–43)**. It is **not**
  the fork reveal — that content moved to **page 38 (Part IV opener: scaling laws / MoE / emergence)**, which must
  align with the **D-17 fork reveal** ("the machine didn't change, the target did"), **LLM-track-first** ordering
  (D-19), and owes the MoE-LoRA `target_parameters` caveat (D-06) to the rest of Part IV.
- **Pages 34–35 (residual stream, transformer block)** assume the *why* of gradient flow / normalization axis was
  taught in the trunk (Parts I–II); confirm those pages exist and cross-reference cleanly.
