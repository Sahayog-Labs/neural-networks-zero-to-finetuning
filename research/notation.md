# notation.md — FROZEN. Prescriptive. Paste into every builder prompt verbatim.

**Status: RATIFIED 2026-07-16.** This table is normative. **No page invents notation.** Where a symbol you need
is absent, flag it to the architect — do not choose one.

ML notation is a genuine mess, and inconsistency across ~50 independently-working agents is a top-three failure
mode for this build. The rule that makes this survivable:

> **Internal consistency beats matching any one paper. But every deviation from the field's convention is
> declared, priced, and mitigated by a Translation Table note on the page where it first bites.**

The deviations are **not apologies**. Teaching the learner that *notation is local and translatable* — rather
than a fixed truth he is failing to know — is itself a durable lesson, and it inoculates against the #1 reason
capable people bounce off their first paper.

---

## 0. THE FIVE HARD INVARIANTS

Violating any of these is a build defect, checkable mechanically.

1. **$W$ is ALWAYS $(d_{\text{out}}, d_{\text{in}})$.** Never (in, out). Never. In both conventions. In every track.
2. **$\theta$ means learnable parameters and nothing else.** Geometric angles are $\vartheta$.
3. **$t$ is the diffusion timestep. $k$ is the optimizer step. $i$ is the sequence position.** Three symbols, three jobs. See §3.
4. **Every equation with >3 distinct symbols carries a Symbol Ledger** (§8). No exceptions, including inside collapsibles.
5. **Every multi-step tensor computation carries a shape ribbon** (§7). Non-negotiable.

---

## 1. ROW OR COLUMN — the central decision

### 1.1 The ground truth

**[VP] PyTorch 2.13 `torch.nn.Linear`** applies

$$y = xA^\top + b$$

with `weight` of shape **`(out_features, in_features)`**, input `(*, in_features)`, output `(*, out_features)`.

Every maths treatment writes $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ with $\mathbf{x}$ a column.

### 1.2 The reconciling insight — a gift, and the course must not squander it

> **The weight matrix has shape (out, in) in BOTH conventions.** `nn.Linear(3, 5).weight.shape == (5, 3)`, and
> the maths's $W$ in $\mathbf{y} = W\mathbf{x}$ is also $(5,3)$. The conventions differ **only** in whether you
> stack the batch on the left. **There is no disagreement about $W$ itself.**

This dissolves the single most common source of shape confusion in the field, and it gets its own page (~p.7),
immediately after the page-6 code moment, when he has just seen both forms and the question is live.

### 1.3 PRESCRIPTION

| Context | Convention | Form |
|---|---|---|
| Single-example maths; all trunk derivations; backprop | **Column vectors** | $\mathbf{y} = W\mathbf{x} + \mathbf{b}$, $\mathbf{x} \in \mathbb{R}^{d_{in}\times 1}$, $W \in \mathbb{R}^{d_{out}\times d_{in}}$ |
| Anything with a batch or sequence axis; **all code**; **all attention** | **Row-batched** | $Y = XW^\top + \mathbf{b}^\top$, $X \in \mathbb{R}^{B\times d_{in}}$ |
| **Everywhere, no exceptions** | $W$ is $(d_{\text{out}}, d_{\text{in}})$ | |

**Why split rather than pick one globally — the honest trade:**
- **Column-only** would force transposing every attention equation in the literature. He *will* read papers.
  Attention is written row-style universally; fighting that spends credibility to buy nothing.
- **Row-only** makes backprop derivations uglier — gradients become row covectors and the Jacobian bookkeeping
  that makes backprop *click* gets muddier. The trunk is where the derivation must land cleanest.
- The split costs **one page** of explicit bridging. The invariant means **the thing he must remember never
  changes** — only the batch placement does, and that is visible in the shape ribbon on every line.

**⚠️ Spec trap [VP]:** a 2→2 layer **cannot** demonstrate the convention (`(2,2)` is ambiguous). **The shape-demo
page MUST use `nn.Linear(2, 3)` so `(out, in) = (3, 2)` is unmistakable.**

---

## 2. TYPOGRAPHY — normative

| Object | Style | Example |
|---|---|---|
| Scalar | italic lowercase | $x$, $\eta$, $r$ |
| Vector | **bold** lowercase, **column** by default | $\mathbf{x}$, $\mathbf{b}$, $\boldsymbol{\epsilon}$ |
| Matrix / 2-D | plain uppercase (bold optional for emphasis) | $W$, $X$, $Q$, $K$, $V$ |
| Tensor (≥3 axes) | plain uppercase; **shape ribbon mandatory** | $X$ with `(B, S, d_model)` |
| Set / space / operator | blackboard | $\mathbb{R}$, $\mathbb{E}$, $\mathbb{1}[\cdot]$ |
| **Scalar element of a vector** | italic + subscript, **not bold** | $x_i$ is the $i$-th component of $\mathbf{x}$ |
| Loss (per-example) | script | $\mathcal{L}$ |
| Loss (batch-averaged) | italic uppercase | $J$ |
| Learnable parameters | theta | $\theta$ |
| Estimate / prediction | hat | $\hat y$, $\hat\epsilon$ |
| Noised quantity (diffusion) | subscript $t$ | $\mathbf{x}_t$ |
| Layer index | parenthesised superscript | $W^{(\ell)}$, $\mathbf{a}^{(\ell)}$ |
| Head index | parenthesised superscript | $W_Q^{(h)}$ |

**$\mathcal{L}$ is per-example; $J$ is batch-averaged.** The factor $1/B$ is exactly where learning-rate
confusion breeds, so the notation makes it **visible** rather than hiding it. State the distinction once, on
the minibatch page, and hold it.

---

## 3. INDEX DISCIPLINE — resolves the three-way $t$ collision

**The pedagogy brief found two $t$ collisions. There are three.** It flagged diffusion-timestep vs
sequence-position, and reserved $t$ for diffusion. It **missed** that the training brief uses $t$ for the
**optimizer step** throughout — $\theta_{t+1} = \theta_t - \eta\nabla\mathcal{L}(\theta_t)$, Adam's $\beta_1^t$,
$T_{\text{warm}}$, $T_{\text{total}}$, the whole cosine schedule. That is pervasive, load-bearing, and it
collides with both of the others.

**Resolve all three at once. This is the ratified assignment:**

| Index | Range | Means | Notes |
|---|---|---|---|
| $b$ | $1..B$ | batch element | $B$ = batch size |
| **$i$** | $1..S$ | **sequence position** (also: query position in attention) | **Not $t$.** |
| $j$ | $1..S$ or $1..S_{\text{txt}}$ | key/value position in attention | |
| **$v$** | $1..V$ | **vocabulary / class index** | **Not $k$** — $v$ for vocab is mnemonic and frees $k$ |
| $\ell$ | $1..L$ | layer index | $L$ = number of layers |
| $h$ | $1..H$ | attention head | $H$ = number of heads |
| $c$ | $1..C$ | channel (conv / latent) | $C$ = channel count |
| $n$ | $1..N$ | dataset example index | $N$ = dataset size |
| **$t$** | $0..T$ | **diffusion timestep / noise level ONLY** | $T = 1000$ by convention |
| **$k$** | $0..K$ | **optimizer step ONLY** | $\theta_{k+1} = \theta_k - \eta\,\mathbf{g}_k$ |
| $r$ | — | LoRA rank (a count, not an index) | |

### 3.1 The three rulings, with their prices

| Ruling | Deviates from | Price | Why it's worth it |
|---|---|---|---|
| **$i$ = sequence position, not $t$** | transformer literature (universal $t$) | one Translation note per attention page | **DiT pages need $t$ and position simultaneously.** Without this, every DiT page needs special pleading. Pedagogy called this "genuinely unavoidable"; it is avoidable, once all three are separated. |
| **$k$ = optimizer step, not $t$** | optimisation literature (universal $t$) | one Translation note on the SGD page | Frees $t$ globally. **Bonus: it makes "step vs epoch" typographically obvious** — a top-5 real bug ("LR schedules are indexed by STEP, not epoch") now has a symbol enforcing it. |
| **$v$ = vocab index, not $k$** | ML convention ($k$ for classes) | negligible | Frees $k$; and **unifies "classes" with "vocab": the LLM loss is cross-entropy over $V$ classes, and $\ln V = 11.93$ is the same fact as $\ln K$.** A win, not a cost. |

### 3.2 Mandatory mitigation — the Translation Table

**Because we deviate, every page introducing a symbol that papers write differently MUST carry a one-line
"papers call this…" note.** Ship it as a persistent course-wide appendix too.

| We write | Papers write | Where he'll meet it |
|---|---|---|
| $i$ (sequence position) | $t$, $i$, $n$, $j$ | every transformer paper |
| $k$ (optimizer step) | $t$, $n$, "step" | every optimisation paper |
| $S$ (sequence length) | $T$, $L$, $N$, $n$, `seq_len`, `max_length` | **four rival letters. This is the worst one.** |
| $L$ (layers) | $N$, $L$, `num_hidden_layers` | |
| $H$ (heads) | $h$, $H$, $n_h$, `num_attention_heads` | |
| $H_{kv}$ (KV heads) | $n_{kv}$, $H_{kv}$, `num_key_value_heads` | |
| $d_{\text{head}}$ | $d_k$, $d_v$, $d_h$, `head_dim` | |
| $d_{\text{model}}$ | $d$, $d_{\text{model}}$, `hidden_size` | |
| $V$ (vocab) | $V$, $K$, `vocab_size` | |
| $\vartheta$ (geometric angle) | $\theta$ — **collides with parameters** | dot-product / RoPE pages |
| $\sigma_t$ (noise std) | $\sigma_t$ — **and $\sigma$ is also sigmoid** | see §5 |
| $w_g$ (CFG scale) | $w$, $s$, `cfg`, `guidance_scale` | |

---

## 4. THE CANONICAL SYMBOL TABLE — course-wide, normative

### 4.1 Trunk

| Symbol | Meaning | Shape | Units / range |
|---|---|---|---|
| $\mathbf{x}$ | input vector, one example | $(d_{in}, 1)$ | data units |
| $X$ | batch of inputs | $(B, d_{in})$ or $(B, S, d_{\text{model}})$ | |
| $y$, $\mathbf{y}$ | target | scalar or $(d_{out}, 1)$ | |
| $\hat y$ | prediction | matches $y$ | |
| $W^{(\ell)}$ | weights, layer $\ell$ | $(d_{out}, d_{in})$ | dimensionless |
| $\mathbf{b}^{(\ell)}$ | bias, layer $\ell$ | $(d_{out}, 1)$ | same as pre-activation |
| $\mathbf{z}^{(\ell)}$ | **pre-activation** (also: logits at the output) | $(d_{out}, 1)$ | log-odds at the output |
| $\mathbf{a}^{(\ell)}$ | post-activation | $(d_{out}, 1)$ | activation's range |
| $\phi(\cdot)$ | **generic activation** | elementwise | — |
| $\sigma(\cdot)$ | **logistic sigmoid — WITH an explicit argument** | elementwise | $(0,1)$ — **see §5** |
| $\mathcal{L}$ | loss, **per-example** | scalar | **nats** |
| $J$ | loss, **batch-averaged** | scalar | nats |
| $\theta$ | all parameters, flattened | $(P,)$ | $P$ = param count |
| $\nabla_\theta\mathcal{L}$, $\mathbf{g}_k$ | gradient at step $k$ | $(P,)$ — **same shape as $\theta$** | loss/param |
| $\boldsymbol\delta^{(\ell)}$ | $\partial\mathcal{L}/\partial\mathbf{z}^{(\ell)}$ | $(d_\ell, 1)$ | **the backprop workhorse** |
| $\eta$ | **learning rate — and nothing else, ever** | scalar | $[10^{-6}, 10^{-1}]$ |
| $\lambda$ | **weight decay — and nothing else in the trunk** | scalar | 0.01–0.1 |
| $\mu$ | momentum coefficient | scalar | 0.9 |
| $\beta_1, \beta_2$ | **Adam moments — always numeric subscripts** | scalar | 0.9 / 0.95 or 0.999 |
| $\varepsilon$ | Adam's / norm's stability floor | scalar | 1e-8, 1e-6 |
| $\boldsymbol\epsilon$ | **Gaussian noise — bold, distinct from $\varepsilon$** | matches $\mathbf{x}$ | $\mathcal{N}(0,I)$ |
| $\odot$ | elementwise (Hadamard) product | — | |
| $\vartheta$ | **geometric angle between two vectors** | scalar | radians/degrees |
| $\kappa$ | condition number $\lambda_{\max}/\lambda_{\min}$ | scalar | $10^4$–$10^6$ for real nets **[EST]** |
| $P$ | total parameter count | scalar | 8,190,735,360 |
| $N$ | dataset size **or** non-embedding params in $C\!=\!6ND$ | scalar | **⚠️ dual use — always say which** |
| $D$ | training tokens (in $C = 6ND$) | scalar | |
| $\mathbb{E}[\cdot]$ | expectation | — | |

### 4.2 Architecture

| Symbol | Meaning | Value (Qwen3-8B) |
|---|---|---|
| $d_{\text{model}}$, or $d$ | residual-stream width | **4096** |
| $d_{\text{ff}}$ | FFN inner width | **12288** ($=3d$) |
| $d_{\text{head}}$ | per-head dim | **128** |
| $L$ | number of layers | **36** |
| $H$ | query heads | **32** |
| $H_{kv}$ | **key/value heads** | **8** |
| $S$ | **sequence length / context** | up to 40,960 |
| $S_{\text{txt}}$ | text-token count (cross-attention source) | 77 / 512 |
| $V$ | vocabulary size | **151,936** |
| $Q, K, V$ | queries / keys / values | $(S, d_{\text{head}})$ per head |
| $A$ | attention weights, row-softmaxed | $(S, S)$ — **rows sum to 1, columns do not** |
| $M$ | additive mask | $(S,S)$; $0$ or $-\infty$ |
| $E$ | embedding matrix | $(V, d)$ |
| $b_{\text{rope}}$ | RoPE base (`rope_theta`) | **1e6** |
| $R_m$ | RoPE rotation for position $m$ | block-diagonal |
| $N_E$ / $K_E$ | MoE: expert count / active experts | |
| $RF_\ell$ | **receptive field after layer $\ell$** | **not $r$** |

**$H$ is the number of attention heads, course-wide.** Image height is **never** $H$ in prose. Latents are
written `(B, C, H_lat, W_lat)`, pixels `(B, 3, H_px, W_px)`. **This costs two subscripts and frees $H$
permanently.** (Both `H = heads` and `H = height` are too entrenched to fight; the subscript is the cheap out.)

### 4.3 Adaptation (trunk-owned; both tracks apply it)

| Symbol | Meaning | Shape |
|---|---|---|
| $W_0$ | frozen pretrained weight | $(d_{out}, d_{in})$ |
| $\Delta W$ | **the update** — *this*, not $W_0$, is what LoRA claims is low-rank | $(d_{out}, d_{in})$ |
| $A$ | LoRA "down" | $(r, d_{in})$, init $\mathcal{N}(0,\varsigma^2)$ |
| $B$ | LoRA "up" | $(d_{out}, r)$, **init zeros** |
| $r$ | **rank** | scalar |
| $\alpha$ | **LoRA scaling (`lora_alpha`) — bare $\alpha$, no subscript** | scalar |
| $\alpha/r$ | the effective scale | scalar |
| $\sigma_i$ | **singular values (SVD)** — subscripted by index $i$ | scalar, descending |
| $U, \Sigma, V^\top$ | SVD factors | — |
| $b_W$ | bytes per stored weight | 4 / 2 / 1 / 0.5 |
| $k_O$ | optimizer bytes per **trainable** param | 0 / 4 / 8 / 12 / 2 |
| $N_t$ | **trainable** parameter count | 43,646,976 (r=16 all-linear) |

**⚠️ $A$ is overloaded** (attention weights **and** LoRA's down-projection). They never co-occur in one
equation, and both are entrenched. **Rule: in any equation containing $B$ and $r$, $A$ is LoRA's. In any
equation containing $Q,K,V$ and softmax, $A$ is attention's.** If a page needs both, write LoRA's as $A_{\text{lo}}$.

**⚠️ $\Sigma$ is overloaded** (SVD's singular-value matrix and a Gaussian covariance). Rule: $\Sigma$ with $U,V^\top$
adjacent is SVD's; $\Sigma$ inside $\mathcal{N}(\mu,\Sigma)$ is a covariance. The course's only covariance is
isotropic ($\sigma^2 I$), so the collision is nearly theoretical.

### 4.4 Diffusion

| Symbol | Meaning | Shape / value |
|---|---|---|
| $\mathbf{x}_0$ | clean data (a **latent**, in LDM) | $(C, H_{\text{lat}}, W_{\text{lat}})$ e.g. $(16,128,128)$ |
| $\mathbf{x}_t$ | data noised to level $t$ | same |
| $t$ | **diffusion timestep** | $0..T$; $T = 1000$ |
| $\beta_t$ | noise variance added **at step $t$** — a **rate** | $10^{-4} \to 0.02$ |
| $\alpha_t \equiv 1-\beta_t$ | signal retention at step $t$ | |
| $\bar\alpha_t \equiv \prod_{s\le t}\alpha_s$ | **cumulative** signal retention | $1 \to 4\times10^{-5}$ |
| $\sigma_t$ | **noise standard deviation — subscripted, no argument. See §5.** | $\sqrt{1-\bar\alpha_t}$, or Karras $\sigma$ |
| $\boldsymbol\epsilon$ | the noise that was added | $\mathcal{N}(0,I)$ |
| $\epsilon_\theta$ | the noise-prediction network | same shape as $\mathbf{x}_t$ |
| $\mathbf{v}$, $v_\theta$ | velocity target / network | same shape |
| $\mathbf{s}_\theta$ | **the score, $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ — bold** | same shape as $\mathbf{x}$ |
| $\psi_t$ | **the rotation angle**, $\psi_t = \arccos\sqrt{\bar\alpha_t}$ | radians |
| $a_t, b_t$ | **the general path**: $\mathbf{x}_t = a_t\mathbf{x}_0 + b_t\boldsymbol\epsilon$ | **the reconciliation's two functions** |
| $\mathbf{c}$ | **conditioning** (text embedding) — **bold** | $(S_{\text{txt}}, d_c)$ |
| $\varnothing$ | the null/empty conditioning | |
| $w_g$ | **CFG guidance scale** | 1.0 – 8.0 |
| $\lambda_{pp}$ | DreamBooth prior-preservation weight | 1.0 |
| $\beta_{\text{KL}}$ | VAE's rate–distortion knob | ~1e-6 for SD's VAE |
| $f$ | VAE spatial downsample factor | **8** |
| $C$ | latent channel count | 4 / 16 / 32 |
| $\theta_E$, $\theta_D$ | VAE encoder / decoder parameters | **not $\phi, \psi$ — both are taken** |

**$a_t, b_t$ is the course's most valuable notational choice.** It makes the DDPM↔flow-matching reconciliation
a *table* rather than two theories:

| Framework | $a_t$ | $b_t$ | Constraint | Endpoint |
|---|---|---|---|---|
| **DDPM / VP-SDE** | $\sqrt{\bar\alpha_t}$ | $\sqrt{1-\bar\alpha_t}$ | $a_t^2+b_t^2=1$ — **a circle** | $a_T \approx 0.006$ (leaky) |
| **Rectified flow** | $1-t$ | $t$ | $a_t+b_t=1$ — **a chord** | $a_1 = 0$ exactly |
| **VE / Karras** | $1$ | $\sigma_t$ | none | $\sigma_{\max}\approx 80$ |
| **Cosine** | $\cos\psi_t$ | $\sin\psi_t$ | circle, uniform sweep | $a_1 = 0$ exactly |

**Everything in the diffusion track is a choice of $(a_t, b_t)$, which coordinate the network reports, and the
loss weighting across $t$. There is no third theory.**

### 4.5 LLM track

| Symbol | Meaning |
|---|---|
| $x_i$ | token ID at position $i$; integer in $[0, V)$ |
| $x_{<i}$ | the prefix |
| $p_\theta(\cdot \mid x_{<i})$ | the softmax over the vocab, shape $(V,)$ |
| $\rho$ | chars-per-token compression ratio ($\approx 4$) |
| $\text{PPL}$ | $e^{\mathcal{L}}$ |
| $m_i$ | loss mask $\in \{0,1\}$; implemented as `label = -100` |
| $T_{\text{samp}}$ | **sampling temperature** — **not $T$, which is the diffusion chain length** |
| $\pi_\theta$, $\pi_{\text{ref}}$ | DPO policy / frozen reference |
| $\beta_{\text{DPO}}$ | DPO's KL strength (~0.1) |
| $y_w$, $y_l$ | preferred / rejected completion |

---

## 5. THE $\sigma$ COLLISION — RESOLVED, and I am overruling the pedagogy brief

### 5.1 The conflict

- **brief-pedagogy §6.4** reserves $\sigma$ for the **logistic sigmoid, exclusively, forever**, and assigns
  $s_t$ to the diffusion noise std. It explicitly requests sign-off from the diffusion author.
- **brief-diffusion** uses $\sigma_t$ pervasively and structurally: Karras sigmas, $\sigma_{\max}=80$, the
  VE-SDE, `FlowMatchEulerDiscreteScheduler`'s `sigmas` array, and ComfyUI's scheduler output — *which is
  literally an array named `sigmas` that the learner has printed.*

### 5.2 The ruling

> **$\sigma(\cdot)$ WITH an explicit argument = the logistic sigmoid. Trunk only.**
> **$\sigma_t$ — subscripted, no argument — = the diffusion noise standard deviation. Diffusion track only.**
> **They may never appear on the same page.** If a diffusion-track page needs a sigmoid, it writes
> `SiLU`, `torch.sigmoid`, or $\text{logistic}(\cdot)$ — never $\sigma$.
> **$s_t$ is banned** (it would be a third symbol for a two-symbol problem, and $\mathbf{s}$ is the score).

### 5.3 Why I am overruling pedagogy, and what it costs

**Pedagogy's own brief says: *"this brief should lose the argument on any individual symbol — but the
resolution must be global, ratified once, and frozen."*** This is that ratification.

**The argument:** forcing $s_t$ means every code snippet in the diffusion track says `sigmas` while the prose
says $s_t$. The track's whole promise is *"theory under the knobs you already turn"* — and the knob is
**literally named sigma**. Renaming it breaks the course's central rhetorical move in exactly the place it
matters most, to buy a collision that is track-scoped anyway.

**What it costs, stated honestly:** two rules instead of one, and a real risk on DiT pages. **Mitigation:** the
sigmoid lives in the trunk (~pages 1–12); $\sigma_t$ lives in the diffusion track (~pages 39–48). The only
overlap surface is SwiGLU/SiLU — which is written `SiLU` anyway, by name, in every config file. **The collision
surface is approximately zero, and the anti-pattern checklist tests for it mechanically (§9, #3).**

**If the architect overrules me back:** do it globally, use $s_t$ everywhere including in code comments, and
accept that the ComfyUI-connection pages will read awkwardly. **Consistency dominates correctness-of-choice
here** — what must not happen is two agents choosing differently.

### 5.4 The related collisions, resolved

| Symbol | Ruling |
|---|---|
| $\alpha$ | **Bare $\alpha$ = LoRA scaling.** $\alpha_t$ / $\bar\alpha_t$ = diffusion schedule — **always subscripted by $t$**. A bare $\alpha$ never means a schedule. |
| $\beta$ | $\beta_t$ = diffusion schedule (**subscript $t$**). $\beta_1,\beta_2$ = Adam (**numeric subscripts**). $\beta_{\text{KL}}$ = VAE. $\beta_{\text{DPO}}$ = DPO. **A bare $\beta$ never appears.** |
| $\eta$ | **= learning rate, exclusively.** DDIM's stochasticity knob is written in code font as `eta` and called *"DDIM's stochasticity parameter"* in prose. **It is never a maths symbol in this course.** |
| $\lambda$ | = weight decay in the trunk. $\lambda_{pp}$ = DreamBooth prior preservation. IP-Adapter's is the API's `weight`. $\lambda_i$ = Hessian eigenvalues (only on the LR/curvature page, with a Ledger). |
| $\phi$ | **= generic activation, exclusively.** Diffusion's rotation angle is $\psi_t$. VAE params are $\theta_E,\theta_D$. |
| $\vartheta$ vs $\theta$ | **$\theta$ = parameters. $\vartheta$ = geometric angle.** `cos ϑ = a·b/(‖a‖‖b‖)`. brief-foundations and brief-architectures both write $\theta$ for the angle — **correct them.** |
| $w$ | **A bare $w$ never appears.** $w_{ij}$ = weight element. $w_g$ = CFG scale. $w_t$ = the ELBO's loss weight (subscripted $t$, diffusion only). |
| $A$ | see §4.3 |
| $s$ | $\mathbf{s}_\theta$ (bold) = the score field. $s$ never means a std. |

---

## 6. THE $t$ CONVENTION IN DIFFUSION — one direction, course-wide

**Course prose, always:** *$t = 0$ is data. $t = T$ (DDPM) or $t = 1$ (flow matching) is noise. Sampling runs $t$
downward.* Everything else goes in the Translation Table.

> **⚠️ Correction to brief-diffusion §6.2.** It asserts SD3/FLUX's *"$t=0$ is data, $t=1$ is noise"* is
> ***"opposite to DDPM's convention."*** **It is not opposite — DDPM also has data at $t=0$.** The claim is
> wrong as stated. What *is* confusing is (a) some flow-matching papers (Lipman's original) put **noise** at
> $t=0$, and (b) diffusers' `sigmas` array runs 1→0 during sampling with $\sigma \equiv t$. **Verify against
> `diffusers` 0.39 and either correct the sentence or delete it.** Do not let it ship — an error inside a
> warning box about confusion is the worst possible place for one.

**Three live conventions, one translation table, printed once and referenced always:**

| Convention | Data | Noise | Where |
|---|---|---|---|
| **DDPM** | $t=0$ | $t=T=1000$ | SD1.5, SDXL |
| **Flow matching (diffusers)** | $t=0$ | $t=1$ | SD3.5, FLUX |
| **Karras** | $\sigma \to 0$ | $\sigma_{\max}\approx 80$ | `karras` scheduler, EDM |

**`num_train_timesteps=1000` in a FLUX config is vestigial** — an indexing convention inherited from DDPM
tooling. Flow matching is continuous in $t\in[0,1]$. [VP: it is a real default in `FlowMatchEulerDiscreteScheduler`.]
Worth its own warning box; have him open the config.

---

## 7. THE SHAPE RIBBON — mandatory

Every multi-step tensor computation carries shapes inline, in this fixed format:

```
x           (B=32, S=128, d_model=4096)
W_q         (4096, 4096)                    # (out, in) — ALWAYS
q = x @ W_q.T                               (32, 128, 4096)
reshape     (32, 128, H=32, d_head=128)
transpose   (32, 32, 128, 128)              # (B, H, S, d_head)
scores = q @ k.T / sqrt(d_head)             (32, 32, 128, 128)   # (B, H, S_q, S_k)
```

**Non-negotiable. Shape errors are the #1 practical failure in real ML work**, and a course whose destination is
"fine-tune models" that does not drill shapes has not reached its destination. **Make shape-tracking a habit the
notation enforces, not a topic in a chapter.**

**Three shape facts to hammer wherever attention appears:**
1. **The output has $S$ rows, one per query.** Attention is a function *of the query set*; the key/value set may
   be any length. **This single fact makes cross-attention obvious.**
2. **Softmax is over the LAST axis** (the keys). **Row $i$ of $A$ sums to 1. Columns do not.** Learners read
   heatmaps backwards constantly.
3. **$d_{\text{head}}$ links $Q$ and $K$. $S$ links $K$ and $V$. $d_{\text{model}}$ never appears after the
   projections.** The operation is dimensionally rigid; there is exactly one legal wiring.

---

## 8. THE SYMBOL LEDGER — mandatory after any equation with >3 distinct symbols

Required format. **The "From" column is the part nobody writes and the part that carries the most** — it tells
the learner *where each thing came from*, which is precisely the information notation deletes.

> $$\boldsymbol\delta^{(\ell)} = \left(W^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}\right)\odot\phi'\!\left(\mathbf{z}^{(\ell)}\right)$$
>
> | Symbol | Is | Shape | From |
> |---|---|---|---|
> | $\boldsymbol\delta^{(\ell)}$ | how much the loss changes per unit change in layer $\ell$'s pre-activation | $(d_\ell, 1)$ | what we're solving for |
> | $W^{(\ell+1)}$ | next layer's weights | $(d_{\ell+1}, d_\ell)$ | forward pass |
> | $\boldsymbol\delta^{(\ell+1)}$ | the same quantity, one layer later | $(d_{\ell+1}, 1)$ | already computed — **this is the recursion** |
> | $\phi'(\mathbf{z}^{(\ell)})$ | slope of the activation at the value it actually took | $(d_\ell, 1)$ | **cached from the forward pass** — this is why training eats VRAM |
> | $\odot$ | elementwise product | — | — |
>
> **Shape check:** $(d_\ell, d_{\ell+1}) \times (d_{\ell+1}, 1) = (d_\ell, 1)$, then $\odot\,(d_\ell,1) \to (d_\ell,1)$. ✓

**Rule: the gradient always has the same shape as the thing it is the gradient of.** Universal law. It is why
`param.grad` has `param.shape`, it makes optimizer code trivially readable, and it is a debugging tool for life.

---

## 9. ANTI-PATTERNS — mechanically checkable. Any page failing these is a defect.

1. ❌ A symbol appears that is not in this table or the page's Ledger
2. ❌ $W$ written as (in, out) anywhere
3. ❌ **$\sigma$ used with an argument on a page that also uses $\sigma_t$** (see §5.2)
4. ❌ **$t$ used for sequence position or optimizer step**
5. ❌ **$k$ used for a vocabulary/class index**
6. ❌ $\theta$ used for a geometric angle
7. ❌ A bare $\alpha$, $\beta$, or $w$ with no subscript and no Ledger entry
8. ❌ An equation with >3 symbols and no Symbol Ledger
9. ❌ A tensor op with no shape ribbon
10. ❌ A worked example that does not terminate in an actual decimal
11. ❌ A demo that does not compute the page's equation live
12. ❌ A demo whose equation is not shown with live values substituted in
13. ❌ A quiz question answerable from vocabulary
14. ❌ A distractor with no named misconception behind it
15. ❌ Content in a collapsible that a later page depends on
16. ❌ A collapsible title that teases rather than states
17. ❌ A thread beat that does not announce itself as one
18. ❌ Code with elided imports or `# ...`
19. ❌ A justification appealing to "germane load"
20. ❌ Re-explaining something the learner demonstrably owns (the −0.428 expertise-reversal tax)
21. ❌ >4 pages since anything last ran
22. ❌ **A number that contradicts `constants.md`**
23. ❌ **A number presented as fact that `constants.md` tags [INF], [EST], or [MEA]**
24. ❌ A deviation from field convention (§3.2) with no Translation note on the page
25. ❌ `nn.Linear(2,2)` used to demonstrate the (out, in) convention

**#22 and #23 are the two a fan-out build will fail most often.** They are also the two that destroy trust
fastest with *this* learner, who owns the hardware and will check.

---

## 10. THE FOUR RECURRING THREADS — tag every beat

Every thread beat MUST be tagged in the spec: `[THREAD:TN1 beat 4/9]`. Each beat states (a) it is the same
object, (b) what is new about this encounter, (c) which prior beat to reopen. **Without explicit tagging,
fan-out agents will silently re-introduce the thread as if new, and the interleaving benefit — the entire reason
threads exist — evaporates. Assume they will. Make the tag mandatory and checkable.**

| Thread | Role | Recurs as |
|---|---|---|
| **The dot product** | **the atom** | neuron → matmul → similarity → attention scores → CFG direction → LoRA's $BA$ |
| **TN-1** | **the object** ← *the spine* | hand arithmetic → JS demo → PyTorch → autograd → one attention head → LoRA target |
| **The chain rule** | **the verb** | derivative → backprop → autograd graph → why activations are cached → the score |
| **The memory budget** | **the reality check** | 16 B/param → activations → KV cache → quantization → LoRA → "does it fit on *my* box?" |

**If forced to cut to one: keep TN-1.** It is the only one that is a *thing* rather than a *technique*, and the
transfer failure it cures ("toy examples that don't transfer") is cured specifically by re-meeting the **same
object** in new guises.

**Canonical visuals: one per object, reused course-wide, never redrawn.** TN-1's graph. The loss surface. The
shape ribbon. The $(a_t,b_t)$ route map. Same colours, same layout, every time. **A re-drawn diagram is a new
object to the reader even when it is the same object to the author — and in a fan-out build, every agent will
redraw it differently unless the spec ships the canonical asset.** Ship the assets, not the descriptions.
