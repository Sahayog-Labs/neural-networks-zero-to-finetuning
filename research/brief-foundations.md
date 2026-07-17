# Research Brief — MATHEMATICAL FOUNDATIONS + THE NEURON

> **⚠️ THIS BRIEF PREDATES TWO VERIFICATION PASSES. `constants.md`, `decisions.md`, `notation.md` OVERRIDE IT.**
> Load-bearing corrections a builder MUST apply here: **(1)** anchor is **Qwen3-8B**, not Llama-3.1-8B — every FFN/
> $d_{ff}$/$V$/$\ln V$/FFN-share/AdamW number is re-derived in `constants.md` (D-01). **(2)** Diffusion LR split
> full 1e-6–5e-6 / LoRA 1e-4 (§4, D-09). **(3)** SVD is NOT cut — trunk gets a full page (D-12). **(4)** "96 GB
> AdamW" → **131.05 GB** (D-03). **(5)** "~31 TFLOPS FP32" is unsourced — delete (D-10c). See `decisions.md` §D-20.

**For:** curriculum architect, ANN course (destination: learner fine-tunes LLMs / diffusion models on own data)
**Date:** 2026-07-16
**Scope:** the minimum math spine + the artificial neuron. Everything before backprop-as-algorithm, before training loops, before transformers.
**Learner:** adult, rusty-but-trained, runs an NVIDIA DGX Spark (GB10, 128 GB LPDDR5X unified, **273 GB/s** bandwidth, ~**1 PFLOP NVFP4** [sparse marketing peak — never size a run off it], ~~~31 TFLOPS FP32~~ **[UNSOURCED — DELETE, D-10c; the inferred dense-BF16 roofline is ~62–125 TF, unresolved, constants §6.3]**), does hands-on ComfyUI diffusion work. Teach at undergrad level. He does not need to be told what a variable is.

---

## 0. THE RUTHLESS CUT LIST — read this first

The single highest-value thing this brief can do is tell you what **not** to teach. A traditional "math for ML" prerequisite chain is 3–4 semesters. The load-bearing subset for *fine-tuning an LLM or a diffusion model* is roughly **12 concepts**. Everything else is cargo cult.

### KEEP (load-bearing — the whole rest of the course leans on these)

| # | Concept | Where it cashes out later |
|---|---|---|
| 1 | Function, composition, parameterized function | "a net is a big composed function" — the framing for everything |
| 2 | Vector / matrix / tensor **shape** and the multiply rule | every debugging session the learner will ever have |
| 3 | Broadcasting | silent-bug source #1 in PyTorch |
| 4 | Dot product as similarity/projection | attention scores `QKᵀ`, embedding similarity, CFG direction in diffusion |
| 5 | Derivative = local slope | learning-rate intuition |
| 6 | Partial derivative + gradient = steepest ascent | gradient descent, guidance in diffusion |
| 7 | **Chain rule** | backprop, LoRA, why gradients vanish, why activations get cached, why VRAM explodes |
| 8 | Distribution / expectation / sampling | dataloaders, minibatch noise, temperature sampling |
| 9 | Gaussian (1-D and isotropic multivariate) | diffusion forward process is *literally* this |
| 10 | log & exp: log-probs, log-space stability, logsumexp | softmax, cross-entropy, NLL loss, why logits not probs |
| 11 | Softmax + cross-entropy + why NLL is the loss | the LLM training objective, verbatim |
| 12 | Neuron = affine map + nonlinearity; linear collapse | why depth exists at all |

### CUT (traditionally taught, not load-bearing here) — with the honest reason

- **Limits, epsilon-delta, continuity proofs.** Never used. Derivative-as-slope is enough. *(If a learner asks "but is ReLU differentiable at 0?" the honest answer is "no, and PyTorch returns 0 there by convention and nobody cares" — that's a warning box, not a chapter.)*
- **Integration techniques** (by parts, substitution, trig substitution). Zero. The only integral in the course is `∫p(x)f(x)dx` as *notation for expectation*, and it is always estimated by sampling. Teach the notation, never the technique.
- **Determinants, eigenvalues, eigenvectors, SVD as a computed object.** Contentious cut — see §14. Verdict: **cut eigen-decomposition; keep one paragraph of SVD intuition, gated, in the LoRA chapter only**, because "low rank" needs a meaning. Do not build an eigen chapter in foundations. PCA is not needed.
- **Matrix inverse, Gaussian elimination, solving `Ax=b`, rank-nullity, vector-space axioms, basis/span formalism.** Nets never invert a matrix. Cut.
- **Cross product, curl, divergence, line/surface integrals, Jacobian determinants for change-of-variables.** Cut. (Normalizing flows would need the last one; this course does not teach normalizing flows.)
- **Lagrange multipliers, KKT, convex optimization theory.** The loss surface is non-convex; the theory doesn't apply. Cut.
- **Taylor series as a topic.** Keep exactly one instance: first-order Taylor `f(x+Δ) ≈ f(x) + f'(x)Δ` **is** the justification for the gradient step. Teach it as one line inside the gradient-descent section, not as a series chapter. Cut remainder terms, radius of convergence, everything.
- **Formal measure-theoretic probability, σ-algebras, Bayes' theorem drill problems.** Cut. Bayes shows up once, in one line, when explaining classifier-free guidance; teach it there, in context.
- **Information theory beyond one paragraph.** Entropy/KL get *exactly* the treatment needed to say "cross-entropy loss = NLL = KL(data‖model) + const". Do not teach mutual information, channel capacity, coding.
- **ODEs/SDEs as a formal subject.** Deliberate flag: the diffusion track in 2026 is written in probability-flow-ODE / SDE language (DDIM, DPM-Solver++, flow matching). **Recommendation: teach the sampler as a discrete update rule `x_{t-1} = f(x_t, ε̂, t)` and mention "in the literature this is written as an ODE" as a *reading key*, not a prerequisite.** The learner can drive ComfyUI samplers competently with the discrete view. *Reconcile with the diffusion brief — if it insists on continuous-time, foundations must add ~1 page on "derivative w.r.t. time" and Euler stepping, and that's the only extra.*
- **Complex numbers.** Cut (RoPE is teachable as a 2-D rotation matrix, which is high-school trig — and that's where his trig knowledge earns its keep).

### The trig payoff (say this explicitly — he was promised trig would be used)
Trig is used **twice** and only twice: (1) sinusoidal / RoPE positional encoding — a 2-D rotation `[[cosθ, −sinθ],[sinθ, cosθ]]`; (2) cosine similarity and the cosine LR schedule. That's it. Foundations should plant `cos θ = (a·b)/(‖a‖‖b‖)` early so the payoff is pre-loaded.

---

## 1. FUNCTIONS, COMPOSITION, AND "A NET IS ONE BIG FUNCTION"

**Intuition first (lead with this):** *A neural network is a single mathematical function with millions of adjustable knobs. Training is turning the knobs. Everything else is bookkeeping.*

### The one framing to establish

$$\hat{y} = f_\theta(x) = f^{(L)}\!\left(f^{(L-1)}\!\left(\cdots f^{(1)}(x)\right)\right)$$

- $x$ — input, a tensor. For an LLM: token IDs, shape `(B, T)`, dtype int64. For a diffusion U-Net: latent, shape `(B, 4, 64, 64)` for SD1.5-class VAEs, `(B, 16, 128, 128)` for SD3/Flux-class 16-channel VAEs.
- $\theta$ — **all** parameters, conceptually one flat vector. $\theta \in \mathbb{R}^P$, $P$ = parameter count. For Llama-3.1-8B, $P = 8.03\times10^9$.
- $f^{(l)}$ — layer $l$, itself $\text{nonlinearity}(W^{(l)}h^{(l-1)} + b^{(l)})$.
- $L$ — depth. Llama-3.1-8B: $L=32$ transformer blocks.
- $\hat{y}$ — output. LLM: logits `(B, T, V)`, $V=128{,}256$ for Llama-3. Diffusion: predicted noise $\hat{\varepsilon}$, same shape as the latent.

**The key mental move:** the learner is used to thinking of $x$ as the variable and $\theta$ as fixed (inference). Training **flips this**: $x$ is fixed (it's the data), $\theta$ is the variable, and the function we differentiate is $\mathcal{L}(\theta)$. State this flip explicitly and early — it is the conceptual pivot of the whole course.

$$\mathcal{L}: \mathbb{R}^P \to \mathbb{R}$$

One number out. $P$ numbers in. **8 billion inputs, one output.** That asymmetry is *exactly why* reverse-mode autodiff (backprop) exists and why forward-mode doesn't — plant the seed here, pay it off in the backprop chapter.

### Misconceptions
- ❌ *"The network 'looks up' or 'stores' the answer."* → It's a function evaluation, deterministic given $\theta$ and $x$. Any randomness in an LLM is *added afterward* at the sampling step, not inside the net. **Correction that lands:** run the same prompt at `temperature=0` twice — bit-identical. Then at `temperature=0.8` — different. The net didn't change.
- ❌ *"Layers are different kinds of things."* → Every layer is the same shape of thing: affine + nonlinearity. Attention is the one genuine structural addition, and even it is built from matmuls.
- ❌ *"Composition means the layers are separate programs."* → They are one function. There is no boundary. The chain rule doesn't respect layer boundaries either.

### Demo: **Function Machine Chain**
Three boxes in a row, each computing $f_i(x) = a_i x + b_i$ or $\tanh(a_i x + b_i)$, toggleable. Sliders for $a_1,b_1,a_2,b_2,a_3,b_3$. Plot the composed curve $f_3(f_2(f_1(x)))$ over $x\in[-3,3]$ live, plus tiny inset plots of each stage's intermediate output.
**Exact math for JS:** evaluate on 200 sample points, `y = x.map(v => f3(f2(f1(v))))`. Nothing more.
**Insight to engineer:** with all boxes linear, *no matter how the six sliders move, the composed plot is always a straight line*. Toggle one `tanh` on and the curve bends. This demo is the setup; §11's linear-collapse demo is the punchline. **Build them as the same widget with a "reveal" step** — reusing the widget is worth more than two separate widgets.

---

## 2. VECTORS, MATRICES, TENSORS — SHAPE IS THE SKILL

**Intuition first:** *A tensor is a rectangular box of numbers plus a tuple saying how it's shaped. 90% of practical deep-learning debugging is arguing with that tuple.*

Say this bluntly: **shape bookkeeping is the #1 practical skill in this course.** Not calculus. Shapes. Every error message he will ever see in fine-tuning is a shape error or an OOM.

### Definitions with shapes and units

- **Scalar** — shape `()`, rank 0.
- **Vector** $v \in \mathbb{R}^{d}$ — shape `(d,)`, rank 1. Example: one token embedding, $d=4096$ in Llama-3.1-8B.
- **Matrix** $W \in \mathbb{R}^{m \times n}$ — shape `(m, n)`, rank 2.
- **Tensor** — rank ≥ 3. LLM activation: `(B, T, d)` = (batch, sequence, model dim). Image batch: `(B, C, H, W)`.

### The multiply rule (the one rule to memorize)

$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}, \qquad A \in \mathbb{R}^{m\times n},\; B \in \mathbb{R}^{n\times p} \;\Rightarrow\; AB \in \mathbb{R}^{m\times p}$$

**Teach it as a shape-cancellation game:** `(m, n) @ (n, p) → (m, p)`. The inner dims must match and they *annihilate*. Write it on the board as `(m, [n) @ (n], p)`. This visual sticks.

**Cost:** $m\cdot n\cdot p$ multiply-adds = $2mnp$ FLOPs (count mul and add separately, as the hardware-vendor convention does). Establish this counting rule here — the FLOP budget chapters later depend on it.

### Worked example, carried to a number — ~~**the Llama-3.1-8B FFN**~~ → **REWRITE ON Qwen3-8B (D-01)**

> **⛔ RETIRED ANCHOR — every number in this worked example is a Llama-3.1-8B figure and must be re-derived on
> Qwen3-8B (`constants.md` §1.2/§1.3, the authoritative derivation). Substitutions:** $d_{ff}$ **14336 → 12288**
> ($=3d$, not 3.5d); $V$ **128,256 → 151,936**; $L$ **32 → 36**; gate/up/down each **58,720,256 → 50,331,648**;
> FFN across the model **5.637B → 5.436B**; FFN share **"~70%" → 66.4% of the model / 78.26% of a block** (say
> which); full-AdamW teaser **"~96 GB" → 131.05 GB** (D-03). The SwiGLU 2/3-rule payoff is *better* on Qwen3:
> 12288 = 3d is closer to the honest 2.67d than Llama's 3.5d, so "why not 16384?" lands harder. Numbers below are
> the OLD anchor — do not print them.

Llama-3.1-8B [RETIRED]: $d_{\text{model}}=4096$, $d_{\text{ff}}=14336$, $L=32$, $V=128{,}256$, heads $=32$, KV heads $=8$ (GQA), head_dim $=128$.

One SwiGLU FFN block has three matrices: `gate_proj (4096→14336)`, `up_proj (4096→14336)`, `down_proj (14336→4096)`.

- gate: $4096 \times 14336 = 58{,}720{,}256$ params
- up: same $= 58{,}720{,}256$
- down: $14336 \times 4096 = 58{,}720{,}256$
- **FFN total per block: $176{,}160{,}768 \approx 176.2$M params**
- Across 32 blocks: $\approx 5.637$B params — **~70% of the whole 8B model is FFN.** This number should recur; it's the reason "the MLP is where the knowledge lives" is more than a slogan, and the reason LoRA-ing only attention leaves most of the model untouched.

Forward FLOPs for one token through one FFN: $2 \times 3 \times 4096 \times 14336 = 352{,}321{,}536 \approx 3.52\times10^8$ FLOPs.

**Memory of one such matrix:** $58{,}720{,}256$ params × 2 bytes (bf16) = $117{,}440{,}512$ B = **112 MiB**. Three of them = 336 MiB per block. ×32 = **10.5 GiB just for FFN weights in bf16.** Whole 8B model in bf16 ≈ **16.1 GB**. On the 128 GB DGX Spark this is the anchor number: *the weights are the cheap part; the optimizer states and activations are what kill you.* (Full AdamW fine-tune of 8B: 16 GB weights + 16 GB grads + 64 GB Adam moments in fp32 = **~96 GB before a single activation** — this is the number that motivates LoRA, and it should be introduced here in foundations as a teaser and paid off later.)

### Broadcasting

**Intuition first:** *Broadcasting is NumPy/PyTorch silently copying a smaller tensor to match a bigger one, so you can add a bias vector to a whole batch without writing a loop. It is a convenience that will eventually cost you a day of debugging.*

**Rule (state precisely):** align shapes from the **right**. For each dim: they're compatible if equal, or one of them is 1 (that one is stretched), or one is missing (treated as 1).

```
(32, 128, 4096)   activations
(          4096)  bias
--------------------------- → (32, 128, 4096)   ✓
```

**The classic disaster:**
```python
pred = torch.randn(64)        # (64,)
target = torch.randn(64, 1)   # (64, 1)
loss = ((pred - target)**2).mean()   # NO ERROR. Shape (64,64). Loss is garbage.
```
This runs. It trains. It converges to nonsense. **This is the single most valuable warning box in the foundations chapter.** The fix is `target.squeeze()` or `pred.unsqueeze(1)`, and the *habit* is: `assert pred.shape == target.shape` before every loss.

### Misconceptions
- ❌ *"`(64,)` and `(64,1)` and `(1,64)` are the same thing."* → They are three different tensors with three different broadcast behaviors. This is *the* rusty-engineer trap.
- ❌ *"Matrix multiply is elementwise-ish."* → `A*B` (Hadamard, elementwise) and `A@B` (matmul) are different operators and both are legal on some shapes. SwiGLU uses both, on purpose, in one line.
- ❌ *"Transposing is free / is a real operation."* → In PyTorch it's a stride change, no data moves. Relevant later when a kernel demands `.contiguous()`.
- ❌ *"The batch dimension is mathematically meaningful."* → It's a hardware convenience. The math treats each batch element independently (until BatchNorm — flag that BatchNorm is the exception, and that it's largely absent from modern LLMs/diffusion, which use LayerNorm/RMSNorm precisely because they don't couple across batch).

### Demo: **Shape Detective**
A drag-and-drop where the learner assembles a chain of ops (`Linear(in,out)`, `reshape`, `transpose`, `@`, `+`) and a live shape tracer shows the tensor shape after each op, going red with the actual PyTorch error message when it breaks.
**Exact math for JS:** implement only the shape algebra, not the numbers. `matmul(a, b)`: check `a[-1] === b[-2]`, output `[...batchBroadcast(a.slice(0,-2), b.slice(0,-2)), a[-2], b[-1]]`. `broadcast(a,b)`: right-align, per-dim `max` if one is 1, else error.
**Insight:** shapes are a little type system, and he can run it in his head. Include a "silent broadcast" mode that highlights in **amber** (not red) any op that *succeeded via broadcasting* — the amber highlight is the lesson: the dangerous ops are the ones that don't error.

---

## 3. THE DOT PRODUCT — SIMILARITY AND PROJECTION

**Intuition first:** *The dot product is one number that answers "how much do these two vectors agree?" Big positive = same direction. Zero = unrelated/perpendicular. Negative = opposed.* This is the most reused single idea in the course.

$$a\cdot b = \sum_{i=1}^{d} a_i b_i = \|a\|\,\|b\|\cos\theta$$

- $a, b \in \mathbb{R}^d$, shape `(d,)`
- $\|a\| = \sqrt{\sum_i a_i^2}$, the L2 norm — length
- $\theta$ — angle between them. **This is where his trig gets used.**

**Projection form:** $\text{proj}_b(a) = \dfrac{a\cdot b}{\|b\|^2}\,b$ — "the shadow $a$ casts on $b$'s direction."

**Cosine similarity:** $\text{cossim}(a,b) = \dfrac{a\cdot b}{\|a\|\|b\|} \in [-1, 1]$.

### Why this recurs — say all four out loud, now, in foundations
1. **A neuron is a dot product.** $z = w\cdot x + b$ — a neuron measures *how much the input resembles the pattern $w$*. This reframing is the whole point of §10.
2. **Attention is dot products.** $\text{score}_{ij} = \dfrac{q_i \cdot k_j}{\sqrt{d_k}}$ — "how much does token $i$'s question match token $j$'s label?"
3. **Embeddings are compared by dot product.** RAG, retrieval, the "king − man + woman" thing.
4. **The output logit is a dot product.** $\text{logit}_v = h \cdot E_v$ — the final hidden state dotted against every row of the vocab embedding matrix. "Which word does my thought most resemble?"

### Worked example — carried to a number
$a = [3, 4]$, $b = [4, 3]$.
$a\cdot b = 12+12 = 24$. $\|a\| = \sqrt{9+16}=5$. $\|b\| = 5$.
$\cos\theta = 24/25 = 0.96 \Rightarrow \theta = \arccos(0.96) \approx 16.26°$. Nearly aligned. ✓

Now $c = [-4, 3]$: $a\cdot c = -12 + 12 = 0 \Rightarrow \theta = 90°$. Orthogonal. **The "unrelated concepts are perpendicular" intuition starts here** and is the seed of superposition later.

### The $\sqrt{d_k}$ preview (do this in foundations — it's a probability+dot-product fact, not an attention fact)
If $q, k \in \mathbb{R}^{d_k}$ have i.i.d. entries with mean 0, variance 1, then
$$\mathbb{E}[q\cdot k] = 0, \qquad \text{Var}(q\cdot k) = d_k$$
so $q\cdot k$ has std $\sqrt{d_k}$. With $d_k = 128$ (Llama-3 head dim), typical scores are ~±11. Push ±11 through softmax and you get a near-one-hot distribution with near-zero gradient. Dividing by $\sqrt{128}\approx 11.31$ restores unit variance. **This is a two-line derivation the learner can do himself, and it explains a magic constant he's seen in every transformer diagram.** High pedagogical ROI — put it in foundations, reference it from attention.

### Misconceptions
- ❌ *"High dot product = similar magnitude."* → It's magnitude AND angle entangled. Cosine similarity strips magnitude; dot product doesn't. This matters: attention uses raw dot product (magnitude matters, it's a "confidence"), RAG usually uses cosine (magnitude is noise).
- ❌ *"In high dimensions, intuition from 2-D carries over."* → It doesn't, and the failure is *useful*: two random unit vectors in $\mathbb{R}^{4096}$ have cosine similarity ~$\mathcal{N}(0, 1/4096)$, i.e. std ≈ 0.0156 — **essentially always orthogonal**. That's why 4096 dims can hold vastly more than 4096 "concepts" (Johnson–Lindenstrauss / superposition). This is a genuinely delightful fact for this learner. Demo it.

### Demo: **Dot Product Dial**
Left panel: 2-D, drag vector $a$ (fixed at $[1,0]$) and $b$ freely. Live readout of $a\cdot b$, $\|a\|\|b\|$, $\cos\theta$, $\theta$ in degrees, and a drawn projection line (the "shadow").
Right panel: dimension slider $d \in \{2, 8, 64, 512, 4096\}$. Sample 5000 pairs of random unit vectors in $\mathbb{R}^d$, histogram their cosine similarity, overlay the theoretical $\mathcal{N}(0, 1/d)$.
**Exact math for JS:** random unit vector = sample $d$ Gaussians (Box–Muller: `u1,u2 ~ U(0,1)`; `z = sqrt(-2*ln(u1))*cos(2*PI*u2)`), then divide by norm. Cosine = dot of two such.
**Insight:** *drag the dimension slider from 2 to 4096 and watch the histogram collapse to a spike at zero.* "In 2-D, random vectors are all over the place. In 4096-D, everything is perpendicular to everything. That's not a bug; that's the storage capacity."

---

## 4. DERIVATIVES — REBUILT FROM SLOPE

**Intuition first:** *The derivative is "if I nudge the input a tiny bit, how much does the output move, and in which direction?" That's it. It's a sensitivity number.*

$$f'(x) = \frac{df}{dx} = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}$$

Teach the limit as *notation for "when h is small enough that shrinking it further doesn't change the answer"*. Do not do epsilon-delta. He does not need it and it will cost you a page and his goodwill.

**Units matter, say them:** if $\mathcal{L}$ is in nats (loss) and $w$ is dimensionless (a weight), then $\partial\mathcal{L}/\partial w$ is in nats-per-unit-weight. The learning rate $\eta$ then has units of weight²/nats. This sounds pedantic but it's the honest answer to "why is $\eta=3\times10^{-4}$ and not 0.5?" — **because the units are set by the scale of the loss and the weights, and both are conventions.** This defuses a lot of magic.

### The only derivatives he needs (a table, not a chapter)

| $f(x)$ | $f'(x)$ | Where used |
|---|---|---|
| $c$ | $0$ | |
| $x^n$ | $nx^{n-1}$ | MSE loss ($n=2$) |
| $e^x$ | $e^x$ | softmax, Gaussians |
| $\ln x$ | $1/x$ | cross-entropy |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | sigmoid gate, and the vanishing-gradient story |
| $\tanh x$ | $1-\tanh^2 x$ | old RNNs |
| $\max(0,x)$ | $\mathbb{1}[x>0]$ (0 or 1) | ReLU — **note the derivative is a switch, not a scale** |

That table is complete. Everything else in the course is these seven plus the chain rule plus matrix bookkeeping.

**Derive $\sigma'$ in full**, because it's the one derivation that pays for itself:
$\sigma(x)=(1+e^{-x})^{-1}$, so $\sigma'(x) = -(1+e^{-x})^{-2}\cdot(-e^{-x}) = \dfrac{e^{-x}}{(1+e^{-x})^2} = \sigma(x)\bigl(1-\sigma(x)\bigr)$.
Max value at $x=0$: $\sigma(0)=0.5$, so $\sigma'(0)=0.25$. **Remember 0.25.** It's the number that kills deep sigmoid networks (§12).

### Partial derivatives and the gradient

**Intuition first:** *You're standing on a hillside in fog. The partial derivatives are "how steep is it if I step exactly north?" and "how steep is it if I step exactly east?" The gradient is the arrow pointing straight uphill, assembled from those two.*

$$\nabla_\theta \mathcal{L} = \left[\frac{\partial\mathcal{L}}{\partial\theta_1}, \frac{\partial\mathcal{L}}{\partial\theta_2}, \ldots, \frac{\partial\mathcal{L}}{\partial\theta_P}\right]^\top \in \mathbb{R}^P$$

**Same shape as $\theta$.** Say this loudly — it's why `param.grad` has `param.shape` in PyTorch, and it's the thing that makes optimizer code trivially readable.

**Why "steepest ascent" is true (one line, worth doing):** the directional derivative in unit direction $u$ is $\nabla f\cdot u = \|\nabla f\|\cos\theta$, maximized when $\cos\theta=1$, i.e. $u \parallel \nabla f$. **The proof is the dot product from §3.** Reusing §3 here is worth a lot — it shows the math spine is actually a spine.

**Gradient descent, with the Taylor justification inline:**
$$f(\theta + \Delta) \approx f(\theta) + \nabla f\cdot\Delta \quad\Rightarrow\quad \text{pick } \Delta = -\eta\nabla f \quad\Rightarrow\quad f \text{ decreases by } \approx \eta\|\nabla f\|^2$$
$$\theta_{t+1} = \theta_t - \eta\,\nabla_\theta \mathcal{L}(\theta_t)$$
- $\eta$ — learning rate, scalar. **Real numbers to plant now and reuse forever:** full fine-tune of an 8B LLM: $\eta \approx 1\text{–}2\times10^{-5}$. LoRA fine-tune: $\eta \approx 1\text{–}2\times10^{-4}$ (10× higher — flag it, explain later that it's because LoRA's B is zero-init and the effective update is scaled by $\alpha/r$). Pretraining from scratch: $\eta \approx 3\times10^{-4}$ with warmup + cosine decay. Diffusion fine-tune: ~~$\eta \approx 1\times10^{-5}$ to $1\times10^{-6}$~~ **← CORRECTED (D-09 / constants §9.4): that is the FULL fine-tune figure only. Split it: diffusion FULL FT / DreamBooth $\eta \approx 1\times10^{-6}$ to $5\times10^{-6}$; diffusion LoRA $\eta \approx 1\times10^{-4}$ — a 100× difference.** Applying 1e-5/1e-6 to a diffusion LoRA trains ~100× too slow, and the noise-dominated loss curve won't reveal it (the learner burns days). Primary source: diffusers `README.md`: *"When using LoRA we can use a much higher learning rate... 1e-4 instead of the usual 2e-6."* The single-source-of-truth table lives on the memory-ledger page (D-09), generated by the principle **fewer trainable parameters ⇒ higher LR**. Adam's $\varepsilon = 10^{-8}$, $\beta_1=0.9$, $\beta_2=0.999$ (or 0.95 for large-scale LLM pretraining — genuine practitioner split, flag it).

### Misconceptions
- ❌ *"The gradient points toward the minimum."* → **No.** It points *locally* uphill; the negative points locally downhill. In a curved valley the negative gradient points at the *wall*, not the exit. This is exactly why momentum and Adam exist. **This is the misconception to attack hardest** — it's the one that makes the whole optimizer chapter feel arbitrary if left uncorrected.
- ❌ *"Bigger gradient = closer to / further from the answer."* → Gradient magnitude says nothing about distance to the minimum. Zero gradient means flat, which could be a minimum, a maximum, a saddle, or a plateau. In $\mathbb{R}^{8\times10^9}$, **saddle points massively outnumber local minima** — flag this as the modern understanding that overturned the old "local minima are the problem" folklore (Dauphin et al. 2014, and the field has broadly accepted it; genuine consensus, not controversy).
- ❌ *"Gradient descent finds the global minimum."* → It does not, it is not trying to, and — genuinely surprising — **it doesn't matter.** Flag as a real open question: *why* SGD on a wildly non-convex 8B-dim surface reliably finds solutions that generalize is **not settled** in 2026. Implicit regularization, the lottery-ticket view, the loss-landscape-connectivity view, and the NTK/feature-learning views all have adherents and none is decisive. Do not paper over this. The learner is an engineer; he will respect "we don't fully know why this works, but here is what is reliably true empirically" far more than a fake explanation.

### Demo: **Slope Explorer / LR Playground**
Plot $f(x) = x^4 - 3x^2 + x + 2$ (has two minima and a local max — good). Draggable point on the curve, live tangent line drawn, readout of $f(x)$ and $f'(x) = 4x^3 - 6x + 1$. Second panel: a "Run gradient descent" button with an LR slider, log scale, $\eta \in [10^{-4}, 1]$, and a starting-$x$ slider; animate the iterates as dots with trails.
**Exact math for JS:** literally `x = x - lr * (4*x**3 - 6*x + 1)` in a loop, 200 steps, record trajectory.
**Insight to engineer, in three drags:** (a) $\eta = 0.01$ → slow, smooth, converges; (b) $\eta = 0.1$ → fast, converges; (c) $\eta = 0.35$ → **oscillates and diverges to NaN.** *He has now personally seen a loss go NaN from an LR that was only 3.5× too big.* Then: change the starting point from $x=-2$ to $x=+2$ and land in a *different* minimum with the same LR. "Same function, same optimizer, different answer. Initialization matters." Add a NaN counter that displays `Inf` in red — it should look exactly like a real training log.

---

## 5. THE CHAIN RULE — THE MOST LOAD-BEARING IDEA IN THE COURSE

Treat this as a full section. If the learner leaves with only one thing, it's this.

**Intuition first:** *Sensitivities multiply along a chain. If turning knob A moves B by 3×, and moving B moves C by 2×, then turning A moves C by 6×. That's the whole idea. Backprop is this, applied a few million times, in the efficient order.*

Second intuition (the gears one — use it, it's better than the "rate" one for this learner): *gears. A 3:1 gear feeding a 2:1 gear gives 6:1 overall. The chain rule is gear ratios. And a gear with ratio 0.25 in the chain — that's a sigmoid — means everything upstream barely turns at all.* This single image pre-loads vanishing gradients.

### Single variable
$$\frac{d}{dx}f(g(x)) = f'(g(x))\cdot g'(x)$$

### Multivariable / multipath — **this is the version backprop actually uses**
$$\frac{\partial \mathcal{L}}{\partial x} = \sum_{i} \frac{\partial \mathcal{L}}{\partial u_i}\cdot\frac{\partial u_i}{\partial x}$$
where $u_1,\dots,u_k$ are all the intermediate quantities that $x$ directly feeds.

**Say the rule in English and make him repeat it:** *"To find how a knob affects the loss: for every path from the knob to the loss, multiply the sensitivities along the path; then add up all the paths."* **Multiply along paths, add across paths.** That sentence *is* backpropagation. Everything in the autodiff chapter is an efficiency argument about the order of operations, not a new idea.

The "add across paths" half is the one everybody forgets, and it's exactly what makes residual connections work (two paths: one through the block, one straight through with derivative exactly **1** — the gradient superhighway).

### Worked example, carried all the way to numbers

One neuron, one input, MSE loss. $x = 2$, $w = 0.5$, $b = 0.1$, target $y = 1$.

Forward:
1. $z = wx + b = 0.5(2) + 0.1 = 1.1$
2. $a = \sigma(z) = 1/(1+e^{-1.1}) = 1/(1+0.33287) = 0.75026$
3. $\mathcal{L} = (a-y)^2 = (0.75026 - 1)^2 = (-0.24974)^2 = 0.062370$

Backward — multiply along the path:
1. $\dfrac{\partial\mathcal{L}}{\partial a} = 2(a-y) = 2(-0.24974) = -0.49948$
2. $\dfrac{\partial a}{\partial z} = \sigma(z)(1-\sigma(z)) = 0.75026 \times 0.24974 = 0.18739$
3. $\dfrac{\partial z}{\partial w} = x = 2$; $\dfrac{\partial z}{\partial b} = 1$; $\dfrac{\partial z}{\partial x} = w = 0.5$

Chain:
$$\frac{\partial\mathcal{L}}{\partial w} = (-0.49948)(0.18739)(2) = \mathbf{-0.18718}$$
$$\frac{\partial\mathcal{L}}{\partial b} = (-0.49948)(0.18739)(1) = \mathbf{-0.09359}$$

Update with $\eta = 0.5$: $w \leftarrow 0.5 - 0.5(-0.18718) = \mathbf{0.59359}$, $b \leftarrow 0.1 + 0.04680 = \mathbf{0.14680}$.

Check it worked: $z' = 0.59359(2)+0.14680 = 1.33398$; $a' = \sigma(1.33398) = 0.79155$; $\mathcal{L}' = (0.79155-1)^2 = \mathbf{0.043452}$. Down from 0.062370. **The loss went down. He did that by hand.**

**Have him verify it in PyTorch.** This is the moment the course earns trust:
```python
import torch
x = torch.tensor(2.0)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
y = torch.tensor(1.0)
z = w*x + b
a = torch.sigmoid(z)
loss = (a - y)**2
loss.backward()
print(w.grad, b.grad)   # tensor(-0.1872) tensor(-0.0936)
```
Same numbers. **Autograd is not magic; it is the arithmetic you just did.** This is the single most important trust-building moment in the entire course — protect it, and make sure the printed numbers match the brief to 4 decimals.

### Misconceptions (this list matters more than most)
- ❌ *"Backprop is a different thing from the chain rule."* → It is the chain rule + memoization + a specific traversal order (reverse topological). Nothing else.
- ❌ *"The gradient flows backward through the same weights."* → It flows through $W^\top$, not $W$. Shape-check it: forward `(B,d_in) @ (d_in,d_out) → (B,d_out)`; backward `(B,d_out) @ (d_out,d_in) → (B,d_in)`. **The transpose is forced by the shapes, not chosen.** Good moment to reuse §2.
- ❌ *"Autograd differentiates my Python code symbolically."* → No. It records the *ops that actually ran* on this input into a graph (that's why `if` statements and Python loops just work), then walks it backward. Explains why `.detach()`, `torch.no_grad()`, and in-place ops matter.
- ❌ *"Intermediate activations are discarded after the forward pass."* → **They are kept**, because $\partial a/\partial z$ needs $z$. This is *why* VRAM scales with batch × sequence × depth and *why* gradient checkpointing (recompute instead of store) is a real speed/memory knob he will actually turn on the DGX Spark. **Land this connection in foundations, not later** — it converts an abstract math fact into a hardware fact he already half-knows.
- ❌ *"Vanishing gradients are about the network being deep."* → Depth is necessary but not sufficient. The cause is **repeated multiplication by factors < 1**. Sigmoid's max derivative is 0.25; ten layers gives $0.25^{10} \approx 9.5\times10^{-7}$. With $\eta=10^{-3}$ the first layer moves by $\sim10^{-9}$ per step — **it is frozen.** ReLU's derivative is exactly 1 on the positive side, so the product is 1, not $10^{-7}$. **That is the real reason ReLU displaced sigmoid** — not "it's faster to compute", which is the folk answer and is a rounding error by comparison. (§12.)

### Demo: **Live Chain-Rule Visualizer** ⭐ flagship demo of this chapter
Render the computation graph of the worked example above as nodes: `x → [×w] → [+b] → z → [σ] → a → [(·−y)²] → L`. Each node shows its live forward value. Sliders for $w$, $b$, $x$, $y$.
- **Forward mode:** values propagate left→right, animate on slider drag.
- **Backward mode:** press a button; each **edge** lights up displaying its local partial (the "gear ratio"), and each **node** displays the accumulated $\partial\mathcal{L}/\partial(\text{node})$. Edge thickness ∝ |local partial|. The running product is displayed as a literal multiplication string that builds up: `-0.4995 × 0.1874 × 2 = -0.1872`.
- **Depth slider:** stack $N \in [1,12]$ sigmoid layers. Display $\partial\mathcal{L}/\partial w_1$ in scientific notation. **Watch it die.** Toggle σ→ReLU and watch it not die. Toggle "add residual connections" and watch the gradient hold at ~1 even with sigmoid — because the skip path contributes a $+1$ term to the sum.
**Exact math for JS:** hardcode the forward ops and their local derivative formulas; do a real reverse topological traversal with accumulation (`node.grad += upstream * localPartial`). ~80 lines. Do **not** fake it with precomputed values — the accumulation with `+=` over multiple paths is exactly the thing the learner must see.
**Insight:** three separate ones, in order: (1) backprop is arithmetic; (2) the gradient is a product and products of small things vanish; (3) residual connections are an *addition* in the backward sum, which is why they fix it. Insight (3) is worth the whole widget.

---

## 6. JUST ENOUGH PROBABILITY

**Intuition first:** *A probability distribution is a shape describing which values are likely. Expectation is the balance point. Sampling is reaching in and grabbing one.*

### Distribution, expectation, variance
$$\mathbb{E}[X] = \sum_x x\,p(x) \quad\text{or}\quad \int x\,p(x)\,dx$$
$$\text{Var}(X) = \mathbb{E}[(X-\mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

**The one thing to hammer: expectations are estimated by averaging samples.**
$$\mathbb{E}[f(X)] \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i), \qquad x_i \sim p$$
**This is what a minibatch IS.** The true loss is an expectation over the whole dataset; a minibatch of $B=32$ is a 32-sample Monte Carlo estimate of it. The standard error of that estimate falls as $1/\sqrt{B}$: going $B=32\to128$ (4×) cuts gradient noise only 2×, for 4× the compute. **That's the honest reason big batches have diminishing returns**, and it's a two-line argument. It also frames "SGD noise" correctly: the noise isn't a defect of the algorithm, it's sampling error, and it turns out to help generalization (mechanism genuinely debated — flag it).

### The Gaussian — do this properly, diffusion is built from it

$$p(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- $\mu$ — mean, same units as $x$. $\sigma$ — std, same units. $\sigma^2$ — variance.
- $\pm1\sigma$: 68.3%. $\pm2\sigma$: 95.4%. $\pm3\sigma$: 99.7%.

**Isotropic multivariate Gaussian** (the only multivariate case needed):
$$\mathcal{N}(x;\mu,\sigma^2 I), \qquad x,\mu\in\mathbb{R}^d$$
"Isotropic" = a round fuzzy ball, same spread in every direction, no correlations. **This is the only covariance structure the course needs**, which is exactly why we can cut eigen-decomposition: $\Sigma = \sigma^2 I$ has no interesting structure.

**The reparameterization trick — teach it HERE, in foundations:**
$$x \sim \mathcal{N}(\mu,\sigma^2) \iff x = \mu + \sigma\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,1)$$
**Intuition:** *sample the randomness once, from a fixed standard bell curve, then shift and stretch it deterministically.* This is (a) how you sample a Gaussian in code, (b) how VAEs get gradients through a sampling step, and (c) **literally the diffusion forward process**:
$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$$
The learner should meet $\mu + \sigma\varepsilon$ in foundations so that when the diffusion brief's forward equation appears, **he recognizes it as an old friend with new letters.** *Coordinate this with the diffusion brief — this is the single highest-value handoff in the whole course.*

**The concentration-of-measure fact (worth including, it's counterintuitive and load-bearing for diffusion):** a sample from $\mathcal{N}(0, I_d)$ in $d=4096$ has expected norm $\approx\sqrt{d} = 64$, with std $\approx 1/\sqrt{2}\approx0.71$. So **essentially no sample is near the origin**, even though the origin is the highest-density point. High-dimensional Gaussians are thin shells, not balls. This explains why "the model just outputs the mean" is not what happens, and why diffusion starting from pure noise starts from a *shell*.

### Misconceptions
- ❌ *"Everything is Gaussian / the CLT means my data is Gaussian."* → No. The Gaussian is used in diffusion because it's *chosen* — it's closed under addition (sum of Gaussians is Gaussian, which is exactly what makes the closed-form $x_t$ from $x_0$ possible) and its score has a clean form. It's a design decision, not a discovered fact about images.
- ❌ *"More samples = the mean gets more accurate, linearly."* → $1/\sqrt{N}$. 100× the samples = 10× the precision.
- ❌ *"Variance and std are interchangeable."* → Units. $\sigma$ has $x$'s units; $\sigma^2$ has $x$'s units squared. Matters when reading $\beta_t$ schedules (variances) vs $\sigma_t$ (stds) in diffusion papers, which use both and rarely say which.
- ❌ *"Sampling from a model = the model is uncertain."* → Sampling is a decision procedure applied to the model's output distribution. The distribution itself is deterministic.

### Demo: **Sampling Bench**
Panel A: a Gaussian PDF with $\mu$, $\sigma$ sliders, and a "draw 1 / draw 100 / draw 10000" button that drops samples as rug ticks and builds a histogram. Live readout of sample mean vs true $\mu$, and the shrinking $\pm 1.96\sigma/\sqrt{N}$ CI band.
Panel B: the same distribution built by $\mu + \sigma\varepsilon$, with $\varepsilon$ drawn from a *fixed* standard normal shown in its own little plot, and arrows showing the shift-and-stretch. Slider on $\sigma$ from 0 to 3 — watch the SAME $\varepsilon$ samples stretch.
Panel C: dimension slider $d$; histogram of $\|x\|$ for $x\sim\mathcal{N}(0,I_d)$, with $\sqrt{d}$ marked.
**Exact math for JS:** Box–Muller for the normals (seed it so Panel B's "same $\varepsilon$" claim is literally true). PDF eval directly from the formula.
**Insight:** Panel B is the one that matters: *the randomness lives in $\varepsilon$; $\mu$ and $\sigma$ are just a deterministic affine map applied to it. That's why you can backprop through a "random" node.* Panel C: *in 4096-D the bell curve is a soap bubble.*

---

## 7. LOGS AND EXPONENTIALS — AS ACTUALLY USED

**Intuition first:** *Logs turn multiplication into addition and turn tiny numbers into manageable negative ones. Neural nets multiply a lot of probabilities together, and floating point can't hold the result. Logs are the fix. They are not a mathematical nicety; they are a numerical necessity.*

Only these identities. Nothing else:
$$\log(ab) = \log a + \log b, \qquad \log(a^n) = n\log a, \qquad e^{\log x} = x, \qquad \log_b x = \frac{\ln x}{\ln b}$$

Convention: **$\log = \ln$ (natural log) always, in ML, unless explicitly stated.** Loss is therefore in **nats**, not bits. Divide by $\ln 2 = 0.6931$ for bits. *(Perplexity $= e^{\mathcal{L}}$; a loss of 2.0 nats = perplexity 7.39 = "the model is as confused as if picking uniformly among 7.4 words.")*

### Worked example — underflow, carried to a number
A 500-token sequence, each token predicted with probability 0.1. The sequence probability is
$$\prod_{i=1}^{500} 0.1 = 10^{-500}$$
Smallest normal float32 is $\approx 1.18\times10^{-38}$; smallest subnormal $\approx 1.4\times10^{-45}$. **$10^{-500}$ is exactly 0.0 in float32.** In bf16 (which has float32's exponent range, ~$10^{-38}$) it's also exactly 0.0. Then `log(0) = -inf`, and `-inf * 0 = nan`, and your run is dead.

In log space:
$$\sum_{i=1}^{500}\ln(0.1) = 500 \times (-2.302585) = -1151.29$$
A perfectly ordinary float. **This is not an optimization, it's the difference between working and not working.** Show him the `nan`.

### The log-sum-exp trick (he will meet this in every framework's source)
$$\log\sum_i e^{z_i} = m + \log\sum_i e^{z_i - m}, \qquad m = \max_i z_i$$
Why: if $z_i = 800$, $e^{800}$ overflows float32 (max $\approx 3.4\times10^{38}$, and $e^{88.7}$ already overflows). Subtracting the max makes the largest exponent exactly $e^0 = 1$, and the rest are $\le 1$ so they can only *underflow to 0*, which is harmless. **The identity is exact, not an approximation** — that's the elegant part, and worth stating, because the learner will assume it's a hack.

This is why `F.cross_entropy(logits, targets)` takes **logits** and not probabilities: it fuses softmax+log+NLL into one numerically stable kernel. **Using `log(softmax(x))` as two separate ops is a real bug**, and it's a bug that only shows up on some inputs. Warning box.

### Misconceptions
- ❌ *"log-probs are just probs in different clothes."* → They live on a scale where addition is the natural operation and there's no underflow floor. The change of scale IS the point.
- ❌ *"Negative loss values mean something's broken."* → For discrete cross-entropy, loss $\ge 0$ always. But for **continuous** densities (diffusion, VAE ELBO terms), a *density* can exceed 1, so log-density can be positive and NLL can be negative. Legitimate. Confuses everyone once. Flag it in the diffusion track too.
- ❌ *"bf16 is just a smaller float16."* → bf16 has 8 exponent bits (same range as fp32, ~$10^{\pm38}$) and 7 mantissa bits; fp16 has 5 exponent bits (range only ~$6\times10^{-5}$ to $65504$) and 10 mantissa bits. **bf16 trades precision for range, and range is what training needs** — that's why fp16 needs a GradScaler and bf16 doesn't. On his Blackwell GB10, bf16 is the default and this is a live, practical fact.

### Demo: **Log-Space Rescue**
Two side-by-side counters. The learner slides "sequence length" $1\to1000$ and "per-token probability" $0.01\to0.99$.
- Left: naive product, computed with `Math.fround` to simulate float32. It hits `0` and the box goes red, and then `log` of it shows `-Infinity`.
- Right: the sum of logs. Shows a comfortable number.
Third panel: logits $[z_1, z_2, z_3]$ with sliders spanning $[-1000, 1000]$; naive `exp(z)/Σexp(z)` (shows `NaN` when $z>88$) vs max-subtracted softmax (correct).
**Exact math for JS:** simulate fp32 with `Math.fround()` on each partial product. `naive = z.map(Math.exp)` → `Infinity/Infinity = NaN`. Stable: subtract `Math.max(...z)` first.
**Insight:** *the two panels compute the same mathematical quantity. One returns NaN. Numerical analysis isn't pedantry; it's the difference between a training run and a crash.*

---

## 8. SOFTMAX AND CROSS-ENTROPY — THE LLM OBJECTIVE

**Intuition first:** *Softmax turns any list of scores into a list of probabilities that sum to 1, while keeping the ordering and exaggerating the gaps. Cross-entropy then asks one question: "what probability did you assign to the right answer?" — and penalizes you by the log of it.*

$$p_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$
- $z \in \mathbb{R}^V$ — **logits**, unnormalized scores, dimensionless, any real value. Shape `(B, T, V)` for an LLM, $V=128{,}256$ for Llama-3.
- $p \in \mathbb{R}^V$ — probabilities, $p_i > 0$, $\sum p_i = 1$.

**Two properties to state, both load-bearing:**
1. **Shift invariance:** $\text{softmax}(z + c) = \text{softmax}(z)$. The absolute logit scale is meaningless; only *differences* matter. (This is also what licenses the max-subtraction trick in §7 — same fact, and pointing that out is a nice moment of coherence.)
2. **Temperature:** $\text{softmax}(z/T)$. $T\to0$ → argmax (greedy); $T=1$ → the model's own distribution; $T\to\infty$ → uniform. **He has turned this exact slider in ComfyUI and in LLM UIs. Name it.**

### Cross-entropy / NLL
$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{V} y_i \log p_i \;\xrightarrow{\text{one-hot } y}\; \mathcal{L} = -\log p_{\text{correct}}$$

**The whole loss collapses to one term.** Say it plainly: *"For a one-hot target, cross-entropy is the negative log of the single probability you assigned to the right token. All the other $V-1$ terms are multiplied by zero."* That demystifies an intimidating formula in one sentence.

### Why log-likelihood is the loss (the derivation, done once, properly)
Maximize the probability of the data under the model:
$$\theta^* = \arg\max_\theta \prod_{n=1}^{N} p_\theta(y_n \mid x_n)$$
Products underflow (§7), so take logs — monotone, so the argmax is unchanged:
$$= \arg\max_\theta \sum_{n=1}^{N}\log p_\theta(y_n\mid x_n) = \arg\min_\theta \underbrace{-\sum_{n=1}^{N}\log p_\theta(y_n\mid x_n)}_{\text{negative log-likelihood} \;=\; \text{cross-entropy}}$$
**Cross-entropy isn't a loss someone invented. It's "make the data likely," rewritten so a computer can do it.** That sentence is the payoff of §7 and §6 combined. Put it in a box.

### The beautiful gradient — derive it, it's the best result in the chapter
For softmax + cross-entropy with one-hot target $y$:
$$\frac{\partial\mathcal{L}}{\partial z_i} = p_i - y_i$$
**Predicted minus actual.** That's the entire gradient w.r.t. the logits. No sigmoid derivative, no 0.25 factor, no vanishing. **The softmax's ugly derivative and the log's $1/x$ cancel exactly.** This is *another* reason CE beat MSE for classification: with MSE+sigmoid, a confidently-wrong output has a *tiny* gradient (because $\sigma'\to0$ when saturated) — the network is confidently wrong and learns slowly. With CE+softmax, a confidently-wrong output has gradient magnitude $\to 1$ — **maximally wrong means maximally corrected.** Worth a full paragraph; it's a genuinely satisfying "oh, that's *why*" moment.

### Worked example — carried to numbers
$V=4$, logits $z = [2.0, 1.0, 0.1, -1.0]$, true class = 0.
$e^{2.0}=7.3891$, $e^{1.0}=2.7183$, $e^{0.1}=1.1052$, $e^{-1.0}=0.3679$. Sum $= 11.5805$.
$p = [0.6381, 0.2347, 0.0954, 0.0318]$. (Check: sums to 1.0000 ✓)
$\mathcal{L} = -\ln(0.6381) = \mathbf{0.4491}$ nats.
Gradient: $\partial\mathcal{L}/\partial z = p - y = [0.6381-1, 0.2347, 0.0954, 0.0318] = \mathbf{[-0.3619, 0.2347, 0.0954, 0.0318]}$. Sums to 0 ✓ (always — it's a consequence of shift invariance; nice cross-check to point out).

**Anchor numbers for LLM loss** (plant these; they recur in every training log he'll ever read):
- Random init, ~~$V=128{,}256$: $\mathcal{L} = \ln(128256) = \mathbf{11.76}$~~ → **Qwen3-8B $V=151{,}936$: $\mathcal{L} = \ln(151936) = \mathbf{11.93}$ nats (D-01).** **If step 0 doesn't print ~11.9, something is wrong.** This is a real, usable sanity check.
- A well-trained 8B model on general text: $\mathcal{L} \approx 1.8$–$2.2$ nats (perplexity ~6–9).
- A LoRA fine-tune on a narrow domain: often $\mathcal{L} \approx 0.5$–$1.2$.
- $\mathcal{L} < 0.1$ on train while eval rises: **memorization.** Real diagnostic.

### Misconceptions
- ❌ *"Softmax outputs are confidences / calibrated probabilities."* → They sum to 1 by construction; that's arithmetic, not calibration. Modern nets are systematically **overconfident**. (Guo et al. 2017 established this for image classifiers; whether large modern LLMs are well-calibrated is **genuinely contested in 2026** — base models appear reasonably calibrated on next-token prediction, and there's substantial evidence RLHF/instruction-tuning degrades calibration, but the picture is not settled and depends heavily on what you're measuring. **Flag as open.**)
- ❌ *"Temperature changes what the model knows."* → It changes the sampling decision only. The logits are identical. He can verify: same prompt, same seed, different `T`.
- ❌ *"Softmax is a soft version of max."* → It's a soft version of **argmax** (returns a distribution over indices, not a value). The name is a historical error and it confuses people. Just say so.
- ❌ *"CE loss is only for classification, so it's irrelevant to diffusion."* → An LLM's next-token prediction is a $V$-way classification, and the *only* loss it has. Diffusion uses MSE on noise instead — and connecting them (both are NLL under different assumed output distributions; MSE **is** the NLL of a Gaussian with fixed variance) is a genuinely unifying insight. **Foundations should plant it; the diffusion brief should collect on it.**

### Demo: **Softmax + Temperature Bench**
Four (or eight) logit sliders, $z_i\in[-5,5]$. Bar chart of $p$ live. Temperature slider $T\in[0.01, 5]$, log scale. A "true class" radio selection; live readout of $\mathcal{L} = -\ln p_{\text{true}}$ and the gradient bar chart $p - y$ (signed, so the true class bar points down and the rest point up — visually, *"push the right one up, push the others down"*).
Add: a **"+3 to all logits"** button — the bars **don't move**. Shift invariance, seen not stated.
**Exact math for JS:** `const m = Math.max(...z); const e = z.map(v => Math.exp((v - m)/T)); const s = e.reduce((a,b)=>a+b); const p = e.map(v => v/s);`
**Insight:** three, in order: (1) $T\to0$: one bar goes to 1, everything else to 0 — *that's greedy decoding, and it's why `temperature=0` output is boring and repetitive*; (2) the gradient bar chart literally shows "push the right answer up, everyone else down, in proportion to how wrong you were"; (3) the +3 button proves logits are only meaningful relative to each other.

---

## 9. THE ARTIFICIAL NEURON

**Intuition first:** *A neuron measures how much its input looks like a pattern it's looking for, adds a bias to set how eager it is to fire, and then passes the result through a nonlinear squash. The weights ARE the pattern.*

That reframing — **the weight vector is a template, and the dot product is the match score** — is the payoff of §3 and is far more useful than the biological-neuron story.

$$a = \phi(z), \qquad z = w\cdot x + b = \sum_{i=1}^{n} w_i x_i + b$$
- $x \in \mathbb{R}^n$ — input, shape `(n,)`
- $w \in \mathbb{R}^n$ — weights, shape `(n,)`, one per input. **The template.**
- $b \in \mathbb{R}$ — bias, scalar. **The threshold, negated.** $z>0 \iff w\cdot x > -b$.
- $z \in \mathbb{R}$ — pre-activation / logit
- $\phi$ — nonlinearity
- $a \in \mathbb{R}$ — activation

**Geometric intuition (essential, do not skip):** $w\cdot x + b = 0$ is a **hyperplane**. $w$ is its normal vector (perpendicular). $b$ shifts it off the origin — specifically, the plane's distance from origin is $|b|/\|w\|$. **A neuron cuts space in half.** $\phi$ decides how sharply. That's the whole geometric content, and it makes the next section obvious.

### A layer of neurons — the shape story
$$h = \phi(Wx + b), \qquad W\in\mathbb{R}^{m\times n},\; b\in\mathbb{R}^m,\; x\in\mathbb{R}^n,\; h\in\mathbb{R}^m$$
Batched, PyTorch convention (**note: PyTorch's `nn.Linear` stores `weight` with shape `(out, in)` and computes $xW^\top + b$** — a real and confusing detail, address it once explicitly rather than letting him trip on it):
$$H = \phi(XW^\top + b), \qquad X\in\mathbb{R}^{B\times n} \to H\in\mathbb{R}^{B\times m}$$
Params: $mn + m$. **$m$ neurons, each with its own template row of $W$.** $Wx$ computes all $m$ match scores in one matmul — that's the only reason GPUs are relevant.

### The Perceptron (5 minutes, historical, but earn it)
Rosenblatt 1958: $\phi = \text{step}$, update rule $w \leftarrow w + \eta(y - \hat{y})x$. It provably converges **iff** the data is linearly separable (Novikoff 1962).
Minsky & Papert 1969 showed a single perceptron cannot compute **XOR**. Show the 4 points; show that no single line separates them:
| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |
$(0,0)$ and $(1,1)$ are class 0 and sit on a diagonal; $(0,1)$ and $(1,0)$ are class 1 on the other diagonal. **Any line puts two same-class points on opposite sides.** This is a 30-second visual proof and it's genuinely convincing.

Then: **two layers solve it, with 2 hidden ReLU neurons.** Give the explicit weights and *make him verify by hand* — this is the single most convincing 10 minutes in the entire foundations chapter:
$$h_1 = \text{ReLU}(x_1 + x_2 - 0), \quad h_2 = \text{ReLU}(x_1 + x_2 - 1), \quad y = h_1 - 2h_2$$
- $(0,0)$: $h_1=\text{ReLU}(0)=0$, $h_2=\text{ReLU}(-1)=0$, $y = 0 - 0 = \mathbf{0}$ ✓
- $(0,1)$: $h_1=\text{ReLU}(1)=1$, $h_2=\text{ReLU}(0)=0$, $y = 1 - 0 = \mathbf{1}$ ✓
- $(1,0)$: $h_1=1$, $h_2=0$, $y = \mathbf{1}$ ✓
- $(1,1)$: $h_1=\text{ReLU}(2)=2$, $h_2=\text{ReLU}(1)=1$, $y = 2 - 2 = \mathbf{0}$ ✓

**Four lines of arithmetic and he has personally built a network that does something a linear model provably cannot.** Then note: the ReLU is doing *all* the work — $h_1$ and $h_2$ are the *same linear function* shifted, and only the kink at 0 distinguishes them. This is the perfect bridge into §10.

**Historical honesty:** the "AI winter caused by Minsky & Papert" story is a **simplification the course should not repeat uncritically**. They knew multilayer nets existed; the actual problem was that nobody had a training algorithm for them until backprop was popularized (Rumelhart, Hinton & Williams 1986 — though Linnainmaa had the method in 1970 and Werbos applied it to nets in 1974). The bottleneck was **credit assignment**, i.e. the chain rule applied at scale. That framing is both truer and better pedagogy — it makes §5 the hero of the historical narrative too.

### Misconceptions
- ❌ *"Neurons are like brain neurons."* → The analogy is 1943-vintage marketing and it actively misleads. Real neurons spike in time, have dendritic computation, and don't do backprop. **Say it once, drop it, never use brain language again.** For this learner specifically, the biological framing is *worse* than useless — he's an engineer; give him "template matcher + hyperplane" and he's better equipped.
- ❌ *"The bias is a minor detail."* → Without $b$, every hyperplane passes through the origin. A layer without bias can't even represent "output 1 when input is 0." *(But do note the genuine wrinkle: modern LLMs — Llama, Qwen, most 2026 architectures — **omit biases in the linear layers** entirely, because RMSNorm + the residual stream make them redundant and they cost params and a little stability. Teach bias as essential-in-principle, note it's often dropped in practice. Anticipating his "wait, I've read Llama code and there are no biases" is exactly the kind of thing that builds credibility.)*
- ❌ *"More neurons = more layers."* → Width and depth are different axes with different effects. Width adds hyperplanes to one cut; depth composes cuts. Depth is exponentially more expressive per parameter (Telgarsky 2016, Montúfar et al. 2014 — for ReLU nets, the number of linear regions grows polynomially in width but exponentially in depth).

---

## 10. LINEAR COLLAPSE — WHY THE NONLINEARITY IS THE WHOLE BALLGAME

Give this its own section. It is the single argument that justifies the existence of deep learning, and it takes four lines.

**Intuition first:** *Stacking linear layers is like stacking sheets of glass — no matter how many you use, you can still see straight through, and you could have used one sheet. The nonlinearity is what bends the light.*

### The proof (four lines, do it)
$$h_1 = W_1 x + b_1$$
$$h_2 = W_2 h_1 + b_2 = W_2(W_1x + b_1) + b_2 = \underbrace{W_2W_1}_{W'}x + \underbrace{(W_2b_1 + b_2)}_{b'} = W'x + b'$$

**A 100-layer linear network is a 1-layer linear network.** Composition of affine maps is affine. Closed under composition. Done.

### The counting that makes it land
Two linear layers, $784 \to 512 \to 10$ (MNIST-shaped):
- Params: $784\times512 + 512 + 512\times10 + 10 = 401{,}408 + 512 + 5{,}120 + 10 = \mathbf{407{,}050}$
- Equivalent single layer $784\to10$: $784\times10 + 10 = \mathbf{7{,}850}$
- **407,050 parameters expressing a function that 7,850 parameters can express exactly.** 98% of the parameters are *provably* redundant. Not approximately — exactly. There is a $W' = W_2W_1$ that reproduces it bit-for-bit (modulo float rounding).

**Then the flip:** insert ONE ReLU between them and the collapse is *impossible*. There is no $W'$ that reproduces $W_2\,\text{ReLU}(W_1x+b_1)+b_2$. The function class jumps from "$\le$ 7,850 params' worth of straight lines" to "arbitrary piecewise-linear functions with up to $\sim2^{512}$ linear regions."

**The best one-liner in the chapter:** *"The nonlinearity contributes zero parameters and does all the work."* ReLU has no parameters. Not one. And without it there is no deep learning.

### Bonus honesty (include it — this learner will appreciate the rigor)
A linear net isn't *totally* useless: the *optimization dynamics* of deep linear nets are genuinely different from a single linear layer (implicit regularization toward low-rank — Saxe et al., Arora et al.), and this is an active research area. **But the function class is identical.** Distinguish "what can it represent" from "what will gradient descent find." That distinction recurs and is worth naming early.

### Demo: **Linear Collapse Playground** ⭐ flagship
A 2-D classification canvas with a toggleable dataset (two moons / XOR / concentric rings — all classic non-linearly-separable). Controls: depth slider $L\in[1,5]$, width slider $[1,16]$, activation dropdown `{identity, sigmoid, tanh, ReLU, GELU}`. Train live in-browser with plain SGD and show the decision boundary evolving as a colored heatmap.
**Exact math for JS:** a real ~100-line MLP with manual forward + backprop. Forward: `h = act(W@h + b)` per layer. Backward: reverse loop, `dW = delta.outer(h_prev)`, `delta_prev = (W.T @ delta) * actPrime(z_prev)`. Loss = binary cross-entropy on a sigmoid output. Batch of ~200 points, 500 epochs, LR ~0.1. This is small enough to run at 60 fps in a `requestAnimationFrame` loop. **It must be real training, not a canned animation.**
Add a live text readout: **"Effective function class: LINEAR"** (red) vs **"PIECEWISE LINEAR, ~N regions"**, and a live-computed count of distinct ReLU activation patterns across the grid — that number is the "number of linear regions" and watching it grow with depth is *the* depth-vs-width lesson made visible.
**Insights, in the order he'll get them:**
1. Set activation to `identity`. Crank depth to 5, width to 16. **The boundary is a straight line and it stays a straight line no matter what.** 1,400 parameters, one line. He will try to break it. He can't. That's the lesson.
2. Switch to `ReLU`, depth 2, width 8. The boundary bends. Moons get separated.
3. Depth 2, width 16 vs depth 4, width 4 (comparable params, ~200 each): **the deeper one carves a much more intricate boundary.** Depth > width, per parameter.
4. Switch to `sigmoid`, depth 5: **it trains visibly slower and often stalls** — the 0.25 factor from §5, felt, not asserted. Show the layer-1 gradient norm as a live number: `~1e-5` for sigmoid-depth-5 vs `~1e-1` for ReLU.

### Demo: **"Draw the function a 2-layer net can fit"** ⭐ flagship
Free-draw canvas: the learner draws any 1-D curve $y = g(x)$ with the mouse over $x\in[-1,1]$. A 1→H→1 ReLU MLP fits it live with Adam. Hidden-width slider $H \in [1, 64]$.
**Crucially, also render each hidden unit's individual contribution** as a faint colored hinge function $v_j\cdot\text{ReLU}(w_jx + b_j)$, with the sum drawn in bold.
**Exact math for JS:** 1→H→1, MSE loss, Adam ($\eta=0.01$, $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$), 128 sampled points from the drawn curve, ~2000 steps at 60fps. Manual backprop, ~60 lines.
**Insights:**
1. $H=1$: one hinge. Two straight segments. That's *all* one ReLU neuron is.
2. $H=4$: four hinges summing to a crude piecewise-linear approximation. **He can see each neuron's kink and count them on the fitted curve.**
3. $H=64$: fits almost anything he draws — but zoom in and **it's still made of straight segments.** "Universal approximation" is a stack of hinges. Nothing smooth is happening.
4. Draw something with a sharp spike and $H=4$: it can't. Crank to 64: it can. **Capacity is a real, visible, countable thing.**
5. Draw only over $x\in[-1,0]$, then extend the plot range to $x\in[-1,2]$: **the net extrapolates as a straight line, confidently, and it's garbage.** → §11's honesty section, felt.

---

## 11. ACTIVATION FUNCTIONS — WHAT EACH IS ACTUALLY FOR IN 2026

Lead with the honest summary: **in 2026 you use SwiGLU inside LLM/DiT FFNs, GELU or SiLU everywhere else, softmax at the output for classification, and sigmoid only for gates and binary outputs. Sigmoid and tanh as hidden activations are dead.** Everything below explains why.

| Name | Formula | Range | $\phi'$ max | 2026 role |
|---|---|---|---|---|
| **Sigmoid** | $\sigma(x)=\frac{1}{1+e^{-x}}$ | $(0,1)$ | **0.25** | Binary output; gates (LSTM legacy, SiLU internals). **Never** a hidden activation. |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}} = 2\sigma(2x)-1$ | $(-1,1)$ | **1.0** | Legacy RNNs; GELU's tanh approx; bounded outputs. Basically retired. |
| **ReLU** | $\max(0,x)$ | $[0,\infty)$ | **1.0** | CNNs, tons of production vision, anything where speed rules. Still everywhere. |
| **GELU** | $x\Phi(x)$ | $\approx(-0.17,\infty)$ | ~1.13 | BERT/GPT-2 lineage, ViTs, many diffusion U-Nets. |
| **SiLU/Swish** | $x\sigma(x)$ | $\approx(-0.28,\infty)$ | ~1.10 | The `Swish` in SwiGLU; also standalone in many diffusion nets. |
| **SwiGLU** | $(\text{SiLU}(xW_g))\odot(xW_u)$ | — | — | **The FFN of essentially every post-2023 LLM**: Llama 1/2/3, Mistral, Mixtral, Qwen, Gemma, DeepSeek, Phi. Verified current as of 2026. |

### The real reason ReLU displaced sigmoid — say this loudly, the folk answer is wrong

**The folk answer: "ReLU is cheaper to compute."** It is cheaper (a `max`, vs an `exp` and a divide), but on a GPU running a $4096\times14336$ matmul, the activation is a **memory-bandwidth-bound elementwise op costing a rounding error** of total time. If that were the reason, nobody would have switched. **This is a satisfying myth to kill and the course should kill it explicitly.**

**The real answer: gradient flow.** $\sigma'$ peaks at 0.25 and decays to ~0 for $|x|\gtrsim 5$.
- Depth 10, sigmoid, all units near their best case: $\prod \sigma' \le 0.25^{10} = 9.54\times10^{-7}$
- Depth 32 (Llama-3-8B depth): $0.25^{32} = 5.4\times10^{-20}$ — **below float32's smallest subnormal.** The gradient at layer 1 is **literally zero** in fp32.
- ReLU: $\phi'\in\{0,1\}$. On the active path the product of derivatives is $1^{32} = 1$. **Exactly 1.** The signal is preserved, not attenuated.

**Sigmoid's second sin (less famous, real):** its output is not zero-centered — it's always in $(0,1)$, so all inputs to the next layer are positive, so all weight gradients in a row share a sign, so updates zigzag. Tanh fixed *this* problem (zero-centered, $\phi'$ max = 1.0) and tanh still lost — **because $\phi'\to0$ for $|x|>2$ anyway.** The saturation is the killer, and only a non-saturating positive side fixes it. That's the argument that makes it click.

**ReLU's own sin: dying ReLU.** If $z<0$ for every input in the dataset, $\phi'=0$ always, gradient is 0 always, and **that unit never recovers.** It's dead permanently. Typical dead-unit rates in a well-initialized trained net: ~10–20% of units (varies enormously; treat as an order of magnitude, not a constant — **medium confidence, don't over-quote**). Big learning rates and large negative biases make it much worse. Leaky ReLU ($\max(0.01x, x)$) and GELU/SiLU (smooth, nonzero gradient for small negative $x$) both address it. **This is why GELU/SiLU won in transformers rather than plain ReLU.**

### GELU — both forms, and the honest note

Exact (verified against the ONNX spec and Hendrycks & Gimpel 2016):
$$\text{GELU}(x) = x\,\Phi(x) = x\cdot\frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt2}\right)\right]$$
Tanh approximation (what BERT and GPT-2 actually shipped, and what `nn.GELU(approximate='tanh')` gives — PyTorch's **default is `'none'`, i.e. exact erf**; this mismatch is a real reproduction footgun):
$$\text{GELU}_{\text{tanh}}(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

**Intuition (better than the usual hand-waving):** *ReLU multiplies $x$ by a hard 0/1 switch. GELU multiplies $x$ by the probability that a standard normal is below $x$ — a soft switch. It's a stochastic-gate expectation: "keep this value with probability $\Phi(x)$", averaged.* That's the actual motivation in the original paper, and it's much better than "it's a smooth ReLU."

**Numbers:** $\text{GELU}(0)=0$. $\text{GELU}(1) = 1\cdot\Phi(1) = 0.8413$. $\text{GELU}(-1) = -1\cdot\Phi(-1) = -0.1587$. $\text{GELU}(-2) = -2 \times 0.02275 = -0.0455$. Minimum $\approx -0.170$ near $x\approx-0.7517$. **Note it's non-monotonic** — it dips below zero and comes back. ReLU can't do that; whether the non-monotonicity matters is genuinely unclear, and the ablations are noisy. Say so.

**SiLU/Swish:** $\text{SiLU}(x)=x\sigma(x)$. Nearly identical to GELU in shape (min $\approx -0.278$ at $x\approx-1.278$) but uses a plain sigmoid instead of $\Phi$ — cheaper, no `erf`. Swish was *found by neural architecture search* (Ramachandran et al. 2017), which is a nice honesty beat: **the field's most-used activations were partly discovered by brute-force search, not derived.**

### SwiGLU — do the full accounting, he's going to see this in every model config

$$\text{SwiGLU}(x) = \bigl(\text{SiLU}(xW_{\text{gate}})\bigr) \odot \bigl(xW_{\text{up}}\bigr), \qquad \text{FFN}(x) = \text{SwiGLU}(x)\,W_{\text{down}}$$
- $x$ — `(B, T, 4096)`; $W_{\text{gate}}, W_{\text{up}}$ — `(4096, 14336)`; $W_{\text{down}}$ — `(14336, 4096)`
- $\odot$ — **elementwise** (Hadamard) product. This is the §2 warning cashing in: one line uses `@` twice and `*` once and they're different operators.

**Intuition:** *Two parallel projections. One is the content ("here's what I computed"), the other is a gate ("here's how much of it to let through, per channel"). The network learns to route information, not just transform it.* Multiplicative gating gives the layer a way to express "if A then B" — a conditional — which an additive layer cannot cheaply do.

**The 2/3 rule — a real number he'll see in every config file.** SwiGLU needs 3 matrices where vanilla FFN needs 2. To keep the param count fair, shrink $d_{ff}$ by $2/3$:
- Vanilla FFN with $d_{ff}=4d = 16384$: $2 \times 4096\times16384 = 134.2$M params
- SwiGLU with $d_{ff} = \frac{2}{3}\times16384 = 10923$: $3\times4096\times10923 = 134.2$M params. **Same.**
- Llama-3-8B actually uses $d_{ff}=14336$, not 10923 — because they also round to a multiple of 256 for hardware efficiency and chose to spend a bit more. **$14336 = 3.5 \times 4096 = 56\times256$.** *Explaining why a config file says 14336 is exactly the kind of demystification this course exists for.*

**Honest caveat that the field's own author gave:** Shazeer (2020), "GLU Variants Improve Transformer," ends with — and this is worth quoting because it's rare intellectual honesty — that he offers no explanation for why these variants work, attributing their success to divine benevolence. **The most-used activation in 2026 has no accepted theoretical justification. It won on benchmarks.** This is exactly the kind of thing the brief was asked to flag rather than paper over, and this learner will *trust the course more* for saying it.

### Misconceptions
- ❌ *"The activation choice is a big accuracy lever."* → Between GELU/SiLU/SwiGLU the differences are ~0.1–0.5% on benchmarks — real at scale, noise for his fine-tuning. **Between sigmoid and any of them at depth, the difference is "trains" vs "doesn't."** The lever is huge at the bad end and tiny at the good end.
- ❌ *"ReLU is obsolete."* → It's still in enormous amounts of production vision and is often the right call. It's not a hidden activation in LLMs, that's all.
- ❌ *"You must match the activation to your data."* → No. This isn't a hyperparameter to tune. **Use what the architecture you're fine-tuning uses. Changing it on a pretrained model destroys it.** (Practical: if he swaps GELU→ReLU in a downloaded checkpoint, the weights are now wrong for the function. Say it.)
- ❌ *"GELU is smooth so it's differentiable everywhere so it's better."* → ReLU's non-differentiability at exactly 0 is a measure-zero non-issue; PyTorch returns 0 and float inputs hit exactly 0.0 essentially never. **Smoothness is not the reason GELU wins.** Not-dying is.
- ❌ *"SwiGLU is an activation function."* → It's a **layer** — it has learned parameters ($W_{\text{gate}}$). ReLU/GELU are parameterless functions. Calling them the same kind of thing is a category error that confuses param counting. This is worth a warning box; the naming in the field is genuinely bad.

### Demo: **Nonlinearity Playground**
Panel A: all six functions overlaid on $x\in[-5,5]$, toggleable, with their **derivatives** on a second linked axis below. Hover for exact values.
Panel B: **the gradient-product simulator.** Depth slider $N\in[1,40]$, activation dropdown. Compute $\prod_{i=1}^{N}\phi'(z_i)$ with $z_i \sim \mathcal{N}(0,1)$ (resample button), display in scientific notation, plus a bar showing where it sits relative to fp32 min-normal ($1.18\times10^{-38}$) and fp16 min-normal ($6.1\times10^{-5}$). **Sigmoid at depth 32 goes under the fp32 line and the bar turns red.** ReLU hovers around 1.
Panel C: **dead-ReLU counter.** 64 ReLU units with random $w,b$; a slider for "bias init mean" from $-2$ to $+2$; feed 1000 random inputs; count units that never fire. Watch the count go from 0 to 60 as bias init goes negative.
**Exact math for JS:** the six formulas directly (`erf` via Abramowitz–Stegun 7.1.26, max error $1.5\times10^{-7}$ — good enough and worth showing the source). Derivatives analytically where clean, central differences otherwise.
**Insight:** Panel B is the payload. *"You don't choose ReLU because it's fast. You choose it because at depth 32, sigmoid's gradient is a number float32 cannot represent."* Panel C: *"and ReLU's own failure mode is a unit that shuts up forever — which is why GELU exists."*

---

## 12. THE MLP AND UNIVERSAL APPROXIMATION — WHAT IT DOES AND DOESN'T PROMISE

**Intuition first:** *A wide enough single-hidden-layer net can approximate any reasonable function on a bounded region, to any accuracy you like. This theorem is true, famous, and much less useful than it sounds.*

### The MLP
$$\text{MLP}(x) = W_2\,\phi(W_1x + b_1) + b_2$$
$x\in\mathbb{R}^{n}$, $W_1\in\mathbb{R}^{H\times n}$, $W_2\in\mathbb{R}^{m\times H}$. $H$ = hidden width.

In a transformer block the MLP is the FFN, and per §2 it's **~70% of the parameters** of Llama-3-8B. It is not a footnote; it's the bulk of the model. That number should recur here.

### The theorem, stated honestly
**Cybenko (1989) / Hornik (1991):** For any continuous $f$ on a compact set $K\subset\mathbb{R}^n$ and any $\varepsilon>0$, there exists a width $H$ and parameters such that
$$\sup_{x\in K}\bigl|\text{MLP}(x) - f(x)\bigr| < \varepsilon$$
for any non-polynomial continuous activation.

### What it does NOT promise — this is the section that matters
1. **It doesn't tell you $H$.** $H$ can be exponential in $n$ — e.g. $\mathcal{O}(\varepsilon^{-n})$ for some function classes. For $n=4096$ that's a number with no physical meaning. **The theorem is an existence result with no useful bound.**
2. **It doesn't tell you that gradient descent can FIND those parameters.** They exist. Finding them is a separate, unsolved question. **(Training a general 3-node net to optimality is NP-hard — Blum & Rivest 1992.)** Existence ≠ reachability. This is the deepest point in the section.
3. **It doesn't promise generalization.** Approximating $f$ *on your training points* is exactly what memorization is. The theorem is silent on new data. Zhang et al. 2017 showed deep nets can perfectly fit **random labels** — capacity is not the constraint, and classical VC-dimension bounds are vacuous here.
4. **It's about a compact set.** **Outside your training region, all bets are off** — and §10's draw-a-function demo showed him a ReLU net extrapolating as a confident straight line into garbage. **Every hallucination and every weird diffusion artifact is, in one sense, this.**
5. **It says nothing about depth.** And depth is what actually made deep learning work. **The most famous theorem in the field is about the architecture nobody uses.** Blunt, true, memorable.

### The genuine open question — flag it, don't resolve it
**Why do overparameterized nets generalize at all?** An 8B-param model trained on ~15T tokens should, by classical statistical learning theory, overfit catastrophically. It doesn't. As of 2026 this is **not settled.** Candidate explanations with real adherents: implicit regularization of SGD toward flat/low-norm minima; the double-descent picture (Belkin et al. 2019 — well-replicated *phenomenon*, contested *explanation*); benign overfitting; NTK/lazy-training (broadly agreed to be an incomplete picture of real nets, since real nets learn features and the NTK regime by construction doesn't); the lottery ticket hypothesis (Frankle & Carbin 2019 — findings replicate, interpretation debated). **The course should say: "we can build these things, they reliably work, and the theory of why is an active research area with no consensus."** For an engineer this is *reassuring*, not alarming — he works with turbulence and empirical correlations. Do not manufacture false closure. The fake explanation is worse than the honest gap.

### Misconceptions
- ❌ *"Universal approximation means one hidden layer is enough, so why go deep?"* → Enough *in principle*, at absurd width, with parameters you cannot find. Depth gives exponentially more expressiveness per parameter and — empirically, this is the part theory doesn't capture — is far easier to optimize.
- ❌ *"Bigger model = always better."* → Chinchilla (Hoffmann et al. 2022) established compute-optimal training needs **~20 tokens per parameter**. Modern practice deliberately *overtrains* well past that for inference efficiency — Llama-3-8B saw **15T tokens ≈ 1875 tokens/param**, ~94× "compute-optimal." **This is a real, deliberate departure from the theory and shows the field trades training compute for inference cost.** Good number for him: it explains why 8B models in 2026 are so much better than 8B models in 2022 at the same size.
- ❌ *"The net learns the true function."* → It learns *a* function that matches on the training distribution. Off-distribution it does whatever the architecture's inductive bias says. For ReLU nets: linear extrapolation, confidently.

---

## 13. RECOMMENDED DEPENDENCY SPINE

Strict order. Each arrow means "the second genuinely cannot be understood without the first," not "conventionally taught after."

```
[1] Functions & composition
     └─> the θ-vs-x flip (θ is the variable)  ──────────────┐
                                                            │
[2] Vectors/matrices/tensors: SHAPE + matmul rule           │
     ├─> Broadcasting (+ the silent-bug warning)            │
     └─> [3] Dot product = similarity/projection            │
              ├─> neuron-as-template-matcher (needed by 9)  │
              ├─> gradient=steepest-ascent proof (needed by 4)
              └─> attention preview, √d_k  (pay off later)  │
                                                            │
[4] Derivative = slope  <─────────────────────────────────  │
     ├─> the 7-derivative table (incl. σ' max = 0.25)       │
     ├─> partials -> gradient (SAME SHAPE as θ)             │
     └─> gradient descent + 1st-order Taylor + η numbers    │
              │                                             │
[5] ★ CHAIN RULE ★  "multiply along paths, add across paths"
     ├─> hand-worked example -> verified in PyTorch  ← trust anchor
     ├─> why activations are cached -> VRAM -> grad checkpointing
     ├─> vanishing gradient = product of small numbers
     └─> (later: backprop = this + memoization + reverse order)
              │
[6] Probability: distribution, E[·], sampling, 1/√B
     ├─> Gaussian, isotropic N(μ, σ²I)
     └─> reparameterization x = μ + σε  ──> ★ HANDOFF TO DIFFUSION ★
              │
[7] Logs & exps: log-probs, underflow, logsumexp, bf16 vs fp16
              │            (7 MUST precede 8 — CE is unmotivated without it)
[8] Softmax + cross-entropy + NLL derivation
     ├─> ∂L/∂z = p − y   (the cancellation)
     └─> L = ln(V) = 11.93 sanity check (Qwen3, V=151936; ~~11.76~~ D-01)  ──> ★ HANDOFF TO LLM TRACK ★
              │
[9] The neuron: w·x + b, hyperplane, perceptron, XOR by hand
              │
[10] LINEAR COLLAPSE  (needs 2 for W₂W₁ = W′, needs 9 for the setup)
              │
[11] Activations (needs 4 for φ′, needs 5 for the product-of-φ′ argument,
     needs 10 to know why a nonlinearity is needed at all)
              │
[12] MLP + universal approximation + the honest limits
              │
      ──> HANDOFF: backprop-as-algorithm / training loop chapter
```

**Non-obvious ordering claims, with the reason:**
- **§3 (dot product) must precede §4 (gradient)** — the steepest-ascent argument *is* $\nabla f \cdot u = \|\nabla f\|\cos\theta$. Teaching gradient first forces you to assert steepest-ascent instead of proving it in one line.
- **§7 (logs) must precede §8 (cross-entropy)** — CE looks arbitrary until you've watched $\prod p_i$ underflow to zero. The motivation is numerical, and it must be *felt* first.
- **§9 (neuron) must precede §10 (collapse) must precede §11 (activations)** — the standard order (activations right after neuron) is **wrong**: it presents ReLU/GELU as a menu of options before establishing that a nonlinearity is *mandatory*. Collapse first makes activations feel necessary rather than decorative. **This is my strongest ordering recommendation.**
- **§5 (chain rule) must precede §11** — the "$0.25^{32}$" argument is a chain-rule argument; without §5, "ReLU beat sigmoid" is folklore.
- **§6 (reparameterization) is early on purpose** — earlier than most curricula. It's cheap here and it makes the diffusion forward process a recognition rather than a derivation.

**Time budget suggestion (of ~45–50 pages total, foundations should be ~8–10):** chain rule 2 pages (it earns it), shapes/broadcasting 1.5, dot product 1, probability+Gaussian 1.5, logs+softmax+CE 1.5, neuron+collapse 1.5, activations 1.5, UAT 0.5. **If the page budget is squeezed, cut the perceptron history and the UAT section — not the chain rule.**

---

## 14. FLAGGED DISAGREEMENTS, UNCERTAINTIES, AND THINGS I COULD NOT VERIFY

**Genuine open questions in the field (present as open, do not resolve):**
1. **Why overparameterized nets generalize.** No consensus in 2026. (§12)
2. **Why SwiGLU/GLU variants work.** The originating paper explicitly declines to explain. (§11)
3. **Whether SGD noise helps generalization, and by what mechanism.** Correlation is well-established; the mechanism is contested. (§6)
4. **LLM calibration.** Base models look reasonably calibrated on next-token; RLHF appears to degrade it; the literature is not unanimous and results are metric-dependent. (§8)
5. **Whether smooth activations' non-monotonicity matters.** Ablations are within noise. (§11)
6. **Loss-landscape geometry (flat vs sharp minima).** Widely cited, and the sharpness measures are known to be reparameterization-sensitive — a real methodological critique (Dinh et al. 2017) that is often ignored. Don't lean on it.

**My editorial calls that the architect may want to overturn — each is a real judgment, not a fact:**
- **Cutting eigenvalues/SVD from foundations.** Defensible counter: LoRA's "rank" is meaningless without it. My call: one gated paragraph inside the LoRA chapter ("rank = how many independent directions this update can move in"), not a foundations section. **This is the cut I'm least certain of.**
- **Cutting continuous-time ODE/SDE framing.** Depends entirely on how the diffusion brief is written. **Must be reconciled.**
- **Teaching §10 (collapse) before §11 (activations).** Non-standard. I'm confident it's right.
- **Introducing the reparameterization trick in the probability section.** Non-standard placement, high payoff.

**Verified this session (2026-07-16):** PyTorch **2.13.0**, released **2026-07-08** (latest stable; FlexAttention on Apple MPS, deterministic FlexAttention backward on CUDA). DGX Spark: GB10, 128 GB LPDDR5X unified, 273 GB/s, ~1 PFLOP NVFP4 (sparse marketing peak), ~~~31 TFLOPS FP32~~ **[UNSOURCED — deleted per D-10c; dense-BF16 roofline is an unpublished inference, ~62–125 TF, constants §6.3]**, 20-core Arm (10× Cortex-X925 + 10× Cortex-A725). SwiGLU confirmed as the standard FFN in essentially all post-2023 LLMs (Llama 1/2/3, Mistral, Mixtral, Qwen, Gemma, DeepSeek, Phi). GELU exact-vs-tanh forms confirmed against the ONNX operator spec; **PyTorch `nn.GELU` defaults to `approximate='none'` (exact erf)**.

**High confidence, not re-verified this session (standard published architecture specs):** Llama-3.1-8B: $P=8.03$B, $d=4096$, $d_{ff}=14336$, $L=32$, $V=128256$, 32 heads / 8 KV heads, head_dim 128, ~15T training tokens. All arithmetic in this brief was recomputed by hand and is self-consistent; the architect should spot-check the FFN param counts (§2) against a `config.json` before they go in the book, since **these numbers are supposed to recur and become familiar — an error would propagate through the entire course.**

**Medium/low confidence — quote loosely or drop:**
- Dead-ReLU rates (~10–20%): order of magnitude only, wildly setup-dependent.
- Typical fine-tuning loss values (0.5–1.2 for narrow-domain LoRA): illustrative, not authoritative.
- Diffusion fine-tune LR range: **RESOLVED (D-09) — full FT 1e-6–5e-6, LoRA 1e-4. Owned by the memory-ledger page. See §4 correction above.**

**⚠️ ANCHOR-MODEL REWRITE (D-01): this brief anchored on Llama-3.1-8B, which is RETIRED. Every worked number
below must be re-derived on Qwen3-8B per `constants.md` — $d_{ff}$ 14336→**12288** (=3d), $V$ 128256→**151936**,
$\ln V$ 11.76→**11.93**, "~70% FFN"→**66.4% of model / 78.26% of block**, "96 GB AdamW"→**131.05 GB**. See D-01, D-20.**

**Handoffs the architect must reconcile with other briefs (corrected):**
| Foundations plants | Which brief collects |
|---|---|
| $x = \mu + \sigma\varepsilon$ (§6) | **Diffusion** — forward process is this verbatim |
| $\sqrt{d_k}$ variance argument (§3) | **Attention/transformer** |
| $q\cdot k$ as "question meets label" (§3) | **Attention** |
| ~~$\mathcal{L}=\ln V = 11.76$~~ → **$\ln(151936) = 11.93$** at init (§8) | **LLM training** — step-0 sanity check (D-01) |
| MSE = NLL of a fixed-variance Gaussian (§8) | **Diffusion** — unifies the two tracks' losses |
| ~~96 GB AdamW-on-8B~~ → **131.05 GB** teaser (§2) | **Fine-tuning / LoRA** — the motivating number (D-03) |
| Cached activations → VRAM → grad checkpointing (§5) | **Training-loop / hardware** |
| ~~~70% of params are FFN~~ → **66.4% of model / 78.26% of block** (§2, §12) | **LoRA target-module choice** (D-01) |
| SVD/rank, ~~deliberately deferred~~ → **OVERRULED: trunk gets a full SVD page** (§0) | **LoRA** — D-12 |
| $\eta$ table: 1e-5 full / 1e-4 LoRA / **diffusion full 1e-6–5e-6, LoRA 1e-4** (§4) | **Both tracks** — D-09 |
