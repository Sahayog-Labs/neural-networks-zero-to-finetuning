# BUILD SPEC — PART II · HOW IT LEARNS (pages 12–24)

<!-- ============================================================================
REPAIR LOG — 2026-07-16, Part II restored against decisions.md §D-21a (defect #2).
Authority: hardware-ground-truth > constants > notation > decisions(§D-21a) > briefs.
Content allocation is D-21a's; titles paraphrased; existing craftsmanship salvaged wholesale.

REALLOCATION MAP (old page → new page). Pages 12–18 unchanged in allocation.
  old 19 (schedules + warmup)              → NEW 21  (schedules and warmup)          [build S per §D-21a]
  old 20 (initialization)                  → NEW 19  (init + vanishing/exploding + clipping, MERGED half A)
  old 21 (vanishing/exploding + clipping)  → NEW 19  (init + vanishing/exploding + clipping, MERGED half B)
  old 22 (normalization + pre-norm)        → NEW 20  (normalization and pre-norm)
  old 23 (regularization + double descent) → SPLIT:  NEW 22 (regularization + the two curves)  [build S]
                                                   +  NEW 23 (bias–variance + double descent, OWN page) [build S]
  old 24 (first real fine-tune / debugging)→ NEW 24  (debugging: twelve failures; milestone closer KEPT/approved)

  The old 20+21 split of init/vanishing was a fidelity defect (§D-21a defect #2): re-merged into ONE page 19.
  The old 23 merge of regularization+double-descent was a lead-REJECTED cut (§D-21a defect #2): un-merged;
  double descent gets its OWN page (23) with the λ-slider-that-dissolves-the-phenomenon demo the arbiter designed.

CODE ARTIFACTS renumbered: 19_schedule.py→21_schedule.py; 20_init_variance.py→19_init_variance.py;
  21_grad_flow.py→19_grad_flow.py; 22_norm_axes.py→20_norm_axes.py; 23_double_descent.py stays 23,
  new 22_regularization.py added.

FLOAT BEAT REVERTED HOME TO PAGE 15. The previous integrator relocated the "exactly zero / 1.402e-08" beat to
  Part I p.10 on constants §8.4's stale "page ~10" pointer. Under §D-21a, Part I p.10 = XOR/linear-collapse (NO
  autograd); the beat's home is PAGE 15 again — full verbatim §8.4 treatment restored, "callback to p.10" framing
  removed. (§8.4's own "page ~10 / three pages early" wording is stale vs §D-21a and is not reprinted.)

PART I CROSS-REFS fixed to §D-21a numbering: chain rule = p.5; TN-1 forward = p.6; probability/Gaussian/
  reparameterization = p.8; logs/logsumexp/softmax/CE = p.9; XOR/linear-collapse = p.10; activations/MLP-limits = p.11.
  Page 12's MSE-from-Gaussian-NLL now explicitly cashes the seed p.8 plants.

Also corrected in this pass: page 16 build letter O → S (matches §D-21a; the rest of 12–18 verified correct).
Note: another agent is deleting Part I's backprop/autograd/one-step pages and exporting salvageable blocks here;
  pages 14/15/17 are written COMPLETE on their own terms per §D-21a — the integrator reconciles any overlap.
============================================================================ -->

**Track:** trunk (no track class on any Part II page — the LLM/diffusion fork is after page 24).
**Anchor model:** Qwen3-8B (D-01). **Canonical network:** TN-1 (D-02, `constants.md` §8).
**Author's mandate:** every entry below is implementable 1:1 without opening a brief. Every number is quoted
verbatim from `constants.md` with its confidence tag. Never launder an estimate into a fact (notation §9 #22/#23).

> **D-21 numbering.** Part II pages are 12–24 and there is **no shift** in this range — the GRPO/diffusion
> renumbering only affects pages 52+. Each entry prints its own number; the parenthetical "(D-21: NN)" equals the
> page number itself for all of Part II. Filenames are `NN-slug.html`; `data-page` MUST equal the filename and the
> `course-map.js` `file:` field or the sidebar/pager break (TEMPLATE lines 25–33).

---

## GLOBAL CONVENTIONS FOR PART II (apply to every page 12–24)

**Notation (frozen, notation.md).** $\mathcal{L}$ = per-example loss (nats); $J$ = batch-averaged loss;
$W$ is always $(d_{\text{out}},d_{\text{in}})$; $\mathbf{z}^{(\ell)}$ pre-activation, $\mathbf{a}^{(\ell)}$
post-activation; $\boldsymbol\delta^{(\ell)}=\partial\mathcal{L}/\partial\mathbf{z}^{(\ell)}$ (the backprop
workhorse); **$k$ = optimizer step** (NOT $t$ — notation §3, add the Translation note on page 13 and 17);
$\eta$ = learning rate and nothing else; $\lambda$ = weight decay; $\mu$ = momentum; $\beta_1,\beta_2$ = Adam
moments (numeric subscripts always); $\varepsilon$ = stability floor; $v$ = vocab/class index (NOT $k$).
Every equation with >3 distinct symbols carries a **Symbol Ledger** (notation §8); every tensor op a **shape
ribbon** (notation §7). Gradient always has the same shape as the thing it differentiates.

**Page rhythm (brief-pedagogy):** predict → intuition → math → worked number → demo → quiz. Open pages that
introduce a mechanism with a **PREDICT** prompt (a `.box try` phrased as a question, answered later on the page).
Correct every named misconception in a `.box warn`. Push heavy derivations into `<details class="deepdive">`
whose `<summary>` **states** the result (never teases; notation §9 #16), and never put load-bearing content in a
collapsible (#15).

**Engine (nn.js / viz.js / course.js) — exact contracts used in Part II:**
- `NN.worked221({W1,b1,W2,b2,x,y})` → live `Value`s + `.forward` + `.grads`/`.gradArray` + `.loss` +
  `.sgdStep(eta)`. **⚠️ CRITICAL TRAP: `worked221()` called with NO args reproduces the RETIRED §5.4 network
  (`W1=[[0.5,-0.3],[0.2,0.8]]`, `W2=[0.7,-0.4]`, `x=[0.6,-0.2]`, default `sgdStep` η=0.5). Every Part II page
  that uses it MUST pass explicit TN-1 config** (see the two frozen configs below) **and MUST pass the explicit
  η** to `sgdStep`. Do not rely on any default.
- TN-1 **input 1** config (dead-unit case): `{W1:[[0.5,-0.3],[0.8,0.2]], b1:[0.1,-0.1], W2:[0.6,-0.9], b2:0.2, x:[1.0,2.0], y:1}` → `sgdStep(0.1)`.
- TN-1 **input 2** config (spread case): same weights, `x:[0.60,-0.20]`, `y:1` → `sgdStep(0.1)`.
- `NN.MLP([nin,...,nout],{act:'tanh'|'relu', outAct:'identity', rng, init, initScale})`; output is LOGITS by
  default (pair with `bceWithLogits`/`crossEntropyLoss`). `.getParameters()/.setParameters(arr)/.zeroGrad()`.
- `NN.SGD(params,{lr,momentum,weightDecay,nesterov})`; `NN.Adam(params,{lr,beta1,beta2,eps,weightDecay,decoupled})`
  — `decoupled:true` == AdamW. `.step()`, `.zeroGrad()`, `.t`, `.m`, `.v` exposed.
- `NN.Trainer({model,X,y,loss:'bce'|'mse'|'ce',opt,batchSize,rng,skipZeroGrad})` — `.step()`, `.steps(n)`,
  `.evalLoss(X,y)`, `.accuracy()`, `.history`, `.step_`. **`batchSize` and `skipZeroGrad` are the two flags the
  zero_grad demo (page 15) requires.**
- `NN.makeDataset('moons'|'circles'|'xor'|'spiral'|'blobs', n, {rng,noise,labelNoise})`,
  `NN.trainTestSplit`, `NN.gradCheck(buildLoss, params, {eps,tol})`, `NN.crossEntropyLoss(logits, targetIdx)`,
  `NN.bceWithLogits(z,y)`, `NN.logSumExp`, `NN.softmaxValues`, `NN.layerNorm(M,{eps,gamma,beta})`,
  `NN.rmsNorm(M,{eps,gamma})`.
- viz: `new Plot(canvas,{xmin,xmax,ymin,ymax,xlog,ylog,xlabel,ylabel})` → `.clear/.grid/.trace(xs,ys,color,w)/
  .hline/.vline/.label`; `new Surface3D(canvas,{f,xRange,yRange,mode:'contour'|'wireframe',resolution,wireRes,
  az,el})` → `.addPoint(x,y)` (draws a trajectory), drag-to-rotate in wireframe; `new Timeline(canvas,{ylog,
  ylabel,xlabel})` → `.push(step,value)`, `.addMarker(step,label,color)`; `new NetGraph(canvas,{layers,weights,
  activations,labels})` → `.animateForward(acts)`, `.animateBackward(grads)`, `.setWeights/.setActivations`;
  `new Heatmap(canvas,{data,rowLabels,colLabels,colormap,cellText,hover})`; `vizUtils.cssVar(name,fallback)` for
  theme-aware canvas colors.
- `makeCtrl(parentEl,{label,min,max,step,value,unit,fmt})` → `{input,get(),onInput(fn)}` (slider only; log
  sliders carry an exponent, compute `10**v` inside `draw`). `renderQuiz(id, questions, title)`:
  `{q,opts,a,why}` (MC) or `{q,num,tol,unit,why}` (numeric; always pass explicit `tol`; escape `\\` in JS
  strings). `eng(value,unit,digits)` for engineering-notation readouts.

**Recurring-thread tags (mandatory, notation §10; announce every beat):**
`[THREAD:TN1]`, `[THREAD:CHAIN]` (the chain rule), `[THREAD:MEM]` (the memory ledger), `[THREAD:QWEN]`.

**Cross-references into Part I (state, don't re-derive; numbering per `decisions.md` §D-21a — the single truth):**
the **chain rule is p.5**; **TN-1's architecture + forward pass to 0.3727 is p.6** (`constants.md` §8.1);
**probability / the Gaussian / μ+σε reparameterization is p.8** (Part II page 12 cashes this seed); **logs,
logsumexp, softmax, and CE = NLL is p.9**; **XOR / the linear-collapse argument is p.10**; **activations + the
MLP's honest limits is p.11**. The transformer block, attention, and the LoRA/SVD trunk pages come in Part III+
(page 25 onward). ⚠️ Do **not** cite Part I p.10 for TN-1's forward pass or for any autograd beat — p.10 is
XOR/linear-collapse and carries no autograd (§D-21a).

---

# 12 — loss-from-mle.html  (D-21: 12)
**Title:** "What 'Learning' Optimizes: Loss from Maximum Likelihood" · **Part II — How It Learns** · build: **O**

**Learning objectives**
1. Define a loss function as the negative log-likelihood of the data under the model's predictions.
2. Derive cross-entropy (Branch C, categorical) and MSE (Branch A, Gaussian) from **one** principle.
3. Compute cross-entropy on the course's canonical logits and read $\ln V$ as the random-init loss.
4. Explain why the LogSumExp trick exists (the math on paper ≠ the math in the machine — foreshadows page 15).

**PREDICT (`.box try`, open the page):** "A freshly-initialised Qwen3-8B, before any training, is asked to
predict the next token from a vocabulary of 151,936. What loss (in nats) do you expect on the very first batch?
Write down a number before scrolling." (Answer on the page: $\ln 151{,}936 = 11.93$.) `[THREAD:QWEN]`

**Section outline**
- *Intuition.* Learning = making the data we saw **probable** under the model. A loss is a scalar that is small
  when the model is confident and right, large when confident and wrong. One derivation, two branches.
- *MLE, one line.* Maximise $\prod_n p_\theta(y_n\mid x_n)$ ⇔ minimise $-\sum_n \ln p_\theta(y_n\mid x_n)$.
  Negative-log-likelihood **is** the loss. Symbol Ledger required.
- *Branch C — categorical → cross-entropy.* For a softmax over $V$ classes,
  $\mathcal{L} = -\ln \hat y_c = -\ln \dfrac{e^{z_c}}{\sum_v e^{z_v}}$. Units: **nats**.
- **Worked number (frozen, `constants.md` §9.2, [DER]):** logits $z=[2.0,\ 1.0,\ 0.1]$, true class $c=0$.
  $e^z=[7.389056,\ 2.718282,\ 1.105171]$, $\sum=11.212509$; $\hat y=[0.659001,\ 0.242433,\ 0.098566]$;
  $\mathcal{L}=-\ln(0.659001)=\mathbf{0.417030}$ nats; $\partial\mathcal{L}/\partial z=\hat y-y=
  [-0.340999,\ +0.242433,\ +0.098566]$ — **sums to zero, always.** Reuse this exact triple course-wide (D-08).
- *Branch A — Gaussian → MSE (cashes the p.8 seed, `.box key`).* **Part I page 8 planted the Gaussian
  $y\sim\mathcal N(\mu,\sigma^2)$ and the $\mu+\sigma\varepsilon$ reparameterization; this is the page that spends
  it.** Put a model's prediction $\hat y$ in the mean slot ($y\sim\mathcal N(\hat y,\sigma^2)$) and the negative
  log-likelihood is $-\ln p=\tfrac{1}{2\sigma^2}(y-\hat y)^2+\text{const}$, i.e. NLL $\propto \tfrac12(y-\hat y)^2$.
  **MSE is not a separate fact — it is Branch A of the same derivation, and it is literally the log of the p.8
  Gaussian.** State the callback in one sentence: *"the bell curve you met on page 8 is where squared error comes
  from."* (Diffusion's loss will be a further corollary of this same Gaussian, D-14; foreshadow one sentence — and
  note the diffusion track cites p.8's reparameterization directly.)
- *Loss anchors (`.box rule`, `constants.md` §9.1, [DER]):* random-init loss $=\ln V=\mathbf{11.93}$ nats for
  Qwen3-8B ($V=151{,}936$); perplexity at init $=V$ exactly ($\text{PPL}=e^{\mathcal L}$); coin-flip
  $=\ln 2=0.6931$; **always plot $\ln V$ as a horizontal line on every loss curve** — a curve that sits on it has
  learned nothing. `[THREAD:QWEN]` ⚠️ Not 11.76 (that is Llama-3's $V=128{,}256$; retired, D-01).
- *LogSumExp (why-this-exists).* $\ln\sum_v e^{z_v}=m+\ln\sum_v e^{z_v-m}$, $m=\max_v z_v$. Naïve `exp` overflows;
  this is exact and stable. **This is the same lesson as page 15's "float that isn't zero," arriving early:**
  the fused `crossEntropyLoss`/`bceWithLogits` in this course never forms the probability explicitly.
- *D-17 foreshadow (one sentence, `.box key`):* "Hold onto this — every model in this course is this same
  machine; **the only thing that will change is what we ask it to predict.**"

**Misconceptions (`.box warn`)**
- "Cross-entropy and MSE are unrelated losses." → Both are NLL; they differ only in the assumed noise
  distribution (categorical vs Gaussian).
- "Higher accuracy always means lower loss." → Loss is calibrated confidence, not just correctness; a right-but-
  unconfident prediction still carries loss.
- "The softmax gradient is complicated." → It is exactly $\hat y - y$, and it sums to zero.

**Demo — "Cross-entropy explorer"** (primitives: `makeCtrl` ×3 + `Plot` + a `Heatmap` or bar readout).
Three sliders drive the logits $z_0,z_1,z_2$ (range $-4..4$, step 0.1, defaults $2.0,1.0,0.1$). JS calls
`NN.softmaxValues([z0,z1,z2])` live and `NN.crossEntropyLoss` for $c=0$. Plot the softmax bars and print
$\hat y$, $\mathcal L$, and $\partial\mathcal L/\partial z$. **The math shown with live values substituted in**
(notation §9 #12). *Aha:* drag $z_0$ up — loss falls toward 0; drag it down — loss rises without bound while
$\partial\mathcal L/\partial z$ stays bounded in $[-1,1]$. A second toggle sets all logits equal → $\mathcal L=\ln 3=1.0986$,
the "random guess" value, mirroring $\ln V$.

**Code artifact (`.box try`):** `code/12_cross_entropy.py` — reproduces the frozen triple with
`torch.nn.functional.cross_entropy` on **logits** (never `log(softmax(...))`) and asserts
`abs(loss.item() - 0.417030) < 1e-5`.

**Quiz (6; ≥1 numeric)**
1. (num) CE for $z=[2.0,1.0,0.1]$, $c=0$ → **0.4170** nats (tol 0.005). Distractor rationale: 0.242 = $\hat y_1$
   confused for a loss.
2. (MC) Units of this course's loss → nats. Distractor: "bits" (that's $\log_2$).
3. (num) Random-init loss of Qwen3-8B → **11.93** nats (tol 0.05). Distractor list includes 11.76 (Llama-3 trap).
4. (MC) $\partial\mathcal L/\partial z$ for softmax+CE → $\hat y - y$, sums to 0. Distractor: "$-1/\hat y_c$" (the
   log term only, missing the softmax Jacobian).
5. (MC) Why LogSumExp → prevents `exp` overflow; identical value. Distractor: "makes it differentiable" (it
   already is).
6. (MC) MSE relates to MLE how → Gaussian-noise NLL. Distractor: "unrelated, chosen for smoothness."

**Cross-refs:** ← Part I p.9 (logs/logsumexp/softmax/CE=NLL primitives), ← Part I p.8 (the Gaussian +
$\mu+\sigma\varepsilon$ reparameterization — Branch A cashes exactly this seed). → page 13 (this loss is the
surface we now descend); → page 15 (LogSumExp = the float-reality beat, in full); → Part III LLM loss ("you
already know it — Branch C, per position", D-14); → Part VI diffusion loss (Branch A's Gaussian again, D-14).

---

# 13 — gradient-descent-landscape.html  (D-21: 13)
**Title:** "Downhill: Gradient Descent and the Loss Landscape" · **Part II** · build: **O**

**Learning objectives**
1. State the update rule $\theta_{k+1}=\theta_k-\eta\,\mathbf g_k$ and identify each symbol (note: $k$ = step).
2. Explain why $-\nabla\mathcal L$ is the direction of **steepest** descent.
3. Recognise the four visual signatures of a learning rate that is too small, right, too big, and unstable.
4. Compute the critical learning rate on a quadratic and predict divergence above it.

**PREDICT (`.box try`):** "On a long narrow valley, the gradient points *downhill*. Does it point toward the
minimum? Sketch the arrow before the demo." (Answer: no — on an anisotropic bowl the gradient is ⊥ to the
contour, which points mostly **across** the valley, not along it. This sets up momentum on page 17.)

**Section outline**
- *Intuition.* The loss is a landscape over parameter space; training rolls downhill. `[THREAD:CHAIN]` — the
  gradient is assembled by the chain rule (page 14 does it by hand).
- *Update rule (`.box rule`).* $\theta_{k+1}=\theta_k-\eta\,\mathbf g_k$, $\mathbf g_k=\nabla_\theta\mathcal L(\theta_k)$.
  **Translation note (notation §3.2):** "optimisation papers write this step index $t$; we reserve $t$ for the
  diffusion timestep, so **our step index is $k$**."
- *Why $-\nabla\mathcal L$ is steepest.* First-order Taylor: $\mathcal L(\theta+\mathbf u)\approx\mathcal L(\theta)+\mathbf g^\top\mathbf u$;
  over unit $\mathbf u$, $\mathbf g^\top\mathbf u$ is minimised at $\mathbf u=-\mathbf g/\|\mathbf g\|$ (Cauchy–Schwarz). Deep-dive
  collapsible: the full Cauchy–Schwarz argument.
- *Learning rate — what too-big/too-small actually look like.* Too small: smooth, glacial, monotone. Right:
  brisk fall then plateau. Too big: bounces on a plateau, never settles. Unstable: **diverges to NaN in a few
  steps.** Tie each to a curve shape (this seeds page 24's debugging table).
- **Worked number (`constants.md` §9.3, [DER]):** on a quadratic with curvature $\lambda_{\max}=20$ (the demo
  ravine), the critical step is $\eta_{\text{crit}}=2/\lambda_{\max}=\mathbf{0.1}$ **exactly**; above it the
  iterate's error is multiplied by $|1-\eta\lambda|>1$ each step and blows up.
- *The 2026 landscape, honestly (`.box key`).* High-dimensional loss surfaces are dominated by **saddle points**,
  not local minima; most minima that SGD finds are about equally good; the "getting stuck in a bad local
  minimum" fear from 1990s neural-net lore is largely obsolete at scale. Say this plainly.

**Misconceptions (`.box warn`)**
- "The gradient points at the minimum." → It points ⊥ to the local contour; on a stretched valley that is mostly
  across, not toward. (The zigzag demo, and the reason momentum exists.)
- "Smaller LR is always safer." → Too small stalls indefinitely and looks identical to a dead/frozen network.
- "Training gets stuck in local minima." → Saddles dominate; and overparameterised minima are mostly equivalent.

**Demo 1 ⭐ — "Gradient descent on a loss surface" (flagship; primitives: `Surface3D` contour + `makeCtrl` +
trajectory).** Surface is an anisotropic quadratic $f(x,y)=\tfrac12(a x^2+b y^2)$ with $a=1,b=20$ (so
$\lambda_{\max}=20$). Slider: $\eta$ (0.005–0.25, step 0.005). JS runs **real** GD from a fixed start
$(-4,3)$: each frame `p = p - η·∇f(p)` and `surface.addPoint(px,py)` draws the path on the contour. Readout:
step count to $\|\nabla f\|<10^{-3}$, or "**DIVERGED**" when it passes 0.1. *Aha:* below 0.1 it converges (zigzag
across the valley, worse near 0.1); **exactly at/above 0.1 it diverges** — the learner *sees* $\eta_{\text{crit}}$.
Optional `mode:'wireframe'` toggle to rotate the bowl. The equation $\theta_{k+1}=\theta_k-\eta\nabla f$ printed
with the live $\eta$ substituted in.

**Code artifact (`.box try`):** `code/13_lr_sweep.py` — 1-D quadratic, sweeps $\eta\in\{0.05,0.1,0.15\}$, prints
convergence/divergence, confirms the $2/\lambda_{\max}$ threshold on his machine.

**Quiz (6; ≥1 numeric)**
1. (num) $\eta_{\text{crit}}$ for $\lambda_{\max}=20$ → **0.1** (tol 0.005).
2. (MC) $-\nabla\mathcal L$ is → steepest-descent direction (Cauchy–Schwarz). Distractor: "the line to the minimum."
3. (MC) LR too small looks like → smooth, flat, glacial. Distractor: "bouncing plateau" (that's too big).
4. (MC) Why the gradient zigzags on a valley → it is ⊥ to elongated contours.
5. (MC) What dominates high-dim loss surfaces → saddle points. Distractor: "many bad local minima."
6. (num) With $\eta=0.15,\lambda=20$, the error multiplier $|1-\eta\lambda|$ per step → **2.0** (tol 0.1) →
   diverges.

**Cross-refs:** ← page 12 (the surface is the loss). → page 14 (how $\mathbf g_k$ is actually computed);
→ page 17 (momentum fixes the zigzag; the $\eta_{\text{crit}}$ idea reappears). → page 43 (roofline reuses
`Surface3D`).

---

# 14 — backprop-by-pencil.html  (D-21: 14)  ⭐ THE COURSE SPINE
**Title:** "Backpropagation, by Pencil: TN-1's Nine Gradients" · **Part II** · build: **O**

> **This is the spine of the course (D-02).** Nine downstream pages name TN-1. Every number here is frozen in
> `constants.md` §8 (input 1) and §8.7 (input 2). **Do not recompute, do not round differently, do not
> re-derive.** `[THREAD:TN1 — the object]` `[THREAD:CHAIN — the verb, made mechanical]`

**Learning objectives**
1. Define $\boldsymbol\delta^{(\ell)}=\partial\mathcal L/\partial\mathbf z^{(\ell)}$ and state the four backprop equations.
2. Hand-compute all nine parameter gradients of TN-1 for input 1 and read the dead-unit lesson.
3. Hand-compute all nine for input 2 (no dead unit) and take one SGD step, watching the loss fall.
4. Explain why backprop is cheap: one backward pass, same cost as one forward pass.

**PREDICT (`.box try`):** "TN-1's first hidden pre-activation is $z_{1,1}=0.5(1)+(-0.3)(2)+0.1$. Compute it.
Then predict: what is the gradient of the loss w.r.t. the weight $W_{2}$ that multiplies this unit's output?"
(Answer: $z_{1,1}=0$, so $a_{1,1}=\tanh 0=0$, so $\partial\mathcal L/\partial W_{2,1}=0$ — that weight **cannot
move this step**. But the *upstream* gradient still flows, because $\tanh'(0)=1$. Two lessons, one dead unit.)

**Section outline** — TN-1 recap (`.box rule`, from Part I **p.6**, the forward pass to 0.3727; the chain rule
itself is Part I **p.5**): $2\to2(\tanh)\to1(\text{sigmoid})\to\text{BCE}$,
9 params. $W_1=[[0.5,-0.3],[0.8,0.2]]$, $\mathbf b_1=[0.1,-0.1]$, $W_2=[0.6,-0.9]$, $b_2=0.2$, $\eta=0.1$.

- *The central object $\boldsymbol\delta$.* $\delta^{(\ell)}_i=\partial\mathcal L/\partial z^{(\ell)}_i$ — "how much
  the loss changes per unit change in this pre-activation." Everything else is $\delta$ times a cached forward
  value.
- *The four backprop equations (`.box rule`, Symbol Ledger mandatory):*
  (BP1) output: $\boldsymbol\delta^{(L)}=\nabla_{\mathbf a}\mathcal L\odot\phi'(\mathbf z^{(L)})$ — for BCE+sigmoid this
  collapses to $\hat y-y$.
  (BP2) recurse: $\boldsymbol\delta^{(\ell)}=(W^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)})\odot\phi'(\mathbf z^{(\ell)})$.
  (BP3) weights: $\partial\mathcal L/\partial W^{(\ell)}=\boldsymbol\delta^{(\ell)}\mathbf a^{(\ell-1)\top}$.
  (BP4) biases: $\partial\mathcal L/\partial\mathbf b^{(\ell)}=\boldsymbol\delta^{(\ell)}$.
  **Note BP3 needs the cached $\mathbf a^{(\ell-1)}$ — this is why training eats VRAM** (foreshadow page 18,
  `[THREAD:MEM]`, and the activation-cache line of the Symbol Ledger in notation §8).

**WORKED — INPUT 1 (`constants.md` §8.1–8.3, [DER, mpmath 30 d.p.]) — the dead-unit case.**
Forward: $z_1=[0.0,\ 1.1]$; $a_1=[0.0,\ 0.8005]$; $z_2=-0.5205$; $\hat y=0.3727$; $\mathcal L=\mathbf{0.9869}$.
Backward: $\partial\mathcal L/\partial z_2=\hat y-y=-0.6273$; $\partial\mathcal L/\partial W_2=[0.0,\ -0.5021]$
(first entry dead because $a_{1,1}=0$); $\partial\mathcal L/\partial\mathbf a_1=[-0.3764,\ 0.5645]$;
$\tanh'(0)=1.0$, $\tanh'(1.1)=0.3592$; $\boldsymbol\delta_1=[-0.3764,\ 0.2028]$;
$\partial\mathcal L/\partial W_1=[[-0.3764,-0.7527],[0.2028,0.4056]]$; $\partial\mathcal L/\partial\mathbf b_1=[-0.3764,0.2028]$.
One SGD step ($\eta=0.1$): $\hat y:0.3727\to0.4395$, $\mathcal L:0.9869\to0.8222$, $\Delta\mathcal L=\mathbf{-0.1646}$.
**"It learned. He did it with a pencil."** ⚠️ Use these exact last digits (§8.3 corrected six brief roundings;
e.g. $\mathcal L=0.9869$ not 0.9870, $W_1[0][1]=-0.7527$ not $-0.7528$).

- *The dead-unit lesson (`.box key`).* $a_{1,1}=0\Rightarrow\partial\mathcal L/\partial W_{2,1}=0$ (visible in the
  result — that weight is frozen this step), **yet** $\tanh'(0)=1$ so the gradient flows *back through* the unit
  undiminished. Two contrasting facts in one hand computation.

**WORKED — INPUT 2 (`constants.md` §8.7, [DER, verified three ways]) — the healthy case.** Same weights,
$x=[0.60,-0.20]$. Forward: $z_1=[0.46,0.34]$; $a_1=[0.4301,0.3275]$; $z_2=0.1633$; $\hat y=0.5407$;
$\mathcal L=\mathbf{0.6148}$. $\tanh'(0.46)=0.8150$, $\tanh'(0.34)=0.8928$ (both healthy — no dead unit, no
saturation). The **nine gradients (frozen):**
$\partial\mathcal L/\partial W_1=[[-0.1348,+0.0449],[+0.2214,-0.0738]]$;
$\partial\mathcal L/\partial\mathbf b_1=[-0.2246,+0.3690]$;
$\partial\mathcal L/\partial W_2=[-0.1975,-0.1504]$; $\partial\mathcal L/\partial b_2=-0.4593$.
One SGD step ($\eta=0.1$): $\hat y:0.5407\to0.5695$, $\mathcal L:0.6148\to0.5630$, $\Delta\mathcal L=-0.0518$.
**These nine live gradients are handed forward to page 17** (the gradient-spread → Adam beat). State that here.

- *Why backprop is cheap (`.box rule`, `constants.md` §5 flavour).* One backward pass reuses the forward pass's
  cached values and costs ~the same as one forward pass — $O(1)$ passes, not $O(P)$. Finite differences would
  need one forward pass **per parameter**.

**Misconceptions (`.box warn`)**
- "A zero gradient means a dead/broken network." → Here $\partial\mathcal L/\partial W_{2,1}=0$ is correct and
  local; the unit is merely off *for this input*.
- "Backprop is a different algorithm from the chain rule." → It **is** the chain rule, ordered to reuse
  intermediate results (reverse mode).
- "You must store the whole graph to get gradients." → You store the forward **activations**; that is the memory
  cost, and it is exactly BP3's $\mathbf a^{(\ell-1)}$.

**Demo — "NetGraph: forward and backward over TN-1" (primitives: `NN.worked221` + `viz.NetGraph`).**
Call `NN.worked221({W1:[[0.5,-0.3],[0.8,0.2]], b1:[0.1,-0.1], W2:[0.6,-0.9], b2:0.2, x:[1.0,2.0], y:1})`
(explicit TN-1 config — never the default). Build `NetGraph` with `layers:[2,2,1]`, weights and activations from
the result, node grads from `.nodes`. Buttons **Animate forward →** (signal L→R, node fills = activations) and
**← Animate backward** (`.animateBackward` with $[\,[0,0],[\text{a1\_1.grad},\text{a1\_2.grad}],[\text{yhat.grad}]\,]$).
A toggle switches to input 2's config so the learner watches the dead unit *light up*. Readout prints
$\hat y$, $\mathcal L$, and $\delta^{(2)}=\partial\mathcal L/\partial z_2$ live. Every displayed number equals the
frozen table (notation §9 #22).

**Code artifact (`.box try`):** `code/14_backprop_tn1.py` — computes both inputs' nine gradients by explicit NumPy
chain rule (no autograd), prints them to match the two frozen tables. Used again on page 15 for the autograd
cross-check.

**Quiz (7; ≥1 numeric)**
1. (num) Input 1, $\partial\mathcal L/\partial b_2$ → **-0.6273** (tol 0.001).
2. (num) Input 1, first entry of $\partial\mathcal L/\partial W_2$ → **0.0** (tol 0.001). Distractor rationale: not
   because the network is broken — because $a_{1,1}=0$.
3. (num) Input 2, $\mathcal L$ before the step → **0.6148** (tol 0.002).
4. (MC) BP3 caches which forward quantity → the previous layer's activation $\mathbf a^{(\ell-1)}$. Ties to memory.
5. (MC) $\tanh'(0)$ → **1** (so the dead unit still passes gradient). Distractor: 0.
6. (MC) Backprop vs finite differences → one backward pass vs one pass per parameter.
7. (num) Input 1, $\Delta\mathcal L$ after one SGD step ($\eta=0.1$) → **-0.1646** (tol 0.001).

**Cross-refs:** ← Part I p.6 (TN-1's forward pass to 0.3727), ← Part I p.5 (the chain rule this mechanises).
→ page 15 (verify these nine numbers in PyTorch autograd; hit the float reality); → page 17 (input 2's nine
gradients → the 10.22× spread → Adam); → Part III (TN-1 recast as one attention head; TN-1 as a LoRA target).

---

# 15 — autograd-float-reality.html  (D-21: 15)  ⭐ "THE FLOAT THAT ISN'T ZERO"
**Title:** "What PyTorch Actually Does: Autograd, and the Float That Isn't Zero" · **Part II** · build: **O**

> **Mandated framing (D-13, `constants.md` §8.4) — agents may not regress it.** This lands where autograd stops
> being magic. `[THREAD:TN1]` `[THREAD:CHAIN]`

**Learning objectives**
1. Describe the computation graph / tape and that `.backward()` is reverse-mode autodiff, i.e. backprop.
2. Reproduce TN-1's nine gradients with `loss.backward()` and reconcile them with page 14's pencil values.
3. Explain why a gradient that is "exactly 0" on paper prints `1.4e-08` in float32 (non-associativity).
4. Explain why `zero_grad()` exists, and see it fail — **with mini-batches** — when omitted.

**PREDICT (`.box try`):** "You just computed $\partial\mathcal L/\partial W_2$'s first entry as **exactly 0** by
hand. When PyTorch computes the same gradient in float32, what will it print?" (Answer: `1.4020191230201817e-08`.
Both are right.)

**Section outline**
- *The three things people confuse (`.box rule`).* (a) autograd ≠ symbolic differentiation (no formula is
  produced); (b) autograd ≠ numerical/finite differences (no $\epsilon$ perturbation); (c) it is the **chain rule,
  executed on a recorded graph, in reverse**. `[THREAD:CHAIN]`
- *The tape.* Each op records its inputs and a local `_backward`. `.backward()` walks the graph in reverse
  topological order accumulating grads. (nn.js's `Value` engine does exactly this; the learner can read it.)
- *Reverse vs forward mode (deep-dive collapsible, summary states the result):* "Reverse mode costs one pass to
  get all $P$ gradients; forward mode would cost $P$ passes — that is the whole reason we use it."
- **★ THE FLOAT THAT ISN'T ZERO — THIS PAGE IS ITS HOME (`[THREAD:TN1]` `[THREAD:CHAIN]`, `constants.md` §8.4).**
  ✅ **HOME RESTORED (2026-07-16, per `decisions.md` §D-21a).** A previous integrator relocated this beat's home
  to Part I page 10 on §8.4's stale "page ~10" pointer. Under §D-21a, **Part I p.10 is XOR / the linear-collapse
  argument and carries NO autograd** — so the beat lives **here, on page 15, in full**, at the exact moment
  `.backward()` reproduces page 14's nine pencil numbers and autograd stops being magic. Ship the mandated `.box
  key` **verbatim** (`constants.md` §8.4 — agents may not regress it), presented as the discovery it is, **not**
  as a callback to page 10:
  > *"By hand you got exactly 0. Torch prints `1.4e-08`. **Both are right.** `0.5 − 0.6 + 0.1` is not zero in
  > binary floating point — it is the textbook non-associativity demo, and you just hit it for free. This is the
  > same lesson as the logsumexp page, arriving made-literal: **the math on paper and the math in the machine are
  > not the same math.** The gradient is zero to seven decimals, the unit is dead, and the lesson stands — but the
  > number on your screen will not be `0.0`, and a course that told you it would be is a course that has never run
  > its own code."*
  Then carry the full float depth (`constants.md` §8.4, verbatim values): the source expression
  `z_{1,1} = 0.5*1.0 + (-0.3)*2.0 + 0.1` → **float64 `2.7755575615628914e-17`**, **float32 `-2.2351742e-08`**, and
  **`torch float32 dW2 → [1.4020191230201817e-08, -0.5021151900291443]`** — the first entry is the "exactly zero"
  gradient, alive as `1.402e-08`. Then explain *why the graph produced it*: the tape accumulates the local
  products in reverse-topological order, and `0.5 − 0.6 + 0.1` is evaluated left-to-right in binary, so the
  rounding residue survives. The logsumexp lesson from page 12 (and Part I p.9) is the **same** paper-vs-machine
  theme — now made literal on a gradient he computed by hand. (§8.4's own "page ~10 / arriving three pages early"
  wording predates §D-21a and is deliberately not reprinted; the beat is six pages *after* logsumexp now, not
  before it.)
- **The mandated autograd assertions (`.box rule`, exact tolerances — do not change).**
  Input 1 (`constants.md` §8.5): `assert torch.allclose(model[0].weight.grad, torch.tensor([[-0.3764,-0.7527],[0.2028,0.4056]]), atol=1e-4)` — **`-0.7527` not `-0.7528`** (the brief's value survives only by a 10% tolerance coincidence).
  Input 2 (`constants.md` §8.7): `assert torch.allclose(model[0].weight.grad, torch.tensor([[-0.13475,0.04492],[0.22140,-0.07380]]), atol=1e-5)` — **5 d.p., `atol=1e-5`, NOT `1e-4`** (at 4 d.p. the margin is only 2.1×, the same trap).
- *`zero_grad()` — why it exists (`constants.md`-adjacent, `[THREAD:MEM]` callback).* Gradients **accumulate** by
  default (a feature: gradient accumulation for large effective batches on limited memory; foreshadow page 18).
  Omit `zero_grad()` and step $k$ carries ~$k\times$ the intended gradient.

**Misconception (`.box warn`) — the load-bearing one.** "You will see the gradient come out as exactly 0.0." →
**False in every float.** Full framing above. Also: "float error means the computation is wrong" → it is
correct to 7 digits; the paper value is the idealisation.

**Demo — "Forgot zero_grad() — with mini-batches" (primitives: `NN.MLP`+`NN.Trainer`+`viz.Timeline`).**
⚠️ **Must use mini-batches** — the nn.js test suite proved full-batch **hides** the bug (a constant-scaled
gradient still points downhill; only the *stochastic* per-batch accumulation visibly diverges). Build two
`Trainer`s on `NN.makeDataset('moons', 200, {rng})` with **`batchSize:16`**: one normal, one with
**`skipZeroGrad:true`**. Two `Timeline` loss curves (log-y) on the same axes, stepped from rAF. *Aha:* the
correct run descends; the `skipZeroGrad` run **rises then NaNs within tens of steps**. A reset re-seeds. Readout
prints both current losses and the step at which the buggy run diverged.

**Code artifact (`.box try`):** `code/15_autograd_check.py` — builds TN-1 as `nn.Sequential`, sets the frozen
weights, runs `loss.backward()` for both inputs, prints the nine grads, runs both mandated asserts, and prints
the `1.4020191230201817e-08` first entry so the learner sees it on **his** box.

**Quiz (6; ≥1 numeric)**
1. (MC) `.backward()` implements → reverse-mode autodiff = the chain rule on a recorded graph. Distractors:
   symbolic diff; finite differences.
2. (num) PyTorch's float32 value for the "exactly zero" gradient entry → **1.4e-08** (tol 5e-9). Distractor: 0.
3. (MC) Why isn't it 0 → binary floating-point non-associativity of `0.5-0.6+0.1`.
4. (MC) Why `atol=1e-5` on input 2, not `1e-4` → the small-magnitude gradient (0.0449) makes 4 d.p. a tolerance
   coincidence.
5. (MC) Omitting `zero_grad()` with mini-batches → loss rises then NaNs (accumulating gradients). Distractor:
   "no effect" (the full-batch illusion).
6. (MC) Reverse mode is preferred because → all $P$ grads in one pass vs $P$ passes for forward mode.

**Cross-refs:** ← page 14 (the nine pencil numbers being verified). → page 12 (LogSumExp, the same
paper-vs-machine lesson). → page 18 (`zero_grad` accumulation = the gradient-accumulation memory trick).

---

# 16 — minibatch-and-noise.html  (D-21: 16)
**Title:** "Batch, Mini-Batch, Stochastic — and Why Noise Is a Feature" · **Part II** · build: **S** (§D-21a)

**Learning objectives**
1. Distinguish full-batch, mini-batch, and stochastic gradient descent, and define an epoch vs a step.
2. State the $\mathcal L$-vs-$J$ distinction and locate the $1/B$ factor exactly.
3. Give three real mechanisms by which mini-batch noise **helps** generalisation and speed.
4. Read a noisy loss curve without mistaking variance for divergence.

**PREDICT (`.box try`):** "You can compute the exact gradient over all your data every step (full batch), or a
noisy estimate over 32 examples. Which trains a better model — and which trains faster? Commit before reading."

**Section outline**
- *The three regimes.* Full batch (exact $\nabla J$, expensive, one step/epoch); stochastic (1 example, maximal
  noise); **mini-batch** (the universal compromise, $B\in[8,\text{few}\,{\times}10^3]$). Define **epoch** (one
  pass over the data) vs **step** (one optimizer update) — and warn that LLM training often runs **<1 epoch**, so
  schedules are indexed by **step $k$**, not epoch (ties to page 21).
- *$\mathcal L$ vs $J$ (`.box rule`, notation §2).* $\mathcal L$ = per-example loss; $J=\frac1B\sum_{b}\mathcal L_b$
  = batch-averaged. **The $1/B$ is where learning-rate confusion breeds** — state it once here and hold it
  course-wide. A gradient summed (not averaged) over the batch scales the effective LR by $B$.
- *Why noise is a FEATURE — three real mechanisms.* (1) **Escapes saddles/sharp minima:** stochastic kicks push
  the iterate off saddle plateaus. (2) **Implicit regularisation:** SGD's noise biases toward flatter, better-
  generalising minima. (3) **Throughput:** more, cheaper, slightly-wrong steps beat few exact ones — you cover
  more ground per unit compute. All three are real; name them separately.
- *Reading a noisy curve.* Mini-batch loss is jagged by construction; judge trend over a window, not step-to-step.
  The determinism tell (page 24): a spike at the **same** step every seed is data; a random-step spike is
  numerics.

**Misconceptions (`.box warn`)**
- "Bigger batch is strictly better." → Larger $B$ = less noise = often **worse** generalisation and diminishing
  returns; and it changes the effective LR.
- "Noise is a necessary evil we tolerate." → It is a **feature** with three mechanisms; pure full-batch GD on
  neural nets generalises worse.
- "Epoch and step are interchangeable." → Schedule by **step**; epochs are meaningless on a token stream.

**Demo — "Batch-size / noise explorer" (primitives: `NN.Trainer` with `batchSize` + `viz.Timeline` + `Surface3D`).**
Train `NN.MLP([2,16,1])` on `NN.makeDataset('moons',256,{noise:0.2})`. Slider: `batchSize` (1, 8, 32, 256 via a
4-step slider with `fmt`). Left: `Timeline` loss curve (log-y); right: the parameter trajectory of two chosen
weights on a `Surface3D` contour of the local loss. *Aha:* `batchSize:1` → very jagged path that still descends
and escapes a plateau; `batchSize:256` → smooth but can stall on a saddle. Readout: final train loss and
wall-steps to threshold, so smaller batches visibly reach a comparable loss in fewer *epochs* though noisier.

**Code artifact (`.box try`):** `code/16_batch_regimes.py` — same MLP on a toy set at $B\in\{1,8,32,\text{full}\}$;
logs steps-to-target and final val loss; shows the effective-LR-scales-with-$B$ effect when the loss is summed
vs meaned.

**Quiz (6; ≥1 numeric)**
1. (MC) The $1/B$ factor lives in → $J$ (batch-averaged loss), not $\mathcal L$.
2. (num) If loss is **summed** not averaged over $B=32$, the effective LR is off by → **32×** (tol 1).
3. (MC) One real reason noise helps → escapes saddles / flatter minima / throughput (any one). Distractor:
   "prevents overflow."
4. (MC) Schedule by → step $k$. Distractor: epoch.
5. (MC) A jagged mini-batch loss curve → normal; judge the trend. Distractor: "diverging, lower the LR."
6. (MC) Larger batch → less noise, often worse generalisation, changed effective LR. Distractor: "strictly
   better."

**Cross-refs:** ← page 13 (saddles), page 15 (the zero_grad demo already used mini-batches). → page 17 (optimizers
average these noisy gradients over time), page 21 (schedule by step), page 24 (reading curves while debugging).

---

# 17 — optimizers-sgd-to-adamw.html  (D-21: 17)  ⭐
**Title:** "Optimizers: SGD → Momentum → RMSProp → Adam → AdamW" · **Part II** · build: **O**

**Learning objectives**
1. Explain each optimizer as fixing one specific failure of the previous one.
2. State momentum's conditioning effect and the $1/(1-\mu)$ amplification.
3. Derive Adam's bias correction and explain why $\beta_2=0.999$ implies a ~1000-step memory.
4. Explain why AdamW decouples weight decay, and why gradient spread (TN-1's 10.22×) motivates per-parameter
   scaling.

**PREDICT (`.box try`):** "On TN-1's input 2 you computed nine gradients. The largest in magnitude is
$\partial\mathcal L/\partial b_2=-0.4593$; the smallest is $\partial\mathcal L/\partial W_1[0][1]=+0.0449$. By
what factor do they differ — and what does that imply for a single global learning rate?" (Answer: **10.22×**;
one $\eta$ either over-steps the big gradient or starves the small one. That is what Adam fixes.)
`[THREAD:TN1 — input 2's nine gradients, reopened from page 14]`

**Section outline** — the lineage, each fixing one thing (`.box rule`):
- **SGD.** $\theta_{k+1}=\theta_k-\eta\mathbf g_k$. Fails on ravines (zigzag, page 13's demo).
- **Momentum.** $\mathbf v_{k}=\mu\mathbf v_{k-1}+\mathbf g_k$, $\theta_{k+1}=\theta_k-\eta\mathbf v_k$. Damps
  oscillation across a ravine while accumulating along it — a **conditioning** fix, not a speed hack. **Worked
  (`constants.md` §9.3, [DER]):** $\mu=0.9\Rightarrow$ steady-state amplification $1/(1-\mu)=\mathbf{10\times}$.
- **RMSProp.** Per-parameter scaling by $1/\sqrt{\text{EMA}(g^2)}$ — gives each parameter its own effective step,
  directly attacking the 10.22× spread.
- **Adam.** Momentum + RMSProp + **bias correction**. $\hat m_k=m_k/(1-\beta_1^k)$, $\hat v_k=v_k/(1-\beta_2^k)$.
  **Worked (`constants.md` §9.3):** at $k=1$ with $\beta_2=0.999$, $v$ is **1000× too small**; the correction
  $1/(1-\beta_2^k)$ removes exactly that init artifact. And $1/(1-0.999)=\mathbf{1000}$ steps of memory — **the
  mechanistic reason warmup exists** (page 21).
- **AdamW.** Decouple weight decay from the gradient: apply $\lambda\theta$ **straight to the weights**, not
  folded into $\mathbf g$ (so Adam's per-parameter scaling never touches it). `NN.Adam(..., {decoupled:true})`
  **is** AdamW; that flag is the entire content of the paper.
- *Translation note (notation §3.2):* "papers index the step $t$; we use $k$."
- **2026 AdamW defaults (`.box rule`, `constants.md` §9.4 / training §7.7, [VP]):** $\beta=(0.9,\mathbf{0.95})$ —
  **not 0.999**: text has huge gradient spikes from rare tokens and a 1000-step memory poisons the denominator;
  $\varepsilon=$ 1e-8; **wd = 0.1**; grad clip 1.0; cosine to $\eta_{\max}/10$. No-decay list: biases, norm
  gains, usually embeddings.
- *Muon (sidebar, one paragraph, `[EST]`/reported).* One momentum buffer (½ Adam's state), ~52% of AdamW FLOPs,
  2-D params only — every real "Muon" run is an AdamW hybrid. Present as "optimisation is not settled," attributed,
  not as the learner's path.

**The spread beat (`.box key`).** $\dfrac{\max|g|}{\min|g|}=\dfrac{0.459260}{0.044917}=\mathbf{10.2246\times}$
(`constants.md` §8.7). **"The largest gradient is ten times the smallest"** — one order of magnitude, all nine
gradients live. And the honest caveat (§8.7): the spread is **structural** — even at a perfectly balanced 1:1
feature ratio it is still 8.64×, so "therefore Adam" survives the "just normalise your inputs" objection.
⚠️ 12.7× is retired (it belonged to the retired §5.4 network).

**Misconceptions (`.box warn`)**
- "Adam adapts the learning rate." → It adapts the **per-parameter scaling of the gradient**; $\eta$ is untouched
  and you still need a schedule.
- "Momentum just makes it faster." → It changes the dynamics (conditioning), damping ravine oscillation.
- "Weight decay = L2." → **For SGD, yes; for Adam, no** — the divergence is the whole reason AdamW exists.
- "Adam is strictly better than SGD." → Well-tuned SGD+momentum still generalises better on vision; Adam wins on
  transformers and wins *untuned*.

**Demo — "Optimizer race" (primitives: `NN.SGD`/`NN.Adam` + `viz.Timeline` + `Surface3D`).** Four runners on the
**same** anisotropic quadratic as page 13 ($\lambda_{\max}=20$) from the same start: SGD, SGD+momentum(0.9),
Adam, AdamW. Each is a real optimizer over 2 params; draw all four trajectories on one `Surface3D` contour and
their four loss curves on one `Timeline`. Sliders: $\eta$ (shared), $\mu$, $\beta_2$. *Aha:* SGD zigzags,
momentum straightens it, Adam/AdamW march nearly straight down the valley because per-parameter scaling erases the
10.22×-style anisotropy. Toggle $\beta_2$ to 0.999 vs 0.95 and watch the early-step behaviour change (seeds page 21).

**Code artifact (`.box try`):** `code/17_optimizer_race.py` — trains the TN-1 MLP (input-2 config) to convergence
under SGD/Adam/AdamW; logs steps-to-target; prints the per-parameter update magnitudes to show Adam equalising
the 10.22× spread.

**Quiz (7; ≥1 numeric)**
1. (num) TN-1 input-2 gradient spread $\max|g|/\min|g|$ → **10.22×** (tol 0.1).
2. (num) Momentum amplification at $\mu=0.9$ → **10×** (tol 0.5).
3. (num) With $\beta_2=0.999$, Adam's memory length $1/(1-\beta_2)$ → **1000** steps (tol 10).
4. (MC) AdamW's one change vs Adam → decoupled weight decay. Distractor: "different LR."
5. (MC) Why LLMs use $\beta_2=0.95$ → rare-token gradient spikes; faster forgetting. Distractor: "saves memory."
6. (MC) "Adam adapts the learning rate" → false; it scales the gradient per parameter.
7. (MC) The spread survives input normalisation because → it is structural (8.64× even at 1:1). Distractor:
   "no — normalise and you don't need Adam."

**Cross-refs:** ← page 14 (input 2's nine gradients), page 13 (ravine + $\eta_{\text{crit}}$). → page 18
(Adam's $m,v$ are the optimizer state that dominates the memory ledger), page 21 (the 1000-step memory → warmup).

---

# 18 — memory-ledger.html  (D-21: 18)  ⭐ THE MEMORY LEDGER · predict-then-measure
**Title:** "The Memory Ledger: 16 Bytes per Parameter, and Does It Fit on YOUR Box?" · **Part II** · build: **O**

> **[THREAD:MEM — the reality check] · [THREAD:QWEN].** The page **MEASURES first, then computes** (D-04, §6.8).
> **It must NOT pre-announce the outcome.** `constants.md` §6.8 / `hardware-ground-truth.md` §2 know what he will
> see (121.6875 GiB, missing 122.05 by 0.36) — the spec knows; the page has him find out.

**Learning objectives**
1. Build the 16-bytes-per-parameter ledger for bf16 mixed-precision + AdamW and explain each row.
2. Compute Qwen3-8B's full-fine-tune state in both GB and GiB (units discipline).
3. **Measure** his own usable memory with `torch.cuda.mem_get_info()` and compare it to the computed need.
4. Explain the escape ladder and the sentence "LoRA is optimizer-state compression, not model compression."

**PREDICT (`.box try`, the whole page hinges on it):** "You are about to compute how much memory a full AdamW
fine-tune of Qwen3-8B needs. Your box was sold as **128 GB**. Before you run anything: predict — does it fit?
Write GiB numbers, not GB. Then run the script and read your machine's *actual* usable memory." **Do not state the
answer on the page above this line.**

**Section outline**
- *Bytes per parameter (`.box rule`, `constants.md` §2.1, [DER]) — the course's most reused table:*
  weights bf16 **2** · gradients bf16 **2** · **fp32 master 4** (bf16's 7 mantissa bits round a 2e-5 update to an
  O(1) weight to **zero**) · Adam $m$ fp32 **4** · Adam $v$ fp32 **4** · **TOTAL 16**. State **both** framings
  (a two-blog-post reader hits an apparent contradiction otherwise): 16 B/param is **4× the fp32 weight size**
  *and* **8× the bf16 weight size** — same 16 bytes. `[THREAD:CHAIN]` callback: the master/grad rows exist
  because backprop (page 14) produced one gradient per weight and page 15's float reality is why fp32 master is
  needed.
- **THE number (`.box worked`, `constants.md` §2.2, [DER]).** $8{,}190{,}735{,}360\times16=131{,}051{,}765{,}760$ B
  = **131.05 GB (decimal)** = **122.05 GiB (binary).** Per component: weights 16.38 GB · grads 16.38 · master
  32.76 · $m$ 32.76 · $v$ 32.76. **Adam's $m+v$ alone = 65.5 GB = 43% of the total** — bigger than the model.
  ⚠️ **State only.** Activations are a **separate, separately-labelled `[EST]` line, always:** 2–6 GB at $B{=}1$,
  $S{=}2048$ with gradient checkpointing; more without. `[THREAD:MEM]`
- *Units discipline (`.box warn`, `constants.md` §0).* Anything compared to hardware capacity is **GiB**; weights
  quoted alone are **GB**. Never mix the two in one comparison. $1\,\text{GiB}=1.0737\,\text{GB}$.
- **★ MEASURE (`.box try`, do not skip, do not spoil).** He runs `code/measure_your_box.py --no-gpu` (already in
  `code/`). Step 1 prints `MemTotal` in B / GB / GiB and `MemAvailable`; it then prints the p.18 question and the
  slack against 122.05 GiB and a **FITS / DOES NOT FIT** verdict **computed from his number, live**. The page's
  job is to have him read the verdict off his own screen, then discuss it — not to have printed it first.
  > *Spec-only, DO NOT put on the page as an assertion (`constants.md` §6.8 [MEA-DEV]): he will see MemTotal
  > = **121.6875 GiB** (firmware carveout eats 6.3125 GiB = 4.9%), so 122.05 needed vs 121.69 available =
  > **−0.36 GiB, does not fit — by 0.3%, before activations.** With ComfyUI up, `mem_get_info` free is only
  > ~19.41 GiB.* The page discusses **after** he measures: "published capacity is never usable capacity; the spec
  > sheet is off by ~5% before you allocate a byte." The near-miss is his, and unfalsifiable by unit convention.
- *The escape ladder (`.box rule`, `constants.md` §2.3, [DER]).* Full AdamW 16 B/param → 131.05 GB / 122.05 GiB.
  8-bit Adam 10 → 81.91 GB / 76.28 GiB (fits). **LoRA r=16 all-linear → 17.08 GB / 15.91 GiB (fits, 7.67×).**
  QLoRA ≈ 4.9–7 GB (19–26×). Reduction claim, exact words (D-05): **trainable state 114.67 GB → 0.61 GB = 187×,
  and 187 is exactly the parameter ratio $8{,}190{,}735{,}360/43{,}646{,}976$**, because optimizer state is linear
  in trainable params.
- **The bridge sentence (`.box key`, ship verbatim, D-05).** *"LoRA is not a model compression technique. LoRA is
  an optimizer-state compression technique. The base model didn't shrink — all 16.38 GB of it is still resident.
  The trainable state shrank by 187×, which is exactly 8,190,735,360 ÷ 43,646,976, because optimizer state is
  linear in trainable parameters. That's the whole trick."* (Full LoRA math is a later trunk page; this is the
  foreshadow.)
- *The LR-vs-trainable-params principle (`.box rule`, D-09, `constants.md` §9.4) — taught here, both tracks
  reuse it.* **Fewer trainable parameters ⇒ higher learning rate.** Full FT moves 8.19B params that already
  encode everything → tiptoe (1e-5–2e-5); textual inversion moves 768 → stride (5e-4–5e-3); LLM LoRA 1e-4–3e-4;
  diffusion **LoRA 1e-4** (NOT the 1e-6 full-FT figure — a 100× trap, D-09).

**Misconceptions (`.box warn`)**
- "128 GB is what you have." → **Measure it.** Usable `MemTotal` is materially less; the carveout is invisible.
- "LoRA shrinks the model / saves activation memory." → It freezes weights but **still backprops the full graph**;
  activations are unchanged (D-03). It shrinks **optimizer state**, by 187×.
- "Adam state is a minor overhead." → It is 4× the bf16 model, 43% of the full-FT footprint.
- Retired numbers to never print: 96 GB AdamW, 224×, "130 GB → 14.3 GB," 128-vs-131 "3 GB over" (all D-03/D-04/D-05).

**Demo — "Memory calculator" (primitives: `makeCtrl` ×3 + `Plot` bar chart + `eng`).** Sliders: parameter count
$P$ (0.5B–70B, log slider), bytes/param method (full 16 / 8-bit 10 / LoRA-state / QLoRA via a stepped slider),
sequence length for the activation `[EST]` band. JS computes each ledger row live and draws a stacked bar
(weights / grads / master / m / v / activations) with a **horizontal line at the learner's measured `MemTotal`**
(entered once, or defaulted with a "measure yours" note — never hard-code 128). Readout in `eng()`: total GB and
GiB, and FITS/OOM vs his line. *Aha:* drag $P$ to 8.19B in full mode and the bar's top **kisses** his measured
line; switch to LoRA and it collapses to ~16% while the base bar stays put.

**Code artifact (`.box try`):** `code/measure_your_box.py` (ships) — run `--no-gpu` here for the memory facts and
the live FITS verdict; the compute/roofline half is used again on page 43. Read-only, installs nothing, and its
comment header explicitly points at ComfyUI's venv if system torch is absent (`hardware-ground-truth.md` §3).

**Quiz (7; ≥1 numeric)**
1. (num) Full-FT state of Qwen3-8B in GiB → **122.05** (tol 0.1).
2. (num) Bytes per parameter, bf16 mixed AdamW → **16** (tol 0). Distractor: 8 (forgot master + one moment).
3. (num) Optimizer-state shrink full→LoRA → **187×** (tol 2). Distractor: 224× (retired).
4. (MC) Which line is state-only vs includes activations → the ledger is state-only; activations are a separate
   `[EST]` line.
5. (MC) What does LoRA compress → optimizer state, not the model. Distractor: "the weights."
6. (MC) Why measure usable memory → published 128 GB ≠ usable; carveout is invisible and unpublished.
7. (num) Adam $m+v$ share of the full-FT footprint → **43%** (tol 2). Distractor: 40% (retired).

**Cross-refs:** ← page 17 (Adam's $m,v$ are these rows), page 15 (`zero_grad` accumulation = the gradient-
accumulation memory trick), page 14 (cached activations = BP3). → page 43 (the compute half of
`measure_your_box.py`: roofline, ridge 227 — **only foreshadowed here, never asserted**); → the trunk LoRA/SVD
pages (they pick up 187× and "rank follows the dataset," D-11 — do **not** re-derive this ledger there).

---

# 19 — init-and-gradient-flow.html  (D-21: 19)
**Title:** "Setting the Gain on a 100-Stage Amplifier: Initialization, Vanishing/Exploding Gradients, and Clipping" · **Part II** · build: **O**

> **ONE page (`decisions.md` §D-21a).** Init, vanishing/exploding gradients, and clipping are a **single
> geometric-product story** told forward (init) and backward (gradients). The fan-out split this across two pages;
> §D-21a re-merges it. **Trunk owns the WHY** (residual-as-identity-term); Part III owns the WHERE (the block
> diagram). `[THREAD:CHAIN]`

**Learning objectives**
1. Explain why zero/constant initialisation fails: symmetry never breaks.
2. Derive the variance-propagation rule $\mathrm{Var}(w)=1/n_{\text{in}}$ and the Xavier/He corrections; compute
   He's std for $d=4096$ and recognise it as GPT-2's ~0.02.
3. Show that backprop through $\ell$ layers multiplies $\ell$ Jacobian factors — the **same** $g^{\ell}$ geometric
   product as init ($1.1^{100}$ vs $0.9^{100}$), now on the backward pass — and compute the sigmoid-derivative
   product that kills deep sigmoid nets.
4. State the historical fixes as **one lineage** (better activations → careful init → **residuals as an identity
   term** $I+\partial F/\partial x$ → normalization) and give the residual gradient argument.
5. Apply gradient clipping correctly and in the right order; know why fp16 never belongs on the Spark.
6. Recognise the bad-init and vanishing/exploding failure signatures (seeds page 24's tables).

**PREDICT (`.box try`, open the page):** "You stack 100 layers. Each multiplies the signal's scale by a factor
$g$. If $g=1.1$, what happens to the signal by layer 100? If $g=0.9$? Compute both before reading." (Answer:
$1.1^{100}=13{,}781$ — clips; $0.9^{100}=2.7\times10^{-5}$ — vanishes. You need $g\approx1.000$, forward **and**
backward — and this single number governs both init and gradient flow.)

**Section outline — PART A: initialization (the forward pass)**
- *Intuition (`.box key`).* Init sets the per-stage gain before any data. **Worked (`constants.md` §9.3, [DER]):**
  $1.1^{100}=\mathbf{13{,}781}$, $0.9^{100}=\mathbf{2.7\times10^{-5}}$ — two numbers that do more than a page of
  prose, and they set up Part B (vanishing/exploding is this same geometric product on the backward pass).
- *Symmetry breaking (`.box warn`).* All-zeros (or any constant) → every neuron in a layer gets an identical
  gradient and stays identical forever; a 4096-wide layer collapses to one neuron. Loss drops to the prior and
  flatlines. **Biases CAN be zero** (the random $W$ already breaks symmetry) — do not over-generalise "never zero."
- **Variance-propagation derivation (`.box rule`, training §9.2).** For $z_j=\sum_i w_{ji}x_i$ with $w\perp x$,
  zero-mean: $\mathrm{Var}(z)=n_{\text{in}}\mathrm{Var}(w)\mathrm{Var}(x)$. To hold $\mathrm{Var}(z)=\mathrm{Var}(x)$:
  **$\mathrm{Var}(w)=1/n_{\text{in}}$.** Three lines, the whole of Xavier/He. Symbol Ledger required.
  - **Xavier/Glorot** (tanh/sigmoid): $\mathrm{Var}(w)=2/(n_{\text{in}}+n_{\text{out}})$.
  - **He/Kaiming** (ReLU): $\mathrm{Var}(w)=2/n_{\text{in}}$ — the "2" is literally "because ReLU throws away half
    the signal."
- **Worked number (`constants.md` §9.3, [DER]).** He init at $n_{\text{in}}=d=4096$: $\mathrm{Var}(w)=2/4096$, so
  **std $=0.0221$** ≈ GPT-2's 0.02. "The theory predicts the config file." `[THREAD:QWEN]`
- *Depth scaling (deep-dive, summary states it).* GPT-2 scales residual projections by $1/\sqrt{2L}$ so the
  accumulated residual-stream variance doesn't grow with depth — same geometric-product logic on a sum.
- *μP (one paragraph, `[EST]`/med confidence).* Rescales init + per-layer LR so the optimal LR on a small proxy
  transfers to a large model — why labs tune on 40M and apply to 40B, and the theory behind "LR shrinks with
  width." Mention, don't derive.
- **The four bad-init failure modes (`.box rule`, training §9.3).** all-zeros → flatline at prior (symmetry);
  too small (std 1e-3) → signal vanishes with depth, looks like a dead LR; too large (std 1.0) → saturation/NaN —
  **two opposite symptoms (NaN *or* stuck) from one cause**; constant nonzero → same as zeros.

**Section outline — PART B: vanishing/exploding gradients (the backward pass — same geometric product)**
- *The math (`.box rule`, training §10.2).* $\boldsymbol\delta^{(1)}=\big(\prod_{\ell}W^{(\ell+1)\top}\,\mathrm{diag}\,\phi'(\mathbf z^{(\ell)})\big)\boldsymbol\delta^{(L)}$
  — a product of $L$ factors. If the typical factor magnitude is $<1$, it **vanishes**; if $>1$, it **explodes**.
  **Same $g^{L}$ story as Part A's $1.1^{100}/0.9^{100}$, now on the backward pass** — one lesson, two directions.
  `[THREAD:CHAIN]` (this is BP2/the $\delta$ recursion of page 14, iterated).
- **Worked numbers (`constants.md` §9.3, [DER]).** Max sigmoid derivative $\sigma'(0)=\mathbf{0.25}$;
  $0.25^{10}=9.54\times10^{-7}$; $0.25^{32}=\mathbf{5.4\times10^{-20}}$ — still representable in fp32 but
  effectively dead. This is why deep sigmoid stacks could not be trained and why ReLU/tanh + residuals mattered.
- *The fixes, in historical order (`.box rule`) — each a whole architecture idea (D-14: WHY here, WHERE in
  Part III).* (1) Better activations (ReLU: derivative 1 on the active half, no saturation for $z>0$).
  (2) Careful init (Part A). (3) **Residual connections — the identity term:** $\partial y/\partial x=I+\partial F/\partial x$
  — the $+I$ guarantees a gradient path of magnitude ~1 through any depth; **worked: $0.9^{36}=0.0225$ vs ≈1 with
  the residual** (foreshadow the transformer block + residual stream, D-14). (4) Normalization (page 20). Present
  as one lineage solving one problem.

**Section outline — PART C: gradient clipping (the operational patch)**
- **Gradient clipping (`.box rule`, `constants.md` §9.3, [VP]).** Global-norm clip at **1.0** (near-universal):
  if $\|\mathbf g\|>c$, rescale $\mathbf g\leftarrow c\,\mathbf g/\|\mathbf g\|$. **Order matters:** clip **after**
  `backward()` and **before** `step()`; clipping the wrong thing or at the wrong time silently does nothing.
- *fp16 vs bf16 (`.box warn`, `constants.md` §9.3/§9.7).* fp16 max = **65,504**; an activation past it → `inf` →
  NaN. **bf16 has fp32's range (8 exponent bits); on Blackwell there is no reason to use fp16 — the DGX Spark
  should never see it.** (Ties to page 24's NaN table.)

**Misconceptions (`.box warn`)**
- "Init just needs to be random." → It needs a **specific variance** tied to fan-in and activation; wrong-scale
  fails like zeros, less obviously.
- "He is for everything." → He assumes ReLU; He + tanh over-scales by $\sqrt2$/layer → $2^{25}$ over 50 layers.
- "Why all-zeros fails is that gradients are zero." → They are **identical**, not necessarily zero; symmetry never
  breaks. Biases *can* be zero.
- "Vanishing gradients are a solved, historical problem." → The mechanism is the same geometric product; residuals
  + norm + init *manage* it, they don't repeal it; very deep or recurrent stacks still hit it.
- "Clipping fixes a high learning rate." → It caps rare spikes; a persistently-too-high $\eta$ still diverges.
- "Exploding and vanishing are opposite bugs needing opposite fixes." → **Same product**; the fixes (init,
  residual, norm) address both.

**Demo 1 — "Init explorer: watch the signal live" (primitives: `NN.MLP` with `init`/`initScale` + `Plot`).** Build
a deep MLP (e.g. `[64,64,...,64]`, 20 layers) with `NN.MLP(sizes,{act, init, initScale})`; push one random input
through and plot **the std of activations vs layer depth**. Sliders: `initScale` (log, 1e-3–3), activation
(tanh/relu toggle). JS computes real per-layer activation std. *Aha:* wrong scale → the std curve exponentially
grows or decays across depth; the He/Xavier scale (auto-marked) holds it flat. A "zero init" button flatlines
every layer identically (symmetry). Readout: std at the last layer, and the implied $g$ per stage.

**Demo 2 — "Gradient through depth" (primitives: `NN.MLP` + `Plot`).** Build an $L$-layer MLP with a chosen
activation; run one backward pass and plot **the gradient norm reaching each layer** vs depth (log-y). Sliders:
depth $L$ (2–40), activation (sigmoid/tanh/relu), a **residual toggle**, and a **clip toggle** (off / clip@1.0).
JS computes the real per-layer grad norms from the engine. *Aha:* sigmoid → grad norm decays ~$0.25^\ell$ into the
floor; relu + residual toggle → flat (the $+I$ path); clip caps the exploding case. Readout: grad norm at layer 1,
and whether it underflowed. **This is the same geometric product as Demo 1, run backward** — flag it on-page.

**Code artifacts (`.box try`)**
- `code/19_init_variance.py` — measures per-layer activation std for zeros / too-small / too-large / He on a deep
  net; prints the std=0.0221 He value for $d=4096$ and checks it against `torch.nn.init.kaiming_normal_`.
- `code/19_grad_flow.py` — deep MLP, logs per-layer grad norm for sigmoid vs relu vs residual; demonstrates
  `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` and prints the pre/post norm and the $0.25^{32}$ number.

**Quiz (8; ≥1 numeric)**
1. (num) He-init std for $n_{\text{in}}=4096$ → **0.0221** (tol 0.001). Distractor: 0.02 accepted within tol
   (that's GPT-2's, deliberately close).
2. (num) $0.9^{100}$ → **2.7e-5** (tol 5e-6).
3. (num) $0.25^{32}$ → **5.4e-20** (tol 1e-20). Distractor: 0 (it's tiny, not zero — the machine-vs-math theme).
4. (num) Max sigmoid derivative → **0.25** (tol 0.005).
5. (MC) Why all-zeros fails → symmetry never breaks (identical, not necessarily zero gradients).
6. (MC) The "2" in He init → ReLU discards half the signal. Distractor: "two moments."
7. (MC) Residual's gradient identity → $I+\partial F/\partial x$, the $+I$ preserves a magnitude-~1 path.
   Distractor: "multiplies gradients."
8. (MC) Clip order → after backward, before step. Distractor: "before backward." · (MC bonus) Why never fp16 on
   the Spark → max 65,504 → overflow; bf16 has fp32 range.

**Cross-refs:** ← page 13 (the amplifier/landscape framing), page 14 (BP2 / the $\delta$ recursion this iterates).
→ page 20 (normalization keeps the per-stage gain ~1 *during* training, and pre-norm preserves this residual
highway), page 21 (warmup partly compensates early instability), Part III (residual stream + pre-norm block: the
WHERE, D-14 — cite back here, do not re-derive).

---

# 20 — normalization.html  (D-21: 20)
**Title:** "Normalization: The Only Question Is Which Axis" · **Part II** · build: **O**

**Learning objectives**
1. State that BatchNorm, LayerNorm, and RMSNorm differ **only** in which axis they reduce over.
2. Compute a LayerNorm and an RMSNorm on a small vector and see RMSNorm drops mean-centering.
3. Explain why BatchNorm lost to LayerNorm/RMSNorm in transformers.
4. Explain pre-norm vs post-norm and why one bracket decides whether an 80-layer model trains.

**PREDICT (`.box try`):** "LayerNorm and RMSNorm both stabilise activations. RMSNorm does strictly less arithmetic.
What one operation does it drop — and why can it get away with it?" (Answer: it drops **mean subtraction**,
keeping only RMS scaling; empirically the re-centering contributes little, so Llama/Qwen use RMSNorm.)

**Section outline** — this page teaches the **WHY** (D-14); the block diagram and RMSNorm's config line are Part
III's **WHERE**. Say so, to avoid looking like the tracks disagree.
- *Intuition.* Normalization re-centres/re-scales activations so each layer sees a well-conditioned input —
  keeping the per-stage gain (page 19) near 1 *during* training, not just at init.
- **The three, precisely — the axis question front and centre (`.box rule`, training §11.2).**
  Given activations $X$ with shape `(B, ..., d)`: **BatchNorm** normalises over the **batch** axis (per feature);
  **LayerNorm** over the **feature** axis $d$ (per example): $\mathrm{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta_i$,
  $\mu,\sigma^2$ over the $d$ features; **RMSNorm** = LayerNorm **minus mean-centering**:
  $\mathrm{RMS}(x)_i=\gamma_i\frac{x_i}{\sqrt{\frac1d\sum_j x_j^2+\varepsilon}}$. **"The only difference is which
  axis you reduce over."** Shape ribbon + Symbol Ledger mandatory. `rms_norm_eps=1e-6` for Qwen3 (`constants.md`
  §1.1). `[THREAD:QWEN]`
- **Worked number.** Take $x=[2.0,1.0,0.1]$ (the canonical triple). Mean $=1.0333$, so LayerNorm centres then
  scales; RMS $=\sqrt{(4+1+0.01)/3}=\sqrt{1.6700}=1.2923$, so $\mathrm{RMS}(x)=[1.548,0.774,0.077]$ (with
  $\gamma=1$). Compute both live with `NN.layerNorm`/`NN.rmsNorm` on a `Mat`.
- *Why BatchNorm lost in transformers (`.box rule`, training §11.4).* (1) Sequence models have variable length and
  small effective batch per position; (2) train/eval mismatch (running stats); (3) it couples examples in a batch
  (bad for generation); (4) LayerNorm/RMSNorm are per-example and batch-independent. Hence transformers use
  LayerNorm/RMSNorm.
- **Pre-norm vs post-norm — the diagram that matters (`.box key`, training §11.5, D-14 WHY).**
  Post-norm: $x\leftarrow\mathrm{Norm}(x+F(x))$. Pre-norm: $x\leftarrow x+F(\mathrm{Norm}(x))$. **Pre-norm keeps a
  clean residual highway** (the $+x$ is un-normalised, so the identity gradient path of page 19 survives to any
  depth); post-norm puts a norm *on* that path and deep post-norm models are hard to train without careful warmup.
  "One bracket decides whether an 80-layer model trains." Modern LLMs are pre-norm.

**Misconceptions (`.box warn`)**
- "The normalizers are fundamentally different mechanisms." → Same operation, different reduction axis.
- "RMSNorm is an approximation that loses accuracy." → Empirically the mean-centering term contributes little;
  it's a deliberate, near-free simplification.
- "Normalization removes the need for good init/residuals." → It complements them; deep nets still need the
  residual highway (pre-norm) and scaled init (page 19).

**Demo — "Which axis? LayerNorm vs RMSNorm vs BatchNorm" (primitives: `NN.layerNorm`/`NN.rmsNorm` + `Heatmap`).**
A small activation matrix `Mat` shape `(B=4, d=6)` shown as a `Heatmap`. Toggle: none / BatchNorm (reduce columns)
/ LayerNorm (reduce rows) / RMSNorm (rows, no centre). JS recomputes with the real engine functions and re-renders
the heatmap plus per-row/col mean & std readouts. *Aha:* LayerNorm makes every **row** mean 0 / std 1; BatchNorm
makes every **column** so; RMSNorm makes every row's **RMS** 1 but leaves the mean nonzero — the axis is the whole
story, visible in the heatmap. A second panel: a pre-norm vs post-norm deep-net grad-norm-through-depth plot
(reuse page 19's engine) showing pre-norm's flat gradient.

**Code artifact (`.box try`):** `code/20_norm_axes.py` — computes all three on a random tensor with
`torch.nn.functional.layer_norm` / a hand RMSNorm, asserts LayerNorm rows are mean-0/var-1 and RMSNorm rows are
RMS-1; opens Qwen3's `config.json` line `rms_norm_eps: 1e-6`.

**Quiz (6; ≥1 numeric)**
1. (MC) BatchNorm vs LayerNorm differ in → the reduction axis (batch vs feature).
2. (num) RMS of $[2.0,1.0,0.1]$ → **1.292** (tol 0.01).
3. (MC) RMSNorm drops → mean-centering. Distractor: "the scale $\gamma$."
4. (MC) Why transformers dropped BatchNorm → per-example, variable length, no train/eval stat mismatch.
5. (MC) Pre-norm keeps → a clean un-normalised residual path. Distractor: "a smaller model."
6. (MC) Qwen3's `rms_norm_eps` → 1e-6.

**Cross-refs:** ← page 19 (init sets the per-stage gain; the residual highway the $+x$ preserves). → page 21
(pre-norm reduces — but does not remove — warmup's necessity), Part III (RMSNorm in the transformer block — WHERE,
D-14; do not re-derive the axis question there).

---

# 21 — lr-schedules-warmup.html  (D-21: 21)
**Title:** "Learning-Rate Schedules and Why Warmup Exists" · **Part II** · build: **S**

**Learning objectives**
1. Write the linear-warmup → cosine-decay schedule and index it by **step $k$**, not epoch.
2. Give the three mechanistic reasons warmup exists, led by Adam's early-variance problem.
3. Explain why we decay to $\eta_{\max}/10$, not to zero.
4. State honestly that warmup is a known patch the field keeps because it is cheap and works.

**PREDICT (`.box try`):** "You computed on page 17 that Adam's variance estimate $\hat v$ is garbage for ~1000
steps. Given that, what should the learning rate do for the first ~1000 steps of training?" (Answer: **ramp up
from ~0** — warmup — so no large step corrupts the running statistics.)

**Section outline**
- *Intuition.* Big steps to reach the right neighbourhood; small steps to find the right house. Annealing, like
  metallurgy.
- **The standard 2026 schedule (`.box rule`, training §8.2, reuse this exact form):**
  $\eta_k=\eta_{\max}\cdot k/T_{\text{warm}}$ for $k\le T_{\text{warm}}$; then
  $\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\big(1+\cos(\pi\frac{k-T_{\text{warm}}}{T_{\text{total}}-T_{\text{warm}}})\big)$.
  Symbol Ledger required. Typical: $T_{\text{warm}}$ = 1–10% of steps (or a flat 2000), $\eta_{\min}=\eta_{\max}/10$.
- **Why warmup — three reasons (`constants.md` §9.3 / training §8.3), lead with #1:**
  (1) **Adam's second moment is unreliable at $k\lesssim1000$** — bias correction fixes the mean of $\hat v$ but
  not its variance; $1/\sqrt{\hat v}$ has huge variance early (RAdam's thesis: warmup is variance reduction).
  (2) **A bad early step poisons Adam for ~1000 steps** — one spike enters $v$ and, at $\beta_2=0.999$
  ($1/(1-0.999)=1000$, from page 17), takes ~1000 steps to decay out. **Most memorable framing; he computed the
  1000 himself.** (3) **LayerNorm + large batch** produce early instability that warmup resolves (and why pre-norm,
  page 20, reduces but doesn't remove the need).
- *Decay to $\eta_{\max}/10$, not 0.* Zero LR = zero learning for the tail of the run.
- *The honest state of 2026 (`.box key`).* Warmup is a **patch** — RAdam, T-Fixup, GradInit reduce or remove it —
  "but everyone still uses it because it costs nothing and works." A valuable, honest thing to hear about how the
  field operates.

**Misconceptions (`.box warn`)**
- "Warmup is because the model is 'fragile' at start." → Specifically because **Adam's variance estimate** is
  unreliable; pure SGD needs warmup far less. Tie it to the optimizer, not mysticism.
- "Warmup and warm restarts are the same." → Unrelated (SGDR re-raises LR periodically).
- "Schedule by epoch." → By **step $k$** — LLM runs are often <1 epoch (page 16).
- "Decay to zero." → To $\eta_{\max}/10$–$/20$.
- "Adam is adaptive, so no schedule needed." → Adam scales direction, not $\eta$ (page 17).

**Demo — "Schedule shaper" (primitives: `makeCtrl` ×4 + `Plot`).** Plot $\eta_k$ vs $k$ over $T_{\text{total}}$.
Sliders: $\eta_{\max}$ (log, 1e-5–1e-2), $T_{\text{warm}}$ (0–3000), $T_{\text{total}}$ (5k–100k),
$\eta_{\min}/\eta_{\max}$ ratio (0–0.5). JS computes the piecewise curve live. A second toggle overlays a real
short training run's loss where the LR follows the shaped schedule (small MLP via `NN.Trainer`, LR set each step
from the schedule) so warmup's effect on early loss is visible, not asserted. *Aha:* set $T_{\text{warm}}=0$ and a
high $\eta_{\max}$ → the early loss spikes; add warmup → it doesn't. Equation printed with live values.

**Code artifact (`.box try`):** `code/21_schedule.py` — implements the warmup-cosine schedule as a
`torch.optim.lr_scheduler.LambdaLR`, plots $\eta_k$, and **plots it against the step index to expose the classic
"scheduled by epoch not step" off-by-one** (page 24's instrumentation panel item 3).

**Quiz (6; ≥1 numeric)**
1. (num) At $\beta_2=0.999$, warmup should cover ~ how many steps → **~1000** (tol 200).
2. (MC) The strongest reason for warmup → Adam's early $\hat v$ variance (RAdam). Distractor: "the model is
   fragile."
3. (MC) Schedule index → step $k$. Distractor: epoch.
4. (MC) Decay target → $\eta_{\max}/10$. Distractor: 0.
5. (MC) Warmup ≠ warm restarts → true; different things.
6. (MC) Honest 2026 status of warmup → a cheap patch everyone keeps. Distractor: "a solved, first-principles
   requirement."

**Cross-refs:** ← page 17 (the 1000-step $\beta_2$ memory), page 16 (step vs epoch). → page 20 (pre-norm reduces
warmup's necessity), page 24 (plot the actual LR — the silent-scheduler bug).

---

# 22 — regularization.html  (D-21: 22)
**Title:** "Generalization I: Regularization and Reading the Two Curves" · **Part II** · build: **S**

> **Un-merged from double descent (`decisions.md` §D-21a).** The fan-out folded regularization and double descent
> onto one page — a **lead-REJECTED cut**. §D-21a restores two pages: **this one owns the practitioner's toolkit +
> the two-curve diagnostic**; page 23 owns bias–variance and double descent. **Written for his regime: ~500
> examples, a few hundred at fine-tune time.** `[THREAD:MEM]` (trainable-count discipline recurs here.)

**Learning objectives**
1. Read the train/val curve as the primary generalisation diagnostic and split data by the right unit.
2. Name **six** two-curve shapes and the **six** diagnoses they map to.
3. Apply weight decay, dropout, and early stopping, and know which subset this learner actually needs.
4. Name catastrophic forgetting as a failure the two-curve plot alone will **not** reveal.

**PREDICT (`.box try`, open the page):** "You fine-tune on 500 examples. After 3 epochs train loss is near 0 and
val loss has started rising. You have four knobs — more epochs, weight decay, dropout, early stopping. Rank them
for *this* situation before reading." (Answer on the page: early stopping + fewer epochs + low LR do the work;
dropout is largely irrelevant in 2026 LLM fine-tuning; more epochs is the wrong direction.)

**Section outline**
- *The two curves (`.box rule`, training §12.2).* Plot train and val loss on the same axes with the baseline
  ($\ln V=11.93$, page 12) drawn. **Split by the right unit** (dedupe; no leakage across the split; for
  instruction data, split by *source document*, not by row).
- **Six shapes, six diagnoses (`.box rule` — the diagnostic table).**
  (1) **both high, flat** → underfitting / LR too low / model too small (page 13, page 19-init) → raise capacity or
  LR. (2) **both low, small gap** → healthy; stop when val flattens. (3) **train→0, val rises** → overfitting →
  regularize / stop earlier / fewer epochs. (4) **train ≫ val (val lower than train)** → leakage or a val set that
  is too small / too easy → fix the split. (5) **both plateau high after a few steps** → a wiring bug or dead LR,
  not generalisation → go to page 24's overfit-one-batch test. (6) **val jagged, no trend** → val set too small to
  estimate → enlarge it or smooth over a window (page 16's "judge the trend"). **Each shape names one fix.**
- *The regularizers (`.box rule`).* **Weight decay** $\lambda$ (0.1 for LLMs, 0.01 elsewhere; excluded from
  biases/norm gains/embeddings — page 17's no-decay list). **Dropout** (rare in 2026 LLM fine-tuning; still used in
  some vision/LoRA-dropout settings). **Early stopping** — evaluate every ~50 steps, keep the best checkpoint.
  **For this learner fine-tuning a few hundred examples, early stopping + few epochs + low rank/LR is the whole
  toolkit** (page 23 explains *why* via the regime).
- *Catastrophic forgetting (`.box warn`, LLM-specific — flagged here, developed on page 24).* Task train/val can
  look perfect while the model gets **worse at everything else**. It does **not** appear on the task's two-curve
  plot. Mitigations: mix in general data, lower rank/LR, fewer epochs.

**Misconceptions (`.box warn`)**
- "A rising val curve always means overfitting." → Shape (4) (val below train) is leakage; shape (5) is a bug.
  Read the *shape*, not just the direction.
- "Dropout is essential regularization." → In 2026 LLM fine-tuning it is usually off; early stopping + low rank do
  more.
- "Val loss looks fine, so fine-tuning is safe." → Catastrophic forgetting hides behind a healthy task curve.
- "More epochs will fix a rising val curve." → It makes overfitting worse; stop earlier.

**Demo — "Read the two curves" (primitives: `NN.MLP`+`NN.Trainer`+`NN.trainTestSplit`+`viz.Timeline`).** Train a
small MLP on `NN.makeDataset('moons', 200, {noise, labelNoise})` with a train/val split; draw train + val loss on
one `Timeline` (log-y) with the $\ln V$ baseline line. Sliders: dataset size $N$ (40–500), weight decay $\lambda$
(0–0.1), epochs cap, val-split fraction. Buttons that *force* each of the six shapes (e.g. **tiny val set** →
shape 6; **shuffle val labels into train** → shape 4). *Aha:* the learner produces each diagnostic shape on demand
and reads the fix off the table. Readout: current train/val loss, the gap, and the best-checkpoint step (early
stopping marker).

**Code artifact (`.box try`):** `code/22_regularization.py` — trains a small model with/without weight decay and
early stopping on a noisy set; logs the two curves and the best-checkpoint step; prints the leakage check (val/
train overlap count) so the split discipline is concrete.

**Quiz (6; ≥1 numeric)**
1. (MC) Overfitting signature on the two-curve plot → train→0, val rises (shape 3).
2. (MC) Train loss below val by a wide margin, val lower than train → leakage / bad split (shape 4). Distractor:
   "great generalisation."
3. (num) The random-init baseline line to draw on every loss plot, for Qwen3 → **11.93** nats (tol 0.05).
4. (MC) His actual regularization toolkit (few-hundred examples) → early stopping + few epochs + low rank/LR.
   Distractor: "heavy dropout."
5. (MC) Catastrophic forgetting shows up on the task's two-curve plot → **no** — it hides; check held-out general
   data. Distractor: "yes, val loss rises."
6. (MC) Split instruction data by → source document (avoid leakage). Distractor: "randomly by row."

**Cross-refs:** ← page 12 ($\ln V$ baseline), page 17 (the no-decay list), page 16 (judging a noisy curve's
trend). → page 23 (WHY the toolkit is what it is — bias–variance + the regime), page 24 ("overfits instantly" is
his failure mode; the instrumentation panel logs these curves), the trunk LoRA/SVD pages (rank as the capacity
knob, D-11).

---

# 23 — bias-variance-double-descent.html  (D-21: 23)
**Title:** "Generalization II: Bias–Variance and the Double-Descent Surprise, Honestly" · **Part II** · build: **S**

> **Its OWN page (`decisions.md` §D-21a).** The fan-out merged this into regularization — a **lead-REJECTED cut**.
> §D-21a gives double descent the full page the arbiter designed, **built around the λ slider that dissolves the
> phenomenon** — the demo that shows the mid-model peak is largely an under-regularization artifact. Page 22 gave
> the practitioner's toolkit; this page gives the *theory that explains which knob to reach for and why.*

**Learning objectives**
1. State the bias–variance decomposition as an algebraic identity and read the classical U-curve.
2. Explain double descent across all three axes and the "choice, not complexity" intuition.
3. **Watch the double-descent peak dissolve under regularization** and state honestly what that means.
4. Locate his own fine-tune in the $P$-vs-$N$ regime and connect the whole picture to the LoRA rank knob.

**PREDICT (`.box try`, open the page):** "Classical theory says test error is U-shaped in model size: too small
underfits, too big overfits. GPT-scale models have *far* more parameters than training examples per task. By
classical logic they should overfit catastrophically. Do they? What's going on?"

**Section outline**
- **Bias–variance — an identity, not a law (`.box rule`, training §13.1).**
  $\mathbb E[(y-\hat f)^2]=\underbrace{(\mathbb E[\hat f]-f)^2}_{\text{Bias}^2}+\underbrace{\mathrm{Var}[\hat f]}_{\text{Variance}}+\underbrace{\sigma^2}_{\text{irreducible}}$.
  Bias = consistently wrong the same way (a line through a curve); variance = wrong differently each dataset (a
  wiggly curve). It is **algebraically always true** for squared loss. Classical prediction: **U-shaped** test
  error in complexity. Correct and useful for 30 years — for $P\ll N$.
- **Double descent (`.box key`, training §13.2).** Past the interpolation threshold ($P\approx N$), test error
  goes up (the classical U), **peaks at the threshold, then descends again** — often below the first minimum.
  Three axes (Nakkiran 2019): **model-wise**, **sample-wise (more data can hurt near the threshold — call it
  out)**, **epoch-wise**. Intuition: at $P\approx N$ there is ~one (brittle) interpolant; past it there are
  infinitely many and SGD's implicit bias (page 16) picks the smoothest. **Overparameterization buys *choice*, not
  complexity.**
- **★ THE HONEST PART — the λ slider dissolves the peak (`.box warn`, training §13.3; do not present double descent
  as settled).** (1) The peak largely **disappears with proper regularization** (optimal early stopping / weight
  decay $\lambda$) — it may be substantially an under-regularization artifact, and the demo below *shows* this by
  turning up $\lambda$ until the peak flattens. (2) The x-axis (parameter count) is a **bad** complexity measure;
  plot against a norm-based one and the second descent can vanish. (3) LLM/ViT scaling curves are **monotone power
  laws — no double descent where it "should" appear.** (4) Only the claim that variance must rise *monotonically*
  with $P$ is dead; the identity survives. **Ship the framing:** "The classical U is real (small model, small data
  — you'll be there). Double descent is real (why big models exist). Both are true in their regimes; the field has
  no single clean theory. Hold both, and know which regime you're in."
- *The "which regime" heuristic (`.box rule`).* $P\ll N$ → classical, U-curve, regularize modestly. $P\gg N$ (a
  7B on 500 examples — **this is him**) → overparameterized with the interpolation threshold *right there*; you
  hit train loss 0 in minutes; **regularization (LoRA's low rank, early stopping, small LR, few epochs) does the
  heavy lifting; double descent will not save you because you can't grow the dataset.**
- **★ The LoRA connection (`.box key`, D-11 foreshadow).** **LoRA's rank $r$ IS the capacity knob — bias–variance
  in a 2026 costume.** $r=4$ → high bias, learns little, won't overfit; $r=256$ on 20 images → low bias, high
  variance, **memorizes and forgets**. The classical U-curve lives in his LoRA config, and its location is set by
  the dataset's information content (the trunk LoRA page, D-11, makes this the rule).

**Misconceptions (`.box warn`)**
- "Bigger models must overfit." → Not in the overparameterized regime; that's the second descent.
- "The bias–variance tradeoff is obsolete." → It's an identity; only the monotonic-variance claim is dead. Honor
  the question mark.
- "More data always helps." → Sample-wise double descent: near the threshold, more data can hurt.
- "Double descent is a settled law you can rely on." → The peak is largely a regularization artifact (see the
  slider); at LLM scale the curves are monotone. Do not bank on it.

**Demo ⭐ — "The λ slider that dissolves double descent" (flagship; primitives: `NN.MLP` sweep +
`NN.trainTestSplit` + `Plot`).** Fit MLPs of increasing hidden width to a small noisy dataset
(`NN.makeDataset('moons',40,{labelNoise:0.15})`, train/test split). For each width, train to interpolation and
record train & test loss; plot **test loss vs model size** with the interpolation threshold ($P\approx N$) marked.
Sliders: dataset size $N$ (20–200), label noise, and **weight decay $\lambda$ (0–0.1) — the star control**.
*Aha:* $\lambda=0$ → the classic U **then a second descent** past $P\approx N$, with a sharp peak at the
threshold; **drag $\lambda$ up and the peak flattens away** — the phenomenon dissolves before his eyes, making the
honest-part claim (1) something he *saw*, not something he was told. Readout: threshold location, the two minima,
and the peak height as a function of $\lambda$.

**Code artifact (`.box try`):** `code/23_double_descent.py` — reproduces model-wise double descent on a small
synthetic set; sweeps weight decay to show the peak flattening (the slider, in code); prints the regime test
($P$ vs $N$) for his actual fine-tune numbers.

**Quiz (7; ≥1 numeric)**
1. (num) The number of terms in the bias–variance decomposition → **3** (bias², variance, irreducible) (tol 0).
   Distractor: 2.
2. (MC) Bias–variance is → an algebraic identity (always true for squared loss). Distractor: "an empirical law
   that fails for big models."
3. (MC) Interpolation threshold at → $P\approx N$.
4. (MC) Double descent's second-descent intuition → many interpolants, SGD picks the smoothest (choice, not
   complexity).
5. (MC) What the λ slider shows → the mid-model peak largely dissolves under regularization (an artifact).
   Distractor: "λ has no effect on the peak."
6. (MC) His regime ($P\gg N$) prescription → LoRA rank + early stopping + few epochs + small LR. Distractor:
   "grow the dataset to ride double descent."
7. (MC) LoRA rank $r$ is → the bias–variance capacity knob.

**Cross-refs:** ← page 22 (the two-curve toolkit this explains), page 12 ($\ln V$ baseline), page 16 (SGD's
implicit bias picks the smooth interpolant). → page 24 ("overfits instantly" is his failure mode), the trunk
LoRA/SVD pages (rank follows dataset information content, D-11 — the rule this page foreshadows).

---

# 24 — first-real-finetune.html  (D-21: 24)  ⭐ MILESTONE — before the fork
**Title:** "Your First Real Fine-Tune — and the Twelve Ways Training Fails" · **Part II** · build: **S**
(scaffold: the browser demo trains live; the `.box try` script runs a real small fine-tune on his box.)

> **The halfway milestone (brief-pedagogy §motivation): he should be able to say "I fine-tuned a model" at the
> midpoint, before the LLM/diffusion split.** This page assembles pages 12–23 into one working loop and hands him
> the diagnostic checklist that turns training from magic into engineering. `[THREAD:TN1]` `[THREAD:MEM]`
> `[THREAD:CHAIN]` `[THREAD:QWEN]` — all four threads converge here.

**Learning objectives**
1. Assemble loss + backprop + optimizer + schedule + regularization into a correct training loop and read every
   line.
2. Run the universal first move (overfit one batch) to bisect the entire space of training failures.
3. Diagnose the three canonical failures (NaN, won't-go-down, overfits-instantly) from their signatures.
4. Instrument a run and **predict-then-measure** its peak memory against page 18's ledger.

**PREDICT (`.box try`):** "You will start a real fine-tune in a moment. Before you press go: what is the single
60-second test that tells you whether your forward/loss/backward/optimizer are correctly wired — *before* you
touch a single hyperparameter?" (Answer: **overfit one batch of 8** to ~0 loss. If it won't, you have a **bug**,
not a hyperparameter problem.)

**Section outline** — the training loop grown by accretion (brief-pedagogy §code-by-accretion): show the ~15-line
loop with each line annotated by **why it exists**, dimming lines from earlier pages and highlighting the new ones
(`class="line hl"`). Every line traces to a Part II page: `zero_grad()` → page 15; `loss = ...` → page 12;
`loss.backward()` → pages 14–15; `clip_grad_norm_(., 1.0)` → page 19; `opt.step()` → page 17; `scheduler.step()`
→ page 21; the eval-every-N + keep-best → page 22.

- **14.0 The universal first move (`.box key`).** Overfit 8 examples, dropout/aug/WD off, 200 steps. Loss→0 ⇒
  wiring correct, problem is data/generalisation. Loss↛0 ⇒ a bug; no LR tuning fixes it. **Bisects the whole
  failure space in 60 s.** Lead with it.
- **"Loss is NaN" (`.box warn` table, training §14.1).** LR too high (#1 cause: loss rose then NaN, grad norm
  exploded — ÷10); `log(0)` in hand-rolled CE (use fused, page 12); ÷0 (add $\varepsilon$); **fp16 overflow**
  (max 65,504 — use bf16, page 19; the Spark should never see fp16); bad data (NaN at the **same step every run**
  — determinism is the tell, page 24.5); missing `zero_grad()` (page 15); mis-ordered clipping (page 19). Tool:
  `torch.autograd.set_detect_anomaly(True)`.
- **"Loss won't go down" (`.box warn` table, training §14.2).** LR too low (smooth, flat — ×10, run an LR range
  test); forgot `step()` (loss **exactly** constant); wrong param group (print trainable count — **for LoRA it
  should be ~43.6M, not 8.19B and not 0**, `[THREAD:MEM]`); **labels shuffled → loss sits exactly at $\ln V$**
  (the clean diagnostic; for Qwen3 that's **11.93**, page 12; put the $\ln V$ line on every plot); data not
  normalised.
- **"Overfits instantly" (`.box warn` table, training §14.3) — HIS failure mode.** $P\ggg N$ → lower LoRA rank
  (page 23); **1–3 epochs, not 50**; LR too high for fine-tuning; add early stopping; check the split;
  **catastrophic forgetting** (LLM-specific — train/val on the task look fine while the model gets worse at
  everything else; does not show on the two-curve plot). 
- **The instrumentation panel (`.box rule`, training §14.4) — log every run, always.** (1) train + val loss with
  the $\ln V$ baseline; (2) **gradient global norm — grad norm leads loss**; (3) the actual LR from the scheduler
  (plot it — catches the scheduled-by-epoch bug, page 21); (4) **update ratio $\|\Delta\theta\|/\|\theta\|$ ≈
  $10^{-3}$** (`[EST]`, a rule of thumb — Karpathy CS231n; $10^{-1}$ ⇒ LR 100× too high, $10^{-6}$ ⇒ 1000× too
  low); (5) dead-unit fraction; (6) **VRAM high-water `torch.cuda.max_memory_allocated()` — predict from page 18's
  ledger, then measure** `[THREAD:MEM]`; (7) throughput; (8) wall-clock/step.
- **The determinism trick (`.box key`, training §14.5).** Same-step failure with a fixed seed ⇒ **data**; random-
  step ⇒ **numerics/race**. One bisection, saves hours.
- *Milestone marker (`<div class="milestone" data-progress="37">`).* "You can now read a training loop and know
  what every line does — and you just fine-tuned a model." (~37% of 65 pages.)
- *The fork ahead (`.box key`, D-17 foreshadow — not the reveal, which is Part III's fork page).* "Next: the same
  machine, pointed at two targets. Predict the next token → an LLM. Predict the noise that was added → a diffusion
  model. **The machine won't change. The target will.**"

**Misconceptions (`.box warn`)**
- "Training failures are mysterious." → There are ~twelve, with distinct signatures; learn all twelve.
- "If the loss is going down, it's working." → Catastrophic forgetting and data leakage both hide behind a healthy
  task curve.
- "Tune the LR first." → **Overfit one batch first**; LR tuning can't fix a wiring bug.

**Demo — "Live fine-tune with the instrumentation panel" (primitives: `NN.MLP`+`NN.Trainer`+`viz.Timeline`+
`viz.NetGraph`).** Train a small MLP on a real 2-D dataset (`NN.makeDataset('spiral'|'moons', 300)`, train/test
split) live from rAF. Panel of live readouts: train loss, val loss (with the baseline line), **grad global norm**,
current LR (a shaped schedule from page 21), and the **update ratio $\|\Delta\theta\|/\|\theta\|$**. Buttons:
**Overfit-one-batch** (train on 8 points → watch it hit ~0, proving the wiring), **Train** (full run), **Break it**
(inject LR×50 → watch grad norm spike and loss NaN; or shuffle labels → watch loss pin to $\ln K$). *Aha:* each
"break" reproduces exactly one row of the diagnostic tables, on a curve he is watching. This is the run→derive→
**break** rhythm the pedagogy contract demands.

**Code artifact (`.box try`) — the milestone deliverable.** `code/24_first_finetune.py` — a **real** LoRA
fine-tune the learner runs on his box: `Qwen3-0.6B` (the smoke-test sibling, D-01/D-21 — finishes in ~1 min so the
milestone is reachable before the split; the arithmetic transfers to the 8B anchor by swapping four config
numbers) on a tiny instruction set, via `peft` `LoraConfig(target_modules="all-linear", r=16)` (D-06) with
`SFTTrainer` and `model_init_kwargs={"dtype": torch.bfloat16}` (the fp32-default trap, `constants.md` §7.3).
It (a) runs overfit-one-batch first, (b) prints the trainable-param count (~its ~4–5M for 0.6B; the 8B figure is
43.6M), (c) logs the full instrumentation panel, (d) **predicts peak VRAM from the ledger then measures it** with
`max_memory_allocated()`. **Installs into a fresh venv, never ComfyUI's** (`hardware-ground-truth.md` §3);
`peft`/`trl` are not on his box, so the setup page (Part III) installs them — reference it. Pin
`trl>=1.8,<2`, `peft==0.19.1`.

**Quiz (7; ≥1 numeric)**
1. (MC) The universal first move → overfit one batch to ~0. Distractor: "sweep the LR."
2. (num) Trainable params for Qwen3-8B LoRA r=16 all-linear (the count to expect in the panel) → **43,646,976**
   (tol 100000). Distractor: 8,190,735,360 (froze the wrong thing).
3. (MC) Loss pinned exactly at $\ln V$ means → labels shuffled / no signal (learned the prior). For Qwen3,
   $\ln V=11.93$.
4. (MC) NaN, loss rose first → LR too high (÷10). Distractor: "bad data" (that's same-step-every-run).
5. (MC) Same-step failure with fixed seed → data, not numerics.
6. (num) Healthy update ratio $\|\Delta\theta\|/\|\theta\|$ → **1e-3** (tol 5e-4; label it a rule of thumb).
7. (MC) Overfits-instantly fix for $P\gg N$ → lower rank + fewer epochs + early stopping. Distractor: "more
   epochs."

**Cross-refs:** ← all of pages 12–23 (each loop line traces to one). → page 18 (predict-then-measure VRAM), the
Part III fork page (the D-17 reveal, the setup page that installs peft/trl), the LLM SFT/LoRA track pages (this
loop, scaled to the 8B anchor).

---

## APPENDIX — CROSS-PART SEAMS (for the other spec writers)

**Into Part II — Part I page numbers per `decisions.md` §D-21a (the single truth; do NOT use the old ~page-10
pointer):** chain rule = **p.5**; TN-1 architecture + **forward pass to 0.3727** (`constants.md` §8.1) = **p.6**;
probability / the Gaussian / μ+σε reparameterization = **p.8** (Part II page 12 cashes this seed for MSE);
logs/logsumexp/softmax/CE=NLL = **p.9**; XOR / linear collapse = **p.10**; activations + MLP limits = **p.11**.
Part I must ship TN-1's forward pass (p.6) and the softmax/CE primitives (p.9) or pages 12/14 have no on-ramp.
- **★ FLOAT BEAT HOME REVERTED TO PAGE 15 (2026-07-16, `decisions.md` §D-21a).** A previous integrator moved the
  "exactly zero is false" gradient beat's home to Part I **page 10** on the strength of `constants.md` §8.4's
  "page ~10, the Early Real Thing" pointer. **That pointer is stale.** Under §D-21a, **Part I p.10 is XOR /
  linear-collapse and carries NO autograd** — there is nowhere on p.10 for the beat to live. `decisions.md` §D-21a
  outranks `constants.md`'s prose pointer (authority chain), and the ratified table assigns the autograd + float
  beat to **page 15**. **Resolution:** **Page 15 OWNS the beat in full** — the mandated `.box key` framing
  (verbatim §8.4), the gradient value `1.4020191230201817e-08`, the float64/float32 depth, the input-1 `atol=1e-4`
  and input-2 `atol=1e-5` assertions, and the autograd-graph mechanics (tape, reverse-vs-forward mode,
  `zero_grad()` accumulation with mini-batches). It is presented as a **discovery**, not a callback to p.10. The
  "callback to page 10" framing is removed everywhere in this file. (§8.4's own "page ~10 / three pages early"
  wording predates §D-21a; the beat now lands six pages *after* the logsumexp page, not before it — the temporal
  phrasing is adjusted, the mandated *content* is shipped intact. Part I's author: p.10 must NOT reprint the
  float-beat `.box key`; it is page 15's.)

**Out of Part II (Part III+ authors, page 25 onward):**
- **worked221 trap:** the shared `NN.worked221()` default reproduces the RETIRED §5.4 network and `sgdStep`
  defaults η=0.5. Any page reusing it (and the TEMPLATE's Demo C, which currently uses defaults) MUST pass
  explicit TN-1 config. Flagged for whoever owns the template/reference page.
- **Ownership handoffs (D-14):** **page 19** teaches the residual-connection WHY (gradient flow, $I+\partial F/\partial x$,
  $0.9^{36}$) and **page 20** teaches the normalization WHY (axis question, pre/post-norm); the **WHERE** (block
  diagram, RMSNorm config line) is Part III's architecture pages — do not re-derive, and cite back to 19/20.
- **The memory ledger (page 18)** owns 16 B/param, 122.05 GiB, the escape ladder, the 187× identity, and the
  LR-vs-trainable-params principle (D-09). The trunk **LoRA/SVD** pages (D-12) must pick up 187× and "rank follows
  the dataset" (D-11) **without re-deriving the ledger**; page 23 already foreshadows rank as the bias–variance knob.
- **The roofline is page 43, not page 18.** Page 18 measures *memory* and shares `code/measure_your_box.py` with
  page 43; page 18 must NOT assert the ~62 TF ceiling or the ridge (227/458) — those are [INF] and page 43's
  predict-then-measure lab (`constants.md` §6.3/§6.4/§10).
- **The D-17 fork reveal** ("the machine didn't change, the target did") is the Part III **fork page**, not page
  24; page 24 only foreshadows it. Track order is **LLM first** (D-19). The setup page (Part III) installs
  peft/trl/accelerate/bitsandbytes into a **fresh venv**; page 24's `.box try` depends on it.
