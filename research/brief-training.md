# RESEARCH BRIEF — HOW NETWORKS LEARN: LOSS, BACKPROP, OPTIMIZATION

> **⚠️ THIS BRIEF PREDATES TWO VERIFICATION PASSES. `constants.md`, `decisions.md`, `notation.md` OVERRIDE IT.**
> Load-bearing corrections a builder MUST apply: **(1)** the anchor is **Qwen3-8B**, not generic-7B/Llama-2-7B —
> every "7B", 112 GB, 130 GB, 56 GB-Adam, 224× figure is retired and re-derived in `constants.md` (D-01, D-03, D-05).
> **(2)** §5.4's rival 2-2-1 network is **RETIRED**; TN-1 is the sole canonical network — but §5.4's *input*
> `[0.60,−0.20]` survives as **TN-1's frozen second input** (`constants.md` §8.7), which carries the "12.7× → Adam"
> beat now recomputed as **10.22×** (D-02). **(3)** memory table: full FT state = **131.05 GB / 122.05 GiB**;
> optimizer-shrink is **187×** (= the param ratio), not 224×. **(4)** the "130 GB → 14.3 GB" callback is retired →
> **131.05 GB → 17.08 GB**, state-to-state (D-03). See `decisions.md` §D-20.

**Audience for this document:** the curriculum architect writing the build spec. Not the learner.
**Scope:** the shared trunk's core. Everything downstream (LLM track, diffusion track, fine-tuning) reuses this vocabulary, these symbols, and these numbers.
**Date grounded:** 2026-07-16. PyTorch 2.10 is the current stable line. Confidence notes are marked `[CONF: high/med/low]` where I could not verify.

---

## 0. THE ONE ORGANIZING IDEA

Every section below is a variation on a single sentence, and the course should say it out loud early and then keep pointing back at it:

> **Training is: guess, measure how wrong, ask each knob "which way should you move to make that wrongness smaller," move every knob a little bit that way, repeat.**

Loss = "measure how wrong." Backprop = "ask each knob." Optimizer = "move a little bit that way." Learning rate = "how little." Everything else in this brief — normalization, init, clipping, schedules, regularization — is damage control for the fact that this loop is run on a chaotic, high-dimensional, badly-conditioned function.

The learner already has hands-on intuition from ComfyUI: he has watched a LoRA train, seen a loss curve, and probably seen one diverge. **Anchor on that.** The course's job is not to tell him gradient descent exists — it's to make him able to look at a loss curve and *diagnose* it.

---

## 1. DEPENDENCY SPINE (recommended order, with the "why")

This is a strict order. Each item is load-bearing for the next.

1. **Derivative as sensitivity** — "if I nudge this by ε, how much does that move?" Not "slope of tangent line." The sensitivity framing is what makes backprop obvious later.
2. **Partial derivative & gradient as a vector of sensitivities** — the gradient is a *list of answers to independent nudge questions*, one per parameter. It is NOT a direction the loss "wants to go"; it is the direction of steepest *increase*, which is why there's a minus sign.
3. **Chain rule as "sensitivity multiplies along a path"** — must be rock solid. This is the single highest-leverage high-school-refresh in the whole course.
4. **Loss as a scalar function of parameters** — the crucial reframe: data is fixed, *weights are the variables*. Almost every learner initially thinks of $f(x)$; you need them thinking $L(\theta)$.
5. **Gradient descent + learning rate** (§3)
6. **Loss functions from maximum likelihood** (§2) — can come before or after 5; I recommend MSE-first-then-MLE (see §2.0).
7. **Backprop: the 2-2-1 hand-worked example** (§5) — THE SPINE. Nothing after this is comprehensible without it.
8. **Autodiff / PyTorch mechanics** (§6) — only after 7, so `.backward()` is demystification, not magic.
9. **Stochasticity: batch → minibatch → SGD noise as feature** (§4)
10. **Optimizers** (§7) — requires 9 (momentum only makes sense once gradients are noisy).
11. **Initialization + vanishing/exploding + clipping** (§9, §10) — requires 7 (you need the product-of-Jacobians picture).
12. **Normalization** (§11) — requires 11 (norm is a fix for what 11 breaks).
13. **LR schedules incl. warmup** (§8) — requires 10 (warmup is an *Adam* pathology) and 12 (pre-norm interacts).
14. **Regularization + bias-variance + double descent** (§12, §13) — can be taught last; it's the "does it generalize" layer, orthogonal to "does it train."
15. **Debugging checklist** (§14) — synthesis. This is where the learner feels competent.

**Ordering disagreement to resolve with the architect:** conventional courses teach regularization before optimizers. I recommend *after*. Reason: this learner's felt pain is "my LoRA diverged / my loss is NaN," not "my model overfits MNIST." Front-load the mechanics of training actually working, then handle generalization.

---

## 2. LOSS FUNCTIONS

### 2.0 Pedagogical sequencing note
Do **not** open with "the loss for classification is cross-entropy." That's an assertion and it's the #1 place learners' understanding goes hollow. Open with MSE (obvious, physical), then ask *"where did that even come from?"*, then derive **both** MSE and cross-entropy from the *same* maximum-likelihood principle. The payoff moment is: **MSE and cross-entropy are the same idea under two different noise assumptions.** That single realization is worth a page.

### 2.1 Intuition first
- **MSE:** "Pretend your model's prediction is the center of a dartboard and reality scattered the dart with Gaussian noise. Loss = squared miss distance."
- **Cross-entropy:** "Loss = how *surprised* you are by the correct answer. If you said 90% and it happened, you're barely surprised. If you said 0.1% and it happened, you're stunned — and stunned costs a lot."
- **NLL:** "Cross-entropy and NLL are the same number wearing different hats. NLL is the model-fitting name; cross-entropy is the information-theory name."

### 2.2 Maximum likelihood — the derivation to actually run

Setup. Dataset $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}$. Model $p_\theta(y \mid x)$ with parameters $\theta \in \mathbb{R}^{P}$.

Assume the $N$ samples are i.i.d. Then

$$
p_\theta(\mathcal{D}) \;=\; \prod_{i=1}^{N} p_\theta\!\left(y^{(i)} \mid x^{(i)}\right)
$$

| symbol | meaning | shape / units |
|---|---|---|
| $N$ | number of training examples | scalar, dimensionless |
| $P$ | number of scalar parameters | scalar |
| $\theta$ | all weights & biases, flattened | $\mathbb{R}^{P}$ |
| $x^{(i)}$ | input | $\mathbb{R}^{d_{\text{in}}}$ |
| $y^{(i)}$ | target | $\mathbb{R}$ or class index $\in\{1..K\}$ |
| $p_\theta(\cdot)$ | probability *density* (regression) or *mass* (classification) | density has units $1/[y]$; mass is dimensionless |

Maximize likelihood ⇒ maximize log-likelihood (log is monotone, turns the product into a sum, and rescues float underflow — a product of 10,000 probabilities each ~0.5 is $\approx 10^{-3010}$, which is **zero** in float64. This is a real, concrete reason, not a convenience. Say it.)

$$
\hat\theta_{\text{MLE}} \;=\; \arg\max_\theta \sum_{i=1}^N \log p_\theta\!\left(y^{(i)} \mid x^{(i)}\right)
\;=\; \arg\min_\theta \underbrace{-\frac{1}{N}\sum_{i=1}^N \log p_\theta\!\left(y^{(i)} \mid x^{(i)}\right)}_{\textstyle \text{this is the loss } L(\theta)}
$$

The $-$ turns max into min. The $1/N$ makes the loss a *per-example average* so its scale doesn't depend on dataset size (and so the learning rate transfers between batch sizes). **Both of those choices are cosmetic, and the course should say so** — they change nothing about $\arg\min$.

#### Branch A → MSE
Assume $y \mid x \sim \mathcal{N}(f_\theta(x), \sigma^2)$ with $\sigma$ fixed and known.

$$
p_\theta(y\mid x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y - f_\theta(x))^2}{2\sigma^2}\right)
$$

$$
-\log p_\theta(y\mid x) = \frac{(y-f_\theta(x))^2}{2\sigma^2} + \underbrace{\tfrac{1}{2}\log(2\pi\sigma^2)}_{\text{constant in }\theta}
$$

Drop the constant, absorb $1/(2\sigma^2)$ into the learning rate:

$$
\boxed{\;L_{\text{MSE}}(\theta) = \frac{1}{N}\sum_{i=1}^N \left(y^{(i)} - f_\theta(x^{(i)})\right)^2\;}
$$

**The punchline to state explicitly:** MSE is not "the natural way to measure error." MSE *is the assumption that your errors are Gaussian with constant variance.* When that assumption is wrong (heavy tails, outliers, heteroscedastic noise), MSE is wrong, and that's exactly why Huber loss and learned-variance heads exist.

#### Branch B → Binary cross-entropy
Assume $y \in \{0,1\}$, $y \mid x \sim \text{Bernoulli}(\hat y)$ where $\hat y = \sigma(z) = 1/(1+e^{-z})$ and $z = f_\theta(x)$ is the **logit** (units: log-odds, i.e. natural-log of $p/(1-p)$; dimensionless but *not* a probability — hammer this).

The Bernoulli pmf written as one expression (the "exponent trick" — one line, worth pausing on):

$$
p_\theta(y \mid x) = \hat y^{\,y}\,(1-\hat y)^{1-y}
$$

Check: $y=1 \Rightarrow \hat y$. $y=0 \Rightarrow 1-\hat y$. Take $-\log$:

$$
\boxed{\;L_{\text{BCE}} = -\Big[\,y\log \hat y + (1-y)\log(1-\hat y)\,\Big]\;}
$$

**This is not a formula to memorize — it is `-log(probability you assigned to what actually happened)`, algebraically flattened into one line so it's differentiable everywhere.** Say that.

#### Branch C → Categorical cross-entropy (the LLM loss)
$K$ classes. Model emits logits $z \in \mathbb{R}^{K}$. Softmax:

$$
\hat y_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

With one-hot target $y$, class index $c$:

$$
\boxed{\;L_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \hat y_k = -\log \hat y_c = -z_c + \log\sum_{j=1}^{K} e^{z_j}\;}
$$

That last form is the one PyTorch actually computes (`log_softmax` + `nll_loss` fused inside `F.cross_entropy`).

**Worked number — carry it through:** $z = [2.0,\; 1.0,\; 0.1]$, true class $c=0$.

- $e^{z} = [7.389056,\; 2.718282,\; 1.105171]$, $\sum = 11.212509$
- $\hat y = [0.659001,\; 0.242433,\; 0.098566]$
- $L = -\ln(0.659001) = \mathbf{0.417030}$ nats
- Gradient: $\partial L/\partial z = \hat y - y = [\mathbf{-0.340999},\; \mathbf{+0.242433},\; \mathbf{+0.098566}]$ — sums to zero, always, and *that* is worth a callout box (softmax gradients live on the simplex tangent; you can add a constant to all logits with zero effect).

**Reuse this exact triple $[2.0, 1.0, 0.1]$ everywhere in the course** — temperature demos, top-k demos, the diffusion-track classifier-free-guidance discussion. Familiar numbers are pedagogical infrastructure.

**Units of loss.** Cross-entropy in $\log_e$ is measured in **nats**. Divide by $\ln 2 = 0.693147$ for **bits**. A random-guessing LLM over the **Qwen3 151,936-token vocabulary** has $L = \ln(151936) = \mathbf{11.93}$ nats (~~128k → 11.76~~ was the Llama-3 vocab, retired; D-01, constants §9.1). **Perplexity** $= e^{L}$ = "effective number of options the model is choosing between." A well-trained 2026 frontier-scale model on held-out English text sits roughly $L \approx 2.0$–2.4 nats ⇒ PPL ≈ 7–11. **State this early**: it makes "loss went from 11.7 to 2.1" mean something physical. `[CONF: med on the exact 2026 frontier PPL — vocab and tokenizer dependent, and labs report on incomparable corpora. Present as an order-of-magnitude anchor, not a benchmark.]`

### 2.3 The killer argument: why not MSE on a sigmoid?

This is the best five minutes in the whole loss section. Compare $\partial L/\partial z$ (gradient w.r.t. the *logit*) for a confidently-wrong prediction ($y=1$, $z$ very negative):

$$
\frac{\partial L_{\text{MSE}}}{\partial z} = (\hat y - y)\,\sigma'(z), \qquad
\frac{\partial L_{\text{BCE}}}{\partial z} = \hat y - y
$$

| $z$ | $\sigma(z)$ | $\sigma'(z)$ | $\partial L_{\text{MSE}}/\partial z$ | $\partial L_{\text{BCE}}/\partial z$ |
|---|---|---|---|---|
| 0.0 | 0.500000 | 2.50e-01 | −1.25e-01 | −0.500000 |
| −2.0 | 0.119203 | 1.05e-01 | −9.25e-02 | −0.880797 |
| −4.0 | 0.017986 | 1.77e-02 | −1.73e-02 | −0.982014 |
| −6.0 | 0.002473 | 2.47e-03 | −2.46e-03 | −0.997527 |
| −8.0 | 0.000335 | 3.35e-04 | −3.35e-04 | −0.999665 |

**Read the table out loud:** with MSE, the *more confidently wrong* the model is, the *less* it learns — the gradient dies as $e^{-|z|}$. With BCE the gradient saturates at −1: maximally wrong ⇒ maximally corrective. The $\sigma'$ in BCE's chain rule is *exactly cancelled* by the $1/\hat y(1-\hat y)$ from $\frac{d}{d\hat y}\log$. **Cross-entropy is the loss designed to cancel the sigmoid's saturation.** That is why we use it. Not "because it's for classification."

This table is a **required course table.** Reuse the numbers.

### 2.4 Misconceptions (→ warning boxes)

| Misconception | The correction that actually fixes it |
|---|---|
| "Cross-entropy is the classification loss, MSE is the regression loss — they're different species." | They're the same principle (negative log-likelihood) under Gaussian vs. Bernoulli/Categorical noise. Show the shared derivation. |
| "Logits are unnormalized probabilities." | Logits are **log-odds**, live in $(-\infty, \infty)$, and can be shifted by any constant with zero effect on the output. Demo: add +100 to all three logits in the worked example; softmax is unchanged. |
| "Loss going down means the model is getting better." | Loss going down on the **train** set means the model is memorizing. Only val loss tells you about "better." (Hook to §12.) |
| "You should apply softmax then take log then NLL." | `F.cross_entropy` takes **raw logits**. Applying softmax first is the single most common silent bug — it doesn't crash, it just trains badly (double-softmax squashes the distribution). Warning box with a code diff. |
| "$L=0$ is the goal." | For CE, $L=0$ requires infinite logits. It is unreachable, and *approaching* it is overfitting. |
| "Perplexity and loss are different metrics." | $\text{PPL} = e^{L}$. Same number, different scale. |

### 2.5 Numerical stability — LogSumExp (a genuine why-this-exists moment)

Naive softmax on $z = [1000, 1001, 1002]$: $e^{1000}$ **overflows float64** (verified: `math.exp(1000)` → OverflowError; in float32 anything above $z\approx 88.7$ overflows). The fix, $m = \max_j z_j$:

$$
\log\sum_j e^{z_j} \;=\; m + \log\sum_j e^{z_j - m}
$$

Every exponent is now $\le 0$, so every term is in $(0, 1]$. Verified: stable LSE of $[1000,1001,1002]$ = **1002.407606**. This is a two-line identity that makes the difference between a training run and a NaN, and it's a great first taste of "the math on the page and the math in the machine are not the same math."

---

## 3. GRADIENT DESCENT AND THE LOSS LANDSCAPE

### 3.1 Intuition first
> "You're on a foggy hillside at night with an altimeter and a spirit level. You can't see the valley. You can only feel which way is downhill *right here*. So: feel the slope, take a step, feel again. The learning rate is your stride length."

Then immediately break the metaphor, because it's the metaphor's failure that teaches: **the hillside has $P$ dimensions, not 2.** For a 7B model, $P = 7\times10^9$. Your "spirit level" reports 7 billion numbers.

### 3.2 The update rule

$$
\theta_{t+1} = \theta_t - \eta\, \nabla_\theta L(\theta_t)
$$

| symbol | meaning | shape | units |
|---|---|---|---|
| $\theta_t$ | parameter vector at step $t$ | $\mathbb{R}^P$ | [param] |
| $\eta$ | learning rate | scalar | [param]²/[loss] — see below |
| $\nabla_\theta L$ | gradient | $\mathbb{R}^P$, **same shape as $\theta$** | [loss]/[param] |
| $t$ | optimizer step (NOT epoch) | integer | — |

**Do the dimensional analysis. Nobody does this and it's clarifying:** $\eta \nabla L$ must have units of $\theta$. $\nabla L$ has units $[\text{loss}]/[\theta]$. So $\eta$ has units $[\theta]^2/[\text{loss}]$. **The learning rate is not dimensionless and it is not "a percentage."** It is a physical conversion factor whose correct value depends on the scale of your loss, the scale of your weights, and the curvature. This kills the "just use 0.001" reflex and explains why Adam (which normalizes the gradient's scale away) has a much more transferable $\eta$ than SGD.

### 3.3 Why $-\nabla L$ is the *steepest* descent direction
Worth one short derivation because it's cheap and it retires a misconception. For a unit vector $u$, the directional derivative is $\nabla L \cdot u = \|\nabla L\| \cos\phi$, minimized at $\phi = 180°$, i.e. $u = -\nabla L/\|\nabla L\|$.

**Misconception it retires:** "the gradient points at the minimum." It does not. It points at the *locally* steepest direction, which in a curved valley points mostly at the *wall*, not down the valley. This is the entire motivation for momentum and for Adam, and setting it up here pays off in §7.

### 3.4 Learning rate: what too-big and too-small *actually look like*

Use the 1-D quadratic $L(w) = \tfrac{1}{2}\alpha w^2$ so the dynamics are exactly solvable — this is the rare case where the course can *prove* the folklore rather than assert it. Gradient $= \alpha w$, so:

$$
w_{t+1} = w_t - \eta \alpha w_t = (1 - \eta\alpha)\, w_t \quad\Longrightarrow\quad w_t = (1-\eta\alpha)^t w_0
$$

Everything follows from the multiplier $r = 1 - \eta\alpha$:

| regime | condition | $r$ | behavior | what you SEE |
|---|---|---|---|---|
| too small | $\eta \ll 1/\alpha$ | $r \lesssim 1$ | crawl | loss decreases, but a nearly flat line; wastes compute |
| optimal | $\eta = 1/\alpha$ | $r = 0$ | **one step to the exact minimum** | — |
| oscillating-but-converging | $1/\alpha < \eta < 2/\alpha$ | $-1 < r < 0$ | zig-zag, shrinking | loss bounces but trends down; a sawtooth |
| edge of stability | $\eta = 2/\alpha$ | $r = -1$ | ping-pong forever | loss flat at a nonzero value, oscillating |
| **divergence** | $\eta > 2/\alpha$ | $\|r\| > 1$ | explode | loss goes up, then to `inf`, then `NaN` in ~5–30 steps |

**$\eta_{\text{crit}} = 2/\alpha$ where $\alpha$ is the curvature (2nd derivative).** State this as a real theorem the learner can verify in the demo.

Then generalize honestly: in $P$ dimensions the curvature is the **Hessian** $H = \nabla^2 L \in \mathbb{R}^{P\times P}$ (symmetric, so it has real eigenvalues $\lambda_1 \ge \dots \ge \lambda_P$). Along each eigen-direction the 1-D story runs independently. So:

$$
\eta_{\text{max}} = \frac{2}{\lambda_{\max}}
$$

**And here is the whole tragedy of optimization in one inequality:** $\eta$ is bounded above by the *sharpest* direction, but progress along the *flattest* direction goes at rate $\eta\lambda_{\min}$. The ratio $\kappa = \lambda_{\max}/\lambda_{\min}$ (the **condition number**) is how many times slower the flat direction is than it could be. **Real deep nets have $\kappa$ in the $10^4$–$10^6$ range** `[CONF: med — Hessian spectra of real nets are measured only approximately, via Lanczos/Hutchinson; the qualitative claim (very ill-conditioned, heavy-tailed spectrum with a few huge outlier eigenvalues) is well-established, the specific magnitude is regime-dependent]`. **Every optimizer after plain SGD is an attempt to reduce the effective $\kappa$.** Frame §7 with this sentence.

### 3.5 The loss landscape — say what's actually true in 2026

The course must not repeat the 2013-era "local minima are the enemy" story. Current understanding `[CONF: high on 1–3, med on 4]`:

1. **Local minima are largely a non-problem in high dimensions.** For a critical point to be a local min, *all* $P$ eigenvalues must be positive. If eigenvalue signs were even loosely independent, that's a $\sim2^{-P}$ event. Almost every critical point is a **saddle**. The enemy is saddles and, more importantly, **plateaus** — huge flat regions where $\|\nabla L\| \approx 0$ and nothing happens.
2. **The minima that exist are mostly connected.** Mode connectivity results show distinct SGD solutions are joined by low-loss paths. The "isolated pits" mental image is wrong.
3. **Flat minima generalize better than sharp minima** — this is *believed* and *widely acted on* (it's the stated motivation for SAM, for large-LR training, for small batch sizes). **But flag it:** it is contested. Sharpness is not reparameterization-invariant — you can rescale weights in a ReLU net to make any minimum arbitrarily "sharp" without changing the function at all (Dinh et al. 2017). So "flat = generalizes" cannot be literally true as stated. The honest framing: *some* sharpness-like quantity correlates with generalization; nobody has nailed which. **This is a genuine open disagreement — do not paper over it.**
4. **2-D loss-surface plots are lies you agree to.** Every pretty 3-D loss surface (including the one this course will build) is a 2-D random or filter-normalized slice through a $10^9$-D object. It is a shadow. **Show one, then say it's a shadow.** The honesty is itself the lesson: a learner who thinks he's seen the loss landscape will make bad predictions forever.

### 3.6 Misconceptions (→ warning boxes)

| Misconception | Correction |
|---|---|
| "Training gets stuck in local minima." | It gets stuck on saddles and plateaus. High-$P$ makes true local minima vanishingly rare. |
| "Smaller learning rate is safer." | Safer against divergence, but it *can't* find the flat solutions and it wastes compute; too-small LR is a real failure mode that looks like "it's training, just slowly, forever." |
| "The gradient tells you how far to go." | It tells you a **direction and a local rate**, valid only infinitesimally. Step size is a separate, unrelated decision. This is why $\eta$ exists at all. |
| "The loss surface looks like the 3-D picture." | It's a 2-D slice of a $10^9$-D object. |
| "If loss is flat, the model has converged." | Flat loss = small gradient *or* a plateau *or* a dead LR *or* dead ReLUs. Four different diseases, one symptom. (Hook to §14.) |

---

## 4. BATCH, MINI-BATCH, STOCHASTIC — AND WHY NOISE IS A FEATURE

### 4.1 Intuition first
> "You don't need to poll every voter to know which way the country leans. A random sample of 32 gets you the direction, roughly, for 1/1000th of the cost — and you can take 1000 noisy steps in the time it takes to make one perfect one. **A thousand drunk steps downhill beat one sober step.**"

### 4.2 The three regimes

$$
\nabla L_{\mathcal{B}}(\theta) = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \nabla_\theta \ell\!\left(f_\theta(x^{(i)}), y^{(i)}\right)
$$

| name | $|\mathcal{B}|$ | gradient property | practical |
|---|---|---|---|
| Full-batch (GD) | $N$ | exact $\nabla L$ | 1 update/epoch; infeasible for $N=10^{12}$ tokens; also *generalizes worse* |
| Mini-batch SGD | 8 – 4096 (typ. 32–256 for vision) | unbiased, $\text{Var} \propto 1/|\mathcal{B}|$ | what everyone does |
| Stochastic (true SGD) | 1 | unbiased, huge variance | rare; no GPU parallelism |

**Key statistical fact, stated precisely:** $\mathbb{E}[\nabla L_{\mathcal{B}}] = \nabla L$ (unbiased — the mini-batch gradient is *right on average*), and $\text{Var}[\nabla L_{\mathcal{B}}] = \sigma^2/|\mathcal{B}|$. So **the noise scales as $1/\sqrt{|\mathcal{B}|}$**: quadrupling the batch only halves the noise. Diminishing returns are built into the square root. This is why batch size 4096 isn't 128× better than 32.

### 4.3 Why the noise is a FEATURE — three separate mechanisms, all real

Don't hand-wave "noise helps escape local minima." Give the three actual mechanisms:

1. **Saddle escape.** At a saddle, $\nabla L = 0$ exactly, so full-batch GD *stops dead* — it is a fixed point. Mini-batch noise means $\nabla L_{\mathcal{B}} \ne 0$, so you get kicked off the saddle along the negative-curvature direction. **Noise is what makes saddles not-fatal.** (Confidence: high; this is the cleanest of the three.)
2. **Implicit regularization / flat-minimum seeking.** Gradient noise acts like a temperature. A sharp minimum is a narrow well; noise of amplitude $\propto \eta/|\mathcal{B}|$ bounces you out of narrow wells and lets you settle only in wide ones. This motivates the **linear scaling rule**: if you multiply batch size by $k$, multiply $\eta$ by $k$ to hold the noise scale $\eta/|\mathcal{B}|$ constant (Goyal et al. 2017 — ImageNet in 1 hour, batch 8192, held to ~76% top-1). `[CONF: med — the rule is empirically load-bearing and widely used, but it breaks down above some batch size (the "critical batch size"), and $\sqrt{k}$ scaling is preferred by some for Adam. Present the rule AND its breakdown.]`
3. **Compute efficiency, which is the real reason.** Be honest: the dominant reason for mini-batching is that a GPU is a wide SIMD machine that is idle at batch 1 and saturated around batch 32–256. The regularization benefits are a happy accident that people rationalized afterward. **Say this.** It's true, and the learner (who owns a DGX Spark and thinks about throughput) will trust the course more for it.

### 4.4 Vocabulary the course must nail down and then never violate
- **step / iteration** = one optimizer update = one mini-batch.
- **epoch** = one full pass over the training set = $N/|\mathcal{B}|$ steps.
- **LR schedules are indexed by STEP, not epoch.** Getting this wrong is a top-5 real bug.
- **Gradient accumulation:** run $k$ micro-batches, sum grads, step once. Effective batch $= k \cdot |\mathcal{B}_{\text{micro}}|$. Mathematically identical to a big batch (modulo BatchNorm, which it breaks — see §11). This is *the* technique the learner will use on his own hardware, so introduce it here, in the trunk, not later.

### 4.5 Misconceptions

| Misconception | Correction |
|---|---|
| "SGD is an approximation to 'real' gradient descent — a compromise." | SGD *outperforms* full-batch GD on test error. The noise is doing work. It's not a compromise, it's a different (better) algorithm. |
| "Bigger batch = better/faster training." | Bigger batch = better *hardware utilization* and *lower variance*, but past the critical batch size you get almost no reduction in steps-to-target, and you may generalize worse without an LR adjustment. |
| "Noise helps escape local minima." | Mostly it helps escape **saddles** and **narrow** minima. Local minima weren't the problem. |
| "Batch size and learning rate are independent knobs." | They are coupled through the noise scale $\eta/\|\mathcal{B}\|$. Change one, reconsider the other. |

---

## 5. ⭐ BACKPROPAGATION — THE FULL WORKED EXAMPLE (THE SPINE OF THE COURSE)

> **This section is the single most important thing in the brief. Budget 4–6 pages. The learner should be able to do this by hand on paper afterward, and should be MADE to.**

### 5.1 Intuition first (three framings, use all three)

1. **Blame assignment.** "The output was wrong by −0.36. Whose fault? Backprop is a rigorous procedure for distributing blame backwards through the network, so every weight learns exactly how much *it* contributed to the error."
2. **Sensitivity flows backwards.** "Forward pass computes *values*. Backward pass computes *sensitivities*. They travel in opposite directions along the same wires."
3. **The chain rule, but you're not stupid about it.** "Backprop is not a new idea. Backprop is the chain rule **plus one engineering insight**: don't recompute shared sub-expressions. That's it. That's the whole invention." — This framing is crucial. Learners think backprop is exotic. It's the chain rule with memoization. Say so, plainly, and their fear drops.

### 5.2 The central object: $\delta$

Define, for every neuron, the **local error signal**

$$
\delta^{(l)}_j \;\equiv\; \frac{\partial L}{\partial z^{(l)}_j}
$$

*"How much would the loss change if I nudged this neuron's pre-activation?"*

**Why define it at $z$ and not at $a$ or at $w$?** Because $\delta$ at the pre-activation is the *reusable* quantity — once you have it, every weight gradient into that neuron is a one-multiply away:

$$
\frac{\partial L}{\partial w^{(l)}_{ji}} = \delta^{(l)}_j \cdot a^{(l-1)}_i, \qquad \frac{\partial L}{\partial b^{(l)}_j} = \delta^{(l)}_j
$$

**Every weight gradient in every neural network ever trained has this form: `(error signal at the output end of the wire) × (activation at the input end of the wire)`.** If the learner takes one sentence from the course, this is a strong candidate. It's also the sentence that makes the $\partial L/\partial W = \delta a^\top$ outer-product structure obvious later.

### 5.3 The four backprop equations (in $\delta$ form)

$$
\begin{aligned}
\text{(BP1) output layer:}\quad & \delta^{(L)} = \nabla_{a}L \odot \sigma'\!\left(z^{(L)}\right) \\
\text{(BP2) recursion:}\quad & \delta^{(l)} = \left(\left(W^{(l+1)}\right)^{\!\top} \delta^{(l+1)}\right) \odot \sigma'\!\left(z^{(l)}\right) \\
\text{(BP3) bias grad:}\quad & \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)} \\
\text{(BP4) weight grad:}\quad & \frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^{\!\top}
\end{aligned}
$$

where $\odot$ is elementwise (Hadamard) product.

**Read BP2 aloud, it's the whole algorithm:** *"Take the error from the layer above. Push it back through the transpose of the weights (that's 'which of my neurons fed into that error, and how strongly'). Then multiply by how responsive my neuron was (that's $\sigma'$ — a saturated neuron passes no blame)."*

**Shape check — make the learner do this, it catches 90% of implementation bugs:**
- $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$, $a^{(l-1)} \in \mathbb{R}^{n_{l-1}}$, $z^{(l)}, b^{(l)}, \delta^{(l)} \in \mathbb{R}^{n_l}$
- BP2: $(n_{l+1}\times n_l)^\top \cdot (n_{l+1}) = (n_l \times n_{l+1})\cdot(n_{l+1}) = (n_l)$ ✓
- BP4: $(n_l)\cdot(n_{l-1})^\top = (n_l \times n_{l-1})$ = shape of $W^{(l)}$ ✓
- **The gradient always has the same shape as the thing it's the gradient of.** Universal law. Reuse it as a debugging tool forever.

### 5.4 ⭐ THE 2-2-1 EXAMPLE — every number verified

> **⛔ THIS NETWORK IS RETIRED — DO NOT USE THESE NUMBERS (D-02). The prose/beats re-host on TN-1; the numbers do
> NOT.** This section's weights ($W^{(2)}=[0.7,-0.4]$, second hidden unit) differ from the canonical TN-1
> (`constants.md` §8). **The input `x=[0.60,−0.20]` survives as TN-1's frozen SECOND input, but on TN-1's weights
> the arithmetic is DIFFERENT** — use `constants.md` §8.7, NOT the values below:
> `δ² = −0.4593` (not −0.364312) · `a¹ = [0.4301, 0.3275]` (not [0.4301, −0.1391]) ·
> `∂L/∂W¹ = [[−0.1348, 0.0449],[0.2214, −0.0738]]` (not [[−0.1247,…]]) · `L = 0.6148` (not 0.453048) ·
> **gradient spread = 10.22×** (not 12.7×). **Every callout below (the sign-flip, the "this line IS backprop",
> the vanishing-gradient setup, the "loss went down with a pencil" beat) transfers verbatim; only the decimals
> change.** Retained beats, re-hosted on TN-1's §8.7 numbers. A builder copying the numbers below ships a page
> that contradicts nine others.

**All numbers below were computed and cross-checked against central finite differences ($\epsilon = 10^{-6}$); analytic and numeric gradients agree to 8 decimal places. ~~Use them verbatim.~~ RETIRED — see the banner above; use `constants.md` §8.7.**

#### Setup

Architecture: 2 inputs → 2 hidden (**tanh**) → 1 output (**sigmoid**) → **BCE loss**.

Chosen deliberately: tanh gives a clean $\sigma' = 1-a^2$; sigmoid+BCE gives the beautiful $\delta = \hat y - y$ cancellation from §2.3, which the learner has *already seen* and will now watch pay off.

$$
x = \begin{bmatrix}0.60 \\ -0.20\end{bmatrix},\quad y = 1
$$
$$
W^{(1)} = \begin{bmatrix}0.50 & -0.30\\ 0.20 & 0.80\end{bmatrix},\quad
b^{(1)} = \begin{bmatrix}0.10\\ -0.10\end{bmatrix},\quad
W^{(2)} = \begin{bmatrix}0.70 & -0.40\end{bmatrix},\quad
b^{(2)} = 0.20
$$

Parameter count: $4 + 2 + 2 + 1 = \mathbf{9}$. Nine numbers. Nine partial derivatives. **The learner will compute all nine.** Then point out: GPT-scale models do exactly this, unchanged, $10^9$ times.

#### FORWARD PASS

$$
z^{(1)}_1 = 0.50(0.60) + (-0.30)(-0.20) + 0.10 = 0.30 + 0.06 + 0.10 = \mathbf{0.46}
$$
$$
z^{(1)}_2 = 0.20(0.60) + 0.80(-0.20) + (-0.10) = 0.12 - 0.16 - 0.10 = \mathbf{-0.14}
$$

(Note: these came out exact to 2 decimals — the numbers were chosen for that. Good for hand-work.)

$$
a^{(1)}_1 = \tanh(0.46) = \mathbf{0.430084}, \qquad a^{(1)}_2 = \tanh(-0.14) = \mathbf{-0.139092}
$$

$$
z^{(2)} = 0.70(0.430084) + (-0.40)(-0.139092) + 0.20 = 0.301059 + 0.055637 + 0.20 = \mathbf{0.556696}
$$

$$
\hat y = \sigma(0.556696) = \frac{1}{1+e^{-0.556696}} = \mathbf{0.635688}
$$

$$
L = -[\,1\cdot\ln(0.635688) + 0\,] = \mathbf{0.453048}
$$

*Sanity anchor for the learner:* a coin-flip guess ($\hat y = 0.5$) gives $L = \ln 2 = 0.693$. We're at 0.453, so we're already better than chance — the random init happened to lean the right way.

#### BACKWARD PASS

**Step 1 — output error (BP1).** Because sigmoid+BCE, the whole $\sigma'$ mess cancels:

$$
\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = \hat y - y = 0.635688 - 1 = \mathbf{-0.364312}
$$

**Do NOT let this cancellation go by silently.** Derive it once, in full, on the page:
$$
\frac{\partial L}{\partial \hat y} = -\frac{y}{\hat y} + \frac{1-y}{1-\hat y} = \frac{\hat y - y}{\hat y (1-\hat y)}, \qquad \frac{\partial \hat y}{\partial z} = \hat y(1-\hat y)
$$
$$
\Rightarrow\quad \frac{\partial L}{\partial z} = \frac{\hat y - y}{\cancel{\hat y (1-\hat y)}} \cdot \cancel{\hat y(1-\hat y)} = \hat y - y
$$
The learner should feel the click. *This is why we paired sigmoid with BCE.* It is design, not coincidence. (And it's why §2.3's table exists.)

Sign reading: $\delta^{(2)} < 0$ means "increasing $z^{(2)}$ decreases the loss" — we want the output **higher**, and $y=1$. ✓ Make the learner check the sign's meaning; it's the difference between following a recipe and understanding.

**Step 2 — output-layer gradients (BP3, BP4).**

$$
\frac{\partial L}{\partial W^{(2)}_1} = \delta^{(2)} a^{(1)}_1 = (-0.364312)(0.430084) = \mathbf{-0.156685}
$$
$$
\frac{\partial L}{\partial W^{(2)}_2} = \delta^{(2)} a^{(1)}_2 = (-0.364312)(-0.139092) = \mathbf{+0.050673}
$$
$$
\frac{\partial L}{\partial b^{(2)}} = \delta^{(2)} = \mathbf{-0.364312}
$$

**Callout — the sign flip is the lesson.** Both weights get gradients, but with *opposite signs*, purely because $a^{(1)}_1 > 0$ and $a^{(1)}_2 < 0$. Hidden unit 1 fired positive, so to raise the output we raise its weight. Hidden unit 2 fired negative, so to raise the output we *lower* its weight (a negative number times a more-negative weight = more positive). **This is `error × activation` doing real, interpretable work.** Spend a paragraph here.

**Step 3 — backprop through the layer (BP2).**

tanh derivative: $\tanh'(z) = 1 - \tanh^2(z) = 1 - a^2$. (Derive it once — it's a nice, short, high-school-doable exercise and it makes the learner trust the machinery.)

$$
1 - (a^{(1)}_1)^2 = 1 - 0.430084^2 = \mathbf{0.815028}, \qquad 1 - (a^{(1)}_2)^2 = 1 - (-0.139092)^2 = \mathbf{0.980653}
$$

$$
\delta^{(1)}_1 = \delta^{(2)}\cdot W^{(2)}_1 \cdot (1-(a^{(1)}_1)^2) = (-0.364312)(0.70)(0.815028) = \mathbf{-0.207847}
$$
$$
\delta^{(1)}_2 = \delta^{(2)}\cdot W^{(2)}_2 \cdot (1-(a^{(1)}_2)^2) = (-0.364312)(-0.40)(0.980653) = \mathbf{+0.142906}
$$

**Callout — this line IS backprop.** Three factors, three meanings:
- $\delta^{(2)}$ — "how much error arrived from above"
- $W^{(2)}_j$ — "how strongly did I contribute to it" (and the sign flips the blame — hidden unit 2 has a *negative* weight, so it gets *opposite-signed* blame)
- $(1-a^2)$ — "how responsive am I right now" — **the gate**. Notice $0.815$ and $0.981$: both units are in the responsive part of tanh, so blame passes through nearly intact. **Now ask the learner: what if $a^{(1)}_1$ had been $0.99$?** Then $1-a^2 = 0.0199$, and 98% of the blame is destroyed by that one multiply. **That is vanishing gradient, and the learner just found it himself, in a 9-parameter network, before we ever name it.** ← This is the setup for §10. Do it exactly here.

**Step 4 — input-layer gradients (BP3, BP4).**

$$
\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} x^\top =
\begin{bmatrix}-0.207847\\ +0.142906\end{bmatrix}
\begin{bmatrix}0.60 & -0.20\end{bmatrix}
= \begin{bmatrix}\mathbf{-0.124708} & \mathbf{+0.041569}\\ \mathbf{+0.085743} & \mathbf{-0.028581}\end{bmatrix}
$$

$$
\frac{\partial L}{\partial b^{(1)}} = \delta^{(1)} = \begin{bmatrix}\mathbf{-0.207847}\\ \mathbf{+0.142906}\end{bmatrix}
$$

**All nine gradients, collected:**

| parameter | value | gradient | interpretation |
|---|---|---|---|
| $W^{(1)}_{11}$ | 0.50 | −0.124708 | increase it |
| $W^{(1)}_{12}$ | −0.30 | +0.041569 | decrease it |
| $W^{(1)}_{21}$ | 0.20 | +0.085743 | decrease it |
| $W^{(1)}_{22}$ | 0.80 | −0.028581 | increase it |
| $b^{(1)}_{1}$ | 0.10 | −0.207847 | increase it |
| $b^{(1)}_{2}$ | −0.10 | +0.142906 | decrease it |
| $W^{(2)}_{1}$ | 0.70 | −0.156685 | increase it |
| $W^{(2)}_{2}$ | −0.40 | +0.050673 | decrease it |
| $b^{(2)}$ | 0.20 | −0.364312 | increase it |

**⚠️ RE-HOST ON TN-1's §8.7 NUMBERS (D-02): the spread is `10.22×`, NOT 12.7×** (12.7× was this retired
network's; TN-1's second input gives $|\partial L/\partial b_2| = 0.4593$ largest vs $|\partial L/\partial W_1[0][1]| = 0.0449$
smallest → **10.22×, all nine gradients live, no dead unit**, constants §8.7). The lesson is identical and the
sentence still costs one line: the largest gradient is ~10× the smallest — **that spread is the condition-number
problem from §3.4, visible in nine numbers.** Extrapolate: across 36 transformer layers it is astronomical.
**This is why per-parameter adaptive step sizes (Adam) exist.** ← The single best motivation for §7. Use it, with 10.22×.

#### ONE SGD STEP, $\eta = 0.5$

$$
\theta \leftarrow \theta - 0.5 \cdot \nabla_\theta L
$$

| parameter | old | new |
|---|---|---|
| $W^{(1)}_{11}$ | 0.50 | **0.562354** |
| $W^{(1)}_{12}$ | −0.30 | **−0.320785** |
| $W^{(1)}_{21}$ | 0.20 | **0.157128** |
| $W^{(1)}_{22}$ | 0.80 | **0.814291** |
| $b^{(1)}_{1}$ | 0.10 | **0.203924** |
| $b^{(1)}_{2}$ | −0.10 | **−0.171453** |
| $W^{(2)}_{1}$ | 0.70 | **0.778342** |
| $W^{(2)}_{2}$ | −0.40 | **−0.425337** |
| $b^{(2)}$ | 0.20 | **0.382156** |

**Re-run the forward pass on the new weights:**

$$
\hat y: 0.635688 \;\longrightarrow\; \mathbf{0.711643}, \qquad L: 0.453048 \;\longrightarrow\; \mathbf{0.340179} \quad (\Delta L = \mathbf{-0.112869})
$$

**The loss went down. The learner did that with a pencil.** End the section here, on that beat. It's the emotional payoff of the entire trunk and everything after is elaboration.

#### GRADIENT CHECKING (include this — it's a superpower)

$$
\frac{\partial L}{\partial \theta_k} \approx \frac{L(\theta + \epsilon e_k) - L(\theta - \epsilon e_k)}{2\epsilon}
$$

**Central** differences, error $O(\epsilon^2)$, not forward differences ($O(\epsilon)$). Verified on this network with $\epsilon = 10^{-6}$:

| parameter | finite-difference | analytic backprop | agree to |
|---|---|---|---|
| $W^{(1)}_{11}$ | −0.12470832 | −0.12470832 | 8 d.p. |
| $W^{(1)}_{22}$ | −0.02858112 | −0.02858112 | 8 d.p. |

Teach the check, and teach its failure mode: **use float64 for gradient checking.** In float32, $\epsilon=10^{-6}$ is near machine epsilon ($\approx 1.19\times10^{-7}$) and the subtraction $L(\theta+\epsilon)-L(\theta-\epsilon)$ catastrophically cancels — you get noise, and you'll blame your correct backprop. `torch.autograd.gradcheck` defaults to float64 for exactly this reason. This is a great second instance of the §2.5 lesson (the math on paper ≠ the math in the machine).

### 5.5 Why backprop is *cheap* — the complexity argument

Must be stated; it's why deep learning exists at all.

- **Naive finite differences:** to get all $P$ gradients you need $2P$ forward passes. For $P=7\times10^9$: **14 billion forward passes per training step.** At ~1 second each that's ~440 years. Per step.
- **Backprop:** one forward + one backward. The backward pass costs about **2×** the forward (each weight participates in two matmuls going back: one for $\delta^{(l)}$, one for $\partial L/\partial W^{(l)}$). So **the full gradient costs ~3× a forward pass, independent of $P$.**

**The famous rule: `training FLOPs ≈ 6 × N_params × N_tokens`** (2 FLOPs per param per token forward — a multiply and an add — × 3 for fwd+bwd = 6). Give the learner one worked instance and let it stick:

> A 7B model on 1 trillion tokens: $6 \times 7\times10^9 \times 1\times10^{12} = 4.2\times10^{22}$ FLOPs. An H100 does ~$10^{15}$ bf16 FLOP/s at maybe 40% MFU ⇒ $4\times10^{14}$ effective FLOP/s ⇒ $1.05\times10^{8}$ s ≈ **3.3 GPU-years**, or ~12 days on 100 GPUs.

This number recurs in the LLM track and in the "why you fine-tune instead of pretrain" argument. **Reuse it verbatim.**

**The trade backprop makes: memory for time.** You must *store every activation* from the forward pass to use in BP4 ($\partial L/\partial W^{(l)} = \delta^{(l)} (a^{(l-1)})^\top$ needs $a^{(l-1)}$). **This is why training needs vastly more VRAM than inference, and it is the thing the learner has already hit in practice.** Name it here; it makes gradient checkpointing (§ handoff) and LoRA make sense later.

### 5.6 Misconceptions (→ warning boxes)

| Misconception | Correction that fixes it |
|---|---|
| "Backprop is a learning algorithm." | **Backprop computes gradients. Full stop.** SGD/Adam is the learning algorithm. You can backprop and then do nothing. Conflating them is why learners can't understand why there are many optimizers but one backprop. **This is the #1 misconception in the whole topic.** |
| "Backprop is the chain rule." | It's the chain rule *plus dynamic programming*. Applying the chain rule naively to a graph re-computes shared subpaths exponentially often. The $\delta$ recursion is the memoization. That's the invention. |
| "The backward pass runs the network in reverse / inverts the weights." | Nothing is inverted. It's $W^\top$, not $W^{-1}$. The transpose appears because of how sums distribute in the chain rule, not because we're undoing anything. (Many learners silently believe the inverse thing.) |
| "Each layer's gradient is computed independently." | $\delta^{(l)}$ depends on $\delta^{(l+1)}$ — it's a strict backward recursion. That sequential dependency is exactly why the backward pass can't be parallelized across depth, and why pipeline parallelism has bubbles. |
| "Gradients tell you the error of each neuron." | They tell you the **sensitivity of the loss to** each neuron. A neuron with zero gradient isn't "correct" — it's *irrelevant* (or dead). |
| "You need the labels during the backward pass." | You need them only to compute $\delta^{(L)}$ at the very top. After that, no labels appear anywhere. |

---

## 6. AUTOMATIC DIFFERENTIATION — WHAT PYTORCH ACTUALLY DOES

### 6.1 Intuition first
> "You already did backprop by hand for 9 parameters and it took ten minutes. PyTorch does it for 7 billion, in 40 milliseconds, and *nobody wrote the derivative code*. Here's the trick: every single operation in the library ships with its own tiny hand-written derivative rule. PyTorch just records which operations you actually ran, then plays the tape backwards, applying each rule. **It is not doing calculus. It is doing bookkeeping.**"

### 6.2 The three things people confuse

Make this table; it retires a whole class of confusion:

| method | what it is | why it's not used |
|---|---|---|
| **Symbolic diff** (Mathematica) | manipulates the *formula* into another formula | expression swell — the symbolic derivative of a 100-layer net is astronomically large |
| **Numerical diff** (finite differences) | $(L(\theta+\epsilon)-L(\theta-\epsilon))/2\epsilon$ | $O(P)$ forward passes; catastrophic cancellation |
| **Automatic diff** (PyTorch) | applies exact chain rule to the *executed trace*, numerically | ✓ exact to float precision, $O(1)$ passes |

**Autodiff is exact.** It is *not* an approximation. Learners routinely believe autodiff is a numerical estimate. It isn't — it's the same exact answer you got by hand in §5.4, computed by machine. That's why the finite-difference check matched to 8 decimals.

### 6.3 Forward-mode vs reverse-mode — the *real* reason we use reverse

$f: \mathbb{R}^{n} \to \mathbb{R}^{m}$, Jacobian $J \in \mathbb{R}^{m\times n}$.

- **Forward mode** computes a **JVP** ($J v$): one pass gives you one *column* of $J$ = the sensitivity of *all outputs* to *one input*. Cost to get the full Jacobian: $n$ passes.
- **Reverse mode** computes a **VJP** ($v^\top J$): one pass gives you one *row* of $J$ = the sensitivity of *one output* to *all inputs*. Cost to get the full Jacobian: $m$ passes.

**Deep learning has $n = P = 10^9$ and $m = 1$ (the loss is a scalar).** So reverse mode costs **one** pass and forward mode costs **a billion**. That asymmetry — *and nothing else* — is why we use reverse mode. **Backprop is just reverse-mode AD applied to a scalar loss.** Say that sentence; it collapses two concepts into one.

Nice corollary the learner will meet later: forward mode isn't useless — it wins when $m \gg n$, and it's what `torch.func.jvp` is for.

### 6.4 The tape / computation graph — the mechanics

- Every tensor with `requires_grad=True` gets a `.grad_fn` pointing at the op that created it.
- The graph is a **DAG built dynamically, during the forward pass** ("define-by-run"). This is the actual difference from TF1: the graph is a *record of what happened*, not a *program you declared*. If your forward has an `if`, the graph differs between iterations. That's fine, and it's why PyTorch won.
- `loss.backward()` walks that DAG from `loss` back to every leaf, calling each op's registered VJP, and **accumulates** into `.grad` on leaves.
- The graph is **freed after `.backward()`** by default. Calling `.backward()` twice → `RuntimeError: Trying to backward through the graph a second time`. The fix people reach for (`retain_graph=True`) is *usually the wrong fix* and masks a real bug: you probably meant to detach something. **Warning box.**

### 6.5 `.zero_grad()` — why it exists and what breaks without it

**PyTorch accumulates `.grad` with `+=`, not `=`.** This surprises everyone. It is a deliberate design choice, not an oversight, and it exists so that:
1. **gradient accumulation** works for free (§4.4) — run $k$ micro-batches, `.backward()` each, step once;
2. multiple losses can be summed into one backward (multi-task, GAN discriminators);
3. a parameter used twice in a forward pass (weight tying — e.g. an LLM's tied embedding/unembedding matrix!) correctly gets the **sum** of both paths' gradients, which is what the multivariable chain rule demands.

**Point 3 is the deep reason and the course should lead with it:** if a node has two outgoing edges in the DAG, its gradient is the *sum over paths*. `+=` isn't a convenience — it's the chain rule's sum rule, implemented.

**What actually breaks without `zero_grad()`:** step $t$'s gradient is the sum of *all* gradients since the last zeroing. So by step 100, your gradient is ~100× too large and points at a stale average of 100 different mini-batches. **Symptom: loss rises, then NaN, within ~20–100 steps, on a config that "should work."** This is a top-3 beginner bug and *the course must show it failing*, not just warn.

**The canonical 2026 loop** (verify against PyTorch 2.10 docs at spec-writing time):

```python
for x, y in loader:
    optimizer.zero_grad(set_to_none=True)   # default is True since PyTorch 2.0
    loss = criterion(model(x), y)           # forward: BUILDS the graph
    loss.backward()                          # backward: WALKS the graph, fills .grad
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()                         # optimizer READS .grad, mutates params
```

**Teach the loop as four separate responsibilities, and make the learner say which line owns which:** zero (housekeeping) → forward+loss (§2) → backward (§5) → clip (§10) → step (§7). Each line maps to a section of this brief. That mapping is the course's structural payoff.

`set_to_none=True` sets `.grad = None` instead of a zero tensor: saves memory and one kernel launch. Gotcha to flag: `.grad` is then `None`, not `0`, so code that inspects `p.grad` between epochs must handle `None`.

### 6.6 The gotchas that will actually bite this learner

| gotcha | what happens | fix |
|---|---|---|
| `total_loss += loss` in a logging accumulator | **keeps the whole graph alive** for every iteration → VRAM climbs linearly → OOM at epoch 1, step ~500 | `total_loss += loss.item()` or `.detach()` |
| `with torch.no_grad():` missing at eval | builds a graph you never use; ~2× the memory, slower | wrap eval in `torch.no_grad()` (or `torch.inference_mode()`, which is strictly faster — it also disables version counting) |
| in-place ops (`x += 1`, `relu_()`) on a tensor needed for backward | `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` | remove the underscore |
| calling `.backward()` on a non-scalar | `RuntimeError: grad can be implicitly created only for scalar outputs` | reduce it, or pass an explicit `v` for the VJP |
| optimizer constructed before `.to(device)` | optimizer holds params on the wrong device; silent-ish weirdness | `.to(device)` first, then build the optimizer |
| detaching by accident (`.data`, `.numpy()`) | gradient silently becomes zero; **no error**, just no learning | never use `.data`; use `.detach()` deliberately |

**"Silent zero-gradient" is the worst failure class in the field** — no traceback, model just doesn't learn. Give it a warning box and a diagnostic: after `.backward()`, assert that gradients exist and aren't all zero.

### 6.7 Where autodiff bites the diffusion track (flag for cross-brief reconciliation)
The stop-gradient / `.detach()` discipline in diffusion training (detaching the noise target, EMA weights being updated *outside* autograd) is where the LLM and diffusion tracks first diverge in a way the trunk must prepare for. **The trunk should teach `.detach()` as "cutting a wire in the graph on purpose," so the diffusion brief can just say "detach the target" and be understood.**

---

## 7. OPTIMIZERS: SGD → MOMENTUM → RMSPROP → ADAM → ADAMW

### 7.0 The frame to open with
> "Every optimizer from here is answering one question: **the raw gradient is a bad step direction — how do we fix it without paying for second derivatives?**" (Callback to §3.4: the Hessian is $P\times P$; for $P=7\times10^9$ that's $4.9\times10^{19}$ entries ≈ **200 exabytes** in fp32. Newton's method is not merely slow, it is *physically impossible*. Compute that number on the page. It permanently ends the "why not just use second-order methods" question.)

### 7.1 The lineage — each fixes one specific thing

| # | optimizer | intuition (one sentence) | what it fixes | state/param | year |
|---|---|---|---|---|---|
| 1 | **SGD** | step downhill | — | 0 | 1951 |
| 2 | **+ Momentum** | a heavy ball has inertia; it averages out the zig-zag and coasts across plateaus | ravines (high $\kappa$), noise, plateaus | 1 buffer | 1964/1983 |
| 3 | **RMSProp** | give each parameter its own step size, inversely scaled by how big its gradients have been | the **10.22×** gradient-scale spread from TN-1's second input (constants §8.7; ~~12.7×~~ retired with the old §5.4 network, D-02) | 1 buffer | 2012 |
| 4 | **Adam** | momentum + RMSProp, plus a fix for the cold start | both at once | **2 buffers** | 2014 |
| 5 | **AdamW** | decouple weight decay from the adaptive scaling, because L2-in-the-gradient gets *divided by* $\sqrt{v}$ and stops being L2 | broken regularization in Adam | 2 buffers | 2017/19 |

**Present it as a story of five fixes, not five algorithms.** That's the difference between memorizing and understanding.

### 7.2 Momentum

$$
v_{t} = \mu\, v_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t
$$

($g_t = \nabla L_{\mathcal{B}}(\theta_t)$; $\mu$ = momentum coefficient, dimensionless; $v$ = velocity, $\in\mathbb{R}^P$, same shape as $\theta$.)

**The number that makes momentum click:** on a constant gradient $g$, the velocity converges to a geometric series
$$
v_\infty = g\,(1 + \mu + \mu^2 + \cdots) = \frac{g}{1-\mu}
$$
So **$\mu = 0.9$ multiplies your effective step by $1/(1-0.9) = 10\times$.** That single fact explains (a) why momentum is fast, and (b) **why turning on momentum without lowering $\eta$ makes your run diverge** — you just secretly 10×'d your learning rate. Learners hit this constantly. Give it a warning box. $\mu=0.99 \Rightarrow 100\times$.

Effective averaging window $\approx 1/(1-\mu)$ steps. $\mu=0.9$ → ~10 steps. This "$1/(1-\beta)$ = memory length" rule recurs for $\beta_1$, $\beta_2$, and EMA in diffusion. **Establish it once here and reuse the phrase.**

**PyTorch gotcha worth naming:** `torch.optim.SGD(momentum=0.9)` implements $v_t = \mu v_{t-1} + g_t$ (gradient *not* scaled by $1-\mu$), so the 10× amplification is real and exposed. Some frameworks/papers use $v_t = \mu v_{t-1} + (1-\mu)g_t$, which normalizes it away. **Same name, different scaling, LRs don't transfer.** This is a real cross-framework footgun.

Nesterov (`nesterov=True`): look ahead before you leap — evaluate the gradient at $\theta_t - \eta\mu v_{t-1}$. Modest, real gain. Mention, don't dwell.

### 7.3 RMSProp

$$
s_t = \rho\, s_{t-1} + (1-\rho)\, g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon}\, g_t
$$

($g_t^2$ = **elementwise** square; $s_t \in \mathbb{R}^P$; $\rho \approx 0.9$–0.99; $\epsilon \approx 10^{-8}$.)

**Intuition:** $\sqrt{s_t}$ is a running RMS estimate of each parameter's gradient magnitude, so $g_t/\sqrt{s_t}$ is **roughly ±1 for every parameter**. **Adaptive methods are approximately sign descent with a smoothed magnitude.** That's the honest one-liner, and it's the key to why they're robust: they *throw away gradient magnitude and keep direction*.

**Consequence to state explicitly** (it will save the learner a week later): because the update is $\approx \eta \cdot \text{sign}$, **the parameter-space step size in Adam is $\approx \eta$ regardless of the loss scale.** That's why Adam LRs transfer across models and SGD LRs don't. It's also why **loss scaling in mixed precision doesn't change Adam's effective step** (the scale cancels in $g/\sqrt{g^2}$) — a genuinely confusing point that this framing resolves in one sentence.

The $\epsilon$ is not a rounding detail: it's the *floor* on the step size. It's inside the sqrt in some implementations and outside in others, and it matters at low precision.

### 7.4 Adam (full, with bias correction derived)

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t &&\text{(1st moment: momentum, EMA of }g) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 &&\text{(2nd moment: RMSProp, EMA of }g^2) \\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} &&\text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{aligned}
$$

**Derive the bias correction. It takes four lines and it's the part everyone skips.**

$m_0 = 0$. After one step: $m_1 = (1-\beta_1)g_1$. With $\beta_1 = 0.9$, $m_1 = 0.1 g_1$ — **the estimate of the mean gradient is 10× too small**, purely because we initialized the EMA at zero. Unrolling: $m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i} g_i$, and if $g_i \approx g$ then $\mathbb{E}[m_t] \approx g(1-\beta_1^t)$. Dividing by $(1-\beta_1^t)$ makes it unbiased. As $t\to\infty$, $\beta_1^t \to 0$ and the correction fades to 1.

**Concrete numbers to put on the page:**

| $t$ | $1-\beta_1^t$ ($\beta_1{=}0.9$) | $1-\beta_2^t$ ($\beta_2{=}0.999$) | correction to $v$ |
|---|---|---|---|
| 1 | 0.1 | 0.001 | **1000×** |
| 10 | 0.651 | 0.00996 | 100× |
| 100 | 0.99997 | 0.0952 | 10.5× |
| 1000 | ≈1 | 0.632 | 1.58× |
| 5000 | 1 | 0.9933 | 1.007× |

**Read the $\beta_2$ column.** At $t=1$, $v$ is **1000× too small**, so $\sqrt{\hat v}$ needs a 31.6× correction. **Without bias correction Adam's first step is enormous.** And $\beta_2 = 0.999$ has a memory of $1/(1-0.999) = 1000$ steps — meaning **Adam's second-moment estimate is not trustworthy until roughly step 1000.** Circle that. **It is the mechanistic reason warmup exists (§8), and it lands with force because the learner just computed it.**

### 7.5 AdamW — and why decoupling matters

**Adam + L2** (the wrong thing) folds the penalty into the gradient, $g_t \leftarrow g_t + \lambda\theta_t$, so the penalty then gets **divided by $\sqrt{\hat v_t}$**. Result: parameters with large historical gradients get *less* regularized than parameters with small ones. **That is backwards** — it's arbitrary, coupled to gradient history, and it's not L2 regularization at all anymore.

**AdamW** applies decay directly to the weights, outside the adaptive machinery:

$$
\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\, \theta_t\right)
$$

so every parameter shrinks by the same relative amount $\eta\lambda$ per step, independent of its gradient history. **That's the whole fix, and it's why AdamW is the 2026 default and plain Adam is essentially never used for serious training.** Note `torch.optim.Adam(weight_decay=...)` is the *coupled* (wrong-ish) version — **the flag name is the same, the semantics differ, and this trips up people reading two codebases.** Warning box.

⚠️ **Naming trap:** PyTorch's AdamW multiplies decay by $\eta$ (as above), so **changing your LR silently changes your effective weight decay.** Under a cosine schedule, your regularization decays along with your LR. Some implementations (and the original paper's "fully decoupled" variant) don't do this. `[CONF: high on PyTorch's behavior; verify the exact 2.10 formula against docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html when writing the spec.]`

### 7.6 ⭐ OPTIMIZER MEMORY — the number that makes LoRA inevitable

> **⚠️ RE-ANCHOR ON Qwen3-8B, all-linear (D-01/D-03/D-05/D-06). The 7B / Llama-2 / attention-only numbers below
> are RETIRED — use `constants.md` §2–§3:** full FT state **131.05 GB / 122.05 GiB** (not ~130); LoRA r=16
> **all-linear** = **43,646,976 params → 17.08 GB total** (not attn-only 16,777,216 → 14.3 GB); optimizer-state
> shrink **187×** = the exact param ratio (not 224×, which mismatched numerator/denominator). Headline pair:
> **131.05 GB → 17.08 GB, 7.67×, state-to-state** (the "130 GB → 14.3 GB" pair credited LoRA with an activation
> saving it does NOT deliver — LoRA freezes weights but still backprops the full graph, so activations are
> unchanged; D-03). The `.detach()`/`b/g/master` structure and the "optimizer-state compression, not model
> compression" punchline all transfer verbatim; only the decimals change.

**This is the highest-value table in the brief for this specific learner. It is the bridge from the trunk to the fine-tuning payoff.**

Per trainable parameter, in a standard bf16 mixed-precision setup:

| item | bytes/param | why |
|---|---|---|
| weights (bf16) | 2 | what the GPU computes with |
| gradients (bf16) | 2 | one per weight, same shape |
| **fp32 master weights** | 4 | bf16 has only 8 mantissa bits; a tiny update to a big weight rounds to *nothing* |
| **Adam $m$ (fp32)** | 4 | ← optimizer state |
| **Adam $v$ (fp32)** | 4 | ← optimizer state |
| **total** | **16** | |

**Adam optimizer states alone = 8 bytes/param = 4× the size of the bf16 model.** The "Adam states = 2× params" shorthand refers to *two extra full-size tensors*; in bytes against a bf16 model it's 4×. **Be precise about which you mean — this is a place courses are routinely sloppy.**

**For a 7B model (verified arithmetic, GB = $10^9$ bytes):**

| component | GB | GiB |
|---|---|---|
| bf16 weights | 14.0 | 13.0 |
| bf16 grads | 14.0 | 13.0 |
| fp32 master | 28.0 | 26.1 |
| Adam $m$ + $v$ | **56.0** | **52.2** |
| **subtotal (16 B/param)** | **112.0** | **104.3** |
| + activations (seq 2048, bs 8, no checkpointing) | ~10–30 | |
| **total full fine-tune** | **~125–145** | |

**Inference on the same model: 16.38 GB. Training: 131.05 GB. An 8× multiplier — and 43% of it is Adam's $m$ and $v$** (65.5/131.05; ~~"40%"~~ was slightly off, D-20). ← *This* is the sentence that makes the whole fine-tuning track make sense. **⚠️ The learner OWNS the box — do NOT assert "over his 128 GB edge." The state is 122.05 GiB, which FITS in 128 GiB (the "3 GB over" was a GB/GiB artifact, D-04). Have him MEASURE usable memory (`torch.cuda.mem_get_info()`) and compute 122.05 GiB against *that*.** That measured moment is the one that becomes personal.

**Now LoRA, in the same units (verified):** LoRA rank $r=16$ on **all linear layers** (D-06; attention-only is the retired example), **Qwen3-8B** geometry ($d=4096$, 36 layers, 7 linears/block):

$$
\text{params} = 43{,}646{,}976 = \mathbf{0.533\%}\text{ of } 8.19\text{B} \quad(\text{constants §3; ~~attn-only } 16{,}777{,}216 = 0.24\%~~ \text{ retired})
$$

Trainable state (grads+master+m+v) for 43.6M params ≈ **0.61 GB.** Frozen base in bf16 = **16.38 GB.** LoRA total ≈ **17.08 GB**.

> **Full fine-tune: 131.05 GB. LoRA (r=16, all-linear): 17.08 GB. The base model didn't shrink — the *optimizer
> state* did, by 187×** (= 8,190,735,360 / 43,646,976 exactly, because optimizer state is linear in trainable
> params). ~~~130 / ~14.3 GB / 224×~~ retired (D-03/D-05).

**LoRA is not a model compression technique. LoRA is an optimizer-state compression technique.** ← If the trunk establishes exactly one bridge to the fine-tuning track, make it this sentence. It is *earned* by §7.6, and it will not land without these numbers. `[CONF: high on arithmetic; the "~0.25 GB" figure matches published LoRA memory accountings.]`

### 7.7 2026 default hyperparameters — what people actually use

`[CONF: high on the LLM row (this configuration is near-universal across 2025–2026 pretraining papers); med on the diffusion row (more variance in practice).]`

| setting | LLM pretrain/finetune | Diffusion (UNet/DiT) | small MLP/CNN | why |
|---|---|---|---|---|
| optimizer | **AdamW** | AdamW | AdamW or SGD+mom | — |
| $\beta_1$ | 0.9 | 0.9 | 0.9 | ~10-step gradient memory |
| $\beta_2$ | **0.95** | 0.999 | 0.999 | **LLMs use 0.95 (≈20-step memory) not 0.999 (≈1000): text has huge gradient spikes from rare tokens, and a 1000-step memory takes forever to forget a spike, poisoning the denominator. Shorter memory = faster recovery.** This is a real, motivated deviation and the course should explain it rather than list it. |
| $\epsilon$ | 1e-8 | 1e-8 | 1e-8 | step-size floor; some use 1e-15 for very small updates |
| weight decay | **0.1** | 0.01 | 0.01–0.05 | LLMs use aggressively high WD |
| peak LR | 3e-4 (~1B) → **1e-4** (7B) → 1.5e-4..3e-4 varies | 1e-4 | 1e-3 | **LR shrinks as models grow — roughly $\propto 1/\sqrt{\text{width}}$, which is what μP formalizes** |
| LoRA LR | **1e-4 – 3e-4** (often 10× the full-FT LR) | 1e-4 | — | LoRA's B init is 0 so the adapter starts as a no-op and tolerates/needs a bigger LR |
| grad clip | **1.0** | 1.0 | 1.0 or off | near-universal |
| warmup | 1–10% of steps (or a flat 2000 steps) | ~1000 steps | often none | see §8 |
| schedule | cosine to $\eta_{\max}/10$ or $/20$ | cosine or constant | cosine | |
| batch (tokens) | 1M–4M tokens | 256–2048 images | 32–256 | |

**No-decay list — a real practice worth teaching:** biases, LayerNorm/RMSNorm gains, and (usually) embeddings are **excluded from weight decay**. Reason: decaying a bias just shifts the function with no capacity benefit; decaying a norm gain toward 0 actively fights the normalization. Every serious codebase has this param-group split and beginners' code never does. It's a great "read a real training script" exercise.

### 7.8 2026 frontier: Muon (flag honestly)

**Muon** (momentum orthogonalized via Newton–Schulz) is the first credible challenger to AdamW's decade-long monopoly, and by mid-2026 it has moved from curiosity to production. Verified claims:
- **Memory:** stores **one** momentum buffer, not two ⇒ **~50% less optimizer state than Adam** (~14 GB saved on a 7B in fp32).
- **Efficiency:** ~**52% of AdamW's training FLOPs** for comparable performance (Moonshot's Moonlight report, arXiv:2502.16982).
- **Throughput at scale (2026):** Kimi K2 — 1,080 TFLOP/s/GPU with Muon vs 1,051 with AdamW; Qwen3-30B — 721 vs 713. NVIDIA integrated Muon into Megatron (blog dated 2026-04-22) reporting near-parity throughput with better model quality.
- Applies only to **2-D (matrix) parameters**; embeddings, biases, and norm gains still use AdamW. Every real "Muon" run is a hybrid. Say so — this is routinely misreported.

**How to teach it:** as a **sidebar, not a main path.** The learner will use AdamW for his fine-tuning. But mentioning Muon (a) makes the course visibly 2026-current, (b) shows optimization is *not settled science*, and (c) reinforces §7.6 (its main selling point is memory — the same lens as LoRA). **Genuine open question to state:** whether Muon's advantage survives at every scale and whether Adam-pretrained models can be Muon-fine-tuned is actively contested (see arXiv:2605.10468, "Can Muon Fine-tune Adam-Pretrained Models?"). `[CONF: med-high on the cited numbers — they're from vendor/lab reports, not independent replications. Present as "reported," with attribution.]`

### 7.9 Misconceptions

| Misconception | Correction |
|---|---|
| "Adam is strictly better than SGD." | Well-tuned SGD+momentum still **generalizes better on vision** (many ResNet SOTA results are SGD). Adam is better on transformers, and better *untuned*. Adam's real superpower is that it works at the default. |
| "Adam adapts the learning rate." | Adam adapts the **per-parameter scaling of the gradient**. Your $\eta$ is untouched — you still need a schedule. "Adaptive" ≠ "you don't need a schedule." This one is pervasive. |
| "Momentum just makes it faster." | It changes the *dynamics*: it damps oscillation across a ravine while accumulating along it. It's a **conditioning** fix, not a speed hack. |
| "Weight decay and L2 are the same." | They are — **for SGD.** For Adam they diverge, and that's the entire reason AdamW exists. |
| "Optimizer state is a minor overhead." | It's **4× your bf16 model.** Bigger than the model. Bigger than the gradients. §7.6. |
| "$\epsilon$ is just to avoid divide-by-zero." | It sets the maximum step size ($\eta/\epsilon$) and it's a real, tuned hyperparameter in low-precision regimes. |

---

## 8. LEARNING-RATE SCHEDULES

### 8.1 Intuition first
> "Big steps to get to the right neighborhood; small steps to find the right house. Annealing, exactly like metallurgy: high temperature to explore, cooled slowly to settle into a good crystal."

### 8.2 The standard 2026 schedule (this exact one — reuse it)

**Linear warmup → cosine decay.**

$$
\eta_t = \begin{cases}
\eta_{\max}\cdot \dfrac{t}{T_{\text{warm}}} & t \le T_{\text{warm}} \\[2ex]
\eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\left(1 + \cos\!\left(\pi\,\dfrac{t - T_{\text{warm}}}{T_{\text{total}} - T_{\text{warm}}}\right)\right) & t > T_{\text{warm}}
\end{cases}
$$

| symbol | typical (7B pretrain) | note |
|---|---|---|
| $\eta_{\max}$ | 1e-4 | |
| $\eta_{\min}$ | $\eta_{\max}/10$ or $/20$ = 1e-5 / 5e-6 | **not zero** — a zero final LR wastes the last steps |
| $T_{\text{warm}}$ | 2000 steps, or 1–10% of total | |
| $T_{\text{total}}$ | fixed in advance | **← the cosine's fatal flaw** |
| $t$ | **steps**, not epochs | §4.4 |

**Worked example — make the learner compute one value:** $\eta_{\max}=1\text{e-}4$, $\eta_{\min}=1\text{e-}5$, $T_{\text{warm}}=2000$, $T_{\text{total}}=100{,}000$. At $t = 51{,}000$: progress $= 49000/98000 = 0.5$, $\cos(\pi/2)=0$, so $\eta = 1\text{e-}5 + 0.5(9\text{e-}5)(1+0) = \mathbf{5.5\times10^{-5}}$. **Halfway through training, the LR is at its midpoint** — that's the property cosine has and step-decay doesn't, and it's why the curve looks the way it does.

**Cosine's real flaw, stated honestly:** you must commit to $T_{\text{total}}$ up front. Stop early and you're at a high LR with a badly-converged model; continue past it and $\eta$ goes negative (or you clamp and lose the schedule). **This is why 2024–2026 saw a shift toward WSD (Warmup–Stable–Decay / trapezoidal): warm up, hold constant, then decay sharply over the last ~10%. You can branch a checkpoint at any point and anneal it, so it's continuation-friendly and scaling-law-friendly.** MiniCPM and several 2025–2026 runs use it. `[CONF: med-high on WSD's adoption breadth; high on its existence and motivation.]` **Flag as a live methodological disagreement** — cosine is still the default, WSD is gaining, and there's no consensus.

### 8.3 ⭐ Why warmup exists — three reasons, and the course should give all three

**This is a great section because the learner already has the ammunition from §7.4.**

1. **Adam's second moment is garbage at $t < \sim1000$.** From §7.4's table: at $t=1$, $v$ is 1000× too small; even at $t=100$ the correction is still 10.5×. The bias correction fixes the *mean* but not the *variance* — $\hat v$ is estimated from a handful of samples, so $1/\sqrt{\hat v}$ has huge variance early. The RAdam paper (arXiv:1908.03265) showed the variance of the adaptive term is unbounded in the first steps and argued **warmup is literally a variance-reduction technique.** RAdam's whole thesis: rectify that variance and you don't need warmup. **This is the strongest and most mechanistic explanation. Lead with it.**
2. **A bad early step poisons Adam for ~1000 steps.** One large early gradient enters $v$, and with $\beta_2=0.999$ it takes ~1000 steps to decay out. So the damage isn't one bad step — it's one bad step *plus a thousand steps of a corrupted denominator*. **Warmup keeps the early steps small enough that nothing terrible gets into the running statistics.** ← This is the most *memorable* framing; the learner has computed the 1000-step memory himself.
3. **Interaction with LayerNorm + large batch.** Post-LN transformers have large gradients at the output layers at init; combining LN with adaptive optimizers produces early instability that warmup resolves. (This is the Transformer-specific part, and it's exactly why pre-LN — §11.5 — reduces but does not eliminate warmup's necessity.)

**The honest state of 2026:** warmup is a *patch*, and the field knows it. RAdam, T-Fixup, GradInit, and better init all reduce or remove the need for it, and there's recent work analyzing/reducing warmup for GPT training (arXiv:2410.23922). **But everyone still uses it, because it costs nothing and it works.** *That* sentence — "we do it because it's cheap insurance against a failure we understand but haven't cleanly fixed" — is a valuable and honest thing for a learner to hear about how the field actually operates.

### 8.4 Misconceptions

| Misconception | Correction |
|---|---|
| "Warmup is because the model is 'fragile' at the start." | It's specifically because **Adam's variance estimate** is unreliable at the start. Pure SGD needs warmup far less. Tie it to the optimizer, not to mysticism. |
| "Warmup and warm restarts are related." | Unrelated. Warmup = the linear ramp at step 0. Warm **restarts** (SGDR) = re-raising the LR periodically. Confusing names, confusing them is common. |
| "Adam is adaptive so it doesn't need a schedule." | See §7.9. Adam adapts *direction scaling*, not $\eta$. |
| "Schedule by epoch." | Schedule by **step**. In LLM training you often do <1 epoch, and "epoch" is meaningless on a token stream. |
| "Decay to zero." | Decay to $\eta_{\max}/10$–$/20$. Zero LR = zero learning for the tail of the run. |

---

## 9. INITIALIZATION

### 9.1 Intuition first
> "You are setting the volume knob on a 100-stage amplifier chain. If each stage has gain 1.1, the signal is amplified $1.1^{100} = 13{,}781\times$ — it clips. If each stage has gain 0.9, the signal is $0.9^{100} = 2.7\times10^{-5}$ — it's gone. **You need gain ≈ 1.000 at every stage, and you need it for the signal going forward AND the blame going backward.** Init is how you set that gain before you've seen any data."

**Compute $1.1^{100}$ and $0.9^{100}$ on the page.** Those two numbers do more work than a page of prose, and they set up §10 (vanishing/exploding is the *same* geometric-product argument, applied to the backward pass) for free.

### 9.2 The variance-propagation derivation

For $z_j = \sum_{i=1}^{n_{\text{in}}} w_{ji} x_i$ with $w$ i.i.d. zero-mean, $x$ i.i.d. zero-mean, and $w \perp x$:

$$
\text{Var}(z_j) = \sum_{i=1}^{n_{\text{in}}}\text{Var}(w_{ji})\,\text{Var}(x_i) = n_{\text{in}}\,\text{Var}(w)\,\text{Var}(x)
$$

**To hold $\text{Var}(z) = \text{Var}(x)$ we need $\text{Var}(w) = 1/n_{\text{in}}$.** That's the entire derivation. It's three lines and it's the whole of Xavier/He.

- **Xavier/Glorot** (for tanh/sigmoid, roughly linear near 0): compromise between forward ($1/n_{\text{in}}$) and backward ($1/n_{\text{out}}$): $\text{Var}(w) = \dfrac{2}{n_{\text{in}}+n_{\text{out}}}$
- **He/Kaiming** (for ReLU): ReLU zeroes half the inputs, halving the output variance. Compensate by doubling: $\text{Var}(w) = \dfrac{2}{n_{\text{in}}}$

**The "2" in He init is literally "because ReLU throws away half your signal."** That's the whole story and it's satisfying.

**Worked number:** a layer with $n_{\text{in}} = 4096$ (a transformer's $d_{\text{model}}$). He: $\text{Var}(w) = 2/4096 = 4.88\times10^{-4}$, so $\text{std}(w) = 0.0221$. **Every weight in an LLM starts as a Gaussian with std ≈ 0.02.** Sanity-check that against what he's seen: yes — GPT-2's init is $\mathcal{N}(0, 0.02^2)$, essentially this number. **That's a lovely "the theory predicts the config file" moment; use it.**

**LLM-specific init to mention (it's a nice depth-scaling instance):** GPT-2 and descendants scale residual-projection weights by $1/\sqrt{2L}$ ($L$ = layer count) so that the *accumulated* residual-stream variance doesn't grow with depth. Same geometric-product logic, applied to a sum instead of a product.

### 9.3 What a bad init actually does — the four failure modes

| init | what happens | what you SEE |
|---|---|---|
| **all zeros** | **symmetry never breaks.** Every neuron in a layer gets an identical gradient, so they stay identical forever. Your 4096-wide layer is a 1-neuron layer. | loss drops to the "predict the mean/prior" value and flatlines *forever* |
| **too small** (e.g. std 0.001) | signal geometrically vanishes with depth; grads too | loss flat at the prior; deeper = worse; looks like a dead LR |
| **too large** (e.g. std 1.0) | activations blow up; tanh/sigmoid saturate at ±1 where $\sigma' \approx 0$ | loss NaN in a few steps, **or** loss stuck (saturated ⇒ no gradient) — **two opposite symptoms from one cause, which is why this is so confusing** |
| **all same nonzero constant** | same symmetry problem as zeros | flatline |

**Biases CAN be zero** (and are, by default) — symmetry is already broken by the random $W$. Learners frequently over-generalize "never init to zero" to biases. Warning box. (Historical exception: ReLU biases were sometimes set to 0.01 to avoid dead units at init; largely abandoned.)

### 9.4 μP — flag it, don't teach it
**Maximal Update Parametrization (μP)** rescales init and per-layer LRs so that **the optimal LR found on a small proxy model transfers to a large model without retuning.** This is how labs tune hyperparameters on a 40M model and apply them to a 40B one, saving enormous compute. `[CONF: med-high — μP/μTransfer is real, published, and used, but adoption breadth across 2026 frontier labs is not something I can verify.]` **Mention in one paragraph as "why the LR-shrinks-with-width rule in §7.7 has actual theory behind it." Do not derive it.**

### 9.5 Misconceptions

| Misconception | Correction |
|---|---|
| "Init just needs to be random." | It needs a **specific variance**, tied to fan-in and to the activation function. Random-but-wrong-scale fails as badly as zeros, just less obviously. |
| "With BatchNorm/LayerNorm, init doesn't matter." | Norm layers make init *much* more forgiving — this is true and is a large part of why they were adopted. But init still sets the effective LR and the early dynamics, and *deep* nets still need scaled residual init. "Doesn't matter" is too strong. |
| "Init is a solved detail." | It's the thing that decides whether a 100-layer net trains at all, and $1/\sqrt{2L}$-style tricks are still being invented. |
| "He is for everything." | He assumes ReLU. Using He with tanh over-scales by $\sqrt2$ per layer; over 50 layers that's $2^{25}$. Match the init to the nonlinearity. |

---

## 10. VANISHING / EXPLODING GRADIENTS AND CLIPPING

### 10.1 Intuition first
> "**It's the amplifier chain from §9, but running backwards.** Blame gets multiplied by a factor at every layer on its way back. Multiply by 0.8 fifty times and the blame arriving at layer 1 is $0.8^{50} = 1.4\times10^{-5}$ of what left layer 50 — layer 1 learns nothing. Multiply by 1.2 fifty times and it's $9{,}100\times$ — layer 1 gets shredded."

**The learner already met this in §5.4** when we asked "what if $a^{(1)}_1$ had been 0.99, so $1-a^2 = 0.0199$?" **Call that back explicitly.** He found the mechanism himself in a 9-parameter network; now we're just naming it and iterating it 50 times.

### 10.2 The math

Unrolling BP2 across layers:

$$
\delta^{(1)} = \left(\prod_{l=2}^{L} \left(W^{(l)}\right)^{\!\top} \text{diag}\!\left(\sigma'(z^{(l-1)})\right)\right) \delta^{(L)}
$$

**The gradient at layer 1 is a product of $L-1$ matrices.** Products of many matrices are governed by their singular values: if the typical $\|W^{(l)}\|\cdot|\sigma'| < 1$, the product $\to 0$ **geometrically**; if $>1$, $\to \infty$ **geometrically**. Geometric, not linear. **There is no "mild" version of this at depth 50** — that's the point.

**Why sigmoid is the historic villain, with numbers:** $\max_z \sigma'(z) = \sigma'(0) = 0.25$. **The very best case is a 4× attenuation per layer.** So even with perfectly scaled weights, a 10-layer sigmoid net attenuates gradients by at least $0.25^{10} = 9.5\times10^{-7}$. **Sigmoid nets deeper than ~5 layers cannot be trained by backprop. That's not a tuning problem — it's arithmetic.** Compute $0.25^{10}$ on the page. `tanh'(0) = 1.0`, which is exactly why tanh beat sigmoid, and ReLU's derivative is exactly 1 on the positive side, which is why ReLU beat tanh. **The entire history of activation functions is this one number.** That is a genuinely great paragraph and the course should write it.

### 10.3 The fixes, in historical order (each is a whole architecture idea)
1. **Better activations** — ReLU: $\sigma' \in \{0, 1\}$, no attenuation when active.
2. **Better init** — §9, set the per-layer gain to 1.
3. **Normalization** — §11, forcibly reset the scale every layer.
4. ⭐ **Residual connections** — $a^{(l+1)} = a^{(l)} + F(a^{(l)})$ ⇒ $\frac{\partial a^{(l+1)}}{\partial a^{(l)}} = I + \frac{\partial F}{\partial a^{(l)}}$. **The identity term guarantees a gradient path of gain exactly 1, all the way down, no matter what $F$ does.** The product-of-matrices becomes a product of (I + small), which is $\approx$ 1 + sum of smalls, not a geometric decay. **This is why 100-layer networks exist. It is arguably the single most important architectural idea in deep learning, and it is a two-line consequence of backprop.** ← This must be in the trunk, because both tracks (transformers *and* diffusion UNets/DiTs) are built entirely on residual streams. **Handoff flag: coordinate with the architecture brief so this isn't taught twice or, worse, zero times.**
5. **Gradient clipping** — §10.4, the blunt instrument for the exploding side.

**Asymmetry to state:** clipping fixes *exploding* completely and cheaply. **Nothing fixes vanishing cheaply** — vanishing requires architecture (residuals). That asymmetry explains why clipping is a one-liner everyone uses and residual connections are a Turing-award-adjacent idea.

### 10.4 Gradient clipping

**Clip by global norm** (the one everyone uses):

$$
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\|g\|_2}\right), \qquad \|g\|_2 = \sqrt{\sum_{k=1}^{P} g_k^2}
$$

$c = 1.0$ is the near-universal 2026 default across LLM and diffusion training.

**Two properties worth stating precisely:**
- **It preserves direction, only shrinks magnitude.** It is not clamping individual values. Contrast `clip_grad_value_`, which clips elementwise and **does** change the direction — it's a different (and worse-behaved) operation with a confusingly similar name. Warning box.
- **The norm is over ALL parameters concatenated**, not per-tensor. So one exploding layer throttles the whole model's step. That's intended: a step where one layer is insane is a step you don't want to take fully.

**Diagnostic gold — teach this as the primary instrument:** **log $\|g\|_2$ every step.** It is a far more informative signal than the loss, and it's the one thing beginners never plot.
- Healthy: fluctuates in a stable band (often ~0.1–1.0 for a converged LLM run), slowly declining.
- **Spikes to 10–100:** you're hitting bad batches. Clipping is earning its keep. If they're frequent, suspect data (a corrupted example, a duplicated document, a tokenizer bug).
- **Grows monotonically:** you're diverging. Loss is about to follow. **Grad norm leads loss.** You can see divergence coming ~50–200 steps before the loss shows it.
- **→ 0:** vanishing, dead ReLUs, or a detached graph (§6.6).

**Ordering bug — real, common, silent:** clipping must happen **after `.backward()`** and **before `.step()`**. Put it after `.step()` and it does literally nothing (you clip gradients that have already been consumed) — **no error, no warning, just a config that "has clipping" and doesn't.** With AMP you must additionally `scaler.unscale_(optimizer)` first, or you're clipping the *scaled* gradient and your threshold of 1.0 is really 65536.0. **These two bugs are a warning box each.**

### 10.5 Misconceptions

| Misconception | Correction |
|---|---|
| "Clipping fixes exploding gradients." | It fixes the *symptom*, per step. If your grad norm is 1000 every step, clipping means you're taking a normalized-random-direction step. **You have a bug, and clipping is hiding it.** Clipping is for rare spikes, not chronic explosion. |
| "Vanishing and exploding are opposite problems with opposite fixes." | Same mechanism (geometric products), opposite sign of the exponent. Both are cured by controlling per-layer gain. |
| "ReLU solved vanishing gradients." | It solved the *saturation* contribution. Deep ReLU nets still vanish through the $W$ product (and die — dead ReLU is vanishing's cousin). **Residuals** solved it. |
| "Clipping changes the direction of the step." | `clip_grad_norm_` doesn't. `clip_grad_value_` does. Different functions. |

---

## 11. NORMALIZATION

### 11.1 Intuition first
> "Every layer is trying to learn a function of its input while its input keeps changing scale underneath it — like tuning a radio while someone randomly turns the volume knob. Normalization nails the volume knob to a fixed setting, so each layer can learn *shape* instead of chasing *scale*."

### 11.2 The three, precisely — with the axis question front and center

**The ONLY thing that distinguishes them is: which axis do you compute the statistics over?** Everything else is identical. Say that first; then the table is trivial instead of three separate memorizations.

Given a batch of activations $x \in \mathbb{R}^{B \times T \times d}$ (batch, time/tokens, features):

**BatchNorm** — normalize each feature across the **batch** (and time):
$$
\mu_j = \frac{1}{BT}\sum_{b,t} x_{btj}, \quad \sigma_j^2 = \frac{1}{BT}\sum_{b,t}(x_{btj}-\mu_j)^2, \quad \hat x_{btj} = \gamma_j \frac{x_{btj}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}} + \beta_j
$$
Learnable: $\gamma, \beta \in \mathbb{R}^{d}$. **Reduction over $(b,t)$.**

**LayerNorm** — normalize each token across its **features**:
$$
\mu_{bt} = \frac{1}{d}\sum_{j} x_{btj}, \quad \sigma_{bt}^2 = \frac{1}{d}\sum_j (x_{btj}-\mu_{bt})^2, \quad \hat x_{btj} = \gamma_j\frac{x_{btj}-\mu_{bt}}{\sqrt{\sigma_{bt}^2+\epsilon}} + \beta_j
$$
**Reduction over $j$.** **No dependence on other examples in the batch — that's the whole point.**

**RMSNorm** — LayerNorm minus the mean-centering:
$$
\hat x_{btj} = \gamma_j \frac{x_{btj}}{\text{RMS}(x_{bt})}, \qquad \text{RMS}(x_{bt}) = \sqrt{\frac{1}{d}\sum_{j=1}^{d}x_{btj}^2 + \epsilon}
$$
**No $\mu$, no $\beta$.** Empirically the re-centering does nothing useful; only the re-scaling matters. **~7–64% faster than LayerNorm** depending on shape `[CONF: med — the RMSNorm paper reports 7–64% run-time reduction; real speedup is hardware- and fusion-dependent]`, and it saves $d$ parameters per norm layer.

### 11.3 The 2026 status — say it plainly
**RMSNorm + pre-norm is the standard for LLMs in 2026** (Llama family, Qwen family, Mistral, and essentially everything descended from them). BatchNorm is **for CNNs, and essentially only for CNNs.** `[CONF: high]`

### 11.4 Why BatchNorm lost — four concrete reasons
1. **It makes the loss depend on other examples in the batch.** Prediction for sample $i$ changes if you change sample $j$. Philosophically ugly; practically it means train ≠ eval behavior.
2. **Train/eval mismatch.** Training uses batch stats; eval uses a running average. **Forgetting `model.eval()` is the classic BatchNorm bug: your metrics change depending on your eval batch size.** This is a genuinely great bug to demonstrate — it's silent and confusing.
3. **It breaks at small batch.** At batch 2, $\sigma^2$ is estimated from 2 samples. Garbage. And large-model training *is* small-per-device-batch training (§4.4 gradient accumulation).
4. **It's awkward for variable-length sequences.** Padding tokens pollute the batch statistics.

**LayerNorm has none of these.** It's per-token, so batch size is irrelevant, and train == eval **exactly** (no running stats, no `model.eval()` behavior change). **That last property is why transformers use it.**

### 11.5 ⭐ Pre-norm vs post-norm — the diagram that matters

$$
\textbf{Post-LN (original, 2017):}\quad x_{l+1} = \text{LN}\big(x_l + F(x_l)\big)
$$
$$
\textbf{Pre-LN (everything since ~2020):}\quad x_{l+1} = x_l + F\big(\text{LN}(x_l)\big)
$$

**The difference is one bracket and it decides whether your 80-layer model trains.**

**In pre-norm, there is a clean identity path from the loss to layer 0 with NO normalization on it.** Look at the equation: $x_{l+1} = x_l + (\text{stuff})$. Differentiate: $\partial x_{l+1}/\partial x_l = I + \partial(\text{stuff})/\partial x_l$. **The gradient at layer 0 gets a term of exactly 1.** In post-norm, the LN is *on* the residual path, so the gradient passes through $L$ normalization Jacobians on the way down and gets rescaled every time — the geometric-product problem returns.

**Consequences, stated as trade-offs (this is a real trade-off, not a strict win):**
- **Pre-norm:** trains stably at depth, **needs much less warmup, tolerates larger LR.** Cost: the residual stream's variance **grows with depth** (you keep adding to it, un-normalized), so the last layers contribute proportionally less — an effective-depth loss. Needs a **final LN** before the LM head to tame the accumulated stream.
- **Post-norm:** slightly better final quality when it trains; **needs careful warmup** and is fragile at depth.
- **2026 status:** pre-norm dominates. But **flag the genuine ongoing debate** — Mix-LN (arXiv:2412.13795) argues pre-LN makes *deep layers* under-contribute and proposes combining post-LN in early layers with pre-LN in later ones; some 2025+ models use hybrid/sandwich norms (e.g. norms on both sides of each sublayer). **This is not settled.** `[CONF: med-high]`

**This section is a direct dependency of the transformer-architecture brief. Coordinate.** The trunk should own *why normalization exists and why pre- beats post-*; the architecture brief should own *where the norms sit in a block diagram*.

### 11.6 Misconceptions

| Misconception | Correction |
|---|---|
| "BatchNorm works by reducing internal covariate shift." | **The original explanation is now widely regarded as wrong.** Santurkar et al. (2018) showed you can *inject* covariate shift after BN and it still helps. The better-supported story: **it smooths the loss landscape** (reduces the Lipschitz constant of the gradient), which permits larger LRs. **Teach the correction, and teach that the field ran with a wrong explanation for years. That is an honest and valuable lesson about how ML actually works.** |
| "LayerNorm normalizes over the batch." | It normalizes over **features**, within one token. The names are genuinely confusing; the axis diagram is the fix. Draw it. |
| "Normalization prevents overfitting." | BN has a *mild* regularizing side-effect (batch-statistic noise). LN/RMSNorm have **none** — no batch, no noise. Norms are an **optimization** tool, not a regularization tool. |
| "RMSNorm is an approximation to LayerNorm." | It's a *different* layer that works as well or better. The mean-subtraction was never doing much. Not an approximation — a simplification that lost nothing. |
| "Norm layers have no parameters." | $\gamma$ (and $\beta$ for LN/BN) are learned — $d$ or $2d$ params per layer. **And they're on the no-decay list (§7.7).** |

---

## 12. REGULARIZATION, AND TRAIN/VAL/TEST DISCIPLINE

### 12.1 Intuition first
> "A student who memorizes the answer key aces the practice exam and fails the real one. Overfitting is memorizing the answer key. Every regularizer is a different way of making memorization *harder than* understanding."

### 12.2 The two curves — the diagnostic picture the course must burn in

Plot **train loss and val loss on the same axes, versus step.** This one plot is the diagnostic instrument for everything below.

| shape | diagnosis | fix |
|---|---|---|
| both high, both flat | **underfitting** (or a bug) | bigger model, higher LR, train longer, check for §14 bugs |
| both decreasing together | **healthy** | keep going |
| train ↓, val ↑ (diverging fork) | **overfitting** | more data, augmentation, dropout, WD, early stop |
| train ↓ fast to ~0, val ↑ immediately | **memorizing** — model ≫ data | far more data, far more regularization, or a much smaller model |
| val < train | **not a bug** — usually dropout (off at eval) or an easier val split | verify the split isn't leaking |
| both spiky | LR too high, batch too small, or bad data | §14 |

**Train/val/test discipline — the part that's about integrity, not math:**
- **train** — gradients come from here.
- **val** — you look at it and make decisions (which model, when to stop, what LR). **Every time you look, you leak a bit of information into your model selection.**
- **test** — you look **once**, at the end, and you report that number. **If you tune on test, you don't have a test set; you have a second val set and a paper you should retract.**
- Split **before** any preprocessing. Fitting a normalizer/tokenizer/PCA on the full dataset then splitting = **leakage** = optimistic results = a model that fails in production. This is the #1 real-world ML failure and it has nothing to do with neural networks.
- **Split by the right unit.** Random row-splitting time-series or splitting by *image* when you have multiple images *per patient* leaks. **This is the bug that ends careers, and it's four lines of code.**

**Relevance for this learner:** he's fine-tuning on his own dataset. He will have a few hundred to a few thousand examples. **He is in the most overfitting-prone regime that exists, and the course must say so directly.** The val-set discipline section should be written *for* someone with 500 examples, not for someone with ImageNet.

### 12.3 The regularizers

**Dropout** — during training, zero each activation independently with probability $p$; scale survivors by $1/(1-p)$ ("inverted dropout") so $\mathbb{E}[\text{output}]$ is unchanged and **eval needs no rescaling** (that's why it's done this way). At eval, dropout is **off** — this is what `model.eval()` does.
- **Intuition:** "You can't rely on any one neuron, because it might not show up. So build redundant, distributed features." Also: it's an implicit ensemble over $2^n$ subnetworks sharing weights.
- $p = 0.1$ for transformers, $0.5$ for old-style dense layers.
- ⚠️ **2026 reality check the course must state:** **large LLM pretraining runs typically use dropout = 0.** When you train on trillions of tokens for <1 epoch, you *cannot* overfit — every example is new — so dropout is pure harm. **Dropout comes back for fine-tuning on small data**, which is exactly this learner's situation. **So: "dropout is dead" is wrong, "dropout is for the data-poor regime" is right.** `[CONF: high]`

**Weight decay** — §7.5. Intuition: "small weights = a simpler, smoother function; a smooth function can't wiggle to hit every noisy point." Use **AdamW's** decoupled version. $\lambda = 0.1$ for LLMs, 0.01 elsewhere. Exclude biases and norm gains.

**Early stopping** — stop when val loss stops improving (patience 3–10 evals). **The deep point:** early stopping is *itself* a regularizer, and for a quadratic loss it's provably equivalent to L2 with a $\lambda$ that depends on when you stopped. "Train time" is a capacity knob. That equivalence is a genuinely satisfying thing to tell a learner, and it explains why early-stopped runs need less WD.

**Data augmentation** — "if you know a cat is still a cat when mirrored, you just doubled your data for free. **Augmentation is how you inject the invariances you know about into a model that can't be told them directly.**"
- Vision: flips, crops, color jitter, RandAugment, mixup, cutmix.
- **Text: much harder** — there is no "mirror" for a sentence. This is a real asymmetry and the course should own it rather than pretend augmentation is universal.
- **Diffusion-track hook:** for image fine-tuning, horizontal flip is often the *only* safe augmentation (crops change composition, color jitter changes what you're teaching). **Coordinate with the diffusion brief.**

**More data** — the only regularizer with no downside. Say it. Everything else is a trade.

### 12.4 Misconceptions

| Misconception | Correction |
|---|---|
| "Overfitting means the model is too big." | Modern practice says otherwise (§13). Overfitting is a mismatch between capacity, data, **and regularization** — and the usual fix is more data/regularization, **not** a smaller model. |
| "Dropout should always be on." | Zero for large-scale pretraining; 0.05–0.1 for small-data fine-tuning. Regime-dependent. |
| "`model.eval()` is just for logging / disabling gradients." | It **changes the computation**: dropout off, BatchNorm switches to running stats. It does **not** disable gradients — that's `torch.no_grad()`. **Two different functions, universally confused. Warning box.** |
| "Val loss going up means stop." | Val loss can rise while val *accuracy* rises (the model gets more confident and wrong-when-wrong; CE punishes confidence, accuracy doesn't). **Watch the metric you care about**, not the loss you optimize. Sharply relevant to LLM fine-tuning, where val loss and downstream quality routinely disagree. |
| "Regularization makes the model better." | It trades train fit for generalization. On an underfitting model, it makes things **worse**. Diagnose first. |

---

## 13. BIAS–VARIANCE, AND WHY DEEP LEARNING BROKE IT

### 13.1 The classical story (teach it — it's not wrong, it's incomplete)

$$
\underbrace{\mathbb{E}\!\left[(y - \hat f(x))^2\right]}_{\text{expected test MSE}} = \underbrace{\left(\mathbb{E}[\hat f(x)] - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\text{Var}\!\left[\hat f(x)\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}}
$$

Expectation over draws of the training set. **Bias** = "your model class is too rigid to represent the truth" (underfit). **Variance** = "your model chases the noise in whichever training set it happened to get" (overfit). **$\sigma^2$** = "the world is noisy and no model fixes that."

**Intuition:** *Bias = you're consistently wrong the same way (a straight line through a curve). Variance = you're wrong differently every time (a wiggly curve through every point, different wiggles for each dataset).*

**Nice concrete demo:** fit polynomials of degree 1, 3, 15 to 10 noisy points from a sine, resampling the 10 points. Degree 1 is wrong the same way every time (bias). Degree 15 is wildly different every time (variance). Degree 3 is right.

Classical prediction: test error is **U-shaped** in model complexity. Pick the bottom of the U. **This was correct, useful, and taught for 30 years.**

### 13.2 ⭐ Double descent — where it breaks

**The empirical fact:** as you keep increasing model size past the interpolation threshold (where the model can fit the training set exactly, roughly $P \approx N$), **test error goes up (the classical U), peaks at the threshold, and then goes DOWN again — often below the first minimum.** The curve is not a U. It's a U followed by a second descent.

**Three axes it appears along** (Nakkiran et al. 2019 — "Deep Double Descent"):
- **model-wise:** more parameters
- **sample-wise:** ⚠️ **more data can make things WORSE near the threshold** — genuinely counterintuitive; call it out
- **epoch-wise:** train longer and error goes up then down

**The intuition for the second descent:** at exactly $P \approx N$ there is essentially **one** way to fit the data, and the optimizer is forced to take it — noise and all. It's a maximally-constrained, maximally-brittle solution. **Past the threshold there are infinitely many perfect fits, and SGD's implicit bias picks the smoothest / minimum-norm one among them.** *More capacity gives the optimizer freedom to choose a good interpolant instead of being forced into a bad one.* Overparameterization isn't buying complexity — it's buying **choice**.

**This reframes the entire modern era:** "just make it bigger" is not brute force. It's exploiting the second descent. GPT-scale models sit far to the right of the interpolation threshold, where the classical U-curve's predictions are simply inapplicable.

### 13.3 ⭐ THE HONEST PART — this is where the brief earns its keep

**Do not present double descent as settled.** The 2025–2026 state:

1. **Double descent is fragile.** The peak largely **disappears with appropriate regularization** — optimal early stopping and optimal WD flatten it. Nakkiran et al.'s own follow-up work on optimal regularization found this. **So double descent may be, substantially, an artifact of *under-regularized* training.** `[CONF: med-high — this is a well-supported and increasingly mainstream critique]`
2. **The x-axis is the problem.** Double descent is defined with "complexity" = **parameter count**. Parameter count is a *terrible* complexity measure for nonlinear models (you can add parameters that do nothing; you can share parameters; you can reparameterize freely). **Plot against a norm-based or effective-capacity measure and the second descent can vanish.** The phenomenon may be partly an artifact of a bad choice of horizontal axis. `[CONF: med-high]`
3. **It does not show up where it "should."** **Empirical scaling analyses of LLMs and ViTs do not exhibit double descent** — those curves are monotone, clean power laws. If double descent were a fundamental property of overparameterized learning, you'd expect to see it in the most overparameterized models ever built. You don't. `[CONF: med-high]`
4. **Bias-variance is not "dead."** The decomposition is an *identity* — it's algebra, it's always true for squared loss. What's dead is the **claim that variance must increase monotonically with parameter count.** Precision matters here. **"A Farewell to the Bias-Variance Tradeoff?" (arXiv:2109.02355) has a question mark in the title, and the course should honor that question mark.**

**What to actually tell the learner** (this is the recommended framing, and I'd fight for it):

> "The classical U-curve is real, and it's what you'll see when you fit a polynomial or a small MLP to a small dataset — which is a situation you will absolutely be in. Double descent is also real, and it's why the enormous models you use exist at all. **Both are true in their own regimes, and the field does not have one clean theory that covers both.** Anyone who tells you the bias-variance tradeoff is obsolete is overclaiming; anyone who tells you bigger models must overfit hasn't looked at a 2026 model. **Hold both, and know which regime you're in.**"

**The "which regime am I in" heuristic — practical, and what he actually needs:**
- $P \ll N$ (small model, lots of data) → **classical regime.** U-curve. Bias-variance reasoning works. Regularize modestly.
- $P \gg N$ (fine-tuning a 7B on 500 examples — **this is him**) → **overparameterized regime, but with the interpolation threshold RIGHT THERE.** You can hit train loss 0 in minutes. **Regularization (LoRA's low rank is itself a capacity constraint! early stopping! small LR! few epochs!) is doing the heavy lifting.** Double descent will not save you, because you can't grow the dataset or ride the curve. **Early stopping is the tool.**

⭐ **The LoRA connection to make explicit:** **LoRA's rank $r$ IS a capacity knob, and it's the bias-variance tradeoff wearing a 2026 costume.** $r=4$ → high bias, can't learn much, won't overfit. $r=256$ → low bias, high variance, memorizes your 500 examples and forgets everything it knew. **The classical U-curve is alive and well and living in your LoRA config, and the learner will tune this exact knob next week.** ← This is the payoff that connects the most "theoretical" section of the trunk to the most practical thing he does. **Do not lose it.**

---

## 14. PRACTICAL TRAINING DEBUGGING — THE DIAGNOSTIC CHECKLIST

> **Frame it as: "Training failures are not mysterious. There are about twelve of them, they have distinct signatures, and you can learn all twelve."** This section is where the learner stops feeling like training is magic. **It's probably the highest-value section for him personally.** Make it a reference table he'll come back to.

### 14.0 The universal first move
**Overfit a single batch.** Take 8 examples. Turn off dropout, augmentation, weight decay, shuffling. Train on those 8 examples for 200 steps.
- **If loss does not go to ~0, you have a bug. Not a hyperparameter problem. A bug.** No amount of LR tuning fixes it. Stop and find it.
- **If it does go to 0, your forward, loss, backward, and optimizer are all wired correctly.** You've just isolated the problem to data or generalization.

**This one test bisects the entire space of training failures in 60 seconds, and almost nobody does it.** Lead the section with it.

### 14.1 "The loss is NaN"

**First: NaN is not random. It has exactly a few causes.** Also: **check whether it was `inf` first** — `inf - inf = NaN`, `0 × inf = NaN`, `0/0 = NaN`. Usually the disease is overflow and NaN is the corpse.

| cause | how to confirm | fix |
|---|---|---|
| **LR too high** (the #1 cause, by far) | loss rose for a few steps before NaN; grad norm exploded | ÷10 and retry. If 1e-6 still NaNs, it's not the LR. |
| **`log(0)`** in a hand-rolled CE | you wrote `torch.log(softmax(x))` | use `F.cross_entropy` on **logits** (it's the fused, stable version — §2.5) |
| **÷0** | a variance, a norm, a count that hit zero | add $\epsilon$; find the empty thing |
| **fp16 overflow** | fp16 max is **65504**; any activation past it → `inf` → NaN | **use bf16** (range = fp32's, ~$3.4\times10^{38}$); on Ampere+/Hopper/Blackwell there is essentially no reason to use fp16. **The learner's DGX Spark should never see fp16.** |
| **bad data** | NaN appears at the *same step* every run (deterministic!) → it's an example, not the math | **that determinism is the tell.** Find batch $k$, inspect it. `torch.isnan(x).any()` on your inputs. |
| **`zero_grad()` missing** | grads accumulate → step $t$ has ~$t\times$ the gradient → blows up in 20–100 steps | §6.5 |
| **missing/mis-ordered clipping** | grad norm spikes to 1e4 | §10.4, and check the order |

**Tool:** `torch.autograd.set_detect_anomaly(True)` gives you the *forward* op that produced the NaN, with a stack trace to your source line. It is ~10× slower — debug only. **Most people don't know this exists.**

### 14.2 "The loss won't go down"

| cause | signature | fix |
|---|---|---|
| **LR too low** | perfectly smooth, nearly flat, monotone-but-glacial | ×10. Run an **LR range test**: ramp $\eta$ from 1e-7 to 1 over ~100 steps, plot loss vs $\log\eta$; pick ~1 order of magnitude below the minimum. Cheap, 2 minutes, and it turns LR from folklore into measurement. |
| **LR too high** | loss bouncing on a plateau, never settling | ÷10 |
| **you forgot `optimizer.step()`** | loss is **exactly** constant | 🙂 it happens |
| **you forgot `zero_grad()`** | loss rises then NaN | §6.5 |
| **wrong LR / wrong param group** | some layers frozen unintentionally | print `sum(p.numel() for p in model.parameters() if p.requires_grad)` — **for LoRA this should be ~16.8M, not 7B and not 0** |
| **detached graph** | `p.grad is None` or all-zero after `.backward()` | §6.6 — check every param has a nonzero grad |
| **dead ReLUs** | fraction of zero activations climbs toward 1.0 | lower LR, or use GELU/SiLU. **Instrument it: log the zero-fraction per layer.** |
| **labels shuffled/misaligned** | loss sits **exactly** at $\ln K$ (the random-guess value) forever | ⭐ **This is the clean diagnostic: loss == $\ln K$ means the model has learned the class prior and nothing else — i.e., there is NO signal from x to y. Your labels are wrong, or your input is wrong.** For $K=1000$, that's 6.908. **Compute your $\ln K$ and put it on the plot as a horizontal line, always.** |
| **model too small** | train loss plateaus high; it's underfitting | bigger model. Confirm with §14.0. |
| **loss is not connected to the params** | `loss.grad_fn is None` | you detached something |
| **data not normalized** | inputs have std 255 instead of 1 | normalize |

### 14.3 "It overfits instantly"

Train loss → 0 in a few hundred steps, val loss rockets. **For this learner fine-tuning on a few hundred examples, this is THE failure mode he will hit.**

| cause | fix |
|---|---|
| $P \ggg N$ | **lower LoRA rank** (§13.3 — $r$ is a capacity knob), fewer trainable modules |
| too many epochs | **1–3 epochs is typical for fine-tuning, not 50.** This alone fixes most cases. |
| LR too high for fine-tuning | 1e-4–3e-4 for LoRA, and lower for full FT |
| no early stopping | add it; evaluate every ~50 steps, keep the best |
| **data leakage** | check the split by the right unit (§12.2) |
| duplicate examples across the split | dedupe |
| **catastrophic forgetting** (LLM-specific) | train loss down, but the model becomes worse at *everything else* — the val loss on your task looks fine! **This does not show up on the two-curve plot.** Mix in general data, or lower rank/LR. **Genuinely important and easy to miss.** |

### 14.4 The instrumentation panel — what to log, every run, always

**Almost everyone logs only the loss. That's like driving with only a speedometer.** The minimum panel:

1. **train loss** (per step) and **val loss** (per N steps), on the same axes, with $\ln K$ / the baseline drawn as a horizontal reference
2. **gradient global norm** ← the most under-used and most informative signal (§10.4). **Grad norm leads loss.**
3. **learning rate** (actual, from the scheduler — verify your warmup/decay is doing what you think; **plot it**, because scheduler-off-by-one and scheduled-by-epoch-not-step bugs are silent)
4. **weight update ratio:** $\dfrac{\|\Delta\theta\|}{\|\theta\|}$ per layer. **Healthy is ~$10^{-3}$.** If it's $10^{-1}$ your LR is 100× too high; if it's $10^{-6}$ it's 1000× too low. ⭐ **This is the single best-calibrated, most scale-free number in all of training diagnostics, and it's nearly unknown outside a few blog posts. Teach it. It turns "is my LR right?" from vibes into a number with a target.** `[CONF: med — the $10^{-3}$ heuristic traces to Karpathy's CS231n notes; widely repeated, rarely formally validated. Present as a well-tested rule of thumb, not a theorem.]`
5. **fraction of dead units** per layer (for ReLU)
6. **VRAM high-water mark** (`torch.cuda.max_memory_allocated()`) — cross-check against §7.6's prediction. **Predict the number, then measure it.** That closes the loop between theory and his machine.
7. **throughput** (tokens/s or samples/s) — he cares, and it catches silent CPU-bound dataloader stalls
8. **wall-clock per step** — a sudden increase means you started swapping

### 14.5 The determinism trick (worth its own box)
**If the failure happens at the same step every time with a fixed seed, it's DATA. If it happens at a random step, it's NUMERICS or a race.** That single bisection saves hours, costs nothing, and almost nobody thinks of it. Pair it with: set the seed, and `torch.use_deterministic_algorithms(True)` when hunting.

---

## 15. INTERACTIVE DEMOS — CONCRETE SPECS

**Design principle for all of them: the JavaScript must compute the actual math live. No pre-baked animations, no fake curves.** The learner must be able to open devtools and see the gradient being computed. Where possible, **show the numbers, not just the picture** — this learner will trust a number.

### DEMO 1 ⭐ — Gradient descent on a loss surface (LR slider)
**The flagship. Build this first; it carries §3.**

- **Plot:** filled contour map of $L(w_1, w_2)$ over $[-3,3]^2$, plus the optimizer's trajectory as a polyline with dots at each step. Also a small inset: loss vs step.
- **Surface selector (this is what makes it teach):**
  - **(a) Isotropic bowl:** $L = \tfrac12(w_1^2 + w_2^2)$. $\kappa=1$. Straight to the middle. Boring on purpose — it's the control.
  - **(b) ⭐ Ill-conditioned ravine:** $L = \tfrac12(w_1^2 + 20 w_2^2)$. $\lambda_{\max}=20$, $\lambda_{\min}=1$, $\kappa=20$. **This is the money surface.**
  - **(c) Rosenbrock:** $L = (1-w_1)^2 + 100(w_2 - w_1^2)^2$. Curved valley; momentum's showcase.
  - **(d) Saddle:** $L = w_1^2 - w_2^2$. Watch GD stall on the ridge, and watch noise rescue it.
  - **(e) Two-minima:** $L = (w_1^2-1)^2 + w_2^2$ — for exploring init dependence.
- **Sliders:** $\eta \in [0.001, 0.15]$ **log scale**; momentum $\mu \in [0, 0.99]$; gradient noise $\sigma \in [0, 0.5]$; toggle SGD / Momentum / Adam. Click the contour to set $\theta_0$ (do NOT skip this — clicking to set the start is what makes it feel like an instrument).
- **Exact math for the JS** — analytic gradients, no autodiff:
  ```js
  // (b) ravine
  const L    = (w1,w2) => 0.5*(w1*w1 + 20*w2*w2);
  const grad = (w1,w2) => [w1, 20*w2];
  // (c) Rosenbrock
  const Lr    = (w1,w2) => (1-w1)**2 + 100*(w2-w1*w1)**2;
  const gradR = (w1,w2) => [-2*(1-w1) - 400*w1*(w2-w1*w1), 200*(w2-w1*w1)];
  // step (momentum; mu=0 → plain SGD)
  let [g1,g2] = grad(w1,w2);
  g1 += sigma*gaussRandom(); g2 += sigma*gaussRandom();   // SGD noise
  v1 = mu*v1 + g1;  v2 = mu*v2 + g2;
  w1 -= eta*v1;     w2 -= eta*v2;
  ```
- **Live readout (essential):** current $\eta$, $\|\nabla L\|$, $L$, step count, **and a red "DIVERGED" banner when $L > 10^6$.**
- **⭐ The insight, engineered precisely:** on surface (b), $\lambda_{\max}=20$ ⇒ **$\eta_{\text{crit}} = 2/20 = 0.1$ exactly.** Print that on screen. The learner drags $\eta$ up and at **exactly 0.1** it goes unstable. **The theory from §3.4 predicts the demo to two decimal places, and he watches it happen.** Then: at $\eta=0.09$ (just under), it zig-zags violently across the ravine while creeping along it — *he sees the condition number*. Then he turns momentum to 0.9 and the zig-zag cancels and it shoots down the valley — *he sees why momentum exists.* Then he switches to Adam and it walks in nearly a straight line — *he sees why Adam exists.* **Three of this brief's core claims, verified by hand, in ninety seconds.**
- **Also required:** on the saddle (d), set noise = 0 → GD **stops dead** on the ridge, forever. Turn noise to 0.1 → it escapes. **That is §4.3's claim, demonstrated, not asserted.**

### DEMO 2 ⭐ — Backprop calculator (**TN-1**, live — NOT the retired §5.4 network)
**This demo IS the spine section. It must exist.** **⚠️ Load it with TN-1's weights and the load-time assertion
from `constants.md` §8.5/§8.7 (D-02), not §5.4's retired constants.** TN-1 offers both inputs: `[1.0,2.0]` (the
dead-unit case) and `[0.60,−0.20]` (the 10.22× gradient-spread case for the "therefore Adam" beat).

- **Plot:** the 2-2-1 network as an SVG graph. Nodes show $z$ and $a$. Edges show $w$. **Every number is live.**
- **Two-phase mode:** a **Forward** button lights up nodes left→right, printing $z$ then $a$ at each. A **Backward** button lights up right→left in a **different color**, printing $\delta$ at each node and $\partial L/\partial w$ on each edge. **The two-color, two-direction animation is the entire pedagogical point: values flow →, sensitivities flow ←.**
- **Sliders:** all 9 parameters, plus $x_1, x_2$, plus $y \in \{0,1\}$, plus $\eta$. **Init the sliders to the exact §5.4 values so the screen matches the page.** A "Reset to worked example" button is mandatory.
- **JS math:** literally the equations — `tanh`, `sigmoid`, BCE, $\delta^{(2)}=\hat y - y$, $\delta^{(1)}_j = \delta^{(2)}W^{(2)}_j(1-(a^{(1)}_j)^2)$, outer products. ~40 lines. **⚠️ Use TN-1's constants (D-02). Verify at load that it reproduces `constants.md` §8's values — for input `[1.0,2.0]`: $L = 0.9869$, $\delta^{(2)} = -0.6273$; for input `[0.60,−0.20]` (§8.7): $L = 0.6148$, $\delta^{(2)} = -0.4593$.** NOT the retired §5.4 network's 0.453048 / −0.364312.
- **"Take a step" button:** applies SGD, re-runs forward, shows $L$. For input `[0.60,−0.20]`: **0.6148 → 0.5630** ($\Delta L = -0.0518$). For `[1.0,2.0]`: 0.9869 → 0.8222. ~~0.453048 → 0.340179~~ (retired net).
- **⭐ The insight:** drag $W^{(1)}_{11}$ until $a^{(1)}_1 \to 0.99$ (saturate the tanh). **Watch $(1-a^2)$ collapse to 0.02 and watch every gradient in layer 1 collapse with it — in real time, live numbers.** *The learner discovers vanishing gradients by dragging a slider*, before §10 names them. **Then §10 says "remember that thing you saw?" That callback is worth more than any lecture.**
- **Second insight:** an **"Add finite-difference check"** toggle that computes $(L(w+\epsilon)-L(w-\epsilon))/2\epsilon$ live and displays it next to each analytic gradient. **They match to 8 decimals.** *"Backprop is not an approximation"* — proven, on screen, by his own slider.

### DEMO 3 — Live-training tiny MLP in the browser
- **Plot:** left = 2-D scatter of a toy dataset with a **live decision boundary** (evaluate the net on a 60×60 grid, render as a heatmap, ~3600 forward passes/frame — trivially fast in JS). Right = loss curve vs step.
- **Datasets:** two-moons, concentric circles, XOR, spiral, **and a "noisy" variant with 20% flipped labels** ← needed for Demo 5.
- **Controls:** hidden layers (1–3), width (2–16), activation (tanh / ReLU / sigmoid), $\eta$, batch size, optimizer, **init scale** (0.001 / He / 1.0), Play/Pause/Step/Reset.
- **JS:** hand-written forward + backward using §5.3's BP1–BP4 (no library — **the point is that he can read the source and it's the same equations**). Fully connected, ~200 lines.
- **⭐ Insights, each a specific slider action:**
  - **Set init scale to 0.001 → nothing happens, ever.** (§9.3)
  - **Set init scale to 1.0 with sigmoid → nothing happens either** (saturated). **Two opposite settings, one identical symptom** — this is the §9.3 warning box, felt.
  - **Width 2 on the spiral → underfits visibly. Width 16 → clean.** Capacity, seen.
  - **Sigmoid at 3 layers → visibly slower than tanh → visibly slower than ReLU.** §10.2's history of activation functions, in one A/B.
  - **Batch size 1 vs 32:** batch 1's loss curve is *hairy* but gets there; batch 32 is smooth. §4.

### DEMO 4 — Optimizer race
- **Plot:** the *same* surface (Rosenbrock, and the ravine) with **5 simultaneous trajectories in 5 colors**: SGD, SGD+momentum, RMSProp, Adam, AdamW. Same $\theta_0$, same seed. Inset: 5 loss curves.
- **Sliders:** shared base $\eta$, $\beta_1$, $\beta_2$, $\mu$, $\epsilon$; **noise level** ⭐; step counter.
- **JS:** all five update rules verbatim from §7, **including Adam's bias correction**, and a **"disable bias correction" toggle**.
- **⭐ Insights:**
  - On the ravine, SGD zig-zags, Adam walks straight. Expected.
  - **⭐ The bias-correction toggle: turn it off and watch Adam's FIRST STEP fly off the screen.** §7.4's "1000× too small at $t=1$" claim, made visible. **This is a claim nobody demonstrates and everybody asserts.**
  - **⭐ Turn noise up.** SGD flails. Adam is nearly unbothered (it normalizes magnitude away). **Then: on the *clean* Rosenbrock, well-tuned SGD+momentum actually WINS.** ← **Build this in deliberately.** It teaches §7.9's "Adam is not strictly better" honestly, and it teaches that a demo that always has one winner is a demo that's lying to you.
  - **Set $\mu$ from 0 → 0.9 with $\eta$ fixed and watch it diverge.** The $1/(1-\mu) = 10\times$ rule (§7.2), felt as a crash.

### DEMO 5 — Overfitting visualizer / double descent
**Two panels; the second is the one that earns §13.**

- **Panel A (classical):** polynomial fit, degree slider 1→20, to 12 noisy sine points. Shows fitted curve + **live train MSE and test MSE**, plus a **U-curve** of test error vs degree building up as you drag. Use the exact normal-equation / least-squares solve in JS (Vandermonde + Gaussian elimination, ~30 lines; use a tiny ridge $\lambda=10^{-8}$ for numerical stability at high degree). **The classical U, reproduced live.**
- **Panel B ⭐ (double descent):** same setup, but sweep degree **1 → 40 with only 12 data points**, so the interpolation threshold at **degree ≈ 11** is *on screen*. Use the **minimum-norm** least-squares solution (pseudo-inverse / SVD) past the threshold — **this is essential, and it's the whole mechanism: the minimum-norm choice is what produces the second descent.** Plot test error vs degree. **The learner watches error spike at degree 11 and then come back down.**
  - **A "regularization" slider ($\lambda$ from 0 → 0.1) that VISIBLY FLATTENS THE PEAK.** ⭐⭐ **This is the single most valuable demo in the whole brief**, because it makes §13.3's honest critique *tangible*: the learner drags a slider and watches the famous phenomenon *dissolve*. **He will remember "double descent is partly an artifact of under-regularization" forever, because he made it disappear with his own hand.** No amount of prose does that.
- **Panel C:** MLP on noisy two-moons, with train/val curves diverging live, and an **early-stopping marker** that plants itself at the val minimum.
- **⭐ Insight:** the classical U and double descent are both real, in different regimes, and regularization is the knob that connects them.

### DEMO 6 (bonus, cheap, high-value) — Loss function explorer
- **Plot:** $L$ vs $\hat y \in (0,1)$ for $y=1$: **BCE vs MSE, overlaid.** Second panel: $\partial L/\partial z$ vs $z$ for both, over $z\in[-8,8]$.
- **⭐ Insight:** **it's the §2.3 table, as a picture.** He sees MSE's gradient flatline toward 0 at $z=-8$ while BCE's holds at −1. **Two curves, one point, ten seconds.** Cheapest insight-per-line-of-code in the course. Build it.

### DEMO 7 (bonus) — Memory calculator ⭐
- **Inputs:** param count (slider or preset: 1B/3B/7B/13B/70B), precision (fp32/bf16), optimizer (SGD/SGD+mom/Adam/Muon), LoRA on/off + rank, batch size, sequence length, gradient checkpointing on/off.
- **Output:** a **stacked bar chart** — weights / grads / master / optimizer / activations — with a red horizontal line at **128 GB (his DGX Spark)**.
- **JS:** the arithmetic from §7.6. Twenty lines.
- **⭐ Insight:** he slides to 7B + Adam + full FT and **the bar goes over the red line.** He toggles LoRA and **the bar collapses to ~14 GB — and he watches the Adam segment, which was the biggest one, essentially vanish.** ⭐ **This is the emotional climax of the trunk and the exact moment the fine-tuning track becomes inevitable rather than arbitrary.** It costs 20 lines of JS. **Build it, and place it at the very end of the training chapter as the bridge.**

---

## 16. NUMBERS TO REUSE VERBATIM (the course's shared vocabulary)

Recurrence is how numbers become intuition. **Every one of these is verified.** The architect should treat this as a fixed table and enforce it across chapters.

| quantity | value | first appears |
|---|---|---|
| worked-example (**TN-1**, input 2) input | $x=[0.60, -0.20]$, $y=1$ | constants §8.7 (was §5.4, retired) |
| worked-example loss | $L = \mathbf{0.6148 \to 0.5630}$ after one step ($\Delta L=-0.0518$) — ~~0.453048→0.340179~~ retired net | constants §8.7, D-02 |
| worked-example output error | $\delta^{(2)} = \mathbf{-0.4593}$ — ~~−0.364312~~ | constants §8.7 |
| worked-example prediction | $\hat y = \mathbf{0.5407 \to 0.5695}$ — ~~0.635688→0.711643~~ | constants §8.7 |
| gradient spread in a 2-layer net | **10.22×** (0.4593 vs 0.0449, TN-1 2nd input) — ~~12.7× (retired §5.4 net)~~ | constants §8.7, D-02 |
| canonical logits | $z = [2.0, 1.0, 0.1] \Rightarrow \hat y=[0.659, 0.242, 0.099]$, $L=0.417030$ | §2.2 / constants §9.2 |
| coin-flip loss | $\ln 2 = 0.693147$ | §2.2 |
| random-guess loss, $V$ classes | $\ln V$ (1000 classes → 6.908; **Qwen3 151,936 vocab → 11.93**, ~~128k→11.76~~) | §2.2, §14.2 / constants §9.1 |
| max sigmoid derivative | 0.25 → $0.25^{10} = 9.5\times10^{-7}$ | §10.2 |
| the amplifier numbers | $1.1^{100}=13{,}781$; $0.9^{100}=2.7\times10^{-5}$ | §9.1 |
| He init at $d=4096$ | std = 0.0221 ≈ GPT-2's 0.02 | §9.2 |
| momentum amplification | $\mu=0.9 \Rightarrow 1/(1-\mu) = 10\times$ | §7.2 |
| Adam bias correction at $t=1$ | $v$ is **1000×** too small ($\beta_2=0.999$) | §7.4 |
| Adam $\beta_2$ memory | $1/(1-0.999) = 1000$ steps | §7.4, §8.3 |
| training memory | **16 bytes/param** (2+2+4+4+4) | §7.6 / constants §2.1 |
| **Qwen3-8B** full fine-tune | **131.05 GB / 122.05 GiB** (state) — ~~7B ~112 GB → ~130~~ retired | constants §2.2, D-01/D-03 |
| 8B Adam states alone ($m+v$) | **65.5 GB** — ~~7B 56 GB~~ | constants §2.2 |
| 8B inference (bf16 weights) | **16.38 GB** — ~~7B 14 GB~~ | constants §1.1 |
| LoRA $r=16$, **all-linear**, 8B | **43,646,976 params (0.533%)**, 0.61 GB trainable state — ~~attn-only 16,777,216~~ | constants §3, D-06 |
| LoRA total | **17.08 GB vs 131.05 GB (7.67×)** — ~~14.3 vs 130~~ | constants §2.3, D-03 |
| training FLOPs | $6 \times N_{\text{params}} \times N_{\text{tokens}}$ | §5.5 |
| 7B × 1T tokens | $4.2\times10^{22}$ FLOPs ≈ 3.3 H100-years @ 40% MFU | §5.5 |
| ravine critical LR | $\eta_{\text{crit}} = 2/\lambda_{\max} = 2/20 = 0.1$ | §3.4, Demo 1 |
| healthy update ratio | $\|\Delta\theta\|/\|\theta\| \approx 10^{-3}$ | §14.4 |
| standard grad clip | 1.0 | §7.7, §10.4 |
| 2026 LLM AdamW | $\beta=(0.9, 0.95)$, wd=0.1, $\eta_{\max}$=1e-4 (7B), clip=1.0, cosine to $\eta/10$ | §7.7 |
| fp16 max | 65504 (→ use bf16) | §14.1 |
| float32 epsilon | $1.19\times10^{-7}$ (→ gradient-check in float64) | §5.4 |

---

## 17. GENUINE UNCERTAINTY / DISAGREEMENT — DO NOT PAPER OVER

The architect should ensure each of these is presented **as** a live question, not resolved. **A course that admits uncertainty is trusted; a course that doesn't gets caught.** This learner is an adult with a DGX Spark — he will notice.

1. **Flat minima ⇒ generalization.** Widely believed, widely acted on (SAM, large LR, small batch), **not proven, and provably not-quite-right as literally stated** (sharpness isn't reparameterization-invariant — Dinh et al. 2017). §3.5.
2. **Double descent's status.** Real phenomenon; **largely vanishes under proper regularization; depends on parameter-count as the complexity axis; absent from LLM/ViT scaling curves.** §13.3. **The most important honesty flag in the brief.**
3. **Why BatchNorm works.** The original "internal covariate shift" story is **wrong** and was believed for years. The landscape-smoothing story is better-supported but not the last word. §11.6. **This is also a meta-lesson about the field, and it should be taught as one.**
4. **Why warmup works.** Adam-variance (RAdam) is the best-supported mechanism, but warmup persists even where that explanation is weakened, and multiple competing explanations coexist. §8.3.
5. **Pre-norm vs post-norm.** Pre-norm dominates; Mix-LN and hybrid/sandwich-norm work argues pre-LN under-trains deep layers. **Not settled.** §11.5.
6. **Cosine vs WSD schedules.** Cosine is the incumbent; WSD is gaining on continuation- and scaling-law grounds. **No consensus in 2026.** §8.2.
7. **Muon vs AdamW.** Promising, deployed at scale (Kimi K2, Megatron), impressive reported numbers — but the reports are from labs/vendors with an interest, independent replication is thin, and Muon-fine-tuning-Adam-pretrained-models is an open question. §7.8.
8. **The linear scaling rule.** Works, until the critical batch size, which nobody can predict a priori. $\sqrt{k}$ vs $k$ scaling for Adam is contested. §4.3.
9. **Adam vs SGD generalization.** SGD still wins on some vision benchmarks. The reason is not fully understood. §7.9.
10. **Whether "why does deep learning generalize at all" is answered.** **It is not.** Classical learning theory's bounds are vacuous for these models (they predict test error > 100%). This is a genuine open problem, and it is *fine and good* to tell an adult learner that the most successful technology of the decade is not understood by the people who build it. **That's honest, it's true, and it's more interesting than a fake answer.**

---

## 18. HANDOFFS — WHAT THE ARCHITECT MUST RECONCILE WITH OTHER BRIEFS

| topic | this brief's position | reconcile with |
|---|---|---|
| **Residual connections** | I motivate them in §10.3 as the *gradient-flow* fix — a two-line consequence of backprop. **This is the right place to first meet them.** | **Architecture brief**, which will want them as a structural element. Suggest: **trunk owns "why residuals exist" (gradient flow); architecture owns "where they sit."** Do not let this fall between the two briefs. |
| **Normalization** | §11 owns the math, the axis question, and pre- vs post-norm's *gradient* justification. | **Architecture brief** (block diagrams). Suggest: trunk owns why; architecture owns where. |
| **Optimizer memory / LoRA** | §7.6 is deliberately built as **the bridge** to fine-tuning. The punchline — **"LoRA is optimizer-state compression, not model compression"** — is *earned* here and will not land without §7.6's numbers. | **Fine-tuning brief.** ⚠️ **Make sure the fine-tuning brief doesn't re-derive it (redundant) or assume it (unearned). It should open by CALLING BACK to §7.6's **131.05 GB → 17.08 GB (7.67×, state-to-state)** — ~~130 GB → 14.3 GB~~, which D-03 retires because it silently credits LoRA with an activation saving it does not deliver.** Reuse the exact numbers from constants §2.3. |
| **Cross-entropy** | §2 derives it from MLE, for both binary and categorical. | **LLM brief** — next-token prediction is *literally* §2's Branch C with $K$ = vocab size, applied per position. **The LLM brief should be able to say "you already know the LLM loss; it's the softmax CE from §2, run once per token" and be RIGHT.** That's a big win if the trunk sets it up. |
| **MSE** | §2 Branch A derives it from a Gaussian likelihood. | **Diffusion brief** — the diffusion loss $\|\epsilon - \epsilon_\theta(x_t,t)\|^2$ **is exactly MSE under a Gaussian assumption**, which §2.2 Branch A already justified. **The diffusion brief should get this for free.** ⭐ **This is the strongest structural argument for deriving MSE from MLE rather than asserting it — it makes the diffusion loss a corollary instead of a new fact.** Flag prominently. |
| **`.detach()` / stop-gradient** | §6.6 teaches it as "cutting a wire on purpose." | **Diffusion brief** (detached targets, EMA outside autograd). Trunk should establish the concept so the diffusion brief can use it in one sentence. |
| **Bias-variance** | §13.3 lands it on **LoRA rank as the capacity knob.** | **Fine-tuning brief** — rank selection. Coordinate so rank is introduced *as* a bias-variance decision, not as a magic number. |
| **Gradient checkpointing** | §5.5 sets it up (activations must be stored for BP4 — that's *why* training needs memory) but I **do not** teach the technique. | **Fine-tuning/practical brief** should own it. **The trunk must plant the "why," in §5.5, or the technique is unmotivated.** |
| **Mixed precision** | §7.6 (fp32 master weights) and §14.1 (bf16 vs fp16, 65504) touch it. **I do not own it.** | **Practical/hardware brief.** Suggest it owns AMP, GradScaler, and the `unscale_`-before-clip ordering bug — but must **not** re-explain why master weights exist (§7.6 owns that). |
| **Attention/transformers** | Not mine. | Architecture brief. |
| **The 6ND FLOPs rule** | §5.5 derives it from the fwd/bwd cost ratio. | **LLM brief** (scaling laws, Chinchilla). Reuse the 7B×1T = $4.2\times10^{22}$ number. |

**One structural recommendation for the architect.** §5.4 (the 2-2-1 worked example) and §7.6 (the memory table) are the two load-bearing pages of the trunk. Everything else can be compressed under page pressure. **These two cannot.** §5.4 is where the learner *understands*; §7.6 is where he *sees why it matters to him*. If the trunk has to shrink, shrink §12 and §13 first.

---

## 19. SOURCES

- [torch.optim — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/optim.html)
- [AdamW — PyTorch main documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.adamw.AdamW_class.html)
- [Muon is Scalable for LLM Training (Moonshot/Moonlight, arXiv:2502.16982)](https://arxiv.org/pdf/2502.16982)
- [MoonshotAI/Moonlight (GitHub)](https://github.com/MoonshotAI/Moonlight)
- [NVIDIA Megatron Boosts LLM Training With Muon Optimizer (2026-04-22)](https://blockchain.news/news/nvidia-megatron-muon-llm-training)
- [Can Muon Fine-tune Adam-Pretrained Models? (arXiv:2605.10468)](https://arxiv.org/pdf/2605.10468)
- [On the Variance of the Adaptive Learning Rate and Beyond — RAdam (arXiv:1908.03265)](https://arxiv.org/pdf/1908.03265)
- [Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training (arXiv:2410.23922)](https://arxiv.org/html/2410.23922v1)
- [Improving Transformer Optimization Through Better Initialization — T-Fixup (ICML 2020)](https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf)
- [Double descent — Wikipedia](https://en.wikipedia.org/wiki/Double_descent)
- [A Farewell to the Bias-Variance Tradeoff? (arXiv:2109.02355)](https://arxiv.org/pdf/2109.02355)
- [Understanding the Double Descent Phenomenon in Deep Learning (review)](https://www.themoonlight.io/en/review/understanding-the-double-descent-phenomenon-in-deep-learning)
- [Mix-LN: Combining Pre-LN and Post-LN (arXiv:2412.13795)](https://arxiv.org/pdf/2412.13795)
- [Demystifying Synthetic Data in LLM Pre-training (arXiv:2510.01631)](https://arxiv.org/pdf/2510.01631) — 2026 AdamW hyperparameter conventions
- [SPAM: Spike-Aware Adam with Momentum Reset (arXiv:2501.06842)](https://arxiv.org/pdf/2501.06842) — LLM gradient-spike behavior
- [GPU VRAM Requirements to Fine-Tune LLMs in 2026 (Spheron)](https://www.spheron.network/blog/gpu-vram-requirements-fine-tune-llm-2026/)
- [Parameter-Efficient Fine-Tuning: The Math & Code Behind LoRA (2026)](https://medium.com/@utkrisht14/parameter-efficient-fine-tuning-the-math-code-behind-lora-adapters-more-fefe55341607)
- [LoRA Learns Less and Forgets Less (arXiv:2405.09673)](https://arxiv.org/pdf/2405.09673)

**Verification note:** all arithmetic in §2.2, §2.3, §2.5, §5.4, §5.5, §7.2, §7.4, §7.6, §8.2, §9.2, §10.2 was computed numerically, not recalled. The §5.4 backprop gradients were independently cross-checked against central finite differences and agree to 8 decimal places. Hyperparameter conventions in §7.7 and the Muon figures in §7.8 are web-verified against 2025–2026 sources and carry the confidence flags noted inline.
