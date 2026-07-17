# BRIEF: PEDAGOGY — The Teaching Contract

**Audience:** the curriculum architect writing the build spec.
**Status:** prescriptive. Where I say MUST, treat it as a spec constraint that every page and every fan-out agent honors. Where I say SHOULD, it is a strong default with a stated escape hatch.
**Date:** 2026-07-16. Evidence current as of this date; confidence flags in §14.

The other briefs tell you WHAT to teach. This brief tells you HOW, and — more importantly — it sets the invariants that keep a fan-out build from producing 50 pages that each make sense alone and contradict each other in aggregate.

---

## 0. The one-paragraph contract

Every page leads with a physical or geometric intuition in one sentence. Every symbol that appears has been defined, with its shape, in the canonical notation table (§6) — no page invents notation. Every concept is carried to at least one real number the learner can check by hand. Every interactive demo computes live math whose formula is written on the page next to it. Every quiz question is answerable only by someone who could do the thing, not by someone who read the words. The learner runs real code by page 6 and never goes more than ~4 pages without running something. Assistance is available everywhere and mandatory nowhere.

---

## 1. The learner model: rusty ≠ naive, and rustiness is *patchy*

This is the single most consequential fact in the brief, and the most common way a course like this fails.

### 1.1 The evidence

The **expertise reversal effect** is the finding that instructional assistance that helps novices actively *harms* learners with high prior knowledge. The 2025 meta-analysis (Tetzlaff et al., *Learning and Instruction* vol. 98, 2025; 176 effect sizes, 60 experiments, N = 5,924, PRISMA, metafor/R with dependency-corrected effect sizes) reports:

| Learner | Instruction | Effect |
|---|---|---|
| Low prior knowledge | High assistance (worked examples, full scaffolds) | **d = +0.505** |
| High prior knowledge | High assistance | **d = −0.428** |

Moderated by prior-knowledge assessment type, educational status, and content domain (effects stronger in higher education; weaker in humanities/language). Two things follow.

**(a) The effect is real and it is large.** d = −0.428 is not a rounding error. Over-explaining to this learner is not "harmlessly thorough" — it measurably degrades his learning. Every redundant re-derivation of something he already owns is a tax.

**(b) The effect is asymmetric.** +0.505 > |−0.428|. Giving assistance to someone who didn't need it costs *less* than withholding it from someone who did. The meta-analysis's own practical implication is: **when uncertain, default to providing assistance.**

### 1.2 Why (a) and (b) look contradictory, and the design that resolves them

They only conflict if assistance is a single global dial. It isn't, and shouldn't be.

Our learner is **not uniformly expert or uniformly novice — he is heterogeneous per topic**:

| Topic | Likely state | Correct assistance level |
|---|---|---|
| Sampling schedulers, CFG, VAE latents, LoRA loading | **Expert-by-practice** (ComfyUI daily) — has procedural fluency, may lack the model | LOW assistance on *what it does*; but he needs the *why*, which is new content, not review |
| GPU memory, quantization, hardware trade-offs | **Expert** (runs a DGX Spark) | LOW — use as an anchor to teach *other* things |
| Chain rule, partial derivatives | **Rusty** — recognizes it, can't wield it | MEDIUM, learner-selectable |
| Matrix multiplication, shapes | **Rusty** — remembers rows-times-columns, has no geometric picture | MEDIUM |
| Probability, KL divergence, ELBO, log-likelihood | **Probably genuinely thin** | HIGH |
| Backprop as an algorithm, autograd | **Novice** | HIGH |

**Therefore: assistance MUST be per-topic and learner-selectable, not global.** The mechanism is the collapsible deep-dive (§5.4). This is not a cosmetic choice — it is the direct instructional-design consequence of a d = −0.428 effect on a learner whose expertise is patchy. The main column is written for the expert case (low assistance); the collapsible carries the novice case (high assistance), present but not imposed.

> **Honest caveat the architect must plan around.** Learner-controlled assistance assumes accurate self-monitoring, and self-assessment of knowledge is notoriously poorly calibrated — people routinely mistake *familiarity* ("I've seen the chain rule") for *fluency* ("I can apply it to a 3-layer composition"). This learner is at maximal risk for exactly that error, because rustiness *feels* like knowing. If we rely on "expand it if you need it," he will under-expand and then stall three pages later.
>
> **The fix is diagnostic, not exhortative.** Each part opens with a 3–5 item **prerequisite check** that is a *retrieval* task, not a self-rating. Never ask "how comfortable are you with the chain rule? (1–5)". Ask him to actually differentiate `f(x) = tanh(3x² + 1)` and grade it. The result routes him: pass → collapsed by default; miss → that part's deep-dives ship pre-expanded. The course tells him where he's rusty; he doesn't have to guess. This also gives the build a place to put spaced retrieval (§3.3) for free.

### 1.3 What "rusty but once trained" specifically means for prose style

- **Do not motivate that math is useful.** He knows. Cut every "as we will see, this powerful technique…"
- **Do not teach vocabulary he has** ("a vector is a list of numbers"). Do teach the *picture* he probably never had ("a vector is a point; a matrix is a machine that moves every point at once, and W's rows are where the basis vectors land").
- **Assert-then-verify beats build-from-zero for known objects.** For a rusty learner, "You remember the chain rule: dz/dx = dz/dy · dy/dx. Here's the one thing about it that everyone forgets and that backprop lives on:" is worth five pages of re-derivation.
- **Respect his procedural expertise by cashing it in.** He has run thousands of diffusion samples. That's a huge asset: he has a rich store of *phenomena* awaiting explanation ("why does CFG > 12 fry the image?", "why do the first few steps matter most?"). Every one of those is a free motivation hook — a question he *already has*. Use them. This is the closest thing this course has to a superpower, and a generic ANN course throws it away.

---

## 2. Where courses like this actually fail

Four named failure modes. The spec should be checkable against each.

### 2.1 Hiding behind notation
Symptom: `$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$` dropped on the page with the words "the ELBO" and no further comment. The learner pattern-matches, nods, and has learned nothing.
Diagnosis: notation is a *compression* of an idea. Presenting the compressed form to someone who lacks the decompressor transmits zero bits.
**Fix (spec rule): the Symbol Ledger.** Any equation with more than three distinct symbols MUST be immediately followed by a per-symbol table: symbol → plain-English name → shape → units/range → "where it came from." No exceptions, including in collapsibles. See §6.6 for the required format. This is boring to write and it is the highest-value rule in the brief.

### 2.2 Asserting instead of deriving
Symptom: "It can be shown that the gradient of the softmax-cross-entropy is `ŷ − y`."
Diagnosis: this specific result is *the* moment of grace in the whole subject — the reason the loss/activation pairing isn't arbitrary — and asserting it converts a memorable insight into a memorized fact.
**Fix:** distinguish **load-bearing derivations** (must be inline, must be complete) from **reassurance derivations** (collapsible). Criterion in §5.4. `ŷ − y` is load-bearing. The closed form of the tanh derivative is reassurance.

### 2.3 Toy examples that don't transfer
Symptom: XOR with two neurons, then a jump to a 7B transformer, with nothing in between. The learner can do XOR and cannot see how XOR is the same thing as a transformer, because *it wasn't presented as the same thing*.
Diagnosis: transfer doesn't happen by proximity. It happens when the *same object* is re-encountered in a new guise, and the course explicitly narrates the correspondence.
**Fix:** the recurring-thread architecture (§9). The tiny network is not "a toy example in chapter 3." It is the **same nine parameters** that get re-instantiated as a PyTorch module, then as an autograd graph, then as one attention head's worth of arithmetic, then as the thing LoRA decomposes. Every reappearance MUST explicitly say "this is TN-1 again, now wearing X."

### 2.4 The inverse failure: 40 pages of theory before anything runs
Symptom: perceptrons → linear algebra → calculus → probability → information theory → *then* code, on page 41.
Diagnosis: motivation decays; and worse, the theory is *unfalsifiable to the learner* — he has no way to test whether he understood it, so misconceptions accrete silently for 40 pages and then all detonate at once.
**Fix: the Early Real Thing (§4).** Something real must run by **page 6**, and the gap between runnable artifacts must never exceed ~4 pages. This is non-negotiable and it constrains the whole sequence.

> **Note the tension.** 2.4 pushes toward code-first; 2.2 pushes toward derive-properly. These are reconciled by *interleaving* (§3.3), not by compromise: run the thing, then derive why it worked, then break it. Do not "balance" by half-deriving. A half-derivation has the costs of both.

---

## 3. The evidence base — what's solid, what's contested

Be honest with the architect about the strength of each lever, because a couple of the most-cited ones are weaker than their reputation.

### 3.1 Solid enough to build on

| Principle | Best estimate | Source |
|---|---|---|
| Expertise reversal | d = +0.505 (novice/assist), −0.428 (expert/assist) | Tetzlaff et al. 2025 meta-analysis, N=5,924 |
| Worked-example effect | Large for novices; **reverses** for experts | Same; Kalyuga et al. lineage |
| Problem-solving-before-instruction (productive failure) | g = 0.36 (95% CI 0.20–0.51), 53 studies / 166 comparisons; g = 0.37–0.58 at high design fidelity; **g = 0.87** for conceptual knowledge/transfer after publication-bias correction | Sinha & Kapur 2021, *Review of Educational Research* |
| Spacing (mathematics specifically) | g = 0.28 overall; g = 0.43 isolated, **g = 0.24 course-embedded** | 2025 meta-analysis, *Educational Psychology Review*, 27 studies / 53 ES |
| Interleaving | g = 0.42 overall; **g = 0.34 for maths tasks** | Brunmair & Richter 2019 and successors |
| Retrieval practice → transfer | **d ≈ 0.4** vs. non-practice, moderated by response congruency, initial accuracy, elaborated retrieval | transfer meta-analysis |

### 3.2 Genuinely contested — do not paper over

- **Cognitive load theory's internal machinery is on shakier ground than its practical advice.** The germane/extraneous/intrinsic triad drew a serious falsifiability objection: when total load dropped, researchers could post-hoc attribute it to extraneous load (if learning improved) or germane load (if it didn't), with no way to falsify either — a circular interpretation. There is also no agreed unit or valid instrument for germane load. **Sweller et al. (2019) responded by removing germane load from the additive equation**, recasting it as "germane *processing*" — a redirection of resources rather than a load that adds to the total. Element interactivity is now the defining mechanism for intrinsic load and increasingly for extraneous load too.
  **What this means for us:** use CLT's *design heuristics* (segmenting, worked examples, split-attention, redundancy, expertise reversal) — those are empirically supported on their own terms. **Do not** write pages that reason about "germane load" as a quantity, and do not let any agent justify a design choice by claiming it "increases germane load." That sentence is not falsifiable and in current CLT it isn't even well-formed. Justify by element interactivity instead: *how many things must this learner hold in mind simultaneously to follow this step?* That question is concrete, countable, and actionable.

- **Retrieval practice does not automatically transfer upward.** This is the finding that should most change our quiz design. Fact quizzes improve fact recall and **do not** produce higher-order learning, contra the popular Bloom's-taxonomy intuition that facts are the foundation higher-order thinking is built on. The governing principle is **transfer-appropriate processing**: retrieval benefits appear when the *practice format matches the target format*. Learners who practice with factual questions and are then tested on higher-order questions perform **indistinguishably from learners who did no retrieval practice at all**. Far transfer *is* achievable — recent work finds it after a ~1-week delay and attributes it to rule-based learning rather than item memory — but it requires sufficient practice volume, delay, and elaborated retrieval.
  **What this means for us:** see §8. Our target format is "fine-tune a model and diagnose it when it misbehaves." So our quiz format must be prediction, diagnosis, and shape/budget arithmetic. **Vocabulary quizzes are worse than nothing** — they consume the retrieval-practice budget and buy zero transfer.

- **Productive failure is conditional, not magic.** The g = 0.36 headline is solid, and the bias-corrected g = 0.87 for conceptual/transfer outcomes is striking, but two caveats are load-bearing: (i) it is **unclear whether failure itself is necessary** — the benefit may come from activating prior knowledge and noticing the gap, not from failing per se; (ii) benefits are **highly conditional** on task structure, consolidation opportunity, and design fidelity. A predict-then-check step with no consolidation afterward is just a wrong answer.
  **What this means for us:** every predict-then-check MUST be followed by explicit consolidation that names the gap ("you probably said the loss would go down; here's why it went up"). Prediction without resolution is worse than no prediction.

- **Spacing is weaker inside a course than in the lab.** g = 0.43 isolated vs **g = 0.24 course-embedded**. Our setting is course-embedded. Plan for the smaller number and don't over-engineer spacing infrastructure. It's worth doing; it is not worth building a spaced-repetition scheduler for.

- **Learning styles remain unsupported.** No visual/verbal/kinesthetic routing. Dual coding (§5.5) is a *different, supported* claim — everyone benefits from coordinated verbal + visual, which is not the same as matching a modality to a person.

### 3.3 The levers we will actually pull

Ranked by expected value in this specific build:

1. **Expertise-adaptive assistance** (collapsibles + diagnostics) — biggest effect, directly matched to this learner. §5.4
2. **Transfer-appropriate retrieval** (quizzes shaped like the destination) — cheap, and the alternative is actively wasteful. §8
3. **Predict-then-check before instruction**, on concepts with a strong hook — g = 0.36–0.87. §5.2
4. **Interleaving via recurring threads** — g ≈ 0.34, and it's nearly free once threads exist. §9
5. **Worked examples with real arithmetic** — for the genuinely-novice topics only. §7
6. **Spacing** — g ≈ 0.24 embedded; get it free via thread callbacks, don't build machinery.

---

## 4. Sequencing: the Early Real Thing, and the ladder

### 4.1 The rule

**Something real runs by page 6. The gap between runnable artifacts never exceeds ~4 pages.**

"Real" means: it executes, on his machine or in his browser, on data he can change, and produces an output he can be surprised by. A static code listing is not real. A pre-rendered animation is not real.

### 4.2 Why page 6 and not page 1

Page 1 is too early for *his own code* — he'd be typing incantations. But it is not too early for **a live demo he can drag**. So the ladder is:

| Page | Artifact | Kind | What it buys |
|---|---|---|---|
| 1–2 | Drag a single neuron's weights, watch the decision boundary rotate | Browser demo | The geometric picture, before any notation |
| 3–5 | Hand-compute TN-1's forward pass; browser demo checks his arithmetic | Hand + browser | Ownership of the object |
| **6** | **Same TN-1, ~15 lines of PyTorch, on his DGX Spark; matches his hand arithmetic to 4 decimals** | **Real code** | **"The framework is not magic — it's my arithmetic, faster."** |
| 8–10 | Autograd computes the gradient he computed by hand; `assert torch.allclose(...)` passes | Real code | Trust in autograd, earned not asserted |
| ~12 | Train TN-1 to convergence; watch loss curve live | Real code | First real training loop |
| ~18 | Same loop, MNIST-scale, on the Spark | Real code | Scale is a parameter, not a new subject |
| ~24 | Fine-tune something small end-to-end (pre-split) | Real code | **Mid-course milestone — see §12** |
| ~32 / ~40 | Track-specific fine-tune (LLM / diffusion) | Real code | The payoff |

The page-6 moment is the highest-leverage single event in the course. **Do not let it slip.** It is what converts "I'm reading a textbook" into "I'm doing this." If the spec has to cut content to protect page 6, cut content.

### 4.3 The general sequencing principle

**Run → derive → break → generalize.** Not derive → run.

- **Run**: it works, and he sees it work. Buys motivation and a concrete referent.
- **Derive**: *why* it worked. Now the derivation is about something he has *seen*, so every symbol has a referent he can point at. This is the difference between notation and hieroglyphics.
- **Break**: change the learning rate to 10, watch it diverge. Set all weights to zero, watch symmetry never break. **Breaking is where the understanding actually lands**, and it is the most-skipped step in every ML course ever written. A thing you have only seen work is a thing you don't understand the boundaries of. Budget for it explicitly.
- **Generalize**: the same structure at 7B parameters.

---

## 5. The page rhythm

### 5.1 The proposed default, and my amendment

The brief proposes: **intuition → math → worked number → interactive demo → quiz.**

That ordering is good and I'd keep it as the *default*, with **one amendment and one restriction.**

**Amendment — add a PREDICT step at the top, on hook-rich pages:**

> **intuition → *predict* → math → worked number → demo (check) → quiz**

The productive-failure evidence (g = 0.36; g = 0.87 for conceptual/transfer, bias-corrected) says a brief attempt *before* instruction beats instruction-then-practice. But the fidelity caveat is real, so this is narrow: the predict step is a **single, concrete, committed guess** — 15 seconds, not an exploration session. "Before you read on: if I double the learning rate, does the loss curve get to the minimum twice as fast, roughly as fast, or not at all? Commit to one." Then the demo resolves it.

**Restriction — when NOT to invert.** Predict-then-check requires a hook the learner can reason *from*. Use it when:
- ✅ he has relevant experience or intuition to mobilize (anything touching diffusion sampling, anything touching memory/hardware, anything with a geometric picture already established);
- ✅ the prediction is falsifiable in one demo drag;
- ✅ there's a consolidation beat immediately after.

Do NOT use it when:
- ❌ the page is definitional or notational (predicting the convention for index order is noise, and flailing there is pure extraneous element interactivity);
- ❌ he has no basis to predict from (asking him to guess the ELBO is cruelty theater);
- ❌ there's no room to consolidate — an unresolved wrong prediction is a *misconception you installed yourself*.

Rough target: predict-then-check on **~40%** of concept pages, concentrated where he has prior experience — i.e., heavily in the diffusion track and the memory/hardware threads, sparsely in the probability foundations.

### 5.2 What each element must accomplish

| Element | Job | Failure mode | Hard rule |
|---|---|---|---|
| **Intuition** | Install a referent the later symbols can point at. Physical or geometric, one sentence, no jargon. | Restating the definition in friendlier words ("the gradient tells us how to improve") — that's a synonym, not an intuition. | Must be **falsifiable-ish and mechanical**: "the gradient is the direction you'd walk to climb the loss surface fastest; backprop is the chain rule run right-to-left so you pay for each layer once." One sentence. If it takes three, the page is doing too much. |
| **Predict** | Activate prior knowledge; create the gap. | Open-ended musing. | One question, ≤3 options, committed before scrolling. |
| **Math** | Make the intuition *precise and computable*. | Notation dump. | Every symbol in the Ledger (§6.6). Every equation annotated with shapes. |
| **Worked number** | Prove the math is arithmetic, not incantation. Give him a checkable anchor. | Symbolic "example" with no numbers; or numbers that don't resolve. | **Must terminate in an actual decimal.** Must be reproducible by hand. Must match the demo and the code to ≥4 significant figures. |
| **Demo** | Let him vary what the math holds fixed. | Decoration. See §7. | Computes the page's equation live, in JS, from the same constants as the worked example. |
| **Quiz** | Retrieval in the *destination format*. | Vocabulary recall. | See §8. |

### 5.3 Ordering argument: why intuition before math (and why this is not just niceness)

Because of **element interactivity**, which is the one CLT construct that survived the germane-load purge intact. The difficulty of a step is the number of elements that must be held and related *simultaneously*. An equation presented cold forces the learner to simultaneously hold: what each symbol denotes, what shape it is, why those operations, and what the whole thing means. That's 4 interacting element-classes at once.

The intuition *pre-loads the "what it means" slot* from long-term memory, so when the symbols arrive, the learner has a referent to hang each on and the interactivity drops to ~2. Same equation, roughly half the load, no content removed. That's the entire mechanism, and it's why "intuition first" is not a stylistic preference but a load-management technique.

Corollary: **the intuition must be the same idea as the math**, not a vibe adjacent to it. A poetic intuition that doesn't map symbol-for-symbol onto the equation makes things *worse* — now there are two things to reconcile.

### 5.4 Inline vs collapsible: the decision rule

This is where expertise reversal gets operationalized. **The main column is written for the low-assistance (expert) reader. Collapsibles carry the high-assistance (novice) content.** Because the effect is asymmetric (+0.505 vs −0.428), assistance should be *present but collapsed*, never absent — and the §1.2 diagnostic decides the default state per part.

**Inline (never collapsed):**
1. The result **changes what the learner does next** (`∂L/∂z = ŷ − y` → this is why you never stack softmax on BCE; `Var` scaling → this is why init matters).
2. The derivation is **the insight itself**, not support for it. Backprop *is* the chain rule applied in an order. Hiding that leaves nothing behind.
3. It's a **thread beat** (§9) — thread continuity must never be optional.
4. It's **≤ 5 lines** and the reader would spend more effort deciding whether to expand it than reading it.

**Collapsible deep-dive:**
1. **Rusty-prerequisite reconstruction** — "Refresher: the chain rule." Expert skips (avoiding the −0.428 tax), rusty expands. This is the single biggest use.
2. **Reassurance derivations** — algebra that confirms a result the learner will use as a black box either way (deriving `d/dx tanh(x) = 1 − tanh²(x)`).
3. **Rigor the result doesn't depend on** — why the reparameterization trick is valid measure-theoretically.
4. **Alternative framings** — "if you prefer the information-theoretic view of this."
5. **Historical/why-not-otherwise** — good content, breaks flow.

**Anti-rules (things the spec should forbid):**
- ❌ Never collapse something a later page depends on. If page 30 needs it, page 12 states it inline. **Collapsibles must be strictly optional for course continuity** — a reader who expands nothing must still be able to finish. This is a hard invariant and a fan-out build will violate it accidentally unless it's checked.
- ❌ Never put the *punchline* in a collapsible. Collapsibles hold support, never payload.
- ❌ Never nest collapsibles. Nesting is a signal the page is mis-scoped.
- ❌ Collapsible titles must state the *content*, not tease it. "Deep dive: why the tanh derivative is 1 − tanh²" — not "Want to know more?" He needs to decide without expanding; a teasing title forces an expand to evaluate, which defeats the entire mechanism.

### 5.5 Dual coding

Supported claim: coordinated verbal + visual presentation beats either alone — *for everyone*, not matched to a "visual learner." Two constraints that matter more than the principle:

- **Split-attention:** the label goes *on* the diagram, not in a legend below it. If the learner must hold "series 3" in working memory while hunting for the blue line, that's manufactured element interactivity.
- **Redundancy effect:** narrating a diagram *and* captioning it with the same words is *worse* than either. Text and picture must carry **complementary** information. This is the most-violated multimedia principle in technical writing.

Concretely: every core object gets **one canonical visual** reused course-wide, not redrawn per page. TN-1's graph. The loss surface. The shape ribbon. Same colors, same layout, every time. A re-drawn diagram is a new object to the reader even when it's the same object to the author — and in a fan-out build, *every agent will redraw it differently unless the spec ships the canonical asset*.

---

## 6. CANONICAL NOTATION — normative

**This section is the most important deliverable in this brief for a fan-out build.** Different agents writing different pages *will* produce incompatible notation unless it is specified to the character. ML notation is genuinely inconsistent in the literature; we do not get to be inconsistent internally.

### 6.1 The central decision: row or column?

This is the real fork, so let me state the actual ground truth first.

**Verified from the PyTorch 2.13 docs (`torch.nn.Linear`):** the module applies

$$y = xA^T + b$$

with `weight` of shape **`(out_features, in_features)`**, input of shape `(*, in_features)`, output `(*, out_features)`.

Meanwhile every math treatment writes $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ with $\mathbf{x}$ a column.

**The reconciling insight — and it is a gift, so the course must not squander it:**

> **The weight matrix has shape (out, in) in BOTH conventions.** `nn.Linear(3, 5).weight.shape == (5, 3)`, and the math's $W$ in $\mathbf{y} = W\mathbf{x}$ is also (5, 3). The conventions differ **only** in whether you stack the batch on the left. There is no disagreement about $W$ itself.

That is not a coincidence to be papered over — it's a **load-bearing teaching moment** that dissolves the single most common source of shape confusion in the field. It gets its own page, early (~page 7, right after the page-6 code moment, when he has just seen both forms and the question is live).

**PRESCRIPTION:**

| Context | Convention | Form |
|---|---|---|
| Single-example math, all trunk derivations, backprop | **Column vectors** | $\mathbf{y} = W\mathbf{x} + \mathbf{b}$, $\mathbf{x} \in \mathbb{R}^{d_{\text{in}} \times 1}$, $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ |
| Anything with a batch or sequence axis; **all code**; attention | **Row-batched** | $Y = XW^\top + \mathbf{b}^\top$, $X \in \mathbb{R}^{B \times d_{\text{in}}}$, $Y \in \mathbb{R}^{B \times d_{\text{out}}}$ |
| **Invariant, everywhere, no exceptions** | $W$ is **always** $(d_{\text{out}}, d_{\text{in}})$ | Never write $W$ as (in, out). Never. |

**Why this split rather than picking one globally:**
- Column-only would force us to transpose every attention equation in the literature, and he will read papers. Attention is written row-style *universally*; fighting that spends the course's credibility to buy nothing.
- Row-only makes backprop derivations uglier (gradients become row covectors, and the Jacobian bookkeeping that makes backprop *click* gets muddier). The trunk is where the derivation must land cleanest.
- The split costs one page of explicit bridging. The invariant ($W$ is always (out, in)) means the *thing he must remember* never changes — only the batch placement does, and that's visible in the shape ribbon on every line.

**Honest cost, stated plainly:** this is a real trade-off and reasonable people pick differently. Two conventions means two chances to confuse. I judge the bridge cheaper than either monoculture *because the (out, in) invariant holds across both*, which makes the bridge a one-liner rather than a translation table. **But the architect should treat the invariant as more important than my specific choice.** If you overrule me and go row-only, do it globally and keep $W$ at (out, in). Consistency dominates correctness-of-choice here.

### 6.2 Typography — normative

| Object | Style | Example |
|---|---|---|
| Scalar | italic lowercase | $x$, $\eta$, $\alpha_t$ |
| Vector | **bold** lowercase, **column** by default | $\mathbf{x}$, $\mathbf{b}$, $\boldsymbol{\epsilon}$ |
| Matrix / 2-D | **bold** uppercase, or plain uppercase for canonical ones | $W$, $X$, $Q$, $K$, $V$ |
| Tensor (≥3 axes) | plain uppercase, shape ribbon **mandatory** | $X$ with `(B, S, d_model)` |
| Set / space | blackboard | $\mathbb{R}$, $\mathbb{E}$ |
| Scalar element of a vector | italic + subscript, **not bold** | $x_i$ is the $i$-th component of $\mathbf{x}$ |
| Loss | script | $\mathcal{L}$ |
| Learned parameters (collective) | theta | $\theta$ |
| Estimate / prediction | hat | $\hat{y}$ |
| Noised quantity (diffusion) | subscript t | $\mathbf{x}_t$ |

### 6.3 Index discipline — resolves the t-collision

**There is a real collision the architect must not discover on page 38:** transformer literature indexes sequence position with $t$; diffusion literature indexes noise level with $t$. The tracks are disjoint — *until they aren't*. A 2026 course covering DiT / diffusion transformers has pages where **both** appear. Resolve it globally, up front:

| Index | Range | Means | Notes |
|---|---|---|---|
| $b$ | $1..B$ | batch element | $B$ = batch size |
| $i$ | $1..S$ | **sequence position** (also: query position in attention) | **Not $t$.** |
| $j$ | $1..S$ | key/value position in attention | |
| $S$ | — | **sequence length** / context length | Papers call this $T$, $L$, or $N$ — we say $S$ |
| $t$ | $0..T$ | **diffusion timestep only** | $T = 1000$ by convention |
| $\ell$ | $1..L$ | layer index | $L$ = number of layers |
| $h$ | $1..H$ | attention head | $H$ = number of heads |
| $k$ | $1..V$ | vocabulary index | $V$ = vocab size |
| $n$ | $1..N$ | dataset example index | $N$ = dataset size |

$t$ is **reserved for diffusion time, course-wide.** Sequence position is $i$. This deviates from the transformer literature and that is a deliberate, priced choice: internal consistency beats matching any one paper, *and* it means DiT pages need no special pleading.

**Mandatory mitigation — the Translation Table.** Because we deviate, every page introducing a symbol that papers write differently MUST carry a one-line "papers call this…" note. This is not an apology for the deviation; it is **inoculation against notation shock**, which is one of the top reasons capable people bounce off their first paper. Teaching him that notation is *local and translatable* — rather than a fixed truth he's failing to know — is itself a durable lesson. Ship it as a persistent course-wide appendix too.

### 6.4 The canonical symbol table (course-wide, normative)

| Symbol | Meaning | Shape | Units / range |
|---|---|---|---|
| $\mathbf{x}$ | input vector, one example | $(d_{\text{in}}, 1)$ | data units |
| $X$ | batch of inputs | $(B, d_{\text{in}})$ or $(B, S, d_{\text{model}})$ | |
| $y$, $\mathbf{y}$ | target | scalar or $(d_{\text{out}}, 1)$ | |
| $\hat{y}$ | prediction | matches $y$ | |
| $W^{(\ell)}$ | weights of layer $\ell$ | $(d_{\text{out}}, d_{\text{in}})$ | dimensionless |
| $\mathbf{b}^{(\ell)}$ | bias of layer $\ell$ | $(d_{\text{out}}, 1)$ | same as pre-activation |
| $\mathbf{z}^{(\ell)}$ | pre-activation | $(d_{\text{out}}, 1)$ | |
| $\mathbf{a}^{(\ell)}$ | post-activation | $(d_{\text{out}}, 1)$ | activation's range |
| $\sigma(\cdot)$ | logistic sigmoid **only** | elementwise | $(0,1)$ |
| $\phi(\cdot)$ | generic activation | elementwise | — |
| $\mathcal{L}$ | loss, **per-example** | scalar | nats |
| $J$ | loss, **batch-averaged** | scalar | nats |
| $\eta$ | learning rate | scalar | $[10^{-5}, 10^{-1}]$ |
| $\theta$ | all parameters, flattened | $(P,)$ | $P$ = param count |
| $\nabla_\theta \mathcal{L}$ | gradient | $(P,)$ | loss/param |
| $\boldsymbol{\delta}^{(\ell)}$ | $\partial \mathcal{L}/\partial \mathbf{z}^{(\ell)}$ | $(d_\ell, 1)$ | **the backprop workhorse** |
| $\odot$ | elementwise (Hadamard) product | — | |
| $d_{\text{model}}$ | residual stream width | scalar | e.g. 4096 |
| $d_{\text{head}}$ | per-head dim | scalar | $= d_{\text{model}}/H$ |
| $\boldsymbol{\epsilon}$ | Gaussian noise | matches $\mathbf{x}$ | $\mathcal{N}(0,I)$ |
| $\alpha_t, \bar{\alpha}_t$ | noise schedule | scalar | $(0,1)$ |
| $r$ | LoRA rank | scalar | typically 8–64 |

**Two rules that prevent the most likely fan-out collisions:**
- $\sigma$ means **logistic sigmoid, and nothing else, ever.** Not standard deviation (use $s$), not a permutation, not a generic activation (use $\phi$). Diffusion pages are the risk: they will want $\sigma_t$ for noise std. **Use $s_t$.** Flag this explicitly to the diffusion-track agents.
- $\mathcal{L}$ is **per-example**; $J$ is **batch-averaged**. The factor of $1/B$ is exactly where learning-rate confusion breeds, so the notation must make it visible rather than hide it.

### 6.5 The shape ribbon — mandatory

Every multi-step computation carries shapes inline, in a fixed format:

```
x         (B=32, S=128, d_model=4096)
W_q^T     (4096, 4096)
q = x W_q^T   (32, 128, 4096)
reshape   (32, 128, H=32, d_head=128)
transpose (32, 32, 128, 128)          # (B, H, S, d_head)
scores = q k^T / sqrt(d_head)   (32, 32, 128, 128)   # (B, H, S_q, S_k)
```

Non-negotiable. **Shape errors are the #1 practical failure in real ML work**, and a course whose destination is "fine-tune models" that doesn't drill shapes has not reached its destination. Make shape-tracking a *habit* the notation enforces, not a topic in a chapter.

### 6.6 The Symbol Ledger format (fixes §2.1)

Required after any equation with >3 distinct symbols:

> $$\boldsymbol{\delta}^{(\ell)} = \left(W^{(\ell+1)\top}\boldsymbol{\delta}^{(\ell+1)}\right) \odot \phi'\!\left(\mathbf{z}^{(\ell)}\right)$$
>
> | Symbol | Is | Shape | From |
> |---|---|---|---|
> | $\boldsymbol{\delta}^{(\ell)}$ | how much the loss changes per unit change in layer $\ell$'s pre-activation | $(d_\ell, 1)$ | what we're solving for |
> | $W^{(\ell+1)}$ | next layer's weights | $(d_{\ell+1}, d_\ell)$ | forward pass |
> | $\boldsymbol{\delta}^{(\ell+1)}$ | same quantity, one layer later | $(d_{\ell+1}, 1)$ | already computed — **this is the recursion** |
> | $\phi'(\mathbf{z}^{(\ell)})$ | slope of the activation at the value it actually took | $(d_\ell, 1)$ | cached from forward pass |
> | $\odot$ | elementwise product | — | — |
>
> **Shape check:** $(d_\ell, d_{\ell+1}) \times (d_{\ell+1}, 1) = (d_\ell, 1)$, then $\odot (d_\ell,1) \to (d_\ell, 1)$. ✓

That "From" column is the part nobody writes and the part that carries the most. It tells the learner *where each thing came from*, which is precisely the information notation deletes.

---

## 7. Interactive demos: teaching vs. decorating

### 7.1 The criteria (the spec can be held to these)

A demo **teaches** iff:

1. **It computes the page's actual equation, live, in JS.** No pre-baked frames, no lookup tables, no canned animation. If the page says $\hat{y} = \sigma(W_2\tanh(W_1\mathbf{x} + \mathbf{b}_1) + b_2)$, the JS evaluates exactly that.
2. **The equation is visible next to the demo, with the live values substituted in.** The learner sees `z₂ = 0.6(0.000) + (−0.9)(0.8005) + 0.2 = −0.5205` update as he drags. This is the single highest-value demo feature in the course and it is what makes a demo *math* rather than *a toy*. It also kills split-attention.
3. **There is something to be surprised by.** If every slider position confirms what he'd guess, the demo is a screensaver. Design demos around the *misconception* (§8.2): the slider should be able to reach the state where the naive model breaks.
4. **It has a defensible default state**, and a reset. He must be able to get back to the worked example's exact numbers.
5. **It answers one question.** A demo with 6 sliders answers nothing. Ruthless: ≤ 3 controls. If it needs 4, it's 2 demos.
6. **It shares constants with the worked example and the PyTorch code.** Same numbers, three representations. This is what makes the thread a thread.
7. **Degradation is honest**: if a config diverges to NaN, show NaN. Don't clamp it into looking reasonable — divergence is a *lesson*.

A demo **decorates** if: it animates on load and ignores input; sliders change only cosmetics; it shows a result without the mechanism; it's a video in a `<canvas>`; or the "insight" is one you could get from a static figure. **If a static figure would do, ship a static figure** — it's cheaper and it loads faster. Interactivity must earn itself.

### 7.2 Three worked demo specs

**DEMO A — "The neuron is a ruler and a bend" (page ~2, before any notation)**

- **Plot:** 2-D plane, ~40 scattered points, two classes. A line. Background shaded by output value, blue→red.
- **Controls:** $w_1$, $w_2$, $b$ (3 sliders, range [−3, 3]).
- **JS computes:** for every pixel $(x_1, x_2)$: $z = w_1x_1 + w_2x_2 + b$; $a = \sigma(z) = 1/(1+e^{-z})$; color = lerp(blue, red, $a$). Also live-display accuracy over the 40 points.
- **The insight to engineer:** the learner drags $b$ and the boundary **translates**; drags $w_1, w_2$ and it **rotates *and* sharpens**. The intended realization: *$\mathbf{w}$'s direction sets the boundary's orientation; $\mathbf{w}$'s magnitude sets how sharp the transition is; $b$ just slides it.* Two facts for the price of one drag, and the second one (magnitude = confidence/sharpness) is the one nobody tells you and that later explains temperature, logit scale, and why init scale matters.
- **Sharp corner:** scale all of $\mathbf{w}, b$ by 10×. Boundary doesn't move; transition becomes a cliff. **This is the intuition for softmax temperature and for CFG scale — flag it for callback in both tracks.**

**DEMO B — "Watch the gradient be the chain rule" (page ~10, TN-1)**

- **Plot:** TN-1's graph (the canonical visual), each edge labeled with its current weight; each node showing forward value; **a second set of red labels flowing right-to-left showing $\delta$ values.** A "step" button.
- **Controls:** $\eta$ slider (log scale, $10^{-3}$ to $10^{1}$); a "step" button; reset.
- **JS computes:** the exact TN-1 forward/backward of §9.2, from the exact starting constants. On step: full backprop, SGD update, re-render, append to the loss curve.
- **The insight:** the red numbers are *the same chain rule he did by hand*, and they move **right to left**, and each node's red number is built only from the red number to its right times a local slope. **Backprop is not an algorithm to memorize; it is bookkeeping that reuses the number you already computed.**
- **The break:** set $\eta = 3$ and step. Loss oscillates, then explodes. Set $\eta = 0.001$: 200 steps, barely moves. He *feels* the learning-rate trade-off in 20 seconds — a thing that costs most people a week of real experiments.
- **Callback:** the $\delta$ at the hidden layer is already tiny at $\tanh$ saturation. **This is the vanishing-gradient demo, prepaid.** When that concept arrives 15 pages later, reopen this demo — don't build a new one. That's interleaving for free.

**DEMO C — "The memory budget" (recurring, first at ~page 18)**

- **Plot:** a stacked horizontal bar, 0→128 GB, with a hard line at 128 (his Spark). Segments: weights / gradients / optimizer state / activations / KV cache.
- **Controls:** parameter count $P$ (log slider, 1M→200B); precision (fp32 / bf16 / fp8 / int4); method (full fine-tune / LoRA / inference-only).
- **JS computes** (this is the exact arithmetic the course reuses everywhere):
  - weights $= P \times \text{bytes\_per\_param}$
  - gradients $= P \times \text{bytes}$ (full FT) or $\approx 0$ (LoRA — **frozen base has no grad**)
  - Adam state $= 2 \times P \times 4$ bytes (fp32 moments) — full FT only
  - LoRA params $= 2 \cdot r \cdot d \cdot n_{\text{layers}} \cdot n_{\text{matrices}}$
  - bar turns red past 128 GB
- **The insight:** he drags to 7B / fp32 / full FT → **~112 GB**, nearly the whole machine, for a *small* model. Switches to bf16+LoRA → **~14 GB**. *LoRA's win is not mainly the parameter count — it's that it deletes the optimizer state and the gradients, which were 3× the weights.* That is the actual reason LoRA exists and it is almost always taught wrong.
- **Why this demo is special:** it's about **his machine**. He can verify it with `nvidia-smi`. A demo the learner can check against physical reality is worth ten that he can't.

### 7.3 The bandwidth insight — the best predict-then-check in the course

Verified DGX Spark specs: **GB10 Grace Blackwell, 128 GB unified LPDDR5x, 273 GB/s, ~1 PFLOP FP4, 31 TFLOPS FP32.** NVIDIA positions it for fine-tuning up to ~70B and inference up to ~200B.

The ratio is startling and it is *pedagogical gold*:

$$\frac{1 \times 10^{15}\ \text{FLOP/s}}{273 \times 10^{9}\ \text{B/s}} \approx 3{,}663\ \text{FLOP per byte}$$

To be compute-bound, you must do ~3,663 floating-point operations for every byte you move. **Single-token decoding does about 2.** It is memory-bound by a factor of ~1,800.

**The predict-then-check:** *"Your Spark does 1 petaFLOP. A 7B model in bf16 needs ~14 GFLOP per token — so, naively, ~71,000 tokens/sec. Predict what you'll actually get."* Then:

$$\text{tokens/s} \le \frac{273\ \text{GB/s}}{14\ \text{GB/token-pass}} \approx 19.5\ \text{tokens/s}$$

**He can run this on his own machine and watch it land near 19.5**, ~3,600× below the FLOP-based guess. Nothing else in the course will make "the bottleneck is memory bandwidth, not arithmetic" stick like that. It reframes quantization, batching, KV cache, and speculative decoding as *one* idea — and it's grounded in hardware he owns. **Flag to the hardware/systems brief: this number must be consistent everywhere.**

---

## 8. Quiz design

### 8.1 The governing principle (and it overturns the obvious approach)

**Transfer-appropriate processing.** Retrieval benefits appear when practice format matches target format. Fact quizzes → fact recall, and produce **no** higher-order gain; learners who practice facts and are tested on higher-order questions score **indistinguishably from those who never practiced at all**.

Our target format is: *fine-tune a model for your own application and diagnose it when it misbehaves.*

**Therefore every question must be one of these six types.** No vocabulary. Not one.

| Type | Example |
|---|---|
| **Predict the number** | "TN-1, $\eta$=0.1, after one step, does $\hat y$ go up or down, and by roughly how much?" |
| **Predict the shape** | "`x` is (32, 128, 4096), `W_q.weight` is (4096, 4096). What's `x @ W_q.weight.T`? What breaks if you drop the `.T`?" |
| **Diagnose the failure** | "Loss goes to NaN at step 40. Three candidate causes are listed. Which is ruled out by the fact that it trained fine for 39 steps?" |
| **Budget it** | "Fine-tune a 13B model on your 128 GB Spark. Full FT in bf16 — does it fit? Show the arithmetic." |
| **Read the code** | "This training loop is missing `optimizer.zero_grad()`. What does the loss curve look like? *Why that shape specifically?*" |
| **Transfer across tracks** | "You know CFG from diffusion. What's the closest analogue in LLM sampling, and where does the analogy break?" |

### 8.2 Distractor design — target the actual misconception

A distractor is only useful if a **specific wrong model** produces it. Random wrong answers test reading. Diagnostic distractors *find the bug in his head.* Every distractor should be annotated in the spec with the misconception it detects, so feedback can address that misconception by name.

Misconception catalog with distractors — these double as the course's **warning boxes**:

| Misconception | Why it's seductive | Distractor that catches it | The correction that actually works |
|---|---|---|---|
| "The gradient points toward the minimum" | Everyone's mental image; usually roughly true in 2-D bowls | Offer "toward the minimum" as an option on an elongated-valley problem | **A demo on an anisotropic quadratic.** The gradient is ⊥ to the contour, which on a stretched valley points mostly *across* the valley, not down it. Show the zigzag. *Then* momentum is obviously the fix, not an arbitrary trick. Words alone never fix this one — he must see the zigzag. |
| "More layers = more capacity = better" | True-ish 1989–2014 | Ask which of 4 nets trains best; make the 40-layer plain net an option | Plain-net degradation is an **optimization** failure, not capacity. This is why residuals exist. |
| "Learning rate = how fast you learn" | The name says so | "10× the LR → 10× faster convergence" | Demo B at $\eta=3$. LR is **step size**, and past a curvature-set threshold you overshoot and *diverge*. Speed and stability trade off. |
| "Softmax outputs probabilities, so they're calibrated confidences" | They sum to 1 | "The 0.97 means 97% likely correct" | They're a normalized exponential of arbitrary logits. Show the same input at 10× logit scale → 0.9999. Callback to **Demo A's magnitude-sharpens-the-boundary insight.** |
| "Backprop is a different algorithm from the chain rule" | It's got its own name and its own chapter | "Backprop computes numerical derivatives" | It **is** the chain rule, with the multiplication order chosen so you pay once per layer. Show both orders; count the multiplies. |
| "LoRA works because fewer parameters = less memory" | Half true, and the half that's true is the boring half | "LoRA saves memory mainly because it has fewer weights" | Demo C. The weights barely shrink (base is still resident!). **Gradients and Adam state vanish** — those were 3× the weight memory. |
| "Diffusion models denoise an image" | The name; the visualization | "Each step removes some noise from the picture" | The net predicts the noise $\boldsymbol\epsilon$ (or the score); the *sampler* does the removing, and the schedule decides how much. He'll feel this — it explains why swapping schedulers in ComfyUI changes everything while the model is untouched. **Very high-value: connects directly to his existing practice.** |
| "The embedding dimension is the number of concepts" | Intuitive | "d_model=4096 → the model knows 4096 things" | Superposition: features are *directions*, not coordinates, and there are far more near-orthogonal directions than dimensions. |
| "Attention attends to words" | Every blog diagram | "The attention head looks up the relevant word" | It's a soft, weighted average over *value vectors* at positions, with weights from a query-key dot product. The dot product is the same one from page 2. **Thread callback.** |
| "Fine-tuning teaches the model new facts" | It's called *training* | "Fine-tune to make it know your database" | Mostly it teaches **form/behavior**; facts are unreliable this way. This misconception is **directly load-bearing for his stated goal**, so it needs a whole page, not a box — and an honest treatment of when to fine-tune vs. retrieve. |

### 8.3 Mechanics

- **2–4 questions per page**, not 10.
- **Answer required before feedback.** No peeking — the generation effect requires a committed attempt.
- **Feedback names the misconception**, doesn't just say "incorrect." "You picked B. That's the answer you get if you think the gradient points at the minimum. Here's the zigzag that shows it doesn't."
- **~30% of questions are callbacks** to earlier threads, ≥3 pages back. That's how spacing (g≈0.24 embedded) and interleaving (g≈0.34) get in essentially for free — no scheduler, just a rule that the spec enforces per page.
- **Difficulty is desirable, frustration isn't.** Target ~70–85% first-attempt success. Below ~60%, the page failed, not the learner — that's a spec bug and should be treated as one.

---

## 9. Recurring threads — the recommendation

**Recommendation: adopt all four candidates, but with explicitly different structural roles.** They're not four competing options; they're four different *kinds* of recurrence, and a course this long needs all four. The failure mode isn't having four — it's having four without assigned roles, so agents cite them inconsistently.

| Thread | Role | Recurs as |
|---|---|---|
| **Dot product** | **The atom.** The single arithmetic operation the whole subject is made of. | neuron → matmul → similarity → attention scores → CFG direction → LoRA's $BA$ |
| **TN-1, the tiny network** | **The object.** ← *the spine* | hand-arithmetic → JS demo → PyTorch → autograd → a "1-head attention" → LoRA target |
| **The chain rule** | **The verb.** | derivative refresher → backprop → autograd graph → BPTT → diffusion's score → why gradient checkpointing works |
| **The memory budget** | **The reality check** — and the bridge to his hardware. | Demo C → activations → KV cache → quantization → LoRA → "will it fit on the Spark?" |

**If forced to cut to one: keep TN-1.** It's the only one that's a *thing* rather than a *technique*, and §2.3's transfer failure is specifically cured by re-meeting the same object in new guises.

### 9.1 The dot product as the atom — the argument

The single most unifying sentence available to this course:

> **"A dot product asks: how much of this is in that? Every neural network is that question, asked billions of times, with the answers rearranged."**

$\mathbf{w}\cdot\mathbf{x} = \|\mathbf{w}\|\|\mathbf{x}\|\cos\vartheta$ — the neuron computes alignment; attention scores are alignment; embedding similarity is alignment; a matmul is just many dot products stacked. If the learner ends the course owning the dot product *geometrically*, he can reconstruct most of the rest. Introduce it on page 1–2 with real trig (he has trig — **use it, it's the one advanced-ish tool he definitely still has**), and call back **every single time** one appears. Literally every time. That drumbeat is the interleaving.

### 9.2 TN-1: the spine — full specification

> **⚠️ CORRECTED — use `constants.md` §8, not the numbers below (D-02/D-13, verified to 50 d.p.).** Seven of this
> section's printed values are last-digit mis-roundings (the brief propagated 4-d.p. intermediates):
> `L 0.9870 → 0.9869` · `∂L/∂W₂ −0.5022 → −0.5021` · `∂L/∂a₁[1] 0.5646 → 0.5645` · `∂L/∂W₁[0][1] −0.7528 → −0.7527` ·
> `z₂(verify) −0.2433 → −0.2434` · `ΔL 0.1648 → 0.1646`. **And the "exactly zero" claim is FALSE in floating point
> (see the rewritten box below).** The autograd assertion must use **−0.7527** (at `atol=1e-4`, −0.7528 has only a
> 10% margin; −0.7527 gives 30×). **TN-1 now also has a frozen SECOND input `x=[0.60,−0.20]` (`constants.md` §8.7)**
> carrying the gradient-spread beat (**10.22×**, no dead unit) — hand it to the "therefore Adam" page.

**The architect should treat `constants.md` §8's numbers as canonical and hand them to every relevant agent verbatim.**

**Architecture:** 2 → 2 (tanh) → 1 (sigmoid). **9 parameters.** ($W_1$: 4, $\mathbf{b}_1$: 2, $W_2$: 2, $b_2$: 1.)

**Canonical initial state:**

$$W_1 = \begin{pmatrix} 0.5 & -0.3 \\ 0.8 & 0.2 \end{pmatrix},\quad \mathbf{b}_1 = \begin{pmatrix} 0.1 \\ -0.1\end{pmatrix},\quad W_2 = \begin{pmatrix} 0.6 & -0.9 \end{pmatrix},\quad b_2 = 0.2$$

$$\mathbf{x} = \begin{pmatrix}1.0 \\ 2.0\end{pmatrix}, \qquad y = 1$$

**Forward:**

$$z_{1,1} = 0.5(1.0) + (-0.3)(2.0) + 0.1 = 0.5 - 0.6 + 0.1 = \mathbf{0.0}$$
$$z_{1,2} = 0.8(1.0) + 0.2(2.0) - 0.1 = 0.8 + 0.4 - 0.1 = \mathbf{1.1}$$
$$a_{1,1} = \tanh(0.0) = \mathbf{0.0}, \qquad a_{1,2} = \tanh(1.1) = \mathbf{0.8005}$$
$$z_2 = 0.6(0.0) + (-0.9)(0.8005) + 0.2 = \mathbf{-0.5205}$$
$$\hat{y} = \sigma(-0.5205) = \frac{1}{1 + e^{0.5205}} = \frac{1}{2.6828} = \mathbf{0.3727}$$

**Loss (binary cross-entropy):** $\mathcal{L} = -\ln(0.3727) = \mathbf{0.9870}$ nats.

> **⚠️ REWRITTEN — "exactly zero" is FALSE in floating point (D-13, constants §8.4 has the MANDATED verbatim
> framing).** $z_{1,1} = 0.5 - 0.6 + 0.1$ is *not* zero in binary: **float64 → `2.78e-17`, float32 → `-2.24e-08`,
> and torch f32 prints `dW2 = [1.4020191e-08, -0.5021...]`.** By hand you get exactly 0; torch prints `1.4e-08`;
> **both are right** — it is the textbook float non-associativity demo (the logsumexp lesson, arriving three pages
> early, for free). The gradient is zero *to seven decimals*, the unit is dead, the lesson stands — **but a course
> that promised the learner "you will see 0.0" on his own box breaks its spine.** $\tanh'(0)=1$, so the upstream
> gradient flows fine (two contrasting lessons in one example). **Preserve the constants; fix the claim.**

**Backward.** The grace moment — BCE ∘ sigmoid collapses to:

$$\frac{\partial \mathcal{L}}{\partial z_2} = \hat{y} - y = 0.3727 - 1 = \mathbf{-0.6273}$$

$$\frac{\partial\mathcal{L}}{\partial W_2} = \frac{\partial\mathcal{L}}{\partial z_2}\mathbf{a}_1^\top = -0.6273 \begin{pmatrix} 0.0 & 0.8005\end{pmatrix} = \begin{pmatrix} \mathbf{0.0} & \mathbf{-0.5022}\end{pmatrix}$$

$$\frac{\partial\mathcal{L}}{\partial b_2} = \mathbf{-0.6273}$$

$$\frac{\partial\mathcal{L}}{\partial \mathbf{a}_1} = W_2^\top \frac{\partial\mathcal{L}}{\partial z_2} = \begin{pmatrix}0.6 \\ -0.9\end{pmatrix}(-0.6273) = \begin{pmatrix}-0.3764 \\ 0.5646\end{pmatrix}$$

$$\boldsymbol{\delta}_1 = \frac{\partial\mathcal{L}}{\partial\mathbf{z}_1} = \frac{\partial\mathcal{L}}{\partial\mathbf{a}_1}\odot(1 - \tanh^2(\mathbf{z}_1)) = \begin{pmatrix}-0.3764 \times 1.0 \\ 0.5646 \times 0.3592\end{pmatrix} = \begin{pmatrix}\mathbf{-0.3764} \\ \mathbf{0.2028}\end{pmatrix}$$

($\tanh'(0) = 1 - 0^2 = 1$; $\tanh'(1.1) = 1 - 0.8005^2 = 1 - 0.6408 = 0.3592$.)

$$\frac{\partial\mathcal{L}}{\partial W_1} = \boldsymbol{\delta}_1\mathbf{x}^\top = \begin{pmatrix}-0.3764 \\ 0.2028\end{pmatrix}\begin{pmatrix}1.0 & 2.0\end{pmatrix} = \begin{pmatrix}-0.3764 & -0.7528 \\ 0.2028 & 0.4056\end{pmatrix}$$

**SGD step, $\eta = 0.1$:**

$$W_2 \leftarrow \begin{pmatrix}0.6 & -0.8498\end{pmatrix},\quad b_2 \leftarrow 0.2627,\quad W_1 \leftarrow \begin{pmatrix}0.5376 & -0.2247 \\ 0.7797 & 0.1594\end{pmatrix},\quad \mathbf{b}_1 \leftarrow \begin{pmatrix}0.1376 \\ -0.1203\end{pmatrix}$$

**Verification forward pass (the payoff — this is what the demo must reproduce):**

$$z_{1,1} = 0.2258,\ z_{1,2} = 0.9783 \Rightarrow a_{1,1} = 0.2221,\ a_{1,2} = 0.7523$$
$$z_2 = -0.2433 \Rightarrow \hat{y} = \mathbf{0.4395}$$
$$\mathcal{L} = -\ln(0.4395) = \mathbf{0.8222}$$

**$\hat y$: 0.3727 → 0.4395 (toward the target of 1). Loss: 0.9870 → 0.8222, down 0.1648.** *It learned.* Nine numbers, one step, by hand, and he watched it work.

> **Note $W_{2,1}$ barely moved** (gradient is zero to seven decimals — torch prints `1.4e-08`, not `0.0`; see the rewritten box above — because $a_{1,1}\approx0$). Point at it. That's the lesson from the box above, now visible in the *result*.

**Mandated reuse:**
- **Page ~3–5:** by hand (above).
- **Page ~6:** PyTorch, `nn.Linear(2,2)` / `nn.Linear(2,1)`, weights set to these exact constants. Assert forward output ≈ 0.3727. **This is the Early Real Thing.**
- **Page ~8–10:** `loss.backward()`; `assert torch.allclose(model[0].weight.grad, torch.tensor([[-0.3764, -0.7527],[0.2028, 0.4056]]), atol=1e-4)` — **note `−0.7527`, NOT −0.7528** (constants §8.5: −0.7528 survives only because `atol=1e-4` absorbs the rounding error; at `atol=1e-5` it fails). **Autograd reproduces his hand arithmetic, and he proves it** — the moment autograd stops being magic.
- **Page ~7:** the row/column bridge — note `nn.Linear(2,2).weight.shape == (2,2)` here is ambiguous; **use `nn.Linear(2,3)` for the shape demo so (out, in) = (3,2) is unmistakable.** *(Small but real spec trap — a 2→2 layer cannot demonstrate the convention.)*
- **Page ~12:** train it to convergence.
- **Attention chapter:** recast TN-1's hidden layer as a 1-head, 2-dim attention computation — same dot products, new name.
- **LoRA:** decompose TN-1's $W_1$ as $BA$ with $r=1$. Nine parameters → count the LoRA params by hand. The whole idea, at a scale he can hold in his head.

### 9.3 Thread hygiene for a fan-out build

Every thread beat must be tagged in the spec: `[THREAD:TN1 beat 4/9]`. Each beat states (a) it's the same object, (b) what's new about this encounter, (c) which prior beat to reopen. **Without explicit tagging, fan-out agents will silently re-introduce the thread as if new, and the interleaving benefit — the entire reason threads exist — evaporates.** Assume they will; make the tag mandatory and checkable.

---

## 10. The trunk → two-track split

### 10.1 Make the split a payoff, not a fork

**The framing that works: "you already built both."**

The split should land as a *reveal*, not a menu. By the time it arrives, the learner has built every component both tracks use. The split page's job is to say: *the trunk gave you a machine that maps vectors to vectors and learns by gradient descent. Point it at "predict the next token" and you get an LLM. Point the same machine at "predict the noise that was added" and you get a diffusion model. **The machine didn't change. The target did.*** Same transformer blocks, same attention, same AdamW, same LoRA — **different loss function.**

That is *true*, it is *deep*, and it is the strongest single sentence in the course. It converts a structural inconvenience (two tracks) into the payoff for everything before it. Build toward it explicitly: the trunk should foreshadow it from ~page 10 ("hold on to this — the only thing that will change is what we ask it to predict").

### 10.2 What MUST be shared before the split

Non-negotiable, or the tracks will duplicate and drift:

1. Tensors, shapes, the two conventions, the shape ribbon
2. The dot product, geometrically
3. Linear layers, activations, TN-1 complete
4. Loss functions **as log-likelihoods** — *critical*: both tracks' losses are negative log-likelihoods of different models. Teach the frame once, in the trunk, or you'll teach it badly twice.
5. Backprop + autograd
6. SGD → momentum → Adam/AdamW
7. Init, normalization, residual connections
8. **The transformer block** — both tracks use it in 2026 (DiT/MMDiT are transformer-based). Putting the transformer *after* the split would be a serious architectural error: it would duplicate ~6 pages and guarantee drift.
9. Embeddings and the residual stream
10. Tokenization *or* latent encoding, framed together as **"how continuous vectors get made from discrete/high-dim stuff"**
11. The memory budget + precision + quantization
12. LoRA/PEFT **mechanics** (the math), before either track's application of it
13. The training loop, dataset/dataloader, checkpointing

**Track-specific after the split:**
- **LLM:** causal masking, KV cache, sampling (temp/top-k/top-p), instruction tuning, chat templates, RLHF/DPO family, eval
- **Diffusion:** forward noising, score/ε-prediction, samplers/schedulers, CFG, VAE latents, conditioning/cross-attention, ControlNet/adapters, eval

### 10.3 Keep the tracks in dialogue (this is what makes equal depth work)

Equal-depth parallel tracks risk feeling like two shorter courses stapled together. Fix: **mandatory cross-references at matched beats.** Structure them as *sibling pages* — the same page number in each track covers the analogous concept:

| Concept | LLM | Diffusion |
|---|---|---|
| What's predicted | next token (softmax over $V$) | the noise $\boldsymbol\epsilon$ (or score) |
| The loss | cross-entropy | MSE on $\boldsymbol\epsilon$ |
| Steering at inference | temperature / top-p | CFG scale / scheduler |
| Conditioning | the prompt, in-context | cross-attention on text embeddings |
| Iterative generation | autoregressive decode, $S$ steps | denoising, $T$ steps |
| The KV cache analogue | KV cache | — (**genuinely disanalogous — say so**) |

**The last row matters most.** Forced analogies are worse than no analogies. Where the correspondence breaks, *say it breaks and why* — that's where the real understanding lives, and an architect who lets agents smooth it over will produce two tracks of comfortable falsehoods. And note §8.1's cross-track question type exists precisely to exploit this table.

**Sequencing recommendation:** LLM track first. Not because it's more important — because his diffusion intuition is his *strongest asset*, and reading the LLM track first means he arrives at the diffusion track with a fresh formal frame to re-interpret things he already does by hand. Maximum "oh, *that's* what CFG was doing." Diffusion-first would waste that.

---

## 11. Integrating real code without a wall

### 11.1 The problem
A 60-line training script is a wall. The learner reads it top to bottom, understands each line, and cannot tell you what it does. The issue is that a script's *reading order* is not its *conceptual order*.

### 11.2 The rules

1. **Code grows by accretion; it is never revealed whole.** Page 6: 15 lines. Page 12: those 15 lines plus 8 more, **with the new lines highlighted and the old lines dimmed**. By page 24 he has a 60-line script he wrote in 6 sittings, and he knows every line because he *added* it for a reason he had at the time.
2. **Every code block has a "why this line exists" for each new line.** Not "what it does" — he can read Python. *Why it's there.* `optimizer.zero_grad()` — because gradients **accumulate** by default, which is a feature (grad accumulation for large batches on limited memory) that becomes a bug if you forget it. **That's a thread callback to the memory budget.**
3. **Code must be runnable exactly as printed**, with no elided imports. `# ...` in a code block is a defect.
4. **Every code block states its hardware cost** — "runs in 0.3 s on CPU" / "≈ 4 min, 18 GB on the Spark." He has good hardware; tell him what to expect so he can tell working from hung. A learner who can't distinguish "slow" from "broken" will thrash.
5. **Pin versions.** **PyTorch 2.13.0 (released 2026-07-08, requires Python ≥ 3.10)** — verified current stable as of today. State it once, up front, and note the release cadence (~3 minor releases/year) so he can reason about drift when it inevitably happens.
6. **Demo ↔ code parity.** When a JS demo and a PyTorch snippet compute the same thing, say so and show they agree numerically. TN-1's `0.3727` must appear in the hand-arithmetic, the demo readout, and the code output. **Three representations, one number** — that identity is the whole point of the spine thread.

### 11.3 The browser/code division of labor

| Use the browser demo when | Use real code when |
|---|---|
| The point is a *relationship* (how does output vary with this?) | The point is *it actually works* |
| Response must be instant to support dragging | Scale is the point (real model, real data) |
| Scale would obscure (2 params, not 7B) | The hardware is the point (Demo C's claims → `nvidia-smi`) |
| You want him to *break* it safely | You want him to own a real artifact |

**They must meet.** The best pages have both computing the same thing — the demo for the intuition, the code for the reality — with the numbers matching. That match is what tells him the browser toy and the real system are the *same subject*, which is exactly the §2.3 transfer problem, solved.

---

## 12. Momentum across 45–50 pages

### 12.1 The trough is real and it has a location

Motivation dips hardest where **the novelty is gone and the payoff isn't visible**. In this course that's predictably **pages 20–30**: past the thrill of the first training loop, deep in optimizers/normalization/regularization, and the LLM/diffusion payoff hasn't arrived.

**Three countermeasures:**

1. **Put the biggest milestone at ~page 24** — an end-to-end fine-tune of something small but *real*, on his data, on his hardware, **before the split.** He should be able to say "I fine-tuned a model" at the halfway mark. This single placement decision does more for completion than any amount of encouraging prose.
2. **Front-load the payoff *language*.** From ~page 10, every optimizer/normalization page carries one line naming the destination: "this is the exact optimizer you'll use on your LoRA in Part 5." Cheap, and it converts drudgery into visible scaffolding.
3. **Interleave a "his world" page every ~5 pages** in the trough — a page that explains something he *already sees in ComfyUI* using the machinery just learned. Why sampler choice changes output. Why the first steps matter most. **These are pure motivation with zero content cost**, because the explanandum is already in his head. This is the trough-killer, and it's available only because we know this learner.

### 12.2 Milestones

| Page | Milestone | Sayable as |
|---|---|---|
| 6 | First real code, matches hand arithmetic | "PyTorch is my arithmetic, faster" |
| 10 | Autograd reproduces his gradient | "I know what `.backward()` does" |
| 12 | First training loop converges | "I trained a network" |
| 18 | MNIST-scale on the Spark | "Scale is a parameter" |
| **24** | **End-to-end fine-tune** | **"I fine-tuned a model"** ← halfway |
| 32 | LLM fine-tune on his data | "I made an LLM mine" |
| 40 | Diffusion fine-tune on his data | "I made a diffusion model mine" |
| 45+ | Chooses the method for a new problem | "I can decide" |

### 12.3 Pacing rules

- **One page = one idea.** If it has two, it's two pages. Page count is free; confusion isn't.
- **No page is only prose.** Every page has a number, a demo, or a runnable thing.
- **Every part opens with a diagnostic** (§1.2) and closes with a synthesis that is a *task*, not a summary.
- **Never two derivation-heavy pages in a row** without something running in between.

---

## 13. Anti-patterns — a checklist the spec can be audited against

Any page failing these is a defect:

1. ❌ A symbol appears that isn't in the notation table or the page's Ledger
2. ❌ $W$ written as (in, out) anywhere
3. ❌ $\sigma$ used for anything but logistic sigmoid; $t$ used for sequence position
4. ❌ An equation with >3 symbols and no Symbol Ledger
5. ❌ A tensor op with no shape ribbon
6. ❌ A worked example that doesn't terminate in a decimal
7. ❌ A demo that doesn't compute the page's equation live
8. ❌ A demo whose equation isn't shown with live values substituted
9. ❌ A quiz question answerable from vocabulary
10. ❌ A distractor with no named misconception behind it
11. ❌ Content in a collapsible that a later page depends on
12. ❌ A collapsible title that teases rather than states
13. ❌ A thread beat that doesn't announce itself as one
14. ❌ Code with elided imports or `# ...`
15. ❌ A justification appealing to "germane load"
16. ❌ Re-explaining something the learner demonstrably owns (the −0.428 tax)
17. ❌ >4 pages since anything last ran
18. ❌ A number that contradicts the same number on another page (**TN-1's 0.3727; the Spark's 128 GB / 273 GB/s; ~19.5 tok/s**)

**#18 is the one a fan-out build will fail most often.** Recommend a shared `constants.md` that every agent must read and none may edit without a global pass.

---

## 14. Confidence and verification

**Verified this session (high confidence):**
- Expertise reversal meta-analysis: 176 ES / 60 studies / N=5,924; d = +0.505 and −0.428; PRISMA + metafor. **Caveat:** sources disagree on the journal — the ScienceDirect ISSN (0959-4752) and the pedocs record both indicate ***Learning and Instruction* vol. 98 (2025)**, while a secondary summary said *Educational Research Review*. **Cite as Tetzlaff et al., *Learning and Instruction*, 98 (2025); the ISSN is decisive.** Full text was paywalled (403), so effect sizes come from consistent secondary reports, not the PDF.
- Productive failure: Sinha & Kapur 2021, 53 studies / 166 comparisons, g = 0.36 (CI 0.20–0.51), g = 0.87 bias-corrected for conceptual/transfer.
- Spacing in maths: g = 0.28 overall / 0.43 isolated / 0.24 course-embedded (2025 *Educ Psych Rev*, 27 studies / 53 ES).
- Interleaving: g = 0.42 overall, 0.34 maths.
- Retrieval→transfer: d ≈ 0.4; transfer-appropriate processing; fact-quiz→higher-order ≈ no-practice.
- CLT: germane load removed from the additive equation in Sweller et al. (2019) → "germane processing"; falsifiability and measurement critiques as described.
- **`torch.nn.Linear` applies $y = xA^T + b$; `weight` shape `(out_features, in_features)`; input `(*, in_features)`.** Read directly from the PyTorch 2.13 docs. **High confidence — this is the load-bearing fact under §6.1.**
- **PyTorch 2.13.0, released 2026-07-08**, Python ≥3.10. Verified on PyPI, with version history (2.12.0 May 2026, 2.11.0 Mar 2026, 2.10.0 Jan 2026) consistent with ~3 minor releases/year.
- **DGX Spark: GB10 Grace Blackwell, 128 GB unified LPDDR5x, 273 GB/s, ~1 PFLOP FP4, 31 TFLOPS FP32**; NVIDIA positions it for fine-tuning ≤70B, inference ≤200B.

**My own arithmetic (checked, moderate-high confidence):**
- All TN-1 numbers in §9.2 — computed and verified by hand this session, including the second forward pass. **Recommend one independent recomputation** before they're frozen as course constants, since ~9 pages will depend on them.
- 3,663 FLOP/byte; ~19.5 tok/s for 7B bf16 at batch 1. Arithmetic is simple; the *model* (weights-read-bound decode) is a standard first-order roofline and ignores KV cache traffic and achievable-vs-peak bandwidth (real bandwidth is typically 70–85% of peak). **Present ~19.5 as an upper bound; expect measured ~14–18.** Say so, or the predict-then-check backfires when his measurement comes in under.
- Memory-budget figures (7B fp32 full FT ≈ 112 GB; bf16 LoRA ≈ 14 GB) are first-order and exclude activations/fragmentation. **Fine as teaching numbers; label them as estimates.**

**Not verified / architect should confirm with other briefs:**
- Specific 2026 model names, sizes, and library APIs (transformers/peft/diffusers versions) — **deliberately out of scope here**; I've kept this brief free of them so it doesn't rot. Where I mention LoRA/DiT/MMDiT, treat as structural, not version-pinned.
- Page numbers throughout are *proportional targets* for a 45–50 page course, not commitments.

---

## 15. What the architect must reconcile with other briefs

1. **The notation table (§6) must be adopted verbatim by every brief and every agent, or the build fails.** This is the top structural risk. If the maths or LLM/diffusion briefs use a conflicting convention, **this brief should lose the argument on any individual symbol — but the resolution must be global, ratified once, and frozen before fan-out.** Recommend the architect publish the ratified table as `notation.md` and make it a required read.
2. **`constants.md`.** TN-1's parameters and outputs, the Spark's specs, the memory-budget formulas, the tok/s figure. One source, no local edits. Anti-pattern #18 is otherwise unavoidable.
3. **The transformer must be in the trunk, not the tracks** (§10.2 item 8). If another brief assumes the transformer lives in the LLM track, that must be resolved *before* page allocation — it's a ~6-page duplication and a guaranteed drift point.
4. **$\sigma$ collision.** The diffusion brief will want $\sigma_t$ for noise std. §6.4 reserves $\sigma$ for the logistic sigmoid and assigns $s_t$. Needs explicit sign-off from the diffusion author.
5. **$t$ vs $i$.** §6.3 reserves $t$ for diffusion time and uses $i$ for sequence position, against transformer convention. Needs sign-off from the LLM author. **Non-negotiable in one respect:** whatever is chosen must be collision-free on DiT pages.
6. **The ~19.5 tok/s figure and the 273 GB/s roofline** must match the hardware/systems brief exactly, including the upper-bound caveat.
7. **Page-6 code.** Requires that the maths brief not demand a prerequisite chain longer than 5 pages. If it does, that's a direct conflict with §4.1 and **I'd argue §4.1 wins** — prerequisites can be collapsibles and just-in-time refreshers; the Early Real Thing cannot be deferred without changing what kind of course this is.
8. **Quiz format (§8.1) forbids vocabulary questions.** If another brief specifies terminology checks, the transfer-appropriate-processing evidence says they're not merely low-value but *budget-consuming with zero transfer*. Worth an explicit ruling.
