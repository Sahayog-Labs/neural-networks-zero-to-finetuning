<!--
================================================================================
REPAIR LOG — spec-part1.md restored against decisions.md §D-21a (2026-07-16)
================================================================================
DEFECT #1 (the big one): the fan-out replaced canonical pages 08–11 (probability,
softmax, XOR/collapse, activations) with a duplicate backprop/autograd/one-step
arc that already lives in Part II (pages 14–15, 17). Restored per D-21a.

WHAT MOVED WHERE (salvage map; old file page → new home):
- old 01 what-is-a-network      → 01 (kept; build O→S per D-21a; cross-refs repaired)
- old 02 vectors-and-shapes      → 02 (kept; build O→S per D-21a; SwiGLU/⊙ ref 04→11)
- old 03 the-dot-product         → 03 (kept; trig-payoff forward-refs corrected to
                                   33 AND 54 — was 53; attention-score ref → 29/30)
- old 05 derivatives-and-descent → 04 (salvaged; θ-vs-x flip promoted to a named
                                   beat; steepest-ascent still proved FROM p.3;
                                   rigorous η_crit=2/λmax + saddles DEFERRED to p.13)
- old 08 the-chain-rule          → 05 (salvaged wholesale; refs 06→06, 09→14)
- old 06 tn1-early-real-thing    → 06 (kept forward-only; float-beat pointer 10→15;
                                   added "how does it learn? → Part II" hook)
- old 07 row-or-column           → 07 (kept; back-pass refs 08–09 → 05 + Part II 14)
- NEW probability/Gaussian/μ+σε  → 08 (WRITTEN FRESH — the diffusion-track dependency
                                   the fan-out deleted; foundations §6)
- NEW logs/logsumexp/softmax/CE  → 09 (WRITTEN FRESH; salvaged the logsumexp/CE
                                   fragments the fan-out had scattered; foundations §7–8)
- old 04 neuron-and-nonlinearity → SPLIT: hyperplane + linear collapse + XOR → 10;
                                   activation taxonomy + ReLU-won + SwiGLU + UAT → 11
- old 04 activation menu + UAT   → 11 (salvaged; build O→S per D-21a)

EXPORT LIST FOR PART II (blocks removed here; integrator must confirm they landed):
  → PART II p.14 (Backprop: TN-1's nine gradients): from old 09 backprop-by-hand —
    the grace moment ∂L/∂z2 = ŷ−y = −0.6273 (§8.2); the full nine-gradient .box worked
    (§8.2, seven corrected roundings incl. −0.7527); the δ^(ℓ)=(W^{(ℓ+1)T}δ^{(ℓ+1)})⊙φ'(z)
    recursion + Symbol Ledger; the DEAD-UNIT beat (a_{1,1}=0 ⇒ ∂L/∂W2[0]=0 but tanh'(0)=1);
    the "TN-1 backward" viz.NetGraph.animateBackward demo; its 7-question quiz.
  → PART II p.15 (Autograd + the float that isn't zero): from old 10 autograd —
    the MANDATED "exactly-zero-is-false" .box key (§8.4, verbatim), value
    1.4020191230201817e-08, the atol=1e-4 assertion on −0.7527 (§8.5); the float32/64
    toggle + NN.gradCheck demo; the tn1.py .backward() accretion. NOTE §8.4 names p.15
    as this beat's HOME (constants §8.4 "page ~10" pointer is STALE — D-21a moves it to 15).
  → PART II p.17 (Optimizers → "therefore Adam"): from old 11 one-step-of-learning —
    the 10.22× gradient-spread beat on the second input [0.60,−0.20] (§8.7), incl. the
    "structural, 8.64× even at 1:1 features" honesty and the retired-12.7× warn.
  → PART II p.13/14 (placement integrator's call): the one-SGD-step-by-hand block
    (§8.3, L: 0.9869→0.8222, ŷ:0.3727→0.4395) — applies the nine gradients; natural at
    14 (right after they're derived) or as p.13's worked step.
  → THE FORK / PART II p.12 opener: the D-17 LLM/diffusion foreshadow ("the machine
    won't change, the target will") from old 11.

CROSS-PART SEAMS FLAGGED FOR PARALLEL AGENTS:
- p.09 underflow is now the trunk's FIRST "paper-math ≠ machine-math" beat and it now
  PRECEDES the autograd 1.4e-08 beat (p.15). constants §8.4's note "same lesson as the
  logsumexp page, arriving three pages early" was written when autograd=10/logsumexp=13;
  the order is now REVERSED (9 before 15). p.15 must forward-reference BACK to p.09, not
  call it "later."
- p.08 owns the FIRST teaching of reparameterization x=μ+σε. brief-diffusion §2 line 218
  ("this [VAE page] is the first place the course should teach it") is SUPERSEDED by D-21a
  p.8; the diffusion VAE page (53) reopens, not re-teaches.
- MSE-as-Gaussian-NLL is deliberately NOT on p.08/09 — p.12 owns it. Both fresh pages
  plant the seed and cite forward to 12.

COULD-NOT-FULLY-SPEC: none. All four restored owner-topics (8,9,10,11) are fully specced.
Build-model note: pages 01,02,11 set to S and 08,09,10 to O per D-21a's printed column,
overriding the old file's reconstructed O/S guesses (decisions.md > briefs).
================================================================================
-->

# BUILD SPECIFICATION — PART I: THE MACHINE (pages 01–11)

**Course:** *Artificial Neural Networks — From Algebra to Fine-Tuning on Your Own Box.* 65-page
interactive HTML course, static `file://`, vanilla JS + local KaTeX. One learner: adult, rusty-but-once-trained,
owns the DGX Spark the ground-truth was measured on, heavy ComfyUI video-diffusion user. Today: 2026-07-16.

**Scope of this spec (restored to D-21a §D-21a):** the trunk's opening arc — *the forward machine and the
mathematical vocabulary it runs on*. From "what a network is" (no notation) through the dot product, derivatives,
the chain rule, a real 9-parameter network run forward by pencil **and** PyTorch, the two shape conventions, the
probability/Gaussian handoff the diffusion track depends on, the softmax/cross-entropy objective the LLM track
depends on, and finally why a linear stack collapses and what the activation that saves it really buys. **Backprop
is NOT here — it is Part II's, and only Part II's (pages 14–15).** Part I teaches *forward + chain rule* only.

## How to read this spec
Each page entry gives: **filename · title · part · build model · track class · objectives · PREDICT prompt ·
section outline (every equation in the frozen notation, every worked number with its `constants.md` citation,
every misconception→`.box warn`, inline-vs-deepdive calls) · demos (primitives, plotted quantities, slider
ranges/steps/units, the exact `nn.js`/`viz.js` math, the intended "aha") · code artifact (`.box try`) · quiz
(5–8 Q, distractors keyed to named misconceptions, ≥1 numeric+tolerance) · thread touchpoints (tagged) ·
cross-references by final page number.**

## Frozen-file compliance (non-negotiable — see `notation.md §9` anti-patterns)
- **Every number cited below is verbatim from `constants.md`; its section is named.** Never launder an [INF]/[EST]/[MEA]/[DER]/[VP] tag into a fact.
- **Notation is `notation.md`.** $W$ is always $(d_{\text{out}}, d_{\text{in}})$. Geometric angle is $\vartheta$, **never** $\theta$ (params). $\sigma(\cdot)$ = logistic sigmoid, with an explicit argument; $\sigma_t$ (diffusion std) is absent from Part I. $\eta$ = learning rate only. $\mathcal{L}$ = per-example loss (nats); $J$ = batch-averaged. Optimizer step index is $k$, sequence position $i$ — **not $t$** (that is diffusion time, absent from Part I).
- **TN-1 numbers come from `constants.md §8` (two frozen inputs).** Do not recompute, do not round differently. The corrected last-digit roundings and the mandated "exactly-zero" framing are load-bearing — **but the backward pass they feed is Part II's**; Part I page 06 uses only the forward numbers (`§8.1`).
- **Template contract:** copy `TEMPLATE-example.html`. Boxes `.box key/rule/warn/worked/try`, `.deepdive`, `.milestone`, `.track-banner`. Primitives `makeCtrl`, `Plot`, `renderQuiz`, `eng`, `cssVar`/`vizUtils.cssVar`, `wireCodeblocks`; globals `NN.*` (incl. `worked221()`, `Mat` (`Mat.randn(n,m,rng,scale)`), `RNG` (seeded), `softmax`/`softmaxUnstable`, `attention`, `MLP`, `SGD`, `Adam`, `Trainer`, `gradCheck`, `makeDataset`, `xorData`) and `viz.*` (`NetGraph`, `Heatmap`, `TensorViz`, `Surface3D`, `Timeline`, `hl`). Demos compute the page's real equation live — nothing is faked. `<body data-page>` must equal the filename; wire the page script inside one `DOMContentLoaded`.

## Build model key (`O`/`S`) — **now taken verbatim from `decisions.md §D-21a`'s printed column**
**`O` = Opus build** (bespoke page: a flagship live-math demo or a spine beat — build fresh, protect it). **`S` = Sonnet build** (copies the template's demo primitives with page-specific parameters). §D-21a prints the per-page assignment; it **supersedes** the earlier reconstruction that had guessed 01/02 as O. Final Part I column: **01 S · 02 S · 03 O · 04 O · 05 O · 06 O · 07 S · 08 O · 09 O · 10 O · 11 S.** Because pages 1–51 are unaffected by the GRPO renumber, each page's final number equals its D-21a number.

## Part I page map (the spine of the arc — D-21a order)
| # | slug | one-idea | Early-Real-Thing anchor | build |
|---|---|---|---|---|
| 01 | what-is-a-network | a net is one function with knobs; a neuron cuts space in half | drag a boundary (no notation) | S |
| 02 | vectors-and-shapes | shape is the skill; $W$ is $(d_{\text{out}},d_{\text{in}})$ | Qwen3-8B FFN, carried to a number | S |
| 03 | the-dot-product | the atom: $\mathbf{w}\cdot\mathbf{x}=\lVert\mathbf w\rVert\lVert\mathbf x\rVert\cos\vartheta$; near-orthogonality | high-dim orthogonality collapse | O |
| 04 | derivatives-and-the-flip | gradient = steepest ascent (proved from p.3); the $\vartheta$-vs-$x$ flip | LR→NaN in three drags | O |
| 05 | the-chain-rule | multiply along paths, add across paths | gears; residual $+1$ preview | O |
| 06 | tn1-early-real-thing | **TN-1 forward → 15 lines of PyTorch, matching to the digit. NO backprop** | **the page-6 moment** | O |
| 07 | row-or-column | $W$ is $(d_{\text{out}},d_{\text{in}})$ in *both* conventions | `nn.Linear(2,3)` | S |
| 08 | probability-and-the-gaussian | distributions/expectation/sampling; $x=\mu+\sigma\varepsilon$ you can differentiate | the same $\varepsilon$ cloud, re-stretched | O |
| 09 | logs-softmax-cross-entropy | underflow felt first; softmax; CE = NLL; $\partial\mathcal L/\partial z=p-y$ | $[2.0,1.0,0.1]\to0.417030$ | O |
| 10 | neuron-xor-collapse | a linear stack collapses; the nonlinearity is the whole ballgame | XOR by hand in 4 lines | O |
| 11 | activations-and-limits | why ReLU won ($0.25^{32}$); SwiGLU; UAT honestly | activation zoo + gradient flow | S |

---

# 01 — what-is-a-network.html
**Title:** "What a Neural Network Actually Is" · **Part I — The Machine** · **build: S** · no track class.

> **⚠️ HARD CONSTRAINT: this page uses NO mathematical notation, by design.** No $W$, no $\sigma$, no subscripts,
> no equations. The learner decides on this page whether the course respects him. Prose + one draggable demo only.
> The single Greek letter allowed is none. (This is the one page exempt from the Symbol-Ledger rules because it
> introduces no symbols.) Notation begins on page 02.

**Learning objectives**
1. State that a neural network is one function with millions of adjustable knobs, and that *training turns the knobs* (input fixed, knobs variable) — the $\vartheta$-vs-$x$ flip, said in English (its symbolic form is page 04).
2. See a single neuron as a line that cuts a plane in half, and feel that its *direction* sets orientation while its *magnitude* sets sharpness.
3. Understand "a network is functions composed" without a composition symbol.
4. Leave able to predict what dragging a weight does before dragging it.

**PREDICT (opens the page).** "Below is one artificial neuron separating blue points from red. If you increase
*both* its weights by the same large factor — steepening it — does the dividing line (a) move to a new place,
(b) stay put but get sharper, or (c) rotate? Commit to one, then drag." *(Consolidation after Demo A resolves it:
answer (b) — magnitude sharpens, direction rotates, bias moves. `brief-pedagogy §7.1 Demo A`.)*

**Section outline**
- *One machine, many knobs.* A network is a single function: numbers in, numbers out, with a few billion tunable knobs in between. Inference = knobs fixed, feed inputs. Training = **flip it**: the inputs are fixed (they're your data), the knobs are what you change, and you change them to make one output number — how wrong the machine is — smaller. State the flip in one sentence; it is the conceptual pivot of the whole course (`brief-foundations §1`). No equation. (Page 04 gives this flip its symbols: the derivative that matters is with respect to the *knobs*, not the input.)
- *A neuron is a ruler and a bend.* Draw the picture: a neuron measures how much an input points along a direction it cares about (a "ruler"), then squashes the reading (a "bend"). The ruler's **direction** decides which way the dividing line faces; the ruler's **length** decides how abruptly the output flips from "no" to "yes"; a separate offset slides the line without turning it. (`brief-foundations §9` geometric intuition, stated wordly.)
- *A network is neurons feeding neurons.* Layers are not different kinds of thing — every layer is the same move (measure, bend), and the output of one is the input of the next. "Deep" just means many in a row. Forward-ref: *"hold on to this — later, the only thing we'll change is what we ask the machine to predict"* — the LLM/diffusion split, planted (`decisions.md D-17 item 1`; cross-ref the fork and page 11).
- **`.box key`** — "A neural network is one function with millions of knobs. Training is turning the knobs so the machine is less wrong. Everything else in this course is *how you turn them fast enough to matter.*"
- **Misconceptions → `.box warn`:**
  - *"The network looks up or stores the answer."* It is a deterministic function evaluation. Any randomness in an LLM is added *afterward*, at sampling. **Correction that lands:** same prompt at temperature 0 twice → identical; at temperature 0.8 → different; the net never changed (`brief-foundations §1`). (Names a phenomenon he already sees in ComfyUI's seed; the mechanism is page 09's temperature.)
  - *"Layers are separate programs."* One function, no internal boundary (`brief-foundations §1`).
- **`.deepdive`** (title states content): "Why one output and billions of inputs forces the shape of everything later." One paragraph: loss maps billions of knobs → one number, and that asymmetry is *why* the gradient is computed backward. Seed only; the "backward is forced" payoff is page 05 (chain rule) and Part II's backprop. (Collapsible carries no continuity dependency — page 05 restates it inline.)

**Demo A — "The neuron is a ruler and a bend"** *(flagship; the whole page rests on it).* Build S (uses template primitives; the page carries no bespoke math beyond the live sigmoid).
- **Primitive:** raw `<canvas>` + `makeCtrl` (no `Plot` — this is a shaded 2-D field). Optionally `viz` heat shading.
- **Plot:** a 2-D plane, ~40 fixed scattered points in two classes (seed the layout so it's reproducible; use `new NN.RNG(7)` or `NN.makeDataset('blobs', …)`). Background shaded blue→red by the neuron's output at every pixel. A drawn dividing line where output = ½. Live-display classification accuracy over the 40 points as a percentage.
- **Controls (3, the named-max):** weight-x `w₁` [−3, 3] step 0.1; weight-y `w₂` [−3, 3] step 0.1; offset `b` [−3, 3] step 0.1. *(Labels use plain words "horizontal weight / vertical weight / offset," not symbols — this page is notation-free.)*
- **Exact math (JS, live every frame):** for each pixel $(x_1,x_2)$: `z = w1*x1 + w2*x2 + b`; `a = 1/(1+Math.exp(-z))`; color = lerp(blue, red, a). This is the logistic sigmoid computed directly — it is the page's real equation even though the symbol is withheld. Accuracy = fraction of the 40 points whose `a>0.5` matches their class.
- **The aha (engineer it):** dragging `b` **translates** the line; dragging `w₁,w₂` **rotates** it; scaling both weights up **sharpens** the transition into a cliff without moving the line. Add a "×10" button that multiplies `w₁,w₂,b` by 10: boundary stays, transition becomes a wall. *Two facts for one drag; the second (magnitude = sharpness/confidence) is the seed for temperature, logit scale, and CFG.* Flag it in the readout: "Notice: steeper isn't a different boundary — it's a *more confident* one." (`brief-pedagogy §7.1`; callback targets named on pages 03, 09 temperature, 60 CFG.)
- **Reset** to a defensible default (`w₁=1, w₂=0, b=0`) that visibly under-fits, inviting the drag.

**Code artifact (`.box try`):** none yet — page 01 is browser-only by design (`brief-pedagogy §4.2`: page 1 is a *demo he can drag*, not his own code; his code starts page 06). The `.box try` here points *at Demo A*: "Drag until you separate the two clusters, then hit ×10 and watch confidence, not the boundary, change."

**Quiz (5; all prediction/diagnosis, zero vocabulary — `brief-pedagogy §8.1`)**
1. *(predict)* You scale a neuron's two weights up by 5× and leave the offset ratio fixed. The dividing line: **[stays put, transition sharpens ✓ / moves toward the origin / rotates 45°]**. Distractor "moves" ← the *"weights set position"* confusion; distractor "rotates" ← conflating magnitude with direction.
2. *(diagnose)* Same LLM prompt, temperature 0, run twice, different answers. Most likely cause: **[the network is nondeterministic internally / you did not actually set temperature 0 or seed ✓ / the weights drifted]**. Distractor "nondeterministic internally" ← the *"net stores/rolls the answer"* misconception.
3. *(predict)* Increasing only the offset `b` **[rotates / translates ✓ / sharpens]** the boundary.
4. *(concept, applied)* "Deep" most nearly means: **[each layer solves a different task / the same measure-and-bend move repeated many times, each feeding the next ✓ / more knobs per neuron]**.
5. *(transfer, his world)* In ComfyUI, changing the seed changes the image but not the model file. The closest idea on this page is: **[training changed the weights / the randomness is added at sampling, outside the deterministic function ✓ / the boundary moved]**.

**Thread touchpoints**
- `[THREAD:dot-product beat 0/…]` — the "ruler" *is* the dot product, named geometrically without the symbol; page 03 gives it its name and its trig. Announce: "the thing you called a ruler has a name — the dot product — and we meet it next page."
- `[THREAD:memory beat 0]` — one line only: "these knobs are the memory story too; on page 18 you'll count them against your own box." No numbers yet.

**Cross-references:** → 03 (dot product names the ruler), → 04 (the flip, in symbols), → 11 & the fork (the "target is the only thing that changes" plant), → 18 (memory ledger), → 09 & 60 (magnitude→sharpness callback: temperature, CFG).

---

# 02 — vectors-and-shapes.html
**Title:** "Vectors, Matrices, Tensors — Shape Is the Skill" · **Part I — The Machine** · **build: S** · no track class.

**Learning objectives**
1. Read a tensor's shape tuple and apply the matmul cancellation rule $(m,n)@(n,p)\to(m,p)$.
2. State and *use* the course invariant: $W$ is always $(d_{\text{out}}, d_{\text{in}})$.
3. Predict a broadcasting outcome and spot the silent `(64,)`-vs-`(64,1)` bug.
4. Carry the multiply rule to a real number on the anchor model (Qwen3-8B's FFN) and read the memory that implies.

**PREDICT.** "You add a bias of shape `(4096,)` to a batch of activations of shape `(32, 128, 4096)`. Does it
(a) error, (b) add the bias to every row correctly, or (c) run but silently produce a `(32,128,4096,4096)` monster?
Commit, then check the shape tracer." *(Resolves to (b); the monster answer is the `(64,)`-vs-`(64,1)` trap on a
different shape — set up the warn box.)*

**Section outline** *(notation begins here; every symbol is from `notation.md §4`, and this page installs the shape ribbon `notation.md §7`)*
- *Shapes, with units.* Scalar `()`; vector $\mathbf{x}\in\mathbb{R}^{d}$, shape `(d,)`; matrix $W\in\mathbb{R}^{m\times n}$, shape `(m,n)`; tensor rank ≥3, e.g. an activation `(B, S, d_model)` (`brief-foundations §2`). Say it bluntly: **shape bookkeeping is the #1 practical skill in this course — not calculus.** Every fine-tuning error he'll hit is a shape error or an OOM.
- *The one rule.* $$(AB)_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj},\qquad A\in\mathbb R^{m\times n},\,B\in\mathbb R^{n\times p}\Rightarrow AB\in\mathbb R^{m\times p}.$$ **Symbol Ledger required** (>3 symbols): $A$ `(m,n)` from left operand; $B$ `(n,p)` from right; inner $n$ **annihilates**; result `(m,p)`. Teach as the cancellation game `(m,[n)@(n],p)→(m,p)`. FLOP cost = $2mnp$ (count mul + add separately, vendor convention) — establish it here; the roofline pages depend on it (`constants.md §5`; cross-ref 43).
- **THE INVARIANT → `.box rule`:** "$W$ is $(d_{\text{out}}, d_{\text{in}})$. Always. In the maths, in the code, in both conventions. Rows of $W$ are output neurons; columns are inputs." (`notation.md §0 invariant 1`.) This is the sentence page 07 will pay off.
- *Shape ribbon* (mandatory format, `notation.md §7`): show the canonical block once —
  ```
  x        (B=32, S=128, d_model=4096)
  W        (12288, 4096)        # (out, in) — ALWAYS
  h = x @ W.T                    (32, 128, 12288)
  ```
  Announce that every multi-step tensor op in this course carries this ribbon inline.
- **Worked example, carried to a number — Qwen3-8B's FFN** *(the anchor model's first appearance; all values `constants.md §1.2–1.3`, [DER]/[VP].)* One SwiGLU FFN block has three matrices, all built from $d=4096$, $d_{\text{ff}}=12288$ ($=3d$):
  - `gate_proj` $4096\times12288 = \mathbf{50{,}331{,}648}$; `up_proj` = same; `down_proj` $12288\times4096$ = same.
  - **FFN per block $=3\,d\,d_{\text{ff}} = 150{,}994{,}944$**, which is **78.26 %** of the block's $192{,}946{,}432$ params (`constants.md §1.3`) — *and* **66.4 %** of the whole model. **`.box warn`: those are two different denominators; say which. Never "~70 %" — that was the retired Llama figure (`constants.md §1.3`, D-01).**
  - Whole model: $\times36$ blocks $=6{,}946{,}071{,}552$ non-embedding; $+2\times622{,}329{,}856$ embeddings $+$ norms $=\mathbf{8{,}190{,}735{,}360}$ total params (`constants.md §1.2`).
  - **Memory seed (label the unit, `constants.md §0`):** bf16 weights $=\mathbf{16.38\ GB}$ (decimal, `constants.md §1.1`). One line only, then defer: "weights are the *cheap* part; on page 18 you'll add gradients and optimizer state and watch it not fit on your box." Do **not** state a budget here (it is [MEA], `constants.md §6.8`).
- *Broadcasting.* Align shapes from the right; a dim is compatible if equal, is 1 (stretched), or missing (treated as 1) (`brief-foundations §2`). **The classic disaster → `.box warn` (the single highest-value warning in the chapter):**
  ```python
  pred   = torch.randn(64)      # (64,)
  target = torch.randn(64, 1)   # (64, 1)
  loss = ((pred - target)**2).mean()   # NO ERROR. Broadcasts to (64,64). Garbage.
  ```
  It runs, it trains, it converges to nonsense. Habit: `assert pred.shape == target.shape` before every loss.
- **Misconceptions → `.box warn`:** `(64,)`≠`(64,1)`≠`(1,64)` (the rusty-engineer trap); `A*B` (Hadamard $\odot$) ≠ `A@B` (matmul) — SwiGLU uses both in one line, on purpose (forward-ref page 11); the batch axis is a hardware convenience, mathematically each example is independent (`brief-foundations §2`).

**Demo B — "Shape Detective / TensorViz"** *(build S; the page's real op is shape algebra + one live matmul).*
- **Primitive:** `viz.TensorViz` in `mode:'matmul'` for the animated cell-by-cell product, plus a `makeCtrl`-driven shape tracer.
- **Plot / behaviour:** learner picks shapes for $A$ and $B$ from sliders; the tracer shows the result shape or turns **red with the real PyTorch error** when inner dims mismatch, and **amber** when an op only succeeded via broadcasting (the amber highlight is the lesson — the dangerous ops don't error). `TensorViz.animateMatmul({A:{shape:[m,n]},B:{shape:[n,p]}})` animates one legal product.
- **Controls (≤3):** `m` [1,8] step 1; shared inner `n` [1,8] step 1; `p` [1,8] step 1. A checkbox "add a trailing 1 to B" to force the broadcast/mismatch teachable state.
- **Exact math (JS):** shape algebra only — `matmul(a,b)` checks `a[-1]===b[-2]`, output `[...,a[-2],b[-1]]`; `broadcast` right-aligns per-dim. `TensorViz` supplies the visual product of small random integer matrices so the cancellation is watchable.
- **The aha:** "shapes are a tiny type system you can run in your head; the bugs that bite are the ones that *don't* raise."

**Code artifact (`.box try`):** `code/shape_gym.py` — a ~20-line script: builds `(64,)` and `(64,1)` tensors, prints `((pred-target)**2).mean()` for both the buggy and `.squeeze()`-fixed versions so he sees the garbage broadcast happen and the assert catch it. Runs on CPU in <1 s (state the cost). Uses only `torch`.

**Quiz (6; ≥1 numeric)**
1. *(numeric, tol exact)* `A` is `(32, 128, 4096)`, `W` is `(12288, 4096)`. Shape of `A @ W.T`? Enter the last dim. **num 12288, tol 0**. `why`: `(…,4096)@(4096,12288)→(…,12288)`.
2. *(numeric, tol 0)* Qwen3-8B FFN params per block $=3\times4096\times12288$. **num 150994944, tol 0** (`constants.md §1.3`).
3. *(predict shape)* `pred (64,)` minus `target (64,1)` gives shape **[(64,) / (64,1) / (64,64) ✓ / error]**. Distractor "error" ← *"shapes must match to subtract."*
4. *(diagnose)* Loss trains but plateaus at nonsense; you find the loss line broadcasts `(64,)`−`(64,1)`. The fix is **[lower LR / `target.squeeze()` or `pred.unsqueeze(1)` ✓ / more epochs]**.
5. *(concept)* $W$ for `nn.Linear(4096, 12288)` has shape **[(4096,12288) / (12288,4096) ✓]**. Distractor ← the (in,out) error (anti-pattern #2).
6. *(numeric, tol 0.01)* FFN's share of the *whole* Qwen3-8B model. **num 66.4, tol 0.5, unit %** — and name the block-share too (78.26%) in `why` (`constants.md §1.3`).

**Thread touchpoints**
- `[THREAD:Qwen3-8B beat 1]` — the anchor's first arithmetic; the fourteen recurring numbers begin ($d{=}4096$, $L{=}36$, $d_{\text{ff}}{=}12288$, $P{=}8{,}190{,}735{,}360$, 16.38 GB). Announce it as the model he will actually fine-tune later.
- `[THREAD:memory beat 1]` — 16.38 GB bf16 weights, one line, deferred to 18.

**Cross-references:** → 03 (a single row of $W$ times $\mathbf x$ is the dot product), → 07 (the invariant → the row/column bridge), → 11 (the $\odot$ warning cashes in at SwiGLU), → 18 (memory ledger), → 43 (the $2mnp$ FLOP rule → roofline).

---

# 03 — the-dot-product.html
**Title:** "The Dot Product — the Atom" · **Part I — The Machine** · **build: O (flagship)** · no track class.

**Learning objectives**
1. Compute a dot product two ways and read it as "how much of this is in that" — magnitude × alignment.
2. Use the trig identity $\mathbf a\cdot\mathbf b=\lVert\mathbf a\rVert\lVert\mathbf b\rVert\cos\vartheta$ (his trig, cashed in) and cosine similarity — **trig payoff #1** (`decisions.md §D-21a`, pays again on 33 and 54).
3. Derive $\operatorname{std}(\mathbf q\cdot\mathbf k)=\sqrt{d_{\text{head}}}$ for unit-variance inputs — the magic constant in every transformer.
4. Feel that in high dimensions random vectors are almost always orthogonal, and see *why that is storage capacity.*

**PREDICT.** "Two random arrows in 2-D: their angle is all over the place. Two random arrows in 4096-D: the angle
between them is almost always about **[0° / 45° / 90° ✓]**. Commit, then drag the dimension slider." *(Resolves via
the Demo's right panel: the cosine histogram collapses to a spike at 0 — near-orthogonal.)*

**Section outline** *(the atom thread's formal birth; angle is $\vartheta$ per `notation.md §5.4`, never $\theta$)*
- *Two forms, one meaning.* $$\mathbf a\cdot\mathbf b=\sum_{i=1}^{d}a_i b_i=\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert\cos\vartheta.$$ **Symbol Ledger:** $\mathbf a,\mathbf b$ `(d,1)`; $\lVert\mathbf a\rVert=\sqrt{\sum a_i^2}$ (length); $\vartheta$ the angle between them (**his trig, used here — `brief-foundations §3, §0 trig payoff`; this is trig payoff #1 of three**). Big positive = same direction; 0 = perpendicular/unrelated; negative = opposed. Projection: $\operatorname{proj}_{\mathbf b}(\mathbf a)=\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf b\rVert^2}\mathbf b$ — "the shadow $\mathbf a$ casts on $\mathbf b$." Cosine similarity $\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\lVert\mathbf b\rVert}\in[-1,1]$.
- **`.box key` — the unifying sentence (`brief-pedagogy §9.1`):** *"A dot product asks: how much of this is in that? Every neural network is that question, asked billions of times, with the answers rearranged."*
- *Why it recurs — say all four out loud, now (`brief-foundations §3`):* (1) **a neuron is a dot product** $z=\mathbf w\cdot\mathbf x+b$ — the weight vector is a *template*, $z$ is the match score (this is page 10's whole reframing); (2) **attention scores are dot products** $\text{score}_{ij}=\mathbf q_i\cdot\mathbf k_j/\sqrt{d_{\text{head}}}$ — **explicit forward-reference: this line returns as attention on pages 29–30**; (3) **embeddings compared by dot product** — RAG, "king − man + woman" (page 27); (4) **the output logit is a dot product** $h\cdot E_v$ (page 09's softmax input). **And the *geometry itself* returns twice more as trig: as rotations that compose — RoPE, page 33 (trig payoff #2) — and as the noise-schedule geometry of the diffusion forward process, page 54 (trig payoff #3).** He'll recognize the second from ComfyUI.
- **Worked example — carried to a number (`brief-foundations §3`):** $\mathbf a=[3,4]$, $\mathbf b=[4,3]$: $\mathbf a\cdot\mathbf b=12+12=24$; $\lVert\mathbf a\rVert=\lVert\mathbf b\rVert=5$; $\cos\vartheta=24/25=0.96\Rightarrow\vartheta=\arccos(0.96)\approx16.26^\circ$ (nearly aligned). Then $\mathbf c=[-4,3]$: $\mathbf a\cdot\mathbf c=-12+12=0\Rightarrow\vartheta=90^\circ$ — orthogonal. "Unrelated concepts are perpendicular" starts here; it is the seed of superposition.
- **The $\sqrt{d_{\text{head}}}$ derivation — do it here, it's a probability+dot-product fact (`brief-foundations §3`, value `constants.md §5`):** if $\mathbf q,\mathbf k\in\mathbb R^{d_{\text{head}}}$ have i.i.d. mean-0, variance-1 entries, then $\mathbb E[\mathbf q\cdot\mathbf k]=0$ and $\operatorname{Var}(\mathbf q\cdot\mathbf k)=d_{\text{head}}$, so $\operatorname{std}=\sqrt{d_{\text{head}}}$. With $d_{\text{head}}=128$ (Qwen3-8B, `constants.md §1.1`): $\sqrt{128}=\mathbf{11.3137}$ (`constants.md §5`). Two-line derivation the learner does himself; it explains the divide-by-$\sqrt{d_k}$ he's seen in every diagram. **Inline (load-bearing), not deepdive.** Forward-ref the attention scaling on page 30. *(The "why variance adds" step is formalized on page 08 — here just use it.)*
- **Misconceptions → `.box warn`:**
  - *"High dot product = similar magnitude."* It entangles magnitude *and* angle. Cosine strips magnitude; the raw dot product keeps it. Attention uses the raw dot product (magnitude = confidence); RAG usually uses cosine (`brief-foundations §3`).
  - *"2-D intuition survives into high dimensions."* It doesn't — two random unit vectors in $\mathbb R^{4096}$ have cosine $\sim\mathcal N(0,1/4096)$, std ≈ 0.0156: essentially always orthogonal. That is *why* 4096 dims hold far more than 4096 near-orthogonal "concepts" (superposition) (`brief-foundations §3`). *(The same $\mathcal N(0,1/d)$ shows up again on page 08 as concentration-of-measure — flag it.)*
- **`.deepdive`:** "Why variance adds: the two-line proof that $\operatorname{Var}(\mathbf q\cdot\mathbf k)=d_{\text{head}}$." Independence ⇒ variances of the $d_{\text{head}}$ product terms add. (Reassurance-grade; the *result* is inline above; the machinery is page 08.)

**Demo — "Dot Product Dial"** *(flagship; two linked panels — build O.)*
- **Left panel (2-D geometry):** `Plot` on a fixed 2-D axis; vector $\mathbf a$ pinned at $[1,0]$, $\mathbf b$ draggable (two sliders: $\lVert\mathbf b\rVert$ [0.2,3] step 0.05, angle $\vartheta_b$ [−180,180]° step 1). Draw both arrows from origin and the projection line (shadow) of $\mathbf a$ onto $\mathbf b$. **Readout:** $\mathbf a\cdot\mathbf b$, $\lVert\mathbf a\rVert\lVert\mathbf b\rVert$, $\cos\vartheta$, $\vartheta$ in degrees — with the live equation printed and substituted (`brief-pedagogy §7.1 rule 2`): e.g. `a·b = 1.00·2.30·cos(37°) = 1.84`.
- **Right panel (high-dim collapse):** dimension slider $d\in\{2,8,64,512,4096\}$ (5-step); a histogram drawn on a second `Plot`. Sample 5000 pairs of random **unit** vectors in $\mathbb R^d$, histogram their cosine similarity, overlay the theoretical $\mathcal N(0,1/d)$ curve.
- **Exact math (JS):** random unit vector = $d$ Box–Muller Gaussians via `NN.Mat.randn(d,1,rng,1)`, divided by norm; cosine = dot of two such (`brief-foundations §3`). Left panel computes the two dot-product forms directly and checks they agree (they must, to machine precision — a mini "three representations, one number" beat).
- **The aha:** "drag $d$ from 2 to 4096 and watch the histogram collapse to a spike at zero. In 2-D random vectors point anywhere; in 4096-D everything is perpendicular to everything — and *that's the storage capacity, not a bug.*"

**Code artifact (`.box try`):** `code/orthogonality.py` — ~15 lines: sample two random unit vectors in $\mathbb R^{4096}$ with `torch`, print their cosine (≈0.01–0.02); loop 10000 times and print the empirical std ≈ $1/\sqrt{4096}=0.0156$. CPU, <1 s. "You just measured why a 4096-wide model isn't limited to 4096 ideas."

**Quiz (6; ≥1 numeric)**
1. *(numeric, tol 0.01)* $\mathbf a=[3,4]$, $\mathbf b=[4,3]$: $\cos\vartheta$? **num 0.96, tol 0.01** (`brief-foundations §3`).
2. *(numeric, tol 0.01)* $\operatorname{std}$ of $\mathbf q\cdot\mathbf k$ at $d_{\text{head}}=128$, unit-variance inputs. **num 11.3137, tol 0.05** (`constants.md §5`). `why`: $\sqrt{128}$; this is why attention divides by it (page 30).
3. *(diagnose)* Attention scores before the $1/\sqrt{d_k}$ divide are ~±11; softmax over them is near one-hot with ~zero gradient. Removing the divide would **[help gradients / saturate softmax and kill gradients ✓ / no effect]**. Distractor "no effect" ← *"the scale factor is cosmetic."*
4. *(predict)* Two random unit vectors in $\mathbb R^{4096}$: their cosine is most likely near **[1 / 0 ✓ / −1]**.
5. *(concept, thread)* $z=\mathbf w\cdot\mathbf x+b$ measures **[the length of x / how much x resembles the template w ✓ / the number of inputs]**. Distractor ← the "neuron sums inputs" surface reading.
6. *(transfer)* RAG retrieval usually uses cosine, attention uses the raw dot product, because **[cosine is faster / attention wants magnitude as a confidence, RAG wants magnitude treated as noise ✓ / they're identical]**.

**Thread touchpoints**
- `[THREAD:dot-product beat 1/…]` — the atom, formally named, with its trig and its high-dim behaviour. Announce: "this single operation returns as the neuron (page 10), as attention scores (pages 29–30), as rotations that compose (RoPE, page 33), and as the geometry of the diffusion forward process you already steer in ComfyUI (page 54)."
- `[THREAD:trig-payoff beat 1/3]` — trig payoff #1; #2 is RoPE (33), #3 is the forward process (54).
- `[THREAD:Qwen3-8B beat 2]` — $d_{\text{head}}=128$ enters via the $\sqrt{d_{\text{head}}}$ derivation.

**Cross-references:** → 04 (steepest-ascent proof *is* this dot product), → 08 (variance-adds, formalized; $\mathcal N(0,1/d)$ again), → 10 (neuron = dot product + bend), → 27 (embeddings/RAG), → 30 (attention scaling $\sqrt{d_{\text{head}}}$), → 33 (RoPE, trig payoff #2), → 54 (forward process, trig payoff #3).

---

# 04 — derivatives-and-the-flip.html
**Title:** "Derivatives, the $\vartheta$-vs-$x$ Flip, and the Downhill Direction" · **Part I — The Machine** · **build: O** · no track class.

> This is the page `brief-training` calls "what p.4 sets up": it installs the **derivative-as-sensitivity** reading,
> the 7-derivative table backprop needs, the $\vartheta$-vs-$x$ flip **in symbols** (page 01 said it in English),
> and proves **gradient = steepest ascent FROM page 03's dot product** (task-pinned). The *rigorous* stability
> story — $\eta_{\text{crit}}=2/\lambda_{\max}$ on a quadratic, saddles-not-minima, the anisotropic ravine — is
> **page 13's** (Part II); this page installs only the intuition + the 1-D LR playground that makes it visceral.
> Optimizer-step index is $k$, **never $t$** (`notation.md §3`).

**Learning objectives**
1. Read a derivative as a sensitivity ("nudge in, how much out"), and use the 7-derivative table the whole course needs.
2. State the $\vartheta$-vs-$x$ flip precisely: training differentiates w.r.t. the **knobs** $\boldsymbol\theta$ (with $x$ fixed), not w.r.t. the input — the symbolic form of page 01's "flip it."
3. Assemble partial derivatives into the gradient and **prove gradient = steepest ascent using the dot product from page 03.**
4. Know $\eta$ is a step size, not a speed — past a curvature-set threshold it diverges — and watch a loss go to NaN from a learning rate only ~3.5× too large.

**PREDICT.** "If I double the learning rate, does the loss reach the minimum (a) twice as fast, (b) roughly as
fast, or (c) not at all / it diverges? Commit, then run the LR playground." *(Resolves: past a threshold set by the
curvature it *diverges* — speed and stability trade off; `brief-pedagogy §5.1`, `§8.2` LR misconception. The exact
threshold $\eta_{\text{crit}}=2/\lambda_{\max}$ is page 13.)*

**Section outline**
- *Derivative = slope = sensitivity.* $$f'(x)=\frac{df}{dx}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}.$$ Teach the limit as notation for "$h$ small enough that shrinking it further doesn't change the answer." No epsilon-delta (`brief-foundations §4`). **Units, said plainly:** if $\mathcal L$ is in nats and $w$ is dimensionless, $\partial\mathcal L/\partial w$ is nats-per-unit-weight and $\eta$ carries the inverse units — which is the honest answer to "why is $\eta=3\times10^{-4}$ and not 0.5" (defuses the magic).
- **The $\vartheta$-vs-$x$ flip, in symbols (inline, load-bearing — the page's title beat).** The same function $f(\mathbf x;\boldsymbol\theta)$ has *two* kinds of input. **Inference** asks $\partial f/\partial\mathbf x$ (fixed knobs, vary the data) — the course almost never needs it. **Training** asks $\partial\mathcal L/\partial\boldsymbol\theta$ (fixed data, vary the knobs) — the course is *about* it. Page 01 said "flip it" in English; here it is one line: **we hold $\mathbf x$ constant and differentiate the loss with respect to $\boldsymbol\theta$.** Everything downstream — backprop, autograd, `requires_grad`, `param.grad` — is that single choice made mechanical. *(This reframes why page 01's "training turns the knobs" is not a metaphor: the knobs are literally the differentiation variable.)*
- **The only derivatives he needs (a table, `.box rule` — `brief-foundations §4`):** $c\to0$; $x^n\to nx^{n-1}$ (MSE, $n{=}2$); $e^x\to e^x$; $\ln x\to1/x$; $\sigma(x)\to\sigma(x)(1-\sigma(x))$; $\tanh x\to1-\tanh^2 x$; $\max(0,x)\to\mathbb 1[x>0]$ (**a switch, not a scale**). "That table is complete. Everything else is these seven + the chain rule + shape bookkeeping." **Derive $\sigma'$ inline** (it pays for itself): $\sigma'(x)=\frac{e^{-x}}{(1+e^{-x})^2}=\sigma(x)(1-\sigma(x))$; max at $x=0$ is $\mathbf{0.25}$ (`constants.md §9.3`) — **remember 0.25, it's the number that kills deep sigmoid nets** (paid off on page 11). $\tanh'$ and $\sigma'$ here are exactly the derivatives TN-1's forward (page 06) will later be differentiated through — but that differentiation is Part II's.
- *Partial derivatives and the gradient.* $$\nabla_{\boldsymbol\theta}\mathcal L=\Big[\tfrac{\partial\mathcal L}{\partial\theta_1},\dots,\tfrac{\partial\mathcal L}{\partial\theta_P}\Big]^\top\in\mathbb R^{P}.$$ **`.box rule`: the gradient has the same shape as $\boldsymbol\theta$** — this is why `param.grad` has `param.shape` (`notation.md §8`). **Steepest ascent, proved in one line using page 03 (task-pinned):** the directional derivative in unit direction $\mathbf u$ is $\nabla f\cdot\mathbf u=\lVert\nabla f\rVert\cos\vartheta$, maximized at $\cos\vartheta=1$, i.e. $\mathbf u\parallel\nabla f$. **The proof *is* the dot product** (`brief-foundations §4`) — announce it: "the spine is a spine; the atom from page 03 just proved which way is uphill." Inline, load-bearing.
- *The downhill step (intuition + vocabulary; rigor is page 13).* Taylor line inline: $f(\boldsymbol\theta+\boldsymbol\Delta)\approx f(\boldsymbol\theta)+\nabla f\cdot\boldsymbol\Delta$; choose $\boldsymbol\Delta=-\eta\nabla f$ ⇒ $f$ decreases by $\approx\eta\lVert\nabla f\rVert^2$. $$\boldsymbol\theta_{k+1}=\boldsymbol\theta_k-\eta\,\nabla_{\boldsymbol\theta}\mathcal L(\boldsymbol\theta_k).$$ State the rule as vocabulary; **the exact stability threshold $\eta_{\text{crit}}=2/\lambda_{\max}$ and the saddle/ravine geometry are page 13 — forward-ref, do not derive here.** Plant the real LR numbers to reuse forever (**`constants.md §9.4`, [EST], label them estimates**): LLM full FT $\eta\approx1$–$2\times10^{-5}$; **LLM LoRA $\eta\approx1$–$3\times10^{-4}$ (~10× higher)**; diffusion full FT/DreamBooth $1$–$5\times10^{-6}$; **diffusion LoRA $1\times10^{-4}$**. **`.box warn`: a diffusion LoRA at the full-FT rate trains ~100× too slow and the noise-dominated loss curve won't tell you — days burned (`decisions.md D-09`, `constants.md §9.4`).** The generating principle (state it; both tracks reuse it): **fewer trainable parameters ⇒ higher learning rate.** The full table lives on the page-18 ledger — forward-ref, don't duplicate.
- **Misconceptions → `.box warn` (attack the first hardest — `brief-foundations §4`):**
  - *"The gradient points toward the minimum."* **No** — it points locally *uphill*; the negative points locally downhill, which in a curved valley points at the *wall*, not the exit. This is exactly why momentum and Adam exist (the anisotropic-valley zigzag is page 13/17). *Words alone never fix this — the zigzag must be seen* (`brief-pedagogy §8.2`).
  - *"Bigger gradient = closer to the answer."* Magnitude says nothing about distance; zero gradient could be a min, max, saddle, or plateau — and in $\mathbb R^{8\times10^9}$ saddles vastly outnumber minima (`brief-foundations §4`; the count is page 13).
  - *"Learning rate = how fast you learn."* It's a *step size*; past a curvature-set threshold you overshoot and diverge — speed and stability trade off (`brief-pedagogy §8.2`).
- **`.deepdive`:** "Why gradient descent finds solutions that generalize is *not settled* in 2026." Implicit regularization / lottery-ticket / landscape-connectivity views all have adherents; none decisive. Honest gap, not fake closure (`brief-foundations §4, §12`).

**Demo — "Slope Explorer / LR Playground"** *(build O; `brief-foundations §4`. The 1-D playground; the 2-D anisotropic ravine is deliberately page 13's.)*
- **Primitive:** `Plot` (two linked panels) + `makeCtrl`.
- **Panel 1:** $f(x)=x^4-3x^2+x+2$ (two minima, one local max), draggable point, live tangent line, readout $f(x)$ and $f'(x)=4x^3-6x+1$.
- **Panel 2:** "Run gradient descent" button; LR slider $\eta$ log-scale [1e-4, 1] (slider carries the exponent; `fmt` shows `10**v`); starting-$x$ slider [−2.5, 2.5]. Animate iterates as dots with trails; a NaN counter that prints `Inf` in red like a real training log.
- **Exact math (JS):** literally `x = x - lr*(4*x**3 - 6*x + 1)` for 200 steps, record the trajectory. Nothing precomputed; divergence must be shown honestly (no clamping — `brief-pedagogy §7.1 rule 7`).
- **The aha (three drags):** $\eta=0.01$ slow-smooth-converges; $\eta=0.1$ fast-converges; $\eta=0.35$ **oscillates and diverges to NaN** — "a loss went NaN from an LR only 3.5× too big." Then move the start from $-2$ to $+2$ and land in a *different* minimum with the same LR: "same function, same optimizer, different answer — initialization matters." (Page 13 names the exact $\eta$ where this flips.)

**Code artifact (`.box try`):** `code/descend.py` — ~20 lines: the same quartic and its derivative in `torch` (or plain Python), a descent loop with `--lr` argument, printing the trajectory and detecting `Inf`. He runs it at `--lr 0.35` and watches it blow up exactly like the demo. CPU, <1 s. "The NaN in your real training logs later is *this*, at scale."

**Quiz (6; ≥1 numeric)**
1. *(numeric, tol 0.001)* $\sigma'(0)$. **num 0.25, tol 0.001** (`constants.md §9.3`).
2. *(numeric, tol 0.01)* $f(x)=x^4-3x^2+x+2$, $f'(x)$ at $x=1$: $4-6+1$. **num -1, tol 0.01**.
3. *(predict / diagnose)* You double a working LR and the loss goes to NaN by step 20. That's consistent with **[a code bug / the LR crossed the stability threshold set by curvature ✓ / too little data]**. Distractor "code bug" ← the *"LR = speed, more is faster"* misconception.
4. *(concept, spine)* "Gradient = steepest ascent" is proved from **[the chain rule / the dot product $\nabla f\cdot\mathbf u=\lVert\nabla f\rVert\cos\vartheta$ ✓ / the Taylor series]** (`brief-foundations §4`; callback page 03).
5. *(concept, the flip)* Training differentiates the loss with respect to **[the input $\mathbf x$ / the knobs $\boldsymbol\theta$, with $\mathbf x$ held fixed ✓ / both equally]**, which is why `param.grad` exists and inputs usually don't carry gradient.
6. *(budget/transfer)* A diffusion LoRA using $\eta=2\times10^{-6}$ (the full-FT number) will most likely **[overfit / train ~100× too slow with a loss curve that hides it ✓ / diverge]** (`decisions.md D-09`, `constants.md §9.4`).

**Thread touchpoints**
- `[THREAD:dot-product beat 2/…]` — steepest ascent proved via the dot product; the atom does structural work again.
- `[THREAD:chain-rule beat 0]` — the derivative table (esp. $\sigma',\tanh'$) is the toolkit page 05 will chain.
- `[THREAD:memory beat 2]` — the LR-vs-trainable-params principle names trainable-param count, the ledger's subject (page 18).

**Cross-references:** → 01 (the flip, in English), → 03 (dot product; the steepest-ascent proof), → 05 (the chain rule uses this table), → 11 ($\sigma'{=}0.25$ kills deep sigmoid), → 13 ($\eta_{\text{crit}}=2/\lambda_{\max}$, saddles, the ravine — the rigorous descent page), → 18 (LR table & ledger).

---

# 05 — the-chain-rule.html
**Title:** "The Chain Rule — Multiply Along Paths, Add Across Paths" · **Part I — The Machine** · **build: O (flagship — the ★ spine beat)** · no track class.

> **★ Spine beat (`decisions.md §D-21a`).** This page teaches the chain rule as a *forward-facing* tool — the rule
> that Part II's backprop will run in reverse. Part I stops at "this sentence *is* backpropagation"; the mechanical
> reverse pass over TN-1 is page 14. Cached activations → memory is planted here (the VRAM hook), collected on 18.

**Learning objectives**
1. Apply the single-variable and multipath chain rule, and say the English rule: *multiply sensitivities along each path, add across paths.*
2. Recognize that that sentence *is* backpropagation — everything in autodiff is an efficiency argument about ordering (the mechanics are Part II).
3. Carry a one-neuron example all the way to $\partial\mathcal L/\partial w=-0.1872$ and verify it in PyTorch.
4. See why products of factors < 1 vanish, why a residual's $+1$ path fixes it, and why *cached activations* are what training pays for in VRAM.

**PREDICT.** "A gear train: a 3:1 gear feeds a 2:1 gear. Overall ratio? And if one gear in a deep chain has ratio
0.25 (a sigmoid), what happens to everything upstream of it after 10 such gears? Commit a number for each."
*(Resolves: 6:1; and $0.25^{10}\approx10^{-6}$ — upstream barely turns. Seeds vanishing gradients, `brief-pedagogy §9.2 callback`.)*

**Section outline**
- *Intuition — gears (`brief-foundations §5`).* Sensitivities multiply along a chain: a 3× link feeding a 2× link makes 6× overall. Backprop is this a few million times, in the efficient order. A 0.25 gear in the chain means everything upstream barely turns — pre-loads vanishing gradients (paid off page 11's activation story).
- *Single variable.* $\frac{d}{dx}f(g(x))=f'(g(x))\,g'(x)$.
- *Multivariable / multipath — the version backprop uses.* $$\frac{\partial\mathcal L}{\partial x}=\sum_i\frac{\partial\mathcal L}{\partial u_i}\frac{\partial u_i}{\partial x},$$ where $u_1,\dots,u_k$ are every intermediate $x$ directly feeds. **Symbol Ledger.** **`.box key`, and make him repeat it:** *"For every path from a knob to the loss, multiply the sensitivities along the path; then add across all paths. Multiply along paths, add across paths — that sentence is backpropagation."* The "add across paths" half is the one everyone forgets, and it is exactly what makes residual connections work: two paths, one through the block, one straight through with derivative exactly **1** — the gradient superhighway (`brief-foundations §5`; forward-ref residuals, page 34).
- **Worked example, carried to numbers (`.box worked`, `brief-foundations §5`).** One neuron, one input, MSE: $x=2$, $w=0.5$, $b=0.1$, $y=1$. Forward: $z=wx+b=1.1$; $a=\sigma(1.1)=0.75026$; $\mathcal L=(a-y)^2=0.062370$. Backward (multiply along the path): $\frac{\partial\mathcal L}{\partial a}=2(a-y)=-0.49948$; $\frac{\partial a}{\partial z}=\sigma(z)(1-\sigma(z))=0.18739$; $\frac{\partial z}{\partial w}=x=2$. Chain: $\frac{\partial\mathcal L}{\partial w}=(-0.49948)(0.18739)(2)=\mathbf{-0.18718}$; $\frac{\partial\mathcal L}{\partial b}=\mathbf{-0.09359}$. Update $\eta=0.5$: $w\leftarrow0.59359$, $b\leftarrow0.14680$; recheck $\mathcal L'=0.043452$ — **down from 0.062370. He did that by hand.** *(This is the generic warm-up; TN-1's full backward is page 14.)*
- **Verify in PyTorch (`.codeblock`, `brief-foundations §5`).** The 8-line snippet: `w=torch.tensor(0.5, requires_grad=True)`, …, `loss.backward()`, `print(w.grad, b.grad)` → `tensor(-0.1872) tensor(-0.0936)`. **Same numbers. Autograd is the arithmetic you just did.** Protect the 4-decimal match. *(This is the ONE `.backward()` call Part I permits — it is a verification of the pencil, not a teaching of the reverse pass; the mechanized graph walk is page 15.)*
- **Misconceptions → `.box warn` (this list matters most — `brief-foundations §5`):** *"backprop is a different thing from the chain rule"* → it's the chain rule + memoization + reverse-topological order, nothing else (mechanized on page 15); *"the gradient flows back through the same weights"* → it flows through $W^\top$, and the transpose is *forced by the shapes* (forward `(B,d_in)@(d_in,d_out)`, backward `(B,d_out)@(d_out,d_in)`) — reuse page 02/07 (paid off page 14); *"autograd differentiates my Python symbolically"* → it records the ops that *actually ran* into a graph then walks it backward (why `if`/loops just work, why `.detach()`/`no_grad()` matter — page 15); *"intermediate activations are discarded after forward"* → **they're kept**, because $\partial a/\partial z$ needs $z$ — **this is why VRAM scales with batch×sequence×depth and why gradient checkpointing is a real knob on his Spark** (land the hardware connection here, `[THREAD:memory]`, forward-ref 18); *"vanishing gradients are about depth"* → the cause is repeated multiplication by factors <1 ($0.25^{32}=5.4\times10^{-20}$, `constants.md §9.3`); ReLU's factor of 1 is the fix (page 11).
- **`.deepdive`:** "Why reverse mode, not forward mode." One output, $P$ inputs (page 01's seed): reverse-mode computes all $P$ derivatives in one backward pass; forward mode would need $P$ passes. That asymmetry is *why backprop exists* (`brief-foundations §1`; the reverse pass itself is page 14).

**Demo — "Live Chain-Rule Visualizer"** *(⭐ flagship of the chapter; build O; `brief-foundations §5`.)*
- **Primitive:** raw computation-graph render (`viz.NetGraph` or a custom node/edge canvas) + `makeCtrl`.
- **Plot:** the graph `x → [×w] → [+b] → z → [σ] → a → [(·−y)²] → L`, each node showing its live forward value.
- **Controls:** sliders $w,b,x,y$; a "Backward" button; a depth slider $N\in[1,12]$ that stacks $N$ sigmoid layers; an activation toggle σ↔ReLU; a "add residual connections" checkbox.
- **Exact math (JS, ~80 lines, real reverse traversal — do NOT fake it, `brief-foundations §5`):** hardcode the forward ops and their local-derivative formulas; do a real reverse topological pass with accumulation `node.grad += upstream * localPartial`. The running product prints as a literal building string: `-0.4995 × 0.1874 × 2 = -0.1872`. Edge thickness ∝ |local partial|. *(This is a pedagogical hand-rolled reverse walk on a toy graph, not the trunk's backprop lesson — that's page 14.)*
- **The aha (three, in order):** (1) the chain rule is arithmetic — the numbers flowing right→left are the pencil; (2) the gradient is a *product*, and products of small things vanish — crank $N$ with sigmoid and watch $\partial\mathcal L/\partial w_1$ die in scientific notation, toggle ReLU and watch it not die; (3) **add residuals and the gradient holds at ~1 even with sigmoid — because the skip path adds a $+1$ term to the backward sum.** Insight (3) is worth the whole widget.

**Code artifact (`.box try`):** `code/chain_rule.py` — the 8-line one-neuron PyTorch verification above, plus a loop that stacks $N$ sigmoid layers and prints the layer-1 gradient for $N=1..12$ so he watches it underflow. CPU, <1 s. "Autograd printed the pencil's `-0.1872`; then it printed the death of a deep sigmoid gradient."

**Quiz (7; ≥1 numeric)**
1. *(numeric, tol 0.001)* One-neuron example: $\partial\mathcal L/\partial w=(-0.49948)(0.18739)(2)$. **num -0.1872, tol 0.001** (`brief-foundations §5`).
2. *(numeric, tol 1e-6 relative)* $0.25^{10}$. **num 9.54e-7, tol 1e-7** (`constants.md §9.3`). `why`: ten 0.25-gears in a chain.
3. *(concept)* The chain rule + memoization + reverse-topological order is **[a numerical-derivative method / backpropagation ✓ / symbolic differentiation of your code]** — all three distractors are named misconceptions; the mechanics are page 14–15.
4. *(shape/diagnose)* In the backward pass the gradient multiplies by **[$W$ / $W^\top$ ✓]**, and you know because **[the paper says so / the shapes force it ✓]** (`brief-foundations §5`).
5. *(predict)* Stack 32 sigmoid layers; the layer-1 gradient is roughly **[0.25 / $10^{-20}$, i.e. zero in fp32 ✓ / 1]** (`constants.md §9.3`).
6. *(concept, residual)* A residual connection fixes vanishing gradients because **[it removes layers / it adds a path whose derivative is exactly 1 into the backward sum ✓ / it lowers the LR]**.
7. *(memory/transfer)* Autograd keeps intermediate activations because **[to save time / $\partial a/\partial z$ needs the stored $z$, which is why training eats VRAM and gradient checkpointing exists ✓ / for logging]** (`brief-foundations §5`; forward-ref 18).

**Thread touchpoints**
- `[THREAD:chain-rule beat 1/…]` — the verb, formally. Announce it returns as TN-1's mechanized backward (14), autograd (15), and later the diffusion score.
- `[THREAD:memory beat 3]` — activations-are-cached → VRAM → gradient checkpointing, seeded for page 18.

**Cross-references:** → 04 (the derivative table), → 06 (TN-1's forward is the chain's top — but its backward is Part II), → 11 (the stalled sigmoid, explained), → 14 (TN-1 backward, the reverse pass), → 15 (autograd), → 18 (checkpointing/VRAM), → 34 (residuals).

---

# 06 — tn1-early-real-thing.html
**Title:** "TN-1: Your First Real Network, by Pencil and by PyTorch" · **Part I — The Machine** · **build: O (the page-6 moment — protect it)** · no track class.

> **★★ THE EARLY REAL THING (`brief-pedagogy §4.1–4.2`, task-pinned, `decisions.md §D-21a`).** The single
> highest-leverage event in the course. He hand-computes TN-1's **forward** pass, then runs ~15 lines of PyTorch on
> his own box and the two match to the digit. **FORWARD ONLY — no backprop, no `.backward()`, no gradients on this
> page.** All numbers are `constants.md §8.1` verbatim. The in-page demo calls **`NN.worked221()`**, reading only
> its forward fields. **Milestone page.** It ends by posing "…and how does it learn?" and pointing at Part II.

**Learning objectives**
1. State TN-1's architecture (2 → 2 tanh → 1 sigmoid → BCE, 9 params) and hand-compute its forward pass to $\hat y=0.3727$, $\mathcal L=0.9869$.
2. Read and run ~15 lines of PyTorch that set those exact weights and reproduce $\hat y$ to 4 decimals.
3. Believe, from evidence, that "the framework is my arithmetic, faster" — not magic.
4. Set up the venv reality of his own box without touching ComfyUI's.

**PREDICT.** "TN-1's first hidden unit computes $0.5(1)+(-0.3)(2)+0.1$. Before you simplify: is that (a) exactly 0,
(b) a small positive number, or (c) a small negative number? Commit." *(Keep this — it *quietly* seeds the float
beat. On paper it is 0; the machine's answer is set up now and **detonated on page 15** (autograd, `constants.md §8.4`),
NOT here. Do NOT reveal the float subtlety on this page. The old spec pointed this at "page 10"; under §D-21a the
float beat's home is **page 15**.)*

**Section outline**
- *Meet TN-1 (the spine's first full appearance).* Architecture: $\mathbf x\in\mathbb R^2\to$ 2 tanh units $\to$ 1 sigmoid unit $\to$ BCE. **9 parameters.** Canonical initial state (**`constants.md §8`, frozen — do not alter a digit**): $$W_1=\begin{pmatrix}0.5&-0.3\\0.8&0.2\end{pmatrix},\ \mathbf b_1=\begin{pmatrix}0.1\\-0.1\end{pmatrix},\ W_2=\begin{pmatrix}0.6&-0.9\end{pmatrix},\ b_2=0.2,\quad \mathbf x=\begin{pmatrix}1.0\\2.0\end{pmatrix},\ y=1,\ \eta=0.1.$$ **Symbol Ledger** for all six. Note every $W$ is $(d_{\text{out}},d_{\text{in}})$: $W_1$ is $(2,2)$, $W_2$ is $(1,2)$. *(tanh and sigmoid are used here purely because they are small, bounded, and hand-computable; the full activation menu and "why these are dead as hidden units" is page 11 — forward-ref.)*
- **Forward, carried to numbers (`.box worked`, `constants.md §8.1` — print exactly):** $z_{1,1}=0.5(1)-0.3(2)+0.1=\mathbf{0.0}$; $z_{1,2}=0.8(1)+0.2(2)-0.1=\mathbf{1.1}$; $a_{1,1}=\tanh(0)=\mathbf{0.0}$; $a_{1,2}=\tanh(1.1)=\mathbf{0.8005}$; $z_2=0.6(0)-0.9(0.8005)+0.2=\mathbf{-0.5205}$; $\hat y=\sigma(-0.5205)=\mathbf{0.3727}$. Loss (BCE, positive example $y{=}1$): $\mathcal L=-\ln(\hat y)=\mathbf{0.9869}$ nats. **`.box warn`: it's 0.9869, not 0.9870** — the brief's 0.9870 propagated a rounded intermediate (`constants.md §8, D-13`). Introduce BCE minimally here as "the surprise: $-\ln$ of the probability you assigned the true label"; the full cross-entropy-from-MLE derivation is page 12, and the softmax/CE family is page 09 (forward-ref both).
- **PyTorch — ~15 lines, runnable exactly as printed (`.codeblock`, no elided imports, `brief-pedagogy §11.2`).** Build `nn.Linear(2,2)` (tanh) → `nn.Linear(2,1)` (sigmoid), set `weight`/`bias` to the constants above (mind `nn.Linear` stores `(out,in)` — page 07 explains why), forward $[1.0,2.0]$, `assert abs(yhat - 0.3727) < 1e-4`. **No `loss.backward()` — that is page 14/15.** **State the hardware cost:** "<0.1 s on CPU; you don't even need the GPU for this." **`.box warn` (his box, `hardware-ground-truth.md §3`):** torch lives only in ComfyUI's venv (`~/ComfyUI/.venv`); run this with that Python, or make a fresh venv — **never pip-install the course's later stack into ComfyUI's venv.** For *this* page torch alone suffices.
- **`.box key`:** "Three representations, one number. The pencil said 0.3727. The demo says 0.3727. PyTorch says 0.3727. The framework isn't magic — it's the arithmetic you just did, faster."
- **`.milestone` (data-progress ~9):** "You built and ran a real neural network. 'PyTorch is my arithmetic, faster.'"
- **The hook → Part II (`.box key`, the page's closer — task-pinned).** *"You have a machine that maps two numbers to one, and pencil, canvas, and PyTorch all agree on what it outputs. Two questions are left. First: what other shapes does this same machine take? — that's the rest of Part I (the two shape conventions, the probability it runs on, the softmax it decides with, why the bend is non-negotiable). Second, and bigger: **how does it learn?** That is the whole of Part II."* Hold the second question open; do not answer it here.
- **Misconceptions → `.box warn`:** *"the framework does something I couldn't."* No — it does exactly your arithmetic; the only thing it adds is speed and (Part II) the *backward* pass. *"tanh/sigmoid are outdated so this is a toy."* They're chosen here precisely because they're small, bounded, and hand-computable — the *object* transfers, the activations are incidental (paid off page 11).

**Demo — "TN-1 forward, live"** *(build O; calls the frozen engine, forward fields only.)*
- **Primitive:** `viz.NetGraph` fed by `NN.worked221()`.
- **Plot:** TN-1's canonical graph (the reused course visual — same colours/layout everywhere, `brief-pedagogy §5.5`): nodes show live forward values, edges labelled with the 9 weights. A "Animate forward →" button flows the signal left→right. **No backward animation on this page** (that visual is page 14).
- **Controls:** none required for the canonical run (this demo is about *matching*, not exploring); optionally a read-only display of the nine constants. Keep it a faithful reproduction, not a slider toy.
- **Exact math (JS):** `const w = NN.worked221();` then `NetGraph({layers:[2,2,1], weights:[[[w.P.W1_11,…]],…], activations:[w.x, w.forward.a1, [w.forward.yhat]]})`. Readout: `ŷ = ${w.forward.yhat.toFixed(4)}` and `L = ${w.loss.toFixed(4)}` — **must print 0.3727 and 0.9869**, matching the pencil and the PyTorch above (the parity is the whole point).
- **The aha:** the number on the canvas equals the number he wrote by hand equals the number PyTorch printed. "Same object, three representations."

**Code artifact (`.box try`):** `code/tn1.py` — the ~15-line forward script above (this file *accretes* across pages 06→14→15→17: forward now; `.backward()` + the nine-gradient check added on page 14; the autograd assertion + the `1.4e-08` on page 15; the SGD step on page 17; each later page shows new lines highlighted and old lines dimmed, `brief-pedagogy §11.2 rule 1`). Ship it complete-and-runnable-forward at this stage. "Run it with your ComfyUI venv's python; it prints `ŷ = 0.3727`."

**Quiz (6; ≥1 numeric)**
1. *(numeric, tol 0.001)* TN-1 forward, $\hat y$. **num 0.3727, tol 0.001** (`constants.md §8.1`).
2. *(numeric, tol 0.001)* TN-1's loss $\mathcal L=-\ln(0.3727)$. **num 0.9869, tol 0.002** (`constants.md §8.1`). `why`: BCE for a positive example is $-\ln\hat y$; note it's 0.9869 not 0.9870.
3. *(predict shape)* `nn.Linear(2,2).weight.shape` is **[(2,2) / it's ambiguous here ✓ / (4,)]**. `why`: a 2→2 layer *can't* reveal (out,in) — that's exactly why page 07 uses `nn.Linear(2,3)`.
4. *(diagnose)* Your PyTorch forward prints 0.41, not 0.3727. Most likely: **[wrong seed / you set `weight` as (in,out) instead of (out,in) ✓ / floating point]**. Distractor ← the transpose bug page 07 addresses.
5. *(concept)* The claim "the framework is not magic" means **[it guesses outputs / it mechanically redoes the arithmetic you just did ✓ / it's slow]**.
6. *(his box)* You should run the course's *later* fine-tuning stack **[inside ComfyUI's venv / in a fresh separate venv ✓ / as root]** (`hardware-ground-truth.md §3`).

**Thread touchpoints**
- `[THREAD:TN1 beat 1/9]` — first full appearance: architecture + forward, by hand and in code. Announce it is the object that returns as its backward (14), autograd (15), one training step (17), a training loop (12+), one attention head, and a LoRA target.
- `[THREAD:chain-rule beat 1.5]` — the loss $\mathcal L(\hat y)$ is the top of the chain Part II walks backward; page 05 gave the verb, page 14 runs it here.
- `[THREAD:memory beat 4]` — one line: "this 9-param toy is the *shape* of the 8.19-billion-param model on page 18; the arithmetic is identical, only $P$ changes."

**Cross-references:** → 04 (the σ/tanh derivatives), → 05 (the chain rule its backward will use), → 07 (why `weight` is stored (out,in)), → 09 & 12 (BCE/cross-entropy properly), → 11 (why tanh/sigmoid are dead as hidden units), → 14 (its backward pass), → 15 (autograd on this exact net; the float beat), → 17 (its first learning step).

---

# 07 — row-or-column.html
**Title:** "Row or Column? The Shape Bridge" · **Part I — The Machine** · **build: S** · no track class.

> **Placed exactly here by design (`notation.md §1.2`, `brief-pedagogy §6.1`): right after the page-6 code, when
> he has *just* seen both $\mathbf y=W\mathbf x$ (maths) and PyTorch's `xW.T` and the question is live.** The
> reconciling gift: **$W$ is $(d_{\text{out}},d_{\text{in}})$ in BOTH conventions** — the only thing that differs
> is whether the batch stacks on the left.

**Learning objectives**
1. State both conventions and see they disagree *only* about batch placement, never about $W$.
2. Use the invariant $W:(d_{\text{out}},d_{\text{in}})$ to predict `nn.Linear` weight shapes and `.T` placement.
3. Read a shape ribbon fluently and translate paper notation ($T$/$L$/$N$ for sequence length, etc.).

**PREDICT.** "`nn.Linear(3, 5)` — what's `.weight.shape`? **[(3,5) / (5,3) ✓]**. Commit, then we'll see why the
maths and the code agree even though one writes $W\mathbf x$ and the other writes $xW^\top$." *(The **spec trap**,
`notation.md §1.2`: this page MUST use `nn.Linear(2,3)`/`(3,5)` shapes — a 2→2 layer is ambiguous and cannot
demonstrate the convention. Anti-pattern #25.)*

**Section outline**
- *The ground truth (`notation.md §1.1`, [VP] PyTorch 2.13).* `nn.Linear` applies $\mathbf y=xA^\top+\mathbf b$ with `weight` of shape `(out_features, in_features)`. The maths writes $\mathbf y=W\mathbf x+\mathbf b$ with $\mathbf x$ a column. **Both put $W$ at $(d_{\text{out}},d_{\text{in}})$.** `nn.Linear(3,5).weight.shape == (5,3)`, and the maths's $W$ in $\mathbf y=W\mathbf x$ is also $(5,3)$. **`.box key`: There is no disagreement about $W$. The conventions differ only in whether you stack the batch on the left.**
- *The prescription (`notation.md §1.3`, `.box rule`).* Single-example maths / all trunk derivations / backprop → **column form** $\mathbf y=W\mathbf x+\mathbf b$, $\mathbf x\in\mathbb R^{d_{\text{in}}\times1}$. Anything with a batch or sequence axis / **all code** / **all attention** → **row-batched** $Y=XW^\top+\mathbf b^\top$, $X\in\mathbb R^{B\times d_{\text{in}}}$. Invariant everywhere: $W$ is $(d_{\text{out}},d_{\text{in}})$.
- *Why split rather than pick one (the honest trade, `notation.md §1.3`).* Column-only forces transposing every attention equation he'll read (attention is written row-style universally); row-only muddies the backprop Jacobian bookkeeping that makes page 14 click. The split costs *one page* — this one — because the invariant means the thing he must remember never changes.
- *Shape ribbon, drilled (`notation.md §7`).* Re-show the canonical ribbon with TN-1's own numbers and with a batch: `x (B, 2)`, `W1 (2,2) # (out,in)`, `h = x @ W1.T → (B, 2)`. Three facts to hammer (they recur at attention): the output has one row per input example; `d_model` disappears after the projection; the operation is dimensionally rigid.
- **Translation Table (`.box rule`, mandatory per `notation.md §3.2`):** "papers call these differently" — $S$ (sequence length) is written $T$/$L$/$N$/`seq_len`/`max_length` (the worst offender, four rival letters); $L$ (layers) is `num_hidden_layers`; $\vartheta$ (our geometric angle) is papers' $\theta$, which *collides with parameters* — hence our split. Ship it as prose here and note it recurs as a course-wide appendix.
- **Misconceptions → `.box warn`:** *"the maths and the code use different weight matrices."* Same $W$, transposed multiply — `xW.T` and `Wx` are the same numbers (`notation.md §1.2`). *"`(2,2)` proves I understand the convention."* It can't — it's ambiguous; that's why this page uses `(2,3)`. *"transpose moves data."* In PyTorch `.T` is a stride change, no copy (until a kernel demands `.contiguous()`) (`brief-foundations §2`).
- **`.deepdive`:** "If your architect had gone row-only: what the backprop page would look like." Shows gradients as row covectors — muddier — justifying the column choice for the trunk. Reassurance-grade.

**Demo — "Shape bridge"** *(build S; reuses `viz.TensorViz`.)*
- **Primitive:** `TensorViz` `mode:'matmul'` animating both $W\mathbf x$ (column) and $XW^\top$ (row-batched) side by side on the *same* $W$.
- **Controls (≤3):** `out` [2,6] step 1, `in` [2,6] step 1 (defaults 3 and 2 so `(out,in)=(3,2)` is unmistakable), batch $B$ [1,4] step 1.
- **Exact math (JS):** shape algebra + a small integer matmul both ways; assert the per-example outputs are identical between the two conventions (they must be — the parity is the lesson). Readout prints `weight.shape = (out, in) = (3, 2)` and the two products' equality.
- **The aha:** "same $W$, same answer, batch just moves to the left. The (out,in) shape never budged."

**Code artifact (`.box try`):** `code/shapes_bridge.py` — ~12 lines: build `nn.Linear(2,3)`, print `weight.shape` → `(3,2)`; do `W @ x` (column) and `x @ W.T` (row) by hand with `torch` and `assert torch.allclose(...)`. CPU, <1 s. "The two forms are the same numbers; only the batch axis moved."

**Quiz (5; ≥1 numeric-ish)**
1. *(numeric, tol 0)* `nn.Linear(4096, 12288).weight.shape` — enter dim 0. **num 12288, tol 0** (`notation.md §1.1`).
2. *(predict)* `nn.Linear(2,3)` weight shape is **[(2,3) / (3,2) ✓]**. Distractor ← the (in,out) error.
3. *(concept)* The maths $W\mathbf x$ and code $xW^\top$ differ in **[the weight matrix / only where the batch axis sits ✓ / the result values]**.
4. *(diagnose)* Your `nn.Linear` layer errors with a shape mismatch after you hand-loaded weights transposed. Fix: **[retrain / store `weight` as (out,in) not (in,out) ✓ / change LR]**.
5. *(transfer)* A paper writes "sequence length $T$." In this course that symbol is **[$t$ diffusion time / $S$ ✓ / $L$]** (`notation.md §3.2`). Distractor "$t$" ← the reserved-for-diffusion collision.

**Thread touchpoints**
- `[THREAD:TN1 beat 2/9]` — TN-1's forward recast in explicit shapes (`(B,2)@(2,2).T`), the same object in code-shape clothing. Callback page 06.
- `[THREAD:Qwen3-8B beat 4]` — the `(12288,4096)` FFN weight shape reappears to make the invariant concrete on the anchor.

**Cross-references:** → 02 (the invariant introduced), → 06 (why the PyTorch forward set weights the way it did), → 14 ($W^\top$ in the backward pass), → 30 (row-style attention).

---

# 08 — probability-and-the-gaussian.html
**Title:** "Probability, the Gaussian, and Sampling You Can Differentiate Through" · **Part I — The Machine** · **build: O** · no track class.

> **★ WRITTEN FRESH — the diffusion track's trunk dependency (`decisions.md §D-21a` page 8; the fan-out deleted
> it and pages 53–57 were left citing a reparameterization trick that appeared nowhere in the trunk).** This is
> **THE handoff to diffusion.** Teach $x=\mu+\sigma\varepsilon$ **early**, as "sampling you can differentiate
> through," so the diffusion forward process is a *recognition*, not a derivation. **MSE-as-Gaussian-NLL is NOT
> here — page 12 owns it; plant the seed and cite forward.** Source: `brief-foundations §6`.

**Learning objectives**
1. Read a probability distribution as "a shape saying which values are likely," expectation as its balance point, and sampling as "reach in and grab one."
2. State that **an expectation is estimated by averaging samples** — the fact a minibatch is built on (the $1/\sqrt{B}$ consequence is page 16).
3. Write the Gaussian PDF, know the 68/95/99.7 rule, and know why *isotropic* $\mathcal N(\mu,\sigma^2 I)$ is the only covariance the course needs.
4. Own the reparameterization trick $x=\mu+\sigma\varepsilon$: sample the randomness once from a fixed bell curve, then shift-and-stretch deterministically — **that is why you can backprop through a "random" node.**

**PREDICT.** "You draw one sample from a 4096-dimensional standard Gaussian $\mathcal N(0, I)$ — the highest-density
point is the origin. How far from the origin will your sample land? **[right at the origin / about distance 64 ✓ /
about distance 1]**. Commit, then run Panel C." *(Resolves: $\mathbb E\lVert x\rVert\approx\sqrt{d}=64$, std ≈0.71
— a high-D Gaussian is a thin shell, not a ball; `brief-foundations §6`. This is *why* diffusion starts from a
shell of noise, and it is the same $\mathcal N(0,1/d)$ orthogonality he already met on page 03.)*

**Section outline** *(the probability thread's formal birth; all symbols `notation.md §5`; $\varepsilon$ is the standard-normal draw, reserved.)*
- *Distribution, expectation, variance (`brief-foundations §6`).* $$\mathbb E[X]=\sum_x x\,p(x)\ \text{ or }\ \int x\,p(x)\,dx,\qquad \operatorname{Var}(X)=\mathbb E[(X-\mathbb E[X])^2]=\mathbb E[X^2]-\mathbb E[X]^2.$$ **Symbol Ledger.** Intuition first: *a distribution is a shape describing which values are likely; expectation is the balance point; sampling is reaching in and grabbing one.*
- **The one thing to hammer — expectations are estimated by averaging samples (`.box key`, `brief-foundations §6`):** $$\mathbb E[f(X)]\approx\frac1N\sum_{i=1}^N f(x_i),\qquad x_i\sim p.$$ **This is what a minibatch IS:** the true loss is an expectation over the whole dataset, and a batch of $B$ is a $B$-sample Monte-Carlo estimate of it. Say only that here; **the $1/\sqrt{B}$ standard-error law and "noise as a feature" are page 16 — forward-ref, do not derive.**
- *The Gaussian — do it properly, diffusion is built from it (`brief-foundations §6`).* $$p(x)=\frac1{\sigma\sqrt{2\pi}}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big).$$ $\mu$ mean (units of $x$), $\sigma$ std (units of $x$), $\sigma^2$ variance (units$^2$). **$\pm1\sigma$: 68.3 %; $\pm2\sigma$: 95.4 %; $\pm3\sigma$: 99.7 %** ([DER]). **Isotropic multivariate** $\mathcal N(x;\mu,\sigma^2 I)$, $x,\mu\in\mathbb R^d$ — "a round fuzzy ball, same spread every direction, no correlations." **This is the only covariance structure the course needs**, which is exactly why eigen-decomposition was cut (`brief-foundations §0`): $\Sigma=\sigma^2 I$ has no interesting structure.
- **★ The reparameterization trick — the page's spine beat, taught EARLY (`.box key`, `brief-foundations §6`):** $$x\sim\mathcal N(\mu,\sigma^2)\iff x=\mu+\sigma\varepsilon,\quad \varepsilon\sim\mathcal N(0,1).$$ **Intuition:** *sample the randomness once, from a fixed standard bell curve, then shift and stretch it deterministically.* Frame it exactly as **"sampling you can differentiate through":** you cannot backprop through "draw a sample from a distribution whose parameters depend on $\theta$" — a draw is not a function of $\mu$. The trick **moves the randomness out of the path**: $\varepsilon$ becomes an *external input*, drawn once, and $x=\mu+\sigma\varepsilon$ is now a plain differentiable function with $\partial x/\partial\mu=1$, $\partial x/\partial\sigma=\varepsilon$. **Gradients flow; the dice are still rolled, just outside the graph** (`brief-diffusion §2`). Announce the payoff without spending it: "this exact move is (a) how you sample a Gaussian in code, (b) how a VAE gets gradients through its sampling step, and (c) *literally* the diffusion forward process $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$ — you will recognize it on page 54 as an old friend with new letters."
- *Concentration of measure (inline, load-bearing for diffusion — `brief-foundations §6`).* A sample from $\mathcal N(0,I_d)$ at $d=4096$ has $\mathbb E\lVert x\rVert\approx\sqrt d=\mathbf{64}$, std $\approx1/\sqrt2\approx0.71$ ([DER]). **Essentially no sample is near the origin, even though the origin is the densest point** — high-D Gaussians are thin shells, not balls. This is why "the model just outputs the mean" is not what happens, and why diffusion starts from a *shell*. Same $\mathcal N(0,1/d)$ fact as page 03's orthogonality collapse — flag the coherence.
- **The seed for page 12 (one line, `.box rule` — do NOT derive):** "later you'll see that plain squared-error loss (MSE) is *exactly* the negative log-likelihood of a Gaussian with fixed variance — that's how the LLM's cross-entropy and the diffusion track's MSE turn out to be one principle. That unification is page 12; here just hold that MSE and 'a Gaussian' are secretly the same story."
- **Misconceptions → `.box warn` (`brief-foundations §6`):**
  - *"Everything is Gaussian / the CLT means my data is Gaussian."* No. The Gaussian is *chosen* in diffusion because it's closed under addition (sum of Gaussians is Gaussian — exactly what makes the closed-form $x_t$ from $x_0$ possible) and its score has a clean form. A design decision, not a discovered fact about images.
  - *"More samples = the mean gets more accurate, linearly."* $1/\sqrt N$: 100× the samples = 10× the precision (page 16 makes this the batch story).
  - *"Variance and std are interchangeable."* Units differ. $\sigma$ has $x$'s units; $\sigma^2$ has them squared. **This bites when reading diffusion papers** that write $\beta_t$ (variances) and $\sigma_t$ (stds) and rarely say which.
  - *"Sampling from a model = the model is uncertain."* Sampling is a decision procedure applied to the model's (deterministic) output distribution — same point as page 01's temperature.
- **`.deepdive`:** "Why a high-D Gaussian is a soap bubble." The $\lVert x\rVert^2=\sum_i x_i^2$ is a sum of $d$ independent unit-variance terms; by the law of large numbers it concentrates at $d$, so $\lVert x\rVert\approx\sqrt d$ with vanishing relative spread. Reassurance-grade; the *result* is inline.

**Demo — "Sampling Bench"** *(flagship; three panels — build O; `brief-foundations §6`. Panel B is the one that matters.)*
- **Primitive:** `Plot` (three linked panels) + `makeCtrl` + a seeded `new NN.RNG(seed)` and `NN.Mat.randn` (Box–Muller) so "the same $\varepsilon$" is *literally* true frame to frame.
- **Panel A — the distribution:** a Gaussian PDF drawn from the formula with $\mu$ [−3,3] step 0.1 and $\sigma$ [0.2,3] step 0.05 sliders; a "draw 1 / draw 100 / draw 10000" button drops samples as rug ticks and builds a histogram. Live readout: sample mean vs true $\mu$, and the shrinking $\pm1.96\,\sigma/\sqrt N$ CI band (the $1/\sqrt N$ made visible, named fully on 16).
- **Panel B — reparameterization (the flagship, the task's named demo):** the *same* distribution built as $x=\mu+\sigma\varepsilon$. A **fixed** cloud of $\varepsilon$ samples is drawn once from a standard normal (shown in its own little plot, seeded); the $\mu$ [−3,3] and $\sigma$ [0,3] sliders then **shift and stretch the SAME $\varepsilon$ cloud**, with arrows showing the affine map. Drag $\sigma$ from 0 to 3 and watch every one of the identical $\varepsilon$ points slide outward together.
- **Panel C — concentration:** dimension slider $d\in\{1,2,8,64,512,4096\}$; histogram of $\lVert x\rVert$ for $x\sim\mathcal N(0,I_d)$, with $\sqrt d$ marked (at $d=4096$ the spike sits at 64).
- **Exact math (JS):** Box–Muller normals via `NN.Mat.randn(N,1,rng,1)` (seed fixed for Panel B); PDF evaluated directly from the formula; $\lVert x\rVert$ computed from a fresh `Mat.randn(d,1,rng,1)` per sample in Panel C.
- **The aha (three, Panel B first):** (1) **the randomness lives entirely in $\varepsilon$; $\mu$ and $\sigma$ are just a deterministic affine map applied to it — *that* is why you can backprop through a "random" node**; (2) Panel A: the sample mean crawls toward $\mu$ as $1/\sqrt N$, never faster; (3) Panel C: in 4096-D the bell curve is a soap bubble at radius 64.

**Code artifact (`.box try`):** `code/reparam.py` — ~18 lines: draw $\mathcal N(\mu,\sigma^2)$ two ways in `torch` — `mu + sigma*torch.randn(n)` vs `torch.normal(mu, sigma, (n,))` — and show the histograms match. Then the payoff: make `mu` and `sigma` `requires_grad=True`, form `x = mu + sigma*eps` with a *detached* `eps`, call `x.sum().backward()`, and print `mu.grad` (all ones) and `sigma.grad` (equals `eps`) — **gradients flow through the sample**. Contrast with trying to differentiate a direct `torch.normal` draw (no graph). CPU, <1 s. "You just backpropagated through randomness. That single trick is the whole VAE and the whole diffusion forward process."

**Quiz (7; ≥1 numeric)**
1. *(numeric, tol 1)* Expected distance of a sample from the origin for $\mathcal N(0,I_d)$ at $d=4096$. **num 64, tol 2** (`brief-foundations §6`). `why`: $\sqrt{4096}$ — the thin-shell fact.
2. *(numeric, tol 0.5)* Fraction of a Gaussian's mass within $\pm2\sigma$. **num 95.4, tol 0.5, unit %** (`brief-foundations §6`).
3. *(concept, the trick)* $x=\mu+\sigma\varepsilon$ lets you backprop through a sample because **[it removes the randomness / the randomness lives in the external $\varepsilon$ and $x$ is a differentiable function of $\mu,\sigma$ ✓ / autograd approximates the draw]** (`brief-foundations §6`).
4. *(numeric, tol 0)* In $x=\mu+\sigma\varepsilon$, the derivative $\partial x/\partial\mu$. **num 1, tol 0** — and note $\partial x/\partial\sigma=\varepsilon$ in `why`.
5. *(diagnose)* A diffusion paper's forward equation $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$ looks new. It is **[a fresh derivation / the reparameterization trick $x=\mu+\sigma\varepsilon$ with $\mu=\sqrt{\bar\alpha_t}x_0$, $\sigma=\sqrt{1-\bar\alpha_t}$ ✓ / unrelated]** (forward-ref 54).
6. *(concept)* The course only ever needs the **isotropic** Gaussian $\mathcal N(\mu,\sigma^2 I)$ because **[data is always round / $\Sigma=\sigma^2 I$ has no cross-correlations, so no eigen-decomposition is needed ✓ / it's an approximation]**.
7. *(transfer, seed for 12)* MSE loss is secretly **[unrelated to probability / the negative log-likelihood of a Gaussian with fixed variance ✓ / only for regression]** — the unification page 12 will pay off (`brief-foundations §8`).

**Thread touchpoints**
- `[THREAD:reparameterization beat 1/… — ★ diffusion handoff]` — $x=\mu+\sigma\varepsilon$ taught in the trunk, EARLY. Announce: "this returns as VAE sampling and as the diffusion forward process; pages **53–57** all lean on what you just did (`decisions.md §D-21a` page 8)."
- `[THREAD:probability beat 1]` — distribution/expectation/sampling; the sample-average fact whose $1/\sqrt B$ payoff is page 16.
- `[THREAD:loss-unification beat 0]` — MSE = Gaussian NLL, planted for page 12.
- `[THREAD:Qwen3-8B beat —]` — $\mathcal N(0,1/d)$ at $d=4096$ ties back to page 03's orthogonality.

**Cross-references:** → 03 (variance-adds; the $\mathcal N(0,1/d)$ shell), → 12 (MSE and CE from one principle; the unification), → 16 ($1/\sqrt B$ minibatch noise), → 53 (VAE reparameterization reopened), → 54 (the forward process is this, trig payoff #3), → 55 (reverse/ELBO leans on it).

---

# 09 — logs-softmax-cross-entropy.html
**Title:** "Logs, Softmax, and Why Cross-Entropy Is Just 'Make the Data Likely'" · **Part I — The Machine** · **build: O** · no track class.

> **★ WRITTEN FRESH (salvaging the logsumexp/CE fragments the fan-out scattered); `decisions.md §D-21a` page 9.**
> Underflow is **FELT first** — nn.js ships a stable AND a deliberately-unstable softmax for exactly this. The
> canonical logits are `[2.0, 1.0, 0.1]` → $\mathcal L=0.417030$ (`constants.md §9.2`, FROZEN — two rival vectors
> retired). $\partial\mathcal L/\partial z=p-y$ is stated here as the **miracle cancellation**; its *derivation* is
> page 12 (from MLE) and page 14 (the backprop). This page is also the trunk's **first "paper-math ≠ machine-math"
> beat** — it now *precedes* the autograd `1.4e-08` beat (page 15), which must reference back to it.

**Learning objectives**
1. Use the four log identities and know why log-space is a numerical *necessity*, not a nicety — feel the underflow before the fix.
2. Apply the log-sum-exp trick and know it is *exact*, not an approximation.
3. Compute softmax on the canonical logits, know shift-invariance and temperature (the ComfyUI slider), and read $\mathcal L=-\log p_{\text{correct}}$ as "the probability you gave the right answer."
4. State the miracle result $\partial\mathcal L/\partial z=p-y$ ("predicted minus actual") and know it is *why* cross-entropy beat MSE for classification.

**PREDICT.** "A 500-token sequence, each token predicted with probability 0.1. You multiply the 500 probabilities
in float32 to score the sequence. The result is **[a tiny number like 1e-40 / exactly 0.0, and then the log is
−inf and the run dies ✓ / about 0.1]**. Commit, then watch it happen." *(Resolves: $\prod 0.1=10^{-500}$, which is
**exactly 0.0** in float32 — smallest subnormal is ~$1.4\times10^{-45}$; then `log(0)=-inf`, `-inf*0=nan`. The fix
is to sum logs: $500\ln0.1=-1151.29$, a perfectly ordinary float. `brief-foundations §7`.)*

**Section outline** *(the log/softmax thread; symbols `notation.md §5`; $\log=\ln$ always in ML, loss in nats.)*
- *Logs, only the identities that earn their place (`brief-foundations §7`).* $$\log(ab)=\log a+\log b,\quad \log(a^n)=n\log a,\quad e^{\log x}=x,\quad \log_b x=\tfrac{\ln x}{\ln b}.$$ Convention: **$\log=\ln$ (nats), not bits** — divide by $\ln2=0.6931$ for bits. Perplexity $=e^{\mathcal L}$: "a loss of 2.0 nats = perplexity 7.39 = as confused as picking uniformly among 7.4 words."
- **Underflow, FELT first (`.box worked`, `brief-foundations §7`).** $\prod_{i=1}^{500}0.1=10^{-500}$. Smallest normal float32 $\approx1.18\times10^{-38}$, smallest subnormal $\approx1.4\times10^{-45}$ — **so $10^{-500}$ is exactly 0.0 in float32, and also in bf16** (same exponent range). Then `log(0)=-inf`, `-inf*0=nan`, run dead. In log space: $\sum_{i=1}^{500}\ln(0.1)=500\times(-2.302585)=\mathbf{-1151.29}$, an ordinary float. **`.box key`: this is not an optimization — it's the difference between working and not working. Show him the `nan`.** *(This is the trunk's first encounter with "the math on paper and the math in the machine are not the same math" — page 15 reopens it with autograd's `1.4e-08`.)*
- **The log-sum-exp trick (he meets it in every framework's source, `brief-foundations §7`):** $$\log\sum_i e^{z_i}=m+\log\sum_i e^{z_i-m},\qquad m=\max_i z_i.$$ Why: $e^{88.7}$ already overflows float32 (max $\approx3.4\times10^{38}$); subtracting the max makes the largest exponent $e^0=1$ and the rest $\le1$ (they can only *underflow to 0*, which is harmless). **The identity is exact, not an approximation** — state it, because he'll assume it's a hack. **`.box warn`:** this is why `F.cross_entropy(logits, targets)` takes **logits**, not probabilities — it fuses softmax+log+NLL into one stable kernel; **`log(softmax(x))` as two ops is a real bug that only shows up on some inputs.**
- *Softmax (`brief-foundations §8`).* $$p_i=\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}}.$$ $z\in\mathbb R^V$ logits (dimensionless, any real; shape `(B,S,V)`, $V=151{,}936$ for Qwen3, `constants.md §9.5`); $p$ probabilities, $p_i>0$, $\sum p_i=1$. **Two load-bearing properties:** (1) **shift invariance** $\operatorname{softmax}(z+c)=\operatorname{softmax}(z)$ — only *differences* matter, and this is the *same fact* that licenses the max-subtraction above; (2) **temperature** $\operatorname{softmax}(z/T)$: $T\to0$ argmax (greedy), $T=1$ the model's own distribution, $T\to\infty$ uniform — **he has turned this exact slider in ComfyUI and every LLM UI; name it** (callback page 01's magnitude→confidence).
- **Worked example — the canonical logits, FROZEN (`.box worked`, `constants.md §9.2`, [DER]):** $z=[2.0,1.0,0.1]$, true class $c=0$. $e^z=[7.389056,2.718282,1.105171]$, $\sum=11.212509$; $\hat y=[\mathbf{0.659001},0.242433,0.098566]$ (sums to 1 ✓); $\mathcal L=-\ln(0.659001)=\mathbf{0.417030}$ nats. **`.box warn`: use `[2.0,1.0,0.1]` everywhere — the 4-way `[2.0,1.0,0.1,−1.0]`→0.4491 and the 3-way `[2,1,0]` are RETIRED (`constants.md §9.2`, D-08).**
- *Cross-entropy = NLL = "make the data likely" (the derivation's shape, result stated; the full MLE derivation is page 12).* $$\mathcal L_{\text{CE}}=-\sum_{i=1}^V y_i\log p_i\ \xrightarrow{\text{one-hot }y}\ \mathcal L=-\log p_{\text{correct}}.$$ **`.box key`: for a one-hot target, cross-entropy is the negative log of the single probability you assigned to the right token — all other $V-1$ terms are multiplied by zero.** One sentence demystifies the formula. Then the *why*: maximizing $\prod_n p_\theta(y_n\mid x_n)$ underflows (see above), so take logs → minimizing $-\sum_n\log p_\theta(y_n\mid x_n)$ = NLL = cross-entropy. **"Cross-entropy isn't a loss someone invented — it's 'make the data likely,' rewritten so a computer can do it."** The formal MLE derivation and the MSE-from-the-same-principle unification are **page 12** — forward-ref, don't duplicate.
- **The miracle cancellation — stated here, derived later (`.box key`, `constants.md §9.2`, `brief-foundations §8`):** $$\frac{\partial\mathcal L}{\partial z_i}=p_i-y_i\quad(\text{"predicted minus actual"}).$$ For the canonical logits: $\partial\mathcal L/\partial z=\hat y-y=[\mathbf{-0.340999},+0.242433,+0.098566]$ — **sums to zero, always** (a consequence of shift invariance; nice cross-check). No sigmoid derivative, no 0.25 factor, no vanishing: the softmax's ugly derivative and the log's $1/x$ **cancel exactly.** This is *why* CE beat MSE for classification — a confidently-wrong output has gradient magnitude $\to1$ (**maximally wrong = maximally corrected**), whereas MSE+sigmoid gives it a *tiny* gradient. **State the result and its consequence; the derivation is page 12 (from MLE) and page 14 (the backprop), which is where it is earned.**
- **Anchor number — the step-0 sanity check (`constants.md §9.1`):** a randomly-initialized Qwen3-8B has loss $\ln(151{,}936)=\mathbf{11.93}$ nats, perplexity exactly $V$ by construction. **If step 0 doesn't print ~11.9, something is wrong** — a real, usable check. (⚠️ NOT 11.76 — that is Llama-3's $V=128{,}256$, retired, `constants.md §9.1`.)
- **Misconceptions → `.box warn` (`brief-foundations §8`):** *"softmax outputs are calibrated confidences"* → they sum to 1 by construction, that's arithmetic not calibration; modern nets are systematically overconfident, and whether large LLMs are calibrated in 2026 is **genuinely contested — flag as open** (Guo et al. 2017 for image classifiers; RLHF appears to degrade it, but unsettled); *"temperature changes what the model knows"* → it changes the sampling decision only, the logits are identical (verify: same prompt+seed, different $T$); *"softmax is a soft max"* → it's a soft **argmax** (a distribution over indices, not a value) — historical misnomer; *"CE is only for classification, irrelevant to diffusion"* → an LLM's next-token prediction is a $V$-way classification and its *only* loss; diffusion uses MSE, and **both are NLL under different assumed output distributions** (MSE = NLL of a fixed-variance Gaussian) — the unification page 12 collects (seeded on page 08).
- **`.deepdive`:** "bf16 vs fp16, and why range beats precision for training." bf16 has 8 exponent bits (fp32's range, ~$10^{\pm38}$) and 7 mantissa; fp16 has 5 exponent (range only ~$6\times10^{-5}$–$65504$) and 10 mantissa. **bf16 trades precision for range, and range is what training needs — that's why fp16 needs a GradScaler and bf16 doesn't.** On his Blackwell GB10 bf16 is the default (`brief-foundations §7`; `constants.md §9.3` fp16 max 65,504). Reassurance-grade but practical.

**Demo 1 — "Underflow, felt"** *(build O; nn.js's two softmaxes are the point; `brief-foundations §7`.)*
- **Primitive:** two side-by-side counters on `Plot`/DOM + `makeCtrl`; `NN.softmax` (stable) vs `NN.softmaxUnstable` (deliberately naive).
- **Controls:** "sequence length" $N$ [1,1000] step 1; "per-token probability" $q$ [0.01,0.99] step 0.01; a logit-scale slider for the softmax panel [−1000,1000].
- **Exact math (JS):** left counter = naive product $\prod q$ computed through `Math.fround` (simulating fp32) — hits `0`, the box goes red, `log` of it prints `-Infinity`; right counter = $\sum\ln q$, a comfortable number. Third strip: feed a large logit (e.g. `[800,1,1]`) to `NN.softmaxUnstable` → `NaN` (from `exp(800)=Infinity`, `Infinity/Infinity`), and to `NN.softmax` (max-subtracted) → correct.
- **The aha:** "the two panels compute the *same* mathematical quantity; one returns NaN. Numerical analysis isn't pedantry — it's the difference between a training run and a crash. And this is the same lesson you'll meet again as a gradient that prints `1.4e-08` instead of zero (page 15)."

**Demo 2 — "Softmax + Temperature Bench"** *(flagship; build O; `brief-foundations §8`.)*
- **Primitive:** `makeCtrl` + a live bar chart on `Plot`; `NN.softmax`.
- **Controls:** three logit sliders $z_i\in[-5,5]$ (defaults `[2.0,1.0,0.1]` — the canonical vector); temperature $T\in[0.01,5]$ log-scale; a "true class" radio; a **"+3 to all logits"** button.
- **Plot:** the probability bar chart live; below it the **gradient bar chart $p-y$** (signed — the true-class bar points down, the rest up: *"push the right one up, push the others down, in proportion to how wrong you were"*); readout of $\mathcal L=-\ln p_{\text{true}}$ (must read **0.417030** at the defaults with class 0).
- **Exact math (JS):** `const m=Math.max(...z); const e=z.map(v=>Math.exp((v-m)/T)); const s=e.reduce((a,b)=>a+b); const p=e.map(v=>v/s);` — i.e. `NN.softmax(z, {T})`; gradient `p.map((pi,i)=>pi-(i===cls?1:0))`.
- **The aha (three):** (1) $T\to0$: one bar → 1, rest → 0 — *that's greedy decoding, why `temperature=0` output is repetitive*; (2) the gradient chart literally shows "predicted minus actual"; (3) the **+3 button moves nothing** — shift invariance, seen not stated, and it's the same fact as the max-subtraction that saved Demo 1.

**Code artifact (`.box try`):** `code/softmax_stable.py` — ~18 lines: compute softmax two ways on `[2.0,1.0,0.1]` (they agree; print $p_0=0.659001$, $\mathcal L=0.417030$, gradient $p-y$ summing to 0) and on `[800.,1.,1.]` (naive `exp/Σexp` → `nan`; stable → correct). Then one line showing `F.cross_entropy(logits, target)` reproduces $\mathcal L$ from *logits* directly, and a comment: this is why you never hand it probabilities. CPU, <1 s. "The stable and unstable versions are the same math; only one survives a big logit."

**Quiz (7; ≥2 numeric)**
1. *(numeric, tol 0.001)* Softmax of $[2.0,1.0,0.1]$, entry 0. **num 0.659001, tol 0.001** (`constants.md §9.2`).
2. *(numeric, tol 0.001)* Cross-entropy loss for those logits, true class 0. **num 0.417030, tol 0.002** (`constants.md §9.2`). `why`: $-\ln(0.659001)$.
3. *(numeric, tol 0.05)* Step-0 loss of a randomly-initialized Qwen3-8B ($V=151{,}936$). **num 11.93, tol 0.1** (`constants.md §9.1`). `why`: $\ln V$; NOT 11.76 (Llama-3).
4. *(concept, the miracle)* $\partial\mathcal L/\partial z=p-y$ holds because **[a coincidence / softmax's derivative and the log's $1/x$ cancel exactly ✓ / we approximated]**, which makes a confidently-wrong output learn **[slowly / maximally — gradient magnitude →1 ✓]** (`brief-foundations §8`).
5. *(diagnose)* Your loss prints `nan` on step 1. You compute it as `(-target * log(softmax(logits))).sum()`. Likely fix: **[lower LR / use `F.cross_entropy(logits, target)` — fuse softmax+log+NLL, don't split them ✓ / more data]** (`brief-foundations §7`).
6. *(concept)* Adding a constant $c$ to every logit **[changes the prediction / leaves softmax unchanged — only differences matter ✓ / doubles the loss]**, the same fact that licenses subtracting the max.
7. *(transfer)* The underflow you felt here (product → 0.0, `log` → −inf) is the trunk's first case of **[a framework bug / paper math ≠ machine math, which returns as the gradient that prints `1.4e-08` on page 15 ✓ / slow code]**.

**Thread touchpoints**
- `[THREAD:softmax-CE beat 1/…]` — softmax, CE=NLL, and the $p-y$ miracle, stated. Announce the derivation lands on 12 (from MLE) and 14 (backprop), and the objective is the LLM track's entire loss (page 45+).
- `[THREAD:paper-vs-machine beat 0]` — underflow is the FIRST such beat; reopened at page 15 (autograd `1.4e-08`). **Ordering note for the Part II agent: 09 precedes 15.**
- `[THREAD:loss-unification beat 1]` — CE and MSE as two NLLs; collected on page 12.
- `[THREAD:Qwen3-8B beat —]` — $V=151{,}936$, step-0 loss 11.93.

**Cross-references:** → 01 (temperature = the magnitude/confidence slider), → 03 (the output logit is a dot product $h\cdot E_v$), → 08 (MSE = Gaussian NLL, the other half of the unification), → 12 (CE and MSE from one MLE principle — the derivation), → 14 ($p-y$ earned in the backprop), → 15 (paper-vs-machine reopened), → 45+ (next-token CE is the LLM's whole loss).

---

# 10 — neuron-xor-collapse.html
**Title:** "The Neuron, XOR, and Why a Linear Stack Collapses" · **Part I — The Machine** · **build: O** · no track class.

> **Ordering ruling (`decisions.md D-19`, preserved): linear collapse is proved BEFORE any activation menu** — so
> the nonlinearity feels *necessary*, not like a decorative list of options. This page proves collapse and shows
> XOR needs *a* bend (ReLU, as the worked example); the full activation taxonomy and "why ReLU won" is page 11.
> Salvaged from the old page 04 (its activation-menu half was moved to 11).
>
> **NOTE (`constants.md §8.4` staleness):** constants §8.4's "page ~10" pointer for the float-beat is STALE under
> §D-21a — the float beat's home is **page 15**, NOT this page. Do **not** put the "exactly zero is false" beat here.

**Learning objectives**
1. Write a neuron $z=\mathbf w\cdot\mathbf x+b$, $a=\phi(z)$ and read it geometrically as a hyperplane with a squash — the symbolic form of page 01's dragged neuron and page 03's "the weight vector is a template."
2. Prove that stacked affine maps collapse to one affine map (4 lines), and feel the parameter waste exactly (not approximately).
3. Verify by hand that a 2-neuron ReLU network computes XOR — build something a linear model provably cannot.
4. Leave *needing* a nonlinearity — which page 11 then supplies and taxonomizes.

**PREDICT.** "Stack five linear layers, 16 units wide each, no activation — 1400 knobs. The decision boundary it
can draw is **[an arbitrary wiggly curve / always a single straight line ✓ / a circle]**. Commit, then try to
break it in the demo." *(Resolves: identity activation ⇒ boundary is always a line, no matter the depth/width.)*

**Section outline**
- *The neuron, precisely (callback pages 01, 03).* $$a=\phi(z),\qquad z=\mathbf w\cdot\mathbf x+b=\sum_{i=1}^{n}w_i x_i+b.$$ **Symbol Ledger:** $\mathbf x,\mathbf w$ `(n,1)`; $b$ scalar (the threshold, negated: $z>0\iff\mathbf w\cdot\mathbf x>-b$); $\phi$ generic activation; $a$ output. **Geometric read (inline, essential):** $\mathbf w\cdot\mathbf x+b=0$ is a hyperplane; $\mathbf w$ is its normal; distance from origin $=|b|/\lVert\mathbf w\rVert$. *A neuron cuts space in half; $\phi$ decides how sharply* (`brief-foundations §9`). **This is Demo A (page 01) with its symbols filled in, and the dot product (page 03) is the ruler — announce both callbacks.** The weight vector is a *template*, $z$ the match score.
- *A layer.* $$\mathbf h=\phi(W\mathbf x+\mathbf b),\quad W\in\mathbb R^{m\times n},\ \mathbf b\in\mathbb R^{m},\ \mathbf h\in\mathbb R^{m}.$$ $m$ neurons, each a template row of $W$; one matmul computes all $m$ match scores (the only reason GPUs matter). Shape ribbon required. ($W$ is $(m,n)=(d_{\text{out}},d_{\text{in}})$ — the invariant, pages 02/07.)
- **Linear collapse — the proof (4 lines, inline, load-bearing — `brief-foundations §10`):** $$\mathbf h_2=W_2(W_1\mathbf x+\mathbf b_1)+\mathbf b_2=\underbrace{W_2W_1}_{W'}\mathbf x+\underbrace{(W_2\mathbf b_1+\mathbf b_2)}_{\mathbf b'}.$$ **A 100-layer linear network is a 1-layer linear network.** The counting that lands (`brief-foundations §10`): $784\to512\to10$ has $401{,}408+512+5{,}120+10=\mathbf{407{,}050}$ params; the equivalent single $784\to10$ layer has $784\times10+10=\mathbf{7{,}850}$. **98 % of the parameters are provably redundant** — exactly, not approximately. Insert one nonlinearity between the layers and no such $W'$ exists. **`.box key`: "The nonlinearity contributes zero parameters and does all the work."**
- *XOR by hand (the most convincing 10 minutes — `brief-foundations §9`).* Show the 4 points; no single line separates them. Then give explicit weights and make him verify: $h_1=\mathrm{ReLU}(x_1+x_2)$, $h_2=\mathrm{ReLU}(x_1+x_2-1)$, $y=h_1-2h_2$. Carry all four: $(0,0)\to0$; $(0,1)\to1$; $(1,0)\to1$; $(1,1)\to2-2=0$. "Four lines of arithmetic and you built something a linear model provably cannot." The ReLU does *all* the work — $h_1,h_2$ are the same line shifted; only the kink distinguishes them. *(ReLU's definition/derivative came from page 04's table; the full menu of alternatives is page 11.)*
- **`.box rule` — the bridge to page 11:** "You now *need* a bend, and you have proof. Next page: which bend, and why one of them (ReLU) quietly won the 2010s." Do not preview the taxonomy here; leave the need standing.
- **Misconceptions → `.box warn`:** *neurons ≠ brain neurons* (1943 marketing; say once, drop it — `brief-foundations §9`); *bias is a minor detail* → without $b$ every hyperplane passes through the origin — teach it as essential-in-principle (its frequent absence in modern LLMs is a page-11 point); *"more neurons = more layers"* → width and depth are different axes (`brief-foundations §9`); *"a deep linear net is more expressive than a shallow one"* → the 4-line proof says no, identical function class.
- **`.deepdive`:** "Depth vs width, per parameter." Two ReLU nets with ≈ equal params — deep-narrow vs shallow-wide — carve boundaries of different intricacy; the deeper one composes more linear regions per parameter. Empirical, honest (the UAT caveat is page 11). Reassurance-grade.

**Demo — "Linear Collapse Playground"** *(flagship; real in-browser training — build O; `brief-foundations §10`.)*
- **Primitive:** raw `<canvas>` heatmap for the decision surface + `makeCtrl`; a real MLP trained live (`NN.MLP` + `NN.Trainer` + `NN.makeDataset`, `rAF` stepper).
- **Plot:** 2-D classification field, dataset toggle {moons, circles, xor, spiral} via `NN.makeDataset`. Decision boundary rendered as a blue→red heatmap, redrawn as training proceeds.
- **Controls (≤3 + dropdown):** depth $L$ [1,5] step 1; width [1,16] step 1; activation dropdown {**identity, ReLU**} (deliberately just these two — the sigmoid/GELU gradient-flow comparison is page 11's demo, not duplicated here). Train/Reset buttons.
- **Exact math (JS):** `NN.MLP([2, …hidden…, 1], {act})`, `NN.Trainer({loss:'bce', opt:new NN.SGD(...)})`, stepped from `requestAnimationFrame`. Live readout: **"Effective function class: LINEAR"** (red) when activation=identity, else **"PIECEWISE-LINEAR, ~N regions"** with a live count of distinct ReLU activation patterns across the grid.
- **The aha (in his hands, in this order):** (1) identity + depth 5 + width 16 = 1400 knobs, boundary is a straight line and *stays* one no matter what he does — "he will try to break it; he can't" (this is the 4-line proof, felt); (2) switch to ReLU, depth 2 separates the moons; (3) depth 4 × width 4 vs depth 2 × width 16 (≈ equal params) — the deeper one carves a more intricate boundary (depth > width per param).

**Code artifact (`.box try`):** `code/xor_by_hand.py` — hardcodes the ReLU XOR weights above, runs all four inputs through `torch`, prints `[0,1,1,0]`. Then a 3-line "now collapse it": stacks two `nn.Linear` with no activation and shows it *cannot* reproduce XOR (loss floors). CPU, <1 s.

**Quiz (7; ≥1 numeric)**
1. *(numeric, tol 0)* Params in a $784\to512\to10$ two-layer *linear* net. **num 407050, tol 0**; `why`: collapses to a $784\to10$ map worth 7,850 (`brief-foundations §10`).
2. *(numeric, tol 0)* The ReLU XOR net on input $(1,1)$: $y=h_1-2h_2$. **num 0, tol 0** (`brief-foundations §9`).
3. *(diagnose the failure)* Your 5-layer net with `identity` activation won't separate two moons no matter the width. Cause: **[too few epochs / affine∘affine is affine — no nonlinearity ✓ / learning rate]**. Distractor "epochs" ← *"more capacity/time fixes it."*
4. *(concept)* A 100-layer linear network has the same function class as **[a 100-layer ReLU net / a 1-layer linear net ✓ / a random net]** (`brief-foundations §10`).
5. *(concept, geometry)* $\mathbf w\cdot\mathbf x+b=0$ is **[a point / a hyperplane with normal $\mathbf w$ ✓ / a curve]**, and $b$ sets its **[slope / distance from the origin $|b|/\lVert\mathbf w\rVert$ ✓]**.
6. *(concept, thread)* A neuron's $z=\mathbf w\cdot\mathbf x$ is largest when $\mathbf x$ **[is longest / points along the template $\mathbf w$ ✓ / is orthogonal to $\mathbf w$]** (callback page 03).
7. *(predict)* Insert one ReLU between two linear layers. A single collapsed $W'$ reproducing the network **[still exists / no longer exists ✓]** (`brief-foundations §10`).

**Thread touchpoints**
- `[THREAD:dot-product beat 3/…]` — $z=\mathbf w\cdot\mathbf x+b$ names the neuron as a dot product; callback to pages 01 and 03.
- `[THREAD:activation beat 0]` — the *need* for a bend is proved here; page 11 supplies and taxonomizes it.

**Cross-references:** → 01 (the dragged neuron, now in symbols), → 03 (dot product = the ruler/template), → 04 (ReLU's definition and derivative), → 06 (TN-1's own tanh/sigmoid layers), → 11 (which bend, and why ReLU won).

---

# 11 — activations-and-limits.html
**Title:** "Activations, Why ReLU Won, and the MLP's Honest Limits" · **Part I — The Machine** · **build: S** · no track class.

> **Closes Part I.** Salvaged from the old page 04's activation menu + UAT deepdive (the collapse/XOR half is now
> page 10). Names **SwiGLU before page 36 needs it** (Qwen3's FFN counting) and tells the universal-approximation
> story *honestly*. Build S (`decisions.md §D-21a`): copies the template's plotting primitives.

**Learning objectives**
1. Name what each 2026 activation is for, and know the *real* reason ReLU beat sigmoid (gradient flow, not speed).
2. Feel the $0.25^{32}$ underflow that killed deep sigmoid nets, and name ReLU's own sin (dying units) and its fix (GELU/SiLU).
3. Know SwiGLU is a *layer with learned parameters*, not an activation — the category error that breaks param counting — and that it is Qwen3's FFN.
4. State the universal approximation theorem *and its four silences* — no manufactured closure.

**PREDICT.** "Two hidden-layer choices, same everything else: one uses sigmoid, one uses ReLU, both 32 layers deep.
At the start of training, layer-1's gradient in the sigmoid net compared to the ReLU net is about **[the same / a
million times smaller / $10^{-20}$ — effectively zero in fp32 ✓]**. Commit, then run the gradient-flow demo."
*(Resolves: $0.25^{32}=5.4\times10^{-20}$, below fp32's smallest normal — the sigmoid net's first layer gets no
signal; `constants.md §9.3`.)*

**Section outline**
- *The honest summary first (`brief-foundations §11`).* **In 2026: SwiGLU inside LLM/DiT FFNs; GELU or SiLU everywhere else; softmax at the output for classification; sigmoid only for gates and binary outputs. Sigmoid and tanh as hidden activations are dead.** Everything below explains why.
- *The menu — what each is for (table, `brief-foundations §11`).* **Sigmoid** $\sigma(x)=\frac1{1+e^{-x}}$, range $(0,1)$, $\phi'_{\max}=0.25$ — binary output & gates only, **never** hidden. **Tanh** $(-1,1)$, $\phi'_{\max}=1.0$ — legacy/retired, **but it is TN-1's hidden activation on page 06** (a deliberate teaching choice: small, bounded, hand-computable). **ReLU** $\max(0,x)$ — still everywhere in vision. **GELU** $x\Phi(x)$, **SiLU** $x\sigma(x)$, **SwiGLU** $(\mathrm{SiLU}(xW_g))\odot(xW_u)$ — the FFN of essentially every post-2023 LLM incl. Qwen3.
- **The real reason ReLU won → `.box warn` (kill the myth, `brief-foundations §11`):** *not* "it's cheaper to compute" (the activation is a bandwidth-bound rounding error next to the matmul). The real reason is **gradient flow**: $\sigma'_{\max}=0.25$, so depth 32 gives $0.25^{32}=\mathbf{5.4\times10^{-20}}$ — **below fp32's smallest normal** (`constants.md §9.3`); ReLU's derivative is exactly 1 on the active path, so the product stays 1. This is the vanishing-gradient beat page 05's demo pre-loaded — now cashed. **Sigmoid's second sin:** not zero-centered ⇒ all next-layer inputs positive ⇒ same-sign gradients ⇒ zigzag; tanh fixed *that* (zero-centered, $\phi'_{\max}=1.0$) and still lost, because $\phi'\to0$ for $|x|>2$ — the saturation is the killer. **ReLU's own sin: dying ReLU** — a unit with $z<0$ for all inputs has gradient 0 forever — which is why **GELU/SiLU** (smooth, nonzero negative-side gradient) won in transformers. GELU's intuition (better than "smooth ReLU"): it multiplies $x$ by $\Phi(x)$, the probability a standard normal is below $x$ — a soft stochastic gate (`brief-foundations §11`).
- **SwiGLU is a layer, not an activation → `.box warn` (`brief-foundations §11`, name it before page 36).** $(\mathrm{SiLU}(xW_g))\odot(xW_u)$ has **learned parameters** $W_g$ (and $W_u$) — calling it "an activation" is a category error that breaks param counting. It uses both $\odot$ (Hadamard) and $@$ (matmul) in one line — the cash-in of page 02's $\odot\ne@$ warning. **Qwen3's FFN is SwiGLU; page 36 counts on this name already being on the table.**
- **Misconceptions → `.box warn`:** *ReLU won because it's cheap* (the named folk myth — it's gradient flow); *SwiGLU is an activation* (the category error above); *"bias is always present"* → modern LLMs (Qwen3: `attention_bias:false`, `constants.md §1.1`) drop biases because RMSNorm makes them redundant — teach bias as essential-in-principle (page 10) but note its frequent absence (anticipates "I've read the config, there are no biases"); *"a wide enough net can learn anything, so architecture doesn't matter"* → the UAT trap, addressed next.
- **The MLP's honest limits — UAT stated properly (inline, the page's closing beat, `brief-foundations §12`).** Cybenko/Hornik: a wide-enough one-hidden-layer net approximates any continuous $f$ on a compact set. **But the theorem is silent on four things that matter:** it doesn't bound the *width* (could be astronomically large); it doesn't say gradient descent can *find* the parameters; it promises nothing about *generalization* to unseen data; and it says nothing about *depth* (why deep beats wide in practice). **`.box key`: "Universal approximation says a solution exists. It does not say you can afford it, train to it, or trust it off the training set. Everything after Part I is about those four gaps."** Flag the genuine open question (why overparameterized nets generalize) as unsettled — do not manufacture closure (`brief-foundations §12`, `brief-foundations §4` echo).
- **`.deepdive`:** "The 2/3 rule — why a config says $d_{\text{ff}}=12288$." SwiGLU needs 3 matrices where vanilla FFN needs 2; to match params you shrink $d_{\text{ff}}$ by $2/3$. Qwen3 uses $12288=3d$ (closer to the honest $2.67d$ than Llama's $3.5d$ — it "spent less"). (`constants.md §1.1`, D-01; reassurance-grade.)
- **`.milestone` (data-progress ~11, closes Part I):** "You've built the whole forward machine — from one dragged neuron to a real network, the probability it samples, the softmax it decides with, and the bend that makes it more than linear algebra. Part II answers the question page 6 left open: **how does it learn?**"

**Demo — "Activation Zoo + Gradient Flow"** *(build S; the $0.25^{32}$ death, felt; `brief-foundations §11`.)*
- **Primitive:** `Plot` (two panels) + `makeCtrl`; a real forward/backward through a stack of identical layers using `NN.MLP` (or a hand-rolled per-layer derivative product).
- **Panel 1 — the function and its derivative:** activation dropdown {sigmoid, tanh, ReLU, GELU, SiLU}; plot $\phi(x)$ and $\phi'(x)$ over $[-6,6]$; mark $\phi'_{\max}$ (0.25 for sigmoid, 1.0 for tanh/ReLU-active). Readout of $\phi'_{\max}$.
- **Panel 2 — deep gradient flow:** depth slider $N$ [1,40]; compute the layer-1 gradient magnitude as the product of per-layer $\phi'$ factors (unit inputs); plot it log-scale vs $N$ for the chosen activation. Overlay the fp32 min-normal floor ($1.18\times10^{-38}$).
- **Exact math (JS):** for sigmoid, `Math.pow(0.25, N)`; general, product of sampled $\phi'$ at typical pre-activations; compare against `1.18e-38`. At $N=32$ sigmoid reads `5.4e-20`; ReLU holds at ~1.
- **The aha:** "drag depth to 32 with sigmoid and the layer-1 gradient crosses into scientific-notation death ($5.4\times10^{-20}$); switch to ReLU and it stays at 1. ReLU didn't win because it's fast — it won because it's the only one whose gradient survives the trip."

**Code artifact (`.box try`):** `code/activations.py` — ~20 lines: tabulate $\phi'_{\max}$ for sigmoid/tanh/ReLU/GELU; print $0.25^{32}=5.4\times10^{-20}$ and compare to `torch.finfo(torch.float32).tiny`; then build a 32-layer sigmoid stack and a 32-layer ReLU stack in `torch`, one backward each, and print the layer-1 grad norms (sigmoid ≈ 0 in fp32, ReLU healthy). CPU, <1 s. "The myth said 'ReLU is faster.' Your own box says 'ReLU is the one whose gradient isn't zero.'"

**Quiz (7; ≥1 numeric)**
1. *(numeric, tol 1e-20 relative — accept order of magnitude)* $\sigma'_{\max}$ raised to depth 32: $0.25^{32}$. **num 5.4e-20, tol 5e-20** (`constants.md §9.3`). `why`: below fp32's smallest normal — layer-1's gradient is *zero* in fp32.
2. *(concept, myth)* ReLU displaced sigmoid mainly because **[it's cheaper to compute / its derivative is 1 on the active path so gradients don't vanish ✓ / it's smoother]**. Distractor "cheaper" ← the named folk myth; "smoother" ← the GELU confusion.
3. *(concept)* SwiGLU is **[an activation function / a layer with learned parameters $W_g$ ✓]**. Distractor ← the named category error that breaks param counting (page 36).
4. *(diagnose)* A ReLU unit outputs 0 for every input in your dataset and never recovers. That's **[a bug in autograd / a dead ReLU: $z<0$ always ⇒ gradient 0 forever ✓ / expected and harmless]**, and a smooth activation like **[sigmoid / GELU or SiLU ✓]** avoids it.
5. *(concept, UAT)* The universal approximation theorem guarantees **[that gradient descent will find the weights / that a wide-enough net can *represent* the function ✓ / that the net will generalize]**. The other two are exactly what it does *not* promise (`brief-foundations §12`).
6. *(transfer)* Qwen3's config has `attention_bias: false`. That's fine because **[bias is never useful / RMSNorm + the residual stream make it redundant ✓ / it's a bug]** (`constants.md §1.1`).
7. *(concept)* Why is tanh (zero-centered, $\phi'_{\max}=1.0$) still dead as a hidden activation? **[It isn't / because $\phi'\to0$ for $|x|>2$ — it still saturates ✓ / because it's slow]** (`brief-foundations §11`).

**Thread touchpoints**
- `[THREAD:activation beat 1/…]` — the taxonomy and the real ReLU story; SwiGLU named for page 36.
- `[THREAD:chain-rule beat 0 reopened]` — page 05's stacked-sigmoid stall is now explained as $0.25^{N}$ and closed.
- `[THREAD:Qwen3-8B beat 5]` — SwiGLU / no-bias / $d_{\text{ff}}=12288$ tie the activation story to the anchor before the page-36 count.

**Cross-references:** → 02 ($\odot\ne@$, cashed at SwiGLU), → 04 ($\sigma'=0.25$ introduced), → 05 (the stalled sigmoid, now explained), → 06 (why tanh/sigmoid are fine for TN-1 but dead in general), → 10 (the collapse that made a bend necessary), → 36 (SwiGLU FFN counted), → 12+ (Part II: how it learns).

---

## SPEC-WRITER NOTES (for the architect / adjacent spec writers)

**Cross-part seams the Part II / diffusion spec writers must match (all against `decisions.md §D-21a`):**
- **Backprop is Part II's alone.** Part I ends at "the chain rule *is* backpropagation" (page 05) and one verifying `.backward()` (page 05) — the reverse pass over TN-1 is **page 14**, autograd + the float beat is **page 15**, the first SGD step + the 10.22× spread is **page 17**. The exported blocks (see REPAIR LOG at the top of this file) must land there; page 15 **owns** the `constants.md §8.4` "exactly zero is false" `.box key` and value `1.4020191230201817e-08` (constants §8.4's "page ~10" pointer is stale).
- **Page 09 precedes page 15 on the "paper-math ≠ machine-math" thread.** Underflow (09) is beat 0; autograd's `1.4e-08` (15) is the reprise. Page 15 must reference *back* to 09, not forward.
- **Page 08 owns the first teaching of reparameterization $x=\mu+\sigma\varepsilon$.** `brief-diffusion §2` (line ~218) calling the VAE page "the first place" is superseded by §D-21a page 8; the VAE page (53) and forward-process page (54, trig payoff #3) **reopen**, not re-teach, and may cite "trunk page 8." Pages 53–57 depend on it.
- **MSE = Gaussian NLL is page 12's**, seeded on both 08 and 09. Neither Part I page derives it.
- **Page 12 (Part II)** derives CE and MSE from one MLE principle and must reuse the canonical logits `[2.0,1.0,0.1]`→0.417030 (`constants.md §9.2`) and the $p-y$ result stated on page 09; it inherits `[THREAD:softmax-CE]` and `[THREAD:loss-unification]`.
- **Page 13 (Part II)** owns rigorous gradient descent: $\eta_{\text{crit}}=2/\lambda_{\max}$ on a quadratic (`constants.md §9.3` ravine $\lambda_{\max}=20\Rightarrow0.1$), saddles-not-minima, and the anisotropic-valley zigzag that page 04 deliberately deferred. Page 04 installs only the 1-D LR→NaN playground and the update-rule vocabulary.
- **Page 30 (attention)** and **pages 33/54 (RoPE / forward process)** are the payoffs of page 03's dot-product/trig thread. Attention reopens $\sqrt{d_{\text{head}}}=11.3137$ (`constants.md §5`); 33 is trig payoff #2, 54 is trig payoff #3. Page 03 forward-references all three.
- **Page 18 (memory ledger)** inherits `[THREAD:memory]` beats seeded on 01/02/04/05 (16.38 GB weights; activations-cached→VRAM; LR-vs-trainable-params) and owns the full LR table (`constants.md §9.4`) and the measured-budget beat (`constants.md §6.8`). **Part I states no memory budget.**
- **Page 36 (Qwen3 count)** needs "SwiGLU" already named — page 11 does that.
- **Canonical visuals** (TN-1's graph; the shape ribbon) are established on pages 06/02 via `NN.worked221()` + `viz.NetGraph`; every later TN-1 beat reuses that asset, never redraws it (`brief-pedagogy §5.5`, `notation.md §10`). Page 06 uses only the forward fields; page 14 adds the backward animation.
