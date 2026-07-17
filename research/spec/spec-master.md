# spec-master.md — THE GLOBAL BUILD HEADER · read this BEFORE your part spec

**Status: ASSEMBLED & CONSISTENCY-CHECKED 2026-07-16.** This is the front matter every page-builder reads first,
then opens its part spec (`spec-part1..7.md`) and the code ladder (`spec-code.md`). It fixes the things a fan-out
build gets wrong: the authority chain, the unit rule, the notation pointer, the page rhythm, the shared vocabulary,
the QA checklist, and the **final 66-entry page table** (index + 65) that `assets/course-map.js` is generated from.

> **The course in one line:** *Artificial Neural Networks — From Algebra to Fine-Tuning on Your Own Box.* 65
> interactive HTML pages + a home page, static `file://`, no build step, vanilla JS + local KaTeX. **One learner:**
> adult, rusty-but-once-trained, owns the DGX Spark the ground-truth was measured on, heavy ComfyUI video-diffusion
> user. **He owns the hardware and will check every number.** Today: 2026-07-16.

---

## 1. THE AUTHORITY CHAIN — non-negotiable, in this order

1. **`hardware-ground-truth.md`** — measured on his actual DGX Spark this session. **Outranks everything** for
   hardware facts (usable memory 121.6875 GiB, no sm_121 binary yet it runs, ComfyUI-only venv, his 12 GB of LoRAs).
2. **`constants.md`** — frozen numbers. **Cite verbatim with the confidence tag.** No agent edits it. If a number
   you need is absent, flag the architect — never invent it.
3. **`notation.md`** — frozen, prescriptive notation. No page invents a symbol.
4. **`decisions.md`** — ratified rulings (D-01…D-21) + the page budget.
5. **The briefs** (`brief-*.md`) — raw material only. **Where a brief contradicts a frozen file, the frozen file
   wins** (briefs predate two verification passes; retired claims carry banners).

**The prime directive (constants.md preamble): never launder an estimate into a fact.** Three of the most quotable
numbers in the source corpus (the "1 PFLOP", the "128 GB budget", the "2.6 s FLUX image") were laundered estimates
or category errors, now corrected. If a page needs a crisper number than the frozen files give, **the honest page
is the one that says "measure it."**

**Confidence tags — print the tag to the learner wherever the course prints the number:**
`[VP]` verified-primary → as fact · `[DER]` derived → as fact, show the derivation · `[INF]` inferred →
**must be labelled "inferred, not published"** · `[EST]` estimate/folklore → **must be labelled an estimate, never
printed as bare fact** · `[MEA]` learner measures it, course ships the script → state the method, not the number ·
`[MEA-DEV]` measured on his box this session → as fact.

---

## 2. THE UNIT CONVENTION — a hard rule (constants.md §0)

The single most consequential error in the source corpus was a GB/GiB mismatch. Therefore:

| Quantity | Unit |
|---|---|
| Anything compared against **hardware capacity** | **GiB** ($2^{30}$ B) |
| Model weights / optimizer state / dataset **quoted on their own** | **GB** ($10^9$ B) |
| Bandwidth | **GB/s** ($10^9$ B/s) |
| KV cache | **KiB/token, GiB total** |

`1 GiB = 1.0737 GB` · `1 GB = 0.9313 GiB`. **Every page stating a memory figure states the unit and never mixes
the two in one comparison.** When comparing a footprint to the box, convert to GiB first and say so.

---

## 3. NOTATION — pointer + the five hard invariants (notation.md)

Read `notation.md` in full; it is normative. The five invariants, checkable mechanically:

1. **$W$ is ALWAYS $(d_{\text{out}}, d_{\text{in}})$** — in both the column-vector maths ($\mathbf y = W\mathbf x + \mathbf b$)
   and the row-batched code ($Y = XW^\top + \mathbf b^\top$). Demo the convention with `nn.Linear(2,3)` → `(3,2)`, **never `(2,2)`**.
2. **$\theta$ = learnable parameters only. Geometric angles are $\vartheta$.**
3. **$t$ = diffusion timestep · $k$ = optimizer step · $i$ = sequence position.** Three symbols, three jobs.
4. **Every equation with >3 distinct symbols carries a Symbol Ledger** (with the "From" column), including inside collapsibles.
5. **Every multi-step tensor computation carries a shape ribbon.**

**The $\sigma$ rule (notation §5, ratified):** $\sigma(\cdot)$ *with an explicit argument* = logistic sigmoid, **trunk
only**; $\sigma_t$ *subscripted, no argument* = diffusion noise std, **diffusion track only**; they never share a page.
Diffusion pages needing a sigmoid write `SiLU`/`torch.sigmoid`. **Diffusion timestep convention, course-wide:** $t=0$
is data, $t=T$ (DDPM) or $t=1$ (flow matching) is noise; sampling runs $t$ downward. (DDPM is **not** "opposite" to
flow matching — both put data at $t=0$; see notation §6.) Every deviation from field convention carries a one-line
**Translation Table** note on the page where it first bites.

---

## 4. THE PAGE RHYTHM & SEQUENCING (brief-pedagogy)

**Per-page rhythm:** **predict → intuition → math → worked number → demo → quiz** (open with a PREDICT prompt where
the rhythm calls for one; a quiz of 5–8 closes). **Sequencing principle:** **run → derive → break → generalize**,
never derive → run. **The Early Real Thing:** something real runs by **page 6**, and the gap between runnable
artifacts never exceeds ~4 pages. Front-load the payoff language from ~page 10 ("this is the exact optimizer you'll
use on your LoRA in Part V"). **Expertise-reversal tax:** do not re-explain what the learner demonstrably owns
(anti-pattern #20) — later encounters of a recurring object **reopen** it as a tagged callback, they do not re-teach it.

**The four recurring threads — tag every beat `[THREAD:NAME beat n]`** (notation §10): **the dot product** (the atom),
**TN-1** (the object / the spine), **the chain rule** (the verb), **the memory budget** (the reality check). Each beat
states: same object, what's new, which prior beat to reopen. Canonical visuals (TN-1's graph, the loss surface, the
shape ribbon, the $(a_t,b_t)$ route map) are drawn **once** and reused, never redrawn.

---

## 5. THE TEMPLATE CONTRACT (what builders copy)

Copy `TEMPLATE-example.html`. **Page shell:** `<body data-page="NN-slug.html">` (must equal the filename **and** the
`file:` field in `course-map.js`) → `#layout > main#content > div.inner`; `<span class="kicker">`, `<h1>`,
`<p class="subtitle">`. **Scripts at end of body, in this exact order:** `course.js` → `course-map.js` → `nn.js` →
`viz.js` → the page IIFE inside **one** `DOMContentLoaded` (it fires after course.js's own, so `buildNav()` /
`wireCodeblocks()` / `wireMilestones()` have already run). Math: `$inline$` / `$$display$$` (KaTeX pre-wired;
double-escape backslashes inside JS quiz strings, e.g. `"\\sqrt"`).

**Assets available:**
- **`course.js`** — `makeCtrl(parent,{label,min,max,step,value,unit,fmt})`, `new Plot(canvas,{xmin,xmax,ymin,ymax,xlog,ylog,xlabel,ylabel})`
  (`.clear/.grid/.trace/.hline/.vline/.label`), `renderQuiz(id,[{q,opts,a,why}|{q,num,tol,unit,why}],title)`, `eng()`,
  `vizUtils.cssVar`, `wireCodeblocks()`, `buildNav()`.
- **`nn.js`** (`window.NN`) — scalar-autograd `Value` (PyTorch accumulation semantics), `Mat` tensor layer (teaching-grade
  shape errors), `MLP`/`Trainer` that really train in-page (rAF stepper), `SGD`/`Adam`, seeded `RNG`,
  `makeDataset(moons/circles/xor/spiral/blobs)`, `gradCheck`, **`worked221()`** (⚠️ its default reproduces the RETIRED
  §5.4 network — always pass explicit TN-1 config), `attention(Q,K,V,{causal,temperature})` (weights + entropy +
  gradient-health), stable/unstable `softmax`, `layerNorm`/`rmsNorm`, `bceWithLogits`, `crossEntropy`.
- **`viz.js`** — `Heatmap`, `NetGraph` (forward **and** backward), `Surface3D` (contour default + wireframe),
  `TensorViz` (matmul animation), `Timeline` (training strip), `hl.block` (syntax highlighter for `.codeblock`).

**Demos compute the page's real equation live — nothing is faked** (anti-pattern #11); the demo shows the equation with
live values substituted (#12).

**Box / track / milestone vocabulary:**
- **Boxes:** `.box key` (the one-sentence takeaway) · `.box rule` (a hard rule/invariant) · `.box warn` (a
  misconception being corrected — each distractor must trace to a named misconception) · `.box worked` (a worked number
  terminating in a decimal) · `.box try` (the `.box try` code artifact the learner runs; names a `code/…` file from
  `spec-code.md`). `.deepdive` = collapsible whose title **states** its content (never teases) and which **nothing later
  depends on**.
- **Tracks:** `.track-llm` / `.track-diffusion` modifiers on `<body>`, `.box`, `.demo`, or a `.track-banner`. A
  `course-map.js` section whose pages all share one track gets that track's rail colour automatically.
- **Milestones:** `<div class="milestone" data-progress="N">` on the page + `milestone: true` in `course-map.js` (→ 🚩
  in the sidebar). Kept sparing: **six total** — pages 06, 24, 36, 52, 62, 65 (one per phase of the spiral).

---

## 6. THE BUILDER QA CHECKLIST — a page passes only if all hold

A page is a defect if it fails any of these (notation §9 is the master list; the two a fan-out fails most are #22/#23):

- [ ] **No number contradicts `constants.md`** (#22). Every cited number is verbatim with its section named.
- [ ] **No `[INF]`/`[EST]`/`[MEA]` number is printed as bare fact** (#23) — the label ships to the learner.
- [ ] Units follow §2; no GB/GiB mix in one comparison.
- [ ] $W$ is $(d_{\text{out}},d_{\text{in}})$; no `nn.Linear(2,2)` used to show the convention; $\vartheta$ not $\theta$
      for angles; $t$/$k$/$i$ used correctly; no $\sigma(\cdot)$ on a $\sigma_t$ page.
- [ ] Every >3-symbol equation has a Symbol Ledger; every tensor op has a shape ribbon.
- [ ] Every worked example terminates in an actual decimal; every demo computes the page's equation live with values shown.
- [ ] The quiz has 5–8 questions, none answerable from vocabulary, each distractor tied to a named misconception, ≥1
      numeric with a tolerance.
- [ ] Every recurring-thread beat is tagged `[THREAD:… beat n]` and reopens (not re-teaches) prior beats.
- [ ] Every `.box try` names a file that `spec-code.md` defines; no code has elided imports or `# ...`.
- [ ] `<body data-page>` equals the filename equals the `course-map.js` `file:` field.
- [ ] No collapsible holds content a later page depends on; no "germane load" justifications; nothing re-explains what
      the learner owns.
- [ ] Cross-references cite the **final** page numbers in §7's table.

### 6b. PILOT-RATIFIED POLICIES (2026-07-16 — added after the 3-page pilot; binding on every builder)

1. **The spec is NOT authoritative on numbers — constants.md is.** For every numeric your spec entry hands
   you, diff it against constants.md (or independently recompute it) BEFORE transcribing. On any mismatch:
   use the constants-consistent value, keep the page internally consistent with it, and emit a loud
   FROZEN-NUMBER DISCREPANCY note in your return. (Observed: spec-part6 shipped √ᾱ₅₀₀=0.358 — a spec
   arithmetic slip; the true schedule value is 0.280. Now corrected, but assume more exist.)
2. **One PREDICT component.** Every PREDICT prompt uses `.box predict` (🔮 title bar) — never `.box try`
   (which stamps "RUN THIS"), never `.box key`, never a bespoke div.
3. **Every PREDICT is RESOLVED on-page.** The learner must find out whether their prediction was right
   without leaving the page — via the demo landing on it or a stated answer. Deferring the *reproduction*
   to a code artifact is fine; deferring the *answer* is a defect.
4. **Sig-fig policy for live readouts.** When a demo surfaces a value that also appears as a frozen constant
   on the same page, display it at the frozen value's precision (or one fewer sig-fig) so rounding can never
   make the demo contradict the table by a last digit. If the live and frozen last digits diverge, round the
   readout to match or suppress that readout (as p.14 does with z₂). Never let the learner see 0.0064-vs-0.0063.
5. **One Symbol Ledger component.** Every Symbol Ledger uses `table.ledger` (styled in course.css) — no
   bespoke local styles, no bare tables.

---

## 7. THE FINAL PAGE TABLE — index + 65 (the source of `assets/course-map.js`)

**REGENERATED 2026-07-16 against `decisions.md §D-21a` (THE CANONICAL PAGE TABLE) after the fidelity repair.** Titles
paraphrase §D-21a's owner-topics; **numbering and content allocation are §D-21a's and are FROZEN.** **Final # = D-21 #**
for pages 1–51; for 52–65 the GRPO renumber applies (GRPO capstone = 52; D-21 diffusion 52–61 → 53–62; D-21 capstone
62–64 → 63–65). Track: LLM ◆ / DIFF ◇.

**Build model:** `O` = Original (bespoke flagship demo / spine beat — build fresh, protect it) · `S` = Standard (copies
template primitives with page-specific parameters). **Build letters for pages 1–43 are taken from §D-21a's printed
O/S column** (the repair set them there). ⚠️ **Known convention collision (escalated, repair-agent-3 flag):** §D-21a's
O/S column names the intended *build agent* (Opus/Sonnet); this master's `O`/`S` historically named *page complexity*
(Original/Standard). They correlate but are not the same axis. Pages 44–65 keep each part spec's self-declared
complexity letter (those tracks were not part of the fidelity repair). **The course-wide meaning of the `Build:` field
is the lead's to fix; it does not block the build.**

| # | (D-21) | file | title | Part | Build | Track | 🚩 |
|--|--|--|--|--|--|--|--|
| — | — | `index.html` | Start Here | Welcome | — | — | |
| — | — | `TEMPLATE-example.html` | ◈ Template & Component Gallery | Welcome | — | — | |
| 1 | 1 | `01-what-is-a-network.html` | What a Neural Network Actually Is | I — The Machine | S | | |
| 2 | 2 | `02-vectors-and-shapes.html` | Vectors, Matrices, Tensors | I — The Machine | S | | |
| 3 | 3 | `03-the-dot-product.html` | The Dot Product — the Atom | I — The Machine | O | | |
| 4 | 4 | `04-derivatives-and-the-flip.html` | Derivatives & the θ-vs-x Flip | I — The Machine | O | | |
| 5 | 5 | `05-the-chain-rule.html` | The Chain Rule | I — The Machine | O | | |
| 6 | 6 | `06-tn1-early-real-thing.html` | TN-1: Your First Real Network | I — The Machine | O | | 🚩 |
| 7 | 7 | `07-row-or-column.html` | Row or Column? The Shape Bridge | I — The Machine | S | | |
| 8 | 8 | `08-probability-and-the-gaussian.html` | Probability, the Gaussian & μ+σε | I — The Machine | O | | |
| 9 | 9 | `09-logs-softmax-cross-entropy.html` | Logs, Softmax & Cross-Entropy | I — The Machine | O | | |
| 10 | 10 | `10-neuron-xor-collapse.html` | The Neuron, XOR & Linear Collapse | I — The Machine | O | | |
| 11 | 11 | `11-activations-and-limits.html` | Activations & the MLP's Limits | I — The Machine | S | | |
| 12 | 12 | `12-loss-from-mle.html` | Loss from Maximum Likelihood | II — How It Learns | O | | |
| 13 | 13 | `13-gradient-descent-landscape.html` | Gradient Descent & the Landscape | II — How It Learns | O | | |
| 14 | 14 | `14-backprop-by-pencil.html` | Backprop by Pencil: Nine Gradients | II — How It Learns | O | | |
| 15 | 15 | `15-autograd-float-reality.html` | Autograd & the Float That Isn't Zero | II — How It Learns | O | | |
| 16 | 16 | `16-minibatch-and-noise.html` | Batch, Mini-Batch & Noise | II — How It Learns | S | | |
| 17 | 17 | `17-optimizers-sgd-to-adamw.html` | Optimizers: SGD → AdamW | II — How It Learns | O | | |
| 18 | 18 | `18-memory-ledger.html` | The Memory Ledger | II — How It Learns | O | | |
| 19 | 19 | `19-init-and-gradient-flow.html` | Initialization & Gradient Flow | II — How It Learns | O | | |
| 20 | 20 | `20-normalization.html` | Normalization | II — How It Learns | O | | |
| 21 | 21 | `21-lr-schedules-warmup.html` | LR Schedules & Warmup | II — How It Learns | S | | |
| 22 | 22 | `22-regularization.html` | Regularization & the Two Curves | II — How It Learns | S | | |
| 23 | 23 | `23-bias-variance-double-descent.html` | Bias-Variance & Double Descent | II — How It Learns | S | | |
| 24 | 24 | `24-first-real-finetune.html` | Your First Real Fine-Tune | II — How It Learns | S | | 🚩 |
| 25 | 25 | `25-architecture-inductive-bias.html` | Architecture Is Inductive Bias | III — Architecture | S | | |
| 26 | 26 | `26-convolutions-the-unet-atom.html` | Convolutions: The U-Net Atom | III — Architecture | O | | |
| 27 | 27 | `27-embeddings-and-the-dot-product.html` | Embeddings & the Dot Product | III — Architecture | S | | |
| 28 | 28 | `28-rnns-why-attention.html` | RNNs & Why Attention Had to Exist | III — Architecture | S | | |
| 29 | 29 | `29-attention-soft-lookup.html` | Attention I: The Soft Lookup | III — Architecture | O | | |
| 30 | 30 | `30-attention-scale-and-heads.html` | Attention II: √d_head & Heads | III — Architecture | O | | |
| 31 | 31 | `31-self-vs-cross-attention.html` | Attention III: Self vs Cross | III — Architecture | O | both | |
| 32 | 32 | `32-causal-masking.html` | Causal Masking: The LLM Hinge | III — Architecture | S | | |
| 33 | 33 | `33-positional-encoding-rope.html` | Positional Encoding & RoPE | III — Architecture | O | | |
| 34 | 34 | `34-residuals-and-normalization.html` | Residuals & the Residual Stream | III — Architecture | O | | |
| 35 | 35 | `35-the-transformer-block.html` | The Transformer Block, Assembled | III — Architecture | O | | |
| 36 | 36 | `36-parameter-counting-qwen3.html` | Parameter Counting on Qwen3-8B | III — Architecture | O | | 🚩 |
| 37 | 37 | `37-attention-at-scale.html` | Attention at Scale: O(S²) & KV | III — Architecture | O | | |
| 38 | 38 | `38-scaling-laws-moe-emergence.html` | Scaling Laws, MoE & Emergence | IV — Adaptation | S | | |
| 39 | 39 | `39-rank-and-svd.html` | Rank, SVD & Low-Rank Approximation | IV — Adaptation | O | | |
| 40 | 40 | `40-lora.html` | LoRA: Low-Rank Adaptation | IV — Adaptation | O | | |
| 41 | 41 | `41-number-formats-qlora.html` | Number Formats & QLoRA | IV — Adaptation | O | | |
| 42 | 42 | `42-three-fine-tunings.html` | 'Fine-Tuning' Names Three Things | IV — Adaptation | S | | |
| 43 | 43 | `43-your-box-roofline.html` | The Roofline: Your Box, Measured | IV — Adaptation | O | | |
| 44 | 44 | `44-tokenization.html` | Tokenization | V — LLM Fine-Tuning | S | ◆ | |
| 45 | 45 | `45-pretraining-base-instruct.html` | Pretraining, Base vs Instruct | V — LLM Fine-Tuning | S | ◆ | |
| 46 | 46 | `46-decoding-kv-roofline.html` | Decoding, KV Cache & the Roofline | V — LLM Fine-Tuning | O | ◆ | |
| 47 | 47 | `47-prompt-rag-finetune.html` | Prompt vs RAG vs Fine-Tune | V — LLM Fine-Tuning | O | ◆ | |
| 48 | 48 | `48-sft-preference-alignment.html` | SFT & Preference Alignment | V — LLM Fine-Tuning | O | ◆ | |
| 49 | 49 | `49-memory-ledger-setup.html` | Does It Fit? Ledger & Setup | V — LLM Fine-Tuning | O | ◆ | |
| 50 | 50 | `50-lora-applied.html` | LoRA on an LLM | V — LLM Fine-Tuning | O | ◆ | |
| 51 | 51 | `51-serving-eval-hardware.html` | Serving, Eval & Hardware Reality | V — LLM Fine-Tuning | O | ◆ | |
| 52 | (GRPO) | `52-grpo-rlvr-capstone.html` | Capstone: GRPO & Verifiable Rewards | V — LLM Fine-Tuning | O | ◆ | 🚩 |
| 53 | 52 | `53-the-vae-his-comfyui-runs-on.html` | Latent Space: The VAE | VI — Diffusion | O | ◇ | |
| 54 | 53 | `54-the-forward-noising-process.html` | The Forward Process: A Rotation | VI — Diffusion | O | ◇ | |
| 55 | 54 | `55-the-reverse-process-and-elbo.html` | The Reverse Process & the ELBO | VI — Diffusion | O | ◇ | |
| 56 | 55 | `56-score-matching-and-the-sde.html` | Score Matching & the SDE | VI — Diffusion | S | ◇ | |
| 57 | 56 | `57-flow-matching-your-scheduler.html` | Flow Matching & Your Scheduler | VI — Diffusion | O | ◇ | |
| 58 | 57 | `58-the-unet-in-depth.html` | The Denoiser I: The U-Net | VI — Diffusion | S | ◇ | |
| 59 | 58 | `59-from-unet-to-dit-mmdit.html` | The Denoiser II: DiT & MMDiT | VI — Diffusion | O | ◇ | |
| 60 | 59 | `60-latent-diffusion-and-the-roofline.html` | Latent Diffusion & the Roofline | VI — Diffusion | O | ◇ | |
| 61 | 60 | `61-conditioning-cross-attention-cfg.html` | Conditioning, Cross-Attn & CFG | VI — Diffusion | O | ◇ | |
| 62 | 61 | `62-fine-tuning-diffusion-on-your-data.html` | Fine-Tuning Diffusion on Your Data | VI — Diffusion | O | ◇ | 🚩 |
| 63 | 62 | `63-predict-your-box.html` | Predicting Your Own Box | VII — Capstone | O | | |
| 64 | 63 | `64-tracks-rejoin.html` | The Tracks Grew Back Together | VII — Capstone | O | both | |
| 65 | 64 | `65-where-next.html` | Where This Goes Next | VII — Capstone | O | | 🚩 |

**Ownership anchors (so tracks don't re-derive trunk material — D-14):** p.18 owns the memory ledger (16 B/param,
122.05 GiB / 121.6875 GiB [MEA-DEV]); p.40 owns the LoRA math (187× = the 8,190,735,360 / 43,646,976 param ratio);
p.41 owns number formats + QLoRA (NF4 = 0.516 B/param = 4.127 bits [VP]); p.43 owns the roofline (ridge **227 working
[INF]** / **458 alt if the ceiling is ~125 TF** — the 62-vs-125 TF question is genuinely unresolved and measured by
the learner, constants §6.3/§6.4). The LLM/diffusion adaptation pages **open with "as the trunk derived on p.40/41…"
and re-derive nothing.**

> **⚠️ Diffusion internal-ordering divergence (Part VI, flagged here per §D-21a's explicit instruction).** §D-21a lists
> the diffusion owner-topics as 58=U-Net→DiT→MMDiT, 59=latent-diffusion+samplers, 60=CFG, 61=LoRA-applied+datasets,
> 62=eval — and *permits* a different internal order for pages 56–60 **provided every owner-topic keeps a full page.**
> `spec-part6.md`'s chosen order (the built pages above) instead **splits the denoiser across 58 (The U-Net, in depth)
> + 59 (DiT & MMDiT)** and **folds LoRA-applied + datasets + the eval protocol together on the milestone page 62
> (Fine-Tuning Diffusion on Your Data).** All ten owner-topics are present; the "eval, why his eyes win" material lives
> as a full section + `.deepdive`s + `eval_lora.py` on p.62 rather than a standalone page, and the U-Net gets an extra
> page of depth. Cross-part references to diffusion pages (from Parts I/VII) only touch 53/54 (VAE, forward), which
> are stable under both orders, so nothing mispoints. **Recorded for the lead; not a blocking defect.**
>
> **⚠️ Filenames/titles provenance:** each part spec's page headers and `.box try` filenames are authoritative and
> match `course-map.js` verbatim. Content allocation and page numbering are **frozen** by §D-21a, this table, and
> `course-map.js`.

---

## 8. RESOLVED — the backprop/autograd double-home (closed by §D-21a)

**CLOSED.** The earlier open item (backprop/autograd appearing at both Part I 09/10 and Part II 14/15 in the divergent
fan-out) is resolved by §D-21a: **backprop lives HERE and only here — pages 14–15. Part I teaches forward + the chain
rule only** (`... Backprop lives HERE and only here — Part I teaches forward + chain rule only. Resolves the spec
integrator's escalation: ONE backprop/autograd home, pages 14–15.`). The fidelity repair restored Part I so that page
08 = probability/reparameterization, 09 = logs/softmax/CE, 10 = XOR/linear-collapse (NO autograd), 11 = activations;
the reverse pass over TN-1 is **page 14**, autograd + the "float that isn't zero" beat is **page 15** (which OWNS the
mandated `constants.md §8.4` framing, the `1.4020191230201817e-08`, and the `atol=1e-4`/`-0.7527` assertion), and the
first SGD step is **page 17**. `constants.md §8.4`'s "page ~10" pointer is **stale** (documented in spec-part1 p.10's
note, spec-part2's appendix, and spec-code §F.7) — the value/framing ship at page 15. `code/tn1.py` (p.06) ships
forward-only and accretes into `14_backprop_tn1.py` / `15_autograd_check.py` / `17_optimizer_race.py`. No structural
ambiguity remains.
