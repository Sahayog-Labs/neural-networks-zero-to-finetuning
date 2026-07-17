# BUILD SPECIFICATION — PART VI · TRACK B: DIFFUSION (pages 53–62)

**Scope.** The ten diffusion-track teaching pages, **final numbers 53–62** (= **D-21: 52–61**). Track class
`.track-diffusion` on every page (add `.track-banner` at the top of page 53 only, the track opener). The
GRPO/RLVR capstone is page **52**; the capstone/rejoining pages (D-21 62–64 → **63–65**) follow this track and
are **not** in this spec — see the SEAMS section at the end for the handoff they depend on.

**Audience calibration — read before building any page.** This learner is **not a diffusion beginner**
(`hardware-ground-truth.md` §5: Wan2.2-14B video, FLUX.2-dev, Qwen-Image, `z_image_turbo`, `flux1-dev-kontext_fp8`,
`krea2_turbo_fp8`, **12 GB of LoRAs / 9 files** in `~/ComfyUI/models/loras/`, his own `bench_baseline/` renders).
He owns the **DGX Spark** the ground-truth was measured on. Every page here gives **theory under knobs he already
turns**: CFG scale, samplers, schedulers, LoRA strength, `denoise`, bucketing. **Expertise-reversal tax applies
hard** (anti-pattern #20): never explain what CFG *does* as if new — explain *why the knob behaves as he has
observed*. His FLUX.2 folder on disk is the primary exhibit for four separate pages (VAE facts, the flow-match
scheduler, the DiT config, the roofline).

**How to read this spec.** Each page entry is self-contained; a builder implements it 1:1 without opening any
brief. Build model **O** (Opus-built: novel/heavy live-math or a convergence demo) or **S** (Sonnet-built:
standard rhythm) is given per page — see the ⚠️ note in SEAMS: the literal D-21 per-page O/S column was not
present in the frozen file, so these are assigned by demo/math weight and must be reconciled if D-21's table
surfaces.

---

## GLOBAL CONVENTIONS FOR THIS TRACK (apply to every page 53–62)

**Notation (frozen — `notation.md`; conform exactly, anti-patterns #1–#9, #22–#24 are mechanically checked):**
- **General path:** $x_t = a_t\,\mathbf{x}_0 + b_t\,\boldsymbol\epsilon$. This $(a_t,b_t)$ table *is* the track's
  spine (notation §4.4). $\mathbf{x}_0$ clean latent, $\mathbf{x}_t$ noised, $\boldsymbol\epsilon\sim\mathcal N(0,I)$.
- **$t$ = diffusion timestep ONLY** ($0..T$, $T=1000$). Not sequence position ($i$), not optimizer step ($k$).
- **$\sigma_t$** (subscript, no argument) = noise std, diffusion track only. **Never write $\sigma(\cdot)$ with an
  argument on any of these pages** (anti-pattern #3) — for a sigmoid write `torch.sigmoid` / `logistic(·)` / `SiLU`.
- **$\psi_t \equiv \arccos\sqrt{\bar\alpha_t}$** is the rotation angle (notation §4.4). ⚠️ **The brief-diffusion
  source uses $\phi_t$; the FROZEN symbol is $\psi_t$. Use $\psi_t$ everywhere and add a one-line Translation note
  ("papers write $\phi_t$ or $\alpha_t$").**
- **$w_g$ = CFG guidance scale** (frozen; papers write $w$, $s$, `cfg`, `guidance_scale`). A bare $w$ never appears.
- $\bar\alpha_t=\prod_{s\le t}\alpha_s$, $\alpha_t=1-\beta_t$; $\epsilon_\theta$ noise net, $v_\theta$ velocity net,
  $\mathbf s_\theta$ score (bold). $\mathbf c$ conditioning (bold), $\varnothing$ null cond. $f$=VAE downsample=8,
  $C$=latent channels. $\eta$ = **learning rate only** — DDIM's stochasticity knob is code-font `eta`, prose
  "DDIM's stochasticity parameter", never a math symbol (notation §5.4).
- **Every equation with >3 symbols carries a Symbol Ledger** (notation §8); **every tensor op carries a shape
  ribbon** (notation §7). Height is $H_{px}/H_{lat}$, never bare $H$ ($H$ = attention heads).
- **Every printed number carries its `constants.md` confidence tag** and, for [INF]/[EST]/[MEA], the label the
  legend mandates. Never launder an estimate into a fact (the prime directive).

**Recurring threads to tag on these pages** (`[THREAD:…]`, notation §10):
- **Memory ledger** — 16 B/param; his measured **121.6875 GiB** MemTotal [MEA-DEV], **6.3125 GiB** carveout.
- **Chain rule** — the score $\mathbf s_\theta=\nabla_{\mathbf x}\log p$ is a gradient *w.r.t. the image*; CFG is a
  statement about scores; backprop reappears as denoising-score-matching.
- **The dot product (the atom)** — attention scores, the CFG direction $\epsilon_c-\epsilon_\varnothing$, LoRA's $BA$.
- **Qwen3-8B / the LLM track** — resurfaces where the text encoder *became an LLM* (page 61) and where FLUX.2's
  encoder is a 40-layer GQA Mistral-3 (foreshadow → the capstone cashes it).

**Rhythm (brief-pedagogy contract):** predict → intuition → math → worked number → demo → quiz. Every page opens
with a `PREDICT` box and closes with `renderQuiz`. No page is prose-only; demos compute the page's equation live
with values substituted in (anti-patterns #10–#12).

**Code artifacts** live in `G:/…/ANN-Course/code/` and **target models already on his disk** — FLUX.2-dev,
`flux1-dev-kontext_fp8`, `z_image_turbo`, his `models/loras/`. Each `.box try` names the file and the exact thing
he runs and checks against a printed number.

---

# 53 — the-vae-his-comfyui-runs-on.html  (D-21: 52)  ·  Part VI Track B · build: **O** · `.track-diffusion .track-banner`

**Title:** "Latent space: the VAE your ComfyUI already runs on"

**Objectives (4):**
1. State the generative goal $p(\mathbf x)$ and why the intractable normalizer $Z_\theta$ is what score/diffusion
   route around (callback to the trunk's MLE/likelihood material).
2. Read a VAE as encoder→bottleneck→decoder, and say what the KL term buys (a *navigable* latent, not "generative").
3. Recite the real latent shapes and compression ratios of the VAEs on his disk, from `constants.md` §9.6 verbatim.
4. Demonstrate that the SD/FLUX latent prior is **not** $\mathcal N(0,I)$ — which is precisely why diffusion is
   needed on top.

**PREDICT box:** "You have `VAEDecode` in every workflow. Sample a tensor of pure Gaussian noise at the FLUX latent
shape $[1,32,128,128]$ and decode it. Do you get (a) a plausible image, (b) structured but wrong, (c) colored
static? Write your guess."

**Section outline:**
- *Generative framing (inline, brief — he knows the goal).* We want to sample from $p(\mathbf x)$. Max-likelihood
  needs $p_\theta(\mathbf x)=\tilde p_\theta(\mathbf x)/Z_\theta$; $Z_\theta$ is an integral over all images —
  intractable. **This is the wall every later page climbs.** [THREAD:chain-rule beat — the escape is to work with
  $\nabla_{\mathbf x}\log p$, where $Z_\theta$ differentiates away; foreshadow page 56.]
- *GANs — 90 seconds, in a `.deepdive`* (D-14: GANs are diffusion-track, ½ page): mode collapse, no likelihood,
  and the fact he should keep — **the SD/FLUX VAE decoder is GAN-trained** (that's why his outputs aren't blurry
  even though a plain VAE is). State, don't dwell; "do not teach WGAN-GP."
- *Plain autoencoder → VAE (math, inline).* Encoder $q_{\theta_E}(\mathbf z\mid\mathbf x)=\mathcal N(\mu,\sigma^2 I)$,
  decoder $p_{\theta_D}(\mathbf x\mid\mathbf z)$. Loss = reconstruction + $\beta_{\text{KL}}\,D_{\mathrm{KL}}(q\,\|\,\mathcal N(0,I))$.
  **Reparameterization trick is trunk-owned — point back, do not re-derive** (D-19: taught early in the trunk's
  probability section); one line: $\mathbf z=\mu+\sigma\odot\boldsymbol\epsilon$. Symbol Ledger required.
  ⚠️ **Notation:** VAE params are $\theta_E,\theta_D$, not $\phi,\psi$ (both taken).
- *What KL actually buys — `.box key`.* Three answers, give all three briefly: (1) regularizes codes toward a
  known prior, (2) makes the latent **navigable** (no dead voids between codes), (3) it is **not** what makes the
  model generative. `.box warn` **Misconception:** "KL makes it generative." Truth: in LDM the KL is *weak*
  ($\beta_{\text{KL}}\approx$1e-6) and $\mathcal N(0,I)$ is a *bad* prior for the latent — which is exactly why you
  need diffusion on top. The predict-box payoff.
- *Real numbers — the VAEs on his disk (`.box worked`, cite `constants.md` §9.6 verbatim with tags):*
  - SD1.5 latent $[1,4,64,64]$ [VP]; compression $786{,}432/16{,}384=\mathbf{48\times}$ [DER].
  - FLUX.1 latent $[1,16,128,128]=262{,}144\to\mathbf{12\times}$ [VP]; **4096 tokens** after 2×2 patchify, width 64
    → projected to $d=3072$ [DER].
  - **FLUX.2 VAE (his disk): $f=2^{4-1}=\mathbf 8$ [VP], `latent_channels: 32` [VP], latent $[1,32,128,128]=524{,}288$
    → $\mathbf{6.0\times}$ compression [VP]; 4096 tokens after 2×2 patchify, width 128 [DER]; VAE = 84.0M params
    [VP, `hardware-ground-truth.md` §4].** The `.box try` has him `cat` his own `vae/config.json` and match every
    figure. ⚠️ **The DeepWiki "32:1 / [B,128,16,16]" claim is wrong** (it reported VAE∘patchify) — flag it so he
    doesn't trust it.
  - *Why 4→16→32 channels over time (inline):* the field is **buying back fidelity** it spent on compression.

**Demos:**
- **D53.1 — Latent shape/compression calculator.** `makeCtrl` sliders: model radio {SD1.5, FLUX.1, FLUX.2},
  resolution {512,1024,2048}. JS computes latent dims $=\lceil H_{px}/f\rceil$, element count $C\cdot H_{lat}\cdot W_{lat}$,
  compression $=(3\cdot H_{px}^2)/(C H_{lat} W_{lat})$, token count $(H_{lat}/2)(W_{lat}/2)$. `readout` via `eng()`.
  **Aha:** FLUX.2's 32 channels *cost* compression (6× vs FLUX.1's 12×) to *buy* fidelity — draggable.
- **D53.2 — "The prior is not Gaussian."** Precomputed: two decoded tiles side by side — (a) a real photo
  round-tripped through his FLUX.2 VAE (crisp), (b) $\mathbf z\sim\mathcal N(0,I)$ at $[1,32,128,128]$ decoded
  (colored static). This is the D2 code artifact rendered. **Aha (the predict payoff):** answer is (c); the latent
  manifold is a thin sheet, $\mathcal N(0,I)$ misses it → diffusion exists to *find* the sheet. Use `viz.js`
  `Heatmap` for the residual at 4× gain.

**Code artifact (`.box try`):** `code/vae_ceiling.py` (manifest D2) — encode/decode a real photo through **his SD1.5
(4ch) and FLUX.2 (32ch) VAEs**; prints shapes, compression **48× / 6.0×**, residual; then samples $\mathcal N(0,I)$
at latent shape and decodes → colored noise. "Your VAE round-trips your photo; your VAE's own prior gives static.
That gap is the whole rest of the track." *(30 s.)*

**Quiz (6; ≥1 numeric):**
1. Numeric: FLUX.2 compression ratio at 1024² (num **6.0**, tol 0.3). Distractor pull: 12× (FLUX.1), 48× (SD1.5).
2. MC: what does decoding $\mathcal N(0,I)$ at FLUX latent shape give? → colored static (distractor: "a random real
   image" = the "prior is Gaussian" misconception).
3. MC: KL term's job → makes the latent navigable (distractor: "makes it generative").
4. Numeric: FLUX.2 latent element count at 1024² (num **524288**, tol 0). 
5. MC: why the SD/FLUX VAE decoder isn't blurry like a textbook VAE → it's GAN-trained.
6. MC: FLUX.2 token count after 2×2 patchify at 1024² → 4096 (distractor: 16384 = forgetting the patchify).

**Threads:** [THREAD:memory-ledger — VAE is 84M params ≈ 0.17 GB, "rounding error" against his 121.69 GiB; the
weights are the mass, not the latent]. [THREAD:chain-rule — foreshadow score].
**Cross-refs:** trunk reparam-trick page; trunk MLE/likelihood page; → 54 (forward process), 60 (the latent-vs-pixel
attention ratio uses these shapes).

---

# 54 — the-forward-noising-process.html  (D-21: 53)  ·  build: **O** · `.track-diffusion`

**Title:** "The forward process: noising is a rotation" — **the trig payoff page (payoff #3)**

**Objectives (4):**
1. Write the Markov noising chain and prove variance-preservation in two lines of high-school algebra.
2. Derive the one-shot reparameterization (Eq. 3.3) and read $(\sqrt{\bar\alpha_t},\sqrt{1-\bar\alpha_t})$ as a
   point on the unit circle — the trig payoff.
3. Read the real $\bar\alpha_t$/SNR table and explain the nonzero-terminal-SNR bug from one row.
4. Separate the two objects his tooling confuses: the **$\beta$ schedule** (baked in the checkpoint) vs the
   ComfyUI **`scheduler`** dropdown (the sampling grid).

**PREDICT box:** "At $t=500$ of 1000 (linear DDPM), how much of the picture is left? Guess the fraction of *signal*
$\sqrt{\bar\alpha_{500}}$ before you read on." (Answer: 0.280 — far less than 'half'. [Corrected per constants §9.6 endpoints; the earlier 0.358 was a spec arithmetic slip — pilot review 2026-07-16.])

**Section outline:**
- *The chain has zero learned parameters — `.box warn`.* $q(\mathbf x_t\mid\mathbf x_{t-1})=\mathcal N(\sqrt{1-\beta_t}\,\mathbf x_{t-1},\beta_t I)$
  is a *ruler*, not a model — "you could have written it in 1950." Its job: a free, infinite supply of
  (noisy input $\mathbf x_t$, known answer $\boldsymbol\epsilon$) pairs. **Diffusion turns unsupervised generative
  modeling into supervised regression** — state this as the strategic core.
- *Variance preservation (inline, two lines).* $\mathrm{Var}(\mathbf x_t)=(1-\beta_t)\mathrm{Var}(\mathbf x_{t-1})+\beta_t$;
  if the input has unit variance, output is exactly 1. **This is why it's "variance-preserving (VP)": signal is
  rotated into noise at constant total energy** — a crossfader, not a volume knob.
- *★ The rotation / trig payoff (`.box key`, headline).* $(\sqrt{\bar\alpha_t})^2+(\sqrt{1-\bar\alpha_t})^2=1$ →
  the pair traces the **unit circle**. Define $\psi_t=\arccos\sqrt{\bar\alpha_t}$; then
  $\mathbf x_t=\cos\psi_t\,\mathbf x_0+\sin\psi_t\,\boldsymbol\epsilon$. **The forward process is a rotation; the
  noise schedule is just how fast you sweep the angle.** Translation note: papers write $\phi_t$. [This sets up
  $v$-prediction (55) and the cosine schedule for free.]
- *Eq. 3.3 — the most important line in the track (`.box key`, boxed & numbered):*
  $$\mathbf x_t=\sqrt{\bar\alpha_t}\,\mathbf x_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon.$$
  Derive by two steps + induction; the merge rule as its own `.box rule`:
  $\mathcal N(0,\sigma_a^2I)+\mathcal N(0,\sigma_b^2I)=\mathcal N(0,(\sigma_a^2+\sigma_b^2)I)$ — **variances add,
  not std** ($\sqrt9+\sqrt{16}=7$ but $\sqrt{9+16}=5$; "in quadrature", "Pythagoras"). Symbol Ledger required.
  Consequence: training is $O(1)$ not $O(T)$; $t$ is a free uniform input; one network for all noise levels.
- *The $\bar\alpha$ / SNR table (`.box worked`, cite the linear-schedule shape from brief §3.4, label the 3rd sig
  fig [EST], endpoints [DER]).* Rows $t\in\{0,100,250,500,750,1000\}$ with $\sqrt{\bar\alpha_t}$, $\sqrt{1-\bar\alpha_t}$,
  $\mathrm{SNR}=\bar\alpha_t/(1-\bar\alpha_t)$. **Endpoints frozen:** $\bar\alpha_{1000}\approx\mathbf{4\times10^{-5}}$,
  $\sqrt{\bar\alpha_{1000}}=\mathbf{0.0063}$ [DER, `constants.md` §9.6] — **nonzero terminal SNR.**
- *Read the table aloud — three observations (inline):* (1) $\sqrt{\bar\alpha_T}=0.0063\ne0$ → SD1.5/SDXL **cannot
  make a pure-black image** (leaks training-set mean luminance); "zero-terminal-SNR" (Lin 2024) is the real fix and
  `noise_offset` in his configs is a hack for it. (2) The linear schedule wastes ~25% of steps where nothing
  changes → the reason cosine exists. (3) The interesting band is $t\in[300,700]$ where SNR≈1 — where *composition*
  is decided; this is why a bad timestep distribution learns style but not structure.
- *The $\beta$-schedule vs `scheduler` confusion — `.box warn`, highest-priority.* The $\beta$ schedule is **baked
  into the checkpoint** — changing it makes the model wrong. ComfyUI's `scheduler` dropdown (karras/normal/
  exponential/sgm_uniform/beta/simple) picks the **sampling timestep grid** — an inference-time choice. "Two
  objects, confusingly similar names. He almost certainly has this conflated." Cosine schedule and the
  resolution-dependence bug (why 2048² gives two heads) go in a `.deepdive`.

**Demos:**
- **D54.1 — Schedule Explorer (`Plot`, log-y twin axis).** Curves vs $t\in[0,1000]$: $\bar\alpha_t$,
  $\sqrt{\bar\alpha_t}$, $\sqrt{1-\bar\alpha_t}$, and $\log_{10}\mathrm{SNR}_t$. Controls: schedule radio
  {linear, scaled-linear, cosine, flow-match+shift}; $\beta_{\min}\in[10^{-5},10^{-3}]$, $\beta_{\max}\in[0.005,0.05]$,
  $T\in\{100,250,500,1000\}$, $s\in[0,0.05]$, `shift`$\in[0.5,6]$. **JS implements exactly:** running cumulative
  product `alphabar[t]=alphabar[t-1]*(1-beta[t])` (NOT `Math.pow`), `snr=alphabar/(1-alphabar)`; FM curve uses
  $t'=\text{shift}\cdot t/(1+(\text{shift}-1)t)$. `readout`: $\bar\alpha_T$ in scientific notation + a red flag when
  $\bar\alpha_T>10^{-6}$: "nonzero terminal SNR — this model cannot make a pure black image." **Aha:** drag
  $\beta_{\max}$ 0.02→0.008 and watch $\bar\alpha_T$ leap $4\text{e-}5\to0.02$ — the endpoint stops being noise.
- **D54.2 — Noise ladder on a real image (`Heatmap`/canvas filmstrip).** 11 frames at $t\in\{0,100,…,1000\}$,
  computed live: Box–Muller $\boldsymbol\epsilon$, per pixel `x_t=sqrt(alphabar[t])*x0+sqrt(1-alphabar[t])*eps` on
  $[-1,1]$ data. Resolution toggle {64²,256²,1024²} same image, same schedule. **Aha #1:** at $t=500$ you still see
  the picture. **Aha #2:** at $t=800$ the 64² tile is dead static and the 1024² tile still shows the subject — the
  resolution bug, discovered by dragging.
- **D54.3 — "Jump vs Walk" verifier (proves Eq. 3.3).** Path A: 500-step loop with fresh draws. Path B: one line
  of Eq. 3.3. Two tiles + two overlaid pixel histograms + printed mean/std. **Aha:** different pictures, histograms
  coincide to ~3 decimals — "statistically identical."

**Code artifact (`.box try`):** `code/diffusion_2d.py` (manifest D3) — DDPM from scratch on 8-Gaussians, ~200 lines,
3-layer MLP, no U-Net: implements Eq. 3.3, $\mathcal L_{\text{simple}}$ (page 55), and the ancestral sampler by
hand. "The whole of the next two pages, in one readable file." *(~40 s CPU.)*

**Quiz (7; ≥1 numeric):**
1. Numeric: $\sqrt{\bar\alpha_{500}}$ signal fraction (num **0.280**, tol 0.03).
2. MC: what does ComfyUI's `scheduler` dropdown set? → the sampling timestep grid (distractor: "the $\beta$
   schedule" = the named confusion).
3. MC: how many parameters does the forward process learn? → zero.
4. Numeric: two independent noises, std 3 and 4, combined std (num **5**, tol 0.1) — the quadrature/Pythagoras rule
   (distractor: 7 = "std adds").
5. MC: why can't SD1.5 make a pure-black image? → nonzero terminal SNR, $\sqrt{\bar\alpha_T}=0.0063$.
6. MC: at $t=500$ the picture is → still visible (distractor: "half noise" = the linear-perception misconception).
7. MC: `noise_offset` in his training config is → a hack for the terminal-SNR bug.

**Threads:** [THREAD:chain-rule — Eq. 3.3 is what page 56 differentiates to get the score]. Dot-product atom absent.
**Cross-refs:** ← 53 (latent shapes). → 55 (reverse is the inversion of this), 56 (score from Eq. 3.3), 57 (flow
matching just picks different $a_t,b_t$).

---

# 55 — the-reverse-process-and-elbo.html  (D-21: 54)  ·  build: **O** · `.track-diffusion`

**Title:** "The reverse process and the ELBO collapse: where the monster dies"

**Objectives (5):**
1. Explain why the true reverse step is intractable and why 1000 *small* steps are each Gaussian (Feller).
2. Show that $\mathbf x_t,\mathbf x_0,\boldsymbol\epsilon,v$ are one rigid triangle — the network only picks a
   coordinate to report; read $v$ off the unit circle.
3. Watch the ELBO (a thousand KLs over $\mathbb R^{3{,}145{,}728}$) collapse to a single MSE via the equal-covariance
   KL lemma.
4. Write the six-line training loop and see the diffusion loss is MSE — a corollary of trunk MSE-from-MLE.
5. Explain the $\epsilon$-pred pole at high $t$ and why $v$-pred cures it (setup for flow matching).

**PREDICT box:** "The training objective for every image you've ever generated in ComfyUI — is it (a) an
adversarial game, (b) reinforcement learning, (c) `MSELoss` on a regression problem? Commit before reading."

**Section outline:**
- *Why the reverse is a dead end, then Feller — `.box key`.* $q(\mathbf x_{t-1}\mid\mathbf x_t)$ needs the unknown
  data marginal. **Feller (1949):** for small $\beta_t$, $q(\mathbf x_{t-1}\mid\mathbf x_t)$ is approximately
  Gaussian, so parameterize $p_\theta(\mathbf x_{t-1}\mid\mathbf x_t)=\mathcal N(\mu_\theta,\Sigma_\theta)$ and learn
  the mean. **The "why 1000 steps" answer — a full page-worth `.box key` + figure:** the reverse of a *big* step is
  as multimodal as $p_{\text{data}}$; the reverse of a *tiny* step is a small Gaussian blob. "1000 small steps are
  each individually Gaussian; one big step is not. Diffusion trades one impossible problem for a thousand easy
  ones." The tractable cousin $q(\mathbf x_{t-1}\mid\mathbf x_t,\mathbf x_0)$ and the completing-the-square algebra
  go in a `.deepdive` (state $\tilde\mu_t,\tilde\beta_t$; verify numerically in D55.1).
- *★ The rigid triangle — $\epsilon$ vs $x_0$ vs $v$ (`.box key`, the definitive fix).* Eq. 3.3 is one equation, two
  unknowns, one known: give any of $\{\mathbf x_0,\boldsymbol\epsilon,v\}$ and solve for the rest. Print the
  conversions. Then the trig picture (payoff #2 of page 54): with $\sqrt{\bar\alpha_t}=\cos\psi_t$,
  $\sqrt{1-\bar\alpha_t}=\sin\psi_t$,
  $$\mathbf x_t=\cos\psi_t\,\mathbf x_0+\sin\psi_t\,\boldsymbol\epsilon,\qquad v=\cos\psi_t\,\boldsymbol\epsilon-\sin\psi_t\,\mathbf x_0.$$
  $v$ is $\mathbf x_t$ rotated a further 90° — the *velocity* around the circle, $d\mathbf x_t/d\psi_t$. **The whole
  $\epsilon/x_0/v$ zoo is a choice of axis on a circle.** Symbol Ledger. Why the choice matters: the loss weighting
  differs by $\mathrm{SNR}_t$: $\|\epsilon-\hat\epsilon\|^2=\mathrm{SNR}_t\|\mathbf x_0-\hat{\mathbf x}_0\|^2$,
  spanning $10^4\!\to\!4\times10^{-5}$ — nine orders of magnitude.
- *The $\epsilon$-pred pole (`.box warn`).* $\hat{\mathbf x}_0=(\mathbf x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon)/\sqrt{\bar\alpha_t}$;
  at $t=999$, $\sqrt{\bar\alpha_{999}}\approx0.0063$ → **divide by 0.0063 = multiply error by $\mathbf{159\times}$**
  [DER, `constants.md` §9.6]. $v$-pred's conversions have coefficients $\cos\psi,\sin\psi\le1$ — **no pole**, stable
  both ends. "$v$-pred was invented to fix this and is the ancestor of flow matching's velocity — do not skip it, it
  prepays for page 57."
- *★ The ELBO collapse — the emotional center (`.box key`, engineer the page break).* Show the monster:
  $-\log p_\theta(\mathbf x_0)\le\mathbb E_q[L_T+\sum_{t\ge2}L_{t-1}+L_0]$, a thousand KLs over $\mathbb R^{3,145,728}$.
  Take it apart: $L_T$ has no $\theta$ (drop; and note $L_T\ne0$ is *the same fact* as the terminal-SNR bug); $L_0$
  is one term. $L_{t-1}$: both Gaussians, same fixed covariance → the **equal-covariance KL lemma (trunk-owned,
  point back, D-14):** $D_{\mathrm{KL}}(\mathcal N(\mu_1,\sigma^2I)\|\mathcal N(\mu_2,\sigma^2I))=\|\mu_1-\mu_2\|^2/2\sigma^2$.
  "A KL over $\mathbb R^{3M}$ becomes a squared distance divided by a number. This one lemma is where the monster
  dies." Substitute the $\epsilon$-parameterization; $\mathbf x_t$ terms cancel →
  $L_{t-1}=w_t\|\boldsymbol\epsilon-\epsilon_\theta\|^2$. Then Ho et al.'s honest hack: **set $w_t:=1$** — "not
  principled, makes the bound wrong, makes images much better." Box the result:
  $$\mathcal L_{\text{simple}}=\mathbb E_{\mathbf x_0,\boldsymbol\epsilon,t\sim\mathcal U\{1,T\}}\big[\|\boldsymbol\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}\mathbf x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon,\,t)\|_2^2\big].$$
  `.box warn` **genuine uncertainty:** dropping $w_t$ is not theory-justified; it re-weights toward mid-noise
  (composition, what people look at). $w_t$ is a **live 2026 research knob** (min-SNR-$\gamma$, logit-normal, EDM,
  P2). [THREAD:memory-ledger — $\phi'$-style cached activations are why training eats VRAM; and this MSE is *simpler*
  than the LLM track's cross-entropy.]
- *The six-line loop + the VAE payoff (`.box worked`).* Print the boxed algorithm (sample $\mathbf x_0$, uniform $t$,
  $\boldsymbol\epsilon$, one line of Eq. 3.3, MSE, SGD). "There is no loop over $t$." Then: **a diffusion model is a
  VAE with 1000 latent layers and a frozen encoder** — §53 cashes out. Sampling and *why $\sigma_t z$ is
  re-injected* (mean-only = mode-seeking mush) go inline; foreshadow DDIM($\eta{=}0$) works anyway (page 59).

**Demos:**
- **D55.1 — Rigid-triangle verifier (`assert`-style readout).** One $(\mathbf x_0,\boldsymbol\epsilon,t)$; compute
  $\mathbf x_t$; derive $\mathbf x_0,\boldsymbol\epsilon,v$ from each other; show round-trip error $<10^{-6}$.
  Slider over $t$; print $1/\sqrt{\bar\alpha_t}$ and flag **159×** at $t=999$. Exact math, no NN.
- **D55.2 — The unit-circle / $v$ figure (`Plot` + slider, the best asset in the track).** Draw the unit circle with
  $\mathbf x_0,\boldsymbol\epsilon$ as the orthonormal basis, $\mathbf x_t$ the rotated point, $v$ the 90°-further
  vector. Slider $\psi_t\in[0,\pi/2]$; readout $\sqrt{\bar\alpha_t}=\cos\psi_t$, $\sqrt{1-\bar\alpha_t}=\sin\psi_t$,
  $\mathrm{SNR}$. **Aha:** $\epsilon/x_0/v$ are three axes on one circle.
- **D55.3 — ELBO term-by-term collapse (interactive).** A column of the 1001 terms; click $L_T$ (no $\theta$ →
  greys out), click a middle $L_{t-1}$ (expands to the KL lemma → the MSE), click $w_t{=}1$ (the weight vanishes).
  Ends with $\mathcal L_{\text{simple}}$ alone. Convergence-point demo — build it as the page's spine.

**Code artifact (`.box try`):** `code/prediction_targets.py` (manifest D5) — take one $(\mathbf x_0,\boldsymbol\epsilon,t)$,
derive $\mathbf x_0,\boldsymbol\epsilon,v$ from each other, `torch.allclose` to machine precision; print the
error-amplification $1/\sqrt{\bar\alpha_t}$ at $t{=}999$ → **159×**. "The triangle, verified, not asserted." *(instant.)*

**Quiz (7; ≥1 numeric):**
1. MC (predict payoff): the training objective is → `MSELoss` on regression (distractors: adversarial game, RL —
   both named misconceptions).
2. Numeric: $\epsilon$-pred error-amplification factor at $t{=}999$ (num **159**, tol 5).
3. MC: why 1000 steps at *training* time? → small steps are Gaussian, big steps aren't (distractor: "more steps =
   more quality" — that's a *sampling* count, different reason).
4. MC: $\epsilon$-pred, $x_0$-pred, $v$-pred are → the same info, different coordinate/loss-weighting (distractor:
   "different models / capabilities").
5. MC: what makes the ELBO's KL tractable? → equal covariances → squared distance.
6. MC: is $\mathcal L_{\text{simple}}$ the ELBO? → no, $w_t$ was deleted (distractor: "yes, it's the bound").
7. MC: why re-inject noise during sampling? → it's a distribution not a point; mean-only = mode-seeking mush.

**Threads:** [THREAD:chain-rule], [THREAD:memory-ledger]. **Cross-refs:** ← 54; trunk KL-lemma page, trunk
MSE-from-MLE page; → 56 (score), 57 ($v$ = FM velocity), 59 (DDIM $\eta{=}0$ puzzle), 62 (this loss is what LoRA
minimizes).

---

# 56 — score-matching-and-the-sde.html  (D-21: 55)  ·  build: **S** · `.track-diffusion`

**Title:** "Score matching and the SDE view: you already trained a score model"

**Objectives (4):**
1. Define the score $\mathbf s_\theta=\nabla_{\mathbf x}\log p_t(\mathbf x)$ and distinguish it from $\nabla_\theta$.
2. Derive $\mathbf s_\theta=-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ in three lines — the "we already trained a score
   model" reveal.
3. Explain why noise is *necessary* (the manifold has no score off-sheet) — the schedule is a smoothing curriculum.
4. Read the forward/reverse SDEs and name which ComfyUI setting (Karras sigmas) is the VE-SDE.

**Ordering:** taught **after DDPM** (D-19) — the reveal only lands once he owns Eq. 3.3 and $\mathcal L_{\text{simple}}$.

**PREDICT box:** "The score points *toward* the data; $\boldsymbol\epsilon$ is what was *added* to leave the data.
Predict the sign relating them before you see the three-line derivation."

**Section outline:**
- *What the score is / isn't (`.box key`).* $\mathbf s(\mathbf x)=\nabla_{\mathbf x}\log p(\mathbf x)$, same shape as
  $\mathbf x$ — a gradient **w.r.t. the image**, not the parameters. [THREAD:chain-rule — same operator, different
  variable.] Analogy: $\log p$ is a potential landscape, the score is the force, sampling is rolling downhill with
  jitter = Langevin dynamics.
- *★ The three-line reveal (`.box key`).* Differentiate $\log q(\mathbf x_t\mid\mathbf x_0)$ (a Gaussian log-density)
  w.r.t. $\mathbf x_t$; use Eq. 3.3 to replace $\mathbf x_t-\sqrt{\bar\alpha_t}\mathbf x_0=\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$:
  $$\mathbf s_\theta(\mathbf x_t,t)=-\frac{\epsilon_\theta(\mathbf x_t,t)}{\sqrt{1-\bar\alpha_t}}.$$
  "The intractable $\nabla\log p$ of page 53 is the network we already have, divided by a lookup number." The
  negative sign is the predicted sign. Symbol Ledger. *Practical payoff:* this makes the CFG derivation (61)
  possible and the ODE/SDE view (below) possible; and it unifies NCSN (Song 2019) with DDPM (Ho 2020) — same
  algorithm, independent discovery — priming the page-57 identical lesson.
- *Why noise was necessary anyway (inline).* Off the data sheet $p=0$, $\log p=-\infty$, score undefined — adding
  noise **inflates the sheet into a solid** so every point has an arrow. And a *ladder* of noise levels is a
  smoothing curriculum: coarse arrows find the neighborhood, fine arrows find the house. "The $\beta$ schedule is a
  curriculum of smoothing scales; sampling is annealing down it." Tweedie's formula (one boxed line) recovers the
  $\hat{\mathbf x}_0$ conversion — and names $\hat{\mathbf x}_0$ as a *posterior mean* (why it's blurry at high $t$).
- *The SDE view (`.box key` + `.deepdive` for the algebra).* $T\to\infty$: staircase → ramp. Forward VP-SDE
  $d\mathbf x=-\tfrac12\beta(t)\mathbf x\,dt+\sqrt{\beta(t)}\,dw$; note $\mathrm{Var}[dw]\propto dt$ so noise scales
  as $\sqrt{dt}$ (random-walk $\sqrt N$ — he knows this). VE-SDE ($f=0$) = NCSN = **"Karras sigmas" in ComfyUI** —
  name the setting. The reverse-time SDE (Anderson 1982) with the score inside is the theorem that makes the field
  work; box it, keep the derivation in the `.deepdive`.

**Demos:**
- **D56.1 — Score field on a 2-D toy (`Plot` quiver, closed-form).** Mixture-of-Gaussians $p(\mathbf x)$ with
  closed-form score in JS (no NN). Draw $\mathbf s(\mathbf x)$ as arrows; a slider anneals the smoothing $\sigma$
  from large→small; overlay a few Langevin trajectories. **Aha:** at large $\sigma$ arrows point at the center of
  mass; at small $\sigma$ they're sharp and local — the annealing curriculum, visible.
- **D56.2 — Numerical score check (`assert` readout).** Finite-difference $\nabla_{\mathbf x}\log q(\mathbf x_t\mid\mathbf x_0)$
  and compare to $-\boldsymbol\epsilon/\sqrt{1-\bar\alpha_t}$; print max abs error → ~$10^{-6}$. The identity,
  verified.

**Code artifact (`.box try`):** `code/score_check.py` (manifest D6) — finite-difference the Gaussian log-density,
`assert torch.allclose` against $-\epsilon/\sqrt{1-\bar\alpha_t}$. "The score identity, verified, not asserted."
*(instant.)*

**Quiz (5; ≥1 numeric):**
1. MC: the score is a gradient with respect to → the image, not the parameters.
2. MC: sign relating score and $\boldsymbol\epsilon$ → opposite (score toward data, $\epsilon$ away).
3. Numeric: if $\sqrt{1-\bar\alpha_t}=0.5$ and $\epsilon_\theta=1.0$, the score magnitude (num **2.0**, tol 0.1).
4. MC: why is noise necessary for score matching? → off-manifold the score is undefined; noise inflates the sheet.
5. MC: "Karras sigmas" in ComfyUI correspond to → the VE-SDE / NCSN parameterization.

**Threads:** [THREAD:chain-rule — the score is $\nabla_{\mathbf x}\log p$, the same differentiation, and it *is*
$\epsilon_\theta$ rescaled]. **Cross-refs:** ← 55; → 57 (velocity/ODE), 61 (CFG is a score statement — this page is
its hard prerequisite).

---

# 57 — flow-matching-your-scheduler.html  (D-21: 56)  ·  build: **O** · `.track-diffusion`

**Title:** "Flow matching and rectified flow — your FLUX scheduler, decoded"

His own `FlowMatchEulerDiscreteScheduler` (from `~/…/FLUX.2-dev/scheduler/scheduler_config.json`, shift 3.0,
dynamic shifting, 1000 train steps [VP, `hardware-ground-truth.md` §4]) is **exhibit A** of this page.

**Objectives (4):**
1. Construct the rectified-flow path and velocity; see the sampler is one line, no $\bar\alpha$/$\tilde\beta$.
2. Reconcile DDPM and flow matching as one object via the $(a_t,b_t)$ table — "there is no third theory."
3. Explain why the learned FM field is *curved* even though training paths are straight (why base FM isn't one-step).
4. Open his own scheduler config and name every parameter: `shift`, `use_dynamic_shifting`, the vestigial
   `num_train_timesteps=1000`.

**PREDICT box:** "SD1.5 needs `dpmpp_2m` to look good at 20 steps; FLUX looks fine with plain `euler`. Same step
count, different sampler requirement. Guess why before the page tells you." (Answer: flow matching straightened the
path; Euler is exact on straight lines.)

**Section outline:**
- *The construction (`.box key`).* SD3/FLUX convention: **$t=0$ data, $t=1$ noise**. ⚠️ Translation/`.box warn`
  (notation §6): this is **not** "opposite to DDPM" — DDPM *also* has data at $t=0$; the genuine confusions are
  (a) Lipman's original FM puts noise at $t=0$, (b) diffusers' `sigmas` runs 1→0 during sampling. State the
  course's one convention and hold it.
  $$\mathbf x_t=(1-t)\mathbf x_0+t\,\boldsymbol\epsilon,\qquad u_t=\tfrac{d\mathbf x_t}{dt}=\boldsymbol\epsilon-\mathbf x_0.$$
  Constant velocity along the path. Train $\mathcal L_{\text{FM}}=\mathbb E\|v_\theta-(\boldsymbol\epsilon-\mathbf x_0)\|^2$.
  Sample by integrating backward: $\mathbf x_{t-\Delta}=\mathbf x_t-\Delta\,v_\theta$. **One line. No $\bar\alpha$,
  no posterior mean, no $\sigma_t z$** — point at page 55's sampler and contrast.
- *Why $v_\theta$ still takes $t$ / isn't one-step (`.box warn`, the most-missed point).* $v_\theta$ sees only
  $\mathbf x_t$; many $(\mathbf x_0,\boldsymbol\epsilon)$ give the same $\mathbf x_t$, so it learns the **conditional
  average** $v_\theta^\star=\mathbb E[\boldsymbol\epsilon-\mathbf x_0\mid\mathbf x_t]$. **Averaging straight lines
  gives a curved field** (lines cross). "Rectified flow is straight" is half true — training paths are straight, the
  learned field is not. That's why **Reflow** exists (re-couple pairs, retrain, straighten → 1–4 step viability, the
  ancestry of every turbo/schnell/klein). `.deepdive`.
- *★ The reconciliation — the $(a_t,b_t)$ table (`.box key`, the section the whole track builds to).*
  $\mathbf x_t=a_t\mathbf x_0+b_t\boldsymbol\epsilon$; everything is a choice of two scalar functions (notation §4.4,
  frozen):

  | Framework | $a_t$ | $b_t$ | Constraint | Endpoint |
  |---|---|---|---|---|
  | DDPM / VP-SDE | $\sqrt{\bar\alpha_t}$ | $\sqrt{1-\bar\alpha_t}$ | $a_t^2+b_t^2=1$ (circle) | $a_T\approx0.006$ (leaky) |
  | Rectified flow | $1-t$ | $t$ | $a_t+b_t=1$ (chord) | $a_1=0$ exactly |
  | VE / Karras | $1$ | $\sigma_t$ | none | $\sigma_{\max}\approx80$ |
  | Cosine | $\cos\psi_t$ | $\sin\psi_t$ | circle, uniform sweep | $a_1=0$ exactly |

  **DDPM is the arc, rectified flow is the chord.** Of course the chord integrates in one step and hits $a=0$
  exactly (no terminal-SNR bug, *structurally*). And the boxed bridge: on the circle $a=\cos\psi,b=\sin\psi$ the FM
  velocity **is $v$-prediction from page 55** — "$v$-pred is flow matching's velocity on the circular path instead
  of the straight one." Conversions on the RF path: $\hat{\mathbf x}_0=\mathbf x_t-t\hat v$,
  $\hat{\boldsymbol\epsilon}=\mathbf x_t+(1-t)\hat v$ — **no square roots, no small division**; have him verify by
  substitution. Print the summary box: **"DDPM and flow matching differ in (1) the route $(a_t,b_t)$, (2) which
  coordinate the network reports, (3) the loss weighting across $t$. There is no third theory."**
- *What his FLUX actually does — open the config (`.box worked` + `.box try`).* Logit-normal timestep sampling:
  $t=\text{sigmoid}(z)$, $z\sim\mathcal N(0,1)$ — concentrates near 0.5. **Worked number:** $P(0.27<t<0.73)=P(-1<z<1)=68\%$
  — two-thirds of every batch in the SNR≈1 band (vs uniform's 68% over $[0.16,0.84]$). The `shift` fix
  $t'=\text{shift}\cdot t/(1+(\text{shift}-1)t)$ for resolution. **Verified defaults:** `shift=1.0`,
  `use_dynamic_shifting=False`, `base_shift=0.5`, `num_train_timesteps=1000` **vestigial** (`.box warn`: FM is
  continuous in $[0,1]$; the 1000 is inherited DDPM bookkeeping, not a chain length). **His FLUX.2-dev scheduler:
  `shift=3.0`, dynamic shifting on** [VP, his disk] — have him `cat` it and read `use_dynamic_shifting: true`. "This
  is page 54's resolution theory shipped as production code in your own folder." *Sampler payoff:* Euler works for
  FLUX because the path is straighter (the predict-box answer); the full ODE-solver menu is deferred to page 59.
- *Honest uncertainty (`.box warn`).* Is FM *better* than DDPM or better-*packaged*? Contested: the "same thing"
  camp (FM is a reparameterization; win is defaults + cleaner code) vs the "genuinely helps" camp (zero terminal
  SNR free, resolution-scalable via one `shift`, first-order-solvable, distills readily; the whole 2026 frontier
  chose it). Not in dispute: FM's code is dramatically simpler and results at least as good. "The field largely
  stopped asking because the practical answer is clear."

**Demos:**
- **D57.1 — The $(a_t,b_t)$ route map (`Plot`, must-build; it *is* the reconciliation).** The $(a,b)$ plane, unit
  circle faint. Three routes $(1,0)\to(0,1)$: DDPM-linear (lumpy arc from real $\bar\alpha_t$), cosine (uniform arc),
  rectified flow (chord). A dot per route animated in $t$; side panel shows the same real image noised per route at
  the current $t$. **Aha:** at $t=0.5$ the three routes give visibly different noise levels from the *same* $t$ —
  "$t$ means different things in different frameworks" — yet share endpoints.
- **D57.2 — Straight vs curved trajectories (`Plot`, the "why few-step" demo).** 2-D 8-Gaussians; two panels
  (DDPM prob-flow ODE vs rectified-flow ODE) with **closed-form** scores/velocities in JS. Step-count slider
  $N\in\{1,2,4,8,16,32,64\}$; overlay the $N{=}1000$ reference; readout W2 distance to truth. **Aha:** drag $N$ to 4
  — DDPM particles fly off the curve into the void between modes (exactly a 4-step SD1.5 image); rectified-flow
  particles land almost correctly. "Why the whole field switched, in one slider." (Optional D57.3 timestep-sampler
  histogram if space allows.)

**Code artifact (`.box try`):** `code/flow_matching_2d.py` (manifest D7) — the **same** 2-D file as `diffusion_2d.py`
with three lines changed ($a_t,b_t$; the target; the sampler). **`diff diffusion_2d.py flow_matching_2d.py` IS the
reconciliation, in code.** Then sweep $N$ 1→64 for both, plot W2. "No other course does this." *(~40 s CPU.)*

**Quiz (6; ≥1 numeric):**
1. Numeric: logit-normal, % of $t$ in $(0.27,0.73)$ (num **68**, tol 3).
2. MC: why does plain `euler` suffice for FLUX where SD1.5 needed `dpmpp_2m`? → FM straightened the path.
3. MC: "rectified flow paths are straight so one step works" → false; the *learned field* is curved (crossing
   lines) — named misconception.
4. MC: `num_train_timesteps=1000` in his FLUX config means → vestigial indexing, not a 1000-step chain.
5. MC: DDPM vs FM differ in → route/coordinate/weighting, no third theory (distractor: "rival theories").
6. MC: $v$-prediction is → FM's velocity on the circular path (distractor: "a DDPM-only, unrelated thing").

**Threads:** [THREAD:chain-rule — velocity is a derivative along the path]. [THREAD:Qwen3-8B absent]. **Cross-refs:**
← 55 ($v$-pred), 56 (ODE view), 54 (shift = resolution theory); → 59 (samplers, distillation), 62 (`discrete_flow_shift`,
`model_prediction_type=raw`, `timestep_sampling` are these knobs).

---

# 58 — the-unet-in-depth.html  (D-21: 57)  ·  build: **S** · `.track-diffusion`

**Title:** "The denoiser's body I: the U-Net, in depth"  — **U-Net first and in depth (ruled, decisions §Z-6)**

**Objectives (4):**
1. State the network's job as a shape constraint: (noisy latent, noise level, text) → a tensor the *same shape* as
   the latent — dense prediction, not classification.
2. Explain the U-Net's down/up structure and why **skip connections carry spatial information a bottleneck
   destroyed** — not the ResNet gradient story.
3. Show where the timestep enters (FiLM/sinusoidal embedding, trunk-owned) and why gain, not signal.
4. Distinguish self- vs cross-attention by their shapes; read the $[1024\times77]$ cross-attention matrix as "where
   the prompt becomes a picture."

**PREDICT box:** "A U-Net without skip connections: does it train *slower*, or produce *permanently blurry* output?
Commit before the ablation demo."

**Section outline:**
- *The shape constraint (`.box key` + shape ribbon).*
  $\epsilon_\theta:\mathbb R^{C\times H_{lat}\times W_{lat}}\times\mathbb R\times\mathbb R^{S_{\text{txt}}\times d_c}\to\mathbb R^{C\times H_{lat}\times W_{lat}}$.
  Output shape = input shape. Not a classifier, not an LLM — dense prediction. Two families satisfy it (both on his
  disk): U-Net (SD1.5, SDXL) and DiT (SD3.5, FLUX, Z-Image, Qwen-Image).
- *The U-Net structure (`.box worked`, the SD1.5 level table, mark [EST]/verify-against-checkpoint).* Levels
  64²→8²→64², channels 320→1280→320, ResBlocks + Transformer2D blocks, skip *concat* on the up path. Note the
  up-path convs take **double** input channels — the concat, visible in the config.
- *★ Skips carry information, not gradients (`.box key`, correct the usual wrong explanation).* At 8×8 each
  activation covers a 64×64-pixel patch — it *knows* "face, upper-left" but **cannot know** where eyelashes go; that
  was averaged away three downsamples ago and is **physically not in the tensor**. The skip hands high-res
  activations *around* the bottleneck. "The skip is not an optimization aid; it is a wire carrying information that
  would otherwise not exist." One-liner: **"Down-path decides *what*. Up-path decides *where*. Skips are how *where*
  survives the trip."** `.box warn`: the ResNet "skips help gradients" story is a *different* mechanism (and ResNet
  *adds*; U-Net *concats*). Without skips: not slower — **structurally blurry, forever** (the predict payoff).
- *Where $t$ enters (inline, point to trunk).* FiLM modulation $h\leftarrow\text{GroupNorm}(h)\cdot(1+\gamma(t))+\beta(t)$,
  with $t$ first mapped by a **sinusoidal embedding — the exact same function as the LLM track's positional
  encoding, applied to a scalar noise level** (trunk-owned; D-14; point back, don't re-derive). Why gain not signal:
  $t$ is a *mode switch* ("which job — coarse composition vs fine texture"), and a mode switch belongs in the gain.
- *Self vs cross-attention (`.box key`, shape ribbon table — trunk owns attention mask-agnostically; this is the
  cash-out).* Flatten $[B,C,H,W]\to[B,HW,C]$. **Self-attn:** $Q,K,V$ all image, $[1024\times1024]$, cost $O((HW)^2)$
  — "let distant pixels talk." **Cross-attn:** $Q$ image, $K,V$ **text**, $[1024\times77]$, cost $O(HW\cdot S_{\text{txt}})$
  — "inject the prompt." Print the shapes table (SD1.5 level 2, 32×32). **Point at $[1024,77]$:** row $i$, col $j$ =
  "how much does patch $i$ care about prompt token $j$?" — "the 77 columns are his prompt; this matrix is the entire
  mechanism of text-to-image; everything else is plumbing." Reshape column $j$ → a $32\times32$ heatmap of where
  "dog" attends (this is the mechanism behind the attention-map nodes in his ComfyUI).

**Demos:**
- **D58.1 — U-Net skip ablation (precomputed + slider).** Side-by-side outputs of two tiny MNIST diffusion U-Nets,
  identical except skips severed; slider "skips cut (0/1/2/3)"; a receptive-field diagram. **Aha:** cut 3 skips →
  digits are right-shaped blobs with no edges. "The *where* was lost — structurally impossible output, not slow
  training."
- **D58.2 — Cross-attention heatmap (`viz.js` `Heatmap`, must-build, the payoff).** A real 512² generation; for
  "a red cube on a blue sphere," click a word → its $[1024]$ attention column reshaped to $32\times32$, upsampled,
  heat-overlaid. Sliders over **layer** and **timestep**. Data: precomputed maps from a real diffusers run with
  hooks (the JS does reshape+normalize+colormap — honest, exact). **Aha #1:** "red" lights up on the cube. **Aha #2:**
  drag the timestep — high $t$ maps are diffuse blobs (deciding *roughly where*), low $t$ tight and object-shaped
  (**composition early, detail late** — page 54's SNR≈1 band, page 56's annealing, page 57's logit-normal, all
  visible in one drag). Convergence-point demo; give it room.

**Code artifact (`.box try`):** `code/ddpm_mnist.py` (manifest D4) — a ~1.5M-param **U-Net** on 28×28 MNIST
(deliberately a U-Net: at 1.5M params the hierarchy is an asset, not a cage — page 59's counterpoint), with a
**skip-severing flag** for D58.1. Real images from noise on his box in ~10 min.

**Quiz (6):**
1. MC (predict payoff): U-Net without skips → permanently blurry output (distractor: "trains slower" = ResNet
   confusion).
2. MC: U-Net skips vs ResNet residuals → carry spatial info around a bottleneck (concat), not gradient flow (add).
3. MC: cross-attention $K,V$ come from → the text tokens (distractor: "the image, like self-attn").
4. Numeric: cross-attention matrix inner dim for SD1.5 CLIP prompt (num **77**, tol 0).
5. MC: why does $t$ enter as FiLM gain, not as a token? → it's a global mode switch, belongs in the gain.
6. MC: the network's output shape equals → the input latent shape (dense prediction), not $K$ class logits.

**Threads:** [THREAD:dot-product — attention scores are dot products; the cross-attention matrix is his prompt].
[THREAD:Qwen3-8B — sinusoidal embeddings are shared with the LLM track]. **Cross-refs:** trunk attention page
(mask-agnostic, cross-attention first-class — hard prerequisite, D-14), trunk sinusoidal-embedding page; ← 55; → 59
(DiT), 61 (cross-attention IS the conditioning), 62 (LoRA targets `to_q/k/v/out` — you must know what they are).

---

# 59 — from-unet-to-dit-mmdit.html  (D-21: 58)  ·  build: **O** · `.track-diffusion`

**Title:** "The denoiser's body II: DiT, MMDiT, and the samplers"  — DiT/MMDiT **after** the U-Net (ruled)

**Objectives (5):**
1. Describe the DiT: patchify latents → tokens → transformer → unpatchify; the 4096 tokens now doing work.
2. Explain why **MMDiT's bidirectional text↔image attention** is why FLUX writes text and binds attributes and
   SD1.5 couldn't — a capability change, not a refactor.
3. Read his own FLUX.2 DiT config off disk (48 heads×128, 8 dual + 48 single, `joint_attention_dim` 15360=3×5120).
4. Decode the sampler dropdown as textbook ODE solvers; explain why ancestral samplers don't converge and why
   `dpmpp_2m` is the free lunch.
5. State the 2026 landscape (transformers + flow matching + LLM encoders + distilled-beats-big) as a
   clearly-marked, ages-fast appendix.

**PREDICT box:** "`euler_a` at 20 steps vs 100 steps — same image sharper, or a *different* image? And `dpmpp_2m`
gets 2nd-order accuracy at how many model calls per step?" (Answers: different image; 1 call.)

**Section outline:**
- *The DiT (`.box key` + shape ribbon).* Patchify $[B,16,128,128]\xrightarrow{2\times2}[B,4096,64]\xrightarrow{\text{linear}}[B,4096,3072]$;
  $N$ transformer blocks; unpatchify back. 4096 tokens = the number from page 53. Peebles & Xie 2023: quality tracks
  Gflops monotonically (the scaling-law behavior); the U-Net's inductive biases were a *constraint* at scale.
- *How conditioning enters — three designs, the field converged (`.box worked` table).* **adaLN-Zero** ($t$+pooled
  text → per-block $\gamma,\beta,\alpha$; $\alpha$ init 0 → each block starts as identity — the same B=0 no-op trick
  as LoRA); **cross-attention** (PixArt); **MMDiT (joint attention)** — concatenate text+image into one sequence,
  self-attend over the union, **separate weight matrices per modality, one shared softmax** (SD3/3.5, FLUX.1/2 — the
  2026 standard).
- *★ Why MMDiT won (`.box key`).* Cross-attention is one-directional (image queries text; text is shouted once,
  frozen). MMDiT lets **text tokens attend to image tokens too**, so the prompt's representation *adapts to what's
  being drawn* — "a red cube on a blue sphere" resolves which is which by looking at the canvas. **This is why
  SD3/FLUX render text and bind attributes and SD1.5 constantly swapped them — he has lived this exact jump.**
- *His FLUX config, off disk (`.box worked`, `.box try`, [VP] `hardware-ground-truth.md` §4).* FLUX.1-dev: 12B,
  $d=3072$, MMDiT, guidance-distilled, 19 double-stream + 38 single-stream blocks [VP `constants.md` §9.6 / brief].
  **FLUX.2-dev (his disk): `Flux2Transformer2DModel`, in_ch 128 (=32×2² packing ✓), 48 heads×128, 8 dual + 48
  single, mlp_ratio 3.0, `rope_theta` 2000, `joint_attention_dim` 15360 = 3×5120** [VP]. Have him `cat
  transformer/config.json`. Shape ribbon for MMDiT joint attention (concatenate $4096$ image + $S_{\text{txt}}$ text
  tokens → self-attend). SD3.5-Large shape walk (4096+154=4250 tokens, $[B,H,4250,4250]$ attn) in a `.deepdive`.
- *Samplers — the ODE-solver menu (`.box key` table; his daily dropdown).* `ddpm/euler_a/euler/ddim/heun/dpmpp_2m/…`
  as order + model-calls/step. **Three takeaways not in any tutorial:** (1) **`dpmpp_2m` is the free lunch** — 2nd
  order by reusing the previous step's output (Adams–Bashforth), *half* Heun's model calls. (2) **Ancestral ($\_a$)
  samplers don't converge** — `euler_a`@20 ≠ `euler_a`@100 improved; it's a *different image* (fresh noise each
  step); `euler`/`ddim`($\eta{=}0$)/`dpmpp_2m` do converge. This resolves the DDIM-$\eta{=}0$ puzzle from page 55.
  (3) **Step count is step size**, error $\sim O(h^p)$; FLUX needs fewer Euler steps *because flow matching
  straightened the path* (page 57). **`denoise` decoded (`.box key`):** `denoise=0.6` = skip the first 40% of the
  schedule, noise the input to $t=0.6T$, run the last 12 of 20 steps. "Where on the ladder you jump in" — not a
  mystery dial.
- *Distillation & few-step (inline + table).* Progressive/Reflow/Consistency-LCM/ADD/DMD/**guidance distillation**
  (bakes CFG into weights — FLUX.1-dev/schnell; sets up page 61's `guidance`≠`cfg`). **The tradeoff (`.box warn`):**
  distilled models are **less steerable** — `guidance_scale=1.0` in the klein example is a *constraint*; CFG largely
  doesn't work on a guidance-distilled model; narrower diversity; harder to LoRA-train. "The step count isn't free;
  you paid in controllability."
- *2026 landscape (`.deepdive`, marked "expected to age"; cite `constants.md` §9.6 verbatim).* Backbone/params
  table: SD1.5 U-Net 860M → SDXL U-Net 2.6B → SD3.5 MMDiT 8B → FLUX.1-dev 12B → **FLUX.2-dev 32B + Mistral-3 24B**
  [VP] → **FLUX.2-klein-4B Apache-2.0, ~13 GB, 4 steps @ guidance_scale 1.0** [VP] → **Z-Image-Turbo 6B, 8 NFE,
  Apache-2.0, #1 open-weights above FLUX.2-dev (32B)** [VP]. Four trends: everything's a transformer; everything's
  flow matching; **text encoders became LLMs** (→ page 61); **small+distilled beats big** — *the opposite of the LLM
  track's still-scaling story* (D-17 divergence, flag deliberately). "Teach on SD1.5, work on FLUX.1, flex on
  FLUX.2." `.box warn` **counterpoint:** at <1B/256² a U-Net still wins — "the right architecture depends on scale"
  beats "transformers won" (why the D4 code artifact is a U-Net).

**Demos:**
- **D59.1 — Sampler shootout (`Plot`, reuse D57.2's closed-form field, zero marginal cost).** Sampler radio
  {euler, euler_a, heun, dpmpp_2m, ddim(η slider)}; step-count $N\in[1,64]$; a **"model calls" counter** (the honest
  x-axis). Plot W2-to-truth vs **model calls**. **Aha #1:** Heun and dpmpp_2m reach the same accuracy but Heun
  spends 2× the calls — the free lunch, measured. **Aha #2:** crank $N$ on `euler_a` → W2 plateaus at a nonzero
  floor; on `euler` → W2→0. "Ancestral samplers don't converge," in five seconds.
- **D59.2 — MMDiT vs cross-attention token budget (`makeCtrl` + `readout`).** Sliders: resolution, encoder {77, 512,
  long}. Outputs: token count $N$, cross-attn $[N\times S_{\text{txt}}]$ vs MMDiT joint $[(N+S_{\text{txt}})^2]$
  entries, memory in MB. **Aha:** CLIP(77)→T5(512) multiplies cross-attention 6.6×; joint attention costs
  $(4096+512)^2$ vs $4096\times512$ ≈ 4.4× more. "Better prompt understanding isn't free — here's the invoice."

**Code artifact (`.box try`):** `code/sample_flux.py` (manifest D8, adapted to his disk) — real generation with
`diffusers` on **his `flux1-dev-kontext_fp8`** exposing seed/steps/`guidance`/scheduler; **prints latent shape
$[1,16,128,128]$, token count 4096, peak memory** — verifying pages 53 & 60 on his own machine. *(~3 s/image.)*

**Quiz (7; ≥1 numeric):**
1. Numeric: model calls per step for `dpmpp_2m` (num **1**, tol 0) vs the free-lunch claim.
2. MC: `euler_a` at 100 vs 20 steps → a different image, not a sharper one (named misconception: "more steps always
   better").
3. MC: why can FLUX render text and bind attributes when SD1.5 couldn't? → MMDiT bidirectional text↔image attention.
4. MC: `denoise=0.6` means → start from a 60%-noised input, skip the first 40% of steps (distractor: "removes 60% of
   noise").
5. MC: a guidance-distilled model at `guidance_scale=1.0` → a constraint (CFG doesn't work on it), not a default.
6. Numeric: FLUX.2-dev `joint_attention_dim` (num **15360**, tol 0) = 3×5120.
7. MC: at <1B params / 256², the better backbone is → the U-Net (its hierarchy is a useful prior).

**Threads:** [THREAD:dot-product — joint attention]. [THREAD:Qwen3-8B — the text encoder became an LLM; FLUX.2's is a
40-layer GQA Mistral-3, foreshadow the capstone]. **Cross-refs:** ← 57 (why euler suffices), 58 (attention shapes),
55 (DDIM η=0 puzzle resolved here); → 61 (guidance distillation → `guidance`≠`cfg`), 62 (distilled models are harder
to LoRA).

---

# 60 — latent-diffusion-and-the-roofline.html  (D-21: 59)  ·  build: **O** · `.track-diffusion`

**Title:** "Latent diffusion and the roofline: why it runs on *your* desk"

**The deleted showpiece stays deleted (D-16).** The honest FLUX.1-Schnell/4-step MFU calculation replaces it; the
learner benchmarks his own box.

**Objectives (5):**
1. *Compute* (not assert) the 65,536× attention saving of latent vs pixel space.
2. Read the real inference memory table and the "the model is essentially all model" observation.
3. Apply the frozen roofline formula $I=S_{\text{fwd}}$ to place diffusion firmly compute-bound (the Spark is
   strong) — the deliberate LLM/diffusion contrast.
4. Reproduce the honest FLUX.1-Schnell/4 FLOP count and MFU; understand why the old "we predicted 2.6 s" showpiece
   was invalid.
5. Predict his own FLUX.2-dev capacity from the 16-B/param ledger against his *measured* 121.69 GiB.

**PREDICT box:** "Pixel-space self-attention at 1024², one head, one layer, fp16: how many bytes is the attention
matrix? Order of magnitude only. (You have a 128 GB box.)" (Answer: **2.2 TB** — off by 17× before you multiply by
heads×layers×steps.)

**Section outline:**
- *★ Do the pixel-space arithmetic (`.box worked`, cite `constants.md` §9.6 verbatim).* Pixel tokens
  $1024^2=1{,}048{,}576$; attention matrix $N^2=\mathbf{1.10\times10^{12}}$ entries = **2.2 TB** per head/layer/step
  fp16 [DER]. Latent (FLUX, $f=8$, 2×2 patchify): $64\times64=\mathbf{4096}$ tokens; $4096^2=\mathbf{1.68\times10^7}$
  = **33.5 MB** [DER]. Box the ratio: $\mathbf{65{,}536\times=16^4}$ [DER] — "a 65,536× reduction in the expensive
  operation for a 12× lossy compression. That trade is why Stable Diffusion exists and why your Spark runs it."
  Historical check inline: Imagen/DALL·E 2 diffused in pixel space and needed 3-model cascades — "the cascade was
  the workaround for not having a VAE."
- *The inference memory table (`.box worked`, mark activation rows [EST]).* FLUX.1-dev ≈ 34–38 GB; **FLUX.2-dev ≈
  112–120 GB**; SDXL ≈ 8 GB; Z-Image ≈ 14 GB. **The latent row is 0.5–1.0 MB** — "the thing you're generating is
  half a megabyte; the weights are 24 GB, 48,000× larger. Diffusion inference is streaming 24 GB of weights past a
  half-megabyte of state, 28 times." Mitigations he should know cold: precompute text embeddings (−48 GB on
  FLUX.2), `enable_model_cpu_offload()` (**near-free on the Spark — unified coherent memory, no PCIe copy**), FP8
  (−50%), NVFP4 (native FP4 tensor cores — what the "1000 TOPS" is for).
- *★ The roofline — the frozen formula and the deliberate contrast (`.box key`, D-15).* Trunk owns the derivation;
  point back and plug in. $I=\dfrac{2N\cdot S_{\text{fwd}}}{2N}=S_{\text{fwd}}$ — **arithmetic intensity = tokens per
  forward.** Ridge $I^\star=P_{\text{peak}}/BW$. **Print with the [INF] label and both live values
  (`constants.md` §6.4):** $62\text{e}12/273\text{e}9=\mathbf{227}$ FLOP/byte (working) **or 458 if the BF16 ceiling
  is ~125 TF** — 62-vs-125 TF is genuinely **unresolved [INF]** (the SDXL BF16 datapoint leans 125). **Do not print
  either ridge as fact.** The verdicts hold under both: diffusion at $S_{\text{fwd}}=4096$ tokens sits **18×/9×
  above** the ridge → **compute-bound → the Spark is strong**; LLM decode ($S_{\text{fwd}}=1$) sits 227×/458×
  *below* → bandwidth-bound → the Spark is weak. "Same machine, same ridge, opposite verdicts, because the tracks
  sit on opposite sides of $I^\star$." `.box warn`: correct brief-diffusion §8.4 — bf16 FLUX is **compute-bound**,
  not bandwidth-bound; compute needs $2\times12\text{e}9\times4096/62\text{e}12=\mathbf{1585}$ ms/step ≫ the 88 ms
  bandwidth floor nobody is near.
- *The honest Schnell/4 replacement (`.box worked`, cite `constants.md` §9.6 verbatim — the showpiece stays
  deleted).* Label the NVIDIA number **"FLUX.1-Schnell, 4 steps, 1024², batch 1, FP4 = 23 img/min = 2.6 s/img"**
  [VP] — **NOT FLUX.1-dev/28.** Show why the dev/28 reading is impossible: $28\times2\times12\text{e}9\times4096/2.6=\mathbf{1.06}$
  PFLOP/s = 106% of the sparse-FP4 *marketing* peak on a workload using neither FP4 nor sparsity. Honest Schnell/4
  FLOP: per step @1024² (4096 img + 512 txt tokens) linear $2NS=1.097\text{e}14$ + attention $4S^2d\cdot57=1.487\text{e}13$
  (11.9%) = $1.245\text{e}14$/step → 4 steps $=\mathbf{4.98\times10^{14}}$ FLOP → $/2.609$ s $=\mathbf{191}$ TF/s → **30–38%
  MFU** vs the ~500 TF dense-FP4 ceiling — plausible [DER]. "You cannot honestly predict a 2.6 s wall-clock when the
  FP4 dense ceiling is itself an unpublished ±2× inference. So you'll measure your own."
- *His capacity, from the ledger (`.box worked`, [MEA-DEV]).* [THREAD:memory-ledger] Full FT = 16 B/param:
  FLUX.1-dev $12\text{e}9\times16=\mathbf{192}$ GB + ~4 act ≈ **196 GB** [DER] ❌; FLUX.2-dev ≈ **520 GB** ❌. But
  **FLUX.2-dev LoRA** with precomputed embeddings (bf16 base) ≈ **70 GB** ✅; **FP8 base** ≈ 38 GB ✅ — "the 32B model
  is trainable *on your desk* and on no consumer GPU (a 5090 has 32 GB; the base weights alone are 64 GB)." Against
  his **measured** budget: `MemTotal` = **121.6875 GiB** [MEA-DEV], carveout **6.3125 GiB (4.9%)**; with ComfyUI up,
  `mem_get_info` free was **19.41 GiB**. The `.box try` has him measure, not trust.
- *"Capacity, not speed" (`.box key`).* 273 GB/s vs a 5090's ~1.8 TB/s — "~1/7th the bandwidth, 4× the memory,
  unified." For jobs that *fit in 32 GB* (SDXL/FLUX.1 LoRA) a 5090 is **faster**; the Spark wins only on jobs the
  5090 can't start. Say it plainly — he owns the box.

**Demos:**
- **D60.1 — Pixel-vs-latent attention calculator (`makeCtrl` + `Plot` log-y).** Sliders: resolution {512…2048}, VAE
  factor $f$ {1(pixel),8}, patchify {1,2}. JS computes tokens, $N^2$ entries, bytes fp16, and the ratio. `readout`:
  the pixel-space bytes flagged red when > 128 GB. **Aha:** at 1024² pixel space is 2.2 TB; latent is 33.5 MB;
  drag resolution and watch the *quadratic* bite (1024²→2048² is 16× the attention).
- **D60.2 — The roofline plot (`Plot` log-log, the contrast demo).** $x$ = arithmetic intensity $I$ (log), $y$ =
  achievable FLOP/s (log). Draw the roof: bandwidth-bound slope $BW=273$ GB/s and the compute ceiling as a
  **band 62–125 TF [INF]** (shaded, labeled "unresolved"). Plot points: LLM decode $I{=}1$, decode batch-32 $I{=}32$,
  prefill $I{=}2048$, **diffusion $I{=}4096$**. Vertical ridge band 227–458. `readout`: which side of the ridge each
  point is on + verdict. **Aha:** diffusion is deep in compute-bound territory under *both* ceilings — the Spark's
  strong side. Toggle to overlay his measured GEMM number once `bench_spark.py` has run.
- **D60.3 — Schnell/4 MFU bar (`Plot` bar + `readout`).** Bars: the impossible dev/28 reading (1.06 PFLOP/s, red,
  "106% of marketing peak") vs the honest Schnell/4 (191 TF/s, green, "30–38% MFU"). No sliders — a static honesty
  exhibit paired with the deleted-showpiece note.

**Code artifact (`.box try`):** `code/bench_spark.py` (manifest D16, targets his disk) — measures **his** machine:
achieved memory bandwidth, ms/step forward and forward+backward at bf16/fp8/fp4, s/image on **FLUX.2-dev / FLUX.2-dev-Turbo**,
against the predictions on this page. Also runs the big bf16 GEMM against **both** the 62 TF and 125 TF ceilings in
both accumulate modes to **settle the ridge himself** (the course's best lab, `constants.md` §10 exercise #2). "The
prediction you check is your own." *(~5 min. ⚠️ ComfyUI is live on his box — the script frees VRAM first / warns.)*

**Quiz (7; ≥1 numeric):**
1. Numeric: pixel-vs-latent attention ratio at 1024² (num **65536**, tol 0) — accept 16^4.
2. MC: latent diffusion is primarily a → **compute** optimization (65,536× on attention), not a memory one (named
   misconception).
3. Numeric: FLUX.1-dev full-FT state at 16 B/param (num **192**, tol 5, unit GB).
4. MC: at bf16, FLUX diffusion is → compute-bound (distractor: "bandwidth-bound at 88 ms/step" = the corrected §8.4
   claim).
5. MC: the NVIDIA "2.6 s / 1K image FP4" figure is → FLUX.1-**Schnell**, 4 steps (distractor: "FLUX.1-dev, 28 steps"
   = the deleted showpiece).
6. Numeric: his measured usable `MemTotal` (num **121.69**, tol 0.1, unit GiB) — not 128.
7. MC: for an SDXL LoRA that fits in 32 GB, a 5090 vs the Spark → the 5090 is faster (capacity, not speed).

**Threads:** [THREAD:memory-ledger — 16 B/param, his 121.69 GiB]. [THREAD:dot-product — attention is the expensive
$N^2$ op]. **Cross-refs:** trunk roofline/ridge page (formula owned there); ← 53 (latent shapes), 59 (his FLUX
config); → 62 (LoRA/QLoRA memory), the LLM-track hardware page (the *other* side of the same ridge — D-15 contrast).

---

# 61 — conditioning-cross-attention-cfg.html  (D-21: 60)  ·  build: **O** · `.track-diffusion`

**Title:** "Conditioning: text encoders, cross-attention, and CFG — your most-turned knob, derived"

**The highest-value page for this learner** — CFG is the knob he's turned most and understands least. Derive it, do
not assert it.

**Objectives (5):**
1. Show conditioning changes the loss by one symbol ($\mathbf c$) — everything from pages 54–57 is untouched.
2. Say what the two ComfyUI text-encoder boxes contain and why the encoder is *frozen*; name the 2026 trend (the
   encoder became an LLM — the tracks converged).
3. **Derive** CFG in five lines from Bayes + the score identity, and read it geometrically as extrapolation.
4. Explain the three real mechanisms of "fried" output and why the one-line std fix (RescaleCFG) repairs it.
5. Separate FLUX's `guidance` (a baked-in embedded number, 1 pass) from real `cfg` (2 passes, extrapolation).

**PREDICT box:** "At `cfg 7.5`, are you sampling from (a) the conditional distribution $p(\mathbf x\mid\mathbf c)$,
(b) a sharper version of it, or (c) a distribution *no real image belongs to*? And how many forward passes per step
does `cfg 7.5` cost?" (Answers: c; 2.)

**Section outline:**
- *Conditioning is one symbol (`.box key`).* $\mathcal L=\mathbb E\|\boldsymbol\epsilon-\epsilon_\theta(\mathbf x_t,t,\mathbf c)\|^2$
  — add $\mathbf c$, train on (image, caption) pairs; forward process, objective, samplers all unchanged. "That's
  why diffusion took over: you can condition on text, depth, pose, another image — anything — by concatenating it."
  The *how* is cross-attention / MMDiT joint attention (page 58/59).
- *The two encoder boxes (`.box worked` table).* SD1.5 CLIP-L (77 tokens, frozen); SDXL CLIP-L+bigG; SD3.5/FLUX.1
  add **T5-XXL 4.7B**; **FLUX.2 = Mistral-3 24B VLM** [VP]. `.box warn` "**frozen**": CLIP/T5 were trained by other
  people for other tasks; the diffusion model learns to *read* a fixed language. Why 77? — CLIP's 2021 context
  length, a historical accident that spawned every prompt-chunking hack he's used. Why T5 alongside CLIP? — CLIP is
  bag-of-words-ish (can't bind "red cube/blue sphere"); T5 has syntax → "CLIP for vibe (pooled→adaLN), T5 for
  content (sequence→attention)". **[THREAD:Qwen3-8B]** The 2026 headline (`.box key`): **the text encoder became an
  LLM** — CLIP-123M → T5-4.7B → **Mistral-3-24B, twice the size of the entire FLUX.1 model, just to read the
  prompt.** His FLUX.2 encoder on disk is a **40-layer GQA (32 heads/8 KV) RoPE Mistral-3** [VP,
  `hardware-ground-truth.md` §4] — "every component the LLM track taught you." Foreshadow the capstone's rejoining.
- *★ Derive CFG (`.box key`, five lines the learner follows).* Bayes: $p(\mathbf x\mid\mathbf c)\propto p(\mathbf c\mid\mathbf x)p(\mathbf x)$;
  take $\log$, $\nabla_{\mathbf x}$ ($p(\mathbf c)$ dies — the $Z$ trick again). Rearrange:
  $\nabla\log p(\mathbf c\mid\mathbf x)=\nabla\log p(\mathbf x\mid\mathbf c)-\nabla\log p(\mathbf x)$ = (conditional
  score) − (unconditional score) — "we have both, and we accidentally have a classifier gradient without training a
  classifier: that's the *classifier-free* in the name." Amplify by $w_g$: $\tilde p_{w_g}\propto p(\mathbf x)p(\mathbf c\mid\mathbf x)^{w_g}$,
  convert scores→$\epsilon$ via the page-56 identity (the $-$ and $\sqrt{}$ cancel), box the result:
  $$\tilde\epsilon_\theta=\epsilon_\theta(\mathbf x_t,t,\varnothing)+w_g\big[\epsilon_\theta(\mathbf x_t,t,\mathbf c)-\epsilon_\theta(\mathbf x_t,t,\varnothing)\big].$$
  "That is `cfg` in the KSampler node, derived from Bayes and one substitution. Page 56 was worth it." Symbol
  Ledger. [THREAD:dot-product/chain-rule — the CFG direction $\epsilon_c-\epsilon_\varnothing$ is the atom; CFG is a
  score statement.]
- *Read it geometrically (`.box key`).* $\tilde\epsilon=\epsilon_\varnothing+w_g(\epsilon_c-\epsilon_\varnothing)$ =
  `lerp(uncond, cond, w_g)` — "he has written this line." Table: $w_g{=}0$ prompt ignored; $w_g{=}1$ the model's
  honest $p(\mathbf x\mid\mathbf c)$ ("CFG off"); **$w_g>1$ extrapolation, off the end of the line**; $w_g{=}7.5$ is
  6.5 lengths past $\epsilon_c$. **"CFG generates images from a distribution that does not exist"** — and it works,
  "one of the most embarrassing facts in generative modeling." Use the frozen canonical logits $z=[2.0,1.0,0.1]$
  ($\hat y=[0.659,0.242,0.099]$, `constants.md` §9.2) for the softmax/temperature side-note if a probabilities
  illustration is wanted (do not introduce a rival triple).
- *Why too high looks "fried" — three real mechanisms (`.box worked`).* (1) **Variance/contrast blow-up** (dominant,
  arithmetic): $\|\tilde\epsilon\|$ grows ~linearly in $w_g$, exceeds the std-1 that $\epsilon$ should be →
  $\hat{\mathbf x}_0$ over-subtracts → blown highlights, crushed blacks. **The one-line fix confirms the diagnosis:**
  RescaleCFG renormalizes $\mathrm{std}(\tilde\epsilon)$ back — "a std fix repairs the artifact, so the artifact was
  a std problem. He may have used the `RescaleCFG` node without knowing it was this." (2) **Off-manifold
  extrapolation** — straight line in $\epsilon$-space leaves the curved data manifold. (3) **Diversity collapse** —
  $p(\mathbf c\mid\mathbf x)^{w_g}$ is mode-seeking. Real ranges (`.box worked`): SD1.5 7–8, SDXL 5–7, SD3.5 3.5–4.5,
  FLUX.1-dev `cfg` 1.0, klein 1.0. "CFG requirements fell monotonically as models improved — CFG is a crutch for a
  weak conditional model, being engineered out of existence via distillation."
- *★ `guidance` ≠ `cfg` (`.box warn`, highest priority — "he almost certainly has this wrong").* FLUX.1-dev's
  `FluxGuidance` node `guidance` (default 3.5) is **not CFG**: it's the CFG scale the guidance-distilled model was
  trained to emulate, **fed in as a conditioning input, embedded like $t$** — one forward pass, no unconditional
  branch, the extrapolation compiled into the weights. Consequences he can act on: setting `cfg=3.5` on FLUX.1-dev
  **doubles time and makes it worse** (double-guidance); `guidance` is cheap (an embedding); FLUX.1-schnell ignores
  it; **FLUX LoRAs must be trained with `guidance_scale=1.0`** (→ page 62). *Cost:* CFG = 2 passes/step → 20 steps @
  cfg 7.5 = 40 evaluations; "the steps number in the UI is half the truth." Negative prompt = replace $\varnothing$
  with $\mathbf c^-$ (repurposed branch, costs nothing; does nothing on FLUX.1-dev — no second pass).
- *Genuine uncertainty (`.box warn`).* Nobody has a satisfying theory of why $w_g\approx7$ beats $w_g{=}1$;
  Karras's "autoguidance" (guiding with a *worse* model works better) suggests CFG corrects model *error*, not the
  conditional. "You're turning a knob the field's best theorists don't fully understand, in a direction that
  provably leaves the data distribution, and it works. That's the honest state of the art."

**Demos:**
- **D61.1 — CFG on a 2-D toy (`Plot` quiver + samples, the flagship, closed-form).** $p(\mathbf x)$ = 5-blob
  mixture; $p(\mathbf x\mid\mathbf c)$ = concentrated on 2. Single **$w_g$ slider 0→20**. Left: the $\tilde\epsilon$
  field ($\epsilon_\varnothing,\epsilon_c$ faint, $\tilde\epsilon$ bold). Right: 500 samples from Euler-integrating
  the field, over the true $p(\mathbf x\mid\mathbf c)$ contours. Readouts: $\|\tilde\epsilon\|/\|\epsilon_c\|$, sample
  std, **fraction of samples outside the true support**. **Aha (drag order):** $w_g{=}0$ all 5 blobs; $w_g{=}1$
  matches the true contours (correct, unimpressive); $w_g{=}4$ tightens; $w_g{=}10$ collapses to two dots, norm 3×;
  $w_g{=}20$ shoots into empty space, "outside support" → 90%. "That's what fried means, with two Gaussians and a
  slider." All three mechanisms visible at once.
- **D61.2 — The CFG line (`Plot`, 10-second permanent fix).** Two fixed points $\epsilon_\varnothing,\epsilon_c$;
  a dot at $\tilde\epsilon$; $w_g$ slider. **Aha:** past $w_g{=}1$ the dot **visibly leaves the segment.** Pair with a
  real image strip at $w_g\in\{1,3,7,12,20\}$ so geometry and frying share the screen.
- **D61.3 — RescaleCFG (`Plot` + $\phi$ slider).** D61.1 plus a rescale-mix $\phi$ slider and the
  $\mathrm{std}(\tilde\epsilon)$ readout. **Aha:** at $w_g{=}12$ drag $\phi$ 0→1, std snaps to 1.0, samples stay in
  the right blobs. "The frying was a scale bug; this is the one-line fix."

**Code artifact (`.box try`):** `code/cfg_by_hand.py` (manifest D9, his disk) — run **his `flux1-dev-kontext_fp8`**
twice manually (cond + uncond), apply $\tilde\epsilon=\epsilon_\varnothing+w_g(\epsilon_c-\epsilon_\varnothing)$ by
hand, `assert torch.allclose` against `pipe(guidance_scale=w_g)`; print $\|\tilde\epsilon\|/\|\epsilon_c\|$ vs $w_g$
(grows linearly → the frying mechanism, measured); then implement CFG-rescale and show the norm snap back. *(~1 min.)*

**Quiz (7; ≥1 numeric):**
1. MC (predict payoff): at `cfg 7.5` you sample from → a distribution no real image belongs to (distractor: "the
   conditional distribution").
2. Numeric: forward passes per step at `cfg 7.5` (num **2**, tol 0).
3. MC: FLUX's `guidance` vs `cfg` → `guidance` is an embedded number, 1 pass; `cfg` is real extrapolation, 2 passes
   (named highest-priority confusion).
4. MC: the dominant cause of "fried" output → variance/contrast blow-up (fixed by RescaleCFG).
5. MC: why is the text encoder *frozen*? → it was trained by others for another task; the model learns to read it.
6. MC: why did CFG requirements fall from 7.5 to 1.0 across model generations? → CFG is a crutch for a weak
   conditional model.
7. Numeric: 20 steps at `cfg 7.5` costs how many model evaluations (num **40**, tol 0).

**Threads:** [THREAD:chain-rule — CFG is a score/gradient statement, page 56 is its prerequisite]. [THREAD:dot-product
— the CFG direction]. [THREAD:Qwen3-8B — the encoder became a Mistral-3, the tracks converge]. **Cross-refs:** ← 56
(score identity, hard prerequisite), 58/59 (cross-attention/MMDiT), 59 (guidance distillation); trunk Bayes page,
`constants.md` §9.2 canonical logits; → 62 (`guidance_scale=1.0` training gotcha; strength=CFG frying analogy), the
capstone (the Mistral reveal).

---

# 62 — fine-tuning-diffusion-on-your-data.html  (D-21: 61)  ·  build: **O** · `.track-diffusion`

**Title:** "Fine-tuning diffusion: putting *your* thing in the model" — **the destination ("I made a diffusion
model mine")**

Carries the fine-tuning taxonomy, LoRA/DreamBooth/TI/ControlNet-IP-Adapter, the dataset craft, and the eval
protocol. Dense by design — it is the milestone page; the dataset/eval depth rides in `.deepdive`s and the code
artifacts. Hands off to the capstone (pages 63–65).

**Objectives (5):**
1. Place every adaptation method on the three-axis taxonomy (WHERE / WHAT / WHICH-recipe) and apply the "is the base
   model different?" test (ControlNet/IP-Adapter are **not** fine-tuning).
2. Derive diffusion LoRA memory and see the win is optimizer-state, not weights; connect `network_alpha`,
   `strength`, and the CFG-frying analogy.
3. State the rank/LR/example-count rules as consequences of *one* principle (information content), reconciled with
   the LLM track.
4. Apply the invariance principle and caption-what-you-vary rule — *derived from the loss*, not folklore.
5. Evaluate a LoRA by a protocol (fixed grid + Mars probe + drift control + strength sweep + eyes), because the loss
   curve is noise.

**PREDICT box:** "You train a LoRA on 20 photos in 20 minutes; you download a ControlNet someone trained on 100k+
images. One changed the model, one didn't. Which is which — and why are there 500,000 LoRAs on Civitai but ~15
useful ControlNets?"

**Section outline:**
- *★ The taxonomy (`.box key` table — trunk owns D-18; instantiate for diffusion).* WHERE the change goes: all
  weights (full FT, 24 GB) · low-rank correction (**LoRA**, 10–200 MB) · the vocabulary (**textual inversion**, 3 KB)
  · **nowhere — you change the input** (ControlNet, IP-Adapter). Orthogonal WHAT: form vs facts (facts → you can't;
  that's RAG). WHICH recipe: SFT/**DreamBooth**/DPO. **The test (`.box key`):** *after you're done, is the base
  model different?* LoRA/DreamBooth/full-FT → **yes**; ControlNet/IP-Adapter → **no** (bit-identical; you bolted on
  an input). The Civitai ratio *is* the distinction (the predict payoff). "DreamBooth vs LoRA is a category error —
  DreamBooth is *what you train* (recipe), LoRA is *how you store it* (parameterization); modern practice is both:
  point at `train_dreambooth_lora_flux.py`."
- *Why not full FT (`.box worked`, [THREAD:memory-ledger]).* FLUX.1-dev full FT ≈ **196 GB** [DER] ❌ on his 121.69
  GiB; FLUX.2-dev ≈ 520 GB. And it's the wrong tool: catastrophic forgetting (20 images vs 1B), 24 GB/concept, can't
  stack. The 16-B/param rule again.
- *★ LoRA, derived (`.box key` — trunk owns the math $W_0+\frac{\alpha}{r}BA$, $B{=}0$ no-op, $\alpha/r$ decouples
  rank from LR, the 187× identity; point back, then instantiate).* Worked param arithmetic on **his** models: one
  FLUX.1-dev attention projection $3072\times3072=9{,}437{,}184$; LoRA $r{=}16$: $16(3072+3072)=98{,}304$ = **1.04%,
  96× fewer** [DER]. Whole model attn-only ≈ **30M = 0.25% of 12B → 60 MB bf16** [EST — verify vs
  `Flux2Transformer2DModel`]. Memory (`.box worked`): FLUX.1-dev LoRA r=16 ≈ **29–31 GB** [DER]+[EST act] vs full FT
  196 GB — **optimizer state 96 GB → 0.24 GB, a factor of ~400**, because AdamW state scales with *trainable*
  params. `.box warn` (frequently botched): **all 24 GB of frozen base weights are still resident — the win is not
  the weights.** **`network_alpha`/`strength` (`.box key`, his daily knobs):** effective scale $\alpha/r$;
  `network_alpha=r/2` **halves** your effect (→ 2× the LR) — "the most common 'why isn't my LoRA learning?' cause."
  ComfyUI `strength_model` is a further multiplier: $W_0+\text{strength}\cdot\frac{\alpha}{r}BA$; **`strength>1.0` is
  extrapolation past what you trained — the same move as CFG$>1$, same frying** (page 61). [THREAD:dot-product — $BA$
  is an outer-product sum, $r$ directions.]
- *Where to attach + variants (`.deepdive`).* `to_q,to_k,to_v,to_out` = the standard (diffusion default is
  attention, D-06/D-11 fork: this is about *what* you teach — attention is where cross-modal binding lives — not a
  parameter budget); cross-attn-only = "teach a new word"; +conv (LoCon) = "teach a new *look*" (styles). QLoRA on a
  4-bit base is the **FLUX.2-dev** path (NF4 = **0.516 B/param = 4.127 bits, DQ on** [VP]; `bnb_4bit_use_double_quant=True`,
  else 0.5625 [VP]). DoRA/LoHa/LoKr/LoRA+ named in a table; "plain LoRA r=16 is 95% of shipped LoRAs."
- *★ Rank/LR/examples = one principle (`.box key`, reconcile with LLM track, D-09/D-11).* **Rank follows the
  *information content of the dataset*, not the model size.** Diffusion concept LoRA: 15–20 images, one point on a
  manifold the model already knows → **r=8–32**; high $r$ on 20 images **memorizes** them. (LLM SFT sits at the
  other end: r=64–256, high $r$ nearly free.) **LR follows trainable-param count** (trunk principle): diffusion full
  FT/DreamBooth **1e-6–5e-6**, diffusion **LoRA 1e-4** (⚠️ **not** 1e-6 — a 100× error that makes the LoRA silently
  fail to learn, D-09), textual inversion **5e-4–5e-3** [EST, `constants.md` §9.4]. "Three apparent hyperparameter
  mysteries, one question: how much are you asking the model to learn?"
- *DreamBooth & textual inversion (`.box worked`).* DreamBooth: rare token (`sks`), **prior-preservation loss**
  $\mathcal L_{\text{DB}}=\mathbb E\|\boldsymbol\epsilon-\epsilon_\theta(\cdot,\mathbf c_{\text{sks}})\|^2+\lambda_{pp}\mathbb E\|\boldsymbol\epsilon'-\epsilon_\theta(\cdot,\mathbf c_{\text{class}})\|^2$,
  $\lambda_{pp}=1.0$, **class images generated by the base model itself** (a snapshot of its own prior, held as a
  mirror — not real photos). 30-second drift diagnostic: generate "a photo of a dog" with **no trigger**; if it's
  *your* dog → drift → add reg images (else skip — live community disagreement). **Textual inversion (`.box key`, the
  intellectual climax):** freeze everything, optimize **one embedding vector** $v^\star\in\mathbb R^{d_c}$; SD1.5
  $d_c=768$ → **768 floats × 4 B = 3 KB**. "The model already knew how to draw your thing — TI finds an *address*
  and labels it; the 3 KB is a pointer, not a payload." Rule: **TI retrieves; LoRA teaches.**
- *ControlNet / IP-Adapter — NOT fine-tuning (`.box key`).* ControlNet: clone the encoder, feed it a hint
  (edge/depth/pose), wire back via **zero-convs** (the third appearance of the "init the new thing to zero" pattern —
  LoRA $B{=}0$, adaLN-Zero $\alpha{=}0$, zero-conv — name it once, point thrice); ~360M ≈ 1.4 GB, "not a LoRA-scale
  object." IP-Adapter: **decoupled cross-attention**, $Z=\text{Attn}(Q,K_{\text{text}},V_{\text{text}})+\lambda\,\text{Attn}(Q,K_{\text{img}},V_{\text{img}})$,
  the text branch untouched, image branch added (new $W_K',W_V'$ from a CLIP image encoder) — "a mixing desk, not a
  surgery; $\lambda$ is the ComfyUI `weight` slider." **The punchline (`.box key`):** he can stack a LoRA ($\theta$)
  + ControlNet ($\mathbf c$) + IP-Adapter ($\mathbf c$) + prompt ($\mathbf c$) — "they don't collide because they're
  not the same kind of object. His messy node graph has a clean type signature."
- *Dataset craft (`.deepdive`, derived from the loss, not folklore — mark folklore numbers [EST]).* **Invariance
  principle:** the model learns whatever is **constant** across the dataset → make the concept constant, vary
  everything else; 15–20 images (label [EST]); "more is often worse" (100 one-shoot photos = "face + that lighting"
  bundle). **Caption-what-you-vary (`.box key`):** $\boxed{\text{caption what you want to VARY, omit what you BAKE
  IN}}$ — *derived*: any feature in $\mathbf c$ is attributed to the words; anything absent has nowhere to go but the
  trigger/weights. Subject → describe context, omit the subject; style → describe content, omit the style (triggerless
  style LoRAs). Bucketing = a batching constraint (constant pixel count → constant token count → constant step time);
  dims divisible by 64 = $f{=}8\times$patch-2$\times$4. "Read your captions — 10 minutes beats a week of hp search."
- *Evaluation (`.box warn` + `.box key`).* **The loss curve is nearly useless** — dominated by the random $t$ draw
  (page 55); flat with huge variance whether learning or memorizing. **The protocol:** fixed prompt grid at fixed
  seeds every 250 steps → contact sheet; the **Mars probe** ("sks man as an astronaut on Mars, full body" — OOD
  generalization test); the **drift control** ("a photo of a man", no trigger); ArcFace IdentitySim for faces
  (>0.6 good, >0.4 recognizable [EST]); a **strength sweep** (healthy LoRA usable at 1.0); "your eyes are the
  metric — turn the vibe into a protocol." Six overfitting signs (background bleed → captioning; pose lock →
  variety; adherence collapse/contrast burn/memorization → over-trained) in a `.deepdive`. FID/CLIP-score critique
  (Gaussian lie, small-$N$ bias, gameable by cranking CFG) briefly in a `.deepdive` — "the judge and the defendant
  share a brain."
- *Tooling handoff (inline).* Train in **ai-toolkit / kohya** → output is a `.safetensors` LoRA → drop in
  `~/ComfyUI/models/loras/` → `LoraLoader`. The annotated `config.toml` where every line links to a page: last four
  rows — `timestep_sampling=flux_shift` (57), `model_prediction_type=raw` (55/57), `guidance_scale=1.0` (61, the
  gotcha), `discrete_flow_shift=3.0` (57). "That config file is the entire course."

**Demos:**
- **D62.1 — The rank slider (`viz.js` `Heatmap` + `Plot` log-y; trunk owns the SVD demo — reuse it here on an
  image).** Real 256×256 grayscale; rank-$r$ reconstruction $\sum_{i\le r}\sigma_i u_i v_i^\top$ (SVD precomputed to
  JSON); residual at 4× gain; log-scale scree plot with a vline at $r$. Readouts: stored $r(256{+}256{+}1)$ vs
  65,536, compression, relative error. **Aha:** $r{=}8$ recognizable (6.3%), $r{=}32$ hard to distinguish (25%, ~3%
  error), the $\sigma_i$ fall off a cliff — "real matrices are approximately low-rank; that's the LoRA hypothesis,
  verified on a photograph." (If the trunk ships the three-matrix dropdown — random / $W_0$ / $\Delta W$ — reference
  it, don't rebuild.)
- **D62.2 — Where the memory goes (`Plot` stacked bar, the calculator he'll actually use).** Segments {frozen
  weights, trainable, gradients, optimizer, activations} vs a red line at his **121.69 GiB** [MEA-DEV] (not 128 —
  use the measured value). Radios: model {SD1.5, SDXL, FLUX.1-dev, **FLUX.2-dev**}; method {full FT, LoRA r=16, r=64,
  QLoRA-4bit}; optimizer {AdamW fp32, 8-bit, Adafactor}; toggles {grad ckpt, precompute embeddings, precompute
  latents}. JS = the exact formulas (frozen $= P\cdot b_W$; trainable/grads $=N_t\cdot2$; opt $=N_t\cdot k_O$).
  **Aha:** FLUX.1-dev full FT leaps past the line (196 GB); switch to LoRA — the optimizer segment vanishes while
  the frozen-weights segment doesn't move; toggle "precompute embeddings" on FLUX.2-dev → −48 GB, bar drops under
  the line.
- **D62.3 — The four places (interactive diagram).** Pipeline `prompt→encoder→c` and `x_T→[θ]→x_0→VAE→image`. Radio
  {LoRA, TI, ControlNet, IP-Adapter, full FT} highlights **exactly what changes** in red + prints the byte count.
  **Aha:** LoRA/full-FT light up $\theta$; TI lights one token in $\mathbf c$; ControlNet/IP-Adapter light $\mathbf c$
  and **$\theta$ stays grey** — "two change the model, three change the message."

**Code artifacts (`.box try`, multiple):**
- `code/lora_from_scratch.py` (manifest D10) — `LoRALinear` in ~30 lines (no `peft`) wrapping a real **FLUX.2**
  attention module; prints full `9,437,184` vs LoRA `98,304` = **1.04%**; asserts bit-identical output at step 0
  ($B{=}0$); then swaps in `peft` for the same numbers.
- `code/train_lora_flux.py` (manifest D11, **the main artifact**) — DreamBooth-LoRA on **20 of his own images**
  against **FLUX.1-dev** (or FP8 FLUX.2-dev): precomputes text embeddings + latents to disk (prints GB saved), bf16,
  grad checkpointing, AdamW8bit, `guidance_scale=1.0`, logit-normal timestep sampling; **prints the step-0 memory
  breakdown matching D62.2**; saves a ComfyUI-compatible `.safetensors` into his `models/loras/`. *(~20–40 min.)*
- `code/caption_ab.py` (manifest D17, the experiment nobody ships) — trains **two SD1.5 LoRAs on the identical 20
  images**, Strategy A vs B captions, runs the Mars probe on both. "You *measure* the captioning rule instead of
  believing it." *(~30 min.)*
- (name in a list, don't detail) `textual_inversion.py` (D13, prints `Trainable parameters: 768`, saves the 3 KB
  file), `eval_lora.py` (D14, the protocol → contact sheet), `comfy_export.py` (D18, closes the loop into
  `models/loras/`).

**Quiz (8; ≥1 numeric):**
1. MC (predict payoff): ControlNet is → not fine-tuning; the base model is bit-identical (named category error).
2. Numeric: diffusion LoRA learning rate order of magnitude (num **1e-4**, tol 3e-5) — distractor 1e-6 = the D-09
   100× error.
3. MC: LoRA's memory win comes from → optimizer state (96 GB→0.24 GB), not the weights (named misconception; base
   weights still resident).
4. Numeric: textual-inversion trainable params for SD1.5 (num **768**, tol 0).
5. MC: `network_alpha = r/2` → halves your LoRA's effect (need ~2× LR).
6. MC: "caption what you want to..." → VARY (omit what you bake in) — derived from the loss (distractor: "describe
   accurately/completely").
7. MC: the diffusion LoRA training loss curve is → nearly useless (dominated by the random $t$ draw) — use a fixed
   prompt grid.
8. MC: `strength_model = 1.4` in ComfyUI → extrapolation past what you trained, same failure as CFG$>1$.

**Threads:** [THREAD:memory-ledger — 16 B/param, his 121.69 GiB, the ~400× optimizer-state win]. [THREAD:dot-product
— $BA$, $r$ directions]. [THREAD:chain-rule — you're minimizing the page-55 loss]. [THREAD:Qwen3-8B — LoRA is the
same object the LLM track fine-tuned Qwen3-8B with; the 187× identity is shared]. **Cross-refs:** trunk SVD/rank
page + LoRA-math page (hard prerequisites), trunk three-things-taxonomy page (D-18), the LLM-track LoRA page (rank
reconciliation, D-11); ← 58/59 (`to_q/k/v/out`), 60 (the memory ledger), 61 (`guidance_scale=1.0`, strength=CFG
analogy); → **the capstone (pages 63–65)**: the rejoining beat (his FLUX.2 encoder is the LLM track's subject) and
the "choose the method for a new problem" synthesis.

---

## SEAMS — what other spec writers must match

**⚠️ Two frozen-file gaps I could not fully honor (flag to the architect):**
1. **The literal D-21 page-by-page outline (titles + O/S build-model column) is NOT in `decisions.md`.** D-21 (lines
   784–796) ratifies the *64-page-no-merges budget* and the GRPO capstone, but contains no numbered page list. I
   reconstructed the 10-page diffusion arc from `brief-diffusion.md`'s sections and the ratified orderings (D-19
   DDPM-before-score/SDE; §Z-6 U-Net-before-DiT; trig-payoff on the forward page; his FlowMatchEuler scheduler as
   the flow-matching exhibit; D-16 deleted showpiece → honest Schnell/4). **O/S assignments are my inference by
   demo/math weight** (O = 53,54,55,57,59,60,61,62; S = 56,58) and must be reconciled if D-21's table surfaces.
2. **Trunk/LLM page *numbers* for cross-references are unknown** (same missing outline). I referenced trunk pages
   descriptively; whoever holds the D-21 map should backfill numbers on: the reparam-trick page, MLE/likelihood
   page, KL(𝒩‖𝒩)-lemma page, MSE-from-MLE page, attention page (mask-agnostic + cross-attention first-class),
   sinusoidal-embedding page, SVD/rank page + rank-slider demo, LoRA-math page (187× identity), the three-things
   taxonomy page (D-18), the roofline/ridge page ($I=S_{\text{fwd}}$, 227/458), number-formats page (NF4/FP8/NVFP4),
   and the LR-vs-trainable-params principle page.

**Upstream dependency (trunk + LLM track, pages 1–52 — these MUST be delivered before page 53, per D-14):** attention
taught **mask-agnostically with cross-attention first-class** (pages 58/59/61 break otherwise — the single biggest
cross-brief dependency); the **KL-between-equal-covariance-Gaussians lemma** and **Jensen** (page 55's collapse is a
wall without them); **MSE-from-MLE** (the diffusion loss is a corollary); **reparameterization trick** taught early;
**sinusoidal embeddings** (page 58 reuses them); **SVD/rank + the rank slider** and **LoRA's math** ($W_0+\frac\alpha
r BA$, $B{=}0$, $\alpha/r$, the **187×** identity) (pages 62); the **three-fine-tuning-things taxonomy** (D-18); the
**roofline formula $I=S_{\text{fwd}}$ and ridge 227/458 [INF]** (page 60 plugs in); **number formats** (page 62's
QLoRA); the **LR-vs-trainable-params principle** (page 62). The GRPO capstone (page 52) precedes page 53.

**Downstream handoff (capstone, pages 63–65 = D-21 62–64 — NOT in this spec):** page 62 ends pointing forward and
plants the seed (foreshadowed on pages 59 and 61) that **FLUX.2's text encoder is a 40-layer GQA RoPE Mistral-3 —
every component the LLM track taught** [VP, `hardware-ground-truth.md` §4]. The capstone owns D-17 beat 3 (the
rejoining: "in 2026 FLUX.2's encoder is a Mistral — the branches grew back together"), the economic-divergence
observation (LLMs still scaling, image models consolidating — Z-Image 6B > FLUX.2 32B), and the final "choose the
method for a new problem" synthesis. **The capstone writer must not re-teach cross-attention or LoRA — page 62 owns
the diffusion instantiation; the capstone reinterprets, not re-introduces** (anti-pattern #20, the expertise-reversal
tax).

**Intra-track invariant to hold:** the $(a_t,b_t)$ table (page 57) and the unit-circle/$\psi_t$ figure (page 55) are
**canonical, drawn once, reused** (notation §10); the roofline point-set (page 60) reuses the trunk's ridge. Do not
redraw them per page.
