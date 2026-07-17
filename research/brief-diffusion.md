# Research Brief: Diffusion Models and Fine-Tuning Them

> **⚠️ THIS BRIEF PREDATES TWO VERIFICATION PASSES. `constants.md`, `decisions.md`, `notation.md` OVERRIDE IT.**
> Load-bearing corrections: **(1)** §8.4's "we predicted a published benchmark" showpiece is **INVALID — DELETE**
> (dev/28 arithmetic vs a Schnell/4-step measurement; and bf16 is **compute-bound**, not bandwidth-bound). Replace
> with `bench_spark.py` (D-16/D-15). **(2)** FLUX.2 VAE is now **VERIFIED [VP]**: $f=8$, 32ch, 4096 tokens (§2.5).
> **(3)** "53,657 tok/s" → **6,969.59** (D-10m). **(4)** diffusion LR: full 1e-6–5e-6 / LoRA 1e-4 (D-09).
> **(5)** anchor for any LLM-side arithmetic is Qwen3-8B. See `decisions.md` §D-20.

**Audience for this document:** the curriculum architect writing the build spec. Not the learner.
**Track:** Destination Track B (equal weight to the LLM track), branching off the shared trunk.
**Date of research:** 2026-07-16. All time-sensitive claims were web-verified; confidence flags are inline.

---

## 0. Framing this track for THIS learner

He already drives ComfyUI. He has turned the CFG slider, swapped `dpmpp_2m` for `euler_a`, loaded LoRAs,
watched a generation "fry" at CFG 14. **He has the phenomenology and not one line of the theory.**

That is an enormous pedagogical asset and the track must exploit it. The rhetorical move for nearly every
section is:

> "You already know what this knob *does*. Here is what it *is*."

Concretely, the track should repeatedly cash out theory into a ComfyUI node he has clicked:

| ComfyUI thing he already touches | Theory the course owes him |
|---|---|
| `KSampler` → `steps` | ODE discretization error; truncation error vs. step count |
| `KSampler` → `cfg` | Classifier-free guidance; score extrapolation; why >8 fries |
| `KSampler` → `sampler_name` | DDIM / DPM-Solver++ / Euler as ODE integrators of different order |
| `KSampler` → `scheduler` (karras/normal/beta/simple) | The $\sigma$ discretization grid, not the noise schedule |
| `KSampler` → `denoise` (img2img) | Starting the reverse chain at $t<T$ from a real $x_0$ |
| `VAEEncode` / `VAEDecode` | Latent diffusion's compression stage; why 4 vs 16 vs 32 channels matters |
| `Empty Latent Image` 128×128 for a 1024px image | The factor $f=8$ made literal |
| `CLIPTextEncode` two boxes (pos/neg) | The two forward passes CFG requires |
| `FluxGuidance` node (≠ `cfg`!) | Guidance **distillation** — a genuinely confusing point, see §9.4 |
| `LoraLoader` → `strength_model` | The $\alpha/r$ scaling and why strength 0.7 ≠ "70% trained" |
| `ControlNetApply` vs a LoRA | Conditioning vs. weight modification (see §11.6) |
| Latent is `[1,4,128,128]` in the node tooltip | Actual tensor shapes, actual memory |

**The single "aha" of the whole track**, which the architect must protect and build toward:
*an intimidating variational bound over a 1000-step Markov chain collapses into "predict the noise, MSE loss."*
Everything before §4 exists to set that up. Everything after §4 exists to exploit it.

**The second "aha"**, which the 2026 state of the art makes mandatory: *DDPM and flow matching are the same
object seen from two chart choices.* Most course material online presents them as rival paradigms. They are not.
§6 is where this track earns its keep versus every tutorial on the internet.

---

## 1. Generative modeling framing

### 1.1 Intuition first

> **One sentence:** A generative model is a machine that has been shown a huge pile of examples and has
> internalized *what makes something a member of that pile* well enough to manufacture new members that
> were never in it.

Second sentence, geometric, and this is the load-bearing one for the whole track:

> **The manifold picture.** A 1024×1024 RGB image is a point in $\mathbb{R}^{3{,}145{,}728}$. Almost every
> point in that space is TV static. Real photographs occupy a vanishingly thin, curved, connected-ish sheet
> inside it — a *manifold* of maybe a few thousand intrinsic dimensions. Generative modeling is learning the
> **shape and location of that sheet**, then walking to a random spot on it.

This picture pays for itself later, repeatedly:
- **Diffusion** = a procedure for finding your way back to the sheet from anywhere in the ambient space.
- **The score** = the direction that points toward the sheet (literally, §5).
- **CFG frying** = you extrapolated past the edge of the sheet (§9.3).
- **LoRA** = the sheet you want is a small deformation of the sheet you have (§11.2).
- **Latent diffusion** = don't work in $\mathbb{R}^{3M}$ when the sheet is only ~$10^4$-dimensional (§8).
- **Overfitting a LoRA** = you learned 20 isolated *points* instead of a *patch* of sheet (§12.4).

The course should draw this manifold cartoon once, early, and then *point back at it by name* at each of
those six moments. Recurring imagery is how intuition sticks in a 45-page document.

### 1.2 The math of "learn the distribution"

Data: i.i.d. samples $x^{(1)},\dots,x^{(N)} \sim p_{\text{data}}(x)$, $x \in \mathbb{R}^D$.

We want a model $p_\theta(x)$ with parameters $\theta \in \mathbb{R}^P$ such that $p_\theta \approx p_{\text{data}}$,
and — critically — from which we can **sample cheaply**.

Symbols:
- $x$ — a data point. Shape $[3, H, W]$ for RGB images; $D = 3HW$. Units: pixel intensity, normalized to $[-1,1]$.
- $p_{\text{data}}(x)$ — the true density. **Never known.** Only accessible through samples.
- $p_\theta(x)$ — the model density. Units: probability per unit volume of $\mathbb{R}^D$, i.e. $\text{nats}^{-1}$-ish; in practice we only ever handle $\log p_\theta$.
- $\theta$ — parameters. $P = 12\times10^9$ for FLUX.1-dev, $32\times10^9$ for FLUX.2-dev (§10).
- $N$ — dataset size. LAION-5B scale for pretraining ($5\times10^9$ image-text pairs); **15–30** for a concept LoRA (§12.1). That six-order-of-magnitude gap is the whole reason fine-tuning is a separate topic.

The classical objective is maximum likelihood, equivalently forward KL:

$$
\theta^\star=\arg\max_\theta \frac{1}{N}\sum_{i=1}^{N}\log p_\theta\!\left(x^{(i)}\right)
\;\;\Longleftrightarrow\;\;
\theta^\star=\arg\min_\theta\; D_{\mathrm{KL}}\!\left(p_{\text{data}}\,\|\,p_\theta\right)
$$

with

$$
D_{\mathrm{KL}}(p\|q)=\int p(x)\log\frac{p(x)}{q(x)}\,dx \quad [\text{nats}]
$$

**Why this is hard, stated concretely** (the course must state the obstacle before the solutions, or VAEs and
diffusion look like arbitrary complications):

Write $p_\theta(x) = \tilde p_\theta(x)/Z_\theta$ where $\tilde p_\theta \ge 0$ is anything a neural net can output
and $Z_\theta = \int \tilde p_\theta(x)\,dx$ is the normalizing constant. Then
$\log p_\theta(x) = \log \tilde p_\theta(x) - \log Z_\theta$, and $Z_\theta$ is an integral over
$\mathbb{R}^{3{,}145{,}728}$. It is not merely expensive. It is not computable by any means, ever.

So every generative family is a **dodge around $Z_\theta$**, and the course should present them as exactly that —
four answers to one question:

| Family | The dodge | Consequence |
|---|---|---|
| **Autoregressive** (GPT, PixelCNN) | Factorize $p(x)=\prod_i p(x_i\mid x_{<i})$; each factor is a softmax over a *small* vocab, so $Z$ is a sum over ~$10^5$ terms, not an integral | Exact likelihood; **sequential** sampling ($D$ passes). Fine for 2k tokens, fatal for 3M pixels |
| **VAE** | Don't compute $\log p_\theta$; **lower-bound** it (the ELBO) | Tractable, one-shot sampling; bound is loose → blurry |
| **GAN** | Abandon likelihood entirely; train a critic to say "real or fake" | Sharp; unstable; mode collapse; no likelihood |
| **Diffusion / flow** | Don't model $p$; model $\nabla_x \log p(x)$ — the **gradient** of the log-density, in which $Z_\theta$ *differentiates away* | This is the trick. §5. |

That last row is the punchline and should be flagged as such the moment it is written:

$$
\nabla_x \log p_\theta(x) = \nabla_x \log \tilde p_\theta(x) - \underbrace{\nabla_x \log Z_\theta}_{=\,0,\ \text{$Z_\theta$ has no $x$ in it}}
$$

**This is the mathematical reason diffusion models exist.** The intractable object vanishes under a gradient.
The course should let that land for a full paragraph before moving on. It is one line of high-school calculus
($\frac{d}{dx}\text{const}=0$) doing all the work — exactly the kind of moment this learner will enjoy, and
exactly the kind of moment that gets skipped in tutorials.

### 1.3 GANs — 90 seconds, then move on, but say why

> **Intuition:** A forger and a detective locked in a room. The forger gets better because the detective gets
> better. Nobody ever computes a probability.

$$
\min_G \max_D \; \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim\mathcal{N}(0,I)}[\log(1-D(G(z)))]
$$

- $G:\mathbb{R}^{d_z}\to\mathbb{R}^D$, generator. $d_z$ typically 128–512.
- $D:\mathbb{R}^D\to(0,1)$, discriminator/critic.

Why the course mentions GANs at all — three specific reasons, not for completeness:
1. **Mode collapse** is the cleanest possible illustration of what "cover the whole distribution" means, and diffusion's freedom from it is a real selling point. Show the 8-Gaussians toy where a GAN nails one mode.
2. **The adversarial loss did not die — it moved.** It is the core of modern **few-step distillation**: ADD (SDXL-Turbo), LADD (SD3-Turbo), DMD2 all use a discriminator or a distribution-matching term. When the learner runs Z-Image-Turbo at 8 steps (§10.2), he is running something with GAN DNA. This is a real, non-decorative connection — make it.
3. **The VAE decoder inside every latent diffusion model is GAN-trained.** The SD/FLUX VAEs use an LPIPS + patch-discriminator objective, not pure MSE. That is *why* `VAEDecode` output looks crisp rather than blurry (§2.5). Without this, the "VAEs are blurry" lesson contradicts his lived experience and he will rightly disbelieve the course.

Timing note: 5–10 minutes of a 45-page course. Do not teach WGAN-GP.

### 1.4 Misconceptions — §1

| Misconception | What's actually true | The correction that fixes it |
|---|---|---|
| "The model memorizes a huge lookup of images and blends them." | It learns a *function*. FLUX.1-dev is 24 GB of weights trained on ~$10^9$ images averaging ~100 KB each ≈ $10^{14}$ bytes. **Ratio ≈ 4,000,000 : 1.** | Do the division on screen. You cannot store 100 TB in 24 GB. It is *compression to a rule*, not storage. (Caveat honestly: verbatim memorization of duplicated images *is* documented (Carlini et al. 2023) at rates ~$10^{-5}$ — the rule has exceptions where an image appeared 1000× in the training set. Don't hide this; the learner may have seen the Ann Graham Lotz result.) |
| "Generative = it makes stuff up = it's random." | Sampling is random; the *distribution* is a precise learned object. A fixed seed gives a bit-identical image. | He already knows this from seed-locking in ComfyUI. Say: "the seed is the $z$; the model is a deterministic function of $(z, c)$ for $\eta=0$ samplers." Instant connection. |
| "Diffusion is a totally different idea from VAEs/GANs." | All four are dodges around the same integral $Z_\theta$. Diffusion is *literally* a hierarchical VAE with a fixed encoder (Sohl-Dickstein 2015; Kingma's VDM). | Present the four-row table above as a *single* question with four answers. |
| "More parameters ⇒ better images." | Z-Image (6B) ranked #1 open-weights on Artificial Analysis Image Arena, above FLUX.2-dev (32B) and Qwen-Image (20B) (§10.2). | A concrete, current, verifiable counterexample. Use it — it kills the belief in one shot. |

---

## 2. The VAE — taught properly, because latent diffusion runs on one

This section is **not** background. Every image the learner has ever made in ComfyUI passed through a VAE
decoder on the way out. He has seen the `VAEDecode` node. He may have seen the "VAE" dropdown and had no idea
what it was. Teach it as a first-class citizen.

### 2.1 Intuition first

> **One sentence:** An autoencoder is a lossy codec that *invents its own file format* by being forced to
> squeeze an image through a narrow pipe and rebuild it; the VAE additionally forces the codec's output space
> to be *smooth and fully occupied*, so that random points in it decode to plausible images instead of garbage.

The second clause is the entire content of the KL term and the course must foreshadow it here.

**The physical analogy that works for this learner:** a plain autoencoder is a *hash table with a learned
hash* — it stores each training image at some arbitrary address and knows how to look it up. The addresses
can be scattered anywhere, with vast empty gaps. A VAE is a *coordinate system* — addresses near each other
mean images that look like each other, and there are no gaps, because the training procedure deliberately
smears every image over a small blob of addresses and penalizes the blobs for drifting apart.

### 2.2 Plain autoencoder

$$
\hat x = D_\psi(E_\phi(x)), \qquad \mathcal{L}_{\text{AE}} = \|x - \hat x\|_2^2
$$

- $E_\phi: \mathbb{R}^{D} \to \mathbb{R}^{d}$, encoder, $d \ll D$.
- $D_\psi: \mathbb{R}^{d} \to \mathbb{R}^{D}$, decoder.
- $z = E_\phi(x) \in \mathbb{R}^d$ — the **code** / latent.

**Why this fails as a generative model — make the learner discover it, don't assert it.**
Train an AE on MNIST with $d=2$. Scatter-plot the 60,000 training codes. You see islands with big voids
between them. Now sample $z \sim \mathcal{N}(0,I)$ and decode. You get gray mush, because you landed in a void
and the decoder was never trained on voids — its behavior there is *arbitrary*, it is extrapolating.
This is a **runnable 40-line experiment** (§14, artifact D1) and it should be run before the VAE math appears.
The learner must feel the void.

### 2.3 The VAE: two changes

**Change 1 — the encoder outputs a distribution, not a point.**

$$
q_\phi(z\mid x)=\mathcal{N}\!\left(z;\ \mu_\phi(x),\ \operatorname{diag}\big(\sigma_\phi^2(x)\big)\right)
$$

The encoder network emits $2d$ numbers: $\mu_\phi(x)\in\mathbb{R}^d$ and $\log\sigma^2_\phi(x)\in\mathbb{R}^d$
(log-variance, for positivity and numerical range). Training decodes a *sample*, not the mean:

$$
z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
$$

This is the **reparameterization trick** — and this is the *first* place the course should teach it, because
the *identical* algebraic move reappears in §3.3 as the diffusion jump-to-$t$ formula. Flag the connection
explicitly and forward-reference it. Reusing one trick twice is worth two teachings of two tricks.

Why the trick is needed, stated so it actually lands: you cannot backprop through "draw a sample from a
distribution whose parameters depend on $\theta$" — sampling is not a differentiable operation, there is no
$\partial(\text{sample})/\partial\mu$ because the sample isn't a function of $\mu$, it's a *draw*. The trick
**moves the randomness out of the path**: $\epsilon$ becomes an *input*, drawn once, external, and $z$ becomes
a plain differentiable function $z(\mu,\sigma,\epsilon) = \mu + \sigma\epsilon$ of $\mu$ and $\sigma$, with
$\partial z/\partial\mu = 1$ and $\partial z/\partial\sigma = \epsilon$. Gradients flow. The dice are still
rolled — they're just rolled *outside* the computation graph and handed in.

> **Warning box material.** "Reparameterization" means two different-looking things in this course and they are
> the same thing: (a) moving noise out of the gradient path in a VAE, (b) collapsing 1000 noising steps into
> one formula in diffusion. Both are the identity *"a Gaussian is an affine function of a standard Gaussian"*:
> $\mathcal{N}(\mu,\sigma^2) \equiv \mu + \sigma\,\mathcal{N}(0,1)$. That one line is the whole idea, twice.

**Change 2 — a prior, and a penalty for deviating from it.**

$$
\boxed{\;\mathcal{L}_{\text{VAE}}
=\underbrace{\mathbb{E}_{q_\phi(z|x)}\big[\|x-D_\psi(z)\|_2^2\big]}_{\text{reconstruction}}
\;+\;\beta\,\underbrace{D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\big\|\,p(z)\right)}_{\text{regularizer}}\;}
$$

with $p(z)=\mathcal{N}(0,I_d)$. For diagonal Gaussians the KL has a **closed form** — no sampling, no
estimator, just arithmetic on the encoder's own outputs:

$$
D_{\mathrm{KL}}\big(\mathcal{N}(\mu,\operatorname{diag}\sigma^2)\,\|\,\mathcal{N}(0,I)\big)
=\frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right) \quad[\text{nats}]
$$

The course should have the learner **verify this formula is a distance-like thing** by plugging in $\mu=0,\sigma=1$:
$\frac12(0 + 1 - 0 - 1) = 0$. Zero, as it must be. Ten seconds, and it makes the formula trustworthy instead of decorative.

### 2.4 Why the KL term is there — three answers, give all three

The question "why the KL term?" is the one the brief was explicitly asked to nail. There are three correct
answers at three levels, and the course should give them **in this order**:

**(a) The mechanical answer — it does two jobs, and you can see both.**

$$
\frac{1}{2}\sum_j \underbrace{\mu_j^2}_{\text{job 1: pull blobs to the origin}} + \underbrace{\sigma_j^2-\log\sigma_j^2-1}_{\text{job 2: keep blobs from shrinking to points}}
$$

- **Job 1** ($\mu_j^2$): stops the encoder from fleeing to $\|\mu\|=10^6$ where the blobs never overlap. It **packs** the codes around the origin, filling the voids from §2.2.
- **Job 2** ($\sigma_j^2 - \log\sigma_j^2 - 1$): this expression is $\ge 0$ with **equality exactly at $\sigma_j=1$** — verify by calculus: $\frac{d}{d\sigma^2}(\sigma^2-\log\sigma^2-1) = 1 - 1/\sigma^2 = 0 \Rightarrow \sigma^2=1$. As $\sigma_j\to0$, $-\log\sigma_j^2 \to +\infty$. So the penalty **explodes** if the encoder tries to become deterministic. Without job 2, the model just sets $\sigma\to0$ and you are back to a plain autoencoder with all its voids.

Job 2 is the one people never explain and it is the more important half. The VAE is *forced to be blurry in
latent space*, and that forced blur is what makes nearby codes decode to similar images — because during
training, a whole *blob* of codes around $\mu_\phi(x)$ had to decode to $x$. **Continuity is not hoped for; it
is manufactured, by noise.**

> **Foreshadow hard here.** "Make the code space smooth by forcing a Gaussian blob of codes to decode to the
> same thing." Now re-read that sentence and notice: **that is diffusion.** Diffusion is this idea taken to its
> logical extreme — not one blob at one scale, but a continuum of blobs at 1000 scales, from $\sigma\approx 0$
> to $\sigma \gg \text{data spread}$. The VAE is a one-step diffusion model. Diffusion is a 1000-step VAE. The
> course should plant this flag in §2 and harvest it in §4.5.

**(b) The variational answer — where it actually comes from.**
$\log p_\theta(x)$ is intractable (§1.2). Introduce any $q_\phi(z|x)$ and derive, by inserting
$1 = q_\phi(z|x)/q_\phi(z|x)$ and applying Jensen:

$$
\log p_\theta(x) \;=\; \underbrace{\mathbb{E}_{q_\phi}\!\left[\log p_\psi(x\mid z)\right] - D_{\mathrm{KL}}\!\left(q_\phi(z|x)\|p(z)\right)}_{\textstyle \mathrm{ELBO}(\phi,\psi;x)} \;+\; \underbrace{D_{\mathrm{KL}}\!\left(q_\phi(z|x)\,\|\,p_\theta(z|x)\right)}_{\ \ge 0,\ \text{unknown but non-negative}}
$$

Therefore $\log p_\theta(x) \ge \mathrm{ELBO}$, always. Maximizing the ELBO pushes up a floor under the
likelihood. The gap is exactly how wrong $q_\phi$ is about the true posterior.

And now the payoff: if $p_\psi(x|z) = \mathcal{N}(x; D_\psi(z), \tfrac{1}{2}I)$ — a Gaussian decoder with fixed
variance — then $-\log p_\psi(x|z) = \|x - D_\psi(z)\|^2 + \text{const}$. **The reconstruction MSE is not a
design choice; it falls out of assuming a Gaussian decoder.** The whole ELBO becomes the boxed loss of §2.3.

Do this derivation *properly*, at undergrad pace, because **§4.4 replays the identical structure** and by then
the learner will recognize the moves instead of drowning. The ELBO appears twice in this track. Teach it once,
carefully, on the easy object.

**(c) The information-theoretic answer — the one that sticks.**
$D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))$ is, in nats, **the number of nats of information about $x$ that the code
$z$ carries** beyond what the prior already told you. The ELBO is literally a rate–distortion tradeoff:

$$
\underbrace{\text{distortion}}_{\text{recon error}} + \beta\cdot\underbrace{\text{rate}}_{\text{nats spent}}
$$

$\beta$ is the exchange rate. This is a codec. Say the word "codec" — this learner has encoded video, he knows
what a rate-distortion knob is, and this framing lands instantly with him where "variational inference" would not.

### 2.5 Real numbers — the VAEs actually in his ComfyUI folder

This table should be **printed in the course and referenced by number throughout**. These are the recurring
numbers the brief asked for.

| Model | VAE downsample $f$ | Latent ch. $c$ | 1024² latent shape | Latent elements | Compression vs 3.15M | Confidence |
|---|---|---|---|---|---|---|
| SD 1.5 | 8 | 4 | $[1,4,128,128]$ | 65,536 | **48×** | High |
| SDXL | 8 | 4 | $[1,4,128,128]$ | 65,536 | **48×** | High |
| SD 3 / 3.5 | 8 | **16** | $[1,16,128,128]$ | 262,144 | **12×** | High (verified: SD3.5 uses a 16-ch VAE) |
| FLUX.1 | 8 | 16 | $[1,16,128,128]$ | 262,144 | **12×** | High |
| FLUX.2 | 8 | **32** | $[1,32,128,128]$ | 524,288 | **6×** | Medium — see note |
| Qwen-Image | 8 | 16 (Wan-2.x video VAE lineage) | $[1,16,128,128]$ | 262,144 | 12× | Medium |

> **Note on FLUX.2 VAE — NOW VERIFIED [VP] (§Z-7, constants §9.6). Confidence: HIGH, not Medium.**
> Read directly from the shipped `vae/config.json` of two ungated repos (`black-forest-labs/FLUX.2-klein-4B`,
> `diffusers/FLUX.2-dev-bnb-4bit`) and `AutoencoderKLFlux2.__init__` in diffusers `main`: **`latent_channels: 32`,
> `block_out_channels: [128,256,512,512]` → spatial factor $f = 2^{4-1} = \mathbf{8}$** (exactly how the pipeline
> computes it: `vae_scale_factor = 2 ** (len(block_out_channels) - 1)`). So FLUX.2 VAE is $f=8$, $c=32$ →
> $[1,32,128,128]$ = **524,288 elements = 6.0× compression**; the *transformer* then applies a $2\times2$ patchify
> → $[1,128,64,64]$ → **4096 tokens of width 128** (128 = 32×2×2) — same token count as FLUX.1, doubled per-token width.
> **The DeepWiki "32:1 / $[B,128,16,16]$" claim is WRONG** — it reported VAE ∘ patchify, not the raw VAE factor.
> **The confusion is worth a warning box**: *the VAE's downsample and the transformer's patchify are two different
> compressions and get conflated constantly, including by documentation — DeepWiki is exactly the documentation meant here.*

**Worked example — carry the arithmetic all the way, do not hand-wave:**

SD 1.5, 512×512 image:
- Pixels: $3 \times 512 \times 512 = 786{,}432$ values. At fp32: $786{,}432 \times 4 = 3.146$ MB.
- Latent: $4\times64\times64 = 16{,}384$ values. At fp16: $32.8$ KB.
- **Compression: $786{,}432 / 16{,}384 = 48\times$.** Storage: 3.15 MB → 32.8 KB = **96× fewer bytes** (48× fewer values, half the bytes each).

FLUX.1-dev, 1024×1024:
- Pixels: $3\times1024\times1024 = 3{,}145{,}728$.
- Latent: $16\times128\times128 = 262{,}144$. **12× compression.**
- Patchify $2\times2$: $\frac{128}{2}\times\frac{128}{2} = 64\times64 = \mathbf{4096\ tokens}$, each of width $16\times2\times2=64$, projected up to the model width $d_{\text{model}}=3072$.
- **4096 tokens.** Say that number out loud and connect it to the LLM track: *this is a sequence length of 4096, the same order as a chat context.* The attention cost is $O(4096^2) = 1.68\times10^7$ pairs per head per layer. **In pixel space it would be $(1024\cdot1024)^2 = 1.1\times10^{12}$ pairs — 65,536× more.** This single ratio is the entire justification for latent diffusion (§8) and the course should print it in 24pt.

**Why the SD1.5 → SD3 jump from 4 to 16 channels happened**, since he will have noticed SD1.5 mangles faces
and text and SD3/FLUX don't: 4 channels at $f=8$ is a 48× compression and it is **too aggressive** — the VAE
physically cannot represent small high-frequency structure, so no amount of diffusion-model improvement can
fix hands, eyes at distance, or text. **The VAE is a hard ceiling on the whole pipeline.** Going 4→16 dropped
compression 48×→12×, and 16→32 (FLUX.2) drops it to 6×. The trend is unambiguous: **the field is buying image
quality with latent capacity, and paying in tokens and compute.**

> **Demo (D-VAE-CEILING), high value, cheap to build.** Take a real photo. Encode → decode with the SD1.5 VAE.
> Show the residual $|x - \hat x|$ at 4× gain. Text becomes illegible; the residual lights up on eyes and
> lettering. Now the same image through the SD3/FLUX 16-ch VAE — residual is dramatically flatter.
> **The insight:** "The best diffusion model in the world, given a perfect latent, still can only produce what
> the VAE decoder can express. When SD1.5 wrote garbled text, that was *partly the VAE*, before the U-Net even
> got a vote." This directly explains something he has *seen a hundred times* and never had explained.
> This is the single highest-value-per-line-of-code demo in §2.

**VAE parameter counts and cost:** SD/FLUX VAEs are ~84M params (~168 MB bf16) — **0.7% of FLUX.1-dev's 12B**.
Encoding/decoding is a rounding error next to the 28-step denoising loop. But note the memory *spike*: the
decoder materializes full-resolution activations, and at high resolution this is where ComfyUI OOMs even though
the sampler fit fine. He has almost certainly hit this. `--tiled-vae` exists for exactly this reason. Naming
this out loud will make him trust the course.

### 2.6 Why VAEs alone are blurry — and why his ComfyUI output isn't

**The honest mechanism.** Minimizing $\mathbb{E}\|x - \hat x\|^2$ over a conditional distribution $p(x|z)$
yields, at the optimum, the **conditional mean** $\hat x^\star = \mathbb{E}[x\mid z]$. If the code $z$ is
ambiguous about the exact placement of a high-frequency detail — say, hair strands — then the mean over all
plausible strand placements is *gray*. **MSE + ambiguity = blur, necessarily.** This is not a training failure;
it is what the loss *asks for*. This exact point recurs in §4.7 (why single-step $x_0$-prediction is blurry)
and in §12.4 — teach it once here, properly, with a 1-D worked example:

> Suppose given $z$, the true $x$ is equally likely to be $+1$ or $-1$. The MSE-optimal single prediction is
> $\hat x = 0$ — a value **that never occurs in the data**. The model is not confused; it is correct, for the
> loss it was given. Average of two valid images is an invalid image.

**But — and this must be said or the learner will call BS —** the SD/FLUX VAE decoder is **not** trained on
pure MSE. It is trained with MSE + **LPIPS** perceptual loss + a **patch discriminator** (adversarial). The GAN
term specifically punishes "plausible-on-average" and rewards "committed to *a* specific plausible detail."
That is why `VAEDecode` gives crisp output. **The GAN he was told is obsolete in §1.3 is running inside every
generation he has ever made.** Close that loop explicitly — it is satisfying, it is true, and it rewards paying
attention.

### 2.7 Misconceptions — §2

| Misconception | Truth | Correction |
|---|---|---|
| "The VAE in ComfyUI is a filter / post-processor / 'makes colors better'." | It is the **codec**. The diffusion model *never sees an image*. It only ever sees latents. The VAE is the only thing that has ever touched a pixel. | Show the pipeline diagram with the VAE at both ends and the U-Net/DiT strictly in the middle, touching nothing but $[1,16,128,128]$ tensors. |
| "Latents are a compressed image — like a small JPEG." | Partly true, dangerously so. Latents are $c$-channel, and $c\ne3$. There is no RGB. The `Latent Preview` you see in ComfyUI is a **cheap linear approximation** (a learned $16\times3$ matrix — "TAESD"/approx-decoder), not the latent itself. | Show all 16 channels of a real latent as 16 grayscale images. They look like weird edge/color/texture maps. Nothing is "the red channel." |
| "Mixing up the VAE with another checkpoint's VAE is harmless." | It is a **format mismatch** — SD1.5's 4-ch and FLUX's 16-ch latents are not the same object; even 4-ch↔4-ch across finetunes shifts scale factors and gives washed/purple output. | The `vae_scale_factor` / `shift_factor` constants (e.g. SD1.5 `0.18215`) exist to normalize latents to ~unit variance for the diffusion model. Wrong VAE = wrong normalization = wrong distribution. |
| "$\beta=1$ is just a default." | $\beta$ *is* the rate-distortion knob. SD's VAE uses a **tiny** KL weight (~$10^{-6}$) — it is barely a VAE at all, deliberately: it's an autoencoder with a whiff of regularization, because reconstruction fidelity matters far more than a clean prior. **The prior's job is taken over by the diffusion model.** | This is a genuinely clarifying point: *in latent diffusion, the VAE is not asked to be a good generative model. The diffusion model is the generative model. The VAE just has to be a good, mildly-regularized codec.* |
| "KL makes it generative." | KL makes the latent space *navigable*. Being generative requires a prior you can sample. In LDM, the KL is weak and $\mathcal{N}(0,I)$ is a bad prior for the latent — which is exactly why you need diffusion on top. | Sample $z\sim\mathcal{N}(0,I)$ at the SD latent shape and `VAEDecode` it. You get colored noise, not an image. **Run this.** It proves the SD VAE's prior is *not* $\mathcal{N}(0,I)$ and motivates every remaining page of the course. |

That last row is a **fantastic 5-line demo** and should absolutely ship: it takes the learner's implicit belief
("the latent space is Gaussian, right?") and destroys it in one cell, creating the exact question that §3–§7 answer.

### 2.8 Dependency note for the architect

§2 requires from the shared trunk: MLPs, backprop, MSE loss, softmax-free basics, and — for §2.4(b) — the
learner must have seen $\log$, expectation, and Jensen's inequality at least once. **Jensen is likely a gap.**
Recommend a half-page inset: *for a concave function, the function of the average ≥ the average of the function*,
with the $\log$ picture and one numeric check ($\log(\frac{1+9}{2})=\log 5=1.61 \ge \frac{\log 1 + \log 9}{2}=1.10$).
That inset is used twice (§2.4b, §4.4) and is load-bearing both times. Everything else in §2 is high-school
algebra plus one derivative.

---

## 3. The forward (noising) process

### 3.1 Intuition first

> **One sentence:** Take a photograph and stir in a little static; repeat 1000 times; you end up with pure
> static, and — this is the point — **you have built a smooth, labeled road** from every image to the same
> featureless fog, which a network can then learn to walk backwards.

The forward process has **no learned parameters**. It is not a model. It is a *ruler* — a fixed, hand-designed
corruption ladder whose only job is to manufacture (noisy input, known answer) training pairs for free, from
unlabeled images. That is worth stating explicitly and early, because learners assume everything in a deep
learning pipeline is learned:

> **Warning box.** $q(x_t\mid x_{t-1})$ contains **zero parameters**. Nothing about the forward process is
> trained. You could have written it in 1950. Its entire purpose is to be a *free, infinite supply of
> supervised training data*: it hands you an input ($x_t$) and the exact answer ($\epsilon$) for every one of
> $1000\times N$ pairs, with no human labeling. **Diffusion turns unsupervised generative modeling into plain
> supervised regression.** That sentence is the strategic core of the entire method.

### 3.2 The Markov chain

$$
\boxed{\;q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\;x_{t-1},\ \beta_t I\right)\;}
$$

$$
q(x_{1:T}\mid x_0) = \prod_{t=1}^{T} q(x_t\mid x_{t-1}) \qquad\text{(Markov: each step sees only the last)}
$$

Every symbol, with shape and units:

| Symbol | Meaning | Shape | Range / typical value |
|---|---|---|---|
| $x_0$ | clean data (a latent, in LDM) | $[c,H',W']$ e.g. $[16,128,128]$ | $\approx\mathcal{N}(0,1)$-scaled |
| $x_t$ | data noised to level $t$ | same as $x_0$ | — |
| $t$ | timestep index | scalar int | $1\ldots T$ |
| $T$ | chain length | scalar | **1000** (DDPM, SD1.5/SDXL); continuous $[0,1]$ for flow models |
| $\beta_t$ | variance of noise **added at step $t$** | scalar | $10^{-4}\to 0.02$ (linear, DDPM) |
| $\alpha_t \equiv 1-\beta_t$ | signal retention at step $t$ | scalar | $0.9999 \to 0.98$ |
| $\bar\alpha_t \equiv \prod_{s=1}^{t}\alpha_s$ | **cumulative** signal retention | scalar | $\bar\alpha_0=1 \to \bar\alpha_{1000}\approx 4\times10^{-5}$ |
| $I$ | identity, i.e. noise is **isotropic**: independent per pixel/channel, same variance | $[D,D]$ conceptually | — |

**The $\sqrt{1-\beta_t}$ shrink is the subtle part and the course must not skate past it.**

Why not simply $x_t = x_{t-1} + \sqrt{\beta_t}\,\epsilon$ (just add noise)? Because then variance would *grow
without bound*: $\mathrm{Var}(x_T) = \mathrm{Var}(x_0) + \sum_t \beta_t$, and $x_T$'s scale would depend on $T$
and on the data. Instead, shrinking the signal by $\sqrt{1-\beta_t}$ **exactly compensates** the added
variance. Prove it in two lines — this is high-school algebra and it is deeply satisfying:

$$
\mathrm{Var}(x_t) = (1-\beta_t)\,\mathrm{Var}(x_{t-1}) + \beta_t
$$

If $\mathrm{Var}(x_{t-1}) = 1$, then $\mathrm{Var}(x_t) = (1-\beta_t)\cdot 1 + \beta_t = 1$. **Exactly 1.**

> **This is why it is called "variance preserving" (VP).** The total variance is pinned at 1 forever. The
> process doesn't *add* noise so much as **rotate signal into noise at constant total energy** — a fixed budget,
> continuously reallocated. That is the picture: a fader crossfading from image to static, not a volume knob
> turning static up.
>
> This intuition has a beautiful geometric form the course should draw: $(\sqrt{\bar\alpha_t}, \sqrt{1-\bar\alpha_t})$
> traces the **unit circle** in the first quadrant, since $(\sqrt{\bar\alpha_t})^2 + (\sqrt{1-\bar\alpha_t})^2 = 1$.
> Define $\phi_t = \arccos\sqrt{\bar\alpha_t}$. Then $x_t = \cos(\phi_t)\,x_0 + \sin(\phi_t)\,\epsilon$.
> **The forward process is a rotation.** $\phi=0$ is the image, $\phi=\pi/2$ is pure noise, and the noise
> schedule is nothing but a choice of *how fast to sweep the angle*. Given that this course promises
> high-school trigonometry as a prerequisite, **this is the payoff moment for trig** and the architect should
> treat it as a headline, not a footnote. It also makes v-prediction (§4.3) and the cosine schedule (§3.4)
> fall out for free instead of arriving as arbitrary definitions.

### 3.3 The reparameterization: jump to any $t$ in one step

> **Intuition:** A Gaussian blurred by a Gaussian is a Gaussian. Stirring static in a thousand times and
> stirring the *right amount* in once give **statistically identical** results. So never simulate the chain.

$$
\boxed{\;q(x_t\mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right)
\quad\Longleftrightarrow\quad
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\;\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)\;}
$$

**This equation is the most important line in the entire track.** It should be boxed, numbered, and referred to
by number ("Eq. 3.3") for the next 20 pages. Everything downstream is a consequence of it.

**Derive it, don't assert it.** Two steps, then induct — the learner can follow every line:

$$
x_1 = \sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\,\epsilon_1
$$
$$
x_2 = \sqrt{\alpha_2}x_1 + \sqrt{1-\alpha_2}\,\epsilon_2
= \sqrt{\alpha_2\alpha_1}\,x_0 + \underbrace{\sqrt{\alpha_2(1-\alpha_1)}\,\epsilon_1 + \sqrt{1-\alpha_2}\,\epsilon_2}_{\text{sum of two independent Gaussians}}
$$

Now the key fact — **the merge rule** — which the course should state as its own boxed lemma because it is used
constantly and is genuinely not obvious to a rusty learner:

$$
\mathcal{N}(0,\sigma_a^2 I) + \mathcal{N}(0,\sigma_b^2 I) \;=\; \mathcal{N}\!\left(0,(\sigma_a^2+\sigma_b^2)I\right)
\qquad\text{— \emph{variances} add, not standard deviations}
$$

> **Warning box.** Variances add; standard deviations **do not**. $\sqrt{9}+\sqrt{16}=7$ but
> $\sqrt{9+16}=5$. Noise combines in quadrature, like the hypotenuse of a right triangle — **and it is
> literally the Pythagorean theorem**, because independent Gaussians are orthogonal directions. This learner
> has done RMS and vector addition; say "in quadrature" and "Pythagoras" and it lands in one sentence. Every
> $\sqrt{\;}$ in this entire track is Pythagoras. Say that once and a page of square roots stops looking arbitrary.

Applying the merge rule:

$$
\sigma^2 = \alpha_2(1-\alpha_1) + (1-\alpha_2) = \alpha_2 - \alpha_2\alpha_1 + 1 - \alpha_2 = 1-\alpha_1\alpha_2
$$

so $x_2 = \sqrt{\bar\alpha_2}\,x_0 + \sqrt{1-\bar\alpha_2}\,\epsilon$ with $\bar\alpha_2 = \alpha_1\alpha_2$.
Induction gives Eq. 3.3. **Note the elegance the learner should be invited to admire:** the $\alpha_2$ terms
cancel perfectly. That cancellation is *why* the $\sqrt{1-\beta_t}$ shrink was chosen. The schedule was
designed to make this telescope.

**Why Eq. 3.3 matters — three consequences, all practical:**
1. **Training is $O(1)$, not $O(T)$.** To make a training example at $t=734$, you do *not* run 734 steps. You draw one $\epsilon$, compute one weighted sum, done. Without this, DDPM training would cost 1000× more and the field would not exist.
2. **$t$ becomes a free, uniformly-sampled input.** Each training example picks $t\sim\mathrm{Uniform}\{1,\dots,T\}$. One network handles all noise levels. This is *why* it's one network and not 1000.
3. **It defines the whole game.** $\sqrt{\bar\alpha_t}$ and $\sqrt{1-\bar\alpha_t}$ are the only two numbers that matter. §6 will show that **flow matching just picks different ones**, and *nothing else changes*.

### 3.4 The $\beta$ schedule — what it is and what it is not

> **Warning box, high priority.** In ComfyUI, `scheduler` (karras / normal / exponential / sgm_uniform / beta /
> simple) does **NOT** set the $\beta$ schedule. The $\beta$ schedule is **baked into the trained checkpoint** —
> change it and the model is wrong. The ComfyUI `scheduler` dropdown chooses **which subset of timesteps to
> visit during sampling** — the discretization grid, an *inference-time* choice (§7.5). These are two different
> objects with confusingly similar names and this confusion is nearly universal. He almost certainly has it.
> **Fix it explicitly, in a box, with both names side by side.**

**Linear schedule** (Ho et al. 2020, DDPM; SD1.5 uses the closely-related "scaled linear"):

$$
\beta_t = \beta_{\min} + \frac{t-1}{T-1}\left(\beta_{\max}-\beta_{\min}\right),\qquad
\beta_{\min}=10^{-4},\;\beta_{\max}=0.02,\;T=1000
$$

**Cosine schedule** (Nichol & Dhariwal 2021, iDDPM) — defined directly on $\bar\alpha$, which is the honest
parameterization:

$$
\bar\alpha_t = \frac{f(t)}{f(0)},\qquad f(t)=\cos^2\!\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right),\qquad s=0.008
$$

$$
\beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}, \quad\text{clipped to } \beta_t \le 0.999
$$

Note $s=0.008$ is a small offset preventing $\beta_1$ from being exactly 0 (which would make $t=1$ a no-op and
break the low-noise end). Cosine is exactly the "sweep the angle $\phi_t$ linearly" idea from §3.2 —
$\bar\alpha_t = \cos^2(\phi_t)$ means $\phi_t \propto t$. **The name is not a coincidence and the trig
connection is real.** Reward the learner for the §3.2 setup by cashing it here.

**Worked example — real numbers, carried to a number.** Linear DDPM, $T=1000$:

| $t$ | $\beta_t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$ (signal) | $\sqrt{1-\bar\alpha_t}$ (noise) | SNR $=\frac{\bar\alpha_t}{1-\bar\alpha_t}$ | What it looks like |
|---|---|---|---|---|---|---|
| 0 | — | 1.0 | 1.000 | 0.000 | $\infty$ | clean |
| 1 | 0.0001 | 0.9999 | 1.000 | 0.010 | $10^4$ | clean |
| 100 | 0.0021 | 0.906 | 0.952 | 0.307 | 9.6 | slightly grainy |
| 250 | 0.0051 | 0.596 | 0.772 | 0.636 | 1.47 | clearly noisy, obviously an image |
| 500 | 0.0101 | 0.128 | 0.358 | 0.934 | 0.147 | **noise with a ghost** — you can just make out shapes |
| 750 | 0.0151 | 0.0072 | 0.085 | 0.996 | 0.0073 | looks like noise |
| 900 | 0.0181 | 0.00046 | 0.021 | 0.9998 | $4.6\times10^{-4}$ | noise |
| 1000 | 0.0200 | $4.0\times10^{-5}$ | 0.0063 | 1.0000 | $4.0\times10^{-5}$ | noise |

*(Values computed from $\bar\alpha_t = \prod_{s\le t}(1-\beta_s)$ with the linear schedule; the architect should
regenerate these with a 3-line script and print the exact figures — I have rounded, and the course should ship
numbers it can verify. Confidence: high on the shape and endpoints, medium on my 3rd significant figure.)*

**Read the table out loud to the learner — three observations that matter:**

1. **$\bar\alpha_{1000} = 4\times10^{-5}$, not 0.** The signal is never *completely* gone. $\sqrt{\bar\alpha_T} = 0.0063$ — there is 0.6% of the image still in there at the "pure noise" end. This is a real, known defect: **SD1.5 and SDXL have a nonzero terminal SNR**, so at $t=T$ the model still sees a whisper of the mean brightness of its training data. **This is why SD1.5 cannot make a genuinely pure-black or pure-white image** — it leaks the training-set average luminance. The fix is "zero terminal SNR" (Lin et al. 2024) which rescales the schedule so $\bar\alpha_T = 0$ exactly. **He may well have hit this** — it's the notorious "SD can't do dark images" bug. Explaining a known, famous artifact from first principles, in one line of a table he's looking at, is exactly the kind of moment that converts a reader into a believer.

2. **Most of the action is at low $t$.** From $t=0$ to $t=250$, $\bar\alpha$ falls 1.0 → 0.60. From $t=750$ to $t=1000$ it falls 0.0072 → 0.00004 — a change that is *visually nothing*, both ends are static. **The linear schedule wastes ~25% of its steps in a regime where nothing happens.** That is precisely Nichol & Dhariwal's complaint and the reason for cosine.

3. **The interesting band is roughly $t\in[300,700]$**, where SNR crosses 1. This is where the model is deciding *composition* — what goes where. Low $t$ is texture and detail. **The course should nail this to something he knows:** this is exactly why LoRA training with a badly-chosen timestep distribution learns style but not structure, and it is why §6.4's timestep-sampling choice is a real hyperparameter and not a formality.

### 3.5 Resolution dependence — the bug nobody expects

> **Intuition:** Downscaling an image *averages* pixels, which *destroys noise faster than it destroys signal*
> (noise is uncorrelated between neighbors and cancels; signal is correlated and survives). So the same
> $\beta$ schedule is **much less destructive at high resolution.**

Concretely: a $2\times2$ average of 4 i.i.d. noise samples has variance $\sigma^2/4$ (std $\sigma/2$), while a
$2\times2$ average of 4 nearly-identical signal pixels is unchanged. So at 1024², $x_{999}$ can still contain
recoverable low-frequency structure that at 64² would be long gone. **The "pure noise" endpoint is a lie that
gets worse as resolution grows.**

This is not academic. It is the reason for:
- the **`shift` parameter** in SD3/FLUX flow schedules (§6.4), and
- `use_dynamic_shifting` / `base_shift` / `max_shift` in `FlowMatchEulerDiscreteScheduler` (**verified**: these are real parameters in the current `diffusers` API, defaults `shift=1.0`, `base_shift=0.5`), and
- why naively generating at 2048² with an SD1.5-era model produces **duplicated subjects** — two heads, two horizons. The model was never trained at a noise level that destroys structure at that scale, so it never learned to *invent* global composition at 2048²; it only ever learned to refine it.

He has certainly seen the two-heads artifact. Explaining it via "high-res noise isn't noisy enough" is a
genuinely satisfying, non-obvious, correct explanation that no ComfyUI tutorial gives him.

### 3.6 Demos — §3

**D3.1 — Schedule Explorer (essential, cheap, high value).**
- **Plot:** four curves vs $t\in[0,1000]$ on one axes: $\bar\alpha_t$, $\sqrt{\bar\alpha_t}$, $\sqrt{1-\bar\alpha_t}$, and $\log_{10}\mathrm{SNR}_t$ on a twinned right axis.
- **Controls:** radio {linear, scaled-linear, cosine, flow-matching-with-shift}; sliders $\beta_{\min}\in[10^{-5},10^{-3}]$, $\beta_{\max}\in[0.005,0.05]$, $T\in\{100,250,500,1000\}$, $s\in[0,0.05]$, `shift`$\in[0.5,6]$.
- **JS must implement, exactly:** `beta[t]` per the selected schedule; `alphabar[t] = alphabar[t-1]*(1-beta[t])` (cumulative product, running loop, not `Math.pow`); `snr[t]=alphabar[t]/(1-alphabar[t])`. For the FM curve: $\sqrt{\bar\alpha_t}\!\to\!1-t'$, $\sqrt{1-\bar\alpha_t}\!\to\!t'$ with $t'=\frac{\text{shift}\cdot t}{1+(\text{shift}-1)t}$, $t\in[0,1]$.
- **Readout:** a live numeric box showing $\bar\alpha_T$ in scientific notation, and a red flag when $\bar\alpha_T > 10^{-6}$ labeled **"nonzero terminal SNR — this model cannot make a pure black image."**
- **Insight:** drag $\beta_{\max}$ from 0.02 down to 0.008 and watch $\bar\alpha_T$ leap from $4\times10^{-5}$ to ~0.02 — *the endpoint stops being noise at all*. The learner discovers by dragging that the schedule's job is to land $\bar\alpha_T$ at (nearly) zero, and that linear-vs-cosine is a fight about **how to spend the middle**, not the ends.

**D3.2 — Noise Ladder on a real image (the visceral one).**
- **Plot:** an 11-frame filmstrip of one real 256×256 image at $t\in\{0,100,\dots,1000\}$, computed live in a `<canvas>`.
- **JS:** Box–Muller for $\epsilon$ (`u1,u2~U(0,1)`; `z = sqrt(-2*ln(u1))*cos(2*PI*u2)`); then per pixel, per channel: `x_t = sqrt(alphabar[t])*x0 + sqrt(1-alphabar[t])*eps`, on data mapped to $[-1,1]$, then mapped back to $[0,255]$ for display.
- **Controls:** schedule radio (shared state with D3.1), plus a **resolution toggle {64², 256², 1024²}** rendering the *same* image at three sizes with the *same* schedule.
- **Insight #1:** at $t=500$ you can still see the picture. The learner's mental model of "halfway = half noise" is wrong; the schedule is wildly nonlinear in perceptual terms.
- **Insight #2 (the money shot):** with the resolution toggle at $t=800$, the 64² tile is dead static and the 1024² tile **still visibly shows the subject**. §3.5, discovered by dragging, in three seconds. Make this the demo that closes §3.

**D3.3 — "Jump vs. Walk" verifier (proves Eq. 3.3).**
- **Plot:** two histograms overlaid, plus two image tiles.
- **JS:** Path A: loop $t=1..500$ applying $x \leftarrow \sqrt{1-\beta_t}\,x + \sqrt{\beta_t}\,\epsilon_t$ with 500 fresh draws. Path B: one line, $x = \sqrt{\bar\alpha_{500}}x_0 + \sqrt{1-\bar\alpha_{500}}\epsilon$. Show both tiles and both pixel histograms; print measured mean/std of each.
- **Insight:** the tiles differ (different noise draws) but the **histograms lie on top of each other** and the measured stds agree to ~3 decimals. *"Different pictures, same distribution — and that is the only thing that matters."* This is the demo that teaches what "statistically identical" means, which is a concept the learner needs and probably does not have crisply. It also makes Eq. 3.3 *believed* rather than *accepted*.

### 3.7 Misconceptions — §3

| Misconception | Truth | Correction |
|---|---|---|
| "The forward process is a neural network / is learned." | Zero parameters. Fixed. Hand-designed in advance. | Show the 3-line implementation. There is nothing to train. |
| "The model runs the forward process during training." | It runs **one** application of Eq. 3.3 at a random $t$. Never the chain. | The training loop is 6 lines (§4.5) and there is no loop over $t$ in it. |
| "$t$ is time / the model runs forward then backward." | $t$ is a **noise-level index**, nothing more. Forward is only used to *make training data*. At inference the forward process is never executed at all. | Rename it in your head: call it "the noise dial." Then `denoise=0.6` in img2img means "start the dial at 60%," which is exactly true. |
| "$\beta_t$ is the amount of noise in $x_t$." | $\beta_t$ is the noise added **at step $t$ alone** — a *rate*. The noise **in** $x_t$ is $\sqrt{1-\bar\alpha_t}$ — a *level*. At $t=500$: $\beta_{500}=0.0101$ (tiny) but $\sqrt{1-\bar\alpha_{500}}=0.934$ (nearly all noise). | Rate vs. level. He knows this distinction from every physical system he's ever touched (flow rate vs. tank level). Use that. |
| "$x_T$ is pure noise." | $\bar\alpha_{1000}=4\times10^{-5}\ne 0$. It is 99.998% noise with a measurable DC leak — which causes the famous SD1.5 dark-image failure. | The table in §3.4. Then: "zero-terminal-SNR" is a real fix to a real bug, not a research curiosity. |
| "ComfyUI's `scheduler` dropdown picks the $\beta$ schedule." | It picks the **sampling timestep grid**. The $\beta$ schedule is frozen in the checkpoint. | §3.4 warning box. Highest-priority correction in this section — it's a near-universal confusion and he's had 100 chances to acquire it. |
| "Gaussian noise is added to make it robust / for regularization." | Gaussian specifically, because **only** Gaussians (a) merge in closed form (Eq. 3.3), (b) have a linear score $\nabla\log p = -\epsilon/\sqrt{1-\bar\alpha}$, and (c) make the reverse posterior Gaussian for small $\beta$. Swap in uniform noise and the entire mathematical apparatus collapses. | The choice is forced by tractability at three separate points. Name all three. It converts "why Gaussian?" from a shrug into a design constraint. |

### 3.8 Dependency order — §3

**Requires:** §2 (reparameterization trick; the *same* algebra, already seen once), plus from the trunk:
Gaussian mean/variance, "variances add in quadrature," and the geometric series / cumulative product idea.
**Nothing else.** §3 is deliberately reachable with high-school algebra + trig and the architect should
protect that — it is the section that proves the course's premise.

**Enables:** §4 (which is *only* an inversion of §3), §5, §6, §8.

**Do not teach before §3:** the U-Net (§7), CFG (§9), or anything about samplers. Learners who see samplers
early conclude that diffusion "is" the sampler. It is not; the sampler is a numerical-integration afterthought
bolted onto §4.

---

## 4. The reverse (denoising) process — where the whole thing pays off

### 4.1 Intuition first

> **One sentence:** If you knew, for a noisy image, *exactly which static was stirred in*, you could subtract
> it and get the picture back — so train a network to look at static-plus-picture and point at the static.

And the strategic sentence, which the architect should treat as the thesis of the track:

> **The forward process handed us a supervised dataset for free.** We know $x_t$ because we built it. We know
> $\epsilon$ because we drew it. **So predict $\epsilon$ from $x_t$.** That is a regression problem — the most
> boring, best-understood problem in machine learning. The entire mystique of generative modeling reduces to
> fitting a curve.

### 4.2 The reverse chain

We want $q(x_{t-1}\mid x_t)$ — un-stir one step. **We cannot have it**: by Bayes,
$q(x_{t-1}|x_t) = q(x_t|x_{t-1})q(x_{t-1})/q(x_t)$, and $q(x_{t-1})$ is the marginal data distribution at
noise level $t-1$ — the very thing we don't know (§1.2). Dead end.

**The escape, and it is a genuinely beautiful one:**

> **Theorem (Feller, 1949).** If $\beta_t$ is small enough, then $q(x_{t-1}\mid x_t)$ is **approximately
> Gaussian**.

So: parameterize it as a Gaussian and *learn the mean*.

$$
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\!\left(x_{t-1};\ \mu_\theta(x_t,t),\ \Sigma_\theta(x_t,t)\right)
$$

> **Why this is the crux, in one paragraph the course must include.** The true reverse of a big noising step is
> a horror — given a completely static-y image, the set of clean images it could have come from is
> astronomically large and shaped like the whole data manifold. That distribution is as complicated as
> $p_{\text{data}}$ itself. **But the reverse of a *tiny* step is simple**: given $x_t$, the possible
> $x_{t-1}$ form a small, roughly-Gaussian blob, because you only removed $\beta_t = 0.01$ worth of variance
> and there just isn't room for multimodality in that little ball. **This is the entire reason $T=1000$.**
> Not for accuracy. Not for quality. Because **1000 small steps are each individually Gaussian, and one big
> step is not.** Diffusion trades one impossible problem for a thousand easy ones. That trade is the invention.

That paragraph is worth a full page of the course with a figure: one drawing showing the reverse of a big step
(a wild multimodal cloud) next to the reverse of a small step (a tidy little ellipse). **Nothing else in the
track explains "why 1000 steps?" correctly**, and every learner asks.

**The tractable cousin.** While $q(x_{t-1}|x_t)$ is unknown, $q(x_{t-1}\mid x_t, x_0)$ — the reverse step **if
you were told the answer** — is *exactly* Gaussian, in closed form, with no approximation:

$$
q(x_{t-1}\mid x_t,x_0) = \mathcal{N}\!\left(x_{t-1};\ \tilde\mu_t(x_t,x_0),\ \tilde\beta_t I\right)
$$

$$
\tilde\mu_t(x_t,x_0)=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0
+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,
\qquad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
$$

Note $\tilde\mu_t$ is a **convex combination** of $x_0$ and $x_t$ (the two coefficients are positive and — the
learner should check — sum to 1 only in the appropriate limit; the exact statement is that it's the posterior
mean of a Gaussian conditional, derived by completing the square). The course should present this as: *"the
best guess for $x_{t-1}$ is somewhere on the line between where you are and where you're going, and the
schedule tells you how far."*

**Derivation note for the architect:** this comes from Bayes + completing the square in the exponent. It is
a page of algebra with no ideas in it. **Recommendation: state the result, verify it numerically in a demo
(D4.1), and put the algebra in a collapsible appendix.** Spending a page on completing-the-square will cost
more learners than it teaches. Do spend the page on the Feller paragraph above instead — that one has an idea in it.

### 4.3 What the network actually predicts — $\epsilon$ vs $x_0$ vs $v$

**This is the confusion the brief flagged, and it deserves its own headline. Here is the fix.**

> **The one sentence that dissolves it:** $x_t$, $x_0$, $\epsilon$, and $v$ are **four points on one rigid
> triangle**. Eq. 3.3 says $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ — that is **one
> equation with two unknowns and one known**. Give me *any one* of $\{x_0, \epsilon, v\}$ and I can solve for
> the others in one line of algebra. **The network is not choosing what to know. It always knows the same
> thing. It is only choosing which coordinate to report.**

The conversions — the course should print this box and let the learner verify one of them by hand:

$$
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\;\hat\epsilon}{\sqrt{\bar\alpha_t}}
\qquad\Longleftrightarrow\qquad
\hat\epsilon = \frac{x_t - \sqrt{\bar\alpha_t}\;\hat x_0}{\sqrt{1-\bar\alpha_t}}
$$

$$
v \equiv \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0
\qquad\Longrightarrow\qquad
\hat x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,\hat v,
\qquad
\hat\epsilon = \sqrt{1-\bar\alpha_t}\,x_t + \sqrt{\bar\alpha_t}\,\hat v
$$

**And now the trig picture from §3.2 makes $v$ obvious instead of arbitrary** — this is the second payoff of
the unit-circle setup, and the architect must not skip it:

With $\sqrt{\bar\alpha_t} = \cos\phi_t$, $\sqrt{1-\bar\alpha_t} = \sin\phi_t$:

$$
x_t = \cos\phi_t\; x_0 + \sin\phi_t\;\epsilon
\qquad\qquad
v = \cos\phi_t\;\epsilon - \sin\phi_t\; x_0
$$

**Stare at those two lines.** $(x_0,\epsilon)$ is an orthonormal basis. $x_t$ is that basis rotated by $\phi_t$.
And $v$ is **$x_t$ rotated a further 90°** — it is $dx_t/d\phi_t$, the *velocity* of the point as it sweeps
around the circle. Hence the name.

> **The whole $\epsilon$/$x_0$/$v$ zoo is a choice of which axis to measure against, on a circle.**
> $x_0$ is "the direction you came from." $\epsilon$ is "the direction you're going." $v$ is "the direction
> you're currently moving." **All three are the same triangle.** This is a $\sin/\cos$ picture a high-schooler
> can draw, and it collapses what is, in the literature, three papers' worth of apparent disagreement.
> **Recommend making this a full-page figure with the unit circle, the four labeled vectors, and $\phi_t$
> swept by a slider (D4.2).** It is, in my judgment, the single best explanatory asset available to this track.

**So if they're equivalent, why does the choice matter at all?** Because the *loss weighting* differs.
Predicting $\epsilon$ and predicting $x_0$ give losses that differ by a $t$-dependent factor:

$$
\|\epsilon - \hat\epsilon\|^2 = \frac{\bar\alpha_t}{1-\bar\alpha_t}\,\|x_0-\hat x_0\|^2 = \mathrm{SNR}_t\cdot\|x_0-\hat x_0\|^2
$$

**That factor is the SNR from the §3.4 table, and it spans $10^4$ to $4\times10^{-5}$ — nine orders of
magnitude.** So "$\epsilon$-pred vs $x_0$-pred" is *identical models trained with wildly different emphasis
across noise levels*. That is the entire content of the debate, and stating it this way makes it a
one-paragraph topic instead of a mystery.

Practical consequences — verified against how the real models behave:

| Target | Fails where | Why | Used by |
|---|---|---|---|
| $\epsilon$-pred | **High $t$** (near-pure noise). $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\hat\epsilon)/\sqrt{\bar\alpha_t}$, and $\sqrt{\bar\alpha_{1000}}=0.0063$ → **divide by 0.0063 = multiply the error by 159×.** Catastrophic error amplification. | The conversion has a pole at $\bar\alpha\to0$ | DDPM, SD1.5, SDXL |
| $x_0$-pred | **Low $t$**. At $t=1$ the answer is ≈ the input; the network learns identity and contributes nothing. Also: MSE-on-$x_0$ at high $t$ = predicting the *conditional mean of the data* = **the dataset average** = brown mush (§2.6!). | Trivial at one end, blurry at the other | rare alone |
| $v$-pred | **Nowhere badly.** $v$'s conversions have coefficients $\cos\phi,\sin\phi$, both bounded by 1 — **no division by anything small, no pole.** Numerically stable at both ends by construction. | It's the *rotation-equivariant* choice | SD 2.1-768, SDXL-refiner-ish, distillation, **and it is the ancestor of flow matching's velocity** |

> **The punchline, and it should be delivered as one:** $v$-prediction was invented (Salimans & Ho 2022) to fix
> $\epsilon$-pred's numerical blowup at high noise. Flow matching (§6) independently arrived at predicting a
> velocity. **They are nearly the same object.** $v$-prediction is the bridge, and if the course teaches $v$
> properly in §4, then §6 costs almost nothing. **Architect: this is a deliberate setup. Do not cut $v$-pred
> from §4 to save space — it is prepaying for §6.**

### 4.4 The ELBO collapse — the "aha"

**This is the emotional center of the track. Budget 3–4 pages. Do not rush it.**

The setup should be theatrical and the architect should let it be. Show the monster first:

$$
-\log p_\theta(x_0) \;\le\;
\mathbb{E}_q\Bigg[\underbrace{D_{\mathrm{KL}}\big(q(x_T|x_0)\,\|\,p(x_T)\big)}_{L_T}
+\sum_{t=2}^{T}\underbrace{D_{\mathrm{KL}}\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)}_{L_{t-1}}
\underbrace{-\log p_\theta(x_0|x_1)}_{L_0}\Bigg]
$$

Let the learner recoil. **A thousand KL divergences between distributions over a 3-million-dimensional space.**
Then take it apart, term by term, and watch it evaporate:

**Term 1: $L_T$ — gone.** $q(x_T|x_0)=\mathcal{N}(\sqrt{\bar\alpha_T}x_0, (1-\bar\alpha_T)I)$ with
$\sqrt{\bar\alpha_T}=0.0063$, and $p(x_T)=\mathcal{N}(0,I)$. These are **almost the same distribution**, and
crucially $L_T$ **contains no $\theta$**. It is a constant. Drop it. *(And note: $L_T \ne 0$ exactly — its
size is precisely the terminal-SNR bug of §3.4. The bug and this term are the same fact. Nice closure.)*

**Term 3: $L_0$ — one term out of 1001. Ignore for now.** (In practice it's a discretized Gaussian likelihood
for the final pixel quantization.)

**Term 2: $L_{t-1}$ — this is the whole thing, and it collapses.** Both arguments are Gaussians with the
**same, fixed** covariance ($\Sigma_\theta = \tilde\beta_t I$, chosen not learned). And the KL between two
Gaussians with identical covariance is — this is worth its own boxed lemma —

$$
D_{\mathrm{KL}}\big(\mathcal{N}(\mu_1,\sigma^2 I)\,\|\,\mathcal{N}(\mu_2,\sigma^2 I)\big)=\frac{\|\mu_1-\mu_2\|^2}{2\sigma^2}
$$

> **Stop. Read that.** A KL divergence — an integral over $\mathbb{R}^{3{,}145{,}728}$, the intractable
> nightmare of §1.2 — **is a squared distance between two vectors, divided by a number.** When the covariances
> match, the integral has a closed form and it is just Euclidean distance. **This one lemma is where the
> monster dies.**

So:

$$
L_{t-1} = \frac{1}{2\tilde\beta_t}\left\|\tilde\mu_t(x_t,x_0) - \mu_\theta(x_t,t)\right\|^2
$$

A **plain MSE between two means**. Now substitute the $\epsilon$-parameterization —
$\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$
and the matching expression for $\tilde\mu_t$ — and the $x_t$ terms **cancel identically**, leaving:

$$
L_{t-1} = \underbrace{\frac{\beta_t^2}{2\tilde\beta_t\,\alpha_t\,(1-\bar\alpha_t)}}_{w_t,\ \text{a scalar you can look up}}\;\left\|\epsilon - \epsilon_\theta(x_t,t)\right\|^2
$$

And then the last move — Ho et al.'s empirical finding, which the course must present **honestly as a hack that
worked**, not as a derivation:

> **Set $w_t := 1$.** Just... delete the weight. It is not principled. It is not the ELBO anymore. It makes the
> bound *wrong*. **And it makes the images much better.**

$$
\boxed{\;\mathcal{L}_{\text{simple}}(\theta)=\mathbb{E}_{\,x_0\sim p_{\text{data}},\;\epsilon\sim\mathcal{N}(0,I),\;t\sim\mathcal{U}\{1,T\}}
\left[\;\left\|\;\epsilon-\epsilon_\theta\!\left(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\;t\right)\right\|_2^2\;\right]\;}
$$

**That is it. That is the loss. Everything above became this.**

The architect should engineer the page break so the monster ELBO and $\mathcal{L}_{\text{simple}}$ face each
other across a spread. This is a *visual* argument as much as a mathematical one.

> **Flag genuine uncertainty — required by the brief, and this is a real one.** Dropping $w_t$ is not
> justified by theory. The honest story: $w_t\propto \mathrm{SNR}_t$-ish weighting makes the objective a valid
> likelihood bound but produces worse-looking images, because likelihood spends capacity on imperceptible
> high-frequency detail that human eyes ignore. **Setting $w_t=1$ re-weights toward mid/high noise —
> toward composition and structure — which is what people actually look at.** Kingma & Gao's "Understanding
> Diffusion Objectives as ELBO with Simple Data Augmentation" (2023) later showed $\mathcal{L}_{\text{simple}}$
> *is* an ELBO for a noise-augmented data distribution — so it's less unprincipled than it looked in 2020, but
> the honest summary remains: **the field found this by trying it, and the theory caught up afterward.**
> The loss weighting $w_t$ is still an **active research knob in 2026** (min-SNR-$\gamma$, sigmoid weighting,
> P2, EDM's weighting, `logit_normal` timestep sampling in SD3/FLUX are all attacks on the same question).
> **This is a place to show the learner that the field is not finished.**

### 4.5 The training loop — six lines

Print it in the course as a boxed algorithm, then again as real PyTorch (§14, artifact D4):

```
repeat:
  x0  ← sample a batch from the dataset          # [B, 16, 128, 128] latents
  t   ← sample uniform integers in [1, T]        # [B]
  ε   ← sample from N(0, I)                      # [B, 16, 128, 128]
  xt  ← sqrt(ᾱ[t]) * x0 + sqrt(1-ᾱ[t]) * ε       # Eq. 3.3 — ONE line
  loss ← ‖ε − ε_θ(xt, t)‖²                       # MSE. That's it.
  θ   ← θ − η ∇_θ loss
until converged
```

**Six lines. Point at them.** The architect should force the comparison:

> "This is the algorithm that made every image you have ever generated in ComfyUI. There is no adversarial
> game. There is no reinforcement learning. There is no reject-and-resample. **It is `MSELoss()` on a
> regression problem.** The complexity of diffusion models lives entirely in (a) the network architecture,
> (b) the sampler, and (c) the conditioning. **The learning objective is the simplest thing in the entire
> course** — simpler than the LLM track's cross-entropy, arguably."

And then the §2 payoff:

> **Look back at the VAE.** Encoder: fixed, no parameters (it's Eq. 3.3). Decoder: $\epsilon_\theta$. Prior:
> $\mathcal{N}(0,I)$, and now it's *actually correct* because $\bar\alpha_T\approx0$. Loss: reconstruction +
> KL, which collapsed to MSE. **A diffusion model is a VAE with 1000 latent layers and a frozen encoder.**
> Everything in §2 was preparation, and now it cashes out.

### 4.6 Sampling — and why $\eta$ exists

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\right)+\sigma_t z,
\qquad z\sim\mathcal{N}(0,I)\ \text{for}\ t>1,\ z=0\ \text{at}\ t=1
$$

with $\sigma_t^2 = \tilde\beta_t$ (or $\beta_t$; **both work, and that they both work is itself informative** —
it means the sampler is robust to this choice, which foreshadows §7's whole "samplers are interchangeable
integrators" theme).

**Why re-inject noise ($\sigma_t z$) when we just spent the whole section removing it?** Because
$p_\theta(x_{t-1}|x_t)$ is a *distribution*, not a point. Take only the mean at every step and you are doing
gradient ascent to the mode — you get the single most-likely image, over and over, low-diversity and
oversmoothed. **The noise is what makes it a sample instead of an optimum.** This is exactly the §2.6
mean-vs-sample lesson for the third time, and by now the learner should be able to say it before the course does.

*(Then §7 will reveal that DDIM sets $\sigma_t=0$ and works **fine** — which is a genuine puzzle worth flagging
here and resolving there. The resolution: with $\sigma=0$ you're solving the probability-flow ODE, and the
diversity comes entirely from $x_T$ instead of being sprinkled along the way. Foreshadow it; don't resolve it yet.)*

### 4.7 Misconceptions — §4

| Misconception | Truth | Correction |
|---|---|---|
| **"The model removes the noise."** (the big one) | It **predicts** the noise. You subtract it. And it doesn't subtract all of it — it takes one small step and re-noises. | Watch the $\hat x_0$ preview during sampling: at step 1 of 28 the model *already* predicts a full (blurry, wrong) image. It is not "revealing" an image hiding in the noise. **It is hallucinating a guess and then refining the guess.** ComfyUI's latent preview shows exactly this — he has watched it happen. |
| "The image is 'in there' and gets uncovered." | $x_T$ is $\mathcal{N}(0,I)$ — it contains **no information about the output** beyond the seed. The image comes from $\theta$, not from $x_T$. | The **same** $x_T$ with two different prompts gives two completely different images. He has done this. It proves $x_T$ is not a hidden picture; it's a random *address*. |
| "$\epsilon$-pred, $x_0$-pred, $v$-pred are different models / different capabilities." | Same information, different coordinate, different implied loss weighting. | The rigid-triangle picture (§4.3) + the unit-circle figure. This is the definitive fix. |
| "The ELBO is the loss." | $\mathcal{L}_{\text{simple}}$ **is not the ELBO** — $w_t$ was deleted. It is a deliberately mis-weighted bound that produces better images. | Say it plainly. The learner respects honesty and this is a great "the field is empirical" moment. |
| "1000 steps because more steps = more quality." | **1000 steps at training time because small steps are Gaussian and big steps are not.** Sampling can use 20. These are different numbers for different reasons. | The §4.2 Feller paragraph. Then note: SD1.5 was trained with $T=1000$ and he samples it at 20. Nothing breaks. That is only possible because §5/§6 reveal the chain was secretly continuous all along. |
| "Uniform $t$ sampling is obviously right." | It is not obvious and it is not universal. SD3/FLUX use **logit-normal** $t$ sampling (§6.4) — deliberately oversampling the middle where SNR≈1 and composition is decided. This is a live tuning knob **in LoRA training** and he will set it. | Connect forward to §13's `timestep_sampling` hyperparameter. It's not a formality; it's this. |
| "Diffusion is fundamentally different from a VAE." | It is a hierarchical VAE with a fixed encoder. Historically it *came from* that lineage (Sohl-Dickstein 2015). | §4.5's closing paragraph. |

### 4.8 Dependency order — §4

**Requires:** §3 complete (especially Eq. 3.3 and the $\bar\alpha$ table); §2.4b (the ELBO structure — the
learner must have seen "insert $q$, apply Jensen, get a bound" **once already**, on the easy object, or §4.4
will be a wall).

**Enables:** §5, §6, §7, §9, and — critically — **§11–§13**. Fine-tuning is incomprehensible without knowing
what the loss is. The architect must resist any temptation to move LoRA earlier for motivation; a learner who
meets LoRA before $\mathcal{L}_{\text{simple}}$ learns a UI, not a method.

**Hard prerequisite from the trunk:** the KL-between-equal-covariance-Gaussians lemma. If the trunk doesn't
have it, §4.4 needs a half-page inset. It is one line to state and the whole collapse hinges on it.

---

## 5. Score matching and the SDE view

### 5.1 Intuition first

> **One sentence:** At every point in image-space there is an arrow pointing "toward more plausible" — uphill
> on the probability density — and if you learn that arrow field, you can start anywhere and *walk home to the
> data manifold*.

That arrow is the **score**. And §1.2 already told the learner why we want it: $Z_\theta$ differentiates away.

$$
s(x) \equiv \nabla_x \log p(x) \qquad \in \mathbb{R}^D,\ \text{same shape as } x
$$

**Say what it is not.** This is *not* $\nabla_\theta$ — not a gradient w.r.t. parameters, not something an
optimizer uses. It is a gradient **with respect to the image itself**: "if I nudge this pixel up, does the
image get more or less plausible?" $D$ numbers, one per pixel. **It is a vector field over image space.**

For a rusty-but-trained learner the right analogy is a **potential field**: $\log p(x)$ is a potential-energy
landscape (upside down — high $\log p$ = low energy = the valley floors are where images live), and the score
is the force. Sampling is letting a particle roll downhill with thermal jitter. That is not a metaphor; it is
literally Langevin dynamics and the course can say so.

### 5.2 The punchline: we already trained a score model

This is the second-biggest "aha" after §4.4 and it costs three lines.

For the forward process, $q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$. Take the log of a
Gaussian density and differentiate w.r.t. $x_t$ — a **high-school calculus exercise**, and the course should
make the learner do it:

$$
\log q(x_t|x_0) = -\frac{\|x_t-\sqrt{\bar\alpha_t}x_0\|^2}{2(1-\bar\alpha_t)} + \text{const}
$$
$$
\nabla_{x_t}\log q(x_t|x_0) = -\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}
= -\frac{\sqrt{1-\bar\alpha_t}\;\epsilon}{1-\bar\alpha_t}
= -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
$$

(using Eq. 3.3 to replace $x_t - \sqrt{\bar\alpha_t}x_0$ with $\sqrt{1-\bar\alpha_t}\,\epsilon$.)

Therefore:

$$
\boxed{\;s_\theta(x_t,t) \;=\; \nabla_{x_t}\log p_t(x_t) \;=\; -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}\;}
$$

> **Read it.** The noise-prediction network **is** the score model, up to a scalar. We trained a score model
> and didn't know it. The intractable $\nabla\log p$ of §1.2 — the thing that was supposed to be the hard part —
> **is the output of the network we already have, divided by a number from a lookup table.**

The negative sign is worth a beat: $\epsilon$ points *away* from the data (it's what was added); the score
points *toward* the data. **Of course** they're negatives. The course should let the learner predict the sign
before revealing it.

**Why this matters practically — it is not just elegance:**
1. It makes the **CFG derivation** (§9.3) possible. CFG is a statement about scores; without §5 it's a magic formula.
2. It makes the **ODE/SDE view** (§5.4) possible, which is what makes 20-step sampling legitimate.
3. It unifies two literatures. Song & Ermon's NCSN/score-matching (2019) and Ho et al.'s DDPM (2020) were developed **independently, with different notation, different motivation, and no shared authors** — and turned out to be the same algorithm. That's a good story and it is **true**, and it primes the learner for §6's identical lesson.

### 5.3 Denoising score matching — why noise was necessary anyway

The learner should be told why you can't just score-match the clean data directly, because it explains why the
noise schedule isn't merely a training-data trick.

Two reasons, both concrete:

1. **The manifold problem.** If the data lives on a thin sheet (§1.1), then $p_{\text{data}}$ is **zero** off the sheet, so $\log p = -\infty$ and $\nabla\log p$ is **undefined almost everywhere**. There is no arrow to learn at the points where you actually start. **Adding noise inflates the sheet into a solid**, giving every point in $\mathbb{R}^D$ a finite density and a well-defined arrow. The noise isn't a nuisance — it is what makes the target *exist*.
2. **The coverage problem.** Even with a smoothed density, at $\sigma$ small the arrows far from the data are astronomically weak and were never trained (you never sample there). So you need **a ladder of noise levels**: at high $\sigma$ the arrows are broad, smooth and point roughly at the data's center of mass; at low $\sigma$ they're sharp and local. **Anneal from high to low and you get from anywhere to the manifold.**

> **That's the schedule.** The $\beta$ schedule of §3.4 is not bookkeeping — it is *a curriculum of smoothing
> scales*, and sampling is **annealing** down it. This reframing should be stated explicitly because it makes
> "why a schedule at all?" answerable in one sentence: *you need coarse arrows to find the neighborhood and
> fine arrows to find the house.*

Tweedie's formula is worth one boxed line for the learner who likes the algebra clicking shut:

$$
\mathbb{E}[x_0 \mid x_t] = \frac{x_t + (1-\bar\alpha_t)\,s(x_t)}{\sqrt{\bar\alpha_t}}
$$

Substituting the boxed score identity recovers **exactly** the $\hat x_0$ conversion of §4.3. Nothing new — but
it proves the pieces are consistent, and it names the fact that **$\hat x_0$ is a posterior mean** (which is
*why* it's blurry at high $t$ — §2.6, fourth appearance of the same lesson).

### 5.4 The SDE view — the chain was continuous all along

> **Intuition:** Take $T=1000$ and let $T\to\infty$ with the steps getting proportionally smaller. The staircase
> becomes a ramp. The Markov chain becomes a **differential equation**.

Song et al. (2021) forward SDE:

$$
dx = \underbrace{f(x,t)}_{\text{drift }[\,\mathbb{R}^D\,]}dt + \underbrace{g(t)}_{\text{diffusion }[\text{scalar}]}\,dw
$$

- $dw$ — Wiener process increment (Brownian motion); $\mathbb{E}[dw]=0$, $\mathrm{Var}[dw]=dt\cdot I$. *Physically: the random kick.* Note $\mathrm{Var}\propto dt$ not $dt^2$ — **noise scales as $\sqrt{dt}$, not $dt$**, which is the one genuinely unfamiliar thing here and deserves a sentence: this is why a random walk goes as $\sqrt{N}$, which he knows.
- $f(x,t) = -\frac{1}{2}\beta(t)\,x$ — the shrink-toward-zero drift (this is the $\sqrt{1-\beta_t}$ of §3.2 in the continuum limit).
- $g(t)=\sqrt{\beta(t)}$.

This particular choice is the **VP-SDE**, and it is *exactly* DDPM's chain as $\Delta t\to0$. **VE-SDE** ($f=0$,
$g=\sqrt{d\sigma^2/dt}$) is Song's NCSN and is what "Karras sigmas" in ComfyUI live in. Naming both, and saying
which ComfyUI setting corresponds to which, is a real service.

**The reverse-time SDE** (Anderson 1982) — the theorem that makes the field work:

$$
\boxed{\;dx = \Big[f(x,t) - g(t)^2\,\underbrace{\nabla_x\log p_t(x)}_{\text{the score — we have this!}}\Big]dt + g(t)\,d\bar w\;}
$$

where $dt$ is negative (time runs backward) and $d\bar w$ is a reverse-time Wiener process.

> **This is a stunning theorem and the course should say so.** *Any* diffusion process can be run backwards
> exactly, and the **only** extra ingredient you need is the score. Not the density. Not $Z$. Just the arrows.
> Anderson proved this in 1982, for reasons having nothing to do with images, and it sat there for 37 years.

**The probability-flow ODE** — same marginals, zero noise:

$$
\boxed{\;\frac{dx}{dt} = f(x,t) - \tfrac{1}{2}g(t)^2\,\nabla_x\log p_t(x)\;}
$$

> **The key fact, and it is genuinely surprising:** this deterministic ODE has the **exact same** $p_t(x)$ at
> every $t$ as the stochastic SDE. Same distribution at every time. Different individual trajectories.

**Why the learner should care — this is where §4.6's puzzle resolves:**
1. **Sampling is now numerical integration of an ODE.** Every sampler in the ComfyUI dropdown is a textbook ODE solver: `euler` = Euler's method (1768). `heun` = Heun/RK2. `dpmpp_2m` = a 2nd-order multistep exponential integrator. **He has been choosing between Euler and Runge–Kutta this whole time.** For someone with an engineering background, this single sentence recontextualizes the entire sampler menu into something he already understands.
2. **Step count = step size.** 20 steps vs 50 steps is $h$ in an ODE solver, and quality-vs-steps is **truncation error**. Not mysticism. §7.
3. **$\eta=0$ (DDIM) is legitimate**, answering §4.6: it's the ODE, and the ODE has the same marginals as the SDE. You don't need the noise. Diversity lives entirely in $x_T$.
4. **Deterministic ⇒ invertible.** Run the ODE *forward* from a real image to get its $x_T$ — **DDIM inversion**. That's how real-image editing, style transfer, and several ComfyUI workflows actually work. Only possible because $\eta=0$.
5. **It sets up §6.** Once sampling is "integrate an ODE," the obvious question is *why is our ODE so curved?* — and flow matching is the answer.

### 5.5 Demos — §5

**D5.1 — The Arrow Field (flagship for §5).**
- **Plot:** a 2-D plane. Background: heatmap of $\log p_t(x)$ for a known toy mixture (e.g. two Gaussians, or the 8-Gaussians ring). Overlay: a quiver plot of $\nabla\log p_t$ on a 25×25 grid. A draggable particle.
- **JS math (all closed-form — no network needed, which is why this demo is cheap and exact):** for a mixture $p_t(x)=\sum_k \pi_k\,\mathcal{N}(x;\sqrt{\bar\alpha_t}\mu_k, (\bar\alpha_t\Sigma_k + (1-\bar\alpha_t))I)$, the score is the responsibility-weighted average $\nabla\log p_t(x) = \sum_k r_k(x)\cdot\big({-}(x-\tilde\mu_k)/\tilde\sigma_k^2\big)$, with $r_k(x)=\pi_k\mathcal{N}(x;\tilde\mu_k,\tilde\sigma_k^2)/p_t(x)$. **Compute this exactly in JS** — it's 15 lines and it means the arrows are *true*, not approximated.
- **Control:** a $t$ slider, $t: 1000 \to 0$.
- **Insight (the money shot):** at $t=1000$ the field is a smooth radial funnel pointing at the origin — **it knows nothing but "go to the middle."** As you drag $t$ down, the funnel *splits*, and two basins separate. **The learner watches the model's "opinion" go from generic to specific.** That is what a diffusion model does, made visible in one slider drag. Every prompt-adherence, composition-vs-detail, and timestep-weighting discussion later in the course can point back at this drag.

**D5.2 — Langevin walker.**
- Same field. Button: release 200 particles from $\mathcal{N}(0,I)$ and integrate $x \leftarrow x + \frac{\eta}{2}s(x) + \sqrt{\eta}\,z$ at **fixed** $t$ (pure Langevin, no annealing), then a second button for **annealed** (sweep $t$ down).
- **Insight:** fixed-$t$ Langevin at small $t$ **fails** — particles get stuck in whichever basin they started near, and the mixture weights come out wrong. Annealed Langevin nails the proportions. **This is §5.3's coverage problem, demonstrated, not asserted.** It's also the single clearest demonstration of *why there is a schedule at all*.

**D5.3 — ODE vs SDE, same marginals.**
- Two panels, same seed distribution. Left: reverse SDE (stochastic). Right: probability-flow ODE (deterministic). Show trajectories as trails.
- **Insight:** trajectories look completely different — SDE paths are jagged and cross; ODE paths are smooth and never cross (uniqueness of ODE solutions!). **But the final histograms match.** "Same destination distribution, different roads." This is what makes DDIM legitimate and it's a 60-second demo.

### 5.6 Misconceptions — §5

| Misconception | Truth | Correction |
|---|---|---|
| "The score is a gradient of the loss w.r.t. weights." | $\nabla_x$, not $\nabla_\theta$. It's a gradient in **image space**, one number per pixel, and it has nothing to do with training. | Print the shapes: score is $[16,128,128]$ — *the same shape as the latent*. Weight gradients are $[12\times10^9]$. Different objects entirely. |
| "Score matching is an alternative to DDPM." | Same algorithm. Two independent 2019–2020 derivations that converged. $s_\theta = -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$. | Show the one-line conversion. Then note both papers are cited in every modern paper *because both notations are still in use* — which is the actual reason the literature looks fragmented. |
| "The SDE view is advanced/optional theory." | It's the reason 20-step sampling exists, the reason DDIM inversion works, and the reason the sampler dropdown has 20 entries. **It is the most practically consequential section in the track.** | Name each ComfyUI sampler as its ODE method. `euler`→Euler. `heun`→RK2. `dpmpp_2m`→2nd-order multistep. |
| "Deterministic sampling loses diversity." | Diversity comes from $x_T$. The ODE map $x_T\mapsto x_0$ is a bijection; feed it a diverse $x_T$, get a diverse $x_0$. | D5.3 — the histograms match. Also: he already knows different seeds give different images with `dpmpp_2m` ($\eta=0$). His own experience disproves it. |
| "Adding noise is a regularizer / data augmentation." | Without noise the score is **undefined** (the manifold has measure zero). Noise creates the target. | §5.3, reason 1. This is a *necessity* argument, not a *helpfulness* argument, and the distinction matters. |
| "`karras` scheduler is a different math." | It's a different **spacing of $\sigma$ values** on the same ODE — a nonuniform grid that puts more solver steps where curvature is high. Pure numerics. | Plot the Karras $\sigma_i = \big(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho})\big)^{\rho}$, $\rho=7$, against uniform. It's a change of variables, nothing more. |

### 5.7 Dependency — §5

**Requires:** §3 (Eq. 3.3), §4 ($\epsilon_\theta$ must exist before you can reveal it's secretly the score).
From the trunk: partial derivatives, $\nabla$ as a vector of partials, $\log$ of a Gaussian. **The SDE/Wiener
material needs a half-page inset** — recommend framing $dw$ as "a random kick whose *variance* is proportional
to $dt$" and explicitly **not** teaching Itô calculus. The learner needs to read the reverse-SDE equation and
know what each symbol means; he does not need to manipulate it.

**Enables:** §6 (flow matching is a different ODE), §9.3 (CFG **cannot** be honestly derived without the score),
§7 (samplers as integrators).

**Architect's call — flag:** §5 could be presented *before* §4 (score-first pedagogy, as MIT's course does) or
after (DDPM-first, as most tutorials do). **Recommendation: after.** DDPM-first gives the learner a concrete
6-line algorithm to hold onto before the abstraction arrives, and the "we already had it" reveal in §5.2 is
much more powerful than the alternative ordering permits. **This is a real fork in the road and the architect
should decide it consciously, once, and be consistent — reconcile with the LLM track's stance on
concrete-then-abstract if there is one.**

---

## 6. Flow matching and rectified flow — and reconciling it with DDPM

**This section is where the track distinguishes itself.** Every 2026 frontier open-weights model — FLUX.1,
FLUX.2, SD3.5, Z-Image, Qwen-Image — is trained with **flow matching**, not DDPM. And essentially every free
tutorial online teaches DDPM and stops. If the course does that, the learner will read a FLUX config file and
find nothing he recognizes: no `betas`, no `alphas_cumprod`, no `num_train_timesteps=1000` in the way he
expects. **He will conclude the course lied to him.** It must not.

### 6.1 Intuition first

> **One sentence:** Instead of a curved, noise-schedule-dependent path from noise to image, just draw the
> **straight line** between them and teach the network the velocity along it.

That's the whole idea. It is almost insultingly simple, and the course should present it that way, because the
learner's reaction should be *"wait — that's allowed?"*

> **The second sentence, which is the actual engineering payoff:** a straight path can be traversed in **one
> big step** without error. A curved path cannot. **All of few-step generation traces back to this.**

**The picture:** DDPM's probability-flow ODE trajectories are *curved* — a particle released at $x_T$ swerves
as it comes in. To follow a curved path with Euler's method you need small steps, or you fly off the curve.
Rectified flow says: **why are we integrating a curve we designed ourselves? Design a straight one.**

### 6.2 The construction

Define the path (SD3/FLUX convention: **$t=0$ is data, $t=1$ is noise** — flag this, it is *opposite* to
DDPM's convention and it is a constant source of sign errors):

$$
\boxed{\;x_t = (1-t)\,x_0 + t\,\epsilon,\qquad t\in[0,1],\ \ x_0\sim p_{\text{data}},\ \ \epsilon\sim\mathcal{N}(0,I)\;}
$$

Linear interpolation. That's it. Differentiate w.r.t. $t$:

$$
\boxed{\;u_t = \frac{dx_t}{dt} = \epsilon - x_0\;}
$$

**The velocity is constant along the path** — it doesn't depend on $t$ at all. The path is a straight line
traversed at constant speed. Train:

$$
\boxed{\;\mathcal{L}_{\text{FM}}(\theta)=\mathbb{E}_{x_0,\,\epsilon,\,t\sim p(t)}\Big[\big\|\,v_\theta(x_t,t)-(\epsilon-x_0)\,\big\|_2^2\Big]\;}
$$

Sample by integrating **backward** from $t=1$ to $t=0$ with Euler:

$$
x_{t-\Delta} = x_t - \Delta\cdot v_\theta(x_t,t)
$$

**Point at that.** One line. No $\bar\alpha$, no $\tilde\beta_t$, no posterior mean, no $\sigma_t z$.
Compare it to §4.6's sampler. **The simplification is not cosmetic** — this is why the field moved.

**The subtlety the course must not skip, or the learner will be confused later.** If the velocity is constant,
why does $v_\theta$ take $t$ as input, and why isn't sampling exactly one step? Because
$v_\theta(x_t,t)$ cannot know $(\epsilon - x_0)$ — it only sees $x_t$, and **many** $(x_0,\epsilon)$ pairs
produce the same $x_t$. So it learns the **conditional average**:

$$
v_\theta^\star(x_t,t) = \mathbb{E}\big[\,\epsilon - x_0 \;\big|\; x_t\,\big]
$$

**Averaging straight lines gives a curved field.** The *individual training paths* are straight; the *learned
marginal velocity field* is curved — because it's an average over crossing lines. **This is exactly the §2.6
"MSE gives the conditional mean" lesson for the fifth time**, and by now the learner should be able to
anticipate it. The architect should let him: pose the question, pause, then answer.

> **This is the single most-missed point about flow matching, in tutorials and in practice.** "Rectified flow
> is straight" is **half true**. The *paths you train on* are straight. The *field you learn* is not, because
> lines cross. And that is precisely why **Reflow** exists (§6.5): re-couple $(x_0,\epsilon)$ so the lines stop
> crossing, and *then* the field really does straighten. **Warning box, high priority.**

### 6.3 The reconciliation — DDPM and flow matching are the same object

**This is the section the brief specifically demanded, and here is the clean way to do it.**

Write the general Gaussian probability path:

$$
\boxed{\;x_t = a_t\,x_0 + b_t\,\epsilon\;}
$$

**Everything** in this track is a choice of the two scalar functions $a_t, b_t$:

| Framework | $a_t$ | $b_t$ | Constraint | Endpoint |
|---|---|---|---|---|
| **DDPM / VP-SDE** | $\sqrt{\bar\alpha_t}$ | $\sqrt{1-\bar\alpha_t}$ | $a_t^2+b_t^2=1$ — a **circle** | $a_T\approx0.006$ (leaky!) |
| **Rectified flow** | $1-t$ | $t$ | $a_t+b_t=1$ — a **straight line** | $a_1=0$ exactly |
| **VE / NCSN (Karras)** | $1$ | $\sigma_t$ | none — variance explodes | $\sigma_{\max}\approx80$ |
| **Cosine ("$\cos/\sin$")** | $\cos(\tfrac{\pi t}{2})$ | $\sin(\tfrac{\pi t}{2})$ | circle, swept uniformly | $a_1 = 0$ exactly |

> **The figure the architect must commission.** One plot: the $(a_t, b_t)$ plane. DDPM traces the **quarter
> unit circle** from $(1,0)$ to $(0,1)$. Rectified flow traces the **straight chord** between the same two
> endpoints. Cosine traces the *same circle as DDPM but at constant angular speed*. **They all start at the
> image and end at noise. They differ only in the route.** This one figure is the entire reconciliation and it
> is a picture a 15-year-old can read.
>
> Note what it makes obvious: **rectified flow is the chord, DDPM is the arc.** Of *course* the chord is easier
> to integrate. Of *course* the chord hits $a=0$ exactly and the arc doesn't (§3.4's terminal-SNR bug — **which
> flow matching simply does not have, structurally**). And the "cosine schedule is better than linear" result
> of §3.4 becomes: *sweep the arc at constant speed rather than lurching.*

**Now the network targets.** Given the path $x_t = a_t x_0 + b_t \epsilon$, the true marginal velocity is
$u_t = \dot a_t x_0 + \dot b_t \epsilon$. For rectified flow, $\dot a = -1, \dot b = +1$, giving $u = \epsilon - x_0$.
For DDPM's circle with $a=\cos\phi, b=\sin\phi$: $u = \frac{d\phi}{dt}(-\sin\phi\, x_0 + \cos\phi\,\epsilon)$ —
and the bracket is **exactly $v$-prediction from §4.3**.

$$
\boxed{\;\text{$v$-prediction is flow matching's velocity, on the circular path instead of the straight one.}\;}
$$

**That is the bridge, and it is why §4.3 taught $v$-pred properly.** The architect should collect the payoff loudly.

And the conversions are trivial. On the rectified-flow path, given $x_t = (1-t)x_0 + t\epsilon$ and
$v = \epsilon - x_0$, solve the 2×2 system:

$$
\boxed{\;\hat x_0 = x_t - t\,\hat v \qquad\qquad \hat\epsilon = x_t + (1-t)\,\hat v\;}
$$

**No square roots. No division by anything small.** Two lines of algebra the learner can verify by
substitution in 30 seconds — and the course should make him do it, because it's the moment he realizes flow
matching isn't a new theory he has to learn, it's the *same triangle with nicer numbers*.

**The summary sentence the course should print in a box:**

> **DDPM and flow matching differ in (1) the route $(a_t,b_t)$, (2) which coordinate the network reports, and
> (3) the loss weighting across $t$. That is all. There is no third theory. Every diffusion/flow paper in
> 2020–2026 is a choice of those three things**, and once you see that, the literature stops being a zoo and
> becomes a table.

This is a **genuinely useful, genuinely true, and genuinely rare** framing. Karras et al.'s EDM paper (2022)
made exactly this argument for the pre-flow era; Lipman et al. (2022) and Albergo & Vanden-Eijnden's
"stochastic interpolants" made it general. **Confidence: high.** The architect can lean on it hard.

### 6.4 What SD3 / FLUX actually do — the practical config

**Verified against the current `diffusers` `FlowMatchEulerDiscreteScheduler` API (v0.39.0, released 2026-07-03).**

**Timestep sampling — logit-normal.** Uniform $t$ is a poor choice for FM: at $t\approx0$ the task is trivial
and at $t\approx1$ it's hopeless. SD3 (Esser et al. 2024) samples:

$$
t = \operatorname{sigmoid}(z) = \frac{1}{1+e^{-z}},\qquad z\sim\mathcal{N}(m,\;s^2),\qquad m=0,\ s=1
$$

This concentrates $t$ near **0.5** — where SNR ≈ 1, where composition is decided (§3.4's third observation).
**Worked example:** with $m=0,s=1$, $P(0.27 < t < 0.73) = P(-1<z<1) = 68\%$. **Two-thirds of every training
batch lands in the middle band.** Compare to uniform, which puts 68% across $[0.16,0.84]$ and wastes ~30% of
compute on the trivial and hopeless ends. *(That's the point of the choice, stated as a number he can check.)*

**The `shift` parameter — the resolution fix from §3.5.** Higher resolution needs *more* noise for the same
perceptual destruction, so shift the schedule toward noisier $t$:

$$
\boxed{\;t' = \frac{\mathrm{shift}\cdot t}{1 + (\mathrm{shift}-1)\,t}\;}
$$

**Verified parameters** in `FlowMatchEulerDiscreteScheduler`: `shift` (default **1.0** = no shift),
`use_dynamic_shifting` (bool, default False), `base_shift` (default **0.5**), `max_shift`,
`num_train_timesteps` (default **1000** — kept for indexing convention even though $t$ is continuous; **this is
itself a nice warning-box item**: the 1000 is vestigial bookkeeping, not a chain length).

- **SD3.5** uses `shift = 3.0` (fixed). *(Confidence: medium-high — widely reported, architect should confirm from the HF `scheduler_config.json`.)*
- **FLUX.1** uses `use_dynamic_shifting = True` — the shift is computed **per-image from the token count**, via `base_shift`/`max_shift` interpolated on sequence length. That's why FLUX handles 1024² and 1536² gracefully from one checkpoint. **This is §3.5's theory shipped as production code**, and the architect should point at it: *the resolution-noise problem was real enough that BFL wired the fix into the scheduler.*

> **This is a fantastic "theory has consequences" moment.** A learner who read §3.5 can now open
> `scheduler_config.json` in his FLUX folder, see `"use_dynamic_shifting": true`, and **know exactly what it is
> and why it's there.** The course should literally instruct him to open that file. That is the moment the
> course stops being a course.

**Sampling.** Euler, and that's usually it:

```
sigmas = linspace(1, 0, N+1)              # the "timesteps"; for FM, sigma == t
sigmas = shift*sigmas / (1 + (shift-1)*sigmas)
for i in range(N):
    v = model(x, sigmas[i], cond)
    x = x + (sigmas[i+1] - sigmas[i]) * v    # note: sigmas decreasing → negative step
```

**Note what's absent:** no $\bar\alpha$ lookup, no posterior variance, no noise re-injection. The reason
`euler` is the default sampler for FLUX in ComfyUI — and works fine, where SD1.5 needed `dpmpp_2m` to look
good at 20 steps — is **that the path is straighter, so first-order integration suffices.** He has observed
this. He has no idea why. Now he does. **Make this connection explicit; it's one of the strongest
"theory-under-the-knob" moments available.**

### 6.5 Reflow, and honest uncertainty

**Reflow** (Liu et al., "Rectified Flow", 2022) attacks §6.2's crossing-lines problem:
1. Train $v_\theta$ on random pairs $(x_0,\epsilon)$.
2. **Generate** a big set of pairs $(\hat x_0, \epsilon)$ by running the model — now $x_0$ and $\epsilon$ are *coupled* by the ODE map, not random.
3. **Retrain** on those pairs. The lines no longer cross, so the learned field straightens.
4. Repeat. Each round straightens further and costs a full training run.

After 1–2 reflow rounds, **1–4 step sampling becomes viable**. This is the ancestry of every turbo/schnell/klein
model (§7.6, §10).

> **Flag genuine disagreement — the brief asked for this and here is a real one.**
> **Is flow matching actually better than DDPM, or is it better-packaged?** The honest 2026 answer is
> *contested*:
> - **The "it's the same thing" camp** (defensible, and I lean here): §6.3 is a proof that FM is a
>   reparameterization. Kingma & Gao (2023) and the EDM line show the objectives differ only in path and
>   weighting, both of which are tunable *within* DDPM. On this reading, FM's win is **a better default and much
>   cleaner code**, not a new capability. Notably, a well-tuned EDM (Karras 2022, VE-flavored, not FM) still
>   posts competitive numbers.
> - **The "it genuinely helps" camp:** the straight path gives exactly-zero terminal SNR for free, is
>   resolution-scalable via one `shift` scalar, integrates with 1st-order solvers, and distills to few steps
>   far more readily. Those are four real engineering wins, and **the entire 2026 frontier chose it** — SD3.5,
>   FLUX.1/.2, Z-Image, Qwen-Image. Revealed preference is evidence.
> - **What is NOT in dispute:** FM's *code* is dramatically simpler, and its *empirical results* at scale are
>   at least as good. Nobody is going back.
>
> **Recommended course stance:** teach the reconciliation (§6.3) as the truth, present the disagreement as a
> real one about *how much* of FM's win is the math vs. the defaults, and be explicit that **this is a question
> the field has not settled and largely stopped asking because the practical answer is clear.** This learner
> will *appreciate* being told the difference between "we know" and "it works and nobody's sure why." Papering
> over it would be the worse error.

### 6.6 Demos — §6

**D6.1 — The $(a_t,b_t)$ route map (must-build; it *is* §6.3).**
- **Plot:** the $(a,b)$ plane, unit circle drawn faintly. Three routes from $(1,0)$ to $(0,1)$: DDPM-linear (a lumpy arc — plot the *actual* $\bar\alpha_t$ values from §3.4), cosine (a uniform arc), rectified flow (the chord). A dot per route, all animated in $t$ together.
- **Side panel:** the same real image, noised per each route, at the current $t$. **Three tiles, live.**
- **Insight:** at $t=0.5$, the three routes have produced **visibly different noise levels from the same $t$.** The learner sees, physically, that "$t$" means different things in different frameworks — the source of half the confusion in the literature — and that they nonetheless share endpoints. **This is the reconciliation, draggable.**

**D6.2 — Straight vs. curved trajectories (the "why few-step works" demo).**
- **Setup:** 2-D toy (8 Gaussians in a ring). Two panels: DDPM probability-flow ODE vs. rectified flow ODE, both with **exact closed-form scores/velocities computed in JS** (mixture-of-Gaussians closed forms as in D5.1 — no training needed, so the demo is exact and instant).
- **Control:** a **step-count slider $N \in \{1,2,4,8,16,32,64\}$**, and a toggle to overlay the exact solution (N=1000).
- **Plot:** the trajectories for 50 particles, plus a **numeric readout of Wasserstein-2 distance to the true distribution** and the endpoint error vs. the N=1000 reference.
- **Insight — this is the one:** drag $N$ down to 4. The DDPM panel's particles **fly off the curve and land in the void between modes** (visible, ugly, and exactly what a 4-step SD1.5 image looks like). The rectified-flow panel's particles **land almost correctly**, because the path was nearly straight and Euler is exact on straight lines. **The learner discovers, by dragging one slider, why the whole field switched to flow matching and why turbo models exist.** Highest-value demo in §6, arguably in the track.

**D6.3 — Reflow, in two clicks.**
- Same toy. Button 1: "train round 1" (fit a tiny JS MLP, or use the closed-form field). Show trajectories: **crossing**. Button 2: "reflow" — regenerate pairs by integrating, re-fit. Show trajectories: **noticeably straighter, fewer crossings.**
- **Insight:** "straight paths" was a *goal*, not a *given*, and reflow is the procedure that pays for it — with a whole extra training run. This kills the "rectified flow is automatically straight" misconception permanently.

**D6.4 — Timestep sampler.**
- **Plot:** histogram of 10,000 draws of $t$, plus the resulting histogram of the noise level $b_t$.
- **Controls:** radio {uniform, logit-normal, logit-normal + shift}; sliders $m\in[-2,2]$, $s\in[0.2,2]$, `shift`$\in[0.5,6]$.
- **JS:** Box–Muller → $z$; $t=1/(1+e^{-(m+sz)})$; then $t' = \text{shift}\cdot t/(1+(\text{shift}-1)t)$.
- **Readout:** live "% of samples in the SNR≈1 band $t\in[0.3,0.7]$".
- **Insight:** he will set `timestep_sampling` and `shift` as **actual hyperparameters** in kohya/ai-toolkit (§13). This demo shows what those two strings do to the training distribution — and the readout makes the §6.4 worked example ("68% in the middle") something he *produced* rather than read.

### 6.7 Misconceptions — §6

| Misconception | Truth | Correction |
|---|---|---|
| **"Flow matching is a different/rival theory to diffusion."** | Different $(a_t,b_t)$ on the same construction. §6.3. | The route-map figure. One picture, permanent fix. |
| **"Rectified flow paths are straight, so one step works."** | The *training* paths are straight. The *learned field* is curved because it averages crossing lines. One step does **not** work for a base FM model — that's what reflow/distillation are for. | D6.3. Also: FLUX.1-dev needs ~28 steps. If FM were one-step-able for free, it wouldn't. His own experience is the proof. |
| "$t$ means the same thing everywhere." | **DDPM: $t=0$ data, $t=T$ noise. SD3/FLUX: $t=0$ data, $t=1$ noise (continuous). Karras: parameterized by $\sigma$, and $\sigma=80$ is the noisy end.** Three conventions, all live, all in current code. | A conventions table, printed once, referenced always. **Architect: pick ONE convention for the course's own prose and stick to it religiously; show the others only in a translation table.** Convention drift will destroy this track faster than any conceptual gap. |
| "`num_train_timesteps=1000` in a FLUX config means it's a 1000-step chain." | Vestigial. FM is continuous in $t\in[0,1]$; the 1000 is an indexing convention inherited from DDPM tooling. | Have him open the config. Verified: it is a real default in `FlowMatchEulerDiscreteScheduler`. |
| "`shift` is a quality knob to tune at inference." | It's a **schedule-matching** parameter. The model was trained with a shift; sampling with a very different one is a train/test mismatch. FLUX computes it *from resolution*. | Show `use_dynamic_shifting=True` in the real config. It's automatic for a reason. |
| "FM removed the noise / FM is deterministic." | The noise is still there — it's the $t\,\epsilon$ term, and $x_1=\epsilon$ is pure noise. What's gone is the **noise re-injection during sampling**, because the default FM sampler integrates the ODE. There are stochastic FM samplers too. | Point at $x_t=(1-t)x_0+t\epsilon$. The $\epsilon$ is right there. |
| "$v$-prediction is a DDPM thing, unrelated to FM." | $v$-pred **is** the velocity on the circular path. FM's $v$ is the velocity on the straight path. Same concept, two routes. | §6.3's boxed bridge. This is why §4.3 taught $v$. |

### 6.8 Dependency — §6

**Requires:** §3 (Eq. 3.3 → generalized to $x_t = a_tx_0+b_t\epsilon$), §4.3 (**$v$-prediction — non-negotiable
prerequisite**; without it the FM velocity arrives from nowhere), §5.4 (ODE view — without it, "integrate the
velocity field" is meaningless).

**Enables:** §7.6 (distillation/few-step), §10 (**every** 2026 model), §13 (`timestep_sampling`, `shift`,
`model_prediction_type` are real hyperparameters he will type).

**Architect's warning — the sequencing trap.** There is a temptation to teach flow matching *instead of* DDPM,
since it's simpler and it's what the models use. **Resist it.** Three reasons: (1) the $\epsilon$-prediction/
noise-prediction insight (§4) is the conceptual heart and it's much more legible in DDPM; (2) SD1.5/SDXL are
DDPM and are still the best *learning* platforms (fast, small, and the entire LoRA-tutorial ecosystem assumes
them); (3) the reconciliation in §6.3 is only satisfying if you already own the thing being reconciled. **Teach
DDPM, then reveal FM as the same thing with better numbers. The "aha" is the product.**

---

## 7. Architecture: U-Net → DiT, and the 2026 shift

### 7.1 What the network's job actually is — state it before any architecture

> **One sentence:** $\epsilon_\theta$ (or $v_\theta$) is a function from
> **(noisy latent, noise level, text conditioning) → a tensor exactly the same shape as the latent.**

$$
\epsilon_\theta:\ \underbrace{\mathbb{R}^{c\times H'\times W'}}_{x_t}\times\underbrace{\mathbb{R}}_{t}\times\underbrace{\mathbb{R}^{L\times d_c}}_{\text{text}}\ \longrightarrow\ \underbrace{\mathbb{R}^{c\times H'\times W'}}_{\hat\epsilon}
$$

**This is the constraint that dictates everything.** Output shape = input shape. That is *not* a classifier
(which collapses to $K$ logits) and *not* an LLM (which maps a sequence to a distribution over the next token).
It's **dense prediction** — every input element gets an output element. Two families of architecture satisfy
this constraint and both are in his ComfyUI folder:

- **U-Net** — go down in resolution, come back up, wire the levels together. (SD1.5, SDXL.)
- **Transformer (DiT)** — chop into tokens, run attention, un-chop. (SD3.5, FLUX, Z-Image, Qwen-Image.)

The course should frame it as: *both solve "same shape out," and they differ in how information travels
between distant pixels.*

### 7.2 The U-Net — intuition first

> **One sentence:** To decide what a pixel should be, you need to know both *"what texture goes here"* (local,
> cheap) and *"what is this a picture of"* (global, expensive) — so shrink the image until "global" becomes
> "local," think there, then blow it back up while re-injecting the fine detail you set aside on the way down.

**Structure** for SD1.5's U-Net (860M params — *verify against the checkpoint; confidence medium-high*):

| Level | Resolution | Channels | Blocks | Attention? |
|---|---|---|---|---|
| in | 64×64 | 4 → 320 | conv_in | — |
| down 1 | 64×64 | 320 | 2× ResBlock + Transformer | ✓ self + **cross** |
| down 2 | 32×32 | 640 | 2× ResBlock + Transformer | ✓ self + **cross** |
| down 3 | 16×16 | 1280 | 2× ResBlock + Transformer | ✓ self + **cross** |
| down 4 | 8×8 | 1280 | 2× ResBlock | ✗ |
| **mid** | 8×8 | 1280 | ResBlock + Transformer + ResBlock | ✓ |
| up 4..1 | 8→64 | 1280→320 | mirror, **+ skip concat** | ✓ |
| out | 64×64 | 320 → 4 | conv_out | — |

**Why skip connections matter — and the course must get this right because the usual explanation is wrong.**

The usual explanation is "skips help gradients flow," which is the *ResNet* story and it is **not** the U-Net
skip story. The real story:

> **The bottleneck destroys spatial precision, by design.** At 8×8, each activation covers a 64×64-pixel
> patch of the original image. That level *knows* "there's a face, upper-left, looking right." It **cannot
> know** where the individual eyelashes go — that information was averaged away three downsamples ago and is
> **physically not present** in the tensor.
>
> The upsampling path has to put the eyelashes back. **It cannot invent them from an 8×8 tensor.** So the
> U-Net **hands the high-resolution activations forward, around the bottleneck**, and concatenates them on the
> way up. The skip is not an optimization aid. **It is a wire carrying information that would otherwise not
> exist.**

The distinction matters: without skips, a U-Net doesn't train *slower* — it produces **structurally blurry
output**, forever, no matter how long you train. The information is gone. **This is an ablation the course
should show a picture of** (and there are canonical figures in the literature to reference; or generate one
with artifact D6, §14).

The one-sentence version: **"Down-path decides *what*. Up-path decides *where*. The skips are how *where*
survives the trip."**

**Mechanically:** at level $\ell$, `up_ℓ_input = concat([upsample(from_below), skip_from_down_ℓ], dim=channels)`.
Note **concat, not add** (that's the U-Net/ResNet difference) — which is why the up-path convs have double the
input channels.

**Where the timestep enters.** Not as a token. Via **FiLM-style modulation** in every ResBlock:

$$
h \leftarrow \text{GroupNorm}(h)\cdot(1+\gamma(t)) + \beta(t)
$$

where $t$ is first embedded via a **sinusoidal positional embedding** — *the exact same function as the LLM
track's positional encoding*, applied to a scalar noise level instead of a position index — then passed through
a small MLP to produce per-channel $\gamma,\beta$.

$$
\text{emb}(t)_{2i} = \sin\!\left(\frac{t}{10000^{2i/d}}\right),\quad \text{emb}(t)_{2i+1} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)
$$

> **Flag the connection to the architectures/LLM brief explicitly.** Sinusoidal embeddings appear in both
> tracks, for *different reasons* (position vs. noise level) with *identical math*. This is a shared-trunk
> asset and the architect should make sure the trunk teaches it once, so both tracks can just point at it.
> **Reconcile this with the architectures brief.**

Why FiLM and not concatenation: the noise level is a *global* property — it must reach every layer and every
spatial position identically. Multiplying the whole feature map by a $t$-dependent scale is the cheapest way to
say "you are at noise level 734" to 320 channels at once. **The intuition:** $t$ is not *content*, it's a
*mode switch* — it tells the network which job it's doing (coarse composition vs. fine texture), and a mode
switch belongs in the gain, not in the signal.

### 7.3 Where attention sits — and the payoff for the architectures brief

**Inside each `Transformer2DBlock` in the U-Net, there are two attention layers, and they are different animals.
Confusing them is the #1 architecture misconception.**

Flatten the spatial grid at level $\ell$ into a sequence: $[B, C, H, W] \to [B, HW, C]$. Then:

1. **Self-attention:** $Q,K,V$ all from the image tokens. Job: **let distant pixels talk.** "This is a face, so the patch 40 pixels right should be the other eye, at the same skin tone." Cost $O((HW)^2)$.
2. **Cross-attention:** $Q$ from image tokens, $K,V$ from **text** tokens. Job: **inject the prompt.** Cost $O(HW\cdot L)$, $L=77$ for CLIP.
3. **FFN.**

$$
\text{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**The shapes, which are the whole lesson** (SD1.5, level 2, 32×32):

| | $Q$ | $K$ | $V$ | Attn matrix |
|---|---|---|---|---|
| **Self-attn** | $[B,\ 1024,\ 640]$ | $[B,\ 1024,\ 640]$ | $[B,\ 1024,\ 640]$ | $[B,h,1024,1024]$ |
| **Cross-attn** | $[B,\ 1024,\ 640]$ | $[B,\ \mathbf{77},\ 640]$ | $[B,\ \mathbf{77},\ 640]$ | $[B,h,\mathbf{1024},\mathbf{77}]$ |

> **Point at the cross-attention matrix: $[1024, 77]$.** Row $i$, column $j$ = *"how much does image patch $i$
> care about prompt token $j$?"* **That matrix is where the prompt becomes a picture.** It is 1024×77 numbers,
> it is computed 16 times per layer per step, and it is the entire mechanism of text-to-image. Everything else
> is plumbing.
>
> **This is the payoff of the attention material in the architectures brief and the architect must flag the
> dependency both ways.** The learner spent pages on attention for the LLM track. Here it is doing something he
> can *see*: those 77 columns are his prompt, and this matrix decides where each word lands. **Cross-attention
> is the single most reusable concept between the two destination tracks** and the trunk should be built to
> serve both.

**And it's visualizable, which is rare and valuable.** Reshape column $j$ back to $32\times32$ and you get a
**heatmap of where the word "dog" is attending.** This is not a toy — it is exactly the mechanism behind
Prompt-to-Prompt editing, Attend-and-Excite, DAAM, and (importantly for him) **the reason attention-map
manipulation nodes exist in ComfyUI**. Demo D7.2.

### 7.4 The DiT — and the 2026 shift away from U-Nets

> **Intuition:** If the U-Net's multi-scale hierarchy exists to let distant pixels talk cheaply, and attention
> lets *any* token talk to *any* token directly... **why keep the hierarchy?** Just make it all tokens and all
> attention.

**Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT), 2023.** Findings that mattered:
- Replace the U-Net with a plain transformer over latent patches. **Quality tracks compute (Gflops) monotonically**, cleanly, predictably — the *scaling-law* behavior the LLM world already had.
- The inductive biases of the U-Net (locality, hierarchy) were **not necessary**. At scale, they were a constraint.

**Structure:**
1. **Patchify.** $[B,16,128,128] \xrightarrow{2\times2\ \text{patches}} [B,\ 4096,\ 64] \xrightarrow{\text{linear}} [B,\ 4096,\ 3072]$. **4096 tokens** — the number from §2.5, now doing work.
2. $N$ transformer blocks. Self-attention over 4096 tokens; conditioning injected (see below); FFN.
3. **Unpatchify.** $[B,4096,3072] \to [B,4096,64] \to [B,16,128,128]$.

**How conditioning enters a DiT — three designs, and the field converged:**

| Design | Mechanism | Used by |
|---|---|---|
| **adaLN-Zero** | $t$ (and pooled text) → MLP → per-block $\gamma,\beta,\alpha$ scaling every LayerNorm and residual. **Initialized so $\alpha=0$** → each block starts as the identity → stable training of very deep stacks. | DiT, and the $t$-path of essentially everything since |
| **Cross-attention** | separate $K,V$ from text (as in the U-Net) | SD3's predecessors, PixArt-$\alpha$ |
| **MMDiT (joint/"concat") attention** | **Concatenate text tokens and image tokens into ONE sequence**; run self-attention over the union; use **separate weight matrices** for the two modalities. | **SD3/3.5, FLUX.1/.2 — the 2026 standard** |

> **MMDiT is the important one and the course should explain why it won**, because it's a satisfying idea:
>
> Cross-attention is **one-directional** — image queries text, text never queries image. The text
> representation is computed once, frozen, and shouted at the image. **MMDiT lets the text tokens attend to
> the image tokens too**, so the prompt's representation *adapts to what's being drawn*. The text for "a red
> cube on a blue sphere" can resolve which object is which **by looking at the canvas.**
>
> **This is why SD3/FLUX are dramatically better at (a) text rendering and (b) compositional binding
> ("red cube, blue sphere" not "blue cube, red sphere") than SD1.5/SDXL were.** He has *lived* this — SD1.5
> could not write a word and constantly swapped attributes; FLUX writes signs correctly. **The architecture
> table above is the reason.** That is a big, concrete, verifiable "theory explains my experience" moment and
> the architect should build a page around it.

**Shape walk-through for MMDiT (SD3.5-Large, verified: 8B params, 38 layers, CLIP-L + OpenCLIP-G + T5-XXL-4.7B):**
- Text: T5-XXL → $[B, 154, 4096]$ → projected to $[B,154,d]$. Plus pooled CLIP embeddings → the adaLN vector.
- Image: $[B,16,128,128]$ → patchify 2×2 → $[B,4096,d]$.
- **Concatenate:** $[B,\ 4096+154 = 4250,\ d]$.
- Self-attention over **4250 tokens**, with separate $W_Q,W_K,W_V$ and separate LayerNorms/FFNs per modality, but **one shared softmax**.
- Attention matrix: $[B,h,4250,4250]$ — **18.1M entries per head per layer.**

**FLUX.1-dev's variant** (12B, $d_{\text{model}}=3072$): **19 "double-stream" MMDiT blocks** (separate weights per modality) followed by **38 "single-stream" blocks** (shared weights, text and image fully merged). *(Confidence: medium-high — widely reported and consistent with the released code; architect should confirm against `FluxTransformer2DModel` config in diffusers.)* The double→single progression is itself a nice idea: keep the modalities distinct while their representations are still modality-specific, then merge once they've become a common "scene" representation.

**The 2026 verdict — verified across the current landscape:**

| Model | Backbone | Params | Year |
|---|---|---|---|
| SD 1.5 | U-Net | 860M | 2022 |
| SDXL | U-Net | 2.6B | 2023 |
| SD 3.5 Large | **MMDiT** | 8B (38 layers) | 2024 |
| FLUX.1-dev | **MMDiT (double+single)** | 12B | 2024 |
| FLUX.2-dev | **rectified-flow transformer** | **32B** | Nov 2025 |
| FLUX.2-klein | **rectified-flow transformer** | **4B** / 9B | Jan 2026 |
| Qwen-Image | **MMDiT** | 20B | 2025 |
| Z-Image | **single-stream DiT** | **6B** | Nov 2025 |

> **The U-Net is over for frontier text-to-image.** Every model released after ~2024 is a transformer. But be
> precise about *why*, and it's not "transformers are better at images" — it's:
> 1. **Scaling behavior.** DiT quality tracks Gflops predictably. U-Net scaling was a craft.
> 2. **Infrastructure reuse.** The entire LLM stack — FlashAttention, tensor/sequence parallelism, FSDP, kernel fusion, quantization — is *transformer-shaped*. Adopting DiT means inheriting a decade of optimization for free. **This is arguably the biggest reason and it is an engineering reason, not a scientific one.** The learner should hear that.
> 3. **MMDiT's bidirectional conditioning** genuinely fixed text and binding.
>
> **Honest counterpoint, flag it:** U-Nets are *not* worse at small scale. For a <1B model at 256², a U-Net is
> often better and always cheaper — the hierarchy is a genuine prior and priors help when data/compute are
> short. **The course's own from-scratch MNIST model (§14, D3) should be a U-Net**, and the course should say
> why: at 1.5M parameters, the U-Net's inductive bias is an asset, not a cage. **"The right architecture depends
> on scale" is a more valuable lesson than "transformers won."**

### 7.5 Samplers — the ODE-solver menu, decoded

**Intuition first:**

> **One sentence:** The model gives you a direction at every point (§5, §6). A sampler is a **numerical
> recipe for taking steps along that direction field** — and every recipe in the ComfyUI dropdown is a method
> from a 19th- or 20th-century numerical-analysis textbook.

| ComfyUI name | What it is | Order | Model calls / step | Typical steps | Notes |
|---|---|---|---|---|---|
| `ddpm` | the full stochastic chain | — | 1 | 250–1000 | ancestral; the original |
| `euler_a` | Euler + noise re-injection ("ancestral") | 1 | 1 | 20–40 | $\eta>0$: **never converges** — see below |
| `euler` | Euler on the prob-flow ODE | 1 | 1 | 20–50 | $\eta=0$, deterministic. **FLUX default** |
| `ddim` | Euler on the ODE, DDPM parameterization | 1 | 1 | 20–50 | $\eta$ knob: 0=ODE, 1=DDPM. **Invertible at $\eta=0$** |
| `heun` | Heun / RK2 (predictor-corrector) | 2 | **2** | 15–30 | 2× cost per step |
| `dpmpp_2m` | DPM-Solver++ 2nd-order **multistep** | 2 | **1** | **20–30** | reuses the previous step's output → 2nd order at 1st-order cost. **The workhorse.** |
| `dpmpp_2m_sde` | + stochastic term | 2 | 1 | 20–30 | |
| `dpmpp_3m_sde` | 3rd-order multistep | 3 | 1 | 20–30 | |
| `lcm` | few-step distilled | — | 1 | **4–8** | requires an LCM-distilled model |
| `uni_pc` | unified predictor-corrector | 2–3 | 1 | 10–20 | |

**The three things the learner should take away, none of which are in any ComfyUI tutorial:**

1. **DPM-Solver++ 2M is the free lunch and here's the mechanism.** Heun gets 2nd order by evaluating the model **twice per step** (predict, then correct). DPM-Solver++ 2M gets 2nd order by **reusing the model output from the previous step** as the second data point — a linear multistep method (Adams–Bashforth family). Same accuracy, **half the model calls.** *That* is why it's everyone's default. It's not magic; it's that you already paid for that gradient evaluation, so use it.

2. **"Ancestral" ($\_a$) samplers do not converge, and this is a real and confusing fact.** `euler_a` at 20 steps and `euler_a` at 100 steps give **different images**, not "the same image, better." Because $\eta>0$ injects fresh noise at every step, the trajectory depends on the step count. `euler`, `ddim` ($\eta=0$), `dpmpp_2m` **do** converge — crank the steps and the image stops changing. **He has certainly noticed that some samplers "change the image" with more steps and some "sharpen" it, and been baffled.** Here's the answer: *ancestral samplers are solving a different problem at every $N$.* One paragraph, and a genuine mystery from his practice dissolves.

3. **Step count is step size, and the tradeoff is truncation error.** For a $p$-th order method, global error $\sim O(h^p)$ with $h\propto 1/N$. So Euler ($p{=}1$): halve the steps, double the error. DPM++2M ($p{=}2$): halve the steps, **4×** the error — but from a much lower baseline. **This is why the quality-vs-steps curves have different shapes for different samplers**, and it's why the answer to "how many steps?" is "depends on the sampler and on how curved your model's ODE is" (§6!). FLUX needs fewer Euler steps than SD1.5 needs *because flow matching straightened the path.*

**The `denoise` parameter, decoded** (img2img — he uses this constantly): `denoise=0.6` with 20 steps means:
**skip the first 40% of the schedule.** Take the input image, noise it to $t = 0.6\cdot T$, and run only the last
12 steps. That's it. **Low denoise = start close to the image = small change.** Now `denoise` is not a mystery
dial; it's "where on the ladder do I jump in." One sentence, permanent understanding.

### 7.6 Distillation and few-step models — 2026

**Intuition:**

> **One sentence:** A 28-step model has a *teacher's* trajectory; distillation trains a *student* to jump
> straight to the answer, trading the ability to be guided for a 5–10× speedup.

The families, and what's actually shipping in 2026:

| Method | Idea | Steps | Examples |
|---|---|---|---|
| **Progressive distillation** (Salimans & Ho 2022) | student takes 1 step = teacher's 2; halve, repeat | 1024→512→…→4 | historical; introduced $v$-pred |
| **Reflow** (§6.5) | re-couple pairs, retrain, straighten | 1–4 | rectified-flow lineage |
| **Consistency / LCM** | train $f(x_t,t)\to x_0$ to be **the same for all $t$ on a trajectory** | 2–8 | LCM-LoRA (**note: distributed as a LoRA — he's used it**) |
| **ADD / LADD** (adversarial diffusion distillation) | student + **discriminator** (in pixel space / latent space) | 1–4 | SDXL-Turbo, SD3-Turbo |
| **DMD / DMD2** (distribution matching) | match the *distribution*, not the trajectory | 1–4 | strong 2026 results |
| **Guidance distillation** | **bake CFG into the weights** so one forward pass = two | — | **FLUX.1-dev, FLUX.1-schnell** ← see §9.4 |

**Verified 2026 data points — real numbers the course can use:**
- **Z-Image-Turbo** (Alibaba Tongyi, Apache 2.0, Nov 26 2025): **6B, 8 NFEs**, sub-second on an H800, **fits in 16 GB**. Ranked **#1 open-weights** on Artificial Analysis Image Arena — above FLUX.2-dev (32B) and Qwen-Image (20B). **A distilled 6B beating a 32B is the headline of 2026's open-weights year** and it is a fact, not a vibe.
- **FLUX.2-klein-4B** (Apache 2.0, Jan 15 2026): official model card shows **`num_inference_steps=4, guidance_scale=1.0`**, ~13 GB VRAM, sub-second. *(Verified from the HF model card.)*
- **FLUX.2-klein-9B**: ~29 GB at FP16, non-commercial license. FP8 and **NVFP4** quantized checkpoints published by BFL, reported at **−40% (FP8)** and **−55% (NVFP4)** VRAM. *(Confidence: medium — reported by secondary sources; architect should verify against the HF repo.)*

> **The tradeoff the course must state, because he will hit it.** Distilled models are **less steerable**.
> `guidance_scale=1.0` in the klein example is not a suggestion — **CFG largely does not work on a
> guidance-distilled model**, because the model no longer has a separate unconditional branch to extrapolate
> from. Turbo/schnell/klein models also have **narrower output diversity** (distillation is mode-seeking) and
> are **harder to LoRA-train on** (§11). **The step count is not free; you paid for it in controllability.**
> This is exactly the kind of engineering tradeoff this learner will appreciate being told plainly rather than
> discovering at 2 a.m.

### 7.7 Demos — §7

**D7.1 — U-Net skip ablation.**
- **Plot:** side-by-side outputs of two pre-trained tiny MNIST/CIFAR diffusion models — identical except one has skips severed — plus a receptive-field diagram.
- Precomputed images (training in-browser is too slow); the *interactive* part is a slider over "how many skip levels are cut (0/1/2/3)."
- **Insight:** cut 3 skips and the digits are **blobs with the right rough shape and no edges.** The "where" was lost. Not slower training — *structurally impossible* output.

**D7.2 — Cross-attention heatmap (must-build; the payoff demo).**
- **Plot:** a real 512² generation, and next to it, for a prompt like *"a red cube on a blue sphere"*, a **per-token heatmap** overlaid on the image.
- **Controls:** click a word → its $[1024]$ attention column, reshaped to $32\times32$, upsampled, shown as a heat overlay. Slider over **layer** and over **timestep**.
- **Data:** precomputed attention maps (a real diffusers run with hooks, exported as JSON/PNG) — the *math* the JS does is the reshape + normalization + colormap, which is honest, cheap, and exact.
- **Insight #1:** the word "red" lights up **on the cube**, spatially. *You can see the prompt land.*
- **Insight #2 (the good one):** drag the **timestep** slider. At high $t$ the maps are **diffuse blobs** — the model is still deciding *roughly where things go*. At low $t$ they're **tight and object-shaped**. **Composition is decided early; detail late.** This is §3.4's SNR≈1 band, §5.5's splitting funnel, and §6.4's logit-normal timestep sampling — **all three, visible in one drag.** The architect should recognize this demo as a *convergence point* for four separate sections and give it a full page.

**D7.3 — Sampler shootout (make the ODE-solver claim testable).**
- **Setup:** the 2-D toy again (reuse D6.2's closed-form field — zero marginal cost).
- **Controls:** sampler radio {euler, euler_a, heun, dpmpp_2m, ddim($\eta$ slider)}; step-count slider $N\in[1,64]$; a **"model calls" counter** (the honest x-axis).
- **Plot:** trajectories + W2 distance to truth, plotted **against model calls, not steps.**
- **Insight #1:** Heun and DPM++2M reach the same accuracy — but Heun spends **2× the calls.** The multistep free lunch, measured.
- **Insight #2:** crank $N$ on `euler_a`. The result **keeps wandering** and W2 plateaus at a nonzero floor. Crank $N$ on `euler`: W2 → 0. **Ancestral samplers don't converge**, demonstrated in five seconds. This is the demo that resolves a real confusion from his practice.

### 7.8 Misconceptions — §7

| Misconception | Truth | Correction |
|---|---|---|
| "The U-Net's skips are for gradient flow (like ResNet)." | ResNet's residuals are for gradients. **U-Net's skips carry spatial information around a bottleneck that physically destroyed it.** Different purpose, and they're **concat** not **add**. | D7.1. Also point at the channel counts: the up-path convs take *double* input channels. That's the concat, visible in the config. |
| "Self-attention and cross-attention are the same layer." | Self: $Q,K,V$ all from the image, $[1024\times1024]$, job = pixels talk to pixels. Cross: $Q$ image, $K,V$ **text**, $[1024\times77]$, job = **prompt injection**. | The shapes table in §7.3. The 77 is the giveaway — it's his prompt. |
| "The DiT is just 'a transformer instead of a U-Net,' a lateral move." | The MMDiT change (bidirectional text↔image attention) is **why FLUX can write text and SD1.5 couldn't.** That's a capability change, not a refactor. | §7.4. He has direct experience of this exact quality jump. |
| "Transformers beat U-Nets at everything now." | At scale, yes. **At 1.5M params on MNIST, the U-Net wins** — the hierarchy is a useful prior when data/compute are scarce. The DiT's win is a *scaling* win plus an *infrastructure-reuse* win. | The course's own from-scratch model is a U-Net. Say why, explicitly. |
| "More sampler steps always = better." | Only for **converging** samplers ($\eta=0$). `euler_a` at 100 steps ≠ `euler_a` at 20 steps improved — it's a **different image**. And for `dpmpp_2m` past ~30 steps the improvement is below the noise floor of human perception. | D7.3, insight #2. |
| "Samplers are secret sauce / model-specific magic." | Euler (1768). Heun (1900). Adams–Bashforth (1883). **They are textbook ODE solvers applied to §5.4's equation.** | The table in §7.5. For an engineer, this is a relief and a revelation. |
| "`denoise` controls how much noise is removed." | It controls **where on the schedule you start.** `denoise=0.6` = skip the first 40% of steps and begin from a 60%-noised version of your input. | One sentence. Permanent fix. |
| "Turbo/schnell/klein are just faster — free lunch." | You paid: **CFG stops working** (guidance-distilled), diversity narrows, LoRA training gets harder. `guidance_scale=1.0` in the klein docs is a **constraint**, not a default. | §7.6's boxed tradeoff, then §9.4. |

### 7.9 Dependency — §7

**Requires:** from the trunk — convolutions, receptive fields, residual connections, LayerNorm/GroupNorm,
**and the full attention stack (QKV, softmax, multi-head, sinusoidal embeddings)**. §4 (so the learner knows
what the network is *for* before seeing how it's built). §2.5 (latent shapes — the 4096 tokens). §5.4/§6 for §7.5.

**Enables:** §8 (memory arithmetic needs the architecture), §9 (cross-attention *is* the conditioning
mechanism), §11 (**LoRA targets specific matrices — you cannot say "rank-16 on `to_q,to_k,to_v,to_out`" to
someone who doesn't know what those are**).

> **Architect: this is the biggest cross-brief dependency in the whole course.** §7.3 and §7.4 are cashing
> cheques written by the architectures brief. The trunk **must** deliver: QKV attention with shapes, multi-head,
> softmax temperature/scaling, sinusoidal embeddings, LayerNorm+FiLM-style modulation, and residual/skip
> topology. If the trunk teaches attention only in an LLM/causal-mask framing, **this track breaks** — diffusion
> attention is **bidirectional** (no causal mask), and cross-attention with $K,V$ from a *different modality* is
> a structure the LLM track may never show. **Reconcile explicitly with the architectures brief: request
> (a) attention taught mask-agnostically, (b) cross-attention as a first-class case, not a footnote.**

---

## 8. Latent diffusion — the idea that made it run on his desk

### 8.1 Intuition first

> **One sentence:** Don't do expensive thinking in a representation that is 90% redundant — compress the image
> down to its meaningful content first, diffuse *there*, and decompress at the end.

Or, in his language: **you don't edit video by manipulating raw frames; you work in a codec's coefficient
domain because most of the raw signal is perceptually irrelevant.** Latent diffusion is that, learned.

Rombach et al. (2022), "High-Resolution Image Synthesis with Latent Diffusion Models" — the paper that turned
diffusion from a Google-datacenter thing into a thing on a gaming GPU. **This is the paper Stable Diffusion is.**

### 8.2 Why pixel-space diffusion was infeasible — do the arithmetic

The course must not assert this. It must **compute** it, because the numbers are shocking and shocking numbers
are memorable.

**Setup:** generate a 1024×1024 image. The network must run **28 times** (steps), and each run does attention.

**Pixel space, naive:**
- Tokens (1 per pixel): $1024\times1024 = 1{,}048{,}576$.
- Self-attention matrix: $N^2 = 1.049\times10^6{}^2 = \mathbf{1.10\times10^{12}}$ entries.
- **Per head, per layer, in fp16:** $1.10\times10^{12}\times 2\ \text{bytes} = \mathbf{2.2\ \text{TB}}$.

**Stop there.** 2.2 terabytes. For **one attention head**, in **one layer**, at **one step**. A DGX Spark has
128 GB. You are off by a factor of **17**, before you multiply by 16 heads, 38 layers, and 28 steps
(a further **17,000×**). *(FlashAttention avoids materializing the matrix, so the memory figure is the naive
one — but the **compute** is irreducible: the FLOPs scale as $N^2 d$ regardless. The point stands.)*

**Latent space (FLUX.1, $f=8$, 2×2 patchify):**
- Tokens: $\frac{1024}{8\cdot 2}\times\frac{1024}{8\cdot 2} = 64\times64 = \mathbf{4096}$.
- Attention matrix: $4096^2 = \mathbf{1.68\times10^{7}}$ entries = **33.5 MB** in fp16. **On a phone.**

$$
\boxed{\;\frac{1.10\times10^{12}}{1.68\times10^{7}} = \mathbf{65{,}536\times}\;\text{less attention work.}\;}
$$

**65,536 = $16^4$.** Because attention is $O(N^2)$ and we reduced each spatial dimension by 16, we reduced $N$
by $16^2=256$ and the cost by $256^2 = 65{,}536$. **The quadratic bites, and here it bites in our favor.**

> **Print this in 24pt.** *Latent diffusion is a 65,536× reduction in the cost of the expensive operation, in
> exchange for a 12× lossy compression of the image.* **That trade — four orders of magnitude of compute for
> one order of magnitude of fidelity — is why Stable Diffusion exists and why his RTX/Spark can run it.**
> Every subsequent design decision in the field is downstream of that ratio.

**Historical check that this is the real story, not a retrofit:** Google's **Imagen** (2022) and OpenAI's
**DALL·E 2** did diffuse in pixel space — and both needed **cascades**: a base model at 64×64, then two
super-resolution diffusion models (64→256→1024). Three models, three training runs, three sets of artifacts at
the seams. **The cascade was the workaround for not having a VAE.** Latent diffusion replaced three models with
one. That's the argument, and it's historical fact, not hindsight.

### 8.3 The real memory numbers — the table to reuse all track

**This table should be printed once and referenced by number for the rest of the course.** These are the
"numbers the course can reuse verbatim" the brief asked for.

**Inference, 1024×1024, batch 1, bf16:**

| Component | FLUX.1-dev | FLUX.2-dev | SDXL | Z-Image | Notes |
|---|---|---|---|---|---|
| Transformer/U-Net weights | 12B → **24.0 GB** | 32B → **64.0 GB** | 2.6B → **5.2 GB** | 6B → **12.0 GB** | $2$ bytes/param |
| Text encoder(s) | T5-XXL 4.7B + CLIP-L 123M → **9.6 GB** | **Mistral-3 24B → 48.0 GB** | CLIP-L + OpenCLIP-G ≈ **1.6 GB** | Qwen-VL-ish (varies) | |
| VAE | 84M → **0.17 GB** | ~0.2 GB | 84M → **0.17 GB** | ~0.2 GB | negligible |
| Latent $[1,c,128,128]$ | 262k × 2 = **0.5 MB** | 524k × 2 = **1.0 MB** | 65k × 2 = **0.13 MB** | — | **rounding error!** |
| Activations (peak, 4096 tok) | ~2–4 GB | ~4–8 GB | ~1–2 GB | ~1–2 GB | est. |
| **Total, all resident, bf16** | **≈ 34–38 GB** | **≈ 112–120 GB** | **≈ 8 GB** | **≈ 14 GB** | |

> **Look at the latent row: 0.5 MB.** The thing we are actually generating — the image — is **half a
> megabyte**. The weights are **24 gigabytes**, 48,000× larger. **The model is essentially all model.** This is
> a genuinely clarifying observation: diffusion inference is not "processing an image," it's "streaming 24 GB
> of weights past a half-megabyte of state, 28 times." **That reframing directly predicts §15's bandwidth
> analysis** — and predicts why offloading the *weights* is painful and offloading the *latent* is free.

**And now the FLUX.2 problem, which is the learner's problem:**

> **FLUX.2-dev at bf16 needs ~112–120 GB, all resident.** The DGX Spark has **128 GB unified**. It **fits** —
> barely, with the OS and CUDA context eating into it — and only if nothing else is running. **This is the
> single most important number in this track for this specific learner**, and the course should land on it
> hard: *the reason his DGX Spark is interesting is that FLUX.2-dev does not fit on any consumer GPU on Earth
> (a 5090 has 32 GB) and does fit on his desk.*
>
> The standard mitigations, which he should know cold:
> 1. **Precompute text embeddings** → drop the 48 GB Mistral-3 encoder entirely after encoding. **−48 GB.** Biggest single win, costs nothing, and is mandatory for training (§13).
> 2. **`enable_model_cpu_offload()`** → hold one component on GPU at a time. On the Spark this is nearly free because memory is **unified and coherent** — there is no PCIe copy. *This is a genuine Spark advantage over a discrete GPU and worth naming.*
> 3. **FP8 weights** → **−50%** on the transformer (64 → 32 GB). BFL ships FP8 checkpoints.
> 4. **NVFP4** → reported **−55%** vs FP16 (medium confidence). Blackwell (GB10) has **native FP4 tensor cores** — this is exactly what the Spark's "1000 TOPS at FP4" spec is for.

### 8.4 Worked example — carry it all the way

**"How much memory to generate one 1024² image with FLUX.1-dev on the DGX Spark, bf16, and how long?"**

*Memory:*
- Transformer: $12.0\times10^9 \times 2\ \text{B} = 24.0$ GB
- T5-XXL: $4.7\times10^9\times 2 = 9.4$ GB; CLIP-L: $0.123\times10^9\times2 = 0.25$ GB
- VAE: $84\times10^6\times2 = 0.17$ GB
- Latent: $16\times128\times128\times2 = 524{,}288$ B $= 0.5$ MB
- Activations: ~3 GB
- **Total ≈ 36.8 GB.** ✅ Fits in 128 GB with 91 GB to spare. *(Independent check: a published Spark benchmark reports FLUX using **29 GB** — consistent, and lower because the text encoder is freed after encoding. **The arithmetic predicts the measurement.** Show that.)*

*Time:*
- Spark memory bandwidth: **273 GB/s** (LPDDR5X, unified). *(Verified.)*
- The 88 ms/step bandwidth *floor* ($24.0/273$) is real but **NOT the bound** — see the correction below.

> **⚠️ SHOWPIECE INVALID — DELETE IT, do not feature it (D-16 / D-15 / constants §9.6, confirmed this session).**
> Two errors compound here:
> 1. **The benchmark is the WRONG MODEL.** "2.6 s / 1K image, FLUX.1 12B FP4" is **FLUX.1-Schnell, 4 steps, 1024²,
>    batch 1** — NOT FLUX.1-dev at 28 steps. Reading it as dev/28 implies $28\times2\times12\text{e}9\times4096 /
>    2.6\text{s} = \mathbf{1.06\ PFLOP/s}$ = 106% of the *sparse-FP4 marketing peak*, on a workload using neither
>    FP4 nor sparsity — **impossible.** The "prediction" was a coincidence between two unrelated quantities. You
>    cannot honestly predict a 2.6 s wall-clock when the FP4 dense ceiling is itself an unpublished ±2× inference.
> 2. **"bf16 is bandwidth-bound at 88 ms/step" is WRONG (D-15).** Compute needs $2\times12\text{e}9\times4096/62\text{e}12
>    = \mathbf{1585\ ms/step}$ ≫ the 88 ms bandwidth floor. **FLUX at bf16 is COMPUTE-bound, comfortably.** The 88 ms
>    floor is a floor nobody is near.
>
> **Honest replacement = `bench_spark.py`:** the learner runs FLUX.1-dev at 28 steps himself (bf16 and FP4) and
> checks his own prediction against both rooflines — *the prediction he checks is his own.* Label the NVIDIA
> number "FLUX.1-Schnell, 4 steps, 1024², batch 1, FP4" wherever it appears. (For the record, the honest Schnell/4
> FLOP count is 4.98e14 → ~150–191 TF/s = 30–38% MFU — plausible.)

### 8.5 Misconceptions — §8

| Misconception | Truth | Correction |
|---|---|---|
| "Latent diffusion is a memory optimization." | It's a **compute** optimization, and a 65,536× one on attention. Memory falls out. | §8.2's ratio. Memory would be solvable with offloading; $10^{12}$ FLOPs are not. |
| "You could just use a smaller image." | Then you have a smaller image. The point is 1024² **output** at 64² **cost**. The VAE decoder does the upsampling, and it's a *learned, GAN-trained* upsampler, not bilinear. | Compare: latent-diffusion 1024² vs. pixel-diffusion at 64² then `Resize`. Not close. |
| "The VAE is lossy so latent diffusion is worse than pixel diffusion." | **In principle yes; in practice the 4 orders of magnitude bought a model 100× bigger, which more than repaid the 12× compression.** And the field is *buying back* the fidelity: 4→16→32 channels (§2.5). | The honest framing: it's a **trade**, and the trade was worth it. Also note pixel-space diffusion needed 3-model cascades (§8.2), whose seam artifacts were their own fidelity loss. |
| "The latent is where the memory goes." | The latent is **0.5 MB**. The weights are **24 GB**. Ratio 48,000:1. | The §8.3 table. Then: "you are not memory-bound on the image. You are memory-bound on the *model*." |
| "FLUX.2 needs 64 GB because it's 32B params." | 64 GB is *just the transformer*. The **Mistral-3 24B text encoder adds another 48 GB** — the encoder is nearly as big as the generator. **Precompute embeddings and it vanishes.** | The §8.3 table, then mitigation #1. Non-obvious, high practical value, and it's the difference between "doesn't fit" and "fits comfortably." |
| "Higher resolution = proportionally more compute." | **Quadratically** more, via attention. 1024²→2048² is 4× the tokens and **16×** the attention. | The $16^4$ arithmetic of §8.2, run in reverse. |

### 8.6 Dependency — §8

**Requires:** §2.5 (VAE shapes), §7.3–7.4 (what the attention matrix is and how many tokens there are).
**Enables:** §15 (hardware), §13 (training memory), and every practical decision he makes.
This section is mostly *arithmetic*, and that is deliberate: it is the section where the course proves that
"real numbers" wasn't a slogan.

---

## 9. Conditioning: text encoders, cross-attention, and CFG

### 9.1 Intuition first

> **One sentence:** Everything so far builds a machine that makes *some* image from the training distribution;
> conditioning is how you tell it *which*.

Formally, we go from learning $p(x)$ to learning $p(x\mid c)$, where $c$ is the prompt. **And the change to the
loss is one symbol:**

$$
\mathcal{L} = \mathbb{E}\left[\left\|\epsilon - \epsilon_\theta(x_t, t, \mathbf{c})\right\|^2\right]
$$

**That's the entire modification.** Add $c$ as an input. Train on (image, caption) pairs instead of images.
Everything from §3–§6 is untouched — same forward process, same objective, same samplers. **The course should
make this explicit and celebratory:** conditioning is nearly free, mathematically, which is *why* diffusion
took over — you can condition on text, depth maps, poses, other images, class labels, audio, anything, by
concatenating it to the input. The architecture question ("*how* does $c$ get in?") is §7.3's answer:
**cross-attention** (or MMDiT joint attention).

### 9.2 Text encoders — what the two boxes in ComfyUI actually contain

**The pipeline:** `"a red cube"` → tokenizer → token IDs → **frozen** text encoder → $[L, d_c]$ → the $K,V$ of
cross-attention.

> **"Frozen" is the key word and the first misconception.** The text encoder is **not trained** during
> diffusion training (with rare exceptions). CLIP was trained by OpenAI on 400M image-text pairs for a
> *completely different task* (contrastive matching). SD just **borrows the embeddings**. The diffusion model
> learns to *read* a fixed language; it does not learn the language.

**The encoders in play, verified:**

| Model | Text encoder(s) | $L$ (tokens) | $d_c$ | Params | Notes |
|---|---|---|---|---|---|
| SD 1.5 | CLIP ViT-L/14 | **77** | 768 | 123M | the infamous 77-token limit |
| SDXL | CLIP-L **+** OpenCLIP ViT-bigG | 77 | 768+1280 = **2048** (concat) | 123M + 694M | plus a **pooled** vector |
| SD 3.5 | CLIP-L + OpenCLIP-bigG + **T5-XXL** | 77 / 77 / **154+** | 4096 (T5) | 123M + 694M + **4.7B** | T5 optional at inference! |
| FLUX.1 | CLIP-L + **T5-XXL** | 77 / **512** | 4096 | 123M + **4.7B** | |
| **FLUX.2** | **Mistral-3 24B VLM** | long | — | **24B** | *a full VLM, verified* |
| Qwen-Image | **Qwen2.5-VL** | long | — | multi-B | *verified* |
| Z-Image | single-stream, text+semantic+image tokens share one transformer | — | — | (6B total) | *verified* |

**The 2026 trend is unmissable and the course should name it:**

> **The text encoder became an LLM.** SD1.5 used a 123M contrastive encoder that understood roughly a bag of
> concepts. FLUX.2 uses **Mistral-3, a 24B vision-language model** — *twice the size of the entire FLUX.1
> model*, just to read the prompt. **The two destination tracks of this course have converged.**
>
> This is the **strongest possible argument for why the trunk is shared** and why he shouldn't skip the LLM
> track. The architect should place a bidirectional flag here: **the diffusion track's conditioning stage is
> the LLM track's subject matter.** In 2022 they were separate fields. In 2026, FLUX.2's text encoder is a
> Mistral. **Reconcile with the LLM brief.**

**Why 77 tokens?** Because CLIP's context length was 77 and nobody at OpenAI was thinking about image
generation. **An arbitrary number from an unrelated 2021 design decision constrained prompt length for the
entire SD 1.5/SDXL era**, spawning the whole ecosystem of prompt-chunking hacks (`BREAK`, weighted chunk
concatenation, `A1111`'s 75×N scheme) that he has definitely used. Historical accident, permanent consequences.
**This is a good story and a real lesson about how the field actually works.**

**Why did T5 get added alongside CLIP?** CLIP embeddings are trained for *matching*, and they're famously
**bag-of-words-ish** — CLIP genuinely struggles with "a red cube on a blue sphere" vs "a blue cube on a red
sphere" because the contrastive objective never forced it to encode binding. T5 is trained on *language*,
with real syntax. **So SD3/FLUX use CLIP for "vibe" (the pooled vector → adaLN) and T5 for "content"
(the sequence → cross-attention/joint-attention).** That division of labor is a real design and it explains why
SD3.5 lets you drop T5 at inference: you lose prompt *precision*, keep the *style*. He may have seen this
option and wondered. Now he knows.

### 9.3 Classifier-free guidance — derive it, don't assert it

**This is the highest-value section in the track for this learner, because CFG is the knob he has turned the
most and understands the least.**

#### Intuition first

> **One sentence:** Ask the model twice — once with your prompt, once with nothing — and then **walk in the
> direction that leads *away* from the generic answer and *toward* the prompted one, but walk farther than the
> model actually suggested.**

CFG is **extrapolation**, not interpolation. That word is the whole section. Everything about CFG's behavior —
why it helps, why it fries, why $w=1$ means "off," why it costs 2× — follows from *extrapolation*.

#### The derivation, in five lines the learner can follow

Start from Bayes:

$$
p(x\mid c) = \frac{p(c\mid x)\,p(x)}{p(c)}
$$

Take $\log$, then $\nabla_x$ (and **notice $p(c)$ dies — it has no $x$ in it**, the §1.2 trick a second time):

$$
\nabla_x\log p(x\mid c) = \nabla_x\log p(c\mid x) + \nabla_x\log p(x)
$$

Rearrange for the **"implicit classifier"** — the thing that says how much this image looks like the prompt:

$$
\underbrace{\nabla_x\log p(c\mid x)}_{\text{"how prompt-y is it?" gradient}} = \nabla_x\log p(x\mid c) - \nabla_x\log p(x)
$$

> **Look at the right-hand side: it's (conditional score) − (unconditional score).** We have **both** — a
> conditional model $\epsilon_\theta(x_t,t,c)$ and an unconditional one $\epsilon_\theta(x_t,t,\varnothing)$
> (trained by randomly dropping the caption ~10% of the time — **one line in the training loop**). **We
> accidentally have a classifier gradient without ever training a classifier.** *That's the "classifier-free"
> in the name*, and the course should make sure the learner sees why the name means what it means.

Now **overdo it on purpose.** Define a sharpened distribution that amplifies the classifier term by $w$:

$$
\tilde p_w(x\mid c) \;\propto\; p(x)\,\cdot\,p(c\mid x)^{\,w}
$$

Take $\nabla_x\log$:

$$
\nabla_x\log\tilde p_w = \nabla_x\log p(x) + w\big[\nabla_x\log p(x|c) - \nabla_x\log p(x)\big]
$$

Convert scores to $\epsilon$ via §5.2's boxed identity ($s = -\epsilon/\sqrt{1-\bar\alpha_t}$ — **the minus and
the $\sqrt{}$ cancel on both sides**, which the learner should verify):

$$
\boxed{\;\tilde\epsilon_\theta(x_t,t,c) = \epsilon_\theta(x_t,t,\varnothing) + w\cdot\big[\epsilon_\theta(x_t,t,c) - \epsilon_\theta(x_t,t,\varnothing)\big]\;}
$$

**That is the CFG formula, and it is `cfg` in the KSampler node.** Derived, in five lines, from Bayes'
theorem and one substitution. **§5 was worth it. Say so.**

#### Read the formula geometrically — this is what makes it stick

Rewrite it as a point on a line:

$$
\tilde\epsilon = \epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)
$$

This is `lerp(uncond, cond, w)`. **He has written this exact line of code.**

| $w$ | Where you land | Behavior |
|---|---|---|
| $w=0$ | exactly $\epsilon_\varnothing$ | **prompt ignored** — a random training-set-ish image |
| $w=1$ | exactly $\epsilon_c$ | **the model's honest opinion of $p(x\mid c)$.** "CFG off" |
| $0<w<1$ | between them | interpolation — weaker than the prompt |
| $\mathbf{w>1}$ | **past $\epsilon_c$, on the far side** | **EXTRAPOLATION — off the end of the line** |
| $w=7.5$ | 6.5 lengths past $\epsilon_c$ | SD1.5 default |

> **The one sentence that fixes CFG forever:** at $w=7.5$ you are **not** at the conditional distribution. You
> are at a point **six and a half segment-lengths beyond it**, in a direction the model only ever validated at
> $w=1$. **You are asking the model to make the image "more prompt-y than any real image."** No image in the
> training set was ever *that* prompt-y. **CFG generates images from a distribution that does not exist.**
>
> And it *works*, which is one of the most embarrassing facts in generative modeling: **the single most
> important quality knob in the field is a mathematically unjustified overshoot.** Ho & Salimans introduced it
> in a 3-page workshop paper. Nobody has a satisfying theory for why $w\approx7$ beats $w=1$ so decisively.
> **Flag this as genuine, live uncertainty** — see §9.5.

#### Why too high looks "fried" — three mechanisms, all real

The brief specifically asked for this. There are **three** distinct things going wrong and the course should
separate them, because they have different fixes:

**1. Variance/contrast blow-up (the dominant one, and it's arithmetic).**
$\tilde\epsilon$ is an extrapolation, so its **norm grows roughly linearly in $w$**. If
$\epsilon_c - \epsilon_\varnothing$ has std $\delta$ and $\epsilon_\varnothing$ has std ~1, then
$\mathrm{std}(\tilde\epsilon) \approx \sqrt{1 + (w^2-1)\cdot(\text{stuff})}$ — the point is it **exceeds 1**,
and $\epsilon$ is *supposed* to be a standard normal. **You are feeding the sampler an over-scaled noise
estimate**, so $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\hat\epsilon)/\sqrt{\bar\alpha_t}$ **over-subtracts**,
and $\hat x_0$ lands **outside the valid latent range.** Decode that and you get **blown highlights, crushed
blacks, oversaturated color** — **exactly the "fried" look.**

> **The fix confirms the diagnosis.** *CFG rescale* (Lin et al. 2024): compute
> $\tilde\epsilon_{\text{rescaled}} = \phi\cdot\tilde\epsilon\cdot\frac{\mathrm{std}(\epsilon_c)}{\mathrm{std}(\tilde\epsilon)} + (1-\phi)\tilde\epsilon$
> — literally *just renormalize the standard deviation back* — and the frying largely goes away while the
> prompt adherence stays. **A one-line std fix repairs the artifact, which proves the artifact was a std
> problem.** ComfyUI has a `RescaleCFG` node. **He may have used it without knowing it was this.**

**2. Off-manifold extrapolation (the §1.1 payoff).**
The linear extrapolation is in $\epsilon$-space, and the data manifold is **curved**. Walk far enough along a
straight line from a curved surface and **you leave the surface**. The result is high-contrast, high-frequency,
overconfident texture that is locally plausible and globally not-a-photograph. **This is the manifold cartoon
from §1.1, cashed.** Point back at it by name.

**3. Diversity collapse.**
$p(c|x)^w$ with large $w$ is a **sharpened** distribution — it concentrates on the mode. High CFG images all
look the same: same composition, same pose, same lighting. **This is the mode-seeking behavior the course
warned about in §1.3 (GANs), §2.6 (blur), §4.6 (why re-inject noise).** Sixth appearance of "mean vs. sample."

**Real ranges, verified against practice — a table he can act on:**

| Model | Typical `cfg` | Why |
|---|---|---|
| SD 1.5 | **7.0–8.0** | classic; frying starts ~12 |
| SDXL | **5.0–7.0** | lower than SD1.5 — better conditional model needs less overshoot |
| SD 3.5 | **3.5–4.5** | lower still |
| **FLUX.1-dev** | **`guidance` 3.5**, `cfg` **1.0** | ← **see §9.4. This is not the same knob.** |
| **FLUX.1-schnell** | **1.0** | guidance-distilled, 4 steps |
| **FLUX.2-klein-4B** | **`guidance_scale=1.0`, 4 steps** | *verified from the model card* |
| **Z-Image-Turbo** | low, **8 NFE** | distilled |

> **Read that column top to bottom.** **CFG requirements have fallen monotonically as models improved: 7.5 →
> 6 → 4 → 1.** That is not a coincidence and the course should say what it means: **CFG is a crutch for a
> weak conditional model.** A model that truly nailed $p(x|c)$ would need $w=1$. The 2026 frontier is
> approaching that. **The knob he has spent years tuning is being engineered out of existence** — and it's
> being replaced by *distillation*, which bakes the overshoot into the weights. That's a real, current,
> observable trend, visible in one column of numbers.

#### The cost — and why his generations are 2× slower than the step count implies

CFG requires **two forward passes per step** (conditional and unconditional). ComfyUI batches them, so:

- SD1.5, 20 steps, CFG 7.5 → **40 model evaluations.**
- **The "steps" number in the UI is half the truth.**
- This is **exactly why guidance distillation exists** (§7.6, §9.4): bake it in, get 2× for free.

**Worked example, tying §8.4 together:** FLUX.1-dev, 28 steps. If it needed real CFG, that would be 56 passes ×
88 ms = **4.9 s** on the Spark. It's guidance-distilled, so it's **28 passes = 2.5 s**. **Guidance distillation
is worth exactly 2× wall-clock, and the §8.4 arithmetic predicts it.** Beautiful closure — use it.

### 9.4 FLUX's `guidance` is NOT CFG — the confusion that costs people days

> **Warning box. Highest priority in §9. He almost certainly has this wrong.**
>
> In ComfyUI, for FLUX.1-dev, there are **two** things that look like CFG:
> - `KSampler` → **`cfg`** — real CFG. **Set to 1.0.** (i.e. **off**)
> - `FluxGuidance` node → **`guidance`, default 3.5** — ***not CFG.***
>
> **What `guidance` actually is:** FLUX.1-dev was **guidance-distilled**. During distillation, the student was
> trained to *directly output* what the teacher produced *at CFG scale $w$* — and $w$ was fed in **as a
> conditioning input**, embedded exactly like the timestep $t$ (§7.2's sinusoidal embedding + FiLM/adaLN).
>
> So `guidance=3.5` is **a number embedded and injected into the network**, saying *"pretend you're a CFG-7.5-ish
> teacher."* It is **one forward pass.** There is no unconditional branch. There is no extrapolation happening
> at inference. **The extrapolation happened during training and got compiled into the weights.**
>
> **Consequences he can act on immediately:**
> - Setting `cfg=3.5` on FLUX.1-dev **doubles your generation time and makes the image worse.** (You're doing real CFG *on top of* baked-in CFG — double-guidance, and it fries.)
> - `guidance` is **cheap** (it's an input embedding) — 1.0 and 10.0 cost identically.
> - `guidance` has a **different useful range** than `cfg` — roughly 1.5–5 for FLUX.1-dev, and it saturates rather than frying the same way.
> - **FLUX.1-schnell ignores it entirely** (fully distilled).
> - **This is why FLUX LoRAs must be trained with a `guidance_scale` of 1.0 fed to the model** (§13) — a real, gotcha-laden training hyperparameter that comes *directly* from this section.

This single box may be the highest practical-value paragraph in the entire brief for this specific learner.

### 9.5 Genuine uncertainty — CFG

**Flag this honestly, as the brief demands:**

- **Nobody has a satisfying theory of why $w \approx 7$ beats $w=1$.** The $p(x)p(c|x)^w$ story tells you what
  distribution you're *targeting*, but not why that distribution produces better-looking images than the true
  $p(x|c)$. The standard hand-wave — "it trades diversity for fidelity" — is a *description*, not an explanation.
- **Competing accounts, all partially right, none complete:** (a) it corrects for the model's under-fitting of
  the conditional; (b) it's an implicit form of the mode-seeking that human raters prefer; (c) Karras et al.
  (2024) showed **guiding with a *smaller, worse* model instead of the unconditional model works better** —
  "autoguidance" — which strongly suggests CFG's benefit is about **correcting model error**, not about the
  conditional distribution at all. That last result is genuinely destabilizing to the textbook story and the
  course should mention it.
- **Practical corollary:** CFG interacts with the schedule. Guidance intervals (apply CFG only for
  $t\in[t_1,t_2]$, skip it at the extremes) measurably improve quality — which nobody predicted from theory.
- **The 2026 direction:** the field is routing around CFG entirely — via distillation (bake it in) and via
  better conditional models (need less of it). **The `cfg` slider may be a historical artifact within a few
  years.**

> **Tell the learner all of this.** He is turning a knob that the field's best theorists do not fully
> understand, in a direction that provably leaves the data distribution, and it works better. **That is the
> honest state of the art**, and saying so builds far more trust than a tidy fiction. It also models the right
> epistemic stance: *the practice ran ahead of the theory, and it still is.*

### 9.6 Demos — §9

**D9.1 — CFG on a 2-D toy (the flagship; this is where CFG becomes obvious).**
- **Setup:** two known 2-D distributions with **closed-form scores** (reuse D5.1's machinery — zero new math): $p(x)$ = a broad mixture of 5 blobs (the "unconditional" data); $p(x|c)$ = a distribution concentrated on 2 of them (the "prompt").
- **Plot:** left panel = the $\tilde\epsilon$ vector field at the current $w$, drawn as a quiver, with $\epsilon_\varnothing$ and $\epsilon_c$ shown as **two faint arrows and $\tilde\epsilon$ as the bold one at each grid point**. Right panel = 500 sampled points from integrating with that field, overlaid on faint contours of the *true* $p(x|c)$.
- **Control:** a single **$w$ slider, 0 → 20.** Live readouts: `‖ε̃‖ / ‖ε_c‖`, the sample cloud's **std**, and **fraction of samples outside the true support**.
- **JS math (exact, no NN):** closed-form mixture scores for both distributions; $\tilde\epsilon = \epsilon_\varnothing + w(\epsilon_c-\epsilon_\varnothing)$; Euler-integrate the prob-flow ODE.
- **Insights, in the order he'll get them by dragging:**
  - $w=0$: samples spread over **all 5 blobs** — the prompt did nothing.
  - $w=1$: samples match the **true** conditional contours. *Correct, and unimpressive-looking.*
  - $w=4$: samples **tighten inside the 2 blobs**. Sharper. Better!
  - $w=10$: samples **collapse to two tiny dots** — diversity gone — **and the readout shows `‖ε̃‖` at 3×.**
  - $w=20$: samples **shoot past the blobs entirely** into empty space where **no data ever was.** The "outside true support" readout goes to 90%.
- **The insight, stated:** *"That last thing — samples landing where no data exists, with an inflated norm — is what 'fried' means. You just watched it happen with two Gaussians and a slider."* **This demo makes §9.3's three mechanisms visible simultaneously, and it costs almost nothing because the scores are closed-form.** Highest-value demo in §9; arguably top-3 in the track.

**D9.2 — The CFG line.**
- **Plot:** a literal number line / 2-D vector diagram. Two fixed points $\epsilon_\varnothing$ and $\epsilon_c$. A dot at $\tilde\epsilon$.
- **Control:** $w$ slider.
- **Insight:** when $w$ passes 1, **the dot visibly leaves the segment.** A ten-second demo that permanently installs "CFG is extrapolation." Pair it with a real image strip at $w\in\{1,3,7,12,20\}$ so the geometry and the frying are on the same screen.

**D9.3 — CFG rescale.**
- Same as D9.1 plus a $\phi$ slider for the rescale mix, and the `std(ε̃)` readout.
- **Insight:** at $w=12$, drag $\phi$ from 0 → 1 and watch `std` snap back to 1.0 while the samples **stay in the right blobs**. *"The frying was a scale bug, and this is the one-line fix."*

**D9.4 — Cross-attention token budget.**
- Sliders: resolution, encoder choice (77 / 512 / long). Outputs: token count $N$, $[N\times L]$ cross-attn shape, memory in MB, and the MMDiT joint-attention $[N+L]^2$ alternative for comparison.
- **Insight:** switching from CLIP (77) to T5 (512) multiplies cross-attention cost **6.6×** — and going to MMDiT joint attention makes it $(4096+512)^2$ vs $4096\times512$, a **4.4× increase**. **Better prompt understanding is not free**, and here's the invoice. Connects §7.4's "why MMDiT won" to a cost he can see.

### 9.7 Misconceptions — §9

| Misconception | Truth | Correction |
|---|---|---|
| **"CFG makes the model follow the prompt more."** | It makes the model follow the prompt **more than the model itself thinks is realistic.** At $w=1$ you get the model's actual $p(x|c)$. Above 1 you are outside it. | The CFG line, D9.2. "More prompt-y than any real image." |
| **"`cfg` and FLUX's `guidance` are the same knob."** | **They are not.** `cfg` = 2 passes, real extrapolation. `guidance` = 1 pass, a *number fed into the network*. Setting `cfg`>1 on FLUX.1-dev doubles cost and degrades output. | §9.4's warning box. **Highest-priority correction in the section.** |
| "CFG is free / costs one pass." | **2× the model calls.** 20 steps at CFG 7.5 = 40 evaluations. | §9.3's cost note + the §8.4 arithmetic (4.9 s vs 2.5 s). |
| "The negative prompt is where CFG's unconditional branch comes from — so negatives are essential." | The unconditional branch is $\epsilon_\theta(x_t,t,\varnothing)$ — the **empty** prompt. A "negative prompt" **replaces $\varnothing$ with a different conditioning $c^-$**, so you extrapolate *away from $c^-$* instead of away from generic. It's a repurposing of an existing branch, **not a separate mechanism** — which is why negatives cost nothing extra. | Show the formula with $\varnothing \to c^-$. One substitution. Also explains why negatives do nothing on FLUX.1-dev: **there's no second pass to put them in.** |
| "The text encoder is trained with the diffusion model." | **Frozen.** CLIP/T5 were trained by other people, for other tasks. The diffusion model learns to read a language it did not design. | The `requires_grad=False` in every training script. Also: this is *why* textual inversion works (§11.4) — you can't change the encoder, so you add a **word**. |
| "Cross-attention is a detail of the architecture." | **It is the entire text-to-image mechanism.** The $[4096\times512]$ matrix *is* where the prompt becomes a picture. | D7.2's heatmap. Click "red," see the cube light up. |
| "Higher CFG = more detail." | Higher CFG = **higher contrast and saturation**, which reads as "detail" and is not. Real detail comes from steps, resolution, and the VAE. | D9.3: rescale the std back and the "detail" **vanishes** while adherence stays. It was contrast all along. |
| "77 tokens is a meaningful design choice." | It's CLIP's context length from a 2021 paper about image *retrieval*. Historical accident. | §9.2. A good lesson about how the field actually accretes. |

### 9.8 Dependency — §9

**Requires:** §5.2 (**the score identity — CFG cannot be derived without it**; without §5 the CFG formula is
an incantation), §7.3 (cross-attention), §4 (the loss it modifies).
From the trunk: Bayes' theorem, $\log$ of a product, and vector `lerp`.

**Enables:** §11 (LoRA/DreamBooth/TI all manipulate conditioning), §12 (**captioning strategy is a statement
about $c$** — you cannot explain "caption what you want to vary" without knowing what $c$ does), §13
(`guidance_scale` during training).

> **Cross-brief flag,重要:** §9.2's "the text encoder became a 24B LLM" is the **convergence point of the two
> destination tracks.** The architect should consider making this an explicit *joining* moment in the course
> structure — the two tracks are taught as branches, but FLUX.2 proves the branches grew back together.
> **Reconcile with the LLM brief on:** who teaches CLIP (contrastive training is neither track's natural
> home — recommend the trunk), and whether the LLM track's coverage of T5/encoder-decoder is sufficient for
> this track to just point at it.

---

## 10. The 2026 model landscape — verified

**All of this was web-verified on 2026-07-16.** Confidence flags are per-row. The architect should re-verify
before publication — **this section has the shortest shelf life in the course** and should be structured as a
clearly-marked, easily-replaced appendix table, not woven into prose.

> **Course design note:** teach *principles* in §1–§9 and put the *landscape* here, isolated. When FLUX.3
> ships, one table changes and 40 pages don't. The architect should make this an explicit architectural
> decision and mark the section "expected to age."

### 10.1 The open-weights frontier

| Model | Org | Released | Params | Backbone | Text encoder | License | VRAM (bf16) | Confidence |
|---|---|---|---|---|---|---|---|---|
| **FLUX.2 [dev]** | Black Forest Labs | **2025-11-25** | **32B** | rectified-flow transformer | **Mistral-3 24B VLM** | non-commercial (comm. license sold) | ~64 GB + 48 GB enc | **High** |
| **FLUX.2 [klein] 4B** | BFL | **2026-01-15** | **4B** | rectified-flow transformer | (Mistral-3 lineage) | **Apache 2.0** | **~13 GB** | **High** (model card verified) |
| **FLUX.2 [klein] 9B** | BFL | 2026-01-15 | 9B | rectified-flow transformer | — | non-commercial | ~29 GB FP16 | Medium |
| **FLUX.2 VAE** | BFL | 2025-11 | ~0.1B | VAE, **32 latent ch., $f=8$** | — | **Apache 2.0** | — | **High (32ch AND $f=8$ — VERIFIED [VP], shipped config; §2.5)** |
| **Z-Image / Z-Image-Turbo** | Alibaba **Tongyi-MAI** | **2025-11-26** | **6B** | **single-stream DiT** (text+semantic+image tokens share one transformer) | — | **Apache 2.0** | **~16 GB** | **High** |
| **Qwen-Image** | Alibaba Qwen | 2025-08 | **20B** | MMDiT | **Qwen2.5-VL** | **Apache 2.0** | ~40 GB | High |
| **Qwen-Image 2.0** | Alibaba Qwen | **2026-02-10** | **7B** | unified gen+edit, native **2K (2048²)** | Qwen-VL | Apache 2.0 (presumed) | — | **Low–Medium** — single source; **verify before printing** |
| **Stable Diffusion 3.5 Large** | Stability AI | 2024-10 | **8B**, 38 layers | MMDiT | CLIP-L + OpenCLIP-bigG + **T5-XXL 4.7B** | Stability Community | ~16 GB | High |
| **SDXL** | Stability AI | 2023-07 | 2.6B | U-Net | CLIP-L + OpenCLIP-bigG | OpenRAIL++ | ~8 GB | High |
| **SD 1.5** | RunwayML/Stability | 2022 | 860M | U-Net | CLIP-L | CreativeML OpenRAIL-M | ~4 GB | High |
| **HunyuanImage 3.0** | Tencent | 2025 | large | DiT | — | open weights | — | Medium |

**Verified headline facts the course can state:**
1. **Z-Image-Turbo (6B) ranked #1 open-weights** on the Artificial Analysis Image Arena, **above FLUX.2-dev (32B), HunyuanImage 3.0, and Qwen-Image (20B)** — acknowledged by Tongyi Lab publicly. **A 6B model beat a 32B model.** *(High confidence.)*
2. **FLUX.2-dev = 32B rectified-flow transformer + Mistral-3 24B VLM + the FLUX.2 VAE** — verified from BFL's own announcement. **2.7× larger than FLUX.1-dev's 12B.**
3. **FLUX.2-klein-4B is Apache 2.0**, runs in **~13 GB**, and its official example uses **4 steps at guidance_scale 1.0**, sub-second. *(Verified from the HF model card.)*
4. **The open-weights frontier is now genuinely competitive with closed models** on photorealism, text rendering, and editing.
5. **BFL ships FP8 and NVFP4 quantized checkpoints**, reported at −40% / −55% VRAM. *(Medium confidence.)*

### 10.2 What the learner should take from this table

**Four trends, all readable directly off the columns — the course should walk them:**

1. **Everything is a transformer.** No U-Net has been released at the frontier since SDXL (2023). §7.4.
2. **Everything is flow matching / rectified flow.** No DDPM at the frontier since SDXL. §6.
3. **Text encoders became LLMs.** CLIP-123M (2022) → T5-XXL-4.7B (2024) → **Mistral-3-24B / Qwen2.5-VL (2025–26)**. §9.2. **The tracks converged.**
4. **Small + distilled is beating big.** Z-Image 6B at 8 steps > FLUX.2 32B. FLUX.2-klein-4B at 4 steps in 13 GB. **The 2026 story is efficiency, not scale** — which is the *opposite* of the LLM track's story, and **the architect should flag that contrast deliberately.** It's a genuinely interesting divergence between the two destinations and worth a paragraph: *text generation is still scaling; image generation is consolidating.*

> **Recommended course platform choices — and the reasoning, which the learner should see:**
>
> | Purpose | Model | Why |
> |---|---|---|
> | **Learn the mechanics** | **SD 1.5** | 860M, 4 GB, DDPM, U-Net, CLIP-77. Trains a LoRA in **10 minutes**. **Every concept in §3–§7 is visible and every knob still works.** The entire tutorial ecosystem assumes it. |
> | **Do real work** | **FLUX.1-dev** or **SDXL** | The LoRA-tooling sweet spot. FLUX.1-dev = flow matching + MMDiT + guidance distillation = **§6, §7, §9.4 all in one checkpoint he already has.** |
> | **Push the hardware** | **FLUX.2-dev** or **Qwen-Image** | 112–120 GB — **the reason he owns a DGX Spark.** §8.3. |
> | **Feel the 2026 frontier** | **Z-Image-Turbo**, **FLUX.2-klein-4B** | 8 and 4 steps. §7.6. Sub-second. Apache 2.0. |
>
> **Teach on SD1.5, work on FLUX.1, flex on FLUX.2.** That progression maps exactly onto the course's own
> conceptual progression, which is not a coincidence — it's chronological.

### 10.3 Licensing — say it plainly, once

He will train and possibly ship. **This is a real, practical, non-technical thing the course owes him**:

| License | Models | Commercial use? |
|---|---|---|
| **Apache 2.0** | **FLUX.2-klein-4B**, **Z-Image**, **Qwen-Image**, **FLUX.2 VAE** | **Yes, unrestricted.** |
| **FLUX.1/2 [dev] non-commercial** | FLUX.1-dev, FLUX.2-dev, klein-9B | **No** — commercial license purchasable from BFL |
| **FLUX.1 [schnell]** | schnell | **Apache 2.0** |
| **Stability Community** | SD 3.5 | free under a revenue threshold |
| **OpenRAIL++ / CreativeML** | SDXL, SD1.5 | yes, with use restrictions |

> **The gotcha that matters and that people get wrong:** **a LoRA trained on FLUX.1-dev inherits FLUX.1-dev's
> license.** The LoRA is a derivative work — you cannot launder a non-commercial base model by training an
> adapter on it and shipping the adapter. **If he wants to sell what he makes, train on FLUX.2-klein-4B,
> Z-Image, Qwen-Image, or FLUX.1-schnell.** One paragraph, real consequences, and almost no tutorial says it.
> *(Not legal advice; the course should say that too.)*

### 10.4 Misconceptions — §10

| Misconception | Truth | Correction |
|---|---|---|
| "Bigger model = better images." | **Z-Image 6B > FLUX.2-dev 32B** on the open-weights arena. | §10.1 fact #1. A current, verifiable, one-line counterexample. |
| "Open weights lag closed models by years." | As of 2026 the open frontier matches or exceeds closed systems on photorealism, text rendering, and editing. | §10.1 fact #4. |
| "Stable Diffusion is the state of the art." | **SD is not the frontier and hasn't been since ~2024.** It remains the best *ecosystem* (LoRAs, ControlNets, tooling) and the best *teaching platform*. Those are different claims. | Separate "best model" from "best ecosystem" from "best to learn on." All three answers differ. |
| "Open weights = I can use it commercially." | **FLUX.1-dev and FLUX.2-dev are non-commercial.** "Open weights" ≠ "open source" ≠ "free to sell with." | §10.3. |
| "My LoRA is my own work, so its license is mine." | It's a derivative of the base model. | §10.3's gotcha box. |
| "This table is the state of the art." | It is the state of the art **on 2026-07-16** and will be wrong within months. | **The course should say this about itself.** Then point at §1–§9, which won't age. That's an honest and valuable meta-lesson: *learn the invariants, look up the models.* |

---

## 11. Fine-tuning diffusion models — the destination

### 11.0 The framing that organizes the whole section

Every method below answers **one question**: *"The base model knows how to make images. It doesn't know **my
thing**. Where do I put **my thing**?"*

**There are exactly four places to put it**, and once the learner sees this, the zoo becomes a list:

| Where | Method | What changes | Size |
|---|---|---|---|
| **In all the weights** | full fine-tune | $\theta \to \theta'$ | 24 GB |
| **In a low-rank correction to some weights** | **LoRA** | $W \to W + \frac{\alpha}{r}BA$ | **10–200 MB** |
| **In the vocabulary** | textual inversion | one new embedding vector | **~4 KB** |
| **In the input, at inference** | **ControlNet / IP-Adapter** | **nothing** — you add a *signal* | (extra network, but no base change) |

> **That table is the section.** Print it first, then fill it in. And note the last row is a **different kind of
> thing** — it's not fine-tuning at all, which is §11.6's whole point and one of the most persistent confusions
> in the community.
>
> Orthogonal to "where" is **"what"**: DreamBooth is not a *place*, it's a **training recipe** (subject + rare
> token + prior preservation) that is almost always *implemented as* a LoRA. **"DreamBooth vs LoRA" is a
> category error** and half the internet makes it. §11.3.

### 11.1 Full fine-tuning — and why he won't do it

> **Intuition:** Just keep training the model, on your images instead of the internet.

Same loss, same everything:
$$\mathcal{L} = \mathbb{E}_{x_0\sim \mathcal{D}_{\text{yours}},\epsilon,t}\left[\|\epsilon - \epsilon_\theta(x_t,t,c)\|^2\right]$$

**Do the memory arithmetic — this is the argument, and it's decisive:**

**FLUX.1-dev, 12B params, AdamW, bf16 compute with fp32 optimizer states:**

| Item | Formula | Size |
|---|---|---|
| Weights (bf16) | $12\times10^9 \times 2$ | **24 GB** |
| Gradients (bf16) | $12\times10^9 \times 2$ | **24 GB** |
| AdamW moment $m$ (fp32) | $12\times10^9 \times 4$ | **48 GB** |
| AdamW variance $v$ (fp32) | $12\times10^9 \times 4$ | **48 GB** |
| fp32 master weights | $12\times10^9 \times 4$ | **48 GB** |
| Activations (grad ckpt, bs1, 1024²) | — | ~4 GB |
| **TOTAL** | | **≈ 196 GB** |

> **196 GB. The DGX Spark has 128.** ❌ **Full fine-tuning FLUX.1-dev does not fit on his machine.** And
> FLUX.2-dev (32B) would need **~520 GB.**
>
> **The rule of thumb the course should print and have him memorize:** **full fine-tuning with AdamW costs
> ~16 bytes per parameter** (2 weights + 2 grads + 4+4 optimizer + 4 master). **A 12B model needs ~192 GB.
> Multiply params by 16 and you have your answer in bytes.** That one heuristic answers "can I fine-tune X?"
> forever, for both tracks.
>
> *(Mitigations exist — 8-bit Adam (−72 GB), Adafactor, fused backward pass, CPU/NVMe offload, LOMO/gradient-free
> tricks. With aggressive 8-bit + fused-backward + grad-checkpointing it can be squeezed onto 128 GB. **But it
> is a fight, it is slow, and it is the wrong tool** — see below.)*

**And even if it fit, it's the wrong tool. Three reasons:**
1. **Catastrophic forgetting.** 20 images vs. 1 billion. Training all 12B params on 20 images destroys the model's general knowledge in a few hundred steps. You get a model that can only make *your* thing, badly.
2. **Distribution:** a 24 GB file per concept. He'd have 50 of them.
3. **Composability:** you cannot stack two full fine-tunes. You can stack five LoRAs. (He does this in ComfyUI constantly.)

**When full fine-tuning IS right** (be honest, don't just dismiss it): you have **≥ 100k images**, you're
teaching a genuinely new *domain* (medical imaging, a new artistic medium, a new modality), and you have
multi-GPU. **That is not a hobby-scale operation and it is not what he's doing** — but he should know the line
exists and where it is.

### 11.2 LoRA — the dominant approach, derived

#### Intuition first

> **One sentence:** The change you need to make to a giant weight matrix, to teach it one new concept, is
> **not a giant change** — it's a *simple* change, and simple changes are **low-rank**, so store only the
> low-rank part.

**The physical analogy for this learner:** a 4096×4096 matrix has 16.8M knobs. But "make everything you draw
look like *my* dog" is not 16.8M independent facts — it's **one coherent adjustment**, applied everywhere. A
coherent adjustment is exactly what a low-rank matrix *is*: a small number of "directions," each scaled.
**Rank = how many independent ideas the correction contains.**

#### The math

$$
\boxed{\;W' = W + \Delta W = W + \frac{\alpha}{r}\,B\,A\;}
$$

| Symbol | Meaning | Shape | Init |
|---|---|---|---|
| $W$ | frozen pretrained weight | $[d_{\text{out}}, d_{\text{in}}]$ | pretrained, `requires_grad=False` |
| $B$ | "up" projection | $[d_{\text{out}}, r]$ | **zeros** |
| $A$ | "down" projection | $[r, d_{\text{in}}]$ | $\mathcal{N}(0,\sigma^2)$ / Kaiming |
| $r$ | **rank** | scalar | **4–128**, typically **16–32** |
| $\alpha$ | scaling ("`network_alpha`") | scalar | typically $r$ or $r/2$ |

**Why $B$ starts at zero — a detail worth a sentence because it's elegant:** $BA = \mathbf{0}\cdot A = \mathbf{0}$
at step 0, so $W' = W$ **exactly**. **Training starts as a perfect no-op.** The model is unchanged until the
first gradient step. No warm-up shock, no initial degradation. *(Same trick as adaLN-Zero in §7.4 — flag the
repeat.)* And it must be $B=0$ with $A$ random, not both zero, or the gradient is zero forever and nothing ever
trains — a nice little exercise the learner can reason through.

**What $\alpha/r$ is for.** The magnitude of $BA$ scales with $r$. Without the $\alpha/r$ factor, changing $r$
from 16 to 64 would change the *effective learning rate* and you'd have to re-tune everything. The $\alpha/r$
normalization **decouples rank from learning rate**, so you can sweep $r$ without re-tuning `lr`.

> **The practical consequence he needs:** **`network_alpha = r`** ⇒ scale = 1.0 ⇒ "what you trained is what you
> get." **`network_alpha = r/2`** ⇒ scale = 0.5 ⇒ **the LoRA's effect is halved**, which means you need ~2× the
> learning rate for the same result. **This is the single most common LoRA-config confusion**, and it makes
> people think a rank change broke their training when it was really an alpha change.
>
> **And `strength_model` in ComfyUI's `LoraLoader` is yet another multiplier on top:**
> $W' = W + \text{strength}\cdot\frac{\alpha}{r}BA$. So `strength=0.7` does **not** mean "70% trained." It means
> **"scale my learned correction by 0.7"** — a *linear* scaling of a *nonlinear* model's behavior, which is why
> 0.7 is not "70% of the way" to the 1.0 result. It's an extrapolation/interpolation on $\Delta W$, and
> `strength=1.4` (which he has probably tried) is **extrapolating past what you trained** — **exactly the same
> move as CFG (§9.3), with exactly the same failure mode.** *That connection is worth its own box.* Frying a
> LoRA by cranking strength and frying an image by cranking CFG are **the same phenomenon**: linear
> extrapolation off a curved manifold.

#### Why low-rank works — the honest version

**The hypothesis** (Hu et al. 2021, LoRA, originally for LLMs): the weight *update* during adaptation has low
**intrinsic rank**. Not the weights — the **update**.

**The evidence:** it works, everywhere, for everything, at $r=4$–$32$ on models from 100M to 500B. That's
strong empirical evidence.

**The honest caveats — flag these:**
- **It is a hypothesis, and it's approximate.** LoRA does *not* match full fine-tuning quality on hard tasks. For learning a **style** or a **subject**, the gap is invisible. For teaching a genuinely **new capability**, LoRA underperforms. This is well-documented in the LLM literature (and is a point the **LLM brief will also make — reconcile**).
- **The rank you need scales with the complexity of the concept, not the size of the model.** A face: $r=8$–$16$. A style: $r=16$–$32$. A broad domain shift: $r=64$–$128$, and at that point ask whether you should be full-fine-tuning.
- **Higher rank is not "better."** It's more capacity **and more capacity to overfit** on 20 images. $r=128$ on 20 images will memorize them. §12.4.

#### The parameter arithmetic — worked, with real numbers

$$
|\Delta W|_{\text{full}} = d_{\text{out}}\times d_{\text{in}}
\qquad\qquad
|\Delta W|_{\text{LoRA}} = r\,(d_{\text{out}} + d_{\text{in}})
$$

**Worked example — one attention projection in FLUX.1-dev ($d_{\text{model}} = 3072$):**
- Full $W_Q$: $3072\times3072 = \mathbf{9{,}437{,}184}$ params.
- LoRA, $r=16$: $16\times(3072+3072) = \mathbf{98{,}304}$ params.
- **Ratio: $98{,}304 / 9{,}437{,}184 = 1.04\%$.** **96× fewer parameters for that matrix.**

**Scaling to the whole model.** FLUX.1-dev: 19 double-stream + 38 single-stream blocks, $d=3072$. Target
$\{q,k,v,o\}$ in every attention:
- Single-stream-equivalent count ≈ 57 blocks × 4 matrices × 98,304 ≈ **22.4M params**.
- Double-stream blocks have *two* sets (text + image) → add ~19 × 4 × 98,304 ≈ 7.5M → **≈ 30M params.**
- **In bf16: $30\times10^6\times2 = 60$ MB.** ✅ **Consistent with real community FLUX rank-16 LoRAs at ~40–170 MB** (the spread is whether MLP/FFN layers are also targeted, which roughly triples it).
- **As a fraction of 12B: 30M / 12,000M = 0.25%.**

> **Print this:** **you are training one quarter of one percent of the model, and it is enough to teach it your
> dog.** That number is the whole argument for LoRA and it should recur.

**And the memory arithmetic — the payoff:**

**FLUX.1-dev LoRA fine-tune, $r=16$, bs 1, 1024², bf16, AdamW, on the DGX Spark:**

| Item | Formula | Size |
|---|---|---|
| Frozen base weights (bf16) | $12\times10^9\times2$ | 24.0 GB |
| LoRA params (bf16) | $30\times10^6\times2$ | **0.06 GB** |
| LoRA gradients | $30\times10^6\times2$ | **0.06 GB** |
| AdamW states (fp32, $m$+$v$) | $30\times10^6\times8$ | **0.24 GB** |
| Text encoders **(precomputed → freed!)** | — | **0 GB** |
| VAE **(latents precached → freed)** | — | **0 GB** |
| Activations (grad checkpointing) | — | ~4–6 GB |
| **TOTAL** | | **≈ 29–31 GB** |

> **Compare to §11.1's 196 GB.** **A 6.5× reduction, and it fits with 97 GB to spare on the Spark.** The
> optimizer states went from **96 GB to 0.24 GB — a factor of 400** — because AdamW state scales with
> *trainable* params, not total params. **That is where LoRA's memory win actually comes from**, and most
> people think it's the weights. **It isn't — the frozen base weights are still all there, all 24 GB.** Warning
> box. This is a genuinely non-obvious and frequently-botched point.

#### Where to attach LoRA — and it matters

| Target | Effect | Notes |
|---|---|---|
| **`to_q, to_k, to_v, to_out`** (attention) | **the standard.** Attention is where *relationships* live | default in kohya/ai-toolkit/peft |
| **cross-attention only** (`attn2`) | **binds the concept to the trigger word** — pure "this word means this thing" | classic minimal-LoRA; very small files |
| + FFN / MLP | more capacity, ~3× the size | for styles and harder concepts |
| **text encoder too** | lets the *word* itself shift meaning | **often skipped for FLUX** (T5 is huge); common for SD1.5/SDXL |
| **`conv` layers** (LoCon/LyCORIS) | textural/stylistic detail the attention layers can't reach | better for **styles**, worse for subjects |

> **The intuition worth stating:** **cross-attention-only LoRA = "teach the model a new word." Full-attention
> LoRA = "teach the model a new thing." Conv LoRA = "teach the model a new *look*."** That's why style LoRAs
> want conv layers and character LoRAs don't.

#### LoRA variants he'll see in 2026

| Variant | Idea | When |
|---|---|---|
| **LoRA** | $W + \frac{\alpha}{r}BA$ | **the default. Start here. Usually end here.** |
| **DoRA** | decompose into magnitude + direction; LoRA only the direction | modest, real quality gain; slower |
| **LoHa / LoKr** (LyCORIS) | Hadamard / Kronecker products instead of plain low-rank | more expressive per param |
| **LoCon** | LoRA on conv layers too | **styles** |
| **QLoRA** | LoRA on a **4-bit-quantized** frozen base | when the base doesn't fit — **relevant for FLUX.2-dev**, §15 |
| **LoRA+** | different LRs for $A$ and $B$ | small free gain |

**Recommendation for the course: teach plain LoRA properly, name the rest in a table, and say plainly that
plain LoRA at $r=16$ is what 95% of shipped LoRAs are.** Don't let variant-tourism displace understanding.

### 11.3 DreamBooth — a *recipe*, not an architecture

> **The correction first, because this is the confusion:** **"Should I use DreamBooth or LoRA?" is a category
> error.** DreamBooth is *what you train* (a recipe). LoRA is *how you store the change* (a parameterization).
> **Modern practice is "DreamBooth training, LoRA parameterization" — you do both, simultaneously, and that's
> what every FLUX/SDXL character LoRA actually is.** The `train_dreambooth_lora_flux.py` script in diffusers is
> named exactly that for exactly this reason. **Point at the filename.**

> **Intuition:** Pick a word the model barely knows, staple your subject to it, and simultaneously remind the
> model what the *general category* looks like so it doesn't forget.

**Ruiz et al. 2022. Three ingredients:**

**1. A rare identifier token.** Caption images as `"a photo of sks dog"`. Why a nonsense token like `sks`?
Because if you use `"a photo of dog"` you overwrite the model's concept of *dog*. If you use `"a photo of
Rex"`, the model has priors about "Rex" that fight you. **`sks` is chosen for having almost no prior** — it's
a near-empty slot in the embedding space that you fill with your subject. *(Amusing and true: `sks` is a
Finnish assault rifle in some tokenizers' training data, which is a mild prior. "ohwx" and "zwx" are common
alternatives. This is folklore-level engineering and the course should say so.)*

**2. Prior preservation loss — the actual innovation.**

$$
\boxed{\;\mathcal{L}_{\text{DB}} = \underbrace{\mathbb{E}\left[\|\epsilon-\epsilon_\theta(x_t,t,c_{\text{sks}})\|^2\right]}_{\text{learn my dog}} + \lambda\,\underbrace{\mathbb{E}\left[\|\epsilon' - \epsilon_\theta(x'_{t},t,c_{\text{class}})\|^2\right]}_{\text{don't forget what "dog" means}}\;}
$$

- $x'$ — **regularization images**: images of the *generic class* ("dog"), typically **generated by the base model itself** (100–200 of them), *not* real photos.
- $c_{\text{class}}$ — `"a photo of dog"`.
- $\lambda$ — **prior preservation weight, typically 1.0.**

**Why generate the class images from the model rather than use real dog photos?** **This is the subtle and
beautiful part.** The second term says *"whatever you used to predict for 'a photo of dog', keep predicting
that."* It is a **self-distillation / anchor term** — it pins the model to *its own prior behavior*. If you
used real photos, you'd be *training* it on dogs (changing it), not *preserving* it. **The regularization
images are a snapshot of the model's own beliefs, held up as a mirror.** That framing makes $\lambda$ obvious:
it's how hard you hold the mirror.

**What it prevents — two named failure modes he will recognize:**
- **Language drift:** train on 20 photos of your dog with the caption `"sks dog"`, and the token **`dog`** starts meaning *your* dog. Now *every* dog you generate is your dog. **He has seen this in downloaded LoRAs.**
- **Reduced output diversity:** the model collapses to your 20 poses.

**3. Low LR, few steps.** $\text{lr} \approx 1\text{–}5\times10^{-6}$ for full DreamBooth (not LoRA!), ~800–1500 steps.

> **The honest 2026 assessment, which the course should give:** prior preservation is **less necessary with
> LoRA** than it was with full DreamBooth, because LoRA's low rank is *already* a strong regularizer — there
> simply isn't enough capacity to destroy the prior. **Many excellent modern LoRAs skip regularization images
> entirely.** It costs 2× training time (you're training on 2× the images) for a benefit that LoRA partly
> provides for free.
>
> **Recommended guidance:** **skip reg images by default. Add them if you observe language drift** (test:
> generate `"a photo of a dog"` with **no trigger word** — if you get *your* dog, you have drift, and reg
> images are the fix). **That's a diagnostic he can run in 30 seconds**, and it turns a folklore debate into a
> test. **Flag as a genuine live disagreement in the community** — practitioners split on this and both camps
> have good LoRAs.

### 11.4 Textual inversion — the 4 KB approach

> **Intuition:** Don't change the model **at all.** The model already knows how to draw everything. Just find
> the **word** — a point in embedding space, which need not be a real word — that means *your thing*.

**Gal et al. 2022.** You freeze **everything** — U-Net, text encoder, all of it — and optimize a **single new
embedding vector** $v^\star$ that gets inserted into the token stream:

$$
v^\star = \arg\min_{v}\ \mathbb{E}_{x_0,\epsilon,t}\left[\left\|\epsilon - \epsilon_\theta\big(x_t,\,t,\,\Gamma(c \text{ with } \texttt{<my-token>} \mapsto v)\big)\right\|^2\right]
$$

- $v \in \mathbb{R}^{d_c}$ — **that's it. $d_c = 768$ for SD1.5.**
- **768 floats × 4 bytes = 3,072 bytes = 3 KB.**

> **Look at that number.** **You are training 768 numbers.** Not 30 million (LoRA). Not 12 billion (full).
> **768.** And it is enough to reliably summon a specific object. **That fact is worth a full page of the
> course** because of what it *implies*:
>
> **The model already knew how to draw your thing.** It was in there, in the 12 billion parameters, all along —
> a point on the manifold (§1.1) with **no name**. Textual inversion doesn't *teach*; it **finds an address**
> and assigns it a label. **The 3 KB is a pointer, not a payload.**
>
> That is a genuinely profound observation about what these models contain, and it's the strongest available
> argument for the manifold picture the course opened with. **The architect should treat this as the
> intellectual climax of §11**, even though LoRA is the practical answer.

**Multi-vector TI:** use $N$ tokens (`<my-token-1> ... <my-token-N>`), $N\in[1,8]$ → $N\times768$ floats.
More capacity, still tiny.

**Honest assessment — when to use it:**

| | Textual inversion | LoRA |
|---|---|---|
| Size | **3–24 KB** | 40–200 MB |
| Trains in | ~3000 steps / ~20 min | ~1500 steps / ~20 min |
| Can learn a **style** | okay | **better** |
| Can learn a **specific face** | **poorly** — 768 numbers can't encode a face the model doesn't already have | **well** |
| Can learn something **genuinely new** | **no** — it can only point at what exists | **yes** — it changes the function |
| **Composability** | **excellent** — it's just a word, stacks freely | good, but LoRAs interfere |
| Portability across models | **no** — embeddings are encoder-specific | no (rank/shape-specific) |

> **The rule:** **TI can only retrieve; LoRA can teach.** If your concept is *inside* the model but unnamed
> (a specific art style, a vibe, a compositional habit) → TI, and it's magic. If your concept is *outside* the
> model (your specific dog's face) → LoRA. **That one line predicts every empirical result about TI.**
>
> **2026 status: mostly displaced by LoRA**, but still alive for negative embeddings and as a **composable
> style token**. Worth teaching primarily for the *insight* above, not the practice. Say that honestly.

### 11.5 Hypernetworks — teach the concept, note the deprecation

> **Intuition:** Instead of learning the weight change directly, learn a small network that *generates* the
> weight change.

$$
\Delta W = h_\varphi(\text{context})
$$

A small MLP $h_\varphi$ emits (or modulates) the attention $K,V$ projections. Ha et al. 2016; adopted by the
NovelAI-era SD community in 2022.

**2026 status: dead.** LoRA won, decisively, because:
- LoRA's $\Delta W$ is **static** — computed once, **merged into $W$** with zero inference cost. A hypernetwork must **run** at every forward pass.
- LoRA is a linear correction with a clean interpretation (§11.2). Hypernetworks are an opaque function of an arbitrary context.
- LoRA composes; hypernetworks fight.

> **Teach it in one paragraph, mark it deprecated, and use it to make a point:** *this is what a dead branch
> looks like.* In 2022 hypernetworks and LoRA were both plausible. **The field ran the experiment and LoRA won
> on inference cost and composability — engineering properties, not accuracy.** He may still see the
> `Hypernetwork` dropdown in old A1111 installs. **That is a good, honest lesson about how technical choices
> actually get made**, and this learner — who has watched tooling churn — will recognize it.

### 11.6 ControlNet and IP-Adapter — **NOT fine-tuning.** The distinction that confuses everyone

> **The correction, stated as bluntly as possible:**
> **LoRA changes what the model *is*. ControlNet and IP-Adapter change what the model is *told*.**
>
> A LoRA modifies $\theta$. **ControlNet and IP-Adapter modify $c$.** They are **conditioning mechanisms**
> (§9), and they belong in the learner's head next to "the prompt," not next to "the LoRA."

**Sharpen it with a test the learner can apply:**

> **The test:** *After you're done, is the base model different?*
> - **LoRA / DreamBooth / full FT:** **Yes.** $W$ changed. The model now knows something it didn't.
> - **ControlNet / IP-Adapter / T2I-Adapter:** **No.** The base model is bit-identical. You bolted an extra input onto it.
>
> **Corollary he can use immediately:** you **train** a LoRA on *your* images and it takes 20 minutes and 20
> photos. You **download** a ControlNet, and it took someone 100k+ images and a datacenter — but it works on
> *any* image forever, because it's not about content, it's about *a kind of control*. **That's why there are
> 500,000 LoRAs on Civitai and about 15 useful ControlNets.** That ratio *is* the distinction, made
> quantitative, and it explains something he's observed.

#### ControlNet

> **Intuition:** Clone the encoder half of the model, feed the clone a **structural map** (edges/depth/pose),
> and let it whisper spatial corrections into the frozen original at every level.

**Zhang et al. 2023.** Mechanism:
1. **Copy** the U-Net's encoder blocks → a trainable copy. Freeze the original entirely.
2. Feed the copy: $x_t$ **+ the hint** (Canny edges, depth map, OpenPose skeleton, scribble...).
3. Connect the copy's outputs into the frozen U-Net's **decoder** via **zero-initialized convolutions**.

$$
y = \mathcal{F}(x;\Theta) + \mathcal{Z}_2\Big(\mathcal{F}\big(x + \mathcal{Z}_1(\text{hint};\Theta_{z1});\ \Theta_c\big);\ \Theta_{z2}\Big)
$$

where $\mathcal{Z}$ are **zero convs** — $1\times1$ convs initialized to **all zeros**.

> **The zero-conv is the same trick as LoRA's $B=0$ and adaLN-Zero's $\alpha=0$ (§7.4, §11.2). Third
> appearance.** At step 0, the ControlNet contributes **exactly zero** — the model is bit-identical to the
> base. Training starts as a perfect no-op and the control fades *in*. **The architect should collect this:
> "initialize the new thing to zero so the old thing is untouched at step 0" is a design pattern that appears
> three separate times in this track.** Name it once, point at it thrice. That's how a pattern becomes
> intuition.

**Size:** a ControlNet is ~**half a U-Net** — for SD1.5, ~360M params ≈ **1.4 GB**. **That is not a LoRA-scale
object** and its size alone tells you it's a different kind of thing.

#### IP-Adapter

> **Intuition:** "A picture is worth a thousand words" — so let the model take a **picture** as (part of) its
> prompt, by giving cross-attention a second, image-derived $K,V$ stream.

**Ye et al. 2023. The mechanism is *decoupled cross-attention*, and it's elegant:**

$$
\boxed{\;Z = \underbrace{\text{Attn}(Q, K_{\text{text}}, V_{\text{text}})}_{\text{the original, untouched}} + \lambda\cdot\underbrace{\text{Attn}(Q, K_{\text{img}}, V_{\text{img}})}_{\text{new, trainable } W_K', W_V'}\;}
$$

- $K_{\text{img}}, V_{\text{img}}$ come from a **CLIP image encoder**'s embedding of your reference image, through **new** $W_K', W_V'$ — the *only* trained parameters (~22M for SD1.5, ~**100 MB**).
- **$Q$ is shared.** The image and text streams answer the *same question* from the image tokens.
- $\lambda$ = the `weight` slider in ComfyUI.

> **Read the formula.** The text branch is **completely unmodified** — it's literally the original layer. The
> image branch is **added**. That's why IP-Adapter composes with everything (LoRAs, ControlNets, other
> IP-Adapters) and why the `weight` slider behaves so linearly and predictably. **It's a mixing desk, not a
> surgery.** He has used this. Now he knows why it behaves the way it does.

#### The decision table — this is what he actually needs

| I want... | Use | Why |
|---|---|---|
| The model to know **my specific dog** | **LoRA (DreamBooth recipe)** | new content → must change $\theta$ |
| The model to draw in **my art style** | **LoRA** (+ conv layers / LoCon) | new content |
| A **named handle** on a style the model already knows | **Textual inversion** | retrieval, not teaching. 3 KB. |
| The output to **match this pose / depth / edge map** | **ControlNet** | spatial control → new *input* |
| The output to **look like this reference image** | **IP-Adapter** | image-as-prompt → new *input* |
| **This face**, on any body, zero training | **IP-Adapter FaceID / InstantID** | new input |
| **This face**, maximum fidelity, willing to train | **LoRA** | LoRA still wins on identity fidelity |
| A **new domain** (100k+ images) | **full fine-tune** | genuinely new $\theta$ |

> **And the punchline that makes it click:** **you can stack all of them at once**, because they operate on
> different things. `FLUX.1-dev + my-dog-LoRA (θ) + depth-ControlNet (c) + IP-Adapter style ref (c) +
> a text prompt (c)` is a **completely normal** ComfyUI graph, and it works *because* the LoRA edits the
> weights and the other three edit the conditioning. **They don't collide because they're not the same kind of
> object.** When he sees that — that his messy node graph has a clean type signature — the distinction is
> permanent.

### 11.7 Misconceptions — §11

| Misconception | Truth | Correction |
|---|---|---|
| **"DreamBooth vs. LoRA — which is better?"** | **Category error.** DreamBooth = a training *recipe* (rare token + prior preservation). LoRA = a *parameterization*. **You do both at once.** | The filename: `train_dreambooth_lora_flux.py`. Then the §11.0 table: "where" vs "what." |
| **"ControlNet is a kind of fine-tuning."** | It **does not change the base model at all.** It's conditioning. | The test: *"is the base model different afterward?"* ControlNet: no. LoRA: yes. |
| **"LoRA saves memory because you only load part of the model."** | **All 24 GB of frozen base weights are still resident.** The saving is in **gradients + optimizer states** (96 GB → 0.24 GB, **a factor of 400**). | The §11.2 memory table next to §11.1's. The base-weights row is *identical* in both. |
| "Higher rank = better LoRA." | Higher rank = more capacity **and more overfitting** on 20 images. $r=128$ on 20 images memorizes them. | §12.4. Have him train $r=4, 16, 64$ on the same 20 images and compare. The $r=64$ one will be worse. |
| "`network_alpha` is a minor setting." | $\alpha/r$ is a **direct multiplier on your LoRA's effect.** $\alpha = r/2$ **halves it** — you need 2× the LR for the same result. | §11.2's box. The most common "why isn't my LoRA learning?" cause. |
| "`strength_model 0.7` = 70% trained." | It's a **linear scale on $\Delta W$**: $W + 0.7\cdot\frac{\alpha}{r}BA$. And **`strength > 1.0` is extrapolation — the same move as CFG > 1, with the same frying.** | The CFG connection (§9.3). Two knobs, one phenomenon. |
| "Textual inversion is a weak LoRA." | It's a **different thing**: TI **retrieves** what's already in the model (3 KB of pointer); LoRA **adds** what isn't (60 MB of payload). TI cannot learn a genuinely new face **at any size**. | §11.4's rule: *retrieve vs. teach.* |
| "Prior preservation is mandatory for DreamBooth." | It was, for **full** DreamBooth. With **LoRA**, low rank is already a regularizer and many great LoRAs skip it. | The 30-second diagnostic: generate `"a photo of a dog"` with **no trigger**. If it's your dog → drift → add reg images. Otherwise don't. **Live community disagreement — flag it.** |
| "I can sell my LoRA trained on FLUX.1-dev." | Derivative work; inherits the non-commercial license. | §10.3. |
| "IP-Adapter replaces the text prompt." | It **adds a second attention stream**; the text branch is untouched and both are summed with weight $\lambda$. | The decoupled cross-attention formula. The text term is literally unchanged. |

### 11.8 Dependency — §11

**Requires:** §4 (the loss you're minimizing), §7.3 (**you cannot say "LoRA on `to_q,to_k,to_v,to_out`" to
someone who doesn't know what those matrices are** — hard prerequisite), §9 (conditioning, for §11.4 and §11.6),
§8.3 (the memory table this section extends).
From the trunk: **matrix rank and SVD** — see the demo below; **this is likely a gap and it is load-bearing.**

**Enables:** §12, §13, §14.

> **Trunk request — flag for the architect.** §11.2 needs the learner to *feel* that "rank $r$" means "$r$
> independent directions" and that **real matrices are approximately low-rank**. If the trunk covers SVD, this
> track just points at it. If not, **request a 1-page SVD inset in the trunk**, because the LLM brief will need
> exactly the same thing for *its* LoRA section. **This is a shared dependency and should be built once.**

### 11.9 Demos — §11

**D11.1 — The rank slider (must-build; this is what makes LoRA *obvious*).**
- **Setup:** load a real grayscale image as a matrix, e.g. **256×256**.
- **Compute the SVD in JavaScript** (256×256 is small; a Jacobi one-sided SVD is ~80 lines and runs in <1 s — or precompute $U,\Sigma,V$ server-side and ship the factors as JSON, which is cheaper and just as honest).
- **Plot:** left = original. Middle = the **rank-$r$ reconstruction** $\sum_{i=1}^{r}\sigma_i u_i v_i^\top$. Right = the residual at 4× gain. Below = a **log-scale plot of the singular values $\sigma_i$**, with a vertical line at $r$.
- **Control:** **rank slider $r \in [1, 256]$.** Live readouts: **stored numbers** = $r(256+256+1)$ vs. **full** = 65,536; **compression ratio**; **relative error** $\|M - M_r\|_F/\|M\|_F$.
- **Insights, in order, and they're devastating:**
  - $r=1$: garbage — a single outer product, all horizontal/vertical banding.
  - $r=8$: **recognizable.** Storage: $8\times513 = 4{,}104$ vs 65,536 = **6.3%**.
  - $r=32$: **hard to distinguish from the original.** Storage: 16,416 = **25%**, error ~**3%**.
  - Look at the $\sigma_i$ plot: it **falls off a cliff.** The first 20 singular values carry ~90% of the energy; the last 200 carry almost nothing.
- **The insight, stated:** *"Real matrices are approximately low-rank. The information isn't spread evenly across 65,536 numbers — it's concentrated in a few dozen directions. **That is the entire LoRA hypothesis, and you just verified it on a photograph.**"* This demo converts LoRA from a trick into an obvious consequence. **Top-3 demo in the track.**

**D11.2 — Where the memory goes.**
- **Plot:** a live stacked bar chart, GB on the y-axis, with segments {frozen weights, trainable weights, gradients, optimizer states, activations}, against a horizontal red line at **128 GB (DGX Spark)**.
- **Controls:** model radio {SD1.5 860M, SDXL 2.6B, FLUX.1-dev 12B, FLUX.2-dev 32B}; method radio {full FT, LoRA r=16, LoRA r=64, QLoRA-4bit}; optimizer radio {AdamW fp32, AdamW 8-bit, Adafactor}; toggles {grad checkpointing, precompute text embeddings, precompute latents}.
- **JS math (exact, the formulas from §11.1/§11.2):** `frozen = P*bytes_w`; `trainable = T*2`; `grads = T*2`; `opt = T*(8 | 2 | ~4)`; `master = T*4`; with $T$ computed from rank and target modules.
- **Insights:**
  - Switch FLUX.1-dev from LoRA to full FT: the bar **leaps past the red line** (196 GB vs 128). *"That's why you use LoRA."*
  - With full FT selected, watch which segment is huge: **the optimizer states, 96 GB.** Then switch to LoRA and watch that segment **vanish** while the frozen-weights segment **doesn't move at all.** ***That*** **is the LoRA memory story**, and it kills the §11.7 misconception permanently.
  - Toggle "precompute text embeddings" on FLUX.2-dev: **−48 GB, and the bar drops under the line.** He can *see* the mitigation work.
- **This demo is a calculator he will actually use** before every training run. Ship it standalone.

**D11.3 — The four places (conceptual, interactive diagram).**
- A pipeline diagram: `prompt → text encoder → c` and `x_T → [transformer θ] → x_0 → VAE → image`.
- Radio: {LoRA, TI, ControlNet, IP-Adapter, full FT}. Selecting one **highlights in red exactly what changes** and prints the byte count.
- **Insight:** LoRA and full-FT light up **$\theta$**. TI lights up **one token inside $c$**. ControlNet and IP-Adapter light up **$c$ / an extra input path** — and **$\theta$ stays grey**. *"Two of these change the model. Three of these change the message. That's the whole taxonomy."* **This single diagram fixes the §11.6 confusion better than any amount of prose.**

**D11.4 — LoRA merge/strength.**
- Take a tiny 2-layer toy net with a known 2-D output; a real (precomputed) $\Delta W$; sliders for `strength` ∈ [−1, 2].
- **Plot:** the output as strength varies, with the **trained point marked at 1.0**.
- **Insight:** at strength 1.4 you're **past** anything you trained. Compare it side-by-side with **D9.2 (the CFG line)** — **same picture.** *"Over-cranking LoRA strength and over-cranking CFG are the same mistake."*

---

## 12. Datasets for diffusion — where LoRAs actually succeed or fail

> **The thesis of this section, and the architect should open with it:** **90% of LoRA quality is dataset, not
> hyperparameters.** The community obsesses over learning rates and rank because those are *knobs*, and
> obsesses less over captions and image selection because those are *work*. **The knobs are nearly saturated;
> the dataset is where the variance lives.** This learner, who has surely read a hundred contradictory
> hyperparameter recommendations, needs to be told this **first**, plainly.

### 12.1 How few images actually suffice

> **Intuition:** You are not teaching the model to draw. It already draws. You are teaching it **which point on
> a manifold it already knows** (§1.1, §11.4). **That needs examples, not a corpus.**

**Real, actionable numbers** *(these are practitioner-consensus figures; confidence: medium-high — they are
well-corroborated across community sources and my own reading, but they are **folklore, not published
benchmarks**, and the course should label them as such):*

| Goal | Images | Steps | Rank | Notes |
|---|---|---|---|---|
| **A specific face** | **10–20** | 1000–2000 | 8–16 | more is often **worse** — see below |
| A specific object/product | **15–30** | 1500–2500 | 16 | needs varied angles |
| A **character** (face + body + outfit) | **25–50** | 2000–3000 | 16–32 | |
| An **art style** | **50–150** | 2000–4000 | 32–64 | + conv layers (LoCon) |
| A broad **aesthetic** | **200–1000** | 4000–10000 | 64–128 | |
| A new **domain** (medical, satellite) | **10k–100k+** | — | — | **full fine-tune territory** (§11.1) |

> **The headline number: 15–20 images.** State it, then state why it's shocking: **the base model saw
> ~1,000,000,000 images. You are adding 20 — a 0.000002% increase in the data — and it changes the model's
> behavior categorically.** That is only possible because you're not teaching it to see; **you're teaching it a
> name for something it already knows how to draw** (§11.4's insight, cashed for the second time).

> **The counterintuitive one, and it's the most useful sentence in §12: MORE IMAGES IS OFTEN WORSE.**
> 100 photos of a face from the same photoshoot — same lighting, same lens, same expression, same background —
> teaches the model **"this face + that lighting + that background"** as an *inseparable bundle*. **20 photos
> across 20 different contexts** teaches it the **face alone**, because the context varies and averages out
> while the face persists.
>
> **The principle: the model learns whatever is CONSTANT across your dataset.** So **make the thing you want
> constant, and make everything else vary.** That single sentence is the entire theory of dataset construction
> for LoRA, it predicts every rule below, and the course should print it in a box and refer to it by name.
> **Call it "the invariance principle."**

**What to vary, concretely** (a checklist he can use):
lighting (indoor/outdoor/flash/golden hour) · background · distance (close-up / medium / full) · angle (front /
3⁄4 / profile / above) · expression · **camera and focal length** · clothing (unless the outfit *is* the concept).

**What must stay constant:** exactly and only **the concept**.

**Quality gates — non-negotiable:**
- **No duplicates / near-duplicates.** Two copies = that image trained 2×. This is the #1 cause of "my LoRA memorized one photo."
- **No other people/objects in frame** unless you want them.
- **No watermarks, no text overlays** — the model will learn them as part of the concept. (Watermark-bearing LoRAs that emit ghost watermarks are a well-known and hilarious failure.)
- **Consistent-ish subject state.** Training a face on photos spanning 20 years teaches an average face.
- **Sharp.** The model will learn the blur.

### 12.2 Captioning — the most underrated variable

> **Intuition, and it inverts what everyone assumes:** **The caption is not a description. It is a list of
> things you want to be able to CHANGE.**

**The rule, boxed, and it follows directly from the invariance principle (§12.1):**

$$
\boxed{\;\textbf{Caption what you want to VARY. Omit what you want to BAKE IN.}\;}
$$

**Why — and this is where §9's conditioning theory pays off directly.** Training minimizes
$\|\epsilon - \epsilon_\theta(x_t,t,c)\|^2$. Any visual feature **mentioned in $c$** can be explained by $c$ —
the model attributes it to the words and hangs it on that token. Any feature **present in the image but absent
from $c$** must be explained by... **the only other thing that's constant: the trigger token and the LoRA
weights themselves.** So it gets baked in.

> **The mechanism in one sentence:** *unmentioned constants become part of the concept, because there is nowhere
> else for them to go.* This is not a heuristic — **it falls out of the loss.** The course must derive it, not
> assert it, because deriving it makes it memorable and lets him reason about novel cases.

**The worked example that makes it concrete — same 20 photos, two captioning strategies:**

**Strategy A — caption everything:**
> `"a photo of sks man, brown hair, blue shirt, smiling, outdoors, park background, sunny day"`

**Result:** hair/shirt/smile/park are all *explained by the words*. The token `sks` absorbs only what's left:
**the face geometry.** ✅ **You get a flexible face LoRA.** Prompt `sks man in a spacesuit on Mars` and it
works.

**Strategy B — caption minimally:**
> `"a photo of sks man"`

**Result:** the model must explain the brown hair, the blue shirt, the smile, and the park **using `sks`**,
because nothing else is available. **`sks` now means "this guy, in a blue shirt, smiling, in a park."** ❌
Prompt `sks man in a spacesuit on Mars` and you get **a park in the background** and a **blue shirt under the
spacesuit.** *(He has downloaded LoRAs that do exactly this and been annoyed. **Now he knows it was a
captioning bug, not a training bug.**)*

**But now the inversion — and this is why the rule is "caption what you want to vary," not "caption
everything":**

**Training a STYLE.** You *want* the style baked in — it should apply to everything, unconditionally.
So: **caption the *content*, never the style.**
> ✅ `"a woman standing by a window"` (for an image in your style)
> ❌ `"a woman standing by a window, in the style of sks, thick impasto brushstrokes, muted palette"` ← **this
> tells the model the brushstrokes belong to those words, so they'll only appear when you say them, and
> weakly.**

> **Same rule, opposite application.** For a **subject**: describe the context, omit the subject's features
> (so the subject binds to the token). For a **style**: describe the content, omit the style (so the style
> binds to *everything*). **The rule didn't change — what you want to vary changed.** The course should walk
> both examples side by side; the symmetry is what makes the rule stick.

**Trigger word — do you need one?**
- **Subject LoRA: yes.** You need a handle. Use a rare token.
- **Style LoRA: often no.** A style LoRA with **no trigger** and content-only captions applies its style to everything at strength 1.0 — which is usually what you want. **Triggerless style LoRAs are a legitimate and common design**, and people find this surprising.

**Captioning tooling, 2026** *(confidence: medium — verify current versions)*:
- **VLM auto-captioning is the default now**: Florence-2, JoyCaption, InternVL, Qwen2.5-VL, CogVLM. **Then hand-edit.** Auto-captions are a *first draft*, not a product.
- **Natural language for FLUX/SD3/Qwen** (T5/LLM encoders parse syntax). **Tag-soup (booru-style) for SD1.5/SDXL/anime models** (CLIP is bag-of-words-ish anyway — §9.2). **Matching the caption style to the encoder's training distribution matters** and this is a real, mechanistic reason, not a preference.
- **Caption length:** FLUX likes **1–3 sentences**. SD1.5 is capped at **77 tokens** anyway (§9.2) — a long caption is *silently truncated*, which is a classic silent failure.

> **The strongest single piece of advice in §12:** **read your captions.** Ten minutes of hand-editing 20
> captions beats a week of hyperparameter search. **The most common "bad LoRA" cause is a caption that
> mentions the thing you were trying to teach** (or an auto-caption that hallucinated a detail).

### 12.3 Resolution and bucketing

> **Intuition:** The model was trained at a specific resolution and specific aspect ratios. Feed it something
> else and you're off-distribution. But your photos aren't all square. **Bucketing is how you keep the
> aspect ratios without cropping the subject's head off.**

**Native training resolutions:**

| Model | Native | Notes |
|---|---|---|
| SD 1.5 | **512×512** | 768 for SD2.x |
| SDXL | **1024×1024** | trained across a bucket set |
| SD 3.5 / FLUX.1 | **1024×1024** | flexible via `shift` (§6.4) |
| FLUX.2 | up to **4 MP** for editing | |
| Qwen-Image 2.0 | native **2K (2048²)** | *low-medium confidence* |

**Aspect-ratio bucketing** (kohya's `enable_bucket`): instead of center-cropping everything to square (which
**decapitates portraits** — a real and common problem), sort images into **buckets** of similar aspect ratio,
each with **approximately constant total pixel count**, and batch within a bucket.

**A real SDXL bucket set** (each ≈ 1,048,576 px = 1024²):

| Bucket | Aspect | Pixels | Latent (÷8) | Tokens (÷2 patch) |
|---|---|---|---|---|
| 1024×1024 | 1:1 | 1,048,576 | 128×128 | 4096 |
| 1152×896 | 1.29:1 | 1,032,192 | 144×112 | 4032 |
| 896×1152 | 1:1.29 | 1,032,192 | 112×144 | 4032 |
| 1216×832 | 1.46:1 | 1,011,712 | 152×104 | 3952 |
| 832×1216 | 1:1.46 | 1,011,712 | 104×152 | 3952 |
| 1344×768 | 1.75:1 | 1,032,192 | 168×96 | 4032 |
| 1536×640 | 2.4:1 | 983,040 | 192×80 | 3840 |

**Why constant pixel count?** So every bucket produces a **similar token count** (3840–4096 above) → **similar
memory and similar step time**. **Buckets are a batching constraint, not an aesthetic one** — you cannot batch
tensors of different shapes, so you group by shape. Note the divisibility: **every dimension must be divisible
by 64** (= VAE's $f{=}8$ × patchify's 2 × a safety factor of 4). **That's why the numbers look weird — they're
$8\times2\times$ integers, not arbitrary.** He has seen `1216×832` in a hundred configs and wondered. It's
arithmetic.

**Settings that matter:** `enable_bucket`, `min_bucket_reso` (256), `max_bucket_reso` (2048),
`bucket_reso_steps` (**64**), `bucket_no_upscale` (**true** — don't upscale small images into big buckets; you're
just training on blur).

> **Practical rule:** **train at the resolution you'll generate at.** A 512-trained LoRA applied to a
> 1024-generation is a distribution mismatch and looks soft. **This is a train/test-mismatch argument** — the
> same species of argument as §3.5 and §6.4's `shift`. Third appearance; name the pattern.

### 12.4 Overfitting — the signs, the causes, the fixes

> **Intuition (§1.1's manifold, final appearance):** You wanted the model to learn a **patch of manifold** —
> "the region of image-space containing my dog, from any angle, in any light." Overfitting means it learned
> **20 isolated points** — your 20 exact photos — and nothing in between.

**The signs, in the order they appear** — this should be a **diagnostic checklist**, because he will need it:

| Sign | What you observe | Underlying cause |
|---|---|---|
| **1. Background bleed** | The training set's background shows up **everywhere**, even in "on Mars" | **Captioning** (§12.2, Strategy B). The background got bound to the token. |
| **2. Pose lock** | Every output has the same head angle / stance | Not enough pose variety (§12.1) |
| **3. Prompt adherence collapse** | The LoRA **ignores the rest of the prompt**. `sks man riding a horse` → a portrait, no horse | **Over-trained.** $\Delta W$ is so large it dominates the base model's response to everything |
| **4. Contrast burn** | Outputs look **fried** even at CFG 5 | $\Delta W$ magnitude too large → over-scaled $\epsilon$ → §9.3's mechanism #1 **again** |
| **5. Verbatim memorization** | You can **regenerate a training image** nearly exactly, from the trigger alone | Terminal. Too many steps, too high rank, too few / duplicated images |
| **6. Flexibility loss at low strength** | Needs `strength 0.5` to be usable; at 1.0 it's unusable | Over-trained; you're manually undoing it |

**The definitive test — and it's 60 seconds:**

> **The out-of-distribution probe.** Generate: **`sks man as an astronaut on the surface of Mars, full body`**
> — something **certainly not in your dataset**.
> - ✅ **Healthy:** you get your guy, in a spacesuit, on Mars. **The concept generalized.**
> - ⚠️ **Background bleed:** your guy, spacesuit, and *your living room* somehow.
> - ❌ **Overfit:** a **head-and-shoulders portrait**, indoors, no spacesuit, no Mars. The LoRA ate the prompt.
>
> **Run this at every checkpoint.** It is worth more than the loss curve. Which brings us to:

> **⚠️ Warning box, and this is important: the training loss curve is nearly useless for diffusion LoRA.**
>
> **Why**, mechanistically: the loss is $\mathbb{E}_{t}[\|\epsilon - \epsilon_\theta\|^2]$ with $t$ **sampled
> randomly every step** (§4.5). The variance *across timesteps* is enormous — the loss at $t=900$ and $t=100$
> differ by orders of magnitude (§3.4's SNR table). **So the loss you plot is dominated by which $t$'s you
> happened to draw**, not by whether the model improved. It is a **noise-dominated signal.** A diffusion LoRA
> loss curve is essentially flat with huge variance from step 50 to step 5000, and it looks the same whether
> you're learning beautifully or memorizing catastrophically.
>
> **This confuses everyone**, because in every other area of ML the loss curve is the primary instrument.
> **Here it is not.**
>
> **What to do instead: save checkpoints every 250 steps and generate a fixed prompt grid at a fixed seed from
> each.** Look at them. **Your eyes are the metric.** (§14's `eval_lora.py`, artifact D11.) This connects
> directly to §14's "human eval still rules" — **and it is the same reason.**

**The fixes, ranked by effectiveness:**
1. **Fix the captions.** (Sign 1 → almost always this.) Free, and the highest-leverage action available.
2. **Train fewer steps.** Use an earlier checkpoint. (Signs 3, 4, 5, 6.)
3. **Lower the rank.** $r=32 \to r=8$. Less capacity to memorize.
4. **Add image variety.** (Sign 2.)
5. **Lower the LR.** $1\times10^{-4} \to 5\times10^{-5}$.
6. **Add regularization images.** (§11.3 — but check for drift first with the diagnostic.)
7. Lower `strength` at inference. ← **This is a bandage, not a fix.** If you need 0.5, retrain.

> **The meta-point the course should make:** **overfitting in diffusion LoRA is almost always a *data* problem
> wearing a *hyperparameter* costume.** Sign 1 is captions. Sign 2 is image selection. Signs 3–6 are
> "too much training on too little variety." **Only one of the seven fixes above is a hyperparameter, and it's
> ranked fifth.**

### 12.5 Misconceptions — §12

| Misconception | Truth | Correction |
|---|---|---|
| **"More training images = better."** | **Often worse.** 100 photos from one shoot teaches "face + that lighting + that room" as a bundle. 20 varied photos teach the face. | **The invariance principle**: the model learns what's **constant**. Make only the concept constant. |
| **"Captions should describe the image accurately."** | Captions should list what you want to **vary**. Accurate-but-complete captions are right for *subjects* and **wrong for styles**. | The Strategy A/B worked example + the style inversion. Derive it from the loss (§12.2). |
| "Caption the style so the model learns the style." | **Backwards.** Naming the style binds it to those words → it only appears when you say them. **Omit** the style and it binds to the weights → it applies always. | The style example. The symmetry with the subject case is what makes it click. |
| **"Watch the loss curve to know when to stop."** | **The loss is dominated by the random $t$ draw** and is essentially uninformative. | The warning box in §12.4. Then: generate a fixed prompt grid every 250 steps. **Eyes, not curves.** |
| "Overfitting means memorizing images." | That's the *terminal* stage. It starts as **background bleed** and **prompt-adherence collapse** — both visible long before verbatim memorization. | The 6-sign diagnostic table + the Mars probe. |
| "I'll fix it with `strength 0.6` at inference." | You're manually undoing over-training. It **compresses your usable range** and degrades quality. | Retrain. Strength is for *blending*, not *correcting*. |
| "Bucketing is an aesthetic feature." | It's a **batching constraint** — you can't batch different tensor shapes, so you group by shape at constant pixel count for constant step time. | The bucket table. The 64-divisibility is $f{=}8\times$patch$2\times4$. **It's arithmetic, not taste.** |
| "Regularization images should be real photos of the class." | They should be **generated by the base model** — they're a snapshot of *its own prior*, held up as an anchor. Real photos would *train* it, not *preserve* it. | §11.3. Genuinely subtle and universally botched. |
| "512-trained LoRAs work fine at 1024." | Train/test mismatch — soft, doubled, or off-composition. **Train at your generation resolution.** | Same argument species as §3.5 and §6.4's `shift`. Third appearance — name the pattern. |

### 12.6 Dependency — §12

**Requires:** §9 (**conditioning — §12.2's captioning rule is derived from what $c$ does in the loss; without §9
it's folklore**), §4 (the loss), §11 (what you're training), §2.5 (the $f{=}8$ that makes buckets divisible by 64).

**Enables:** §13, §14.

> **Architect's note:** §12 is the section with the **worst signal-to-noise ratio in the public literature.**
> Almost everything online is folklore, contradictory, and undated. **The course's differentiator here is
> deriving the rules from §4 and §9 instead of listing them.** The invariance principle and the
> caption-what-you-vary rule are both *consequences of the loss function*, and presenting them that way is what
> makes this course worth reading over a Reddit thread. **Protect that.** Where I've given folklore numbers
> (§12.1's table), the course should **label them as practitioner consensus, not measurement** — and, ideally,
> the course should ship an experiment (§14, D12) that lets him **measure them himself** on SD1.5 in an hour.
> That would be genuinely novel and it is cheap to build.

---

## 13. Tooling and hyperparameters — verified 2026

### 13.1 The training-tool landscape

**Verified as of 2026-07.** *(Confidence: medium-high — this ecosystem churns fast and my sources are a mix of
primary repos and dated secondary guides. **The architect must re-verify before publication.**)*

| Tool | What it is | Best for | Status 2026 |
|---|---|---|---|
| **kohya_ss / sd-scripts** | The foundational LoRA training scripts. **"Almost every other local image-LoRA tool wraps these."** | **SDXL, SD1.5**; the reference implementation; best generalization across diverse prompts | **The bedrock.** v0.9.0 (Jan 2025) added the **fused backward pass** — the single most impactful VRAM optimization for consumer training |
| **ai-toolkit** (Ostris) | Modern trainer, clean configs | **The primary tool for FLUX.2, Z-Image, and Qwen-Image training.** Reliable balanced results | **The 2026 default for new models.** |
| **OneTrainer** | Kohya fork + all-in-one GUI | **Photographic realism**; easier than raw sd-scripts | Strong, actively developed |
| **SimpleTuner** | HF-ecosystem trainer | Large-scale, multi-GPU, research | Trains FLUX in **~20 GB** with Optimum-Quanto |
| **FluxGym** | Web UI wrapping kohya | Beginners on **FLUX.1**; explicit 12/16/20 GB VRAM presets | Alive; a wrapper, not an engine |
| **diffusers** `train_*` scripts | HF reference implementations | **Learning / reading the code** | **v0.39.0, released 2026-07-03** *(verified)* |
| **ComfyUI native LoRA training** | Built-in node | Quick iteration without leaving Comfy | **Exists** but sparsely documented *(medium confidence)* |
| **ComfyUI-Realtime-Lora** (community) | Train/edit/block-merge LoRAs in-graph | **SDXL, SD1.5, FLUX, Z-Image, Qwen-Image, Wan 2.2**; SD1.5 in **<2 min** | Notable 2026 development |

> **ComfyUI's role — state it precisely, because he'll ask.** ComfyUI is an **inference and workflow engine**.
> Training in it is possible in 2026 (native node + community nodes) and is genuinely convenient for fast
> iteration, but **the serious training tools are separate**. **Recommended workflow for him:**
> **train in ai-toolkit or kohya → the output is a `.safetensors` LoRA → drop it in `ComfyUI/models/loras/` →
> `LoraLoader`.** The handoff is a **file**, and that's the whole integration. The course should show that
> handoff explicitly (§14, D13) because "how does my trained thing get back into the tool I use?" is a real
> question that tutorials skip.

> **Recommended course choice:** **`diffusers` for the from-scratch teaching code** (readable, canonical, and
> the learner can *read the loss*), **`ai-toolkit` or `kohya_ss` for the real training artifact** (because
> that's what he'll actually use). **Show both. Explain that they implement the same six lines from §4.5.**
> That's the moment the course's theory and his practice fuse.

### 13.2 The hyperparameters that matter — real ranges

**LoRA hyperparameters, by model.** *(Confidence: medium-high. These are practitioner-consensus ranges,
corroborated across tools and sources. Label as such; they are not benchmarks.)*

| Parameter | SD 1.5 | SDXL | **FLUX.1-dev** | Notes |
|---|---|---|---|---|
| `learning_rate` (UNet/DiT) | **1e-4** | **1e-4** | **1e-4** | remarkably stable across models |
| `learning_rate` (text encoder) | 5e-5 | 5e-5 | — | **usually off for FLUX** (T5 is 4.7B) |
| `network_dim` (rank $r$) | 8–32 | **16–32** | **16** | §11.2 |
| `network_alpha` | **= $r$** | = $r$ | = $r$ | **$\alpha=r/2$ halves your effect** (§11.2) |
| `batch_size` | 1–4 | 1–2 | **1** | grad accumulation instead |
| `gradient_accumulation_steps` | 1 | 1–4 | **4** | effective batch = bs × accum |
| `max_train_steps` | 1000–2000 | 1500–3000 | **2000–4000** | |
| `optimizer` | **AdamW8bit** | AdamW8bit | **AdamW8bit** / Adafactor | 8-bit: **−72 GB** on a 12B full FT |
| `lr_scheduler` | cosine / constant | cosine_with_restarts | **constant** | FLUX practice favors constant |
| `lr_warmup_steps` | 0–100 | 100 | 100 | |
| `resolution` | **512** | **1024** | **1024** | §12.3 |
| `mixed_precision` | fp16 | **bf16** | **bf16** | bf16 on Blackwell |
| `gradient_checkpointing` | optional | **true** | **true** | trades ~30% speed for big VRAM savings |
| `noise_offset` | 0.05–0.1 | 0.05 | — | **a hack for §3.4's terminal-SNR bug!** |
| `clip_skip` | 2 (anime) / 1 | 1 | — | SD1.5-era |
| **`timestep_sampling`** | — | — | **`flux_shift`** / `sigmoid` | **§6.4** |
| **`model_prediction_type`** | `epsilon` | `epsilon` | **`raw`** (velocity) | **§4.3 / §6.3** |
| **`guidance_scale`** (training) | — | — | **1.0** | **§9.4 — critical gotcha** |
| `discrete_flow_shift` | — | — | **3.0–3.2** | **§6.4's `shift`** |

> **Look at the last four rows.** `timestep_sampling=flux_shift`, `model_prediction_type=raw`,
> `guidance_scale=1.0`, `discrete_flow_shift=3.0`. **A learner who has not read §4.3, §6.4, and §9.4 is typing
> magic words into a config file.** A learner who *has* read them knows: *"I'm using logit-normal timestep
> sampling with a resolution shift of 3.0, training on the velocity target, and feeding guidance=1.0 because
> FLUX.1-dev is guidance-distilled and I must not double-guide it."*
>
> **That sentence is the entire course, in one config file.** **The architect should build toward exactly this
> moment** — put an annotated `config.toml` at the end of the track where **every single line links back to a
> section number.** That artifact would be the single most valuable page in the course, and it makes the
> track's promise ("theory under the knobs you already turn") literal and checkable.

**Adaptive optimizers — worth a note:** **Prodigy** and **DAdaptAdam** estimate the LR themselves; set
`learning_rate = 1.0` (yes, 1.0 — it's a multiplier on the estimate) and `d_coef = 1.0`. Popular for people who
don't want to tune. **Cost: more VRAM and ~20% slower.** Legitimate; not magic.

**Other approaches' hyperparameters:**

| Method | Key params |
|---|---|
| **Full DreamBooth** (not LoRA) | `lr` **1e-6 – 5e-6** (100× lower than LoRA!), 800–1500 steps, `prior_loss_weight` **1.0**, 100–200 class images |
| **Textual inversion** | `lr` **5e-4 – 5e-3** (much higher — it's one vector), **~3000 steps**, `num_vectors` 1–8 |
| **LoRA (above)** | `lr` **1e-4** |

> **Note the LR spread: 1e-6 → 1e-4 → 5e-3. Three orders of magnitude across three methods.** Why? **Fewer
> trainable parameters ⇒ higher LR.** Full FT touches 12B params and must tiptoe. TI touches 768 and can
> stride. **That's a real, transferable principle** (and one **the LLM brief will need too — reconcile**), not
> a lookup table.

### 13.3 Dependency — §13

**Requires:** §11 (what the parameters parameterize), §12 (dataset), and §4/§6/§9 for the last four rows of
the table.
**Enables:** §14, §15.

> **Architect's warning:** §13 is the **fastest-aging section in the track** after §10. **Structure it as a
> replaceable appendix** with the *reasoning* (why LR scales inversely with trainable params; why alpha/r
> matters; why guidance=1.0) in the durable body and the *numbers* in the table. When ai-toolkit renames a
> flag, one cell changes.

---

## 14. Evaluation — and why his eyes are still the instrument

> **Note on cross-references:** earlier sections cite runnable code as "artifact D1…D13." Those live in
> **§16**. This section (§14) is what §12.4 refers to as "human eval still rules."

### 14.1 Intuition first

> **One sentence:** You are trying to put a single number on "does this look good and match the prompt," which
> is a question about **human perception and human intent** — so every automated metric is a *proxy*, and every
> proxy is gameable.

**The structural problem, stated once:** for a classifier you have labels, so accuracy is *the* answer. For a
generative model **there is no ground truth**. There is no correct image for "a cat astronaut." You are
comparing **two distributions**, one of which you only have samples from and the other of which is *in
someone's head*. **That is a fundamentally harder measurement problem**, and the course should say so plainly
rather than presenting a metrics menu as if one of them were the answer.

### 14.2 FID — the standard, and its well-known limits

**Fréchet Inception Distance** (Heusel et al. 2017). Still the most-reported number in the literature.

**How it works:**
1. Push $N$ real images and $N$ generated images through **Inception-V3**, pretrained on ImageNet.
2. Take the **pool3** layer activations: a **2048-dimensional** vector per image.
3. **Assume both sets are Gaussian** in that 2048-D space. Fit $(\mu_r,\Sigma_r)$ and $(\mu_g,\Sigma_g)$.
4. Compute the **Fréchet (Wasserstein-2) distance between the two Gaussians**:

$$
\boxed{\;\text{FID} = \underbrace{\|\mu_r - \mu_g\|_2^2}_{\text{means differ}} + \underbrace{\operatorname{Tr}\!\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r\Sigma_g\right)^{1/2}\right)}_{\text{covariances differ}}\;}
$$

- $\mu_r,\mu_g \in \mathbb{R}^{2048}$ — mean feature vectors.
- $\Sigma_r,\Sigma_g \in \mathbb{R}^{2048\times2048}$ — feature covariances (**4.2M entries each**).
- **Lower is better.** **Units: none** — it's an arbitrary scale.

**Typical values** *(confidence: medium — FID values are notoriously implementation-dependent; see limitation 5)*:
COCO-30k zero-shot FID ≈ **20–30** for early SD, high single digits to low teens for strong modern models,
~**2–3** on ImageNet-256 class-conditional for the best models. **Do not let the course quote cross-paper FIDs
as if comparable** — see below.

**The limits — and the course must be blunt, because FID is quoted everywhere as if it meant something:**

1. **The Gaussian assumption is false.** Image features in 2048-D are **not** Gaussian, not remotely. FID fits two Gaussians to two non-Gaussian clouds and measures the distance between the *fits*. Two genuinely different distributions can share $(\mu,\Sigma)$.
2. **It's biased at small $N$.** FID is a **biased estimator** and the bias depends on $N$. FID@1k and FID@50k are **different quantities**. You need **≥ 10k, ideally 50k** samples for stability. **This makes FID structurally useless for evaluating a 20-image LoRA** — you'd need 50,000 generations to get a number about 20 photos.
3. **Inception-V3 is an ImageNet classifier from 2015.** Its features are tuned to discriminate 1000 ImageNet classes — dogs, mushrooms, trucks. **It has no features for "is the text spelled correctly," "do the hands have five fingers," or "is the lighting coherent"** — the exact failure modes people actually care about in 2026. **FID is blind to the things that distinguish modern models.**
4. **It conflates fidelity and diversity.** A model that produces beautiful images of only one thing, and a model that produces mediocre images of everything, can post the same FID. You cannot tell which failure you have. *(Precision/Recall for generative models, and Density/Coverage, were invented to split these apart, and the course should name them as the fix.)*
5. **It is implementation-sensitive to an embarrassing degree.** Resize interpolation (bilinear vs. bicubic vs. Lanczos), JPEG compression, the exact Inception weights (TF vs. PyTorch ports differ!), and float precision **all move FID by amounts comparable to real model differences.** Parmar et al. (2022, "On Aliased Resizing and Surprising Subtleties in GAN Evaluation") documented this and it is a genuine scandal. **Cross-paper FID comparisons are frequently meaningless**, and everyone knows, and everyone keeps doing it.
6. **It correlates poorly with human judgment for modern models.** This is the killer. In the GAN era FID tracked quality usefully. By 2026 the models are good enough that FID differences are dominated by *style and dataset-matching*, not quality. **A model can have worse FID and be obviously better to every human who looks at it.**

> **The honest summary the course should give:** **FID is a legacy metric that the field continues to report
> out of convention and reviewer expectation, and that essentially nobody makes decisions with anymore.** It
> is still worth teaching because (a) he will see it in every paper and must know what it does and doesn't
> say, and (b) **its failure modes are a superb lesson in what "measuring a distribution" actually requires**.
> *(This is a real disagreement in the field — plenty of researchers still defend FID as a coarse sanity check,
> and that's a fair position. What's *not* defensible is treating a 0.5 FID delta as a result.)*

**The 2026 replacement worth naming: CMMD** (Jayasumana et al., CVPR 2024) — **CLIP** features instead of
Inception, and **MMD** (Maximum Mean Discrepancy) instead of a Gaussian fit. This fixes limitations 1, 2, and 3
simultaneously: MMD is **unbiased** and works at **small $N$**, and CLIP features actually encode modern
semantics. It has not displaced FID socially, but it is the technically correct choice.
*(Confidence: medium-high on the method; the course should verify adoption status.)*

### 14.3 CLIP score — measures alignment, not quality

$$
\text{CLIPScore} = \max\!\left(0,\ 100\cdot\cos\!\big(E_{\text{img}}(x),\ E_{\text{text}}(c)\big)\right)
$$

- $E_{\text{img}}, E_{\text{text}}$ — CLIP's image and text encoders → a shared embedding space.
- $\cos(a,b) = \frac{a\cdot b}{\|a\|\|b\|}$ — **plain cosine similarity**, which he can compute by hand.
- Typical range for good text-to-image: **~0.25–0.35** raw cosine (≈ 25–35 on the ×100 scale).

**What it measures:** *"does the image match the prompt?"* — **alignment only.**
**What it does NOT measure:** quality, realism, aesthetics, anatomy, or coherence.

**Limits:**
- **It saturates.** Above ~0.32, differences are noise. Modern models are all saturated. **A metric that everyone maxes out is not a metric.**
- **It uses CLIP, so it inherits CLIP's bag-of-words weakness** (§9.2). **CLIPScore cannot reliably tell "a red cube on a blue sphere" from "a blue cube on a red sphere"** — *precisely the failure it's supposed to catch.* **The judge has the same blind spot as the defendant.** That's a beautiful and damning point and the course should make it.
- **Gameable:** you can raise CLIP score by cranking CFG, at the cost of frying the image (§9.3). **The metric rewards the artifact.**

> **The classic and instructive result:** plot **CLIP score vs. FID** as you sweep the CFG scale. You get a
> **Pareto curve** — CFG↑ improves CLIP score (alignment) and worsens FID (realism/diversity). **The two
> standard metrics directly trade off against each other, and CFG is the knob that moves you along the curve.**
> That single plot teaches more about both metrics — and about CFG — than any definition. **Ship it as a demo
> (D14.1).**

**Better alternatives, 2026** *(confidence: medium — verify current SOTA)*:

| Metric | What it does | Why better |
|---|---|---|
| **HPSv2 / HPSv3**, **PickScore**, **ImageReward** | reward models **trained on human preference pairs** | They predict what humans *actually chose*. Directly optimizes the real target. |
| **GenEval**, **DPG-Bench**, **T2I-CompBench** | **compositional** benchmarks: object counting, spatial relations, attribute binding | Uses an object **detector** to *check* "two cats on the left" — a **verifiable** claim, not a vibe |
| **VQAScore** | ask a **VLM**: "does this image show X?" | 2026's best automated alignment proxy; a VLM can parse syntax where CLIP can't |
| **Artificial Analysis Image Arena** | **human ELO**, pairwise | **The 2026 de facto leaderboard.** This is what ranked Z-Image #1 (§10.1). |

> **Note the trajectory: the field replaced its metrics with (a) models trained on human preferences and
> (b) actual humans voting.** That's the honest arc, and it should be stated.

### 14.4 Evaluating a LoRA — none of the above works

**This is what he actually needs, and the metrics literature has nothing for him.**

**Why the standard metrics fail here, specifically:**
- **FID needs 10k–50k samples** to compare **distributions**. You have **20 images** and you care about **a concept**, not a distribution. Structurally inapplicable (limitation 2).
- **CLIP score** cannot evaluate `sks` — **it's a nonsense token to CLIP** (that's the whole point of choosing it, §11.3). The metric literally cannot see your trigger word.
- **The training loss is noise** (§12.4's warning box).

**What actually works — the protocol the course should teach and ship as code:**

1. **A fixed prompt grid at fixed seeds.** Build **8–12 prompts** before you start training, covering:
   - **in-distribution:** `"a photo of sks man"`
   - **pose/context change:** `"sks man riding a bicycle"`
   - **strong OOD (the Mars probe, §12.4):** `"sks man as an astronaut on Mars, full body"`
   - **style transfer:** `"an oil painting of sks man"`
   - **the drift control (no trigger!):** `"a photo of a man"` ← **detects language drift** (§11.3)
   - **the prior control:** `"a photo of a dog"` ← detects general degradation
   Same seeds every time. **Generate this grid from every checkpoint (every 250 steps).** Contact-sheet it.

2. **Identity similarity, if it's a face — one place where a number genuinely helps.** Use an off-the-shelf face-recognition embedder (**ArcFace / InsightFace**) and compute
$$
\text{IdentitySim} = \cos\!\big(f_{\text{ArcFace}}(x_{\text{gen}}),\ \tfrac{1}{N}\textstyle\sum_i f_{\text{ArcFace}}(x^{(i)}_{\text{train}})\big)
$$
   Typical: **> 0.6** is a good likeness, **> 0.4** is recognizable, **< 0.3** failed. *(Confidence: medium — thresholds are model-and-implementation-dependent; the course should have him calibrate on his own data.)* **Why this one works when CLIP doesn't:** ArcFace was trained to answer *exactly* the question "are these the same person," so it's not a proxy — it's a purpose-built instrument. **The lesson: a narrow, purpose-built metric beats a broad, general one.**

3. **Diversity check.** Generate 16 images from **one prompt, 16 seeds**. If they're all near-identical → **mode collapse from over-training** (§12.4, sign 3).

4. **The strength sweep.** Same prompt/seed at `strength` ∈ {0.4, 0.6, 0.8, **1.0**, 1.2}. **A healthy LoRA is usable at 1.0.** If you need 0.5, you over-trained (§12.4, sign 6).

5. **Look at it.** Then look at it again tomorrow.

> **"Human eval still rules" — and here is *why*, mechanistically, not as a platitude.** Every automated metric
> is a **model** of human judgment. Models of human judgment are **exactly as flawed as the models generating
> the images** — often more so, and often **flawed in correlated ways** (CLIPScore and the SD text encoder are
> *the same network*, §9.2/§14.3 — the judge and the defendant share a brain). Meanwhile the *target* is
> definitionally "what a human wants." **When your metric is a worse model of humans than your generator is,
> you should just ask the human.** In 2026, for LoRA evaluation at $N=20$, **that is unambiguously the state of
> the art**, and it is not a cop-out — it is the correct engineering decision.
>
> **And it's what he already does.** He has been evaluating his ComfyUI outputs by looking at them for years.
> **The course's contribution is not to replace that with a metric — it's to make his looking *systematic*:**
> fixed prompts, fixed seeds, checkpoints, contact sheets, controls. **Turn a vibe into a protocol.** That's a
> real upgrade and it's honest about what's achievable.

### 14.5 Demos — §14

**D14.1 — The CFG Pareto curve (the demo that teaches both metrics at once).**
- **Plot:** a scatter/line in (CLIP score, FID) space, one point per CFG value, precomputed from a real sweep (CFG ∈ {1,2,3,5,7,9,12,15,20}), with a **thumbnail** of the image at the hovered point.
- **Control:** the **same $w$ slider as D9.1** — literally reuse it, so §9 and §14 share a control.
- **Insight:** drag $w$ up. **CLIP score rises. FID rises (worse).** They move in *opposite* directions. Hover at $w=20$: CLIP score is near its max and the image is **fried garbage.** *"You just maximized the alignment metric by destroying the image. That's what 'the metric is gameable' means."* **This one demo teaches: FID's diversity-blindness, CLIP score's gameability, CFG's tradeoff, and why human eval survives — simultaneously.** Extremely high value per line of code.

**D14.2 — FID's Gaussian lie.**
- **Plot:** a 2-D feature space (a stand-in for the 2048-D one). Two point clouds with **identical mean and covariance** but obviously different shapes — e.g. a **ring** vs. a **cross**, or four corner-blobs vs. a diagonal band.
- **Readout:** the computed **FID between them: ≈ 0.0**, next to the two visibly different pictures.
- **Insight:** *"These two distributions share a mean and a covariance. FID says they are identical. Look at them."* **One screen, and the Gaussian assumption is dead forever.** Cheap to build (the point clouds can be constructed analytically to match moments) and devastating.

**D14.3 — FID vs. N.**
- **Plot:** FID (computed in JS on a small synthetic feature set) against sample count $N$, log-x, from 100 to 50,000, with error bars from repeated draws.
- **Insight:** the curve **doesn't flatten until ~10k** and the small-$N$ values are **wildly wrong and biased in one direction** — it's not noise, it's bias. *"This is why you cannot FID a 20-image LoRA."* Then overlay **CMMD**, which is flat from $N\approx100$. The fix, visible.

### 14.6 Misconceptions — §14

| Misconception | Truth | Correction |
|---|---|---|
| **"FID measures image quality."** | It measures the **Fréchet distance between two Gaussians fitted to Inception-V3 features**. It conflates fidelity with diversity, is blind to hands and text, and is biased at small $N$. | D14.2. Two identical FIDs, two obviously different distributions. |
| "Lower FID = better model." | Not reliably, for modern models. **FID correlates poorly with human judgment above a quality threshold**, and is dominated by dataset/style matching. | The Artificial Analysis Arena (human ELO) is what actually ranks models in 2026, and it ranked a 6B above a 32B (§10.1). |
| "FID numbers are comparable across papers." | **Frequently not** — resize interpolation, JPEG, and the Inception weights port move FID by amounts comparable to real differences (Parmar et al. 2022). | Limitation 5. A genuine, documented scandal. |
| "I'll use FID to evaluate my LoRA." | **Needs 10k–50k samples.** You have 20 images and you're evaluating a *concept*, not a distribution. | D14.3 + §14.4's protocol. |
| **"CLIP score measures whether the image is good."** | It measures **alignment only** — and it saturates, and **it shares CLIP's bag-of-words blindness**, so it can't catch attribute-binding errors, *which is the thing it's for.* | The judge and the defendant share a brain (§9.2). |
| "Higher CLIP score = better image." | **Crank CFG to 20: CLIP score goes up, image becomes garbage.** | **D14.1.** The single most efficient misconception-killer in §14. |
| **"Human eval is a cop-out until we get better metrics."** | Every metric is a **model of human judgment**, and in 2026 those models are worse than the generators. **When your judge is dumber than your defendant, ask the human.** | §14.4's boxed argument. It's an engineering conclusion, not resignation. |
| "The training loss tells me if it's working." | It's dominated by the random $t$ draw (§12.4). Nearly uninformative. | Generate a fixed grid every 250 steps. |
| "There's a number for 'is my LoRA good.'" | **There isn't.** There's a protocol: fixed prompt grid + OOD probe + drift control + diversity check + strength sweep + your eyes. | §14.4. **Turn the vibe into a protocol** — that's the achievable win. |

### 14.7 Dependency — §14

**Requires:** §9 (CFG, for D14.1 and the gameability argument), §9.2 (CLIP — **§14.3's central critique is that
CLIPScore inherits the text encoder's weakness, which requires knowing what the text encoder is**), §12.4
(the loss-curve warning), §11 (what a LoRA is).
From the trunk: cosine similarity, mean/covariance. **The matrix square root $(\Sigma_r\Sigma_g)^{1/2}$ in the
FID formula should be named and waved past** — it's a real operation but explaining it costs more than it
teaches. Say "there is a matrix analogue of $\sqrt{\;}$, it exists, move on."

---

## 15. Hardware reality — diffusion fine-tuning on a DGX Spark

### 15.1 The machine — verified specs

| Spec | Value | Source confidence |
|---|---|---|
| Chip | **NVIDIA GB10 Superchip** (Blackwell) | **High** |
| Unified memory | **128 GB LPDDR5X**, coherent CPU+GPU | **High** |
| **Memory bandwidth** | **273 GB/s** | **High** |
| AI performance | **~1000 AI TOPS** at **FP4** with sparsity | **High** |
| Tensor cores | **5th gen, native FP4** | **High** |
| CPU | **20 Arm cores** (10× Cortex-X925 + 10× Cortex-A725) | **High** |
| Storage | 4 TB NVMe | High |
| Networking | dual QSFP, **200 Gb/s** aggregate; **2 Sparks can be linked** | High |

> **The one number that defines this machine: 273 GB/s.** For comparison, an RTX 5090 has ~1.8 TB/s of GDDR7
> and a datacenter B200 has ~8 TB/s of HBM3e. **The Spark has roughly 1/7th the bandwidth of a consumer
> flagship and ~1/30th of a datacenter part — but 4× the memory of the 5090 and it's unified.**
>
> **So the Spark's identity is: capacity, not speed.** It is a machine for running and training things that
> **do not fit anywhere else** at this price. **That framing should govern every recommendation in §15**, and
> it's the honest one — the course should not oversell it. It's a 128 GB box that thinks at a moderate pace.

### 15.2 What fits — the capacity table

**Inference, bf16, from §8.3:**

| Model | Total | Fits in 128 GB? |
|---|---|---|
| SD 1.5 | ~4 GB | ✅ trivially |
| SDXL | ~8 GB | ✅ trivially |
| SD 3.5 Large | ~16 GB | ✅ |
| Z-Image-Turbo | ~14 GB | ✅ |
| FLUX.2-klein-4B | **~13 GB** | ✅ |
| FLUX.1-dev | **~29–37 GB** | ✅ comfortably |
| Qwen-Image 20B | ~40 GB + encoder | ✅ |
| **FLUX.2-dev** | **~112–120 GB** | ✅ **barely — and this is the point** |

**Training:**

| Task | Memory | Fits? |
|---|---|---|
| SD1.5 LoRA r=16 | ~6 GB | ✅ |
| SDXL LoRA r=32 | ~14 GB | ✅ |
| **SDXL full fine-tune** (AdamW8bit + fused backward) | ~25–40 GB | ✅ **— and this is notable** |
| **FLUX.1-dev LoRA r=16** | **~29–31 GB** | ✅ **comfortably** (§11.2) |
| FLUX.1-dev LoRA r=64 | ~33 GB | ✅ |
| **FLUX.1-dev full fine-tune** (AdamW fp32) | **~196 GB** | ❌ (§11.1) |
| FLUX.1-dev full FT (AdamW8bit + fused bwd + ckpt) | ~90–110 GB | ⚠️ **tight but plausible** — *estimate, confidence low-medium* |
| **FLUX.2-dev LoRA r=16** (precomputed embeddings, bf16 base) | **~70 GB** | ✅ |
| FLUX.2-dev LoRA (FP8 base) | ~38 GB | ✅ comfortably |
| **FLUX.2-dev full fine-tune** | **~520 GB** | ❌ **not close** |

> **The headline for him: he can LoRA-train FLUX.2-dev (32B), which is impossible on any consumer GPU** — an
> RTX 5090 has 32 GB and the base weights alone are 64 GB at bf16. **He can also full-fine-tune SDXL**, which
> is a real capability most people never get. **Those two facts justify the machine for this work.**
>
> **And the arithmetic that gets him there is §11.2's, not magic:** precompute the text embeddings (**−48 GB**),
> precache the latents (**−0.2 GB and a big speedup**), and the 32B model becomes trainable. **The course
> taught him to do that in §8.3 and §11.2 and now it pays off on his actual hardware.**

### 15.3 How fast — the honest numbers

**Verified:**
- **DGX Spark generates a 1K image every 2.6 s with FLUX.1-Schnell, 4 steps, 1024², batch 1, FP4** — **NOT
  FLUX.1-dev at 28 steps** (D-16 correction; the old unlabelled "FLUX.1 12B at FP4" wording was the wrong-model
  trap). *(NVIDIA's own figure, correctly attributed.)*
- One independent benchmark reports FLUX generation at **82 s** with **171 s of loading** and **29 GB** of memory. *(Medium confidence — almost certainly bf16/unoptimized, and note the **171 s load time**: at 273 GB/s, pulling ~30 GB off NVMe and into memory *is* the bottleneck. **First-run latency on this machine is dominated by loading, not computing** — a real UX fact he'll notice.)*
- ~~Llama 3.1 8B LoRA fine-tuning peaked at **53,657 tokens/s**.~~ **← CORRECTED to 6,969.59 tok/s** (D-10m /
  constants §6.7): NVIDIA's own blog says 6,969.59; the 53,657 figure was an aggregator inflation. Trust the primary.

**The §8.4 roofline analysis, applied to training** *(these are **my estimates**, clearly labeled — confidence
low-medium; **the course should ship a benchmark script (§16, D14) and have him measure his own machine**,
which is far better pedagogy than trusting my arithmetic):*

**FLUX.1-dev LoRA, 1024², batch 1, bf16, gradient checkpointing:**
- Forward pass: must stream **24 GB** of weights → $24/273 = \mathbf{88\ \text{ms}}$ bandwidth floor.
- Backward ≈ 2× forward in FLOPs; **with gradient checkpointing, add a second forward** → total ≈ **3–4× forward**.
- **Estimated: ~0.4–1.0 s/step.**
- **2000 steps → ~15–35 minutes.**

> **Sanity-check that against the verified anchor:** NVIDIA's 2.6 s / 28 steps = **93 ms/step at FP4
> inference.** bf16 is 4× the bytes → ~350 ms/step forward if bandwidth-bound. Training ≈ 3× that ≈ **1.0 s/step**
> → **2000 steps ≈ 33 min.** **The two estimates bracket each other.** **Call it 20–40 minutes for a FLUX.1-dev
> LoRA on 20 images, and tell him to measure it.**

**Rough expectations table** *(estimates — label clearly):*

| Task | Steps | Est. time on Spark |
|---|---|---|
| **SD1.5 LoRA**, 20 images, 512² | 1500 | **~8–15 min** |
| **SDXL LoRA**, 20 images, 1024² | 2000 | **~30–50 min** |
| **FLUX.1-dev LoRA**, 20 images, 1024² | 2000 | **~20–40 min** |
| FLUX.2-dev LoRA (FP8 base), 20 img | 2000 | **~1.5–3 hr** *(low confidence)* |
| SDXL full fine-tune, 10k images | 20k | **~1–2 days** |

> **The honest comparison he deserves:** an **RTX 5090** (32 GB, ~1.8 TB/s) will train an **SDXL or
> FLUX.1-dev** LoRA **substantially faster than the Spark** — bandwidth wins, and these fit in 32 GB. **The
> Spark's win is not speed; it's the jobs the 5090 cannot start at all** (FLUX.2-dev LoRA, SDXL full FT,
> multi-model pipelines held resident simultaneously). **Say this plainly.** He owns the machine; he deserves
> an accurate model of it, not marketing. **"Capacity, not speed"** — and for *learning* and *experimenting
> with things that don't fit*, capacity is the right purchase.

### 15.4 Spark-specific practicalities

1. **Unified memory changes the offloading calculus.** `enable_model_cpu_offload()` on a discrete GPU means a **PCIe round-trip** (~25 GB/s, painful). On the Spark, CPU and GPU **share coherent memory** — "offloading" is close to a pointer change. **The standard advice about avoiding offload does not apply here**, and that's a genuine, non-obvious advantage worth a box.
2. **FP4 is native.** The GB10's 5th-gen tensor cores do FP4 in hardware — that's what the "1000 TOPS" is. **BFL ships NVFP4 checkpoints** (§10.1). **On this machine, using them is not a compromise; it's using the hardware you bought.** And per §8.4, at bf16 he is **bandwidth-bound** — so FP4/FP8 buys **real wall-clock**, not just capacity. That's a strong, arithmetic-backed recommendation.
3. **ARM (aarch64), not x86.** Some Python wheels, some custom ComfyUI nodes, and some CUDA extensions will not have aarch64 builds. **This is a real, current friction point** and the course should warn him rather than let him discover it at 1 a.m. *(Confidence: high that it's a general ARM issue; medium on which specific packages break in 2026.)* NVIDIA's **DGX Spark Playbooks** repo is the canonical starting point — there is an official **FLUX.1 DreamBooth LoRA fine-tuning playbook** for the Spark. **Point him at it directly; it's exactly his use case.**
4. **Load times dominate short runs.** 171 s to load ~30 GB. Precache latents and embeddings **once**, to disk, and reuse — this is a bigger win here than on a machine with fast storage-to-VRAM paths.
5. **Two Sparks link at 200 Gb/s** (= 25 GB/s). For **inference** on a huge model, plausible. For **training**, 25 GB/s is **slow relative to 273 GB/s of local bandwidth** — gradient sync would dominate. **Don't oversell clustering for this workload.**

### 15.5 Misconceptions — §15

| Misconception | Truth | Correction |
|---|---|---|
| "128 GB unified = like a 128 GB GPU." | **273 GB/s vs. a 5090's ~1.8 TB/s.** You have **capacity**, not **speed**. | The §15.1 box. Then §15.3's honest comparison: a 5090 beats the Spark on jobs that fit in 32 GB. |
| "A 5090 is strictly worse because it has less memory." | For SDXL/FLUX.1 LoRA — **jobs that fit** — the 5090 is **faster**. The Spark wins only on jobs the 5090 **can't start**. | §15.3. Give him the accurate model. |
| "Quantization is a quality compromise." | On this machine, **bf16 inference is bandwidth-bound** (§8.4): 24 GB/step ÷ 273 GB/s = 88 ms of *pure waiting*. FP8 halves the bytes and **halves the wait**. **FP4 is native in the GB10's tensor cores.** | The §8.4 roofline. Quantization here buys wall-clock, not just capacity — and the hardware was **designed** for FP4. |
| "I can full-fine-tune FLUX.1-dev, I have 128 GB." | **196 GB with AdamW.** Needs 8-bit optimizer + fused backward + checkpointing to *maybe* fit — and it's the wrong tool anyway (§11.1). | **The 16-bytes-per-parameter rule.** $12\text{B} \times 16 = 192$ GB. Memorize it. |
| "Offloading will kill my performance." | **Not on unified coherent memory.** No PCIe round-trip. | §15.4.1. A real Spark advantage. |
| "It's a Linux box, everything just works." | **aarch64.** Some wheels and ComfyUI custom nodes have no ARM builds. | §15.4.3 + the DGX Spark Playbooks repo. |
| "Two Sparks = 2× training speed." | 200 Gb/s = 25 GB/s interconnect vs. **273 GB/s** local. Gradient sync would dominate. Fine for inference; poor for data-parallel training. | §15.4.5. |

### 15.6 Dependency — §15

**Requires:** §8.3–8.4 (the memory table and the roofline arithmetic — **§15 is §8 applied to one machine**),
§11.1–11.2 (the training memory formulas), §13 (the knobs).
**This section is the course's final "cash the theory" moment** and should be positioned as such: *everything
you learned about tensor shapes, parameter counts, and bytes-per-parameter now tells you exactly what your own
desk can and cannot do.* **Nothing new is taught here — that's the point.** It's a capstone by arithmetic.

---

## 16. Runnable code artifacts — the manifest

**Design principles for the whole set, and the architect should enforce them:**
1. **Every artifact runs on the DGX Spark**, and D1–D5 also run on a CPU laptop in under 5 minutes.
2. **Every artifact prints tensor shapes and memory** — the numbers from §2.5, §8.3, §11.2 should appear in the *program's own output*, so the tables in the course are *verified by the code the learner runs*. **That is the single best thing this course can do for credibility.**
3. **From-scratch first, library second.** Every concept gets a NumPy/bare-PyTorch version before a `diffusers` version, and the course explicitly shows they agree.
4. **No artifact is a black box demo.** Each one answers a question the text posed.

| # | File | Demonstrates | Sections | Runtime |
|---|---|---|---|---|
| **D1** | `autoencoder_void.py` | Train a 2-latent AE on MNIST; **scatter the codes and see the voids**; sample the prior → mush. **Then** add the KL term and watch the voids fill. **The VAE motivated by failure.** | §2.2–2.4 | ~2 min CPU |
| **D2** | `vae_ceiling.py` | Encode/decode a real photo through **the actual SD1.5 (4ch) and FLUX (16ch) VAEs**; print shapes `[1,4,64,64]` / `[1,16,128,128]`, compression ratios **48×/12×**, and show the residual at 4× gain. **Also:** sample $z\sim\mathcal{N}(0,I)$ at latent shape and decode → colored noise, proving the SD VAE's prior is not Gaussian. | §2.5–2.7 | 30 s |
| **D3** | `diffusion_2d.py` | **DDPM from scratch on a 2-D Swiss roll / 8-Gaussians.** ~200 lines, no U-Net, a 3-layer MLP. Implements Eq. 3.3, $\mathcal{L}_\text{simple}$, and the ancestral sampler **by hand**. **The whole of §3–§4 in one readable file.** | §3, §4 | ~40 s CPU |
| **D4** | `ddpm_mnist.py` | A **~1.5M-param U-Net** (deliberately a U-Net — §7.4's "hierarchy helps at small scale") on 28×28 MNIST. Real images from noise, on a laptop. Includes a **skip-severing flag** for the §7.2 ablation. | §3, §4, §7.2 | ~10 min Spark |
| **D5** | `prediction_targets.py` | Take one $(x_0,\epsilon,t)$; compute $x_t$; then **derive $x_0$, $\epsilon$, and $v$ from each other** and `assert torch.allclose(...)` to machine precision. **Proves the §4.3 rigid triangle numerically.** Prints the error-amplification factor $1/\sqrt{\bar\alpha_t}$ at $t=999$ → **159×**. | §4.3 | instant |
| **D6** | `score_check.py` | Numerically differentiate $\log q(x_t\mid x_0)$ with finite differences; compare to $-\epsilon/\sqrt{1-\bar\alpha_t}$. **`assert allclose`.** The §5.2 identity, verified, not asserted. | §5.2 | instant |
| **D7** | `flow_matching_2d.py` | **The same 2-D toy as D3, rectified flow.** Deliberately structured as a **near-identical file with three lines changed** ($a_t,b_t$; the target; the sampler). **A `diff D3 D7` IS the §6.3 reconciliation, in code.** Then: sweep step count 1→64 for both and plot W2. **The straight path wins at low $N$.** | §6 | ~40 s CPU |
| **D8** | `sample_flux.py` | Real generation with `diffusers` + FLUX.1-dev. Exposes seed/steps/`guidance`/scheduler. **Prints the latent shape `[1,16,128,128]`, the token count `4096`, and peak memory** — verifying §2.5 and §8.3 **on his machine**. | §2.5, §7, §8 | ~3 s/image |
| **D9** | `cfg_by_hand.py` | Run the transformer **twice manually** (cond + uncond), apply $\tilde\epsilon = \epsilon_\varnothing + w(\epsilon_c-\epsilon_\varnothing)$ by hand, and **`assert allclose` against `pipe(guidance_scale=w)`.** Then: **print $\|\tilde\epsilon\|/\|\epsilon_c\|$ vs $w$** and show it growing linearly → **§9.3's frying mechanism, measured.** Then implement **CFG-rescale** and show the norm snap back. | §9.3 | ~1 min |
| **D10** | `lora_from_scratch.py` | Implement `LoRALinear` in **~30 lines** — no `peft`. Wrap a real FLUX/SDXL attention module. **Print:** full params `9,437,184`, LoRA params `98,304`, ratio **1.04%**; assert output is **bit-identical to base at step 0** (because $B=0$). **Then** swap in `peft` and show the same numbers. §11.2, made real. | §11.2 | instant |
| **D11** | `train_lora_flux.py` | **The main artifact.** DreamBooth-LoRA on 20 images. **Precomputes text embeddings and latents to disk** (§8.3, §15.4 — and prints the GB saved), bf16, gradient checkpointing, AdamW8bit, `guidance_scale=1.0` (§9.4), logit-normal timestep sampling (§6.4). **Prints the full memory breakdown at step 0, matching §11.2's table.** Saves ComfyUI-compatible `.safetensors`. | §11–§13 | ~30 min |
| **D12** | `train_lora_dreambooth_prior.py` | D11 + generated class images + the prior-preservation term $\lambda=1.0$. **Ships the 30-second drift diagnostic** (generate `"a photo of a dog"`, no trigger). **Lets him A/B whether reg images help — resolving §11.3's live disagreement on his own data.** | §11.3 | ~60 min |
| **D13** | `textual_inversion.py` | Optimize **768 floats**. Print `Trainable parameters: 768`. Save the **3 KB** file. **The §11.4 climax, made undeniable.** | §11.4 | ~15 min |
| **D14** | `eval_lora.py` | The **§14.4 protocol**, automated: fixed prompt grid (incl. the **Mars probe** and the **drift control**) × every checkpoint × fixed seeds → a **contact sheet PNG**. Plus ArcFace identity similarity, a diversity check, and a strength sweep. **The instrument the course argues for.** | §14.4 | ~5 min |
| **D15** | `memory_calculator.py` | Prints the §8.3 / §11.1 / §11.2 / §15.2 tables **from formulas**, for any (model, method, rank, optimizer, precision). **The CLI twin of demo D11.2.** He will actually use this. | §8, §11, §15 | instant |
| **D16** | `bench_spark.py` | Measures **his** machine: achieved memory bandwidth, ms/step for forward and forward+backward at bf16/fp8/fp4, and s/image — **against the §15.3 predictions.** | §15 | ~5 min |
| **D17** | `caption_ab.py` | **The §12 experiment nobody ships.** Train **two SD1.5 LoRAs on the identical 20 images** — one with Strategy A captions, one with Strategy B (§12.2) — then run the **Mars probe** on both. **He *measures* the captioning rule instead of believing it.** | §12.2 | ~30 min |
| **D18** | `comfy_export.py` | Convert/verify a trained LoRA's key naming for ComfyUI's `LoraLoader`, and copy it into `models/loras/`. **Closes the loop back to the tool he actually uses.** | §13.1 | instant |

> **The three artifacts that matter most, if the architect must cut:** **D7** (the `diff` that *is* the
> reconciliation — no other course does this), **D11** (the destination), and **D17** (the course's one genuine
> contribution to the LoRA-folklore problem — it converts §12's most important rule from advice into a
> measurement). **D5, D6, D9, and D10 are `assert`-based and near-free to build**, and each one converts a
> claimed identity into a *verified* one. **Ship all four; they're the cheapest credibility in the course.**

---

## 17. The dependency spine — and what the architect must reconcile

### 17.1 The recommended order

```
 TRUNK (shared with LLM track)
   MLP · backprop · MSE/CE · Adam · Gaussians (mean/var, "variances add") ·
   ∇ as vector of partials · Jensen's inequality · KL(N‖N) closed form ·
   ATTENTION (QKV, multi-head, softmax scaling, mask-AGNOSTIC) ·
   cross-attention as a first-class case · sinusoidal embeddings ·
   LayerNorm/GroupNorm · FiLM-style modulation · residual/skip topology ·
   MATRIX RANK & SVD · cosine similarity
        │
        ▼
  §1  Generative framing · the manifold picture · the Z_θ dodge table
        │
        ▼
  §2  VAE (encoder/decoder, reparameterization #1, the KL's TWO jobs,
      real latent shapes, why SD's VAE is barely a VAE)
        │
        ▼
  §3  Forward process (Eq. 3.3 ← THE equation; the unit-circle/trig picture;
      the ᾱ table; terminal-SNR; resolution dependence)
        │
        ▼
  §4  Reverse process · ε/x₀/v as ONE triangle · ELBO COLLAPSE ← the aha ·
      the 6-line training loop
        │
        ├──────────────┐
        ▼              ▼
  §5  Score/SDE    §7  U-Net → DiT (needs trunk attention)
      (needs §4)       │
        │              │
        ▼              │
  §6  Flow matching ←──┘  (needs §4.3's v-pred + §5.4's ODE)
      · the (aₜ,bₜ) route map ← THE reconciliation
        │
        ▼
  §8  Latent diffusion memory arithmetic (needs §2.5 + §7)
        │
        ▼
  §9  Conditioning · cross-attention · CFG DERIVED (needs §5.2's score!)
        │
        ▼
  §10 The 2026 landscape  ← isolated, expected to age
        │
        ▼
  §11 Fine-tuning: the four places · LoRA · DreamBooth · TI ·
      ControlNet/IP-Adapter are NOT fine-tuning
        │
        ▼
  §12 Datasets · the invariance principle · captioning DERIVED from §9
        │
        ▼
  §13 Tooling · the annotated config.toml where every line cites a §
        │
        ▼
  §14 Evaluation · why human eval survives
        │
        ▼
  §15 DGX Spark ← capstone by arithmetic; nothing new is taught
```

### 17.2 The five hard orderings, and why

1. **§2 before §4.** The ELBO must be met **once, on the easy object**, or §4.4 is a wall. The VAE is also *literally* the thing latent diffusion runs on, so it's not a detour.
2. **§3 before §4.** §4 is nothing but an inversion of §3. Eq. 3.3 must be fluent first.
3. **§4.3 ($v$-pred) before §6.** Non-negotiable. $v$-prediction **is** flow matching's velocity on the circular path. Teach $v$ in §4 and §6 costs almost nothing; skip it and §6 arrives from nowhere.
4. **§5.2 (the score identity) before §9.3 (CFG).** **CFG cannot be honestly derived without the score.** Without §5, the CFG formula is an incantation — which is exactly the state the learner is already in, and exactly what the course exists to fix.
5. **§9 before §12.** The captioning rule ("caption what you want to vary") is a **consequence of what $c$ does in the loss**. Derived, it's memorable and generalizes. Asserted, it's one more Reddit rule among a hundred contradictory ones. **This is the course's differentiator in §12 — protect it.**

### 17.3 The one contested ordering — architect must decide

**§5 (score/SDE) before or after §4 (DDPM)?** Score-first is defensible and MIT's 2026 course does something
like it. **My recommendation: §4 first**, because (a) the learner gets a concrete 6-line algorithm to hold
before the abstraction, and (b) §5.2's *"we already trained a score model and didn't know it"* reveal is only
available in this order, and it's one of the best moments in the track. **But decide once and be consistent.**

### 17.4 What the architect must reconcile with other briefs

| # | Issue | With | The ask |
|---|---|---|---|
| **1** | **Attention must be taught mask-agnostically, and cross-attention must be first-class.** Diffusion attention is **bidirectional** (no causal mask), and cross-attention takes $K,V$ from **a different modality**. If the trunk teaches attention only in a causal-LM framing, **§7 and §9 break.** | **architectures brief, LLM brief** | **Highest-priority reconciliation in this brief.** |
| **2** | **Matrix rank + SVD.** §11.2 needs it; the LLM track's LoRA section needs the identical thing. **Build it once in the trunk.** | LLM brief, trunk | ~1 page + demo D11.1 |
| **3** | **Sinusoidal embeddings** appear in both tracks with identical math and different meanings (position vs. noise level). Teach once in the trunk; both tracks point at it. | architectures brief | small |
| **4** | **The tracks converge and the course should say so.** FLUX.2's text encoder is **Mistral-3 24B** — an LLM, twice the size of all of FLUX.1. §9.2. **This is the strongest argument for the shared trunk** and possibly deserves an explicit structural "rejoining" moment. | LLM brief | **conceptual — worth a joint decision** |
| **5** | **LoRA is taught in both tracks.** Same math ($W+\frac{\alpha}{r}BA$), same rank/alpha confusion, same "the memory win is optimizer states, not weights" insight. **Decide who owns the derivation.** Recommend: **trunk owns the math; each track owns its own targets, ranks, and gotchas** (diffusion: `to_q/k/v/out`, conv-for-styles, ComfyUI `strength`; LLM: attention vs. MLP targets, QLoRA). | LLM brief | **must not be taught twice** |
| **6** | **"LR scales inversely with trainable parameter count"** (§13.2: 1e-6 full FT → 1e-4 LoRA → 5e-3 TI). The LLM track will make the same observation. **One principle, stated once.** | LLM brief | small |
| **7** | **The 2026 story diverges between tracks** — LLMs are still scaling; image models are **consolidating** (Z-Image 6B > FLUX.2 32B). **This contrast is interesting and should be drawn deliberately, not left as an accident.** | LLM brief | conceptual |
| **8** | **Jensen's inequality + KL(𝒩‖𝒩) closed form.** §2.4b and §4.4 both need them. Likely a genuine gap for a rusty learner. | trunk | ~½ page |
| **9** | **GANs.** §1.3 wants ~10 minutes (mode collapse, distillation ancestry, and **the VAE decoder is GAN-trained**). If the trunk or another brief covers GANs, this track should point rather than repeat. | trunk / LLM brief | scope check |
| **10** | **Notation convention.** DDPM ($t{:}0{\to}T$), flow matching ($t{:}0{\to}1$, **data at 0**), and Karras ($\sigma$) are three live conventions. **Pick ONE for the course's prose; put the others in a translation table.** Convention drift will destroy this track faster than any conceptual gap. | internal — but flag to all briefs | **critical** |

### 17.5 Honest assessment of confidence in this brief

**High confidence** (verified against primary sources or standard, stable mathematics): all equations in §2–§6,
§9.3's CFG derivation; the FLUX.2-dev 32B / Mistral-3 24B / FLUX.2 VAE 32-channel facts; FLUX.2-klein-4B
(Apache 2.0, ~13 GB, 4 steps @ guidance 1.0); Z-Image 6B Apache 2.0 (2025-11-26) and its #1 open-weights arena
ranking; SD3.5-Large 8B/38-layer/CLIP+T5 config; diffusers **0.39.0** (2026-07-03) and the
`FlowMatchEulerDiscreteScheduler` parameters; **all DGX Spark specs** (GB10, 128 GB, **273 GB/s**, 1000 FP4
TOPS, 20 Arm cores); the 2.6 s/1K-image FLUX.1 FP4 Spark figure; **all memory arithmetic in §8, §11, §15**
(it's multiplication — but the architect should re-run it).

**Medium confidence — verify before printing:** the FLUX.1-dev 19-double + 38-single block structure; SD3.5's
`shift=3.0`; the FLUX.2 VAE's *spatial* factor (**§2.5 documents a real source conflict — I recommend
$f=8$, 32ch, and 4096 tokens at 1024², contra one auto-generated wiki**); FLUX.2-klein-9B's 29 GB and the
FP8/NVFP4 −40%/−55% figures; the SD1.5 U-Net's 860M breakdown; the training-tool landscape (§13.1 — **churns
fast**); CMMD's adoption status; ComfyUI's native training node.

**Low confidence — flagged in place, do not print without verification:** **Qwen-Image 2.0** (7B, 2026-02-10,
native 2K) — **single secondary source**; my **training-time estimates** in §15.3 (**ship D16 and have him
measure**); the §12.1 dataset-size table (**practitioner folklore, not benchmarks — label it as such, and
consider having D17 let him test it**); the ArcFace identity thresholds in §14.4.

**Genuine open questions in the field, flagged in-place rather than papered over:** why $w\approx7$ CFG beats
$w=1$ (§9.5 — and Karras et al.'s autoguidance result, which suggests the textbook story is *wrong*); whether
flow matching's win is mathematical or just better defaults (§6.5); the correct loss weighting $w_t$ (§4.4 —
still an active knob in 2026); whether prior-preservation images help with LoRA (§11.3 — the community is split
and both camps ship good work); and whether FID deserves to survive (§14.2).

**Where I would place the course's genuine differentiators**, having surveyed what's available publicly:
**§4.2's Feller paragraph** (why 1000 steps — almost nobody explains this correctly), **§4.3's rigid triangle +
unit circle** (dissolves the ε/x₀/v confusion permanently), **§6.3's $(a_t,b_t)$ route map** (the DDPM↔FM
reconciliation — essentially absent from free material), **§9.4's FLUX-`guidance`-is-not-CFG box** (highest
immediate practical value for this specific learner), **§12.2's derivation of the captioning rule from the
loss** (turns folklore into a consequence), and **§8.4's prediction of a published benchmark from first
principles** (which this learner, specifically, will love).
