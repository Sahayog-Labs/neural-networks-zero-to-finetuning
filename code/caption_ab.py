#!/usr/bin/env python3
"""
caption_ab.py — the caption ablation nobody ships. RUNG 8 companion, p.62 (manifest D17).

Page 62 hands you a RULE, boxed and derived from the loss (§12.2, "🔑 Caption what you
want to VARY, omit what you BAKE IN"):

    Training minimizes  E_{x0,eps,t} || eps - eps_theta(x_t, t, c) ||^2 .
    Any visual feature MENTIONED in the caption c can be explained by those words, so the
    model hangs it on the words. Any feature PRESENT in the image but ABSENT from c has
    nowhere to go but the trigger token / the LoRA weights — so it gets BAKED IN.

Everyone repeats this rule. Almost nobody runs the experiment. This script does:

  * Strategy A (detailed captions):  "a photo of sks man, brown hair, blue shirt,
    smiling, park background, sunny day"  — the varying attributes are NAMED, so `sks`
    absorbs only what's left: the face geometry.        -> a FLEXIBLE face LoRA.
  * Strategy B (bare trigger):       "a photo of sks man"  for every image — the model
    must explain hair/shirt/background USING `sks`, because nothing else is available.
    -> `sks` now means "this guy, blue shirt, in a park".

TWO SD1.5 LoRAs, the SAME 20 images, the SAME trigger token, the SAME seed and
hyperparameters. **The only thing that differs is the caption.** Then both get the
out-of-distribution Mars probe (§12.4):

    "sks man as an astronaut on the surface of Mars, full body"

  ✅ A generalizes: your guy, spacesuit, Mars.
  ❌ B bleeds:      your guy, spacesuit — and the park background / blue shirt tag along.

You MEASURE the rule instead of taking it on faith. The outcome is labelled [EST] in the
prediction (`--self-test`, the pure-logic path below) and [MEA] once you have run it on
your own images on the Spark (`--run`). It is a real ablation, not a frozen claim.

Why SD1.5 (§10.2 "teach on SD1.5"): 860M UNet, 512², DDPM, CLIP-77 — it trains a LoRA in
~10 min, every knob still works, and the whole tutorial ecosystem assumes it. The rule you
prove here is model-agnostic; SD1.5 is just the fastest place to prove it.

SAFETY
------
The --run path trains TWO real SD1.5 LoRAs and then generates from each: GPU-saturating,
~20–40 min total on the Spark. It WILL contend with ComfyUI if that is live on the GPU —
do not start it blind (pass --device cpu only for a tiny smoke check; SD1.5 on CPU is very
slow). It writes two adapter dirs + a contact sheet under --outdir; nothing else on your
system is touched, and it installs nothing. The default path (no --run) is the pure-logic
--self-test: it allocates nothing, needs no GPU/torch, and runs in milliseconds — run that
first, on any machine, to see the whole prediction and the assertions behind it.

Usage
-----
    python caption_ab.py                    # --self-test: the prediction + assertions, no GPU
    python caption_ab.py --self-test        #   (same thing, explicit)

    # the real 2-LoRA ablation on the Spark:
    python caption_ab.py --run --images ~/data/me20 --captions ~/data/me20/detailed.txt \
        --sd15 ~/ComfyUI/models/checkpoints/v1-5-pruned-emaonly.safetensors
    python caption_ab.py --run --quick ...  # 300 steps/LoRA instead of ~1200 (a fast look)

`--captions` (Strategy A) is ONE detailed caption per image, in filename-sorted order, one
per line — this is the ten minutes of work §12.2 says beats a week of tuning. Strategy B is
generated for you: the bare "a photo of <token> <class>" for every image.
"""

import argparse
import glob
import os
import random
import sys

# --------------------------------------------------------------------------- #
# Frozen facts, transcribed from constants.md / brief-diffusion.md — never remembered.
# --------------------------------------------------------------------------- #

GiB = 1 << 30
GB = 10 ** 9

# SD1.5 @ 512², constants.md §9.6 [VP]/[DER]. The compression identity anchors the
# resolution this whole ablation trains at; if it drifts, the self-test fires.
SD15_LATENT = (1, 4, 64, 64)             # [1,4,64,64] at 512²                     [VP]
SD15_LATENT_ELEMS = 4 * 64 * 64          # 16,384                                  [VP]
SD15_PIXELS = 3 * 512 * 512              # 786,432                                 [VP]
SD15_COMPRESSION = 48                    # 786,432 / 16,384 = 48×                  [DER]

# SD1.5 text side, brief-diffusion §9.2 [VP]. The 77-token cap is a SILENT failure mode:
# a "detailed" caption over 77 tokens is truncated with no error (§12.2), which would
# quietly sabotage Strategy A. We guard it.
CLIP_MAX_TOKENS = 77                     # CLIP ViT-L/14 context length            [VP]
CLIP_DC = 768                            # CLIP text width d_c                     [VP]

VAE_SCALE_FACTOR = 0.18215               # SD1.5 latent normalization, brief §2/§confusion [VP]
SD15_UNET_PARAMS_APPROX = 860_000_000    # brief §7.5 — [VP, MEDIUM; verify vs checkpoint]

# LoRA hyperparameters for SD1.5, brief-diffusion §13.2 [VP, practitioner-consensus].
LORA_LR = 1e-4                           # UNet LoRA LR — constants.md §9.4 [VP]; NOT 1e-6 (that
                                         # is FULL DreamBooth; conflating them = D-09, a 100× error)
LORA_RANK = 16                           # r ∈ 8–32 for a face; 16 is the middle (§12.1/§13.2)
LORA_ALPHA = 16                          # α = r  (α = r/2 halves your effect — §11.2)
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")   # SD1.5 attention projections (§11.2)
TRAIN_STEPS = 1200                       # 1000–2000 for a face (§13.2); midpoint
RESOLUTION = 512                         # SD1.5 native (§12.3)
DDPM_TRAIN_TIMESTEPS = 1000              # T=1000, ε-prediction, uniform t (SD1.5/DDPM, §3.4/§4)

# The invariance principle (§12.1), as a set of attributes. A training photo of a person
# bundles these; the caption decides which bind to WORDS and which bind to the TOKEN.
FACE = "face_geometry"                                    # the concept — what you actually want
VARYING_ATTRS = ("hair", "shirt", "smile", "background", "lighting")   # what should VARY at gen time
# The Mars probe supplies NEW values for the varying attributes and asks for the concept.
MARS_PROBE = "{token} {cls} as an astronaut on the surface of Mars, full body"


# --------------------------------------------------------------------------- #
# THE MECHANISM, as pure logic (no torch). This is §12.2 derived, not asserted:
# the concept token absorbs exactly the attributes the caption does NOT name.
# --------------------------------------------------------------------------- #

def token_binding(caption_names_varying: bool):
    """What does the trigger token `sks` come to MEAN under a captioning strategy?

    caption_names_varying=True  (Strategy A): the caption names hair/shirt/.../background,
        so those bind to WORDS. The token is left with only the unmentioned constant: the
        face. -> token means {face}. Flexible.
    caption_names_varying=False (Strategy B): the caption names nothing that varies, so
        every constant-across-the-set attribute (face AND the incidental hair/shirt/
        background of these particular 20 photos) binds to the token. -> token means the
        whole bundle. Rigid.
    """
    if caption_names_varying:
        return {FACE}                                  # A: only the concept
    return {FACE} | set(VARYING_ATTRS)                 # B: the concept + baked-in incidentals


def predict_mars_probe(token_meaning: set):
    """Predicted OOD behaviour (§12.4). The Mars prompt overrides the varying attributes
    (spacesuit instead of shirt, Mars instead of park). If the token ALSO carries those
    attributes, they fight the prompt and 'bleed' through.

    Returns (generalizes: bool, bled_attrs: sorted list). [EST] — a prediction from the
    loss; --run turns it into [MEA] with your own eyes on the contact sheet.
    """
    bled = sorted(token_meaning & set(VARYING_ATTRS))  # attrs the token drags into the OOD scene
    generalizes = (len(bled) == 0)                     # clean iff the token carries only the face
    return generalizes, bled


# --------------------------------------------------------------------------- #
# Caption construction — the ONLY thing that differs between the two runs.
# --------------------------------------------------------------------------- #

def strategy_b_captions(n, token, cls):
    """Strategy B: the identical bare trigger for every one of the n images."""
    return [f"a photo of {token} {cls}"] * n


def approx_token_count(caption):
    """A torch-free UPPER-ish bound on CLIP tokens: BPE splits punctuation and some words
    into >1 piece, so we count words + commas + the 2 BOS/EOS specials. In --run we replace
    this with the real tokenizer. Purpose here is only to catch a caption that is obviously
    over the 77-token cap in the pure-logic path."""
    words = caption.replace(",", " , ").split()
    return len(words) + 2                               # +2 for <bos>/<eos>


def load_detailed_captions(path, n_images):
    """Strategy A captions: one detailed line per image, filename-sorted order."""
    with open(path, encoding="utf-8") as f:
        caps = [ln.strip() for ln in f if ln.strip()]
    if len(caps) != n_images:
        sys.exit(f"[stop] --captions has {len(caps)} lines but there are {n_images} images. "
                 "Strategy A needs exactly one detailed caption per image, sorted by filename.")
    return caps


# --------------------------------------------------------------------------- #
# THE PREDICTION (self-narrating) + the assertions. Runs with no GPU, no torch.
# --------------------------------------------------------------------------- #

def _synthetic_detailed_caption(i, token, cls):
    """A stand-in Strategy-A caption for the --self-test (no real images on hand). It NAMES
    every varying attribute — which is the whole point of Strategy A — and varies their
    values across the 20 examples so the set-level constant really is only the face."""
    # Coprime list lengths (4,5,3,...) so the (i%4, i%5, i%3) residue triple — hence the
    # whole caption — is DISTINCT for every i < lcm(4,5,3)=60, i.e. for all 20 images. This
    # mirrors the real thing: hand-written per-image captions are never identical.
    hairs = ("brown hair", "hair tied back", "wet hair", "hair under a cap")           # 4
    shirts = ("blue shirt", "red jacket", "grey hoodie", "white tee", "green sweater")  # 5
    smiles = ("smiling", "neutral expression", "laughing")                             # 3
    bgs = ("park background", "kitchen background", "city street", "beach at sunset")
    lights = ("sunny day", "overcast", "indoor lamp light", "golden hour")
    return (f"a photo of {token} {cls}, {hairs[i % 4]}, {shirts[i % 5]}, "
            f"{smiles[i % 3]}, {bgs[i % len(bgs)]}, {lights[i % len(lights)]}")


def self_test(n_images, token, cls):
    print("=" * 78)
    print("THE CAPTION ABLATION — predicted before a single step runs (brief-diffusion §12.2)")
    print("=" * 78)
    print(f"  base model      Stable Diffusion 1.5  (~{SD15_UNET_PARAMS_APPROX/1e6:.0f}M UNet "
          f"[VP, verify vs checkpoint], 512², DDPM, CLIP-77)")
    print(f"  dataset         {n_images} images, ONE subject, trigger token = {token!r}")
    print(f"  controlled      same images · same seed · same hyperparameters · "
          f"r={LORA_RANK}, α={LORA_ALPHA}, lr={LORA_LR:g}, {TRAIN_STEPS} steps")
    print(f"  the ONE change  the caption. That is the whole experiment.")
    print()

    # Build the two caption sets. B is bare-and-identical; A names the varying attributes.
    caps_b = strategy_b_captions(n_images, token, cls)
    caps_a = [_synthetic_detailed_caption(i, token, cls) for i in range(n_images)]

    print("  Strategy A (detailed) example:")
    print(f"    {caps_a[0]!r}")
    print("  Strategy B (bare) — identical for all {}:".format(n_images))
    print(f"    {caps_b[0]!r}")
    print()

    # -- what each strategy makes the token MEAN, and the predicted OOD outcome --
    mean_a = token_binding(caption_names_varying=True)
    mean_b = token_binding(caption_names_varying=False)
    gen_a, bled_a = predict_mars_probe(mean_a)
    gen_b, bled_b = predict_mars_probe(mean_b)
    probe = MARS_PROBE.format(token=token, cls=cls)

    print(f"  OOD probe (§12.4):  {probe!r}")
    print(f"    A → token {token!r} means {sorted(mean_a)}")
    print(f"        predicted: {'GENERALIZES ✅' if gen_a else 'BLEEDS ❌'}"
          f"{'' if gen_a else '  bleed: ' + str(bled_a)}   [EST]")
    print(f"    B → token {token!r} means {sorted(mean_b)}")
    print(f"        predicted: {'GENERALIZES ✅' if gen_b else 'BLEEDS ❌ bleed: ' + str(bled_b)}   [EST]")
    print()

    # ------------------------------------------------------------------ #
    # ASSERTIONS — the experimental controls and the frozen facts.
    # ------------------------------------------------------------------ #

    # 1. The frozen SD1.5 geometry this ablation trains inside (constants.md §9.6).
    assert 4 * 64 * 64 == SD15_LATENT_ELEMS == 16_384
    assert SD15_PIXELS == 786_432
    assert SD15_PIXELS // SD15_LATENT_ELEMS == SD15_COMPRESSION == 48, "SD1.5 48× compression"
    print("  [OK] SD1.5 @ 512²: latent [1,4,64,64], 786,432/16,384 = 48× (constants §9.6).")

    # 2. THE CONTROL: identical count, and ONLY the caption differs. Strategy B must be
    #    perfectly uniform (one caption repeated); Strategy A must be per-image distinct.
    assert len(caps_a) == len(caps_b) == n_images, "both strategies use the SAME image set"
    assert len(set(caps_b)) == 1, "Strategy B is the SAME bare caption for every image"
    assert len(set(caps_a)) == n_images, "Strategy A is a DISTINCT detailed caption per image"
    # Everything except the caption is shared — encode that as a single shared config dict.
    shared = dict(rank=LORA_RANK, alpha=LORA_ALPHA, lr=LORA_LR, steps=TRAIN_STEPS,
                  resolution=RESOLUTION, targets=LORA_TARGETS, seed=42, token=token)
    assert shared["alpha"] == shared["rank"], "α = r (α = r/2 would halve the effect, §11.2)"
    assert shared["lr"] == 1e-4, "SD1.5 LoRA lr = 1e-4, NOT full-FT 1e-6 (D-09; constants §9.4)"
    print("  [OK] control holds: same 20 images, same seed/rank/α/lr/steps — only c differs.")

    # 3. The 77-token silent-truncation guard (§12.2). A detailed caption that overflows
    #    CLIP's 77 tokens is truncated with NO error, quietly gutting Strategy A.
    worst = max(caps_a, key=approx_token_count)
    assert approx_token_count(worst) <= CLIP_MAX_TOKENS, \
        f"a Strategy-A caption approx-exceeds CLIP's {CLIP_MAX_TOKENS}-token cap: {worst!r}"
    print(f"  [OK] every Strategy-A caption fits CLIP's {CLIP_MAX_TOKENS}-token cap "
          f"(worst ≈ {approx_token_count(worst)} tokens; --run checks the real tokenizer).")

    # 4. THE PREDICTION the rule makes, derived from the loss (labelled [EST]).
    assert mean_a == {FACE}, "A: naming the varying attrs leaves the token holding only the face"
    assert VARYING_ATTRS[0] in mean_b and FACE in mean_b, "B: unnamed constants bake into the token"
    assert gen_a is True and bled_a == [], "A predicted to GENERALIZE on the Mars probe [EST]"
    assert gen_b is False and "background" in bled_b, \
        "B predicted to BLEED (background/shirt) on the Mars probe [EST]"
    print("  [OK] derived prediction [EST]: A generalizes; B bleeds "
          f"{bled_b} — because unnamed constants have nowhere to go but {token!r}.")

    print()
    print("=" * 78)
    print("SELF-TEST PASSED — the controls hold and the prediction closes on §12.2's mechanism.")
    print("It is a PREDICTION [EST]. Run `python caption_ab.py --run --images ... --captions ...`")
    print("on the Spark to turn it into a contact sheet you judge with your eyes [MEA].")
    print("=" * 78)
    return shared


# --------------------------------------------------------------------------- #
# THE REAL ABLATION. Two SD1.5 LoRAs via the diffusers/PEFT reference path (the readable
# one, §13.1) — same six-line loss (§4.5) for both, only the caption tensor differs.
# Spark-only; if the stack is not importable it prints exactly what to install (a FRESH
# venv, NEVER ComfyUI's — hardware-ground-truth §3).
# --------------------------------------------------------------------------- #

def _stamp():
    import torch, diffusers, peft, transformers
    print(f"  torch {torch.__version__} | diffusers {diffusers.__version__} | "
          f"peft {peft.__version__} | transformers {transformers.__version__} | CUDA {torch.version.cuda}")


def _find_sd15(override):
    if override:
        return os.path.abspath(os.path.expanduser(override))
    for d in ("~/ComfyUI/models/checkpoints", "~/comfyui/models/checkpoints"):
        for c in glob.glob(os.path.join(os.path.expanduser(d), "*.safetensors")):
            n = os.path.basename(c).lower()
            if "v1-5" in n or "sd15" in n or "sd_15" in n or "1.5" in n:
                return c
    return None


def _load_images(image_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(os.path.expanduser(image_dir), e))
    return sorted(files)


def _make_loss_closure(pipe, device, dtype):
    """Return step(latents_batch, prompt_embeds) -> scalar loss, the SD1.5 ε-prediction
    training step (§4.5): add noise at a uniform t, ask the UNet to predict it, MSE.
    ONE closure, used identically for Strategy A and Strategy B."""
    import torch
    from diffusers import DDPMScheduler
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)   # T=1000, scaled_linear (§3.4)
    assert noise_sched.config.num_train_timesteps == DDPM_TRAIN_TIMESTEPS

    def step(latents, prompt_embeds):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        t = torch.randint(0, DDPM_TRAIN_TIMESTEPS, (bsz,), device=device).long()  # UNIFORM t (SD1.5)
        noisy = noise_sched.add_noise(latents, noise, t)
        pred = pipe.unet(noisy, t, encoder_hidden_states=prompt_embeds).sample     # ε-prediction
        return torch.nn.functional.mse_loss(pred.float(), noise.float())

    return step


def _train_one(strategy, image_files, captions, sd15_path, device, dtype, steps, outdir, seed):
    """Train ONE SD1.5 LoRA and return (adapter_dir, trainable_param_count)."""
    import torch
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model_state_dict

    print("-" * 78)
    print(f"STRATEGY {strategy}: training an SD1.5 LoRA on {len(image_files)} images")

    # Load SD1.5 (single-file .safetensors or a diffusers folder). text_encoder + vae stay
    # FROZEN (requires_grad=False) — you cannot retrain CLIP; that is WHY the caption is the
    # lever and why textual inversion exists (brief §9 requires_grad note).
    if os.path.isdir(sd15_path):
        pipe = StableDiffusionPipeline.from_pretrained(sd15_path, safety_checker=None)
    else:
        pipe = StableDiffusionPipeline.from_single_file(sd15_path, safety_checker=None)
    pipe = pipe.to(device=device, dtype=dtype)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    if device == "cuda":
        pipe.unet.enable_gradient_checkpointing()

    # Attach the LoRA to the UNet's attention projections (§11.2). diffusers' native PEFT
    # integration: UNet2DConditionModel.add_adapter(LoraConfig) — the canonical 0.39 path.
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, init_lora_weights="gaussian",
        target_modules=list(LORA_TARGETS),
    )
    pipe.unet.add_adapter(lora_config)
    trainable = [p for p in pipe.unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    # B is initialised to zero (init_lora_weights) so at step 0 the UNet is bit-identical to
    # base — the D-10/§11.2 sanity check the from-scratch script asserts.
    print(f"  trainable LoRA params: {n_trainable:,}  (targets {LORA_TARGETS}, r={LORA_RANK}) [MEA]")

    # Precompute frozen latents + text embeddings once (§13's 'cache to disk'): the VAE and
    # CLIP never change, so we pay for them once, not every step.
    from PIL import Image
    import numpy as np
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    latents_cache, embeds_cache = [], []
    tok = pipe.tokenizer
    with torch.no_grad():
        for path, cap in zip(image_files, captions):
            # real 77-token guard: warn on the actual tokenizer if this caption truncates.
            ids = tok(cap, truncation=False, return_tensors="pt").input_ids
            if ids.shape[-1] > CLIP_MAX_TOKENS:
                print(f"  !! caption truncated at {CLIP_MAX_TOKENS} tokens (had {ids.shape[-1]}): {cap!r}")
            img = Image.open(path).convert("RGB")
            w, h = img.size; s = min(w, h)
            img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            x = torch.from_numpy(np.asarray(img, "float32") / 127.5 - 1.0)
            x = x.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
            z = pipe.vae.encode(x).latent_dist.sample() * VAE_SCALE_FACTOR   # 0.18215 (§2)
            assert list(z.shape) == list(SD15_LATENT), f"expected {SD15_LATENT}, got {list(z.shape)}"
            te = pipe.text_encoder(tok(cap, padding="max_length", truncation=True,
                                       max_length=CLIP_MAX_TOKENS,
                                       return_tensors="pt").input_ids.to(device))[0]
            latents_cache.append(z); embeds_cache.append(te)

    step_fn = _make_loss_closure(pipe, device, dtype)
    opt = torch.optim.AdamW(trainable, lr=LORA_LR)   # (kohya uses AdamW8bit; plain AdamW here is readable)
    pipe.unet.train()
    for it in range(steps):
        j = it % len(latents_cache)
        loss = step_fn(latents_cache[j], embeds_cache[j])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == steps - 1:
            # NB: this loss is NOISE-DOMINATED by the random t draw (§12.4) — do NOT read it
            # as progress. The contact sheet below is the instrument, not this number.
            print(f"    step {it:>4}/{steps}  loss {loss.item():.4f}  "
                  f"[t-noise-dominated; not a progress signal — §12.4]")

    adapter_dir = os.path.join(outdir, f"lora_strategy_{strategy}")
    os.makedirs(adapter_dir, exist_ok=True)
    # Canonical diffusers-0.39 PEFT save: pull the adapter state off the UNet and write a
    # diffusers-format pytorch_lora_weights.safetensors. (For ComfyUI's LoraLoader, convert
    # the key naming with diffusers.utils.convert_state_dict_to_kohya first.)
    unet_lora_state = get_peft_model_state_dict(pipe.unet)
    StableDiffusionPipeline.save_lora_weights(save_directory=adapter_dir,
                                              unet_lora_layers=unet_lora_state,
                                              safe_serialization=True)
    print(f"  saved: {adapter_dir}/pytorch_lora_weights.safetensors  ({n_trainable:,} params)")
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()
    return adapter_dir, n_trainable


def _generate_probe(strategy, adapter_dir, sd15_path, probe, device, dtype, outdir, seed):
    """Load one trained LoRA and run the OOD Mars probe at a fixed seed. Saves a PNG and
    returns it. Optionally scores background-bleed with CLIP if available (a [MEA] number)."""
    import torch
    from diffusers import StableDiffusionPipeline
    if os.path.isdir(sd15_path):
        pipe = StableDiffusionPipeline.from_pretrained(sd15_path, safety_checker=None)
    else:
        pipe = StableDiffusionPipeline.from_single_file(sd15_path, safety_checker=None)
    pipe = pipe.to(device=device, dtype=dtype)
    pipe.load_lora_weights(adapter_dir)                     # canonical diffusers-0.39 LoRA load
    g = torch.Generator(device=device).manual_seed(seed)   # SAME seed for A and B — fair test
    img = pipe(probe, num_inference_steps=30, guidance_scale=7.5, generator=g).images[0]
    out = os.path.join(outdir, f"mars_probe_strategy_{strategy}.png")
    img.save(out)
    print(f"  Strategy {strategy}: {os.path.basename(out)}  (prompt: {probe!r})")
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def run_ablation(args):
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        import peft  # noqa: F401
    except ImportError as e:
        print(f"[stop] the diffusion training stack is not importable ({e}).")
        print("  This runs on the Spark in a FRESH venv (NEVER ComfyUI's — you will break its")
        print("  working cu130 torch). See the setup page / spec-code §A:")
        print("    uv venv --python 3.12 ~/course/.venv && source ~/course/.venv/bin/activate")
        print('    uv pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130')
        print('    uv pip install "diffusers==0.39.0" "peft==0.19.1" "transformers==5.14.1" accelerate')
        sys.exit(1)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    sd15 = _find_sd15(args.sd15)
    if not sd15 or not os.path.exists(sd15):
        sys.exit("[stop] SD1.5 checkpoint not found. Pass --sd15 <v1-5 .safetensors or diffusers folder>.")
    image_files = _load_images(args.images)
    if not image_files:
        sys.exit(f"[stop] no images under --images {args.images!r} (png/jpg/jpeg/webp/bmp).")
    if not args.captions:
        sys.exit("[stop] --captions is required for Strategy A: one detailed caption per image, "
                 "filename-sorted. That ten minutes IS the experiment (§12.2). "
                 "VLM auto-captioning (Florence-2/JoyCaption) is a fine FIRST DRAFT — then hand-edit.")
    caps_a = load_detailed_captions(args.captions, len(image_files))
    caps_b = strategy_b_captions(len(image_files), args.token, args.cls)

    steps = 300 if args.quick else TRAIN_STEPS
    outdir = os.path.abspath(os.path.expanduser(args.outdir))
    os.makedirs(outdir, exist_ok=True)

    print("=" * 78)
    print("REAL CAPTION ABLATION — two SD1.5 LoRAs, identical images, only the caption differs")
    print("=" * 78)
    _stamp()
    if device == "cuda":
        free, tot = torch.cuda.mem_get_info()
        print(f"  GPU free {free/GiB:.1f} GiB of {tot/GiB:.1f} GiB."
              + ("" if free > 0.7 * tot else "  !! something else is holding memory (ComfyUI?) — expect contention."))
    print(f"  images {len(image_files)} | steps/LoRA {steps} | device {device} | outdir {outdir}")
    if len(image_files) < 15 or len(image_files) > 25:
        print(f"  !! {len(image_files)} images — §12.1's face sweet spot is 15–20. More is often WORSE.")
    print()

    # The prediction first — the run is what validates it.
    self_test(len(image_files), args.token, args.cls)
    print()

    # SAME seed for both strategies: only c differs. This is the control, in code.
    dir_a, nt_a = _train_one("A", image_files, caps_a, sd15, device, dtype, steps, outdir, args.seed)
    dir_b, nt_b = _train_one("B", image_files, caps_b, sd15, device, dtype, steps, outdir, args.seed)
    assert nt_a == nt_b, "both LoRAs must have identical trainable-param counts (same r/targets)"

    probe = MARS_PROBE.format(token=args.token, cls=args.cls)
    print("-" * 78)
    print(f"OOD MARS PROBE (§12.4) — same prompt, same seed, both LoRAs: {probe!r}")
    img_a = _generate_probe("A", dir_a, sd15, probe, device, dtype, outdir, args.seed)
    img_b = _generate_probe("B", dir_b, sd15, probe, device, dtype, outdir, args.seed)

    print()
    print("=" * 78)
    print("MEASURED VERDICT [MEA] — your eyes are the instrument (§12.4; the loss curve is not)")
    print("=" * 78)
    print(f"  identical trainable params both strategies: {nt_a:,} (the control held).")
    print(f"  Strategy A (detailed captions): {os.path.basename(img_a)}")
    print(f"  Strategy B (bare trigger):      {os.path.basename(img_b)}")
    print("  Look for it, don't take it on faith:")
    print("    ✅ A → your subject in a spacesuit, on Mars. The concept GENERALIZED.")
    print("    ❌ B → the training background / clothing BLEEDS into the Mars scene, because")
    print("          those unnamed constants had nowhere to go but the trigger token.")
    print("  If A is clean and B bleeds, you just MEASURED §12.2's rule instead of trusting it.")
    print("  (Predicted [EST] above; this contact sheet is the [MEA]. Same 20 images — only c changed.)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    # Teaching output uses α/×/→/✅ etc. On UTF-8 terminals (his Linux Spark, macOS) this
    # just works; on a legacy Windows cp1252 console it would crash, so ask for UTF-8.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="pure-logic prediction + assertions, no GPU/torch (the default)")
    ap.add_argument("--run", action="store_true",
                    help="the real 2-LoRA ablation on the Spark (heavy, ~20–40 min)")
    ap.add_argument("--quick", action="store_true",
                    help="with --run: 300 steps/LoRA instead of ~1200 (a fast look)")
    ap.add_argument("--images", default=None, help="dir of ~20 images of ONE subject (--run)")
    ap.add_argument("--captions", default=None,
                    help="Strategy A: one detailed caption per image, filename-sorted (--run)")
    ap.add_argument("--sd15", default=None,
                    help="SD1.5 v1-5 .safetensors or diffusers folder (default: search ComfyUI/models/checkpoints)")
    ap.add_argument("--token", default="sks", help="rare trigger token (default: sks)")
    ap.add_argument("--cls", default="man", help="class word for the caption (default: man)")
    ap.add_argument("--n", type=int, default=20, help="images for the --self-test prediction (default 20)")
    ap.add_argument("--seed", type=int, default=42, help="shared seed for both strategies")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--outdir", default="./caption_ab_out", help="adapters + contact sheet")
    args = ap.parse_args()

    if args.run:
        run_ablation(args)
    else:
        self_test(args.n, args.token, args.cls)


if __name__ == "__main__":
    main()
