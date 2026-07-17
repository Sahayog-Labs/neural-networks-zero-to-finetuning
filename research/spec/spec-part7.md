# BUILD SPECIFICATION — Part VII: CAPSTONE (pages 63–65) + index.html

**Scope.** The three closing pages of the ANN course — the shared capstone *after* both tracks and the GRPO/RLVR capstone (page 52) are done — plus the course home page. These are the landing zone: the four recurring threads all terminate here, the two tracks structurally rejoin, and the course's actual destination is cashed in (constants.md §10 meta-thread: *"a learner who can predict his own hardware's behaviour from first principles before running anything"*). **Final numbers 63–65 map to D-21's 62–64** (the GRPO renumber shifts everything ≥52: GRPO capstone = 52, diffusion 53–62 = D-21 52–61, this capstone 63–65 = D-21 62–64). Each entry states both numbers.

**How to read this spec.** Every page entry is fully self-contained: a builder implements it 1:1 without opening any brief. Numbers are quoted from `constants.md`/`hardware-ground-truth.md` with their confidence tags; **print the tag to the learner wherever it is [INF]/[EST]/[MEA]/[MEA-DEV]** (notation anti-pattern #23). Retired numbers are named so no builder reintroduces them.

**Template contract (from `TEMPLATE-example.html` + `assets/course.js` / `nn.js` / `viz.js`).** Page shell: `<body data-page="NN-slug.html">` → `#layout > main#content > div.inner`; `<span class="kicker">`, `<h1>`, `<p class="subtitle">`. Scripts at end of body in order: `course.js`, `course-map.js`, `nn.js`, `viz.js`, then the page IIFE inside one `DOMContentLoaded`. Math: `$inline$` / `$$display$$` (KaTeX pre-wired; double-escape backslashes inside JS quiz strings, e.g. `"\\sqrt"`). Boxes: `.box key|rule|warn|worked|try` each with `<span class="box-title">`; `.deepdive` collapsible with inner `<div>`; `.milestone`; `.track-banner` and `.track-llm|.track-diffusion` modifiers on any `.box`/`.demo`/banner. Demo DOM: `.demo > .demo-head (+ .demo-tag) > .demo-body > canvas[height] + .controls[id] + .readout[id]`. Primitives: `makeCtrl(parent,{label,min,max,step,value,unit,fmt})→{input,get(),onInput(fn)}`; `new Plot(canvas,{xmin,xmax,ymin,ymax,xlog,ylog,xlabel,ylabel})` → `.clear/.grid/.trace(xs,ys,color,width)/.hline/.vline/.label`; `renderQuiz(id,[{q,opts,a,why}|{q,num,tol,unit,why}],title)`; `eng(v,unit,digits)`; `vizUtils.cssVar`; `new Heatmap/NetGraph/Surface3D/TensorViz/Timeline`; `hl.block`. Engine (`window.NN`): `Value, Mat, MLP, Trainer, SGD, Adam, RNG, makeDataset, gradCheck, worked221, attention, softmax, layerNorm, rmsNorm, bceWithLogits, crossEntropy`. **Demos compute real math live — nothing is pre-baked** (anti-pattern #11).

**Build-model note (O/S).** D-21's O/S column is not reproduced in the frozen files I was given (same gap flagged by spec-part4). **Treat all three pages as `O` (original full build)** unless the architect's D-21 master overrides. A defensible alternative the architect may prefer: 65 as `S` (synthesis/send-off, one reflective demo). Flagged in the return summary.

**Part-wide notation lock.** This part re-uses every symbol from the trunk and both tracks; it introduces none. Load-bearing here: $I = S_{\text{fwd}}$ (arithmetic intensity = tokens/forward, bf16; notation §4.1, constants §6.5) · $I^\star$ ridge · $16P$ ledger · KV $= 2LH_{kv}d_{head}bSB$ (note $H_{kv}$, **never** $H$; notation §3, constants §4) · $H_{kv}$ = KV heads · $d_{head}$ · GQA ratio $H/H_{kv}$. **The measured-on-device facts outrank everything** (`hardware-ground-truth.md`): usable `MemTotal` = **121.6875 GiB** [MEA-DEV], carveout **6.3125 GiB (4.9%)**, sm_121 with no sm_121 binary/PTX, 273 GB/s, dense BF16 ceiling **unresolved (~62 TF FP32-acc vs ~125 TF FP16-acc)**.

**Recurring threads — all four TERMINATE in this part** (tag every beat `[THREAD:… beat k/n]`, notation §10). This is their last appearance; each page must *announce the thread is closing* and name the prior beat it reopens:
- **The dot product (the atom):** neuron → matmul → attention score → $BA$ → and finally the same $\langle q,k\rangle$ inside his FLUX.2 text encoder's attention.
- **TN-1 (the object, the spine):** 9-param hand arithmetic → JS demo → PyTorch → autograd → one attention head → LoRA target → **now: the same block, scaled to a 40-layer Mistral-3, is the front half of his image model.**
- **The chain rule (the verb):** derivative → backprop → autograd graph → cached activations → score → **the one verb that ran under every page.**
- **The memory ledger (the reality check):** 16 B/param → activations → KV → quantization → LoRA → **"does it fit on *my* box?" answered by measurement, for the last time.**

---

# 63 — 63-predict-your-box.html  (D-21: 62)

**Title:** "The Whole Machine: Predicting Your Own Box" · **Part VII — Capstone** · build **O** · no track class (trunk-level synthesis; tint the two *verdict* callouts `.track-llm` / `.track-diffusion`).

**Learning objectives**
1. Assemble the memory ledger and the roofline into one predictive model of the learner's DGX Spark.
2. Predict, before running, whether a given workload fits in memory and whether it is compute- or bandwidth-bound.
3. Explain why the *same machine* is weak for LLM decode and strong for diffusion sampling, from one formula.
4. State honestly which of the page's numbers are measured, inferred, or must still be measured — and why the course measured rather than asserted them.

**PREDICT (opens the page; consolidation beat lands at the unified demo).**
> "You have done this piecemeal for 60 pages. Now, cold, before touching the demo: on *your* Spark, (a) does a full fine-tune of Qwen3-8B fit? (b) will LLM decode at batch 1 be fast or slow? (c) will FLUX sampling be fast or slow? Write down three yes/no answers and one sentence of *why* for each. The rest of the page checks them."

**Section outline**
- **Framing (one sentence, inline).** "The course's real destination was never 'operate the tools.' It was *this*: you can now predict what your hardware will do before you run it. This page collects the receipts." (constants §10 meta-thread — verbatim-worthy.)
- **Ledger recap, as a formula not a re-teach (`.box key`, NOT a re-explanation — anti-pattern #20).** State only the closed form and let the demo do the work: state memory $= b_W P + k_O N_t + \text{grads} + \text{activations[EST]}$. Full-FT anchor: $8{,}190{,}735{,}360 \times 16\,\text{B} = 131{,}051{,}765{,}760\,\text{B} = \mathbf{131.05\,GB} = \mathbf{122.05\,GiB}$ [DER, constants §2.2]. **Activations are ALWAYS a separate [EST] line, both sides of every comparison** (D-03; constants §2.4). Escape ladder verbatim from constants §2.3: LoRA r=16 all-linear **17.08 GB / 15.91 GiB**; QLoRA (NF4, DQ on) **~4.93 GB state / base 4.225 GB (3.935 GiB)** [DER, constants §3]. Reduction sentence to ship verbatim (D-05): *"the trainable state shrank by 187×, which is exactly 8,190,735,360 ÷ 43,646,976, because optimizer state is linear in trainable parameters."*
- **The measured wall (`.box worked`, [MEA-DEV]).** 122.05 GiB needed vs **121.6875 GiB** measured `MemTotal` = **−0.36 GiB, does not fit, by 0.3%, before activations** (hardware-ground-truth §2; constants §6.8). The carveout villain, stated as the generalizable lesson: **"published capacity is never usable capacity — the spec sheet is off by 4.9% before you start."** Print the three-number distinction he must not conflate: Published 128 GB / `MemTotal` 121.69 GiB / `MemAvailable` with ComfyUI up **19.41 GiB** (hardware-ground-truth §2.2). **Do not assert his budget on the page — the demo has him measure** (D-04).
- **The roofline recap, as the ONE formula (`.box rule`).** $I = \dfrac{2N\,S_{\text{fwd}}}{2N} = \boxed{S_{\text{fwd}}}$ FLOP/byte (dense, bf16; constants §6.5, D-15). Ridge $I^\star = P_{\text{peak}}/BW$. **Print the honest range, not a fake fact:** ridge = **227** (working, from ~62 TF FP32-accumulate) **or 458** (if the ceiling is ~125 TF FP16-accumulate) — **[INF], genuinely unresolved** (constants §6.4). Both verdicts below hold under either ridge.
- **The opposite verdicts (two callouts, tinted).** `.box track-llm`: LLM decode batch 1 → $S_{\text{fwd}}=1$ → $I=1$ → **227–458× below the ridge → bandwidth-bound → the Spark is weak.** `.box track-diffusion`: diffusion sampling, 4096 latent tokens → $I=4096$ → **9–18× above the ridge → compute-bound → the Spark is strong.** Same machine, same ridge, opposite sides of $I^\star$ (D-15). **The measured confirmation [VP]:** decode 20.5 → 368 tok/s (18×) batch 1→32 while **prefill is flat** (7,991 → 7,949) — the roofline, measured, on his box (constants §6.7).
- **What is still [MEA] (`.box warn`).** Name the two numbers the course refused to fabricate: the **dense BF16 ceiling** (62 vs 125 TF — the SDXL BF16 datapoint 69.7 TF/s leans 125; community GEMM reports lean 62; constants §6.3) and **achieved bandwidth** (273 GB/s is theoretical peak). "A course that told you these would be lying — you measure them, and the measurement fixes the ridge." Points at `09_spark_capability_probe.py`.
- **Deepdive (collapsible): "Why 0.65 and not 1.0 — the decode heuristic's honest error bars."** decode tok/s $\approx 0.65 \times BW/(\text{weight bytes read per token})$; the byte count is **file − embedding table** (§6.6); the MoE punchline verbatim: *"if your heuristic predicts more than 100% of peak bandwidth, your model is an MoE."* Qwen3-8B bf16 prediction: ceiling **16.67 tok/s**, expect **~10.8** (×0.65) — *say which is which* (constants §6.7; the ~19.5 figure is RETIRED). Collapsible because the roofline verdict above does not depend on it.

**Interactive demo — "Predict-your-box console" (the page's spine; unifies the p.18 ledger demo and the roofline demo, but PREDICTIVE).**
- **Primitives:** `makeCtrl` sliders + a workload `<select>`; one `Plot` (roofline, log–log) + an HTML `.readout` ledger table. Reuses `NN` memory arithmetic; no faked numbers.
- **Controls:** workload dropdown {Full-FT AdamW, Full-FT 8-bit Adam, LoRA r=16, QLoRA r=16, LLM decode b=1, LLM decode b=32, LLM prefill S=2048, Diffusion sample 4096-tok}; model preset {Qwen3-8B (default), Qwen3-0.6B, Llama-3.3-70B} filling $P,L,H,H_{kv},d_{head},d_{ff},V$ from constants §1.1/§1.4; **a "measure my box" button** that (in the shipped page) reads a pre-filled `121.6875` GiB but whose *label* tells him to replace it with his own `torch.cuda.mem_get_info()` output; ceiling toggle {62 TF, 125 TF} so he sees the ridge move 227↔458; slider $S_{\text{fwd}}$ (1 → 40960, log) for the decode/prefill cases.
- **Exact math the JS implements (show the equation with live values substituted — anti-pattern #12):** memory = $b_W P + k_O N_t$ (+ grads 2P, + activations shown as a labelled [EST] band 2–6 GB); $I = S_{\text{fwd}}$; ridge $= P_{\text{peak}}/(273\text{e}9)$; verdict = compute-bound iff $I > I^\star$. Plot: attainable FLOP/s $= \min(P_{\text{peak}},\,BW\times I)$ vs $I$ (log–log), a `.vline` at the workload's $I$, an `.hline` at $P_{\text{peak}}$, and a labelled point. Ledger `.readout` prints GiB needed vs the measured ceiling with a red FITS/OOM verdict.
- **The aha:** every number he built across the course lives in one console, and it **predicts before it runs** — the full-FT bar overshoots his measured line by a hair, decode sits far left of the ridge, diffusion far right. His three PREDICT answers get graded here.

**Code artifact (`.box try`).** `code/predict_your_box.py` (**builder creates**): step 1 `free,total = torch.cuda.mem_get_info(); print(free/2**30, total/2**30)` (constants §6.8 verbatim); step 2 compute 122.05 GiB against *his* `total`; step 3 a big bf16 GEMM timed against both 62/125 TF ceilings in both accumulate modes (this is `09_spark_capability_probe.py`'s core). "You measured the wall, the ceiling, and the ridge — the three numbers the course refused to hand you." **Must install nothing into ComfyUI's venv** (hardware-ground-truth §3).

**Quiz (7; renderQuiz, ≥2 numeric).**
1. Numeric — Qwen3-8B full-FT state in GiB. `num: 122.05, tol: 0.1, unit: "GiB"`. Why: $8.19\text{e}9\times16/2^{30}$; cite [DER].
2. Numeric — his measured usable `MemTotal`. `num: 121.6875, tol: 0.01, unit: "GiB"`. Distractor rationale in `why`: 128 GiB is marketing; the 6.3125 GiB carveout is real [MEA-DEV].
3. MC — "LLM decode at batch 1 is slow on the Spark because…" Correct: $I=1 \ll$ ridge, bandwidth-bound. Distractor "the GPU is too slow / low TFLOPs" → catches the compute-bound misconception the roofline exists to kill.
4. MC — "Same Spark, why is diffusion sampling *strong*?" Correct: $I=4096 \gg$ ridge, compute-bound. Distractor "diffusion needs less memory" → catches conflating memory with the roofline.
5. Numeric — ridge if the BF16 ceiling turns out to be 62 TF. `num: 227, tol: 5, unit: "FLOP/byte"`; `why`: $62\text{e}12/273\text{e}9$ [INF] — and it would be 458 at 125 TF.
6. Diagnose — "LoRA r=16 'only trains 43 M params', so its memory footprint is ~0.6 GB total." Ruled out: **17.08 GB total** — LoRA does not shrink the base model or activations; only *optimizer state* shrinks 187× (D-05, constants §2.4). Distractors are the retired 224×/417× and "14.3 GB".
7. Transfer — "Which two numbers on this page are you *not allowed to trust from the course*, and why?" → the dense BF16 ceiling and achieved bandwidth; both [MEA]; unpublished / theoretical-peak (constants §10).

**Thread touchpoints.** `[THREAD:memory-ledger beat FINAL]` — reopen p.18 (16 B/param first stated) and p.40 (LoRA 17.08 GB): "the ledger you started on p.18 pays out here, against a number you measure." `[THREAD:chain-rule beat …]` — the activations line in the ledger *is* why backprop caches (reopen p.5 chain rule / cached-activations→memory, and p.14 backprop): "the VRAM the chain rule eats is the [EST] band in this console." `[THREAD:dot-product]` — $I=S_{\text{fwd}}$ comes from $2N$ FLOP over $2N$ bytes, the same matmul atom.

**Cross-references.** ← p.18 (memory ledger), ← p.40 (LoRA arithmetic), ← the roofline/hardware page that owns D-15 (**final number in the 44–52 landing zone — match its ridge 227/458 and the 62-vs-125 TF framing exactly**), ← p.52 (GRPO capstone, the last thing that ran). → p.64 (the rejoining), → p.65 (the send-off).

---

# 64 — 64-tracks-rejoin.html  (D-21: 63)

**Title:** "The Tracks Grew Back Together" · **Part VII — Capstone** · build **O** · **the merge point** — carry a `.track-banner` styled with *both* track colors (a split banner), no single track class on `<body>`.

**Learning objectives**
1. Recognize the FLUX.2 text encoder as a 40-layer multimodal Mistral-3 — every component the LLM track taught (GQA, RoPE, the transformer block, a KV cache).
2. Verify this on the learner's *own disk* with `cat`/`python` one-liners, not on faith.
3. Derive the Mistral-3 encoder's KV-per-token in the frozen notation and compare it to Qwen3-8B's 144 KiB/token.
4. State the course's central structural claim precisely: *the machine didn't change; the target did* — and see it as a fact on his SSD, not rhetoric.

**PREDICT (opens the page).**
> "Your FLUX.2 model has to *read the prompt* before it can paint. Before you look: what kind of network do you think does that reading — a bag-of-words embedder, a CNN, an RNN, or a full transformer language model? And roughly how big, relative to the 32B image model itself? Commit, then open the config on your disk."

**Section outline**
- **The reveal (`.box key`, [VP], measured-on-disk).** From `model_index.json` on *his* FLUX.2-dev: `"text_encoder": ["transformers", "Mistral3ForConditionalGeneration"]` (hardware-ground-truth §4.1). **His image model's prompt reader is a Mistral — a 24B VLM, roughly twice the size of the entire FLUX.1 model, just to read the prompt** (constants §9.6; D-17). "The tracks grew back together" stops being a metaphor: the LLM track he finished **is the manual for the front half of the diffusion model he just fine-tuned** (D-17 beat 3, ratified verbatim).
- **The component-by-component match (`.box rule`, a table).** Read from `text_encoder/config.json` (hardware-ground-truth §4.1) — every row is something a trunk/LLM page taught:

  | Mistral-3 encoder | Value [VP] | Where the course taught it |
  |---|---|---|
  | hidden_size $d$ | **5120** | p.2 weights/layers; §4.2 notation |
  | num_hidden_layers $L$ | **40** | the transformer block |
  | num_attention_heads $H$ | **32** | attention page |
  | num_key_value_heads $H_{kv}$ | **8** → **GQA 4×** | GQA (same 4× as Qwen3-8B!) |
  | $d_{head}=5120/32$ | **160** | shape ribbon |
  | intermediate_size $d_{ff}$ | **32768** | FFN / SwiGLU |
  | vocab_size $V$ | **131072** | tokenization page |
  | rope_theta | **1e9** | RoPE page |
  | dtype | **bf16** | mixed-precision page |
  | + `vision_config` + `PixtralProcessor` | — | **multimodal, not text-only** — it can read *images* as prompt too |

- **The GQA gift (inline).** Qwen3-8B is 32 heads / 8 KV; Mistral-3 is **also** 32 / 8. "The exact GQA ratio you computed on p.[attention] recurs, unchanged, inside your image model." (constants §1.1 vs §4.1.)
- **Worked example — the encoder's KV cache (`.box worked`, [DER] from the VP-on-disk config).** Same formula, note $H_{kv}$ not $H$ (constants §4): per token per layer $= 2H_{kv}d_{head}b = 2\times8\times160\times2 = \mathbf{5120\,B = 5\,KiB}$; all 40 layers $= 5120\times40 = 204{,}800\,\text{B} = \mathbf{200\,KiB/token}$. **His image model's text encoder carries a *heavier* per-token KV than the Qwen3-8B anchor (200 vs 144 KiB/token).** Symbol Ledger required (>3 symbols).
- **The joint-attention reveal (`.box worked`, [VP]).** The Flux2 transformer's `joint_attention_dim` = **15360**, and $15360 / 5120 = \mathbf{3}$ — the DiT reads a conditioning width that is exactly **3 × the Mistral hidden size** (hardware-ground-truth §4.1). And `in_channels` = **128** = $32 \times 2^2$ (32 latent channels × 2×2 patch packing) — verify the multiply live (constants §9.6). The cross-attention that binds text to image (taught mask-agnostically in the trunk, D-14) takes $K,V$ from *this* Mistral encoder — a different modality, exactly the first-class cross-attention the diffusion track needed.
- **The flow-matching vindication (inline).** His FLUX.2 scheduler is `FlowMatchEulerDiscreteScheduler` (hardware-ground-truth §4.1) — the flow-match Euler the diffusion track derived, on his disk. `num_train_timesteps=1000` is **vestigial** DDPM-tooling indexing; flow matching is continuous in $t\in[0,1]$ (notation §6 — have him note it, don't let him think it's a 1000-step chain).
- **Misconception (`.box warn`).** *"The text encoder is just a CLIP-style embedding lookup / a bag of words."* No — it is a full 40-layer autoregressive-class transformer (multimodal), with attention, RoPE, GQA and a KV cache. A learner who thinks Q/K/V are "three memories" cannot understand cross-attention and therefore cannot understand conditioning at all (D-14, the highest-stakes trunk ask). This page is where that misconception dies on his own hardware.
- **Deepdive (collapsible): "FLUX.1 vs FLUX.2 — why the encoder ballooned."** FLUX.1-dev: T5-XXL 4.7B + CLIP-L, $d=3072$, guidance-distilled. FLUX.2-dev: **32B** MMDiT + **Mistral-3 24B VLM**, 32-ch VAE $f=8$, 6.0× compression, 4096 tokens after 2×2 patchify (width 128) (constants §9.6, all [VP]). The trend: image models put a whole language model in the front. Optional — the reveal above stands without it.

**Interactive demo — "Three blocks, one architecture" (structural diff).**
- **Primitives:** `new NetGraph` (or side-by-side `Heatmap` shape ribbons) rendering three transformer blocks; a `<select>` to overlay; `.readout` computing the KV/param arithmetic live via `NN` helpers.
- **Controls:** dropdown {Qwen3-8B block, FLUX.2 Mistral-3 encoder block, FLUX.2 DiT single-stream block}; a $S$ (sequence length) slider (1 → 40960, log) driving a live KV-cache readout for each; a "highlight" toggle {residual stream, GQA fan-out, RoPE positions, cross-attention $K,V$ source}.
- **Exact math the JS implements:** for the selected block, draw the shape ribbon with its real dims ($d,H,H_{kv},d_{head},d_{ff}$ from the tables above), animate the GQA fan-out (8 KV heads serving 32 query heads — the same picture for Qwen3 and Mistral-3), and compute KV bytes $=2LH_{kv}d_{head}bS$ live (144 KiB/token Qwen3 vs 200 KiB/token Mistral-3 at $S=1$). For the DiT block, show cross-attention pulling $K,V$ from the 15360-wide (=3×5120) conditioning tensor. All numbers recomputed from sliders, none baked.
- **The aha:** flip between his LLM anchor and his image model's prompt reader and the block is *the same object at different scale* — the TN-1 spine, grown to 40 layers of Mistral. `[THREAD:TN1 beat FINAL]`.

**Code artifact (`.box try`).** `code/verify_flux2_encoder.py` (**builder creates**): loads `~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/*/text_encoder/config.json` off *his* disk, prints `hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, rope_theta`, computes the 200 KiB/token KV, and prints it **side-by-side with Qwen3-8B's 144 KiB/token**. Include the two shell one-liners inline (no elided imports — anti-pattern #18):
```bash
D=~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots
cat $D/*/model_index.json | python3 -c "import json,sys; print(json.load(sys.stdin)['text_encoder'])"
# → ['transformers', 'Mistral3ForConditionalGeneration']
cat $D/*/text_encoder/config.json | python3 -c "import json,sys; c=json.load(sys.stdin); t=c.get('text_config',c); print(t['hidden_size'], t['num_hidden_layers'], t['num_attention_heads'], t['num_key_value_heads'])"
# → 5120 40 32 8
```
(Note to builder: FLUX.2's `Mistral3ForConditionalGeneration` config nests the LM fields under `text_config` alongside `vision_config`; the `c.get('text_config',c)` fallback handles both layouts. Have him print the full config first if a key is missing — do not hard-fail.)

**Quiz (7; renderQuiz, ≥2 numeric).**
1. MC — "What is FLUX.2's text encoder?" Correct: a 40-layer multimodal Mistral-3 transformer. Distractors: CLIP embedder / T5 only / a bag-of-words — each a real belief about how prompts get read.
2. Numeric — Mistral-3 encoder $d_{head}$. `num: 160, tol: 0, unit: ""`; `why`: $5120/32$.
3. Numeric — its KV per token, all 40 layers. `num: 200, tol: 1, unit: "KiB"`; `why`: $2\times8\times160\times2\times40 / 1024$ [DER]. Distractor value 800 KiB catches using $H=32$ instead of $H_{kv}=8$ (the field's #1 KV error, constants §4).
4. Numeric — `joint_attention_dim` ÷ hidden_size. `num: 3, tol: 0, unit: ""`; `why`: $15360/5120$ [VP].
5. MC — "Qwen3-8B and Mistral-3 both have 32 heads and 8 KV heads. That ratio means…" Correct: 4× GQA — 4 query heads share each KV head, saving 4× on the KV cache. Distractor "4× more compute".
6. MC — "The scheduler on his disk is `FlowMatchEulerDiscreteScheduler` with `num_train_timesteps=1000`. That means…" Correct: flow matching, continuous $t\in[0,1]$; the 1000 is vestigial DDPM indexing. Distractor "a 1000-step DDPM chain" (notation §6).
7. Transfer — "State the course's thesis in one sentence, using the object on this page." Model answer: *the same transformer block, trained to predict tokens, is an LLM; trained to read a prompt for an image model, it's a Mistral text encoder — the machine didn't change, the target did* (D-17).

**Thread touchpoints.** `[THREAD:TN1 beat FINAL]` — reopen p.6 (TN-1 by hand) and the attention recast: "the 9-param network you ran with a pencil, grown to 40 Mistral layers, is reading your prompts." `[THREAD:dot-product beat FINAL]` — the $\langle q,k\rangle$ atom is inside this encoder's every attention row. `[THREAD:memory-ledger]` — the encoder has its own KV cache (200 KiB/token), the ledger's reach extends into the image model.

**Cross-references.** ← the fork reveal page (D-17 beat 2, "the machine didn't change, the target did" — final number in the fork zone; match its framing), ← the trunk attention page (mask-agnostic, cross-attention first-class, D-14), ← the RoPE and GQA pages (LLM track), ← the diffusion track's MMDiT/VAE page (page 62 = D-21 61 — **the diffusion finale must hand off to this reveal; name FLUX.2's Mistral encoder as the cliffhanger into 64**), ← p.63 (KV formula). → p.65 (the send-off).

---

# 65 — 65-where-next.html  (D-21: 64)

**Title:** "Where This Goes Next" · **Part VII — Capstone** · build **O** (architect may prefer **S**) · `milestone: true` (the final flag) · no track class.

**Learning objectives**
1. Situate what the learner built inside the 2026 field: the tracks converged *architecturally* and diverged *economically*, in the same year.
2. Name the honest state of the tooling he'd reach for next (what is maintained, what is dying, what is containerized).
3. Recall the meta-thread: at every checkpoint he predicted his own hardware and was right.
4. Leave with a concrete, honest "what to do next on this box" using models already on his disk.

**PREDICT (opens the page — the last one).**
> "One trend prediction before the epilogue: over the next two years, do image models get *bigger* (like LLMs did) or *smaller*? Bet one way, then read — the 2026 data is already surprising."

**Section outline**
- **The convergence, then the divergence (`.box key`, D-17).** Architecturally the tracks grew back together (p.64). **Economically they split in 2026:** LLMs are still scaling; **image models are consolidating.** The evidence, all [VP] (constants §9.6): **Z-Image-Turbo, 6B, 8 NFE, Apache-2.0, is #1 open-weights on the AA Image Arena — above FLUX.2-dev (32B).** FLUX.2-**klein-4B** runs **4 steps at guidance 1.0 in ~13 GB**. "A 6B model beat a 32B model, and a 4B model runs on a phone-class budget — in the same year the LLM world was still stacking parameters." Draw this deliberately (D-17): *converged in architecture, diverged in economics.* Better than a tidy "everything is one thing."
- **The tooling, honestly (`.box rule`).** What he'd actually reach for, with the honest caveats (constants §7, hardware-ground-truth §3): **torchtune is wound down** — the best-documented DGX Spark fine-tune (Llama-3.1-8B, seq 16,384, ~8 h/epoch, ~80% of 128 GB) runs on a **dying** library; `torchforge` is *not* a drop-in successor (experimental, RL/agentic). **vLLM works on sm_121 — via the `vllm/vllm-openai:cu130-nightly` container, pinned by digest** (the PyPI aarch64 wheel is CUDA-12 and fails; D-10j). **bitsandbytes ships native sm_121 only into the aarch64+CUDA-13 wheel** — the Spark is *better* served than an x86 box (constants §6.9). The real trap is narrower than "aarch64 is painful": **any package pinning `libcudart.so.12` fails at import** on CUDA-13-only DGX OS. Pin versions (constants §7): torch 2.13, transformers 5.14.1, peft 0.19.1, **trl ≥1.8,<2**, diffusers 0.39, bitsandbytes 0.49.2 — and **install into a fresh venv, never ComfyUI's** (hardware-ground-truth §3).
- **The meta-thread payoff (`.box worked`, constants §10 — the course's actual destination).** Walk the receipts: artifact #1 he counted his tokens; #5 he computed his memory; from the fine-tuning table he predicted his wall-clock; #11 he ran it and **the prediction was right**; then #2/#14 he predicted his serving speed from the roofline and **that was right too.** "A learner who can predict his own hardware's behaviour from first principles before running anything — that was the destination. You are there."
- **What is still yours to measure (inline honesty beat).** The dense BF16 ceiling (62 vs 125 TF) is *still unresolved* even now (constants §6.3) — "the course's biggest open lab is yours to close; whichever it lands at, understanding *why* teaches accumulate precision better than any paragraph." This is not a loose end to hide; it is the final lesson that measurement outranks assertion.
- **A concrete next step on his box (`.box try` framing, not a code artifact).** He already has the material: **his own LoRAs in `~/ComfyUI/models/loras/` (12 GB, 9 files)**, FP8 quantized models, Wan 2.2 I2V / LTX video pipelines (hardware-ground-truth §5). Suggest: (1) train a Qwen3-8B LoRA on his own writing (17.08 GB, ~12 min — constants §2.3/§6.7); (2) re-render one of his `bench_baseline/` FLUX jobs and check the wall-clock against the roofline he built on p.63; (3) run `bench_spark.py` (FLUX.1-dev, 28 steps, bf16 vs FP4, against both rooflines — constants §9.6, D-16). "Nothing here is a toy — every model is one you already own."
- **The farewell (`.milestone` + closing prose).** Short, earned, second person: *"You wrote a `Value` class and made it do calculus. You trained a 9-parameter network with a pencil and watched the loss go down. You read a parameter count off a checkpoint and it matched to the byte. You measured a wall on a machine that says it has memory it doesn't. And your image model's prompt reader turned out to be a Mistral. The machine never changed. Only ever the target did."*

**Interactive demo — "The 2026 map: size vs. capability" (one reflective, real-data plot).**
- **Primitives:** one `Plot` (scatter, log-x on params); optional `new Timeline` strip for "your journey."
- **Controls:** axis toggle {arena rank, steps/NFE, VRAM to run}; a "class" filter {LLM, image}. A checkbox to overlay "your box's 121.69 GiB ceiling" as an `.hline` on the VRAM axis.
- **Exact math/data the JS plots (all [VP], constants §9.6):** points — Z-Image-Turbo (6B, 8 NFE, arena #1), FLUX.2-dev (32B), FLUX.2-klein-4B (~4B, 4 steps, ~13 GB), FLUX.1-dev (12B, 28 steps); LLM side — Qwen3-0.6B/1.7B/8B, Llama-3.3-70B (as size references). No fabricated ranks: plot only the arena facts constants §9.6 states ("Z-Image > FLUX.2-dev"); where a rank is unknown, place the point on the size axis only and label it. The VRAM overlay uses the escape-ladder numbers from p.63.
- **The aha:** the image-model points trend *down-left* (smaller, fewer steps, still winning) while the LLM points trend *up-right* — the divergence, drawn from real 2026 numbers, on the same axes.

**Quiz (5; renderQuiz, ≥1 numeric — kept light; it is the send-off).**
1. MC — "In 2026 the two tracks…" Correct: converged architecturally, diverged economically. Distractor "became the same field" (the tidy-but-wrong answer D-17 warns against).
2. Numeric — the open-weights arena leader's size, in billions of params. `num: 6, tol: 0, unit: "B"` (Z-Image-Turbo, constants §9.6). Distractor context: it beat a 32B model.
3. MC — "You want throughput serving on the Spark. The honest path is…" Correct: the `vllm/vllm-openai:cu130-nightly` container pinned by digest. Distractor "pip install vllm" (the CUDA-12 wheel fails, D-10j).
4. MC — "Which is true of the best-documented DGX Spark fine-tuning result?" Correct: it runs on torchtune, which is wound down/dying (constants §7.4). Distractor "torchforge is its drop-in successor" (it is not).
5. Transfer — "The course refused to print one number to the very end. Which, and why?" → the dense BF16 ceiling (62 vs 125 TF); unpublished/unresolved; the course measures rather than asserts (constants §6.3, §10).

**Thread touchpoints.** All four threads announce their close here. `[THREAD:memory-ledger beat CLOSE]`, `[THREAD:TN1 beat CLOSE]`, `[THREAD:chain-rule beat CLOSE]`, `[THREAD:dot-product beat CLOSE]` — the farewell prose names each object once by its first guise (the `Value` class, TN-1, the checkpoint byte-match, the measured wall).

**Cross-references.** ← p.63 (the roofline he built, reused as the VRAM overlay), ← p.64 (the Mistral reveal, the convergence half), ← p.52 (GRPO/RLVR — "the last new technique you learned"), ← the diffusion track's economic note (page 62 = D-21 61, the divergence source; match constants §9.6). → index.html (the loop closes; the learner can restart any thread).

---

# index.html — Course home (not numbered)

**Shell.** `data-page="index.html"`. Kicker `⚡ Interactive Course`. `<h1>`: "Artificial Neural Networks — From High-School Algebra to Fine-Tuning on Your Own Box". Subtitle: "Sixty-five interactive pages from a single neuron to a Qwen3-8B LoRA and a FLUX.2 fine-tune — every number measured on the hardware you own." Load `course.css`; at end of body `course.js` + `course-map.js` (so the sidebar and this grid share one registry) + a short page IIFE. **No quiz on index.** Mirror PI-SI's index structure (its spec §"index.html").

**Intro section (prose).**
- What an ANN is: a machine that maps vectors to vectors and learns by gradient descent — and *the same machine*, pointed at "predict the next token," is an LLM; pointed at "predict the noise that was added," is a diffusion model (D-17 framing, verbatim-worthy). State the destination explicitly: **the reader trains toward fine-tuning LLMs and diffusion models on his own NVIDIA DGX Spark.**
- The structure: a shared **trunk** (foundations → training → architectures → adaptation) every learner walks, then a **fork** into two tracks — **LLM fine-tuning first**, then **diffusion** (D-17: LLM-first so his strong diffusion intuition gets a fresh formal frame) — then a **capstone** where the tracks rejoin.
- One paragraph on stakes/honesty: every hardware number is measured on his box, not quoted from a spec sheet — "published capacity is never usable capacity" (the 121.6875 GiB beat), and where a number can't be measured yet, the course ships a script instead of a fact.

**"How to use this course" box (`.box key`).** Pages are meant in order. Every content page follows one rhythm: **predict → intuition → math → worked number → demo → quiz** (brief-pedagogy). Demos compute real math live — drag the sliders. Deep-dive derivations are optional collapsibles. Several pages ship a Python script to run on *his* Spark (measure, don't take our word). The four **threads** — the dot product, TN-1 (the 9-parameter network), the chain rule, and the memory ledger — recur by design; watch for them.

**"What you'll be able to do" box (`.box rule`).** Derive backprop by hand on a 9-parameter network; read a model's parameter count off its checkpoint *to the byte*; build the memory ledger and predict what fits on your box before you run it; train a Qwen3-8B LoRA (17.08 GB, ~12 min) and a diffusion LoRA; explain attention, GQA, RoPE and cross-attention; predict your hardware's throughput from the roofline and be right; and recognize your FLUX.2 text encoder as a Mistral. (Draw each clause from a real page destination above.)

**Card grid — DATA-DRIVEN from the registry (do NOT hardcode 65 titles).**
- `<div class="card-grid">` populated by the page IIFE iterating `window.COURSE_MAP` (the same array `course.js` uses for the sidebar — the single source of truth per `course-map.js`'s header). One `<a class="card" href=file>` per entry with `.card-num` (the entry's `num`, or its part+index), `.card-title` (`title`), `.card-desc` (a one-line summary; **add an optional `desc` field to each `COURSE_MAP` entry** drawn from that page's `subtitle`, falling back to empty).
- **Group by `part`:** emit a `<h2>` part heading whenever `part` changes, wrapping each part's cards in a sub-grid. **Skip the `Welcome` entries** (`index.html`, `TEMPLATE-example.html`) — they are navigation, not content cards (or render the Template card in a small "reference" footer).
- **Track colors:** apply `.card.track-llm` / `.card.track-diffusion` per the entry's `track` field (rail color + ◆ chip, matching the sidebar's scheme in `course.css §track`). Trunk and capstone parts stay neutral.
- **The seven parts** (final structure; the capstone's group heading is `Part VII — Capstone`): I Foundations, II Training, III Architectures, IV Adaptation, V — LLM Track (`.track-llm`, ends with the GRPO/RLVR capstone p.52), VI — Diffusion Track (`.track-diffusion`, pp.53–62), VII — Capstone (pp.63–65). Milestone entries (`milestone:true`) get a 🚩 on their card.

**⚠️ Builder note (must ship, and must reach the architect).** `assets/course-map.js` is currently an explicit **placeholder** (its own header says so) listing ~30 provisional entries with `L#`/`D#`/`T#` numbering — **not** the final 65-page, 1–65 + Part I–VII scheme this course settled on (D-21, GRPO renumber). **index.html must not invent the 65 titles; it must render whatever `COURSE_MAP` contains.** The architect must replace `COURSE_MAP` with the final 65-entry list (each with `file`, `num` 1–65, `title`, `part` I–VII, `track` where applicable, `desc`, `milestone`) before ship. Until then this index renders the placeholder correctly and grows automatically when the map is filled — exactly the property `course-map.js` promises ("adding/removing/reordering entries here is all it takes to re-wire the whole course's navigation"). **This is the one cross-cutting dependency in Part VII; it is named in the return summary.**

**Cross-references.** Every card links out; the capstone cards (63–65) link to this part's pages. index.html is the hub the four threads loop back to.
