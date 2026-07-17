/* ==========================================================================
   ANN Course — navigation map (single source of truth for the sidebar).
   Consumed by assets/course.js (buildNav / prev-next pager).

   FINAL MAP — ratified 2026-07-16 by the master-spec assembler from the seven
   part specs (research/spec/spec-part1..7.md) + the code ladder (spec-code.md),
   reconciled against decisions.md §D-21a (THE CANONICAL PAGE TABLE) and the GRPO
   renumber. RE-INTEGRATED 2026-07-16 after the fidelity repair restored Parts I–IV
   to §D-21a: Part I 04=derivatives, 05=chain-rule, 08=probability/reparam,
   09=logs/softmax/CE, 10=XOR/collapse, 11=activations; Part II 19=init,
   20=normalization, 21=schedules, 22=regularization, 23=bias-variance/double-descent;
   Part III 32=causal-masking, 37=O(S²)/KV; Part IV 38=scaling-laws/MoE. The structure
   the course commits to:

     • a shared TRUNK — Part I The Machine (01–11), Part II How It Learns
       (12–24), Part III Architecture (25–37), Part IV Adaptation (38–43) —
       that every learner walks, then
     • a fork into two parallel TRACKS — Part V LLM Fine-Tuning (44–52, ending
       in the GRPO/RLVR capstone) and Part VI Diffusion (53–62) — readable in
       either order, then
     • a shared Part VII Capstone (63–65) where the tracks rejoin.

   Numbering (final = D-21 with the GRPO renumber): pages 1–51 as in D-21; the
   GRPO capstone = 52; D-21's diffusion 52–61 → 53–62; D-21's capstone 62–64 →
   63–65. 65 content pages + index = the 66-entry spine. (TEMPLATE-example.html
   is retained as a dev-reference link under Welcome; it is not a content page.)

   Entry schema (all fields optional except file/title/part):
     file      — the html filename, also the data-page value on that page's <body>
     num       — sidebar number prefix ("" to omit)
     title     — < 40 chars, or the sidebar wraps
     part      — groups consecutive entries into one collapsible <details> section.
                 buildNav opens only the section containing the current page.
     track     — "llm" | "diffusion". A section whose pages ALL share one track
                 gets that track's rail color + chip automatically; mixed
                 sections tint per-link.
     milestone — true → 🚩 flag in the sidebar + eligible for a <div class="milestone">
                 checkpoint on the page. Kept sparing: 06, 24, 36, 52, 62, 65
                 (one per major phase of the spiral).
     devOnly   — true → excluded as a prev/next pager neighbor (buildNav's pager
                 skips it in both directions), but still listed and clickable in
                 the sidebar. Used for TEMPLATE-example.html: it sits in the map
                 for discoverability under Welcome, but it is not a content page
                 and should never appear in the index→01→…→65 pager chain.

   Adding/removing/reordering entries here is all it takes to re-wire the whole
   course's navigation — no page edits, no engine edits.
   ========================================================================== */
window.COURSE_MAP = [
  { file: "index.html",                       num: "",   title: "Start Here",                          part: "Welcome" },
  { file: "TEMPLATE-example.html",            num: "",   title: "◈ Template & Component Gallery",       part: "Welcome", devOnly: true },

  /* ---------------- TRUNK · Part I — The Machine ---------------- */
  { file: "01-what-is-a-network.html",        num: "1",  title: "What a Neural Network Actually Is",    part: "Part I — The Machine", desc: "A machine, a knob, and the question this whole course answers: which knobs, turned how fast?" },
  { file: "02-vectors-and-shapes.html",       num: "2",  title: "Vectors, Matrices, Tensors",          part: "Part I — The Machine", desc: "Notation starts here — almost every real fine-tuning mistake is a shape mistake." },
  { file: "03-the-dot-product.html",          num: "3",  title: "The Dot Product — the Atom",          part: "Part I — The Machine", desc: "One operation, “how much of this is in that,” that a network asks billions of times." },
  { file: "04-derivatives-and-the-flip.html", num: "4",  title: "Derivatives & Steepest Ascent",       part: "Part I — The Machine", desc: "A derivative is a sensitivity — turned into the seven slopes the whole course needs." },
  { file: "05-the-chain-rule.html",           num: "5",  title: "The Chain Rule",                      part: "Part I — The Machine", desc: "One sentence on this page is backpropagation — chain three sensitivities by hand." },
  { file: "06-tn1-early-real-thing.html",     num: "6",  title: "TN-1: Your First Real Network",       part: "Part I — The Machine", milestone: true, desc: "Hand-compute a real 9-parameter network, then watch a live engine agree with you." },
  { file: "07-row-or-column.html",            num: "7",  title: "Row or Column? The Shape Bridge",      part: "Part I — The Machine", desc: "Proves y = Wx on paper and lin1(x) in PyTorch are the same W." },
  { file: "08-probability-and-the-gaussian.html", num: "8", title: "Probability, Gaussian & Reparam",  part: "Part I — The Machine", desc: "A distribution, an expectation, and the reparameterization trick behind every VAE and diffusion model." },
  { file: "09-logs-softmax-cross-entropy.html", num: "9", title: "Logs, Softmax & Cross-Entropy",       part: "Part I — The Machine", desc: "Feel an underflow crash, fix it with logs, meet softmax and the one loss the course keeps." },
  { file: "10-neuron-xor-collapse.html",      num: "10", title: "The Neuron, XOR & Linear Collapse",   part: "Part I — The Machine", desc: "Stack a hundred neurons and prove the stack does nothing a single one couldn't — until you bend it." },
  { file: "11-activations-and-limits.html",   num: "11", title: "Activations & the MLP's Limits",      part: "Part I — The Machine", desc: "The activation menu, why ReLU won the 2010s, and the layer that isn't an activation at all." },

  /* ---------------- TRUNK · Part II — How It Learns ---------------- */
  { file: "12-loss-from-mle.html",            num: "12", title: "Loss from Maximum Likelihood",        part: "Part II — How It Learns", desc: "A loss isn't a menu choice — it's the negative log-likelihood of your data under the model." },
  { file: "13-gradient-descent-landscape.html", num: "13", title: "Gradient Descent & the Landscape",  part: "Part II — How It Learns", desc: "Turns the loss into a landscape over parameter space and gives the one rule the course reuses." },
  { file: "14-backprop-by-pencil.html",       num: "14", title: "Backprop by Pencil: Nine Gradients",  part: "Part II — How It Learns", desc: "Run TN-1 backward by hand and read out all nine gradients, checked against a live engine." },
  { file: "15-autograd-float-reality.html",   num: "15", title: "Autograd & the Float That Isn't Zero", part: "Part II — How It Learns", desc: "Hand PyTorch the same network — it returns the same nine gradients, almost, and floats explain why." },
  { file: "16-minibatch-and-noise.html",      num: "16", title: "Batch, Mini-Batch & Noise",           part: "Part II — How It Learns", desc: "Every step since page 14 was secretly a choice about batch size — now it's out loud." },
  { file: "17-optimizers-sgd-to-adamw.html",  num: "17", title: "Optimizers: SGD → AdamW",             part: "Part II — How It Learns", desc: "Each optimizer in the lineage fixes one specific way the last one failed, ending at AdamW." },
  { file: "18-memory-ledger.html",            num: "18", title: "The Memory Ledger",                   part: "Part II — How It Learns", desc: "Backprop's gradients, the fp32 master copy, and Adam's moments add up to bytes per parameter." },
  { file: "19-init-and-gradient-flow.html",   num: "19", title: "Initialization & Gradient Flow",      part: "Part II — How It Learns", desc: "Vanishing, exploding, and clipping are one number — the per-stage gain — told forward and backward." },
  { file: "20-normalization.html",            num: "20", title: "Normalization",                       part: "Part II — How It Learns", desc: "Init sets the gain once; training moves the weights and the gain drifts — normalization corrects it live." },
  { file: "21-lr-schedules-warmup.html",      num: "21", title: "LR Schedules & Warmup",               part: "Part II — How It Learns", desc: "Adam's second moment is off by 1000× at step one — the entire reason training needs warmup." },
  { file: "22-regularization.html",           num: "22", title: "Regularization & the Two Curves",     part: "Part II — How It Learns", desc: "A training loop, a memory budget, stable init, a good optimizer — now the two curves that reveal overfitting." },
  { file: "23-bias-variance-double-descent.html", num: "23", title: "Bias-Variance & Double Descent",  part: "Part II — How It Learns", desc: "The theory behind page 22's toolkit — and why more parameters can make overfitting vanish." },
  { file: "24-first-real-finetune.html",      num: "24", title: "Your First Real Fine-Tune",           part: "Part II — How It Learns", milestone: true, desc: "Assembles loss, backward pass, optimizer, schedule, and regularization into one readable training loop." },

  /* ---------------- TRUNK · Part III — Architecture ---------------- */
  { file: "25-architecture-inductive-bias.html", num: "25", title: "Architecture Is Inductive Bias",   part: "Part III — Architecture", desc: "Pages 1–24 taught one architecture, the plain MLP. Here's why every other one exists." },
  { file: "26-convolutions-the-unet-atom.html", num: "26", title: "Convolutions: The U-Net Atom",       part: "Part III — Architecture", desc: "You've run U-Nets in every ComfyUI denoise step — this page opens the box." },
  { file: "27-embeddings-and-the-dot-product.html", num: "27", title: "Embeddings & the Dot Product",   part: "Part III — Architecture", desc: "Where attention is won or lost: how much does this vector matter to that one?" },
  { file: "28-rnns-why-attention.html",       num: "28", title: "RNNs & Why Attention Had to Exist",    part: "Part III — Architecture", desc: "Makes the pain that killed recurrent networks visceral, so attention arrives as relief." },
  { file: "29-attention-soft-lookup.html",    num: "29", title: "Attention I: The Soft Lookup",         part: "Part III — Architecture", desc: "Every transformer you've run is built from one operation — the soft lookup, derived here." },
  { file: "30-attention-scale-and-heads.html", num: "30", title: "Attention II: √d_head & Heads",       part: "Part III — Architecture", desc: "Derives the most hand-waved symbol in every attention explainer: the √d_head thermostat." },
  { file: "31-self-vs-cross-attention.html",  num: "31", title: "Attention III: Self vs Cross",         part: "Part III — Architecture", desc: "Change one thing — where K,V come from — and self-attention becomes cross-attention." },
  { file: "32-causal-masking.html",           num: "32", title: "Causal Masking: The LLM Hinge",        part: "Part III — Architecture", desc: "How one sequence, masked by time order, becomes as many training signals as it has tokens." },
  { file: "33-positional-encoding-rope.html", num: "33", title: "Positional Encoding & RoPE",           part: "Part III — Architecture", desc: "Attention can't tell “dog bites man” from “man bites dog” — RoPE tells it the order back." },
  { file: "34-residuals-and-normalization.html", num: "34", title: "Residuals & Normalization",         part: "Part III — Architecture", desc: "A single “+ x” kills the vanishing-gradient death and installs the residual stream." },
  { file: "35-the-transformer-block.html",    num: "35", title: "The Transformer Block, Assembled",     part: "Part III — Architecture", desc: "Bolts RMSNorm, GQA, RoPE, QK-Norm, and causal masking into the block that runs everything." },
  { file: "36-parameter-counting-qwen3.html", num: "36", title: "Parameter Counting on Qwen3-8B",       part: "Part III — Architecture", milestone: true, desc: "Four numbers from a config.json become 8,190,735,360 params, exactly, confirmed on the checkpoint." },
  { file: "37-attention-at-scale.html",       num: "37", title: "Attention at Scale: O(S²) & KV",       part: "Part III — Architecture", desc: "Three different costs hide inside softmax(QKᵀ/√d)V, and conflating them optimizes the wrong one." },

  /* ---------------- TRUNK · Part IV — Adaptation ---------------- */
  { file: "38-scaling-laws-moe-emergence.html", num: "38", title: "Scaling Laws, MoE & Emergence",      part: "Part IV — Adaptation", desc: "A base model is centuries of compute you'll never reproduce — this derives why you adapt, not rebuild." },
  { file: "39-rank-and-svd.html",             num: "39", title: "Rank, SVD & Low-Rank Approximation",  part: "Part IV — Adaptation", desc: "The load-bearing fact under the whole adaptation plan: the change a fine-tune makes is low-rank." },
  { file: "40-lora.html",                     num: "40", title: "LoRA: Low-Rank Adaptation",           part: "Part IV — Adaptation", desc: "You can't move all 8.19B of Qwen3-8B's weights — freeze the base, train two skinny matrices instead." },
  { file: "41-number-formats-qlora.html",     num: "41", title: "Number Formats & QLoRA",              part: "Part IV — Adaptation", desc: "Sixteen million numbers clustered near zero don't need sixteen bits — about four will do." },
  { file: "42-three-fine-tunings.html",       num: "42", title: "'Fine-Tuning' Names Three Things",    part: "Part IV — Adaptation", desc: "Your ComfyUI graph stacks a LoRA, a ControlNet, and an IP-Adapter — and they never fight." },
  { file: "43-your-box-roofline.html",        num: "43", title: "The Roofline: Your Box, Measured",    part: "Part IV — Adaptation", desc: "A chip has two speed limits — compute and feed rate. One number decides which one binds." },

  /* ---------------- FORK · Part V — LLM Fine-Tuning ---------------- */
  { file: "44-tokenization.html",             num: "44", title: "Tokenization",                        part: "Part V — LLM Fine-Tuning", track: "llm", desc: "The model never sees letters — it sees integers from a dictionary built by greedy compression." },
  { file: "45-pretraining-base-instruct.html", num: "45", title: "Pretraining, Base vs Instruct",      part: "Part V — LLM Fine-Tuning", track: "llm", desc: "Next-token prediction, applied 10^13 times, is where everything Qwen3-8B knows comes from." },
  { file: "46-decoding-kv-roofline.html",     num: "46", title: "Decoding, KV Cache & the Roofline",   part: "Part V — LLM Fine-Tuning", track: "llm", desc: "Generation is a different regime than scoring — and the KV cache is why it has its own roofline." },
  { file: "47-prompt-rag-finetune.html",      num: "47", title: "Prompt vs RAG vs Fine-Tune",          part: "Part V — LLM Fine-Tuning", track: "llm", desc: "The most expensive mistake in applied LLMs: fine-tuning a model to make it “know” something." },
  { file: "48-sft-preference-alignment.html", num: "48", title: "SFT & Preference Alignment",          part: "Part V — LLM Fine-Tuning", track: "llm", desc: "The same cross-entropy loss you already own, aimed at a different target: human preference." },
  { file: "49-memory-ledger-setup.html",      num: "49", title: "Does It Fit? Ledger & Setup",         part: "Part V — LLM Fine-Tuning", track: "llm", desc: "Five lines of a table decide whether a full fine-tune of Qwen3-8B fits your Spark. It misses, by a hair." },
  { file: "50-lora-applied.html",             num: "50", title: "LoRA on an LLM",                      part: "Part V — LLM Fine-Tuning", track: "llm", desc: "The adapter files in your ComfyUI folder are the same object you're about to train on Qwen3-8B." },
  { file: "51-serving-eval-hardware.html",    num: "51", title: "Serving, Eval & Hardware Reality",    part: "Part V — LLM Fine-Tuning", track: "llm", desc: "Your fine-tune trained in eleven minutes. Serving it and proving it's better are two harder problems." },
  { file: "52-grpo-rlvr-capstone.html",       num: "52", title: "Capstone: GRPO & Verifiable Rewards", part: "Part V — LLM Fine-Tuning", track: "llm", milestone: true, desc: "The 2026 reasoning frontier, on your desk: no reward model, no human labels, just a checkable answer." },

  /* ---------------- FORK · Part VI — Diffusion ---------------- */
  { file: "53-the-vae-his-comfyui-runs-on.html", num: "53", title: "Latent Space: The VAE",            part: "Part VI — Diffusion", track: "diffusion", desc: "Before a single denoising step, every image is squeezed into a tiny latent grid by a VAE." },
  { file: "54-the-forward-noising-process.html", num: "54", title: "The Forward Process: A Rotation",  part: "Part VI — Diffusion", track: "diffusion", desc: "The whole forward chain is one rotation of a unit vector — the noise schedule just sets the sweep rate." },
  { file: "55-the-reverse-process-and-elbo.html", num: "55", title: "The Reverse Process & the ELBO",  part: "Part VI — Diffusion", track: "diffusion", desc: "The reverse of one big noising step is intractable; the reverse of a tiny step is a Gaussian blob." },
  { file: "56-score-matching-and-the-sde.html", num: "56", title: "Score Matching & the SDE",          part: "Part VI — Diffusion", track: "diffusion", desc: "You trained one network, but three lines of calculus show it's secretly a score estimator too." },
  { file: "57-flow-matching-your-scheduler.html", num: "57", title: "Flow Matching & Your Scheduler",  part: "Part VI — Diffusion", track: "diffusion", desc: "One route change makes the sampler a single subtraction and kills a structural DDPM bug." },
  { file: "58-the-unet-in-depth.html",        num: "58", title: "The Denoiser I: The U-Net",           part: "Part VI — Diffusion", track: "diffusion", desc: "The network was a black box called ε_θ for five pages. Here it's opened." },
  { file: "59-from-unet-to-dit-mmdit.html",   num: "59", title: "The Denoiser II: DiT & MMDiT",        part: "Part VI — Diffusion", track: "diffusion", desc: "Swap the U-Net's hierarchy for a transformer over patch tokens, and quality starts tracking GFLOPs." },
  { file: "60-latent-diffusion-and-the-roofline.html", num: "60", title: "Latent Diffusion & the Roofline", part: "Part VI — Diffusion", track: "diffusion", desc: "The VAE wasn't a convenience — it moved diffusion off a 65,536× cliff." },
  { file: "61-conditioning-cross-attention-cfg.html", num: "61", title: "Conditioning, Cross-Attn & CFG", part: "Part VI — Diffusion", track: "diffusion", desc: "Guidance is the dial you've turned a thousand times and understood least — five lines, derived." },
  { file: "62-fine-tuning-diffusion-on-your-data.html", num: "62", title: "Fine-Tuning Diffusion on Your Data", part: "Part VI — Diffusion", track: "diffusion", milestone: true, desc: "Every LoRA, ControlNet, and IP-Adapter on your disk is one of a small number of ways to change output." },

  /* ---------------- Part VII — Capstone (tracks rejoin) ---------------- */
  { file: "63-predict-your-box.html",         num: "63", title: "Predicting Your Own Box",             part: "Part VII — Capstone", desc: "Assembles the memory ledger and the roofline into one console that predicts your Spark before you run it." },
  { file: "64-tracks-rejoin.html",            num: "64", title: "The Tracks Grew Back Together",       part: "Part VII — Capstone", desc: "Your image model's prompt reader turns out to be a 40-layer Mistral — the LLM track, reopened." },
  { file: "65-where-next.html",               num: "65", title: "Where This Goes Next",                part: "Part VII — Capstone", milestone: true, desc: "The tracks converged architecturally and diverged economically, in the same year — and where you go now." }
];
