# Research Brief — Track A: LLMs and Fine-Tuning Them

> **⚠️ THIS BRIEF PREDATES TWO VERIFICATION PASSES. `constants.md`, `decisions.md`, `notation.md` OVERRIDE IT.**
> Load-bearing corrections: **(1)** the **"131 GB vs 128 GB, 3 GB over" cliffhanger is a GB/GiB ARTIFACT** — the
> state is **122.05 GiB**, which *fits* in a 128 GiB box. The course must **MEASURE** usable memory
> (`torch.cuda.mem_get_info()`) and compute 122.05 GiB against *that*, not assert a budget (D-04, constants §6.8).
> Every "3 GB over" / "128 GB budget" line below is affected. **(2)** QLoRA NF4 = **0.516 B/param (4.127 bits)
> [VP]**, not 0.53 (the 4.25-bit arithmetic on line ~750 is an error; §Z-4). **(3)** MBU heuristic SURVIVES: four
> dense points land 60–75% under decode-traffic bytes; NVFP4 sizes are 6.03/10.54 GB not 4.5/7.5 (D-10e). **(4)**
> 70B QLoRA adapter = **207,093,760 / 3.31 GB / ≈39.7 GB** (D-20). **(5)** §9 base-count reconciliation closes
> **to the byte** (8,190,735,360) once norms are added — not "to 3 s.f." (D-07). **(6)** trl **1.8.0**. See §D-20.

**For:** the curriculum architect writing the build spec
**Date compiled:** 2026-07-16
**Scope:** the LLM destination track. Assumes the shared trunk (tensors, backprop, softmax, attention, transformer block) is already taught.
**Learner:** rusty-but-trained adult; NVIDIA DGX Spark (GB10, 128 GB unified); already fluent in ComfyUI/diffusion; undergrad level, drop to HS algebra only where load-bearing.

**Confidence convention used throughout:**
- **[V]** = verified this session against a primary source (config.json, official docs, vendor benchmark page).
- **[V-2°]** = verified against a credible secondary source (LMSYS review, HF docs summarizing a paper).
- **[U]** = recalled, NOT verified this session. Treat as a to-check item before shipping.

---

## 0. The one-paragraph shape of this track

The learner already knows a transformer maps a sequence of vectors to a sequence of vectors. This track adds five things, in order: (1) how text becomes vectors at all (**tokenization**), (2) what the pretraining objective buys and what it does *not* buy (**next-token prediction, base vs instruct**), (3) how you get text back out and why that costs memory (**decoding, KV cache**), (4) how the model was taught to behave (**SFT, preference alignment**), and (5) **how to change it yourself** — which begins not with LoRA but with an honest decision framework about whether you should be fine-tuning at all. The track's centre of gravity is a **memory ledger**: nearly every practical decision in this domain is a consequence of counting bytes, and the DGX Spark's 128 GB / 273 GB/s makes those bytes unusually legible. Lead with the ledger and the rest follows.

---

## 1. Dependency order (recommended spine)

This is the ordering I'd defend. Rationale for each arrow given.

```
[shared trunk: attention, transformer block, softmax, cross-entropy, AdamW]
   │
   ├─▶ A1. Tokenization ──────────────┐  (must precede everything: defines vocab V,
   │      BPE, vocab size tradeoff    │   which sets embedding params & the loss's
   │      failure modes               │   output dimension)
   │                                  │
   ├─▶ A2. Next-token prediction ◀────┘
   │      loss = mean CE over positions
   │      perplexity; scaling; data pipeline
   │      ⇒ BASE vs INSTRUCT  ◀── (needs A2 to explain what base models *are*)
   │
   ├─▶ A3. Decoding & the KV cache
   │      greedy/temp/top-k/top-p/beam
   │      KV cache size formula  ◀── (needs GQA from trunk; needs A1 for seq length)
   │      decode is memory-bound ⇒ roofline
   │
   ├─▶ A4. Post-training: SFT → preference alignment
   │      chat templates ◀── (needs A1: templates are token sequences)
   │      loss masking ◀── (needs A2: you're masking the SAME loss)
   │      DPO/GRPO/RLVR
   │
   ├─▶ A5. ★ THE DECISION: prompt vs RAG vs fine-tune
   │      ⇒ A5a. RAG properly (chunk/embed/retrieve/rerank)
   │      ⇒ A5b. fine-tune (continue below)
   │
   ├─▶ A6. THE MEMORY LEDGER  ◀── (the hinge of the whole track)
   │      bytes/param for full FT ⇒ 8B doesn't fit ⇒ motivates PEFT
   │
   ├─▶ A7. LoRA ◀── (needs A6 for motivation, needs SVD/low-rank from trunk)
   │      ⇒ QLoRA ◀── (needs A7 + quantization)
   │      ⇒ DoRA, rsLoRA, PiSSA/EVA
   │
   ├─▶ A8. Quantization (NF4, GGUF, AWQ, FP8, NVFP4)
   │      [can also be taught before A7; see note]
   │
   ├─▶ A9. Datasets  ◀── (needs A4's chat templates)
   ├─▶ A10. The training loop (transformers/peft/trl)
   ├─▶ A11. Hyperparameters
   ├─▶ A12. Evaluation  ◀── (needs A2: why loss ≠ quality)
   ├─▶ A13. Serving: merge, vLLM/llama.cpp/Ollama
   └─▶ A14. Hardware reality: what the Spark can and cannot do
```

**Three ordering arguments the architect should weigh:**

1. **A6 (memory ledger) before A7 (LoRA) is non-negotiable.** LoRA taught as "a clever low-rank trick" is forgettable; LoRA taught as "you counted the bytes, you were 3 GB over, here is the fix" is load-bearing. Make the learner *fail* first: have them compute full fine-tuning of Qwen3-8B and discover it's 131 GB against a 128 GB budget. The 3 GB miss is more pedagogically valuable than a 10× miss would be, because it's *infuriating*.

2. **A8 (quantization) is genuinely order-ambiguous.** QLoRA needs NF4, so quantization must precede QLoRA. But quantization-for-inference (GGUF/AWQ) belongs near A13 (serving). I'd **split it**: teach the *number format* idea (A8a: what 4 bits can represent, blockwise scales) immediately before QLoRA, and teach *deployment formats* (A8b: GGUF vs AWQ vs NVFP4, what to pick) at A13. Do not teach it as one 6-page block — it will read as a datasheet.

3. **A5 (the decision) must come before A6/A7, not after.** If the learner arrives at LoRA before they've been forced to ask "should I be doing this at all?", the course has failed at the exact place the field fails. The most common expensive mistake in this domain is fine-tuning when RAG was the answer. Put the fork *before* the machinery.

**Cross-track note:** A1, A2, A3, A6, A7, A8 are the LLM-specific spine. **A6, A7, A8 are shared with the diffusion track** — see §16 for what must be reconciled.

---

## 2. Tokenization

### Intuition first
> **The model never sees letters. It sees integers drawn from a fixed dictionary of roughly 150,000 word-fragments — a dictionary that was itself learned, by a greedy compression algorithm, from a pile of text nobody curated for your use case.**

That single sentence explains, without any further machinery, why models miscount letters, why they're bad at arithmetic, and why your Sanskrit/Python/DNA corpus costs 3× more to process than English.

### The mechanism: BPE

Byte-Pair Encoding. Start with the 256 possible bytes as the base vocabulary. Then repeatedly:

1. Count all adjacent pairs in the corpus.
2. Merge the most frequent pair into a new single token.
3. Append that merge to an ordered **merge list**.
4. Repeat until $|V|$ reaches target.

Encoding a new string = apply the merge list **in learned order**, greedily. This is why tokenization is deterministic and why it's *not* a search for the optimal segmentation — it's a replay of a training-time greedy schedule. That distinction matters and is nearly always skipped.

**Formally.** Let $\Sigma$ = byte alphabet, $|\Sigma| = 256$. The tokenizer is a map
$$
\tau: \Sigma^* \rightarrow \{0, 1, \dots, V-1\}^*
$$
where $V$ = vocabulary size (unitless count). For **Qwen3-8B, $V = 151{,}936$** [V — from `config.json`, `vocab_size: 151936`]. For Llama 3.x, $V = 128{,}256$ [U].

**Compression ratio** (the number the course should make familiar):
$$
\rho = \frac{\text{characters}}{\text{tokens}}
$$
$\rho \approx 3.5$–$4.0$ for English prose with a modern 100k–150k BPE vocab. $\rho \approx 1.5$–$2.5$ for code, non-Latin scripts, and rare technical jargon. **Use $\rho \approx 4$ as the course's standing rule of thumb** and have it recur: it converts "500-word document" → "~650 tokens" in one step.

### SentencePiece vs tiktoken-style BPE
- **SentencePiece** (Google; used by T5, older Llama, Gemma [U]) treats the input as a raw Unicode stream, encodes spaces explicitly as `▁`, and can run either BPE or unigram-LM algorithms. Its selling point is that it needs no pre-tokenization (no language-specific word splitting) — important for Japanese/Chinese/Thai, which don't use spaces.
- **Byte-level BPE** (GPT-2 lineage; tiktoken; Qwen, Llama 3 [U]) maps bytes → tokens, so it **can never produce an `<UNK>`**. Any byte string is representable. This is the dominant 2026 choice.
- The practical difference the learner will hit: SentencePiece models often need `add_special_tokens` care and have an explicit `▁` prefix convention; byte-BPE models put the leading space *inside* the token (`" the"` is a different token from `"the"`). **This single fact causes a large fraction of chat-template bugs.**

### Vocab size tradeoff — the real math

Larger $V$:
- ✅ Fewer tokens per document → shorter sequences → attention cost $O(n^2)$ falls quadratically, KV cache falls linearly.
- ❌ Embedding + output head params grow **linearly** in $V$.
- ❌ The softmax over $V$ at every position dominates activation memory (see §10, `chunked_nll`).
- ❌ Rare tokens get few gradient updates → poorly trained embeddings.

**Worked number [V, derived from Qwen3-8B config]:**
Embedding matrix: $V \times d_{\text{model}} = 151{,}936 \times 4096 = 622{,}329{,}856 \approx 0.622\text{B params}$.
Qwen3 has `tie_word_embeddings: false`, so the LM head is a *separate* matrix of the same size: another 0.622B.
**Total embedding-related params: 1.244B — that is 15.2% of the model's 8.19B.** The Qwen model card states 8.2B total / 6.95B non-embedding [V]; $8.19 - 6.95 = 1.24$ ✓. The arithmetic closes exactly. **Use this.** It is rare to get a published spec that reconciles to the byte.

### Why tokenization causes famous failure modes

| Failure | Real cause | The correction that actually fixes it |
|---|---|---|
| "How many r's in strawberry?" | The model sees ~3 opaque integers, not 10 letters. Letter-counting requires information the input representation *destroyed*. | Not "the model is dumb." The model is being asked to report on a property of a string it was never shown. A model can memorize the spelling of common words — which is why it gets "strawberry" right in 2026 but fails on a novel string like "hyperbolizations". **Test with a nonce word.** |
| Arithmetic on long numbers | `1234567` may split as `123`/`4567` — digit grouping is a *frequency artifact*, misaligned with place value. Carrying across a token boundary is a genuinely hard operation. | Modern tokenizers (Llama 3, Qwen) force **single-digit or 3-digit-group** splitting to fix exactly this [U — verify against the actual `tokenizer.json` regex before shipping]. This is a *tokenizer design fix*, not a model fix — good evidence for "the tokenizer is a design decision, not a given." |
| Reversing a string | Same as above. | Same. |
| Non-English costs 2–3× | The BPE merge list was learned on a corpus that was mostly English. Merges that would compress Hindi were never profitable during training. | Real, structural, and an *equity* issue, not a bug. Quantify it in the demo. |
| Trailing-whitespace prompts break generation | `"The answer is "` ends with a space; the model wants to emit `" Paris"` (with leading space) but the space is already consumed, forcing a rare token path. | Never end a prompt with a trailing space. This is a genuine footgun the learner will hit. |

### ⚠️ Misconception box: "Tokens are words"
Learners map token→word and then can't explain why a 500-word doc is 650 tokens, why `" the"` ≠ `"the"` ≠ `"The"`, or why the vocab is 151,936 (far more than English's word count) yet still splits `hyperbolizations`. **Correction that lands:** show them that `" the"`, `"the"`, `"The"`, `" The"` are **four distinct integers** in the vocab. The vocab isn't a dictionary of words; it's a dictionary of *frequent byte sequences*, and capitalization and leading spaces are part of the byte sequence. Once they see the four integers, the model clicks.

### ⚠️ Misconception box: "I can just add tokens to the vocab"
You can (`tokenizer.add_tokens`), but the new embedding rows are **randomly initialized** and the model has never seen them. Adding tokens requires `model.resize_token_embeddings()` and then enough training to make those rows meaningful — and with LoRA the embedding matrix is frozen by default, so **your new tokens will stay random unless you add them to `modules_to_save`.** This is a real, silent failure. PEFT has `trainable_token_indices` specifically for the "I want to train 5 new token embeddings without a 622M-param `modules_to_save`" case [V — in `LoraConfig`].

### 🎛️ Demo A1: Live BPE trainer + tokenizer
**Plot:** two panels. Top: a text box; the user's text rendered with each token in an alternating background colour, token ID printed beneath. Bottom: a line chart, x = number of BPE merges (0 → 3000), y = tokens-per-100-characters, with **three lines: English, Python source, Devanagari**.
**Slider:** number of merges, 0 → 3000. A second slider: which corpus to train the merges on (English-only / mixed).
**Exact math the JS must implement:** real BPE training. Maintain `Map<string, count>` of adjacent pair frequencies over a bundled ~50 KB corpus; each merge step: find argmax pair, apply merge to the corpus word-splits, push to ordered merge list. Then encode the user's live text by replaying the merge list in order. This is ~80 lines of JS and it must be the real algorithm — a lookup table is worthless here.
**The insight when they drag it:** (a) the tokens-per-char curve is a **steep drop then a long flat tail** — this *is* the vocab-size tradeoff, visible as a curve, and it shows why nobody uses $V = 10^6$; (b) when they switch the training corpus to English-only, the **Devanagari line barely moves** while English plummets — the multilingual tax made visual; (c) typing `strawberry` shows 3 tokens and typing `1234567` shows the digit split, at the exact moment they can explain why.

### 🎛️ Demo A2: The `strawberry` probe (small, 2 minutes, high value)
**Plot:** an input box + the token decomposition + a "count the r's" arithmetic the *learner* does on the token pieces.
**Insight:** the learner performs the counting task *using only the tokens the model sees*, and discovers they can't do it either without re-expanding to characters. This converts "the model is stupid" into "the representation destroyed the information" — which is the actual lesson and generalizes to a dozen other failures.

---

## 3. Pretraining and next-token prediction

### Intuition first
> **Pretraining is not teaching the model facts. It is forcing the model to compress the internet, and the only way to compress text well is to accidentally learn grammar, syntax, world structure, code semantics, and a great many facts as a side effect.**

The compression framing is the one that makes scaling laws feel inevitable rather than magical, and it's the framing that makes "why does next-token prediction produce reasoning?" a tractable question instead of a mystical one.

### The objective

$$
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

- $x_t$ — the token ID at position $t$; integer in $[0, V)$; shape scalar.
- $x_{<t}$ — all preceding tokens; shape $(t-1,)$.
- $p_\theta(\cdot \mid x_{<t})$ — the softmax over the vocab; shape $(V,)$, i.e. **(151936,)** for Qwen3.
- $T$ — sequence length in tokens (unitless).
- $\mathcal{L}$ — units are **nats/token**.
- $\theta$ — all 8.19e9 parameters.

**Batched shapes the course must state explicitly, because learners lose the plot here:**
- input_ids: `(B, T)` int64
- hidden states: `(B, T, d_model)` = `(B, T, 4096)` bf16
- logits: `(B, T, V)` = `(B, T, 151936)` **← this is the memory bomb**
- labels: `(B, T)` int64, shifted left by one, with `-100` on masked positions

**The logits tensor is the single most underappreciated number in this course.** At `B=1, T=2048`: $1 \times 2048 \times 151936 \times 2\text{ bytes} = 622\text{ MB}$ **for one tensor**, and cross-entropy in fp32 makes it 1.2 GB, and you need it *and* its gradient. This is precisely why TRL 1.8's default `loss_type` is `"chunked_nll"` — it drops `labels == -100` positions *before* the `lm_head` matmul and chunks the CE, so peak activation memory no longer scales with $V \times T$ [V — TRL SFTConfig docs]. **Teach this. It's a real, current, non-obvious thing and it directly explains an API default the learner will see.**

### Perplexity — and the conversion the course should drill

$$
\text{PPL} = \exp(\mathcal{L}) \qquad\text{[$\mathcal{L}$ in nats/token]}
$$
$$
\text{bits/token} = \frac{\mathcal{L}}{\ln 2}
$$

**Worked example, carried to a number:**
- A *random* model over Qwen3's vocab: $\mathcal{L} = \ln(151{,}936) = 11.93$ nats/token. $\text{PPL} = 151{,}936$ (it is exactly $V$, by construction — say this, it's a satisfying check). bits/token $= 11.93/0.693 = 17.2$ bits.
- A good 8B model on English web text: $\mathcal{L} \approx 2.0$ nats. $\text{PPL} = e^{2.0} = 7.39$. bits/token $= 2.89$.
- **Interpretation the learner should carry:** the model has gone from "1 in 151,936" to "effectively choosing among ~7.4 options." At $\rho \approx 4$ chars/token, 2.89 bits/token ≈ **0.72 bits per character** — which the learner can compare to Shannon's famous ~1.0–1.3 bits/char estimate for English [U — verify the Shannon figure and its caveats]. The model is compressing English *below* Shannon's human-prediction estimate. That's a genuinely startling number and worth a callout box.
- A fine-tune on your own corpus might reach $\mathcal{L} = 1.5$, $\text{PPL} = 4.48$. **This looks like a big win and may mean nothing** — see §12.

### Scale, and the data pipeline

- **Chinchilla-optimal** [U — verify the exact 20:1 and the current status of this claim]: for a compute budget $C \approx 6ND$ FLOPs ($N$ = params, $D$ = training tokens), loss is minimized at $D \approx 20N$.
- **2026 reality: nobody trains Chinchilla-optimal anymore.** Llama 3 8B was trained on ~15T tokens [U] = **1,875 tokens/param, ~94× Chinchilla**. Why: Chinchilla optimizes *training* compute. If you're going to serve the model billions of times, you want it *small and over-trained*, because inference cost scales with $N$, not $D$. **This is a genuinely important correction to a widely-repeated factoid and the course should make the point sharply: Chinchilla answers a question almost nobody actually has.**
- $C \approx 6ND$: for 8B × 15T = $6 \times 8\times10^9 \times 1.5\times10^{13} = 7.2\times10^{23}$ FLOPs. **On a DGX Spark at a generous sustained 100 TFLOP/s bf16 [U — the 1 PFLOP figure is FP4-with-sparsity, see §14], that is $7.2\times10^9$ s ≈ 228 years.** State this number. It ends the "can I pretrain my own?" question permanently and correctly, and it makes the case for fine-tuning viscerally rather than by assertion.
- **Data pipeline** (what actually happens, compressed to a page): crawl (Common Crawl) → language ID → dedup (MinHash/LSH at document and paragraph level) → quality filter (classifier trained on "good" reference text, e.g. Wikipedia/books) → toxicity/PII filter → decontamination against eval sets → tokenize → pack into fixed-length blocks. **The dedup step is where most of the win is** [U — worth a source]. Emphasize: the ratio of raw crawl to surviving tokens is roughly 10:1 to 100:1 [U].

### ★ Base vs Instruct — the distinction learners constantly miss

This deserves its own boxed section with a demo, because it is the #1 confusion.

> **A base model is not a chatbot that's bad at chatting. It is a document autocompleter that has never been asked a question in its life.**

| | Base (`Qwen3-8B-Base`) | Instruct (`Qwen3-8B`) |
|---|---|---|
| Trained on | raw text, next-token only | base + SFT + preference alignment |
| Given `"What is the capital of France?"` | may emit `"\nWhat is the capital of Germany?\nWhat is the capital of Spain?"` — because it's completing a **quiz document** | emits `"Paris."` |
| Has a chat template? | usually **no** (or an unused vestigial one) | yes, and it's mandatory |
| Knows when to stop? | **no** — no meaningful EOS behaviour | yes, emits `<|im_end|>` |
| When to fine-tune from it | you have a lot of data (>50k) and want to define the behaviour yourself; or you're doing continued pretraining on a domain corpus | almost always — you want to keep its instruction-following and just adjust form |

**The failure mode this predicts:** learner SFTs a **base** model on 500 examples, gets a model that rambles forever and never stops. Cause: the base model has no EOS convention and 500 examples isn't enough to install one. **Correction: start from Instruct, or if you must use Base, set `eos_token` explicitly in `SFTConfig` and use `chat_template_path`.** TRL 1.8 supports exactly this: `SFTConfig(chat_template_path="HuggingFaceTB/SmolLM3-3B")` grafts a template onto a base model and the trainer handles the tokenizer/special-token updates [V — TRL SFT docs]. There's also a documented gotcha: Qwen base models ship a chat template in the tokenizer but with a **mismatched EOS**, so you must pass `eos_token="<|im_end|>"` [V — explicit warning in TRL docs].

### 🎛️ Demo A3: Base vs Instruct side-by-side
**Plot:** two panes, same prompt, token-by-token generation shown streaming with per-token probability bars.
**Implementation note:** this cannot run a real 8B in-browser. **Options:** (a) pre-record the actual token/probability traces for ~12 curated prompts from both models and replay them — this is honest, cheap, and preserves the real numbers; (b) ship a genuinely tiny model (a ~1–5M-param char-level or small-BPE transformer trained on a toy corpus) via ONNX Runtime Web / transformers.js and train the *toy* base/instruct pair yourself. **I recommend (a) for this demo** and (b) only if the shared trunk already ships a tiny in-browser transformer, in which case reuse it.
**The insight:** the base model's continuation is *not gibberish* — it's fluent, plausible, and completing the wrong document. That specific observation ("it's not broken, it's answering a different question") is what makes the distinction stick.

---

## 4. Inference and decoding

### Intuition first
> **The model gives you a probability distribution over 151,936 options. Decoding is the policy you use to pick one. Every decoding "parameter" is just a different way of deciding how much of the tail to throw away before you roll the dice.**

### The math, all of it

Given logits $z \in \mathbb{R}^{V}$ at the final position:

**Temperature:**
$$
p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{V}\exp(z_j / T)}
$$
$T > 0$, dimensionless. $T \to 0^+$ ⇒ argmax (greedy). $T = 1$ ⇒ the model's calibrated distribution. $T \to \infty$ ⇒ uniform over $V$.

**Greedy:** $x_t = \arg\max_i z_i$. Equivalent to $T \to 0$. Deterministic.

**Top-k:** keep the $k$ largest $z_i$, set the rest to $-\infty$, then softmax. $k$ is an integer count.

**Top-p (nucleus):** sort $p$ descending; find the smallest set $S$ such that $\sum_{i \in S} p_i \geq p_{\text{thresh}}$; renormalize over $S$. $p_{\text{thresh}} \in (0, 1]$.

**Min-p** (2024-era, now common in llama.cpp/vLLM [U — verify it's exposed in the 2026 APIs]): keep tokens with $p_i \geq p_{\text{min}} \cdot \max_j p_j$. Scales the cutoff with the model's confidence.

**Beam search:** maintain $B$ partial hypotheses, expand each, keep the top $B$ by cumulative log-prob $\sum_t \log p(x_t \mid x_{<t})$, usually with a length penalty $\frac{1}{|x|^\alpha}$.

### ⚠️ Misconception box: "Temperature adds randomness"
It doesn't add anything. **Temperature is a monotone rescaling of the logits — it never changes the *ranking* of tokens, only the *gaps* between their probabilities.** The randomness was already there (sampling). $T$ controls how much the sampler respects the model's confidence. **Correction that fixes it:** show that greedy at $T=0.1$ and $T=2.0$ produce the *same argmax* — the top token is always the top token. What changes is how often you take the 2nd.

### ⚠️ Misconception box: "Higher temperature = more creative"
Higher temperature = more of the model's low-probability tail, which is mostly **wrong tokens**, not creative ones. Beyond ~$T=1.2$ on most models you get degradation, not creativity. **Correction:** in the demo, at $T=1.5$ the learner will see a genuinely absurd token grab meaningful probability mass. The tail is not a reservoir of good ideas.

### ⚠️ Misconception box: "Beam search gives better answers"
It gives **higher-likelihood** answers, and for open-ended generation that is actively bad — it produces bland, repetitive, degenerate text (the "likelihood trap"). Beam search is for **constrained, single-correct-answer** tasks: translation, structured extraction, code with a known target. It is essentially **not used for chat**, and the KV-cache cost is $B\times$. **Correction:** state the rule directly — *if there is one right answer, beam; if there are many acceptable answers, sample.*

### ★ The KV cache

> **Intuition: the model recomputes attention over the whole prefix at every new token — unless you save the K and V vectors you already computed. The cache is that savings. It turns an $O(n^2)$-per-token disaster into $O(n)$, and in exchange it eats your RAM linearly in context length.**

**Why K and V but not Q:** at step $t$ you need $q_t$ (new, one vector) attended against $K_{1:t}$ and $V_{1:t}$ (all of them). $q_1 \dots q_{t-1}$ are never needed again. That asymmetry is the whole idea and learners rarely get told it explicitly.

**The formula — the course's most reusable equation:**
$$
\text{KV bytes} = 2 \cdot L \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot b \cdot n_{\text{ctx}} \cdot B
$$
- $2$ — one for K, one for V (unitless)
- $L$ — number of transformer layers
- $n_{\text{kv}}$ — **key/value** heads (NOT query heads — this is the GQA saving)
- $d_{\text{head}}$ — head dimension
- $b$ — bytes per element (2 for bf16/fp16, 1 for fp8)
- $n_{\text{ctx}}$ — context length in tokens
- $B$ — batch size (concurrent sequences)

**Worked example — Qwen3-8B [V, all values from `config.json`]:**
$L = 36$, $n_{\text{kv}} = 8$, $d_{\text{head}} = 128$, $b = 2$.

Per token:
$$
2 \times 36 \times 8 \times 128 \times 2 = 147{,}456 \text{ bytes} = \mathbf{144\ KiB/token}
$$

**Make `144 KiB/token` a number the course repeats until it's reflexive.**

- At native context $n_{\text{ctx}} = 32{,}768$: $147{,}456 \times 32{,}768 = 4.83 \times 10^9$ B = **4.5 GiB**
- At YaRN-extended $131{,}072$: **18 GiB**
- At batch 8, 32k context: **36 GiB**

**The GQA punchline:** Qwen3-8B has 32 query heads but only **8** KV heads (`num_attention_heads: 32`, `num_key_value_heads: 8`) [V]. If it used vanilla MHA ($n_{\text{kv}} = 32$), the cache would be **4× larger: 576 KiB/token, 18 GiB at 32k context.** GQA is not a minor optimization — **it is the reason long context is affordable at all**, and this 4× is the cleanest possible demonstration of it. MQA ($n_{\text{kv}}=1$) would be 18 KiB/token, a 32× saving, at a real quality cost.

**Context the learner needs:** at 32k context, the KV cache (4.5 GiB) is **27% of the model's own bf16 weights (16.4 GB)**. At 128k it's 18 GiB — **larger than the model**. That inversion is the thing to make them feel.

### ⚠️ Misconception box: "The KV cache is an optimization I can turn off if I'm short on memory"
Turning it off doesn't save memory in any useful sense — it makes generation $O(n^2)$ per token instead of $O(n)$, so generating 1000 tokens goes from ~1000 forward passes over 1 token each to ~1000 forward passes over an average of 500 tokens each: **~500× more compute.** The cache is not optional. What you actually do when short on memory: shorter context, smaller batch, **fp8 KV cache** (halves it), or GQA/MQA architectures.

### ⚠️ Misconception box: "Prefill and decode are the same operation"
They have **opposite bottlenecks**, and this explains every performance number in §14.
- **Prefill**: process $n$ prompt tokens at once. One big matmul per layer. **Compute-bound.** Parallel. LMSYS measured **7,991 tok/s prefill** for Llama 3.1 8B FP8 on DGX Spark [V-2°].
- **Decode**: one token at a time. Every weight in the model must be read from memory to produce **one** token. **Memory-bandwidth-bound.** Sequential. Same machine, same model: **20.5 tok/s decode** [V-2°].
- **That's a 390× gap on identical hardware and an identical model.** If the learner internalizes nothing else from §4, this is it.

### 🎛️ Demo A4: The decoding sandbox
**Plot:** a horizontal bar chart of the top 30 tokens by probability, with the truncation boundary drawn as a vertical line. Below: cumulative probability curve. Below that: a running "sampled sequence" of ~40 tokens.
**Sliders:** Temperature (0.01 → 2.0), top-k (1 → 100, plus "off"), top-p (0.1 → 1.0), min-p (0 → 0.5). A dropdown selects one of ~8 **real, pre-recorded logit vectors** captured from Qwen3-8B at interesting positions (a confident one: after `"The capital of France is"`; a genuinely uncertain one: after `"The best programming language is"`; a mid-sentence-function-word one).
**Exact math the JS must implement:** real softmax with temperature over the full 151,936-length logit vector (ship the top ~2000 + a lumped tail to keep the payload sane — but *say* you did that, don't hide it). Real top-k, real top-p with sorting and cumsum, real min-p. Real multinomial sampling. Compute and display **entropy** $H = -\sum_i p_i \log_2 p_i$ in bits, live.
**Insight #1 (temperature):** dragging $T$ from 0.1 to 2.0, the bars flatten but **never reorder**. That's the "temperature doesn't add randomness" lesson, delivered by their own hand.
**Insight #2 (top-k vs top-p):** switch between the *confident* logit vector and the *uncertain* one with top-p = 0.9 fixed. On the confident one, top-p keeps **2 tokens**. On the uncertain one it keeps **60**. Now do the same with top-k = 20: it keeps 20 in both, which is far too many for the confident case and too few for the uncertain one. **This is the entire argument for why top-p beat top-k, and it is invisible without the interaction.** This is the single best demo in §4.
**Insight #3 (entropy):** the entropy readout makes "the model is confident here / lost here" a *number*, which sets up §12's calibration discussion.

### 🎛️ Demo A5: KV cache calculator
**Plot:** a stacked horizontal bar: [model weights | KV cache | activations], against a vertical line at **128 GB (your Spark)** and a second at 24 GB (a 4090, for contrast).
**Sliders:** context length (512 → 262,144, log scale), batch size (1 → 64), KV dtype (bf16 / fp8), and a **preset dropdown** (Qwen3-8B / Qwen3-32B [U] / Llama-3.3-70B [U]) that fills $L$, $n_{\text{kv}}$, $d_{\text{head}}$. Plus an **"attention type" toggle: MHA / GQA / MQA** which overrides $n_{\text{kv}}$ to $n_{\text{heads}}$ / 8 / 1.
**Exact math:** the formula above, verbatim. No fudge factors. Show the arithmetic as a rendered expression that updates live — the learner should see `2 × 36 × 8 × 128 × 2 × 32768 × 1 = 4.83 GB` change term by term.
**Insight:** the moment they drag context past ~100k on a 70B and the bar blows through the 128 GB line, "why is long context expensive?" is answered permanently. Then they flip MHA→GQA and watch it drop 4× and understand why every 2026 model is GQA.

### 🎛️ Demo A6: The decode roofline (★ high value, ties §4 to §14)
**Plot:** log-log. x = model bytes-in-memory (1 → 200 GB). y = decode tokens/sec. Draw the **roofline** $\text{tok/s} = \text{BW} / \text{bytes}$ as a straight line of slope −1. Then **overlay real measured points as dots** (from §14's table).
**Slider:** memory bandwidth (100 → 8000 GB/s), preset-marked at **DGX Spark = 273**, RTX 5090 ≈ 1792 [U], H100 SXM ≈ 3350 [U], M3 Ultra ≈ 819 [U].
**Exact math:** $\text{tok/s}_{\max} = \text{BW}_{\text{GB/s}} / (\text{params} \times \text{bytes/param} / 10^9)$. Also compute and display **MBU (Memory Bandwidth Utilization) = measured / ceiling** for each real dot.
**Insight:** the measured dots sit at a **strikingly consistent ~60–70% of the ceiling** (see §14 for the arithmetic — Llama 8B FP8: 60%; Llama 70B FP8: 69%; Llama 8B NVFP4: 64%). The learner discovers that **a single division predicts their hardware's real-world speed to within ~40%**, and that quantization's speed benefit is *almost entirely* a memory-bandwidth effect, not a math effect. This demo retroactively explains most of §8 and all of §14, and it is the most valuable demo in this brief.

---

## 5. Post-training

### Intuition first
> **Pretraining gives the model everything it knows. Post-training gives it manners, a format, and the willingness to answer instead of continue. It changes almost nothing about what the model knows and almost everything about what it does.**

### The pipeline (2026 canonical shape) [V-2°]

```
Base model  ──SFT──▶  Instruction-following  ──Preference──▶  Aligned  ──RLVR──▶  Reasoning
   (15T tok)          (10k–1M examples)         (10k–100k prefs)      (verifiable tasks)
```

The 2026 stack is **modular** and the community has converged on: SFT for instruction-following → preference optimization (DPO / SimPO / KTO) for style and refusals → **RL with verifiable rewards (GRPO / DAPO)** for math/code/reasoning where you can *check* the answer programmatically [V-2°].

### Stage 1: SFT / instruction tuning

Same cross-entropy loss as pretraining. **The only differences are the data and the mask.** Say this plainly — learners expect SFT to be a different algorithm and it is not.

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} m_t \log p_\theta(y_t \mid y_{<t}), \qquad m_t \in \{0,1\}
$$
$m_t$ is the loss mask: 1 on assistant tokens, 0 on system/user tokens (implemented as `label = -100`).

TRL 1.8 gives you three masking regimes [V]:
- `completion_only_loss=True` — default for prompt-completion datasets; loss on completion only.
- `assistant_only_loss=True` — for conversational datasets; loss only on assistant turns. **Requires the chat template to contain `{% generation %}` / `{% endgeneration %}` markers.** TRL auto-patches known families (Qwen3); for others you must check. This is a real, current footgun.
- Neither — loss on the full sequence (correct for continued pretraining, wrong for chat SFT).

### ⚠️ Misconception box: "Loss masking is a minor detail"
If you train on the *user* turns too, you are teaching the model to **generate plausible user questions**, which is (a) not what you want and (b) will make it interview you. On a dataset with long prompts and short answers, the majority of your gradient signal goes to the wrong task. **Correction:** compute the ratio. If your average example is a 400-token prompt and a 50-token answer, **89% of your unmasked loss is on text the model should never generate.** That number ends the argument.

### Stage 2: Preference alignment

**RLHF (the original):** train a reward model $r_\phi$ on human pairwise preferences, then PPO the policy against it with a KL penalty to the reference. Three models in memory (policy, reference, reward) + a value head. Expensive, unstable, and largely **not what the learner should do**.

**DPO — the key insight, stated as intuition first:**
> **You don't need a reward model. The optimal policy under a KL-constrained reward objective has a closed form — which means the policy *is* the reward model, up to a constant. So you can just optimize the policy directly on the preference pairs.**

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

- $x$ — prompt; $y_w$ — preferred ("winning") completion; $y_l$ — rejected.
- $\pi_\theta$ — the policy being trained. $\pi_{\text{ref}}$ — frozen reference (usually the SFT checkpoint).
- $\beta$ — KL strength, typically **0.1** (range 0.01–0.5) [U — verify against TRL DPOConfig default]. Higher $\beta$ = stay closer to reference.
- $\sigma$ — logistic sigmoid.
- Memory: **two** models (policy + frozen ref). With LoRA you get the reference **for free** by disabling the adapter — a genuinely elegant trick worth showing.

**The 2026 successors [V-2°]:**

| Method | Core change | What it buys | Cost |
|---|---|---|---|
| **IPO** | replaces DPO's unbounded log-sigmoid with a regression target | fixes DPO's tendency to drive $\pi(y_l) \to 0$ without bound; has a well-defined stopping point | still needs ref model |
| **KTO** | uses **unpaired** binary 👍/👎 signals | you can use production thumbs-up data, which is *abundant and free*; no pairwise collection | slightly weaker than DPO given equal-quality pairs |
| **ORPO** | folds SFT + preference into **one** objective via an odds-ratio penalty | **no reference model** → halves memory; one training pass instead of two | less controllable; ties you to one stage |
| **SimPO** | uses **length-normalized average log-prob** as the implicit reward; no ref model | no reference model; reported **+6.4 on AlpacaEval 2, +7.5 on Arena-Hard vs DPO** [V-2° — vendor-adjacent claim, treat as directional] | length normalization is doing a lot of the work; see the honest note below |
| **GRPO** | **online** RL; sample a group of $G$ responses, use the group mean/std as the baseline; **no critic/value network** | the workhorse for reasoning; enables RLVR | needs generation in the loop → slow; needs a reward *function* |
| **DAPO** | GRPO with decoupled clipping, dynamic sampling, token-level loss [U — verify details] | stability fixes for long-CoT GRPO | complexity |

**GRPO advantage (the equation that makes it click):**
$$
A_i = \frac{r_i - \text{mean}(r_1 \dots r_G)}{\text{std}(r_1 \dots r_G)}
$$
$G$ = group size, typically **8–64** [V-2°]; TRL's `num_generations` default in the LoRA guide is 8–16 [V]. **The insight: PPO needs a learned value network to know "was this better than average?" GRPO just *asks 16 times and takes the average*. The critic was replaceable by a sample mean.** That is a beautiful, teachable simplification and it is *the* reason GRPO took over.

### ★ RLVR — the thing that actually changed 2025–2026
> **If you can *check* the answer with a program — a unit test, a math grader, a JSON schema validator — you don't need a reward model or human preferences at all. You have a ground-truth reward, and you can do real RL against it.**

This is where reasoning models come from, and it's the most *actionable* alignment idea for the learner, because **his own domain probably has a checker.** If his task has a verifiable output (does the generated config parse? does the extracted JSON match the schema? does the SQL run?), GRPO+RLVR is on the table on a Spark for a small model. TRL ships `trl.rewards.reasoning_accuracy_reward` and supports `--vllm_mode colocate` for in-loop generation [V]. **Flag this to the architect as a possible capstone.**

### ⚠️ Misconception box: "Alignment/RLHF teaches the model facts"
It does not, and this is the single most consequential misconception in the whole track because it's the one that makes people fine-tune when they should RAG. **Post-training reweights behaviours the base model already had.** It cannot install knowledge that isn't in the base model's weights, and attempting to do so via SFT produces **confident hallucination** — you've taught the model that "answering questions about topic X in an authoritative tone" is the expected behaviour, without giving it any way to know the answers. **Correction:** this is not a bug in your training; it is the predicted result. See §6.

### ⚠️ Misconception box: "There's a 'best' alignment algorithm"
There is real, unresolved disagreement here (see §15). The honest framing for 2026: **DPO is the default because it's simple and the tooling is mature; SimPO/ORPO win on memory; KTO wins when your data is binary; GRPO wins when you have a verifiable reward.** Anyone claiming a clean ranking is selling something. Many of the reported margins are within the noise of judge-based eval (§12).

---

## 6. ★★ THE CENTRAL DECISION: prompt vs RAG vs fine-tune

**This is the most important section in the track.** It comes before the memory ledger, before LoRA, before any code.

### The one-sentence rule, which must be repeated until it's reflexive

> ## **Fine-tuning teaches FORM. RAG supplies FACTS.**
> **If you can't answer the question by pasting the answer into the prompt, fine-tuning won't fix it. If you can, RAG is doing it automatically and you're done.**

### The honest decision framework

Ask, **in this order** — and stop at the first "yes":

| # | Question | If YES → | Why |
|---|---|---|---|
| 1 | Does a **better prompt** fix it? Have you actually tried few-shot with 5 real examples? | **Prompt.** Done. | Zero cost, zero latency, instant iteration, no maintenance. **Most people skip this and it works ~40% of the time.** |
| 2 | Is the problem that the model **doesn't know something**? Facts, your documents, your database, anything after its cutoff, anything that changes? | **RAG.** | Facts live in a store you can update in 5 seconds. Fine-tuning bakes them into weights you'd have to retrain to change — and even then unreliably. |
| 3 | Does the answer need to be **attributable / auditable / citable**? | **RAG.** | A fine-tuned model cannot cite. RAG hands you the source document for free. |
| 4 | Does the knowledge **change** — weekly, daily, ever? | **RAG.** | Retraining as a data-update mechanism is insane. |
| 5 | Is the problem **form**? Output format, house style, tone, a domain's idiom, a rigid schema, a specific reasoning trace shape, refusing what you want refused? | **Fine-tune.** | This is exactly and only what fine-tuning is good at. |
| 6 | Is the problem **latency or cost** — your prompt is 4,000 tokens of instructions and examples on every call? | **Fine-tune** (distill the prompt into weights). | Genuinely one of the best reasons to fine-tune and the most underrated. See the arithmetic below. |
| 7 | Is the problem that RAG works but the model **won't use the retrieved context properly** — ignores it, formats wrong, won't cite in your format? | **Both.** Fine-tune on the *RAG-augmented format*. | This is the mature endgame and where most serious systems land. |

### ⚠️ THE misconception — give it the biggest warning box in the course
> **"I'll fine-tune the model on my company documents so it knows them."**

This is the single most common and most expensive mistake in the field, and the learner will want to do exactly this, because his stated goal is "for their own specific application or database."

**Why it fails, mechanically:** SFT on documents optimizes $-\log p_\theta(\text{document tokens})$. That makes the document *likely*, i.e. it teaches the model to **write text that sounds like your documents**. It does not build a retrievable index. Facts get smeared across 8 billion parameters at a learning rate of $2\times10^{-4}$ for 3 epochs. The model will now produce **fluent, confident, correctly-formatted, plausibly-toned wrong answers** — which is strictly worse than before, because the style now signals authority it hasn't earned.

**The correction that actually fixes it:** *"Fine-tuning changes the model's* ***prior***, *not its* ***evidence***. *If you want the model to know a fact at inference time, the fact has to be in the context window at inference time. There is no third option."*

**The empirical backstop:** have the learner fine-tune an 8B on 200 documents and then ask it a factual question whose answer is in document #147. It will hallucinate. Then have them RAG the same corpus. It answers correctly with a citation. **Ship this as a notebook (§13, artifact #6). One hour of their time permanently inoculates them.**

### The cost arithmetic for reason #6 (prompt distillation)
Suppose your prompt carries 4,000 tokens of instructions + few-shot examples, and you make 50,000 calls/month.
- Prefill cost: $4{,}000 \times 50{,}000 = 2\times10^{8}$ tokens/month of pure overhead.
- On a Spark at ~8,000 tok/s prefill [V-2°]: $2\times10^{8} / 8000 = 25{,}000$ s = **6.9 hours/month of pure prefill, just re-reading your own instructions.**
- Fine-tune the behaviour in, drop the prompt to 200 tokens: 20 min/month. **A 20× reduction, from an hour of LoRA training.**
This is a genuinely strong argument for fine-tuning and it has nothing to do with knowledge. Lead reason #6 with it.

### 🎛️ Demo A7: The decision tree, but honest
**Plot:** an interactive flowchart. The learner answers the 7 questions; the path lights up; the leaf shows the recommendation **plus an estimated cost in hours and GB and an estimated maintenance burden**.
**The design requirement that makes this non-decorative:** each leaf must show the **counterfactual** — "you chose fine-tune; here is what RAG would have cost and here is the specific failure you should expect." And it must be capable of telling the learner **"do nothing, your prompt is fine."** A decision tree that never recommends the null action is propaganda.

---

## 7. RAG, properly

The learner's goal explicitly says "for their own specific application or database." Per §6, **RAG is more likely to be the right answer for the "database" half of that goal than fine-tuning is.** Give it real depth, not a token section.

### Intuition first
> **RAG is not AI. RAG is a search engine bolted to a summarizer. The quality of your RAG system is ~80% the quality of your search and ~20% the model. Everyone tunes the model.**

### The pipeline, with real parameters

```
Documents ─chunk─▶ Chunks ─embed─▶ Vectors ─index─▶ Vector DB
                                                        │
Query ─embed─▶ q ──────── top-N retrieve (N≈20) ────────┘
                    │
                    ├─ BM25 keyword retrieve (hybrid) ─┐
                    │                                   │
                    └────── fuse (RRF) ─────────────────┘
                                    │
                          rerank (cross-encoder) → top-5
                                    │
                          stuff into prompt → LLM
```

### Chunking [V-2°]
- **Default that works for ~80% of cases: recursive character splitting, 300–500 tokens, 10–15% overlap.** Ship this as the starting number.
- **Fixed-size:** simple, breaks mid-sentence, mid-table, mid-function. Fine for prose, terrible for code and structured docs.
- **Recursive:** split on `\n\n` → `\n` → `. ` → ` `, taking the largest unit that fits. The sane default.
- **Semantic:** embed sentence-by-sentence, start a new chunk when $\cos(e_i, e_{i+1})$ drops below a threshold. Costs an embedding pass over every sentence; the wins are real but modest.
- **Contextual retrieval** (Anthropic, 2024): prepend an LLM-generated 1–2 sentence summary of *where this chunk sits in the document* to each chunk before embedding. **Reported: up to 67% reduction in top-20 retrieval failures when combined with reranking** [V-2° — this is a vendor-published figure; present it as such, not as an independent result].
- **The tradeoff, stated as a tension the learner must resolve for their own corpus:** small chunks → precise retrieval, but the retrieved chunk may lack the context to be useful. Large chunks → the answer is in there somewhere, but the embedding is a mush of 5 topics and retrieval degrades. **Short factoid queries reward small chunks; multi-hop reasoning queries reward large ones.** There is no universal answer; this is an empirical question about *their* corpus, and the course should say so rather than hand them a number and pretend.

### Embedding
$$
\text{sim}(q, c) = \cos(e_q, e_c) = \frac{e_q \cdot e_c}{\|e_q\|\|e_c\|}
$$
$e_q, e_c \in \mathbb{R}^{d}$; $d$ typically 384 / 768 / 1024 / 1536 / 3072 depending on model.
- **Storage math:** 100,000 chunks × 1024 dims × 4 bytes (fp32) = **410 MB**. At fp16, 205 MB. With binary quantization (1 bit/dim), **12.8 MB** at ~95% recall retention [U — verify the recall figure]. This is a nice, concrete "quantization applies here too" callback.
- **Matryoshka embeddings** [U — verify current model support]: models trained so that the first $k$ dimensions are independently useful, letting you truncate 1024→256 and lose little. Retrieve cheap on 256, rerank on 1024.
- **Pick the embedding model from MTEB, not from vibes** [U — verify MTEB is still the standard leaderboard in 2026 and what currently tops it for retrieval]. **⚠️ Flag: I could not verify the current 2026 SOTA embedding models this session. The architect must fill this in with a live check.**

### ⚠️ Misconception box: "Cosine similarity finds relevant chunks"
It finds **similar** chunks. Similar ≠ relevant, and the gap is where RAG systems die. The classic failure: query "What are the risks of Project Titan?" retrieves five chunks that *talk about Project Titan* and zero that *discuss risks*, because "Project Titan" dominates the embedding. **Correction: this is exactly what the reranker fixes** — a cross-encoder sees the query and chunk *together* and can model relevance, not just similarity. This misconception is the reason reranking is not optional.

### Hybrid retrieval
Dense (embedding) retrieval fails on **exact identifiers**: part numbers, error codes, `SKU-88213`, rare proper nouns. BM25 (sparse, lexical) nails those and fails on paraphrase. **You need both.** Fuse with Reciprocal Rank Fusion:
$$
\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}, \qquad k = 60 \text{ (conventional)}
$$
RRF needs no score calibration between retrievers — it only uses ranks. That's why it's the default. [U — verify $k=60$ is still the convention.]

### Reranking [V-2°]
- **Bi-encoder** (the embedder): encodes $q$ and $c$ **separately**. Fast (chunks pre-computed), less accurate.
- **Cross-encoder** (the reranker): encodes $[q; c]$ **jointly** through a transformer, outputs a scalar relevance. Accurate, and $O(N)$ forward passes at query time — so you can only afford it on a shortlist.
- **The rule of thumb, and it's a good one: retrieve 20, rerank to 5, send 3–5 to the LLM. Reranking 100+ candidates rarely pays.** [V-2°]
- **This is where most enterprise RAG becomes production-grade.** [V-2°] If the learner does one thing beyond naive vector search, it's this.

### Vector DBs — honest guidance
- **< 100k chunks: you do not need a vector database.** A NumPy array and `np.argsort(embeddings @ q)` is exact, instant, and has zero operational burden. **Say this loudly** — the field oversells vector DBs by an order of magnitude. 100k × 1024 fp32 = 410 MB; a brute-force dot product over that is a ~milliseconds-scale matmul on the Spark's GPU.
- **100k–10M:** FAISS (HNSW or IVF-PQ), or a local embedded DB (Chroma / LanceDB / `pgvector` with HNSW) [U — verify these are the 2026 recommendations].
- **> 10M / multi-tenant / needs a server:** Qdrant, Weaviate, Milvus, pgvector [U].
- **HNSW** is approximate. Its `ef_search` parameter trades recall for latency. **The learner should know his retrieval is lossy** — a real and rarely-mentioned fact.

### Long context vs RAG — a genuine 2026 tension (see §15)
With 10M-token context windows claimed [V-2° — Llama 4 Scout; treat the number skeptically, see §15], "just put everything in the prompt" is a real option. **The honest position:** long context is *expensive* (prefill is compute-bound, KV cache is linear — see Demo A5), *lossy* (needle-in-haystack degrades in the middle), and *unattributable*. RAG is cheap, fast, and citable. **The 2026 consensus is that they are complementary: RAG narrows to 50k tokens, long context absorbs the imprecision so your chunking doesn't have to be perfect.** Long context made RAG *easier*, not obsolete. But flag this as contested.

### 🎛️ Demo A8: RAG retrieval sandbox (★ high value)
**Plot:** left, a real ~200-chunk corpus rendered as a 2-D scatter (UMAP/t-SNE precomputed offline, shipped as coordinates). Right, a ranked result list. The query vector is plotted as a star; the top-N are highlighted and connected by lines.
**Sliders/controls:** chunk size (100 → 1000 tokens — requires **pre-computing embeddings at 5 chunk sizes offline** and shipping all 5 sets; be honest in the UI that this is a discrete switch, not continuous); overlap %; top-N; a **hybrid weight slider** (pure BM25 ↔ pure dense); a **rerank on/off** toggle.
**Exact math the JS must implement:** real cosine similarity over the shipped embedding vectors (a 200×384 matrix is trivial in JS); **real BM25** — implement it, it's 20 lines:
$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t)\cdot\frac{f(t,d)\cdot(k_1+1)}{f(t,d) + k_1\cdot(1 - b + b\cdot\frac{|d|}{\text{avgdl}})}, \quad k_1 = 1.2,\ b = 0.75
$$
Real RRF fusion. For the reranker toggle, ship **pre-computed cross-encoder scores** for a fixed set of ~10 canned queries (be explicit that this is canned) — with free-text queries, gray the rerank toggle out and say why.
**The insight:** ship **two specific canned queries** that make the lesson unmissable. Query 1: `"error code E-4471"` — the dense retriever returns five chunks about *errors in general* and misses the one chunk containing the literal string; slide toward BM25 and it snaps to #1. Query 2: `"why did the project fail?"` — BM25 returns nothing useful (no lexical overlap); slide to dense and it works. **The learner discovers hybrid retrieval by being unable to solve both queries with one setting.** That's the demo. Then rerank-on visibly reorders the "similar but not relevant" chunks down, closing the §7 misconception box.

---

## 8. ★ THE MEMORY LEDGER — full fine-tuning, and why it fails

This is the hinge of the track. **Everything after this is a consequence.**

### Intuition first
> **Your model's weights are the *smallest* thing in the room. Training it means also storing its gradient, and Adam's two running averages, and a full-precision master copy — so the ledger is not 16 GB, it's 131 GB, and you have 128.**

### The bytes-per-parameter table (memorize this)

Standard bf16 mixed-precision + AdamW:

| Item | Precision | Bytes/param | Why it exists |
|---|---|---|---|
| Weights | bf16 | 2 | the forward pass |
| Gradients | bf16 | 2 | backprop's output |
| AdamW $m$ (1st moment) | fp32 | 4 | momentum |
| AdamW $v$ (2nd moment) | fp32 | 4 | per-param adaptive LR |
| **Master weights** | fp32 | 4 | **bf16 has ~3 decimal digits; a $2\times10^{-5}$ update to a weight of order 1 is *silently rounded to zero* in bf16. Without the fp32 master copy, training stalls.** |
| **Total** | | **16** | |

**The "4×" the learner will hear** comes from the fp32 framing: 4 (weights) + 4 (grads) + 8 (Adam $m,v$) = 16 bytes/param = **4× the fp32 weight size**. The mixed-precision framing gives **8× the bf16 weight size**. **Both are correct and they describe the same 16 bytes/param. State both explicitly** — the learner will otherwise hit a contradiction between two blog posts and lose trust.

**The fp32 master weights bullet deserves its own paragraph.** It is the least-explained and most surprising line in the table, it is a *numerical precision* argument (which the trunk already taught), and "why not just keep everything in bf16?" is the exact question a rusty-but-trained engineer asks. Answer it properly.

### ★ The worked example — Qwen3-8B on a DGX Spark [V for param count]

$N = 8.19 \times 10^9$ (derived exactly in §2/§9 from the verified config).

$$
\text{Memory} = 8.19\times10^9 \times 16\ \text{bytes} = 1.31\times10^{11}\ \text{B} = \mathbf{131\ GB}
$$

**You have 128 GB. You are 3 GB over — before a single activation.**

**This is the best number in the entire brief. Build the section around it.** It is not a comfortable 10× miss that invites "well, obviously"; it is an *infuriating* near-miss that makes the learner want to fight for those 3 GB — and then discover that even winning that fight leaves no room for activations, the KV-free forward cache, or the dataset. The lesson lands emotionally, which is why it sticks.

Add activations: at `B=1, T=2048` with gradient checkpointing on, roughly 2–6 GB [U — order-of-magnitude; the architect should measure this rather than cite it]. Without checkpointing, far more. **Verdict: full fine-tuning Qwen3-8B on a Spark with plain AdamW does not fit.**

**The escape hatches, and what each costs:**

| Change | Bytes/param | Total (8.19B) | Fits in 128 GB? | Cost |
|---|---|---|---|---|
| Plain AdamW mixed precision | 16 | **131 GB** | ❌ (barely) | — |
| **8-bit Adam** (`bnb` `adamw_8bit`): $m,v$ → 1 byte | 10 | **82 GB** | ✅ | negligible quality impact [U — verify] |
| 8-bit Adam, **no fp32 master** (pure bf16) | 6 | **49 GB** | ✅ | risk of stalled updates; use with caution |
| **SGD + momentum** ($m$ only, no $v$) | 8 (bf16 w+g, fp32 m, fp32 master... ) ≈ 12 | ~98 GB | ✅ | much worse convergence on transformers; not recommended |
| **LoRA r=16, all-linear** | see §9 | **~17 GB** | ✅✅ | see §9 and §15 |
| **QLoRA r=16 (NF4 base)** | see §9 | **~5 GB** | ✅✅✅ | see §9 |

**The pedagogical arc is now complete and it is a good one:** 131 GB → doesn't fit → 82 GB with 8-bit Adam → fits, but you've spent your whole machine on an 8B model and can't touch a 70B → **enter LoRA at 17 GB and QLoRA at 5 GB** → and now the 70B is on the table. The learner has *earned* LoRA rather than been handed it.

### 🎛️ Demo A9: The memory ledger (★★ the most important demo in the track)
**Plot:** a stacked vertical bar, one segment per ledger row (weights / gradients / Adam-m / Adam-v / master / activations / KV), colour-coded, with a **red horizontal line at 128 GB** labelled "your DGX Spark" and a fainter one at 24 GB ("an RTX 4090").
**Controls:** model size slider (0.5B → 405B, log); method radio (Full / LoRA / QLoRA / DoRA); optimizer dropdown (AdamW / adamw_8bit / adafactor / SGD-momentum); base dtype (fp32/bf16/fp8/nf4); LoRA rank (1 → 512); target-modules toggle (attn-only / all-linear); batch size; sequence length; gradient-checkpointing toggle.
**Exact math:** the bytes-per-param table for the frozen/trained split, plus the **real LoRA parameter count computed from the actual per-layer matrix shapes** (see §9 — do NOT use a hand-wave percentage; compute $r(d_{\text{in}} + d_{\text{out}})$ per targeted matrix and sum, using shipped config presets for 5 real models). KV formula from §4. Activations as a stated, labelled estimate — **and label it as an estimate in the UI.** Honesty about which numbers are exact and which are modelled is itself a teaching move here.
**The insight when they drag it:** three distinct discoveries, in order. (1) At 8B/Full/AdamW the bar is **just barely** over the red line — the 131-vs-128 gut-punch, self-inflicted. (2) Switching Full→LoRA collapses four of the seven segments to slivers, and the learner *sees* that LoRA didn't shrink the model, it shrank the *optimizer state* — which is the actual mechanism and the thing everyone gets wrong. (3) Dragging rank 16→512 with LoRA barely moves the bar, but dragging model size 8B→70B with QLoRA still fits — **so rank is nearly free and you should stop being stingy with it**, which is precisely the Thinking Machines finding (§15) discovered by hand.

---

## 9. LoRA and PEFT

### Intuition first
> **Fine-tuning changes the weight matrix by some $\Delta W$. LoRA's bet is that this $\Delta W$ — the *change*, not the weight — is nearly flat: it has a few important directions and the rest is noise. So don't store a 4096×4096 change. Store two skinny matrices whose product is that change.**

The emphasis must be on **"the change is low-rank," not "the model is low-rank."** The pretrained $W$ is emphatically full-rank. This distinction is the whole idea and it is the thing learners get wrong.

### The math

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x
$$

- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ — frozen pretrained weight.
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ — initialized $\mathcal{N}(0, \sigma^2)$ (Kaiming-ish).
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ — **initialized to zero.**
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$ — the rank; unitless integer.
- $\alpha$ — scaling constant; unitless.
- $x \in \mathbb{R}^{d_{\text{in}}}$, $h \in \mathbb{R}^{d_{\text{out}}}$.

**Why $B = 0$ at init:** $BA = 0$, so $\Delta W = 0$, so **the adapted model is bit-identical to the base model at step 0.** Training starts from the base model, not from a randomly perturbed one. Say this — it's why LoRA is safe to attach to a production model. (And it's why `init_lora_weights=False` exists only "for debugging purposes" [V — PEFT docs].)

**Parameter count per adapted matrix:**
$$
r(d_{\text{in}} + d_{\text{out}}) \quad\text{vs.}\quad d_{\text{in}} \cdot d_{\text{out}}
$$
For a 4096×4096 matrix at $r=16$: $16 \times 8192 = 131{,}072$ vs $16{,}777{,}216$. **A 128× reduction on that matrix.**

### ★ Worked example — the exact LoRA parameter count for Qwen3-8B [V, derived from verified config]

From `config.json`: $d_{\text{model}} = 4096$, $n_{\text{heads}}=32$, $n_{\text{kv}}=8$, $d_{\text{head}}=128$, $d_{\text{ff}} = 12{,}288$, $L = 36$.

So the per-layer matrix shapes are:

| Matrix | Shape ($d_{\text{out}} \times d_{\text{in}}$) | Base params | LoRA params at $r=16$ |
|---|---|---|---|
| `q_proj` | 4096 × 4096 | 16,777,216 | $16(4096+4096) = 131{,}072$ |
| `k_proj` | 1024 × 4096 | 4,194,304 | $16(4096+1024) = 81{,}920$ |
| `v_proj` | 1024 × 4096 | 4,194,304 | 81,920 |
| `o_proj` | 4096 × 4096 | 16,777,216 | 131,072 |
| `gate_proj` | 12288 × 4096 | 50,331,648 | $16(4096+12288) = 262{,}144$ |
| `up_proj` | 12288 × 4096 | 50,331,648 | 262,144 |
| `down_proj` | 4096 × 12288 | 50,331,648 | 262,144 |
| **per layer** | | **192,937,984** | **1,212,416** |
| **× 36 layers** | | **6,945,767,424** | **43,646,976** |

**Check the base count:** $6.9458\text{B}$ (layers) $+ 0.6223\text{B}$ (embed) $+ 0.6223\text{B}$ (lm_head) $= \mathbf{8.190\text{B}}$, and the Qwen card says **8.2B total / 6.95B non-embedding** [V]. **It closes to three significant figures.** This is a gift — a fully verified, self-consistent parameter derivation the learner can do by hand and check against a published card. **Build a §9 exercise around reproducing this table.**

**So: LoRA $r=16$, all-linear ⇒ 43.6M trainable params = 0.53% of the model.**

**Memory:** $43.6\text{M} \times 16\ \text{B/param} = 0.70\ \text{GB}$ of optimizer/gradient state, plus the frozen base at bf16 = 16.4 GB.
$$
\textbf{~17.1 GB total, vs 131 GB for full fine-tuning. A 7.7× reduction.}
$$
And note **where** the reduction came from: the frozen 16.4 GB didn't shrink at all. **All of the win is in the optimizer state.** That is the mechanism and Demo A9 shows it.

**At $r=256$ (the Thinking Machines SFT recommendation — see §15):** LoRA params scale exactly linearly in $r$, so $43.6\text{M} \times 16 = \mathbf{698\text{M}}$ trainable = 8.5% of the model. Memory: $698\text{M} \times 16 = 11.2$ GB + 16.4 GB base = **27.6 GB.** **Still trivially fits on a Spark.** This is the number that should make the learner stop using $r=8$ out of superstition.

### What $r$ and $\alpha$ actually do

- **$r$ = capacity.** It is the maximum rank of $\Delta W$, i.e. the number of independent directions your update can move in. The Thinking Machines framing is the useful one: **the adapter can store roughly $2r$ bits per parameter-slot, and if your dataset's information content exceeds the adapter's capacity, LoRA underperforms full fine-tuning. Below that threshold, it matches it.** [V-2° — HF TRL docs restating the TM blog]
- **$\alpha$ = a scaling constant, and it is NOT a second capacity knob.** The effective scale is $\alpha/r$. Changing $\alpha$ at fixed $r$ is *nearly* equivalent to changing the learning rate — which is why the folklore "always set $\alpha = 2r$" exists (it holds the scale at a constant 2.0 as you vary $r$, so your LR doesn't need retuning).

### ⚠️ Misconception box: "α is a strength dial — turn it up to learn harder"
$\alpha$ and the learning rate are **nearly redundant**. Tuning both independently is a waste of a hyperparameter search. **Correction:** pick a convention and hold it — either $\alpha = 2r$ (folklore) or $\alpha = 16$ fixed (Thinking Machines) — and **tune the learning rate instead.** But be aware these two conventions **genuinely disagree** and the disagreement is unresolved — see §15. That's a real fork, not a matter of taste.

### ⚠️ Misconception box: "Attention is where the knowledge is, so target `q_proj` and `v_proj`"
This is the original LoRA paper's setup and **it is the most persistent stale advice in the field.** The 2026 finding is unambiguous:
> **Apply LoRA to *all* linear layers, especially the MLPs. Attention-only LoRA underperforms even when you raise the rank to match the parameter count.** [V-2° — TRL docs / Thinking Machines]

**Why it should have been obvious:** look at the table above. The MLP matrices are **150,994,944 of the 192,937,984 params per layer — 78% of the model lives in the MLPs.** Targeting attention-only means you froze 78% of the model and then wondered why it didn't learn. **Use `target_modules="all-linear"`.** [V — this is a real PEFT value; it selects all linear/Conv1D modules and excludes the output layer for a `PreTrainedModel`.]

This misconception is worth a big box because it is (a) universally repeated, (b) wrong, (c) explained by a number the learner just computed themselves in the §9 table, and (d) fixed by a one-word config change.

### ⚠️ Misconception box: "LoRA is a lossy approximation of real fine-tuning"
The 2024 consensus was "LoRA learns less and forgets less" — a real trade-off. **The 2025–26 refinement:** LoRA matches full fine-tuning **when it isn't capacity-constrained**. A rank-32 adapter on a 7B matched full FT up to ~50,000 examples; beyond that, raising to 64 or 128 restored parity. [V-2°] **Correction: LoRA isn't worse; under-ranked LoRA is worse. And since rank is nearly free in memory (see the r=256 number above), being stingy with rank is the actual error.**

### QLoRA

> **Intuition: the frozen base model is 16.4 GB of read-only weights you never update. Why are you storing read-only data at full precision? Compress it to 4 bits, decompress each block on the fly during the forward pass, and keep the *adapter* — the only thing that gets a gradient — in bf16.**

**NF4 (4-bit NormalFloat):** the key insight is that pretrained weights are approximately $\mathcal{N}(0, \sigma^2)$. So don't use uniformly-spaced 4-bit levels — use the **16 quantiles of a standard normal**, so each of the 16 codes is equally likely. It is information-theoretically optimal *for normally-distributed data*, which is exactly what the weights are. [U — verify the "information-theoretically optimal" framing against the QLoRA paper before asserting it that strongly.]

**Blockwise quantization:** quantize in blocks of 64, each with its own fp32 scale. **Double quantization:** those scales are themselves 8-bit quantized (in blocks of 256), saving ~0.37 bits/param [U — verify the constant]. Effective: **~4.13 bits/param ≈ 0.516 bytes/param** [U — the QLoRA paper's figure; verify].

**Worked example — QLoRA on Qwen3-8B:**
$$
8{,}190{,}735{,}360 \times 0.515869 = \mathbf{4.225\ GB} \text{ (base, NF4 double-quant; } 0.516 \text{ B/param [VP], D-04/§Z-4)}
$$
$+\ 0.61$ GB (r=16 all-linear state) $+$ activations $\approx$ **~5–7 GB.** *(NF4 double-quant defaults OFF; without it, 4.5 bits = 4.607 GB base.)*
**vs 131.05 GB for full fine-tuning. A 19–26× reduction** (say the range, not "~20×"; D-20).

**Worked example — QLoRA on Llama-3.3-70B [DER from fetched config: d=8192, L=80, d_ff=28672, H_kv=8]:**
$70.6\times10^9 \times 0.516 = 36.4$ GB base. Adapters at $r=16$ all-linear = **207,093,760 params (0.294%)** × 16 B = **3.31 GB state**. **≈ 39.7 GB + activations. Comfortably inside 128 GB.** *(The old "~0.4% ≈ 280M → 4.5 GB → ≈41 GB" used the hand-wave % this brief's own demo spec forbids — corrected per D-20.)* This is precisely why NVIDIA can publish a Llama 3.3 70B QLoRA benchmark for the Spark (§14). **128 GB unified memory + QLoRA is the specific combination that puts a 70B on the learner's desk. Make that the payoff moment of the track.**

**Paged optimizers:** `bnb` can page optimizer state to CPU on OOM spikes. On the Spark's **unified** memory this is a much weaker concept than on a discrete GPU (there is no separate host memory to page *to* — it's one pool). **Flag as a place where Spark differs from every tutorial the learner will read.**

### ⚠️ Misconception box: "QLoRA is just LoRA but faster"
**QLoRA is slower than LoRA**, typically ~20–40% [U — verify]. You pay dequantization on every forward pass. QLoRA trades **compute for memory**. You use it when LoRA doesn't fit, not when you want speed. On a 128 GB Spark, **8B QLoRA is the wrong choice** — LoRA fits at 17 GB, so use LoRA. QLoRA is for the 70B.

### DoRA and the 2026 PEFT menagerie [V — all confirmed present in PEFT 0.19.1]

**DoRA (Weight-Decomposed Low-Rank Adaptation)** — decompose the weight into **magnitude** and **direction**:
$$
W' = \underbrace{m}_{\text{learned magnitude}} \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$
where $\|\cdot\|_c$ is the column-wise norm and $m \in \mathbb{R}^{d_{\text{in}}}$ is a separate learnable vector.
- **Intuition: LoRA has to spend rank on changing a weight's *size* and its *direction* at once. DoRA gives size its own free parameter, so all of the rank goes to direction.**
- **Improves LoRA especially at low rank** [V — PEFT docs say exactly this]. Which implies the corollary: **if you're already at $r=256$, DoRA buys you little.** That's a useful, honest, non-obvious inference.
- **Costs:** "bigger overhead than pure LoRA"; PEFT recommends merging for inference; linear and Conv2D only [V]. Enable with `use_dora=True`.

**rsLoRA** — sets the scale to $\alpha/\sqrt{r}$ instead of $\alpha/r$ [V]. The claim: at high rank, $\alpha/r$ over-shrinks the update and the adapter under-trains; $\sqrt{r}$ fixes it. **This matters precisely in the $r=256$ regime the field now recommends.** `use_rslora=True`.

**Smarter initializations** (all real in PEFT 0.19.1 [V], all via `init_lora_weights=`):
- `"pissa"` — initialize $A, B$ from the **principal** singular vectors of $W_0$, so the adapter starts on the directions that already matter. Converges faster; **reduces quantization error vs QLoRA** [V — PEFT docs claim this]. `"pissa_niter_16"` does fast-SVD and initializes a 7B "within seconds" [V].
- `"eva"` — data-driven: SVD of the **layer input activations** on your actual fine-tuning data, then **redistributes rank across layers** (`rho`, default 2.0, caps a layer at $2r$). PEFT calls it "SOTA" [V — vendor claim, present as such].
- `"corda"` — context-oriented; has a **"Knowledge-Preserved Mode"** that explicitly targets catastrophic forgetting [V]. Relevant to §12.
- `"olora"` / `"orthogonal"` — orthogonal init.
- `"loftq"` — jointly quantizes the base and inits LoRA to **minimize quantization error**. Note PEFT also ships `replace_lora_weights_loftq()` to apply it on-the-fly to an already-bnb-quantized model [V].

**LoRA-GA** [V — in PEFT 0.19.1]: init from the SVD of **estimated gradients** (run 64–128 batches first), aligning the initial update direction with full FT's. **Claim: 2–4× faster convergence, same final performance.** Overhead: 1–2 min for 64 batches. **Does not support quantized models** (needs full-precision weights) [V] — so **LoRA-GA and QLoRA are mutually exclusive**, which is a real and useful constraint to state.

**Intruder dimension mitigation** [V — `peft.tuners.lora.intruders.reduce_intruder_dimension`, based on "LoRA vs Full Fine-tuning: An Illusion of Equivalence," arXiv 2410.21228]. This is worth a callout. The finding: LoRA adapters introduce **"intruder dimensions"** — singular directions with near-zero cosine similarity to any pretrained singular vector. They're the mechanism of LoRA's forgetting. PEFT ships a **post-hoc** fix that subtracts them, with a `migration_lambda` (default 0.75) trading task accuracy against recovered general knowledge. **This is a genuinely interesting, current, and rarely-taught result, and it directly contradicts a naive reading of "LoRA forgets less."** Flag for §15.

**aLoRA (Activated LoRA)** [V]: the adapter activates only on tokens **after** an invocation string, so **the KV cache before the invocation is interchangeable with the base model's.** In an agentic pipeline that switches between base and adapter, you skip re-prefilling the shared context — PEFT claims "an order of magnitude or more" inference speedup on vLLM depending on shared-context length [V]. Cannot be merged. Niche but a lovely example of the KV cache (§4) driving an architecture decision.

**`target_parameters`** [V]: for MoE models, HF Transformers implements experts as `nn.Parameter`, not `nn.Linear`, so `target_modules` **cannot reach them**. PEFT's example is literally Llama 4: `target_parameters=['feed_forward.experts.gate_up_proj', 'feed_forward.experts.down_proj']`. **If the course covers MoE fine-tuning at all, this is a must-mention — `all-linear` will silently skip the majority of an MoE's parameters.**

### 🎛️ Demo A10: What rank actually buys (SVD)
**Plot:** three panels — original matrix as a heatmap; rank-$r$ reconstruction as a heatmap; the error. Plus a scree plot of the singular values $\sigma_1 \dots \sigma_n$ on a log axis, with the retained ones highlighted, and a live readout of $\|W - W_r\|_F / \|W\|_F$ and the compression ratio.
**Slider:** $r$, 1 → 64.
**Exact math:** a **real SVD in JS** (a 64×64 Jacobi SVD is ~60 lines and runs instantly). $W_r = \sum_{i=1}^{r}\sigma_i u_i v_i^T$.
**The critical design choice — and this is where most versions of this demo fail:** offer **three** matrices in a dropdown: (1) a random Gaussian matrix, (2) a **real slice of a pretrained $W_0$** (ship a 64×64 crop from a real Qwen3 `q_proj`), (3) a **real slice of a $\Delta W$ from an actual fine-tune** (do the fine-tune, save $W_{\text{after}} - W_{\text{before}}$, crop it, ship it).
**The insight — the whole reason to build this demo:** the scree plots are **completely different shapes**. The random matrix's singular values decay slowly (rank-32 reconstruction still looks like noise). The pretrained $W_0$'s decay slowly too — **it is nearly full rank, and a rank-16 approximation of it destroys the model**. But $\Delta W$'s singular values **fall off a cliff** — rank-8 captures most of it. **The learner sees, in one dropdown flip, that the LoRA hypothesis is a claim about $\Delta W$ and is *false* about $W_0$.** That is exactly the misconception in the box above, killed by direct observation. Nothing else in this brief teaches that as efficiently.

### 🎛️ Demo A11: The α/r scaling explorer
**Plot:** a small real 2-layer MLP trained live in-browser on a 2-D toy task (spiral classification), with a LoRA adapter attached. Show the decision boundary evolving, and a loss curve.
**Sliders:** $r$, $\alpha$, learning rate; a toggle for `use_rslora`.
**Exact math:** real forward/backward on the toy net, real LoRA parametrization with $\frac{\alpha}{r}BA$ (or $\frac{\alpha}{\sqrt r}BA$ when rsLoRA is on), real SGD/Adam. This is small enough to be genuinely live.
**The insight:** the learner sweeps $\alpha$ at fixed $r$ and LR, then sweeps LR at fixed $\alpha$, and **discovers the loss curves are nearly superimposable** — $\alpha$ and LR are redundant. That kills the "α is a strength dial" misconception by experiment rather than assertion. Then they flip rsLoRA on at $r=256$ and see it matters *only* at high rank.

---

## 10. Quantization

*(Recommend splitting per §1: the number-format part here, the deployment-format part at §13.)*

### Intuition first
> **A pretrained weight matrix is 16 million numbers all clustered near zero in a bell curve. You do not need 16 bits to say "this one is a bit left of centre." You need about 4 — as long as you're smart about where you put the 16 available levels, and as long as you re-anchor the scale every 16 to 64 weights so one outlier can't ruin the neighbourhood.**

**"Re-anchor every 16–64 weights" is the load-bearing half of that sentence and it's the half everyone omits.** Blockwise scaling *is* modern quantization; the bit-width is almost a detail.

### The general form
$$
w \approx s \cdot q, \qquad q \in \{\text{16 levels}\},\ s = \text{scale (per block)}
$$
**Effective bits/param** $= \text{bits}(q) + \frac{\text{bits}(s)}{\text{block size}}$.

Worked: NF4 with block 64 and an fp32 scale = $4 + 32/64 = 4.5$ bits. With **double quantization** (8-bit scales, blocks of 256): ~~$4 + 8/64 + 32/256 = 4.25$ bits~~ **← RESOLVED and REFUTED. The correct figure is $4 + 8/64 + 32/(64\cdot256) = 4.127$ bits $= 0.516$ B/param [VP], per `constants.md` §3 / decisions §Z-4.** The `32/256` term was an arithmetic error: the second-level fp32 constant amortizes over the *parameters* its 256 scales cover ($64\times256 = 16{,}384$), not over the 256 scales — so it is $32/16384 = 0.00195$ bits, not $32/256 = 0.125$. My original suspicion was right; the paper (arXiv 2305.14314 §3) wins. **Also note:** double-quant defaults **off**; un-opted-in NF4 is 4.5 bits = 0.5625 B/param. The 0.516 figure requires `bnb_4bit_use_double_quant=True`.

### The 2026 format table

| Format | Bits | Block | Scale format | Where it runs | Quality vs FP16 | Notes |
|---|---|---|---|---|---|---|
| **BF16** | 16 | — | — | everything | baseline | training default |
| **FP8** (E4M3) | 8 | per-tensor/channel | fp32 | Hopper, Blackwell | **0.5–2%** degradation [V-2°] | the safe inference default |
| **MXFP4** | 4 (E2M1) | **32** | **E8M0 (powers of 2!)** | Blackwell | **significant drop** [V-2°] | the power-of-two scale constraint is the problem |
| **NVFP4** | 4 (E2M1) | **16** | **FP8 micro-scale + fp32 global** | **Blackwell (GB10 native!)** | **<1%** on many tasks; DeepSeek-R1 MMLU 90.8→90.7 [V-2°] | **88% lower quantization error than MXFP4** [V-2°] |
| **NF4** | ~4.1–4.25 | 64 | fp32→8-bit (double quant) | anywhere (bnb) | small, for *training* | **QLoRA's format; a training format, not a serving format** |
| **GPTQ** | 3/4/8 | 128 | per-group | wide | ~90% retention [V-2° — low-confidence blog figure] | **largely superseded by AWQ for new releases** [V-2°] |
| **AWQ** | 4 | 128 | per-group | wide | ~95% retention [V-2° — same caveat] | protects the ~1% "salient" channels by scaling |
| **GGUF Q4_K_M** | ~4.5 | mixed | mixed | **llama.cpp/Ollama, CPU+GPU** | ~92% retention [V-2° — same caveat] | the local-inference default; k-quants use **different bit-widths for different tensors** |

> ⚠️ **The "quality retention %" column is from SEO-grade blog sources and the numbers are mutually inconsistent across sources. Do NOT print them as facts.** The *ordering* (FP8 > AWQ > GGUF-Q4_K_M ≳ GPTQ) is probably right; the specific percentages are not defensible. **Recommendation: either re-derive these on the learner's own hardware as a course exercise (which is better pedagogy anyway — see artifact #8) or present them as a qualitative ordering with no numbers.**

### The NVFP4 story is worth telling properly, because the Spark has it in hardware
> **MXFP4 forces every block's scale to be a power of two. That means the scale can only be 1, 2, 4, 8... — so if your block's values want a scale of 3, you round to 2 or 4 and eat up to a 33% error *before you've quantized a single weight*. NVFP4 lets the scale be an arbitrary FP8 value, and halves the block to 16. That's the whole difference, and it's worth 88% of the error.** [V-2°]

This is a rare case where a hardware format difference has a **one-sentence intuitive explanation** and a large measured effect. Use it. It also sets up the general lesson: **quantization error is dominated by scale granularity, not by the mantissa bits.**

### ⚠️ Misconception box: "4-bit means each weight is one of 16 evenly spaced values"
No modern 4-bit format does this. NF4 uses **normal quantiles** (unevenly spaced, dense near zero, matched to the weight distribution). NVFP4/MXFP4 use **E2M1 floating point** (exponent+mantissa, so also unevenly spaced). GGUF k-quants use **different bit-widths for different tensors in the same model**. **Correction: "4-bit" names a budget, not a scheme, and the scheme is where all the quality lives.**

### ⚠️ Misconception box: "Quantization makes the model faster because there's less math"
Almost entirely wrong for **decode**, which is what the learner cares about. **The math is the same amount** (you dequantize and multiply — often you do *more* work). The speedup is that you **read 4× fewer bytes from memory**, and decode is memory-bound (§4, Demo A6). **Correction: quantization is a bandwidth optimization wearing a compute costume.** *(Caveat: on Blackwell, NVFP4 tensor cores do genuinely execute FP4 matmuls natively, so for **prefill/training** — which are compute-bound — the math speedup is real too. State both halves; the distinction is exactly the prefill/decode split from §4.)*

### ⚠️ Misconception box: "I should train in the format I'll serve in"
No. **Train in NF4 (QLoRA) or bf16; serve in GGUF/AWQ/NVFP4.** These are different quantization families optimized for different things (NF4 for gradient-friendly frozen weights; AWQ/GGUF for inference kernels). You **merge your adapter into bf16 and then re-quantize for serving** (§13). **This is a real workflow gap that trips people: your QLoRA NF4 base is not your deployment artifact, and merging a bf16 adapter into an NF4 base is lossy** — you must merge into the *original* bf16 weights.

### 🎛️ Demo A12: The quantizer
**Plot:** top — a histogram of a **real** pretrained weight tensor (ship a real Qwen3 `q_proj` slice) with the quantization levels drawn as vertical lines. Bottom — a scatter of original vs dequantized weight, with the identity line; the spread off-diagonal *is* the error. A readout of RMS error and effective bits/param.
**Controls:** a format dropdown (**INT4-uniform / NF4 / E2M1-MXFP4 / E2M1-NVFP4 / INT8 / FP8-E4M3**), a **block size slider (4 → 4096, log)**, and a scale-format toggle (**fp32 / fp8 / power-of-two**).
**Exact math:** real quantize→dequantize round-trips for each format, computed live on the shipped real tensor. Real level placement (NF4 = normal quantiles — hardcode the 16 published values; E2M1 = enumerate the 16 representable floats). Real per-block scale computation ($s = \max|w| / \max|q|$).
**The insight — three of them, and they're all excellent:**
1. Switch INT4-uniform → NF4 on the **real** weight histogram: the uniform levels waste half their codes in the empty tails while the dense centre gets 3 levels; NF4's levels **cluster exactly where the mass is.** The "match the levels to the distribution" idea becomes obvious in one click.
2. Drag **block size** from 4096 down to 16 and watch RMS error collapse. **This is the single most important quantization lesson and it has nothing to do with bit-width.** Then drag bit-width instead and see it matter *less*. The learner discovers that granularity > precision.
3. Flip the scale format from **fp32 to power-of-two** at block 32 — the error jumps. **That is MXFP4 vs NVFP4, reproduced by hand in the browser, on a real tensor.** The learner just derived NVIDIA's design decision.

This demo is worth building carefully; it does more work than any other in §10.

---

## 11. Datasets — the part everyone underestimates

### Intuition first
> **Your model will learn exactly what your dataset demonstrates, including the parts you didn't mean to demonstrate. If every example in your set is 3 sentences long, you have trained a model that emits 3 sentences. Forever. About everything.**

That failure — "I fine-tuned and now it's weirdly terse" — is the most common surprised complaint, and it always traces to an unintended regularity in the data.

### Chat templates — the #1 source of silent failure

A chat template is a **Jinja template that turns a list of message dicts into one string of tokens.** Get it wrong and **nothing errors** — you just get a mediocre model. This is the worst failure mode there is: silent.

ChatML (Qwen family) [V — the exact token strings appear in TRL's docs]:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>
```

**The rules the learner must internalize:**
1. **Use `tokenizer.apply_chat_template()`. Never hand-roll the string.** Every family differs.
2. **Train-time and inference-time templates must match exactly** — including the trailing `<|im_start|>assistant\n` generation prompt. A mismatched template is the #1 cause of "my fine-tune is worse than the base model."
3. **The EOS token must match the template's turn terminator.** For Qwen bases: `SFTConfig(eos_token="<|im_end|>")` [V — TRL docs warn about this by name].
4. `assistant_only_loss=True` needs `{% generation %}`/`{% endgeneration %}` in the template. TRL auto-patches known families (Qwen3); **for others, check** [V].

TRL 1.8 accepts four dataset shapes [V — verbatim from the docs]:
```python
{"text": "The sky is blue."}                                      # standard LM
{"messages": [{"role": "user", "content": "..."},                 # conversational LM
              {"role": "assistant", "content": "..."}]}
{"prompt": "The sky is", "completion": " blue."}                  # standard prompt-completion
{"prompt": [{"role": "user", "content": "..."}],                  # conversational prompt-completion
 "completion": [{"role": "assistant", "content": "..."}]}
```
**Recommend the course standardize on conversational prompt-completion** — it gives you `completion_only_loss` for free and composes with `assistant_only_loss`.

### ★ How many examples do you actually need?

**The honest answer, in a table** [U for the row boundaries — these are field folklore, calibrated against LIMA and the TM results; the architect should present them as rules of thumb, not measurements]:

| Goal | Examples | Evidence |
|---|---|---|
| Rigid output format (JSON schema, a fixed template) | **50–200** | format is the cheapest thing to teach |
| Tone / house style / persona | **200–1,000** | LIMA territory |
| Domain idiom, task-specific behaviour | **1,000–10,000** | |
| New capability, complex multi-step behaviour | **10,000–100,000** | |
| Change what the model *knows* | **∞ — go do RAG** | §6 |

**The LIMA result [V-2°], stated precisely because it's usually stated wrong:** LIMA fine-tuned a 65B on **1,000** carefully curated examples (750 from Stack Exchange/wikiHow, 250 hand-written) and matched or beat GPT-4 in **43%** of head-to-head comparisons (58% vs Bard, 65% vs DaVinci-003). **The much more important half, which everyone drops:** when the authors scaled to **2,000 examples without maintaining quality, the model got *worse*.** [V-2°]

> **The lesson is not "1,000 examples is enough." It's "adding mediocre examples actively harms you." Data quality isn't a nice-to-have; low-quality data has *negative* value.**

**⚠️ Caveat the course must state:** LIMA was 2023, on a base model, evaluated by human preference on open-ended chat. **Do not over-generalize it.** It is evidence about *alignment/style*, which is exactly the FORM half of §6 — and that is precisely why it's relevant here.

**Reconcile with Thinking Machines [V-2°]:** rank-32 on a 7B matched full FT **up to ~50,000 examples**. So the two results are about different regimes: LIMA says *quality dominates at small N*; TM says *capacity dominates at large N*. Present them together, not as rivals.

### Building a dataset from a personal corpus — the actual recipe

This is what the learner will really do. Give it as a procedure:

1. **Extract.** PDFs/emails/docs → text. Expect this to be 50% of your effort and to be miserable. It is not a modelling problem.
2. **Decide what you're teaching.** *Write the sentence "after this fine-tune, the model will ___" before you write any code.* If that sentence contains the word "know," **stop — go to §7.**
3. **Generate instruction-response pairs.** Realistically: use a strong model to generate questions *from* each chunk, then answers. This is **synthetic data** and it's the standard 2026 practice.
4. **Filter.** This is where the value is. Dedup (exact + near-dup via MinHash). Length outliers. Refusals. Truncations. Anything where the generator model hedged.
5. **Human-review a sample.** 100 examples, by hand, by you. **You will find something horrifying.** Budget for this; it is the highest-ROI hour in the project.
6. **Hold out a real test set — BEFORE you look at anything.** Split by *document*, not by example, or near-duplicate chunks leak across the split and your eval is a lie.
7. **Format** into the chat template.

### Synthetic data — the honest treatment
- It works. It is standard. Most 2026 instruct datasets are substantially synthetic.
- **Model collapse** is real but frequently overstated. The failure mode is *training on your own outputs recursively without fresh data or filtering*. Generating with a **stronger** model and **filtering** is distillation, not collapse, and it works fine.
- **⚠️ Licensing: check the generating model's terms.** Many commercial providers' ToS restrict using outputs to train competing models. **This is a real legal constraint and the course should not hand-wave it.** Open-weight generators (Qwen3 is Apache-2.0 [V]) sidestep it entirely — **which is a good practical reason to recommend an Apache-2.0 generator to the learner.**
- **Diversity is the failure mode, not quality.** An LLM asked for 1,000 questions gives you ~50 questions in 1,000 costumes. **Fix: seed each generation with a different real chunk + a sampled "question type" from an explicit taxonomy + temperature ≥ 1.0.** Then measure: embed all your prompts and look at the pairwise similarity distribution. If the mode is above ~0.8, you have 50 questions.

### Contamination
- **Eval contamination:** your test set leaked into pretraining. You cannot check this for a closed corpus. **Assume public benchmarks are contaminated** and weight your own held-out set accordingly (§12).
- **Train/test leakage in your own split:** the one you *can* control and the one you'll actually get wrong. **Split by document/source, and dedup across the split boundary.**

### 🎛️ Demo A13: The chat template + loss mask visualizer
**Plot:** three synchronized panes. (1) The message list as editable JSON. (2) The rendered template string with **special tokens highlighted in a distinct colour** and non-printing tokens made visible. (3) The **token-by-token loss mask**: every token as a chip, green = contributes to loss, gray = `-100`.
**Controls:** a template dropdown (**ChatML / Llama-3 / Gemma / Mistral** — ship the real Jinja templates), and radio buttons for the masking mode (**full-sequence / completion-only / assistant-only**).
**Exact math:** real Jinja rendering in JS (`nunjucks` is a drop-in), real BPE tokenization (reuse Demo A1's encoder), real mask construction from `{% generation %}` markers. **Live readout: "N of M tokens contribute to the loss (X%)."**
**The insight:** the learner flips masking mode from assistant-only to full-sequence on a long-prompt/short-answer example and watches the percentage jump from **11% to 100%** — and immediately understands that in full-sequence mode, **89% of their gradient is teaching the model to write user questions.** That's the §5 misconception, closed with a number they generated. Then they switch template families and see the special tokens change completely, which makes "never hand-roll the template" self-evident.

---

## 12. Evaluation

### Intuition first
> **Loss going down means the model is better at predicting your dataset. Your dataset is not your goal. These are different sentences, and the gap between them is where fine-tuning projects die.**

### Why loss is not enough
1. **Loss rewards mimicry of surface form.** A model that learned your data's *formatting quirks* has a great loss and no new capability.
2. **Loss is not comparable across tokenizers.** Different $V$, different $\rho$, different nats/token. **Cross-model perplexity comparisons are meaningless unless the tokenizer is identical.** Worth a box — people do this constantly.
3. **Low loss on a contaminated eval is memorization.**
4. **Loss says nothing about what you broke.** See catastrophic forgetting.

### The eval stack, cheapest first
1. **Held-out loss** — free, and *only* answers "am I overfitting?" Nothing else. Use it for early stopping, not for judgment.
2. **`mean_token_accuracy`** — TRL logs this by default [V]. Slightly more interpretable than loss; same limitations.
3. **Task metrics** — if you have a programmatic checker (does the JSON parse? does the SQL run? does the extracted field match?), **this is your best metric by a wide margin.** Free, fast, uncorrupted, no judge bias. **If the learner's task admits a checker, everything below this line is optional.** Push hard on this.
4. **Your own held-out set, eyeballed** — 50 examples, read them. Irreplaceable. Do it every run.
5. **LLM-as-judge** — scalable, biased (below).
6. **Public benchmarks** — mostly for comparing against the field, mostly contaminated.
7. **Human preference** — the gold standard, expensive.

### ★ LLM-as-judge and its biases [V-2°]

The five named biases and their real mitigations:

| Bias | What happens | Mitigation | Does it work? |
|---|---|---|---|
| **Position** | Judges systematically prefer the **first** option in a pairwise comparison | **Evaluate both orderings and average.** Costs 2×. | **Yes — this one is genuinely fixed.** Cheap, reliable, do it always. |
| **Verbosity** | Judges rate longer answers higher **regardless of quality** | Length-Controlled AlpacaEval regresses length out | **Partially.** "Harder to mitigate than position bias because it is embedded in the model's learned representations, not just the prompt framing" [V-2°]. Report answer lengths alongside scores, always. |
| **Self-preference** | Judges prefer **their own** outputs | Use a judge from a **different family** than both the model you tuned and the model that generated your synthetic data | Partially |
| **Format** | Markdown, headers, bullets score higher | Normalize formatting before judging | Partially |
| **Calibration drift** | The judge's scale wanders across runs/versions | **Anchor to a human-labelled golden set**; re-score it every time | Yes, and it's the one people skip |

**The devastating finding worth a callout box** [V-2°]: *"LLM judges' ratings of a model's behavior converge with that model's self-report in a way that human ratings of the same samples do not."* — i.e. **the judge is agreeing with the model rather than evaluating it.** That is a *correlated* error, not noise, and averaging more judges does not fix it.

> **⚠️ The trap: if you generated your synthetic training data with model X and you judge with model X, you have built a closed loop that measures how much your model sounds like X. It will show improvement. It means nothing.** This is a very easy mistake to make and the course must name it explicitly.

**Minimum honest judge protocol:** (1) both orderings, averaged; (2) judge from a different family than the data generator; (3) report mean answer length next to every score; (4) a 30–50 example human-labelled golden set, re-scored every run, reported as judge-vs-human agreement. If agreement drifts, **your metric moved, not your model.**

### ★ Catastrophic forgetting

> **Intuition: the weights that encode "how to write Python" and the weights that encode "how to write your company's incident reports" are the same weights. Moving them toward one is moving them away from the other.**

**How to detect it — and the learner *will* skip this:** keep a **general** eval set (10–20 prompts about things completely unrelated to your fine-tune: a coding question, a translation, a math problem, a general-knowledge question). **Run it before and after. Diff by eye.** Total cost: 15 minutes. It catches the most common way a fine-tune destroys value while the loss curve looks beautiful.

**Mitigations, in order of practicality:**
1. **Fewer epochs, lower LR.** Boring; usually sufficient; try first.
2. **LoRA over full FT.** Constrains the update to a low-rank subspace → structurally less damage. **But see the intruder-dimension result below — this is less protective than folklore claims.**
3. **Mix in general instruction data.** ~5–20% of a general set (e.g. tulu-3 [V — used in the TM reproduction]) alongside your domain data. The most reliable fix. Costs training time.
4. **`init_lora_weights="corda"` in Knowledge-Preserved Mode** [V — real in PEFT 0.19.1]. Purpose-built for this. Under-known.
5. **`reduce_intruder_dimension()` post-hoc** [V]. Tunable via `migration_lambda`.
6. **Merge with a scaling factor < 1** at merge time — blend adapter and base.

**⚠️ Genuine tension the course must not paper over:** "LoRA Learns Less and Forgets Less" (arXiv 2405.09673) says LoRA forgets less. "LoRA vs Full Fine-tuning: An Illusion of Equivalence" (arXiv 2410.21228) says LoRA introduces **intruder dimensions** that are the *mechanism* of forgetting — and PEFT ships a mitigation for exactly that [V]. These are not obviously compatible. **The honest synthesis: LoRA forgets less *in aggregate* but forgets *differently and more structurally* — and the structure is fixable post-hoc, at a cost in task accuracy. Present the tension. Do not resolve it; it isn't resolved.**

### Benchmarks and their limits
- MMLU, GSM8K, HumanEval, HellaSwag are **contaminated**. Assume it.
- Arena-Hard / AlpacaEval 2 are judge-based, so they inherit every bias above.
- **A benchmark measures the benchmark.** For the learner's purpose — a fine-tune for a specific application — **public benchmarks are almost entirely irrelevant except as a forgetting check.** Say this plainly. Their eval is: *does it do my task on my held-out data, and did it break anything I cared about?*

### 🎛️ Demo A14: The judge-bias sandbox
**Plot:** a pairwise-comparison UI. Two canned answers, A and B. A "judge verdict" panel.
**Controls:** a **swap A/B** button; a **verbosity slider** that swaps in pre-generated versions of the *same answer* at 1×/2×/4× length (same content, padded with genuine but redundant elaboration); a **markdown formatting toggle**.
**Implementation:** ship **pre-recorded real judge verdicts** for every cell of the (order × length × format) grid — you must actually run these against a real judge model offline. This is a few hundred API calls and it is worth it. **Do not simulate the bias; measure it and replay it.** A simulated bias demo teaches nothing and is arguably dishonest.
**The insight:** the learner clicks "swap" and **the verdict flips** — with identical content. Then they drag verbosity and watch the *same answer, padded*, win. Their trust in judge scores dies in about 15 seconds, permanently, and correctly. Then the "average both orderings" toggle appears and they watch position bias vanish while **verbosity bias survives** — which is the real, differentiated lesson: some biases are cheap to fix and some aren't.

---

## 13. The training loop — real, current APIs

### ⚠️ Version reality check — this is time-critical and it bites

| Library | Version | Date | Confidence |
|---|---|---|---|
| **transformers** | **5.14.1** | 2026 | [V — PyPI] |
| **trl** | **1.8.0** | **2026-07-09** | [V — PyPI, 7 days before this brief] |
| **peft** | **0.19.1** | **2026-04-16** | [V — PyPI] |
| accelerate | ≥1.4.0 (trl's floor) | | [V] |
| datasets | ≥4.7.0 (trl's floor) | | [V] |
| bitsandbytes | 0.49.2+, **aarch64 wheels exist** | | [V-2°] |

> ## ★★ **transformers is at v5. This is a MAJOR break and it invalidates essentially every fine-tuning tutorial written before ~late 2025.**

**v5 breaking changes that hit this course directly** [V-2°]:
- **PyTorch is the only backend.** TF/Flax gone.
- **`dtype` now defaults to `"auto"`** — models load in whatever precision they were saved in. **Code that assumed fp32 now silently runs in bf16.** Note the sharp edge: TRL's `SFTTrainer` **still defaults to fp32** when `model` is a string and `model_init_kwargs` doesn't specify `dtype` — *"This differs from `from_pretrained`, where (since Transformers v5) the dtype is inferred from the model config"* [V — verbatim from TRL docs]. **This inconsistency between two libraries the learner uses in the same script is a real trap. Call it out by name.**
- **Safetensors only.** `safe_serialization=False` raises.
- **`transformers-cli` → `transformers`** (Typer-based). `transformers chat` no longer starts a server; that's `transformers serve`.
- Python 3.10+, PyTorch 2.4+, `huggingface_hub` 1.3.0–<2.0.
- **`encode_plus()` deprecated** → call the tokenizer directly.

**Recommendation to the architect: the course MUST pin exact versions in every artifact and say so loudly.** A `requirements.txt` with `transformers==5.14.1, trl==1.8.0, peft==0.19.1` and a note that these move fast. Consider `uv` — TRL's own docs now use `uv run <url>` to run scripts straight from a URL [V], which is a genuinely nice pattern for a course.

### The minimal working script (TRL 1.8 API [V])

```python
from datasets import load_dataset
from trl import SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```
**That is the entire quickstart in TRL 1.8** [V — verbatim from the docs]. Three lines. **The course should show this first**, precisely so the learner understands that the other 60 lines below are *choices*, not ceremony.

### The realistic QLoRA script (TRL 1.8 + PEFT 0.19)

**Note the important 2026 API change: `quantization_config` is now a direct `SFTTrainer` argument** [V — "Quantization configuration used when loading the model from a model identifier. Combine with `peft_config` for QLoRA training. Ignored if the model is already instantiated."]. **You no longer need to build the model yourself for QLoRA.** Most tutorials still do the old `AutoModelForCausalLM.from_pretrained(..., quantization_config=...)` + `prepare_model_for_kbit_training()` + `get_peft_model()` dance. **That still works but is no longer necessary, and the course should show the current path.**

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("json", data_files="my_data.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=42)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,            # see §15 — this convention is CONTESTED
    lora_dropout=0.0,
    target_modules="all-linear",   # ← §9: not q_proj/v_proj. All of them.
    use_rslora=True,               # matters at r>=64
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="./qwen3-8b-mydomain",
    model_init_kwargs={"dtype": torch.bfloat16},   # ← §13 trap: TRL defaults to fp32 here
    max_length=2048,
    packing=True,
    packing_strategy="bfd",        # implies padding_free
    assistant_only_loss=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,          # effective batch = 16  (§15: keep < 32)
    learning_rate=2e-4,                     # §11/§15: ~10x full-FT
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    bf16=True,
    report_to=["trackio"],
    seed=42,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-8B",
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    quantization_config=quant_config,    # ← 2026 API: QLoRA without hand-building the model
)
trainer.train()
trainer.save_model()
```

**Defaults in `SFTConfig` that differ from `TrainingArguments` and will surprise the learner** [V — TRL states these explicitly]:
- `logging_steps` = **10** (not 500)
- `gradient_checkpointing` = **True** (not False) ← **already on. Do not "discover" it as an optimization.**
- `bf16` = **True** if `fp16` unset (not False)
- `learning_rate` = **2e-5** (not 5e-5) ← this is the **full-FT** default; **for LoRA you must raise it ~10×**
- `max_length` = **1024** ← **silently truncates.** A very common cause of "why did my long examples not work."
- `loss_type` = `"chunked_nll"` ← the §3 memory optimization, on by default

**TRL 1.8 features worth teaching** [V]:
- `packing=True` + `packing_strategy="bfd"` — best-fit-decreasing bin packing; **implies `padding_free`**. Big throughput win when example lengths vary.
- `padding_free=True` — flattens the batch into one continuous sequence; **needs FlashAttention 2 or 3**.
- `activation_offloading=True` — activations to CPU. **On the Spark's unified memory this is semantically odd** (§14) — flag it.
- `loss_type="dft"` — Dynamic Fine-Tuning (arXiv 2508.05629), an RL-perspective reward-rectified SFT loss. **New, interesting, unproven. Mention; don't build on it.**
- `use_liger_kernel=True` — Triton kernels; claimed ~20% throughput / ~60% memory [V-2° — vendor claim]. **Note: forces `loss_type="nll"`; incompatible with `chunked_nll`.**
- `report_to=["trackio"]` — HF's current local-first tracker. [V]

### 🎛️ Demo A15: The config-to-consequences translator
**Plot:** a form mirroring the real `SFTConfig`/`LoraConfig` above; beside it, a live-updating panel: **trainable params (exact, from §9's shape math), total memory (from §8's ledger), estimated wall-clock on a Spark (from §14's measured tok/s), effective batch size, and total optimizer steps.**
**Exact math:** all of it is arithmetic the course has already taught — this demo is the **integration exercise**, and that's its purpose. `effective_batch = per_device × grad_accum`. `steps = ceil(n_examples × epochs / effective_batch)`. `wall_clock = n_examples × epochs × avg_tokens / measured_tok_per_s`.
**The insight:** the learner changes `r` 16→256 and watches memory move by 10 GB but wall-clock barely move — **rank is cheap.** They change `max_length` 1024→4096 and watch memory move a lot — **sequence length is expensive.** They set `per_device_train_batch_size=32` and get a **red warning: "effective batch 32 ≥ 32; LoRA is less tolerant of large batches (§15)."** The config file stops being a magic incantation and becomes a set of levers with visible, quantified consequences. **This is the demo that converts knowledge into competence.**

---

## 14. ★ Hardware reality — the DGX Spark

### Verified specs [V — NVIDIA's own `docs.nvidia.com/dgx/dgx-spark/hardware.html`]

| Spec | Value |
|---|---|
| **Superchip** | NVIDIA **GB10** Grace Blackwell |
| **CPU** | **20-core Arm** — 10× Cortex-X925 + 10× Cortex-A725 |
| **GPU** | Blackwell, **6,144 CUDA cores**, 5th-gen Tensor Cores, 4th-gen RT cores |
| **Memory** | **128 GB LPDDR5x unified**, 256-bit bus, 4266 MHz |
| **Memory bandwidth** | **273 GB/s** ← **the number that governs everything** |
| **Interconnect** | NVLink-C2C between CPU and GPU |
| **Storage** | 1 TB or 4 TB self-encrypting NVMe M.2 |
| **Networking** | 10 GbE RJ-45, **ConnectX-7 SmartNIC**, Wi-Fi 7, BT 5.4 |
| **Power** | 240 W PSU; **GB10 SoC = 140 W TDP** |
| **Headline perf** | **"up to 1,000 TOPS / 1 PFLOP at FP4 **with sparsity**"** |
| Price | ~$3,000–4,700 depending on SKU/retailer [V-2°] |

> ### ⚠️ The "1 PFLOP" number is marketing and the course must say so, kindly but clearly.
> **1 PFLOP is FP4, *with structured sparsity* (a 2× multiplier you get only if your weights are 2:4 sparse, which yours are not).** Your realistic **dense bf16 training** throughput is **far** lower. **Do not size training runs off the headline number.** Use the measured tok/s below.
>
> ⚠️ **[U] I could not verify a published dense BF16 TFLOPS figure for GB10 this session** — NVIDIA's hardware page gives FP4-with-sparsity and nothing else [V, confirmed absent]. **The architect must either find this number or, better, have the learner measure it** (artifact #9). Measuring it is the better pedagogy anyway.

### ★ Measured numbers — use these, not the marketing

**NVIDIA's own published fine-tuning benchmarks, seq len 2048, PyTorch** [V — `developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/`]:

| Task | Throughput |
|---|---|
| **Llama 3.2 3B — FULL fine-tune** | **13,519.54 tok/s** |
| **Llama 3.1 8B — LoRA** | **6,969.59 tok/s** |
| **Llama 3.3 70B — QLoRA** | **759.79 tok/s** |

**These three numbers are the most valuable thing in §14. Make them recur.** They let the learner answer "how long will this take?" for any dataset with one division, and the course should have them do exactly that, repeatedly.

**Measured inference** [V-2° — LMSYS, SGLang, FP8]:

| Model | Prefill | Decode (batch 1) | Decode (batch 32) |
|---|---|---|---|
| Llama 3.1 **8B** | **7,991 tok/s** | **20.5 tok/s** | **368 tok/s** |
| Llama 3.1 **70B** | 803 tok/s | **2.7 tok/s** | — |
| DeepSeek-R1 14B | 2,074 tok/s (b8) | 83.5 tok/s (b8) | — |
| GPT-OSS 20B (Ollama) | 2,053 tok/s | 49.7 tok/s | — |

**NVIDIA's inference numbers** [V]: Qwen3 14B NVFP4 = **22.71 tok/s**; Llama 3.1 8B NVFP4 = **38.65 tok/s**; Qwen3 235B on **two** Sparks = 11.73 tok/s.

**For contrast** [V-2° — LMSYS]: RTX Pro 6000 Blackwell on GPT-OSS 20B = 10,108 / **215 tok/s** — **~4× faster decode.** RTX 5090 = 8,519 / 205 tok/s. LMSYS attributes the gap directly to the **273 GB/s** unified LPDDR5x.

### ★★ The roofline — the analysis that makes all of the above make sense

$$
\text{tok/s}_{\max} = \frac{\text{BW}}{\text{model bytes}}, \qquad \text{MBU} = \frac{\text{measured}}{\text{tok/s}_{\max}}
$$

**Check it against the measurements:**

> **⚠️ CORRECTED byte column (Task 1 / D-10e / constants §6.6–6.7). NVFP4 checkpoints are really 6.03 / 10.54 GB
> (read from safetensors headers), NOT the 4.5 / 7.5 GB assumed here. The physically-correct byte count is
> decode-traffic = checkpoint − embedding table.** Under it, all four land **60–75% MBU** (the heuristic survives).

| Model | Decode-traffic bytes | Ceiling = 273/bytes | Measured | **MBU** |
|---|---|---|---|---|
| Llama 3.1 8B FP8 | ~8 GB | 34.1 tok/s | 20.5 | **60.3%** |
| Llama 3.1 70B FP8 | ~70 GB | 3.9 tok/s | 2.7 | **69.8%** |
| Llama 3.1 8B NVFP4 | 6.03 − 1.05 = ~4.98 GB | 54.8 tok/s | 38.65 | **70.5%** |
| Qwen3 14B NVFP4 | 10.54 − 1.56 = ~8.98 GB | 30.4 tok/s | 22.71 | **74.7%** |

> ### **Four independent dense batch-1 measurements — three model sizes, two quantization formats, two inference stacks — all land at 60–75% MBU under a consistent decode-traffic byte count.**
>
> **This is the single most valuable analytical result in the brief, and it SURVIVED the audit** (an earlier
> "REFUTED — only two land" verdict was a strawman that swapped in a batch-8 point and an MoE). The learner can
> predict decode speed with **one division and a ×0.65 fudge**, right to within ~13% worst-case. **The deeper
> lesson: the dominant uncertainty is the byte count, not the coefficient** — for a quantized model, weight-bytes-
> per-token is the file *minus the embedding table*, not the file size and not params×bits. The two failure modes
> (an MoE implying >100% of peak → "your model is an MoE"; a batch-8 point) are the punchline, not refutations.

**Recommend the course carry this as a named rule:**
$$
\boxed{\text{decode tok/s} \approx 0.65 \times \frac{\text{memory bandwidth (GB/s)}}{\text{model size (GB)}}}
$$

### ★ Unified memory — what actually changes, honestly

**The win — and it's a real one:**
- **128 GB is 128 GB.** No 24 GB VRAM cliff. **A 70B QLoRA fine-tune (~41 GB, §9) simply fits.** On any consumer discrete GPU it simply does not, at any price under ~$30k.
- **No PCIe transfer.** CPU and GPU share one physical pool over NVLink-C2C. No `.to("cuda")` copy cost, no host↔device shuffling.
- **`device_map="auto"` / CPU offload / paged optimizers become semantically weird** — there's nowhere to offload *to*. Tutorials that lean on offload are solving a problem you don't have. **Flag this explicitly; it will confuse the learner otherwise.**

**The cost — and it is severe:**
- **273 GB/s.** An RTX 5090 is ~1792 GB/s [U]; an H100 SXM ~3350 GB/s [U]. **The Spark is ~6.6× and ~12× slower in bandwidth respectively.**
- **Decode is bandwidth-bound (§4). So decode on the Spark is ~4–12× slower than a discrete GPU** — exactly what LMSYS measured [V-2°].
- **Prefill and training are compute-bound**, so the Spark closes much of the gap there. **This is why it's a fine *fine-tuning* box and a mediocre *serving* box, and that asymmetry is precisely predicted by §4's prefill/decode distinction.** The learner should be able to derive this conclusion themselves before being told it.
- **The honest LMSYS verdict** [V-2°]: large models on the Spark are for *"prototyping and experimentation rather than production."*

### ⚠️ aarch64 — the practical tax, and it's real
> **Everything must be ARM64. There is no x86 fallback layer. Every wheel, every container, every dependency has to exist for aarch64.** [V-2°]

- **bitsandbytes: aarch64 wheels exist** (e.g. `bitsandbytes-0.49.2-py3-none-manylinux_2_24_aarch64.whl`) — **QLoRA on the Spark works** [V-2°], with community reports of successful QLoRA runs on GB10 [V-2°].
- **vLLM: rocky.** Issue #31128 ("Add support of Blackwell SM121 (DGX Spark)") is **closed** with an associated PR **also closed** [V]. The documented workaround was `pip install vllm==0.13.0 --no-deps` + manual deps + **`--enforce-eager`, which costs ~20–30% inference speed** by disabling CUDA graphs [V]. The asks were PyTorch 2.10.x support, CUDA-13 ARM64 wheels, and native sm_120 without eager. **⚠️ [U] I could NOT verify the current state as of 2026-07. The architect must re-check this before shipping — it's exactly the kind of thing that changes monthly, and it determines whether §13's serving recommendation is vLLM or llama.cpp.**
- **llama.cpp/Ollama: works well** — Arm publishes an official DGX Spark + llama.cpp learning path [V-2°]. **Recommend llama.cpp/Ollama as the course's default local serving path for the Spark**, on grounds of it actually working, with vLLM as an "if you need throughput and it's working this month" note.
- **NVFP4 is native to GB10's Blackwell tensor cores** [V-2°]. **This is the Spark's genuinely special capability and the course should lean on it** — it's the one place the hardware beats older discrete GPUs outright.

### ★ The verdict table — what the learner can actually do

| Task | On the Spark? | The arithmetic |
|---|---|---|
| **Pretrain anything meaningful** | ❌ **Never** | §3: 228 years for an 8B |
| Full FT ≤ 3B | ✅ Comfortable | 3B × 16 B = 48 GB; NVIDIA measured 13,519 tok/s |
| **Full FT 8B, plain AdamW** | ❌ **131 GB > 128 GB** | §8 — *the* teaching moment |
| Full FT 8B, `adamw_8bit` | ✅ Tight | 82 GB + activations |
| Full FT 13B+ | ❌ | 13B × 10 B/param = 130 GB even with 8-bit Adam |
| **LoRA 8B** | ✅✅ **Easy — the sweet spot** | ~17 GB; 6,970 tok/s |
| LoRA 32B [U] | ✅ | ~66 GB bf16 base + adapters |
| **QLoRA 70B** | ✅✅ **The headline capability** | ~41 GB; **759.79 tok/s** |
| QLoRA 120B+ | ⚠️ Marginal | 120B × 0.516 = 62 GB base; adapters + activations tight |
| Inference ≤ 14B | ✅ Good | 20–40 tok/s at NVFP4 |
| Inference 70B | ⚠️ **2.7 tok/s** — technically works | bandwidth-bound; painful |
| **Serving to users** | ❌ **Wrong tool** | 4× slower decode than a single 5090 |
| Multi-model dev, big-context experiments | ✅✅ | 128 GB is the point |

### ★★ Worked example — the learner's actual project, end to end

*Fine-tune Qwen3-8B on 2,000 examples from his own corpus, avg 800 tokens each.*

- Tokens/epoch: $2{,}000 \times 800 = 1.6\times10^6$
- 3 epochs: $4.8\times10^6$ tokens
- **LoRA on 8B at 6,969.59 tok/s** [V]: $4.8\times10^6 / 6{,}970 = 689\ \text{s} = \mathbf{11.5\ minutes}$
- Memory: ~17 GB of 128 GB. **He could run seven of these simultaneously.**

*Same data, Llama-3.3-70B QLoRA:*
- $4.8\times10^6 / 759.79 = 6{,}318\ \text{s} = \mathbf{1.75\ hours}$
- Memory: ~41 GB of 128 GB.

> **These two numbers reframe the entire track. Fine-tuning an 8B on your own data is a coffee break. Fine-tuning a 70B is an afternoon. The bottleneck was never the compute — it is, and always was, building the dataset (§11) and knowing whether you should be doing this at all (§6).**
>
> **I would put this box in the course twice: once here, and once at the very front as the track's thesis statement.**

---

## 15. ★ Genuine disagreements and uncertainty — flag these, don't paper over them

The brief asked for this explicitly. These are real, live, and the course loses credibility if it pretends otherwise.

### 15.1 ★ `lora_alpha`: 2r or 16? — **the sharpest, most actionable conflict in this brief**
- **Folklore (2021–2024, still in most tutorials):** $\alpha = 2r$. At $r=16$, $\alpha=32$, **scale = 2.0**.
- **Thinking Machines / current TRL docs [V]:** `LoraConfig(r=256, lora_alpha=16)` — **scale = 16/256 = 0.0625.**
- **These differ by 32×.** They are not reconcilable as "a matter of taste."
- **My read:** TM holds $\alpha$ *fixed* while varying $r$, arguing the $1/r$ scaling makes the **optimal LR approximately rank-independent** [V-2° — TRL states this]. The folklore holds the *scale* fixed instead. **Both can work because $\alpha$ trades off against LR (§9).** But a learner who copies TM's $\alpha=16$ **and** a tutorial's $r=8$ gets scale = 2.0 — accidentally landing on the folklore. And one who copies $\alpha=2r$ **and** TM's $r=256$ gets $\alpha=512$, scale 2.0, with a TM-tuned LR — **and will diverge.**
- **⚠️ This is a live footgun.** **Recommendation: the course picks ONE convention, states it's a convention, shows the other, and shows the $\alpha$–LR redundancy in Demo A11 so the learner can debug the collision when they hit it in the wild.** Do not present either as "the answer."

### 15.2 Does LoRA match full fine-tuning?
- **"LoRA Learns Less and Forgets Less" (2405.09673):** a real trade-off exists.
- **"LoRA Without Regret" (Schulman/TM, 2025, DOI 10.64434/tml.20250929) [V-2°]:** LoRA matches full FT at ~67% of the compute, **if** all-linear + sufficient rank + LR ~10× + effective batch < 32.
- **"LoRA vs Full Fine-tuning: An Illusion of Equivalence" (2410.21228) [V — PEFT implements its mitigation]:** they're *not* equivalent; LoRA creates intruder dimensions.
- **Honest synthesis: they match on *task performance* in the low-regret regime, and differ in *solution structure*, and the structural difference shows up as forgetting.** That's three papers making three compatible-but-different claims and the course should say so rather than pick a winner.

### 15.3 The batch-size finding is odd and under-explained
TM: **LoRA is *less* tolerant of large batch sizes than full FT, and raising the rank does not fix it** [V-2°]. This is genuinely counterintuitive — batch size effects usually interact with capacity. **Nobody has a clean mechanistic story.** Report it as an empirical finding with an honest "we don't fully know why," and keep effective batch < 32. **⚠️ Note it conflicts with general LLM training practice, where large batches are standard and good.**

### 15.4 Long context vs RAG
Genuinely unsettled. Claimed 10M-token windows [V-2° — Llama 4 Scout; **the claim comes from marketing-adjacent sources and I could not verify effective (as opposed to nominal) performance at that length**]. **The nominal/effective gap is the crux and it is not honestly reported by anyone.** Present as: *complementary today, direction uncertain, and be suspicious of any context-length number not accompanied by a retrieval-accuracy-at-length curve.*

### 15.5 Which alignment algorithm
No clean ranking (§5). Most reported margins are **judge-based** and therefore inherit §12's biases. **SimPO's +6.4 AlpacaEval-2 over DPO [V-2°] is a length-sensitive metric and SimPO's whole mechanism is length normalization — that is not obviously a fair fight, and it deserves an explicit caveat in the course rather than a citation.**

### 15.6 Chinchilla
The 20:1 ratio [U] answers a question almost nobody has (§3). It is **still** widely quoted as though it were a law of nature. Present it, then dismantle it.

### 15.7 Quantization quality numbers
**The "AWQ 95% / GGUF 92% / GPTQ 90%" figures are from low-quality secondary sources and are mutually inconsistent.** [V-2°, low confidence] **Do not print them.** The *ordering* is probably right; the numbers are not defensible. **Better: make measuring them an exercise (artifact #8).**

### 15.8 Things I could not verify this session — **the architect must check these**
| Claim | Status |
|---|---|
| **GB10 dense BF16 TFLOPS** | **[U] — NVIDIA publishes only FP4-with-sparsity. Confirmed absent from the hardware page. Best fix: measure it.** |
| **vLLM SM121 support status as of 2026-07** | **[U] — issue closed, PR closed, current state unknown. Re-check; it gates the §13 serving recommendation.** |
| Current SOTA embedding models / MTEB status | **[U] — could not verify. Must fill in for §7.** |
| Llama 3.1/3.3 exact configs | **[U] — HF repo is gated (401). Recommend centering on Qwen3 (Apache-2.0, ungated, fully verified).** |
| Qwen3.5 / Qwen3.6 official lineup | **[U] — Qwen3.5-9B and Qwen3.6-27B/35B-A3B appear on HF, but I could not cleanly separate official Qwen org releases from community re-uploads. Re-check.** |
| QLoRA effective bits (4.127 vs my 4.25 derivation) | **[U] — my arithmetic disagrees with the cited figure. Resolve before printing.** |
| Shannon's bits/char for English | **[U]** |
| Llama 3 15T token count; 128,256 vocab | **[U]** |
| RTX 5090 / H100 / M3 Ultra bandwidths | **[U] — used only for Demo A6 presets.** |
| DPO `beta` default in TRL 1.8 | **[U]** |
| bnb double-quant saving (0.37 bits) | **[U]** |
| QLoRA speed penalty (20–40%) | **[U]** |

---

## 16. ★ For the curriculum architect — reconciliation with other briefs

**Shared with the diffusion track — must be taught ONCE, in the trunk or a shared "Adaptation" chapter:**
1. **LoRA itself.** ComfyUI LoRAs and LLM LoRAs are **the same mathematics** — $\Delta W = \frac{\alpha}{r}BA$, same $r$, same $\alpha$, same merge. **The learner already uses diffusion LoRAs. This is a colossal pedagogical asset: he has hands-on intuition for the artifact before he has the math.** ★★ **Strong recommendation: teach LoRA in a shared chapter and open it with "you already use these — here's what's inside the file."** Then each track covers only what's different: **which modules to target** (LLM: all-linear across 36 transformer blocks; diffusion: attention blocks in the UNet/DiT + optionally the text encoder) and **typical ranks** (LLM SFT: 64–256; diffusion: 4–128 [U — diffusion brief should verify]).
2. **Quantization.** NF4/FP8/NVFP4 are format-identical across tracks. **NVFP4 on Blackwell serves both** (NVIDIA publishes Flux.1 12B FP4 at 2.6 s/1K-image on the Spark [V]). Teach the number formats once.
3. **The memory ledger.** 16 bytes/param under AdamW is architecture-agnostic. **Teach once; instantiate twice.**
4. **The Spark's specs and the 273 GB/s roofline.** One hardware chapter. **⚠️ Note the tracks will emphasize opposite halves: LLM decode is bandwidth-bound (the Spark is weak); diffusion sampling is compute-bound (the Spark is relatively strong). That contrast is a *feature* — it's the prefill/decode lesson generalizing — and the architect should engineer it deliberately.**
5. **Cross-entropy / the training loop / AdamW / gradient checkpointing / mixed precision.** Trunk.

**Genuinely LLM-specific — do not let these leak into the trunk:**
Tokenization; next-token prediction & perplexity; base-vs-instruct; the KV cache; decoding strategies; chat templates; RAG; preference alignment; LLM-as-judge.

**⚠️ Conflicts to watch for:**
- **"LoRA rank" advice will differ between tracks** (LLM: $r=256$ is now recommended and nearly free; diffusion: $r=16$–32 is typical and higher ranks overfit fast on small image sets). **These are both right and they will look contradictory.** Reconcile explicitly with the capacity framing: *rank should scale with the information content of your dataset* — which is large for a 50k-example SFT set and small for 20 photos of a person. **That single sentence resolves it and is a genuinely deep point.** Make sure the diffusion brief uses the same framing.
- **"How many examples"** differs by orders of magnitude (LLM: 1,000+; diffusion LoRA: 10–50 images). **Same resolution: information content, not example count.**
- **The word "fine-tuning" means different things** to a ComfyUI user (mostly: LoRA/DreamBooth on a subject) than in the LLM world (SFT + preference alignment). **Disambiguate early, in the trunk.**

**Ordering recommendation across tracks:** teach the **shared trunk → the memory ledger → LoRA/quantization (shared) → then fork.** This gets the learner to LoRA once, using his existing diffusion intuition as the on-ramp, and makes each destination track shorter and more focused.

**Page budget suggestion for this track (of ~45–50 total, LLM track ≈ 18–20 pages):**
| § | Topic | Pages |
|---|---|---|
| 2 | Tokenization | 2 |
| 3 | Pretraining, base vs instruct | 2 |
| 4 | Decoding + KV cache | 2.5 |
| 5 | Post-training | 2 |
| **6** | **★ prompt vs RAG vs FT** | **1.5** |
| **7** | **RAG** | **2.5** |
| **8** | **★ Memory ledger** | **1.5** |
| **9** | **★ LoRA/PEFT** | **3** |
| 10 | Quantization | 1.5 (split, per §1) |
| 11 | Datasets | 2 |
| 12 | Evaluation | 2 |
| 13 | Training loop | 1.5 |
| 14 | **★ Hardware reality** | **2** |
*(§8–§9 and §14 partially shared with diffusion — negotiate the split.)*

---

## 17. ★ Proposed runnable code artifacts

Ordered by dependency. Every one must be **pinned** (`transformers==5.14.1, trl==1.8.0, peft==0.19.1`) and **run on the Spark** (aarch64, unified memory).

| # | Artifact | Demonstrates | Runtime on Spark |
|---|---|---|---|
| **1** | `01_tokenizer_lab.ipynb` | Load the real Qwen3 tokenizer. Tokenize the learner's own text. **Reproduce the 3-token `strawberry`.** Measure $\rho$ on *his* corpus (English vs code vs his domain jargon). **Compute his corpus's total token count → feeds artifact #5's time estimate.** Train a toy BPE from scratch on his corpus and compare merges to Qwen's. | 2 min |
| **2** | `02_kv_cache_and_roofline.py` | **Measure** the KV cache by watching allocated memory grow as context grows; **verify 144 KiB/token empirically against §4's formula.** Then benchmark decode tok/s at several model sizes and **plot measured vs the 273 GB/s roofline — reproducing the 60–69% MBU result on his own box.** ★ **The single best artifact in this list.** | 20 min |
| **3** | `03_base_vs_instruct.py` | Same prompt into `Qwen3-8B-Base` and `Qwen3-8B`. Print token-level probabilities. **The base model will not stop.** Then apply a chat template to the base and watch it partially recover. | 10 min |
| **4** | `04_decoding_lab.py` | Real logits from Qwen3-8B. Implement greedy/temp/top-k/top-p/min-p **by hand from the logits** (not `generate()`), then verify against `generate()`. **Show that temperature never reorders.** Plot entropy per position over a real generation. | 10 min |
| **5** | `05_memory_ledger.py` | **Compute** the ledger for any HF model id from its `config.json` (reproduce §9's Qwen3-8B table programmatically), then **actually allocate it** and confirm with `torch.cuda.max_memory_allocated()`. **Watch the 8B full-FT attempt OOM at 131 GB.** ★ **Make the learner run the OOM. It is the emotional core of the track.** | 5 min |
| **6** | **`06_rag_vs_finetune_showdown.ipynb`** | ★★ **The inoculation.** Take 200 of his own documents. (a) LoRA-SFT an 8B on them; ask a factual question from doc #147 → **watch it hallucinate fluently.** (b) RAG the same corpus → correct answer + citation. (c) Do both. **Same corpus, same model, same question, three outcomes.** ★★ **If the course ships one artifact, this is it. It permanently prevents the field's most expensive mistake.** | 45 min |
| **7** | `07_rag_pipeline.py` | Chunk → embed → FAISS (and a **NumPy brute-force baseline that beats it under 100k chunks — prove it**) → BM25 → RRF → cross-encoder rerank. **Ablate each stage and report retrieval recall@5.** The learner sees reranking's contribution as a number on *his* data. | 30 min |
| **8** | `08_quantization_lab.py` | Quantize a real Qwen3 weight tensor to NF4/INT4/FP8/NVFP4 by hand; **plot RMS error vs block size** (reproduce Demo A12's key insight numerically). Then quantize the full model and **measure perplexity + his task metric at each format** — **producing the honest numbers §15.7 says not to trust from blogs.** | 30 min |
| **9** | `09_spark_capability_probe.py` | Measure **dense BF16 TFLOPS** (the §15.8 gap!), achieved memory bandwidth (STREAM-like), NVFP4 tensor-core throughput, prefill vs decode. **Output a one-page "what my box actually does" card the learner keeps.** ★ Turns §15.8's unknown into an exercise. | 15 min |
| **10** | `10_build_dataset.py` | His corpus → chunks → synthetic Q/A via a **local Apache-2.0 Qwen3** (sidesteps §11's licensing issue) → dedup (MinHash) → **diversity check via prompt-embedding pairwise similarity** → **document-level** train/test split → chat template → JSONL. **With an assert that fails loudly if any test document's text appears in train.** | 1 hr |
| **11** | **`11_finetune_qlora.py`** | ★ **The main event.** §13's script, parameterized, on his data. Trackio logging. Should complete in **~12 min for 8B LoRA** — matching §14's prediction, which the learner computed in artifact #1+#5. **The prediction landing is the payoff of the whole track.** | 15 min |
| **12** | `12_evaluate.py` | Held-out loss + **his task metric** + **the general-capability forgetting probe (before/after diff)** + LLM-as-judge **with both orderings averaged and answer lengths reported**. **Prints a warning if the judge family == the data-generator family (§12's closed loop).** | 20 min |
| **13** | `13_ablate.py` | Sweep $r \in \{8, 32, 128, 256\}$ and attn-only vs all-linear. **6 runs × ~12 min ≈ 1.2 hr — trivially affordable on his box.** ★ **He personally reproduces the "all-linear beats attention-only at matched param count" result on his own data.** This converts §9's biggest misconception from something he was told into something he measured. | 1.5 hr |
| **14** | `14_merge_and_serve.py` | `merge_and_unload()` **into the original bf16 base (not the NF4 one — §10's trap, with an assert)** → save → convert to GGUF → quantize Q4_K_M → `ollama create` → serve → **measure decode tok/s and check it against artifact #2's roofline.** Closes the loop. | 20 min |
| **15** | `15_grpo_rlvr.py` *(stretch)* | If his task has a programmatic checker: GRPO with a real verifiable reward on a small model (Qwen3-0.6B/1.7B). **The 2026 frontier, on his desk.** ⚠️ Needs in-loop generation (`--vllm_mode colocate`) — **gated on §15.8's vLLM-on-SM121 question.** Have a llama.cpp fallback. | 2 hr |

**Meta-recommendation on the artifacts:** the thread that ties #1 → #2 → #5 → #11 → #14 together is **prediction then measurement**. In #1 he counts his tokens; in #5 he computes his memory; from §14's table he predicts his wall-clock; in #11 he runs it and **the prediction is right**. Then in #2/#14 he predicts his serving speed from the roofline and **that's right too.** *That* — a learner who can predict his own hardware's behaviour from first principles before running anything — is the actual destination of this track, and it's a better destination than "can operate the tools." Build for it.

---

## 18. Sources

- [NVIDIA DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) — **[V]** primary; all §14 specs
- [How NVIDIA DGX Spark's Performance Enables Intensive AI Tasks](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/) — **[V]** the three fine-tuning tok/s numbers
- [NVIDIA DGX Spark In-Depth Review — LMSYS](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) — **[V-2°]** prefill/decode measurements, competitive comparison
- [Qwen/Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) — **[V]** every architectural number in §2/§4/§9
- [Qwen/Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) — **[V]** 8.2B/6.95B, GQA 32/8, Apache-2.0
- [TRL SFT Trainer docs (v1.8.0)](https://huggingface.co/docs/trl/sft_trainer) — **[V]** the entire §13 API surface
- [TRL: LoRA Without Regret](https://huggingface.co/docs/trl/en/lora_without_regret) — **[V]** r=256/α=16, all-linear, 10× LR, batch<32
- [Thinking Machines Lab: LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) — **[V-2°]** the source (Schulman et al., DOI 10.64434/tml.20250929)
- [PEFT LoraConfig reference (v0.19.0)](https://huggingface.co/docs/peft/package_reference/lora) — **[V]** DoRA, rsLoRA, PiSSA/EVA/CorDA/LoftQ/LoRA-GA, aLoRA, intruder dimensions, target_parameters
- [PyPI: trl](https://pypi.org/project/trl/) / [peft](https://pypi.org/pypi/peft/json) / [transformers](https://pypi.org/pypi/transformers/json) — **[V]** versions & dates
- [transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) — **[V-2°]** §13 breaking changes
- [Introducing NVFP4 — NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) — **[V-2°]** NVFP4 vs MXFP4
- [vLLM issue #31128 — Blackwell SM121 support](https://github.com/vllm-project/vllm/issues/31128) — **[V]** status, `--enforce-eager` workaround
- [Arm: llama.cpp on DGX Spark](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_llamacpp/1a_gb10_setup/) — **[V-2°]** aarch64 path
- [LoRA Learns Less and Forgets Less (2405.09673)](https://arxiv.org/abs/2405.09673) — **[V-2°]**
- LoRA vs Full Fine-tuning: An Illusion of Equivalence (2410.21228) — **[V]** via PEFT's implementation
- [LIMA: Less Is More for Alignment](https://openreview.net/pdf?id=KBMOKmX2he) — **[V-2°]** 1,000 examples; the 2,000-worse result
- [Self-Preference Bias in LLM-as-a-Judge (2410.21819)](https://arxiv.org/pdf/2410.21819) — **[V-2°]**
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026) — **[V-2°]** §5 landscape
- [Searching for Best Practices in RAG (2407.01219)](https://arxiv.org/pdf/2407.01219) — **[V-2°]**
