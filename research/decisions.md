# decisions.md — RATIFIED RULINGS. Every cross-brief conflict and its resolution.

**Status: RATIFIED 2026-07-16.** Read with `constants.md` and `notation.md`. Where a brief contradicts a ruling
here, **the ruling wins** — the briefs predate two independent verification passes whose verdicts override them.

**Format:** Conflict · Ruling · Reasoning · Brief edits required · Confidence in my own call.

**How to read the confidence column:** I am the arbiter, not an oracle. Where I write **LOW**, the project lead
should look at it personally — those are collected in §Z.

---

## PART A — THE VERIFICATION VERDICTS (not relitigable; propagate them)

### D-01 ⛔ THE ANCHOR MODEL — the corpus's largest silent conflict

**Conflict.** *Neither verifier caught this.* Four briefs anchor on four different models:

| Brief | Anchor | Numbers it plants |
|---|---|---|
| foundations | **Llama-3.1-8B** | $d_{ff}$=14336, $L$=32, $V$=128,256, "~70% FFN", $\ln V$=11.76, "96 GB AdamW" |
| training | **generic 7B / Llama-2-7B** | 112 GB, 130 GB, 56 GB Adam, 16,777,216 LoRA params, 224× |
| architectures, llm-finetuning | **Qwen3-8B** | $d_{ff}$=12288, $L$=36, $V$=151,936, 8,190,735,360 |
| tooling | **Qwen3-4B** for running | — |

**These briefs do not overlap in a single derived number.** A learner reading foundations then architectures
meets two different 8B models with the same name-shape and no acknowledgement.

**RULING. Qwen3-8B is the course-wide arithmetic anchor. Everywhere. Llama-3.1-8B, Llama-2-7B, and the generic
"7B" are RETIRED from all parameter, memory, KV, and FLOP arithmetic.** Qwen3-0.6B/1.7B appear **only** where a
run must finish in under two minutes. Llama-3.3-70B appears **only** on the LLM hardware page, for the QLoRA
capability demo.

**Reasoning.**
1. **It is the only one that can be verified live.** Llama configs are gated (401). Qwen3's `config.json` and
   `model.safetensors.index.json` were both fetched this session.
2. **The derivation is exact, twice.** 8,190,735,360 from the formula; 16,381,470,720 bytes ÷ 2 (bf16) =
   8,190,735,360 from the checkpoint's own index. **Two independent routes to the same integer.** That is the
   strongest credibility beat available to this course and no other candidate has it.
3. **Apache-2.0, ungated.** No license digression, no waiting on an approval email mid-lesson.
4. **The lead's instruction says "allow a smaller model for what actually runs on the box if genuinely
   necessary." It is not necessary.** Qwen3-8B LoRA = 17.08 GB and ~12 minutes on his box (constants §2.3, §6.7).
   **The anchor and the runnable model are the same object.** This is strictly better than a split — the
   recurrence is the entire point, and the arithmetic he does is arithmetic about the model he trains.
5. **The one genuine split is for speed, not capacity:** Qwen3-0.6B for the 60-second smoke test and the GRPO
   stretch goal (which needs generation in the loop). Same family, same tokenizer, same config *shape* — so the
   arithmetic transfers by substituting four numbers, which is itself the lesson.

**Brief edits.**
- `brief-foundations` §2 — **rewrite the entire FFN worked example against Qwen3-8B.** $d_{ff}$: 14336→**12288**.
  "~70% FFN" → **66.4% of model / 78.26% of block** (say which). The "96 GB AdamW teaser" → **131.05 GB**
  (§D-03). §8's $\ln(128256)=11.76$ → **$\ln(151936)=11.93$**. §13's dependency spine references likewise.
- `brief-foundations` §11 — the SwiGLU 2/3-rule worked example uses Llama's 14336 = 3.5d. **Rewrite for Qwen3's
  12288 = 3d.** The *rule* is unchanged and the "why not 16384?" payoff is *better*: 3d is closer to the honest
  2.67d, so Qwen3 "spent less" than Llama did.
- `brief-training` §5.5, §7.6, §16 — every "7B" → Qwen3-8B; every derived figure recomputed (constants §2).
- `brief-training` §7.6's LoRA example (Llama-2-7B, attn-only, r=16, 16,777,216) → **Qwen3-8B, all-linear, r=16,
  43,646,976** (§D-06 makes all-linear the default, so attn-only was the wrong example anyway).
- `brief-tooling` §3 rung 6 — Qwen3-4B → **Qwen3-8B**. Its own §5.3 verifies 8B LoRA at 6,969 tok/s.
- `brief-tooling` §6.3 Examples A–D — all recomputed against Qwen3-8B; the `[UNVERIFIED] d, n_L` flags disappear.

**Confidence: HIGH.**

---

### D-02 ⛔ THE TWO RIVAL CANONICAL NETWORKS

**Conflict.** Two 2-2-1 tanh/sigmoid/BCE networks with 9 parameters, **both declared canonical in imperative
language**, both shipping a **conflicting load-time unit test for the same demo**:

| | pedagogy §9.2 "TN-1" | training §5.4 |
|---|---|---|
| $W_1$ | `[[0.5,−0.3],[0.8,0.2]]` | `[[0.5,−0.3],[0.2,0.8]]` |
| $W_2$ | `[0.6, −0.9]` | `[0.7, −0.4]` |
| $x$ | `[1.0, 2.0]` | `[0.60, −0.20]` |
| $\eta$ | 0.1 | 0.5 |
| $\mathcal{L}$ | 0.9869 → 0.8222 | 0.453048 → 0.340179 |
| mandate | *"treat these as canonical and hand them to every relevant agent verbatim"* | *"Use them verbatim."* |
| test | `assert allclose(grad, [[−0.3764,−0.7527],[0.2028,0.4056]])` | *"Verify at load that it reproduces L = 0.453048"* |

$W_1$ differs **only by transposing the second row** — near-certainly one brief mis-transcribing the other.
**Pedagogy's own anti-pattern #18 names this exact failure ("a number that contradicts the same number on
another page — TN-1's 0.3727"). The corpus already contains the contradiction.**

**RULING. TN-1 is the single canonical network. brief-training §5.4's network is RETIRED.** Its constants,
corrected, are frozen in `constants.md` §8. **§5.4's prose, pedagogy, and beats are retained and re-hosted on
TN-1.**

**Reasoning.** The arithmetic verifier recommended §5.4 *on arithmetic grounds* (flawless, real FD check, no
degeneracy) but explicitly noted TN-1's dead-unit lesson is worth more and that TN-1 is the 9-page spine. I
agree, and go further:

1. **TN-1 is structurally load-bearing; §5.4 is not.** Nine pages, the Early Real Thing, the autograd assertion,
   the JS demo, the attention recast, and the LoRA decomposition all name TN-1. §5.4 is one section.
2. **The dead unit is the better lesson and it has no substitute.** $a_{1,1}=0 \Rightarrow \partial\mathcal{L}/\partial W_{2,1}=0$
   ⇒ that weight never moves — visible in the *result* — while $\tanh'(0)=1$ means the *upstream* gradient flows
   fine. **Two contrasting lessons in one hand-computed example, on page 10, for free.** §5.4 has no equivalent.
3. **§5.4's arithmetic superiority is now moot** — TN-1's six roundings are corrected and re-verified to 30 d.p.
   (`constants.md` §8), and the FD check works on any network.
4. **What is not acceptable is shipping both**, and that is exactly what the corpus does.

**⚠️ The cost, stated honestly — NOW PAID IN FULL (remedy (a), computed).** Retiring §5.4 cost one good beat:
**"the largest gradient is 12.7× the smallest — therefore Adam."** TN-1's *first* input has a dead unit (spread
contaminated by a zero gradient; the old "3.09×" was itself wrong — it is **3.7119×** over the nine parameter
gradients, a seventh mis-rounding). **RESOLVED via remedy (a):**
- **(a) ADOPTED and COMPUTED.** The **second canonical input** $x = [0.60, -0.20]$ ($z_1=[0.46,0.34]$, **no dead
  unit**) is frozen in `constants.md` §8.7 with all nine gradients, verified three independent ways. **Spread =
  10.22×, all nine live** — a rounder, more quotable "one order of magnitude" than 12.7×. The course now has **one
  network, two inputs**: the dead-unit case (input 1) and the spread case (input 2). Assertion frozen at 5 d.p.
  (`atol=1e-5`) to avoid a tolerance coincidence. ⚠️ Its nine gradients WERE computed (not invented); §Z-1 closed.
- ~~(b) Fallback (live-computed spread)~~ — **not needed; (a) landed.**

**`12.7×` is retired** (it belonged to the retired network); **10.22× replaces it.**

**Brief edits.** `brief-training` §5.4 — replace the network; retain the sign-flip callout, the "this line IS
backprop" callout, the saturation thought-experiment, the FD-check section, and the "loss went down, he did it
with a pencil" beat. `brief-training` §16's table — six rows retired. Demo 2's load-time assertion → TN-1's.
`brief-pedagogy` §9.2 — six roundings corrected; the "exactly zero" box rewritten (D-13).

**Confidence: HIGH** on retiring one and keeping TN-1. **(a)-vs-(b) RESOLVED: (a), computed and frozen** (§Z-1).

---

### D-03 ⛔ THE 130-vs-131 COINCIDENCE, AND THE LoRA HEADLINE

**Conflict.** Two numbers that agree by accident on different bases:
- `brief-training` §7.6: **7B**, state 112 GB, **+ ~18 GB activations** → "~130 GB"
- `brief-llm-finetuning` §8: **8.19B**, state **131 GB**, activations **explicitly excluded** ("before a single activation")

Different model **and** different accounting, landing 1 GB apart. **A learner reading both concludes +1.19B
params costs +1 GB. It costs +19 GB of state (112 → 131).**

**RULING.**
1. **Standardize on Qwen3-8B, state-only = 131.05 GB / 122.05 GiB, everywhere.** (D-01.)
2. **Activations are ALWAYS a separate, separately-labelled line, tagged [EST], on both sides of every comparison.**
3. **The headline pair is `131.05 GB → 17.08 GB` (7.67×), state-to-state.** The "130 GB → 14.3 GB" pair is deleted.
4. `brief-training` §7.6's cross-reference note — *"the fine-tuning brief should open by CALLING BACK to §7.6's
   130 GB → 14.3 GB"* — **would propagate the wrong pair into the next brief. Rewrite it to 131 GB → 17.1 GB.**

**Reasoning.** The 130/14.3 pair silently credits LoRA with an **activation saving it does not deliver.** LoRA
freezes weights; it does not shorten the graph. You still backprop through the full network, so activations are
~unchanged. As written, a learner who measures LoRA's real footprint finds it **well above 14.3 GB** — by
exactly the activation term the comparison hid. `brief-llm-finetuning` §9's "17.1 GB vs 131 GB, 7.7×" is the
correct like-for-like framing (both state-only). **Use that one.**

**Confidence: HIGH.**

---

### D-04 ⛔ THE 131-vs-128 CLIFFHANGER IS A UNIT ARTIFACT

**Conflict.** `brief-llm-finetuning` calls this *"the best number in the entire brief"*; §5 calls it *"the
emotional core of the track"*; and `05_memory_ledger.py` **mandates the learner reproduce the OOM on his own
box.** It must survive contact with reality. **It does not.**

```
8,190,735,360 × 16 B = 131,051,765,760 B
   = 131.05 GB (decimal, 1e9)   = 122.05 GiB (binary, 2^30)

vs 128 GiB physical (DRAM is conventionally binary) →  +5.95 GiB   FITS
vs 128e9 bytes (decimal reading)                    →  −2.84 GiB   OOM
```

**The "3 GB over" compares 131 decimal GB against 128 binary GiB.** DRAM capacity is essentially always binary:
128 GB of LPDDR5X is 128 GiB = 137.44e9 bytes. **Under the correct reading the full fine-tune state fits with
~6 GiB to spare and the cliffhanger evaporates.** The briefs are consistently decimal elsewhere (16.4 GB bf16
weights; the 14.0/13.0 GB/GiB column), which makes the 128 the odd one out.

**RULING.**
1. **`constants.md` §0 fixes one unit convention, course-wide: GiB for anything compared to hardware capacity,
   GB for weights.** Hard rule, mechanically checkable.
2. **The course asserts NO memory budget for the Spark.** Step 1 of the ledger script is
   `torch.cuda.mem_get_info()` / `free -g`. He **measures** his usable memory, then computes **122.05 GiB**
   against *that*.
3. The conclusion — *full FT of an 8B does not fit on a Spark* — is **very likely still true** (usable unified
   memory is meaningfully below 128 GiB after OS/driver reservation, and activations add 2–6 GB on top of
   122.05). **But the course arrives there by measurement, not assertion.**

**Reasoning — and this is the ruling I feel most strongly about.** *He owns the box.* The course explicitly
dares him to disprove a number on hardware he has. As written he will find that the state allocates fine and
OOMs for a different reason — **precisely the "told to check against reality, hits a contradiction" failure this
whole arbitration exists to catch.** The brief's own flag (`[CONF: high — verify exact spec at spec-writing
time]`, `[M]`) **is load-bearing and has not been discharged.** No verifier could discharge it: NVIDIA does not
publish usable-after-reservation memory.

**And the measured version is a better page.** The near-miss becomes **real, personal, and
unfalsifiable-by-unit-convention.** The emotional beat is identical — arguably stronger, because **the number is
his.** He types one line and finds out what his machine actually has. That is the course's thesis in one command.

---

**ADDENDUM (2026-07-16, post-ratification): THE MEASUREMENT WAS TAKEN.** The lead SSH'd into the learner's
actual Spark (`hardware-ground-truth.md`). `MemTotal` = 130,662,936,576 B = **exactly 121.6875 GiB**. The
firmware/driver carveout is **6.3125 GiB (4.9% of the physical 128 GiB)**.

**Ruling point 3 is upgraded from "very likely" to MEASURED FACT — with a twist this ruling did not predict:
the FITS arithmetic above (+5.95 GiB) was itself wrong**, because it assumed the physical 128 GiB is usable.
Measured verdict: **122.05 GiB needed vs 121.69 GiB existing = −0.36 GiB. DOES NOT FIT, by 0.3%, before
activations.** So the original brief's cliffhanger conclusion was right *by accident* (wrong reason: a unit
error), this ruling's correction was wrong *on the merits* (right method: measure), and the measurement
vindicated the method while refuting the interim number. Three layers, each catching the one above.
**Ruling point 2 stands unchanged and is now the page's proven design:** he measures, then computes 122.05
against his own number, and misses by a hair. See `constants.md` §6.8 [MEA-DEV] for the frozen numbers.

**Brief edits.** `brief-llm-finetuning` §8 (the whole "3 GB over" framing), §5, §14's verdict table, Demo A9,
artifact `05_memory_ledger.py`. `brief-architectures` §10's "misses by about 3 GB" cliffhanger and §12.1.
`brief-pedagogy` Demo C's 128 GB line → a *measured* line. `brief-tooling` §5.5/§6.3's "119.2 GiB" and "budget
~110 GB usable" → measured.

**Confidence: HIGH** on the unit artifact and on measuring. **MEDIUM** on whether the OOM actually reproduces —
which is exactly why we measure.

---

### D-05 THE OPTIMIZER-SHRINK FACTOR

**Conflict.** `brief-training` §7.6: *"the optimizer state shrank by 224×."* **Refuted.** Provenance
reproduced: $56\ \text{GB} / 0.25\ \text{GB} = 224$ — but 56 GB is Adam's **$m+v$ only** while 0.25 GB is LoRA's
**full 16 B/param state** (grads + master + m + v). **Mismatched numerator and denominator.**

**RULING. On the Qwen3-8B anchor: the trainable state shrinks by 187×, and 187 is exactly the parameter ratio.**

$$\frac{8{,}190{,}735{,}360}{43{,}646{,}976} = 187.7 \qquad\text{and}\qquad \frac{114.67\ \text{GB}}{0.61\ \text{GB}} = 187.7$$

**Reasoning.** The verifier's like-for-like correction was **417×** — correct, but computed on the retired
Llama-2-7B / attention-only example (7e9 / 16,777,216). D-01 retires that anchor and D-06 makes all-linear the
default, so the correct figure for this course is **187×**. **It is exactly the parameter ratio, which is a
*better sentence* than either:** LoRA's state shrinks by exactly the factor by which the trainable parameter
count shrinks, because state is linear in trainable params. Nothing else is going on. That is the mechanism,
stated as an identity.

> **The sentence to ship:** *"LoRA is not a model compression technique. LoRA is an optimizer-state compression
> technique. The base model didn't shrink — all 16.38 GB of it is still resident. The trainable state shrank by
> 187×, which is exactly 8,190,735,360 ÷ 43,646,976, because optimizer state is linear in trainable parameters.
> That's the whole trick."*

**Brief edits.** `brief-training` §7.6, §16, §18. `brief-tooling` §6.3 Example C's "430×" (a third rival figure,
from the retired 7B/attn-only example) and §15.2.

**Confidence: HIGH.**

---

### D-06 THE LoRA TARGET-MODULE DEFAULT

**Conflict.** `brief-tooling` §4.2: *"attention-only vs attention+MLP is a real, live trade-off — MLP targeting
roughly triples adapter params for typically-modest gain. Present as a trade, not a rule."*
`brief-llm-finetuning` §9: *"**Apply LoRA to all linear layers.** Attention-only underperforms even when you
raise the rank to match the parameter count. This is the most persistent stale advice in the field."* [V-2°]

**RULING. `target_modules="all-linear"` is the course's default for the LLM track. brief-tooling loses.**

**Reasoning.** Not a matter of taste — **the learner computes the refutation himself two pages earlier.**
`constants.md` §3: the MLP matrices are **150,994,944 of the 192,937,984** targetable params per layer = **78%**.
Attention-only means *you froze 78% of the model and then wondered why it didn't learn.* The trade-off framing
survived from the 2021 LoRA paper's setup and is stale. **This misconception is worth a big box because it is
(a) universally repeated, (b) wrong, (c) refuted by a number he just computed, and (d) fixed by a one-word
config change.**

**Diffusion is different and that is not a contradiction** — see D-11. Diffusion's `to_q/to_k/to_v/to_out`
default is about *what* you're teaching (attention is where cross-modal binding lives), not about a parameter
budget. **Say so explicitly at the fork**, or the two tracks look like they disagree.

**⚠️ MoE caveat [VP]:** HF implements experts as `nn.Parameter`, not `nn.Linear`, so **`all-linear` silently
skips the majority of an MoE's parameters.** PEFT's `target_parameters=[...]` reaches them. **If the course
touches MoE fine-tuning at all, this is a must-mention.**

**Brief edits.** `brief-tooling` §4.2's misconception box — reverse it. §14's file list.

**Confidence: HIGH.**

---

### D-07 THE NON-EMBEDDING COUNT DIFFERS BY 304,128

**Conflict.**
- `brief-architectures`: per-block **192,946,432** (norms included) × 36 = **6,946,071,552**; total
  **8,190,735,360** — claims **exact** reconciliation. ✅
- `brief-llm-finetuning` §9: per-layer **192,937,984** (norms **omitted**) × 36 = **6,945,767,424**; total
  **8,190,427,136** — claims it *"closes to three significant figures."*

It does, which is how it went unnoticed. **But both cannot be the reference derivation, and one claims exactness.**

**RULING.** `brief-llm-finetuning` §9's **table keeps its 7 rows** (correct — norms are not LoRA targets). Its
**"check the base count" line must add the norms and cite `8,190,735,360`, not re-derive `8.190B`:**

$$6{,}945{,}767{,}424 + \underbrace{36\times8{,}448}_{304{,}128} + \underbrace{4{,}096}_{\text{final norm}} + 2\times622{,}329{,}856 = \mathbf{8{,}190{,}735{,}360}$$

**No page may say "closes to three significant figures."** It closes **to the byte**, and saying so *is the point*
— it is the moment model cards stop being authority and become arithmetic.

**Reasoning.** The omission is harmless *for the LoRA table's purpose* and actively correct there. But a
norm-free subtotal presented as "the model's parameter count" is the difference between a course that reconciles
exactly and one that hand-waves — on the exact page whose whole purpose is to prove nothing is hidden.

**Confidence: HIGH.**

---

### D-08 THREE RIVAL CANONICAL LOGIT VECTORS

**Conflict.** Nobody flagged it. `brief-training` §2.2 uses $z=[2.0,1.0,0.1]$, computes $\mathcal{L}=0.417030$,
and declares *"Reuse this exact triple everywhere in the course."* `brief-foundations` §8 uses
$z=[2.0,1.0,0.1,-1.0]$ → $\mathcal{L}=0.4491$. `brief-architectures` §F2b uses $z=[2,1,0]$ → $A=[0.665,0.245,0.090]$.

**RULING. `[2.0, 1.0, 0.1]` — training's — is canonical.** Frozen in `constants.md` §9.2. It is used for the CE
worked example, the temperature demo, top-k/top-p, the $\sqrt{d_{\text{head}}}$ saturation demo, and the CFG
discussion. The other two are retired.

**Reasoning.** Training's brief claimed it first and explicitly ("familiar numbers are pedagogical
infrastructure"). Three is worse than any one. Architectures' $[2,1,0]$ needs only a trivial rewrite —
$[2.0,1.0,0.1]$ scaled by 11.31 makes the same one-hot-in-a-trenchcoat point.

**Confidence: HIGH.**

---

### D-09 THE DIFFUSION LEARNING RATE — a 100× error

**Conflict.** Nobody flagged it. `brief-foundations` §4 plants, unqualified: *"Diffusion U-Net / DiT fine-tune:
$\eta \approx 1\times10^{-5}$ to $1\times10^{-6}$."* `brief-diffusion` §13.2: **LoRA `learning_rate` = 1e-4**,
full DreamBooth = 1e-6–5e-6.

**Foundations conflated full fine-tuning with LoRA. The gap is 100×.** A learner following foundations' number
for a diffusion LoRA gets a LoRA that does not learn — and, per `brief-diffusion` §12.4, **the loss curve will
not tell him**, because diffusion LoRA loss is noise-dominated by the random $t$ draw. **He would burn days.**

**RULING.** The LR table is frozen in `constants.md` §9.4, **generated by a stated principle rather than listed**:

> **Fewer trainable parameters ⇒ higher learning rate.** Full FT moves 8.19B parameters that already encode
> everything the model knows; it must tiptoe. Textual inversion moves **768** and can stride.

`1e-6` (diffusion full FT) → `1e-4` (LoRA, both tracks) → `5e-3` (textual inversion). **Three orders of
magnitude, three methods, one principle.** `brief-diffusion` §13.2 and `brief-llm-finetuning` both independently
noticed this and both flagged "the other brief will need this — reconcile." **It is one principle. State it once,
in the trunk, on the memory-ledger page** (where trainable-parameter count is already the subject).

**Brief edits.** `brief-foundations` §4 and §14's "medium/low confidence" list.

**Confidence: HIGH.**

---

### D-10 THE HARDWARE VERDICTS — propagate wholesale

Not relitigable. Summarized here so no agent has to re-read a verifier report; **details in `constants.md` §6–§7.**

| # | Was | Now |
|---|---|---|
| a | 273 GB/s `[M]`, doubted by brief-architectures | **[VP] CONFIRMED.** The doubt is cleared; **brief-tooling was right.** brief-architectures §12.6's "someone must verify" is **discharged**. |
| b | "1 PFLOP" as a compute ceiling | **FP4 with 2:4 sparsity.** Never size a run off it. |
| c | Dense BF16: 125 TF (tooling) / 31 TF FP32 (foundations, pedagogy) / unstated (architectures) | **~62 TF [INF]**, chain shown, **labelled "inferred, not published"**. The 31 TF figure is unsourced — **delete it.** |
| d | ridge 3,663 (pedagogy) / 458 (tooling) | **227 working [INF]** — but **458 is NOT retired: it is correct if the BF16 ceiling is ~125 TF, now flagged unresolved (§6.4, Task 3).** 3,663 (sparse-FP4 peak) stays retired. |
| e | "60–69% MBU across four measurements" | **Heuristic SURVIVES** (Task 1 resolution corrected the specs pass's strawman). The brief's **four dense batch-1 points all land 60–75% MBU** under a consistent decode-traffic byte count (checkpoint − embedding table); NVFP4 checkpoints are really **6.03/10.54 GB**, not 4.5/7.5. The **batch-8 (DeepSeek) and MoE (gpt-oss, implying 234% of peak — impossible)** cases are the two instructive **failure modes**, not failed supports. Heuristic: `0.65 × BW / weight-bytes-read-per-token, dense, batch 1`; MoE blow-up is the punchline; byte-count is the deeper lesson. |
| f | trl 0.29.1 (tooling) vs 1.8.0 (llm-finetuning) | **1.8.0.** brief-tooling is stale, and its recommendation to *avoid* `trl` in favour of `Trainer` was based on that stale version. **Overruled: use `trl>=1.8,<2` + `SFTTrainer`.** The v0→v1 break is small per `MIGRATION.md`. |
| g | transformers v5 = "a hard break" | **mostly soft.** `torch_dtype` has a working BC shim; `batch_decode` survives; `warmup_ratio` was **never removed** (the changelog is wrong — do not trust it). **The real hazard is silent default changes**, esp. `report_to` "all"→"none" and dtype-as-saved. |
| h | "torch runs on GB10 via binary compat + PTX JIT" | **SASS-only. Release wheels ship NO PTX.** Therefore *"the first kernel JIT after a driver change is slow — that's the PTX compile"* **cannot happen on a stock wheel. Delete the story.** |
| i | bitsandbytes "DGX Spark cuda121" garbled | **Resolved and it inverts the framing: bitsandbytes compiles native sm_121 ONLY into the aarch64+CUDA-13 wheel.** x86 does not get it. **The Spark is *better* served than an x86 box here** — and the 70B QLoRA path depends on it. |
| j | vLLM blocked on SM121 | **Supported.** Issue #31128 closed. ⚠️ **Via the `vllm/vllm-openai:cu130-nightly` container, pinned by digest** — the PyPI aarch64 wheel is CUDA-12 and fails. |
| k | "aarch64 is a broad practical tax" | **Overstated for our stack** (torch, transformers, peft, trl, diffusers, bitsandbytes, vllm all ship aarch64). **The real trap is narrower and sharper: any package pinning `libcudart.so.12` fails at import** on CUDA-13-only DGX OS, *before* kernel compatibility is evaluated. |
| l | "Ferrari with a garden hose" (1 PFLOP ÷ 273 GB/s) | **Retired — arithmetically unfair.** The honest version (~62–125 TF vs 273 GB/s → **227–458 FLOP/byte**, ceiling unresolved) is **still bandwidth-starved, still makes the point, and survives his checking it.** |
| m | 53,657 tok/s for Llama-8B LoRA (brief-diffusion §15.3) | **6,969.59** — NVIDIA's own blog. Trust the primary. |

**Serving recommendation.** `brief-tooling` says llama.cpp/Ollama because "vLLM is rocky." **That premise is
dead.** Ruling: **teach llama.cpp/Ollama as the default local path** (it works, it is the GGUF endpoint of the
merge-and-serve loop, and Arm publishes an official DGX Spark learning path), **and name the vLLM cu130
container as the throughput option, pinned by digest.** Both, honestly, with the container caveat stated.

**Confidence: HIGH** (these are the verifiers' calls, not mine).

---

### D-11 ⛔ LoRA RANK: LLM r=256 vs DIFFUSION r=16

**Conflict.** The LLM track says **r=256 is nearly free and being stingy is the error** (`constants.md` §3:
r=256 → 27.55 GB, still trivially fits). The diffusion track says **r=16–32, and r=128 on 20 images memorizes
them.** `brief-tooling` §11.1 says there is **no consensus, present it as a trade**. `brief-training` §13.3 says
**rank IS the bias-variance tradeoff wearing a 2026 costume**. Four positions.

**RULING. One story, and it is `brief-llm-finetuning` §16's, ratified and elevated:**

> ## **Rank should scale with the information content of your dataset — not with the size of your model.**

The rank the task needs is set by **how much there is to learn**, and the two tracks sit at opposite ends of
that axis:

| | LLM SFT | Diffusion concept LoRA |
|---|---|---|
| Dataset | 10k–50k examples of varied behaviour | **15–20 images of one thing** |
| Information to install | a domain, an idiom, a reasoning shape | *one point on a manifold the model already knows* |
| Capacity needed | **r = 64–256** | **r = 8–32** |
| What goes wrong at high $r$ | **nothing — rank is nearly free in memory** | **memorization of your 20 photos** |
| What goes wrong at low $r$ | **underperforms full FT** (capacity-constrained) | nothing — it's plenty |

**Reasoning — why this resolves rather than papers over.**
1. **It makes the two tracks' advice a *consequence* of one rule, not two rules.** A rusty engineer will notice
   r=256 and r=16 in the same course. He must be told *why*, or he concludes the course is stitched together.
2. **It is the same principle as D-09's LR rule and D-13's "how many examples" question** — all three are
   *"how much are you actually asking the model to learn?"* **Three apparent hyperparameter mysteries, one
   question.** That is worth engineering deliberately.
3. **It absorbs `brief-training` §13.3's bias-variance framing exactly.** $r$ *is* the capacity knob. $r=4$ →
   high bias. $r=256$ on 20 images → high variance, memorizes, forgets. **The classical U-curve is alive and
   well and living in his LoRA config** — and the U's *location* is set by dataset information content. Same rule.
4. **The two "the rank is nearly free" claims are both true and not in conflict** — they are *memory* claims.
   `brief-tooling` §8.2 demo #3 and `brief-llm-finetuning` Demo A9 both land "drag rank 16→256, the bar barely
   moves." **True, and the correct follow-through is: *memory is not what constrains your rank — your dataset is.***
   That is a better lesson than either brief's, and it converts a surprising demo into a principle.

**Therefore `brief-tooling` §11.1's "no consensus, try both" is overruled** — "try both, the experiment costs an
hour" remains **excellent practical advice** and stays, but it is now the *epilogue to a rule*, not a substitute
for one.

**Where the rule is taught: the trunk's LoRA page**, once, at the moment $r$ is defined. Both tracks then say
"as the trunk said, r follows the dataset — here's what that means for mine."

**Confidence: HIGH** on the rule. **MEDIUM** on the specific numbers in the table (all [EST] — diffusion's are
explicitly practitioner folklore; label them).

---

### D-12 THE SVD / RANK OWNERSHIP GAP — a real hole

**Conflict.** `brief-foundations` §0 **explicitly CUT** eigen/SVD, calling it *"the cut I'm least certain of"*
and offering *"one gated paragraph inside the LoRA chapter."* Meanwhile:
- `brief-llm-finetuning` §1's dependency spine: *"A7. LoRA ← needs **SVD/low-rank from trunk**"*
- `brief-diffusion` §11.8: *"**this is likely a gap and it is load-bearing** … request a 1-page SVD inset in the
  trunk, because the LLM brief will need exactly the same thing. **This is a shared dependency and should be
  built once.**"*

**Both destination tracks depend on a trunk section that does not exist.** The lead has already ruled that the
trunk owns LoRA's derivation and therefore owns rank.

**RULING. The trunk gets a full page: "Rank, SVD, and low-rank approximation," immediately before the LoRA page.**
Not a gated paragraph. Not an inset. **A page**, with the demo.

**Reasoning.** Foundations' cut was right *for foundations* — nothing before LoRA needs it, and an eigen chapter
in the maths spine would be cargo cult. **But "low rank" is not a word you can use without a meaning, and it is
the single load-bearing claim under the course's entire destination.** The cut is correct; the *placement* is
what was wrong.

**And the payoff is one of the best demos in the corpus, appearing independently in two briefs** — `brief-llm-finetuning`
Demo A10 and `brief-diffusion` D11.1 both invented essentially the same thing. **Build it once:**

> **The rank slider.** A real 256×256 grayscale image. Rank-$r$ reconstruction $\sum_{i\le r}\sigma_i u_i v_i^\top$.
> The residual at 4× gain. A log-scale scree plot of $\sigma_i$. Live readouts: stored numbers $r(256+256+1)$ vs
> 65,536; compression ratio; relative error.
> - $r=8$: **recognizable.** 6.3% of the storage.
> - $r=32$: **hard to distinguish.** 25% of storage, ~3% error.
> - The scree plot **falls off a cliff.**
>
> ***"Real matrices are approximately low-rank. The information isn't spread evenly across 65,536 numbers — it's
> concentrated in a few dozen directions. That is the entire LoRA hypothesis, and you just verified it on a
> photograph."***

**And then the move that makes it honest — merge in `brief-llm-finetuning` Demo A10's dropdown, which is the
better half of the idea:** offer **three** matrices — (1) a random Gaussian, (2) **a real slice of a pretrained
$W_0$**, (3) **a real slice of a $\Delta W$ from an actual fine-tune.** The scree plots are *completely
different shapes.* Random decays slowly. **Pretrained $W_0$ decays slowly too — it is nearly full rank, and a
rank-16 approximation of it destroys the model.** But **$\Delta W$'s singular values fall off a cliff.**

> **In one dropdown flip, the learner sees that the LoRA hypothesis is a claim about $\Delta W$ and is FALSE
> about $W_0$.** That is *the* misconception ("the model is low-rank"), killed by direct observation. Nothing
> else in the corpus teaches it as efficiently.

**Brief edits.** `brief-foundations` §0's cut list and §14's editorial-calls list — record the overrule and the
reason (placement, not existence). Merge Demo A10 and D11.1 into one trunk demo.

**Confidence: HIGH.**

---

### D-13 THE "EXACTLY ZERO" GRADIENT — mandated pre-emption

**Conflict.** `brief-pedagogy` §9.2's designed feature: *"the learner sees $\partial\mathcal{L}/\partial W_2$'s
first entry come out **exactly zero**."* **False in every float.** `constants.md` §8.4: float64 gives
2.78e-17; float32 gives −2.24e-08; torch prints `1.4020191e-08`.

**RULING. `constants.md` §8.4's framing is MANDATED VERBATIM. Agents may not regress it.**

**Reasoning.** This lands on **page ~10, the Early Real Thing**, at the exact moment autograd is supposed to stop
being magic, in a course whose promise is **"three representations, one number."** Left unpatched, ~9 fan-out
agents each write "you will see 0.0" and **the spine visibly breaks on his own box.** It is the single most
likely place the whole design fails.

**But it is fixable into an asset, and the fix is better than the plan was.** `0.5 − 0.6 + 0.1` is *the* textbook
float non-associativity demo. The course's own logsumexp page teaches "the math on paper ≠ the math in the
machine." **This is that lesson, arriving three pages early, for free, discovered rather than told** —
which is exactly the run→derive→break ordering the pedagogy brief argues for.

Then: **the mandated autograd assertion must use `−0.7527`, not `−0.7528`** (`constants.md` §8.5). With the
brief's value the error is 9.66e-05 against a budget of 1.075e-04 — **a 10% margin, surviving only because
`atol=1e-4` happens to absorb the brief's own rounding error.** At `atol=1e-5` it fails. **Fixing the rounding
is what stops a nine-page thread from depending on a tolerance coincidence.**

**Confidence: HIGH.**

---

## PART B — OWNERSHIP RULINGS (who teaches what, once)

### D-14 THE OWNERSHIP MAP

Every "teach once, where?" question, ruled. **The rule generating all of them: whoever owns the *why* is not
necessarily whoever owns the *where*, and both must be named or the topic falls between two briefs.**

| Topic | **WHY** (mechanism) | **WHERE / HOW** (application) | Flagged by |
|---|---|---|---|
| **Residual connections** | **Trunk** — the gradient-flow fix; $\partial y/\partial x = I + \partial F/\partial x$; $0.9^{36}=0.0225$ vs ≈1 | **Architectures** — the residual stream as a shared bus; where the `+x` sits in the block | training §18, architectures §12.7 |
| **Pre-norm vs post-norm** | **Trunk** — one bracket decides whether an 80-layer model trains | **Architectures** — the block diagram; RMSNorm's config line | training §11.5 |
| **Normalization (the axis question)** | **Trunk** — "the ONLY difference is which axis you reduce over" | **Architectures** — RMSNorm in the block | training §18 |
| **Gradient checkpointing** | **Trunk** — §5.5 plants it: BP4 needs $\mathbf{a}^{(\ell-1)}$, *therefore* training eats VRAM | **Tooling** — `gradient_checkpointing_enable()`, `use_cache=False`, the PEFT interaction | training §18, tooling §7.2 |
| **Mixed precision / bf16 vs fp16** | **Trunk** — the fp32-master row of the ledger; 65,504 | **Tooling** — AMP, and GradScaler **in the past tense** | training §18 |
| **Quantization: number formats** | **Trunk** — blockwise scaling; NF4/FP8/NVFP4/MXFP4; *"4-bit names a budget, not a scheme"* | — | llm-finetuning §16, diffusion §15.4 |
| **Quantization: deployment** | — | **LLM track** — GGUF/AWQ, merge-then-requantize | llm-finetuning §1 |
| **Quantization: FP4-native payoff** | — | **Diffusion track** — Blackwell FP4 tensor cores; the wall-clock win | diffusion §15.4 |
| **QLoRA** | **Trunk** (NF4 is a number format) | **LLM track** (it's the 70B path) | |
| **Rank / SVD** | **Trunk** — D-12 | each track's targets & ranks (D-11) | diffusion §11.8, llm-ft §1 |
| **LoRA's math** | **Trunk** — $W_0 + \frac{\alpha}{r}BA$; $B{=}0$; $\alpha/r$; the 187× | **Each track** — targets, ranks, gotchas (LLM: `all-linear`; diffusion: `to_q/k/v/out`, conv-for-styles, ComfyUI `strength`) | lead's ruling; diffusion §17.4 #5 |
| **Attention** | **Trunk — MASK-AGNOSTIC, with cross-attention FIRST-CLASS** | LLM: causal masking. Diffusion: bidirectional + MMDiT | **diffusion §17.4 #1 — highest-priority ask in that brief** |
| **Sinusoidal embeddings** | **Trunk** — once | LLM: RoPE (a *different* thing). Diffusion: $t$-embedding via FiLM | diffusion §17.4 #3 |
| **Cross-entropy from MLE** | **Trunk** — both branches, one derivation | LLM: *"you already know the LLM loss — it's Branch C, per position"* | training §18 |
| **MSE from MLE** | **Trunk** — Branch A, Gaussian likelihood | Diffusion: **the diffusion loss is a corollary, not a new fact** | training §18 |
| **`.detach()` / stop-gradient** | **Trunk** — "cutting a wire on purpose" | Diffusion: detached targets, EMA outside autograd | training §6.7 |
| **Jensen + KL(𝒩‖𝒩) closed form** | **Trunk** — ~½ page. **Likely a genuine gap.** | Diffusion §2.4b and §4.4 both need it | diffusion §17.4 #8 |
| **CLIP / contrastive training** | **Trunk** | Diffusion: the text encoder. LLM: why RAG needs a second model | diffusion §9.8, architectures §12.4 |
| **GANs** | — | **Diffusion track**, ~½ page: mode collapse; distillation ancestry; **the VAE decoder is GAN-trained** | diffusion §17.4 #9 |
| **Embedding geometry / anisotropy** | **Trunk** | LLM: why RAG uses a dedicated bidirectional embedder | architectures §12.4 |
| **The LR-vs-trainable-params principle** | **Trunk** — the ledger page (D-09) | both tracks instantiate | diffusion §17.4 #6, llm-ft |
| **"How many examples"** | **Trunk** — same information-content rule as D-11 | LLM: 1,000+. Diffusion: 15–20 images | llm-ft §16 |
| **The transformer block** | **Trunk** | — | **lead's ruling.** Both tracks need it; splitting it costs ~6 pages and guarantees drift |

**The two asks I am ratifying loudest, because they are load-bearing and easy to lose:**
1. **Attention MUST be taught mask-agnostically and cross-attention MUST be first-class, not a footnote.**
   `brief-diffusion` §17.4 calls this its *highest-priority reconciliation* and it is right. If the trunk teaches
   attention only in a causal-LM framing, **§7 and §9 of the diffusion track break** — diffusion attention is
   bidirectional, and cross-attention takes $K,V$ from a *different modality*. **A learner who thinks Q/K/V are
   "three memories" cannot understand cross-attention at all, which means he cannot understand diffusion
   conditioning at all.** Highest stakes in the trunk.
2. **Jensen's inequality and KL between equal-covariance Gaussians.** Half a page. **Without them the ELBO
   collapse — the emotional centre of the diffusion track — is a wall.** The KL lemma is *the single line where
   the monster dies*: an integral over $\mathbb{R}^{3{,}145{,}728}$ becomes $\|\mu_1-\mu_2\|^2/2\sigma^2$.

---

### D-15 ★ THE SPARK CUTS OPPOSITE WAYS — engineered, with one formula

**Conflict.** `brief-llm-finetuning` §16 asks for this contrast to be *"engineered deliberately"*: LLM decode is
bandwidth-bound (the Spark is weak); diffusion sampling is compute-bound (the Spark is relatively strong).

**But the briefs contradict each other on the diffusion half.** `brief-diffusion` §8.4 asserts FLUX at bf16 is
**bandwidth-bound** (an 88 ms/step floor from 24 GB ÷ 273 GB/s) — and `brief-tooling` §5.2 says training is
compute-bound because "$I$ is in the hundreds." **They cannot both be right, and the corpus offers three rival
ridge points (3,663 / 458 / —) to adjudicate with.**

**RULING. One formula, frozen in `constants.md` §6.5:**

$$I \;=\; \frac{\text{FLOPs}}{\text{bytes}} \;=\; \frac{2N\cdot S_{\text{fwd}}}{2N} \;=\; \boxed{\,S_{\text{fwd}}\ \text{FLOP/byte}\,}\qquad\text{(dense, bf16)}$$

**Arithmetic intensity equals tokens per forward pass.** Ridge $I^\star = 62\text{e}12/273\text{e}9 = \mathbf{227}$
(working value) — **or 458 if the BF16 ceiling is ~125 TF, which is now a live, unresolved possibility (§6.4,
D-16 item 4). The opposite-verdicts contrast holds under both**; only the printed ridge number turns on the ceiling.

| Workload | $S_{\text{fwd}}$ | $I$ | vs 227 | Verdict |
|---|---|---|---|---|
| **LLM decode, batch 1** | 1 | **1** | **227× below** | bandwidth-bound. **Spark weak.** |
| LLM decode, batch 32 | 32 | 32 | 7× below | still bandwidth-bound |
| LLM prefill, S=2048 | 2048 | 2048 | 9× above | compute-bound |
| **Diffusion, 4096 latent tokens** | 4096 | **4096** | **18× above** | compute-bound. **Spark strong.** |

> **Same machine. Same ridge. Opposite verdicts. Because the two tracks sit on opposite sides of $I^\star$.**
> **That is the contrast, and it is one formula rather than two stories.**

**This adjudicates the contradiction: `brief-diffusion` §8.4 is wrong.** A bandwidth *floor* of 88 ms is not a
bound if compute needs 1,585 ms (= $2\times12\text{e}9\times4096 \div 62\text{e}12$). **Diffusion at bf16 is
compute-bound, comfortably.** The 88 ms figure is a floor nobody is near. **`brief-tooling` §5.2 is directionally
right** (though its "$I \approx 2$ for decode" is off by 2× — it is **1** at bf16).

**And the measured data confirms the formula on his own machine [VP]:** decode 20.5 → 368 tok/s (**18×**) going
batch 1→32, while **prefill is flat** (7,991 → 7,949). *Prefill was already compute-bound and gained nothing;
decode was bandwidth-bound and gained ~linearly with $I$.* **The roofline, measured, on the box he owns.**

**How to engineer it in the course.** The trunk's hardware page derives $I \approx S_{\text{fwd}}$ and the ridge,
**once**. Then each track's hardware page plugs in **one number** and gets its verdict — and the two verdicts are
opposite. **The learner should be able to derive his track's conclusion before being told it.** The
batch-1→batch-32 measurement is the confirmation, and it is already verified.

**Confidence: HIGH** on the formula and the ridge. **MEDIUM** on the ~62 TF that sets the ridge — it is [INF].
**Which is precisely why `constants.md` §10 makes measuring it exercise #2 and the course's best lab.**

---

### D-16 ⛔ THE FLUX BENCHMARK IS THE WRONG MODEL — a showpiece is invalid

**Conflict.** *Neither verifier's report connected this to the briefs, though the specs pass supplied the fact.*
Both `brief-diffusion` §8.4 and `brief-tooling` §3/§5.3 quote **"FLUX.1 12B @ FP4 → 23 img/min, 2.6 s per 1K
image"** [VP] and treat it as **FLUX.1-dev at 28 steps.** `brief-diffusion` §8.4 builds a showpiece on it — *"it
predicts a published benchmark from first principles… the architect should feature it."*

**The specs verifier confirms: the benchmark is FLUX.1-*Schnell*, 4 steps, 1024², batch 1. Not dev.**

**The arithmetic proves it independently and the course should show the check:**
$$28\ \text{steps} \times 2 \times 12\text{e}9 \times 4096 = 2.75\times10^{15}\ \text{FLOP} \;\div\; 2.6\ \text{s} = \mathbf{1.06\ \text{PFLOP/s}}$$
— **exactly the sparse-FP4 marketing peak, i.e. 100% MFU on a sparsity mode the model does not use.**
**Impossible.** At 4 steps it is ~150 TF ≈ 30% of the 500 TF dense-FP4 ceiling — **plausible.**

**RULING.**
1. **Label it: FLUX.1-Schnell, 4 steps, 1024², batch 1, FP4.** Everywhere it appears.
2. **`brief-diffusion` §8.4's "we predicted a published benchmark" showpiece is INVALID and must be DELETED, not
   rebuilt as a prediction** (Task 3 resolution, confirmed). You cannot honestly predict a 2.6 s wall-clock when
   the FP4 dense ceiling is itself an unpublished inference (±factor 2). The honest FLOP count for the *correct*
   variant (Schnell, 4 steps) is 4.98e14 FLOP → **~150–191 TF/s = 30–38% MFU** against the ~500 TF dense-FP4
   ceiling — plausible, versus the dev/28 reading's impossible 1.06 PFLOP/s (106% of the sparse-FP4 marketing peak).
3. **The honest replacement: `bench_spark.py`.** He runs FLUX.1-dev at 28 steps himself, at bf16 and FP4, and
   compares to *both* rooflines. **The prediction he checks is his own.**
4. **SDXL 1.0 @ BF16 = 7 img/min = 8.571 s/img** [VP] — **50 steps, batch 2, TensorRT; CFG not stated.** Label or
   measure. **NEW cross-check (Task 3):** the SDXL UNet is 5.977 TFLOP/forward @1024², batch 1 [DER,
   `FlopCounterMode`]. If CFG was on, sustained BF16 = **69.7 TF/s**, which **exceeds the inferred 62.5 TF
   FP32-accumulate ceiling (111%)** and leans toward the ~125 TF FP16-accumulate figure. See the D-15/§6.4 update:
   the ridge (227 vs 458) turns on this, so **62 vs 125 TF is now flagged unresolved [INF]** and settled by
   exercise #2. The qualitative "opposite verdicts" contrast holds under either ceiling.

**Reasoning.** The specs verifier flagged Flux-23-img/min as **"the highest-risk trust number in the set, higher
than the bandwidth figure, because he *will* check it against a machine he already owns."** He runs ComfyUI. If
he benchmarks FLUX **dev** at 20–50 steps he gets ~5–10× worse and concludes the course lied — **and the course
would have lied, in its own showpiece, in the one section that claims to prove theory predicts reality.**

**Confidence: HIGH** that the label is wrong. **HIGH** that the showpiece must go.

---

### D-17 THE TRACKS CONVERGE — exploit it, structurally

**Conflict/opportunity.** `brief-diffusion` §9.2: **FLUX.2's text encoder is Mistral-3, a 24B VLM — twice the
size of the entire FLUX.1 model, just to read the prompt.** The brief calls this *"the strongest possible
argument for why the trunk is shared"* and asks for *"an explicit structural rejoining moment."*
`brief-architectures` §9 independently arrives at the same place from the other side: *"encoder-decoder isn't
dead — it moved to diffusion. A text-to-image model IS an encoder-decoder."*

**RULING. Exploit it, three times, at three scales:**

1. **In the trunk, as foreshadowing** (~p.12): *"hold on to this — the only thing that will change is what we ask
   it to predict."*
2. **At the fork, as the reveal.** `brief-pedagogy` §10.1's framing, ratified verbatim: *"The trunk gave you a
   machine that maps vectors to vectors and learns by gradient descent. Point it at 'predict the next token' and
   you get an LLM. Point the same machine at 'predict the noise that was added' and you get a diffusion model.
   **The machine didn't change. The target did.**"* Same blocks, same attention, same AdamW, same LoRA —
   **different loss function.** That is true, it is deep, and it converts a structural inconvenience (two tracks)
   into the payoff for everything before it.
3. **At the end, as the rejoining — and this is the beat the corpus discovered and nobody planned for.**
   The final diffusion page: *"In 2022 these were two fields. In 2026, FLUX.2's text encoder is a Mistral — and
   the LLM track you finished is the manual for the front half of the diffusion model you just fine-tuned. The
   branches grew back together."*

**Reasoning.** It is true, it is current, it is verifiable, and it retroactively justifies the course's own
architecture. It also converts the course's biggest structural risk — *"two shorter courses stapled together"* —
into its thesis.

**And it decides the track order.** `brief-pedagogy` §10.3 recommends **LLM first**, *not* because it matters
more but because **his diffusion intuition is his strongest asset**, and reading the LLM track first means he
arrives at diffusion with a fresh formal frame to reinterpret things he already does by hand. **Maximum "oh,
*that's* what CFG was doing."** Diffusion-first squanders it. **Ratified: LLM track first.** D-17's rejoining
beat only works in this order — the Mistral reveal needs the LLM track to already be in his hands.

**⚠️ And keep the honest divergence, which is at least as interesting.** `brief-diffusion` §10.2: **the 2026
story diverges between the tracks — LLMs are still scaling; image models are consolidating** (Z-Image 6B > FLUX.2
32B on the arena; klein-4B at 4 steps in 13 GB). **Draw this deliberately, at the same rejoining moment.** The
tracks converged *architecturally* and diverged *economically*, in the same year. That is a real observation and
it is better than a tidy "everything is one thing."

**Confidence: HIGH.**

---

### D-18 "FINE-TUNING" MEANS THREE DIFFERENT THINGS

**Conflict.** `brief-llm-finetuning` §16: *"The word 'fine-tuning' means different things to a ComfyUI user
(mostly: LoRA/DreamBooth on a subject) than in the LLM world (SFT + preference alignment). Disambiguate early,
in the trunk."* And `brief-diffusion` §11.0/§11.6 shows it is worse than a two-way split: **ControlNet and
IP-Adapter are not fine-tuning at all**, and *"DreamBooth vs LoRA"* is a **category error** that half the
internet makes.

**RULING. A trunk page: "Fine-tuning' names three different things. Here is the type signature."**
`brief-diffusion` §11.0's table, generalized to both tracks and ratified as the course's taxonomy:

| Axis | Question | Answers |
|---|---|---|
| **WHERE does the change go?** | | all weights (full FT) · a low-rank correction (**LoRA**) · the vocabulary (**textual inversion**, 3 KB) · **nowhere — you change the input** (ControlNet, IP-Adapter, RAG, the prompt) |
| **WHAT are you teaching?** | | **form** (style, format, idiom, behaviour) vs **facts** (→ **you cannot; go to RAG**) |
| **WHICH recipe?** | | SFT · DreamBooth (rare token + prior preservation) · DPO/GRPO · continued pretraining |

**The test that collapses the whole confusion, ratified from `brief-diffusion` §11.6 and worth its own box:**

> ### **After you're done, is the base model different?**
> - **LoRA / DreamBooth / full FT / SFT / DPO** → **Yes.** $\theta$ changed.
> - **ControlNet / IP-Adapter / RAG / a better prompt** → **No.** The base model is bit-identical. You bolted an
>   extra input onto it.

**And the corollary that makes it stick, because he has observed it:** *you **train** a LoRA on your images in 20
minutes with 20 photos. You **download** a ControlNet — it took someone 100k+ images and a datacenter, but it
works on any image forever, because it isn't about content, it's about a kind of control.* **That is why there
are 500,000 LoRAs on Civitai and about 15 useful ControlNets.** The ratio *is* the distinction, made quantitative.

**Then the punchline that makes his messy node graph legible:** *you can stack all of them at once — a LoRA edits
$\theta$, the ControlNet and the IP-Adapter and the prompt all edit $\mathbf{c}$. **They don't collide because
they aren't the same kind of object.*** When he sees that his ComfyUI graph has a **clean type signature**, the
distinction is permanent.

**"DreamBooth vs LoRA" is a category error.** DreamBooth is *what you train* (a recipe). LoRA is *how you store
the change* (a parameterization). **Modern practice is both, simultaneously. Point at the filename:
`train_dreambooth_lora_flux.py`.**

**Confidence: HIGH.**

---

### D-19 THE ORDERING FORKS — ruled

| Fork | Brief positions | **RULING** | Reasoning |
|---|---|---|---|
| Regularization before or after optimizers? | training §1 recommends **after**, non-standard | **After.** | His felt pain is *"my LoRA diverged / my loss is NaN,"* not *"my model overfits MNIST."* Front-load the mechanics of training working; then generalization. |
| Activations before or after linear collapse? | foundations §13: **collapse first**, its "strongest ordering recommendation" | **Collapse first.** | The standard order presents ReLU/GELU as a *menu of options* before establishing that a nonlinearity is *mandatory*. Collapse first makes activations feel **necessary** rather than decorative. |
| Score/SDE before or after DDPM? | diffusion §17.3: **after**, "a real fork, decide once" | **After (DDPM first).** | (a) He gets a concrete 6-line algorithm to hold before the abstraction. (b) §5.2's *"we already trained a score model and didn't know it"* reveal is **only available in this order** and it's one of the best moments in the track. Consistent with the trunk's run→derive→break rhythm. |
| Quantization before or after LoRA? | llm-ft §1: genuinely ambiguous; recommends **splitting** | **Split.** Number formats in the trunk (before LoRA, since QLoRA needs NF4); **deployment** formats at serving. | Teaching it as one 6-page block reads as a datasheet. |
| The decision (prompt/RAG/FT) before or after the machinery? | llm-ft §1: **before**, non-negotiable | **Before.** | *"If the learner arrives at LoRA before he's been forced to ask 'should I be doing this at all?', the course has failed at the exact place the field fails."* Ratified. |
| Reparameterization trick early? | foundations §13: in the probability section, non-standard | **Yes, early.** | Cheap there; makes the diffusion forward process a **recognition** rather than a derivation. **The single highest-value handoff in the course.** |
| Track order | pedagogy §10.3: **LLM first** | **LLM first.** | D-17. |
| Page-6 code vs prerequisite chains | pedagogy §15.7: *"§4.1 wins"* | **§4.1 wins.** | Prerequisites can be collapsibles and just-in-time refreshers. **The Early Real Thing cannot be deferred without changing what kind of course this is.** |
| Vocabulary quizzes | pedagogy §8.1: **forbidden** | **Forbidden.** | Transfer-appropriate processing: learners who practice facts and are tested on higher-order questions score **indistinguishably from those who never practiced at all.** Vocabulary quizzes are not merely low-value — they are **budget-consuming with zero transfer.** |

---

### D-20 CONTENT NOW INVALIDATED — the demolition list

Everything a fan-out agent might otherwise faithfully reproduce from a brief, and must not:

| Brief | Section | Status |
|---|---|---|
| foundations | §2 (FFN example), §8 (11.76), §11 (2/3 rule), §13 | **Rewrite for Qwen3-8B.** D-01 |
| foundations | §2's "96 GB AdamW on 8B" | **Wrong. 131.05 GB.** D-03 |
| foundations | §4's diffusion LR | **100× wrong for LoRA.** D-09 |
| foundations | §0/§14's SVD cut | **Overruled — trunk page.** D-12 |
| foundations | §8's 4-way logits | **Retired.** D-08 |
| foundations | §6 "~31 TFLOPS FP32" | **Unsourced. Delete.** D-10c |
| training | §5.4's network | **Retired.** D-02 |
| training | §7.6's 7B table, 130 GB, 224×, "40% is Adam's m,v" (it's 43%) | **All retired.** D-01, D-03, D-05 |
| training | §16's numbers table | **~8 rows retired.** |
| training | §18's mandated callback "130 GB → 14.3 GB" | **Would propagate the error into the next brief. Rewrite.** D-03 |
| architectures | §10's "misses by about 3 GB", §12.1 | **Unit artifact.** D-04 |
| architectures | §11's "~380 years, generously assume sustained" | **Inverted. It's a lower bound at 100% MFU.** constants §5 |
| architectures | §12.6's "someone must verify 273 GB/s" | **Discharged. [VP].** D-10a |
| architectures | §3, §F2b's $\theta$ for the geometric angle | **$\vartheta$.** notation §5.4 |
| llm-finetuning | §8's "3 GB over", §5, §14, artifact 05 | **Unit artifact.** D-04 |
| llm-finetuning | §9's "check the base count… closes to 3 s.f." | **Add the norms; it closes to the byte.** D-07 |
| llm-finetuning | §9's 70B QLoRA (280M/4.5/41) | **207,093,760 / 3.31 / 39.7.** Used the hand-wave % its own demo spec forbids. |
| llm-finetuning | §9's "~20× QLoRA" | **19–26×.** |
| llm-finetuning | §14's "four measurements at 60–69% MBU" | **Heuristic SURVIVES: all four dense points land 60–75% under decode-traffic bytes; use real NVFP4 sizes 6.03/10.54 GB.** D-10e, Task 1 |
| llm-finetuning | §14's QLoRA "0.53 B/param" | **0.516 (4.127 bits) [VP]; 0.53 is an arithmetic error.** §Z-4, constants §3 |
| llm-finetuning | §15.8's [U] list | **Mostly discharged** by the specs pass. |
| pedagogy | §9.2's six roundings | **Corrected.** constants §8 |
| pedagogy | §9.2's "exactly zero" | **False.** Mandated framing. D-13 |
| pedagogy | §7.3's 3,663 FLOP/byte and ~19.5 tok/s | **Retired.** D-10d, constants §6.7 |
| pedagogy | §6.4's $\sigma$/$s_t$ rule | **Overruled.** notation §5 |
| pedagogy | §6.3's index table | **Missed the optimizer-step $t$.** notation §3 |
| pedagogy | Demo C's 7B/112 GB/14 GB | **Retired anchor + wrong ledger.** |
| tooling | §1's trl 0.29.1 and §4.3's "prefer Trainer" | **trl 1.8; use SFTTrainer.** D-10f |
| tooling | §4.1's v5 breaking-change table | **Several rows wrong.** D-10g |
| tooling | §5.2's 458 FLOP/byte, "Ferrari with a garden hose", $I\approx2$ | **Ferrari retired; $I=1$. But 458 is NOT retired — it is the correct ridge if the ceiling is ~125 TF, now flagged unresolved (227 working).** D-10d/l, D-15, Task 3 |
| tooling | §6.5's QLoRA 0.127 bits/param | **CORRECT — promote [UNVERIFIED]→[VP].** The corpus was internally inconsistent; tooling had it right. §Z-4 |
| tooling | §5.4's PTX-JIT story | **No PTX in release wheels.** D-10h |
| tooling | §6.3's Examples A–D | **Recompute on Qwen3-8B.** D-01 |
| tooling | §6.3 Example C's "430×" | **187×.** D-05 |
| tooling | §11.1's "no consensus on rank" | **Overruled — the information-content rule.** D-11 |
| tooling | §4.2's "attn vs MLP is a trade" | **Overruled — `all-linear`.** D-06 |
| diffusion | §8.4's benchmark showpiece | **INVALID — Schnell/4-step. DELETE, don't rebuild as a prediction → `bench_spark.py`.** D-16, Task 3 |
| diffusion | §8.4's "bf16 is bandwidth-bound" | **Compute-bound.** D-15 |
| diffusion | §2.5's FLUX.2 VAE "$f{=}8$, 32ch — unverified" | **VERIFIED [VP]: raise confidence.** Shipped `vae/config.json` + `AutoencoderKLFlux2`. §Z-7 |
| diffusion | §15.3's 53,657 tok/s | **6,969.59.** D-10m |
| diffusion | §6.2's "FM's $t$ is opposite to DDPM's" | **Both have data at $t=0$. Verify or delete.** notation §6 |
| diffusion | §15.2's "128 GB" verdicts | **Measure.** D-04 |
| **ALL** | any un-tagged number | **Must carry a `constants.md` confidence tag.** notation §9 #23 |

---

### D-21 THE COURSE IS 64 PAGES (+ GRPO ≈ 65) — project-lead decision, final

**The course is 64 pages, not 48/50.** The arbiter proposed six merges to fit a ~50 budget (see the retired §Z-5
concern); **the lead REJECTED all six.** The full 64-page outline stands. **Nothing is compressed** — not the
CNN section (stays ~3.5 pages, D-21 confirms the intended cost), not foundations, not the diffusion track.

**The GRPO/RLVR capstone is IN.** vLLM is unblocked on sm_121 via the cu130 container (D-10j), which was the only
gate. It is an **additional capstone page in the LLM track's landing zone**, bringing the course to **~65 pages**.
It uses generation-in-the-loop, so it may lean on Qwen3-0.6B for speed (D-01's smoke-test model), with the
arithmetic transferring to the 8B anchor by substituting four config numbers.

**Consequences for builders:** any brief text that assumed a ~50-page budget or a merge is void. The page
allocation is generous; depth is the priority the lead chose. **Confidence: HIGH — this is the lead's call, recorded.**

---

#### D-21a — THE CANONICAL PAGE TABLE (ratified with D-21; persisted 2026-07-16 after the spec fan-out
#### reconstructed it divergently because this table lived only in the arbiter's return, not in this file)

**This is the outline the arbiter produced and the lead ratified. CONTENT ALLOCATION IS FROZEN.** Titles may be
paraphrased; what each page *owns* may not move without a new ruling. Numbering: arbiter pages 1–51 unchanged;
**GRPO capstone = 52**; arbiter 52–61 (diffusion) → **53–62**; arbiter 62–64 (capstone) → **63–65**.
★ = spine beat. Build model: O = Opus, S = Sonnet (27 O / 21 S ratio from the arbiter, GRPO = O).

**PART I — THE MACHINE (1–11), trunk**
| # | Owns | Build |
|---|---|---|
| 1 | The neuron, dragged — 3 sliders, a decision boundary, NO notation | S |
| 2 | Shape is the skill — tensors, matmul rule, broadcasting, the silent `(64,)`/`(64,1)` bug | S |
| 3 | The dot product — similarity/projection; near-orthogonality in high-D. **Trig payoff #1** (pays again on 33, 54) | O |
| 4 | Derivatives and the θ-vs-x flip — slope→sensitivity; gradient=steepest ascent proved FROM p.3 | O |
| 5 | ★ The chain rule — multiply along paths, add across paths; cached activations → memory | O |
| 6 | ★★ TN-1 by hand → PyTorch — forward to 0.3727 (constants §8.1), then 15 matching lines. NO backprop yet | O |
| 7 | Row vs column — the `(out,in)` invariant, `nn.Linear` | S |
| 8 | **Probability, the Gaussian, μ+σε — reparameterization EARLY. ★ THE handoff to diffusion (53–57 depend on it)** | O |
| 9 | Logs, logsumexp, softmax, CE = NLL — underflow felt first; `[2.0,1.0,0.1]` → 0.417030; ∂L/∂z = p−y | O |
| 10 | The neuron, XOR, linear collapse — hyperplane; XOR by hand; collapse BEFORE activations | O |
| 11 | Activations + the MLP's honest limits — why ReLU won (0.25³² underflow); SwiGLU; UAT's broken promises | S |

**PART II — HOW IT LEARNS (12–24), trunk.** Backprop lives HERE and only here — Part I teaches forward + chain
rule only. (Resolves the spec integrator's escalation: ONE backprop/autograd home, pages 14–15.)
| # | Owns | Build |
|---|---|---|
| 12 | Loss from maximum likelihood — MSE and CE from ONE principle; prepays both tracks' losses | O |
| 13 | Gradient descent and the LR — η_crit = 2/λ_max on a quadratic; saddles-not-local-minima | O |
| 14 | ★★ Backprop: TN-1's nine gradients by pencil — constants §8.2 + §8.7 (second input, spread 10.22×) | O |
| 15 | ★ Autograd + the float that isn't zero — `.backward()` reproduces p.14; **torch prints 1.402e-08 and both are right** (§8.4 home) | O |
| 16 | Minibatch, noise, accumulation — 1/√B; noise as feature; grad-accum HERE | S |
| 17 | Optimizers SGD→momentum→Adam→AdamW — five fixes; the 10.22× spread → "therefore Adam" | O |
| 18 | ★★ The memory ledger: 16 B/param — predict-then-MEASURE (constants §6.8 [MEA-DEV]); LR ∝ 1/trainable | O |
| 19 | Init, vanishing/exploding, clipping — 1.1¹⁰⁰ vs 0.9¹⁰⁰; residuals-as-identity-term (trunk owns WHY) | O |
| 20 | Normalization and pre-norm — the axis question; one bracket decides if 80 layers train | O |
| 21 | Schedules and why warmup exists — warmup is an ADAM pathology (he computed β₂'s memory on p.17) | S |
| 22 | Regularization and the two curves — six shapes, six diagnoses; written for 500 examples | S |
| 23 | Bias-variance and double descent, honestly — the λ slider that dissolves the phenomenon | S |
| 24 | Debugging: twelve failures, twelve signatures — overfit one batch; determinism bisects. (The "first real
fine-tune" milestone beat the spec fan-out added here is APPROVED as the page's closer — additive, not a merge) | S |

**PART III — ARCHITECTURE (25–37), trunk**
| # | Owns | Build |
|---|---|---|
| 25 | Architecture is inductive bias | S |
| 26 | CNNs, sized for the U-Net — kernel/stride/RF; 111-layers-vs-12; equivariance ≠ invariance | O |
| 27 | Embeddings and high-D geometry — anisotropy; why RAG needs a second model | S |
| 28 | RNNs and why they lost — 0.5 vs 1000 FLOP/byte; parallelism, not smartness | S |
| 29 | ★ Attention I: Q/K/V as a soft lookup — MASK-AGNOSTIC (D-ruling; diffusion depends on it) | O |
| 30 | ★ Attention II: √d_head = 11.31, multi-head — gradient death, not overflow; QK-Norm | O |
| 31 | ★★ Self vs cross-attention — `attn(Q=x,K=y,V=y)`. **The diffusion hinge.** Own page | O |
| 32 | **Causal masking — 4096 signals per pass. The LLM hinge. OWN PAGE (the fan-out folded it into 31 — restored)** | S |
| 33 | Positional encoding → RoPE — rotations compose. **Trig payoff #2** | O |
| 34 | Residuals and the residual stream — the shared bus (architectures owns WHERE; p.19 owned WHY) | O |
| 35 | The transformer block, assembled — six "why is this line here?" | O |
| 36 | ★★ Counting Qwen3-8B — 8,190,735,360, twice, independently | O |
| 37 | O(S²), FlashAttention, GQA, KV cache — 144 KiB/token; the crossover | O |

**PART IV — ADAPTATION (38–43), shared, before the fork**
| # | Owns | Build |
|---|---|---|
| 38 | Scaling laws, MoE, "emergence" — 6ND; ≥4 centuries at peak; the metric artifact. **(NOT an "adaptation
decision" page — the prompt/RAG/fine-tune decision lives at 47 and ONLY 47)** | S |
| 39 | ★ Rank, SVD, low-rank — the gap foundations cut; W₀ full-rank, ΔW isn't | O |
| 40 | ★★ LoRA derived — W₀+(α/r)BA; B=0; 187× = the parameter ratio; rank ∝ dataset information | O |
| 41 | Number formats: bf16→FP8→NF4→NVFP4 — QLoRA 0.516 B/param [VP]; granularity > mantissa | O |
| 42 | ★ "Fine-tuning" names three things — is the base model different afterward? | S |
| 43 | ★★ Your box, measured — roofline; ridge 227 [INF] / 458 alt, UNRESOLVED, taught honestly; `measure_your_box.py` | O |

**FORK — LLM first (his diffusion intuition is the asset; spend it last).**

**PART V — TRACK A: LLMs (44–52)** — 44 tokenization S · 45 next-token/base-vs-instruct S · 46 decoding+KV cache
(repaired heuristic; MoE punchline) O · **47 prompt vs RAG vs fine-tune — FORM vs FACTS, the ONLY decision page** O ·
48 post-training SFT→DPO→GRPO/RLVR S · 49 ledger-applied + fresh-venv setup (hardware-ground-truth §3) O ·
50 LoRA applied (his 9 ComfyUI LoRAs open it) O · 51 datasets/eval/forgetting + serving S · **52 GRPO/RLVR capstone** O

**PART VI — TRACK B: DIFFUSION (53–62)** — 53 VAE and its ceiling O · 54 forward process (**trig payoff #3**,
terminal-SNR) O · 55 ★★ reverse + ELBO collapse (1000 KLs → MSELoss; ε/x₀/v triangle) O · 56 score + SDE O ·
57 ★ flow matching = the chord (his own FlowMatchEuler scheduler = exhibit A) O · 58 U-Net → DiT → MMDiT O ·
59 latent diffusion 65,536× + samplers O · 60 ★★ CFG derived — extrapolation; FLUX `guidance` ≠ `cfg` O ·
61 LoRA applied + datasets (caption what you want to vary) O · 62 eval, why his eyes win S
*(The fan-out's internal ordering of 56–60 differed page-by-page from this; EITHER internal order is acceptable
provided every listed owner-topic keeps a full page — flag the chosen order in spec-master §7.)*

**PART VII — CAPSTONE (63–65)** — 63 merge/serve/measure: predict-your-box console S→O ok ·
64 the tracks grew back together (FLUX.2's Mistral-3 encoder, verified from his disk) S→O ok · 65 choose the
method for a new problem / where next S

**Fidelity defects in the 2026-07-16 spec fan-out, to be repaired against this table:**
1. Part I pages 8–11 were replaced with backprop/autograd/one-step (duplicating 14–15) — **the probability/
   Gaussian/reparameterization page (8), logs/logsumexp/softmax/CE page (9), XOR/collapse page (10), and
   activations/MLP-limits page (11) must be restored.** Reparameterization currently appears NOWHERE in the
   trunk while the diffusion track cites it 5×.
2. Part II merged regularization+double-descent (a lead-REJECTED cut) and split init/vanishing — restore 19–23
   per this table.
3. Part III folded causal masking into 31 and pulled O(S²)/KV forward to 32 — restore 32 and 37 per this table.
4. Part IV invented an "adaptation decision" page at 38, duplicating 47, and displaced scaling/MoE — restore.

---

## PART Z — ⚠️ RESOLVED BY THE PROJECT LEAD + TWO RESOLUTION PASSES (2026-07-16)

**All eight items are now CLOSED.** Recorded here as the audit trail; the reasoning is kept. Where a resolution
pass computed or verified something, the script/source is named. **Nothing in §Z is open.**

1. **[P0 → CLOSED] TN-1's second input — remedy (a) ADOPTED and computed.** `x=[0.60,−0.20]` on TN-1's weights
   is frozen as the second canonical case in `constants.md` §8.7. **The nine gradients ARE now computed** — three
   independent methods agree (mpmath 50 d.p. analytic ↔ torch autograd f64, max Δ 5.3e-17 ↔ central FD, max Δ
   1.5e-31). **Gradient-spread beat recovered: 10.22× (all nine live, no dead unit)**, replacing the retired
   12.7×. The multiplicative story (3:1 feature ratio × δ-ratio × layer term) is transparent, and the spread is
   *structural* — 8.64× even at a 1:1 feature ratio — so "therefore Adam" survives the "just normalize" objection.
   ⚠️ **A seventh mis-rounding surfaced:** the old "first-input spread 3.09×" was wrong (max taken over the
   intermediate $|dz_2|$, not the nine parameter gradients); the true first-input figure is **3.7119×**, but the
   beat uses the second input regardless. The second-input assertion is frozen at **5 d.p. with `atol=1e-5`** (4
   d.p. gives only a 2.1× margin — the same tolerance-coincidence trap §8.5 warns against). Unblocks the backprop
   page and Demo 2.
2. **[P0 → CLOSED] The Spark budget beat — CONFIRMED measured.** The lead ratifies D-04: the memory beat is
   MEASURED (`mem_get_info()` first), not an asserted cliffhanger. The "131-vs-128 wall" was a GB/GiB artifact;
   **122.05 GiB actually fits in 128 GiB with room to spare.** The course arrives at "does it fit?" by measurement.
3. **[P1 → CLOSED] The $\sigma$ overrule — RATIFIED as written.** The lead confirms the $\sigma(\cdot)$ vs
   $\sigma_t$ disambiguation (notation §5) as ruled; do not revisit. Two track-scoped rules stand.
4. **[P1 → CLOSED] QLoRA's effective bits — RESOLVED against the paper. 0.516 B/param [VP].** `constants.md` §3.
   The 0.53 is **REFUTED as an arithmetic error**, not a rival convention: the brief divided the second-level
   fp32 constant by 256 (scales) instead of 64×256 = 16,384 (parameters those scales cover). Total = 4 + 65/512 =
   **4.127 bits = 0.515869 B/param**; overhead 0.127 bits and DQ-saving 3.03 GB/65B both reproduce arXiv
   2305.14314's own published figures exactly. `brief-llm-finetuning.md:750` fixed; `brief-tooling-hardware.md:650`
   already had 0.127 correct (promoted to [VP]). **New caveat:** double-quant defaults **False**; un-opted-in NF4
   is 4.5 bits = 0.5625 B/param (4.607 GB for Qwen3-8B) — the 4.225 GB figure assumes `bnb_4bit_use_double_quant=True`.
5. **[P2 → CLOSED] Page budget — 64 pages, NO merges.** The lead REJECTED all six proposed merges. The full
   64-page outline stands; nothing is compressed. **Plus the GRPO/RLVR capstone (item 8) → ~65 pages.** The
   arbiter's ~50 budget is void.
6. **[P2 → CLOSED] Diffusion U-Net-first depth — CONFIRMED.** U-Net taught first and in depth, DiT/MMDiT
   following. `brief-architectures`' CNN section **stays at ~3.5 pages**; the lead confirms this is the intended
   cost.
7. **[P2 → CLOSED] FLUX.2 VAE spatial factor — VERIFIED [VP].** $f=8$ (`block_out_channels` length 4 →
   $2^{4-1}$), `latent_channels: 32`, latent $[1,32,128,128]$ = 524,288 elements → 6.0× compression, 4096 tokens
   after 2×2 patchify (width 128). Read from the shipped `vae/config.json` of two ungated repos and
   `AutoencoderKLFlux2.__init__` in diffusers `main`. **The DeepWiki "32:1 / [B,128,16,16]" claim is wrong** — it
   reported VAE ∘ patchify, not the raw VAE factor. `constants.md` §9.6, `brief-diffusion` §2.5 confidence raised.
8. **[P3 → CLOSED] GRPO/RLVR capstone — IN.** vLLM is unblocked on sm_121 via the cu130 container (D-10j). It is
   an **additional capstone page** in the LLM track's landing zone, bringing the course to **~65 pages**. Recorded.
