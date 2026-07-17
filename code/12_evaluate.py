#!/usr/bin/env python3
r"""
12_evaluate.py - did the fine-tune actually help? The evaluation staircase. p.51.

The whole page in one line (the key box, printed verbatim below):

    Loss going down means the model is getting better at predicting YOUR DATASET.
    Your dataset is not your goal.

A held-out loss curve that falls beautifully only tells you the model isn't
memorizing garbage. It says NOTHING about whether the outputs are useful. So this
script climbs the p.51 staircase, cheapest rung first, and stops when a rung
answers your question:

  1. Held-out loss + perplexity + mean_token_accuracy   [free - "am I overfitting?"]
  2. A programmatic TASK METRIC - does the SQL run / JSON validate / test pass?
                                                        [if you have one, the rest is optional]
  3. 50 eyeballed examples                              [~15 min - catches what a metric misses]
  4. LLM-as-judge, both orderings averaged, lengths reported   [cheap, biased - managed here]

...and, running in parallel with ALL of them for almost nothing:

  * the CATASTROPHIC-FORGETTING probe - 10-20 GENERAL prompts unrelated to your
    task, answered by the base and by the fine-tune, diffed by eye. Fifteen minutes
    catches the single most common way a fine-tune quietly destroys value while the
    loss curve looks perfect: it gets great at your task and worse at everything else.

Two things this script REFUSES to launder as fact (constants.md ships zero LLM-judge
bias magnitudes and zero frozen eval numbers for this course - notation §9 #23):

  * The LLM-as-judge win rate is [MEA] on YOUR judge and YOUR data, never a SEO
    "95%". The self-test's transparent scorer (weights 0.9 / 1.1 / 0.5, shared
    verbatim with the p.51 demo) is labelled ILLUSTRATIVE - it exists to prove the
    position-bias-cancels-under-averaging ARITHMETIC, not to stand in for a number.
  * The forgetting verdict is a diff you read, not a scalar this script invents.

The one hard structural guarantee (spec-part5 p.51 self-check): the judge is ALWAYS
run in BOTH orderings and averaged. Position bias - the answer shown first wins more
often, independent of quality - cancels exactly when every candidate is first in one
run and second in the other. A judge that ran one ordering is a broken judge; the
asserts fail loudly if only one ran.

The closed-loop trap (p.51 warn box): if you generated your synthetic training data
with model X and then judge with model X, you are not measuring quality - you are
measuring how much your fine-tune's outputs sound like X. This script warns EXPLICITLY
when the judge's model family matches the data-generator's family.

SAFETY
------
The real path loads your fine-tuned model, optionally the base (for the forgetting
probe), and a judge model - up to three checkpoints, but sequentially where it can,
never all three resident if memory is tight (~16 GB bf16 for an 8B). It contends with
ComfyUI if that is live on the GPU; consult before running. It writes nothing and
installs nothing (read-only w.r.t. the box); it only DOWNLOADS checkpoints into your
HF cache if absent. Set HF_HOME to the NVMe first (constants §7).

Env (THE fresh venv - never ComfyUI's; hardware-ground-truth §3):
    transformers 5.14.1 · torch 2.13.0 · peft 0.19.1 · CUDA 13.0
    (v5: load with dtype=, not torch_dtype; PEFT adapters via PeftModel.from_pretrained)

Usage
-----
    python 12_evaluate.py --self-test          # GPU-free: all eval arithmetic, run + asserted
    python 12_evaluate.py \                     # the real thing (needs a GPU)
        --model ./out/qwen3-8b-lora \           #   merged fine-tune OR a base+adapter (see --adapter)
        --base  Qwen/Qwen3-8B \                 #   for the forgetting probe (before vs after)
        --heldout heldout.jsonl \               #   held-out chat examples: {"messages":[user, assistant]}
        --judge Qwen/Qwen3-8B \                 #   LLM judge - use a DIFFERENT family than your data-gen!
        --generator-family qwen \               #   the family that made your SFT data (for the closed-loop warn)
        --task-metric contains                  #   contains | json | none  (a real checker beats a judge)
"""

import argparse
import json
import math
import sys

# --------------------------------------------------------------------------- #
# Frozen facts vs. illustrative weights - the distinction this whole file turns on.
# --------------------------------------------------------------------------- #

# FROZEN [constants.md]. The loss->perplexity anchor lets the self-test check its
# arithmetic against a real derived number: a random-init CE floor is ln V, and the
# perplexity there is exactly V (an untrained uniform softmax over V classes).
V = 151_936                         # Qwen3 vocab, constants §1.1 [VP]
LN_V = math.log(V)                  # random-init CE floor = 11.9312... nats [DER, §9.1]

# ILLUSTRATIVE [NOT constants]. The transparent judge's bias weights, shared VERBATIM
# with the p.51 demo so page and script cannot drift. They reproduce the DIRECTION and
# survival pattern the literature reports; they are not measured magnitudes. constants.md
# ships none, because there are none to freeze for this course (p.51 disclosed deviation).
Q0 = 5.0                            # baseline "quality" both candidates share
POS_BIAS = 0.9                      # bonus for being read FIRST (the bias we CANCEL)
VERB_BIAS = 1.1                     # coefficient on log2(words / concise_words)
MD_BIAS = 0.5                       # bonus for markdown formatting

IGNORE_INDEX = -100                 # HF ignore_index / the m_i == 0 mask (notation §4.5)

KEY_BOX = ("Loss going down means better at predicting YOUR DATASET; "
           "your dataset is not your goal.")

# Model-id -> family, for the closed-loop / self-preference warning. Substring match on
# the lowercased id. Extend freely; unknown ids return None (we warn we couldn't tell).
_FAMILIES = ("qwen", "llama", "mistral", "gemma", "phi", "deepseek", "gpt", "yi",
             "falcon", "mixtral", "command", "olmo", "granite")


# --------------------------------------------------------------------------- #
# Shared pure arithmetic - imported by NOTHING, exercised by the self-test AND used
# by the real path, so the numbers the page shows and the numbers the script asserts
# are computed by the same code.
# --------------------------------------------------------------------------- #

def perplexity(loss_nats):
    """PPL = e^{loss}. Loss must be a NAT-valued cross-entropy (constants §9.1)."""
    return math.exp(loss_nats)


def family_of(model_id):
    """Best-effort model family from an id/path. None if we can't tell."""
    if not model_id:
        return None
    low = model_id.lower()
    for fam in _FAMILIES:
        if fam in low:
            return fam
    return None


def judge_score(words, is_first, has_md, concise_words):
    """The TRANSPARENT illustrative judge from p.51, to the character:

        score = Q0 + POS_BIAS*first + VERB_BIAS*log2(words/concise_words) + MD_BIAS*hasMD

    Returns the total and the decomposed terms so the self-test can prove which bias
    survives averaging and which dies."""
    verb_term = VERB_BIAS * math.log2(words / concise_words)
    pos_term = POS_BIAS if is_first else 0.0
    md_term = MD_BIAS if has_md else 0.0
    total = Q0 + pos_term + verb_term + md_term
    return {"total": total, "pos": pos_term, "verb": verb_term, "md": md_term, "words": words}


def average_both_orderings(concise_words, padded_words, padded_has_md):
    """Run the transparent judge in BOTH orderings (concise-first, then padded-first)
    and average each candidate's two scores. This is the position-bias fix, in the
    smallest possible arithmetic. Returns (concise_avg, padded_avg, n_orderings)."""
    orderings = []
    # ordering 1: concise is read first
    orderings.append((
        judge_score(concise_words, True, False, concise_words),
        judge_score(padded_words, False, padded_has_md, concise_words),
    ))
    # ordering 2: padded is read first
    orderings.append((
        judge_score(concise_words, False, False, concise_words),
        judge_score(padded_words, True, padded_has_md, concise_words),
    ))
    c_avg = sum(o[0]["total"] for o in orderings) / len(orderings)
    p_avg = sum(o[1]["total"] for o in orderings) / len(orderings)
    return c_avg, p_avg, len(orderings)


def words_of(text):
    """Real word count, markdown structural characters stripped (matches the p.51 demo)."""
    cleaned = text.replace("#", " ").replace("*", " ").replace("_", " ")
    return len([w for w in cleaned.split() if w])


def forgetting_delta(before, after):
    """The diff-by-eye probe, made mechanical enough to flag. Returns a dict:
    changed (bool), word-count delta, and a crude Jaccard on the token sets so a big
    drop in overlap surfaces as "this general answer moved a lot after fine-tuning".
    The VERDICT is still yours to read - this only sorts what deserves your eyes."""
    b_words = before.split()
    a_words = after.split()
    b_set, a_set = set(b_words), set(a_words)
    inter = len(b_set & a_set)
    union = len(b_set | a_set) or 1
    jaccard = inter / union
    return {
        "changed": before.strip() != after.strip(),
        "len_before": len(b_words),
        "len_after": len(a_words),
        "len_delta": len(a_words) - len(b_words),
        "jaccard": jaccard,
    }


def task_pass(gold, generation, kind):
    """A programmatic task metric - the rung that makes everything below it optional.
    'contains': gold answer appears (case-insensitive) in the generation.
    'json'   : the generation parses as JSON (does the structured output validate?).
    'none'   : no metric available (returns None; the caller skips this rung).
    Swap in YOUR real checker (does the SQL run, does the unit test pass)."""
    if kind == "none":
        return None
    if kind == "contains":
        return gold.strip().lower() in generation.lower()
    if kind == "json":
        try:
            json.loads(generation)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    raise ValueError(f"unknown task-metric {kind!r} (use contains | json | none)")


# --------------------------------------------------------------------------- #
# Self-test - every rung's arithmetic, on synthetic inputs, WITHOUT a GPU.
# --------------------------------------------------------------------------- #

def self_test():
    print("=" * 72)
    print("SELF-TEST - the p.51 evaluation staircase, arithmetic only (no GPU)")
    print("=" * 72)
    print(f"  key box: \"{KEY_BOX}\"")
    print()

    # -- Rung 1: held-out loss -> perplexity, anchored to a FROZEN constant. -------- #
    print("  [1] held-out loss -> perplexity (anchored to constants §9.1)")
    assert abs(perplexity(0.0) - 1.0) < 1e-12, "PPL of a perfect (loss=0) fit must be 1"
    assert abs(perplexity(LN_V) - V) < 1e-3, "PPL at the random-init CE floor must equal V"
    assert round(LN_V, 2) == 11.93, f"ln V must round to 11.93, got {LN_V:.4f}"
    print(f"      PPL(loss=0)      = {perplexity(0.0):.4f}   (a certain fit; as it must be)")
    print(f"      PPL(loss=ln V)   = {perplexity(LN_V):,.0f}   = V exactly (untrained floor, §9.1)")
    print(f"      => loss is just log-perplexity; both answer only 'am I overfitting?'")
    print()

    # -- Rung 2: a programmatic task metric. ---------------------------------------- #
    print("  [2] programmatic task metric (the rung that makes the rest optional)")
    assert task_pass("Paris", "The capital is Paris.", "contains") is True
    assert task_pass("Paris", "I am not sure.", "contains") is False
    assert task_pass("", '{"ok": true}', "json") is True
    assert task_pass("", "not json at all", "json") is False
    assert task_pass("x", "anything", "none") is None
    print("      contains / json / none checkers all behave (a real checker beats a judge)")
    print()

    # -- Rung 4: LLM-as-judge - the position-bias cancellation, PROVEN. ------------- #
    print("  [4] LLM-as-judge - both orderings averaged (the position-bias fix)")
    concise_words = 42                       # a stand-in concise answer length
    # (a) IDENTICAL content, single ordering: the winner is PURE position bias.
    c_first = judge_score(concise_words, True, False, concise_words)
    p_second = judge_score(concise_words, False, False, concise_words)   # same words, read 2nd
    gap_single = c_first["total"] - p_second["total"]
    assert abs(gap_single - POS_BIAS) < 1e-12, \
        "on identical content, the single-order gap must be exactly the position bonus"
    print(f"      identical content, one ordering: first-read wins by {gap_single:.2f} "
          f"= POS_BIAS (pure position bias, nothing else)")

    # (b) IDENTICAL content, BOTH orderings averaged: the bias cancels to an honest tie.
    c_avg, p_avg, n_ord = average_both_orderings(concise_words, concise_words, False)
    assert n_ord == 2, "the judge MUST run both orderings (spec-part5 p.51 self-check)"
    assert abs(c_avg - p_avg) < 1e-12, "averaging both orderings must cancel position bias exactly"
    # each candidate now carries HALF the position bonus (first once, second once)
    assert abs(c_avg - (Q0 + POS_BIAS / 2)) < 1e-12, "each candidate should carry +POS_BIAS/2"
    print(f"      identical content, both orderings averaged: gap = {abs(c_avg - p_avg):.2f} "
          f"(honest tie); each carries +{POS_BIAS/2:.2f} (half the bonus) -> position CANCELS")

    # (c) PADDED content: averaging kills position but VERBOSITY survives untouched.
    padded_words = concise_words * 2         # 2x length, same substance
    c_avg2, p_avg2, n_ord2 = average_both_orderings(concise_words, padded_words, False)
    assert n_ord2 == 2, "the judge MUST run both orderings"
    surviving = p_avg2 - c_avg2
    expected_verb = VERB_BIAS * math.log2(padded_words / concise_words)  # = VERB_BIAS*log2(2) = 1.1
    assert abs(surviving - expected_verb) < 1e-12, \
        "after averaging, the ONLY remaining gap must be the verbosity term (position gone)"
    print(f"      padded 2x, both orderings averaged: padded still wins by {surviving:.2f} "
          f"= VERB_BIAS*log2(2) - position fixed, VERBOSITY untouched (report lengths!)")
    print(f"      weights (0.9 / 1.1 / 0.5) are ILLUSTRATIVE - shared with the p.51 demo,")
    print(f"      not measured. Your judge's REAL biases are what the real path measures.")
    print()

    # -- The closed-loop trap: judge family == data-generator family. --------------- #
    print("  [warn] closed-loop trap - judge family vs data-generator family")
    assert family_of("Qwen/Qwen3-8B") == "qwen"
    assert family_of("meta-llama/Llama-3.1-8B-Instruct") == "llama"
    assert family_of("./my-local-checkpoint") is None
    assert family_of("Qwen/Qwen3-8B") == family_of("Qwen/Qwen3-0.6B"), \
        "same family must be detected as a closed loop regardless of size"
    assert family_of("Qwen/Qwen3-8B") != family_of("mistralai/Mistral-7B"), \
        "different families must NOT trip the closed-loop warning"
    print("      qwen==qwen (would warn), qwen!=mistral (safe), unknown id -> can't tell")
    print()

    # -- The catastrophic-forgetting probe: flag what moved. ------------------------ #
    print("  [probe] catastrophic forgetting - diff general answers before/after")
    same = forgetting_delta("The capital of France is Paris.",
                            "The capital of France is Paris.")
    moved = forgetting_delta("The capital of France is Paris.",
                             "SELECT city FROM t WHERE country='France';")
    assert same["changed"] is False and abs(same["jaccard"] - 1.0) < 1e-9, \
        "an unchanged general answer must not be flagged"
    assert moved["changed"] is True and moved["jaccard"] < 0.3, \
        "a general answer that collapsed into task-speak must surface as low overlap"
    print(f"      unchanged answer: jaccard={same['jaccard']:.2f} (not flagged)")
    print(f"      collapsed answer: jaccard={moved['jaccard']:.2f}, len {moved['len_before']}"
          f"->{moved['len_after']} (FLAGGED - read this one by eye)")
    print()

    print("=" * 72)
    print("SELF-TEST OK. Every rung's math checks out; the judge ran BOTH orderings and")
    print("position bias cancelled to zero on identical content. Run without --self-test")
    print("on your box for held-out loss, your task metric, the forgetting probe, and a")
    print("REAL LLM-as-judge pass over your own model.")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# The real path - needs a GPU, torch, transformers, (optionally) peft. Desk-checked
# against the verified July-2026 transformers 5.x API (constants §7.1, mirrors the
# already-shipped 03_base_vs_instruct.py / 10_build_dataset.py usage):
#   - AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)   [v5: dtype=, not torch_dtype]
#   - PeftModel.from_pretrained(base, adapter_dir)                      [peft 0.19]
#   - model(**enc, labels=labels).loss                                  [teacher-forced CE]
#   - tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
#   - model.generate(..., do_sample=False)                             [greedy]
# --------------------------------------------------------------------------- #

def stamp():
    import torch
    line = f"  torch {torch.__version__} | cuda {torch.version.cuda}"
    try:
        import transformers
        line += f" | transformers {transformers.__version__}"
    except Exception:
        pass
    try:
        import peft
        line += f" | peft {peft.__version__}"
    except Exception:
        pass
    print(line)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"  device {torch.cuda.get_device_name(0)} | capability sm_{cap[0]}{cap[1]}")


def load_model(model_id, adapter, dtype):
    """Load a causal LM; attach a PEFT adapter if --adapter was given (base+adapter path)."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def render_prompt_and_target(tok, messages):
    """Split a chat example into (prompt_ids, full_ids). The prompt render (all but the
    final assistant turn, add_generation_prompt=True) MUST be a token prefix of the full
    render - the same discipline 10_build_dataset.py asserts. Labels = full with -100 on
    the prompt prefix, so held-out loss is over the ASSISTANT tokens only."""
    prompt_ids = tok.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=True)
    full_ids = tok.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True)
    # prefix guarantee: if a template quirk breaks it, fall back to masking min-length.
    prefix_len = len(prompt_ids)
    if full_ids[:prefix_len] != prompt_ids:
        # be conservative: mask nothing we can't prove is prompt (loss over all is a
        # pessimistic but honest fallback, and we say so).
        prefix_len = 0
    return full_ids, prefix_len


def heldout_loss(model, tok, examples):
    """Token-weighted held-out CE + perplexity + mean_token_accuracy over ASSISTANT
    tokens only. Returns (loss_nats, perplexity, token_accuracy, n_tokens)."""
    import torch
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for messages in examples:
        full_ids, prefix_len = render_prompt_and_target(tok, messages)
        ids = torch.tensor([full_ids], device=model.device)
        labels = ids.clone()
        labels[0, :prefix_len] = IGNORE_INDEX
        with torch.no_grad():
            out = model(input_ids=ids, labels=labels)
        # next-token loss is over positions [prefix_len-1 .. n-2] predicting [prefix_len .. n-1]
        logits = out.logits[0, :-1, :]
        tgt = labels[0, 1:]
        keep = tgt != IGNORE_INDEX
        n = int(keep.sum().item())
        if n == 0:
            continue
        # token-weighted mean: recompute summed CE so short/long examples weight by tokens
        ce = torch.nn.functional.cross_entropy(
            logits[keep].float(), tgt[keep], reduction="sum")
        total_loss += float(ce.item())
        preds = logits[keep].argmax(dim=-1)
        total_correct += int((preds == tgt[keep]).sum().item())
        total_tokens += n
    if total_tokens == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean_loss = total_loss / total_tokens
    return mean_loss, perplexity(mean_loss), total_correct / total_tokens, total_tokens


def generate_answer(model, tok, question, max_new_tokens=256):
    """Greedy chat generation of a single answer to `question`."""
    import torch
    msgs = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def judge_prefers(judge_model, judge_tok, question, ans_a, ans_b):
    """Ask the REAL judge which of A / B is the better answer. Returns 'A' or 'B'
    (defaults to a coin-free 'tie'->None on an unparseable reply). One ordering only;
    the caller runs this TWICE with A/B swapped and averages (position-bias fix)."""
    prompt = (
        f"You are grading two answers to a question. Reply with a single letter, "
        f"A or B, for the better answer. No explanation.\n\n"
        f"Question: {question}\n\n"
        f"Answer A: {ans_a}\n\n"
        f"Answer B: {ans_b}\n\n"
        f"Better answer (A or B):")
    reply = generate_answer(judge_model, judge_tok, prompt, max_new_tokens=4).upper()
    for ch in reply:
        if ch in ("A", "B"):
            return ch
    return None


def judge_both_orderings(judge_model, judge_tok, question, ft_answer, ref_answer):
    """Run the judge in BOTH orderings and average. ft is the fine-tune's answer, ref is
    the baseline/reference. Returns (ft_win_score, n_orderings) where ft_win_score is in
    {0.0, 0.5, 1.0}: 1.0 = ft wins regardless of position, 0.5 = position-dependent split
    (a wash - exactly what averaging is meant to expose), 0.0 = ref wins regardless."""
    wins = 0.0
    n = 0
    # ordering 1: ft is A, ref is B
    v1 = judge_prefers(judge_model, judge_tok, question, ft_answer, ref_answer)
    n += 1
    if v1 == "A":
        wins += 1.0
    elif v1 is None:
        wins += 0.5
    # ordering 2: ft is B, ref is A  (positions swapped)
    v2 = judge_prefers(judge_model, judge_tok, question, ref_answer, ft_answer)
    n += 1
    if v2 == "B":
        wins += 1.0
    elif v2 is None:
        wins += 0.5
    assert n == 2, "the judge MUST run both orderings (spec-part5 p.51 self-check)"
    return wins / n, n


def load_heldout(path):
    """Load held-out chat examples. Each line is either {"messages":[...]} or a flat
    {"prompt":..., "response":...} / {"question":..., "answer":...} pair."""
    examples, questions, golds = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                msgs = obj["messages"]
                q = next((m["content"] for m in msgs if m["role"] == "user"), "")
                g = next((m["content"] for m in reversed(msgs)
                          if m["role"] == "assistant"), "")
            else:
                q = obj.get("prompt") or obj.get("question") or ""
                g = obj.get("response") or obj.get("answer") or ""
                msgs = [{"role": "user", "content": q},
                        {"role": "assistant", "content": g}]
            examples.append(msgs)
            questions.append(q)
            golds.append(g)
    return examples, questions, golds


# 12 general prompts, deliberately UNRELATED to any plausible fine-tuning task. The
# forgetting probe answers these before (base) and after (fine-tune) and diffs by eye.
GENERAL_PROMPTS = [
    "What is the capital of Japan?",
    "Write a haiku about the ocean.",
    "Explain why the sky is blue in one sentence.",
    "What is 17 times 24?",
    "Who wrote Pride and Prejudice?",
    "Translate 'good morning' into Spanish.",
    "Give me a synonym for 'happy'.",
    "What year did the Apollo 11 moon landing happen?",
    "Summarize photosynthesis in one line.",
    "What is the boiling point of water at sea level in Celsius?",
    "Name three primary colors.",
    "Complete the sentence: 'A stitch in time...'",
]


def run_real(args):
    import torch

    print("=" * 72)
    print("VERSIONS (paste this line into any support request)")
    print("=" * 72)
    stamp()
    if not torch.cuda.is_available():
        print("\n  CUDA not available - the real eval needs a GPU. Stopping.")
        print("  (Run --self-test anywhere for the GPU-free eval arithmetic.)")
        sys.exit(1)
    print()

    print(f"  key box: \"{KEY_BOX}\"")
    print()

    # ---- the closed-loop warning, up front (p.51 warn box) ---------------------- #
    judge_fam = family_of(args.judge)
    gen_fam = family_of(args.generator_family) or (args.generator_family or None)
    print("=" * 72)
    print("CLOSED-LOOP CHECK - is your judge the same family as your data generator?")
    print("=" * 72)
    print(f"  judge model     : {args.judge}   (family: {judge_fam or 'unknown'})")
    print(f"  data generator  : {args.generator_family or 'not given'}   "
          f"(family: {gen_fam or 'unknown'})")
    if judge_fam and gen_fam and judge_fam == gen_fam:
        print(f"  !! WARNING: judge family == data-generator family ({judge_fam}).")
        print(f"     You are measuring how much your fine-tune sounds like {judge_fam},")
        print(f"     NOT quality. The score will improve and mean nothing. Use a")
        print(f"     DIFFERENT-family judge (p.51 closed-loop trap).")
    elif not judge_fam or not gen_fam:
        print(f"  (could not resolve both families - name --generator-family and use a")
        print(f"   recognizable judge id so this check can protect you.)")
    else:
        print(f"  OK: {judge_fam} judge vs {gen_fam} generator - different families.")
    print()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    examples, questions, golds = load_heldout(args.heldout)
    print(f"  held-out set: {len(examples)} examples from {args.heldout}")
    print()

    print("=" * 72)
    print("SAFETY: loading the fine-tune (bf16, ~16 GB for an 8B). The forgetting probe")
    print("and the judge load more checkpoints - freed between stages. Contends with")
    print("ComfyUI if up. Nothing is written or installed.")
    print("=" * 72)
    print()

    ft = load_model(args.model, args.adapter, torch.bfloat16)

    # ---- RUNG 1: held-out loss + perplexity + mean_token_accuracy --------------- #
    print("=" * 72)
    print("RUNG 1 - held-out loss  [free; answers only 'am I overfitting?']")
    print("=" * 72)
    loss, ppl, acc, ntok = heldout_loss(ft, tok, examples)
    print(f"  held-out loss          {loss:.4f} nats   over {ntok:,} assistant tokens")
    print(f"  perplexity  e^loss  =  {ppl:,.2f}")
    print(f"  mean_token_accuracy    {acc*100:.2f} %   (argmax == gold, assistant positions)")
    print(f"  [MEA] on your held-out set. Lower loss = better at predicting YOUR data.")
    print(f"        {KEY_BOX}")
    print()

    # ---- RUNG 2: programmatic task metric --------------------------------------- #
    if args.task_metric != "none":
        print("=" * 72)
        print(f"RUNG 2 - task metric '{args.task_metric}'  [if you have one, the rest is optional]")
        print("=" * 72)
        passes = 0
        counted = 0
        ft_answers = []
        for q, g in zip(questions, golds):
            a = generate_answer(ft, tok, q, args.max_new_tokens)
            ft_answers.append(a)
            r = task_pass(g, a, args.task_metric)
            if r is not None:
                counted += 1
                passes += int(r)
        if counted:
            print(f"  pass rate: {passes}/{counted} = {100*passes/counted:.1f} %   [MEA, your checker]")
        print(f"  a real checker (does the SQL run, does the test pass) beats every rung below.")
        print()
    else:
        ft_answers = [generate_answer(ft, tok, q, args.max_new_tokens) for q in questions]
        print("  (no task metric given - skipping rung 2; push hard for a programmatic one.)\n")

    # ---- The catastrophic-forgetting probe (needs the base) --------------------- #
    forgetting = None
    if args.base:
        print("  freeing the fine-tune to load the base for the forgetting probe...")
        base_answers_ft = [generate_answer(ft, tok, p, args.max_new_tokens)
                           for p in GENERAL_PROMPTS]
        del ft
        torch.cuda.empty_cache()

        base = load_model(args.base, None, torch.bfloat16)
        print()
        print("=" * 72)
        print("PROBE - catastrophic forgetting  [~free; the diff catches silent value loss]")
        print("=" * 72)
        print("  12 GENERAL prompts, base answer vs fine-tune answer. Read the flagged ones.")
        print()
        flagged = 0
        for p, ft_ans in zip(GENERAL_PROMPTS, base_answers_ft):
            base_ans = generate_answer(base, tok, p, args.max_new_tokens)
            d = forgetting_delta(base_ans, ft_ans)
            mark = "  " if d["jaccard"] > 0.4 else "!!"
            if d["jaccard"] <= 0.4:
                flagged += 1
            print(f"  {mark} [{d['jaccard']:.2f} overlap] {p}")
            print(f"        base: {base_ans[:100]!r}")
            print(f"        ft  : {ft_ans[:100]!r}")
        forgetting = flagged
        print()
        print(f"  {flagged}/{len(GENERAL_PROMPTS)} general answers moved a lot (overlap <= 0.40).")
        print(f"  [MEA] Low overlap on GENERAL prompts is the fine-tune quietly forgetting.")
        print(f"        This is a diff you READ, not a scalar to optimize.")
        del base
        torch.cuda.empty_cache()
        print()
    else:
        print("  (no --base given - skipping the forgetting probe. Pass --base to run it;")
        print("   it is the cheapest catch for the most expensive silent regression.)\n")

    # ---- RUNG 4: LLM-as-judge, both orderings averaged -------------------------- #
    if args.judge and args.base:
        print("=" * 72)
        print("RUNG 4 - LLM-as-judge  [cheap, biased; both orderings averaged, lengths reported]")
        print("=" * 72)
        print(f"  loading judge: {args.judge}")
        judge = load_model(args.judge, None, torch.bfloat16)
        judge_tok = AutoTokenizer.from_pretrained(args.judge)

        # we need ref (base) answers to the eval questions to judge ft against.
        reloaded_base = load_model(args.base, None, torch.bfloat16)
        ref_answers = [generate_answer(reloaded_base, judge_tok, q, args.max_new_tokens)
                       for q in questions]
        del reloaded_base
        torch.cuda.empty_cache()

        ft_win_total = 0.0
        ft_len_total = 0
        ref_len_total = 0
        for q, ft_a, ref_a in zip(questions, ft_answers, ref_answers):
            score, n_ord = judge_both_orderings(judge, judge_tok, q, ft_a, ref_a)
            assert n_ord == 2, "judge must average both orderings"
            ft_win_total += score
            ft_len_total += words_of(ft_a)
            ref_len_total += words_of(ref_a)
        m = len(questions) or 1
        print(f"  fine-tune win rate vs base: {100*ft_win_total/m:.1f} %   "
              f"[MEA, averaged over BOTH orderings - position bias cancelled]")
        print(f"  answer lengths (words): fine-tune {ft_len_total/m:.0f} avg | "
              f"base {ref_len_total/m:.0f} avg   <- watch verbosity bias")
        print(f"  a 50% win rate means the judge split by POSITION, not quality (a wash).")
        print(f"  reminder: this number is [MEA] on THIS judge; it is not transferable.")
        del judge
        torch.cuda.empty_cache()
        print()
    else:
        print("  (LLM-as-judge needs both --judge and --base; skipped. When you run it,")
        print("   it will average both orderings and report answer lengths automatically.)\n")

    print("=" * 72)
    print("Done. You climbed the staircase. If a rung answered your question, you can stop")
    print("there - and remember the whole point:")
    print(f"  {KEY_BOX}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="GPU-free: run and assert every eval rung's arithmetic")
    ap.add_argument("--model", help="fine-tuned model id/path (merged, or base + --adapter)")
    ap.add_argument("--adapter", default=None,
                    help="optional PEFT adapter dir to attach to --model (base+adapter path)")
    ap.add_argument("--base", default=None,
                    help="base model id/path - enables the forgetting probe and the judge")
    ap.add_argument("--heldout", help="held-out chat JSONL for loss + task metric")
    ap.add_argument("--judge", default=None,
                    help="LLM-as-judge model id - use a DIFFERENT family than your data generator")
    ap.add_argument("--generator-family", default=None,
                    help="the model/family that GENERATED your SFT data (for the closed-loop warn)")
    ap.add_argument("--task-metric", default="none", choices=["contains", "json", "none"],
                    help="programmatic task checker (swap in your real one)")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="generation cap for task metric / probe / judge answers")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.model or not args.heldout:
        print("The real eval needs at least --model and --heldout. For the GPU-free math:")
        print("    python 12_evaluate.py --self-test")
        sys.exit(2)

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("torch + transformers are required for the real eval and are not importable")
        print("here. Install them into THE fresh venv (never ComfyUI's; hardware-ground-truth")
        print("§3), or run the GPU-free path:")
        print("    python 12_evaluate.py --self-test")
        sys.exit(1)

    run_real(args)


if __name__ == "__main__":
    main()
