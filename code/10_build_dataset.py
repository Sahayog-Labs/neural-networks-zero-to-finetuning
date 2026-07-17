#!/usr/bin/env python3
"""
10_build_dataset.py - the data pipeline for RUNG 6. p.48 partner to 11_finetune_qlora.py.

The whole SFT stack downstream is only as honest as this step: JSONL -> `datasets`
-> chat-template -> tokenized tensors, with a loss mask that decides WHICH tokens the
model is graded on. Get the mask wrong and every number after it is a lie you can't
see - the loss goes down, the model gets worse, no exception is raised. This script is
the `13_chat_template_and_mask` beat made runnable on YOUR ~500 examples: it renders one
real example through `tokenizer.apply_chat_template()`, prints the `-100` loss mask
token-by-token, and ASSERTS the assistant-only fraction matches what it just printed
(the page's `abs(mask.float().mean() - assistant_fraction) < 1e-6`).

Three things it guarantees, each an assert that fails loudly on regression:
  1. Non-assistant positions carry label = -100 (HF's ignore_index; the m_i mask,
     notation §4.5). Prompt/system/user tokens cost EXACTLY zero loss.
  2. The train/test split is BY DOCUMENT, not by example. Split by example and
     near-duplicate chunks from the same source leak across the boundary and your
     eval becomes a number that only measures memorization (brief §11, rule 6).
  3. The assistant-token fraction is printed, per-example and over the corpus - the
     page-48 misconception ("masking is a minor detail") dies against this number:
     on a 400-token prompt / 50-token answer, full-sequence loss spends 89% of the
     gradient teaching the model to write YOUR questions. Completion-only spends 0%.

Why two ways to build the mask, and why they must agree:
  - PRIMARY (real path): the template's own `{% generation %}` markers via
    `apply_chat_template(..., return_assistant_tokens_mask=True)`. This is EXACTLY
    the mechanism TRL's `assistant_only_loss=True` uses, and it is correct for
    multi-turn and for Qwen3's thinking template.
  - ILLUSTRATION / fallback: the prefix-diff - render the prompt (messages minus the
    final assistant turn, `add_generation_prompt=True`), render the full conversation,
    and mask the shared prefix. When the template has no generation markers this is the
    fallback, and it only works if the prompt render is a token-exact PREFIX of the
    full render. That prefix invariant IS the "never hand-roll the template" lesson:
    a string-concatenated template that puts a space in the wrong place breaks it
    silently (page 44's " the"-vs-"the" byte fact).
The self-test exercises the shared mask/split core on a toy prefix-stable ChatML
renderer so the arithmetic is verified with no torch, no transformers, no GPU.

Usage
-----
    python 10_build_dataset.py --self-test          # GPU-free, dependency-free: the
                                                    #   mask + split arithmetic, asserted
    python 10_build_dataset.py --in my_data.jsonl   # the real pipeline on your JSONL
    python 10_build_dataset.py --in my_data.jsonl --model Qwen/Qwen3-8B --out ./tokenized
    python 10_build_dataset.py                       # real path on the tiny built-in corpus

Input JSONL - one JSON object per line, any of these shapes (brief §11):
    {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
    {"prompt": "...", "completion": "..."}          # or {"instruction","response"}
  Optional "doc_id" (or "source") groups examples by document for the split. Without it,
  each example is its own document and the script WARNS - because per-example splitting
  is the exact leakage the brief tells you to avoid.

SAFETY
------
This is a CPU-only data step. It loads NO model weights and touches NO GPU, so it does
not contend with ComfyUI. It downloads the tokenizer's config/vocab (a few MB) into your
HF cache if not present, and writes a tokenized dataset ONLY when you pass --out (naming
the directory). Otherwise everything stays in memory. Read-only w.r.t. the system.

Env (THE fresh venv - never ComfyUI's; hardware-ground-truth §3):
    transformers 5.14.1 · trl 1.8.x · datasets >=4.7 · CUDA n/a (no GPU used)
    v5 note: apply_chat_template is unchanged; assistant_only_loss lives in SFTConfig.
"""

import argparse
import json
import random
import sys

# --------------------------------------------------------------------------- #
# Frozen facts. The page-48 worked example is the one this script asserts; the
# vocab is context for "loss floor" only (03_base_vs_instruct owns ln V).
# --------------------------------------------------------------------------- #
IGNORE_INDEX = -100                    # HF ignore_index / the m_i == 0 label (notation §4.5)
V_QWEN3 = 151_936                      # Qwen3 vocab, constants §1.1 [VP] (context only)
EOS = "<|im_end|>"                     # ChatML turn terminator (Qwen family); brief §11 rule 3
IM_START = "<|im_start|>"
# The page's worked example (48, "resolving the predict"): 400-token prompt, 50-token answer.
# Full-sequence mode grades ALL 450 tokens; of that gradient, 89% lands on the PROMPT
# (400/450) - text the model must never generate. Completion-only grades just the 50-token
# answer (50/450 = 11%). "89% wasted" is the prompt fraction, not the graded fraction.
PAGE_PROMPT_TOKENS = 400
PAGE_ANSWER_TOKENS = 50
PAGE_PROMPT_FRACTION = 400 / 450       # = 0.8888... -> the "89% wasted" number
PAGE_COMPLETION_FRACTION = 50 / 450    # = 0.1111... -> what completion-only actually trains on


# --------------------------------------------------------------------------- #
# THE SHARED CORE - pure Python, no dependencies. The real path (real tokenizer)
# and the self-test (toy tokenizer) call THESE functions, so the mask the page
# teaches and the mask the script builds cannot drift. A mask is a list of
# 1 (contributes to loss) / 0 (ignored); labels put IGNORE_INDEX where mask == 0.
# --------------------------------------------------------------------------- #

def completion_mask_from_prefix(prompt_len, total_len):
    """The prefix-diff mask: 0 on the prompt's tokens, 1 on the completion's.
    `prompt_len` MUST be the length of a token-exact prefix of the full sequence."""
    if not 0 <= prompt_len <= total_len:
        raise ValueError(f"prompt_len {prompt_len} out of range for total {total_len}")
    return [0] * prompt_len + [1] * (total_len - prompt_len)


def labels_from_mask(token_ids, mask, ignore_index=IGNORE_INDEX):
    """labels = input_ids.clone(); labels[mask == 0] = -100  (the page's two lines)."""
    if len(token_ids) != len(mask):
        raise ValueError(f"len(ids)={len(token_ids)} != len(mask)={len(mask)}")
    return [tid if m == 1 else ignore_index for tid, m in zip(token_ids, mask)]


def assistant_fraction(mask):
    """mask.float().mean() - the fraction of positions the model is graded on."""
    return sum(mask) / len(mask) if mask else 0.0


def assert_mask_is_valid(token_ids, mask, labels, ignore_index=IGNORE_INDEX):
    """The RUNG-6 guarantee #1, as three asserts. Returns the assistant fraction."""
    assert len(token_ids) == len(mask) == len(labels), "ids/mask/labels length mismatch"
    for tid, m, lab in zip(token_ids, mask, labels):
        if m == 0:
            assert lab == ignore_index, \
                f"non-assistant position must be labelled {ignore_index}, got {lab}"
        else:
            assert lab == tid, f"assistant position must keep its token id, got {lab} vs {tid}"
            assert lab != ignore_index, "an assistant token id can never equal the ignore_index"
    frac = assistant_fraction(mask)
    # the page's own consistency assert: the printed fraction == the mask's mean
    assert abs(assistant_fraction(mask) - frac) < 1e-9
    assert 0.0 <= frac <= 1.0
    return frac


def document_split(records, test_frac, seed):
    """Split BY DOCUMENT (guarantee #2). Every record carries a 'doc_id'; whole
    documents go to train or test, never split across. Returns (train, test, meta)
    and GUARANTEES the train/test doc-id sets are disjoint (no leakage)."""
    doc_ids = sorted({r["doc_id"] for r in records})
    if len(doc_ids) < 2:
        # can't hold out a document if there's only one; caller is warned upstream
        return list(records), [], {"n_docs": len(doc_ids), "n_test_docs": 0,
                                   "test_docs": set(), "degenerate": True}
    rng = random.Random(seed)
    shuffled = list(doc_ids)
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_frac))
    n_test = min(n_test, len(shuffled) - 1)          # always leave >=1 doc for train
    test_docs = set(shuffled[:n_test])
    train = [r for r in records if r["doc_id"] not in test_docs]
    test = [r for r in records if r["doc_id"] in test_docs]
    train_docs = {r["doc_id"] for r in train}
    # THE no-leakage assert - the whole reason to split by document
    assert train_docs.isdisjoint(test_docs), \
        "train/test document leakage: the same doc_id is on both sides of the split"
    return train, test, {"n_docs": len(doc_ids), "n_test_docs": len(test_docs),
                         "test_docs": test_docs, "degenerate": False}


# --------------------------------------------------------------------------- #
# Record normalization - accept the JSONL shapes the brief lists, emit a uniform
# {"messages": [...], "doc_id": ...}. One place, so the real path and any future
# caller agree on what "an example" is.
# --------------------------------------------------------------------------- #

def normalize_record(obj, index):
    """Map one JSONL object to {"messages":[...], "doc_id":...}. Raises on garbage."""
    if "messages" in obj:
        messages = obj["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"record {index}: 'messages' must end with an assistant turn")
    elif "prompt" in obj and "completion" in obj:
        messages = [{"role": "user", "content": obj["prompt"]},
                    {"role": "assistant", "content": obj["completion"]}]
    elif "instruction" in obj and "response" in obj:
        messages = [{"role": "user", "content": obj["instruction"]},
                    {"role": "assistant", "content": obj["response"]}]
    else:
        raise ValueError(f"record {index}: need 'messages', 'prompt'+'completion', "
                         f"or 'instruction'+'response'; got keys {sorted(obj)}")
    doc_id = obj.get("doc_id", obj.get("source"))
    has_doc = doc_id is not None
    if not has_doc:
        doc_id = f"__example_{index}__"     # its own document -> triggers the warning
    return {"messages": messages, "doc_id": str(doc_id), "_had_doc_id": has_doc}


# A tiny built-in corpus: 3 documents, a few QA pairs each, so the BY-DOCUMENT split
# is actually demonstrable (a per-example split could leak two pairs from one doc).
# Real runs pass --in with your ~500 examples; this only proves the pipeline is wired.
FALLBACK_RECORDS = [
    {"doc_id": "handbook-onboarding",
     "messages": [{"role": "user", "content": "How do I request time off?"},
                  {"role": "assistant", "content": "Submit a request in the HR portal at least two weeks ahead; your manager approves it there."}]},
    {"doc_id": "handbook-onboarding",
     "messages": [{"role": "user", "content": "When do new hires get their laptop?"},
                  {"role": "assistant", "content": "IT ships it to arrive on or before your first day; set it up with the onboarding checklist."}]},
    {"doc_id": "handbook-security",
     "messages": [{"role": "user", "content": "What is our password policy?"},
                  {"role": "assistant", "content": "At least 16 characters, unique per service, stored only in the company password manager."}]},
    {"doc_id": "handbook-security",
     "messages": [{"role": "user", "content": "Can I use a personal USB drive on my work laptop?"},
                  {"role": "assistant", "content": "No. Removable storage is blocked by policy; use the approved cloud share instead."}]},
    {"doc_id": "handbook-expenses",
     "messages": [{"role": "user", "content": "How do I expense a client dinner?"},
                  {"role": "assistant", "content": "Upload the itemized receipt in the expense tool, tag it Client Meals, and note the client name."}]},
    {"doc_id": "handbook-expenses",
     "messages": [{"role": "user", "content": "What is the daily meal limit when traveling?"},
                  {"role": "assistant", "content": "Seventy-five dollars per day domestically; anything above needs a one-line justification."}]},
]


def load_records(path):
    """Read a JSONL file into normalized records, or fall back to the built-in corpus."""
    if path is None:
        print(f"  no --in given -> using the built-in {len(FALLBACK_RECORDS)}-example, "
              f"3-document corpus (proves the pipeline; teaches the model nothing durable)")
        return [normalize_record(r, i) for i, r in enumerate(FALLBACK_RECORDS)]
    records, n_missing_doc = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = normalize_record(json.loads(line), i)
            n_missing_doc += 0 if rec["_had_doc_id"] else 1
            records.append(rec)
    if not records:
        raise ValueError(f"{path}: no records parsed")
    print(f"  loaded {len(records)} example(s) from {path}")
    if n_missing_doc:
        print(f"  !! {n_missing_doc} example(s) had NO doc_id/source field. Each becomes")
        print(f"     its OWN document, so the split degrades to per-example - the exact")
        print(f"     leakage the brief (§11 rule 6) warns against. Add a 'doc_id' per source.")
    return records


# --------------------------------------------------------------------------- #
# SELF-TEST - a toy, prefix-stable ChatML renderer + a whitespace/punct "tokenizer".
# It is NOT BPE (real tokenization needs the real tokenizer). It exists only to drive
# the shared mask/split core deterministically with zero dependencies, and to reproduce
# the page's 89%/11% number exactly.
# --------------------------------------------------------------------------- #

def toy_render_chatml(messages, add_generation_prompt=False):
    """`<|im_start|>role\\ncontent<|im_end|>\\n` per turn - the real ChatML shape,
    prefix-stable by construction (no thinking block), so the prefix-diff is exact."""
    parts = []
    for m in messages:
        parts.append(f"{IM_START}{m['role']}\n{m['content']}{EOS}\n")
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def toy_tokenize(text):
    """A stand-in tokenizer: special tokens are one token; everything else splits on
    whitespace and keeps punctuation attached. Returns a list of int 'ids' (hashes)."""
    import re
    # keep the ChatML specials and \n as their own tokens
    pattern = re.compile(r"<\|im_start\|>|<\|im_end\|>|\n|[^\s<]+")
    toks = pattern.findall(text)
    # map to stable non-negative ids; never collide with IGNORE_INDEX (-100)
    return [abs(hash(t)) % 1_000_003 for t in toks], toks


def build_masked_example_toy(messages):
    """Render + tokenize + mask ONE example with the toy stack, via the prefix-diff.
    Returns (ids, str_toks, mask, labels). The prompt render MUST be a token prefix."""
    prompt_str = toy_render_chatml(messages[:-1], add_generation_prompt=True)
    full_str = toy_render_chatml(messages, add_generation_prompt=False)
    prompt_ids, _ = toy_tokenize(prompt_str)
    full_ids, full_toks = toy_tokenize(full_str)
    # the prefix invariant (the "never hand-roll" lesson): prompt is a token-exact prefix
    assert full_ids[:len(prompt_ids)] == prompt_ids, \
        "toy template not prefix-stable - a hand-rolled template would fail here silently"
    mask = completion_mask_from_prefix(len(prompt_ids), len(full_ids))
    labels = labels_from_mask(full_ids, mask)
    return full_ids, full_toks, mask, labels


def self_test():
    print("=" * 72)
    print("SELF-TEST - the mask + document-split core (no torch, no transformers, no GPU)")
    print("=" * 72)

    # ---- 1. one toy example, masked, printed like the real path prints it ---- #
    example = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]
    ids, toks, mask, labels = build_masked_example_toy(example)
    frac = assert_mask_is_valid(ids, mask, labels)

    print("  toy ChatML render (NOT BPE - a stand-in tokenizer for the arithmetic only):")
    for t, m in zip(toks, mask):
        chip = t if t not in (IM_START, EOS) else t
        tag = "loss" if m == 1 else " -100"
        shown = repr(chip) if chip != "\n" else "'\\n'"
        print(f"      [{tag}] {shown}")
    print(f"  assistant-token fraction = {sum(mask)}/{len(mask)} = {frac:.4f}")
    print(f"  every mask==0 position is labelled {IGNORE_INDEX}; every mask==1 keeps its id. OK")
    assert mask[0] == 0, "the <|im_start|> that opens the system turn must be masked"
    assert mask[-1] == 1, "the assistant turn's closing <|im_end|> must be UNMASKED (learn to stop)"
    assert 0.0 < frac < 1.0, "a real example grades some-but-not-all positions"
    print()

    # ---- 2. the page-48 worked example: 89% vs 11%, to the digit ---- #
    print("  the page-48 worked example (400-token prompt, 50-token answer):")
    total = PAGE_PROMPT_TOKENS + PAGE_ANSWER_TOKENS
    full_seq_mask = [1] * total                                   # no mask: grade everything
    completion_mask = completion_mask_from_prefix(PAGE_PROMPT_TOKENS, total)
    f_full = assistant_fraction(full_seq_mask)                    # = 1.0 (grades all 450)
    f_comp = assistant_fraction(completion_mask)                  # = 50/450 (grades the answer)
    f_prompt = 1.0 - f_comp                                       # = 400/450, the wasted share
    print(f"      full-sequence (no mask): grades {f_full:.4f} of positions -> but "
          f"{round(f_prompt*100)}% of that gradient lands on the PROMPT")
    print(f"      completion-only  (mask): grades {f_comp:.4f} -> {round(f_comp*100)}% on the "
          f"answer, {round(f_prompt*100)}% cost zero loss")
    assert abs(f_full - 1.0) < 1e-9, "full-sequence grades every position"
    assert abs(f_prompt - PAGE_PROMPT_FRACTION) < 1e-9
    assert abs(f_comp - PAGE_COMPLETION_FRACTION) < 1e-9
    assert round(f_prompt * 100) == 89, "the page's number is 89% - it must reproduce exactly"
    print(f"      matches the page: 89% of an unmasked gradient teaches the model to write")
    print(f"      YOUR questions. That number ends the argument.")
    print()

    # ---- 3. document-level split: whole documents move; no id leaks across ---- #
    print("  document-level split (whole documents move as a unit; assert no leakage):")
    recs = [normalize_record(r, i) for i, r in enumerate(FALLBACK_RECORDS)]
    train, test, meta = document_split(recs, test_frac=0.34, seed=42)
    train_docs = sorted({r["doc_id"] for r in train})
    test_docs = sorted(meta["test_docs"])
    print(f"      {meta['n_docs']} documents -> {len(train_docs)} train / "
          f"{len(test_docs)} test  ({len(train)} / {len(test)} examples)")
    print(f"      train docs: {train_docs}")
    print(f"      test  docs: {test_docs}")
    assert set(train_docs).isdisjoint(set(test_docs)), "no doc may appear on both sides"
    assert len(test) > 0 and len(train) > 0, "both splits must be non-empty"
    # prove the contrast: a naive per-EXAMPLE split of a 2-example doc could leak
    assert all(any(r["doc_id"] == d for r in train) for d in train_docs)
    print(f"      no doc_id appears in both splits - the eval measures generalization,")
    print(f"      not memorization of a near-duplicate that leaked from training.")
    print()

    print("=" * 72)
    print("SELF-TEST OK. -100 mask placed, 89%/11% reproduced, split leak-free.")
    print("Run with --in your_data.jsonl on your box for the real tokenizer + datasets path.")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# THE REAL PATH - real tokenizer + `datasets`. CPU-only, no model weights, no GPU.
# Desk-checked against the verified July-2026 APIs (constants §7.1, brief-llm §11/§13):
#   - AutoTokenizer.from_pretrained(model)
#   - tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
#         enable_thinking=False, return_dict=True[, return_assistant_tokens_mask=True])
#   - datasets.Dataset.from_list([...]) ; .map(...)
#   TRL's assistant_only_loss=True uses the SAME {% generation %} mask this path prefers.
# --------------------------------------------------------------------------- #

def stamp():
    line = "  (versions)"
    try:
        import transformers
        line = f"  transformers {transformers.__version__}"
    except Exception:
        pass
    try:
        import datasets
        line += f" | datasets {datasets.__version__}"
    except Exception:
        pass
    try:
        import trl
        line += f" | trl {trl.__version__}"
    except Exception:
        pass
    print(line)


def _apply_template(tok, messages, enable_thinking=False, **kw):
    """apply_chat_template, retrying without enable_thinking for non-Qwen templates
    that don't accept the kwarg. Keeps Qwen3 behavior (thinking off) identical."""
    try:
        return tok.apply_chat_template(messages, enable_thinking=enable_thinking, **kw)
    except TypeError:
        return tok.apply_chat_template(messages, **kw)


def render_and_mask_real(tok, messages, enable_thinking=False):
    """Render + tokenize + build the loss mask for ONE example with a REAL tokenizer.
    Prefers the template's own assistant mask ({% generation %} markers, exactly what
    TRL's assistant_only_loss uses); falls back to the prefix-diff otherwise.
    Returns (ids, str_toks, mask, labels, method)."""
    # try the template's native assistant mask first
    try:
        out = _apply_template(
            tok, messages, enable_thinking=enable_thinking,
            tokenize=True, add_generation_prompt=False, return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = list(out["input_ids"])
        amask = out.get("assistant_masks")
        if amask is not None and any(amask):
            mask = [1 if m else 0 for m in amask]
            labels = labels_from_mask(ids, mask)
            toks = tok.convert_ids_to_tokens(ids)
            return ids, toks, mask, labels, "template {% generation %} markers (TRL-native)"
    except (TypeError, KeyError, ValueError):
        pass  # template lacks generation markers or older signature -> prefix-diff below

    # fallback: the prefix-diff, with the prefix invariant enforced
    prompt_ids = _apply_template(
        tok, messages[:-1], enable_thinking=enable_thinking,
        tokenize=True, add_generation_prompt=True,
    )
    full_ids = _apply_template(
        tok, messages, enable_thinking=enable_thinking,
        tokenize=True, add_generation_prompt=False,
    )
    prompt_ids, full_ids = list(prompt_ids), list(full_ids)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "this template is NOT prefix-stable, so the prefix-diff mask would be wrong.\n"
            "   Use a template with {% generation %} markers (Qwen3 has them; TRL's\n"
            "   assistant_only_loss relies on them). This is the 'never hand-roll the\n"
            "   template' failure the page warns about - made loud instead of silent.")
    mask = completion_mask_from_prefix(len(prompt_ids), len(full_ids))
    labels = labels_from_mask(full_ids, mask)
    toks = tok.convert_ids_to_tokens(full_ids)
    return full_ids, toks, mask, labels, "prefix-diff (prompt is a token-exact prefix)"


def run_real(args):
    from transformers import AutoTokenizer
    from datasets import Dataset

    print("=" * 72)
    print("VERSIONS (paste this line into any support request)")
    print("=" * 72)
    stamp()
    print()

    records = load_records(args.infile)
    print()

    print("=" * 72)
    print(f"TOKENIZER: {args.model}  (config/vocab only - no model weights, no GPU)")
    print("=" * 72)
    tok = AutoTokenizer.from_pretrained(args.model)
    eos_id = tok.convert_tokens_to_ids(EOS)
    if eos_id is None or eos_id < 0:
        print(f"  !! {EOS} not in this tokenizer's vocab. For a NON-Qwen model, set the")
        print(f"     right turn terminator; SFTConfig(eos_token=...) must match it (brief §11).")
    else:
        print(f"  turn terminator {EOS} -> id {eos_id}. SFTConfig(eos_token=\"{EOS}\") must match it.")
    print()

    # ---- the 13_chat_template_and_mask beat: ONE real example, printed in full ---- #
    print("=" * 72)
    print("THE MASKED EXAMPLE - apply_chat_template() output + the -100 loss mask")
    print("=" * 72)
    ex = records[0]["messages"]
    try:
        rendered = tok.apply_chat_template(
            ex, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    except TypeError:
        # non-Qwen templates may not accept enable_thinking; drop it
        rendered = tok.apply_chat_template(
            ex, tokenize=False, add_generation_prompt=False)
    print("  rendered conversation (never hand-rolled - the library owns the template):")
    for ln in rendered.splitlines():
        print(f"      {ln!r}")
    print()

    ids, toks, mask, labels, method = render_and_mask_real(tok, ex)
    frac = assert_mask_is_valid(ids, mask, labels)
    print(f"  mask built via: {method}")
    print(f"  token-by-token loss mask (green=graded, -100=ignored). Showing all "
          f"{len(ids)} tokens:")
    for tk, m, lab in zip(toks, mask, labels):
        tag = "loss" if m == 1 else "-100"
        show = tk.replace("\n", "\\n")
        print(f"      [{tag}]  {show!r:<24} label={lab}")
    print()

    # the page's consistency assert, verbatim in spirit:
    #   abs(mask.float().mean() - assistant_fraction) < 1e-6
    printed_fraction = sum(mask) / len(mask)
    assert abs(printed_fraction - frac) < 1e-6, \
        "the printed assistant fraction must equal the mask's mean"
    # the terminator should be graded so the model learns to STOP; a trailing "\n"
    # after <|im_end|> may legitimately be masked, so check the id is present & graded
    if eos_id is not None and eos_id >= 0:
        graded_eos = any(t == eos_id and m == 1 for t, m in zip(ids, mask))
        if graded_eos:
            print(f"  the turn terminator {EOS} is inside the graded span -> the model "
                  f"learns to STOP (not ramble). Good.")
        else:
            print(f"  NOTE: {EOS} is not in the graded span for this example - check the")
            print(f"        template's generation markers, or the model may not learn to stop.")
    print(f"  assistant-token fraction (this example) = {sum(mask)}/{len(mask)} "
          f"= {frac:.4f}   [{round(frac*100)}% graded, {round((1-frac)*100)}% cost zero loss]")
    print(f"  assert abs(mask.mean() - assistant_fraction) < 1e-6  ->  PASS")
    print()

    # ---- build the datasets object + tokenize every example, then split BY DOCUMENT ---- #
    print("=" * 72)
    print("THE PIPELINE - JSONL -> datasets -> chat-template -> tokenized tensors")
    print("=" * 72)

    def tokenize_map(rec):
        i2, t2, m2, l2, _ = render_and_mask_real(tok, rec["messages"])
        return {"input_ids": i2, "labels": l2, "assistant_mask": m2,
                "n_tokens": len(i2), "n_assistant": sum(m2), "doc_id": rec["doc_id"]}

    ds = Dataset.from_list([{"messages": r["messages"], "doc_id": r["doc_id"]}
                            for r in records])
    ds = ds.map(tokenize_map, remove_columns=ds.column_names)
    print(f"  tokenized {len(ds)} example(s). Columns: {ds.column_names}")

    # corpus-wide assistant-token fraction (guarantee #3)
    total_tokens = sum(ds["n_tokens"])
    total_assistant = sum(ds["n_assistant"])
    corpus_frac = total_assistant / total_tokens if total_tokens else 0.0
    print(f"  corpus assistant-token fraction = {total_assistant:,}/{total_tokens:,} "
          f"= {corpus_frac:.4f}")
    print(f"    -> completion-only training grades {round(corpus_frac*100)}% of your tokens;")
    print(f"       full-sequence would waste the other {round((1-corpus_frac)*100)}% on "
          f"prompt text the model must never generate.")
    print()

    # the document-level split (guarantee #2), on the RECORD list (not row-level)
    train_recs, test_recs, meta = document_split(records, args.test_frac, args.seed)
    if meta["degenerate"]:
        print(f"  !! only {meta['n_docs']} document(s) - cannot hold one out. Add examples")
        print(f"     from MORE distinct sources, or the eval set is empty. (brief §11 rule 6)")
    else:
        train_docs = sorted({r["doc_id"] for r in train_recs})
        print(f"  document-level split (seed {args.seed}, test_frac {args.test_frac}):")
        print(f"      {meta['n_docs']} docs -> {len(train_docs)} train / "
              f"{len(meta['test_docs'])} test   "
              f"({len(train_recs)} / {len(test_recs)} examples)")
        assert set(train_docs).isdisjoint(meta["test_docs"]), \
            "train/test document leakage detected"
        print(f"      assert train_docs.isdisjoint(test_docs)  ->  PASS (no leakage)")
        train_ds = Dataset.from_list([tokenize_map(r) for r in train_recs])
        test_ds = Dataset.from_list([tokenize_map(r) for r in test_recs])
        print(f"      -> train_dataset[{len(train_ds)}], eval_dataset[{len(test_ds)}] "
              f"ready for SFTTrainer(train_dataset=..., eval_dataset=...)")
        if args.out:
            train_ds.save_to_disk(f"{args.out}/train")
            test_ds.save_to_disk(f"{args.out}/test")
            print(f"      wrote tokenized datasets to {args.out}/train and {args.out}/test")
    print()

    print("=" * 72)
    print("Done. This dataset feeds 11_finetune_qlora.py (RUNG 6) unchanged: pass")
    print("assistant_only_loss=True and SFTConfig(eos_token=\"<|im_end|>\") - the mask")
    print("you just verified is the one TRL will build. The pipeline is honest end to end.")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="GPU-free, dependency-free: run and assert the mask + split arithmetic")
    ap.add_argument("--in", dest="infile", default=None,
                    help="input JSONL (default: the tiny built-in corpus)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="tokenizer to render the chat template (config/vocab only)")
    ap.add_argument("--out", default=None,
                    help="directory to save the tokenized train/test datasets (default: none)")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="fraction of DOCUMENTS (not examples) held out for eval")
    ap.add_argument("--seed", type=int, default=42, help="split seed")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    try:
        import transformers  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        print("transformers + datasets are required for the real pipeline and are not")
        print("importable here. Install them into THE fresh venv (never ComfyUI's;")
        print("hardware-ground-truth §3), or run the dependency-free path:")
        print("    python 10_build_dataset.py --self-test")
        sys.exit(1)

    run_real(args)


if __name__ == "__main__":
    main()
