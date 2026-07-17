#!/usr/bin/env python3
"""
15_grpo_rlvr.py - RUNG 7, the capstone. GRPO + a verifiable reward on Qwen3-0.6B. (p.52)

This is the 2026 reasoning frontier, shrunk to fit a two-hour experiment on a box
you own. SFT (page 50) taught imitation; DPO/preference (page 48) taught taste;
RLVR - reinforcement learning from VERIFIABLE rewards - teaches the model to get a
checkable answer RIGHT, with no human labels and no learned reward model.

    GRPO (Group Relative Policy Optimization) is the algorithm DeepSeek actually
    used. It throws PPO's value network away. For one prompt it samples a GROUP of
    G completions, scores each with a program (the "verifiable reward"), and credits
    each completion against the GROUP'S OWN mean and spread:

        A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

    PPO needs a learned critic to ask "was this better than average?" GRPO just asks
    G times and takes the average. The critic was replaceable by a sample mean - and
    that is the whole idea (constants §7.2 context; brief-llm §GRPO).

The frozen quiz numeric this script asserts (page 52, constants context):
    a group with mean 0.4, std 0.5; a completion the verifier PASSED (r_i = 1) gets
        A_i = (1 - 0.4) / 0.5 = 0.6 / 0.5 = 1.2
    a completion that FAILED (r_i = 0) gets  A_i = (0 - 0.4) / 0.5 = -0.8.
    The policy is nudged TOWARD the +1.2 rows and AWAY from the -0.8 rows.

Why this capstone needed a container (the gate that held it - constants §6.9, D-10j):
    GRPO is ONLINE. It must GENERATE the G completions INSIDE the training loop, then
    score and step. Generation-in-the-loop wants vLLM. But the PyPI aarch64 vLLM wheel
    is built against CUDA 12; it pins `libcudart.so.12` and dies at IMPORT on the
    Spark's CUDA-13-only DGX OS - before a single kernel's compute-capability is even
    checked. The validated path is the container `vllm/vllm-openai:cu130-nightly`,
    pinned by digest, which ships its own CUDA-13 userspace. This script runs
    `use_vllm=True, vllm_mode="colocate"` (generation + training in one process, right
    for a single box) INSIDE that container. A `--no-vllm` HF-generation fallback runs
    without it (slower) so the loop is still exercisable if the container isn't up.

The one TRL v1 default to know (constants §7.2): `GRPOConfig.vllm_mode` flipped
    "server" -> "colocate" at trl>=1.8. The rest of the v0->v1 migration is minimal.
    Bump the pin (`trl>=1.8,<2`) and move on. The real run asserts the live default.

Runnable scope (page 52): the capstone runs on Qwen3-0.6B - the smoke-test sibling
    (constants §1.4) - so a real generation-in-the-loop job finishes in minutes, not
    hours. The ARITHMETIC is identical to the 8B anchor; you recover the 8B's memory
    and throughput by swapping four config values - d=4096, L=36, H_kv=8, d_ff=12288
    (VP, constants §1.1) - into the same formulas from page 49. The 0.6B is smaller on
    every axis, which is exactly why it fits a two-hour window.

Self-test (no GPU, no download, no torch): `python 15_grpo_rlvr.py --self-test`. It
    reproduces the A_i = 1.2 / -0.8 frozen arithmetic, proves group advantages centre
    on zero with unit spread (that is what subtracting the mean and dividing by std
    DOES), and unit-tests both verifiable reward functions on fixed strings - all pure
    Python, so a regression fails loudly on any laptop.

Usage
-----
    python 15_grpo_rlvr.py --self-test                          # arithmetic + reward checks, no GPU
    python 15_grpo_rlvr.py --model Qwen/Qwen3-0.6B --reward accuracy    # in the cu130 container
    python 15_grpo_rlvr.py --reward json_schema --num-generations 8
    python 15_grpo_rlvr.py --reward accuracy --no-vllm          # HF-generation fallback (slower)

`--reward` picks a PROGRAMMATIC checker (no human, no reward model):
    accuracy     - math/reasoning grader: extract the final answer, compare to ground truth.
    json_schema  - schema validator: does the completion parse as JSON with the required keys?
Point it at whatever YOUR domain can check (does the config parse? does the SQL run?).

SAFETY: this capstone is HEAVY - generation-in-the-loop saturates the GPU for ~2 hr and,
    with vLLM colocate, holds BOTH the policy and the vLLM engine in the same unified-memory
    pool. If ComfyUI is live on the Spark it WILL contend - consult before launching (HARD
    SAFETY RULE: never run it FOR him on the Spark; he runs it there). It writes a LoRA-sized
    adapter to --output-dir and downloads Qwen3-0.6B (~1.2 GB) on first run. It installs nothing.
    NEVER run vLLM outside the cu130 container on this box - the CUDA-12 wheel import-traps.

Requires (constants §6.9/§7, pin exactly), INSIDE the container:
    torch (cu130) · transformers 5.14.1 · trl>=1.8,<2 · peft 0.19.1 · vllm (cu130-nightly image)
    Verified against these versions 2026-07-16; the API below is transcribed, never remembered.
"""

import argparse
import json
import re
import sys
import time
from statistics import mean, pstdev

GiB = 1 << 30
GB = 10 ** 9

# --------------------------------------------------------------------------- #
# FROZEN (constants / page 52). The quiz numeric and the 8B anchor config the
# 0.6B substitutes into. These are the ONLY numbers the self-test asserts.
# --------------------------------------------------------------------------- #
QUIZ_MEAN = 0.4          # page-52 group mean
QUIZ_STD = 0.5           # page-52 group std
QUIZ_R_PASS = 1.0        # a completion the verifier passed
QUIZ_A_PASS = 1.2        # (1 - 0.4) / 0.5  -> the frozen quiz answer
QUIZ_R_FAIL = 0.0        # a completion the verifier failed
QUIZ_A_FAIL = -0.8       # (0 - 0.4) / 0.5

# Qwen3-8B anchor (constants §1.1, VP). The 0.6B is smaller on every axis; you
# recover these four to move any page-49 formula from the 0.6B back to the 8B.
ANCHOR_8B = {"d_model": 4096, "L": 36, "H_kv": 8, "d_ff": 12288}

# The verifiable-reward scale this script uses. Correct answer scores full; a
# well-formed-but-wrong answer gets a small format credit (maps to the page's
# green/amber/red chips); anything else scores zero. GRPO only needs the ORDERING
# to be right - a program, not a labeller, produced every number here.
R_CORRECT = 1.0
R_FORMATTED_WRONG = 0.1
R_FAIL = 0.0


# --------------------------------------------------------------------------- #
# THE GRPO ADVANTAGE - pure arithmetic, the heart of the page. Shared by the
# self-test and (as a sanity mirror) the real run. No torch: this IS the math
# TRL runs per group, written out so page and script cannot drift.
# --------------------------------------------------------------------------- #

def group_advantages(rewards, eps=1e-8):
    """A_i = (r_i - mean) / std over one group, using the POPULATION std (ddof=0)
    - the same normalization TRL applies with its default scale_rewards=True. By
    construction the returned advantages have mean 0 and (population) std 1: that
    is what centring on the group mean and dividing by the group spread DOES.
    A degenerate group (all rewards equal, std 0) yields all-zero advantages - no
    signal, correctly, because no completion beat the others."""
    m = mean(rewards)
    s = pstdev(rewards)                 # population std, ddof=0
    if s < eps:
        return [0.0 for _ in rewards], m, s
    return [(r - m) / s for r in rewards], m, s


# --------------------------------------------------------------------------- #
# VERIFIABLE REWARD FUNCTIONS - the "R" in RLVR. A PROGRAM scores each completion;
# no human, no learned reward model. These follow the TRL reward-function contract
# (verified July-2026): a reward fn is called with keyword args `prompts`,
# `completions`, and every REMAINING dataset column (here `answer` / `schema`),
# and returns list[float], ONE score per completion. For a standard (text)
# dataset `completions` is list[str]; for a conversational one it is a list of
# message-lists and you would read `c[0]["content"]` instead. We use the text form.
# --------------------------------------------------------------------------- #

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_IS = re.compile(r"answer\s*(?:is|:)\s*([-+]?\d[\d,]*\.?\d*)", re.IGNORECASE)
_LAST_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_final_answer(text):
    """Pull the model's committed answer out of a completion. Priority: \\boxed{},
    then 'answer is/: N', then the last number in the string. Returns a normalized
    numeric string, or None if the completion never commits to a number."""
    m = _BOXED.search(text)
    if m:
        cand = m.group(1)
    else:
        m = _ANSWER_IS.search(text)
        if m:
            cand = m.group(1)
        else:
            nums = _LAST_NUM.findall(text)
            cand = nums[-1] if nums else None
    if cand is None:
        return None
    cand = cand.replace(",", "").strip()
    try:
        f = float(cand)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def _is_formatted(text):
    """Did the completion at least COMMIT to an answer in a checkable format?
    (a \\boxed{} or an explicit 'answer is ...'). Distinguishes amber from red."""
    return bool(_BOXED.search(text) or _ANSWER_IS.search(text))


def accuracy_reward(prompts=None, completions=None, answer=None, **kwargs):
    """Math/reasoning grader. Full credit if the extracted final answer matches the
    ground truth; small format credit if it committed to a well-formed answer that
    is wrong; zero otherwise. `answer` arrives as a per-example list (the dataset
    column), one ground truth per prompt in the group's batch order."""
    scores = []
    for i, comp in enumerate(completions):
        gold = _norm_num(answer[i]) if answer is not None else None
        got = extract_final_answer(comp)
        if gold is not None and got is not None and got == gold:
            scores.append(R_CORRECT)
        elif _is_formatted(comp):
            scores.append(R_FORMATTED_WRONG)
        else:
            scores.append(R_FAIL)
    return scores


def json_schema_reward(prompts=None, completions=None, schema=None, **kwargs):
    """Schema validator. Full credit if the completion parses as a JSON object
    containing every required key; small credit if it is valid JSON of the wrong
    shape; zero if it does not parse at all. `schema` is a per-example list of
    required-key lists. This is the config-parses / JSON-matches / SQL-runs class
    of reward the page tells him his own domain probably has."""
    scores = []
    for i, comp in enumerate(completions):
        required = schema[i] if schema is not None else []
        obj = _try_json(comp)
        if obj is None:
            scores.append(R_FAIL)
        elif isinstance(obj, dict) and all(k in obj for k in required):
            scores.append(R_CORRECT)
        else:
            scores.append(R_FORMATTED_WRONG)
    return scores


REWARDS = {"accuracy": accuracy_reward, "json_schema": json_schema_reward}


def _norm_num(x):
    try:
        f = float(str(x).replace(",", "").strip())
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return None


def _try_json(text):
    """Best-effort JSON parse: whole string first, then the first {...} span (models
    love to wrap JSON in prose). Returns the object or None."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Tiny built-in datasets - prove the loop is wired, teach nothing durable. Real
# use points --data at your own JSONL with a verifiable ground truth per row.
# --------------------------------------------------------------------------- #
FALLBACK_ACCURACY = [
    {"prompt": "What is 17 + 25? Put the final answer in \\boxed{}.", "answer": "42"},
    {"prompt": "A box holds 6 rows of 7 apples. How many apples? \\boxed{} the answer.", "answer": "42"},
    {"prompt": "What is 100 - 37? Put the final answer in \\boxed{}.", "answer": "63"},
    {"prompt": "What is 8 times 9? Put the final answer in \\boxed{}.", "answer": "72"},
    {"prompt": "There are 3 bags of 15 marbles. Total marbles? \\boxed{} it.", "answer": "45"},
    {"prompt": "What is 144 divided by 12? Put the final answer in \\boxed{}.", "answer": "12"},
]
FALLBACK_JSON = [
    {"prompt": 'Return a JSON object with keys "name" and "age" for a 30-year-old named Sam.',
     "schema": ["name", "age"]},
    {"prompt": 'Return JSON with keys "city" and "country" for Paris, France.',
     "schema": ["city", "country"]},
    {"prompt": 'Return a JSON object with keys "sku" and "price" for a $5 widget SKU A1.',
     "schema": ["sku", "price"]},
]


# --------------------------------------------------------------------------- #
# THE SELF-TEST - any laptop, no GPU, no torch, no download. The path the build
# contract requires be executed locally with real output pasted.
# --------------------------------------------------------------------------- #

def self_test():
    print("=" * 72)
    print("SELF-TEST (no GPU) - the GRPO arithmetic and the verifiable rewards")
    print("=" * 72)

    # (1) the frozen quiz numeric - the assertion the page's quiz asks for.
    a_pass = (QUIZ_R_PASS - QUIZ_MEAN) / QUIZ_STD
    a_fail = (QUIZ_R_FAIL - QUIZ_MEAN) / QUIZ_STD
    print("  Frozen quiz (page 52): group mean 0.4, std 0.5")
    print(f"    passed completion r_i=1 -> A_i = (1 - 0.4)/0.5 = {a_pass:.4f}  "
          f"(nudge TOWARD, {a_pass:.1f} std above the group)")
    print(f"    failed completion r_i=0 -> A_i = (0 - 0.4)/0.5 = {a_fail:.4f}  "
          f"(nudge AWAY, {abs(a_fail):.1f} std below)")
    assert abs(a_pass - QUIZ_A_PASS) < 1e-9, f"A_i(pass) must be 1.2, got {a_pass}"
    assert abs(a_fail - QUIZ_A_FAIL) < 1e-9, f"A_i(fail) must be -0.8, got {a_fail}"
    print()

    # (2) a LIVE group: advantages centre on zero with unit spread, always.
    group = [1.0, 1.0, 1.0, 0.0, 0.0]          # 3 of 5 completions passed the verifier
    adv, m, s = group_advantages(group)
    print(f"  Live group of G={len(group)} verifier rewards: {group}")
    print(f"    mean {m:.4f}  std(pop) {s:.4f}")
    print(f"    advantages A_i: {[round(a, 4) for a in adv]}")
    print(f"    sum(A_i) = {sum(adv):.2e}  (~0: subtracting the mean guarantees it)")
    print(f"    std(A_i) = {pstdev(adv):.4f} (=1: dividing by the spread guarantees it)")
    assert abs(sum(adv)) < 1e-9, f"advantages must sum to ~0, got {sum(adv):.2e}"
    assert abs(pstdev(adv) - 1.0) < 1e-9, f"advantage std must be 1, got {pstdev(adv)}"
    # a degenerate group (nobody beat anybody) yields no signal - correctly zero.
    flat_adv, _, flat_s = group_advantages([0.0, 0.0, 0.0])
    assert flat_s == 0.0 and all(a == 0.0 for a in flat_adv), "flat group must give zero advantage"
    print("    degenerate group (all-equal rewards) -> all-zero advantage (no signal). OK")
    print()

    # (3) the ACCURACY reward, unit-tested on fixed strings - a program, not a labeller.
    comps = [
        "Let me add. 17 + 25 = 42. \\boxed{42}",    # correct
        "The answer is 43.",                          # formatted, wrong
        "hmm, forty-two-ish",                          # unformatted, wrong
        "6 * 7 = 42, so \\boxed{42}",                 # correct
    ]
    gold = ["42", "42", "42", "42"]
    r = accuracy_reward(completions=comps, answer=gold)
    print(f"  accuracy_reward on 4 fixed completions (gold 42): {r}")
    assert r == [R_CORRECT, R_FORMATTED_WRONG, R_FAIL, R_CORRECT], f"accuracy reward drifted: {r}"
    assert extract_final_answer("...\\boxed{42}") == "42"
    assert extract_final_answer("the answer is 1,024") == "1024"
    assert extract_final_answer("no number here") is None
    print("    correct->1.0, formatted-wrong->0.1, unformatted-wrong->0.0. OK")
    print()

    # (4) the JSON-SCHEMA reward, unit-tested - the 'does it parse?' class of checker.
    jcomps = [
        '{"name": "Sam", "age": 30}',                 # valid + required keys
        'Here you go: {"name": "Sam"}',                # valid JSON, missing 'age'
        'name = Sam, age = 30',                        # not JSON
    ]
    jschema = [["name", "age"], ["name", "age"], ["name", "age"]]
    jr = json_schema_reward(completions=jcomps, schema=jschema)
    print(f"  json_schema_reward on 3 fixed completions: {jr}")
    assert jr == [R_CORRECT, R_FORMATTED_WRONG, R_FAIL], f"json reward drifted: {jr}"
    print("    valid+keys->1.0, valid-wrong-shape->0.1, not-JSON->0.0. OK")
    print()

    # (5) the 8B-by-substitution anchor (page 52): the four values you swap back in.
    a = ANCHOR_8B
    print("  8B anchor config (constants §1.1, VP) - swap these into any page-49 formula")
    print(f"    d_model={a['d_model']}  L={a['L']}  H_kv={a['H_kv']}  d_ff={a['d_ff']}")
    print(f"    per-token KV at bf16 = 2*L*H_kv*d_head*2 = "
          f"2*{a['L']}*{a['H_kv']}*128*2 = {2*a['L']*a['H_kv']*128*2:,} B = 144 KiB (constants §4)")
    assert 2 * a["L"] * a["H_kv"] * 128 * 2 == 147_456, "8B per-token KV must be 147,456 B = 144 KiB"
    print("    (the 0.6B is smaller on every axis - which is why it fits ~2 hr, page 52)")
    print()

    print("  self-checks passed: A_i = 1.2 / -0.8 ; advantages centre 0, spread 1 ;")
    print("  both verifiable rewards score correctly ; 8B anchor KV = 144 KiB. No GPU touched.")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Version stamp - a support request begins with real versions, not remembered ones.
# --------------------------------------------------------------------------- #

def stamp():
    import torch
    line = f"torch {torch.__version__}"
    for mod in ("transformers", "trl", "peft", "vllm"):
        try:
            line += f" · {mod} {__import__(mod).__version__}"
        except Exception:
            line += f" · {mod} MISSING"
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        line += f" · {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} · CUDA {torch.version.cuda}"
    print(f"  [{line}]")


# --------------------------------------------------------------------------- #
# Data loading - JSONL rows carry the prompt AND the verifiable ground truth
# (`answer` for accuracy, `schema` for json_schema). Extra columns flow through
# to the reward function as kwargs - that is the TRL reward contract.
# --------------------------------------------------------------------------- #

def load_records(data_path, reward_name):
    gt_key = "answer" if reward_name == "accuracy" else "schema"
    if data_path is None:
        fallback = FALLBACK_ACCURACY if reward_name == "accuracy" else FALLBACK_JSON
        print(f"  no --data given; using the {len(fallback)}-example built-in fallback "
              f"(proves the loop is wired, teaches nothing durable)")
        return [dict(r) for r in fallback]
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" not in row or gt_key not in row:
                raise ValueError(
                    f"{data_path}:{lineno}: --reward {reward_name} needs keys "
                    f"'prompt' and '{gt_key}', got {sorted(row.keys())}")
            records.append(row)
    if not records:
        raise ValueError(f"{data_path}: no records found")
    print(f"  loaded {len(records)} examples from {data_path}")
    return records


# --------------------------------------------------------------------------- #
# A quick, program-only accuracy probe (before vs after) - reuses the SAME
# verifiable reward, so the "did it get better" number is itself checkable.
# --------------------------------------------------------------------------- #

def measure_accuracy(model, tokenizer, records, reward_name, n=None, max_new_tokens=256):
    import torch
    reward_fn = REWARDS[reward_name]
    gt_key = "answer" if reward_name == "accuracy" else "schema"
    rows = records if n is None else records[:n]
    prompts = [r["prompt"] for r in rows]
    gts = [r[gt_key] for r in rows]
    outs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        outs.append(tokenizer.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
    scores = reward_fn(completions=outs, **{gt_key: gts})
    solved = sum(1 for s in scores if s >= R_CORRECT)
    return solved / len(rows), scores


# --------------------------------------------------------------------------- #
# THE REAL RUN - GRPO via the current TRL 1.8 API, in the cu130 container.
# API transcribed from the verified July-2026 docs (constants §7.2, brief-llm §GRPO);
# NEVER an API from memory. Line-by-line desk-check notes are inline.
# --------------------------------------------------------------------------- #

def run_grpo(args):
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    print("=" * 72)
    print(f"RUNG 7 - GRPO + RLVR ({args.reward} reward) on {args.model}")
    print("=" * 72)
    stamp()

    # The one TRL v1 default the page makes you memorize (constants §7.2): assert the
    # LIVE default is "colocate" so a silent library regression fails loudly here.
    default_mode = GRPOConfig(output_dir=args.output_dir).vllm_mode
    print(f"  GRPOConfig.vllm_mode default (this trl build): {default_mode!r}")
    assert default_mode == "colocate", (
        f"GRPOConfig.vllm_mode default must be 'colocate' at trl>=1.8 (constants §7.2); "
        f"got {default_mode!r} - you are on a pre-1.8 trl (the v0 default was 'server').")

    reward_fn = REWARDS[args.reward]
    records = load_records(args.data, args.reward)
    ds = Dataset.from_list(records)

    # num_generations (the group size G) MUST divide the global batch. We set the
    # per-device batch = G so one optimizer step is exactly one group - the simplest
    # arrangement and a real TRL footgun if you get it wrong.
    G = args.num_generations
    print(f"\n  group size G (num_generations) = {G}; per-device batch set to G "
          f"so one step = one group (TRL requires batch % G == 0)")

    use_vllm = not args.no_vllm
    if use_vllm:
        print(f"  generation: vLLM, vllm_mode={args.vllm_mode!r} - REQUIRES the cu130 container")
        print(f"    (constants §6.9/D-10j: the PyPI aarch64 vLLM wheel pins libcudart.so.12 and")
        print(f"     import-traps on CUDA-13-only DGX OS; the container ships its own CUDA-13 userspace)")
    else:
        print("  generation: HuggingFace .generate() fallback (--no-vllm) - slower, but needs")
        print("    no container. Use it to smoke-test the loop; switch to vLLM for the real 2-hr run.")

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        model_init_kwargs={"dtype": torch.bfloat16},   # same fp32 trap as SFT (constants §7.3):
                                                        # a string model + no dtype => fp32. Pin bf16.
        num_generations=G,                              # G completions per prompt - the "group"
        per_device_train_batch_size=G,                 # one group per step (batch % G == 0)
        gradient_accumulation_steps=args.accum,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,                   # >0: the group MUST be diverse or std->0
        num_train_epochs=args.epochs,
        learning_rate=args.lr,                          # GRPO runs low (~1e-6): it reweights, not relearns
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        use_vllm=use_vllm,                              # colocate needs this True (constants §7.2)
        vllm_mode=args.vllm_mode,                       # "colocate": gen+train in one process (single box)
        logging_steps=1,
        report_to=["trackio"] if args.trackio else "none",  # v5 default "none" - opt in for the curve
        seed=args.seed,
        save_strategy="no" if args.no_save else "epoch",
    )
    # Guard the two traps so a future edit that drops them fails loudly, not silently.
    assert cfg.model_init_kwargs.get("dtype") is torch.bfloat16, \
        "model_init_kwargs dtype must be bf16 (constants §7.3) or GRPO trains in fp32 and OOMs."
    assert cfg.num_generations == cfg.per_device_train_batch_size, \
        "per-device batch must equal num_generations here (batch % G == 0)."

    # --- an optional LoRA path so the 0.6B run fits comfortably (state, not model) --- #
    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
            target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
        print(f"  LoRA on (r={args.rank}, all-linear): trains adapter state only, not the base.")

    # --- before: the program-scored baseline accuracy (same verifiable reward) ------ #
    baseline_acc = None
    if not args.skip_eval:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("\n  measuring BEFORE accuracy (greedy, program-scored)...")
        tok0 = AutoTokenizer.from_pretrained(args.model)
        m0 = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
        baseline_acc, _ = measure_accuracy(m0, tok0, records, args.reward, n=args.eval_n)
        print(f"    BEFORE: {baseline_acc*100:.1f}% solved on {min(args.eval_n, len(records))} prompts")
        del m0
        torch.cuda.empty_cache()

    # --- SINGLE PATH: string model + reward_funcs + args. TRL builds/samples/scores/steps. --- #
    trainer = GRPOTrainer(
        model=args.model,               # a STRING - TRL loads it (bf16 via model_init_kwargs)
        reward_funcs=reward_fn,         # the VERIFIABLE reward - list[float], one per completion
        args=cfg,
        train_dataset=ds,
        peft_config=peft_config,        # None for full GRPO; a LoraConfig for the adapter path
    )

    print("\n  PREDICTION (before training): the policy will be nudged toward completions the")
    print("  verifier scores above the GROUP mean and away from those below - no critic, no")
    print("  labels. Each nudge is weighted by A_i = (r_i - mean)/std (the 1.2 you asserted).")
    print("  SAFETY: this now saturates the GPU for ~2 hr; if ComfyUI is up it WILL contend.\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    print("\n  MEASURED (after training):")
    print(f"    wall-clock : {dt:.1f} s = {dt/60:.1f} min")
    if torch.cuda.is_available():
        print(f"    peak mem   : {torch.cuda.max_memory_allocated()/GiB:.2f} GiB (max_memory_allocated)")

    # the reward curve from the trainer's own log - did the group's mean reward climb?
    rk = [(h["step"], h.get("reward")) for h in trainer.state.log_history if "reward" in h]
    if rk:
        print("  mean group reward (step: reward):")
        for step, rv in rk[:: max(1, len(rk) // 8)]:
            print(f"    {step:>6}: {rv:.4f}")
        print(f"    first {rk[0][1]:.4f} -> last {rk[-1][1]:.4f}")

    # --- after: re-run the SAME program-scored accuracy on the trained policy ------- #
    if not args.skip_eval and baseline_acc is not None:
        after_acc, _ = measure_accuracy(trainer.model, trainer.processing_class, records,
                                        args.reward, n=args.eval_n)
        print(f"\n  task accuracy: BEFORE {baseline_acc*100:.1f}%  ->  AFTER {after_acc*100:.1f}%  "
              f"(delta {(after_acc-baseline_acc)*100:+.1f} pts, program-verified)")
        print("  Reminder (page 52): RLVR REWEIGHTS checkable behaviours the base already has;")
        print("  it does NOT install facts. Where a program can score the output, it is remarkable;")
        print("  where it can't, it has nothing to optimize.")

    if not args.no_save:
        trainer.save_model()
        print(f"  policy/adapter saved to {args.output_dir}")
    print("=" * 72)
    print("RUNG 7 complete - the 2026 reasoning frontier, on a 240 W box you own.")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the GRPO arithmetic + reward unit checks (no GPU, no torch)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="HF id or local path (constants §1.4: 0.6B/1.7B for the <2 hr stretch)")
    ap.add_argument("--reward", choices=sorted(REWARDS), default="accuracy",
                    help="verifiable reward function: accuracy (math grader) or json_schema")
    ap.add_argument("--data", default=None,
                    help="JSONL with 'prompt' + ('answer' for accuracy | 'schema' for json_schema)")
    ap.add_argument("--num-generations", type=int, default=8,
                    help="group size G (brief-llm: TRL default 8-16; typical 8-64)")
    ap.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"],
                    help="TRL 1.8 default 'colocate' (gen+train one process; constants §7.2)")
    ap.add_argument("--no-vllm", action="store_true",
                    help="use HF .generate() instead of vLLM (slower; needs no container)")
    ap.add_argument("--lora", action="store_true", help="train a LoRA adapter instead of full GRPO")
    ap.add_argument("-r", "--rank", type=int, default=16, help="LoRA rank if --lora (default 16)")
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="GRPO LR - low; it reweights, not relearns (default 1e-6)")
    ap.add_argument("--epochs", type=float, default=1, help="training epochs (default 1)")
    ap.add_argument("--accum", type=int, default=1, help="grad accumulation steps (default 1)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature - MUST be >0 or the group collapses (std->0)")
    ap.add_argument("--max-prompt-length", type=int, default=256, help="max prompt tokens")
    ap.add_argument("--max-completion-length", type=int, default=256, help="max generated tokens")
    ap.add_argument("--eval-n", type=int, default=6, help="prompts in the before/after accuracy probe")
    ap.add_argument("--skip-eval", action="store_true", help="skip the before/after accuracy probe")
    ap.add_argument("--trackio", action="store_true", help="log the reward curve to Trackio")
    ap.add_argument("--seed", type=int, default=42, help="seed (determinism)")
    ap.add_argument("--output-dir", default="./qwen3-0.6b-grpo", help="where the policy is written")
    ap.add_argument("--no-save", action="store_true", help="do not write the policy to disk")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    try:
        import torch
    except ImportError:
        print("torch not importable. This capstone runs INSIDE the cu130 container")
        print("(vllm/vllm-openai:cu130-nightly) - NEVER ComfyUI's venv, NEVER a raw pip vllm")
        print("(the CUDA-12 wheel import-traps on DGX OS - constants §6.9). Start the container,")
        print("then rerun. Or try --self-test: it needs no GPU, no torch, and still checks every")
        print("frozen number (A_i = 1.2, the reward functions, the 8B anchor).")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available - GRPO generates inside the loop and needs a GPU. Run --self-test")
        print("for the arithmetic, or move to the Spark IN the cu130 container (mindful of ComfyUI -")
        print("HARD SAFETY RULE: he runs it there; this script is desk-checked here).")
        sys.exit(1)

    run_grpo(args)


if __name__ == "__main__":
    main()
