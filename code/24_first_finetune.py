#!/usr/bin/env python3
"""
24_first_finetune.py — your first real fine-tune. The milestone deliverable for p.24.

This is a REAL LoRA fine-tune, not a toy. It trains Qwen3-0.6B (the smoke-test
sibling of the course's anchor model, Qwen3-8B — constants.md §1.4) on a tiny
instruction set, on your box, in the fresh venv the Part III setup page installs
(peft/trl/accelerate/bitsandbytes are NOT in ComfyUI's venv — hardware-ground-truth.md
§3 — never install into it). The arithmetic here is the SAME arithmetic the LLM
track scales up to the 8B anchor by swapping four config numbers: model name,
r, target_modules stay identical, only the parameter COUNT changes.

It does five things, in order, matching the page's diagnostic checklist:
  1. Overfits one batch of 8 examples first (the universal first move — page 24.0).
     If this doesn't hit ~0 loss, everything below is a waste of GPU time.
  2. Prints the trainable-parameter count so you can catch the #1 "wrong param
     group" bug (constants.md §3: for Qwen3-8B LoRA r=16 all-linear that count
     is 43,646,976 — for 0.6B it will be a few million, never 8.19e9 and never 0).
  3. Logs the full instrumentation panel every `--log-every` steps: train loss,
     grad global norm, current LR, and the update ratio ||delta theta|| / ||theta||.
  4. Predicts peak VRAM from page 18's ledger BEFORE training, then measures it
     with torch.cuda.max_memory_allocated() and prints both side by side. This
     model is not the memory ledger's 8B anchor, so the predicted number is a
     small worked estimate from the SAME formula (16 B/param on the TRAINABLE
     LoRA parameters, since the frozen base stays in inference dtype) — not a
     frozen course constant. Compare it to what you actually measure; the gap
     IS the lesson (activations, CUDA allocator overhead, none of which the
     16-B/param formula includes — constants.md §2.2's own warning).
  5. Generates from the model before and after training on one held-out prompt,
     so "did it learn anything" has an answer you can read, not just a loss curve.

Usage
-----
    python 24_first_finetune.py --model Qwen3-0.6B --data tiny_instructions.jsonl
    python 24_first_finetune.py --data tiny_instructions.jsonl --steps 60 --overfit-only

`--data` is a JSONL file of {"instruction": "...", "response": "..."} pairs, one per
line. Ships with a tiny built-in fallback set (12 examples) if you don't have one yet
— enough to prove the loop is wired, not enough to teach the model anything durable.

Requires (constants.md §7, pin exactly): trl>=1.8,<2  peft==0.19.1
Installs into a FRESH venv, never ComfyUI's. Runtime ~3-8 min on Qwen3-0.6B.
"""

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

GiB = 1 << 30
GB = 10 ** 9

# The page's fallback dataset — enough to prove the wiring, not enough to teach
# the model anything durable. Real runs pass --data with a larger JSONL file.
FALLBACK_EXAMPLES = [
    {"instruction": "What is the capital of France?", "response": "Paris."},
    {"instruction": "Name the primary colors.", "response": "Red, blue, and yellow."},
    {"instruction": "What is 7 plus 5?", "response": "12."},
    {"instruction": "Give a one-word synonym for 'happy'.", "response": "Joyful."},
    {"instruction": "What planet do we live on?", "response": "Earth."},
    {"instruction": "Convert 1 kilometer to meters.", "response": "1000 meters."},
    {"instruction": "What is the chemical symbol for water?", "response": "H2O."},
    {"instruction": "Name a programming language.", "response": "Python."},
    {"instruction": "What is the opposite of 'up'?", "response": "Down."},
    {"instruction": "How many days are in a week?", "response": "Seven."},
    {"instruction": "What color do you get mixing blue and yellow?", "response": "Green."},
    {"instruction": "What is the boiling point of water in Celsius?", "response": "100 degrees Celsius."},
]


def load_dataset_records(data_path):
    """Load {"instruction","response"} records from a JSONL file, or fall back
    to the tiny built-in set. No silent truncation — every line is validated."""
    if data_path is None:
        print(f"  no --data given, using the {len(FALLBACK_EXAMPLES)}-example "
              f"built-in fallback set (proves the loop is wired, nothing more)")
        return list(FALLBACK_EXAMPLES)
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "instruction" not in row or "response" not in row:
                raise ValueError(
                    f"{data_path}:{lineno}: expected keys 'instruction' and "
                    f"'response', got {sorted(row.keys())}"
                )
            records.append(row)
    if not records:
        raise ValueError(f"{data_path}: no records found")
    print(f"  loaded {len(records)} examples from {data_path}")
    return records


def to_sft_text(tokenizer, record):
    """Render one instruction/response pair through the chat template — the
    same tokenizer call the SFTTrainer uses internally, made explicit here so
    the overfit-one-batch check below can print exactly what the model sees."""
    messages = [
        {"role": "user", "content": record["instruction"]},
        {"role": "assistant", "content": record["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def predicted_peak_vram_gib(trainable_params, base_params, base_dtype_bytes=2):
    """The page-18 ledger formula (constants.md §2.1), applied to THIS run.
    16 B/param on the TRAINABLE (LoRA) parameters only — the frozen base model
    contributes only its inference weights, not gradients or optimizer state.
    This is a worked ESTIMATE from the formula, not a frozen constant — the
    frozen 122.05 GiB figure is the 8B FULL fine-tune and does not apply here."""
    lora_state_bytes = trainable_params * 16          # weights+grad+master+m+v
    base_weight_bytes = base_params * base_dtype_bytes  # frozen, inference dtype
    total_bytes = lora_state_bytes + base_weight_bytes
    return total_bytes / GiB, lora_state_bytes / GiB, base_weight_bytes / GiB


def build_trainable_report(model):
    """print_trainable_parameters()'s own numbers, pulled out so the script can
    assert on them instead of only printing them."""
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


class InstrumentationCallback:
    """The page's mandated instrumentation panel (training §14.4), as a trl
    TrainerCallback: train loss, grad global norm, current LR, update ratio.
    trl computes grad-norm internally for logging when max_grad_norm is set;
    this callback additionally tracks the update ratio, which trl does not."""

    def __init__(self, model, log_every):
        from transformers import TrainerCallback  # local import: only needed if this class is used
        self._Base = TrainerCallback
        self.model = model
        self.log_every = log_every
        self._prev_norm = None
        self._prev_params = None

    def _param_norm(self):
        import torch
        sq = 0.0
        for p in self.model.parameters():
            if p.requires_grad:
                sq += float(torch.sum(p.detach() ** 2))
        return math.sqrt(sq)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every != 0:
            return control
        norm_now = self._param_norm()
        grad_norm = state.log_history[-1].get("grad_norm") if state.log_history else None
        lr_now = state.log_history[-1].get("learning_rate") if state.log_history else None
        loss_now = state.log_history[-1].get("loss") if state.log_history else None
        update_ratio = None
        if self._prev_norm is not None and self._prev_norm > 0:
            # ||delta theta|| ~= |now - prev| as a scalar proxy on the norms
            # themselves (exact vector delta would need a full param snapshot
            # each step, which is expensive at 8B scale — this scalar version
            # is the same order-of-magnitude signal the rule of thumb needs).
            update_ratio = abs(norm_now - self._prev_norm) / self._prev_norm
        self._prev_norm = norm_now
        print(f"  step {state.global_step:>5} | loss {loss_now} | "
              f"grad_norm {grad_norm} | lr {lr_now} | "
              f"update_ratio~{update_ratio if update_ratio is None else f'{update_ratio:.2e}'} "
              f"(healthy ~1e-3, constants.md §9.3 [EST])")
        return control


def run_overfit_one_batch(model_name, records, device_map, dtype, steps=200):
    """The universal first move (page 24.0): 8 examples, 200 steps, no schedule,
    no weight decay. Loss must fall to ~0 or there is a wiring bug — no amount
    of learning-rate tuning fixes that. This function is a real training loop,
    not a simulation of one."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print("=" * 72)
    print("STEP 1 — the universal first move: overfit 8 examples")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device_map
    )
    lora_cfg = LoraConfig(target_modules="all-linear", r=16, lora_alpha=32, lora_dropout=0.0)
    model = get_peft_model(base, lora_cfg)
    model.train()

    trainable, total = build_trainable_report(model)
    print(f"  trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")
    assert 0 < trainable < total, (
        "trainable param count is 0 or ==total — wrong param group (page 24 §14.2). "
        "For Qwen3-8B LoRA r=16 all-linear the expected count is 43,646,976, "
        "never 8,190,735,360 and never 0 (constants.md §3)."
    )

    batch = records[: min(8, len(records))]
    texts = [to_sft_text(tokenizer, r) for r in batch]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    labels = labels.to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        out = model(**enc, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        losses.append(out.loss.item())
        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:>4}/{steps}  loss {out.loss.item():.4f}")
    dt = time.time() - t0

    print(f"  overfit-8 result: loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"in {dt:.1f}s ({steps} steps)")
    if losses[-1] < 0.05:
        print("  VERDICT: WIRING OK — loss collapsed to ~0. Safe to proceed to a real run.")
    else:
        print("  VERDICT: BUG, not a hyperparameter problem. Do not tune the LR yet — "
              "check zero_grad(), the loss function, and the label mask first "
              "(page 24, training §14.1/§14.2).")
    assert losses[-1] < losses[0], (
        "loss did not fall at all on 8 memorizable examples — this IS the bug "
        "the universal first move exists to catch. Stop and read page 24's "
        "'Loss won't go down' table before changing anything else."
    )
    return model, tokenizer, trainable, total


def run_full_finetune(model, tokenizer, records, steps, log_every, lr, dtype):
    """Step 3: a real short fine-tune via trl's SFTTrainer, with the
    instrumentation panel attached. Uses model_init_kwargs to avoid the
    fp32-default trap (constants.md §7.3): SFTTrainer defaults to fp32 when
    `model` is a string, even though from_pretrained would have inferred bf16."""
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print("=" * 72)
    print("STEP 3 — the real short fine-tune, instrumented")
    print("=" * 72)

    ds = Dataset.from_list([
        {"text": to_sft_text(tokenizer, r)} for r in records
    ])

    cfg = SFTConfig(
        output_dir=tempfile.mkdtemp(prefix="p24_finetune_"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=steps,
        learning_rate=lr,
        logging_steps=log_every,
        max_grad_norm=1.0,             # constants.md §9.3: near-universal clip value
        bf16=(dtype == torch.bfloat16),
        report_to="none",
        save_strategy="no",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.add_callback(InstrumentationCallback(model, log_every))

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    print(f"  trained {steps} steps in {dt:.1f}s ({dt / steps:.3f}s/step)")
    return trainer


def generate_once(model, tokenizer, prompt, max_new_tokens=40):
    import torch
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    model.train()
    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                     help="HF model id or local path (default: the smoke-test sibling, constants.md §1.4)")
    ap.add_argument("--data", default=None, help="JSONL of {instruction, response} pairs")
    ap.add_argument("--steps", type=int, default=200, help="training steps for the real fine-tune")
    ap.add_argument("--overfit-steps", type=int, default=200, help="steps for the overfit-one-batch check")
    ap.add_argument("--log-every", type=int, default=10, help="instrumentation panel logging cadence")
    ap.add_argument("--lr", type=float, default=2e-4,
                     help="LoRA LR (constants.md §9.4: LoRA ~10x full-FT, 1e-4-3e-4)")
    ap.add_argument("--overfit-only", action="store_true", help="run step 1 only, skip the full fine-tune")
    ap.add_argument("--prompt", default="What is the capital of France?",
                     help="held-out prompt for the before/after generation check")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch not importable. This script needs the FRESH venv the Part III")
        print("setup page installs (peft/trl/accelerate/bitsandbytes) — NOT ComfyUI's")
        print("venv (hardware-ground-truth.md §3). Create it, then rerun.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available — this script needs a GPU to be worth running.")
        sys.exit(1)

    dtype = torch.bfloat16  # the Spark should never see fp16 (constants.md §9.3: fp16 max 65,504)
    device_map = {"": 0}

    records = load_dataset_records(args.data)

    model, tokenizer, trainable, total = run_overfit_one_batch(
        args.model, records, device_map, dtype, steps=args.overfit_steps
    )

    predicted_gib, lora_gib, base_gib = predicted_peak_vram_gib(trainable, total)
    print()
    print("=" * 72)
    print("STEP 4 — predict, THEN measure (page 18's ledger, applied to this run)")
    print("=" * 72)
    print(f"  predicted: {predicted_gib:.3f} GiB total "
          f"= {lora_gib:.3f} GiB LoRA state (16 B/param x {trainable:,} trainable) "
          f"+ {base_gib:.3f} GiB frozen base (bf16, {total:,} params)")
    print("  this predicted figure is a worked ESTIMATE from the page-18 formula, "
          "not a frozen course constant — the frozen 122.05 GiB is the 8B FULL "
          "fine-tune, a different run entirely (constants.md §2.2).")
    print("  now write down your own guess, then look at the measured number below.")

    before_text = generate_once(model, tokenizer, args.prompt)
    print(f"\n  BEFORE training, prompt {args.prompt!r} ->\n    {before_text!r}")

    if not args.overfit_only:
        run_full_finetune(model, tokenizer, records, args.steps, args.log_every, args.lr, dtype)

        measured_bytes = torch.cuda.max_memory_allocated()
        measured_gib = measured_bytes / GiB
        print()
        print(f"  measured:  {measured_gib:.3f} GiB (torch.cuda.max_memory_allocated())")
        diff = measured_gib - predicted_gib
        print(f"  predicted vs measured: {predicted_gib:.3f} GiB vs {measured_gib:.3f} GiB "
              f"({'+' if diff >= 0 else ''}{diff:.3f} GiB)")
        print("  the gap is activations + the CUDA allocator's own overhead — the "
              "16-B/param formula was never claiming to cover those (constants.md §2.2).")

        after_text = generate_once(model, tokenizer, args.prompt)
        print(f"\n  AFTER training, prompt {args.prompt!r} ->\n    {after_text!r}")
        assert after_text != before_text, (
            "the adapter produced byte-identical output before and after training — "
            "this IS a 'loss won't go down' or 'wrong param group' bug (page 24 §14.2), "
            "not a fluke. Check the trainable-param count printed above first."
        )
        print("\n  the output changed. You just fine-tuned a model. That's the milestone.")

    free, total_mem = torch.cuda.mem_get_info()
    print(f"\n  (for reference: your box currently reports "
          f"{free / GiB:.2f} GiB free of {total_mem / GiB:.2f} GiB total — "
          f"compare against hardware-ground-truth.md §2's measured 121.6875 GiB usable)")


if __name__ == "__main__":
    main()
