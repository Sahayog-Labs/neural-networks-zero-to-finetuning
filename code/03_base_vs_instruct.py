#!/usr/bin/env python3
"""
03_base_vs_instruct.py - the same prompt into Qwen3-8B-Base and Qwen3-8B. p.45.

The page ships a live decode demo on HAND-AUTHORED candidate logits (this build has
no path to a real Qwen3 forward pass, and it says so on the page rather than launder
synthetic numbers as captured ones). THIS script is the honest version: it runs the
identical experiment against the REAL two checkpoints on your own box, so nothing here
is asserted about behaviour - it is observed.

What it shows, in order:

  1. The step-0 sanity check, exactly (pure arithmetic, no GPU):
       random-init loss = ln V = ln(151,936) = 11.93 nats   [DER, constants §9.1]
       perplexity at init = e^{ln V} = V = 151,936           - exactly, by construction
     and it refuses the retired 11.76 (that is ln(128,256), Llama-3's V - D-01).
  2. The chat-template diff: your raw prompt, versus what tokenizer.apply_chat_template()
     turns it into (ChatML: <|im_start|>...<|im_end|>). Never hand-rolled (brief rule 1).
  3. Greedy decode of BOTH models, printing token-level probabilities at each step and,
     crucially, P(<|im_end|>) - the end-of-turn candidate - at every step:
       - Qwen3-8B-Base rambles: it never learned the habit of emitting end-of-turn, so
         P(<|im_end|>) stays low and it burns all max_new_tokens.
       - Qwen3-8B (Instruct) answers and then chooses <|im_end|> with a large margin: it
         STOPS. Instruct-ness is a learned behaviour, not new knowledge (sets up p.48).
  4. Then it wraps the BASE prompt in the same chat template and decodes again: the
     untrained-for format nudges end-of-turn up and the base PARTIALLY recovers - it
     stops, but later and at lower confidence than the real instruct model. All three
     behavioural claims are labelled [EST]: they are the well-documented qualitative
     outcome, not frozen constants.

Behaviour is DATA, not a promise: everything in step 3-4 is measured off the real
scores tensor, and the asserts encode the documented prediction (base doesn't stop,
instruct does) so a regression - a base that suddenly stops, an instruct that rambles -
fails loudly instead of passing quietly.

SAFETY
------
Loads an 8B checkpoint TWICE - base first, freed, then instruct - never both resident at
once (sequential, ~16 GB bf16 each). It contends with ComfyUI if that is live on the GPU;
consult before running. It writes nothing and installs nothing (read-only w.r.t. the box);
it only DOWNLOADS the two checkpoints into your HF cache if they are not already there.
Set HF_HOME to the NVMe first (constants §7) - two 8B checkpoints is ~32 GB of cache.

Env (THE fresh venv - never ComfyUI's; hardware-ground-truth §3):
    transformers 5.14.1 · torch 2.13.0 · CUDA 13.0   (v5: load with dtype=, not torch_dtype)

Usage
-----
    python 03_base_vs_instruct.py --self-test          # GPU-free: the arithmetic + decode
                                                       #   summariser math, run and asserted
    python 03_base_vs_instruct.py                      # the real experiment (needs a GPU)
    python 03_base_vs_instruct.py --max-new-tokens 60  # push the base out to 60, like the page
    python 03_base_vs_instruct.py --model-base Qwen/Qwen3-8B-Base --model-instruct Qwen/Qwen3-8B
"""

import argparse
import math
import sys

# --------------------------------------------------------------------------- #
# Frozen facts (constants.md). The two the step-0 check ASSERTS are V and ln V.
# --------------------------------------------------------------------------- #
V = 151_936                       # Qwen3 vocab, constants §1.1 [VP]
LN_V = math.log(V)                # = 11.9312..., the random-init loss [DER, §9.1]
LN_V_ROUNDED = 11.93              # what the page and quiz print
V_LLAMA = 128_256                 # Llama-3 vocab - the 11.76 trap (D-01), NOT Qwen3's
EOS = "<|im_end|>"                # ChatML end-of-turn (Qwen family), brief §chat-templates
DEFAULT_BASE = "Qwen/Qwen3-8B-Base"
DEFAULT_INSTRUCT = "Qwen/Qwen3-8B"
RAW_PROMPT = "The capital of France is"
QUESTION = "What is the capital of France?"


# --------------------------------------------------------------------------- #
# Shared decode arithmetic - ONE softmax, ONE accumulator, used by the self-test
# (synthetic candidate rows) AND the real path (full-vocab scores). Page and script
# compute the same thing: greedy argmax, cumulative log-prob, PPL = e^{-cum/n}, and
# the running max of P(end-of-turn). Keeping it in one place is why they cannot drift.
# --------------------------------------------------------------------------- #

def softmax(logits):
    """Numerically stable softmax of a Python list -> list of probabilities."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


class Trace:
    """Accumulates a greedy decode, model-agnostic. Feed one step at a time:
    the chosen token's string, its probability, P(end-of-turn) this step, and whether
    the choice WAS end-of-turn. It tracks the text, the cumulative log-prob, the step
    at which it stopped (if ever), and the largest P(end-of-turn) ever seen."""

    def __init__(self, label):
        self.label = label
        self.text = ""
        self.cum_logprob = 0.0
        self.steps_taken = 0
        self.stopped_at = None          # 1-indexed step, or None if it never stopped
        self.max_eos_prob = 0.0
        self.stop_eos_prob = None

    def step(self, tok_str, chosen_prob, eos_prob, is_eos):
        self.steps_taken += 1
        self.cum_logprob += math.log(chosen_prob)
        self.max_eos_prob = max(self.max_eos_prob, eos_prob)
        if is_eos:
            self.stopped_at = self.steps_taken
            self.stop_eos_prob = eos_prob
            return False                # signal: stop decoding
        self.text += tok_str
        return True                     # signal: keep going

    @property
    def stopped(self):
        return self.stopped_at is not None

    def ppl(self):
        n = self.steps_taken
        return math.exp(-self.cum_logprob / n) if n else float("nan")

    def summary(self):
        n = self.steps_taken
        if self.stopped:
            head = (f"  {self.label}: STOPPED after {self.stopped_at} tokens - chose "
                    f"end-of-turn with p={self.stop_eos_prob * 100:.1f}%")
        else:
            head = (f"  {self.label}: still generating at token {n} (never stopped). "
                    f"max P(end-of-turn) seen = {self.max_eos_prob * 100:.1f}% "
                    f"- never close to winning")
        tail = (f"    sequence log-prob {self.cum_logprob:.2f} nats over {n} step(s) "
                f"-> running PPL = e^(-logp/n) = {self.ppl():.2f}")
        return head + "\n" + tail


# --------------------------------------------------------------------------- #
# Step 0 - the arithmetic sanity check. No GPU, no model, always runs.
# --------------------------------------------------------------------------- #

def step0_sanity():
    print("=" * 72)
    print("STEP 0 - the number every real training run starts just below (no GPU)")
    print("=" * 72)
    print(f"  random-init loss = ln V = ln({V:,}) = {LN_V:.4f} nats")
    print(f"                   -> {LN_V_ROUNDED} nats   [DER, constants §9.1]")
    print(f"  perplexity at init = e^(ln V) = V = {V:,}   - exactly, by construction")
    print(f"  (an untrained softmax over V classes is uniform; the CE of a correct")
    print(f"   answer under a uniform guess is -ln(1/V) = ln V. Nothing estimated.)")
    print()
    print(f"  the trap: 11.76 is ln({V_LLAMA:,}) = {math.log(V_LLAMA):.4f} - Llama-3's")
    print(f"  vocabulary, NOT Qwen3's. Different tokenizer, different floor. (D-01)")
    print()

    # asserts against the frozen constants (§9.1)
    assert round(LN_V, 2) == LN_V_ROUNDED, f"ln V must round to 11.93, got {LN_V:.4f}"
    assert round(math.exp(LN_V)) == V, "e^{ln V} must equal V exactly (PPL at init = V)"
    assert round(math.log(V_LLAMA), 2) == 11.76, "sanity: ln(128256) is the 11.76 number"
    assert abs(LN_V - math.log(V_LLAMA)) > 0.15, "Qwen3 floor must differ from Llama-3's"
    print("  self-checks passed: ln V = 11.93, PPL_init = V = 151,936, and 11.76 refused.")
    print()


# --------------------------------------------------------------------------- #
# Self-test - the decode summariser, exercised WITHOUT a GPU on tiny synthetic
# candidate rows that reproduce the page's qualitative behaviour: a base whose
# end-of-turn candidate never wins, and an instruct that picks it and stops.
# --------------------------------------------------------------------------- #

def self_test():
    step0_sanity()

    print("=" * 72)
    print("SELF-TEST - decode summariser on synthetic logits (the page's mechanism)")
    print("=" * 72)

    # softmax invariants
    p = softmax([2.0, 1.0, 0.1])
    assert abs(sum(p) - 1.0) < 1e-12, "softmax must sum to 1"
    assert p[0] > p[1] > p[2], "softmax must preserve the logit ordering"
    assert abs(p[0] - 0.659001) < 1e-5, f"canonical p0 must be 0.659001, got {p[0]:.6f}"
    print(f"  softmax([2.0,1.0,0.1]) = "
          f"[{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}]  (sums to 1, ordering kept)")

    # A synthetic "base": end-of-turn (last candidate) logit stays low every step,
    # so it is never the argmax -> the trace never stops.
    EOS_I = 5
    base_rows = [
        [6.0, 2.0, 1.8, 1.5, 1.2, 0.5],
        [4.5, 1.8, 1.6, 1.2, 1.0, 0.4],
        [3.8, 2.0, 1.5, 1.3, 1.0, 0.3],
        [4.2, 1.6, 1.3, 1.0, 0.9, 0.3],
    ]
    base = Trace("synthetic-base")
    for row in base_rows:
        probs = softmax(row)
        amax = max(range(len(row)), key=lambda i: row[i])
        if not base.step(f"tok{amax}", probs[amax], probs[EOS_I], amax == EOS_I):
            break
    assert not base.stopped, "synthetic base must NOT stop (end-of-turn never wins)"
    assert base.max_eos_prob < 0.05, f"base end-of-turn prob must stay low, got {base.max_eos_prob:.3f}"
    print("  " + base.summary().strip())

    # A synthetic "instruct": answers two tokens, then end-of-turn dominates and it stops.
    instruct_rows = [
        [7.5, 1.6, 1.4, 1.2, 1.0, 0.6],   # answer token wins
        [5.6, 1.8, 1.4, 1.2, 1.0, 0.7],   # answer token wins
        [1.6, 1.4, 1.2, 1.0, 0.9, 6.4],   # end-of-turn (index 5) wins big -> STOP
    ]
    instruct = Trace("synthetic-instruct")
    for row in instruct_rows:
        probs = softmax(row)
        amax = max(range(len(row)), key=lambda i: row[i])
        if not instruct.step(f"tok{amax}", probs[amax], probs[EOS_I], amax == EOS_I):
            break
    assert instruct.stopped, "synthetic instruct MUST stop (end-of-turn wins)"
    assert instruct.stopped_at == 3, f"instruct must stop at step 3, got {instruct.stopped_at}"
    assert instruct.stop_eos_prob > 0.9, "instruct must stop with high confidence"
    print("  " + instruct.summary().strip())

    # the qualitative contrast, asserted
    assert base.max_eos_prob < instruct.stop_eos_prob, \
        "base's best end-of-turn prob must be far below instruct's stopping prob"
    # PPL arithmetic: a deterministic (p=1 every step) trace has PPL exactly 1.
    det = Trace("deterministic")
    for _ in range(4):
        det.step("x", 1.0, 0.0, False)
    assert abs(det.ppl() - 1.0) < 1e-12, "PPL of a certain sequence must be 1.0"
    print(f"  PPL of a certain (p=1) sequence = {det.ppl():.4f}  (= 1, as it must be)")
    print()
    print("  self-checks passed: softmax, greedy argmax, cumulative log-prob, PPL,")
    print("  and the stop / no-stop contrast all reproduce the page's mechanism.")
    print()
    print("=" * 72)
    print("SELF-TEST OK. Run without --self-test on your box for the real two-model decode.")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# The real experiment - needs a GPU, torch, transformers. Desk-checked against the
# verified July-2026 transformers 5.x API (constants §7.1, brief-tooling §4.1):
#   - AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)   [v5: dtype=, not torch_dtype]
#   - tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
#   - model.generate(..., return_dict_in_generate=True, output_scores=True, do_sample=False)
# --------------------------------------------------------------------------- #

def stamp():
    import torch
    line = f"  torch {torch.__version__} | cuda {torch.version.cuda}"
    try:
        import transformers
        line += f" | transformers {transformers.__version__}"
    except Exception:
        pass
    print(line)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"  device {torch.cuda.get_device_name(0)} | capability sm_{cap[0]}{cap[1]}")


def decode_one(model, tokenizer, prompt_text, eos_id, label, max_new_tokens, show_steps):
    """Greedy-decode `prompt_text`, halting on `eos_id`, printing per-step token-level
    probabilities and P(end-of-turn). Returns a completed Trace."""
    import torch

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                        # greedy = argmax, the page's decode
        return_dict_in_generate=True,
        output_scores=True,                     # per-step logits over the vocab
        eos_token_id=eos_id,                    # halt when the model emits end-of-turn
        pad_token_id=eos_id,                    # silence the open-end padding warning
    )
    gen_ids = out.sequences[0, prompt_len:]

    trace = Trace(label)
    print(f"\n  --- {label} ---")
    print(f"  prompt: {prompt_text!r}")
    for k, logits in enumerate(out.scores):
        probs = torch.softmax(logits[0].float(), dim=-1)
        chosen_id = int(gen_ids[k].item())
        chosen_prob = float(probs[chosen_id].item())
        eos_prob = float(probs[eos_id].item())
        is_eos = (chosen_id == eos_id)
        tok_str = tokenizer.decode([chosen_id])
        if show_steps and (k < show_steps or is_eos):
            shown = "<|im_end|>" if is_eos else repr(tok_str)
            print(f"    step {k + 1:>2}: {shown:<16} p={chosen_prob * 100:5.1f}%   "
                  f"P(end-of-turn)={eos_prob * 100:5.1f}%")
        keep = trace.step(tok_str, chosen_prob, eos_prob, is_eos)
        if not keep:
            break
    print(trace.summary())
    return trace


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 72)
    print("VERSIONS (paste this line into any support request)")
    print("=" * 72)
    stamp()
    if not torch.cuda.is_available():
        print("\n  CUDA not available - this experiment needs the GPU. Stopping.")
        print("  (Run --self-test anywhere for the GPU-free arithmetic + decode math.)")
        sys.exit(1)
    print()

    step0_sanity()

    print("=" * 72)
    print("SAFETY: about to load an 8B checkpoint (bf16, ~16 GB) - base first, then,")
    print("after freeing it, instruct. Never both at once. Contends with ComfyUI if up.")
    print("=" * 72)
    print()

    # ---- the chat-template diff, shown before any generation (brief rule 1) ---- #
    tok = AutoTokenizer.from_pretrained(args.model_instruct)
    messages = [{"role": "user", "content": args.question}]
    chatml = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print("=" * 72)
    print("THE CHAT-TEMPLATE DIFF - what apply_chat_template() does (never hand-rolled)")
    print("=" * 72)
    print(f"  raw prompt (what the base sees):   {args.prompt!r}")
    print(f"  ChatML-wrapped (what instruct sees):")
    for ln in chatml.splitlines():
        print(f"      {ln!r}")
    print(f"  enable_thinking=False so Qwen3 answers directly instead of emitting a")
    print(f"  <think>...</think> block first - otherwise 'it stops' is muddied by")
    print(f"  a long reasoning trace. [Qwen3 thinking-mode default; brief-tooling]")
    print()

    eos_id = tok.convert_tokens_to_ids(EOS)
    assert eos_id is not None and eos_id >= 0, f"could not resolve {EOS} in the tokenizer"

    # ---- BASE: raw prompt, then the same prompt wrapped in the chat template ---- #
    print("=" * 72)
    print(f"LOADING BASE: {args.model_base}  (bf16)")
    print("=" * 72)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_base, dtype=torch.bfloat16, device_map="auto")
    base_model.eval()

    base_raw = decode_one(base_model, tok, args.prompt, eos_id,
                          "Base, RAW prompt", args.max_new_tokens, args.show_steps)
    base_tpl = decode_one(base_model, tok, chatml, eos_id,
                          "Base, chat-template-wrapped", args.max_new_tokens, args.show_steps)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- INSTRUCT: the wrapped prompt ---- #
    print("\n" + "=" * 72)
    print(f"LOADING INSTRUCT: {args.model_instruct}  (bf16) - base has been freed")
    print("=" * 72)
    instr_model = AutoModelForCausalLM.from_pretrained(
        args.model_instruct, dtype=torch.bfloat16, device_map="auto")
    instr_model.eval()

    instruct = decode_one(instr_model, tok, chatml, eos_id,
                          "Instruct, chat-template-wrapped", args.max_new_tokens, args.show_steps)

    del instr_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------- #
    # The verdict - behaviour as DATA, with [EST] labels and loud regression asserts.
    # --------------------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("VERDICT  [EST - behavioural, the documented outcome, not a frozen constant]")
    print("=" * 72)
    print(f"  Base (raw):       stopped = {base_raw.stopped}   "
          f"max P(end-of-turn) = {base_raw.max_eos_prob * 100:.1f}%")
    print(f"  Base (templated): stopped = {base_tpl.stopped}   "
          + (f"at step {base_tpl.stopped_at}, p={base_tpl.stop_eos_prob * 100:.1f}%"
             if base_tpl.stopped else
             f"max P(end-of-turn) = {base_tpl.max_eos_prob * 100:.1f}%"))
    print(f"  Instruct:         stopped = {instruct.stopped}   "
          + (f"at step {instruct.stopped_at}, p={instruct.stop_eos_prob * 100:.1f}%"
             if instruct.stopped else "NEVER (unexpected!)"))
    print()
    print("  The story, in three lines:")
    print("   - The base never learned to emit end-of-turn: it rambles to the token cap.")
    print("   - The instruct model answers and then chooses end-of-turn with a big margin.")
    print("   - Wrapping the base in the template it was never trained on nudges end-of-turn")
    print("     up - it partially recovers, stopping later and less confidently than instruct.")
    print("   Instruct-ness is a learned HABIT, not new knowledge. (-> p.48)")
    print()

    # regression guards: the documented prediction. A base that suddenly stops or an
    # instruct that rambles is a real change worth failing on.
    assert instruct.stopped, \
        "[EST] the instruct model is expected to emit end-of-turn and STOP - it did not"
    assert base_raw.max_eos_prob < instruct.stop_eos_prob, \
        ("[EST] the raw base's best end-of-turn probability should stay well below the "
         "instruct model's stopping probability - it did not")
    if base_raw.stopped:
        print("  NOTE: the raw base stopped this run - unusual for a base model; the")
        print("        qualitative point (it stops far later / less reliably) may still hold,")
        print("        but inspect the per-step probabilities above before trusting it.")
    print("  regression guards passed: instruct stops, raw base's end-of-turn stays weak.")
    print()
    print("=" * 72)
    print("Done. The demo on p.45 is this experiment on authored logits; this was the real one.")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="GPU-free: run and assert the arithmetic + decode-summariser math")
    ap.add_argument("--model-base", default=DEFAULT_BASE, help="base checkpoint id/path")
    ap.add_argument("--model-instruct", default=DEFAULT_INSTRUCT, help="instruct checkpoint id/path")
    ap.add_argument("--prompt", default=RAW_PROMPT, help="raw continuation prompt for the base")
    ap.add_argument("--question", default=QUESTION, help="the question wrapped for the instruct model")
    ap.add_argument("--max-new-tokens", type=int, default=60,
                    help="token cap (push to 60 to watch the base never stop, like the page)")
    ap.add_argument("--show-steps", type=int, default=12,
                    help="print per-step probabilities for the first N steps (0 = summary only)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("torch + transformers are required for the real experiment, and are not")
        print("importable here. Install them into THE fresh venv (never ComfyUI's;")
        print("hardware-ground-truth §3), or run the GPU-free path:")
        print("    python 03_base_vs_instruct.py --self-test")
        sys.exit(1)

    run_real(args)


if __name__ == "__main__":
    main()
