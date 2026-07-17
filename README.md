# 🧠 Neural Networks: Zero to Fine-Tuning

An interactive, self-contained course — **65 pages** from high-school algebra to fine-tuning
LLMs and diffusion models on your own hardware. No build step, no server, no dependencies:
**unzip / clone, open `index.html`, learn.**

## What it is

- **Parts I–IV (the trunk, pp. 1–43):** the machine (neurons → shapes → the dot product →
  the chain rule), how it learns (backprop derived by pencil and verified live in your
  browser, optimizers, the memory ledger), architecture (attention in depth, RoPE, the
  transformer block, counting a real 8B model's parameters to the byte), and adaptation
  (rank/SVD, LoRA derived once, quantization, the roofline).
- **The fork:** Track A — **LLM fine-tuning** (tokenization → RAG-vs-fine-tune → QLoRA →
  a GRPO/RLVR capstone) and Track B — **diffusion** (VAE → forward/reverse processes → flow
  matching → CFG → training your own LoRA), at equal depth.
- **Capstone (pp. 63–65):** predict your machine's behavior from first principles, then
  verify that the two tracks literally converged (a frontier image model's text encoder *is*
  an LLM — provable from files on disk).

Every page: an interactive demo that computes **real math live** (a genuine autograd engine,
`assets/nn.js`, trains real networks in the page — nothing is a canned animation), a
worked example carried to actual digits, misconception warnings, and a self-graded quiz.
The `code/` folder holds **72 runnable Python scripts + 2 notebooks** that take the same
ideas to real hardware — sized for a single-GPU box (they were written against an NVIDIA
DGX Spark, and the course teaches you to *measure* your own machine rather than trust
spec sheets).

## Using it

1. Clone, then open `index.html` in any modern browser. That's it. (For the handful of
   demos that fetch local files, serve the folder instead: `python -m http.server` and open
   `http://localhost:8000`.)
2. Pages are meant in order; the sidebar tracks your position. Dark and light themes.
3. The `code/` scripts each have a `--self-test` that runs without a GPU. The fine-tuning
   scripts expect a fresh Python venv (never an existing app's) — `code/00_verify_env.py`
   checks your setup first.

## How this course was built

**Every artifact in this repository — every page, demo, script, spec, and this README — was
written end-to-end by Claude (Anthropic), running in Claude Code.** A human learner directed
the project: chose the scope ("high-school entry to fine-tuning, go deep"), made the
judgment calls at decision gates, and lent the hardware that grounded the measurements — but
authored none of the content.

The build ran over two days (2026-07-16/17) as a pipeline of ten multi-agent workflows: an
orchestrating session (Claude Fable 5) spawning parallel subagents — Claude Opus 4.8 for
research, derivations, and the hard pages; Claude Sonnet 5 for design work, standard pages,
and per-page QA. The exact telemetry, from the workflow run reports:

| Phase | What happened | Agents | Tokens |
|---|---|---:|---:|
| 1 · Research | 7 parallel briefs, ~142k words, web-verified | 7 | 0.87 M |
| 1b · Verify + arbitrate | independent recomputation of all load-bearing math + hardware/version claims; constants/notation/decisions frozen | 3 | 0.78 M |
| 1c · Numerics cleanup | remaining disputes computed three independent ways | 3 | 0.48 M |
| 2 · Design system | in-browser autograd engine (168 gradient-checked tests), canvas viz primitives, visual identity, template | 4 | 0.89 M |
| 3 · Curriculum spec | 8 spec writers + consistency integrator | 9 | 2.31 M |
| 3b · Fidelity repair | spec drift vs the ratified outline caught and repaired | 4 | 0.90 M |
| 4a · Pilot | 3 representative pages + adversarial review → fleet-prompt amendments | 4 | 0.83 M |
| 4b · Page fleet | 63 pages, each build → independent QA → fix; whole-course sweep | 265* | 25.07 M* |
| 5 · Code ladder | 52 scripts/notebooks built, all self-tested; polish; packaging | 55 | 6.88 M |
| — · Standalone agents | component retrofits and fixes | 1 | 0.18 M |
| **Total** | | **~355** | **~39 M** |

\* The fleet ran twice: an auth-token expiry killed the first run mid-flight (102 agents,
9.35 M tokens, 38 builds salvaged), and the second run (163 agents, 15.72 M) replayed the
salvaged work from cache and completed the rest — so the fleet figures include retried work.
Totals exclude the orchestrating session's own reasoning tokens.

The defining feature of the pipeline was **layered distrust**: verifiers recomputed the
researchers' numbers (and refuted several confident claims), an arbiter reconciled the
verifiers (and was itself refuted by on-device measurement), page builders were required to
check their spec against the frozen constants (and caught a spec arithmetic error), QA
agents re-verified builder self-reports (36 of 63 pages needed fixes), and the final gate
re-tested an earlier pass's "fixed" claim and found it false. Every layer caught something
the layer above it got confidently wrong.

## Provenance & rigor

Independent verification passes recomputed every load-bearing number (the frozen values live
in `research/constants.md` with per-value confidence labels), hardware claims were
**measured on a real machine** rather than quoted (`research/hardware-ground-truth.md`),
every page passed headless functional QA, and anything not verifiable is labeled *inferred*
or *estimate* on the page itself — never laundered into fact. The `research/` folder is the
full audit trail; `DESIGN-HANDOFF.md` documents the design system for contributors.

Numbers are anchored to open, reproducible artifacts (Qwen3-8B's config resolves the
parameter count to the exact byte — twice, by independent routes) so you can check the
course's arithmetic yourself. Found an error anyway? Open an issue — this course's whole
ethos is that claims get verified.

## License

MIT — see [LICENSE](LICENSE). KaTeX (vendored under `assets/katex/`) is MIT-licensed by its
own authors.
