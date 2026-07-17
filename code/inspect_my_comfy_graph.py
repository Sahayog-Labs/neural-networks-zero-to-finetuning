#!/usr/bin/env python3
"""
inspect_my_comfy_graph.py — the D-18 taxonomy, applied to your own files. For p.42.

"'Fine-tuning' names three different things" taught the type signature: every
adaptation either edits theta (the base model's weights) or edits c (the
conditioning — everything that goes INTO the frozen model at inference time).
The collapsing test that sorts them: "after you're done, is the base model
different?" LoRA/DreamBooth -> yes, theta changed. ControlNet/IP-Adapter/the
prompt -> no, you bolted an extra input onto an unchanged model.

This script points that test at a REAL saved ComfyUI workflow (yours, or the
bundled example if you don't pass one) and tags every node theta-edit vs
c-edit vs infra (load/sample/save nodes that are neither — they don't touch
weights and they don't touch conditioning, they just move tensors around).

It reads both ComfyUI export formats:
  - API format   (Save (API Format) / what a Python client POSTs): a flat
    dict keyed by node id, each value {"class_type": ..., "inputs": {...}}.
  - UI format    (the default "Save" — a full workflow file): {"nodes": [...],
    "links": [...]}, each node has "type" instead of "class_type".

Usage
-----
    python inspect_my_comfy_graph.py --workflow ~/ComfyUI/user/default/workflows/my_flux_lora.json
    python inspect_my_comfy_graph.py                      # no arg -> bundled example workflow
    python inspect_my_comfy_graph.py --workflow-dir ~/ComfyUI/user/default/workflows --all

No installs, no GPU, read-only. Runtime ~2 s (constants.md §10 exercise budget; spec-code.md D.2).
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# THE TYPE SIGNATURE — the same three buckets p.42's taxonomy draws.
#
#   THETA_EDIT : after this node runs, the model object carries different
#                weights than the checkpoint on disk. D-18: "the base model
#                is different."
#   C_EDIT     : this node changes what goes INTO the (unchanged) model —
#                the conditioning c, or the image being denoised. The base
#                model is bit-identical to the checkpoint on disk.
#   INFRA      : load/sample/decode/save plumbing. Neither edits theta nor c;
#                it moves tensors between the two groups above.
#
# Extend these sets for node classes not covered here -- ComfyUI's ecosystem
# adds new custom nodes constantly. The rule for adding one is the same
# question the page teaches: "after this node runs, is the base model
# different?" Yes -> THETA_EDIT_TYPES. No, but it changed the input -> C_EDIT_TYPES.
# ---------------------------------------------------------------------------
THETA_EDIT_TYPES = {
    "LoraLoader",               # applies BA to the model's weights in-graph
    "LoraLoaderModelOnly",
    "LoraLoaderTagged",         # some custom-node forks name it this
    "HypernetworkLoader",       # older SD1.5-era theta-edit, same bucket
    "DreamBoothLoraLoader",     # a DreamBooth checkpoint IS a full theta edit,
    "CheckpointLoaderSimple_DreamBooth",  # (whether stored as LoRA or full weights)
}

C_EDIT_TYPES = {
    "CLIPTextEncode",           # the prompt -- text -> conditioning vector
    "CLIPTextEncodeSDXL",
    "ConditioningCombine",
    "ConditioningConcat",
    "ConditioningSetArea",
    "ConditioningZeroOut",
    "ConditioningSetTimestepRange",
    "ControlNetApply",          # applies control signal to the SAMPLING process,
    "ControlNetApplyAdvanced",  # not to the model's weights
    "IPAdapterApply",
    "IPAdapterAdvanced",
    "IPAdapterCombineParams",
    "T2IAdapterApply",
    "StyleModelApply",
}

INFRA_TYPES = {
    "CheckpointLoaderSimple", "UNETLoader", "CLIPLoader", "DualCLIPLoader",
    "VAELoader", "ControlNetLoader", "IPAdapterModelLoader", "CLIPVisionLoader",
    "KSampler", "KSamplerAdvanced", "SamplerCustom",
    "EmptyLatentImage", "LatentUpscale", "VAEDecode", "VAEEncode",
    "LoadImage", "SaveImage", "PreviewImage", "Note",
}


# ---------------------------------------------------------------------------
# The bundled example — a realistic small graph, used when the learner
# doesn't pass --workflow: checkpoint -> LoRA (1 theta-edit) -> ControlNet +
# IP-Adapter + two CLIPTextEncode (positive/negative) = 4 c-edits -> sampler
# -> decode -> save. This is the exact "1 place / 4 places" split the page's
# spec calls out (spec-part4.md p.42, spec-code.md D.2).
# ---------------------------------------------------------------------------
EXAMPLE_WORKFLOW_API = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-dev.safetensors"}},
    "2": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "lora_name": "my_subject_lora.safetensors", "strength_model": 0.85}},
    "3": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "flux_canny.safetensors"}},
    "4": {"class_type": "ControlNetApply", "inputs": {"conditioning": ["6", 0], "control_net": ["3", 0]}},
    "5": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": "ip-adapter_flux.safetensors"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a photo of sks person, studio lighting"}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "blurry, low quality"}},
    "8": {"class_type": "IPAdapterApply", "inputs": {"model": ["2", 0], "ipadapter": ["5", 0], "image": ["9", 0]}},
    "9": {"class_type": "LoadImage", "inputs": {"image": "reference_face.png"}},
    "10": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024}},
    "11": {"class_type": "KSampler", "inputs": {"model": ["8", 0], "positive": ["4", 0], "negative": ["7", 0], "latent_image": ["10", 0]}},
    "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
    "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0]}},
}


def load_workflow(path):
    """Parse either ComfyUI export format into a flat list of
    {"id", "class_type", "title"} dicts. Raises ValueError on an unrecognized
    shape rather than guessing -- a silently-wrong classification is worse
    than a loud failure here."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return parse_workflow_dict(raw)


def parse_workflow_dict(raw):
    nodes = []
    if isinstance(raw, dict) and "nodes" in raw and isinstance(raw["nodes"], list):
        # UI export format
        for n in raw["nodes"]:
            nodes.append({
                "id": str(n.get("id", "?")),
                "class_type": n.get("type", "UNKNOWN"),
                "title": n.get("title") or n.get("type", "UNKNOWN"),
            })
    elif isinstance(raw, dict):
        # API export format: every value should look like {"class_type": ...}
        looks_like_api = all(isinstance(v, dict) and "class_type" in v for v in raw.values())
        if not looks_like_api:
            raise ValueError(
                "unrecognized workflow JSON shape: expected a UI export "
                "(top-level 'nodes' list) or an API export (dict of "
                "{id: {class_type, inputs}}). Re-save from ComfyUI with "
                "'Save' or 'Save (API Format)'."
            )
        for node_id, v in raw.items():
            nodes.append({
                "id": node_id,
                "class_type": v["class_type"],
                "title": v.get("_meta", {}).get("title", v["class_type"]),
            })
    else:
        raise ValueError("workflow JSON must be an object at the top level")
    return nodes


def classify(nodes):
    """Tag every node theta-edit / c-edit / infra / unknown. Returns the
    tagged list plus the three counts the page's punchline quotes."""
    tagged = []
    for n in nodes:
        ct = n["class_type"]
        if ct in THETA_EDIT_TYPES:
            bucket = "THETA-EDIT"
        elif ct in C_EDIT_TYPES:
            bucket = "C-EDIT"
        elif ct in INFRA_TYPES:
            bucket = "infra"
        else:
            bucket = "unknown"
        tagged.append({**n, "bucket": bucket})
    return tagged


def report(tagged, source_label):
    n_theta = sum(1 for t in tagged if t["bucket"] == "THETA-EDIT")
    n_c = sum(1 for t in tagged if t["bucket"] == "C-EDIT")
    n_infra = sum(1 for t in tagged if t["bucket"] == "infra")
    n_unknown = sum(1 for t in tagged if t["bucket"] == "unknown")

    print("=" * 78)
    print(f"  {source_label}")
    print("=" * 78)
    print(f"  {'id':>4}  {'bucket':<11} {'class_type':<26} title")
    print("  " + "-" * 74)
    for t in tagged:
        print(f"  {t['id']:>4}  {t['bucket']:<11} {t['class_type']:<26} {t['title']}")
    print("  " + "-" * 74)

    if n_unknown:
        print(f"  ({n_unknown} node(s) not in the classification table above -- "
              f"a custom node this script doesn't know yet. Ask 'after this node "
              f"runs, is the base model different?' and add it to THETA_EDIT_TYPES "
              f"or C_EDIT_TYPES.)")

    print()
    print(f"  your graph changes the base model in {n_theta} place"
          f"{'s' if n_theta != 1 else ''} and the conditioning in {n_c} place"
          f"{'s' if n_c != 1 else ''}.")
    print(f"  ({n_infra} infra node(s): load / sample / decode / save -- "
          f"neither theta nor c, just plumbing.)")
    print()
    if n_theta == 0:
        print("  D-18 collapsing test: with zero theta-edits, this graph never asks "
              "'is the base model different?' -- it's a bit-identical checkpoint "
              "with a stack of c-edits on top. That's most ComfyUI graphs.")
    elif n_theta == 1:
        print("  D-18 collapsing test: exactly one theta-edit -- the base model IS "
              "different after this graph runs, and there is exactly one node "
              "responsible. Every other node you might have called 'fine-tuning' "
              "(ControlNet, IP-Adapter, the prompt itself) left theta untouched.")
    else:
        print(f"  D-18 collapsing test: {n_theta} theta-edits stack -- each one "
              f"changes the SAME base model further (e.g. two chained LoraLoader "
              f"nodes), which is legal because they're the same kind of object.")
    return n_theta, n_c, n_infra, n_unknown


def find_default_workflow_dir():
    for d in (Path.home() / "ComfyUI" / "user" / "default" / "workflows",
              Path.home() / "ComfyUI" / "workflows"):
        if d.is_dir():
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow", default=None,
                     help="path to a saved ComfyUI workflow .json (UI or API export). "
                          "Default: the bundled example graph (no file needed).")
    ap.add_argument("--workflow-dir", default=None,
                     help="scan this directory for *.json workflows instead of one file "
                          "(default when --workflow is omitted and --all is passed: "
                          "~/ComfyUI/user/default/workflows or ~/ComfyUI/workflows)")
    ap.add_argument("--all", action="store_true",
                     help="with --workflow-dir (or its default), report on every .json found, "
                          "not just the first")
    args = ap.parse_args()

    if args.workflow:
        path = Path(args.workflow).expanduser()
        if not path.is_file():
            print(f"  {path} not found. Falling back to the bundled example workflow.\n")
            tagged = classify(parse_workflow_dict(EXAMPLE_WORKFLOW_API))
            report(tagged, "bundled example (checkpoint + LoRA + ControlNet + IP-Adapter + prompt)")
            sys.exit(1)
        paths = [path]
    elif args.workflow_dir or args.all:
        d = Path(args.workflow_dir).expanduser() if args.workflow_dir else find_default_workflow_dir()
        if d is None or not d.is_dir():
            print("  no ComfyUI workflows directory found (looked in "
                  "~/ComfyUI/user/default/workflows and ~/ComfyUI/workflows). "
                  "Pass --workflow-dir explicitly, or --workflow for a single file.")
            sys.exit(1)
        paths = sorted(d.glob("*.json"))
        if not paths:
            print(f"  {d} has no .json workflows.")
            sys.exit(1)
        if not args.all:
            paths = paths[:1]
    else:
        print("  no --workflow given -- using the bundled example graph.\n"
              "  (pass --workflow <path.json> to point this at one of your own "
              "~12 GB of saved LoRA workflows, hardware-ground-truth.md section 5)\n")
        tagged = classify(parse_workflow_dict(EXAMPLE_WORKFLOW_API))
        n_theta, n_c, n_infra, n_unknown = report(tagged, "bundled example")
        _run_self_checks(tagged, n_theta, n_c)
        return

    for p in paths:
        try:
            nodes = load_workflow(p)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  {p.name}: could not parse -- {e}")
            continue
        tagged = classify(nodes)
        report(tagged, p.name)
        print()


def _run_self_checks(tagged, n_theta, n_c):
    """spec-code.md D.2's mandated self-check: LoRA nodes -> theta,
    ControlNet/IP-Adapter/prompt -> c. Only meaningful for the bundled
    example, whose composition is fixed and known in advance."""
    by_type = {t["class_type"]: t["bucket"] for t in tagged}
    assert by_type["LoraLoader"] == "THETA-EDIT", "LoRA must classify as a theta-edit (D-18)"
    assert by_type["ControlNetApply"] == "C-EDIT", "ControlNet must classify as a c-edit (D-18)"
    assert by_type["IPAdapterApply"] == "C-EDIT", "IP-Adapter must classify as a c-edit (D-18)"
    assert by_type["CLIPTextEncode"] == "C-EDIT", "the prompt (CLIPTextEncode) must classify as a c-edit (D-18)"
    assert n_theta == 1, f"bundled example should have exactly 1 theta-edit, got {n_theta}"
    assert n_c == 4, f"bundled example should have exactly 4 c-edits, got {n_c}"
    print("  self-check passed: LoraLoader -> THETA-EDIT; ControlNetApply, "
          "IPAdapterApply, CLIPTextEncode(x2) -> C-EDIT. 1 place / 4 places, as printed above.")


if __name__ == "__main__":
    main()
