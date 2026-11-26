#!/usr/bin/env python
"""Extract last-position hidden states for the frozen goldens.

One forward pass per row; the probe is trained on the cached [N, layers, H]
tensor.  Layer 0 is the embedding output; layer `num_hidden_layers` is after
the final norm, i.e. what the LM head reads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.task import (  # noqa: E402
    CATEGORY_OPTIONS, CATEGORY_QUESTION, CATEGORY_SYSTEM,
    OPTIONS, QUESTION, SYSTEM,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def render_prompts(tokenizer, rows: list[dict], question: str, options: list[str],
                   system: str) -> list[str]:
    from anyjev import Question
    from anyjev.readout import build_prompt, render_chat

    q = Question.choice(question, options, name="risk")
    perms = list(range(len(options)))
    return [render_chat(tokenizer, build_prompt(r["state"], q, perms, system=system))
            for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goldens", default="eval/goldens.jsonl")
    ap.add_argument("--out", default="runs/m0.5/features_binary.npz")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--task", choices=["binary", "category", "v2"], default="binary")
    ap.add_argument("--layers", default="8,16,24,32",
                    help="hidden-state indices; 32 = after final norm for this model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    rows = load_jsonl(Path(args.goldens))
    if args.task == "category":
        question, options, system = CATEGORY_QUESTION, CATEGORY_OPTIONS, CATEGORY_SYSTEM
    elif args.task == "v2":
        from eval.task_v2 import OPTIONS as V2_OPTIONS
        from eval.task_v2 import QUESTION as V2_QUESTION
        from eval.task_v2 import SYSTEM as V2_SYSTEM
        question, options, system = V2_QUESTION, V2_OPTIONS, V2_SYSTEM
    else:
        question, options, system = QUESTION, OPTIONS, SYSTEM
    if args.limit:
        rows = rows[: args.limit]

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    from anyjev.backends.hf import HFBackend

    print(f"loading {args.model} ...", flush=True)
    backend = HFBackend(args.model, device=args.device, dtype=args.dtype,
                        batch_size=args.batch_size)
    prompts = render_prompts(backend.tokenizer, rows, question, options, system)
    print(f"rows={len(rows)} layers={layers} mean_prompt_chars="
          f"{sum(len(p) for p in prompts) / len(prompts):.0f}", flush=True)

    started = time.perf_counter()
    feats, _, _ = backend.hidden_states(prompts, layers=layers)
    elapsed = time.perf_counter() - started
    print(f"extracted {feats.shape} in {elapsed:.1f}s "
          f"({elapsed / max(len(rows), 1):.4f}s/row)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        feats=feats.astype(np.float32),
        ids=np.asarray([r["id"] for r in rows]),
        labels=np.asarray([r["label"] for r in rows]),
        categories=np.asarray(
            [r.get("category") or r.get("scene_category") or "" for r in rows]
        ),
        cohorts=np.asarray([r.get("cohort") or "" for r in rows]),
        splits=np.asarray([r["split"] for r in rows]),
        layers=np.asarray(layers),
        question=np.asarray(question),
        options=np.asarray(options),
        elapsed_s=np.asarray(elapsed),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
