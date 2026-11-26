#!/usr/bin/env python
"""Option-order sensitivity on real dev rows.

Runs the same state with options (正常, 涉诈) and (涉诈, 正常).  Raw readout is
position-sensitive; L0 marginalizes the two permutations and should be
invariant.  The fixed-prior variant applies the same cal prior in both orders.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.harness import (  # noqa: E402
    apply_fixed_prior, load_goldens, make_decider, prior_from_cal,
)
from eval.metrics import flip_rate  # noqa: E402
from eval.task import OPTIONS, QUESTION  # noqa: E402


def run_order(backend, rows, question, config, args):
    decider = make_decider(backend, config, args)
    return decider.decide_batch([r["state"] for r in rows], question)


def flip_by_label(rows, f_ans, r_ans):
    out = {}
    for lab in OPTIONS:
        idx = [i for i, r in enumerate(rows) if r["label"] == lab]
        flips = sum(int(f_ans[i] != r_ans[i]) for i in idx)
        out[lab] = {"n": len(idx), "flips": flips,
                    "rate": (flips / len(idx)) if idx else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goldens", default="eval/goldens.jsonl")
    ap.add_argument("--out", default="runs/m0.5/reversal.json")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-perms", type=int, default=None)
    ap.add_argument("--min-prior-n", type=int, default=8)
    ap.add_argument("--adaptive-min-shifts", type=int, default=2)
    ap.add_argument("--adaptive-target", type=float, default=0.01)
    ap.add_argument("--selection", default="runs/m0.5/selection.json",
                    help="frozen prior strengths; falls back to --prior-strength")
    ap.add_argument("--prior-strength", type=float, default=None)
    ap.add_argument("--limit-normal", type=int, default=32)
    ap.add_argument("--limit-fraud", type=int, default=88)
    ap.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    splits = load_goldens(Path(args.goldens), ["cal", "dev"])
    dev = splits["dev"]
    normal = [r for r in dev if r["label"] == "正常"][: args.limit_normal]
    fraud = [r for r in dev if r["label"] == "涉诈"][: args.limit_fraud]
    rows = normal + fraud
    print(f"dev rows={len(rows)} normal={len(normal)} fraud={len(fraud)}", flush=True)

    from anyjev import Question
    from anyjev.backends.hf import HFBackend

    backend = HFBackend(args.model, device=args.device, dtype=args.dtype,
                        batch_size=args.batch_size)
    q_fwd = Question.choice(QUESTION, OPTIONS, name="risk_fwd")
    q_rev = Question.choice(QUESTION, list(reversed(OPTIONS)), name="risk_rev")

    prior = prior_from_cal(splits["cal"])
    strengths: dict[str, float] = {}
    sel_path = Path(args.selection)
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        strengths = {c: float(p["strength"]) for c, p in sel.get("prior", {}).items()}
    payload = {"n": len(rows), "normal": len(normal), "fraud": len(fraud),
               "cal_prior": prior.tolist(), "configs": {}}

    for config in ("raw", "l0-noprior"):
        fwd = run_order(backend, rows, q_fwd, config, args)
        rev = run_order(backend, rows, q_rev, config, args)
        f_ans = [d.answer for d in fwd]
        r_ans = [d.answer for d in rev]
        payload["configs"][config] = {
            "forward_predicted": {o: f_ans.count(o) for o in OPTIONS},
            "reversed_predicted": {o: r_ans.count(o) for o in OPTIONS},
            "flip": flip_rate(f_ans, r_ans),
            "flip_by_label": flip_by_label(rows, f_ans, r_ans),
        }
        print(f"{config}: flip={payload['configs'][config]['flip']} "
              f"by_label={payload['configs'][config]['flip_by_label']}", flush=True)

        strength = strengths.get(config, args.prior_strength)
        if strength is not None:
            # Apply the same cal prior in both option orders.  Decision.probs
            # follow the question's option order, so the reversed run's probs
            # are [涉诈, 正常]; map them back to canonical [正常, 涉诈] before
            # applying the canonical prior (apply_fixed_prior labels with
            # OPTIONS = [正常, 涉诈]).
            f_rows = [{"id": rows[i]["id"], "label": rows[i]["label"],
                       "probs": fwd[i].probs} for i in range(len(rows))]
            r_rows = [{"id": rows[i]["id"], "label": rows[i]["label"],
                       "probs": list(reversed(rev[i].probs))} for i in range(len(rows))]
            f_adj = apply_fixed_prior(f_rows, prior, strength)
            r_adj = apply_fixed_prior(r_rows, prior, strength)
            f_adj_ans = [x["answer"] for x in f_adj]
            r_adj_ans = [x["answer"] for x in r_adj]
            payload["configs"][f"{config}+prior"] = {
                "strength": strength,
                "flip": flip_rate(f_adj_ans, r_adj_ans),
                "flip_by_label": flip_by_label(rows, f_adj_ans, r_adj_ans),
            }
            print(f"{config}+prior({strength}): flip={payload['configs'][f'{config}+prior']['flip']}",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
