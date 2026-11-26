#!/usr/bin/env python
"""Operating-point frontier for the M0.5 binary task.

For each frozen decision config, sweep the fraud-probability threshold and
report the best achievable point under two criteria:

  * max balanced accuracy (unconstrained);
  * max normal recall subject to fraud recall >= 0.95 (the pre-registered
    operational constraint).

This is a diagnostic over dev (and optionally gate) probabilities.  It does
not select anything that feeds the gate: the gate decisions are frozen by
`eval/harness.py`; this script only reads their probabilities.

Usage:
  python eval/frontier.py --split dev
  python eval/frontier.py --split gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.task import OPTIONS  # noqa: E402


def load_rows(path: str | Path, split: str) -> dict[str, list[dict]]:
    """Return {config: rows} for one split.

    Accepts either a harness results.json (base + derived configs) or a probe
    predictions.jsonl.  Rows must have label, probs, and answer.
    """
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    if "splits" in payload:
        split_payload = payload["splits"].get(split, {})
        for name, cfg in split_payload.items():
            out[name] = cfg["rows"]
        for name, cfg in payload.get("derived", {}).items():
            out[name] = cfg["rows"]
        return out
    # predictions.jsonl
    rows = []
    for line in p.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("split") == split:
            rows.append(rec)
    return {p.stem: rows} if rows else {}


def load_probe(path: str | Path, split: str, name: str) -> list[dict]:
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("split") == split:
            rows.append(rec)
    return rows


def sweep(rows: list[dict]) -> dict:
    y = np.asarray([1 if r["label"] == "涉诈" else 0 for r in rows], dtype=int)
    score = np.asarray([float(r["probs"][1]) for r in rows], dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return {}
    order = np.argsort(-score, kind="stable")
    y_sorted = y[order]
    score_sorted = score[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fraud_recall = tp / n_pos
    normal_recall = 1.0 - fp / n_neg
    bal = (fraud_recall + normal_recall) / 2.0
    precision = tp / np.maximum(tp + fp, 1)
    # Candidate operating points: one per distinct score (threshold = score).
    # Keep the LAST occurrence of each score so the cumulative counts include
    # every row tied at that threshold (prediction is score >= threshold).
    keep = np.ones(len(score_sorted), dtype=bool)
    keep[:-1] = score_sorted[:-1] != score_sorted[1:]
    idx = np.flatnonzero(keep)

    def point(i: int) -> dict:
        return {
            "threshold": float(score_sorted[i]),
            "fraud_recall": float(fraud_recall[i]),
            "normal_recall": float(normal_recall[i]),
            "balanced_accuracy": float(bal[i]),
            "precision": float(precision[i]),
        }

    best_bal = idx[np.argmax(bal[idx])]
    high = idx[fraud_recall[idx] >= 0.95]
    best_high_normal = high[np.argmax(normal_recall[high])] if len(high) else None
    best_high_bal = high[np.argmax(bal[high])] if len(high) else None
    return {
        "n": len(rows),
        "support": {"正常": n_neg, "涉诈": n_pos},
        "max_balanced_accuracy": point(int(best_bal)),
        "fraud_recall_ge_0.95": {
            "max_normal_recall": point(int(best_high_normal)) if best_high_normal is not None else None,
            "max_balanced_accuracy": point(int(best_high_bal)) if best_high_bal is not None else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["dev", "gate"], default="dev")
    ap.add_argument("--dev-results", default="runs/m0.5/dev/results.json")
    ap.add_argument("--gate-results", default="runs/m0.5/gate/results.json")
    ap.add_argument("--probe", default="runs/m0.5/probe/predictions.jsonl")
    ap.add_argument("--probe-riskfirst", default="runs/m0.5/probe/predictions_riskfirst.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    configs: dict[str, list[dict]] = {}
    results_path = args.dev_results if args.split == "dev" else args.gate_results
    if Path(results_path).exists():
        configs.update(load_rows(results_path, args.split))
    if Path(args.probe).exists():
        configs["probe"] = load_probe(args.probe, args.split, "probe")
    if Path(args.probe_riskfirst).exists():
        configs["probe-riskfirst"] = load_probe(args.probe_riskfirst, args.split, "probe-riskfirst")

    if not configs:
        raise SystemExit("no configs found")

    out = {"split": args.split, "configs": {}}
    print(f"split={args.split}")
    print(f"{'config':>18} {'n':>4} {'max-bal point':>34} {'fraud>=.95: normal/bal':>34}")
    for name, rows in configs.items():
        s = sweep(rows)
        if not s:
            continue
        out["configs"][name] = s
        b = s["max_balanced_accuracy"]
        h = s["fraud_recall_ge_0.95"]
        hn = h["max_normal_recall"]
        hb = h["max_balanced_accuracy"]
        print(f"{name:>18} {s['n']:>4} "
              f"t={b['threshold']:.3f} f={b['fraud_recall']:.3f} n={b['normal_recall']:.3f} b={b['balanced_accuracy']:.3f} "
              f"| t={hn['threshold']:.3f} n={hn['normal_recall']:.3f} b={hb['balanced_accuracy']:.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
