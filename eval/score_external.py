#!/usr/bin/env python
"""Score an external feature cache with the frozen v2 probe head.

The head in ``runs/v2/probe/probe_head.npz`` was fitted on cal and selected on
dev; this script never trains or selects anything.  It is meant for external
normal-only batches (e.g. the Chongqing 20261030-1102 normal set) where the
question is the false-positive rate at the already-frozen v2 thresholds.

The thresholds are read from ``runs/v2/gate_results.json``: argmax plus every
dev-frozen operating point (``probe@devR95`` / ``probe@fpr5`` / ``probe@fpr10``).

Usage:
  python eval/score_external.py \
      --features runs/external_cq/features_normal_1030.npz \
      --out-dir runs/external_cq/normal_1030
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import wilson_ci  # noqa: E402


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_thresholds(path: Path) -> dict[str, float]:
    results = json.loads(path.read_text(encoding="utf-8"))
    thresholds = {
        "argmax": float(results["selection"]["thresholds"]["argmax"]),
    }
    for name, point in results.get("operating_points", {}).items():
        thresholds[name] = float(point["threshold"])
    return thresholds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--head", default="runs/v2/probe/probe_head.npz")
    ap.add_argument("--gate-results", default="runs/v2/gate_results.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expected-label", default="非涉诈",
                    help="label whose rate is reported as false positives")
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    head = np.load(args.head, allow_pickle=True)
    feats = data["feats"]
    ids = [str(x) for x in data["ids"]]
    labels = [str(x) for x in data["labels"]]
    splits = [str(x) for x in data["splits"]]
    layers = [int(x) for x in data["layers"]]
    layer = int(head["layer"])
    if layer not in layers:
        raise SystemExit(f"head layer {layer} not in feature layers {layers}")

    x = feats[:, layers.index(layer), :].astype(np.float64)
    mean = head["mean"].astype(np.float64)
    std = np.maximum(head["std"].astype(np.float64), 1e-6)
    z = np.clip((x - mean) / std, -10.0, 10.0)
    logits = z @ head["weight"].astype(np.float64).T + head["bias"].astype(np.float64)
    probs = softmax(logits)
    options = [str(x) for x in head["options"]]
    if len(options) != 2 or options[1] != "涉诈":
        raise SystemExit(f"unexpected head options: {options}")

    thresholds = load_thresholds(Path(args.gate_results))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for i, row_id in enumerate(ids):
            fh.write(json.dumps({
                "id": row_id,
                "split": splits[i],
                "label": labels[i],
                "probs": [float(v) for v in probs[i]],
                "answer": options[int(probs[i].argmax())],
            }, ensure_ascii=False) + "\n")

    summary: dict = {
        "features": args.features,
        "head": args.head,
        "layer": layer,
        "n": len(ids),
        "unique_ids": len(set(ids)),
        "label_counts": {v: labels.count(v) for v in sorted(set(labels))},
        "expected_label": args.expected_label,
        "score_stats": {
            "mean": float(probs[:, 1].mean()),
            "p50": float(np.percentile(probs[:, 1], 50)),
            "p90": float(np.percentile(probs[:, 1], 90)),
            "p95": float(np.percentile(probs[:, 1], 95)),
            "p99": float(np.percentile(probs[:, 1], 99)),
            "max": float(probs[:, 1].max()),
        },
        "thresholds": {},
    }
    n = len(ids)
    for name, threshold in thresholds.items():
        positives = [i for i, p in enumerate(probs) if float(p[1]) >= threshold]
        k = len(positives)
        lo, hi = wilson_ci(k, n)
        rates = {}
        for label in sorted(set(labels)):
            idx = [i for i, v in enumerate(labels) if v == label]
            if idx:
                kk = sum(1 for i in idx if float(probs[i][1]) >= threshold)
                rates[label] = kk / len(idx)
        summary["thresholds"][name] = {
            "threshold": threshold,
            "predicted_positive": k,
            "rate": k / n if n else None,
            "rate_ci95": [lo, hi],
            "rate_by_label": rates,
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"n={n} unique_ids={summary['unique_ids']} labels={summary['label_counts']}")
    print(f"{'threshold':>14} {'t':>8} {'positive':>8} {'rate':>7} {'95% CI':>16}")
    for name, p in summary["thresholds"].items():
        print(f"{name:>14} {p['threshold']:>8.4f} {p['predicted_positive']:>8} "
              f"{p['rate']:>7.3f} [{p['rate_ci95'][0]:.3f}, {p['rate_ci95'][1]:.3f}]")
    print(f"wrote {out_dir / 'predictions.jsonl'} and {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
