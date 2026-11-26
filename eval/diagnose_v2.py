#!/usr/bin/env python
"""Score the frozen v1 probe against the v2 `is_fraud` labels.

The v1 probe was trained on the old coarse `正常 / 涉诈（含违规）` target.
Re-scoring its frozen probabilities against `is_fraud` isolates the effect of
the label redefinition: every 4.1-only violation becomes a negative, so the
old model's false positives are mostly low/medium/high violations, not normal
calls.

This is a diagnostic, not a v2 model result.  It does not retrain or tune
anything.  Only the `human_0729` cohort has frozen v1 predictions; the
`agent_1022` rows in the combined CSV are skipped.

Usage:
  python eval/diagnose_v2.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import average_precision, recall_at_fpr, roc_auc  # noqa: E402

SPLITS = ("cal", "dev", "gate")


def load_predictions(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["id"]] = rec
    return out


def load_labels(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {row["data_id"]: row for row in csv.DictReader(fh)}


def evaluate(ids: list[str], preds: dict, labels: dict) -> dict:
    y = np.asarray([int(labels[i]["is_fraud"]) for i in ids], dtype=int)
    score = np.asarray([float(preds[i]["probs"][1]) for i in ids], dtype=float)
    pred = (score >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    npos = int(y.sum())
    nneg = len(y) - npos
    fp_by_scene = collections.Counter(
        labels[i]["scene_category"] for i, a, b in zip(ids, y, pred)
        if a == 0 and b == 1
    )
    fn_by_scene = collections.Counter(
        labels[i]["scene_category"] for i, a, b in zip(ids, y, pred)
        if a == 1 and b == 0
    )
    return {
        "n": len(ids),
        "fraud": npos,
        "non_fraud": nneg,
        "threshold": 0.5,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "fraud_recall": tp / npos if npos else None,
        "non_fraud_recall": tn / nneg if nneg else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "auroc": roc_auc(y, score),
        "average_precision": average_precision(y, score),
        "recall_at_fpr": {
            f"{p:.2f}": recall_at_fpr(y, score, p)["recall"]
            for p in (0.05, 0.10, 0.20)
        },
        "fp_by_scene": dict(fp_by_scene.most_common()),
        "fn_by_fraud_type": dict(fn_by_scene.most_common()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default="runs/m0.5/probe/predictions.jsonl")
    ap.add_argument("--labels", default="assets/复核数据_v2.csv")
    ap.add_argument("--out", default="runs/labels_v2/probe_v2_diagnostic.json")
    args = ap.parse_args()

    preds = load_predictions(Path(args.predictions))
    labels = load_labels(Path(args.labels))
    missing = set(preds) - set(labels)
    if missing:
        raise SystemExit(f"{len(missing)} predictions have no v2 label")
    skipped = set(labels) - set(preds)

    payload = {
        "predictions": args.predictions,
        "labels": args.labels,
        "note": "v1 probe scored against v2 is_fraud; diagnostic only",
        "cohort": "human_0729 (only rows with frozen v1 predictions)",
        "skipped_labels": len(skipped),
    }
    if skipped:
        print(f"note: ignoring {len(skipped)} labels without v1 predictions")
    print(f"{'split':>5} {'n':>4} {'fraud':>5} {'recall':>6} {'tnr':>5} "
          f"{'prec':>5} {'auroc':>6} {'ap':>6} {'r@fpr10':>7}")
    for split in SPLITS:
        ids = [i for i, r in preds.items() if r["split"] == split]
        if not ids:
            continue
        m = evaluate(ids, preds, labels)
        payload[split] = m
        print(f"{split:>5} {m['n']:>4} {m['fraud']:>5} "
              f"{m['fraud_recall']:>6.3f} {m['non_fraud_recall']:>5.3f} "
              f"{m['precision']:>5.3f} {m['auroc']:>6.4f} "
              f"{m['average_precision']:>6.4f} "
              f"{m['recall_at_fpr']['0.10']:>7.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
