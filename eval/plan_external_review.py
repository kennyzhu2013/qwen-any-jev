#!/usr/bin/env python
"""Plan the external label audit: confirmed rows, stratified sample, rest.

Policy (2026-11-20):
  * all forward (1103-1113) ``贷款相关-ab贷`` agent-fraud rows are user-confirmed
    positives;
  * the remaining agent-fraud rows are stratified by scene and sampled for a
    noise estimate (n<=10: review all; otherwise 20% with min 10 / max 20);
  * rows not sampled stay ``待定`` until the sample supports auto-adoption
    (Wilson lower bound >= 0.70 for the scene's precision).  The probe is NOT
    used as corroboration: it is the model being evaluated.

Usage:
  python eval/plan_external_review.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.external_labels import parse_agent_result, suggested_v2  # noqa: E402

SEED = 20261120
CONFIRMED_PREFIX = "chongqing20261103-1113"
CONFIRMED_SCENE = "贷款相关-ab贷"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in load_jsonl(path)}


def sample_size(n: int) -> int:
    # 10/10 is the smallest perfect sample whose Wilson lower bound reaches
    # 0.70, the auto-adoption threshold; smaller strata are reviewed in full.
    if n <= 10:
        return n
    return min(20, max(10, math.ceil(0.20 * n)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--normal-conflict", default="runs/external_cq/normal_1030_unmatch.jsonl")
    ap.add_argument("--normal-preds",
                    default="runs/external_cq/normal_1030_unmatch/predictions.jsonl")
    ap.add_argument("--forward", default="runs/external_cq/forward_1103_1113.jsonl")
    ap.add_argument("--forward-preds",
                    default="runs/external_cq/forward_1103_1113/predictions.jsonl")
    ap.add_argument("--out-dir", default="runs/external_cq")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        (Path(args.normal_conflict), load_predictions(Path(args.normal_preds))),
        (Path(args.forward), load_predictions(Path(args.forward_preds))),
    ]
    candidates = []
    for gold_path, preds in sources:
        for g in load_jsonl(gold_path):
            if not g["agent_res"].startswith("涉诈"):
                continue
            parsed = parse_agent_result(g["agent_res"])
            sug = suggested_v2(parsed)
            pred = preds.get(g["id"])
            candidates.append({
                "id": g["id"],
                "state": g.get("state") or "",
                "source_file": g.get("source_file") or "",
                "source_date": g.get("source_date") or "",
                "old_label": g.get("old_label") or "",
                "agent_res": g["agent_res"],
                "agent_conf": float(g.get("agent_conf") or 0.0),
                "scene": parsed["scene"],
                "subtype": parsed["subtype"] or "",
                "probe_p": float(pred["probs"][1]) if pred else None,
                "suggested": sug,
            })
    confirmed = [
        r for r in candidates
        if r["source_file"].startswith(CONFIRMED_PREFIX) and r["scene"] == CONFIRMED_SCENE
    ]
    confirmed_ids = {r["id"] for r in confirmed}
    remaining = [r for r in candidates if r["id"] not in confirmed_ids]

    with (out_dir / "confirmed_positive.csv").open(
            "w", encoding="utf-8-sig", newline="") as fh:
        fields = ["id", "source_file", "source_date", "agent_res", "agent_conf",
                  "probe_p", "is_fraud", "is_violation", "risk_level",
                  "scene_category", "scene_subtype", "label_source", "note"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in sorted(confirmed, key=lambda x: x["id"]):
            writer.writerow({
                "id": r["id"], "source_file": r["source_file"],
                "source_date": r["source_date"], "agent_res": r["agent_res"],
                "agent_conf": r["agent_conf"],
                "probe_p": "" if r["probe_p"] is None else f"{r['probe_p']:.6f}",
                "is_fraud": 1, "is_violation": 1, "risk_level": "高风险",
                "scene_category": r["scene"], "scene_subtype": r["subtype"],
                "label_source": "user_confirmed_ab贷",
                "note": "用户确认；其中 2 条 probe P<0.5（conf 0.72 / 0.90），建议快速复核",
            })

    # Stratified sample by scene; use a fixed seed.
    rng = random.Random(SEED)
    by_scene = collections.defaultdict(list)
    for r in remaining:
        by_scene[r["scene"]].append(r)
    sample_ids: set[str] = set()
    summary = []
    for scene, rows in sorted(by_scene.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rows_sorted = sorted(rows, key=lambda r: r["id"])
        k = sample_size(len(rows_sorted))
        picked = rng.sample(rows_sorted, k)
        sample_ids.update(r["id"] for r in picked)
        summary.append({
            "scene": scene, "n": len(rows_sorted), "sample_n": k,
            "remaining_n": len(rows_sorted) - k,
            "probe_mean_p": sum((r["probe_p"] or 0.0) for r in rows_sorted) / len(rows_sorted),
        })

    with (out_dir / "sampling_plan.csv").open(
            "w", encoding="utf-8-sig", newline="") as fh:
        fields = ["id", "source_file", "source_date", "agent_res", "agent_conf",
                  "scene", "subtype", "probe_p", "strategy",
                  "复核_is_fraud", "复核_scene", "复核_subtype", "复核依据", "复核人", "复核日期"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in sorted(remaining, key=lambda x: (x["scene"], -(x["probe_p"] or 0), x["id"])):
            writer.writerow({
                "id": r["id"], "source_file": r["source_file"],
                "source_date": r["source_date"], "agent_res": r["agent_res"],
                "agent_conf": r["agent_conf"], "scene": r["scene"],
                "subtype": r["subtype"],
                "probe_p": "" if r["probe_p"] is None else f"{r['probe_p']:.6f}",
                "strategy": "抽样复核" if r["id"] in sample_ids else "待定",
            })

    with (out_dir / "review_sample.csv").open(
            "w", encoding="utf-8-sig", newline="") as fh:
        fields = ["id", "source_file", "source_date", "agent_res", "agent_conf",
                  "scene", "subtype", "probe_p", "内容",
                  "复核_is_fraud", "复核_scene", "复核_subtype", "复核依据", "复核人", "复核日期"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in sorted(remaining, key=lambda x: (x["scene"], -(x["probe_p"] or 0), x["id"])):
            if r["id"] not in sample_ids:
                continue
            writer.writerow({
                "id": r["id"], "source_file": r["source_file"],
                "source_date": r["source_date"], "agent_res": r["agent_res"],
                "agent_conf": r["agent_conf"], "scene": r["scene"],
                "subtype": r["subtype"],
                "probe_p": "" if r["probe_p"] is None else f"{r['probe_p']:.6f}",
                "内容": r["state"],
            })

    with (out_dir / "sampling_summary.csv").open(
            "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["scene", "n", "sample_n", "remaining_n", "probe_mean_p"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"confirmed positives: {len(confirmed)} (saved confirmed_positive.csv)")
    print(f"remaining: {len(remaining)}, sampled: {len(sample_ids)}")
    print(f"{'scene':<24} {'n':>4} {'sample':>7} {'remain':>7} {'probe_mean':>11}")
    for s in summary:
        print(f"{s['scene']:<24} {s['n']:>4} {s['sample_n']:>7} "
              f"{s['remaining_n']:>7} {s['probe_mean_p']:>11.3f}")
    print(f"wrote {out_dir}/sampling_plan.csv, review_sample.csv, sampling_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
