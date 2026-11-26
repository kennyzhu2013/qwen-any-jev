#!/usr/bin/env python
"""Apply the external stratified sample: estimate noise and auto-adopt.

Reads ``review_sample.csv`` with the manual ``复核_is_fraud`` column filled,
computes per-scene precision with a Wilson 95% interval, and auto-adopts the
remaining (unsampled) rows of a scene only when

    precision >= 0.90 and wilson_low >= 0.70.

The probe column is never used here.  Outputs:

  * ``sample_noise_estimates.csv``: per-scene precision/CI/adoption decision;
  * ``labels_external_final.csv``: confirmed + sampled positives + auto-adopted;
  * ``auto_adopted_positive.csv``: the auto-adopted rows only.

Usage:
  python eval/apply_external_sample.py
  python eval/apply_external_sample.py --sample runs/external_cq/review_sample.xlsx
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import wilson_ci  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    """Read a CSV, or the last (data) sheet of an XLSX review workbook."""
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[-1]]
        iterator = ws.iter_rows(values_only=True)
        headers = ["" if h is None else str(h) for h in next(iterator)]
        return [
            {h: ("" if v is None else str(v)) for h, v in zip(headers, row)}
            for row in iterator
        ]
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def verdict(row: dict) -> int | None:
    value = (row.get("复核_is_fraud") or "").strip()
    if value in ("0", "1"):
        return int(value)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="runs/external_cq/review_sample.csv")
    ap.add_argument("--plan", default="runs/external_cq/sampling_plan.csv")
    ap.add_argument("--confirmed", default="runs/external_cq/confirmed_positive.csv")
    ap.add_argument("--out-dir", default="runs/external_cq")
    ap.add_argument("--min-precision", type=float, default=0.90)
    ap.add_argument("--min-wilson-low", type=float, default=0.70)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sample = read_csv(Path(args.sample))
    plan = read_csv(Path(args.plan))
    confirmed = read_csv(Path(args.confirmed))

    reviewed = [r for r in sample if verdict(r) is not None]
    print(f"sample rows: {len(sample)}, reviewed: {len(reviewed)}")
    if not reviewed:
        print("no verdicts filled yet; nothing to apply")
        return 0

    by_scene: dict[str, list[dict]] = {}
    for r in reviewed:
        by_scene.setdefault(r["scene"], []).append(r)

    estimates = []
    qualify: dict[str, bool] = {}
    for scene, rows in sorted(by_scene.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        n = len(rows)
        k = sum(1 for r in rows if verdict(r) == 1)
        p = k / n
        lo, hi = wilson_ci(k, n)
        ok = (p >= args.min_precision) and (lo >= args.min_wilson_low)
        qualify[scene] = ok
        remaining_n = sum(1 for r in plan if r["scene"] == scene and r["strategy"] == "待定")
        estimates.append({
            "scene": scene, "reviewed": n, "positive": k, "precision": p,
            "wilson_low": lo, "wilson_high": hi,
            "min_precision": args.min_precision, "min_wilson_low": args.min_wilson_low,
            "qualifies": int(ok), "remaining_n": remaining_n,
            "auto_adopted_n": remaining_n if ok else 0,
        })

    fields = ["scene", "reviewed", "positive", "precision", "wilson_low",
              "wilson_high", "min_precision", "min_wilson_low", "qualifies",
              "remaining_n", "auto_adopted_n"]
    with (out_dir / "sample_noise_estimates.csv").open(
            "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for e in estimates:
            writer.writerow({**e, "precision": f"{e['precision']:.4f}",
                             "wilson_low": f"{e['wilson_low']:.4f}",
                             "wilson_high": f"{e['wilson_high']:.4f}"})

    # Final positives: user-confirmed + sampled positives + qualifying remainder.
    final: list[dict] = []
    for r in confirmed:
        final.append({
            "id": r["id"], "source_file": r["source_file"],
            "source_date": r["source_date"], "agent_res": r["agent_res"],
            "agent_conf": r["agent_conf"], "probe_p": r["probe_p"],
            "is_fraud": 1, "is_violation": 1, "risk_level": "高风险",
            "scene_category": r["scene_category"], "scene_subtype": r["scene_subtype"],
            "label_source": "user_confirmed_ab贷",
        })
    for r in reviewed:
        scene = (r.get("复核_scene") or r["scene"]).strip()
        subtype = (r.get("复核_subtype") or r["subtype"]).strip()
        if verdict(r) == 1:
            final.append({
                "id": r["id"], "source_file": r["source_file"],
                "source_date": r["source_date"], "agent_res": r["agent_res"],
                "agent_conf": r["agent_conf"], "probe_p": r["probe_p"],
                "is_fraud": 1, "is_violation": 1, "risk_level": "高风险",
                "scene_category": scene, "scene_subtype": subtype,
                "label_source": "sample_review",
            })
        else:
            normal = scene in ("正常", "合规", "")
            final.append({
                "id": r["id"], "source_file": r["source_file"],
                "source_date": r["source_date"], "agent_res": r["agent_res"],
                "agent_conf": r["agent_conf"], "probe_p": r["probe_p"],
                "is_fraud": 0, "is_violation": 0 if normal else 1,
                "risk_level": "合规" if normal else "",
                "scene_category": scene or "正常",
                "scene_subtype": subtype,
                "label_source": "sample_review_negative",
            })
    auto_adopted = 0
    for r in plan:
        if r["strategy"] != "待定":
            continue
        if qualify.get(r["scene"], False):
            final.append({
                "id": r["id"], "source_file": r["source_file"],
                "source_date": r["source_date"], "agent_res": r["agent_res"],
                "agent_conf": r["agent_conf"], "probe_p": r["probe_p"],
                "is_fraud": 1, "is_violation": 1, "risk_level": "高风险",
                "scene_category": r["scene"], "scene_subtype": r["subtype"],
                "label_source": "sample_auto_adopt",
            })
            auto_adopted += 1

    fields = ["id", "source_file", "source_date", "agent_res", "agent_conf",
              "probe_p", "is_fraud", "is_violation", "risk_level",
              "scene_category", "scene_subtype", "label_source"]
    with (out_dir / "labels_external_final.csv").open(
            "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(final)

    with (out_dir / "auto_adopted_positive.csv").open(
            "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(r for r in final if r["label_source"] == "sample_auto_adopt")

    print(f"{'scene':<24} {'reviewed':>8} {'pos':>4} {'prec':>6} {'wilson_lo':>9} "
          f"{'adopt':>6} {'remaining':>9}")
    for e in estimates:
        print(f"{e['scene']:<24} {e['reviewed']:>8} {e['positive']:>4} "
              f"{e['precision']:>6.2f} {e['wilson_low']:>9.2f} "
              f"{'yes' if e['qualifies'] else 'no':>6} {e['remaining_n']:>9}")
    positives = [r for r in final if r["is_fraud"] == 1]
    negatives = [r for r in final if r["is_fraud"] == 0]
    print(f"final labels: {len(final)} (positive={len(positives)}, negative={len(negatives)})")
    print(f"  positives: confirmed={len(confirmed)}, "
          f"sample={sum(1 for r in reviewed if verdict(r)==1)}, "
          f"auto_adopted={auto_adopted}")
    print(f"wrote {out_dir}/sample_noise_estimates.csv, labels_external_final.csv, "
          f"auto_adopted_positive.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
