#!/usr/bin/env python
"""Merge the 10.22 human review and the Chongqing normal batch into v2.

Inputs
------
- ``assets/复核数据_v2.csv``: current 7,800-row dataset.  The 242-row
  ``spec_review`` adjudication is already backfilled and is treated as final.
- ``assets/10.22agent和人工复核结果.xlsx``: human review of the 6,353-row
  10.22 batch (``human_label`` / ``human_risk_level`` / ``human_comment``).
- ``assets/重庆行业卡人工复核正常数据.xlsx``: 1,582 normal rows (one exact
  duplicate data_id, deduplicated to 1,581).

Rules
-----
1. For 10.22 rows whose current ``label_source`` is ``spec_review`` (107
   rows), keep the dedicated spec adjudication.  All other 10.22 rows take
   the human review.
2. ``human_label`` is the old coarse 正常/涉诈 label; 4.2 fraud is derived
   from the ``human_comment`` prefix (the 4.2 type list).  ``human_risk_level``
   supplies the new risk level.
3. Any ``is_violation=0`` row is normalised to ``scene_category=正常`` and
   ``scene_subtype=正常``.
4. ``proposed_*`` is historical; it is cleared for all reviewed rows.  The
   pre-review proposals remain in ``review_queue_combined.csv``.
5. New normal rows form a ``human_cq_normal`` cohort.  Existing 7,800 split
   assignments are frozen; new rows are assigned by near-duplicate cluster
   to keep 40/20/40.

Usage
-----
  python eval/merge_human_review.py            # dry run
  python eval/merge_human_review.py --write    # backup + write
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sys
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_review_batch import near_duplicate_groups  # noqa: E402

MAIN = Path("assets/复核数据_v2.csv")
HUMAN_REVIEW = Path("assets/10.22agent和人工复核结果.xlsx")
NORMAL = Path("assets/重庆行业卡人工复核正常数据.xlsx")
SPLITS = Path("eval_v2/splits.json")
EXTRA_SPLITS = Path("runs/labels_v2/extra_splits.json")
SUMMARY = Path("runs/labels_v2/human_review_merge_summary.json")

SCENE_OPTIONS = {
    "引流第三方平台", "贷款相关", "法律服务", "企业营销与招商服务",
    "商品推销", "商业地产", "违规催收", "其他", "正常",
}
FRAUD_TYPES = {
    "个体工商户年报补录收费", "引导贷款用户添加第三方微信", "手机租赁套路贷",
    "套路运", "网贷平台退息退费", "证券投资类", "贷款相关-ab贷",
}
FRAUD_SUBTYPE = {
    "贷款相关-ab贷": "ab贷",
    "个体工商户年报补录收费": "年报补录收费",
    "引导贷款用户添加第三方微信": "引导添加第三方微信",
    "证券投资类": "证券投资荐股",
    "手机租赁套路贷": "手机租赁套现",
    "网贷平台退息退费": "网贷退息退费",
    "套路运": "套路运",
}
SPLIT_ORDER = ("cal", "dev", "gate")
RATIOS = {"cal": 0.40, "dev": 0.20, "gate": 0.40}
PROPOSED = (
    "proposed_risk_level", "proposed_scene_category",
    "proposed_subtype", "proposed_is_fraud",
)


def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def load_xlsx(path: Path, sheet: str | None = None) -> list[dict]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.active
    it = ws.iter_rows(values_only=True)
    header = [str(x) for x in next(it)]
    rows = [dict(zip(header, r)) for r in it if r and r[0]]
    wb.close()
    return rows


def category_of(comment: str) -> str:
    text = str(comment or "").strip()
    for sep in ("：", ":"):
        if sep in text:
            return text.split(sep, 1)[0].strip()
    return text[:20]


def review_row(row: dict, human: dict, stats: collections.Counter) -> dict:
    category = category_of(human.get("human_comment"))
    if category not in SCENE_OPTIONS | FRAUD_TYPES:
        raise SystemExit(f"{row['data_id']}: unknown category {category!r}")
    risk = str(human.get("human_risk_level") or "").strip()
    if risk not in ("合规", "低风险", "中风险", "高风险"):
        raise SystemExit(f"{row['data_id']}: bad human risk {risk!r}")
    fraud = int(category in FRAUD_TYPES)
    if fraud and risk != "高风险":
        raise SystemExit(f"{row['data_id']}: fraud but risk={risk}")
    if fraud:
        subtype = FRAUD_SUBTYPE[category]
    elif category == "正常":
        subtype = "正常"
    elif category == row["scene_category"]:
        subtype = row["scene_subtype"]
    else:
        subtype = ""
    new = dict(row)
    new.update({
        "label_v2": "涉诈" if fraud else "非涉诈",
        "is_violation": "0" if (risk == "合规" and not fraud) else "1",
        "is_fraud": str(fraud),
        "risk_level": risk,
        "scene_category": category,
        "scene_subtype": subtype,
        "label_detail": str(human.get("human_comment") or "").strip(),
        "label_source": "human_1022",
        "needs_review": "0",
    })
    if category != row["scene_category"]:
        stats["scene_changed"] += 1
    if risk != row["risk_level"]:
        stats["risk_changed"] += 1
    if fraud != int(row["is_fraud"]):
        stats["fraud_changed"] += 1
    return new


def assign_normal_splits(
    all_rows: list[dict], frozen_split_of: dict[str, str], new_ids: set[str]
) -> tuple[dict[str, str], dict]:
    groups = near_duplicate_groups(
        [{"data_id": r["data_id"], "content": r["content"]} for r in all_rows]
    )
    assigned = dict(frozen_split_of)
    counts: collections.Counter[str] = collections.Counter()
    new_only: list[list[str]] = []
    bridged = 0
    for group in groups:
        ids = [all_rows[i]["data_id"] for i in group]
        new_here = [i for i in ids if i in new_ids]
        if not new_here:
            continue
        existing = {frozen_split_of[i] for i in ids if i in frozen_split_of}
        if existing:
            if len(existing) > 1:
                bridged += 1
                continue
            split = existing.pop()
            for i in new_here:
                assigned[i] = split
            counts[split] += len(new_here)
        else:
            new_only.append(new_here)

    target_n = len(new_ids)
    for group in sorted(new_only, key=lambda g: (-len(g), g[0])):
        split = max(
            SPLIT_ORDER,
            key=lambda s: RATIOS[s] * target_n - counts[s],
        )
        for i in group:
            assigned[i] = split
        counts[split] += len(group)

    missing = new_ids - set(assigned)
    if missing:
        raise SystemExit(f"{len(missing)} new rows unassigned")
    stats = {
        "n_clusters": len(groups),
        "multi_row_clusters": sum(len(g) > 1 for g in groups),
        "largest_cluster": max(len(g) for g in groups),
        "normal_only_clusters": len(new_only),
        "group_ids": len({tuple(g) for g in new_only}),
        "bridged_clusters_skipped": bridged,
        "normal_split_sizes": {s: counts[s] for s in SPLIT_ORDER},
    }
    return assigned, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows, fields = load_csv(MAIN)
    by_id = {r["data_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise SystemExit("duplicate data_id in main CSV")
    human_rows = load_xlsx(HUMAN_REVIEW)
    human = {str(r["data_id"]).strip(): r for r in human_rows}
    if len(human) != len(human_rows):
        raise SystemExit("duplicate data_id in human review xlsx")
    missing = set(human) - set(by_id)
    if missing:
        raise SystemExit(f"{len(missing)} human review ids missing from main")

    stats = collections.Counter()
    merged: list[dict] = []
    for row in rows:
        if row["data_id"] not in human:
            merged.append(dict(row))
            continue
        if row["label_source"] == "spec_review":
            # Dedicated adjudication wins over the batch human review.
            merged.append(dict(row))
            stats["spec_preserved"] += 1
            continue
        merged.append(review_row(row, human[row["data_id"]], stats))
        stats["human_applied"] += 1

    # Normalise compliant rows and retire pre-review proposals.
    for row in merged:
        if row["is_violation"] == "0":
            if row["scene_category"] != "正常" or row["scene_subtype"] != "正常":
                stats["normalised_compliant"] += 1
            row["scene_category"] = "正常"
            row["scene_subtype"] = "正常"
        if any(row[k] for k in PROPOSED):
            stats["proposed_cleared"] += 1
        for key in PROPOSED:
            row[key] = ""
        row["needs_review"] = "0"

    # Chongqing normal batch: exact duplicate data_id kept once.
    normal_rows = load_xlsx(NORMAL)
    seen: set[str] = set()
    normal_unique: list[dict] = []
    for raw in normal_rows:
        did = str(raw["data_id"]).strip()
        if did in seen:
            stats["normal_duplicate_dropped"] += 1
            continue
        seen.add(did)
        if did in by_id:
            raise SystemExit(f"normal id already in main: {did}")
        normal_unique.append(raw)
    for raw in normal_unique:
        merged.append({
            "data_id": str(raw["data_id"]).strip(),
            "content": str(raw["content"]),
            "label_v2": "非涉诈",
            "is_violation": "0",
            "is_fraud": "0",
            "risk_level": "合规",
            "scene_category": "正常",
            "scene_subtype": "正常",
            "label_detail": "正常：重庆行业卡人工复核正常数据",
            "label_source": "human_cq_normal",
            "needs_review": "0",
            "proposed_risk_level": "",
            "proposed_scene_category": "",
            "proposed_subtype": "",
            "proposed_is_fraud": "",
            "old_label": "正常",
            "old_category": "",
            "人工复核": "",
            "备注": "",
            "cohort": "human_cq_normal",
            "source_file": str(NORMAL),
        })
    stats["normal_added"] = len(normal_unique)

    # Final schema checks.
    if len({r["data_id"] for r in merged}) != len(merged):
        raise SystemExit("duplicate data_id after merge")
    for r in merged:
        iv, iff = int(r["is_violation"]), int(r["is_fraud"])
        if iff > iv:
            raise SystemExit(f"{r['data_id']}: fraud without violation")
        if r["label_v2"] != ("涉诈" if iff else "非涉诈"):
            raise SystemExit(f"{r['data_id']}: label_v2 mismatch")
        if iv == 0 and (r["risk_level"] != "合规" or r["scene_category"] != "正常"):
            raise SystemExit(f"{r['data_id']}: bad compliant row")
        if iff and (r["risk_level"] != "高风险" or r["scene_category"] not in FRAUD_TYPES):
            raise SystemExit(f"{r['data_id']}: bad fraud row")
        if iv and not iff and r["scene_category"] not in SCENE_OPTIONS - {"正常"}:
            raise SystemExit(f"{r['data_id']}: bad violation scene")

    # Split assignment.
    splits_payload = json.loads(SPLITS.read_text(encoding="utf-8"))
    frozen = {i: s for s, ids in splits_payload["ids"].items() for i in ids}
    new_ids = {r["data_id"] for r in merged} - set(frozen)
    assigned, cluster_stats = assign_normal_splits(merged, frozen, new_ids)

    extra = json.loads(EXTRA_SPLITS.read_text(encoding="utf-8"))
    extra_ids = {s: set(extra["ids"][s]) for s in SPLIT_ORDER}
    for did in new_ids:
        extra_ids[assigned[did]].add(did)
    extra["ids"] = {s: sorted(extra_ids[s]) for s in SPLIT_ORDER}
    extra["cohort"] = ["agent_1022", "human_cq_normal"]
    extra["source"] = [str(EXTRA_SPLITS), str(NORMAL)]
    extra["split_method"] = (
        "frozen eval_v2 splits for the existing 7,800 rows; new normal rows "
        "assigned by near-duplicate cluster with greedy 40/20/40"
    )
    extra["cluster_stats"] = {
        **cluster_stats,
        "split_sizes": {
            s: len(extra["ids"][s]) for s in SPLIT_ORDER
        },
        "frozen_cross_split_clusters": 0,
    }

    summary = {
        "main_csv": str(MAIN),
        "human_review": str(HUMAN_REVIEW),
        "normal_batch": str(NORMAL),
        "n_main_before": len(rows),
        "n_main_after": len(merged),
        "human_applied": stats["human_applied"],
        "spec_preserved": stats["spec_preserved"],
        "normal_added": stats["normal_added"],
        "normal_duplicate_dropped": stats["normal_duplicate_dropped"],
        "scene_changed": stats["scene_changed"],
        "risk_changed": stats["risk_changed"],
        "fraud_changed": stats["fraud_changed"],
        "normalised_compliant": stats["normalised_compliant"],
        "proposed_cleared": stats["proposed_cleared"],
        "risk_level": dict(collections.Counter(r["risk_level"] for r in merged)),
        "is_fraud": dict(collections.Counter(r["is_fraud"] for r in merged)),
        "is_violation": dict(collections.Counter(r["is_violation"] for r in merged)),
        "cohort": dict(collections.Counter(r["cohort"] for r in merged)),
        "label_source": dict(collections.Counter(r["label_source"] for r in merged)),
        "split_sizes": dict(collections.Counter(assigned.values())),
        "clusters": cluster_stats,
        "precedence": "spec_review rows preserved; other 10.22 rows use human_1022",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.write:
        print("[dry-run] no files written; add --write")
        return 0

    backup = MAIN.with_name("复核数据_v2.pre_human1022.bak.csv")
    if not backup.exists():
        shutil.copy2(MAIN, backup)
        print(f"backup -> {backup}")
    with MAIN.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    EXTRA_SPLITS.write_text(
        json.dumps(extra, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {MAIN} ({len(merged)} rows)")
    print(f"wrote {EXTRA_SPLITS}")
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
