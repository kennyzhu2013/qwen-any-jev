#!/usr/bin/env python
"""Append the 2026-10-22 review batch to the v2 label set.

The 0729 file is human-confirmed.  The new batch (`10.22 复核结果.xlsx`) is
LLM/heuristic reviewed (`cq_agent_*`) and covers 2026-10-23..2026-11-04.
Both cohorts are kept in one canonical CSV; provenance is recorded in
`cohort` and `source_file`.

The 1447 human rows keep their frozen v1 split ids.  New rows are assigned
by near-duplicate cluster (char 3-gram SimHash, Hamming <= 4): whole clusters
go to one split, so a 643-row ASR template cannot leak across cal/dev/gate.
Clusters are balanced 40/20/40 within `(risk_level, scene_category)` strata.
New rows are never treated as human gold: `label_source` starts with
`cq_agent:` and the README / stats keep the two cohorts separate.

Outputs:
  assets/复核数据_v2.csv
  runs/labels_v2/extra_splits.json
  runs/labels_v2/summary_combined.json
  runs/labels_v2/review_queue_combined.csv
  runs/labels_v2/extra_details.jsonl

Usage:
  python eval/merge_review_batch.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import re
from pathlib import Path

from openpyxl import load_workbook

SPLIT_ORDER = ["cal", "dev", "gate"]
RATIOS = {"cal": 0.40, "dev": 0.20, "gate": 0.40}
SALT = "risk-jev-v2-extra-20261105"
NEW_COHORT = "agent_1022"
OLD_COHORT = "human_0729"
SHINGLE_SIZE = 3
HAMMING = 4
RISK_ORDER = {"合规": 0, "低风险": 1, "中风险": 2, "高风险": 3}

RISK_LEVELS = {"合规", "低风险", "中风险", "高风险"}
SCENE_OPTIONS = {
    "引流第三方平台", "贷款相关", "法律服务", "企业营销与招商服务",
    "商品推销", "商业地产", "违规催收", "其他",
}
FRAUD_TYPE_OPTIONS = {
    "个体工商户年报补录收费", "引导贷款用户添加第三方微信", "手机租赁套路贷",
    "套路运", "网贷平台退息退费", "证券投资类", "贷款相关-ab贷",
}

FIELDS = [
    "data_id", "content", "label_v2", "is_violation", "is_fraud", "risk_level",
    "scene_category", "scene_subtype", "label_detail", "label_source",
    "needs_review", "proposed_risk_level", "proposed_scene_category",
    "proposed_subtype", "proposed_is_fraud",
    "old_label", "old_category", "人工复核", "备注",
    "cohort", "source_file",
]
OLD_REQUIRED = set(FIELDS) - {"cohort", "source_file"}
NEW_HEADER = [
    "data_id", "content", "cq_agent_is_violation", "cq_agent_is_fraud",
    "cq_agent_risk_level", "cq_agent_scene_category", "cq_agent_scene_subtype",
    "cq_agent_source", "cq_agent_detail",
]


def validate_row(row: dict, where: str) -> None:
    def fail(msg: str) -> None:
        raise SystemExit(f"{where}: {msg}")

    if not row["data_id"]:
        fail("empty data_id")
    if not row["content"]:
        fail("empty content")
    if row["label_v2"] not in ("非涉诈", "涉诈"):
        fail(f"bad label_v2 {row['label_v2']!r}")
    if row["is_violation"] not in ("0", "1") or row["is_fraud"] not in ("0", "1"):
        fail("bad is_violation/is_fraud")
    iv, iff = int(row["is_violation"]), int(row["is_fraud"])
    if iff > iv:
        fail("is_fraud=1 but is_violation=0")
    if row["risk_level"] not in RISK_LEVELS:
        fail(f"bad risk_level {row['risk_level']!r}")
    if row["label_v2"] != ("涉诈" if iff else "非涉诈"):
        fail("label_v2/is_fraud mismatch")
    if not iv:
        if row["risk_level"] != "合规":
            fail(f"non-violation with risk {row['risk_level']!r}")
        if row["scene_category"] != "正常":
            fail(f"non-violation with scene {row['scene_category']!r}")
    elif not iff and row["scene_category"] not in SCENE_OPTIONS:
        fail(f"bad 4.1 scene {row['scene_category']!r}")
    if iff:
        if row["risk_level"] != "高风险":
            fail(f"fraud row not high risk: {row['risk_level']!r}")
        if row["scene_category"] not in FRAUD_TYPE_OPTIONS:
            fail(f"bad 4.2 type {row['scene_category']!r}")
    if row["needs_review"] not in ("0", "1"):
        fail(f"bad needs_review {row['needs_review']!r}")


def read_old(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = OLD_REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        rows = []
        seen = set()
        for lineno, raw in enumerate(reader, 1):
            row = {k: (v or "").strip() for k, v in raw.items()}
            row["cohort"] = OLD_COHORT
            row["source_file"] = "assets/0729复核人工确认数据.csv"
            validate_row(row, f"{path}:row{lineno}")
            if row["data_id"] in seen:
                raise SystemExit(f"{path}:row{lineno}: duplicate data_id")
            seen.add(row["data_id"])
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def read_new(path: Path) -> tuple[list[dict], list[dict]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    if len(wb.worksheets) != 1:
        raise SystemExit(f"{path}: expected one sheet, got {wb.sheetnames}")
    ws = wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    try:
        header = [str(x).strip() if x is not None else "" for x in next(it)]
    except StopIteration:
        raise SystemExit(f"{path}: empty sheet")
    if header != NEW_HEADER:
        raise SystemExit(f"{path}: unexpected columns {header}")

    rows, details = [], []
    seen = set()
    for lineno, values in enumerate(it, 2):
        if all(v is None or str(v).strip() == "" for v in values):
            continue
        raw = dict(zip(header, values))
        data_id = str(raw["data_id"] or "").strip()
        content = str(raw["content"] or "").strip()
        risk = str(raw["cq_agent_risk_level"] or "").strip()
        scene = str(raw["cq_agent_scene_category"] or "").strip()
        subtype = str(raw["cq_agent_scene_subtype"] or "").strip()
        source = str(raw["cq_agent_source"] or "").strip() or "unknown"
        iv = int(float(str(raw["cq_agent_is_violation"] or "0").strip()))
        iff = int(float(str(raw["cq_agent_is_fraud"] or "0").strip()))

        if not iv:
            scene, subtype = "正常", ""
        if subtype == "-":
            subtype = ""
        if iff and not subtype and scene == "引导贷款用户添加第三方微信":
            subtype = "引导添加第三方微信"

        try:
            detail = json.loads(raw["cq_agent_detail"] or "{}")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:row{lineno}: bad detail JSON: {exc}")
        explanation = str(detail.get("explanation") or "").strip()
        quotes = [
            str(q).strip() for q in (detail.get("evidenceQuotes") or [])
            if str(q).strip()
        ]
        label_detail = explanation
        if quotes:
            label_detail = (label_detail + " | 证据: " + "；".join(quotes)).strip(" |")

        row = {
            "data_id": data_id,
            "content": content,
            "label_v2": "涉诈" if iff else "非涉诈",
            "is_violation": str(iv),
            "is_fraud": str(iff),
            "risk_level": risk,
            "scene_category": scene,
            "scene_subtype": subtype,
            "label_detail": label_detail,
            "label_source": f"cq_agent:{source}",
            "needs_review": "1" if (risk in ("中风险", "高风险") or "heuristic" in source) else "0",
            "proposed_risk_level": "",
            "proposed_scene_category": "",
            "proposed_subtype": "",
            "proposed_is_fraud": "0",
            "old_label": "",
            "old_category": "",
            "人工复核": "",
            "备注": "",
            "cohort": NEW_COHORT,
            "source_file": str(path),
        }
        validate_row(row, f"{path}:row{lineno}")
        if data_id in seen:
            raise SystemExit(f"{path}:row{lineno}: duplicate data_id {data_id}")
        seen.add(data_id)
        rows.append(row)
        details.append({
            "data_id": data_id,
            "cohort": NEW_COHORT,
            "label_source": f"cq_agent:{source}",
            "risk_level": risk,
            "scene_category": scene,
            "scene_subtype": subtype,
            "detail": detail,
        })
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    return rows, details


def _shingles(text: str) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {
        text[i:i + SHINGLE_SIZE]
        for i in range(max(0, len(text) - SHINGLE_SIZE + 1))
    }


def _simhash(text: str) -> int:
    vector = [0] * 64
    for shingle in _shingles(text):
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if vector[bit] > 0:
            out |= 1 << bit
    return out


def near_duplicate_groups(rows: list[dict]) -> list[list[int]]:
    """Group near-duplicate contents (char 3-gram SimHash, Hamming <= 4)."""
    hashes = [_simhash(row["content"]) for row in rows]
    bands: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for i, value in enumerate(hashes):
        for band in range(4):
            bands[(band, (value >> (band * 16)) & 0xFFFF)].append(i)

    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ids in bands.values():
        if len(ids) < 2:
            continue
        for a, b in itertools.combinations(ids, 2):
            if (hashes[a] ^ hashes[b]).bit_count() <= HAMMING:
                union(a, b)

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return list(groups.values())


def _majority_stratum(rows: list[dict], idxs: list[int]) -> tuple[str, str, str]:
    risks = collections.Counter(rows[i]["risk_level"] for i in idxs)
    top_risk_count = max(risks.values())
    risk = max(
        (r for r, c in risks.items() if c == top_risk_count),
        key=lambda r: RISK_ORDER[r],
    )
    majority = [i for i in idxs if rows[i]["risk_level"] == risk]
    scenes = collections.Counter(rows[i]["scene_category"] for i in majority)
    if not scenes:
        scenes = collections.Counter(rows[i]["scene_category"] for i in idxs)
    scene = max(scenes.items(), key=lambda kv: (kv[1], kv[0]))[0]
    subtypes = collections.Counter(
        rows[i]["scene_subtype"] for i in majority
        if rows[i]["scene_category"] == scene
    )
    subtype = max(subtypes.items(), key=lambda kv: (kv[1], kv[0]))[0] if subtypes else ""
    return risk, scene, subtype


def assign_splits(
    rows: list[dict], frozen_split_of: dict[str, str]
) -> tuple[dict[str, str], dict]:
    """Assign new rows by near-duplicate cluster, not by individual row.

    A 643-row ASR template cluster exists in the new batch.  Splitting it
    across cal/dev/gate would leak the template into evaluation, so entire
    clusters are assigned as units.  Clusters that also contain a frozen old
    row are constrained to that old row's split.
    """
    clusters = []
    for idxs in near_duplicate_groups(rows):
        new_idxs = [i for i in idxs if rows[i]["cohort"] == NEW_COHORT]
        if not new_idxs:
            continue
        old_splits = {
            frozen_split_of[rows[i]["data_id"]]
            for i in idxs if rows[i]["cohort"] == OLD_COHORT
        }
        risk, scene, subtype = _majority_stratum(rows, idxs)
        clusters.append({
            "idxs": idxs,
            "new_idxs": new_idxs,
            "n": len(idxs),
            "new_n": len(new_idxs),
            "risk": risk,
            "scene": scene,
            "subtype": subtype,
            "mixed_risk": len({rows[i]["risk_level"] for i in idxs}) > 1,
            "old_splits": old_splits,
            "key": hashlib.sha256(
                "|".join(sorted(rows[i]["data_id"] for i in idxs)).encode("utf-8")
            ).hexdigest(),
        })

    constrained = [c for c in clusters if c["old_splits"]]
    new_only = [c for c in clusters if not c["old_splits"]]

    strata: dict[tuple[str, str], int] = collections.defaultdict(int)
    for c in clusters:
        strata[(c["risk"], c["scene"])] += c["new_n"]
    target = {
        key: {s: n * RATIOS[s] for s in SPLIT_ORDER}
        for key, n in strata.items()
    }
    current = {key: collections.Counter() for key in strata}
    current_all = collections.Counter()
    split_of: dict[str, str] = {}

    for c in constrained:
        split = max(
            SPLIT_ORDER,
            key=lambda s: (
                sum(1 for x in c["old_splits"] if x == s),
                -SPLIT_ORDER.index(s),
            ),
        )
        c["split"] = split
        for i in c["new_idxs"]:
            split_of[rows[i]["data_id"]] = split
        key = (c["risk"], c["scene"])
        current[key][split] += c["new_n"]
        current_all[split] += c["new_n"]

    for c in sorted(new_only, key=lambda x: (-x["n"], x["key"])):
        key = (c["risk"], c["scene"])
        split = max(
            SPLIT_ORDER,
            key=lambda s: (
                target[key][s] - current[key][s],
                c["new_n"] * RATIOS[s] - current_all[s],
                -SPLIT_ORDER.index(s),
            ),
        )
        c["split"] = split
        for i in c["new_idxs"]:
            split_of[rows[i]["data_id"]] = split
        current[key][split] += c["new_n"]
        current_all[split] += c["new_n"]

    mixed = [c for c in clusters if c["mixed_risk"]]
    mixed_rows = 0
    for c in mixed:
        for i in c["new_idxs"]:
            row = rows[i]
            if row["risk_level"] == c["risk"]:
                continue
            mixed_rows += 1
            row["needs_review"] = "1"
            note = "近重复簇内风险等级不一致，待复核"
            row["备注"] = f"{row['备注']}；{note}" if row["备注"] else note
            # 就高不就低：不因簇内多数标签而给出降级建议。
            proposed_risk = max(
                (row["risk_level"], c["risk"]), key=lambda r: RISK_ORDER[r]
            )
            if proposed_risk == row["risk_level"]:
                proposed_scene = row["scene_category"]
                proposed_subtype = row["scene_subtype"]
            else:
                proposed_scene = c["scene"]
                proposed_subtype = c["subtype"]
            row["proposed_risk_level"] = proposed_risk
            row["proposed_scene_category"] = proposed_scene
            row["proposed_subtype"] = proposed_subtype
            row["proposed_is_fraud"] = "1" if proposed_risk == "高风险" else "0"

    stats = {
        "n_clusters": len(clusters),
        "multi_row_clusters": sum(1 for c in clusters if c["n"] > 1),
        "largest_cluster": max((c["n"] for c in clusters), default=0),
        "mixed_risk_clusters": len(mixed),
        "mixed_risk_cluster_rows": sum(c["new_n"] for c in mixed),
        "mixed_risk_rows": mixed_rows,
        "old_new_clusters": sum(1 for c in clusters if c["old_splits"]),
        "old_cross_split_clusters": sum(
            1 for c in clusters if len(c["old_splits"]) > 1
        ),
        "split_sizes": dict(current_all),
    }
    return split_of, stats


def load_frozen_splits(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        rid: split for split, ids in payload["ids"].items() for rid in ids
    }


def count(rows: list[dict], key: str) -> dict[str, int]:
    return dict(collections.Counter(row[key] for row in rows))


def cohort_summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "label_v2": count(rows, "label_v2"),
        "is_violation": count(rows, "is_violation"),
        "is_fraud": count(rows, "is_fraud"),
        "risk_level": count(rows, "risk_level"),
        "scene_category": count(rows, "scene_category"),
        "label_source": count(rows, "label_source"),
        "needs_review": count(rows, "needs_review"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default="assets/0729复核人工确认数据_v2.csv")
    ap.add_argument("--new", default="assets/10.22 复核结果.xlsx")
    ap.add_argument("--frozen-splits", default="eval/splits.json")
    ap.add_argument("--out", default="assets/复核数据_v2.csv")
    ap.add_argument("--extra-splits", default="runs/labels_v2/extra_splits.json")
    ap.add_argument("--summary", default="runs/labels_v2/summary_combined.json")
    ap.add_argument("--queue", default="runs/labels_v2/review_queue_combined.csv")
    ap.add_argument("--details", default="runs/labels_v2/extra_details.jsonl")
    args = ap.parse_args()

    old_rows = read_old(Path(args.old))
    new_rows, details = read_new(Path(args.new))

    old_ids = {r["data_id"] for r in old_rows}
    new_ids = {r["data_id"] for r in new_rows}
    overlap = old_ids & new_ids
    if overlap:
        raise SystemExit(f"{len(overlap)} data_ids appear in both cohorts")

    frozen = load_frozen_splits(Path(args.frozen_splits))
    if set(frozen) != old_ids:
        raise SystemExit(
            f"{args.frozen_splits}: id mismatch "
            f"missing={len(old_ids - set(frozen))} extra={len(set(frozen) - old_ids)}"
        )

    rows = old_rows + new_rows
    extra_split_of, cluster_stats = assign_splits(rows, frozen)
    if set(extra_split_of) != new_ids:
        raise SystemExit("split assignment does not cover all new rows")

    write_csv(Path(args.out), rows)

    extra_ids = {s: [] for s in SPLIT_ORDER}
    for rid, split in extra_split_of.items():
        extra_ids[split].append(rid)
    for split in extra_ids:
        extra_ids[split].sort()
    Path(args.extra_splits).parent.mkdir(parents=True, exist_ok=True)
    Path(args.extra_splits).write_text(json.dumps({
        "version": "m0.5-v2-extra",
        "salt": SALT,
        "ratios": RATIOS,
        "source": args.new,
        "cohort": NEW_COHORT,
        "frozen_from": args.frozen_splits,
        "split_method": "near-duplicate clusters, greedy (risk, scene) balance",
        "cluster_stats": cluster_stats,
        "ids": extra_ids,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Combined split view (old frozen + new assignment).
    split_of = dict(frozen)
    split_of.update(extra_split_of)

    summary = {
        "source_old": args.old,
        "source_new": args.new,
        "n_old": len(old_rows),
        "n_new": len(new_rows),
        "n_rows": len(rows),
        "cohorts": count(rows, "cohort"),
        "label_v2": count(rows, "label_v2"),
        "is_violation": count(rows, "is_violation"),
        "is_fraud": count(rows, "is_fraud"),
        "risk_level": count(rows, "risk_level"),
        "scene_category": count(rows, "scene_category"),
        "label_source": count(rows, "label_source"),
        "needs_review": count(rows, "needs_review"),
        "by_cohort": {
            OLD_COHORT: cohort_summary(old_rows),
            NEW_COHORT: cohort_summary(new_rows),
        },
        "clusters": cluster_stats,
        "by_split": {},
    }
    for split in SPLIT_ORDER:
        sub = [r for r in rows if split_of[r["data_id"]] == split]
        summary["by_split"][split] = {
            "n": len(sub),
            "fraud": sum(int(r["is_fraud"]) for r in sub),
            "violation": sum(int(r["is_violation"]) for r in sub),
            "review": sum(int(r["needs_review"]) for r in sub),
            "cohort": count(sub, "cohort"),
            "risk_level": count(sub, "risk_level"),
        }
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with Path(args.queue).open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if row["needs_review"] == "1":
                writer.writerow(row)

    with Path(args.details).open("w", encoding="utf-8") as fh:
        for record in details:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {args.out}: {len(rows)} rows "
          f"({len(old_rows)} human + {len(new_rows)} agent)")
    print(f"labels: {summary['label_v2']} | is_fraud: {summary['is_fraud']} | "
          f"risk: {summary['risk_level']}")
    for split in SPLIT_ORDER:
        x = summary["by_split"][split]
        print(f"  {split}: n={x['n']} fraud={x['fraud']} "
              f"violation={x['violation']} review={x['review']} cohort={x['cohort']}")
    print(f"review queue: {sum(1 for r in rows if r['needs_review'] == '1')} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
