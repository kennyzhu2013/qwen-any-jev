#!/usr/bin/env python
"""Build v2 evaluation artifacts from the combined relabelled CSV.

The v1 M0.5 artifacts in `eval/` stay frozen.  This script writes a separate
`eval_v2/` tree so the binary target can change from the old coarse
`正常 / 涉诈（含违规）` label to the new standard's `is_fraud` (4.2 only).

The combined CSV contains two cohorts: `human_0729` (1447 human-confirmed
rows) and `agent_1022` (the 2026-10-23..11-04 batch labelled by `cq_agent_*`).
The human rows keep their frozen `eval/splits.json` ids; the agent rows use
`runs/labels_v2/extra_splits.json` (near-duplicate clusters, no cluster
crosses a split).

Outputs (eval_v2/):
  goldens.jsonl             binary 非涉诈 / 涉诈 (is_fraud target)
  goldens_scene.jsonl       8-way 4.1 scene classification (all violations)
  goldens_fraud_type.jsonl  7-way 4.2 fraud type (is_fraud rows only)
  splits.json               frozen human ids + agent batch ids
  stats.json                distributions, cohort audit, cluster stats
  README.md                 label semantics and known limitations

Usage:
  python eval/prepare_v2.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

QUESTION = "请判断该通话是否属于4.2涉诈场景（而非仅属于4.1违规）："
OPTIONS = ["非涉诈", "涉诈"]
SCENE_QUESTION = "请判断该通话的违规业务大类："
SCENE_OPTIONS = [
    "引流第三方平台",
    "贷款相关",
    "法律服务",
    "企业营销与招商服务",
    "商品推销",
    "商业地产",
    "违规催收",
    "其他",
]
FRAUD_TYPE_QUESTION = "请判断该通话的4.2涉诈类型："
# Only the fraud types observed in the 2026-07-29 snapshot.  The new standard
# lists 22 fraud scenarios; the other 15 have no rows in this dataset.
FRAUD_TYPE_OPTIONS = [
    "个体工商户年报补录收费",
    "引导贷款用户添加第三方微信",
    "手机租赁套路贷",
    "套路运",
    "网贷平台退息退费",
    "证券投资类",
    "贷款相关-ab贷",
]
RISK_LEVELS = ["合规", "低风险", "中风险", "高风险"]
LABELS = ["非涉诈", "涉诈"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_v2(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "data_id", "content", "label_v2", "is_violation", "is_fraud",
            "risk_level", "scene_category", "scene_subtype", "label_detail",
            "label_source", "needs_review", "proposed_risk_level",
            "proposed_scene_category", "proposed_subtype", "proposed_is_fraud",
            "old_label", "old_category", "人工复核", "备注",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        rows = []
        seen = set()
        for lineno, raw in enumerate(reader, 1):
            row = {k: (v or "").strip() for k, v in raw.items()}
            row["source_row"] = lineno
            row["cohort"] = row.get("cohort") or (
                "human_cq_normal" if row["label_source"] == "human_cq_normal"
                else "agent_1022"
                if (row["label_source"] == "human_1022"
                    or row["label_source"].startswith("cq_agent:"))
                else "human_0729"
            )
            row["source_file"] = row.get("source_file", "")
            if not row["data_id"] or row["data_id"] in seen:
                raise SystemExit(f"row {lineno}: empty/duplicate data_id")
            seen.add(row["data_id"])
            if row["label_v2"] not in LABELS:
                raise SystemExit(f"row {lineno}: bad label_v2 {row['label_v2']!r}")
            if row["is_violation"] not in ("0", "1") or row["is_fraud"] not in ("0", "1"):
                raise SystemExit(f"row {lineno}: bad is_violation/is_fraud")
            iv, iff = int(row["is_violation"]), int(row["is_fraud"])
            if iff > iv:
                raise SystemExit(f"row {lineno}: is_fraud=1 but is_violation=0")
            if row["risk_level"] not in RISK_LEVELS:
                raise SystemExit(f"row {lineno}: bad risk_level {row['risk_level']!r}")
            if row["label_v2"] != ("涉诈" if iff else "非涉诈"):
                raise SystemExit(f"row {lineno}: label_v2/is_fraud mismatch")
            if iv and not iff and row["scene_category"] not in SCENE_OPTIONS:
                raise SystemExit(f"row {lineno}: bad scene_category {row['scene_category']!r}")
            if iff and row["scene_category"] not in FRAUD_TYPE_OPTIONS:
                raise SystemExit(f"row {lineno}: bad fraud type {row['scene_category']!r}")
            if not iv and row["risk_level"] != "合规":
                raise SystemExit(f"row {lineno}: non-violation with risk {row['risk_level']!r}")
            if iff and row["risk_level"] != "高风险":
                raise SystemExit(f"row {lineno}: fraud row not high risk")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def load_splits(paths: list[Path], ids: set[str]) -> dict[str, str]:
    split_of: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{path}: split file not found")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for split, part in payload["ids"].items():
            if split not in ("cal", "dev", "gate"):
                raise SystemExit(f"{path}: unknown split {split!r}")
            for rid in part:
                if rid in split_of:
                    raise SystemExit(f"{path}: duplicate split id {rid}")
                split_of[rid] = split
    missing = ids - set(split_of)
    extra = set(split_of) - ids
    if missing or extra:
        raise SystemExit(
            "split id mismatch: "
            f"missing={len(missing)} extra={len(extra)} "
            f"files={[str(p) for p in paths]}"
        )
    return split_of


COHORTS = ("human_0729", "agent_1022", "human_cq_normal")


def render_readme(stats: dict, source_sha: str) -> str:
    by_split = stats["by_split"]
    human = stats["by_cohort"]["human_0729"]
    agent = stats["by_cohort"]["agent_1022"]
    normal = stats["by_cohort"].get("human_cq_normal", {
        "n": 0, "fraud": 0, "violation": 0, "compliant": 0, "review": 0,
        "risk": {}, "label": {}, "label_source": {},
    })
    clusters = stats.get("clusters", {})
    src = stats["label_source"]
    split_line = " / ".join(
        f"{s} {by_split[s]['n']}"
        f"（{' + '.join(f'{c} {by_split[s]['cohort'].get(c, 0)}' for c in COHORTS if by_split[s]['cohort'].get(c, 0))}；"
        f"涉诈 {by_split[s]['fraud']}）"
        for s in ("cal", "dev", "gate")
    )
    old_suspect = stats["old_label"].get("涉诈", 0)
    n_fraud = stats["is_fraud"][1]
    n_violation = stats["is_violation"][1]
    lines = [
        "# 新标准评估集（v2，含 10.22 人工复核与重庆正常数据）",
        "",
        f"- 源文件：`{stats['source']}`（sha256 `{source_sha[:16]}…`）",
        "- 生成链：`eval/merge_review_batch.py` → `scripts/backfill_spec_review.py` "
        "→ `eval/merge_human_review.py` → `eval/prepare_v2.py`",
        f"- 样本：{stats['n_rows']} 条 = human_0729 {human['n']} 条 + "
        f"agent_1022 {agent['n']} 条 + human_cq_normal {normal['n']} 条。",
        f"- 拆分：{split_line}",
        "- `human_0729` 的 1447 个 id 沿用 `eval/splits.json` 的冻结拆分；"
        "`agent_1022` 与 `human_cq_normal` 按近重复簇整簇分配（簇不跨 split），"
        "新正常行按 40/20/40 补入。",
        "",
        "## 三个 cohort（重要）",
        "",
        f"- `human_0729`：{human['n']} 条，2026-07-29 人工确认；"
        "主门禁与 v1/v2 配对比较只用这个 cohort。",
        f"- `agent_1022`：{agent['n']} 条，来自 `assets/10.22 复核结果.xlsx`，"
        "覆盖 2026-10-23..2026-11-04。其中 "
        f"{agent['label_source'].get('human_1022', 0)} 条采用 "
        "`assets/10.22agent和人工复核结果.xlsx` 的人工复核，"
        f"{agent['label_source'].get('spec_review', 0)} 条保留 242 条队列的 spec 复核裁定。"
        f"风险分布：{agent['risk']}；涉诈 {agent['fraud']} 条。",
        f"- `human_cq_normal`：{normal['n']} 条，来自 "
        "`assets/重庆行业卡人工复核正常数据.xlsx`，全部为人工复核正常/合规；"
        "用于补充正常类，尤其是 gate 的正常样本。",
        f"- 全量近重复簇 {clusters.get('n_clusters', 0)} 个"
        f"（多行簇 {clusters.get('multi_row_clusters', 0)} 个，"
        f"最大 {clusters.get('largest_cluster', 0)} 条）；新正常行与既有行"
        f"同簇 {clusters.get('bridged_clusters_skipped', 0)} 个，未跨 split。",
        f"- 主门禁口径：`split=gate` 且 `cohort=human_0729`（"
        f"{by_split['gate']['cohort'].get('human_0729', 0)} 条）；"
        f"`human_cq_normal` gate（{by_split['gate']['cohort'].get('human_cq_normal', 0)} 条）"
        "用于正常类 recall；`agent_1022` gate "
        f"（{by_split['gate']['cohort'].get('agent_1022', 0)} 条）"
        "只作外部人工复核对照。",
        "",
        "## 标签语义（重要）",
        "",
        "新标准把旧的粗标签 `涉诈` 拆成两层：",
        "",
        f"- `is_violation=1`：命中任一违规/涉诈场景，共 {n_violation} 条；"
        f"其中 4.2 涉诈 {n_fraud} 条，仅 4.1 违规 "
        f"{n_violation - n_fraud} 条；",
        f"- `is_fraud=1`：命中 4.2 涉诈场景（均为高风险），共 {n_fraud} 条"
        f"（human_0729 {human['fraud']} + agent_1022 {agent['fraud']} + "
        f"human_cq_normal {normal['fraud']}）；",
        f"- `非涉诈` = 合规 + 仅 4.1 违规，共 {stats['labels']['非涉诈']} 条。",
        "",
        "**二分类任务的目标是 `is_fraud`，不是旧的 `label`。** "
        f"旧人工 `label=涉诈` 的 {old_suspect} 条中，{human['fraud']} 条是 4.2 涉诈，"
        f"{old_suspect - human['fraud']} 条降为仅 4.1 违规（负类）。"
        f"全量涉诈占比 {n_fraud}/{stats['n_rows']} "
        f"= {n_fraud / stats['n_rows'] * 100:.1f}%。",
        "",
        "## 三套任务",
        "",
        f"- `goldens.jsonl`：`{QUESTION}`，选项 {OPTIONS}，全部 "
        f"{stats['n_rows']} 条。",
        f"- `goldens_scene.jsonl`：`{SCENE_QUESTION}`，选项 {SCENE_OPTIONS}，"
        f"仅非涉诈的 4.1 违规行（{stats['scene_goldens']} 条）。",
        f"- `goldens_fraud_type.jsonl`：`{FRAUD_TYPE_QUESTION}`，选项 "
        f"{FRAUD_TYPE_OPTIONS}，仅 `is_fraud=1` 行（{n_fraud} 条）。",
        "",
        "## 复核状态",
        "",
        f"- `needs_review=1`：{stats['needs_review'].get(1, 0)} 条"
        f"（human_0729 {human['review']} + agent_1022 {agent['review']} + "
        f"human_cq_normal {normal['review']}）。",
        f"- 242 条复核队列全部关闭，回填 `label_source=spec_review` "
        f"{src.get('spec_review', 0)} 条；其中 agent_1022 的 107 条"
        "保留 spec 裁定，未被人批复核覆盖。",
        "- 逐条结论与证据见 `runs/labels_v2/review_queue_复核结论.csv`；"
        "两种复核来源的 107 条重叠冲突见 "
        "`runs/labels_v2/spec_review/conflicts_human1022_vs_spec.csv`。",
        "",
        "## 已知限制",
        "",
        "- `agent_1022` 的 6,246 条是 10.22 人工复核，107 条是 spec 复核；"
        "两套口径在 107 条重叠上仍有 19 条风险、17 条场景、2 条涉诈分歧，"
        "当前按「专项 spec 裁定优先」保留。",
        f"- `agent_1022` 只有 {agent['fraud']} 条 4.2 涉诈，"
        "不能单独估计涉诈 recall；它主要补充低/中/高风险违规负类。",
        "- `agent_1022` 与 `human_cq_normal` 都是单批 snapshot；"
        "拆分按近重复簇分组而不是时间外推，簇内模板不跨 split，"
        "但同一业务的不同模板仍可能跨 split。",
        "- `human_0729` 的 4.1 风险等级/类型多为规则推断，只有 16 条有明确人工风险备注；"
        "新标准 22 类涉诈场景中有 15 类在本数据中无样本。",
        f"- `human_cq_normal` 的 gate 正常样本为 "
        f"{by_split['gate']['cohort'].get('human_cq_normal', 0)} 条，"
        "全部来自 2026-08 批次，不是时间外推；正常 recall 仍需在真实时间窗上复核。",
        "",
        "## 审计字段",
        "",
        "- `cohort`：`human_0729` / `agent_1022` / `human_cq_normal`；"
        "`source_file`：原始文件。",
        "- `label`：`非涉诈`/`涉诈`（由 `is_fraud` 派生）；"
        "`is_violation`/`is_fraud`：0/1。",
        "- `risk_level`：合规/低/中/高；`scene_category`/`scene_subtype`：新分类。",
        "- `label_source`：`old_category`/`manual_note`/`rule:*`/`spec_review`/"
        "`human_1022`/`human_cq_normal`。",
        "- `proposed_*`：历史建议字段，复核完成后已清空；原始建议见 "
        "`review_queue_combined.csv`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="assets/复核数据_v2.csv")
    ap.add_argument("--splits", default="eval/splits.json")
    ap.add_argument("--extra-splits", default="runs/labels_v2/extra_splits.json")
    ap.add_argument("--out-dir", default="eval_v2")
    args = ap.parse_args()

    source = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_v2(source)
    split_paths = [Path(args.splits)]
    extra_splits = Path(args.extra_splits)
    extra_payload = {}
    if extra_splits.exists():
        split_paths.append(extra_splits)
        extra_payload = json.loads(extra_splits.read_text(encoding="utf-8"))
    split_of = load_splits(split_paths, {r["data_id"] for r in rows})

    goldens, scene_goldens, fraud_goldens = [], [], []
    for row in rows:
        iv, iff = int(row["is_violation"]), int(row["is_fraud"])
        label = "涉诈" if iff else "非涉诈"
        split = split_of[row["data_id"]]
        record = {
            "id": row["data_id"],
            "source": str(source),
            "source_row": row["source_row"],
            "state": row["content"],
            "question": QUESTION,
            "options": OPTIONS,
            "label": label,
            "is_violation": iv,
            "is_fraud": iff,
            "risk_level": row["risk_level"],
            "scene_category": row["scene_category"] if iv else None,
            "scene_subtype": row["scene_subtype"] or None,
            "label_source": row["label_source"],
            "needs_review": int(row["needs_review"]),
            "proposed_risk_level": row["proposed_risk_level"] or None,
            "proposed_scene_category": row["proposed_scene_category"] or None,
            "proposed_subtype": row["proposed_subtype"] or None,
            "proposed_is_fraud": int(row["proposed_is_fraud"]) if row["proposed_is_fraud"] else 0,
            "old_label": row["old_label"],
            "old_category": row["old_category"] or None,
            "risk_note": row["人工复核"] or None,
            "review_note": row["备注"] or None,
            "cohort": row["cohort"],
            "source_file": row["source_file"] or None,
            "split": split,
        }
        goldens.append(record)
        if iv and not iff:
            scene_goldens.append({
                "id": row["data_id"],
                "source": str(source),
                "source_row": row["source_row"],
                "state": row["content"],
                "question": SCENE_QUESTION,
                "options": SCENE_OPTIONS,
                "label": row["scene_category"],
                "is_fraud": iff,
                "risk_level": row["risk_level"],
                "cohort": row["cohort"],
                "split": split,
            })
        if iff:
            if row["scene_category"] not in FRAUD_TYPE_OPTIONS:
                raise SystemExit(
                    f"{row['data_id']}: fraud type {row['scene_category']!r} "
                    "not in FRAUD_TYPE_OPTIONS"
                )
            fraud_goldens.append({
                "id": row["data_id"],
                "source": str(source),
                "source_row": row["source_row"],
                "state": row["content"],
                "question": FRAUD_TYPE_QUESTION,
                "options": FRAUD_TYPE_OPTIONS,
                "label": row["scene_category"],
                "cohort": row["cohort"],
                "split": split,
            })

    # Audit stats.
    by_split = {}
    for s in ("cal", "dev", "gate"):
        sub = [r for r in goldens if r["split"] == s]
        by_split[s] = {
            "n": len(sub),
            "fraud": sum(r["is_fraud"] for r in sub),
            "violation": sum(r["is_violation"] for r in sub),
            "review": sum(r["needs_review"] for r in sub),
            "risk": dict(collections.Counter(r["risk_level"] for r in sub)),
            "label": dict(collections.Counter(r["label"] for r in sub)),
            "cohort": dict(collections.Counter(r["cohort"] for r in sub)),
        }

    by_cohort = {}
    for c in ("human_0729", "agent_1022", "human_cq_normal"):
        sub = [r for r in goldens if r["cohort"] == c]
        if not sub:
            continue
        by_cohort[c] = {
            "n": len(sub),
            "fraud": sum(r["is_fraud"] for r in sub),
            "violation": sum(r["is_violation"] for r in sub),
            "compliant": sum(1 for r in sub if not r["is_violation"]),
            "review": sum(r["needs_review"] for r in sub),
            "risk": dict(collections.Counter(r["risk_level"] for r in sub)),
            "label": dict(collections.Counter(r["label"] for r in sub)),
            "label_source": dict(collections.Counter(r["label_source"] for r in sub)),
        }

    normal_audit = [r for r in goldens if r["label_source"] == "normal_audit"]
    stats = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "n_rows": len(goldens),
        "labels": dict(collections.Counter(r["label"] for r in goldens)),
        "is_violation": dict(collections.Counter(r["is_violation"] for r in goldens)),
        "is_fraud": dict(collections.Counter(r["is_fraud"] for r in goldens)),
        "risk_level": dict(collections.Counter(r["risk_level"] for r in goldens)),
        "scene_category": dict(collections.Counter(
            r["scene_category"] for r in goldens if r["is_violation"]
        )),
        "old_label": dict(collections.Counter(r["old_label"] for r in goldens)),
        "label_source": dict(collections.Counter(r["label_source"] for r in goldens)),
        "needs_review": dict(collections.Counter(r["needs_review"] for r in goldens)),
        "cohort": dict(collections.Counter(r["cohort"] for r in goldens)),
        "by_cohort": by_cohort,
        "normal_audit": len(normal_audit),
        "normal_audit_fraud": sum(r["proposed_is_fraud"] for r in normal_audit),
        "normal_audit_high": sum(
            r["proposed_risk_level"] == "高风险" for r in normal_audit
        ),
        "normal_audit_low": sum(
            r["proposed_risk_level"] == "低风险" for r in normal_audit
        ),
        "old_to_new": dict(collections.Counter(
            f"{r['old_label']}->{'fraud' if r['is_fraud'] else ('violation' if r['is_violation'] else 'compliant')}"
            for r in goldens if r["old_label"]
        )),
        "scene_goldens": len(scene_goldens),
        "fraud_type_goldens": len(fraud_goldens),
        "by_split": by_split,
        "clusters": extra_payload.get("cluster_stats", {}),
        "split_sources": [str(p) for p in split_paths],
    }

    ids_by_split = {s: [] for s in ("cal", "dev", "gate")}
    for rid, split in split_of.items():
        ids_by_split[split].append(rid)
    for split in ids_by_split:
        ids_by_split[split].sort()

    jsonl_write(out_dir / "goldens.jsonl", goldens)
    jsonl_write(out_dir / "goldens_scene.jsonl", scene_goldens)
    jsonl_write(out_dir / "goldens_fraud_type.jsonl", fraud_goldens)
    (out_dir / "splits.json").write_text(json.dumps({
        "version": "m0.5-v2",
        "reused_from": str(args.splits),
        "extra_splits": str(extra_splits) if extra_splits.exists() else None,
        "ids": ids_by_split,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        render_readme(stats, stats["source_sha256"]), encoding="utf-8"
    )

    print(f"wrote {out_dir}/goldens.jsonl ({len(goldens)} rows)")
    print(f"wrote {out_dir}/goldens_scene.jsonl ({len(scene_goldens)} rows)")
    print(f"wrote {out_dir}/goldens_fraud_type.jsonl ({len(fraud_goldens)} rows)")
    print(f"labels: {stats['labels']} | is_fraud: {stats['is_fraud']} | "
          f"risk: {stats['risk_level']}")
    print(f"cohorts: {stats['cohort']}")
    for s in ("cal", "dev", "gate"):
        x = by_split[s]
        print(f"  {s}: n={x['n']} fraud={x['fraud']} violation={x['violation']} "
              f"review={x['review']} cohort={x['cohort']}")
    print(f"normal_audit: {stats['normal_audit']} "
          f"(fraud {stats['normal_audit_fraud']}, high {stats['normal_audit_high']}, "
          f"low {stats['normal_audit_low']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
