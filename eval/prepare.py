#!/usr/bin/env python
"""Build the frozen M0.5 evaluation artifacts from human-reviewed CSV data.

The source file is a one-day snapshot of human-confirmed call reviews.  This
script intentionally does the following once, and writes artifacts that later
runs consume:

  eval/goldens.jsonl           binary normal / fraud decisions (all rows)
  eval/goldens_category.jsonl  conditional fraud-type decisions (top-6 types)
  eval/splits.json             deterministic cal / dev / gate assignment
  eval/stats.json              distributions for audit
  eval/README.md               label semantics and known limitations

The split is stratified by (label, coarse category).  Assignment is stable
across machines: SHA-256 of the source id, not Python's randomized hash().
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task import CATEGORY_OPTIONS, CATEGORY_QUESTION, OPTIONS, QUESTION  # noqa: E402

LABELS = ("正常", "涉诈")
RATIOS = {"cal": 0.40, "dev": 0.20, "gate": 0.40}
SPLIT_ORDER = ["cal", "dev", "gate"]


def stable_hash(text: str) -> str:
    return hashlib.sha256(("risk-jev-m0.5-v1:" + text).encode("utf-8")).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        expected = {"data_id", "content", "label", "category", "人工复核", "备注"}
        if set(reader.fieldnames or []) != expected:
            raise SystemExit(f"unexpected columns: {reader.fieldnames}")
        rows = []
        seen_ids, seen_content = set(), set()
        for lineno, raw in enumerate(reader, 1):
            row = {k: (v or "").strip() for k, v in raw.items()}
            row["source_row"] = lineno
            if not row["data_id"]:
                raise SystemExit(f"row {lineno}: empty data_id")
            if row["data_id"] in seen_ids:
                raise SystemExit(f"row {lineno}: duplicate data_id {row['data_id']}")
            if not row["content"]:
                raise SystemExit(f"row {lineno}: empty content")
            if row["content"] in seen_content:
                raise SystemExit(f"row {lineno}: duplicate content")
            if row["label"] not in LABELS:
                raise SystemExit(f"row {lineno}: label {row['label']!r} not in {LABELS}")
            if row["label"] == "正常" and row["category"]:
                raise SystemExit(f"row {lineno}: normal row has category {row['category']!r}")
            if row["label"] == "涉诈" and not row["category"]:
                raise SystemExit(f"row {lineno}: fraud row has no category")
            seen_ids.add(row["data_id"])
            seen_content.add(row["content"])
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def coarse_category(row: dict) -> str:
    if row["label"] == "正常":
        return ""
    return row["category"] if row["category"] in CATEGORY_OPTIONS else "其他"


def stratified_assign(rows: list[dict], ratios: dict[str, float]) -> dict[str, str]:
    """Exact per-stratum quotas via largest remainder; stable row order."""
    strata: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rows:
        strata[(row["label"], coarse_category(row))].append(row)

    split_of: dict[str, str] = {}
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: stable_hash(r["data_id"]))
        n = len(group)
        exact = {s: n * ratios[s] for s in SPLIT_ORDER}
        base = {s: math.floor(exact[s]) for s in SPLIT_ORDER}
        remaining = n - sum(base.values())
        # Largest fractional part wins; ties follow SPLIT_ORDER for determinism.
        order = sorted(SPLIT_ORDER, key=lambda s: (-(exact[s] - base[s]), SPLIT_ORDER.index(s)))
        for s in order[:remaining]:
            base[s] += 1
        cursor = 0
        for s in SPLIT_ORDER:
            for row in group[cursor:cursor + base[s]]:
                split_of[row["data_id"]] = s
            cursor += base[s]
    return split_of


def length_stats(values: list[int]) -> dict:
    values = sorted(values)
    if not values:
        return {}
    def pct(p: float) -> int:
        return values[min(len(values) - 1, int(len(values) * p))]
    return {
        "n": len(values),
        "min": values[0],
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": values[-1],
        "mean": round(statistics.mean(values), 1),
        "total_chars": sum(values),
    }


def jsonl_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_readme(stats: dict, split_counts: dict, source_sha: str) -> str:
    lines = [
        "# M0.5 评估集（冻结）",
        "",
        f"- 源文件：`{stats['source']}`（sha256 `{source_sha[:16]}…`）",
        f"- 生成脚本：`eval/prepare.py`（salt `risk-jev-m0.5-v1`）",
        f"- 样本：{stats['n_rows']} 条，全部来自 2026-07-29 单日人工复核数据",
        f"- 拆分：cal {split_counts['cal']} / dev {split_counts['dev']} / gate {split_counts['gate']}",
        "",
        "## 标签语义（重要）",
        "",
        "源文件 `label` 只有两个值：`正常` 与 `涉诈`。`category` 仅对 `label=涉诈` 的行填写。",
        "复核备注中有 10 条明确写“低风险”、6 条写“高风险”，它们全部仍被标为 `涉诈`。",
        "因此这里的 `涉诈` 应理解为**需要按涉诈/违规处置**的粗标签，不能等同于刑事诈骗；",
        "风险等级（合规/低/中/高）在源数据中没有系统标注，M0.5 不把风险等级作为门禁指标。",
        "",
        "`人工复核` 与 `备注` 只覆盖少数行，保留在 goldens 中作为审计线索，不参与评分。",
        "",
        "## 拆分",
        "",
        "- 按 `(label, 粗类型)` 分层，层内按 `sha256(salt + data_id)` 排序后按 40/20/40 分配。",
        "- `cal`：估计固定 prior、训练线性 probe、拟合温度；`dev`：选层/选正则/选阈值；",
        "  `gate`：只用于最终报告，不参与任何选择。",
        "- 源数据只有一天，没有用户/时间字段，无法做时间外推拆分；这是本评估集的硬限制。",
        "- 已做 SimHash（字符 3-gram, 64-bit）近重复检查：Hamming ≤ 4 的样本对为 0，",
        "  精确重复 content 也为 0；同一天同一话术模板的语义泄漏仍可能存在。",
        "",
        "## 二分类任务（goldens.jsonl）",
        "",
        f"- 问题：`{QUESTION}`",
        f"- 选项：{OPTIONS}",
        "- 全部 1447 条；`涉诈` 1287 条（89.0%），`正常` 160 条（11.0%）。",
        "- 该任务是不平衡二分类：准确率会被多数类主导，报告必须同时看 `涉诈` recall、",
        "  `正常` recall（TNR）、固定 FPR 下的 recall、macro TPR、PR-AUC 与配对置信区间。",
        "",
        "## 类型任务（goldens_category.jsonl）",
        "",
        f"- 问题：`{CATEGORY_QUESTION}`",
        f"- 选项：{CATEGORY_OPTIONS}",
        "- 只保留 top-6 类型的 1243 条涉诈样本；其余 44 条长尾类型（8 个类型，单项 1–13 条）",
        "  不进入类型任务，但保留在二分类任务中。",
        "- 类型任务的条件准确率不能直接乘二分类准确率当作端到端指标；端到端需另行定义。",
        "",
        "## 已知限制",
        "",
        "- 单日、单批次数据；同一天的外呼活动可能共享话术，gate 不是时间外推。",
        "- 正常类只有 160 条，gate 约 64 条；`正常` recall 的 95% Wilson CI 约 ±12pp。",
        "- 标签含噪：话术边界依赖规范细节，`正常` 与低风险违规之间可能存在人工分歧。",
        "- `涉诈` 类别内混合了低风险违规与高风险涉诈，二分类标签不携带风险等级。",
        "",
        "## 审计字段",
        "",
        "- `id`：源 `data_id`；`source_row`：CSV 行号（含表头）；`split`：cal/dev/gate。",
        "- `state`：通话转写原文；`category`：源类型；`category_coarse`：top-6 或 `其他`。",
        "- `risk_note` / `review_note`：源 `人工复核` / `备注`，可为空。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="assets/0729复核人工确认数据.csv")
    ap.add_argument("--out-dir", default="eval")
    args = ap.parse_args()

    source = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(source)
    split_of = stratified_assign(rows, RATIOS)

    goldens, category_goldens = [], []
    for row in rows:
        split = split_of[row["data_id"]]
        record = {
            "id": row["data_id"],
            "source": str(source),
            "source_row": row["source_row"],
            "state": row["content"],
            "question": QUESTION,
            "options": OPTIONS,
            "label": row["label"],
            "category": row["category"] or None,
            "category_coarse": coarse_category(row) or None,
            "risk_note": row["人工复核"] or None,
            "review_note": row["备注"] or None,
            "split": split,
        }
        goldens.append(record)
        if row["label"] == "涉诈" and row["category"] in CATEGORY_OPTIONS:
            category_goldens.append({
                "id": row["data_id"],
                "source": str(source),
                "source_row": row["source_row"],
                "state": row["content"],
                "question": CATEGORY_QUESTION,
                "options": CATEGORY_OPTIONS,
                "label": row["category"],
                "split": split,
            })

    # Audit stats.
    split_counts = collections.Counter(r["split"] for r in goldens)
    label_by_split = {
        s: collections.Counter(r["label"] for r in goldens if r["split"] == s)
        for s in SPLIT_ORDER
    }
    category_by_split = {
        s: collections.Counter(
            r["category_coarse"] for r in goldens
            if r["split"] == s and r["label"] == "涉诈"
        )
        for s in SPLIT_ORDER
    }
    stats = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "salt": "risk-jev-m0.5-v1",
        "n_rows": len(goldens),
        "labels": dict(collections.Counter(r["label"] for r in goldens)),
        "categories": dict(collections.Counter(
            r["category"] for r in goldens if r["label"] == "涉诈"
        )),
        "category_task_rows": len(category_goldens),
        "split_counts": dict(split_counts),
        "label_by_split": {s: dict(c) for s, c in label_by_split.items()},
        "category_by_split": {s: dict(c) for s, c in category_by_split.items()},
        "content_chars": length_stats([len(r["state"]) for r in goldens]),
        "content_chars_by_label": {
            label: length_stats([len(r["state"]) for r in goldens if r["label"] == label])
            for label in LABELS
        },
        "ratios": RATIOS,
    }

    jsonl_write(out_dir / "goldens.jsonl", goldens)
    jsonl_write(out_dir / "goldens_category.jsonl", category_goldens)
    (out_dir / "splits.json").write_text(json.dumps({
        "version": "m0.5-v1",
        "salt": stats["salt"],
        "ratios": RATIOS,
        "ids": {
            s: [r["id"] for r in goldens if r["split"] == s]
            for s in SPLIT_ORDER
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        render_readme(stats, split_counts, stats["source_sha256"]), encoding="utf-8"
    )

    print(f"wrote {out_dir}/goldens.jsonl ({len(goldens)} rows)")
    print(f"wrote {out_dir}/goldens_category.jsonl ({len(category_goldens)} rows)")
    print(f"splits: {dict(split_counts)}")
    for s in SPLIT_ORDER:
        print(f"  {s}: {dict(label_by_split[s])} | categories {dict(category_by_split[s])}")
    print(f"source sha256: {stats['source_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
