#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 242 条 spec 复核终判回填到 labels_v2 主表 `assets/复核数据_v2.csv`。

来源：`runs/labels_v2/review_queue_复核结论.csv`（复核_is_fraud / 复核_场景标签 /
复核_风险等级 / 复核_判定说明 / 复核_结论来源）。

覆盖规则（仅动 242 个 data_id，其余 7558 行原样保留）：
  label_v2       = 涉诈 / 非涉诈（由 复核_is_fraud 推导）
  is_fraud       = 复核_is_fraud
  is_violation   = 合规且非涉诈 → 0，否则 1
  risk_level     = 复核_风险等级
  scene_category = 复核_场景标签
  scene_subtype  = 涉诈三类给标准子类；场景改为正常→"正常"；大类变化→清空待补
  label_detail   = 复核_判定说明
  label_source   = spec_review
  needs_review   = 0（已复核完毕）
原 default_low / rule:* / cq_agent:* 等兜底标签被覆盖；回溯可查
`review_queue_combined.csv` 与备份 `.bak.csv`。

用法：
    python scripts/backfill_spec_review.py            # 试运行（只报告，不写文件）
    python scripts/backfill_spec_review.py --write    # 实际回填（先自动备份）
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MAIN_CSV = Path("assets/复核数据_v2.csv")
CONCLUSION_CSV = Path("runs/labels_v2/review_queue_复核结论.csv")
STATS_JSON = Path("runs/labels_v2/spec_review/backfill_summary.json")

# 涉诈场景大类 → 标准 scene_subtype
FRAUD_SUBTYPE = {
    "贷款相关-ab贷": "ab贷",
    "证券投资类": "证券投资荐股",
    "引导贷款用户添加第三方微信": "引导添加第三方微信",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="实际写回主表（默认试运行）")
    args = ap.parse_args()

    with CONCLUSION_CSV.open(encoding="utf-8-sig", newline="") as fh:
        conclusions = {r["data_id"]: r for r in csv.DictReader(fh)}
    if len(conclusions) != 242:
        raise SystemExit(f"结论文件行数异常: {len(conclusions)}")

    with MAIN_CSV.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    before_src = collections.Counter(r["label_source"] for r in rows)
    changed: list[dict] = []
    for r in rows:
        c = conclusions.get(r["data_id"])
        if c is None:
            continue
        fraud = int(c["复核_is_fraud"])
        risk = c["复核_风险等级"]
        scene = c["复核_场景标签"]
        old_scene = r["scene_category"]
        if fraud:
            subtype = FRAUD_SUBTYPE.get(scene, "")
        elif scene == "正常":
            subtype = "正常"
        elif scene != old_scene:
            subtype = ""  # 大类改判且非涉诈：旧子类不再适用，清空待补
        else:
            subtype = r["scene_subtype"]
        new = {
            "label_v2": "涉诈" if fraud else "非涉诈",
            "is_fraud": str(fraud),
            "is_violation": "0" if (risk == "合规" and not fraud) else "1",
            "risk_level": risk,
            "scene_category": scene,
            "scene_subtype": subtype,
            "label_detail": c["复核_判定说明"],
            "label_source": "spec_review",
            "needs_review": "0",
        }
        diff = {k: (old, new[k]) for k, old in r.items() if k in new and old != new[k]}
        r.update(new)
        changed.append({"data_id": r["data_id"], "review_origin": c["复核_结论来源"],
                        "fields_changed": sorted(diff)})

    matched = len(changed)
    after_src = collections.Counter(r["label_source"] for r in rows)
    stats = {
        "main_csv": str(MAIN_CSV),
        "conclusion_csv": str(CONCLUSION_CSV),
        "n_main_rows": len(rows),
        "n_backfilled": matched,
        "not_matched": 242 - matched,
        "label_source_before": dict(before_src),
        "label_source_after": dict(after_src),
        "is_fraud_after": dict(collections.Counter(r["is_fraud"] for r in rows)),
        "risk_level_after": dict(collections.Counter(r["risk_level"] for r in rows)),
        "needs_review_after": dict(collections.Counter(r["needs_review"] for r in rows)),
        "by_review_origin": dict(collections.Counter(c["review_origin"] for c in changed)),
    }
    print(json.dumps({k: v for k, v in stats.items() if k != "rows"}, ensure_ascii=False, indent=2))

    if not args.write:
        print("[dry-run] 未写文件；加 --write 执行回填")
        return 0
    if matched != 242:
        raise SystemExit(f"仅匹配到 {matched}/242，中止写入")

    backup = MAIN_CSV.with_suffix(".pre_specreview.bak.csv")
    if not backup.exists():
        shutil.copy2(MAIN_CSV, backup)
        print(f"backup -> {backup}")
    with MAIN_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(f"wrote {MAIN_CSV}（{len(rows)} 行，回填 {matched} 条）；stats -> {STATS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
