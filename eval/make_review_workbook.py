#!/usr/bin/env python
"""Render the 242-row v2 review queue as a human-review Excel workbook.

The CSV queue (`runs/labels_v2/review_queue_combined.csv`) is the machine
artifact; this script adds a reviewer-friendly view with filters, frozen
headers, risk colouring and dropdowns for the manual decision columns.

Blank manual fields mean "keep the current label".  Only rows marked
`修正` need the corrected risk/scene/fraud values filled in.

Usage:
  python eval/make_review_workbook.py
  python eval/make_review_workbook.py --queue runs/labels_v2/review_queue_combined.csv \
      --out runs/labels_v2/review_queue_combined.xlsx
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

RISK_ORDER = {"高风险": 0, "中风险": 1, "低风险": 2, "合规": 3, "": 4}
COHORT_ORDER = {"human_0729": 0, "agent_1022": 1}
RISK_FILL = {
    "高风险": PatternFill("solid", fgColor="F8CBAD"),
    "中风险": PatternFill("solid", fgColor="FFE699"),
    "低风险": PatternFill("solid", fgColor="FFF2CC"),
    "合规": PatternFill("solid", fgColor="E2EFDA"),
}
HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(color="FFFFFF", bold=True)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP_TOP = Alignment(wrap_text=True, vertical="top")

COLUMNS = [
    ("序号", 6),
    ("cohort", 11),
    ("data_id", 22),
    ("复核类型", 16),
    ("当前风险", 9),
    ("当前大类", 18),
    ("当前子类", 18),
    ("当前涉诈", 8),
    ("建议风险", 9),
    ("建议大类", 18),
    ("建议子类", 18),
    ("建议涉诈", 8),
    ("复核原因", 55),
    ("通话内容", 90),
    ("标签来源", 26),
    ("原人工复核", 20),
    ("原备注", 24),
    ("人工风险", 10),
    ("人工大类/类型", 22),
    ("人工子类", 20),
    ("人工涉诈", 9),
    ("复核结论", 10),
    ("复核备注", 30),
    ("复核人", 10),
    ("复核日期", 12),
]
MANUAL_COLS = {"人工风险", "人工大类/类型", "人工子类", "人工涉诈",
               "复核结论", "复核备注", "复核人", "复核日期"}


def safe(value: str | None) -> str:
    """Keep Excel from interpreting transcript text as a formula."""
    text = "" if value is None else str(value)
    if text and text[0] in "=+-@":
        text = "'" + text
    return text


def review_type(row: dict) -> str:
    if "近重复簇" in (row.get("备注") or ""):
        return "近重复簇少数派"
    if row["cohort"] == "agent_1022" and row["risk_level"] in ("中风险", "高风险"):
        return "agent中高风险"
    src = row.get("label_source") or ""
    if src == "normal_audit":
        return "旧标签正常审计"
    if src.startswith("rule:"):
        return "规则命中"
    if src == "default_low":
        return "旧标签低风险默认"
    if "heuristic" in src:
        return "heuristic"
    return "其他"


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            RISK_ORDER.get(r["risk_level"], 9),
            COHORT_ORDER.get(r["cohort"], 9),
            r["data_id"],
        ),
    )


def write_table(ws, rows: list[dict], with_validation: bool = True) -> None:
    ws.append([c for c, _ in COLUMNS])
    for idx, (name, width) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center",
                                   horizontal="center")
        cell.border = BORDER
        ws.column_dimensions[get_column_letter(idx)].width = width
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{len(rows) + 1}"

    for n, row in enumerate(rows, start=1):
        values = [
            n,
            row["cohort"],
            row["data_id"],
            review_type(row),
            row["risk_level"],
            row["scene_category"],
            row["scene_subtype"],
            row["is_fraud"],
            row.get("proposed_risk_level") or "",
            row.get("proposed_scene_category") or "",
            row.get("proposed_subtype") or "",
            row.get("proposed_is_fraud") or "",
            safe(row.get("label_detail")),
            safe(row.get("content")),
            row.get("label_source") or "",
            safe(row.get("人工复核")),
            safe(row.get("备注")),
            "", "", "", "", "", "", "", "",
        ]
        ws.append(values)
        line = ws.max_row
        ws.row_dimensions[line].height = 62
        for col, cell in enumerate(ws[line], start=1):
            cell.alignment = WRAP_TOP
            cell.border = BORDER
            name = COLUMNS[col - 1][0]
            if name in MANUAL_COLS:
                cell.fill = PatternFill("solid", fgColor="DDEBF7")
        risk_cell = ws.cell(row=line, column=5)
        if risk_cell.value in RISK_FILL:
            risk_cell.fill = RISK_FILL[risk_cell.value]

    if not with_validation or not rows:
        return
    last = len(rows) + 1
    risk_dv = DataValidation(
        type="list", formula1='"合规,低风险,中风险,高风险"', allow_blank=True,
        error="请选择：合规 / 低风险 / 中风险 / 高风险", errorTitle="风险等级无效",
    )
    fraud_dv = DataValidation(
        type="list", formula1='"0,1"', allow_blank=True,
        error="0=非涉诈，1=涉诈", errorTitle="涉诈标记无效",
    )
    verdict_dv = DataValidation(
        type="list", formula1='"维持,修正,存疑"', allow_blank=True,
        error="请选择：维持 / 修正 / 存疑", errorTitle="复核结论无效",
    )
    ws.add_data_validation(risk_dv)
    ws.add_data_validation(fraud_dv)
    ws.add_data_validation(verdict_dv)
    risk_col = get_column_letter([c for c, _ in COLUMNS].index("人工风险") + 1)
    fraud_col = get_column_letter([c for c, _ in COLUMNS].index("人工涉诈") + 1)
    verdict_col = get_column_letter([c for c, _ in COLUMNS].index("复核结论") + 1)
    risk_dv.add(f"{risk_col}2:{risk_col}{last}")
    fraud_dv.add(f"{fraud_col}2:{fraud_col}{last}")
    verdict_dv.add(f"{verdict_col}2:{verdict_col}{last}")


def write_summary(ws, rows: list[dict]) -> None:
    ws.title = "汇总"
    ws.column_dimensions["A"].width = 22
    for col in "BCDEF":
        ws.column_dimensions[col].width = 12

    ws["A1"] = "复核队列汇总（242 条）"
    ws["A1"].font = Font(bold=True, size=14)
    ws.merge_cells("A1:F1")

    ws["A3"] = "按 cohort × 风险等级"
    ws["A3"].font = Font(bold=True)
    risks = ["合规", "低风险", "中风险", "高风险", "合计"]
    ws.append([])
    ws.append(["cohort"] + risks)
    header_row = ws.max_row
    for cell in ws[header_row]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.border = BORDER
        cell.alignment = Alignment(horizontal="center")
    cohorts = ["human_0729", "agent_1022"]
    for cohort in cohorts:
        sub = [r for r in rows if r["cohort"] == cohort]
        counts = collections.Counter(r["risk_level"] for r in sub)
        ws.append([cohort] + [counts.get(k, 0) for k in risks[:-1]] + [len(sub)])
    total = collections.Counter(r["risk_level"] for r in rows)
    ws.append(["合计"] + [total.get(k, 0) for k in risks[:-1]] + [len(rows)])
    for row in ws.iter_rows(min_row=header_row + 1, max_row=ws.max_row):
        for cell in row:
            cell.border = BORDER
            cell.alignment = Alignment(horizontal="center")

    start = ws.max_row + 2
    ws.cell(row=start, column=1, value="按复核类型").font = Font(bold=True)
    ws.cell(row=start + 1, column=1, value="复核类型")
    ws.cell(row=start + 1, column=2, value="条数")
    for cell in ws[start + 1][:2]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.border = BORDER
    counts = collections.Counter(review_type(r) for r in rows)
    line = start + 2
    for name, n in counts.most_common():
        ws.cell(row=line, column=1, value=name)
        ws.cell(row=line, column=2, value=n)
        for col in (1, 2):
            ws.cell(row=line, column=col).border = BORDER
        line += 1

    start = line + 1
    ws.cell(row=start, column=1, value="按标签来源").font = Font(bold=True)
    ws.cell(row=start + 1, column=1, value="label_source")
    ws.cell(row=start + 1, column=2, value="条数")
    for cell in ws[start + 1][:2]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.border = BORDER
    counts = collections.Counter(r.get("label_source") or "" for r in rows)
    line = start + 2
    for name, n in counts.most_common():
        ws.cell(row=line, column=1, value=name)
        ws.cell(row=line, column=2, value=n)
        for col in (1, 2):
            ws.cell(row=line, column=col).border = BORDER
        line += 1


def write_notes(ws) -> None:
    ws.title = "说明"
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 100
    lines = [
        ("文件用途", "v2 标签合并数据集的 242 条待人工复核队列，供逐条复核使用。"),
        ("机器队列", "runs/labels_v2/review_queue_combined.csv（242 行，字段与主数据集一致）"),
        ("主数据集", "assets/复核数据_v2.csv（7800 行）"),
        ("队列构成", "human_0729 135 条 + agent_1022 107 条；"
                     "高风险 45 条 + 中风险 54 条 + 低风险 83 条 + 合规 60 条。"),
        ("复核类型", "近重复簇少数派 46、agent中高风险 56、规则命中 43、"
                     "旧标签低风险默认 77、旧标签正常审计 15、heuristic 5。"),
        ("填写方式", "人工风险 / 人工大类/类型 / 人工子类 / 人工涉诈：留空表示维持当前值，"
                     "仅在需要修正时填写。"),
        ("复核结论", "逐行选择：维持 / 修正 / 存疑。选“修正”时必须填写对应的人工字段。"),
        ("口径提醒", "低风险仍是违规（is_violation=1），只是非涉诈；"
                     "“正常合规”应看 合规 / is_violation=0。"),
        ("来源提醒", "agent_1022 是 cq_agent_* 机器标签（silver），"
                     "human_0729 是人工金标准；复核时请优先保证 human 队列。"),
        ("建议顺序", "先看“高风险45”和“中风险54”，再看 human_0729_135，最后 agent_1022_107。"),
        ("回填", "复核完成后把本工作簿交给数据负责人，合并回 assets/复核数据_v2.csv 并重跑 "
                 "eval/prepare_v2.py 更新 goldens。"),
    ]
    for i, (key, value) in enumerate(lines, start=1):
        ws.cell(row=i, column=1, value=key).font = Font(bold=True)
        ws.cell(row=i, column=2, value=value).alignment = WRAP_TOP
        ws.row_dimensions[i].height = 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", default="runs/labels_v2/review_queue_combined.csv")
    ap.add_argument("--out", default="runs/labels_v2/review_queue_combined.xlsx")
    args = ap.parse_args()

    with Path(args.queue).open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 242:
        print(f"warning: expected 242 rows, got {len(rows)}")

    wb = Workbook()
    write_notes(wb.active)
    write_summary(wb.create_sheet(), rows)

    ordered = sort_rows(rows)
    write_table(wb.create_sheet("全部242"), ordered)
    write_table(wb.create_sheet("高风险45"),
                [r for r in ordered if r["risk_level"] == "高风险"])
    write_table(wb.create_sheet("中风险54"),
                [r for r in ordered if r["risk_level"] == "中风险"])
    write_table(wb.create_sheet("human_0729_135"),
                [r for r in ordered if r["cohort"] == "human_0729"])
    write_table(wb.create_sheet("agent_1022_107"),
                [r for r in ordered if r["cohort"] == "agent_1022"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"wrote {out} ({len(rows)} review rows, "
          f"{len(wb.sheetnames)} sheets: {', '.join(wb.sheetnames)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
