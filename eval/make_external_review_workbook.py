#!/usr/bin/env python
"""Render the external Chongqing review queues as an Excel workbook.

Queues:
  * ``agent涉诈``: every external row whose CQ agent says ``涉诈-...``.
  * ``正常冲突``: old label 正常 but the agent says 违规/涉诈.
  * ``旧涉诈_agent正常``: old label 涉诈 but the agent says 正常 (sample).

The agent label is a *suggestion*, not gold: on the 4,670-row v2 overlap the
agent's 4.2 precision is 0.245 (引导加微 0.017, ab贷 0.75).  Reviewers accept or
correct each row; only the manual columns are written back.

Usage:
  python eval/make_external_review_workbook.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.external_labels import parse_agent_result, suggested_v2  # noqa: E402

HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(color="FFFFFF", bold=True)
MANUAL_FILL = PatternFill("solid", fgColor="DDEBF7")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP_TOP = Alignment(wrap_text=True, vertical="top")
RISK_FILL = {
    "高风险": PatternFill("solid", fgColor="F8CBAD"),
    "中风险": PatternFill("solid", fgColor="FFE699"),
    "低风险": PatternFill("solid", fgColor="FFF2CC"),
    "合规": PatternFill("solid", fgColor="E2EFDA"),
}
PRIORITY_ORDER = {
    "P0_ab贷": 0, "P1_高置信": 1, "P2_其他": 2, "P3_引导加微": 3,
    "P3_4.1硬负例": 4, "抽样100": 5, "备用": 6,
}

COLUMNS = [
    ("序号", 6),
    ("队列", 14),
    ("优先级", 14),
    ("来源文件", 38),
    ("callid", 22),
    ("source_date", 12),
    ("旧标签", 8),
    ("agent结果", 42),
    ("agent判定", 9),
    ("agent风险", 9),
    ("agent场景", 24),
    ("agent子类", 20),
    ("agent置信度", 10),
    ("probe_P涉诈", 10),
    ("probe@FPR5", 10),
    ("建议_is_fraud", 11),
    ("建议_is_violation", 12),
    ("建议_risk", 10),
    ("建议_scene", 24),
    ("建议_subtype", 20),
    ("场景冲突", 8),
    ("通话内容", 100),
    ("复核_is_fraud", 11),
    ("复核_is_violation", 12),
    ("复核_risk", 10),
    ("复核_scene", 24),
    ("复核_subtype", 20),
    ("复核结论", 12),
    ("复核备注", 30),
    ("复核人", 10),
    ("复核日期", 12),
]
MANUAL_COLS = {"复核_is_fraud", "复核_is_violation", "复核_risk", "复核_scene",
               "复核_subtype", "复核结论", "复核备注", "复核人", "复核日期"}


def safe(value) -> str:
    text = "" if value is None else str(value)
    if text and text[0] in "=+-@":
        text = "'" + text
    return text


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in load_jsonl(path)}


def priority(rec: dict) -> str:
    if rec["kind"] == "涉诈":
        if rec["scene"] == "贷款相关-ab贷":
            return "P0_ab贷"
        if rec["conf"] >= 0.9:
            return "P1_高置信"
        if rec["scene"] == "引导贷款用户添加第三方微信":
            return "P3_引导加微"
        return "P2_其他"
    if rec["old_label"] == "正常":
        return "P3_4.1硬负例"
    return "备用"


def build_rows(args) -> list[dict]:
    rows: list[dict] = []
    sources = [
        (Path(args.normal_conflict), Path(args.normal_conflict_preds)),
        (Path(args.forward), Path(args.forward_preds)),
    ]
    for gold_path, pred_path in sources:
        preds = load_predictions(pred_path)
        for g in load_jsonl(gold_path):
            parsed = parse_agent_result(g["agent_res"])
            sug = suggested_v2(parsed)
            pred = preds.get(g["id"])
            p_fraud = float(pred["probs"][1]) if pred else None
            conf = float(g.get("agent_conf") or 0.0)
            rec = {
                "id": g["id"],
                "state": g.get("state") or "",
                "source_file": g.get("source_file") or "",
                "source_date": g.get("source_date") or "",
                "old_label": g.get("old_label") or "",
                "agent_res": g["agent_res"],
                "kind": parsed["kind"],
                "agent_risk": parsed["risk"],
                "scene": parsed["scene"],
                "subtype": parsed["subtype"],
                "conf": conf,
                "p_fraud": p_fraud,
                "suggested": sug,
            }
            if parsed["kind"] == "涉诈":
                rec["queue"] = "agent涉诈"
            elif rec["old_label"] == "正常":
                rec["queue"] = "正常冲突"
            elif parsed["kind"] == "正常":
                rec["queue"] = "旧涉诈_agent正常"
            else:
                continue
            rec["priority"] = priority(rec)
            rows.append(rec)
    # Mark a fixed random sample of the old-fraud/agent-normal queue.
    pool = sorted((r for r in rows if r["queue"] == "旧涉诈_agent正常"),
                  key=lambda r: r["id"])
    sample = set(r["id"] for r in random.Random(20261120).sample(pool, min(100, len(pool))))
    for r in pool:
        r["priority"] = "抽样100" if r["id"] in sample else "备用"
    return rows


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (
        PRIORITY_ORDER.get(r["priority"], 9),
        -(r["p_fraud"] if r["p_fraud"] is not None else -1),
        r["id"],
    ))


def write_table(ws, rows: list[dict]) -> None:
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

    for n, r in enumerate(rows, start=1):
        sug = r["suggested"]
        ws.append([
            n, r["queue"], r["priority"], r["source_file"], r["id"],
            r["source_date"], r["old_label"], r["agent_res"], r["kind"],
            r["agent_risk"] or "", r["scene"], r["subtype"] or "",
            r["conf"], "" if r["p_fraud"] is None else r["p_fraud"],
            "" if r["p_fraud"] is None else int(r["p_fraud"] >= 0.3074820786714554),
            sug["is_fraud"], sug["is_violation"], sug["risk_level"],
            sug["scene_category"], sug["scene_subtype"],
            "是" if sug["scene_conflict"] else "", safe(r["state"]),
            "", "", "", "", "", "", "", "", "",
        ])
        line = ws.max_row
        ws.row_dimensions[line].height = 62
        for col, cell in enumerate(ws[line], start=1):
            cell.alignment = WRAP_TOP
            cell.border = BORDER
            if COLUMNS[col - 1][0] in MANUAL_COLS:
                cell.fill = MANUAL_FILL
            elif COLUMNS[col - 1][0] == "probe_P涉诈" and isinstance(cell.value, float):
                cell.number_format = "0.000"
        risk_cell = ws.cell(row=line, column=18)
        if risk_cell.value in RISK_FILL:
            risk_cell.fill = RISK_FILL[risk_cell.value]

    if not rows:
        return
    last = len(rows) + 1
    fraud_dv = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    risk_dv = DataValidation(type="list", formula1='"合规,低风险,中风险,高风险"',
                             allow_blank=True)
    verdict_dv = DataValidation(type="list", formula1='"接受建议,改判,存疑"',
                                allow_blank=True)
    for dv in (fraud_dv, risk_dv, verdict_dv):
        ws.add_data_validation(dv)
    names = [c for c, _ in COLUMNS]
    fraud_dv.add(f"{get_column_letter(names.index('复核_is_fraud') + 1)}2:"
                 f"{get_column_letter(names.index('复核_is_fraud') + 1)}{last}")
    risk_dv.add(f"{get_column_letter(names.index('复核_risk') + 1)}2:"
                f"{get_column_letter(names.index('复核_risk') + 1)}{last}")
    verdict_dv.add(f"{get_column_letter(names.index('复核结论') + 1)}2:"
                   f"{get_column_letter(names.index('复核结论') + 1)}{last}")


def write_summary(ws, rows: list[dict]) -> None:
    ws.title = "汇总"
    ws.column_dimensions["A"].width = 30
    for col in "BCDE":
        ws.column_dimensions[col].width = 12
    ws["A1"] = "外部重庆批次复核队列汇总"
    ws["A1"].font = Font(bold=True, size=14)
    ws.merge_cells("A1:E1")

    def block(title: str, counts: collections.Counter, row: int) -> int:
        ws.cell(row=row, column=1, value=title).font = Font(bold=True)
        ws.cell(row=row + 1, column=1, value="类别")
        ws.cell(row=row + 1, column=2, value="条数")
        for cell in ws[row + 1][:2]:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.border = BORDER
        line = row + 2
        for name, n in counts.most_common():
            ws.cell(row=line, column=1, value=name)
            ws.cell(row=line, column=2, value=n)
            for col in (1, 2):
                ws.cell(row=line, column=col).border = BORDER
            line += 1
        return line + 1

    line = block("按队列", collections.Counter(r["queue"] for r in rows), 3)
    line = block("按优先级", collections.Counter(r["priority"] for r in rows), line)
    fraud_rows = [r for r in rows if r["queue"] == "agent涉诈"]
    line = block("agent涉诈：按场景", collections.Counter(r["scene"] for r in fraud_rows), line)
    line = block("正常冲突：按 agent 判定", collections.Counter(
        r["kind"] for r in rows if r["queue"] == "正常冲突"), line)


def write_notes(ws, rows: list[dict]) -> None:
    ws.title = "说明"
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 110
    counts = collections.Counter(r["queue"] for r in rows)
    lines = [
        ("用途", "重庆行业卡外部批次的 4.2 涉诈候选与标签冲突复核表。"),
        ("队列", f"agent涉诈 {counts.get('agent涉诈', 0)} 条；"
                 f"正常冲突 {counts.get('正常冲突', 0)} 条（另有 134 条旧标正常、agent 判涉诈的行"
                 f"已归入 agent涉诈 页，合计 469 条冲突）；"
                 f"旧涉诈_agent正常 {counts.get('旧涉诈_agent正常', 0)} 条（其中随机抽样 100 条）。"),
        ("agent 不是金标", "与 v2 重合 4,670 条的核对：agent 4.2 precision 0.245、recall 0.730；"
                            "引导加微 1/60（0.017）、ab贷 24/32（0.75）。必须逐条复核，不能自动并入。"),
        ("建议字段", "由 agent 结果机械映射，仅作参考；发现 agent 判涉诈但场景是 4.1 类目时，"
                     "“场景冲突”列会标“是”（例如 涉诈-高风险-贷款相关 缺 4.2 子类）。"),
        ("优先级", "P0_ab贷 最先看（历史精度 0.75）；P1_高置信（conf≥0.9）；"
                    "P2_其他；P3_引导加微 最后看（历史精度 0.017）。"
                    "正常冲突里的 P3_4.1硬负例 是硬负例候选，不是正例。"),
        ("填写方式", "复核_is_fraud / 复核_is_violation：0/1；复核_risk：合规/低/中/高；"
                     "复核_scene/subtype 按 spec；复核结论：接受建议/改判/存疑。"
                     "接受建议时可只填“复核结论”，改判时填对应字段。"),
        ("标签口径", "4.2 涉诈 = is_fraud=1 且风险=高；仅 4.1 违规 = is_violation=1, is_fraud=0；"
                     "合规 = 两者都为 0。旧 label 只是粗标签，不能作为 v2 目标。"),
        ("probe 列", "probe_P涉诈 是冻结 v2 probe（layer 24 / L2 0.1 / balanced）的概率；"
                     "probe@FPR5 显示是否 ≥0.3075。probe 只用于排序参考，不参与复核结论。"),
        ("probe 与 agent 分歧", "1030 的 134 条中：年报补录 probe 平均 P=0.69（15/21 ≥0.5，可能为真）；"
                                  "引导加微 0.08（2/49）、验证码 0.07（1/23），agent 可能误报；"
                                  "ab贷 0.17（0/4），probe 可能漏。1103-1113 的 73 条中："
                                  "ab贷 12/14 P≥0.5（强一致），引导加微 3/40、手机租赁 0/9。"
                                  "probe 不是金标，只作排序参考。"),
        ("回填", "复核完成后按 callid 合并，label_source=external_review；"
                 "正例才可进入时间外推召回评估，冲突行只能进硬负例或误报分析。"),
        ("当前策略", "1103-1113 的 14 条 ab贷 已由用户确认为正例（confirmed_positive.csv）；"
                     "其余 193 条按场景分层抽样 88 条（review_sample.xlsx），"
                     "抽样精度达标（点估计≥0.90 且 Wilson 下界≥0.70）才自动采纳该场景剩余行，"
                     "否则保持待定。probe 不参与标签采纳。"),
    ]
    for i, (key, value) in enumerate(lines, start=1):
        ws.cell(row=i, column=1, value=key).font = Font(bold=True)
        ws.cell(row=i, column=2, value=value).alignment = WRAP_TOP
        ws.row_dimensions[i].height = 42


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--normal-conflict", default="runs/external_cq/normal_1030_unmatch.jsonl")
    ap.add_argument("--normal-conflict-preds",
                    default="runs/external_cq/normal_1030_unmatch/predictions.jsonl")
    ap.add_argument("--forward", default="runs/external_cq/forward_1103_1113.jsonl")
    ap.add_argument("--forward-preds",
                    default="runs/external_cq/forward_1103_1113/predictions.jsonl")
    ap.add_argument("--out", default="runs/external_cq/review_queue_external.xlsx")
    ap.add_argument("--csv-out", default="runs/external_cq/review_queue_external.csv")
    args = ap.parse_args()

    rows = build_rows(args)
    wb = Workbook()
    write_notes(wb.active, rows)
    write_summary(wb.create_sheet(), rows)

    fraud = sort_rows([r for r in rows if r["queue"] == "agent涉诈"])
    conflict = sort_rows([r for r in rows if r["queue"] == "正常冲突"])
    old_normal = sort_rows([r for r in rows if r["queue"] == "旧涉诈_agent正常"])
    write_table(wb.create_sheet(f"agent涉诈{len(fraud)}"), fraud)
    write_table(wb.create_sheet(f"正常冲突{len(conflict)}"), conflict)
    write_table(wb.create_sheet(f"旧涉诈_agent正常{len(old_normal)}"), old_normal)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)

    csv_path = Path(args.csv_out)
    names = [c for c, _ in COLUMNS]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for n, r in enumerate(sort_rows(rows), start=1):
            sug = r["suggested"]
            writer.writerow({
                "序号": n, "队列": r["queue"], "优先级": r["priority"],
                "来源文件": r["source_file"], "callid": r["id"],
                "source_date": r["source_date"], "旧标签": r["old_label"],
                "agent结果": r["agent_res"], "agent判定": r["kind"],
                "agent风险": r["agent_risk"] or "", "agent场景": r["scene"],
                "agent子类": r["subtype"] or "", "agent置信度": r["conf"],
                "probe_P涉诈": "" if r["p_fraud"] is None else f"{r['p_fraud']:.6f}",
                "probe@FPR5": "" if r["p_fraud"] is None else int(
                    r["p_fraud"] >= 0.3074820786714554),
                "建议_is_fraud": sug["is_fraud"], "建议_is_violation": sug["is_violation"],
                "建议_risk": sug["risk_level"], "建议_scene": sug["scene_category"],
                "建议_subtype": sug["scene_subtype"],
                "场景冲突": "是" if sug["scene_conflict"] else "",
                "通话内容": r["state"],
            })
    print(f"wrote {out} and {csv_path}")
    print(f"queues: " + ", ".join(f"{k}={v}" for k, v in
                                  collections.Counter(r['queue'] for r in rows).items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
