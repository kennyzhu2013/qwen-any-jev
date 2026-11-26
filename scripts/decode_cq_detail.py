# -*- coding: utf-8 -*-
r"""将 cq_agent_detail 列 JSON 中的 \\uXXXX 中文转义解码为正常中文并写回。

- 先备份原文件为 *.bak.xlsx
- 逐单元格 json.loads + json.dumps(ensure_ascii=False)，解析失败则用正则兜底解码
- 仅改写该列，其余列、其他工作表不动
"""
import json
import re
import shutil
import sys
from pathlib import Path

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(__file__).resolve().parents[1] / "assets" / "10.22agent和人工复核结果.xlsx"
BAK = SRC.with_name(SRC.stem + ".bak.xlsx")
COL = "cq_agent_detail"
ESC = re.compile(r"\\u[0-9a-fA-F]{4}")


def decode_escape(s: str) -> str:
    r"""正则兜底：把字符串里的 \\uXXXX 逐个还原为对应字符（含代理对）。"""
    def repl(m):
        code = m.group(0)
        try:
            return json.loads('"' + code + '"')
        except ValueError:
            return code
    return ESC.sub(repl, s)


def convert(v):
    """返回 (新值, 方式说明)。"""
    if not isinstance(v, str) or not ESC.search(v):
        return v, "skip"
    try:
        obj = json.loads(v)
    except ValueError:
        return decode_escape(v), "regex"
    return json.dumps(obj, ensure_ascii=False), "json"


if not BAK.exists():
    shutil.copy2(SRC, BAK)
    print("backup ->", BAK.name)

wb = openpyxl.load_workbook(SRC)
n_json = n_regex = n_skip = n_escaped = n_toolong = 0
for ws in wb.worksheets:
    header = [c.value for c in ws[1]]
    if COL not in header:
        continue
    col = header.index(COL) + 1
    for r in range(2, ws.max_row + 1):
        cell = ws.cell(row=r, column=col)
        new, how = convert(cell.value)
        if how == "skip":
            n_skip += 1
            continue
        if how == "regex":
            n_regex += 1
        else:
            n_json += 1
        if ESC.search(new):
            n_escaped += 1
        if isinstance(new, str) and len(new) > 32767:
            n_toolong += 1
            print(f"  WARN {ws.title}!{cell.coordinate}: length {len(new)} > 32767")
        cell.value = new

wb.save(SRC)
print(f"done: json={n_json} regex={n_regex} skip={n_skip}")
print(f"残留转义单元格={n_escaped} 超长单元格={n_toolong}")
