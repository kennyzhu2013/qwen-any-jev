#!/usr/bin/env python
"""Render the M0.5 report from frozen dev/gate results.

This script never trains or selects anything.  It reads:
  runs/m0.5/dev/results.json
  runs/m0.5/selection.json
  runs/m0.5/gate/results.json
  runs/m0.5/probe/probe_grid.json
  eval/goldens.jsonl  (for error snippets / length buckets)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import mcnemar_exact, paired_bootstrap_diff  # noqa: E402
from eval.task import OPTIONS  # noqa: E402

ORDER = [
    "raw", "raw-temp", "raw-prior",
    "l0-noprior", "l0-noprior-temp", "l0-noprior-prior",
    "l0-batch", "l0-cf",
    "probe", "probe-riskfirst", "probe@fpr5", "probe@fpr10",
]
LABELS = {"raw": "raw", "raw-temp": "raw+温度", "raw-prior": "raw+固定prior",
          "l0-noprior": "L0 无prior", "l0-noprior-temp": "L0+温度",
          "l0-noprior-prior": "L0+固定prior", "l0-batch": "L0 batch prior",
          "l0-cf": "L0 content-free", "probe": "线性 probe",
          "probe-riskfirst": "probe 风险优先",
          "probe@fpr5": "probe@FPR5%", "probe@fpr10": "probe@FPR10%"}


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f(x, d: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def ci(pair) -> str:
    if not pair or len(pair) != 2:
        return "—"
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def frontier_rows(fr: dict) -> list[list[str]]:
    rows = []
    for name, s in fr.get("configs", {}).items():
        if "@" in name:
            # Thresholded variants share the underlying probabilities with the
            # unthresholded config; the sweep would duplicate that row.
            continue
        b = s.get("max_balanced_accuracy")
        h = s.get("fraud_recall_ge_0.95", {}).get("max_normal_recall")
        if not b or not h:
            continue
        rows.append([
            LABELS.get(name, name),
            f(b["threshold"], 3), f(b["fraud_recall"]), f(b["normal_recall"]),
            f(b["balanced_accuracy"]),
            f(h["threshold"], 3), f(h["fraud_recall"]), f(h["normal_recall"]),
            f(h["balanced_accuracy"]),
        ])
    return rows


def get_config(payload: dict, name: str) -> dict | None:
    gate = payload.get("splits", {}).get("gate", {})
    if name in gate:
        return gate[name]
    if name in payload.get("derived", {}):
        return payload["derived"][name]
    return None


def metric_row(gate: dict, name: str) -> list[str]:
    cfg = get_config(gate, name)
    if cfg is None:
        return [LABELS.get(name, name), "—"] + ["—"] * 13
    m = cfg["metrics"]
    raw = get_config(gate, "raw")
    rel = "—"
    if raw and name not in ("probe", "probe@fpr5", "probe@fpr10"):
        base = raw.get("per_row_s")
        cur = cfg.get("per_row_s")
        if base and cur:
            rel = f"{cur / base:.2f}×"
    elif name.startswith("probe"):
        rel = "1× prefill"
    r5 = next((x for x in m["recall_at_fpr"] if abs(x["max_fpr"] - 0.05) < 1e-9), {})
    return [
        LABELS.get(name, name),
        str(m["n"]),
        f(m["accuracy"]),
        f(m["balanced_accuracy"]),
        f"{f(m['fraud_recall'])} {ci(m['fraud_recall_ci95'])}",
        f"{f(m['normal_recall_tnr'])} {ci(m['normal_recall_ci95'])}",
        f(m["fraud_precision"]),
        f(m["mcc"]),
        f(m["pr_auc"]),
        f(m["ece_10bin"]),
        f(m["nll"]),
        f(r5.get("recall")),
        f(m["mean_confidence"]),
        f(cfg.get("per_row_s"), 4) if cfg.get("per_row_s") else "—",
        rel,
    ]


def paired(gate: dict, a: str, b: str) -> dict | None:
    ca, cb = get_config(gate, a), get_config(gate, b)
    if not ca or not cb:
        return None
    rows_a, rows_b = ca["rows"], cb["rows"]
    by_b = {r["id"]: r for r in rows_b}
    labels, ans_a, ans_b = [], [], []
    for r in rows_a:
        other = by_b.get(r["id"])
        if other is None:
            continue
        labels.append(r["label"])
        ans_a.append(r["answer"])
        ans_b.append(other["answer"])
    acc = paired_bootstrap_diff(labels, ans_a, ans_b, OPTIONS, metric="accuracy")
    mtpr = paired_bootstrap_diff(labels, ans_a, ans_b, OPTIONS, metric="macro_tpr")
    mc = mcnemar_exact(ans_a, ans_b, labels)
    return {"accuracy": acc, "macro_tpr": mtpr, "mcnemar": mc}


def error_analysis(gate: dict, name: str, goldens: dict, top: int = 12) -> tuple[list[dict], str]:
    cfg = get_config(gate, name)
    if not cfg:
        return [], ""
    errors = [r for r in cfg["rows"] if r["answer"] != r["label"]]
    out = []
    for r in errors[:top]:
        g = goldens.get(r["id"], {})
        out.append({
            "id": r["id"],
            "label": r["label"],
            "answer": r["answer"],
            "p_fraud": round(float(r["probs"][1]), 4),
            "category": r.get("category"),
            "chars": len(g.get("state", "")),
            "snippet": (g.get("state", "")[:160]).replace("\n", " "),
        })
    counter = Counter((r["label"], r["answer"]) for r in errors)
    lines = [f"- 错误数：{len(errors)} / {len(cfg['rows'])}"]
    for (lab, ans), n in counter.most_common():
        lines.append(f"  - 真值 `{lab}` → 预测 `{ans}`：{n}")
    return out, "\n".join(lines)


def category_recall(gate: dict, names: list[str]) -> str:
    """Fraud recall by source category for selected configs (gate, post-hoc)."""
    cats: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for name in names:
        cfg = get_config(gate, name)
        if not cfg:
            continue
        for r in cfg["rows"]:
            if r["label"] != "涉诈":
                continue
            c = r.get("category") or "其他"
            hit, total = cats[c].get(name, [0, 0])
            cats[c][name] = [hit + int(r["answer"] == "涉诈"), total + 1]
    rows = []
    for c in sorted(cats, key=lambda c: -max(v[1] for v in cats[c].values())):
        row = [c]
        for name in names:
            hit, total = cats[c].get(name, [0, 0])
            row.append(f"{hit / total:.3f} ({total})" if total else "—")
        rows.append(row)
    return table(["类型"] + [LABELS.get(n, n) for n in names], rows)


def bucket_analysis(gate: dict, name: str, goldens: dict) -> str:
    cfg = get_config(gate, name)
    if not cfg:
        return ""
    buckets = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 10**9)]
    rows = []
    for lo, hi in buckets:
        sel = [r for r in cfg["rows"]
               if lo <= len(goldens.get(r["id"], {}).get("state", "")) < hi]
        if not sel:
            continue
        fraud = [r for r in sel if r["label"] == "涉诈"]
        normal = [r for r in sel if r["label"] == "正常"]
        fr = sum(r["answer"] == "涉诈" for r in fraud) / len(fraud) if fraud else float("nan")
        nr = sum(r["answer"] == "正常" for r in normal) / len(normal) if normal else float("nan")
        rows.append([f"{lo}-{hi}", str(len(sel)), str(len(normal)),
                     f(fr), f(nr)])
    return table(["字符长度", "n", "正常数", "涉诈 recall", "正常 recall"], rows)


def main() -> int:
    root = Path("runs/m0.5")
    dev = load(root / "dev/results.json")
    sel = load(root / "selection.json")
    gate = load(root / "gate/results.json")
    grid = load(root / "probe/probe_grid.json")
    goldens = {json.loads(line)["id"]: json.loads(line)
               for line in Path("eval/goldens.jsonl").open(encoding="utf-8")}
    stats = load("eval/stats.json")

    probe = get_config(gate, "probe")
    if not probe:
        raise SystemExit("gate results have no probe config")
    pm = probe["metrics"]
    baseline_names = ["raw", "raw-prior", "l0-noprior", "l0-noprior-prior",
                      "l0-batch", "l0-cf"]
    # Baseline selection is a dev decision: pick the strongest non-probe
    # configuration on dev, then read its frozen gate score.  Selecting the
    # baseline on gate would leak gate labels into the comparison.
    def dev_bal(name: str) -> float | None:
        if name in dev.get("splits", {}).get("dev", {}):
            return float(dev["splits"]["dev"][name]["metrics"]["balanced_accuracy"])
        if name.endswith("-prior"):
            base = name[: -len("-prior")]
            p = sel.get("prior", {}).get(base)
            if p:
                row = next((r for r in p["table"] if r["strength"] == p["strength"]), None)
                if row:
                    return float(row["balanced_accuracy"])
        return None

    baseline_name = max(
        (n for n in baseline_names if dev_bal(n) is not None and get_config(gate, n)),
        key=lambda n: dev_bal(n),
    )
    baseline = (baseline_name, get_config(gate, baseline_name)["metrics"])
    vs = paired(gate, "probe", baseline[0])
    delta = vs["accuracy"] if vs else None   # placeholder; use metric specific below
    probe_vs_base_bal = None
    if vs:
        probe_vs_base_bal = vs["macro_tpr"]["point"]
    gate_pass = (
        pm["balanced_accuracy"] >= 0.70
        and pm["fraud_recall"] >= 0.95
        and pm["normal_recall_tnr"] >= 0.50
        and pm["ece_10bin"] <= 0.10
        and probe_vs_base_bal is not None
        and probe_vs_base_bal >= 0.05
        and vs["macro_tpr"]["ci95"][0] > 0
    )

    lines = []
    lines.append("# M0.5 · 真实人工复核数据上的 off-ramp 对比")
    lines.append("")
    lines.append(f"- 模型：`Qwen/Qwen3.5-4B`；评测集：`eval/goldens.jsonl`（{stats['n_rows']} 条，2026-07-29 单日）")
    lines.append(f"- 拆分：cal {stats['split_counts']['cal']} / dev {stats['split_counts']['dev']} / gate {stats['split_counts']['gate']}")
    lines.append(f"- gate 标签：正常 {stats['label_by_split']['gate'].get('正常', 0)} / 涉诈 {stats['label_by_split']['gate'].get('涉诈', 0)}")
    lines.append("- 协议：所有选择（probe 层/L2/类权重、prior 强度、温度）只在 cal/dev 上做；gate 只评分一次。")
    lines.append("- 任务定义与 rubric：`eval/task.py`；标签语义与限制：`eval/README.md`。")
    lines.append("")
    lines.append("## 1. 结论")
    lines.append("")
    lines.append(f"- 最佳非 probe baseline：**{LABELS.get(baseline[0], baseline[0])}**，"
                 f"balanced accuracy **{baseline[1]['balanced_accuracy']:.3f}**，"
                 f"涉诈 recall **{baseline[1]['fraud_recall']:.3f}**，"
                 f"正常 recall **{baseline[1]['normal_recall_tnr']:.3f}**。")
    lines.append(f"- 线性 probe（dev 上按 balanced accuracy 选）：balanced accuracy **{pm['balanced_accuracy']:.3f}**，"
                 f"涉诈 recall **{pm['fraud_recall']:.3f}**，"
                 f"正常 recall **{pm['normal_recall_tnr']:.3f}**，"
                 f"ECE **{pm['ece_10bin']:.3f}**。")
    rf = get_config(gate, "probe-riskfirst")
    if rf:
        rm = rf["metrics"]
        lines.append(f"- 线性 probe（风险优先：dev 涉诈 recall≥0.95 下最大化正常 recall）："
                     f"balanced accuracy **{rm['balanced_accuracy']:.3f}**，"
                     f"涉诈 recall **{rm['fraud_recall']:.3f}**，"
                     f"正常 recall **{rm['normal_recall_tnr']:.3f}**。")
    if probe_vs_base_bal is not None:
        lines.append(f"- probe 相对最佳 baseline 的 balanced accuracy 差：**{probe_vs_base_bal:+.3f}** "
                     f"(95% CI {ci(vs['macro_tpr']['ci95'])})。")
    lines.append(f"- 预注册门禁：**{'通过' if gate_pass else '未通过'}**。"
                 f"（bal≥0.70, 涉诈 recall≥0.95, 正常 recall≥0.50, ECE≤0.10, "
                 f"相对 baseline bal 提升≥5pp 且 CI 下界>0）")
    criteria = [
        ("balanced accuracy ≥ 0.70", pm["balanced_accuracy"] >= 0.70,
         f"{pm['balanced_accuracy']:.3f}"),
        ("涉诈 recall ≥ 0.95", pm["fraud_recall"] >= 0.95,
         f"{pm['fraud_recall']:.3f}"),
        ("正常 recall ≥ 0.50", pm["normal_recall_tnr"] >= 0.50,
         f"{pm['normal_recall_tnr']:.3f}"),
        ("ECE ≤ 0.10", pm["ece_10bin"] <= 0.10, f"{pm['ece_10bin']:.3f}"),
        ("相对 baseline bal ≥ +5pp",
         probe_vs_base_bal is not None and probe_vs_base_bal >= 0.05,
         f"{probe_vs_base_bal:+.3f}" if probe_vs_base_bal is not None else "—"),
        ("paired bootstrap CI 下界 > 0",
         vs is not None and vs["macro_tpr"]["ci95"][0] > 0,
         ci(vs["macro_tpr"]["ci95"]) if vs else "—"),
    ]
    for label, ok, val in criteria:
        lines.append(f"  - {'✅' if ok else '❌'} {label}：{val}")
    alt = get_config(gate, "l0-noprior-prior")
    if alt and alt["metrics"]["balanced_accuracy"] > baseline[1]["balanced_accuracy"]:
        alt_vs = paired(gate, "probe", "l0-noprior-prior")
        lines.append(f"- baseline 按协议在 dev 上冻结为 {LABELS.get(baseline[0], baseline[0])}；"
                     f"gate 上 L0+固定 prior 的 bal 反而更高"
                     f"（{alt['metrics']['balanced_accuracy']:.3f} vs "
                     f"{baseline[1]['balanced_accuracy']:.3f}）。不改选 baseline；"
                     f"若以 L0+固定 prior 为参照，probe 的 Δbal 仍为 "
                     f"{alt_vs['macro_tpr']['point']:+.3f} "
                     f"（95% CI {ci(alt_vs['macro_tpr']['ci95'])}）。")
    lines.append("- 关键限定：冻结 probe 的唯一失败项是**涉诈 recall**（0.875），"
                 "不是平衡性；gate 上事后把阈值扫到涉诈 recall 0.951 时正常 recall 恰为 0.500，"
                 "但 dev 在同一约束下只有 0.219 —— 这是 32/64 条正常样本下的边界波动，"
                 "不是一个稳健的可用操作点。")
    lines.append("")
    if gate_pass:
        lines.append("**判定：不做 LoRA。** 线性头在 gate 上过门禁，项目应以「基座 + AnyJev/线性头」结束 off-ramp。")
    else:
        lines.append("**判定：线性头未过门禁，暂不进入 LoRA 训练。** 按预注册协议，LoRA 是下一层评估对象；"
                     "但 dev/gate 的错误结构显示瓶颈更像是「标签依赖通话外业务上下文」与操作点，"
                     "而不是模型容量。前置动作是补齐风险等级/违规类型与业务归属标签、重定义漏判代价，"
                     "再用同一套 gate 复评；直接开训只会把同样的标签噪声蒸馏进学生模型。")
    lines.append("")
    lines.append("## 2. Dev 选择（gate 未参与）")
    lines.append("")
    lines.append("### 2.1 Probe 网格（前 12）")
    grid_rows = sorted(grid["grid"], key=lambda r: (-r["dev_score"], -r["dev_fraud_recall"]))[:12]
    lines.append(table(
        ["layer", "L2", "class weight", "dev bal/macroF1", "dev acc",
         "dev 涉诈 recall", "dev 正常 recall", "dev ECE"],
        [[str(r["layer"]), f"{r['l2']:g}", r["class_weight"],
          f(r["dev_score"]), f(r["dev_accuracy"]),
          f(r["dev_fraud_recall"]), f(r["dev_normal_recall"]), f(r["dev_ece"])]
         for r in grid_rows],
    ))
    lines.append(f"\n选中：layer **{grid['selected']['layer']}**, L2 **{grid['selected']['l2']:g}**, "
                 f"class weight **{grid['selected']['class_weight']}**, "
                 f"thresholds `{grid['selected']['thresholds']}`。")
    if grid.get("selected_riskfirst"):
        r = grid["selected_riskfirst"]
        lines.append(f"\n风险优先备选：layer **{r['layer']}**, L2 **{r['l2']:g}**, "
                     f"class weight **{r['class_weight']}**，"
                     f"dev 涉诈 recall **{r['dev_metrics']['fraud_recall']:.3f}**，"
                     f"正常 recall **{r['dev_metrics']['normal_recall_tnr']:.3f}**。")
    lines.append("")
    lines.append("### 2.2 固定 prior 强度（dev）")
    for config in ("raw", "l0-noprior"):
        if config not in sel.get("prior", {}):
            continue
        p = sel["prior"][config]
        lines.append(f"\n**{LABELS.get(config, config)}**，cal prior = "
                     f"`{p['prior']}`，选中 strength = **{p['strength']}**")
        lines.append(table(
            ["strength", "bal acc", "acc", "涉诈 recall", "正常 recall", "ECE", "NLL"],
            [[f(r["strength"]), f(r["balanced_accuracy"]), f(r["accuracy"]),
              f(r["fraud_recall"]), f(r["normal_recall"]), f(r["ece"]), f(r["nll"])]
             for r in p["table"]],
        ))
    lines.append("")
    lines.append("### 2.3 温度（cal 上拟合 NLL）")
    lines.append("")
    lines.append(table(
        ["config", "temperature"],
        [[LABELS.get(k, k), f(v, 2)] for k, v in sel.get("temperature", {}).items()],
    ))
    lines.append("")
    frontier_path = root / "frontier_dev.json"
    if frontier_path.exists():
        fr = load(frontier_path)
        lines.append("### 2.4 Dev 操作点前沿（post-hoc，不参与选择）")
        lines.append("")
        lines.append("对每个已冻结配置的 P(涉诈) 扫阈值；"
                     "`max-bal` 是无约束的最佳平衡点，"
                     "`fraud≥0.95` 是在预注册召回约束下的最佳正常 recall。")
        lines.append("")
        lines.append(table(
            ["config", "max-bal t", "涉诈 R", "正常 R", "bal",
             "fraud≥.95 t", "涉诈 R", "正常 R", "bal"],
            frontier_rows(fr),
        ))
        lines.append("")
        lines.append("> 这张表只读 dev 概率，用来判断「同一模型换操作点」能否过门禁；"
                     "它不改变 gate 的冻结决策。")
        lines.append("")
    lines.append("## 3. Gate 结果")
    lines.append("")
    lines.append(table(
        ["config", "n", "acc", "bal acc", "涉诈 recall [95% CI]",
         "正常 recall [95% CI]", "涉诈 prec", "MCC", "PR-AUC", "ECE", "NLL",
         "recall@FPR5", "mean conf", "s/row", "相对 raw"],
        [metric_row(gate, n) for n in ORDER],
    ))
    lines.append("")
    lines.append("> `probe@FPR5/10` 的 threshold 在 dev 上按正常类经验 FPR 选取；"
                 "其 ECE/PR-AUC 仍基于未阈值化的概率，解读时只看 recall/精度。")
    lines.append("")
    frontier_gate_path = root / "frontier_gate.json"
    if frontier_gate_path.exists():
        frg = load(frontier_gate_path)
        lines.append("### 3.1 Gate 操作点前沿（post-hoc）")
        lines.append("")
        lines.append("与 §2.4 同一扫阈值方法，但只读 gate 已冻结的概率，用于回答"
                     "「同一模型换阈值能否同时满足涉诈 recall≥0.95 与正常 recall≥0.50」。")
        lines.append("")
        lines.append(table(
            ["config", "max-bal t", "涉诈 R", "正常 R", "bal",
             "fraud≥.95 t", "涉诈 R", "正常 R", "bal"],
            frontier_rows(frg),
        ))
        lines.append("")
        feasible = []
        for name, s in frg.get("configs", {}).items():
            if "@" in name:
                continue
            h = s.get("fraud_recall_ge_0.95", {}).get("max_normal_recall") or {}
            if h.get("normal_recall", 0.0) >= 0.50:
                feasible.append(LABELS.get(name, name))
        lines.append("- 同时满足「涉诈 recall≥0.95、正常 recall≥0.50」的配置："
                     + ("、".join(feasible) if feasible else "**无**")
                     + "。")
        lines.append("")
        if feasible:
            lines.append("> 注意：这是**扫 gate 概率**得到的事后操作点，不是预注册的决策规则。"
                         "冻结的 probe 用 argmax（0.5），其涉诈 recall 不达标；"
                         "若把阈值移到 frontier 给出的点，正常 recall 恰好卡在 0.50 边界。"
                         "dev 上同一约束（fraud≥0.95）只能得到正常 recall "
                         "0.219，说明该边界点在小样本下不稳定，不能据此判定门禁通过。"
                         "门禁结论以冻结决策为准。")
            lines.append("")
    triage_path = root / "triage.json"
    if triage_path.exists():
        tr = load(triage_path)
        lines.append("### 3.2 三路分流操作点（post-hoc）")
        lines.append("")
        lines.append("规则：`s ≤ lo` 自动正常，`s ≥ hi` 自动涉诈，其余转人工；"
                     "阈值只在 dev 上按下列策略选取，gate 只做对照评估、不参与选点。"
                     "策略菜单（复核预算档位、10% 正常误拒上限）是在看到 gate 前沿后"
                     "设定的诊断口径，因此整节只作 post-hoc 分析，"
                     "不改变 §1 的预注册门禁结论，也不作为上线依据。")
        lines.append("")
        policy_names = {"zero_escape": "零漏放优先", "balanced": "平衡",
                        "budget20": "复核≤20%", "budget30": "复核≤30%",
                        "budget40": "复核≤40%", "budget50": "复核≤50%"}
        rows = []
        for name, p in tr.get("policies", {}).items():
            for split in ("dev", "gate"):
                m = p.get(split)
                if not m:
                    continue
                rows.append([
                    policy_names.get(name, name),
                    "关闭" if p["lo"] < 0 else f(p["lo"]),
                    "关闭" if p["hi"] > 1 else f(p["hi"]),
                    split,
                    f(m["review_rate"]),
                    f(m["fraud_escape"]),
                    f(m["fraud_auto_catch"]),
                    f(m["normal_auto_pass"]),
                    f(m["normal_false_reject"]),
                    f(m["auto_accuracy"]),
                ])
        lines.append(table(
            ["策略", "lo", "hi", "split", "复核率", "漏放涉诈", "自动拦截涉诈",
             "正常自动通过", "正常误拒", "自动判对率"],
            rows,
        ))
        lines.append("")
        lines.append("dev 复核预算前沿（每个预算内最小化漏放）：")
        lines.append("")
        fr = tr.get("frontier_dev", [])
        lines.append(table(
            ["复核预算", "实际复核率", "lo", "hi", "漏放涉诈", "自动拦截涉诈",
             "正常自动通过", "正常误拒"],
            [["≤" + f"{x['review_budget'] * 100:.0f}%", f(x["review_rate"]),
              "关闭" if x["lo"] < 0 else f(x["lo"]),
              "关闭" if x["hi"] > 1 else f(x["hi"]),
              f(x["fraud_escape"]), f(x["fraud_auto_catch"]),
              f(x["normal_auto_pass"]), f(x["normal_false_reject"])]
             for x in fr],
        ))
        lines.append("")
        lines.append("- 前沿上所有点都取 `lo = 关闭`：当前 probe 的低分端没有足够干净的"
                     "正常类，设自动正常区间要么引入漏放，要么在同等漏放下不节省复核。")
        lines.append("- gate 上若复核容量 40%，`hi≈0.713` 可自动拦截 69.6% 涉诈，"
                     "但 17.2% 正常被误自动拒绝；容量 50% 时自动拦截 65.0%、"
                     "正常误拒 14.1%（同一阈值 dev 仅 6.2%，正常类 32 条不足以定阈值）。")
        lines.append("- 「零漏放优先」在 gate 上漏放 0、自动拦截 67.5% 涉诈、复核 38.2%，"
                     "代价是 15.6% 正常被自动拒绝。三路分流目前只能做"
                     "「高置信涉诈自动拦截 + 其余全复核」，不能做自动正常。")
        lines.append("- 采用前需要业务给出**误拒一个正常客户 vs 人工复核一条**的成本比，"
                     "并把正常类样本扩到 300–500 条后重定阈值；现有 dev 的 32 条正常"
                     "无法支撑阈值外推。")
        lines.append("- 代价陷阱：gate 涉诈占比 88.9%，「全部自动拒绝」的正常误拒成本只有"
                     " 0.111×C_FP/行；只要单条复核成本高于 0.111×C_FP，无约束的代价"
                     "最小化就会退化为全部拒绝。操作点必须先设硬约束（如正常误拒≤10%），"
                     "不能直接做代价最优。")
        lines.append("")
    lines.append("## 4. 配对比较（gate）")
    lines.append("")
    comps = [("probe", baseline[0]), ("probe", "raw"),
             (baseline[0], "raw"), ("l0-batch", "l0-noprior")]
    rows = []
    for a, b in comps:
        p = paired(gate, a, b)
        if not p:
            continue
        rows.append([
            f"{LABELS.get(a, a)} vs {LABELS.get(b, b)}",
            f"{p['accuracy']['point']:+.3f} {ci(p['accuracy']['ci95'])}",
            f"{p['macro_tpr']['point']:+.3f} {ci(p['macro_tpr']['ci95'])}",
            f"b={p['mcnemar']['n01']}, c={p['mcnemar']['n10']}, p={p['mcnemar']['p_value']:.3g}",
        ])
    lines.append(table(["比较", "Δaccuracy [95% CI]", "Δmacro TPR [95% CI]", "McNemar"], rows))
    lines.append("")
    lines.append("## 5. 错误分析")
    lines.append("")
    for name in ("probe", baseline[0]):
        errs, text = error_analysis(gate, name, goldens)
        lines.append(f"### {LABELS.get(name, name)}")
        lines.append("")
        lines.append(text)
        lines.append("")
        lines.append(table(
            ["id", "真值", "预测", "P(涉诈)", "类型", "字符", "片段"],
            [[e["id"], e["label"], e["answer"], str(e["p_fraud"]),
              e["category"] or "—", str(e["chars"]), e["snippet"]]
             for e in errs],
        ))
        lines.append("")
    lines.append("### 按长度分桶（probe）")
    lines.append("")
    lines.append(bucket_analysis(gate, "probe", goldens))
    lines.append("")
    lines.append("### 按违规类型分桶（gate 涉诈 recall）")
    lines.append("")
    lines.append(category_recall(gate, ["raw", baseline[0], "probe", "probe-riskfirst"]))
    lines.append("")
    rev_path = root / "reversal.json"
    if rev_path.exists():
        rev = load(rev_path)
        lines.append("### 选项顺序敏感性（dev 子集，post-hoc）")
        lines.append("")
        lines.append(f"dev 抽样 {rev['n']} 条（正常 {rev['normal']} / 涉诈 {rev['fraud']}），"
                     "比较选项顺序 `(正常, 涉诈)` 与 `(涉诈, 正常)` 的答案翻转率。")
        lines.append("")
        rrows = []
        for name, c in rev.get("configs", {}).items():
            bl = c.get("flip_by_label", {})
            rrows.append([
                name, str(c["flip"]["n"]), str(c["flip"]["flips"]),
                f(c["flip"]["rate"]),
                f(bl.get("正常", {}).get("rate")),
                f(bl.get("涉诈", {}).get("rate")),
            ])
        lines.append(table(["config", "n", "flips", "flip rate", "正常 flip", "涉诈 flip"], rrows))
        lines.append("")
    lines.append("## 6. 判读与下一步")
    lines.append("")
    lines.append("- 本评估集只有一天，gate 不是时间外推；正常类 gate 只有 "
                 f"{stats['label_by_split']['gate'].get('正常', 0)} 条，"
                 "正常 recall 的 95% CI 约 ±12pp，不能支撑细粒度上线承诺。")
    n_gate = stats["split_counts"]["gate"]
    n_gate_fraud = stats["label_by_split"]["gate"].get("涉诈", 0)
    raw_gate = get_config(gate, "raw")
    if raw_gate:
        lines.append(f"- 类别极不平衡（gate 涉诈 {n_gate_fraud}/{n_gate} = "
                     f"{n_gate_fraud / n_gate:.1%}）：raw 的 gate accuracy "
                     f"{raw_gate['metrics']['accuracy']:.3f} 低于「全判涉诈」常数基线 "
                     f"{n_gate_fraud / n_gate:.3f}；accuracy 在本任务上没有判别力，"
                     "必须看 balanced accuracy 与两类 recall。")
    lines.append("- 源标签 `涉诈` 包含低风险违规（备注中 10 条明确写“低风险”），"
                 "不是严格的刑事诈骗标签；风险等级未系统标注，未做门禁。")
    lines.append("- dev 上正常类与涉诈类的贷款词（贷款/额度/信贷/资金/利息等）覆盖率分别为 "
                 "65.6%（21/32）与 66.5%（171/257），单看通话文本几乎不可分；"
                 "大量正常样本是合规贷款营销，而部分涉诈样本是违规催收/荐股。"
                 "这说明标签很可能取决于通话外的业务归属/资质，而不是话术本身——"
                 "这是任何 transcript-only 模型（probe 或 LoRA）的共同上限。")
    lines.append("- 三路分流（§3.2）：当前概率只能支撑「高置信涉诈自动拦截 + 其余转人工」，"
                 "不能自动放行正常；复核容量 40% 时自动拦截 69.6% 涉诈，"
                 "但 17.2% 正常被误自动拒绝。采用前必须先定成本比与正常误拒上限。")
    lines.append("- 如果 probe 未过门禁，下一步不是直接训 LoRA，而是："
                 "（a）补齐风险等级/违规类型标签；（b）按类型分层检查教师质量；"
                 "（c）确定漏判/误判代价与操作点；再进入 M1。")
    lines.append("")
    lines.append("## 7. 复现")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv312/bin/python eval/prepare.py")
    lines.append(".venv312/bin/python eval/features.py --out runs/m0.5/features_binary.npz")
    lines.append(".venv312/bin/python eval/probe.py --features runs/m0.5/features_binary.npz")
    lines.append(".venv312/bin/python -m eval.harness run --splits cal,dev --out-dir runs/m0.5/dev")
    lines.append(".venv312/bin/python -m eval.harness select")
    lines.append(".venv312/bin/python -m eval.harness gate")
    lines.append(".venv312/bin/python eval/frontier.py --split dev --out runs/m0.5/frontier_dev.json")
    lines.append(".venv312/bin/python eval/frontier.py --split gate --out runs/m0.5/frontier_gate.json")
    lines.append(".venv312/bin/python eval/reversal.py")
    lines.append(".venv312/bin/python eval/triage.py")
    lines.append(".venv312/bin/python eval/report.py")
    lines.append("```")
    lines.append("")

    out = root / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(f"probe gate pass: {gate_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
