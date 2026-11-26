#!/usr/bin/env python
"""Score the frozen v2 probe exactly once on gate and render a v2 report.

The probe grid and thresholds are selected on cal/dev by ``eval/probe.py``.
This script never trains or selects anything; it reads the frozen predictions
and computes the gate metrics, including the primary gate
(``split=gate`` and ``cohort=human_0729``) and a paired comparison against the
frozen v1 probe re-scored on the v2 labels.

Usage:
  python eval/report_v2.py
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.frontier import sweep  # noqa: E402
from eval.metrics import (  # noqa: E402
    mcnemar_exact, paired_bootstrap_diff, summarize_binary,
)

CONFIG_LABELS = {
    "probe": "probe（argmax）",
    "probe-riskfirst": "probe 风险优先（argmax）",
    "probe@devR95": "probe@dev-R95",
    "probe@fpr5": "probe@FPR5%",
    "probe@fpr10": "probe@FPR10%",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_predictions(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in load_jsonl(path)}


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


def answers_at(probs: np.ndarray, threshold: float) -> list[str]:
    return ["涉诈" if float(p[1]) >= threshold else "非涉诈" for p in probs]


def binom_le(k: int, n: int, p: float) -> float:
    """Exact P(X <= k) for X ~ Binomial(n, p); no scipy dependency."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def fpr_of(m: dict) -> float | None:
    c = m["confusion"]
    denom = c["fp"] + c["tn"]
    return c["fp"] / denom if denom else None


def metrics_for(
    ids: list[str], preds: dict[str, dict], labels: dict[str, str],
    goldens: dict[str, dict], options: list[str], threshold: float,
) -> dict:
    rows = [preds[i] for i in ids]
    y = [labels[i] for i in ids]
    probs = np.asarray([r["probs"] for r in rows], dtype=float)
    ans = answers_at(probs, threshold)
    m = summarize_binary(y, ans, probs, options)
    m["by_cohort"] = {}
    for cohort in sorted({goldens[i].get("cohort", "") for i in ids}):
        sel = [j for j, i in enumerate(ids) if goldens[i].get("cohort", "") == cohort]
        if not sel:
            continue
        yy = [y[j] for j in sel]
        mm = summarize_binary(yy, [ans[j] for j in sel], probs[sel], options)
        m["by_cohort"][cohort] = {
            "n": mm["n"], "support": mm["support"],
            "fraud_recall": mm["fraud_recall"],
            "normal_recall_tnr": mm["normal_recall_tnr"],
            "balanced_accuracy": mm["balanced_accuracy"],
            "accuracy": mm["accuracy"],
            "ece_10bin": mm["ece_10bin"],
        }
    def scene_of(i: str) -> str:
        g = goldens[i]
        if g.get("scene_category"):
            return g["scene_category"]
        # goldens.jsonl stores compliant rows with scene_category=null.
        return "正常" if int(g.get("is_violation") or 0) == 0 else "其他"

    m["fp_by_scene"] = dict(collections.Counter(
        scene_of(i)
        for i, a, b in zip(ids, y, ans) if a == "非涉诈" and b == "涉诈"
    ).most_common())
    m["fn_by_scene"] = dict(collections.Counter(
        scene_of(i)
        for i, a, b in zip(ids, y, ans) if a == "涉诈" and b == "非涉诈"
    ).most_common())
    m["errors"] = [
        {
            "id": i, "label": a, "answer": b, "p_fraud": float(p[1]),
            "cohort": goldens[i].get("cohort"), "scene": scene_of(i),
            "snippet": (goldens[i].get("state") or "")[:180].replace("\n", " "),
        }
        for i, a, b, p in zip(ids, y, ans, probs)
        if a != b
    ]
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="runs/v2/features_binary.npz")
    ap.add_argument("--probe-dir", default="runs/v2/probe")
    ap.add_argument("--goldens", default="eval_v2/goldens.jsonl")
    ap.add_argument("--v1-predictions", default="runs/m0.5/probe/predictions.jsonl")
    ap.add_argument("--out-dir", default="runs/v2")
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    ids = [str(x) for x in data["ids"]]
    labels = dict(zip(ids, [str(x) for x in data["labels"]]))
    splits = dict(zip(ids, [str(x) for x in data["splits"]]))
    cohorts = dict(zip(ids, [str(x) for x in data["cohorts"]]))
    options = [str(x) for x in data["options"]]
    goldens = {r["id"]: r for r in load_jsonl(Path(args.goldens))}
    missing = set(ids) - set(goldens)
    if missing:
        raise SystemExit(f"{len(missing)} feature ids missing from goldens")
    for i in ids:
        goldens[i].setdefault("cohort", cohorts[i])

    probe_dir = Path(args.probe_dir)
    grid = json.loads((probe_dir / "probe_grid.json").read_text(encoding="utf-8"))
    preds = {"probe": load_predictions(probe_dir / "predictions.jsonl")}
    thresholds = {
        "probe": float(grid["selected"]["thresholds"]["argmax"]),
        "probe@fpr5": float(grid["selected"]["thresholds"]["fpr_5"]),
        "probe@fpr10": float(grid["selected"]["thresholds"]["fpr_10"]),
    }
    configs = ["probe", "probe@fpr5", "probe@fpr10"]
    if (probe_dir / "predictions_riskfirst.jsonl").exists() and grid.get("selected_riskfirst"):
        preds["probe-riskfirst"] = load_predictions(
            probe_dir / "predictions_riskfirst.jsonl")
        thresholds["probe-riskfirst"] = float(
            grid["selected_riskfirst"]["thresholds"]["argmax"])
        configs.insert(1, "probe-riskfirst")

    # Dev-only operating-point frontier (post-hoc over dev, never touches gate).
    dev_ids = [i for i in ids if splits[i] == "dev"]
    dev_frontier = sweep([preds["probe"][i] for i in dev_ids])
    dev_r95 = (dev_frontier.get("fraud_recall_ge_0.95") or {}).get("max_normal_recall")
    if dev_r95:
        thresholds["probe@devR95"] = float(dev_r95["threshold"])
        configs.insert(1, "probe@devR95")

    # The threshold variants use the primary head's probabilities.
    for name in ("probe@devR95", "probe@fpr5", "probe@fpr10"):
        if name in thresholds:
            preds[name] = preds["probe"]

    gate_ids = [i for i in ids if splits[i] == "gate"]
    primary_ids = [i for i in gate_ids if cohorts[i] == "human_0729"]
    normal_ids = [i for i in gate_ids if cohorts[i] == "human_cq_normal"]

    results = {
        "features": args.features,
        "probe_dir": str(probe_dir),
        "options": options,
        "selection": grid["selected"],
        "selection_riskfirst": grid.get("selected_riskfirst"),
        "gate_full": {},
        "gate_primary": {},
        "gate_normal_cohort": {},
    }
    for name in configs:
        if gate_ids:
            results["gate_full"][name] = metrics_for(
                gate_ids, preds[name], labels, goldens, options, thresholds[name])
        if primary_ids:
            results["gate_primary"][name] = metrics_for(
                primary_ids, preds[name], labels, goldens, options, thresholds[name])
        if normal_ids:
            results["gate_normal_cohort"][name] = metrics_for(
                normal_ids, preds[name], labels, goldens, options, thresholds[name])

    results["dev_frontier"] = dev_frontier

    # Frozen v1 probe re-scored on the v2 labels, same primary-gate rows.
    v1 = load_predictions(Path(args.v1_predictions))
    common = [i for i in primary_ids if i in v1]
    v1_probs = np.asarray([v1[i]["probs"] for i in common], dtype=float)
    v1_ans = answers_at(v1_probs, 0.5)
    v1_metrics = summarize_binary(
        [labels[i] for i in common], v1_ans, v1_probs, ["非涉诈", "涉诈"])
    new_ans = [preds["probe"][i]["answer"] for i in common]
    paired_bal = paired_bootstrap_diff(
        [labels[i] for i in common], new_ans, v1_ans, options, metric="macro_tpr")
    paired_acc = paired_bootstrap_diff(
        [labels[i] for i in common], new_ans, v1_ans, options, metric="accuracy")
    mcnemar = mcnemar_exact(new_ans, v1_ans, [labels[i] for i in common])
    results["v1_probe_primary_gate"] = {
        "n": len(common),
        "metrics": v1_metrics,
        "paired_v2_minus_v1_macro_tpr": paired_bal,
        "paired_v2_minus_v1_accuracy": paired_acc,
        "mcnemar": mcnemar,
    }

    if "probe" not in results["gate_primary"]:
        raise SystemExit("primary gate has no rows")
    pm = results["gate_primary"]["probe"]
    criteria = [
        ("balanced accuracy ≥ 0.70", pm["balanced_accuracy"] >= 0.70,
         f(pm["balanced_accuracy"])),
        ("涉诈 recall ≥ 0.95", pm["fraud_recall"] >= 0.95,
         f(pm["fraud_recall"])),
        ("非涉诈 recall ≥ 0.50", pm["normal_recall_tnr"] >= 0.50,
         f(pm["normal_recall_tnr"])),
        ("ECE ≤ 0.10", pm["ece_10bin"] <= 0.10, f(pm["ece_10bin"])),
        ("相对 v1 probe（诊断基线）的 bal 提升 ≥ +5pp",
         paired_bal["point"] >= 0.05, f"{paired_bal['point']:+.3f} {ci(paired_bal['ci95'])}"),
        ("配对 bootstrap CI 下界 > 0", paired_bal["ci95"][0] > 0,
         ci(paired_bal["ci95"])),
    ]
    gate_pass_absolute = all(ok for _, ok, _ in criteria[:4])
    gate_pass = all(ok for _, ok, _ in criteria)
    results["gate_pass_absolute"] = gate_pass_absolute
    results["gate_pass"] = gate_pass
    results["criteria"] = [
        {"criterion": c, "pass": bool(ok), "value": v} for c, ok, v in criteria
    ]

    # Secondary verdicts for the FPR operating points that were frozen on dev
    # before gate.  These do not change the primary argmax verdict.
    op_points = {}
    for name in ("probe@devR95", "probe@fpr5", "probe@fpr10"):
        if name not in results["gate_primary"]:
            continue
        mm = results["gate_primary"][name]
        checks = [
            ("balanced accuracy ≥ 0.70", mm["balanced_accuracy"] >= 0.70,
             f(mm["balanced_accuracy"])),
            ("涉诈 recall ≥ 0.95", mm["fraud_recall"] >= 0.95,
             f(mm["fraud_recall"])),
            ("非涉诈 recall ≥ 0.50", mm["normal_recall_tnr"] >= 0.50,
             f(mm["normal_recall_tnr"])),
            ("ECE ≤ 0.10", mm["ece_10bin"] <= 0.10, f(mm["ece_10bin"])),
        ]
        op_points[name] = {
            "threshold": thresholds[name],
            "metrics": {k: mm[k] for k in (
                "n", "balanced_accuracy", "fraud_recall", "normal_recall_tnr",
                "fraud_precision", "ece_10bin", "confusion")},
            "criteria": [
                {"criterion": c, "pass": bool(ok), "value": v}
                for c, ok, v in checks
            ],
            "pass_absolute": all(ok for _, ok, _ in checks),
        }
    results["operating_points"] = op_points

    # Row-level sampling uncertainty for the pre-registered recall criterion,
    # plus the realized FPR of every dev-frozen threshold on each gate subset.
    dev_rows = [preds["probe"][i] for i in dev_ids]
    dev_neg = [r for r in dev_rows if r["label"] != "涉诈"]
    uncertainty = {}
    for name in ("probe", "probe@devR95", "probe@fpr5", "probe@fpr10"):
        if name not in results["gate_primary"]:
            continue
        mm = results["gate_primary"][name]
        n_pos = int(mm["support"].get("涉诈", 0))
        tp = int(mm["confusion"]["tp"])
        fn = int(mm["confusion"]["fn"])
        dev_fp = sum(
            1 for r in dev_neg if float(r["probs"][1]) >= thresholds[name])
        uncertainty[name] = {
            "threshold": thresholds[name],
            "n_pos": n_pos,
            "tp": tp,
            "fn": fn,
            "max_fn_allowed_at_0.95": int(math.floor(0.05 * n_pos + 1e-9)),
            "fraud_recall_ci95": mm["fraud_recall_ci95"],
            "normal_recall_ci95": mm["normal_recall_ci95"],
            "p_value_recall_ge_0.95": binom_le(tp, n_pos, 0.95),
            "dev_fpr": dev_fp / len(dev_neg) if dev_neg else None,
            "gate_fpr": {
                "primary": fpr_of(results["gate_primary"][name]),
                "cq_normal": fpr_of(results["gate_normal_cohort"][name])
                if name in results["gate_normal_cohort"] else None,
                "full": fpr_of(results["gate_full"][name])
                if name in results["gate_full"] else None,
            },
        }
    results["primary_gate_uncertainty"] = uncertainty

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- Markdown report -------------------------------------------------
    L: list[str] = []
    L.append("# v2（4.2 涉诈）· 线性 probe 复评")
    L.append("")
    L.append(f"- 模型：`Qwen/Qwen3.5-4B`，4 层隐藏状态（8/16/24/32），最后一 token。")
    L.append(f"- 数据：`eval_v2/goldens.jsonl`（{len(ids)} 条）；"
             f"cal/dev/gate = {sum(splits[i] == 'cal' for i in ids)}/"
             f"{sum(splits[i] == 'dev' for i in ids)}/{len(gate_ids)}。")
    L.append(f"- 主门禁：`split=gate` 且 `cohort=human_0729`（{len(primary_ids)} 条）；"
             f"另有 `human_cq_normal` gate {len(normal_ids)} 条用于正常类 recall。")
    L.append("- 协议：层/L2/类权重只在 cal 上拟合、dev 上选择；gate 只在冻结后评分一次。"
             "v2 任务与 rubric 见 `eval/task_v2.py`。")
    L.append("")
    L.append("## 1. 结论")
    L.append("")
    L.append(f"- 主门禁（argmax）：balanced accuracy **{f(pm['balanced_accuracy'])}**，"
             f"涉诈 recall **{f(pm['fraud_recall'])}**，"
             f"非涉诈 recall **{f(pm['normal_recall_tnr'])}**，"
             f"ECE **{f(pm['ece_10bin'])}**。")
    L.append(f"- 绝对门禁（前 4 项）：**{'通过' if gate_pass_absolute else '未通过'}**；"
             f"含「相对 v1 probe」参考条件的完整门禁：**{'通过' if gate_pass else '未通过'}**。")
    for c, ok, v in criteria:
        L.append(f"  - {'✅' if ok else '❌'} {c}：{v}")
    if op_points:
        L.append("")
        L.append("**操作点敏感性（阈值均在 dev 上冻结，未看 gate）**：")
        L.append("")
        op_rows = []
        for name, p in op_points.items():
            mm = p["metrics"]
            op_rows.append([
                CONFIG_LABELS[name], f(p["threshold"], 4), f(mm["balanced_accuracy"]),
                f(mm["fraud_recall"]), f(mm["normal_recall_tnr"]),
                f(mm["ece_10bin"]), "通过" if p["pass_absolute"] else "未通过",
            ])
        L.append(table(
            ["config", "threshold", "bal", "涉诈 R", "非涉诈 R", "ECE",
             "四项绝对门禁"], op_rows))
        L.append("")
        L.append("> argmax（0.5）是主决策。`probe@dev-R95` 的阈值由 dev 上「涉诈 recall ≥ 0.95 "
                 "前提下最大化非涉诈 recall」冻结；`@FPR5`/`@FPR10` 由 dev 正常类经验 FPR 冻结。"
                 "三者都是 gate 之前可复算的 dev-only 操作点，但事后在 gate 上比较它们仍引入选择"
                 "偏差；采用哪个点取决于误报成本，不能用 gate 结果反向选点。")
        passing = [
            (name, fpr_of(results["gate_primary"][name]),
             results["gate_primary"][name]["fraud_recall"])
            for name in ("probe@devR95", "probe@fpr5", "probe@fpr10")
            if name in results["gate_primary"]
            and results["gate_primary"][name]["fraud_recall"] >= 0.95
        ]
        if passing:
            L.append("")
            L.append("- 过 recall 门禁的 dev 冻结点在主门禁上的实际误报：" +
                     "、".join(f"{CONFIG_LABELS[n]} {fpr*100:.1f}%" for n, fpr, _ in passing) +
                     f"；argmax 误报最低（{fpr_of(results['gate_primary']['probe'])*100:.1f}%）"
                     "但 recall 未达标。")
    L.append("")
    L.append(f"- 对比参考：冻结 v1 probe 在同一批 {len(common)} 条 gate 行上按 v2 标签重评分，"
             f"balanced accuracy **{f(v1_metrics['balanced_accuracy'])}**，"
             f"涉诈 recall **{f(v1_metrics['fraud_recall'])}**，"
             f"非涉诈 recall **{f(v1_metrics['normal_recall_tnr'])}**。")
    L.append("")
    L.append("## 2. Gate 结果")
    L.append("")
    for subset_name, subset in (
        ("全量 gate", results["gate_full"]),
        ("主门禁 human_0729", results["gate_primary"]),
        ("human_cq_normal gate", results["gate_normal_cohort"]),
    ):
        if not subset:
            continue
        L.append(f"### {subset_name}")
        L.append("")
        rows = []
        for name in configs:
            m = subset[name]
            rows.append([
                CONFIG_LABELS[name], str(m["n"]),
                f(m["accuracy"]), f(m["balanced_accuracy"]),
                f(m["fraud_recall"]), f(m["normal_recall_tnr"]),
                f(m["fraud_precision"]), f(m["pr_auc"]), f(m["ece_10bin"]),
                f"{m['confusion']['tp']}/{m['confusion']['fn']}/"
                f"{m['confusion']['fp']}/{m['confusion']['tn']}",
            ])
        L.append(table(
            ["config", "n", "acc", "bal", "涉诈 R", "非涉诈 R", "涉诈 P",
             "PR-AUC", "ECE", "tp/fn/fp/tn"], rows))
        L.append("")
    if results.get("primary_gate_uncertainty"):
        L.append("### 主门禁采样不确定性（涉诈类 "
                 f"{results['primary_gate_uncertainty']['probe']['n_pos']} 条）")
        L.append("")
        rows = []
        for name, u in results["primary_gate_uncertainty"].items():
            rows.append([
                CONFIG_LABELS[name], f(u["threshold"], 4), str(u["fn"]),
                str(u["max_fn_allowed_at_0.95"]),
                f"{u['fraud_recall_ci95'][0]:.3f}–{u['fraud_recall_ci95'][1]:.3f}",
                f"{u['normal_recall_ci95'][0]:.3f}–{u['normal_recall_ci95'][1]:.3f}",
                f"{u['p_value_recall_ge_0.95']:.3f}",
            ])
        L.append(table(
            ["config", "threshold", "FN", "允许 FN（R≥0.95）",
             "涉诈 R [95% CI]", "非涉诈 R [95% CI]",
             "P(观测≤tp \\| 真实 R=0.95)"], rows))
        L.append("")
        L.append("> Wilson 区间与二项尾概率按行独立近似；数据存在近重复簇，区间偏窄，"
                 "只用于判断「点估计差 1.6pp」是否超出采样噪声。门禁仍按预注册的点估计判定。")
        L.append("")
        L.append("### 操作点的实际误报率（dev 选点 vs gate 实测）")
        L.append("")
        rows = []
        for name, u in results["primary_gate_uncertainty"].items():
            gf = u["gate_fpr"]
            rows.append([
                CONFIG_LABELS[name], f(u["dev_fpr"]), f(gf["primary"]),
                f(gf["cq_normal"]), f(gf["full"]),
            ])
        L.append(table(
            ["config", "dev FPR（选点用）", "主门禁负类 FPR",
             "human_cq_normal FPR", "全量 gate FPR"], rows))
        L.append("")
        L.append("> `@FPR5`/`@FPR10` 的名字只表示 dev 上的选点目标，不是线上 FPR 保证。"
                 "dev 与 gate 的负类构成/批次不同：主门禁负类中 167/230 是仅 4.1 违规，"
                 "`human_cq_normal` 是另一批纯合规数据，两者上 FPR5 都达到约 23%。"
                 "阈值若要上线，必须用目标流量重新校准误报率。")
        L.append("")
    if "probe" in results["gate_full"]:
        rows = []
        for cohort, m in results["gate_full"]["probe"]["by_cohort"].items():
            rows.append([cohort, str(m["n"]), f(m["balanced_accuracy"]),
                         f(m["fraud_recall"]), f(m["normal_recall_tnr"]),
                         str(m["support"].get("涉诈", 0))])
        L.append("### 按 cohort（probe argmax，全量 gate）")
        L.append("")
        L.append(table(["cohort", "n", "bal", "涉诈 R", "非涉诈 R", "涉诈数"], rows))
        L.append("")
        L.append("> `human_cq_normal` 的涉诈数为 0，该子集的涉诈 recall 不可估计，只看非涉诈 recall。")
        L.append("")
    L.append("## 3. Dev 选择（gate 未参与）")
    L.append("")
    sel = grid["selected"]
    L.append(f"- 选中：layer **{sel['layer']}**，L2 **{sel['l2']:g}**，"
             f"class weight **{sel['class_weight']}**，"
             f"thresholds `{sel['thresholds']}`。")
    L.append(f"- dev：balanced accuracy **{f(grid['dev_metrics']['balanced_accuracy'])}**，"
             f"涉诈 recall **{f(grid['dev_metrics']['fraud_recall'])}**，"
             f"非涉诈 recall **{f(grid['dev_metrics']['normal_recall_tnr'])}**，"
             f"ECE **{f(grid['dev_metrics']['ece_10bin'])}**。")
    rf = grid.get("selected_riskfirst")
    if rf:
        L.append(f"- 风险优先备选：layer **{rf['layer']}**，L2 **{rf['l2']:g}**，"
                 f"class weight **{rf['class_weight']}**；dev 涉诈 recall "
                 f"**{f(rf['dev_metrics']['fraud_recall'])}**，非涉诈 recall "
                 f"**{f(rf['dev_metrics']['normal_recall_tnr'])}**。")
    L.append("")
    fr = results["dev_frontier"]
    if fr:
        b = fr["max_balanced_accuracy"]
        h = fr["fraud_recall_ge_0.95"]["max_normal_recall"]
        L.append("### Dev 操作点前沿（post-hoc，不参与选择）")
        L.append("")
        L.append(f"- 最大 balanced accuracy：threshold **{b['threshold']:.3f}**，"
                 f"涉诈 recall **{b['fraud_recall']:.3f}**，"
                 f"非涉诈 recall **{b['normal_recall']:.3f}**，bal **{b['balanced_accuracy']:.3f}**。")
        if h:
            L.append(f"- 约束涉诈 recall ≥ 0.95 时：threshold **{h['threshold']:.3f}**，"
                     f"非涉诈 recall **{h['normal_recall']:.3f}**，"
                     f"bal **{h['balanced_accuracy']:.3f}**。")
        else:
            L.append("- dev 上不存在涉诈 recall ≥ 0.95 的操作点。")
    L.append("")
    L.append("## 4. 与 v1 probe 的配对比较（human_0729 gate）")
    L.append("")
    L.append(table(
        ["比较", "Δbalanced accuracy [95% CI]", "Δaccuracy [95% CI]", "McNemar"],
        [["v2 probe − v1 probe",
          f"{paired_bal['point']:+.3f} {ci(paired_bal['ci95'])}",
          f"{paired_acc['point']:+.3f} {ci(paired_acc['ci95'])}",
          f"b={mcnemar['n01']}, c={mcnemar['n10']}, p={mcnemar['p_value']:.3g}"]]))
    L.append("")
    L.append("> v1 probe 是在旧粗标签 + 旧 prompt 上训练的冻结模型，这里只把它的概率按 "
             "`is_fraud` 重评分。v2 probe 同时换了训练标签和 rubric，"
             "因此差值不能全部归因于标签重定义；这是诊断参考，不是预注册 baseline。")
    L.append("")
    L.append("## 5. 错误分析（probe argmax，全量 gate）")
    L.append("")
    if "probe" in results["gate_full"]:
        m = results["gate_full"]["probe"]
        L.append(f"- 错误数：{len(m['errors'])} / {m['n']}。")
        L.append(f"- 假阳性（非涉诈→涉诈）按场景：" +
                 ("、".join(f"{k} {v}" for k, v in list(m["fp_by_scene"].items())[:10]) or "无"))
        L.append(f"- 假阴性（涉诈→非涉诈）按场景：" +
                 ("、".join(f"{k} {v}" for k, v in list(m["fn_by_scene"].items())[:10]) or "无"))
        L.append("")
        L.append(table(
            ["id", "真值", "预测", "P(涉诈)", "cohort", "场景", "片段"],
            [[e["id"], e["label"], e["answer"], f"{e['p_fraud']:.3f}",
              e["cohort"], e["scene"] or "—", e["snippet"]]
             for e in m["errors"][:12]]))
        L.append("")
    L.append("## 6. 限制")
    L.append("")
    L.append("- 主门禁的 normal 类只有 human_0729 gate 的 "
             f"{pm['support'].get('非涉诈', 0)} 条 + human_cq_normal 的 {len(normal_ids)} 条；"
             "前者用于门禁，后者只作补充正常 recall，不能替代时间外推。")
    L.append("- `agent_1022` 的 6,353 条是 10.22 人批复核，107 条保留 spec 裁定；"
             "两套口径在 107 条上仍有分歧（见 conflicts CSV）。")
    L.append("- 本报告是 transcript-only 结论；没有使用业务归属/资质等通话外字段。")
    L.append("")
    L.append("## 7. 复现")
    L.append("")
    L.append("```bash")
    L.append(".venv312/bin/python eval/features.py --task v2 "
             "--goldens eval_v2/goldens.jsonl --out runs/v2/features_binary.npz")
    L.append(".venv312/bin/python eval/probe.py --features runs/v2/features_binary.npz "
             "--out-dir runs/v2/probe")
    L.append(".venv312/bin/python eval/report_v2.py")
    L.append("```")
    L.append("")

    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out_dir / 'gate_results.json'} and {out_dir / 'report.md'}")
    print(f"primary gate pass: {gate_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
