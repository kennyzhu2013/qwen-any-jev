#!/usr/bin/env python
"""Three-way triage operating-point analysis for the frozen probe.

Policy (per row, score s = P(涉诈)):

    s <= lo   -> auto 正常
    s >= hi   -> auto 涉诈
    otherwise -> 人工复核

Thresholds are selected on dev under explicit constraints; gate is evaluated
once as a post-hoc diagnostic.  This does NOT change the pre-registered M0.5
gate verdict (which is based on the frozen argmax probe).

Outputs runs/m0.5/triage.json with selected policies and the dev frontier.

Usage:
  python eval/triage.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_predictions(path: str | Path, split: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("split") == split:
            rows.append(rec)
    if not rows:
        raise SystemExit(f"no {split} rows in {path}")
    y = np.asarray([1 if r["label"] == "涉诈" else 0 for r in rows], dtype=int)
    s = np.asarray([float(r["probs"][1]) for r in rows], dtype=float)
    return rows, y, s


def _count_le(sorted_arr: np.ndarray, t: float) -> int:
    return int(np.searchsorted(sorted_arr, t, side="right"))


def metrics(y: np.ndarray, s: np.ndarray, lo: float, hi: float) -> dict:
    """Counts/rates for the three-way policy."""
    n = len(y)
    npos = int(y.sum())
    nneg = n - npos
    sn = np.sort(s[y == 0])
    sf = np.sort(s[y == 1])
    # auto 正常: s <= lo
    an_n = _count_le(sn, lo)
    an_f = _count_le(sf, lo)
    # auto 涉诈: s >= hi
    af_n = nneg - int(np.searchsorted(sn, hi, side="left"))
    af_f = npos - int(np.searchsorted(sf, hi, side="left"))
    auto_n = an_n + an_f
    auto_f = af_n + af_f
    review = n - auto_n - auto_f
    auto_correct = an_n + af_f
    return {
        "n": n,
        "support": {"正常": nneg, "涉诈": npos},
        "lo": float(lo),
        "hi": float(hi),
        "auto_normal": int(auto_n),
        "auto_fraud": int(auto_f),
        "review": int(review),
        "review_rate": review / n,
        "auto_coverage": 1.0 - review / n,
        "fraud_escape": an_f / npos if npos else None,
        "fraud_auto_catch": af_f / npos if npos else None,
        "normal_auto_pass": an_n / nneg if nneg else None,
        "normal_false_reject": af_n / nneg if nneg else None,
        "auto_accuracy": auto_correct / (auto_n + auto_f) if (auto_n + auto_f) else None,
        "review_fraud_share": (npos - an_f - af_f) / review if review else None,
    }


def candidate_pairs(s: np.ndarray) -> list[tuple[float, float]]:
    vals = np.unique(np.concatenate([[-1.0, 2.0], s])).astype(float)
    pairs = []
    for lo in vals:
        for hi in vals:
            if hi > lo:
                pairs.append((float(lo), float(hi)))
    return pairs


def evaluate_pairs(y, s, pairs) -> list[dict]:
    return [metrics(y, s, lo, hi) for lo, hi in pairs]


def select(m: dict, policy: str) -> dict | None:
    """Pick one dev point for a named policy."""
    if policy == "zero_escape":
        ok = [x for x in m if x["fraud_escape"] <= 0.005
              and x["normal_false_reject"] <= 0.10]
        key = lambda x: (x["auto_coverage"], x["normal_auto_pass"], -x["fraud_escape"])
    elif policy == "balanced":
        ok = [x for x in m if x["fraud_escape"] <= 0.02
              and x["normal_false_reject"] <= 0.10]
        key = lambda x: (x["auto_coverage"], x["normal_auto_pass"], -x["fraud_escape"])
    elif policy.startswith("budget"):
        budget = float(policy[len("budget"):]) / 100.0
        ok = [x for x in m if x["review_rate"] <= budget]
        key = lambda x: (-x["fraud_escape"], x["normal_auto_pass"],
                         -x["normal_false_reject"], x["auto_coverage"])
    else:
        raise ValueError(policy)
    if not ok:
        return None
    return max(ok, key=key)


def frontier(m: dict, budgets: list[float]) -> list[dict]:
    out = []
    for b in budgets:
        ok = [x for x in m if x["review_rate"] <= b]
        if not ok:
            continue
        best = min(ok, key=lambda x: (x["fraud_escape"],
                                      -x["normal_auto_pass"],
                                      x["normal_false_reject"]))
        out.append({"review_budget": b, **best})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default="runs/m0.5/probe/predictions.jsonl")
    ap.add_argument("--dev", default="dev")
    ap.add_argument("--gate", default="gate")
    ap.add_argument("--out", default="runs/m0.5/triage.json")
    args = ap.parse_args()

    _, y_dev, s_dev = load_predictions(args.predictions, args.dev)
    _, y_gate, s_gate = load_predictions(args.predictions, args.gate)

    pairs = candidate_pairs(s_dev)
    dev_m = evaluate_pairs(y_dev, s_dev, pairs)
    print(f"dev pairs={len(dev_m)}", flush=True)

    names = ["zero_escape", "balanced", "budget20", "budget30", "budget40", "budget50"]
    payload = {"dev_n": len(y_dev), "gate_n": len(y_gate),
               "policies": {}, "frontier_dev": []}
    print(f"{'policy':>12} {'lo':>7} {'hi':>7} {'split':>5} {'review':>7} "
          f"{'escape':>7} {'catch':>7} {'npass':>7} {'nfr':>7} {'autoacc':>8}")
    for name in names:
        sel = select(dev_m, name)
        if sel is None:
            print(f"{name:>12}  (no feasible point)")
            continue
        gate_m = metrics(y_gate, s_gate, sel["lo"], sel["hi"])
        payload["policies"][name] = {"lo": sel["lo"], "hi": sel["hi"],
                                     "dev": sel, "gate": gate_m}
        for split, m in (("dev", sel), ("gate", gate_m)):
            print(f"{name:>12} {sel['lo']:>7.4f} {sel['hi']:>7.4f} {split:>5} "
                  f"{m['review_rate']:>7.3f} {m['fraud_escape']:>7.3f} "
                  f"{m['fraud_auto_catch']:>7.3f} {m['normal_auto_pass']:>7.3f} "
                  f"{m['normal_false_reject']:>7.3f} {m['auto_accuracy']:>8.3f}", flush=True)

    payload["frontier_dev"] = frontier(dev_m, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    print("\ndev frontier (min fraud_escape within review budget):")
    for f in payload["frontier_dev"]:
        print(f"  budget={f['review_budget']:.1f} lo={f['lo']:.4f} hi={f['hi']:.4f} "
              f"review={f['review_rate']:.3f} escape={f['fraud_escape']:.3f} "
              f"catch={f['fraud_auto_catch']:.3f} npass={f['normal_auto_pass']:.3f} "
              f"nfr={f['normal_false_reject']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
