#!/usr/bin/env python
"""Run AnyJev decision configs and assemble M0.5 gate tables.

Protocol:
  run --splits cal,dev      # collect calibration and development decisions
  select                    # choose prior strength / temperature on dev
  run --splits gate --selection ...   # freeze choices, score gate once

AnyJev 0.3.0 provides raw and L0.  Fixed-prior L0, temperature scaling and the
linear probe are self-implemented here or in eval/probe.py, matching Plan.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import summarize_binary  # noqa: E402
from eval.task import OPTIONS, QUESTION, SYSTEM  # noqa: E402

SPLIT_ORDER = ["cal", "dev", "gate"]
BASE_CONFIGS = ["raw", "l0-noprior", "l0-batch", "l0-cf"]
ALL_CONFIGS = BASE_CONFIGS + ["l0-adaptive"]


# --------------------------------------------------------------------------- #
# data / deciders
# --------------------------------------------------------------------------- #

def load_goldens(path: Path, splits: list[str]) -> dict[str, list[dict]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    out = {s: [r for r in rows if r["split"] == s] for s in splits}
    for s, sub in out.items():
        if not sub:
            raise SystemExit(f"no rows for split {s!r}")
        if {r["question"] for r in sub} != {QUESTION}:
            raise SystemExit(f"split {s}: question mismatch")
        if {tuple(r["options"]) for r in sub} != {tuple(OPTIONS)}:
            raise SystemExit(f"split {s}: options mismatch")
    return out


def make_decider(backend, config: str, args):
    from anyjev import Decider

    common = dict(system=SYSTEM, canonical_order=True)
    if config == "raw":
        return Decider(backend, level="raw", prior="none", **common)
    if config == "l0-noprior":
        return Decider(backend, level="L0", prior="none",
                       max_permutations=args.max_perms, **common)
    if config == "l0-batch":
        return Decider(backend, level="L0", prior="batch",
                       min_prior_n=args.min_prior_n,
                       max_permutations=args.max_perms, **common)
    if config == "l0-cf":
        return Decider(backend, level="L0", prior="content_free",
                       max_permutations=args.max_perms, **common)
    if config == "l0-adaptive":
        return Decider(backend, level="L0", prior="none", adaptive_shifts=True,
                       adaptive_min_shifts=args.adaptive_min_shifts,
                       adaptive_target=args.adaptive_target,
                       max_permutations=args.max_perms, **common)
    raise ValueError(f"unknown config {config!r}")


def _diagnostic_scalars(diag: dict | None) -> dict:
    keys = ("shifts_used", "permutations", "order_flip_raw", "order_flip_l0",
            "prior_method", "prior_strength", "adaptive", "stop_threshold",
            "stop_stat", "answer_mass")
    out = {}
    for key in keys:
        if not diag or key not in diag:
            continue
        value = diag[key]
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:  # noqa: BLE001
                value = str(value)
        out[key] = value if isinstance(value, (bool, int, float, str)) or value is None else str(value)
    return out


def run_config(backend, rows: list[dict], question, config: str, args) -> dict:
    decider = make_decider(backend, config, args)
    states = [r["state"] for r in rows]
    started = time.perf_counter()
    decisions = decider.decide_batch(states, question)
    elapsed = time.perf_counter() - started
    records = []
    for row, decision in zip(rows, decisions):
        probs = np.asarray(decision.probs, dtype=float)
        records.append({
            "id": row["id"],
            "label": row["label"],
            "category": row.get("category"),
            "answer": decision.answer,
            "confidence": float(decision.confidence),
            "probs": [float(x) for x in probs],
            "level": decision.level,
            "diagnostics": _diagnostic_scalars(decision.diagnostics),
        })
    return {
        "config": config,
        "elapsed_s": elapsed,
        "per_row_s": elapsed / max(len(rows), 1),
        "decider_stats": dict(decider.stats),
        "rows": records,
    }


# --------------------------------------------------------------------------- #
# derived decisions (self-implemented L1 / fixed-prior L0)
# --------------------------------------------------------------------------- #

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(labels: list[str], probs: np.ndarray,
                    grid: np.ndarray | None = None) -> float:
    """Minimize NLL on calibration rows; argmax is unchanged."""
    if grid is None:
        grid = np.round(np.arange(0.50, 5.01, 0.05), 2)
    index = {o: i for i, o in enumerate(OPTIONS)}
    y = np.asarray([index[v] for v in labels], dtype=int)
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        p = softmax(logp / t)
        nll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def apply_temperature(rows: list[dict], temperature: float) -> list[dict]:
    out = []
    for r in rows:
        p = softmax(np.log(np.clip(np.asarray(r["probs"]), 1e-12, 1.0)) / temperature)
        out.append({**r, "probs": p.tolist(),
                    "answer": OPTIONS[int(p.argmax())],
                    "confidence": float(p.max())})
    return out


def apply_fixed_prior(rows: list[dict], prior: np.ndarray, strength: float) -> list[dict]:
    """L0's correction, but with a prior estimated on the calibration split."""
    out = []
    for r in rows:
        p = np.asarray(r["probs"], dtype=float) / np.power(np.clip(prior, 1e-12, 1.0), strength)
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        out.append({**r, "probs": p.tolist(),
                    "answer": OPTIONS[int(p.argmax())],
                    "confidence": float(p.max())})
    return out


def summarize(rows: list[dict]) -> dict:
    return summarize_binary(
        [r["label"] for r in rows],
        [r["answer"] for r in rows],
        [r["probs"] for r in rows],
        OPTIONS,
    )


def prior_from_cal(cal: list[dict]) -> np.ndarray:
    labels = [r["label"] for r in cal]
    return np.asarray([labels.count(o) for o in OPTIONS], dtype=float) / len(labels)


def select_prior_strength(cal: list[dict], dev: list[dict],
                          strengths: list[float]) -> dict:
    prior = prior_from_cal(cal)
    table = []
    best = None
    for s in strengths:
        rows = apply_fixed_prior(dev, prior, s)
        m = summarize(rows)
        row = {"strength": s, "balanced_accuracy": m["balanced_accuracy"],
               "accuracy": m["accuracy"], "fraud_recall": m["fraud_recall"],
               "normal_recall": m["normal_recall_tnr"], "ece": m["ece_10bin"],
               "nll": m["nll"]}
        table.append(row)
        key = (row["balanced_accuracy"], -row["ece"], row["fraud_recall"])
        if best is None or key > best[0]:
            best = (key, s)
    return {"prior": prior.tolist(), "strength": best[1], "table": table}


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_run(args) -> int:
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in ALL_CONFIGS:
            raise SystemExit(f"unknown config {c!r}")
    goldens = load_goldens(Path(args.goldens), splits)
    from anyjev import Question
    from anyjev.backends.hf import HFBackend

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    print(f"loading {args.model} ...", flush=True)
    backend = HFBackend(args.model, device=args.device, dtype=args.dtype,
                        batch_size=args.batch_size)
    question = Question.choice(QUESTION, OPTIONS, name="risk")

    # One raw warmup row; separate Decider so batch-prior state cannot leak.
    warm = make_decider(backend, "raw", args)
    warm.decide_batch([next(iter(goldens.values()))[0]["state"]], question)
    del warm

    payload = {"meta": {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "splits": splits,
        "configs": configs,
        "question": QUESTION,
        "system": SYSTEM,
        "options": OPTIONS,
        "prior_strength": args.prior_strength,
        "temperature_grid": "0.50..5.00 step 0.05, fit on cal",
        "notes": [
            "L0-batch estimates the prior on the evaluation split itself; it is a reference, not a deployment candidate.",
            "L0-noprior is permutation marginalization only; fixed-prior and temperature are self-implemented.",
            "Timing is one pass after a one-row raw warmup; it excludes model load and is not a serving benchmark.",
        ],
    }, "splits": {}}

    for split in splits:
        rows = goldens[split]
        print(f"\n=== split {split}: {len(rows)} rows ===")
        payload["splits"][split] = {}
        for config in configs:
            print(f"[run] {split}/{config}", flush=True)
            result = run_config(backend, rows, question, config, args)
            result["metrics"] = summarize(result["rows"])
            payload["splits"][split][config] = result
            m = result["metrics"]
            print(f"      acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} "
                  f"fraud_recall={m['fraud_recall']:.4f} "
                  f"normal_recall={m['normal_recall_tnr']:.4f} "
                  f"ece={m['ece_10bin']:.4f} s/row={result['per_row_s']:.4f}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {out_dir}/results.json")
    return 0


def cmd_select(args) -> int:
    payload = json.loads(Path(args.dev_results).read_text(encoding="utf-8"))
    if "cal" not in payload["splits"] or "dev" not in payload["splits"]:
        raise SystemExit("select needs cal and dev in --dev-results")
    cal, dev = payload["splits"]["cal"], payload["splits"]["dev"]

    selection = {"configs": {}, "prior": {}, "temperature": {}, "probe": {}}
    for config in ("raw", "l0-noprior"):
        if config not in cal or config not in dev:
            continue
        cal_probs = np.asarray([r["probs"] for r in cal[config]["rows"]])
        t = fit_temperature([r["label"] for r in cal[config]["rows"]], cal_probs)
        selection["temperature"][config] = t
        prior = select_prior_strength(
            cal[config]["rows"], dev[config]["rows"],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )
        selection["prior"][config] = prior
        print(f"{config}: temperature={t:.2f} prior_strength={prior['strength']} "
              f"cal_prior={prior['prior']}")

    probe_dir = Path(args.probe_dir) if args.probe_dir else None
    if probe_dir and (probe_dir / "probe_grid.json").exists():
        grid = json.loads((probe_dir / "probe_grid.json").read_text(encoding="utf-8"))
        selection["probe"] = {
            "dir": str(probe_dir),
            "selected": grid["selected"],
            "dev_metrics": grid["dev_metrics"],
        }
        print(f"probe: layer={grid['selected']['layer']} "
              f"l2={grid['selected']['l2']} cw={grid['selected']['class_weight']} "
              f"thresholds={grid['selected']['thresholds']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


def load_probe_predictions(probe_dir: Path) -> dict[str, list[dict]]:
    path = probe_dir / "predictions.jsonl"
    out: dict[str, list[dict]] = {}
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        out.setdefault(rec["split"], []).append(rec)
    return out


def probe_rows(rows: list[dict], predictions: list[dict],
               threshold: float | None) -> list[dict]:
    by_id = {r["id"]: r for r in predictions}
    out = []
    for r in rows:
        p = by_id[r["id"]]
        probs = np.asarray(p["probs"], dtype=float)
        if threshold is None:
            answer = OPTIONS[int(probs.argmax())]
        else:
            answer = "涉诈" if probs[1] >= threshold else "正常"
        out.append({
            "id": r["id"], "label": r["label"], "category": r.get("category"),
            "answer": answer, "probs": probs.tolist(),
            "confidence": float(probs.max()),
        })
    return out


def cmd_gate(args) -> int:
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    final_out = Path(args.out_dir)
    args.configs = "raw,l0-noprior,l0-batch,l0-cf"
    args.splits = "gate"
    # Re-use cmd_run to collect base decisions, but write to a temp payload first.
    tmp_out = final_out / "_base"
    args.out_dir = str(tmp_out)
    cmd_run(args)
    payload = json.loads((tmp_out / "results.json").read_text(encoding="utf-8"))
    gate = payload["splits"]["gate"]

    derived = {}
    for config in ("raw", "l0-noprior"):
        if config not in gate:
            continue
        rows = gate[config]["rows"]
        t = selection.get("temperature", {}).get(config)
        if t:
            derived[f"{config}-temp"] = apply_temperature(rows, float(t))
        p = selection.get("prior", {}).get(config)
        if p:
            prior = np.asarray(p["prior"], dtype=float)
            derived[f"{config}-prior"] = apply_fixed_prior(rows, prior, float(p["strength"]))

    probe_dir = Path(selection["probe"]["dir"]) if selection.get("probe", {}).get("dir") else None
    if probe_dir:
        preds = load_probe_predictions(probe_dir).get("gate", [])
        if preds:
            derived["probe"] = probe_rows(gate["raw"]["rows"], preds, None)
            for name, key in (("probe@fpr5", "fpr_5"), ("probe@fpr10", "fpr_10")):
                t = selection["probe"]["selected"]["thresholds"].get(key)
                if t is not None:
                    derived[name] = probe_rows(gate["raw"]["rows"], preds, float(t))
        rf_path = probe_dir / "predictions_riskfirst.jsonl"
        if rf_path.exists():
            # load_probe_predictions only reads predictions.jsonl; read the
            # risk-first file directly.
            rf_preds = []
            for line in rf_path.open(encoding="utf-8"):
                rec = json.loads(line)
                if rec["split"] == "gate":
                    rf_preds.append(rec)
            if rf_preds:
                derived["probe-riskfirst"] = probe_rows(gate["raw"]["rows"], rf_preds, None)

    results = {}
    for config, rows in {**{k: v["rows"] for k, v in gate.items()}, **derived}.items():
        results[config] = {"rows": rows, "metrics": summarize(rows)}
        m = results[config]["metrics"]
        print(f"{config:>14}: acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} "
              f"fraud_recall={m['fraud_recall']:.4f} "
              f"normal_recall={m['normal_recall_tnr']:.4f} ece={m['ece_10bin']:.4f}")

    payload["derived"] = results
    payload["selection"] = selection
    final_out.mkdir(parents=True, exist_ok=True)
    (final_out / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {final_out}/results.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--goldens", default="eval/goldens.jsonl")
    run.add_argument("--splits", default="cal,dev")
    run.add_argument("--configs", default="raw,l0-noprior,l0-batch,l0-cf")
    run.add_argument("--out-dir", default="runs/m0.5/dev")
    run.add_argument("--model", default="Qwen/Qwen3.5-4B")
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    run.add_argument("--max-perms", type=int, default=None)
    run.add_argument("--min-prior-n", type=int, default=8)
    run.add_argument("--adaptive-min-shifts", type=int, default=2)
    run.add_argument("--adaptive-target", type=float, default=0.01)
    run.add_argument("--prior-strength", type=float, default=None)
    run.set_defaults(func=cmd_run)

    sel = sub.add_parser("select")
    sel.add_argument("--dev-results", default="runs/m0.5/dev/results.json")
    sel.add_argument("--probe-dir", default="runs/m0.5/probe")
    sel.add_argument("--out", default="runs/m0.5/selection.json")
    sel.set_defaults(func=cmd_select)

    gate = sub.add_parser("gate")
    gate.add_argument("--goldens", default="eval/goldens.jsonl")
    gate.add_argument("--selection", default="runs/m0.5/selection.json")
    gate.add_argument("--out-dir", default="runs/m0.5/gate")
    gate.add_argument("--model", default="Qwen/Qwen3.5-4B")
    gate.add_argument("--device", default="cuda")
    gate.add_argument("--dtype", default="bfloat16")
    gate.add_argument("--batch-size", type=int, default=4)
    gate.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    gate.add_argument("--max-perms", type=int, default=None)
    gate.add_argument("--min-prior-n", type=int, default=8)
    gate.add_argument("--adaptive-min-shifts", type=int, default=2)
    gate.add_argument("--adaptive-target", type=float, default=0.01)
    gate.add_argument("--prior-strength", type=float, default=None)
    gate.set_defaults(func=cmd_gate)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
