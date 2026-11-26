#!/usr/bin/env python
"""Train the self-implemented linear probe (the M0.5 off-ramp candidate).

The probe is a multinomial logistic head on cached last-position hidden
states.  It is deliberately small and closed-form-ish: layer, L2 strength and
class weighting are selected on `dev`; `gate` is never used for selection.

AnyJev 0.3.0 does not ship this head (the docs call it L2); this is the
self-implemented replacement described in Plan.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import summarize_binary, summarize_multiclass  # noqa: E402
from eval.task import CATEGORY_OPTIONS, OPTIONS  # noqa: E402


def standardize(train_x: np.ndarray, other_x: np.ndarray, clip: float = 10.0):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    z_train = np.clip((train_x - mean) / std, -clip, clip)
    z_other = np.clip((other_x - mean) / std, -clip, clip)
    return z_train, z_other, mean.squeeze(0), std.squeeze(0)


def fit_logistic(x: np.ndarray, y: np.ndarray, *, k: int, l2: float,
                 class_weight: str, max_iter: int = 300, seed: int = 20261103):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    model = torch.nn.Linear(x.shape[1], k)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    if class_weight == "balanced":
        counts = np.bincount(y, minlength=k).astype(np.float64)
        w = len(y) / (k * np.maximum(counts, 1.0))
        weight = torch.as_tensor(w, dtype=torch.float32)
    else:
        weight = None
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter, history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        logits = model(x_t)
        loss = F.cross_entropy(logits, y_t, weight=weight)
        loss = loss + 0.5 * l2 * (model.weight ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        probs = torch.softmax(model(x_t), dim=-1).cpu().numpy().astype(np.float64)
    return model, probs


def predict(model, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)


def accuracy_metrics(labels, probs, options, task: str) -> dict:
    answers = [options[int(i)] for i in probs.argmax(axis=1)]
    if task == "binary":
        return summarize_binary(labels, answers, probs, options)
    return summarize_multiclass(labels, answers, probs, options)


def threshold_for_fpr(normal_scores: np.ndarray, max_fpr: float) -> float:
    """Largest threshold whose empirical FPR on normal dev scores is <= max_fpr."""
    scores = np.sort(np.asarray(normal_scores, dtype=float))
    n = len(scores)
    if n == 0:
        return 0.5
    # Allow at most floor(max_fpr*n) false positives; threshold just above the
    # (n - allowed)-th largest normal score.
    allowed = int(np.floor(max_fpr * n))
    idx = n - allowed - 1
    if idx < 0:
        return 0.5
    if idx >= n - 1:
        return float(scores[-1] + 1e-9)
    return float((scores[idx] + scores[idx + 1]) / 2.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="runs/m0.5/features_binary.npz")
    ap.add_argument("--out-dir", default="runs/m0.5/probe")
    ap.add_argument("--task", choices=["binary", "category"], default="binary")
    ap.add_argument("--l2", default="1e-4,1e-3,1e-2,1e-1,1.0")
    ap.add_argument("--class-weight", default="none,balanced")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261103)
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    feats = data["feats"]
    ids = data["ids"].astype(str)
    labels = data["labels"].astype(str)
    splits = data["splits"].astype(str)
    layers = [int(x) for x in data["layers"]]
    options = list(data["options"]) if "options" in data else (
        OPTIONS if args.task == "binary" else CATEGORY_OPTIONS
    )
    if args.task == "binary":
        # v2 uses ['非涉诈', '涉诈']; any two-option negative/fraud task is
        # compatible as long as the positive option is '涉诈'.
        assert len(options) == 2 and options[1] == "涉诈", options
        k = 2
    else:
        assert options == CATEGORY_OPTIONS, options
        k = len(CATEGORY_OPTIONS)
    label_index = {o: i for i, o in enumerate(options)}
    y = np.asarray([label_index[v] for v in labels], dtype=np.int64)

    train_mask = splits == "cal"
    dev_mask = splits == "dev"
    gate_mask = splits == "gate"
    if not train_mask.any() or not dev_mask.any():
        raise SystemExit("need cal and dev splits in features")
    print(f"rows cal={train_mask.sum()} dev={dev_mask.sum()} gate={gate_mask.sum()} "
          f"layers={layers} options={options}", flush=True)

    grid = []
    best = None
    for layer in layers:
        x_all = feats[:, layers.index(layer), :]
        x_train_raw, x_dev_raw = x_all[train_mask], x_all[dev_mask]
        z_train, z_dev, _, _ = standardize(x_train_raw, x_dev_raw)
        y_train, y_dev = y[train_mask], y[dev_mask]
        labels_train = labels[train_mask].tolist()
        labels_dev = labels[dev_mask].tolist()
        for cw, l2 in itertools.product(
            [x.strip() for x in args.class_weight.split(",") if x.strip()],
            [float(x) for x in args.l2.split(",") if x.strip()],
        ):
            model, _ = fit_logistic(z_train, y_train, k=k, l2=l2,
                                    class_weight=cw, max_iter=args.max_iter,
                                    seed=args.seed)
            dev_probs = predict(model, z_dev)
            metrics = accuracy_metrics(labels_dev, dev_probs, options, args.task)
            score = (
                metrics["balanced_accuracy"] if args.task == "binary"
                else metrics["macro_f1"]
            )
            row = {
                "layer": layer, "l2": l2, "class_weight": cw,
                "dev_score": float(score),
                "dev_accuracy": float(metrics["accuracy"]),
                "dev_macro_tpr": float(metrics.get("macro_tpr", metrics.get("macro_recall", 0))),
                "dev_fraud_recall": metrics.get("fraud_recall"),
                "dev_normal_recall": metrics.get("normal_recall_tnr"),
                "dev_ece": metrics.get("ece_10bin"),
                "dev_nll": metrics.get("nll"),
            }
            grid.append(row)
            # Rank by balanced accuracy / macro F1, then fraud recall, then -ECE.
            key = (row["dev_score"], row["dev_fraud_recall"] or 0.0, -row["dev_ece"])
            if best is None or key > best[0]:
                best = (key, layer, l2, cw, metrics)
            print(f"layer={layer:>2} l2={l2:<6g} cw={cw:<8} "
                  f"score={row['dev_score']:.4f} acc={row['dev_accuracy']:.4f} "
                  f"fraud_recall={row['dev_fraud_recall']} "
                  f"normal_recall={row['dev_normal_recall']}", flush=True)

    _, best_layer, best_l2, best_cw, best_dev_metrics = best
    print(f"\nselected layer={best_layer} l2={best_l2} cw={best_cw}")
    print(f"dev metrics: {json.dumps(best_dev_metrics, ensure_ascii=False)[:1000]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def refit(layer: int, l2: float, cw: str):
        x_all = feats[:, layers.index(layer), :]
        z_train, z_other, mean, std = standardize(x_all[train_mask], x_all)
        model, _ = fit_logistic(z_train, y[train_mask], k=k, l2=l2,
                                class_weight=cw, max_iter=args.max_iter,
                                seed=args.seed)
        probs_all = predict(model, z_other)
        thresholds = {"argmax": 0.5}
        if args.task == "binary":
            normal_dev = probs_all[dev_mask][y[dev_mask] == 0, 1]
            for target in (0.05, 0.10):
                thresholds[f"fpr_{int(target * 100)}"] = threshold_for_fpr(normal_dev, target)
        return model, mean, std, probs_all, thresholds

    def save_head(path: Path, model, mean, std, layer, l2, cw):
        np.savez_compressed(
            path,
            weight=model.weight.detach().cpu().numpy(),
            bias=model.bias.detach().cpu().numpy(),
            mean=mean, std=std, layer=np.asarray(layer),
            l2=np.asarray(l2), class_weight=np.asarray(cw),
            options=np.asarray(options),
        )

    def save_predictions(path: Path, probs_all):
        with path.open("w", encoding="utf-8") as fh:
            for i, row_id in enumerate(ids):
                rec = {
                    "id": row_id,
                    "split": splits[i],
                    "label": labels[i],
                    "probs": [float(x) for x in probs_all[i]],
                    "answer": options[int(probs_all[i].argmax())],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    model, mean, std, probs_all, thresholds = refit(best_layer, best_l2, best_cw)
    save_head(out_dir / "probe_head.npz", model, mean, std, best_layer, best_l2, best_cw)
    save_predictions(out_dir / "predictions.jsonl", probs_all)

    # Secondary, risk-first operating point: among configs whose dev fraud
    # recall is already >= 0.95, maximize normal recall.  This is reported
    # separately; it does not change the primary balanced-accuracy selection.
    risk_first = None
    if args.task == "binary":
        candidates = [r for r in grid if (r["dev_fraud_recall"] or 0.0) >= 0.95]
        if candidates:
            rf = max(candidates, key=lambda r: (
                r["dev_normal_recall"] or 0.0, r["dev_score"], -r["dev_ece"]))
            rf_model, rf_mean, rf_std, rf_probs, rf_thresholds = refit(
                rf["layer"], rf["l2"], rf["class_weight"])
            rf_metrics = accuracy_metrics(labels_dev, rf_probs[dev_mask], options, args.task)
            save_head(out_dir / "probe_head_riskfirst.npz", rf_model, rf_mean, rf_std,
                      rf["layer"], rf["l2"], rf["class_weight"])
            save_predictions(out_dir / "predictions_riskfirst.jsonl", rf_probs)
            risk_first = {
                "layer": rf["layer"], "l2": rf["l2"],
                "class_weight": rf["class_weight"], "thresholds": rf_thresholds,
                "dev_metrics": rf_metrics,
            }
            print(f"risk-first: layer={rf['layer']} l2={rf['l2']} cw={rf['class_weight']} "
                  f"dev fraud_recall={rf_metrics['fraud_recall']:.4f} "
                  f"normal_recall={rf_metrics['normal_recall_tnr']:.4f}")

    (out_dir / "probe_grid.json").write_text(json.dumps({
        "task": args.task,
        "features": args.features,
        "grid": grid,
        "selected": {"layer": best_layer, "l2": best_l2,
                     "class_weight": best_cw, "thresholds": thresholds},
        "dev_metrics": best_dev_metrics,
        "selected_riskfirst": risk_first,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Dev-only metrics are printed; gate metrics are computed by the report
    # step, after all configurations are frozen.
    dev_probs = probs_all[dev_mask]
    dev_metrics = accuracy_metrics(labels_dev, dev_probs, options, args.task)
    print(f"\ndev metrics (refit on cal): {json.dumps(dev_metrics, ensure_ascii=False)[:1000]}")
    for name, t in thresholds.items():
        if args.task != "binary" or name == "argmax":
            continue
        pred = (probs_all[dev_mask][:, 1] >= t).astype(int)
        ans = [options[int(x)] for x in pred]
        m = summarize_binary(labels_dev, ans, np.eye(2)[pred], options)
        print(f"dev {name}: threshold={t:.4f} acc={m['accuracy']:.4f} "
              f"fraud_recall={m['fraud_recall']:.4f} normal_recall={m['normal_recall_tnr']:.4f}")
    print(f"wrote {out_dir}/probe_grid.json, probe_head.npz, predictions.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
