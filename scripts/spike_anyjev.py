#!/usr/bin/env python3
"""M0 spike: introspect the installed AnyJev API and Qwen3.5-4B tokenizer/template.

This script is deliberately defensive: AnyJev is pre-alpha and its API has been
changing. `probe` prints what is actually installed; `run` compares the readout
configurations on a goldens JSONL.

Examples
--------
    .venv/bin/python scripts/spike_anyjev.py probe
    .venv/bin/python scripts/spike_anyjev.py probe --model Qwen/Qwen3.5-4B \
        --hf-endpoint https://hf-mirror.com
    .venv/bin/python scripts/spike_anyjev.py tokens --model Qwen/Qwen3.5-4B \
        --hf-endpoint https://hf-mirror.com
    .venv/bin/python scripts/spike_anyjev.py run --model Qwen/Qwen3.5-4B \
        --hf-endpoint https://hf-mirror.com \
        --levels raw,L0,L0-noprior,L0-cf,L0-adaptive,L1 \
        --out runs/spike-m0

Outputs
-------
Human-readable report on stdout, plus JSON at runs/spike-probe/report.json
(override with --json-out).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def sig(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:  # noqa: BLE001
        return f"<signature unavailable: {exc!r}>"


def doc(obj: Any, max_lines: int = 12) -> str:
    text = inspect.getdoc(obj) or ""
    return "\n".join(text.splitlines()[:max_lines])


def fields(cls: Any) -> dict[str, str]:
    try:
        return {f.name: str(f.type) for f in dataclasses.fields(cls)}
    except Exception:  # noqa: BLE001
        return {k: str(v) for k, v in getattr(cls, "__annotations__", {}).items()}


def public_names(obj: Any) -> list[str]:
    return sorted(n for n in dir(obj) if not n.startswith("_"))


def source_of(obj: Any, max_lines: int = 120) -> str:
    try:
        lines = inspect.getsource(obj).splitlines()
        return "\n".join(lines[:max_lines])
    except Exception as exc:  # noqa: BLE001
        return f"<source unavailable: {exc!r}>"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def safe(label: str, fn, report: dict[str, Any]) -> Any:
    """Run fn(), record either the result or the exception, never crash."""
    try:
        value = fn()
        report[label] = value
        print(f"[ok]   {label}")
        return value
    except Exception as exc:  # noqa: BLE001
        report[label] = {"error": repr(exc), "traceback": traceback.format_exc()[-2000:]}
        print(f"[fail] {label}: {exc!r}")
        return None


# --------------------------------------------------------------------------- #
# probe: Python / package versions
# --------------------------------------------------------------------------- #

def probe_versions(report: dict[str, Any]) -> None:
    section("versions")
    import anyjev

    report["python"] = sys.version
    report["python_executable"] = sys.executable
    report["anyjev_version"] = getattr(anyjev, "__version__", None)
    report["anyjev_file"] = getattr(anyjev, "__file__", None)
    report["anyjev_exports"] = public_names(anyjev)
    print(f"python          : {sys.version.split()[0]} ({sys.executable})")
    print(f"anyjev          : {report['anyjev_version']} ({report['anyjev_file']})")
    print(f"anyjev exports  : {', '.join(report['anyjev_exports'])}")

    for modname in ("torch", "transformers", "accelerate", "numpy"):
        try:
            mod = importlib.import_module(modname)
            ver = getattr(mod, "__version__", "?")
            report[f"{modname}_version"] = ver
            print(f"{modname:<15} : {ver}")
        except Exception as exc:  # noqa: BLE001
            report[f"{modname}_version"] = {"error": repr(exc)}
            print(f"{modname:<15} : NOT AVAILABLE ({exc!r})")

    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        report["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else []
        print(f"cuda            : {report['cuda_available']} ({report['cuda_device_count']} devices)")
        for i, name in enumerate(report["cuda_devices"]):
            print(f"  [{i}] {name}")
    except Exception as exc:  # noqa: BLE001
        report["cuda_available"] = {"error": repr(exc)}
        print(f"cuda            : unknown ({exc!r})")


# --------------------------------------------------------------------------- #
# probe: AnyJev API surface
# --------------------------------------------------------------------------- #

def probe_api(report: dict[str, Any]) -> None:
    section("AnyJev API surface")

    # ---- top-level classes -------------------------------------------------
    for name in ("Decision", "Decider", "Question", "levels", "LEVELS"):
        if not hasattr(importlib.import_module("anyjev"), name):
            continue
        obj = getattr(importlib.import_module("anyjev"), name)
        key = f"anyjev.{name}"
        if inspect.isclass(obj):
            report[f"{key}.fields"] = fields(obj)
            report[f"{key}.public"] = public_names(obj)
            report[f"{key}.doc"] = doc(obj)
            print(f"\n--- {key} ---")
            print(f"doc: {doc(obj, 6) or '(none)'}")
            if report[f"{key}.fields"]:
                print(f"fields: {report[f'{key}.fields']}")
            print(f"public: {', '.join(report[f'{key}.public'])}")
        else:
            report[key] = repr(obj)
            print(f"\n{key} = {obj!r}")

    # ---- signatures --------------------------------------------------------
    from anyjev import Decider, Question

    report["Decider.__init__"] = sig(Decider.__init__)
    report["Decider.decide"] = sig(Decider.decide) if hasattr(Decider, "decide") else None
    print(f"\nDecider.__init__ : {report['Decider.__init__']}")
    print(f"Decider.decide   : {report['Decider.decide']}")
    print(f"Decider methods  : {', '.join(n for n in public_names(Decider) if not n.startswith('_'))}")

    report["Decider.calibration_methods"] = [
        n for n in public_names(Decider)
        if any(k in n.lower() for k in ("calibrat", "adapt", "fit", "certif", "rotat"))
    ]
    print(f"calibration-ish  : {', '.join(report['Decider.calibration_methods']) or '(none)'}")

    for qname in ("choice", "noul", "score"):
        if hasattr(Question, qname):
            report[f"Question.{qname}"] = sig(getattr(Question, qname))
            report[f"Question.{qname}.doc"] = doc(getattr(Question, qname), 8)
            print(f"\nQuestion.{qname}{report[f'Question.{qname}']}")
            print(doc(getattr(Question, qname), 8) or "(no doc)")

    # ---- constants ---------------------------------------------------------
    section("constants / module-level config")
    for modname in ("anyjev.question", "anyjev.decider", "anyjev.calibration"):
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001
            report[f"{modname}.import"] = {"error": repr(exc)}
            print(f"[skip] {modname}: {exc!r}")
            continue
        consts = {
            n: repr(getattr(mod, n))
            for n in dir(mod)
            if n.isupper() and not n.startswith("_")
        }
        report[f"{modname}.constants"] = consts
        print(f"{modname}: {consts or '(no uppercase constants)'}")

    # ---- backends ----------------------------------------------------------
    section("backends")
    try:
        hf_mod = importlib.import_module("anyjev.backends.hf")
        report["hf_backend_public"] = public_names(hf_mod)
        print(f"anyjev.backends.hf public: {', '.join(report['hf_backend_public'])}")
        for name in ("HFBackend",):
            if hasattr(hf_mod, name):
                cls = getattr(hf_mod, name)
                report[f"HFBackend.{name}.init"] = sig(cls.__init__)
                report[f"HFBackend.{name}.public"] = public_names(cls)
                report[f"HFBackend.{name}.doc"] = doc(cls, 10)
                print(f"\n{name}.__init__ {report[f'HFBackend.{name}.init']}")
                print(f"public: {', '.join(report[f'HFBackend.{name}.public'])}")
                print("--- source (first 120 lines) ---")
                print(source_of(cls))
    except Exception as exc:  # noqa: BLE001
        report["hf_backend_error"] = repr(exc)
        print(f"[fail] anyjev.backends.hf: {exc!r}")

    # ---- short source excerpts for semantics -------------------------------
    section("source excerpts (Decision / level handling)")
    try:
        from anyjev import Decision

        report["Decision.source"] = source_of(Decision, 80)
        print(report["Decision.source"])
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] Decision source: {exc!r}")

    # ---- package file tree (small) ----------------------------------------
    section("installed package layout")
    try:
        import anyjev

        root = Path(anyjev.__file__).parent
        layout = sorted(
            str(p.relative_to(root))
            for p in root.rglob("*.py")
            if "__pycache__" not in str(p)
        )
        report["package_layout"] = layout
        print("\n".join(layout))
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] layout: {exc!r}")


# --------------------------------------------------------------------------- #
# probe: tokenizer / chat template / model config
# --------------------------------------------------------------------------- #

def probe_model(model: str, hf_endpoint: str | None, trust_remote_code: bool,
                report: dict[str, Any]) -> None:
    section(f"model probe: {model}")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"HF_ENDPOINT={hf_endpoint}")
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default)')}")

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    report["model_config"] = {
        "class": type(config).__name__,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "is_encoder_decoder": getattr(config, "is_encoder_decoder", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "text_config": type(getattr(config, "text_config", None)).__name__,
    }
    print(f"config class      : {report['model_config']['class']}")
    print(f"model_type        : {report['model_config']['model_type']}")
    print(f"architectures     : {report['model_config']['architectures']}")
    print(f"is_encoder_decoder: {report['model_config']['is_encoder_decoder']}")
    print(f"vocab_size        : {report['model_config']['vocab_size']}")
    print(f"text_config       : {report['model_config']['text_config']}")

    tokenizer = AutoTokenizer.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    report["tokenizer_class"] = type(tokenizer).__name__
    report["tokenizer_vocab_size"] = len(tokenizer)
    report["has_chat_template"] = tokenizer.chat_template is not None
    print(f"tokenizer         : {report['tokenizer_class']} (len={report['tokenizer_vocab_size']})")
    print(f"chat_template     : {'present' if report['has_chat_template'] else 'MISSING'}")

    # ---- letter tokenization ----------------------------------------------
    section("letter-token readout check (A-Z)")
    variants = ["A", " A", "A)", " A)", "(A)", "A.", " A.", "a", " a"]
    token_map: dict[str, Any] = {}
    for text in variants:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            token_map[text] = {
                "ids": ids,
                "tokens": [tokenizer.convert_ids_to_tokens(i) for i in ids],
                "single_token": len(ids) == 1,
            }
        except Exception as exc:  # noqa: BLE001
            token_map[text] = {"error": repr(exc)}
    report["letter_variants"] = token_map
    for text, info in token_map.items():
        if "error" in info:
            print(f"  {text!r:8} ERROR {info['error']}")
        else:
            print(f"  {text!r:8} -> {info['tokens']} (single={info['single_token']})")

    single_upper = []
    multi_upper = []
    for code in range(ord("A"), ord("Z") + 1):
        ch = chr(code)
        ids = tokenizer.encode(ch, add_special_tokens=False)
        (single_upper if len(ids) == 1 else multi_upper).append(ch)
    report["single_token_letters"] = single_upper
    report["multi_token_letters"] = multi_upper
    print(f"  single-token A-Z: {len(single_upper)}/26 {single_upper}")
    print(f"  multi-token A-Z : {multi_upper or '(none)'}")

    # ---- chat template with thinking off/on --------------------------------
    section("chat template check (enable_thinking false/true)")
    messages = [{
        "role": "user",
        "content": (
            "请判断该交易应如何处理。\n"
            "状态：用户近30天首次登录，设备与历史一致，金额128元，商户为常用商户。\n"
            "A. 通过\nB. 人工复核\nC. 拒绝\n"
            "只回答一个字母。"
        ),
    }]
    template_report: dict[str, Any] = {}
    for think in (False, True):
        key = f"enable_thinking={think}"
        attempts = [
            ("direct_kwarg", {"enable_thinking": think}),
            ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": think}}),
        ]
        entry: dict[str, Any] = {}
        for attempt_name, kwargs in attempts:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
                entry[attempt_name] = {
                    "ok": True,
                    "tail": text[-600:],
                    "has_empty_think": "<think>\n\n</think>" in text,
                    "ends_with_open_think": text.rstrip().endswith("<think>"),
                }
                break
            except Exception as exc:  # noqa: BLE001
                entry[attempt_name] = {"ok": False, "error": repr(exc)}
        template_report[key] = entry
        print(f"\n--- {key} ---")
        for attempt_name, info in entry.items():
            if info.get("ok"):
                print(
                    f"  [{attempt_name}] OK; empty_think={info['has_empty_think']}, "
                    f"open_think={info['ends_with_open_think']}"
                )
                print("  tail:", info["tail"].replace("\n", "\\n")[-300:])
            else:
                print(f"  [{attempt_name}] FAIL: {info['error']}")
    report["chat_template"] = template_report

    # ---- model class mapping without loading weights -----------------------
    section("AutoModel mapping")
    try:
        from transformers import AutoModelForCausalLM

        cls = AutoModelForCausalLM._model_mapping  # noqa: SLF001
        report["auto_model_causal_lm_mapping"] = str(cls)
        print(f"AutoModelForCausalLM mapping available: {cls is not None}")
    except Exception as exc:  # noqa: BLE001
        report["auto_model_mapping_error"] = repr(exc)
        print(f"[fail] AutoModel mapping: {exc!r}")


# --------------------------------------------------------------------------- #
# run: compare raw / L0 / L0-adaptive / L1 on a goldens JSONL
# --------------------------------------------------------------------------- #

def load_goldens(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"state", "options", "label"} - set(row)
            if missing:
                raise ValueError(f"{path}:{lineno} missing fields {sorted(missing)}")
            if row["label"] not in row["options"]:
                raise ValueError(f"{path}:{lineno} label {row['label']!r} not in options")
            rows.append(row)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def common_question(rows: list[dict[str, Any]]):
    from anyjev import Question

    texts = {r.get("question", "请判断该笔交易应如何处理：") for r in rows}
    option_sets = {tuple(r["options"]) for r in rows}
    if len(texts) != 1:
        raise ValueError(f"run expects one question text, got {len(texts)}; split the file")
    if len(option_sets) != 1:
        raise ValueError(f"run expects one option set, got {len(option_sets)}; split the file")
    text = texts.pop()
    options = list(option_sets.pop())
    return Question.choice(text, options, name="risk"), options


def make_decider(backend, config: str, args: argparse.Namespace):
    from anyjev import Decider

    if config == "raw":
        return Decider(backend, level="raw", prior="none")
    prior = args.prior
    if config == "L0-noprior":
        prior = "none"
    elif config == "L0-cf":
        prior = "content_free"
    elif config == "L0-adaptive-cf":
        prior = "content_free"
    common = dict(
        level="L0",
        prior=prior,
        min_prior_n=args.min_prior_n,
        max_permutations=args.max_perms,
        canonical_order=not args.no_canonical_order,
    )
    if config in ("L0", "L0-noprior", "L0-cf"):
        return Decider(backend, **common)
    if config in ("L0-adaptive", "L0-adaptive-cf"):
        return Decider(
            backend,
            adaptive_shifts=True,
            adaptive_min_shifts=args.adaptive_min_shifts,
            adaptive_target=args.adaptive_target,
            **common,
        )
    raise ValueError(f"unknown config {config!r}")


def _diagnostic_scalars(diag: dict[str, Any] | None) -> dict[str, Any]:
    keys = (
        "shifts_used", "permutations", "order_flip_raw", "order_flip_l0",
        "prior_method", "prior_strength", "adaptive", "stop_threshold",
        "stop_stat", "stop_calibrated", "answer_mass",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if not diag or key not in diag:
            continue
        value = diag[key]
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:  # noqa: BLE001
                value = str(value)
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def run_config(backend, rows, question, config: str, args) -> dict[str, Any]:
    import numpy as np

    states = [r["state"] for r in rows]
    repeats = max(1, args.timing_repeats)
    decisions = None
    times: list[float] = []
    first_stats: dict[str, Any] = {}
    for repeat in range(repeats):
        decider = make_decider(backend, config, args)
        started = time.perf_counter()
        current = decider.decide_batch(states, question)
        times.append(time.perf_counter() - started)
        if repeat == 0:
            decisions = current
            first_stats = dict(decider.stats)
    elapsed = float(np.median(times))
    records = []
    for row, decision in zip(rows, decisions):
        probs = np.asarray(decision.probs, dtype=float)
        records.append(
            {
                "id": row["id"],
                "label": row["label"],
                "answer": decision.answer,
                "confidence": float(decision.confidence),
                "probs": [float(x) for x in probs],
                "level": decision.level,
                "diagnostics": _diagnostic_scalars(decision.diagnostics),
            }
        )
    return {
        "config": config,
        "elapsed_s": elapsed,
        "timing_runs_s": times,
        "per_row_s": elapsed / max(len(rows), 1),
        "decider_stats": first_stats,
        "rows": records,
    }


def run_reversal(backend, rows, question_text, options, config, args,
                 forward_rows) -> dict[str, Any]:
    from anyjev import Question

    reversed_q = Question.choice(question_text, list(reversed(options)), name="risk_reversed")
    decider = make_decider(backend, config, args)
    decisions = decider.decide_batch([r["state"] for r in rows], reversed_q)
    records = []
    for row, decision, forward in zip(rows, decisions, forward_rows):
        flip = int(decision.answer != forward["answer"])
        records.append(
            {
                "id": row["id"],
                "forward_answer": forward["answer"],
                "reversed_answer": decision.answer,
                "flip": flip,
            }
        )
    return {
        "flip_rate": sum(r["flip"] for r in records) / max(len(records), 1),
        "rows": records,
    }


def compute_metrics(records, options) -> dict[str, Any]:
    import numpy as np

    index = {option: i for i, option in enumerate(options)}
    y = np.asarray([index[r["label"]] for r in records], dtype=int)
    pred = np.asarray([index[r["answer"]] for r in records], dtype=int)
    conf = np.asarray([r["confidence"] for r in records], dtype=float)
    probs = np.asarray([r["probs"] for r in records], dtype=float)
    n, k = len(records), len(options)

    per_class = {}
    for j, option in enumerate(options):
        tp = int(((pred == j) & (y == j)).sum())
        fn = int(((pred != j) & (y == j)).sum())
        tn = int(((pred != j) & (y != j)).sum())
        fp = int(((pred == j) & (y != j)).sum())
        per_class[option] = {
            "support": tp + fn,
            "predicted": tp + fp,
            "tpr": (tp / (tp + fn)) if (tp + fn) else None,
            "tnr": (tn / (tn + fp)) if (tn + fp) else None,
        }

    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & ((conf < hi) if i < 9 else (conf <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(
                float((pred[mask] == y[mask]).mean()) - float(conf[mask].mean())
            )

    onehot = np.eye(k)[y]
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    nll = float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, None)).mean())
    tprs = [v["tpr"] for v in per_class.values() if v["tpr"] is not None]
    answers = [r["answer"] for r in records]
    return {
        "n": n,
        "accuracy": float((pred == y).mean()),
        "macro_tpr": float(np.mean(tprs)) if tprs else None,
        "ece_10bin": ece,
        "brier": brier,
        "nll": nll,
        "mean_confidence": float(conf.mean()),
        "answers": {option: answers.count(option) for option in options},
        "per_class": per_class,
    }


def fit_temperature(records, options):
    import numpy as np

    index = {option: i for i, option in enumerate(options)}
    probs = np.clip(np.asarray([r["probs"] for r in records], dtype=float), 1e-12, None)
    y = np.asarray([index[r["label"]] for r in records], dtype=int)
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 5.0, 46):
        p = probs ** (1.0 / t)
        p /= p.sum(axis=1, keepdims=True)
        nll = float(-np.log(p[np.arange(len(y)), y]).mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t, best_nll


def apply_temperature(records, options, temperature: float):
    import numpy as np

    out = []
    for record in records:
        p = np.clip(np.asarray(record["probs"], dtype=float), 1e-12, None) ** (1.0 / temperature)
        p /= p.sum()
        item = dict(record)
        item["probs"] = [float(x) for x in p]
        item["confidence"] = float(p.max())
        item["answer"] = options[int(np.argmax(p))]
        out.append(item)
    return out


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_summary(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    lines = [
        "# AnyJev × Qwen3.5-4B spike summary",
        "",
        f"- model: `{meta['model']}` (revision `{meta['revision']}`)",
        f"- goldens: `{meta['goldens']}` ({meta['n_rows']} rows, synthetic={meta['synthetic']})",
        f"- anyjev {meta['anyjev_version']} / transformers {meta['transformers_version']} / torch {meta['torch_version']}",
        f"- device: {meta['device']}, dtype: {meta['dtype']}, batch_size: {meta['batch_size']}",
        f"- backend: {meta.get('backend', {}).get('model_class')} "
        f"({meta.get('backend', {}).get('n_layers')} layers, hidden "
        f"{meta.get('backend', {}).get('hidden_size')}), tokenizer "
        f"{meta.get('backend', {}).get('tokenizer_class')}",
        f"- question: {meta['question']}",
        f"- options: {', '.join(meta['options'])}",
        f"- prior: {meta['prior']}, max_perms: {meta['max_perms']}, canonical_order: {meta['canonical_order']}",
        "",
        "## Metrics",
        "",
        "| config | acc | macro TPR | ECE | Brier | NLL | mean conf | mean shifts | reversal flip | s/row |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["results"].items():
        m = result["metrics"]
        shifts = [
            r["diagnostics"].get("shifts_used", r["diagnostics"].get("permutations"))
            for r in result["rows"]
            if r.get("diagnostics")
        ]
        shifts = [s for s in shifts if isinstance(s, (int, float))]
        mean_shifts = sum(shifts) / len(shifts) if shifts else None
        reversal = result.get("option_reversal", {}).get("flip_rate")
        lines.append(
            f"| {name} | {_fmt(m['accuracy'])} | {_fmt(m['macro_tpr'])} | "
            f"{_fmt(m['ece_10bin'])} | {_fmt(m['brier'])} | {_fmt(m['nll'])} | "
            f"{_fmt(m['mean_confidence'])} | {_fmt(mean_shifts, 1)} | {_fmt(reversal)} | "
            f"{_fmt(result['per_row_s'], 4)} |"
        )

    lines += ["", "## Per-class TPR / TNR", ""]
    for name, result in payload["results"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| class | support | predicted | TPR | TNR |")
        lines.append("|---|---:|---:|---:|---:|")
        for option, stats in result["metrics"]["per_class"].items():
            lines.append(
                f"| {option} | {stats['support']} | {stats['predicted']} | "
                f"{_fmt(stats['tpr'])} | {_fmt(stats['tnr'])} |"
            )
        lines.append("")

    lines += ["## Notes", ""]
    lines += [f"- {note}" for note in meta.get("notes", [])]
    lines.append("")
    return "\n".join(lines)


def run_spike(args: argparse.Namespace) -> int:
    import numpy as np  # noqa: F401  (kept for parity with the metric helpers)
    import anyjev
    import torch
    import transformers
    from anyjev.backends.hf import HFBackend

    rows = load_goldens(args.goldens, args.limit)
    question, options = common_question(rows)
    print(f"loaded {len(rows)} rows; question={question.text!r}; options={options}")

    backend = HFBackend(
        args.model,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    backend_meta = {
        "model_class": type(backend.model).__name__,
        "config_class": type(backend.model.config).__name__,
        "n_layers": backend.n_layers,
        "hidden_size": backend.hidden_size,
        "tokenizer_class": type(backend.tokenizer).__name__,
        "shared_fallbacks": getattr(backend, "shared_fallbacks", None),
        "param_dtype": str(next(backend.model.parameters()).dtype),
    }
    print(f"backend ready: {backend_meta}")

    # Warm the kernels on a single row so the first timed config is not paying
    # CUDA/cuBLAS initialization. Results are discarded.
    make_decider(backend, "raw", args).decide_batch([rows[0]["state"]], question)

    requested = [c.strip() for c in args.levels.split(",") if c.strip()]
    results: dict[str, Any] = {}
    for config in requested:
        if config == "L1":
            continue
        print(f"\n=== {config} ===")
        result = run_config(backend, rows, question, config, args)
        result["metrics"] = compute_metrics(result["rows"], options)
        if not args.no_reversal:
            result["option_reversal"] = run_reversal(
                backend, rows, question.text, options, config, args, result["rows"]
            )
        results[config] = result
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))

    if "L1" in requested:
        if "L0" not in results:
            raise ValueError("L1 is post-hoc temperature scaling on L0; request L0 too")
        temperature, fit_nll = fit_temperature(results["L0"]["rows"], options)
        l1_rows = apply_temperature(results["L0"]["rows"], options, temperature)
        results["L1"] = {
            "config": "L1",
            "base": "L0",
            "temperature": temperature,
            "fit_nll_in_sample": fit_nll,
            "in_sample": True,
            "elapsed_s": 0.0,
            "per_row_s": 0.0,
            "decider_stats": {},
            "rows": l1_rows,
            "metrics": compute_metrics(l1_rows, options),
        }
        print(f"\n=== L1 (temperature={temperature:.2f}, fit in-sample) ===")

    notes = [
        "Samples are synthetic smoke-test rows, not the Phase 0 goldens; metrics are not a quality claim.",
        "AnyJev 0.3.0 implements raw and L0 only; L1 here is post-hoc temperature scaling fit in-sample.",
        "L0 uses a batch prior estimated from the same rows; no labels are used in the prior, but the eval batch's class composition does enter the correction.",
        "L0-noprior isolates permutation marginalization from prior correction; L0-cf uses content-free probes and costs extra forward passes.",
        "Timing is the median of --timing-repeats runs (fresh Decider each run) after a one-row warmup; it excludes model load and is not a serving benchmark.",
        "A 12-row run cannot support a promotion decision; it validates the API and pipeline.",
    ]
    payload = {
        "meta": {
            "model": args.model,
            "revision": args.revision,
            "goldens": str(args.goldens),
            "n_rows": len(rows),
            "synthetic": all(r.get("synthetic") for r in rows),
            "anyjev_version": getattr(anyjev, "__version__", None),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "timing_repeats": args.timing_repeats,
            "backend": backend_meta,
            "question": question.text,
            "options": options,
            "prior": args.prior,
            "max_perms": args.max_perms,
            "canonical_order": not args.no_canonical_order,
            "notes": notes,
        },
        "results": results,
    }
    outdir = Path(args.out) if args.out else Path("runs") / f"spike-{datetime.now():%Y%m%d-%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )
    (outdir / "summary.md").write_text(render_summary(payload))
    print(f"\nwrote {outdir / 'results.json'} and {outdir / 'summary.md'}")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["probe", "tokens", "run"], nargs="?", default="probe",
        help="probe = API + optional model; tokens = model/tokenizer only; "
             "run = compare readout levels on a goldens JSONL",
    )
    parser.add_argument("--model", default=None, help="HF/ModelScope model id or local path")
    parser.add_argument(
        "--hf-endpoint", default=None,
        help="e.g. https://hf-mirror.com (must be set before transformers import)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--json-out", default="runs/spike-probe/report.json",
        help="where to write the JSON report (probe/tokens)",
    )

    # run-only options
    parser.add_argument(
        "--goldens", default="scripts/spike_samples.jsonl",
        help="JSONL with id/state/options/label (run)",
    )
    parser.add_argument(
        "--levels", default="raw,L0",
        help="comma list: raw, L0, L0-noprior, L0-cf, L0-adaptive, L0-adaptive-cf, L1 (run)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--prior", choices=["batch", "content_free", "none"], default="batch",
        help="L0 prior mode (run)",
    )
    parser.add_argument("--min-prior-n", type=int, default=1)
    parser.add_argument("--max-perms", type=int, default=None)
    parser.add_argument("--no-canonical-order", action="store_true")
    parser.add_argument("--adaptive-min-shifts", type=int, default=2)
    parser.add_argument("--adaptive-target", type=float, default=0.01)
    parser.add_argument(
        "--timing-repeats", type=int, default=3,
        help="timed runs per config, fresh Decider each time; median is reported (run)",
    )
    parser.add_argument("--no-reversal", action="store_true")
    parser.add_argument("--out", default=None, help="output dir for run artifacts")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # HF_ENDPOINT must be set before transformers is imported.
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.command == "run":
        if not args.model:
            print("run requires --model", file=sys.stderr)
            return 2
        return run_spike(args)

    report: dict[str, Any] = {"args": vars(args), "command": args.command}

    if args.command == "probe":
        probe_versions(report)
        probe_api(report)
    if args.model:
        probe_model(args.model, args.hf_endpoint, args.trust_remote_code, report)
    else:
        print("\n[note] no --model given; skipped tokenizer/template/model probe")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nJSON report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
