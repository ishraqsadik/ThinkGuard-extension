#!/usr/bin/env python3
"""Run ThinkGuard / LG3-style evaluations across BeaverTails, ToxicChat, WildGuardMix, OpenAI moderation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from tg_eval.categories import get_categories_for_benchmark
from tg_eval.data import iter_benchmark_rows, row_to_eval_payload
from tg_eval.metrics import aggregate_records, plot_reflect_compare, plot_summary_bar
from tg_eval.models import load_model_and_tokenizer
from tg_eval.parse import parse_prediction
from tg_eval.prompting import evaluate_safety
from tg_eval.reflect import reflective_pipeline


def _should_skip(mode: str, payload: dict) -> bool:
    if mode == "openai" and payload["y_binary"] is None:
        return True
    if mode == "wildguard":
        if payload.get("prompt_harm") is None and payload.get("response_harm") is None:
            return True
    return False


def _run_benchmark(mode: str, args, model, tokenizer, hf_token: str | None) -> tuple[list[dict], dict, dict | None]:
    categories, _, _ = get_categories_for_benchmark(mode)  # type: ignore[arg-type]
    ds = iter_benchmark_rows(mode, max_samples=args.max_samples, hf_token=hf_token)
    records: list[dict] = []
    nc = len(categories)

    for row in tqdm(ds, desc=mode):
        payload = row_to_eval_payload(mode, row)
        if _should_skip(mode, payload):
            continue
        pu, pa = payload["prompt_user"], payload["prompt_agent"]

        if args.reflect:
            outs = reflective_pipeline(
                model,
                tokenizer,
                pu,
                pa,
                categories,
                max_new_tokens=args.max_new_tokens,
                verbose=args.verbose,
            )
            text_i, _, text_r = outs
            vi, mhi = parse_prediction(text_i, nc)
            vr, mhr = parse_prediction(text_r, nc)
            rec = {
                "prompt_user": pu,
                "prompt_agent": pa,
                "meta": payload["meta"],
                "model_output_initial": text_i[:8000],
                "model_output_reflect": text_r[:8000],
                "verdict": vi,
                "verdict_reflect": vr,
                "y_binary": payload["y_binary"],
                "prompt_harm": payload.get("prompt_harm"),
                "response_harm": payload.get("response_harm"),
                "y_multihot_true": payload.get("y_multihot"),
                "y_multihot_pred": mhi.tolist(),
                "y_multihot_pred_reflect": mhr.tolist(),
            }
        else:
            text = evaluate_safety(
                model,
                tokenizer,
                pu,
                pa,
                categories,
                max_new_tokens=args.max_new_tokens,
                verbose=args.verbose,
            )
            v, mh = parse_prediction(text, nc)
            rec = {
                "prompt_user": pu,
                "prompt_agent": pa,
                "meta": payload["meta"],
                "model_output": text[:8000],
                "verdict": v,
                "y_binary": payload["y_binary"],
                "prompt_harm": payload.get("prompt_harm"),
                "response_harm": payload.get("response_harm"),
                "y_multihot_true": payload.get("y_multihot"),
                "y_multihot_pred": mh.tolist(),
            }
        records.append(rec)

    if not records:
        empty = {"benchmark": mode, "n": 0, "note": "no_rows_after_filtering"}
        return records, empty, None

    summary_s = aggregate_records(records)
    summary_r = None
    if args.reflect:
        summary_r = aggregate_records(
            records,
            verdict_field="verdict_reflect",
            multihot_pred_field="y_multihot_pred_reflect",
        )
    return records, summary_s, summary_r


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_metrics_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="ThinkGuard reproduction runner")
    parser.add_argument("--benchmark", choices=["beaver", "toxic", "wildguard", "openai", "all"], default="beaver")
    parser.add_argument("--model-id", default="Rakancorle1/ThinkGuard")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "default_run")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reflect", action="store_true", help="3-step reflective inference (Extension A)")
    parser.add_argument("--quantize-4bit", action="store_true", help="Load model in NF4 (Extension B inference path)")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    benchmarks = ["beaver", "toxic", "wildguard", "openai"] if args.benchmark == "all" else [args.benchmark]

    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        quantize_4bit=args.quantize_4bit,
        hf_token=hf_token,
    )

    all_summaries: list[tuple[str, dict]] = []
    combined_rows: list[dict] = []
    for mode in benchmarks:
        out_dir = args.output_dir / mode
        records, summary_s, summary_r = _run_benchmark(mode, args, model, tokenizer, hf_token)
        _write_json(out_dir / "predictions.json", {"rows": records})
        _write_json(out_dir / "summary_single.json", summary_s)
        flat = {
            "benchmark": mode,
            "setting": "single",
            "n_evaluated": summary_s.get("n"),
            **summary_s.get("binary", {}),
            **{
                k: v
                for k, v in summary_s.items()
                if k not in ("binary", "per_task", "n", "verdict_field")
            },
        }
        if summary_s.get("per_task"):
            flat["per_task_json"] = json.dumps(summary_s["per_task"])
        combined_rows.append(flat)

        all_summaries.append((mode, summary_s))

        if summary_r:
            _write_json(out_dir / "summary_reflect.json", summary_r)
            flat_r = {
                "benchmark": mode,
                "setting": "reflect",
                "n_evaluated": summary_r.get("n"),
                **summary_r.get("binary", {}),
                **{
                    k: v
                    for k, v in summary_r.items()
                    if k not in ("binary", "per_task", "n", "verdict_field")
                },
            }
            if summary_r.get("per_task"):
                flat_r["per_task_json"] = json.dumps(summary_r["per_task"])
            combined_rows.append(flat_r)

            if (
                mode == "wildguard"
                and summary_s.get("macro_f1_tasks") is not None
                and summary_r.get("macro_f1_tasks") is not None
            ):
                plot_reflect_compare(
                    summary_s["macro_f1_tasks"],
                    summary_r["macro_f1_tasks"],
                    args.output_dir / mode / "reflect_vs_single.png",
                )

    _write_metrics_rows(args.output_dir / "metrics_summary.csv", combined_rows)

    if len(all_summaries) > 1:
        plot_summary_bar(all_summaries, args.output_dir / "overview_metrics.png")


if __name__ == "__main__":
    main()
