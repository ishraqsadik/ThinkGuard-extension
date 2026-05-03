#!/usr/bin/env python3
"""Measure mean generation latency for FP16/BF16 vs 4-bit ThinkGuard."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from tg_eval.categories import get_categories_for_benchmark
from tg_eval.data import iter_benchmark_rows, row_to_eval_payload
from tg_eval.latency import benchmark_generations
from tg_eval.models import load_model_and_tokenizer
from tg_eval.prompting import build_formatted_prompt


def collect_prompts(mode: str, n: int, hf_token: str | None) -> list[str]:
    categories, _, _ = get_categories_for_benchmark(mode)  # type: ignore[arg-type]
    ds = iter_benchmark_rows(mode, max_samples=n, hf_token=hf_token)
    prompts: list[str] = []
    for row in ds:
        payload = row_to_eval_payload(mode, row)
        pu, pa = payload["prompt_user"], payload["prompt_agent"]
        prompts.append(build_formatted_prompt(pu, pa, categories))
        if len(prompts) >= n:
            break
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Rakancorle1/ThinkGuard")
    parser.add_argument("--benchmark", default="beaver", choices=["beaver", "toxic", "wildguard", "openai"])
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=ROOT / "results" / "latency.json")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    prompts = collect_prompts(args.benchmark, args.num_prompts, hf_token)
    if not prompts:
        raise RuntimeError("No prompts collected (dataset auth / empty?).")

    results = {}

    def run_case(name: str, quant: bool) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tokenizer = load_model_and_tokenizer(args.model_id, quantize_4bit=quant, hf_token=hf_token)
        stats = benchmark_generations(
            model,
            tokenizer,
            prompts,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        stats["quantize_4bit"] = quant
        results[name] = stats
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_case("fp16_bf16", quant=False)
    run_case("nf4", quant=True)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
