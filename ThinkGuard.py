"""Legacy BeaverTails demo entrypoint.

Prefer `python scripts/run_eval.py` for full reproducibility metrics (four benchmarks,
reflection, quantization flags). This module keeps a tiny runnable smoke test.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from tg_eval.categories import get_categories_for_benchmark
from tg_eval.data import load_beavertails, row_to_eval_payload
from tg_eval.models import load_model_and_tokenizer
from tg_eval.prompting import evaluate_safety


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    max_samples = int(os.environ.get("TG_LEGACY_MAX_SAMPLES", "5"))

    categories, _, _ = get_categories_for_benchmark("beaver")
    model_id = os.environ.get("TG_MODEL_ID", "Rakancorle1/ThinkGuard")
    model, tokenizer = load_model_and_tokenizer(model_id, quantize_4bit=False, hf_token=hf_token)

    ds = load_beavertails(max_samples=max_samples, hf_token=hf_token)
    results = []
    for row in tqdm(ds, total=len(ds), desc="ThinkGuard legacy demo"):
        payload = row_to_eval_payload("beaver", row)
        text = evaluate_safety(
            model,
            tokenizer,
            payload["prompt_user"],
            payload["prompt_agent"],
            categories,
            max_new_tokens=int(os.environ.get("TG_MAX_NEW_TOKENS", "512")),
            verbose=os.environ.get("TG_VERBOSE", "").lower() in ("1", "true", "yes"),
        )
        results.append(
            {
                "prompt": payload["prompt_user"],
                "response": payload["prompt_agent"],
                "model_output": text,
            }
        )

    output_path = Path(os.environ.get("TG_OUTPUT", "ThinkGuard_cla_results_demo.json"))
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()
