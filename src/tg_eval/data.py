from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

from .categories import BEAVERTAILS_CATEGORY_KEYS

MODERATION_LABEL_COLS = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]

BEAVERTAILS_URI = "hf://datasets/PKU-Alignment/BeaverTails/round0/30k/test.jsonl.gz"


def _take_split(ds: DatasetDict | Dataset, split: Optional[str]) -> Dataset:
    if isinstance(ds, DatasetDict):
        if split and split in ds:
            return ds[split]
        # Prefer common split names
        for key in ("test", "validation", "train"):
            if key in ds:
                return ds[key]
        first = next(iter(ds.keys()))
        return ds[first]
    return ds


def load_beavertails(split: Optional[str] = None, max_samples: Optional[int] = None, hf_token: Optional[str] = None) -> Dataset:
    ds = load_dataset("json", data_files={"train": BEAVERTAILS_URI}, token=hf_token)["train"]
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def load_toxic_chat(max_samples: Optional[int] = None, hf_token: Optional[str] = None) -> Dataset:
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", token=hf_token)
    ds = _take_split(ds, "test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def load_wildguard_mix(max_samples: Optional[int] = None, hf_token: Optional[str] = None) -> Dataset:
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", token=hf_token)
    ds = _take_split(ds, None)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def load_openai_moderation(max_samples: Optional[int] = None, hf_token: Optional[str] = None) -> Dataset:
    ds = load_dataset("walledai/openai-moderation-dataset", split="train", token=hf_token)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def iter_benchmark_rows(mode: str, max_samples: Optional[int] = None, hf_token: Optional[str] = None) -> Dataset:
    if mode == "beaver":
        return load_beavertails(max_samples=max_samples, hf_token=hf_token)
    if mode == "toxic":
        return load_toxic_chat(max_samples=max_samples, hf_token=hf_token)
    if mode == "wildguard":
        return load_wildguard_mix(max_samples=max_samples, hf_token=hf_token)
    if mode == "openai":
        return load_openai_moderation(max_samples=max_samples, hf_token=hf_token)
    raise ValueError(f"Unknown benchmark {mode}")


def row_to_eval_payload(mode: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a dataset row into prompt pairs + labels for metrics."""
    if mode == "beaver":
        cat = row["category"]
        y_multi = [1.0 if bool(cat[k]) else 0.0 for k in BEAVERTAILS_CATEGORY_KEYS]
        y_bin = 0 if row["is_safe"] else 1
        return {
            "prompt_user": row["prompt"],
            "prompt_agent": row["response"],
            "y_binary": y_bin,
            "y_multihot": y_multi,
            "meta": {"benchmark": "beaver"},
        }
    if mode == "toxic":
        toxic = int(row.get("toxicity", 0))
        return {
            "prompt_user": row["user_input"],
            "prompt_agent": row["model_output"],
            "y_binary": toxic,
            "y_multihot": None,
            "meta": {"benchmark": "toxic"},
        }
    if mode == "wildguard":
        resp = row.get("response") or ""
        rh = row.get("response_harm_label")
        ph = row.get("prompt_harm_label")
        sub = row.get("subcategory")

        def hb(x):
            if x is None:
                return None
            if x == "harmful":
                return 1
            if x == "unharmful":
                return 0
            return None

        y_prompt = hb(ph)
        y_resp = hb(rh)
        return {
            "prompt_user": row["prompt"],
            "prompt_agent": resp if resp else "(empty assistant)",
            "y_binary": y_resp,
            "prompt_harm": y_prompt,
            "response_harm": y_resp,
            "y_multihot": None,
            "meta": {"benchmark": "wildguard", "subcategory": sub, "response_harm_label": rh, "prompt_harm_label": ph},
        }
    if mode == "openai":
        prompt = row.get("prompt") or row.get("text") or row.get("input")
        labels = []
        for c in MODERATION_LABEL_COLS:
            if c not in row or row[c] is None:
                labels.append(None)
            else:
                labels.append(int(row[c]))
        if None in labels:
            y_bin = None
        else:
            y_bin = 1 if any(labels) else 0
        placeholder = "(no assistant reply)"
        return {
            "prompt_user": prompt,
            "prompt_agent": placeholder,
            "y_binary": y_bin,
            "y_multihot": None,
            "meta": {"benchmark": "openai", "raw_labels": labels},
        }
    raise ValueError(mode)


def assistant_placeholder_for_prompt_only() -> str:
    return "(no assistant reply)"
