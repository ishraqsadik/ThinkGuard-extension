from __future__ import annotations

import re
from typing import List, Literal, Optional, Tuple

import numpy as np


def parse_verdict(text: str) -> Optional[Literal["safe", "unsafe"]]:
    """Extract binary verdict from Llama-Guard-style output (supports critique-heavy ThinkGuard text)."""
    if not text or not text.strip():
        return None
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    verdict_re = re.compile(r"^(safe|unsafe)\b", re.IGNORECASE)

    def pick(line: str) -> Optional[Literal["safe", "unsafe"]]:
        m = verdict_re.match(line.strip())
        if not m:
            return None
        v = m.group(1).lower()
        if v == "safe":
            return "safe"
        return "unsafe"

    for ln in lines:
        got = pick(ln)
        if got:
            return got
    # Fallback: loose scan (critique-heavy outputs)
    lower = text.lower()
    has_u = bool(re.search(r"\bunsafe\b", lower))
    has_s = bool(re.search(r"\bsafe\b", lower))
    if has_u:
        return "unsafe"
    if has_s:
        return "safe"
    return None


def extract_category_codes(text: str, prefix: str = "S") -> List[str]:
    """Find category short codes like S3,S12 (comma-separated blocks or free text)."""
    esc = re.escape(prefix)
    return re.findall(rf"\b{esc}\d+\b", text, flags=re.IGNORECASE)


def multihot_from_codes(codes: List[str], num_categories: int, prefix: str = "S") -> np.ndarray:
    v = np.zeros(num_categories, dtype=np.float32)
    esc = re.escape(prefix)
    for c in codes:
        m = re.match(rf"{esc}(\d+)$", c, flags=re.IGNORECASE)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < num_categories:
            v[idx] = 1.0
    return v


def parse_prediction(text: str, num_categories: int, prefix: str = "S") -> Tuple[Optional[Literal["safe", "unsafe"]], np.ndarray]:
    verdict = parse_verdict(text)
    codes = extract_category_codes(text, prefix=prefix)
    mh = multihot_from_codes(codes, num_categories, prefix=prefix)
    return verdict, mh
