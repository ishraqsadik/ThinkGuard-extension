from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def benchmark_generations(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    *,
    max_new_tokens: int = 64,
    warmup: int = 3,
    repeats: Optional[int] = None,
) -> Dict[str, float]:
    """Wall-clock latency per generation (CUDA events if available)."""
    device = next(model.parameters()).device
    if repeats is None:
        repeats = len(prompts)

    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    if pad is None:
        pad = eos

    times_ms: List[float] = []
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    def one_call(p: str) -> float:
        inputs = tokenizer([p], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        if torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad,
                    eos_token_id=eos,
                    do_sample=False,
                )
            ender.record()
            torch.cuda.synchronize()
            # ms
            return float(starter.elapsed_time(ender))
        import time

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad,
                eos_token_id=eos,
                do_sample=False,
            )
        _ = out  # noqa: F841
        return (time.perf_counter() - t0) * 1000.0

    pool = prompts * (1 + (repeats // max(1, len(prompts))))
    pool = pool[: warmup + repeats]

    for i, p in enumerate(pool):
        dt = one_call(p)
        if i >= warmup:
            times_ms.append(dt)

    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else float("nan")
    )
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "n": len(times_ms),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
        "p50_ms": float(np.percentile(arr, 50)),
        "peak_vram_mb": float(peak_mb),
    }
