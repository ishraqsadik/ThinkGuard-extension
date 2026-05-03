from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore


def load_model_and_tokenizer(
    model_id: str,
    *,
    quantize_4bit: bool = False,
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    model_kw: dict = {"device_map": "auto", "token": hf_token, "trust_remote_code": True}
    if quantize_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes/transformers quantization config unavailable.")
        model_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kw)
    model.eval()
    return model, tokenizer


def resolve_pad_eos(tokenizer) -> Tuple[int, int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = 128009
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos
    return int(pad), int(eos)
