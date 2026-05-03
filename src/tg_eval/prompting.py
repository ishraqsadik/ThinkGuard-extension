from __future__ import annotations

from typing import List

import torch

from .lg3_imports import load_prompt_format_utils
from .models import resolve_pad_eos


def build_formatted_prompt(
    prompt_user: str,
    prompt_agent: str,
    categories,
    *,
    with_policy: bool = True,
) -> str:
    p = load_prompt_format_utils()
    formatted = p.build_custom_prompt(
        agent_type=p.AgentType.AGENT,
        conversations=p.create_conversation([prompt_user, prompt_agent]),
        categories=categories,
        category_short_name_prefix=p.LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX,
        prompt_template=p.PROMPT_TEMPLATE_3,
        with_policy=with_policy,
    )
    return formatted


def generate_completion(
    model,
    tokenizer,
    formatted_prompt: str,
    *,
    max_new_tokens: int = 512,
    verbose: bool = False,
) -> str:
    if verbose:
        print("-" * 80)
        print(formatted_prompt[:4000])
        print("-" * 80)
    device = next(model.parameters()).device
    inputs = tokenizer([formatted_prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]
    pad_id, eos_id = resolve_pad_eos(tokenizer)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            do_sample=False,
        )
    gen = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    if verbose:
        print(">>>", gen[:2000])
    return gen


def evaluate_safety(
    model,
    tokenizer,
    prompt_user: str,
    prompt_agent: str,
    categories,
    *,
    max_new_tokens: int = 512,
    verbose: bool = False,
    with_policy: bool = True,
) -> str:
    fp = build_formatted_prompt(prompt_user, prompt_agent, categories, with_policy=with_policy)
    return generate_completion(model, tokenizer, fp, max_new_tokens=max_new_tokens, verbose=verbose)


def generate_from_chat_user(
    model,
    tokenizer,
    user_content: str,
    *,
    max_new_tokens: int = 512,
    verbose: bool = False,
) -> str:
    """Single-turn chat-style prompt (for reflective follow-ups)."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        formatted = user_content
    return generate_completion(model, tokenizer, formatted, max_new_tokens=max_new_tokens, verbose=verbose)
