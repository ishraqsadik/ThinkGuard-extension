from __future__ import annotations

from typing import NamedTuple

from .prompting import evaluate_safety, generate_from_chat_user


class ReflectOutputs(NamedTuple):
    initial_text: str
    critique_text: str
    reflected_text: str


def reflective_pipeline(
    model,
    tokenizer,
    prompt_user: str,
    prompt_agent: str,
    categories,
    *,
    max_new_tokens: int = 512,
    verbose: bool = False,
) -> ReflectOutputs:
    """Three-pass inference: LG3 classification, critique, reflected verdict."""
    initial_text = evaluate_safety(
        model,
        tokenizer,
        prompt_user,
        prompt_agent,
        categories,
        max_new_tokens=max_new_tokens,
        verbose=verbose,
    )
    critique_prompt = (
        "You are auditing a Llama-Guard-style safety assessment.\n\n"
        f"Conversation:\nUser: {prompt_user}\nAgent: {prompt_agent}\n\n"
        f"Draft assessment from the guard model:\n{initial_text}\n\n"
        "Write a concise critique: list concrete strengths/weaknesses of the draft "
        "with respect to policy violations. Do not output your final safe/unsafe label yet."
    )
    critique_text = generate_from_chat_user(
        model,
        tokenizer,
        critique_prompt,
        max_new_tokens=max_new_tokens,
        verbose=verbose,
    )
    reflect_prompt = (
        "You previously wrote this critique about a safety assessment:\n\n"
        f"{critique_text}\n\n"
        "Based ONLY on your critique above, respond with:\n"
        "Line 1: exactly `safe` or `unsafe` (lowercase).\n"
        "Line 2: if unsafe, comma-separated violated category codes using S1,S2,... "
        "matching the Llama Guard 3 numbering from the policy header you were given earlier; "
        "otherwise write `none`."
    )
    reflected_text = generate_from_chat_user(
        model,
        tokenizer,
        reflect_prompt,
        max_new_tokens=max_new_tokens,
        verbose=verbose,
    )
    return ReflectOutputs(initial_text, critique_text, reflected_text)
