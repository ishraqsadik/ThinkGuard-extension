"""Import LG3 prompt helpers from llama-recipes (legacy) or llama-cookbook (current)."""


def load_prompt_format_utils():
    try:
        from llama_recipes.inference import prompt_format_utils as p  # type: ignore
    except ImportError:
        from llama_cookbook.inference import prompt_format_utils as p  # type: ignore
    return p
