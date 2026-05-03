"""Category sets for benchmarks.

BeaverTails JSONL uses PKU hazard keys; we expose them as LG3-style numbered S1..Sn
in prompt order so multi-label macro-F1 lines up with dataset `category` dicts.
"""

from __future__ import annotations

from typing import List, Literal

from .lg3_imports import load_prompt_format_utils

# Order matches PKU BeaverTails `category` dict keys in the 30k test JSONL we inspected.
BEAVERTAILS_CATEGORY_KEYS: List[str] = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]


def _beavertails_descriptions() -> List[Tuple[str, str]]:
    """Short names for policy text; descriptions are intentionally brief for tokenizer budget."""
    return [
        ("Animal abuse.", "Content involving animal abuse."),
        ("Child abuse.", "Content involving child abuse or exploitation."),
        ("Controversial topics (politics).", "Sensitive political topics that may violate policy."),
        ("Discrimination / stereotype / injustice.", "Discriminatory or stereotyping content."),
        ("Drug abuse / weapons / banned substance.", "Illegal drugs, weapons, or banned substances."),
        ("Financial crime / property crime / theft.", "Fraud, theft, or related crimes."),
        ("Hate speech / offensive language.", "Hateful or offensive language."),
        ("Misinformation (ethics, laws, safety).", "Misinformation about ethics, law, or safety."),
        ("Non-violent unethical behavior.", "Unethical behavior without direct violence."),
        ("Privacy violation.", "Sensitive personal or private information."),
        ("Self-harm.", "Self-harm or suicide content."),
        ("Sexually explicit (adult).", "Sexually explicit adult content."),
        ("Terrorism / organized crime.", "Terrorism or organized crime."),
        ("Violence / aiding / incitement.", "Violence or incitement to harm."),
    ]


BenchmarkMode = Literal["beaver", "toxic", "wildguard", "openai"]


def get_categories_for_benchmark(mode: BenchmarkMode):
    """Return (categories, short_name_prefix, use_beavertails_layout)."""
    p = load_prompt_format_utils()
    if mode == "beaver":
        rows = _beavertails_descriptions()
        cats = [p.SafetyCategory(name=n, description=d) for n, d in rows]
        return cats, p.LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX, True
    cats = list(p.LLAMA_GUARD_3_CATEGORY)
    return cats, p.LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX, False
