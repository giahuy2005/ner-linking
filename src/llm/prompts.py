"""Prompt for the whitelisted ontology candidate selector."""

from __future__ import annotations


_CANDIDATE_SELECTOR_SYSTEM = """You select RxNorm or ICD-10 codes for a Vietnamese medical mention.
Choose only codes in the supplied list or abstain with an empty list. Never invent a code.
For RxNorm, match ingredient first, then strength, dose form, and release type; choose at most one.
For ICD-10, choose the most specific directly supported diagnosis; choose two only for explicit
coordination of independent diagnoses. Return JSON only: {"chosen_codes":["code"]}."""


def build_candidate_selector_prompt(
    entity_text: str,
    entity_type: str,
    candidates: list[tuple[str, str]],
    max_choices: int = 2,
    context: str = "",
) -> tuple[str, str]:
    candidate_lines = "\n".join(
        f"- code={code} | {label}" for code, label in candidates
    )
    user_prompt = (
        f'Mention: "{entity_text}" (type={entity_type})\n'
        f'Context: "{context}"\n'
        f"Candidates:\n{candidate_lines}\n"
        f"Choose at most {max_choices} supplied code(s), or []. Return JSON only."
    )
    return _CANDIDATE_SELECTOR_SYSTEM, user_prompt
