"""Rule-based reranker: dense/lexical score + feature so khớp mention <-> candidate.

Khi clonazepam bị phạt sai -> sửa compare_strength().
Khi nystatin bị ép sang SCD -> sửa tty_prior() / TTY_BONUS_TABLE.
Khi felodipine chen vào amlodipine -> sửa ingredient_gate().
"""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from . import config
from .parser import normalize_text
from .schemas import ParsedDrugMention, RxNormCandidate


class RxNormRuleReranker:
    # ------------------------------------------------------------
    # Specificity của mention (dùng để chọn TTY_BONUS_TABLE phù hợp)
    # ------------------------------------------------------------

    @staticmethod
    def mention_specificity(parsed: ParsedDrugMention) -> str:
        has_strength = bool(parsed.strengths)
        has_form = bool(parsed.dose_forms)

        if has_strength and has_form:
            return "full_product"
        if has_strength:
            return "ingredient_strength"
        if has_form:
            return "ingredient_form"
        return "ingredient_only"

    # ------------------------------------------------------------
    # Ingredient gate: candidate có đúng hoạt chất với mention không
    # ------------------------------------------------------------

    def ingredient_gate(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> str:
        names: list[str] = []

        for ingredient in candidate.structured.get("ingredients", []):
            name = ingredient.get("name") if isinstance(ingredient, dict) else ingredient
            if name:
                names.append(str(name))

        for precise in candidate.structured.get("precise_ingredients", []):
            name = precise.get("name") if isinstance(precise, dict) else precise
            if name:
                names.append(str(name))

        for brand in candidate.structured.get("brands", []):
            name = brand.get("name") if isinstance(brand, dict) else brand
            if name:
                names.append(str(name))

        if not names and candidate.name:
            names.append(candidate.name)

        if not parsed.ingredient_core:
            return "unknown"

        core = parsed.ingredient_core
        best = 0.0

        for name in names:
            normalized_name = normalize_text(name)

            # Token-boundary match only.  Plain substring matching promoted
            # short unrelated names (e.g. a 3-letter ingredient hidden inside
            # a Vietnamese adjective) to an exact drug match.
            if normalized_name and re.search(
                rf"(?<!\w){re.escape(normalized_name)}(?!\w)", core
            ):
                return "exact"

            if len(normalized_name) >= 4 and len(core) >= 4:
                best = max(best, fuzz.partial_ratio(normalized_name, core) / 100.0)

        if best >= 0.9:
            return "exact"
        if best >= 0.6:
            return "partial"

        return "mismatch"

    def ingredient_match_score(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> float:
        relation = self.ingredient_gate(parsed, candidate)

        return {"exact": 1.0, "partial": 0.6, "unknown": 0.5, "mismatch": 0.0}[relation]

    # ------------------------------------------------------------
    # So khớp strength / dose form / release type
    # ------------------------------------------------------------

    def compare_strength(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> str:
        candidate_strengths = candidate.structured.get("strengths", [])
        candidate_display = [normalize_text(str(s)) for s in candidate_strengths]

        if not parsed.strengths and not candidate_display:
            return "both_missing"
        if not parsed.strengths:
            return "candidate_more_specific"
        if not candidate_display:
            return "mention_more_specific"

        mention_display = [normalize_text(s) for s in parsed.strengths]

        if set(mention_display) & set(candidate_display):
            return "exact"

        return "order_dose_mismatch"

    def compare_dose_form(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> str:
        candidate_forms = [
            normalize_text(str(f)) for f in candidate.structured.get("dose_forms", [])
        ]
        mention_forms = [normalize_text(f) for f in parsed.dose_forms]

        if not mention_forms and not candidate_forms:
            return "both_missing"
        if not mention_forms:
            return "candidate_more_specific"
        if not candidate_forms:
            return "mention_more_specific"
        if set(mention_forms) & set(candidate_forms):
            return "exact"

        return "mismatch"

    def compare_release(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> str:
        candidate_release = [
            normalize_text(str(r)) for r in candidate.structured.get("release_types", [])
        ]
        mention_release = [normalize_text(r) for r in parsed.release_types]

        if not mention_release and not candidate_release:
            return "both_missing"
        if set(mention_release) & set(candidate_release):
            return "exact"
        if mention_release and not candidate_release:
            return "mention_more_specific"
        if candidate_release and not mention_release:
            return "candidate_more_specific"

        return "mismatch"

    # ------------------------------------------------------------
    # TTY prior theo specificity
    # ------------------------------------------------------------

    def tty_prior(self, specificity: str, tty: str) -> float:
        return config.TTY_BONUS_TABLE.get(specificity, {}).get(tty, 0.0)

    # ------------------------------------------------------------
    # Feature extraction + score
    # ------------------------------------------------------------

    def extract_features(
        self, parsed: ParsedDrugMention, candidate: RxNormCandidate
    ) -> dict[str, Any]:
        return {
            "ingredient_relation": self.ingredient_gate(parsed, candidate),
            "strength_relation": self.compare_strength(parsed, candidate),
            "form_relation": self.compare_dose_form(parsed, candidate),
            "release_relation": self.compare_release(parsed, candidate),
            "specificity": self.mention_specificity(parsed),
        }

    def score_candidate(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> float:
        features = self.extract_features(parsed, candidate)
        candidate.features = features

        ingredient_score = self.ingredient_match_score(parsed, candidate)

        if features["ingredient_relation"] == "mismatch":
            # gate cứng: sai hoạt chất thì không cho lên top dù dense cao
            candidate.final_score = candidate.dense_score * 0.05
            return candidate.final_score

        score = (
            config.DENSE_WEIGHT * candidate.dense_score
            + config.LEXICAL_WEIGHT * candidate.lexical_score
        )

        if candidate.exact_term_match or features["ingredient_relation"] == "exact":
            score += config.INGREDIENT_EXACT_BONUS * ingredient_score

        if features["strength_relation"] == "exact":
            score += 0.10
        elif features["strength_relation"] == "order_dose_mismatch":
            score -= 0.15

        if features["form_relation"] == "exact":
            score += 0.05
        elif features["form_relation"] == "mismatch":
            score -= 0.05

        if features["release_relation"] == "exact":
            score += 0.03
        elif features["release_relation"] == "mismatch":
            score -= 0.03

        score += self.tty_prior(features["specificity"], candidate.tty)

        if candidate.historical:
            score -= 0.02  # ưu tiên nhẹ cho active/current trước historical

        candidate.final_score = score
        return score

    def rerank(
        self, parsed: ParsedDrugMention, candidates: list[RxNormCandidate]
    ) -> list[RxNormCandidate]:
        for candidate in candidates:
            self.score_candidate(parsed, candidate)

        return sorted(candidates, key=lambda c: c.final_score, reverse=True)
