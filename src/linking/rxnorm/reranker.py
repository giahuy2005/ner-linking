"""Structured RxNorm reranking with hard conflicts and calibrated support."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from rapidfuzz import fuzz

from . import config
from .parser import normalize_text
from .schemas import ParsedDrugMention, RxNormCandidate


class RxNormRuleReranker:
    _STRENGTH_RE = re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mcg|mg|g|meq|iu|units?|unit)\b", re.I
    )
    _UNIT_FACTOR = {
        "mcg": Decimal("0.001"), "mg": Decimal("1"), "g": Decimal("1000"),
        "meq": Decimal("1"), "iu": Decimal("1"), "unit": Decimal("1"),
        "units": Decimal("1"),
    }

    @staticmethod
    def mention_specificity(parsed: ParsedDrugMention) -> str:
        if parsed.strengths and parsed.dose_forms:
            return "full_product"
        if parsed.strengths:
            return "ingredient_strength"
        if parsed.dose_forms:
            return "ingredient_form"
        return "ingredient_only"

    @classmethod
    def _strength_values(cls, values: list[str]) -> list[tuple[Decimal, str]]:
        output = []
        for item in values:
            for match in cls._STRENGTH_RE.finditer(normalize_text(str(item))):
                unit = match.group("unit").lower()
                try:
                    number = Decimal(match.group("value")) * cls._UNIT_FACTOR[unit]
                except (InvalidOperation, KeyError):
                    continue
                family = "mass" if unit in {"mcg", "mg", "g"} else unit.rstrip("s")
                output.append((number, family))
        return output

    def ingredient_gate(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> str:
        names = []
        for key in ("ingredients", "precise_ingredients", "brands"):
            for value in candidate.structured.get(key, []):
                name = value.get("name") if isinstance(value, dict) else value
                if name:
                    names.append(str(name))
        if not names and candidate.name:
            names.append(candidate.name)
        if not parsed.ingredient_core:
            return "unknown"
        core = parsed.ingredient_core
        best = 0.0
        aliases = [core, *parsed.ingredient_aliases]
        for name in names:
            normalized = normalize_text(name)
            for alias in aliases:
                if normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", alias):
                    return "exact"
                if len(normalized) >= 4 and len(alias) >= 4:
                    best = max(best, fuzz.partial_ratio(normalized, alias) / 100.0)
        return "exact" if best >= .9 else "partial" if best >= .6 else "mismatch"

    def compare_strength(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> str:
        candidate_strengths = [str(value) for value in candidate.structured.get("strengths", [])]
        if not parsed.strengths and not candidate_strengths:
            return "both_missing"
        if not parsed.strengths:
            return "candidate_more_specific"
        if not candidate_strengths:
            return "mention_more_specific"
        mention_values = self._strength_values(parsed.strengths)
        candidate_values = self._strength_values(candidate_strengths)
        if set(mention_values) & set(candidate_values):
            return "exact"
        if parsed.strength_role == "range" and len(mention_values) >= 2:
            lower, upper = sorted(value for value, _family in mention_values[:2])
            families = {family for _value, family in mention_values[:2]}
            if any(family in families and lower <= value <= upper for value, family in candidate_values):
                return "range_contains"
        if parsed.strength_role == "range":
            match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(mcg|mg|g|meq|iu|units?|unit)", parsed.strengths[0], re.I)
            if match:
                unit = match.group(3).lower()
                factor = self._UNIT_FACTOR.get(unit)
                if factor is not None:
                    lower = Decimal(match.group(1)) * factor
                    upper = Decimal(match.group(2)) * factor
                    family = "mass" if unit in {"mcg", "mg", "g"} else unit.rstrip("s")
                    if any(candidate_family == family and lower <= value <= upper
                           for value, candidate_family in candidate_values):
                        return "range_contains"
        if not parsed.dose_forms and any(
            mf == cf and cv > 0 and cv <= mv
            for mv, mf in mention_values for cv, cf in candidate_values
        ):
            return "dose_interpretation"
        return "conflict"

    @staticmethod
    def _compare_terms(mention: list[str], candidate: list[str]) -> str:
        mention_values = {normalize_text(value) for value in mention}
        candidate_values = {normalize_text(value) for value in candidate}
        if not mention_values and not candidate_values:
            return "both_missing"
        if not mention_values:
            return "candidate_more_specific"
        if not candidate_values:
            return "mention_more_specific"
        return "exact" if mention_values & candidate_values else "conflict"

    def extract_features(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> dict[str, Any]:
        previous = candidate.features or {}
        dose_forms = [normalize_text(str(value)) for value in candidate.structured.get("dose_forms", [])]
        route_support = "unknown"
        if parsed.route == "PO":
            route_support = "exact" if any("oral" in value for value in dose_forms) else "conflict" if dose_forms else "unknown"
        strength_endpoint = None
        if parsed.strength_role == "range" and parsed.strengths:
            match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(mcg|mg|g|meq|iu|units?|unit)", parsed.strengths[0], re.I)
            candidate_values = self._strength_values(candidate.structured.get("strengths", []))
            if match:
                unit = match.group(3).lower(); factor = self._UNIT_FACTOR.get(unit)
                if factor is not None:
                    low, high = Decimal(match.group(1)) * factor, Decimal(match.group(2)) * factor
                    values = {value for value, _family in candidate_values}
                    strength_endpoint = "lower" if low in values else "upper" if high in values else None
        return {
            "ingredient_relation": self.ingredient_gate(parsed, candidate),
            "strength_relation": self.compare_strength(parsed, candidate),
            "form_relation": self._compare_terms(parsed.dose_forms, candidate.structured.get("dose_forms", [])),
            "release_relation": self._compare_terms(parsed.release_types, candidate.structured.get("release_types", [])),
            "specificity": self.mention_specificity(parsed),
            "exact_structured_match": bool(previous.get("exact_structured_match")),
            "query_source_count": len(candidate.retrieval_sources),
            "matched_term_count": len(candidate.matched_terms),
            "active": candidate.active,
            "historical": candidate.historical,
            "route_support": route_support,
            "strength_endpoint": strength_endpoint,
        }

    def score_candidate(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> float:
        features = self.extract_features(parsed, candidate)
        candidate.features = features
        conflicts = []
        if features["ingredient_relation"] == "mismatch": conflicts.append("ingredient_mismatch")
        if parsed.release_types and features["release_relation"] == "conflict": conflicts.append("explicit_release_conflict")
        if parsed.dose_forms and features["form_relation"] == "conflict": conflicts.append("explicit_form_conflict")
        if parsed.strengths and parsed.dose_forms and features["strength_relation"] == "conflict": conflicts.append("explicit_strength_conflict")
        if not candidate.active and not candidate.current_rxcuis: conflicts.append("inactive_without_current_mapping")
        candidate.rejection_reasons = conflicts
        if conflicts:
            candidate.support_level = "rejected"
            candidate.final_score = -1.0 + candidate.dense_score * .01
            return candidate.final_score

        score = config.DENSE_WEIGHT * candidate.dense_score + config.LEXICAL_WEIGHT * candidate.lexical_score
        if candidate.exact_term_match or features["ingredient_relation"] == "exact": score += config.INGREDIENT_EXACT_BONUS
        score += {"exact": .10, "range_contains": .075, "dose_interpretation": .025, "conflict": -.15}.get(features["strength_relation"], 0)
        score += {"exact": .05, "conflict": -.05}.get(features["form_relation"], 0)
        score += {"exact": .03, "conflict": -.03}.get(features["release_relation"], 0)
        if parsed.route == "PO":
            forms = [normalize_text(str(value)) for value in candidate.structured.get("dose_forms", [])]
            if any("oral tablet" in value for value in forms): score += .08
            elif any("oral" in value for value in forms): score += .025
            elif forms: score -= .12
        if features.get("strength_endpoint") == "lower": score += .04
        elif features.get("strength_endpoint") == "upper": score -= .01
        score += config.TTY_BONUS_TABLE.get(features["specificity"], {}).get(candidate.tty, 0.0)
        if candidate.historical: score -= .02
        if features["exact_structured_match"]: score += .05
        brand_exact = any(normalize_text(str(brand)) in parsed.normalized_text for brand in candidate.structured.get("brands", []))
        if candidate.tty == "SBD" and not brand_exact: score -= .08
        candidate.evidence_completeness = sum((
            features["ingredient_relation"] == "exact",
            features["strength_relation"] in {"exact", "range_contains", "dose_interpretation", "both_missing"},
            features["form_relation"] in {"exact", "both_missing", "candidate_more_specific"},
            features["release_relation"] in {"exact", "both_missing", "candidate_more_specific"},
        )) / 4.0
        if candidate.exact_term_match and candidate.evidence_completeness >= .75:
            candidate.support_level = "exact"
        elif features["ingredient_relation"] == "exact" and candidate.evidence_completeness >= .75:
            candidate.support_level = "strong"
        elif features["ingredient_relation"] in {"exact", "partial"} and score >= .45:
            candidate.support_level = "medium"
        else:
            candidate.support_level = "weak"
        if "bare_liquid_unit_without_quantity" in parsed.parse_warnings:
            candidate.support_level = "medium" if candidate.support_level in {"exact", "strong"} else candidate.support_level
        candidate.final_score = score
        return score

    def rerank(self, parsed: ParsedDrugMention, candidates: list[RxNormCandidate]) -> list[RxNormCandidate]:
        for candidate in candidates:
            self.score_candidate(parsed, candidate)
        ranked = sorted(candidates, key=lambda item: item.final_score, reverse=True)
        margin = ranked[0].final_score - ranked[1].final_score if len(ranked) > 1 else float("inf")
        for candidate in ranked:
            candidate.top1_margin = margin
        return ranked
