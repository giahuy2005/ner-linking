"""Structured RxNorm reranking with one authoritative support policy."""

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
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mcg|mg|g|gm|meq|iu|unt|units?|unit)\b",
        re.I,
    )
    _UNIT_FACTOR = {
        "mcg": Decimal("0.001"),
        "mg": Decimal("1"),
        "g": Decimal("1000"),
        "gm": Decimal("1000"),
        "meq": Decimal("1"),
        "iu": Decimal("1"),
        "unt": Decimal("1"),
        "unit": Decimal("1"),
        "units": Decimal("1"),
    }

    @staticmethod
    def mention_specificity(parsed: ParsedDrugMention) -> str:
        if parsed.strengths and parsed.dose_forms:
            return "full_product"
        if parsed.strengths:
            return "ingredient_strength"
        if parsed.dose_forms or parsed.route:
            return "ingredient_form"
        return "ingredient_only"

    @classmethod
    def _strength_values(cls, values: list[str]) -> list[tuple[Decimal, str]]:
        output: list[tuple[Decimal, str]] = []
        for item in values:
            for match in cls._STRENGTH_RE.finditer(normalize_text(str(item))):
                unit = match.group("unit").lower()
                try:
                    number = Decimal(match.group("value")) * cls._UNIT_FACTOR[unit]
                except (InvalidOperation, KeyError):
                    continue
                family = "mass" if unit in {"mcg", "mg", "g", "gm"} else unit.rstrip("s")
                output.append((number, family))
        return output

    @staticmethod
    def _candidate_names(candidate: RxNormCandidate) -> tuple[list[str], list[str]]:
        ingredients: list[str] = []
        brands: list[str] = []
        for key in ("ingredients", "precise_ingredients"):
            for value in candidate.structured.get(key, []) or []:
                name = value.get("name") if isinstance(value, dict) else value
                if name:
                    ingredients.append(normalize_text(str(name)))
        for value in candidate.structured.get("brands", []) or []:
            name = value.get("name") if isinstance(value, dict) else value
            if name:
                brands.append(normalize_text(str(name)))
        if not ingredients and candidate.tty in {"IN", "PIN", "MIN"}:
            ingredients.append(normalize_text(candidate.structured.get("name") or candidate.name))
        return list(dict.fromkeys(filter(None, ingredients))), list(dict.fromkeys(filter(None, brands)))

    @staticmethod
    def _safe_fuzzy(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        left_compact = re.sub(r"\W+", "", left)
        right_compact = re.sub(r"\W+", "", right)
        if not left_compact or not right_compact:
            return 0.0
        left_digits = re.findall(r"\d+", left_compact)
        right_digits = re.findall(r"\d+", right_compact)
        if left_digits != right_digits:
            return 0.0
        # Mixed alpha-numeric clinical names are identifiers, not fuzzy prose.
        # Do not confuse vitamin 3B with vitamin K3/B3 or similarly shaped codes.
        mixed = lambda value: {
            token for token in re.findall(r"[a-z]*\d+[a-z]*", value)
            if any(char.isalpha() for char in token)
        }
        left_mixed, right_mixed = mixed(left_compact), mixed(right_compact)
        if (left_mixed or right_mixed) and left_mixed != right_mixed:
            return 0.0
        if left_compact[0] != right_compact[0] or abs(len(left_compact) - len(right_compact)) > 4:
            return 0.0
        return fuzz.ratio(left_compact, right_compact) / 100.0

    @classmethod
    def _component_match(cls, component: str, names: list[str]) -> float:
        component = normalize_text(component)
        if not component:
            return 0.0
        component_tokens = set(component.split())
        best = 0.0
        for name in names:
            name = normalize_text(name)
            if component == name:
                return 1.0
            name_tokens = set(name.split())
            if component_tokens and name_tokens:
                overlap = len(component_tokens & name_tokens) / max(len(component_tokens), len(name_tokens))
                # Containment is useful only for meaningful multi-character names.
                if min(len(component), len(name)) >= 4 and (component in name or name in component):
                    best = max(best, 0.92 if overlap >= 0.5 else 0.82)
                best = max(best, overlap * 0.85)
            best = max(best, cls._safe_fuzzy(component, name))
        return best

    def ingredient_gate(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> str:
        if "generic_drug_class" in parsed.parse_warnings:
            return "generic_class"
        if "negated_or_excluded_ingredient" in parsed.parse_warnings:
            return "negated"
        components = parsed.ingredient_components or (
            [parsed.ingredient_core] if parsed.ingredient_core else []
        )
        if not components:
            return "unknown"
        ingredients, brands = self._candidate_names(candidate)
        all_names = [*ingredients, *brands, normalize_text(candidate.name)]
        scores = [self._component_match(component, all_names) for component in components]
        matched = sum(score >= 0.88 for score in scores)
        partial = sum(score >= 0.68 for score in scores)

        # A multi-drug span may link only to a true combination concept covering
        # every explicit component. Selecting one constituent is always unsafe.
        if len(components) >= 2:
            if matched == len(components):
                return "exact"
            if partial:
                return "incomplete_combination"
            return "mismatch"

        score = scores[0]
        if score >= 0.88:
            return "exact"
        if score >= 0.68:
            return "partial"
        return "mismatch"

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
        if parsed.strength_role == "range" and mention_values:
            by_family: dict[str, list[Decimal]] = {}
            for value, family in mention_values:
                by_family.setdefault(family, []).append(value)
            for family, values in by_family.items():
                if len(values) < 2:
                    continue
                lower, upper = min(values), max(values)
                if any(cf == family and lower <= cv <= upper for cv, cf in candidate_values):
                    return "range_contains"
        # Some source mentions encode ordered amount rather than concentration.
        # Keep this as soft evidence only when no dose form is explicit.
        if not parsed.dose_forms and any(
            mf == cf and cv > 0 and cv <= mv
            for mv, mf in mention_values for cv, cf in candidate_values
        ):
            return "dose_interpretation"
        return "conflict"

    @staticmethod
    def _compare_terms(mention: list[str], candidate: list[str]) -> str:
        mention_values = {normalize_text(value) for value in mention if value}
        candidate_values = {normalize_text(value) for value in candidate if value}
        if not mention_values and not candidate_values:
            return "both_missing"
        if not mention_values:
            return "candidate_more_specific"
        if not candidate_values:
            return "mention_more_specific"
        for left in mention_values:
            for right in candidate_values:
                if left == right or left in right or right in left:
                    return "exact"
        return "conflict"

    @staticmethod
    def _route_relation(parsed: ParsedDrugMention, candidate: RxNormCandidate) -> str:
        if not parsed.route:
            return "unknown"
        forms = [
            normalize_text(str(value))
            for value in candidate.structured.get("dose_forms", []) or []
        ]
        forms.append(normalize_text(candidate.name))
        haystack = " ".join(forms)
        compatible = config.ROUTE_FORM_KEYWORDS.get(parsed.route, ())
        conflicts = config.ROUTE_CONFLICT_KEYWORDS.get(parsed.route, ())
        if any(keyword in haystack for keyword in compatible):
            return "exact"
        if any(keyword in haystack for keyword in conflicts):
            return "conflict"
        return "unknown"

    def extract_features(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> dict[str, Any]:
        previous = candidate.features or {}
        ingredient_relation = self.ingredient_gate(parsed, candidate)
        strength_relation = self.compare_strength(parsed, candidate)
        form_relation = self._compare_terms(
            parsed.dose_forms,
            [str(value) for value in candidate.structured.get("dose_forms", [])],
        )
        release_relation = self._compare_terms(
            parsed.release_types,
            [str(value) for value in candidate.structured.get("release_types", [])],
        )
        route_relation = self._route_relation(parsed, candidate)
        specificity = self.mention_specificity(parsed)
        product_candidate = candidate.tty in config.PRODUCT_TTYS
        overspecific_product = bool(
            specificity == "ingredient_only"
            and len(parsed.ingredient_components) <= 1
            and product_candidate
            and not candidate.exact_term_match
        )
        return {
            "ingredient_relation": ingredient_relation,
            "ingredient_component_count": len(parsed.ingredient_components),
            "strength_relation": strength_relation,
            "form_relation": form_relation,
            "release_relation": release_relation,
            "specificity": specificity,
            "exact_structured_match": bool(previous.get("exact_structured_match")),
            "query_source_count": len(candidate.retrieval_sources),
            "matched_term_count": len(candidate.matched_terms),
            "active": candidate.active,
            "historical": candidate.historical,
            "route_support": route_relation,
            "overspecific_product": overspecific_product,
        }

    def score_candidate(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> float:
        features = self.extract_features(parsed, candidate)
        candidate.features = features
        conflicts: list[str] = []
        ingredient = features["ingredient_relation"]
        if ingredient == "generic_class":
            conflicts.append("generic_drug_class")
        elif ingredient == "negated":
            conflicts.append("negated_or_excluded_ingredient")
        elif ingredient == "mismatch":
            conflicts.append("ingredient_mismatch")
        elif ingredient == "incomplete_combination":
            conflicts.append("incomplete_combination_coverage")
        if parsed.release_types and features["release_relation"] == "conflict":
            conflicts.append("explicit_release_conflict")
        if parsed.dose_forms and features["form_relation"] == "conflict":
            conflicts.append("explicit_form_conflict")
        if parsed.strengths and features["strength_relation"] == "conflict":
            conflicts.append("explicit_strength_conflict")
        if parsed.route and features["route_support"] == "conflict":
            conflicts.append("explicit_route_conflict")
        if not candidate.active and not candidate.current_rxcuis:
            conflicts.append("inactive_without_current_mapping")
        candidate.rejection_reasons = list(dict.fromkeys(conflicts))
        if candidate.rejection_reasons:
            candidate.support_level = "rejected"
            candidate.final_score = -1.0 + candidate.dense_score * 0.01
            candidate.evidence_completeness = 0.0
            return candidate.final_score

        score = (
            config.DENSE_WEIGHT * candidate.dense_score
            + config.LEXICAL_WEIGHT * candidate.lexical_score
        )
        if candidate.exact_term_match:
            score += 0.30
        if ingredient == "exact":
            score += config.INGREDIENT_EXACT_BONUS
        elif ingredient == "partial":
            score += 0.08
        score += {
            "exact": 0.13,
            "range_contains": 0.09,
            "dose_interpretation": 0.02,
            "mention_more_specific": -0.05,
            "candidate_more_specific": -0.04,
        }.get(features["strength_relation"], 0.0)
        score += {"exact": 0.06, "candidate_more_specific": -0.02}.get(
            features["form_relation"], 0.0
        )
        score += {"exact": 0.04, "candidate_more_specific": -0.01}.get(
            features["release_relation"], 0.0
        )
        score += {"exact": 0.10, "unknown": -0.01}.get(features["route_support"], 0.0)
        score += config.TTY_BONUS_TABLE.get(features["specificity"], {}).get(candidate.tty, 0.0)
        if candidate.historical:
            score -= 0.04
        if features["exact_structured_match"]:
            score += 0.06
        if features["overspecific_product"]:
            score -= 0.12

        normalized_mention = parsed.normalized_text
        brand_exact = any(
            normalize_text(str(brand)) == normalized_mention
            or normalize_text(str(brand)) in parsed.brand_hints
            for brand in candidate.structured.get("brands", []) or []
        )
        if candidate.tty == "SBD" and not brand_exact and not candidate.exact_term_match:
            score -= 0.10

        ingredient_ok = ingredient == "exact"
        strength_ok = features["strength_relation"] in {
            "exact", "range_contains", "dose_interpretation", "both_missing",
        }
        form_ok = features["form_relation"] in {
            "exact", "both_missing", "candidate_more_specific",
        }
        release_ok = features["release_relation"] in {
            "exact", "both_missing", "candidate_more_specific",
        }
        route_ok = features["route_support"] in {"exact", "unknown"}
        candidate.evidence_completeness = sum(
            (ingredient_ok, strength_ok, form_ok, release_ok, route_ok)
        ) / 5.0

        exact_granularity = bool(
            candidate.exact_term_match
            and ingredient_ok
            and not features["overspecific_product"]
        )
        strong_granularity = bool(
            ingredient_ok
            and candidate.evidence_completeness >= 0.8
            and not features["overspecific_product"]
            and (
                candidate.tty in {"IN", "PIN", "MIN", "BN"}
                or features["specificity"] != "ingredient_only"
            )
        )
        if exact_granularity:
            candidate.support_level = "exact"
        elif strong_granularity and score >= 0.52:
            candidate.support_level = "strong"
        elif ingredient in {"exact", "partial"} and score >= 0.45:
            candidate.support_level = "medium"
        else:
            candidate.support_level = "weak"
        if "bare_liquid_unit_without_quantity" in parsed.parse_warnings and candidate.support_level == "strong":
            candidate.support_level = "medium"
        candidate.final_score = score
        return score

    def rerank(self, parsed: ParsedDrugMention, candidates: list[RxNormCandidate]) -> list[RxNormCandidate]:
        for candidate in candidates:
            self.score_candidate(parsed, candidate)
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.support_level != "rejected",
                {"exact": 4, "strong": 3, "medium": 2, "weak": 1, "rejected": 0}.get(item.support_level, 0),
                item.final_score,
                item.dense_score,
            ),
            reverse=True,
        )
        supported = [item for item in ranked if item.support_level in {"exact", "strong", "medium"}]
        margin = (
            supported[0].final_score - supported[1].final_score
            if len(supported) > 1 else float("inf")
        )
        for candidate in ranked:
            candidate.top1_margin = margin
        return ranked