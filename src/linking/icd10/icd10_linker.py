#!/usr/bin/env python3
"""High-recall SapBERT/FAISS ICD-10 retrieval with calibrated evidence.

The linker is intentionally split into two responsibilities:

1. ``link``/``link_many`` retrieve a broad, evidence-rich whitelist.
2. ``predict``/``predict_many`` provide a conservative deterministic policy.

The production Qwen selector consumes the whitelist from ``link_many``. It must
not invent a code and must not reapply an older lexical hard gate after Qwen has
selected a supported code.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

try:
    from ..sapbert_encoder import SapBertEncoder, clean_query_text
except ImportError:  # Support direct execution by file path.
    from viettel_ai_ner.src.linking.sapbert_encoder import SapBertEncoder, clean_query_text

try:
    from . import config
except ImportError:  # Support direct execution by file path.
    from viettel_ai_ner.src.linking.icd10 import config


_SUPPORT_RANK = {"exact": 3, "strong": 2, "medium": 1, "weak": 0, "rejected": -1}
_TERM_TYPE_BONUS = {"preferred": 0.012, "inclusion": 0.008, "alias": 0.006}


def _normalize_alias(text: str) -> str:
    value = unicodedata.normalize("NFC", clean_query_text(text)).casefold()
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:()[]{}")


def _fold_ascii(text: str) -> str:
    value = unicodedata.normalize("NFD", _normalize_alias(text)).replace("đ", "d").replace("Đ", "D")
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def _semantic_forms(text: str) -> list[str]:
    """Return general lexical variants without changing the entity text."""
    normalized = _normalize_alias(text)
    forms = [normalized]
    for source, target in config.SEMANTIC_PHRASE_VARIANTS:
        if source in normalized:
            forms.append(normalized.replace(source, target))
    return list(dict.fromkeys(value for value in forms if value))


def _token_text(text: str) -> str:
    value = _fold_ascii(text)
    value = re.sub(
        r"\bkhong\s+(?:xac\s+dinh|dac\s+hieu|biet\s+dinh|phan\s+loai)\b",
        " ",
        value,
    )
    value = re.sub(r"\bchua\s+phan\s+loai\b", " ", value)
    value = re.sub(
        r"\b(?:unspecified|not\s+specified|not\s+otherwise\s+specified|nos)\b",
        " ",
        value,
    )
    return re.sub(r"\s+", " ", value).strip()


def _meaningful_token_sequence(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", _token_text(text))
        if token not in config.NON_SEMANTIC_TOKENS and not token.isdigit()
    ]


def _meaningful_tokens(text: str) -> set[str]:
    return set(_meaningful_token_sequence(text))


def _is_abbreviation(text: str) -> bool:
    value = text.strip()
    if re.search(r"\s", value):
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    return bool(compact and compact.upper() == compact and 2 <= len(compact) <= 12)


def _acronym_tokens(text: str) -> set[str]:
    result = set()
    for token in re.findall(r"[A-Za-z0-9]+", text):
        if (token.upper() == token and len(token) >= 2) or (
            any(char.isdigit() for char in token) and len(token) >= 2
        ):
            result.add(token.casefold())
    return result


def _best_lexical_features(mention: str, label: str) -> dict[str, Any]:
    """Compare every general semantic form and retain the best alignment."""
    best: dict[str, Any] | None = None
    mention_forms = _semantic_forms(mention)
    label_forms = _semantic_forms(label)
    for mention_form in mention_forms:
        mention_sequence = _meaningful_token_sequence(mention_form)
        mention_tokens = set(mention_sequence)
        for label_form in label_forms:
            label_sequence = _meaningful_token_sequence(label_form)
            label_tokens = set(label_sequence)
            overlap = mention_tokens & label_tokens
            mention_coverage = len(overlap) / len(mention_tokens) if mention_tokens else 0.0
            candidate_coverage = len(overlap) / len(label_tokens) if label_tokens else 0.0
            extra_tokens = sorted(label_tokens - mention_tokens)
            extra_ratio = len(extra_tokens) / max(1, len(label_tokens))
            similarity = SequenceMatcher(None, _fold_ascii(mention_form), _fold_ascii(label_form)).ratio()
            containment = bool(
                _fold_ascii(mention_form) in _fold_ascii(label_form)
                or _fold_ascii(label_form) in _fold_ascii(mention_form)
            )
            mention_acronyms = _acronym_tokens(mention)
            label_folded = _fold_ascii(label_form)
            abbreviation_support = bool(
                (_is_abbreviation(mention) and any(token in label_tokens for token in mention_tokens))
                or any(acronym in label_folded for acronym in mention_acronyms)
            )
            score = (
                0.42 * mention_coverage
                + 0.24 * candidate_coverage
                + 0.20 * similarity
                + 0.08 * float(containment)
                + 0.06 * float(abbreviation_support)
            )
            current = {
                "mention_coverage": mention_coverage,
                "candidate_coverage": candidate_coverage,
                "extra_tokens": extra_tokens,
                "extra_token_ratio": extra_ratio,
                "lexical_similarity": similarity,
                "phrase_containment": containment,
                "abbreviation_support": abbreviation_support,
                "catalogue_exact_match": bool(
                    len(mention_sequence) >= 2 and mention_sequence == label_sequence
                ),
                "mention_token_count": len(mention_sequence),
                "mention_form": mention_form,
                "label_form": label_form,
                "lexical_alignment_score": score,
            }
            if best is None or current["lexical_alignment_score"] > best["lexical_alignment_score"]:
                best = current
    return best or {
        "mention_coverage": 0.0,
        "candidate_coverage": 0.0,
        "extra_tokens": [],
        "extra_token_ratio": 1.0,
        "lexical_similarity": 0.0,
        "phrase_containment": False,
        "abbreviation_support": False,
        "catalogue_exact_match": False,
        "mention_token_count": len(_meaningful_token_sequence(mention)),
        "mention_form": _normalize_alias(mention),
        "label_form": _normalize_alias(label),
        "lexical_alignment_score": 0.0,
    }


def _code_key(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", code).upper()


def _is_ancestor_code(ancestor: str, descendant: str) -> bool:
    left, right = _code_key(ancestor), _code_key(descendant)
    return bool(left != right and right.startswith(left) and len(right) > len(left))


def aggregate_term_results(
    term_results: Iterable[dict[str, Any]],
    *,
    top_k_codes: int,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Aggregate independent term/query evidence by ICD code.

    ``matched_terms`` is deliberately ``list[str]``. Rich evidence remains in
    ``matched_term_details`` so selector prompts stay compact and schema-stable.
    """
    if top_k_codes <= 0:
        raise ValueError("top_k_codes must be positive")

    by_code: dict[str, dict[str, Any]] = {}
    for result in term_results:
        score = float(result["score"])
        if min_score is not None and score < min_score:
            continue
        code = str(result["code"])
        evidence = {
            "text": str(result["text"]),
            "score": score,
            "language": str(result["language"]),
            "term_type": str(result["term_type"]),
            "term_id": str(result["term_id"]),
            "query_source": str(result.get("query_source", "raw")),
        }
        current = by_code.get(code)
        if current is None:
            current = by_code[code] = {
                "code": code,
                "score": score,
                "matched_term": evidence["text"],
                "language": evidence["language"],
                "term_type": evidence["term_type"],
                "term_id": evidence["term_id"],
                "matched_terms": [],
                "matched_term_details": [],
                "query_sources": [],
                "exact_alias_source": result.get("exact_alias_source"),
            }
        if score > float(current["score"]):
            current.update(
                score=score,
                matched_term=evidence["text"],
                language=evidence["language"],
                term_type=evidence["term_type"],
                term_id=evidence["term_id"],
            )
        if result.get("exact_alias_source"):
            current["exact_alias_source"] = result["exact_alias_source"]
        if evidence not in current["matched_term_details"]:
            current["matched_term_details"].append(evidence)
        if evidence["text"] not in current["matched_terms"]:
            current["matched_terms"].append(evidence["text"])
        source = evidence["query_source"]
        if source not in current["query_sources"]:
            current["query_sources"].append(source)

    for item in by_code.values():
        top_scores = sorted(
            (row["score"] for row in item["matched_term_details"]), reverse=True
        )[:3]
        item["top_n_mean"] = sum(top_scores) / len(top_scores)
        item["independent_term_count"] = len(
            {row["term_id"] for row in item["matched_term_details"]}
        )
        item["query_variant_count"] = len(item["query_sources"])
        item["aggregate_components"] = {
            "max_dense": item["score"],
            "top_n_mean": item["top_n_mean"],
            "term_count": item["independent_term_count"],
            "query_count": item["query_variant_count"],
        }
        item["aggregate_score"] = min(
            1.0,
            float(item["score"])
            + min(0.025, 0.005 * (item["independent_term_count"] - 1))
            + min(0.015, 0.005 * (item["query_variant_count"] - 1)),
        )

    ranked = sorted(
        by_code.values(), key=lambda item: (-float(item["aggregate_score"]), item["code"])
    )
    return ranked[:top_k_codes]


def _build_metadata_alias_index(metadata: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        normalized = _normalize_alias(str(row.get("text", "")))
        if not normalized:
            continue
        forms = {normalized}
        if row.get("term_type") == "preferred":
            shortened = re.sub(r"^(?:bệnh|hội chứng|chứng)\s+", "", normalized).strip()
            if len(shortened) >= 3:
                forms.add(shortened)
        for form in forms:
            aliases[form].add(str(row["code"]))
    return dict(aliases)


def _build_metadata_alias_codes(metadata: Iterable[dict[str, Any]]) -> dict[str, str]:
    aliases = _build_metadata_alias_index(metadata)
    return {
        alias: next(iter(codes))
        for alias, codes in aliases.items()
        if len(codes) == 1
    }


def _expected_chapters(mention: str) -> tuple[str, ...] | None:
    normalized = _normalize_alias(mention)
    for phrases, chapters in config.CHAPTER_HINTS:
        if any(phrase in normalized for phrase in phrases):
            return tuple(chapters)
    return None


def _query_variants(mention: str) -> list[str]:
    """Offset-independent retrieval variants; entity text is never changed."""
    raw = clean_query_text(mention)
    variants = [raw, *_semantic_forms(raw)]
    normalized = _normalize_alias(raw)
    stripped = re.sub(r"^(?:bệnh|hội chứng)\s+", "", normalized).strip()
    if len(stripped) >= 3 and stripped != normalized:
        variants.append(stripped)
    suffix_trimmed = re.sub(
        r"\s+(?:ở|trên|dưới)\s+(?:trẻ(?:\s+em)?|người\s+lớn|nam|nữ)$",
        "",
        normalized,
    ).strip()
    if len(suffix_trimmed) >= 3:
        variants.append(suffix_trimmed)
    parenthetical = re.search(r"\(([^()]{2,80})\)", raw)
    if parenthetical:
        variants.extend(
            [
                parenthetical.group(1),
                re.sub(r"\s*\([^()]+\)\s*", " ", raw).strip(),
            ]
        )
    punctuation = re.sub(r"[-_/]+", " ", normalized)
    variants.append(re.sub(r"\s+", " ", punctuation).strip())
    variants.append(_fold_ascii(normalized))
    return list(dict.fromkeys(value for value in variants if len(value) >= 2))


def _support_level(item: dict[str, Any]) -> str:
    if item.get("hard_conflicts"):
        return "rejected"
    if item.get("exact_alias_source") == "configured_exact_alias":
        return "exact"
    if (
        item.get("exact_alias_source") == "metadata_exact_unique"
        and item.get("exact_text_quality", True)
    ):
        return "exact"
    if item.get("normalized_exact_match") or item.get("catalogue_exact_match"):
        return "exact"
    score = float(item.get("support_score", 0.0))
    coverage = float(item.get("mention_coverage", 0.0))
    similarity = float(item.get("lexical_similarity", 0.0))
    if (
        score >= config.SUPPORT_STRONG_MIN_SCORE
        and (coverage >= 0.65 or item.get("abbreviation_support"))
    ) or (
        item.get("technical_containment")
        and score >= 0.70
        and coverage >= 0.90
    ):
        return "strong"
    if score >= config.SUPPORT_MEDIUM_MIN_SCORE and (
        coverage >= 0.42 or similarity >= 0.55
    ):
        return "medium"
    return "weak"


def _derive_hard_conflicts(mention: str, item: dict[str, Any]) -> list[str]:
    """Catalogue-level conflicts that are safe to enforce as hard guards."""
    code = str(item.get("code", "")).upper()
    normalized = _normalize_alias(mention)
    conflicts: list[str] = []
    status_terms = ("sàng lọc", "khám", "tư vấn", "tiền sử", "theo dõi", "tình trạng", "người mang", "tiếp xúc")
    pregnancy_terms = ("mang thai", "thai", "sau đẻ", "sau sinh", "sản phụ")
    neonatal_terms = ("sơ sinh", "chu sinh", "bẩm sinh", "thai nhi")
    injury_terms = ("tai nạn", "chấn thương", "ngộ độc", "tác dụng phụ", "phản ứng có hại", "quá liều")
    if code.startswith("Z") and not any(term in normalized for term in status_terms):
        conflicts.append("encounter_or_status_code_without_context")
    if code.startswith("O") and not any(term in normalized for term in pregnancy_terms):
        conflicts.append("obstetric_code_without_pregnancy_context")
    if code.startswith("P") and not any(term in normalized for term in neonatal_terms):
        conflicts.append("perinatal_code_without_neonatal_context")
    if code[:1] in {"V", "W", "X", "Y"} and not any(term in normalized for term in injury_terms):
        conflicts.append("external_cause_code_without_event_context")
    mention_sequence = _meaningful_token_sequence(mention)
    label_sequence = _meaningful_token_sequence(str(item.get("matched_term", "")))
    if "khong" in label_sequence and "khong" not in mention_sequence:
        conflicts.append("negative_qualifier_not_supported_by_mention")
    return conflicts


def _finalize_term_results(
    mention: str,
    term_results: list[dict[str, Any]],
    *,
    top_k_codes: int,
    min_score: float | None,
) -> list[dict[str, Any]]:
    ranked = aggregate_term_results(
        term_results,
        top_k_codes=max(top_k_codes, 64),
        min_score=min_score,
    )
    chapters = _expected_chapters(mention)
    for item in ranked:
        detail_rows = item.get("matched_term_details", []) or []
        best_detail = None
        best_features = None
        for detail in detail_rows:
            features = _best_lexical_features(mention, str(detail.get("text", "")))
            detail_rank = (
                features["lexical_alignment_score"],
                float(detail.get("score", 0.0)),
                _TERM_TYPE_BONUS.get(str(detail.get("term_type", "")), 0.0),
            )
            if best_detail is None or detail_rank > best_detail[0]:
                best_detail = (detail_rank, detail)
                best_features = features
        if best_detail is not None and best_features is not None:
            detail = best_detail[1]
            item["matched_term"] = detail["text"]
            item["language"] = detail["language"]
            item["term_type"] = detail["term_type"]
            item["term_id"] = detail["term_id"]
            item.update(best_features)
        else:
            item.update(_best_lexical_features(mention, str(item.get("matched_term", ""))))

        normalized_exact_raw = _normalize_alias(mention) == _normalize_alias(
            str(item.get("matched_term", ""))
        )
        exact_text_quality = bool(
            str(item.get("term_type", "")) == "preferred"
            or int(item.get("mention_token_count", 0)) >= 3
            or _is_abbreviation(mention)
            or _acronym_tokens(mention)
        )
        normalized_exact = bool(normalized_exact_raw and exact_text_quality)
        item["normalized_exact_match"] = normalized_exact
        item["exact_text_quality"] = exact_text_quality
        item["catalogue_exact_match"] = bool(
            item.get("catalogue_exact_match") and exact_text_quality
        )
        item["chapter_support"] = (
            item["code"].startswith(chapters) if chapters is not None else None
        )
        exact_bonus = 0.0
        if item.get("exact_alias_source") == "configured_exact_alias":
            exact_bonus = 0.18
        elif item.get("exact_alias_source") == "metadata_exact_unique" and exact_text_quality:
            exact_bonus = 0.14
        elif item.get("exact_alias_source") == "metadata_exact_ambiguous":
            exact_bonus = 0.06
        elif normalized_exact:
            exact_bonus = 0.10
        elif item.get("catalogue_exact_match"):
            exact_bonus = 0.08

        extra_ratio = float(item.get("extra_token_ratio", 1.0))
        abbreviation_support = bool(item.get("abbreviation_support"))
        specificity_penalty = min(0.16, 0.24 * extra_ratio)
        mention_ascii_tokens = _meaningful_tokens(mention)
        technical_containment = bool(
            item.get("phrase_containment")
            and float(item.get("mention_coverage", 0.0)) >= 0.90
            and (
                abbreviation_support
                or any(char.isdigit() for char in mention)
                or any(len(token) >= 10 for token in mention_ascii_tokens)
            )
        )
        item["technical_containment"] = technical_containment
        if abbreviation_support:
            specificity_penalty *= 0.20
        elif technical_containment:
            specificity_penalty *= 0.25
        over_specific = bool(
            bool(item.get("extra_tokens"))
            and not abbreviation_support
            and not normalized_exact
            and not item.get("catalogue_exact_match")
            and not technical_containment
            and item.get("exact_alias_source") != "metadata_exact_unique"
        )
        item["over_specific"] = over_specific
        item["hard_conflicts"] = list(dict.fromkeys([
            *list(item.get("hard_conflicts", [])),
            *_derive_hard_conflicts(mention, item),
        ]))

        dense = float(item.get("aggregate_score", item.get("score", 0.0)))
        support_score = (
            0.55 * dense
            + 0.20 * float(item.get("mention_coverage", 0.0))
            + 0.10 * float(item.get("candidate_coverage", 0.0))
            + 0.08 * float(item.get("lexical_similarity", 0.0))
            + 0.04 * float(bool(item.get("phrase_containment")))
            + 0.03 * min(1.0, float(item.get("query_variant_count", 0)) / 3.0)
            + 0.08 * float(abbreviation_support)
            + exact_bonus
            + _TERM_TYPE_BONUS.get(str(item.get("term_type", "")), 0.0)
            - specificity_penalty
        )
        if item.get("chapter_support") is True:
            support_score += 0.015
        item["support_score"] = max(0.0, min(1.0, support_score))

    # Soft hierarchy evidence: a parent is useful for an underspecified mention,
    # but this never removes a child from the retrieval whitelist.
    codes = [str(item["code"]) for item in ranked]
    for item in ranked:
        descendants = [code for code in codes if _is_ancestor_code(str(item["code"]), code)]
        item["hierarchy_has_descendants"] = bool(descendants)
        if descendants and item.get("over_specific"):
            item["support_score"] = min(1.0, float(item["support_score"]) + 0.012)
        item["support_level"] = _support_level(item)

    ranked.sort(
        key=lambda item: (
            -_SUPPORT_RANK.get(str(item.get("support_level", "weak")), 0),
            -float(item.get("support_score", 0.0)),
            -float(item.get("aggregate_score", item.get("score", 0.0))),
            str(item["code"]),
        )
    )
    for rank, item in enumerate(ranked, 1):
        item["support_rank"] = rank
    return ranked[:top_k_codes]


class Icd10Linker:
    """SapBERT candidate retriever with dense, exact and lexical union."""

    def __init__(
        self,
        index_dir: str | Path = config.DEFAULT_INDEX_DIR,
        *,
        device: str = config.DEFAULT_DEVICE,
        query_batch_size: int = config.DEFAULT_QUERY_BATCH_SIZE,
    ) -> None:
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive")
        self.index_dir = Path(index_dir).resolve()
        self.query_batch_size = query_batch_size
        config_path = self.index_dir / config.INDEX_CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(
                f"Index config not found: {config_path}. Run build_icd10_faiss_index.py first."
            )
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self._validate_config()

        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS is required. Install faiss-cpu (or faiss-gpu) first.") from exc

        artifacts = self.config["artifacts"]
        index_path = self._artifact_path(artifacts["index"])
        metadata_path = self._artifact_path(artifacts["metadata"])
        embeddings_path = self._artifact_path(artifacts["embeddings"])
        for path in (index_path, metadata_path, embeddings_path):
            if not path.exists():
                raise FileNotFoundError(f"Index artifact not found: {path}")

        self.index = faiss.read_index(str(index_path))
        self.metadata = self._load_metadata(metadata_path)
        self.metadata_alias_index = _build_metadata_alias_index(self.metadata)
        self.metadata_alias_codes = {
            alias: next(iter(codes))
            for alias, codes in self.metadata_alias_index.items()
            if len(codes) == 1
        }
        self._alias_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._token_to_vector_ids: dict[str, set[int]] = defaultdict(set)
        for vector_id, row in enumerate(self.metadata):
            forms = set(_semantic_forms(str(row["text"])))
            if row.get("term_type") == "preferred":
                for form in list(forms):
                    shortened = re.sub(r"^(?:bệnh|hội chứng|chứng)\s+", "", form).strip()
                    if len(shortened) >= 3:
                        forms.add(shortened)
            for form in forms:
                self._alias_rows[form].append(row)
                for token in _meaningful_tokens(form):
                    self._token_to_vector_ids[token].add(vector_id)

        expected_count = int(self.config["embedding"]["count"])
        expected_dimension = int(self.config["model"]["dimension"])
        if self.index.ntotal != expected_count or len(self.metadata) != expected_count:
            raise ValueError(
                "Artifact count mismatch: "
                f"index={self.index.ntotal}, metadata={len(self.metadata)}, config={expected_count}"
            )
        if self.index.d != expected_dimension:
            raise ValueError(
                f"Artifact dimension mismatch: index={self.index.d}, config={expected_dimension}"
            )

        model_config = self.config["model"]
        revision = model_config.get("revision")
        if revision in {"null", "None", ""}:
            revision = None
        self.encoder = SapBertEncoder(
            model_config["model_id"],
            revision=revision,
            device=device,
            max_length=int(model_config["max_length"]),
            pooling=model_config["pooling"],
        )
        if self.encoder.dimension != self.index.d:
            raise ValueError(
                f"Query encoder dimension {self.encoder.dimension} != index dimension {self.index.d}"
            )

    def _artifact_path(self, filename: str) -> Path:
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Unsafe artifact filename in config: {filename!r}")
        return self.index_dir / filename

    def _validate_config(self) -> None:
        if self.config.get("config_version") != 1:
            raise ValueError(f"Unsupported config_version: {self.config.get('config_version')}")
        if self.config.get("faiss", {}).get("index_type") != "IndexFlatIP":
            raise ValueError("Only IndexFlatIP configs are supported")
        if self.config.get("faiss", {}).get("metric") != "inner_product":
            raise ValueError("Expected inner_product FAISS metric")
        if self.config.get("embedding", {}).get("l2_normalized") is not True:
            raise ValueError("Expected L2-normalized index embeddings")
        required_artifacts = {"embeddings", "index", "metadata"}
        if not required_artifacts.issubset(self.config.get("artifacts", {})):
            raise ValueError("Config is missing one or more artifact filenames")

    @staticmethod
    def _load_metadata(path: Path) -> list[dict[str, Any]]:
        metadata: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid metadata JSON at {path}:{line_number}: {exc}"
                    ) from exc
                vector_id = len(metadata)
                if row.get("vector_id") != vector_id:
                    raise ValueError(
                        f"metadata vector_id mismatch at line {line_number}: "
                        f"{row.get('vector_id')} != {vector_id}"
                    )
                for field in ("term_id", "code", "text", "language", "term_type"):
                    if not isinstance(row.get(field), str):
                        raise ValueError(
                            f"Invalid metadata field {field} at line {line_number}"
                        )
                metadata.append(row)
        return metadata

    def _term_results_from_search(self, scores: Any, indices: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for score, vector_id in zip(scores, indices):
            vector_id = int(vector_id)
            if vector_id < 0:
                continue
            results.append({"score": float(score), **self.metadata[vector_id]})
        return results

    def _exact_alias_results(self, mention: str) -> list[dict[str, Any]]:
        normalized = _normalize_alias(mention)
        configured = config.EXACT_ALIAS_CODES.get(normalized)
        if configured is not None:
            return [{
                "code": configured,
                "score": 1.0,
                "text": mention,
                "language": "vi",
                "term_type": "alias",
                "term_id": f"configured_alias:{normalized}",
                "query_source": "configured_exact_alias",
                "exact_alias_source": "configured_exact_alias",
            }]
        codes = self.metadata_alias_index.get(normalized, set())
        if not codes:
            return []
        source = "metadata_exact_unique" if len(codes) == 1 else "metadata_exact_ambiguous"
        rows = self._alias_rows.get(normalized, [])
        result = []
        seen_codes: set[str] = set()
        for row in rows:
            code = str(row["code"])
            if code not in codes or code in seen_codes:
                continue
            seen_codes.add(code)
            result.append({
                "score": 1.0,
                **row,
                "query_source": source,
                "exact_alias_source": source,
            })
        return result

    def _lexical_metadata_results(self, mention: str) -> list[dict[str, Any]]:
        best_by_term: dict[str, dict[str, Any]] = {}
        for form in _semantic_forms(mention):
            tokens = _meaningful_tokens(form)
            if not tokens:
                continue
            postings = [self._token_to_vector_ids.get(token, set()) for token in tokens]
            if not postings or any(not values for values in postings):
                continue
            vector_ids = set.intersection(*postings)
            for vector_id in vector_ids:
                row = self.metadata[vector_id]
                features = _best_lexical_features(form, str(row["text"]))
                if features["mention_coverage"] < 0.60:
                    continue
                score = min(
                    0.995,
                    0.60
                    + 0.19 * features["mention_coverage"]
                    + 0.09 * features["candidate_coverage"]
                    + 0.07 * features["lexical_similarity"]
                    + 0.04 * float(features["phrase_containment"])
                    + _TERM_TYPE_BONUS.get(str(row.get("term_type", "")), 0.0),
                )
                value = {
                    "score": score,
                    **row,
                    "query_source": f"metadata_lexical:{form}",
                }
                previous = best_by_term.get(str(row["term_id"]))
                if previous is None or score > float(previous["score"]):
                    best_by_term[str(row["term_id"])] = value
        ranked = sorted(
            best_by_term.values(),
            key=lambda row: (-float(row["score"]), str(row["code"]), str(row["term_id"])),
        )
        return ranked[: config.LEXICAL_RETRIEVAL_LIMIT]

    def search_terms(
        self, mention: str, *, top_k_terms: int = config.DEFAULT_TOP_K_TERMS
    ) -> list[dict[str, Any]]:
        if top_k_terms <= 0:
            raise ValueError("top_k_terms must be positive")
        mention = clean_query_text(mention)
        query = self.encoder.encode(
            [mention], batch_size=1, show_progress=False, normalize=True
        )
        count = min(top_k_terms, int(self.index.ntotal))
        scores, indices = self.index.search(query, count)
        return self._term_results_from_search(scores[0], indices[0])

    def link(
        self,
        mention: str,
        *,
        top_k_terms: int = config.DEFAULT_TOP_K_TERMS,
        top_k_codes: int = config.DEFAULT_TOP_K_CODES,
        min_score: float | None = config.DEFAULT_MIN_SCORE,
    ) -> list[dict[str, Any]]:
        return self.link_many(
            [mention],
            top_k_terms=top_k_terms,
            top_k_codes=top_k_codes,
            min_score=min_score,
        )[0]

    def link_many(
        self,
        mentions: Iterable[str],
        *,
        top_k_terms: int = config.DEFAULT_TOP_K_TERMS,
        top_k_codes: int = config.DEFAULT_TOP_K_CODES,
        min_score: float | None = config.DEFAULT_MIN_SCORE,
    ) -> list[list[dict[str, Any]]]:
        if top_k_terms <= 0 or top_k_codes <= 0:
            raise ValueError("top_k_terms and top_k_codes must be positive")
        mention_list = list(mentions)
        if not mention_list:
            return []

        # Invalid input is isolated to that row instead of aborting every
        # mention in the batch.
        cleaned: list[str | None] = []
        for mention in mention_list:
            try:
                cleaned.append(clean_query_text(mention))
            except (TypeError, ValueError):
                cleaned.append(None)

        variants_by_row = [
            _query_variants(mention) if mention is not None else []
            for mention in cleaned
        ]
        flat_variants = list(
            dict.fromkeys(
                variant for variants in variants_by_row for variant in variants
            )
        )
        search_rows: dict[str, list[dict[str, Any]]] = {}
        if flat_variants:
            queries = self.encoder.encode(
                flat_variants,
                batch_size=self.query_batch_size,
                show_progress=False,
                normalize=True,
            )
            count = min(top_k_terms, int(self.index.ntotal))
            scores, indices = self.index.search(queries, count)
            for variant, row_scores, row_indices in zip(flat_variants, scores, indices):
                search_rows[variant] = self._term_results_from_search(
                    row_scores, row_indices
                )

        output: list[list[dict[str, Any]]] = []
        for mention, variants in zip(cleaned, variants_by_row):
            if mention is None:
                output.append([])
                continue
            term_results: list[dict[str, Any]] = []
            for variant in variants:
                for raw_row in search_rows.get(variant, []):
                    row = dict(raw_row)
                    row["query_source"] = f"dense:{variant}"
                    term_results.append(row)
            term_results.extend(self._lexical_metadata_results(mention))
            term_results.extend(self._exact_alias_results(mention))
            output.append(
                _finalize_term_results(
                    mention,
                    term_results,
                    top_k_codes=top_k_codes,
                    min_score=min_score,
                )
            )
        return output

    def retrieve_many(self, mentions: Iterable[str], **kwargs) -> list[list[dict[str, Any]]]:
        return self.link_many(mentions, **kwargs)

    def rank_many(self, mentions: Iterable[str], **kwargs) -> list[list[dict[str, Any]]]:
        return self.link_many(mentions, **kwargs)

    def predict_many(self, mentions: Iterable[str], **kwargs) -> list[list[dict[str, Any]]]:
        mention_list = list(mentions)
        top_k_codes = max(10, int(kwargs.pop("top_k_codes", 10)))
        ranked_many = self.link_many(mention_list, top_k_codes=top_k_codes, **kwargs)
        outputs: list[list[dict[str, Any]]] = []
        for ranked in ranked_many:
            supported = [
                item
                for item in ranked
                if item.get("support_level") in {"exact", "strong"}
                and not item.get("hard_conflicts")
            ]
            if not supported:
                outputs.append([])
                continue
            top = supported[0]
            second_score = (
                float(supported[1].get("support_score", 0.0))
                if len(supported) > 1
                else 0.0
            )
            margin = float(top.get("support_score", 0.0)) - second_score
            exact_unique = bool(
                top.get("exact_alias_source") == "configured_exact_alias"
                or (
                    top.get("exact_alias_source") == "metadata_exact_unique"
                    and top.get("exact_text_quality", True)
                )
            )
            if exact_unique or (
                not top.get("over_specific")
                and (
                    len(supported) == 1
                    or margin >= config.DETERMINISTIC_STRONG_MARGIN
                )
            ):
                outputs.append([top])
            else:
                outputs.append([])
        return outputs

    def predict(
        self,
        mention: str,
        *,
        top_k_terms: int = config.DEFAULT_TOP_K_TERMS,
        min_score: float | None = config.DEFAULT_MIN_SCORE,
        retrieval_k_codes: int = 10,
        max_candidates: int = 1,
    ) -> list[dict[str, Any]]:
        return self.predict_many(
            [mention],
            top_k_terms=top_k_terms,
            top_k_codes=max(retrieval_k_codes, max_candidates),
            min_score=min_score,
        )[0][:max_candidates]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mention", help="NER mention to link")
    parser.add_argument("--index-dir", type=Path, default=config.DEFAULT_INDEX_DIR)
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)
    parser.add_argument("--top-k-terms", type=int, default=config.DEFAULT_TOP_K_TERMS)
    parser.add_argument("--top-k-codes", type=int, default=config.DEFAULT_TOP_K_CODES)
    parser.add_argument("--min-score", type=float, default=config.DEFAULT_MIN_SCORE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    linker = Icd10Linker(args.index_dir, device=args.device)
    results = linker.link(
        args.mention,
        top_k_terms=args.top_k_terms,
        top_k_codes=args.top_k_codes,
        min_score=args.min_score,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()