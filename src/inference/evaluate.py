"""Stage-wise NER/linking metrics used for honest A/B evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _safe_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_ner(gold_records: Iterable[list[dict]], predicted_records: Iterable[list[dict]]) -> dict:
    exact_tp = exact_fp = exact_fn = 0
    boundary_errors = type_correct = matched_boundaries = 0
    assertion_tp = assertion_fp = assertion_fn = 0
    for gold, predicted in zip(gold_records, predicted_records):
        gold_exact = {(tuple(item["position"]), item["type"]) for item in gold}
        pred_exact = {(tuple(item["position"]), item["type"]) for item in predicted}
        exact_tp += len(gold_exact & pred_exact)
        exact_fp += len(pred_exact - gold_exact)
        exact_fn += len(gold_exact - pred_exact)
        gold_by_span = {tuple(item["position"]): item for item in gold}
        for item in predicted:
            span = tuple(item["position"])
            if span in gold_by_span:
                matched_boundaries += 1
                if item["type"] == gold_by_span[span]["type"]:
                    type_correct += 1
                gold_assertions = set(gold_by_span[span].get("assertions", []))
                pred_assertions = set(item.get("assertions", []))
                assertion_tp += len(gold_assertions & pred_assertions)
                assertion_fp += len(pred_assertions - gold_assertions)
                assertion_fn += len(gold_assertions - pred_assertions)
            elif any(span[0] < g[1] and span[1] > g[0] for g in gold_by_span):
                boundary_errors += 1
    return {
        "exact_span_type": _safe_f1(exact_tp, exact_fp, exact_fn),
        "type_accuracy_on_exact_boundary": type_correct / matched_boundaries if matched_boundaries else 0.0,
        "assertion_micro": _safe_f1(assertion_tp, assertion_fp, assertion_fn),
        "boundary_errors": boundary_errors,
    }


def retrieval_recall_at_k(gold_codes: Iterable[set[str]], ranked_codes: Iterable[list[str]], k: int) -> float:
    rows = [(set(gold), list(ranked)[:k]) for gold, ranked in zip(gold_codes, ranked_codes) if gold]
    if not rows:
        return 0.0
    return sum(bool(gold.intersection(ranked)) for gold, ranked in rows) / len(rows)


def evaluate_final_linking(gold_codes: Iterable[set[str]], predicted_codes: Iterable[list[str]]) -> dict[str, float]:
    rows = [(set(gold), set(predicted)) for gold, predicted in zip(gold_codes, predicted_codes)]
    if not rows:
        return {"weighted_jaccard": 0.0, "false_code_rate": 0.0, "abstention_rate": 0.0}
    intersection = sum(len(gold & predicted) for gold, predicted in rows)
    union = sum(len(gold | predicted) for gold, predicted in rows)
    predicted_count = sum(len(predicted) for _, predicted in rows)
    false_count = sum(len(predicted - gold) for gold, predicted in rows)
    return {
        "weighted_jaccard": intersection / union if union else 1.0,
        "false_code_rate": false_count / predicted_count if predicted_count else 0.0,
        "abstention_rate": sum(not predicted for _, predicted in rows) / len(rows),
    }

