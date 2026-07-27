"""Đánh giá Icd10Linker trên gold BTC.

File này đánh giá hai tầng riêng biệt:

1. Candidate Recall:
   Kiểm tra các mã gold có được FAISS/linker tìm thấy trong top-k hay không.

2. Final Prediction:
   Tính candidates_score bằng weighted Jaccard trên danh sách mã cuối cùng.

Nếu Icd10Linker đã có method predict(), official metric sẽ dùng predict().
Nếu chưa có predict(), chương trình tạm fallback sang link() và in cảnh báo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from .icd10_linker import Icd10Linker
    from . import config

except ImportError:
    # Hỗ trợ chạy trực tiếp:
    # python src/linking/icd10/evaluate.py

    import sys

    SRC_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(SRC_ROOT))

    from linking.icd10.icd10_linker import Icd10Linker
    from linking.icd10 import config


# ============================================================
# GOLD DATA
# ============================================================

# Một mention có thể có nhiều ICD-10 code đúng.
GOLD_DIAGNOSES: list[tuple[str, list[str]]] = [
    (
        "bệnh trào ngược dạ dày - thực quản",
        ["K21.0", "K21.9"],
    ),
]


@dataclass(frozen=True)
class GoldItem:
    mention: str
    gold_codes: frozenset[str]


@dataclass
class RecallReport:
    k: int
    matched_codes: int
    total_gold_codes: int
    misses: list[str]

    @property
    def recall(self) -> float:
        if self.total_gold_codes == 0:
            return 0.0

        return self.matched_codes / self.total_gold_codes


def load_gold() -> list[GoldItem]:
    return [
        GoldItem(
            mention=mention,
            gold_codes=frozenset(codes),
        )
        for mention, codes in GOLD_DIAGNOSES
    ]


# ============================================================
# METRIC
# ============================================================

def jaccard(
    gold: set[str] | frozenset[str],
    predicted: set[str],
) -> float:
    """Jaccard dùng cho candidates_score."""

    if not gold and not predicted:
        return 1.0

    if not gold and predicted:
        return 0.0

    union = set(gold) | predicted

    if not union:
        return 1.0

    intersection = set(gold) & predicted

    return len(intersection) / len(union)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_codes(
    linker: Icd10Linker,
    mention: str,
    *,
    top_k_terms: int,
    top_k_codes: int,
) -> list[dict[str, Any]]:
    """Lấy candidate phục vụ đo Recall@K."""

    # Muốn lấy K code thì phải tìm ít nhất K surface terms.
    effective_top_k_terms = max(
        top_k_terms,
        top_k_codes,
    )

    return linker.link(
        mention,
        top_k_terms=effective_top_k_terms,
        top_k_codes=top_k_codes,
    )


def predict_final_codes(
    linker: Icd10Linker,
    mention: str,
    *,
    top_k_terms: int,
    retrieval_k_codes: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Lấy danh sách code cuối cùng để tính candidates_score.

    Ưu tiên linker.predict() nếu đã triển khai tầng post-processing.
    Fallback sang linker.link() để file vẫn chạy được trong giai đoạn
    chưa hoàn thiện predict().
    """

    predict_method = getattr(
        linker,
        "predict",
        None,
    )

    if callable(predict_method):
        return predict_method(
            mention,
            top_k_terms=top_k_terms,
            retrieval_k_codes=retrieval_k_codes,
            max_candidates=max_candidates,
        )

    # Fallback tạm thời:
    # đây vẫn là retrieval top-k, chưa phải final policy hoàn chỉnh.
    return linker.link(
        mention,
        top_k_terms=max(
            top_k_terms,
            max_candidates,
        ),
        top_k_codes=max_candidates,
    )


# ============================================================
# CANDIDATE RECALL
# ============================================================

def evaluate_code_recall(
    linker: Icd10Linker,
    gold: list[GoldItem],
    *,
    top_k_terms: int,
    top_k_codes: int,
) -> RecallReport:
    """Tính recall thật trên tổng số gold code.

    Ví dụ gold có K21.0 và K21.9:
    - tìm thấy cả hai: 2/2
    - chỉ tìm thấy một: 1/2
    """

    matched_codes = 0
    total_gold_codes = 0
    misses: list[str] = []

    for item in gold:
        retrieved = retrieve_codes(
            linker,
            item.mention,
            top_k_terms=top_k_terms,
            top_k_codes=top_k_codes,
        )

        retrieved_codes = {
            str(result["code"])
            for result in retrieved
        }

        matched = (
            retrieved_codes
            & set(item.gold_codes)
        )

        matched_codes += len(matched)
        total_gold_codes += len(
            item.gold_codes
        )

        missing = sorted(
            set(item.gold_codes)
            - retrieved_codes
        )

        if missing:
            misses.append(
                f"{item.mention}: "
                f"missing={missing}"
            )

    return RecallReport(
        k=top_k_codes,
        matched_codes=matched_codes,
        total_gold_codes=total_gold_codes,
        misses=misses,
    )


# ============================================================
# OFFICIAL CANDIDATE SCORE
# ============================================================

def evaluate_candidates_score(
    linker: Icd10Linker,
    gold: list[GoldItem],
    *,
    top_k_terms: int,
    retrieval_k_codes: int,
    max_final_candidates: int,
) -> dict[str, float]:
    """Tính weighted Jaccard theo candidates_score của BTC."""

    weighted_sum = 0.0
    weight_total = 0.0
    top1_hit = 0
    exact_set_hit = 0

    for item in gold:
        predicted = predict_final_codes(
            linker,
            item.mention,
            top_k_terms=top_k_terms,
            retrieval_k_codes=retrieval_k_codes,
            max_candidates=max_final_candidates,
        )

        predicted_codes = {
            str(result["code"])
            for result in predicted
        }

        gold_codes = set(item.gold_codes)

        weight = len(gold_codes) + 1

        item_jaccard = jaccard(
            gold_codes,
            predicted_codes,
        )

        weighted_sum += (
            item_jaccard * weight
        )

        weight_total += weight

        if (
            predicted
            and str(predicted[0]["code"])
            in gold_codes
        ):
            top1_hit += 1

        if predicted_codes == gold_codes:
            exact_set_hit += 1

    total = len(gold)

    return {
        "candidates_score": (
            weighted_sum / weight_total
            if weight_total
            else 0.0
        ),
        "top1_accuracy": (
            top1_hit / total
            if total
            else 0.0
        ),
        "exact_set_accuracy": (
            exact_set_hit / total
            if total
            else 0.0
        ),
    }


# ============================================================
# ERROR ANALYSIS
# ============================================================

def classify_prediction_status(
    gold_codes: set[str],
    predicted_codes: set[str],
) -> str:
    if predicted_codes == gold_codes:
        return "ĐÚNG"

    missing = gold_codes - predicted_codes
    extra = predicted_codes - gold_codes

    if missing and extra:
        return "THIẾU + DƯ"

    if missing:
        return "THIẾU"

    if extra:
        return "DƯ"

    return "SAI"


def print_error_analysis(
    linker: Icd10Linker,
    gold: list[GoldItem],
    *,
    top_k_terms: int,
    retrieval_k_codes: int,
    max_final_candidates: int,
    display_retrieval_k: int = 5,
) -> None:
    for item in gold:
        gold_codes = set(
            item.gold_codes
        )

        predicted = predict_final_codes(
            linker,
            item.mention,
            top_k_terms=top_k_terms,
            retrieval_k_codes=retrieval_k_codes,
            max_candidates=max_final_candidates,
        )

        predicted_codes_list = [
            str(result["code"])
            for result in predicted
        ]

        predicted_codes = set(
            predicted_codes_list
        )

        status = classify_prediction_status(
            gold_codes,
            predicted_codes,
        )

        top1_correct = bool(
            predicted_codes_list
            and predicted_codes_list[0]
            in gold_codes
        )

        missing = sorted(
            gold_codes - predicted_codes
        )

        extra = sorted(
            predicted_codes - gold_codes
        )

        print(
            f"[{status}] {item.mention}"
        )

        print(
            f"    gold       : "
            f"{sorted(gold_codes)}"
        )

        print(
            f"    predicted  : "
            f"{predicted_codes_list}"
        )

        print(
            f"    top1 đúng  : "
            f"{top1_correct}"
        )

        print(
            f"    jaccard    : "
            f"{jaccard(gold_codes, predicted_codes):.4f}"
        )

        if missing:
            print(
                f"    còn thiếu  : {missing}"
            )

        if extra:
            print(
                f"    dự đoán dư : {extra}"
            )

        print("    final results:")

        if not predicted:
            print("      (không có dự đoán)")

        for rank, result in enumerate(
            predicted,
            start=1,
        ):
            code = str(
                result["code"]
            )

            mark = (
                " <-- gold"
                if code in gold_codes
                else ""
            )

            score = float(
                result.get("score", 0.0)
            )

            matched_term = result.get(
                "matched_term",
                "",
            )

            print(
                f"      {rank:>2}. "
                f"code={code:<8} "
                f"score={score:.4f} "
                f"matched_term={matched_term!r}"
                f"{mark}"
            )

        # In thêm retrieval để biết gold có được tìm thấy
        # nhưng bị final selector loại hay không.
        retrieved = retrieve_codes(
            linker,
            item.mention,
            top_k_terms=top_k_terms,
            top_k_codes=max(
                retrieval_k_codes,
                display_retrieval_k,
            ),
        )

        print(
            f"    retrieval top "
            f"{display_retrieval_k}:"
        )

        for rank, result in enumerate(
            retrieved[:display_retrieval_k],
            start=1,
        ):
            code = str(
                result["code"]
            )

            mark = (
                " <-- gold"
                if code in gold_codes
                else ""
            )

            print(
                f"      {rank:>2}. "
                f"code={code:<8} "
                f"score={float(result['score']):.4f} "
                f"matched_term="
                f"{result['matched_term']!r}"
                f"{mark}"
            )

        print()


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_full_evaluation(
    index_dir: str | Path | None = None,
    *,
    device: str = config.DEFAULT_DEVICE,
    top_k_terms: int = config.DEFAULT_TOP_K_TERMS,
    retrieval_k_codes: int = 20,
    max_final_candidates: int = 3,
) -> None:
    linker = Icd10Linker(
        index_dir=(
            index_dir
            or config.DEFAULT_INDEX_DIR
        ),
        device=device,
        query_batch_size=(
            config.DEFAULT_QUERY_BATCH_SIZE
        ),
    )

    gold = load_gold()

    has_predict = callable(
        getattr(
            linker,
            "predict",
            None,
        )
    )

    print(
        f"=== Gold: {len(gold)} mention ===\n"
    )

    print(
        "Final prediction mode:",
        (
            "linker.predict()"
            if has_predict
            else "fallback linker.link()"
        ),
    )

    if not has_predict:
        print(
            "CẢNH BÁO: Icd10Linker chưa có predict(). "
            "candidates_score hiện chỉ là kết quả tạm "
            "từ retrieval top-k.\n"
        )

    print("=== Code Recall ===")

    for k in (5, 10):
        report = evaluate_code_recall(
            linker,
            gold,
            top_k_terms=max(
                top_k_terms,
                k,
            ),
            top_k_codes=k,
        )

        print(
            f"Recall@{k}: "
            f"{report.recall:.3f} "
            f"({report.matched_codes}/"
            f"{report.total_gold_codes})"
        )

        if report.misses:
            for miss in report.misses:
                print(
                    f"  - {miss}"
                )

    print(
        "\n=== Official Metric "
        "(candidates_score) ==="
    )

    metrics = evaluate_candidates_score(
        linker,
        gold,
        top_k_terms=top_k_terms,
        retrieval_k_codes=(
            retrieval_k_codes
        ),
        max_final_candidates=(
            max_final_candidates
        ),
    )

    print(
        f"candidates_score   : "
        f"{metrics['candidates_score']:.4f}"
    )

    print(
        f"top1_accuracy      : "
        f"{metrics['top1_accuracy']:.4f}"
    )

    print(
        f"exact_set_accuracy : "
        f"{metrics['exact_set_accuracy']:.4f}"
    )

    print("\n=== Error Analysis ===")

    print_error_analysis(
        linker,
        gold,
        top_k_terms=top_k_terms,
        retrieval_k_codes=(
            retrieval_k_codes
        ),
        max_final_candidates=(
            max_final_candidates
        ),
        display_retrieval_k=5,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help=(
            "Thư mục chứa ICD-10 FAISS index. "
            "Mặc định lấy từ config.py."
        ),
    )

    parser.add_argument(
        "--device",
        default=config.DEFAULT_DEVICE,
        help="auto, cpu, cuda hoặc cuda:0",
    )

    parser.add_argument(
        "--top-k-terms",
        type=int,
        default=config.DEFAULT_TOP_K_TERMS,
        help="Số surface term lấy từ FAISS.",
    )

    parser.add_argument(
        "--retrieval-k-codes",
        type=int,
        default=20,
        help=(
            "Số code retrieval trước khi "
            "final post-processing."
        ),
    )

    parser.add_argument(
        "--max-final-candidates",
        type=int,
        default=3,
        help=(
            "Số code tối đa trong output cuối."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k_terms <= 0:
        raise SystemExit(
            "--top-k-terms must be positive"
        )

    if args.retrieval_k_codes <= 0:
        raise SystemExit(
            "--retrieval-k-codes must be positive"
        )

    if args.max_final_candidates <= 0:
        raise SystemExit(
            "--max-final-candidates must be positive"
        )

    run_full_evaluation(
        index_dir=args.index_dir,
        device=args.device,
        top_k_terms=args.top_k_terms,
        retrieval_k_codes=(
            args.retrieval_k_codes
        ),
        max_final_candidates=(
            args.max_final_candidates
        ),
    )


if __name__ == "__main__":
    main()