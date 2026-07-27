"""Đánh giá riêng từng tầng của pipeline trên gold BTC (Viettel AI Race).

Đừng chỉ đo top-1 cuối cùng — nếu chỉ đo top-1, không biết lỗi nằm ở
retrieval (mất candidate ngay từ đầu) hay ở rerank (có candidate đúng
nhưng xếp sai hạng).

GOLD_DRUGS lấy từ ví dụ input/output chính thức của đề bài vòng 1
(1 bệnh án mẫu, 11 thuốc).
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    # Chạy như module trong package: python -m src.linking.rxnorm.evaluate
    from .linker import RxNormLinker
    from .schemas import RxNormCandidate
except ImportError:
    # Chạy trực tiếp file (vd bấm Run trong VS Code): python evaluate.py
    # Tự thêm thư mục cha (linking/) vào sys.path để import bằng tên package.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from linking.rxnorm.linker import RxNormLinker
    from linking.rxnorm.schemas import RxNormCandidate
    from linking.rxnorm.parser import parse_drug_mention
else:
    from .parser import parse_drug_mention

# (mention text, rxcui đúng theo gold BTC)
GOLD_DRUGS: list[tuple[str, str]] = [
    ("amlodipine 10 mg po daily", "308135"),
    ("aspirin 81 mg po daily", "243670"),
    ("metoprolol succinate xl 50 mg po daily", "866436"),
    ("guaifenesin ml po q6h:prn", "392085"),
    ("nystatin oral suspension 5 ml po qid:prn", "7597"),
    ("acetaminophen 325-650 mg po q6h:prn", "313782"),
    ("pravastatin 40 mg po daily", "904475"),
    ("docusate sodium 100 mg po bid", "1099279"),
    ("senna 8.6 mg po bid:prn", "312935"),
    ("clonazepam 0.5 mg po qam:prn", "197527"),
    ("clonazepam 1.5 mg po qhs", "197528"),
]


@dataclass
class RecallReport:
    k: int
    hit: int
    total: int
    misses: list[str]

    @property
    def recall(self) -> float:
        return self.hit / self.total if self.total else 0.0


def evaluate_candidate_recall(
    linker: RxNormLinker, gold: list[tuple[str, str]], k: int
) -> RecallReport:
    """Recall@k của tầng retrieval (trước rerank)."""

    hit = 0
    misses: list[str] = []

    for mention, target_rxcui in gold:
        parsed = parse_drug_mention(mention)
        candidates = linker.retriever.retrieve(parsed)

        ranked_rxcuis = sorted(
            candidates.values(),
            key=lambda c: (c.dense_score, c.lexical_score),
            reverse=True,
        )[:k]

        if any(c.rxcui == target_rxcui for c in ranked_rxcuis) or target_rxcui in candidates:
            # target có mặt trong tập candidate (không nhất thiết đứng
            # trong top-k theo dense/lexical thô) -> tính là "không mất gold"
            if target_rxcui in candidates:
                hit += 1
            else:
                misses.append(mention)
        else:
            misses.append(mention)

    return RecallReport(k=k, hit=hit, total=len(gold), misses=misses)


def evaluate_rule_ranking(
    linker: RxNormLinker, gold: list[tuple[str, str]], top_k: int = 10
) -> dict[str, float]:
    """Rule Recall@top_k và Rule Top-1 sau rerank."""

    recall_at_k = 0
    top1 = 0

    for mention, target_rxcui in gold:
        result = linker.link(mention, top_k=top_k)
        ranked: list[RxNormCandidate] = result["candidates"]

        if ranked and ranked[0].rxcui == target_rxcui:
            top1 += 1

        if any(c.rxcui == target_rxcui for c in ranked):
            recall_at_k += 1

    total = len(gold)
    return {
        "rule_recall_at_k": recall_at_k / total if total else 0.0,
        "rule_top1": top1 / total if total else 0.0,
    }


def print_error_analysis(linker: RxNormLinker, gold: list[tuple[str, str]], top_k: int = 5) -> None:
    for mention, target_rxcui in gold:
        result = linker.link(mention, top_k=top_k)
        ranked: list[RxNormCandidate] = result["candidates"]

        predicted = ranked[0].rxcui if ranked else None
        status = "OK" if predicted == target_rxcui else "SAI"

        print(f"[{status}] {mention}")
        print(f"    gold      : {target_rxcui}")
        print(f"    predicted : {predicted}")

        for rank, candidate in enumerate(ranked, start=1):
            mark = " <-- gold" if candidate.rxcui == target_rxcui else ""
            print(
                f"    {rank:>2}. rxcui={candidate.rxcui:<10} tty={candidate.tty:<5} "
                f"final={candidate.final_score:.4f} dense={candidate.dense_score:.4f} "
                f"features={candidate.features}{mark}"
            )

        print()


def run_full_evaluation(index_dir: str, clean_path: str | None = None) -> None:
    linker = RxNormLinker(index_dir=index_dir, clean_path=clean_path)

    print("=== Candidate Recall ===")
    for k in (50, 200):
        report = evaluate_candidate_recall(linker, GOLD_DRUGS, k=k)
        print(f"Recall@{k}: {report.recall:.3f} ({report.hit}/{report.total})")
        if report.misses:
            print("  misses:", report.misses)

    print("\n=== Rule Ranking ===")
    metrics = evaluate_rule_ranking(linker, GOLD_DRUGS, top_k=10)
    print(f"Rule Recall@10: {metrics['rule_recall_at_k']:.3f}")
    print(f"Rule Top-1    : {metrics['rule_top1']:.3f}")

    print("\n=== Error Analysis ===")
    print_error_analysis(linker, GOLD_DRUGS, top_k=5)


if __name__ == "__main__":
    
    import sys

    try:
        from . import config
    except ImportError:
        from linking.rxnorm import config

    index_dir = sys.argv[1] if len(sys.argv) > 1 else str(config.DEFAULT_INDEX_DIR)
    clean_path = sys.argv[2] if len(sys.argv) > 2 else str(config.DEFAULT_CLEAN_PATH)

    run_full_evaluation(index_dir, clean_path)