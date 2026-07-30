"""CLI chạy two-pass NER, 1.5B fixer, 7B review và 7B-assisted linking.

Qwen 7B được load một lần cho toàn bộ batch: trước tiên review/recover NER,
sau đó chọn trong candidate do RxNorm/ICD-10 retriever trả về, rồi mới unload.

Ví dụ dùng:

    # test nhanh chỉ NER
    python -m src.inference.cli --input data/1.txt --print

    # batch full pipeline (linking, không LLM)
    python -m src.inference.cli --input-dir data/public_test --output-dir output \\
        --with-rxnorm --with-icd10

    # pipeline notebook + 1.5B fixer + 7B review/linking
    python -m src.inference.cli --input-dir data/public_test --output-dir output \\
        --with-llm-fixer --with-llm-7b --with-rxnorm --with-icd10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as cfg
from . import io as inference_io
from .pipeline import InferencePipeline


def _configure_utf8_stdio() -> None:
    """Keep Vietnamese help/log/JSON usable on Windows legacy consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="1 file .txt duy nhất")
    input_group.add_argument("--input-dir", type=Path, help="thư mục chứa nhiều file .txt")

    parser.add_argument("--output-dir", type=Path, default=None,
                         help="thư mục ghi output .json (bắt buộc nếu dùng --input-dir, trừ khi có --print)")
    parser.add_argument("--print", dest="do_print", action="store_true",
                         help="in JSON ra stdout thay vì/thêm vào ghi file")

    parser.add_argument("--with-rxnorm", action="store_true", help="bật linking RxNorm cho THUỐC")
    parser.add_argument("--with-icd10", action="store_true", help="bật linking ICD-10 cho CHẨN_ĐOÁN")
    parser.add_argument("--with-llm-7b", action="store_true",
                        help="bật Qwen 7B cho NER review/recovery và linking rerank")
    parser.add_argument("--with-llm-fixer", action="store_true",
                        help="bật Qwen2.5-1.5B fixer sau NER, trước reviewer 7B")
    parser.add_argument("--with-llm-selector", action="store_true",
                        help="bật riêng Qwen 7B rerank candidate linking")
    parser.add_argument(
        "--no-llm-recall-audit",
        action="store_true",
        help="khi bật fixer, chỉ sửa span bị flag và không audit entity bị sót",
    )

    parser.add_argument("--checkpoint", type=Path, default=cfg.DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--label-dicts", type=Path, default=cfg.DEFAULT_LABEL_DICTS_PATH)
    parser.add_argument("--backbone", default=cfg.DEFAULT_BACKBONE)
    parser.add_argument("--vncorenlp-jar", type=Path, default=cfg.DEFAULT_VNCORENLP_JAR)
    parser.add_argument("--device", default=cfg.DEFAULT_DEVICE)

    parser.add_argument("--no-repair-gate", action="store_true",
                         help="tắt rule filter, để so sánh A/B raw model output")

    return parser.parse_args()


def _load_raw_texts(paths: list[Path]) -> dict[str, str]:
    return {p.stem: inference_io.read_text_file(p) for p in paths}


def run(args: argparse.Namespace, input_paths: list[Path]) -> dict[str, list[dict]]:
    print("[cli] Đang load NER engine...", file=sys.stderr)
    pipeline = InferencePipeline.load(
        with_rxnorm=args.with_rxnorm,
        with_icd10=args.with_icd10,
        checkpoint_path=args.checkpoint,
        label_dicts_path=args.label_dicts,
        backbone=args.backbone,
        vncorenlp_jar=args.vncorenlp_jar,
        device=args.device,
    )
    print("[cli] Load NER engine xong.", file=sys.stderr)

    predict_kwargs = {"apply_repair_gate": not args.no_repair_gate}
    raw_texts_by_id = _load_raw_texts(input_paths)

    # ---- Stage 1: NER cho toàn bộ batch (không LLM) ----
    entities_by_id = pipeline.run_ner_stage(raw_texts_by_id, **predict_kwargs)

    # ---- Stage 2 (optional): notebook Qwen2.5-1.5B guarded fixer ----
    if args.with_llm_fixer:
        from ..llm.backend import LocalLLM
        from ..llm.config import NER_FIXER_CONFIG

        print("[cli] Đang load Qwen2.5-1.5B NER fixer...", file=sys.stderr)
        fixer_llm = LocalLLM(NER_FIXER_CONFIG)
        fixer_llm.load()
        entities_by_id = pipeline.run_fixer_stage(
            raw_texts_by_id,
            entities_by_id,
            fixer_llm,
            audit_missing=not args.no_llm_recall_audit,
            batch_size=NER_FIXER_CONFIG.batch_size,
        )
        fixer_llm.unload()
        print("[cli] 1.5B fixer xong, đã unload trước khi load 7B.", file=sys.stderr)

    # ---- Stage 3 (optional): grouped 7B NER review/recovery ----
    use_7b_reviewer = args.with_llm_7b
    use_7b_selector = args.with_llm_7b or args.with_llm_selector
    reviewer_llm = None
    if use_7b_reviewer or use_7b_selector:
        from ..llm.backend import LocalLLM
        from ..llm.config import NER_REVIEWER_7B_CONFIG

        print("[cli] Đang load Qwen 7B...", file=sys.stderr)
        reviewer_llm = LocalLLM(NER_REVIEWER_7B_CONFIG)
        reviewer_llm.load()

    if use_7b_reviewer:
        entities_by_id = pipeline.run_7b_ner_stage(
            raw_texts_by_id,
            entities_by_id,
            reviewer_llm,
            batch_size=NER_REVIEWER_7B_CONFIG.batch_size,
            retry_rounds=NER_REVIEWER_7B_CONFIG.retry_rounds,
            include_recovery=not args.no_llm_recall_audit,
        )
        print("[cli] 7B NER reviewer xong; giữ model để rerank linking.", file=sys.stderr)

    # ---- Stage 4: retriever hiện tại -> optional 7B chọn trong candidate ----
    candidates_by_id = pipeline.run_linking_stage(
        entities_by_id,
        selector_llm=reviewer_llm if use_7b_selector else None,
        raw_texts_by_id=raw_texts_by_id,
    )

    if reviewer_llm is not None:
        reviewer_llm.unload()
        print("[cli] 7B xong, đã unload.", file=sys.stderr)

    return pipeline.build_outputs(entities_by_id, candidates_by_id)


def main() -> None:
    _configure_utf8_stdio()
    args = parse_args()

    if args.input_dir is not None and args.output_dir is None and not args.do_print:
        print("Cần --output-dir hoặc --print khi dùng --input-dir", file=sys.stderr)
        sys.exit(1)

    input_paths = [args.input] if args.input is not None else sorted(args.input_dir.glob("*.txt"))
    if not input_paths:
        print("Không tìm thấy file .txt nào để chạy.", file=sys.stderr)
        sys.exit(1)

    outputs = run(args, input_paths)

    n_ok = 0
    for rid, record_output in outputs.items():
        if args.output_dir is not None:
            inference_io.write_output_json(record_output, args.output_dir / f"{rid}.json")
        if args.do_print or args.output_dir is None:
            print(json.dumps(record_output, ensure_ascii=False, indent=2))
        n_ok += 1

    print(f"[cli] Xong: {n_ok} record.", file=sys.stderr)


if __name__ == "__main__":
    main()
