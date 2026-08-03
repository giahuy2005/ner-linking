"""CLI for detailed CRF/span NER, Qwen3-8B editing, and ontology linking."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import config as cfg
from . import io as inference_io
from .pipeline import InferencePipeline


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="một file .txt")
    inputs.add_argument("--input-dir", type=Path, help="thư mục chứa file .txt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print", dest="do_print", action="store_true")

    parser.add_argument("--with-rxnorm", action="store_true")
    parser.add_argument("--with-icd10", action="store_true")
    parser.add_argument("--with-llm-8b", action="store_true")
    parser.add_argument("--with-llm-ner-editor", action="store_true")
    parser.add_argument("--with-llm-linking-selector", action="store_true")
    parser.add_argument("--deterministic-linking-only", action="store_true")
    parser.add_argument("--llm-model-id", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--llm-dtype", choices=("auto", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--llm-quantization", choices=("none", "4bit"), default="none",
    )
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--llm-cache-path", type=Path)
    parser.add_argument("--llm-audit-dir", type=Path)
    parser.add_argument("--linking-selector-cache-path", type=Path)
    parser.add_argument("--linking-selector-batch-size", type=int, default=4)
    parser.add_argument("--rxnorm-retrieval-k", type=int, default=50)
    parser.add_argument("--icd10-retrieval-k", type=int, default=50)
    parser.add_argument("--linking-audit-dir", type=Path)
    parser.add_argument("--no-llm-recovery", action="store_true")
    parser.add_argument("--review-only-auto-add-eligible", action="store_true")

    parser.add_argument("--checkpoint", type=Path, default=cfg.DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--label-dicts", type=Path, default=cfg.DEFAULT_LABEL_DICTS_PATH)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="mặc định dùng model_config.json nằm cạnh checkpoint",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help="override model_name trong model config",
    )
    parser.add_argument("--vncorenlp-jar", type=Path, default=cfg.DEFAULT_VNCORENLP_JAR)
    parser.add_argument("--device", default=cfg.DEFAULT_DEVICE)
    parser.add_argument("--no-repair-gate", action="store_true")
    return parser.parse_args()


def _load_raw_texts(paths: list[Path]) -> dict[str, str]:
    return {path.stem: inference_io.read_text_file(path) for path in paths}


def _write_stage_checkpoint(output_dir: Path | None, stage: str, payload) -> None:
    if output_dir is None:
        return
    stage_dir = output_dir / ".stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / f"{stage}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def run(
    args: argparse.Namespace,
    input_paths: list[Path],
    *,
    raw_texts_by_id: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    print("[cli] Đang load NER engine...", file=sys.stderr)
    pipeline = InferencePipeline.load(
        with_rxnorm=args.with_rxnorm,
        with_icd10=args.with_icd10,
        checkpoint_path=args.checkpoint,
        label_dicts_path=args.label_dicts,
        model_config_path=args.model_config,
        backbone=args.backbone,
        vncorenlp_jar=args.vncorenlp_jar,
        device=args.device,
    )
    raw_texts_by_id = raw_texts_by_id or _load_raw_texts(input_paths)
    predict_kwargs = {"apply_repair_gate": not args.no_repair_gate}

    use_editor = args.with_llm_8b or args.with_llm_ner_editor
    use_selector = (args.with_llm_8b or args.with_llm_linking_selector) and not args.deterministic_linking_only
    detailed_by_id = None
    if use_editor:
        detailed_by_id = pipeline.run_ner_stage_detailed(
            raw_texts_by_id, **predict_kwargs,
        )
        entities_by_id = {
            record_id: list(detail.final_entities)
            for record_id, detail in detailed_by_id.items()
        }
    else:
        entities_by_id = pipeline.run_ner_stage(
            raw_texts_by_id, **predict_kwargs,
        )

    qwen_llm = None
    if use_editor or use_selector:
        from ..llm.backend import LocalLLM
        from ..llm.config import QWEN3_8B_EDITOR_CONFIG

        llm_config = replace(
            QWEN3_8B_EDITOR_CONFIG,
            model_id=args.llm_model_id,
            dtype=args.llm_dtype,
            load_in_4bit=args.llm_quantization == "4bit",
            batch_size=args.llm_batch_size,
        )
        pipeline.ner_engine.offload_to_cpu()
        print(f"[cli] Đang load {llm_config.model_id}...", file=sys.stderr)
        qwen_llm = LocalLLM(llm_config)
        qwen_llm.load()
    if use_editor:
        entities_by_id = pipeline.run_qwen8b_ner_editor_stage(
            raw_texts_by_id,
            detailed_by_id,
            qwen_llm,
            batch_size=args.llm_batch_size,
            include_recovery=not args.no_llm_recovery,
            review_only_auto_add_eligible=args.review_only_auto_add_eligible,
            cache_path=args.llm_cache_path,
            model_id=args.llm_model_id,
        )
        if args.llm_audit_dir is not None:
            args.llm_audit_dir.mkdir(parents=True, exist_ok=True)
            for record_id, audit in pipeline.last_editor_audit.items():
                path = args.llm_audit_dir / f"{record_id}.ner_audit.json"
                path.write_text(
                    json.dumps(audit, ensure_ascii=False, indent=2, default=list),
                    encoding="utf-8",
                )

    _write_stage_checkpoint(args.output_dir, "ner", {
        record_id: [entity.to_btc_dict([]) for entity in entities]
        for record_id, entities in entities_by_id.items()
    })

    candidates_by_id = pipeline.run_linking_stage(
        entities_by_id,
        selector_llm=qwen_llm if use_selector else None,
        raw_texts_by_id=raw_texts_by_id,
        selector_batch_size=args.linking_selector_batch_size,
        selector_cache_path=args.linking_selector_cache_path,
        model_id=args.llm_model_id,
        retrieval_k_by_linker={
            "rxnorm": args.rxnorm_retrieval_k,
            "icd10": args.icd10_retrieval_k,
        },
    )
    _write_stage_checkpoint(args.output_dir, "linking", candidates_by_id)
    if args.linking_audit_dir is not None:
        args.linking_audit_dir.mkdir(parents=True, exist_ok=True)
        for record_id, rows in pipeline.last_linking_retrieval.items():
            audit = {str(index): [
                item if isinstance(item, dict) else item.__dict__ for item in candidates
            ] for index, candidates in rows.items()}
            (args.linking_audit_dir / f"{record_id}.linking_audit.json").write_text(
                json.dumps({"retrieval": audit, "selection": pipeline.last_linking_audit.get(record_id, {})},
                           ensure_ascii=False, indent=2, default=str), encoding="utf-8",
            )
    if qwen_llm is not None:
        qwen_llm.unload()
    return pipeline.build_outputs(entities_by_id, candidates_by_id)


def main() -> None:
    _configure_utf8_stdio()
    args = parse_args()
    if args.input_dir is not None and args.output_dir is None and not args.do_print:
        raise SystemExit("Cần --output-dir hoặc --print khi dùng --input-dir")
    input_paths = (
        [args.input]
        if args.input is not None
        else sorted(args.input_dir.glob("*.txt"))
    )
    if not input_paths:
        raise SystemExit("Không tìm thấy file .txt")

    raw_texts_by_id = _load_raw_texts(input_paths)
    outputs = run(args, input_paths, raw_texts_by_id=raw_texts_by_id)
    expected_ids = [path.stem for path in input_paths]
    if set(outputs) != set(expected_ids):
        raise ValueError("Pipeline trả sai tập record ID")

    for record_id in expected_ids:
        output = outputs[record_id]
        raw_text = raw_texts_by_id[record_id]
        inference_io.validate_record_output(output, raw_text=raw_text)
        if args.output_dir is not None:
            inference_io.write_output_json(
                output, args.output_dir / f"{record_id}.json", raw_text=raw_text,
            )
        if args.do_print or args.output_dir is None:
            print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"[cli] Xong: {len(expected_ids)} record.", file=sys.stderr)


if __name__ == "__main__":
    main()
