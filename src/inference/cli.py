"""CLI for detailed CRF/span NER, Qwen3-8B editing, and ontology linking."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
    parser.add_argument("--llm-batch-size", type=int, default=12)
    parser.add_argument("--llm-max-batch-tokens", type=int, default=16384)
    parser.add_argument("--llm-min-batch-size", type=int, default=1)
    parser.add_argument("--no-dynamic-llm-batching", action="store_true")
    parser.add_argument("--llm-device-map", choices=("single_gpu", "auto"), default="single_gpu")
    parser.add_argument("--require-full-gpu", action="store_true")
    parser.add_argument("--llm-local-files-only", action="store_true")
    parser.add_argument("--llm-cache-path", type=Path)
    parser.add_argument("--llm-cache-fsync", action="store_true")
    parser.add_argument("--llm-progress-every", type=int, default=1)
    parser.add_argument("--stage-cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--saved-detailed-ner-dir", type=Path,
        help="bỏ qua hoàn toàn NER và load *.detailed.json để benchmark LLM-only",
    )
    parser.add_argument(
        "--save-detailed-ner-dir", type=Path,
        help="xuất detailed NER portable cho lần benchmark LLM-only sau",
    )
    parser.add_argument("--export-detailed-ner-only", action="store_true")
    parser.add_argument("--llm-audit-dir", type=Path)
    parser.add_argument("--linking-selector-cache-path", type=Path)
    parser.add_argument("--linking-selector-batch-size", type=int, default=4)
    parser.add_argument("--rxnorm-retrieval-k", type=int, default=50)
    parser.add_argument("--icd10-retrieval-k", type=int, default=50)
    parser.add_argument("--linking-audit-dir", type=Path)
    parser.add_argument("--no-llm-recovery", action="store_true")
    parser.add_argument("--review-only-auto-add-eligible", action="store_true")
    parser.add_argument("--speed-profile", choices=("fast", "balanced", "full"), default="balanced")
    parser.add_argument("--max-recovery-proposals-per-record", type=int, default=24)
    parser.add_argument("--max-recovery-requests-per-record", type=int, default=4)

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


def _stable_object(value):
    if isinstance(value, dict):
        return {
            str(key): _stable_object(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_object(item) for item in value]
    if isinstance(value, set):
        return sorted(_stable_object(item) for item in value)
    if hasattr(value, "__dict__"):
        return _stable_object(vars(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fingerprint(value) -> str:
    payload = json.dumps(
        _stable_object(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _editor_evidence_fingerprint(detailed_by_id: dict) -> str:
    fields = (
        "crf_entities",
        "span_candidates",
        "lattice_entities",
        "local_verifications",
        "final_entities",
    )
    payload = {
        record_id: {
            field: getattr(detail, field, [])
            for field in fields
        }
        for record_id, detail in detailed_by_id.items()
    }
    return _fingerprint(payload)


def _entities_fingerprint(entities_by_id: dict) -> str:
    return _fingerprint({
        record_id: [
            {
                "text": entity.text,
                "type": entity.type,
                "assertions": list(entity.assertions),
                "position": list(entity.position),
            }
            for entity in entities
        ]
        for record_id, entities in entities_by_id.items()
    })


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
    run_started = time.perf_counter()
    stage_times: dict[str, float] = {}
    print(
        f"[cli] speed_profile={args.speed_profile} batch={args.llm_batch_size} "
        f"max_batch_tokens={args.llm_max_batch_tokens} dynamic={not args.no_dynamic_llm_batching}",
        file=sys.stderr, flush=True,
    )
    stage_started = time.perf_counter()
    print(
        "[cli] Đang load saved detailed-NER artifacts (không load NER engine)..."
        if args.saved_detailed_ner_dir is not None else "[cli] Đang load NER engine...",
        file=sys.stderr,
    )
    pipeline = InferencePipeline.load_linking_only(
        with_rxnorm=args.with_rxnorm, with_icd10=args.with_icd10,
    ) if args.saved_detailed_ner_dir is not None else InferencePipeline.load(
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
    stage_times["load_ner"] = (
        0.0 if args.saved_detailed_ner_dir is not None
        else time.perf_counter() - stage_started
    )
    predict_kwargs = {"apply_repair_gate": not args.no_repair_gate}
    stage_cache_root = args.stage_cache_dir or (
        args.output_dir / ".stages" if args.output_dir is not None else None
    )

    use_editor = args.with_llm_8b or args.with_llm_ner_editor
    use_selector = (args.with_llm_8b or args.with_llm_linking_selector) and not args.deterministic_linking_only
    if use_editor:
        from .ner.qwen_editor import PROMPT_VERSION as editor_prompt_version
    else:
        editor_prompt_version = "editor_disabled"
    if args.saved_detailed_ner_dir is not None and not use_editor:
        raise ValueError("--saved-detailed-ner-dir requires --with-llm-8b or --with-llm-ner-editor")
    if args.saved_detailed_ner_dir is not None:
        print("[cli] LLM-only benchmark: NER engine/inference skipped", file=sys.stderr, flush=True)
    detailed_by_id = None
    if use_editor or args.export_detailed_ner_only:
        stage_started = time.perf_counter()
        detailed_cache = None
        detailed_by_id = {}
        if args.saved_detailed_ner_dir is not None:
            from .detailed_artifacts import load_artifacts
            detailed_by_id = load_artifacts(args.saved_detailed_ner_dir, raw_texts_by_id)
        if stage_cache_root is not None:
            from .stage_cache import StageCache
            detailed_cache = StageCache(stage_cache_root, "detailed_ner", {
                "stage_schema": "detailed_ner_v2",
                "checkpoint": str(args.checkpoint), "model_config": str(args.model_config),
                "backbone": args.backbone, "predict": predict_kwargs,
            })
            if args.resume:
                detailed_by_id = {
                    record_id: cached for record_id, raw_text in raw_texts_by_id.items()
                    if (cached := detailed_cache.load(record_id, raw_text)) is not None
                }
        missing_raw = {
            record_id: raw_text for record_id, raw_text in raw_texts_by_id.items()
            if record_id not in detailed_by_id
        }
        computed = pipeline.run_ner_stage_detailed(
            missing_raw,
            on_record_complete=(detailed_cache.put if detailed_cache else None),
            **predict_kwargs,
        )
        detailed_by_id.update(computed)
        detailed_by_id = {record_id: detailed_by_id[record_id] for record_id in raw_texts_by_id}
        if args.save_detailed_ner_dir is not None:
            from .detailed_artifacts import save_artifact
            for record_id, detail in detailed_by_id.items():
                save_artifact(
                    args.save_detailed_ner_dir, record_id,
                    raw_texts_by_id[record_id], detail,
                )
        entities_by_id = {
            record_id: list(detail.final_entities)
            for record_id, detail in detailed_by_id.items()
        }
        stage_times["detailed_ner"] = time.perf_counter() - stage_started
    else:
        stage_started = time.perf_counter()
        entities_by_id = pipeline.run_ner_stage(
            raw_texts_by_id, **predict_kwargs,
        )
        stage_times["ner"] = time.perf_counter() - stage_started

    if args.export_detailed_ner_only:
        if args.save_detailed_ner_dir is None:
            raise ValueError("--export-detailed-ner-only requires --save-detailed-ner-dir")
        stage_times["total"] = time.perf_counter() - run_started
        print("[cli] detailed-NER artifacts exported; LLM/linking skipped", file=sys.stderr, flush=True)
        return pipeline.build_outputs(entities_by_id, {record_id: {} for record_id in entities_by_id})

    qwen_llm = None
    if use_editor or use_selector:
        from ..llm.backend import LocalLLM
        from ..llm.batching import VersionedJsonlCache
        from ..llm.config import QWEN3_8B_EDITOR_CONFIG

        for cache_path in (args.llm_cache_path, args.linking_selector_cache_path):
            if cache_path is not None:
                VersionedJsonlCache(cache_path, fsync=args.llm_cache_fsync)
                print(f"[cli] cache ready: {cache_path}", file=sys.stderr, flush=True)

        llm_config = replace(
            QWEN3_8B_EDITOR_CONFIG,
            model_id=args.llm_model_id,
            dtype=args.llm_dtype,
            load_in_4bit=args.llm_quantization == "4bit",
            batch_size=args.llm_batch_size,
            max_batch_tokens=args.llm_max_batch_tokens,
            min_batch_size=args.llm_min_batch_size,
            dynamic_batching=not args.no_dynamic_llm_batching,
            device_map_mode=args.llm_device_map,
            require_full_gpu=args.require_full_gpu,
            local_files_only=args.llm_local_files_only,
        )
        if pipeline.ner_engine is not None:
            pipeline.ner_engine.offload_to_cpu()
        print(f"[cli] Đang load {llm_config.model_id}...", file=sys.stderr)
        qwen_llm = LocalLLM(llm_config)
        qwen_llm.load()
        stage_times["load_llm"] = qwen_llm.load_stats.get("load_seconds", 0.0)
    if use_editor:
        stage_started = time.perf_counter()
        editor_evidence_fingerprint = _editor_evidence_fingerprint(detailed_by_id)
        editor_cache = None
        resumed_entities = {}
        if stage_cache_root is not None:
            from .stage_cache import StageCache
            editor_cache = StageCache(stage_cache_root, "editor", {
                "stage_schema": "editor_entities_v2",
                "model": args.llm_model_id, "profile": args.speed_profile,
                "recovery": not args.no_llm_recovery,
                "prompt": editor_prompt_version,
                "editor_evidence_fingerprint": editor_evidence_fingerprint,
            })
            if args.resume:
                resumed_entities = {
                    record_id: cached for record_id, raw_text in raw_texts_by_id.items()
                    if (cached := editor_cache.load(record_id, raw_text)) is not None
                }
        editor_raw = {key: value for key, value in raw_texts_by_id.items() if key not in resumed_entities}
        editor_details = {key: detailed_by_id[key] for key in editor_raw}
        computed_entities = pipeline.run_qwen8b_ner_editor_stage(
            editor_raw,
            editor_details,
            qwen_llm,
            batch_size=args.llm_batch_size,
            include_recovery=not args.no_llm_recovery,
            review_only_auto_add_eligible=args.review_only_auto_add_eligible,
            cache_path=args.llm_cache_path,
            model_id=args.llm_model_id,
            max_batch_tokens=args.llm_max_batch_tokens,
            min_batch_size=args.llm_min_batch_size,
            dynamic_batching=not args.no_dynamic_llm_batching,
            cache_fsync=args.llm_cache_fsync,
            progress_every=args.llm_progress_every,
            speed_profile=args.speed_profile,
            max_recovery_proposals_per_record=args.max_recovery_proposals_per_record,
            max_recovery_requests_per_record=args.max_recovery_requests_per_record,
        )
        for record_id, value in computed_entities.items():
            if editor_cache:
                editor_cache.put(record_id, raw_texts_by_id[record_id], value)
        if stage_cache_root is not None:
            recovery_cache = StageCache(stage_cache_root, "recovery", {
                "stage_schema": "recovery_entities_v2",
                "editor_prompt": editor_prompt_version,
                "editor_evidence_fingerprint": editor_evidence_fingerprint,
                "profile": args.speed_profile, "enabled": not args.no_llm_recovery,
            })
            for record_id, value in computed_entities.items():
                recovery_cache.put(record_id, raw_texts_by_id[record_id], value)
        resumed_entities.update(computed_entities)
        entities_by_id = {record_id: resumed_entities[record_id] for record_id in raw_texts_by_id}
        stage_times["editor_and_recovery"] = time.perf_counter() - stage_started
        stage_times.update(pipeline.last_editor_stage_times)
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

    stage_started = time.perf_counter()
    entities_fingerprint = _entities_fingerprint(entities_by_id)
    linking_cache = None
    candidates_by_id = {}
    if stage_cache_root is not None:
        from .stage_cache import StageCache
        linking_cache = StageCache(stage_cache_root, "linking_selector", {
            "stage_schema": "linking_selector_v2",
            "rxnorm": args.with_rxnorm, "icd10": args.with_icd10,
            "selector": use_selector, "model": args.llm_model_id,
            "rx_k": args.rxnorm_retrieval_k, "icd_k": args.icd10_retrieval_k,
            "editor_prompt": editor_prompt_version,
            "entities_fingerprint": entities_fingerprint,
            "profile": args.speed_profile, "recovery": not args.no_llm_recovery,
        })
        if args.resume:
            candidates_by_id = {
                record_id: cached for record_id, raw_text in raw_texts_by_id.items()
                if (cached := linking_cache.load(record_id, raw_text)) is not None
            }
    linking_entities = {key: value for key, value in entities_by_id.items() if key not in candidates_by_id}
    linking_raw = {key: raw_texts_by_id[key] for key in linking_entities}
    retrieval_cache = None
    retrieval_by_id = {}
    if stage_cache_root is not None:
        retrieval_cache = StageCache(stage_cache_root, "linking_retrieval", {
            "stage_schema": "linking_retrieval_v2",
            "rxnorm": args.with_rxnorm, "icd10": args.with_icd10,
            "rx_k": args.rxnorm_retrieval_k, "icd_k": args.icd10_retrieval_k,
            "editor_prompt": editor_prompt_version,
            "entities_fingerprint": entities_fingerprint,
            "profile": args.speed_profile, "recovery": not args.no_llm_recovery,
        })
        if args.resume:
            retrieval_by_id = {
                record_id: cached for record_id, raw_text in linking_raw.items()
                if (cached := retrieval_cache.load(record_id, raw_text)) is not None
            }
    retrieval_entities = {
        key: value for key, value in linking_entities.items() if key not in retrieval_by_id
    }
    retrieval_started = time.perf_counter()
    computed_retrieval = pipeline.run_linking_retrieval_stage_batch(
        retrieval_entities,
        top_k_by_linker={
            "rxnorm": args.rxnorm_retrieval_k,
            "icd10": args.icd10_retrieval_k,
        },
    )
    for record_id, value in computed_retrieval.items():
        if retrieval_cache:
            retrieval_cache.put(record_id, raw_texts_by_id[record_id], value)
    retrieval_by_id.update(computed_retrieval)
    pipeline.last_linking_retrieval = retrieval_by_id
    stage_times["linking_retrieval"] = time.perf_counter() - retrieval_started
    selector_started = time.perf_counter()
    computed_candidates = pipeline.run_qwen8b_linking_selector_stage(
        linking_entities,
        retrieval_by_id,
        selector_llm=qwen_llm if use_selector else None,
        raw_texts_by_id=linking_raw,
        batch_size=args.linking_selector_batch_size,
        cache_path=args.linking_selector_cache_path,
        model_id=args.llm_model_id,
    )
    stage_times["linking_selector"] = time.perf_counter() - selector_started
    for record_id, value in computed_candidates.items():
        if linking_cache:
            linking_cache.put(record_id, raw_texts_by_id[record_id], value)
    candidates_by_id.update(computed_candidates)
    candidates_by_id = {record_id: candidates_by_id[record_id] for record_id in raw_texts_by_id}
    stage_times["linking_retrieval_and_selector"] = time.perf_counter() - stage_started
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
        _write_stage_checkpoint(args.output_dir, "llm_workload", pipeline.last_llm_workload)
        _write_stage_checkpoint(args.output_dir, "llm_batch_telemetry", {
            "load": qwen_llm.load_stats,
            "microbatches": qwen_llm.batch_generation_stats,
        })
        qwen_llm.unload()
    outputs = pipeline.build_outputs(entities_by_id, candidates_by_id)
    if stage_cache_root is not None:
        from .stage_cache import StageCache
        final_cache = StageCache(stage_cache_root, "final", {
            "stage_schema": "final_output_v2",
            "editor_prompt": editor_prompt_version,
            "entities_fingerprint": entities_fingerprint,
            "profile": args.speed_profile, "selector": use_selector,
            "rx_k": args.rxnorm_retrieval_k, "icd_k": args.icd10_retrieval_k,
        })
        for record_id, value in outputs.items():
            final_cache.put(record_id, raw_texts_by_id[record_id], value)
    stage_times["total"] = time.perf_counter() - run_started
    print("[cli] stage_runtime_seconds=" + json.dumps(
        {key: round(value, 3) for key, value in stage_times.items()},
        ensure_ascii=False,
    ), file=sys.stderr, flush=True)
    if args.output_dir is not None:
        _write_stage_checkpoint(args.output_dir, "runtime", stage_times)
    return outputs


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