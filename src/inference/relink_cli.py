"""Re-run RxNorm/ICD linking on saved BTC entities without re-running NER.

The command validates every saved span against the original input, applies
the current deterministic cleanup, discards all old candidate codes, then
retrieves/selects candidates again into a separate output directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import config as cfg
from . import io as inference_io
from .pipeline import InferencePipeline
from .rule.clinical import deterministic_cleanup
from .schemas import NerEntity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--entities-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-rxnorm", action="store_true")
    parser.add_argument("--with-icd10", action="store_true")
    parser.add_argument(
        "--with-llm-selector",
        action="store_true",
        help="Use Qwen 7B only for candidate selection; NER is not rerun.",
    )
    return parser.parse_args()


def _natural_key(path: Path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.stem)
        if part
    )


def _load_saved_entities(path: Path, raw_text: str) -> list[NerEntity]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: root phải là JSON list")
    entities = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: entity {index} không phải object")
        position = item.get("position")
        if not (
            isinstance(position, list)
            and len(position) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in position)
        ):
            raise ValueError(f"{path}: entity {index} có position sai")
        start, end = position
        text = item.get("text")
        if not isinstance(text, str) or not (0 <= start < end <= len(raw_text)) \
                or raw_text[start:end] != text:
            raise ValueError(f"{path}: entity {index} không khớp raw offset")
        entities.append(NerEntity(
            text=text,
            type=str(item.get("type", "")),
            assertions=list(item.get("assertions") or []),
            position=(start, end),
            score=1.0,
        ))
    return entities


def _load_linking_pipeline(*, with_rxnorm: bool, with_icd10: bool) -> InferencePipeline:
    rxnorm_linker = None
    if with_rxnorm and cfg.RXNORM_INDEX_DIR is not None:
        from ..linking.rxnorm.linker import RxNormLinker

        rxnorm_linker = RxNormLinker(
            index_dir=str(cfg.RXNORM_INDEX_DIR),
            clean_path=str(cfg.RXNORM_CLEAN_PATH) if cfg.RXNORM_CLEAN_PATH else None,
            device=cfg.LINKER_DEVICE,
        )
    icd10_linker = None
    if with_icd10 and cfg.ICD10_INDEX_DIR is not None:
        from ..linking.icd10.icd10_linker import Icd10Linker

        icd10_linker = Icd10Linker(
            index_dir=cfg.ICD10_INDEX_DIR,
            device=cfg.LINKER_DEVICE,
        )
    return InferencePipeline(
        ner_engine=None,
        rxnorm_linker=rxnorm_linker,
        icd10_linker=icd10_linker,
    )


def run(args: argparse.Namespace) -> dict[str, int]:
    input_dir = args.input_dir.resolve()
    entities_dir = args.entities_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == entities_dir:
        raise ValueError("--output-dir phải khác --entities-dir để không ghi đè submission cũ")
    if not input_dir.is_dir() or not entities_dir.is_dir():
        raise FileNotFoundError("Không tìm thấy input-dir hoặc entities-dir")

    paths = sorted(input_dir.glob("*.txt"), key=_natural_key)
    if not paths:
        raise ValueError(f"Không có file .txt trong {input_dir}")

    raw_texts: dict[str, str] = {}
    entities_by_id: dict[str, list[NerEntity]] = {}
    before_count = 0
    cleanup_count = 0
    for input_path in paths:
        saved_path = entities_dir / f"{input_path.stem}.json"
        if not saved_path.is_file():
            raise FileNotFoundError(f"Thiếu saved entity file: {saved_path}")
        raw_text = inference_io.read_text_file(input_path)
        saved = _load_saved_entities(saved_path, raw_text)
        cleaned, logs = deterministic_cleanup(raw_text, saved)
        raw_texts[input_path.stem] = raw_text
        entities_by_id[input_path.stem] = cleaned
        before_count += len(saved)
        cleanup_count += sum(log.get("status") in {"repair", "drop"} for log in logs)

    pipeline = _load_linking_pipeline(
        with_rxnorm=args.with_rxnorm,
        with_icd10=args.with_icd10,
    )
    selector_llm = None
    if args.with_llm_selector:
        from ..llm.backend import LocalLLM
        from ..llm.config import NER_REVIEWER_7B_CONFIG

        selector_llm = LocalLLM(NER_REVIEWER_7B_CONFIG)
        selector_llm.load()
    try:
        candidates_by_id = pipeline.run_linking_stage(
            entities_by_id,
            selector_llm=selector_llm,
            raw_texts_by_id=raw_texts,
        )
    finally:
        if selector_llm is not None:
            selector_llm.unload()

    outputs = pipeline.build_outputs(entities_by_id, candidates_by_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    linked_count = 0
    for record_id, output in outputs.items():
        inference_io.write_output_json(
            output,
            output_dir / f"{record_id}.json",
            raw_text=raw_texts[record_id],
        )
        linked_count += sum(bool(item.get("candidates")) for item in output)
    stats = {
        "records": len(outputs),
        "entities_before_cleanup": before_count,
        "entities_after_cleanup": sum(len(items) for items in entities_by_id.values()),
        "cleanup_actions": cleanup_count,
        "nonempty_candidates": linked_count,
    }
    print(json.dumps(stats, ensure_ascii=False))
    return stats


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[relink] lỗi: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
