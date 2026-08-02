"""Re-run RxNorm/ICD linking on saved BTC entities without re-running NER.

Default behavior is a *strict relink*:

- validate every saved span against the original input;
- preserve ``text``, ``type``, ``assertions`` and ``position`` exactly;
- discard/recompute candidate codes only;
- write results into a separate output directory.

Entity cleanup is intentionally opt-in via ``--cleanup-entities`` because
cleanup can change WER/assertion inputs and would make a linking A/B test
invalid.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from . import config as cfg
from . import io as inference_io
from .pipeline import InferencePipeline
from .schemas import NerEntity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--entities-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-rxnorm", action="store_true")
    parser.add_argument("--with-icd10", action="store_true")
    parser.add_argument(
        "--with-llm-8b",
        action="store_true",
        help="Use Qwen3-8B only for candidate selection; NER is not rerun.",
    )
    parser.add_argument(
        "--cleanup-entities",
        action="store_true",
        help=(
            "Opt in to deterministic entity cleanup before relinking. "
            "By default saved text/type/assertions/position are preserved exactly."
        ),
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

    entities: list[NerEntity] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: entity {index} không phải object")

        position = item.get("position")
        if not (
            isinstance(position, list)
            and len(position) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in position
            )
        ):
            raise ValueError(f"{path}: entity {index} có position sai")

        start, end = position
        text = item.get("text")
        if (
            not isinstance(text, str)
            or not (0 <= start < end <= len(raw_text))
            or raw_text[start:end] != text
        ):
            actual = (
                raw_text[start:end]
                if isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end <= len(raw_text)
                else "<invalid range>"
            )
            raise ValueError(
                f"{path}: entity {index} không khớp raw offset: "
                f"position={[start, end]!r}, text={text!r}, raw_slice={actual!r}"
            )

        raw_assertions = item.get("assertions") or []
        if not isinstance(raw_assertions, list) or not all(
            isinstance(value, str) for value in raw_assertions
        ):
            raise ValueError(f"{path}: entity {index} có assertions sai")

        entities.append(
            NerEntity(
                text=text,
                type=str(item.get("type", "")),
                assertions=list(raw_assertions),
                position=(start, end),
                score=1.0,
            )
        )

    return entities


def _entity_identity(entity: NerEntity) -> tuple[str, str, tuple[str, ...], tuple[int, int]]:
    """Fields that a strict relink is forbidden to change."""
    return (
        entity.text,
        entity.type,
        tuple(entity.assertions or []),
        tuple(entity.position),
    )


def _output_identity(item: dict[str, Any]) -> tuple[str, str, tuple[str, ...], tuple[int, int]]:
    position = item.get("position")
    assertions = item.get("assertions") or []
    return (
        str(item.get("text", "")),
        str(item.get("type", "")),
        tuple(assertions),
        tuple(position) if isinstance(position, (list, tuple)) else tuple(),
    )


def _snapshot_entities(
    entities_by_id: dict[str, list[NerEntity]],
) -> dict[str, Counter]:
    """Create an order-independent snapshot of immutable NER fields."""
    return {
        record_id: Counter(_entity_identity(entity) for entity in entities)
        for record_id, entities in entities_by_id.items()
    }


def _assert_entities_preserved(
    expected: dict[str, Counter],
    entities_by_id: dict[str, list[NerEntity]],
    *,
    stage: str,
) -> None:
    actual = _snapshot_entities(entities_by_id)
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise RuntimeError(
            f"Strict relink bị vi phạm tại {stage}: "
            f"record thiếu={missing}, record dư={extra}"
        )

    for record_id in expected:
        if actual[record_id] != expected[record_id]:
            removed = expected[record_id] - actual[record_id]
            added = actual[record_id] - expected[record_id]
            raise RuntimeError(
                f"Strict relink bị vi phạm tại {stage}, record={record_id}: "
                f"NER bị xóa/sửa={list(removed.elements())[:5]!r}, "
                f"NER bị thêm/sửa={list(added.elements())[:5]!r}"
            )


def _assert_outputs_preserve_entities(
    expected: dict[str, Counter],
    outputs: dict[str, list[dict[str, Any]]],
) -> None:
    actual = {
        record_id: Counter(_output_identity(item) for item in items)
        for record_id, items in outputs.items()
    }

    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise RuntimeError(
            "Strict relink bị vi phạm khi build output: "
            f"record thiếu={missing}, record dư={extra}"
        )

    for record_id in expected:
        if actual[record_id] != expected[record_id]:
            removed = expected[record_id] - actual[record_id]
            added = actual[record_id] - expected[record_id]
            raise RuntimeError(
                f"Strict relink bị vi phạm khi build output, record={record_id}: "
                f"NER bị xóa/sửa={list(removed.elements())[:5]!r}, "
                f"NER bị thêm/sửa={list(added.elements())[:5]!r}"
            )


def _prepare_entities_for_relink(
    raw_text: str,
    saved: list[NerEntity],
    *,
    cleanup_entities: bool,
) -> tuple[list[NerEntity], list[dict[str, Any]]]:
    """Preserve saved entities by default; cleanup only when explicitly enabled."""
    if not cleanup_entities:
        # Copy the list so later stages cannot add/remove entries from the caller's
        # list object accidentally. NerEntity fields are guarded by snapshots below.
        return list(saved), []

    # Lazy import keeps strict relink independent from semantic cleanup code and
    # makes the opt-in behavior explicit.
    from .rule.clinical import deterministic_cleanup

    cleaned, logs = deterministic_cleanup(raw_text, saved)
    return cleaned, logs


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


def run(args: argparse.Namespace) -> dict[str, int | bool | str]:
    input_dir = args.input_dir.resolve()
    entities_dir = args.entities_dir.resolve()
    output_dir = args.output_dir.resolve()
    cleanup_entities = bool(getattr(args, "cleanup_entities", False))

    if output_dir == entities_dir:
        raise ValueError(
            "--output-dir phải khác --entities-dir để không ghi đè submission cũ"
        )
    if not input_dir.is_dir() or not entities_dir.is_dir():
        raise FileNotFoundError("Không tìm thấy input-dir hoặc entities-dir")

    paths = sorted(input_dir.glob("*.txt"), key=_natural_key)
    if not paths:
        raise ValueError(f"Không có file .txt trong {input_dir}")

    raw_texts: dict[str, str] = {}
    entities_by_id: dict[str, list[NerEntity]] = {}
    loaded_count = 0
    cleanup_count = 0

    for input_path in paths:
        saved_path = entities_dir / f"{input_path.stem}.json"
        if not saved_path.is_file():
            raise FileNotFoundError(f"Thiếu saved entity file: {saved_path}")

        raw_text = inference_io.read_text_file(input_path)
        saved = _load_saved_entities(saved_path, raw_text)
        working_entities, logs = _prepare_entities_for_relink(
            raw_text,
            saved,
            cleanup_entities=cleanup_entities,
        )

        raw_texts[input_path.stem] = raw_text
        entities_by_id[input_path.stem] = working_entities
        loaded_count += len(saved)
        cleanup_count += sum(
            log.get("status") in {"repair", "drop"}
            for log in logs
            if isinstance(log, dict)
        )

    # Snapshot immediately before linking. In strict mode this is exactly the
    # saved NER payload; in opt-in cleanup mode it is the explicit post-cleanup
    # payload. Linking is forbidden to change it in either mode.
    ner_snapshot = _snapshot_entities(entities_by_id)

    pipeline = _load_linking_pipeline(
        with_rxnorm=args.with_rxnorm,
        with_icd10=args.with_icd10,
    )

    selector_llm = None
    if getattr(args, "with_llm_8b", False):
        from ..llm.backend import LocalLLM
        from ..llm.config import QWEN3_8B_EDITOR_CONFIG

        selector_llm = LocalLLM(QWEN3_8B_EDITOR_CONFIG)
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

    _assert_entities_preserved(
        ner_snapshot,
        entities_by_id,
        stage="sau run_linking_stage",
    )

    outputs = pipeline.build_outputs(entities_by_id, candidates_by_id)
    _assert_entities_preserved(
        ner_snapshot,
        entities_by_id,
        stage="sau build_outputs",
    )
    _assert_outputs_preserve_entities(ner_snapshot, outputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    linked_count = 0
    for record_id, output in outputs.items():
        inference_io.write_output_json(
            output,
            output_dir / f"{record_id}.json",
            raw_text=raw_texts[record_id],
        )
        linked_count += sum(bool(item.get("candidates")) for item in output)

    entities_for_linking = sum(len(items) for items in entities_by_id.values())
    stats: dict[str, int | bool | str] = {
        "records": len(outputs),
        "mode": "cleanup_then_relink" if cleanup_entities else "strict_relink",
        "cleanup_enabled": cleanup_entities,
        # Keep the old names so existing scripts parsing these fields do not break.
        "entities_before_cleanup": loaded_count,
        "entities_after_cleanup": entities_for_linking,
        "cleanup_actions": cleanup_count,
        "ner_preserved_during_linking": True,
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
