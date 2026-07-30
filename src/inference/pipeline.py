"""Điều phối: two-pass NER -> 1.5B fixer -> 7B review -> linking -> BTC JSON.

Hai LLM được load tuần tự: Qwen2.5-1.5B sửa batch rồi unload; Qwen 7B review
NER, rerank candidate linking, rồi mới unload.

Luồng dùng đúng cho batch (xem cli.py):
    entities_by_id = pipeline.run_ner_stage(raw_texts_by_id)
    fixer_1_5b.load(); entities_by_id = pipeline.run_fixer_stage(...)
    fixer_1_5b.unload()
    reviewer_7b.load(); entities_by_id = pipeline.run_7b_ner_stage(...)
    candidates_by_id = pipeline.run_linking_stage(entities_by_id, selector_llm=reviewer_7b)
    reviewer_7b.unload()
    outputs = pipeline.build_outputs(entities_by_id, candidates_by_id)

`process_record()` / `process_file()` vẫn giữ lại cho test nhanh 1 file
(1 record thì load/unload 1 lần cũng không sao) — nhận thẳng instance
LocalLLM ĐÃ LOAD (fixer_llm=..., selector_llm=...), không tự load bên
trong pipeline nữa.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from . import config as cfg
from . import io as inference_io
from .ner.postprocessor import clean_text_for_inference
from .schemas import NerEntity

if TYPE_CHECKING:
    from .ner.engine import NerEngine

# Map type NER -> tên linker sẽ dùng trong self._linkers. Chỉ 2 type có
# linker đã build; các type khác luôn có candidates=[] (đúng ví dụ BTC,
# TRIỆU_CHỨNG/TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM không có candidates).
TYPE_TO_LINKER = {
    "THUỐC": "rxnorm",
    "CHẨN_ĐOÁN": "icd10",
}

MAX_OUTPUT_CANDIDATES = {
    "THUỐC": 1,
    "CHẨN_ĐOÁN": 2,
}


def _extract_codes(candidates: list, top_k: int, key_priority: tuple[str, ...] = ("code", "rxcui")) -> list[str]:
    """Rút list code từ output linker — hỗ trợ cả 2 dạng candidate:
      - RxNormCandidate (dataclass, xem linking/rxnorm/schemas.py): field
        code nằm ở `.rxcui`, KHÔNG subscriptable như dict.
      - dict (ICD-10, xem aggregate_term_results trong icd10_linker.py):
        field code nằm ở key "code".
    key_priority thử theo thứ tự, dùng chung cho cả 2 dạng qua getattr
    lẫn __getitem__."""
    codes = []
    for cand in candidates[:top_k]:
        if isinstance(cand, str):
            codes.append(cand)
            continue
        for key in key_priority:
            if hasattr(cand, key):
                codes.append(str(getattr(cand, key)))
                break
            if isinstance(cand, dict) and key in cand:
                codes.append(str(cand[key]))
                break
    return codes


class InferencePipeline:
    def __init__(
        self,
        ner_engine: NerEngine,
        rxnorm_linker=None,
        icd10_linker=None,
        *,
        top_k_candidates: int = cfg.LINKER_TOP_K,
    ):
        self.ner_engine = ner_engine
        self._linkers = {"rxnorm": rxnorm_linker, "icd10": icd10_linker}
        self.top_k_candidates = top_k_candidates
        self.last_two_pass_results = {}
        self.last_handoffs = {}
        self.last_fixer_logs = []
        self.last_7b_logs = []

    @classmethod
    def load(
        cls,
        *,
        with_rxnorm: bool = False,
        with_icd10: bool = False,
        **ner_engine_kwargs,
    ) -> "InferencePipeline":
        """Load NerEngine luôn; RxNormLinker/Icd10Linker chỉ load nếu
        with_rxnorm/with_icd10=True VÀ path tương ứng trong config khác
        None — để test riêng NER không bắt buộc build FAISS index.
        KHÔNG load LLM ở đây — LLM lifecycle do caller (cli.py) quản lý
        riêng, xem docstring module."""
        from .ner.engine import NerEngine

        ner_engine = NerEngine.load(**ner_engine_kwargs)

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

        return cls(ner_engine, rxnorm_linker, icd10_linker)

    # ------------------------------------------------------------
    # Stage 1: NER (không LLM) — an toàn chạy cho cả batch bất kỳ lúc nào
    # ------------------------------------------------------------
    def run_ner_stage(
        self,
        raw_texts_by_id: dict[str, str],
        *,
        two_pass: bool = True,
        maximum_second_pass_regions: int = 24,
        **predict_kwargs,
    ) -> dict[str, list[NerEntity]]:
        from .ner.two_pass import run_two_pass_ner
        from .ner.sectioner import split_sections_by_header
        from .rule.clinical import apply_clinical_rules
        from .rule.routing import build_handoff_requests

        entities_by_id: dict[str, list[NerEntity]] = {}
        for rid, raw_text in raw_texts_by_id.items():
            pass1 = []
            for block in split_sections_by_header(raw_text).values():
                block_text = block["body"]
                if not block_text.strip():
                    continue
                cleaned = clean_text_for_inference(block_text)
                if len(cleaned) != len(block_text):
                    raise ValueError("clean_text_for_inference changed block length")
                local_entities = self.ner_engine.predict_text(cleaned, **predict_kwargs)
                block_start = int(block["start"])
                for entity in local_entities:
                    local_start, local_end = entity.position
                    global_start = block_start + local_start
                    global_end = block_start + local_end
                    if not (0 <= local_start < local_end <= len(block_text)):
                        continue
                    if block_text[local_start:local_end] != entity.text:
                        continue
                    if raw_text[global_start:global_end] != entity.text:
                        continue
                    pass1.append(NerEntity(
                        text=entity.text,
                        type=entity.type,
                        assertions=list(entity.assertions),
                        position=(global_start, global_end),
                        score=entity.score,
                        flag=entity.flag,
                        review_hints=list(entity.review_hints),
                    ))
            pass1.sort(key=lambda entity: (
                entity.position[0], entity.position[1], entity.type,
            ))
            if two_pass:
                def predict_region(region_text: str) -> list[NerEntity]:
                    cleaned_region = clean_text_for_inference(region_text)
                    local = self.ner_engine.predict_text(cleaned_region, **predict_kwargs)
                    return inference_io.remap_entities_to_raw(region_text, cleaned_region, local)

                result = run_two_pass_ner(
                    raw_text,
                    pass1,
                    predict_region,
                    maximum_regions=maximum_second_pass_regions,
                )
                final = result.final_entities
                self.last_two_pass_results[rid] = result
                self.last_handoffs[rid] = build_handoff_requests(
                    raw_text, final, result.regions, request_prefix=rid
                )
            else:
                final, logs = apply_clinical_rules(raw_text, pass1)
                self.last_two_pass_results[rid] = None
                self.last_handoffs[rid] = build_handoff_requests(
                    raw_text, final, [], request_prefix=rid
                )
            entities_by_id[rid] = final
        return entities_by_id

    # ------------------------------------------------------------
    # Stage 2 (optional): grouped 7B NER review/recovery, before linking.
    # ------------------------------------------------------------
    def run_fixer_stage(
        self,
        raw_texts_by_id: dict[str, str],
        entities_by_id: dict[str, list[NerEntity]],
        fixer_llm,
        *,
        audit_missing: bool = True,
        batch_size: int = 4,
    ) -> dict[str, list[NerEntity]]:
        """Run the notebook small-model stage before the separate 7B stage.

        Qwen2.5-1.5B reviews candidates flagged by the deterministic gates in
        batches.  Its output is schema/offset checked, then the optional recall
        audit is also batched.  Final deterministic cleanup prevents an unsafe
        small-model suggestion from reaching 7B/linking unchecked.
        """
        from .ner.llm_fixer import (
            audit_missing_entities_batch,
            fix_flagged_entities_batch,
        )
        from .rule.clinical import deterministic_cleanup
        from .rule.routing import build_handoff_requests

        fixed = fix_flagged_entities_batch(
            raw_texts_by_id,
            entities_by_id,
            fixer_llm,
            batch_size=batch_size,
        )
        if audit_missing:
            fixed = audit_missing_entities_batch(
                raw_texts_by_id,
                fixed,
                fixer_llm,
                batch_size=batch_size,
            )

        cleaned = {}
        logs = []
        for record_id, entities in fixed.items():
            cleaned[record_id], record_logs = deterministic_cleanup(
                raw_texts_by_id[record_id], entities
            )
            logs.extend({"record_id": record_id, **item} for item in record_logs)
            previous = self.last_two_pass_results.get(record_id)
            regions = previous.regions if previous is not None else []
            self.last_handoffs[record_id] = build_handoff_requests(
                raw_texts_by_id[record_id], cleaned[record_id], regions,
                request_prefix=record_id,
            )
        self.last_fixer_logs = logs
        return cleaned

    def run_7b_ner_stage(
        self,
        raw_texts_by_id: dict[str, str],
        entities_by_id: dict[str, list[NerEntity]],
        reviewer_llm,
        *,
        batch_size: int = 4,
        retry_rounds: int = 1,
        include_recovery: bool = True,
    ) -> dict[str, list[NerEntity]]:
        from .ner.reviewer_7b import review_entities_batch

        handoffs = self.last_handoffs
        if not include_recovery:
            handoffs = {
                rid: {**handoff, "region_recoveries": [], "region_recovery_count": 0}
                for rid, handoff in handoffs.items()
            }
        reviewed, logs = review_entities_batch(
            raw_texts_by_id,
            entities_by_id,
            handoffs,
            reviewer_llm,
            batch_size=batch_size,
            retry_rounds=retry_rounds,
        )
        self.last_7b_logs = logs
        return reviewed

    # ------------------------------------------------------------
    # Stage 3: existing retrievers followed by optional 7B candidate selection.
    # ------------------------------------------------------------
    def _get_raw_candidates(self, ent: NerEntity):
        """Trả candidate RAW (list[RxNormCandidate] hoặc list[dict]) CHƯA
        cắt xuống code — dùng chung cho cả nhánh có/không LLM selector."""
        from .rule.clinical import is_linkable_entity

        if not is_linkable_entity(ent):
            return None
        linker_name = TYPE_TO_LINKER.get(ent.type)
        if linker_name is None:
            return None
        linker = self._linkers.get(linker_name)
        if linker is None:
            return None

        try:
            if linker_name == "rxnorm":
                result = linker.link(ent.text, top_k=self.top_k_candidates)
                return result["candidates"]
            if linker_name == "icd10":
                return linker.link(ent.text, top_k_codes=self.top_k_candidates)
        except Exception as exc:
            print(f"[pipeline] linking lỗi cho '{ent.text}' ({ent.type}): {exc}", file=sys.stderr)
            return None

    def attach_candidates(
        self,
        entities: list[NerEntity],
        *,
        selector_llm=None,
        raw_text: str | None = None,
    ) -> dict[int, list[str]]:
        """Attach linker candidates, optionally reranked by the loaded 7B.

        The selector may only choose codes already returned by the existing
        retriever; schema validation in ``candidate_selector`` rejects invented
        codes.  NER is already final before this method is called.
        """
        candidates_by_entity: dict[int, list[str]] = {}
        selector_indexes: list[int] = []
        selector_items: list[dict] = []
        for i, ent in enumerate(entities):
            raw_candidates = self._get_raw_candidates(ent)
            if not raw_candidates:
                continue

            if selector_llm is None:
                output_limit = MAX_OUTPUT_CANDIDATES[ent.type]
                candidates_by_entity[i] = _extract_codes(raw_candidates, output_limit)
                continue
            context = ""
            if raw_text is not None:
                start, end = ent.position
                context = raw_text[max(0, start - 120):min(len(raw_text), end + 120)]
            selector_indexes.append(i)
            selector_items.append({
                "entity_text": ent.text,
                "entity_type": ent.type,
                "candidates": raw_candidates,
                "context": context,
            })

        if selector_items:
            from .selection.candidate_selector import select_candidates_many

            selected_batches = select_candidates_many(
                selector_items,
                selector_llm,
                top_k_context=self.top_k_candidates,
            )
            for index, selected in zip(selector_indexes, selected_batches):
                candidates_by_entity[index] = selected

        return candidates_by_entity

    def run_linking_stage(
        self,
        entities_by_id: dict[str, list[NerEntity]],
        *,
        selector_llm=None,
        raw_texts_by_id: dict[str, str] | None = None,
    ) -> dict[str, dict[int, list[str]]]:
        if selector_llm is None:
            return {
                rid: self.attach_candidates(entities)
                for rid, entities in entities_by_id.items()
            }

        from .selection.candidate_selector import select_candidates_many

        results: dict[str, dict[int, list[str]]] = {rid: {} for rid in entities_by_id}
        destinations: list[tuple[str, int]] = []
        selector_items: list[dict] = []
        raw_texts_by_id = raw_texts_by_id or {}
        for rid, entities in entities_by_id.items():
            raw_text = raw_texts_by_id.get(rid)
            for index, entity in enumerate(entities):
                raw_candidates = self._get_raw_candidates(entity)
                if not raw_candidates:
                    continue
                context = ""
                if raw_text is not None:
                    start, end = entity.position
                    context = raw_text[max(0, start - 120):min(len(raw_text), end + 120)]
                destinations.append((rid, index))
                selector_items.append({
                    "entity_text": entity.text,
                    "entity_type": entity.type,
                    "candidates": raw_candidates,
                    "context": context,
                })
        selected_batches = select_candidates_many(
            selector_items,
            selector_llm,
            top_k_context=self.top_k_candidates,
        )
        for (rid, index), selected in zip(destinations, selected_batches):
            results[rid][index] = selected
        return results

    # ------------------------------------------------------------
    # Build output cuối — thuần ghép dữ liệu, không gọi model gì nữa
    # ------------------------------------------------------------
    def build_outputs(
        self,
        entities_by_id: dict[str, list[NerEntity]],
        candidates_by_id: dict[str, dict[int, list[str]]],
    ) -> dict[str, list[dict]]:
        return {
            rid: inference_io.build_record_output(entities, candidates_by_id.get(rid, {}))
            for rid, entities in entities_by_id.items()
        }

    # ------------------------------------------------------------
    # Tiện ích test nhanh 1 record/file — load/unload LLM (nếu có) NGAY
    # TRONG lệnh gọi này KHÔNG hợp lý cho batch, chỉ dùng cho test đơn lẻ.
    # Muốn dùng LLM ở đây, tự load fixer_llm/selector_llm rồi truyền vào.
    # ------------------------------------------------------------
    def process_record(
        self,
        raw_text: str,
        *,
        fixer_llm=None,
        selector_llm=None,
        audit_missing: bool = True,
        **predict_kwargs,
    ) -> list[dict]:
        entities_by_id = self.run_ner_stage({"_single": raw_text}, **predict_kwargs)

        if fixer_llm is not None:
            entities_by_id = self.run_fixer_stage(
                {"_single": raw_text},
                entities_by_id,
                fixer_llm,
                audit_missing=audit_missing,
            )

        candidates_by_id = self.run_linking_stage(
            entities_by_id,
            selector_llm=selector_llm,
            raw_texts_by_id={"_single": raw_text},
        )
        return self.build_outputs(entities_by_id, candidates_by_id)["_single"]

    def process_file(self, path: str | Path, **kwargs) -> list[dict]:
        raw_text = inference_io.read_text_file(path)
        return self.process_record(raw_text, **kwargs)
