"""Production pipeline: detailed CRF/span NER -> Qwen3 editor -> linking."""

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
        self.last_detailed_results = {}
        self.last_editor_audit = {}
        self.last_linking_retrieval = {}

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
        maximum_second_pass_regions: int = cfg.MAXIMUM_SECOND_PASS_REGIONS,
        **predict_kwargs,
    ) -> dict[str, list[NerEntity]]:
        from .ner.two_pass import run_two_pass_ner
        from .ner.sectioner import split_sections_by_header
        from .rule.clinical import apply_clinical_rules

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
            else:
                final, logs = apply_clinical_rules(raw_text, pass1)
                self.last_two_pass_results[rid] = None
            entities_by_id[rid] = final
        return entities_by_id

    def run_ner_stage_detailed(
        self,
        raw_texts_by_id: dict[str, str],
        **predict_kwargs,
    ):
        """Run offset-preserving section inference and retain detailed evidence."""
        from .ner.evidence import NerDetailedResult, SpanCandidateEvidence, WordEvidence
        from .ner.sectioner import split_sections_by_header

        outputs = {}
        for record_id, raw_text in raw_texts_by_id.items():
            combined = NerDetailedResult(len(raw_text), len(raw_text))
            for block_id, block in split_sections_by_header(raw_text).items():
                body = block["body"]
                if not body.strip():
                    continue
                block_start = int(block["start"])
                detail = self.ner_engine.predict_text_detailed(body, **predict_kwargs)

                def shift_entity(entity):
                    start, end = entity.position
                    global_start, global_end = block_start + start, block_start + end
                    if raw_text[global_start:global_end] != entity.text:
                        raise ValueError("detailed section offset mismatch")
                    return NerEntity(
                        entity.text, entity.type, list(entity.assertions),
                        (global_start, global_end), entity.score, entity.flag,
                    )

                combined.crf_entities.extend(shift_entity(item) for item in detail.crf_entities)
                combined.lattice_entities.extend(shift_entity(item) for item in detail.lattice_entities)
                combined.final_entities.extend(shift_entity(item) for item in detail.final_entities)
                combined.span_candidates.extend(SpanCandidateEvidence(
                    item.start + block_start, item.end + block_start, item.type,
                    item.score, item.word_start, item.word_end, item.source,
                ) for item in detail.span_candidates)
                word_base = len(combined.words)
                combined.words.extend(WordEvidence(
                    word_base + item.index, item.text,
                    item.start + block_start, item.end + block_start,
                    item.line_id, block_id, item.crf,
                    item.span_top_label, item.span_top_score,
                ) for item in detail.words)
                combined.thresholds.update(detail.thresholds)
                combined.logs.extend({"block_id": block_id, **item} for item in detail.logs)
                combined.span_head_enabled = combined.span_head_enabled or detail.span_head_enabled
            for field_name in ("crf_entities", "lattice_entities", "final_entities"):
                getattr(combined, field_name).sort(key=lambda item: (*item.position, item.type))
            combined.validate_offsets(raw_text)
            outputs[record_id] = combined
        self.last_detailed_results = outputs
        return outputs

    def run_qwen8b_ner_editor_stage(
        self,
        raw_texts_by_id: dict[str, str],
        detailed_by_id: dict,
        qwen_llm,
        *,
        batch_size: int = 4,
        include_recovery: bool = True,
        review_only_auto_add_eligible: bool = False,
        cache_path: str | Path | None = None,
        model_id: str = "Qwen/Qwen3-8B",
    ) -> dict[str, list[NerEntity]]:
        """Build stable catalogs and run one locked-editor batch."""
        from .ner.candidates import build_candidate_catalog, build_missing_proposals
        from .ner.qwen_editor import (
            apply_editor_response, apply_missing_decisions, build_editor_request,
            build_missing_request, parse_missing_response,
            generate_with_cache, VersionedJsonlCache,
        )

        record_ids, catalogs, prompts = [], {}, []
        for record_id, detailed in detailed_by_id.items():
            raw_text = raw_texts_by_id[record_id]
            catalog = build_candidate_catalog(record_id, raw_text, detailed)
            catalogs[record_id] = catalog
            if catalog:
                record_ids.append(record_id)
                prompts.append(build_editor_request(record_id, raw_text, 0, catalog))
        cache = VersionedJsonlCache(cache_path) if cache_path is not None else None
        raw_outputs = generate_with_cache(
            qwen_llm, prompts, batch_size=batch_size, model_id=model_id,
            task="ner_editor", cache=cache,
        )
        outputs = {
            record_id: list(detailed.final_entities)
            for record_id, detailed in detailed_by_id.items()
        }
        audit = {}
        proposals_by_id = {}
        for record_id, raw_output in zip(record_ids, raw_outputs):
            result = apply_editor_response(
                raw_texts_by_id[record_id], catalogs[record_id], raw_output,
            )
            outputs[record_id] = result.entities
            proposals = build_missing_proposals(
                record_id, raw_texts_by_id[record_id], catalogs[record_id],
            )
            if review_only_auto_add_eligible:
                proposals = [item for item in proposals if item.auto_add_eligible]
            proposals_by_id[record_id] = proposals
            audit[record_id] = {
                "candidate_catalog": [item.__dict__ for item in catalogs[record_id]],
                "missing_proposals": [item.__dict__ for item in proposals],
                "raw_response": raw_output,
                "applied": result.applied,
                "rejected": result.rejected,
                "unresolved": result.unresolved,
            }
        if include_recovery:
            missing_ids = [record_id for record_id in record_ids if proposals_by_id.get(record_id)]
            missing_prompts = [
                build_missing_request(
                    f"{record_id}:missing", raw_texts_by_id[record_id], 0,
                    proposals_by_id[record_id],
                )
                for record_id in missing_ids
            ]
            missing_outputs = generate_with_cache(
                qwen_llm, missing_prompts, batch_size=batch_size, model_id=model_id,
                task="missing_proposal", cache=cache,
            )
            for record_id, raw_output in zip(missing_ids, missing_outputs):
                decisions, rejected = parse_missing_response(raw_output)
                missing_result = apply_missing_decisions(
                    raw_texts_by_id[record_id], outputs[record_id],
                    proposals_by_id[record_id], decisions,
                )
                outputs[record_id] = missing_result.entities
                audit[record_id]["missing_raw_response"] = raw_output
                audit[record_id]["missing_applied"] = missing_result.applied
                audit[record_id]["missing_rejected"] = rejected + missing_result.rejected
        self.last_editor_audit = audit
        return outputs

    # ------------------------------------------------------------
    # Batched retrievers followed by optional Qwen3 candidate selection.
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
        """Attach linker candidates, optionally selected by the loaded Qwen3 model.

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
                from .selection.candidate_selector import select_supported_top_candidates
                candidates_by_entity[i] = select_supported_top_candidates(
                    ent.text,
                    ent.type,
                    raw_candidates,
                    max_choices=MAX_OUTPUT_CANDIDATES[ent.type],
                )
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

    def run_linking_retrieval_stage_batch(
        self,
        entities_by_id: dict[str, list[NerEntity]],
        *,
        top_k: int | None = None,
    ) -> dict[str, dict[int, list]]:
        """Retrieve all mentions by ontology, using one encoder batch when supported."""
        from .rule.clinical import is_linkable_entity

        top_k = top_k or self.top_k_candidates
        results: dict[str, dict[int, list]] = {rid: {} for rid in entities_by_id}
        grouped: dict[str, list[tuple[str, int, NerEntity]]] = {"rxnorm": [], "icd10": []}
        for rid, entities in entities_by_id.items():
            for index, entity in enumerate(entities):
                linker_name = TYPE_TO_LINKER.get(entity.type)
                if linker_name and self._linkers.get(linker_name) is not None and is_linkable_entity(entity):
                    grouped[linker_name].append((rid, index, entity))

        for linker_name, items in grouped.items():
            if not items:
                continue
            linker = self._linkers[linker_name]
            mentions = [item[2].text for item in items]
            try:
                if hasattr(linker, "link_many"):
                    if linker_name == "icd10":
                        batches = linker.link_many(mentions, top_k_codes=top_k)
                    else:
                        batches = linker.link_many(mentions, top_k=top_k)
                else:
                    batches = []
                    for mention in mentions:
                        value = linker.link(mention, **({"top_k_codes": top_k} if linker_name == "icd10" else {"top_k": top_k}))
                        batches.append(value.get("candidates", []) if isinstance(value, dict) else value)
                if len(batches) != len(items):
                    raise ValueError("link_many returned the wrong number of rows")
            except Exception as exc:
                print(f"[pipeline] batch {linker_name} linking failed: {exc}", file=sys.stderr)
                batches = [[] for _ in items]
            for (rid, index, _entity), candidates in zip(items, batches):
                results[rid][index] = list(candidates or [])
        self.last_linking_retrieval = results
        return results

    def run_qwen8b_linking_selector_stage(
        self,
        entities_by_id: dict[str, list[NerEntity]],
        retrieval_by_id: dict[str, dict[int, list]],
        *,
        selector_llm=None,
        raw_texts_by_id: dict[str, str] | None = None,
    ) -> dict[str, dict[int, list[str]]]:
        from .selection.candidate_selector import select_candidates_many, select_supported_top_candidates

        results: dict[str, dict[int, list[str]]] = {rid: {} for rid in entities_by_id}
        destinations, selector_items = [], []
        raw_texts_by_id = raw_texts_by_id or {}
        for rid, entities in entities_by_id.items():
            for index, entity in enumerate(entities):
                candidates = retrieval_by_id.get(rid, {}).get(index, [])
                if not candidates:
                    continue
                if selector_llm is None:
                    results[rid][index] = select_supported_top_candidates(
                        entity.text, entity.type, candidates,
                        max_choices=MAX_OUTPUT_CANDIDATES[entity.type],
                    )
                    continue
                start, end = entity.position
                raw_text = raw_texts_by_id.get(rid, "")
                context = raw_text[max(0, start - 120):min(len(raw_text), end + 120)] if raw_text else ""
                destinations.append((rid, index))
                selector_items.append({
                    "entity_text": entity.text, "entity_type": entity.type,
                    "candidates": candidates, "context": context,
                })
        if selector_llm is None:
            return results
        selected_batches = select_candidates_many(
            selector_items,
            selector_llm,
            top_k_context=self.top_k_candidates,
        )
        for (rid, index), selected in zip(destinations, selected_batches):
            results[rid][index] = selected
        return results

    def run_linking_stage(
        self,
        entities_by_id: dict[str, list[NerEntity]],
        *,
        selector_llm=None,
        raw_texts_by_id: dict[str, str] | None = None,
    ) -> dict[str, dict[int, list[str]]]:
        retrieval = self.run_linking_retrieval_stage_batch(entities_by_id)
        return self.run_qwen8b_linking_selector_stage(
            entities_by_id, retrieval, selector_llm=selector_llm,
            raw_texts_by_id=raw_texts_by_id,
        )

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
    # The optional selector must already be loaded by the caller.
    # ------------------------------------------------------------
    def process_record(
        self,
        raw_text: str,
        *,
        selector_llm=None,
        **predict_kwargs,
    ) -> list[dict]:
        entities_by_id = self.run_ner_stage({"_single": raw_text}, **predict_kwargs)

        candidates_by_id = self.run_linking_stage(
            entities_by_id,
            selector_llm=selector_llm,
            raw_texts_by_id={"_single": raw_text},
        )
        return self.build_outputs(entities_by_id, candidates_by_id)["_single"]

    def process_file(self, path: str | Path, **kwargs) -> list[dict]:
        raw_text = inference_io.read_text_file(path)
        return self.process_record(raw_text, **kwargs)
