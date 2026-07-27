"""Điều phối end-to-end: raw text -> NER -> (optional) LLM fix -> linking
-> (optional) LLM select candidate -> BTC JSON.

THIẾT KẾ TÁCH STAGE, không phải 1 hàm process_record() làm hết — vì 2
model LLM (fixer 1.7B, selector 7B) PHẢI load 1 LẦN CHO CẢ BATCH rồi mới
loop qua record, không được load/unload theo từng record (batch 100 record
x 600s timeout/lần nộp -> load lại model mỗi record là hết giờ chắc chắn).

Luồng dùng đúng cho batch (xem cli.py):
    entities_by_id = pipeline.run_ner_stage(raw_texts_by_id)
    if dùng fixer:
        fixer_llm.load(); entities_by_id = pipeline.run_fixer_stage(...); fixer_llm.unload()
    if dùng selector:
        selector_llm.load()
    candidates_by_id = pipeline.run_linking_stage(entities_by_id, selector_llm=selector_llm)
    if dùng selector:
        selector_llm.unload()
    outputs = pipeline.build_outputs(entities_by_id, candidates_by_id)

`process_record()` / `process_file()` vẫn giữ lại cho test nhanh 1 file
(1 record thì load/unload 1 lần cũng không sao) — nhận thẳng instance
LocalLLM ĐÃ LOAD (fixer_llm=..., selector_llm=...), không tự load bên
trong pipeline nữa.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import config as cfg
from . import io as inference_io
from .ner.engine import NerEngine
from .ner.postprocessor import clean_text_for_inference
from .schemas import NerEntity

# Map type NER -> tên linker sẽ dùng trong self._linkers. Chỉ 2 type có
# linker đã build; các type khác luôn có candidates=[] (đúng ví dụ BTC,
# TRIỆU_CHỨNG/TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM không có candidates).
TYPE_TO_LINKER = {
    "THUỐC": "rxnorm",
    "CHẨN_ĐOÁN": "icd10",
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
    def run_ner_stage(self, raw_texts_by_id: dict[str, str], **predict_kwargs) -> dict[str, list[NerEntity]]:
        entities_by_id: dict[str, list[NerEntity]] = {}
        for rid, raw_text in raw_texts_by_id.items():
            cleaned = clean_text_for_inference(raw_text)
            entities = self.ner_engine.predict_text(cleaned, **predict_kwargs)
            entities_by_id[rid] = inference_io.remap_entities_to_raw(raw_text, cleaned, entities)
        return entities_by_id

    # ------------------------------------------------------------
    # Stage 2 (optional): LLM sửa entity bị repair_gate flag — nhận
    # fixer_llm ĐÃ LOAD, gọi cho toàn bộ batch trong 1 lượt model ở VRAM.
    # ------------------------------------------------------------
    def run_fixer_stage(
        self,
        raw_texts_by_id: dict[str, str],
        entities_by_id: dict[str, list[NerEntity]],
        fixer_llm,
    ) -> dict[str, list[NerEntity]]:
        from .ner.llm_fixer import fix_flagged_entities

        fixed_by_id: dict[str, list[NerEntity]] = {}
        for rid, entities in entities_by_id.items():
            fixed_by_id[rid] = fix_flagged_entities(raw_texts_by_id[rid], entities, fixer_llm)
        return fixed_by_id

    # ------------------------------------------------------------
    # Stage 3 (+ stage 4 lồng vào): linking, có thể kèm LLM re-rank nếu
    # selector_llm được truyền vào (đã load sẵn).
    # ------------------------------------------------------------
    def _get_raw_candidates(self, ent: NerEntity):
        """Trả candidate RAW (list[RxNormCandidate] hoặc list[dict]) CHƯA
        cắt xuống code — dùng chung cho cả nhánh có/không LLM selector."""
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

    def attach_candidates(self, entities: list[NerEntity], *, selector_llm=None) -> dict[int, list[str]]:
        """selector_llm=None -> chỉ cắt top-k theo score linker (không LLM).
        selector_llm=instance đã load -> gọi candidate_selector re-rank."""
        candidates_by_entity: dict[int, list[str]] = {}

        for i, ent in enumerate(entities):
            raw_candidates = self._get_raw_candidates(ent)
            if not raw_candidates:
                continue

            if selector_llm is not None:
                from .selection.candidate_selector import select_candidates
                candidates_by_entity[i] = select_candidates(
                    ent.text, ent.type, raw_candidates, selector_llm,
                    top_k_context=self.top_k_candidates,
                )
            else:
                candidates_by_entity[i] = _extract_codes(raw_candidates, self.top_k_candidates)

        return candidates_by_entity

    def run_linking_stage(
        self, entities_by_id: dict[str, list[NerEntity]], *, selector_llm=None,
    ) -> dict[str, dict[int, list[str]]]:
        return {
            rid: self.attach_candidates(entities, selector_llm=selector_llm)
            for rid, entities in entities_by_id.items()
        }

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
    def process_record(self, raw_text: str, *, fixer_llm=None, selector_llm=None, **predict_kwargs) -> list[dict]:
        entities_by_id = self.run_ner_stage({"_single": raw_text}, **predict_kwargs)

        if fixer_llm is not None:
            entities_by_id = self.run_fixer_stage({"_single": raw_text}, entities_by_id, fixer_llm)

        candidates_by_id = self.run_linking_stage(entities_by_id, selector_llm=selector_llm)
        return self.build_outputs(entities_by_id, candidates_by_id)["_single"]

    def process_file(self, path: str | Path, **kwargs) -> list[dict]:
        raw_text = inference_io.read_text_file(path)
        return self.process_record(raw_text, **kwargs)