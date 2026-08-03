"""Run the 30-record V3 integration smoke with a schema-valid deterministic LLM stub.

This exercises NER, region editor parsing/isolation, RxNorm/ICD retrieval, selector
V2 parsing/whitelisting, output validation and audit writing. It deliberately does
not claim Qwen model quality; use the same CLI flags on A40 for that measurement.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import config as inference_config
from src.inference import io as inference_io
from src.inference.pipeline import InferencePipeline
from src.linking.icd10.icd10_linker import Icd10Linker
from src.linking.rxnorm.linker import RxNormLinker


class SchemaValidLlmStub:
    def __init__(self):
        self.calls = 0
        self.requests = 0
        self.request_types = {"editor": 0, "recovery": 0, "selector": 0}

    def generate_batch(self, prompts, batch_size=4, **_kwargs):
        self.calls += 1
        self.requests += len(prompts)
        responses = []
        for _system, user in prompts:
            payload = json.loads(user)
            request_id = payload["request_id"]
            if "candidates" in payload and "entity" not in payload:
                self.request_types["editor"] += 1
                response = {"request_id": request_id, "changes": [], "unresolved_ids": []}
            elif "proposals" in payload:
                self.request_types["recovery"] += 1
                response = {"request_id": request_id, "decisions": [{
                    "proposal_id": proposal["proposal_id"], "decision": "REJECT",
                    "type": None, "assertions": [], "confidence": "HIGH",
                    "reason_code": "NOT_AN_ENTITY",
                } for proposal in payload["proposals"]]}
            else:
                self.request_types["selector"] += 1
                response = {
                    "request_id": request_id, "decision": "ABSTAIN", "chosen_codes": [],
                    "confidence": "LOW", "reason_code": "AMBIGUOUS",
                }
            responses.append(json.dumps(response, ensure_ascii=False))
        return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.txt"), key=lambda path: int(path.stem))
    raw = {path.stem: inference_io.read_text_file(path) for path in paths}
    if len(raw) != 30:
        raise ValueError(f"expected 30 input files, found {len(raw)}")
    timings = {}
    started = time.perf_counter()
    pipeline = InferencePipeline.load(with_rxnorm=False, with_icd10=False, device=args.device)
    details = pipeline.run_ner_stage_detailed(raw)
    timings["ner_seconds"] = time.perf_counter() - started

    llm = SchemaValidLlmStub()
    stage = time.perf_counter()
    entities = pipeline.run_qwen8b_ner_editor_stage(
        raw, details, llm, batch_size=8, cache_path=args.output_dir / "editor_cache.jsonl",
        model_id="schema-valid-smoke-stub",
    )
    timings["editor_seconds"] = time.perf_counter() - stage
    pipeline.ner_engine = None
    gc.collect()

    stage = time.perf_counter()
    pipeline._linkers["rxnorm"] = RxNormLinker(
        inference_config.RXNORM_INDEX_DIR, inference_config.RXNORM_CLEAN_PATH, device=args.device,
    )
    pipeline._linkers["icd10"] = Icd10Linker(
        inference_config.ICD10_INDEX_DIR, device=args.device,
    )
    selected = pipeline.run_linking_stage(
        entities, selector_llm=llm, raw_texts_by_id=raw,
        selector_batch_size=8, selector_cache_path=args.output_dir / "selector_cache.jsonl",
        model_id="schema-valid-smoke-stub",
        retrieval_k_by_linker={"rxnorm": 50, "icd10": 50},
    )
    timings["linking_seconds"] = time.perf_counter() - stage
    outputs = pipeline.build_outputs(entities, selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record_id, output in outputs.items():
        inference_io.validate_record_output(output, raw_text=raw[record_id])
        inference_io.write_output_json(output, args.output_dir / f"{record_id}.json", raw_text=raw[record_id])
        (args.output_dir / f"{record_id}.ner_audit.json").write_text(
            json.dumps(pipeline.last_editor_audit[record_id], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    summary = {
        "input_count": len(raw), "output_count": len(outputs),
        "editor_parse_failures": sum(audit["summary"]["parse_failure"] for audit in pipeline.last_editor_audit.values()),
        "llm_batches": llm.calls, "llm_requests": llm.requests,
        "llm_request_types": llm.request_types,
        "generation_microbatches_at_batch_size_8": sum(
            math.ceil(count / 8) for count in llm.request_types.values()
        ),
        "total_entities": sum(len(rows) for rows in outputs.values()),
        "linked_codes": sum(len(row["candidates"]) for rows in outputs.values() for row in rows),
        "timings": timings, "total_seconds": time.perf_counter() - started,
        "llm_note": "schema-valid deterministic stub; not a Qwen quality measurement",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
