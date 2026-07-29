import unittest

from src.inference import io as inference_io
from src.inference.pipeline import InferencePipeline
from src.inference.ner.llm_fixer import _locate_span, audit_missing_entities
from src.inference.ner.repair_gate import filter_entities
from src.inference.schemas import NerEntity
from src.inference.selection.candidate_selector import select_candidates, select_candidates_many


class _StaticLlm:
    def __init__(self, response):
        self.response = response

    def generate(self, _system_prompt, _user_prompt):
        return self.response


class _FailLlm:
    def generate(self, _system_prompt, _user_prompt):
        raise AssertionError("LLM không được gọi cho exact candidate")


class _BatchLlm:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def generate_batch(self, prompts, batch_size=4):
        self.calls += 1
        self.prompt_count = len(prompts)
        self.batch_size = batch_size
        return self.responses


class InferenceRegressionTests(unittest.TestCase):
    def test_low_confidence_entity_is_flagged_for_1_7b(self):
        kept, dropped = filter_entities([{
            "text": "lazer (tbm)",
            "type": "TÊN_XÉT_NGHIỆM",
            "assertions": [],
            "position": [10, 21],
            "score": 0.74,
        }])
        self.assertEqual([], dropped)
        self.assertEqual("low_emission_confidence", kept[0]["flag"])

    def test_recall_audit_adds_only_exact_uncovered_entity(self):
        raw = "Đang dùng aspirin 81 mg daily và không sốt."
        fever_start = raw.index("sốt")
        existing = [NerEntity(
            text="sốt",
            type="TRIỆU_CHỨNG",
            assertions=["isNegated"],
            position=(fever_start, fever_start + 3),
        )]
        llm = _StaticLlm(
            '{"additions":['
            '{"text":"aspirin 81 mg daily","type":"THUỐC","assertions":[]},'
            '{"text":"không tồn tại","type":"CHẨN_ĐOÁN","assertions":[]}'
            ']}'
        )

        audited = audit_missing_entities(raw, existing, llm)

        self.assertEqual(["aspirin 81 mg daily", "sốt"], [entity.text for entity in audited])
        self.assertEqual(raw.index("aspirin"), audited[0].position[0])

    def test_exact_icd_candidate_bypasses_llm(self):
        candidates = [{
            "code": "A82.9",
            "matched_term": "Bệnh dại",
            "score": 0.99,
            "language": "vi",
            "term_type": "preferred",
        }]

        selected = select_candidates("Bệnh dại", "CHẨN_ĐOÁN", candidates, _FailLlm())

        self.assertEqual(["A82.9"], selected)

    def test_validator_rejects_multiple_rxnorm_codes(self):
        with self.assertRaisesRegex(ValueError, "tối đa 1 candidate"):
            inference_io.validate_record_output([{
                "text": "aspirin",
                "type": "THUỐC",
                "candidates": ["1", "2"],
                "assertions": [],
                "position": [0, 7],
            }])

    def test_rule_only_pipeline_limits_output_candidates(self):
        class _RxLinker:
            @staticmethod
            def link(_text, top_k):
                return {"candidates": [
                    {"rxcui": str(index)} for index in range(top_k)
                ]}

        pipeline = InferencePipeline(object(), rxnorm_linker=_RxLinker(), top_k_candidates=10)
        entity = NerEntity(text="aspirin", type="THUỐC")

        attached = pipeline.attach_candidates([entity])

        self.assertEqual(["0"], attached[0])

    def test_raw_offset_remap_preserves_repair_flag(self):
        entity = NerEntity(
            text="thiếu",
            type="CHẨN_ĐOÁN",
            position=(0, 5),
            score=0.2,
            flag="suspect_truncated_diagnosis",
        )

        remapped = inference_io.remap_entities_to_raw("thiếu máu", "thiếu máu", [entity])

        self.assertEqual("suspect_truncated_diagnosis", remapped[0].flag)
        self.assertEqual(0.2, remapped[0].score)

    def test_drug_selector_never_returns_multiple_rxnorm_codes(self):
        candidates = [
            {"rxcui": "1", "matched_term": "Drug A"},
            {"rxcui": "2", "matched_term": "Drug B"},
            {"rxcui": "3", "matched_term": "Drug C"},
        ]
        llm = _StaticLlm('{"chosen_codes":["2","1","3"],"reason":"x"}')

        selected = select_candidates("Drug", "THUỐC", candidates, llm)

        self.assertEqual(["2"], selected)

    def test_candidate_selector_batches_ambiguous_entities(self):
        items = [
            {
                "entity_text": "Drug A",
                "entity_type": "THUỐC",
                "candidates": [{"rxcui": "1"}, {"rxcui": "2"}],
            },
            {
                "entity_text": "viêm phổi",
                "entity_type": "CHẨN_ĐOÁN",
                "candidates": [
                    {"code": "J18.9", "matched_term": "pneumonia"},
                    {"code": "J15.9", "matched_term": "bacterial pneumonia"},
                ],
            },
        ]
        llm = _BatchLlm([
            '{"chosen_codes":["2"],"reason":"x"}',
            '{"chosen_codes":["J18.9"],"reason":"x"}',
        ])

        selected = select_candidates_many(items, llm)

        self.assertEqual([["2"], ["J18.9"]], selected)
        self.assertEqual(1, llm.calls)
        self.assertEqual(2, llm.prompt_count)

    def test_pipeline_uses_one_selector_batch(self):
        class _RxLinker:
            @staticmethod
            def link(text, top_k):
                return {"candidates": [
                    {"rxcui": f"{text}-1"},
                    {"rxcui": f"{text}-2"},
                ]}

        pipeline = InferencePipeline(object(), rxnorm_linker=_RxLinker())
        entities = [
            NerEntity(text="aspirin", type="THUỐC", position=(0, 7)),
            NerEntity(text="metformin", type="THUỐC", position=(12, 21)),
        ]
        llm = _BatchLlm([
            '{"chosen_codes":["aspirin-2"],"reason":"x"}',
            '{"chosen_codes":["metformin-1"],"reason":"x"}',
        ])

        attached = pipeline.attach_candidates(
            entities,
            selector_llm=llm,
            raw_text="aspirin va metformin",
        )

        self.assertEqual({0: ["aspirin-2"], 1: ["metformin-1"]}, attached)
        self.assertEqual(1, llm.calls)

    def test_linking_stage_batches_across_documents(self):
        class _RxLinker:
            @staticmethod
            def link(text, top_k):
                return {"candidates": [
                    {"rxcui": f"{text}-1"},
                    {"rxcui": f"{text}-2"},
                ]}

        pipeline = InferencePipeline(object(), rxnorm_linker=_RxLinker())
        entities_by_id = {
            "1": [NerEntity(text="aspirin", type="THUỐC", position=(0, 7))],
            "2": [NerEntity(text="metformin", type="THUỐC", position=(0, 9))],
        }
        llm = _BatchLlm([
            '{"chosen_codes":["aspirin-2"],"reason":"x"}',
            '{"chosen_codes":["metformin-1"],"reason":"x"}',
        ])

        attached = pipeline.run_linking_stage(
            entities_by_id,
            selector_llm=llm,
            raw_texts_by_id={"1": "aspirin", "2": "metformin"},
        )

        self.assertEqual({"1": {0: ["aspirin-2"]}, "2": {0: ["metformin-1"]}}, attached)
        self.assertEqual(1, llm.calls)
        self.assertEqual(2, llm.prompt_count)

    def test_retrim_uses_nearest_repeated_occurrence(self):
        raw = "thiếu máu đã ổn. Hiện không có thiếu máu."
        old_start = raw.rfind("thiếu")

        span = _locate_span(raw, (old_start, old_start + 5), "thiếu máu", radius=60)

        self.assertEqual((raw.rfind("thiếu máu"), len(raw) - 1), span)


if __name__ == "__main__":
    unittest.main()
