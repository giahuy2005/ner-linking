import unittest

from src.inference import io as inference_io
from src.inference.pipeline import InferencePipeline
from src.inference.ner.llm_fixer import _locate_span, audit_missing_entities
from src.inference.ner import offset_mapper, postprocessor
from src.inference.ner.sectioner import split_sections_by_header
from src.inference.ner.repair_gate import filter_entities
from src.inference.schemas import NerEntity
from src.inference.selection.candidate_selector import select_candidates, select_candidates_many
from src.llm.config import NER_FIXER_CONFIG
from src.linking.rxnorm.reranker import RxNormRuleReranker
from src.linking.rxnorm.schemas import ParsedDrugMention, RxNormCandidate


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
    def test_notebook_offset_validation_checks_every_internal_word(self):
        self.assertFalse(offset_mapper.is_valid_char_span(
            [(0, 3), (None, None), (7, 10)], 0, 3,
        ))
        self.assertFalse(offset_mapper.is_valid_char_span(
            [(0, 5), (4, 8)], 0, 2,
        ))
        self.assertTrue(offset_mapper.is_valid_char_span(
            [(0, 3), (4, 8)], 0, 2,
        ))

    def test_notebook_assertion_threshold_can_be_per_label(self):
        thresholds = {"isHistorical": 0.55, "isNegated": 0.65}
        self.assertEqual(0.55, postprocessor.get_assertion_threshold(
            thresholds, "isHistorical",
        ))
        self.assertEqual(0.5, postprocessor.get_assertion_threshold(
            thresholds, "isFamily",
        ))

    def test_notebook_qa_is_one_offset_preserving_block(self):
        raw = "Hỏi: Tôi bị sốt.\r\nTrả lời: Bạn nên đi khám."
        blocks = split_sections_by_header(raw)
        self.assertEqual(1, len(blocks))
        self.assertEqual(raw, blocks[0]["body"])
        self.assertEqual((0, len(raw)), (blocks[0]["start"], blocks[0]["end"]))

    def test_notebook_repeated_emr_sections_remain_separate_and_keep_heading(self):
        raw = "1. Tiền sử bệnh\nsốt\n1. Tiền sử bệnh\nđau"
        blocks = split_sections_by_header(raw)
        self.assertEqual(2, len(blocks))
        self.assertTrue(all(block["body"].lstrip().startswith("1. Tiền sử bệnh")
                            for block in blocks.values()))
        self.assertEqual(raw, "".join(block["body"] for block in blocks.values()))

    def test_pipeline_predicts_emr_blocks_and_maps_offsets_to_raw(self):
        raw = "1. Tiền sử bệnh\nsốt\n2. Tiền sử bệnh hiện tại\nđau"

        class _BlockEngine:
            def __init__(self):
                self.calls = []

            def predict_text(self, text, **_kwargs):
                self.calls.append(text)
                for surface in ("sốt", "đau"):
                    if surface in text:
                        start = text.index(surface)
                        return [NerEntity(surface, "TRIỆU_CHỨNG", [],
                                          (start, start + len(surface)), 0.9)]
                return []

        engine = _BlockEngine()
        pipeline = InferencePipeline(engine)
        result = pipeline.run_ner_stage({"doc": raw}, two_pass=False)["doc"]

        self.assertEqual(2, len(engine.calls))
        self.assertEqual(["sốt", "đau"], [entity.text for entity in result])
        self.assertTrue(all(raw[start:end] == entity.text for entity in result
                            for start, end in [entity.position]))

    def test_small_fixer_uses_notebook_qwen25_15b_model(self):
        self.assertEqual("Qwen/Qwen2.5-1.5B-Instruct", NER_FIXER_CONFIG.model_id)
        self.assertFalse(NER_FIXER_CONFIG.supports_thinking)

    def test_small_fixer_stage_is_distinct_from_7b_reviewer(self):
        raw = "Bệnh nhân sốt."
        start = raw.index("sốt")
        entity = NerEntity(
            "sốt", "TRIỆU_CHỨNG", [], (start, start + 3),
            score=0.4, flag="low_emission_confidence",
        )
        llm = _BatchLlm([
            '{"action":"keep","text":"sốt","type":"TRIỆU_CHỨNG"}',
        ])
        pipeline = InferencePipeline(object())

        result = pipeline.run_fixer_stage(
            {"doc": raw}, {"doc": [entity]}, llm,
            audit_missing=False, batch_size=4,
        )

        self.assertEqual(["sốt"], [item.text for item in result["doc"]])
        self.assertIsNone(result["doc"][0].flag)
        self.assertEqual(1, llm.calls)

    def test_small_fixer_blocks_unsafe_drop_for_7b_handoff(self):
        raw = "Bệnh nhân đau đầu."
        start = raw.index("đau đầu")
        entity = NerEntity(
            "đau đầu", "TRIỆU_CHỨNG", [], (start, start + len("đau đầu")),
            score=0.60, flag="low_emission_confidence",
        )
        llm = _BatchLlm([
            '{"action":"drop","text":"đau đầu","type":"TRIỆU_CHỨNG"}',
        ])
        pipeline = InferencePipeline(object())

        result = pipeline.run_fixer_stage(
            {"doc": raw}, {"doc": [entity]}, llm,
            audit_missing=False,
        )

        self.assertEqual(["đau đầu"], [item.text for item in result["doc"]])
        self.assertEqual("low_emission_confidence", result["doc"][0].flag)
        self.assertIn("doc", pipeline.last_handoffs)
        target = pipeline.last_handoffs["doc"]["review_regions"][0]["targets"][0]
        self.assertEqual("blocked_unsafe_drop",
                         target["small_llm_review_hints"][0]["status"])

    def test_small_fixer_retype_is_only_a_hint_for_constrained_7b_target(self):
        raw = "Bệnh nhân có hội chứng lạ."
        start = raw.index("hội chứng lạ")
        entity = NerEntity(
            "hội chứng lạ", "CHẨN_ĐOÁN", [], (start, start + len("hội chứng lạ")),
            score=0.50, flag="low_emission_confidence",
        )
        llm = _BatchLlm([
            '{"action":"retype","text":"hội chứng lạ","type":"TRIỆU_CHỨNG"}',
        ])
        pipeline = InferencePipeline(object())

        result = pipeline.run_fixer_stage(
            {"doc": raw}, {"doc": [entity]}, llm,
            audit_missing=False,
        )

        # 1.5B cannot mutate the type; 7B receives the suggestion as evidence.
        self.assertEqual("CHẨN_ĐOÁN", result["doc"][0].type)
        target = pipeline.last_handoffs["doc"]["review_regions"][0]["targets"][0]
        hint = target["small_llm_review_hints"][0]
        self.assertEqual("RETYPE_SUGGEST", hint["requested_action"])
        self.assertEqual("TRIỆU_CHỨNG", hint["suggested_type"])

    def test_low_confidence_entity_is_flagged_for_1_5b(self):
        kept, dropped = filter_entities([{
            "text": "lazer (tbm)",
            "type": "TÊN_XÉT_NGHIỆM",
            "assertions": [],
            "position": [10, 21],
            "score": 0.74,
        }])
        self.assertEqual([], dropped)
        self.assertEqual("low_emission_confidence", kept[0]["flag"])

    def test_short_span_is_reviewed_but_never_rule_dropped(self):
        kept, dropped = filter_entities([{
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [10, 12],
            "score": 0.99,
        }])

        self.assertEqual([], dropped)
        self.assertEqual("short_span_review", kept[0]["flag"])

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

    def test_validator_rejects_more_than_two_icd_codes(self):
        with self.assertRaisesRegex(ValueError, "tối đa 2 candidate"):
            inference_io.validate_record_output([{
                "text": "viêm phổi",
                "type": "CHẨN_ĐOÁN",
                "candidates": ["J18.9", "J15.9", "J12.9"],
                "assertions": [],
                "position": [0, 10],
            }])

    def test_icd_selector_defaults_to_one_and_needs_explicit_coordination_for_two(self):
        ambiguous = [
            {"code": "A", "matched_term": "bệnh x", "score": 0.80},
            {"code": "B", "matched_term": "bệnh y", "score": 0.77},
            {"code": "C", "matched_term": "bệnh z", "score": 0.76},
        ]
        selected = select_candidates(
            "bệnh x", "CHẨN_ĐOÁN", ambiguous,
            _StaticLlm('{"chosen_codes":["A","B","C"],"reason":"x"}'),
            max_choices=3,
        )
        self.assertEqual(["A"], selected)

        coordinated = select_candidates(
            "bệnh x và bệnh y", "CHẨN_ĐOÁN", ambiguous,
            _StaticLlm('{"chosen_codes":["A","B","C"],"reason":"x"}'),
            max_choices=3,
        )
        self.assertEqual(["A", "B"], coordinated)

        separated = [
            {"code": "A", "matched_term": "bệnh x", "score": 0.90},
            {"code": "B", "matched_term": "bệnh y", "score": 0.75},
        ]
        selected = select_candidates(
            "bệnh x", "CHẨN_ĐOÁN", separated,
            _StaticLlm('{"chosen_codes":["A","B"],"reason":"x"}'),
        )
        self.assertEqual(["A"], selected)

    def test_ambiguous_selector_can_abstain_and_invalid_json_does_not_force_code(self):
        candidates = [
            {"code": "C51", "matched_term": "unrelated concept", "score": 0.70},
            {"code": "R05", "matched_term": "another concept", "score": 0.68},
        ]

        abstained = select_candidates(
            "khái niệm không tương ứng", "CHẨN_ĐOÁN", candidates,
            _StaticLlm('{"chosen_codes":[],"reason":"không phù hợp"}'),
        )
        malformed = select_candidates(
            "khái niệm không tương ứng", "CHẨN_ĐOÁN", candidates,
            _StaticLlm('not-json'),
        )

        self.assertEqual([], abstained)
        self.assertEqual([], malformed)

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

    def test_candidate_selector_abstains_from_semantically_unsupported_icd(self):
        candidates = [{
            "code": "C51",
            "matched_term": "U ác âm hộ",
            "score": 0.79,
            "language": "vi",
        }]
        llm = _StaticLlm('{"chosen_codes":["C51"],"reason":"top"}')

        selected = select_candidates("HA", "CHẨN_ĐOÁN", candidates, llm)

        self.assertEqual([], selected)

    def test_candidate_selector_keeps_supported_nonexact_icd(self):
        candidates = [{
            "code": "J18.9",
            "matched_term": "Viêm phổi, không xác định",
            "score": 0.74,
            "language": "vi",
        }]
        llm = _StaticLlm('{"chosen_codes":["J18.9"],"reason":"đúng nghĩa"}')

        selected = select_candidates("viêm phổi", "CHẨN_ĐOÁN", candidates, llm)

        self.assertEqual(["J18.9"], selected)

    def test_pipeline_without_selector_still_abstains_from_bad_icd(self):
        class _IcdLinker:
            @staticmethod
            def link(text, top_k_codes):
                return [{
                    "code": "C51",
                    "matched_term": "U ác âm hộ",
                    "score": 0.79,
                    "language": "vi",
                }]

        pipeline = InferencePipeline(object(), icd10_linker=_IcdLinker())
        entity = NerEntity(text="HA", type="CHẨN_ĐOÁN", position=(0, 2))

        attached = pipeline.attach_candidates([entity])

        self.assertEqual([], attached[0])

    def test_rxnorm_short_substring_is_not_exact_ingredient(self):
        parsed = ParsedDrugMention(
            raw_text="cấp tính",
            normalized_text="cap tinh",
            ingredient_core="cap tinh",
        )
        candidate = RxNormCandidate(
            rxcui="10603",
            tty="IN",
            tier="support",
            name="tin",
            structured={"ingredients": ["tin"]},
        )

        relation = RxNormRuleReranker().ingredient_gate(parsed, candidate)

        self.assertEqual("mismatch", relation)

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
