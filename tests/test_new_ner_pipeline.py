import json
import unittest

from src.inference.ner.reviewer_7b import review_entities_batch
from src.inference.ner.two_pass import SuspiciousRegion
from src.inference.pipeline import InferencePipeline
from src.inference.rule.clinical import apply_clinical_rules, deterministic_cleanup
from src.inference.rule.routing import build_handoff_requests
from src.inference.schemas import NerEntity


GOLD_TEXT = (
    "Danh sách thuốc trước nhập viện chính xác và đầy đủ. "
    "1. amlodipine 10 mg po daily 2. aspirin 81 mg po daily "
    "3. metoprolol succinate xl 50 mg po daily 4. guaifenesin ml po q6h:prn điều trị ho "
    "5. nystatin oral suspension 5 ml po qid:prn điều trị đau nhức "
    "6. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau "
    "7. pravastatin 40 mg po daily 8. docusate sodium 100 mg po bid điều trị táo bón "
    "9. senna 8.6 mg po bid:prn điều trị táo bón "
    "10. clonazepam 0.5 mg po qam:prn điều trị lo âu "
    "11. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ"
)


class _BatchLlm:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_batch(self, prompts, batch_size=4):
        self.calls += 1
        return self.responses.pop(0)


class NewNerPipelineTests(unittest.TestCase):
    def test_gold_btc_medication_list(self):
        entities, logs = apply_clinical_rules(GOLD_TEXT, [])
        drugs = [entity for entity in entities if entity.type == "THUỐC"]
        symptoms = [entity for entity in entities if entity.type == "TRIỆU_CHỨNG"]

        self.assertEqual(11, len(drugs))
        self.assertTrue(all(entity.assertions == ["isHistorical"] for entity in drugs))
        self.assertEqual(
            ["ho", "đau nhức", "sốt đau", "táo bón", "táo bón", "lo âu", "lo âu", "mất ngủ"],
            [entity.text for entity in symptoms],
        )
        self.assertTrue(all(not entity.assertions for entity in symptoms))
        self.assertTrue(all(GOLD_TEXT[s:e] == entity.text for entity in entities
                            for s, e in [entity.position]))
        self.assertTrue(any(log["reason"] == "pre_admission_medication_list" for log in logs))

    def test_gold_rxnorm_ids_are_attached_only_after_final_ner(self):
        entities, _ = apply_clinical_rules(GOLD_TEXT, [])
        expected = {
            "amlodipine 10 mg po daily": "308135",
            "aspirin 81 mg po daily": "243670",
            "metoprolol succinate xl 50 mg po daily": "866436",
            "guaifenesin ml po q6h:prn": "392085",
            "nystatin oral suspension 5 ml po qid:prn": "7597",
            "acetaminophen 325-650 mg po q6h:prn": "313782",
            "pravastatin 40 mg po daily": "904475",
            "docusate sodium 100 mg po bid": "1099279",
            "senna 8.6 mg po bid:prn": "312935",
            "clonazepam 0.5 mg po qam:prn": "197527",
            "clonazepam 1.5 mg po qhs": "197528",
        }

        class _GoldRxNormLinker:
            @staticmethod
            def link(text, top_k):
                return {"candidates": [{"rxcui": expected[text]}]}

        pipeline = InferencePipeline(object(), rxnorm_linker=_GoldRxNormLinker())
        attached = pipeline.attach_candidates(entities)
        output = pipeline.build_outputs({"gold": entities}, {"gold": attached})["gold"]

        drugs = [item for item in output if item["type"] == "THUỐC"]
        self.assertEqual([[expected[item["text"]]] for item in drugs],
                         [item["candidates"] for item in drugs])
        self.assertTrue(all(not item.get("candidates", []) for item in output
                            if item["type"] != "THUỐC"))

    def test_boundary_repeated_token_and_false_positive_rules(self):
        raw = "sốt bn; bn vàng da; Thiếu men G6PD (; chụp chụp ct sọ não; ăn ngủ"
        surfaces = ["sốt bn", "bn vàng da", "Thiếu men G6PD (",
                    "chụp chụp ct sọ não", "ăn ngủ"]
        types = ["TRIỆU_CHỨNG", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN",
                 "TÊN_XÉT_NGHIỆM", "TRIỆU_CHỨNG"]
        entities = []
        for surface, entity_type in zip(surfaces, types):
            start = raw.index(surface)
            entities.append(NerEntity(surface, entity_type, [],
                                      (start, start + len(surface)), 0.7))

        cleaned, _ = deterministic_cleanup(raw, entities)

        self.assertEqual(["sốt", "vàng da", "Thiếu men G6PD", "chụp ct sọ não"],
                         [entity.text for entity in cleaned])

    def test_additional_notebook_boundary_examples(self):
        raw = "bệnh Kawasaki\nMặc; Cấy máu, dịch hầu họng; Chụp lại chụp ct sọ não"
        surfaces = ["bệnh Kawasaki\nMặc", "hầu họng", "Chụp lại chụp ct sọ não"]
        types = ["CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "TÊN_XÉT_NGHIỆM"]
        entities = []
        for surface, entity_type in zip(surfaces, types):
            start = raw.index(surface)
            entities.append(NerEntity(surface, entity_type, [],
                                      (start, start + len(surface)), 0.7))

        cleaned, _ = deterministic_cleanup(raw, entities)

        self.assertEqual(["bệnh Kawasaki", "dịch hầu họng", "chụp ct sọ não"],
                         [entity.text for entity in cleaned])

    def test_7b_batches_requests_and_applies_valid_review_and_recovery(self):
        raw = "Bệnh nhân sốt bn và rối loạn thị lực."
        start = raw.index("sốt bn")
        entities = [NerEntity("sốt bn", "TRIỆU_CHỨNG", [],
                              (start, start + len("sốt bn")), 0.5, "boundary_signal")]
        region = SuspiciousRegion(0, 0, len(raw), start, len(raw), ("boundary_signal",), 3.0)
        handoff = build_handoff_requests(raw, entities, [region])
        review_id = handoff["review_regions"][0]["request_id"]
        recovery_id = handoff["region_recoveries"][0]["request_id"]
        fever_end = start + len("sốt")
        vision_start = raw.index("rối loạn thị lực")
        responses = [[
            json.dumps({"request_id": review_id, "decisions": [{
                "candidate_id": 0, "action": "REPAIR_SPAN", "text": "sốt",
                "type": "TRIỆU_CHỨNG", "global_position": [start, fever_end]
            }]}, ensure_ascii=False),
            json.dumps({"request_id": recovery_id, "new_entities": [{
                "text": "rối loạn thị lực", "type": "TRIỆU_CHỨNG",
                "relative_position": [vision_start, vision_start + len("rối loạn thị lực")],
                "assertions": []
            }]}, ensure_ascii=False),
        ]]
        llm = _BatchLlm(responses)

        result, logs = review_entities_batch(
            {"doc": raw}, {"doc": entities}, {"doc": handoff}, llm,
            batch_size=4, retry_rounds=0,
        )

        self.assertEqual(["sốt", "rối loạn thị lực"], [entity.text for entity in result["doc"]])
        self.assertEqual(1, llm.calls)
        self.assertTrue(any(log["status"] == "decision_applied" for log in logs))

    def test_invalid_batch_retries_only_failed_request_then_falls_back(self):
        raw = "sốt"
        entity = NerEntity(raw, "TRIỆU_CHỨNG", [], (0, len(raw)), 0.5, "boundary_signal")
        handoff = build_handoff_requests(raw, [entity], [])
        request_id = handoff["review_regions"][0]["request_id"]
        llm = _BatchLlm([["not json"], [json.dumps({
            "request_id": request_id,
            "decisions": [{"candidate_id": 0, "action": "KEEP"}],
        })]])

        result, logs = review_entities_batch(
            {"doc": raw}, {"doc": [entity]}, {"doc": handoff}, llm,
            batch_size=4, retry_rounds=1,
        )

        self.assertEqual([raw], [item.text for item in result["doc"]])
        self.assertEqual(2, llm.calls)
        self.assertTrue(any(log["status"] == "response_rejected" for log in logs))

    def test_recovery_rejects_object_assertion_without_crashing(self):
        raw = "Bệnh nhân rối loạn thị lực."
        focus_start = raw.index("rối loạn thị lực")
        focus_end = focus_start + len("rối loạn thị lực")
        region = SuspiciousRegion(
            0, 0, len(raw), focus_start, focus_end,
            ("suspicious_empty_region",), 3.0,
        )
        handoff = build_handoff_requests(raw, [], [region])
        request_id = handoff["region_recoveries"][0]["request_id"]
        llm = _BatchLlm([[
            json.dumps({
                "request_id": request_id,
                "new_entities": [{
                    "text": "rối loạn thị lực",
                    "type": "TRIỆU_CHỨNG",
                    "relative_position": [focus_start, focus_end],
                    "assertions": [{"name": "isHistorical"}],
                }],
            }, ensure_ascii=False),
        ]])

        result, logs = review_entities_batch(
            {"doc": raw}, {"doc": []}, {"doc": handoff}, llm,
            batch_size=4, retry_rounds=0,
        )

        self.assertEqual([], result["doc"])
        self.assertTrue(any(
            log.get("status") == "recovery_rejected"
            and log.get("reason") == "invalid_schema"
            for log in logs
        ))


if __name__ == "__main__":
    unittest.main()
