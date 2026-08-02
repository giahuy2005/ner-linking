import json
import unittest
from pathlib import Path

from src.inference.pipeline import InferencePipeline
from src.inference.rule.clinical import apply_clinical_rules, deterministic_cleanup
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



class NewNerPipelineTests(unittest.TestCase):
    @staticmethod
    def _cleanup_saved_output(record_id: int, output_dir: str = "output"):
        root = Path(__file__).resolve().parents[1]
        raw = (root / "data" / "input" / f"{record_id}.txt").read_text(encoding="utf-8")
        rows = json.loads(
            (root / output_dir / f"{record_id}.json").read_text(encoding="utf-8")
        )
        entities = [NerEntity(
            text=row["text"],
            type=row["type"],
            assertions=row.get("assertions", []),
            position=tuple(row["position"]),
        ) for row in rows]
        cleaned, logs = deterministic_cleanup(raw, entities)
        return raw, cleaned, logs

    def test_gold_btc_medication_list(self):
        symptom_surfaces = ["ho", "đau nhức", "sốt đau", "táo bón", "táo bón",
                            "lo âu", "lo âu", "mất ngủ"]
        symptom_entities = []
        cursor = 0
        for surface in symptom_surfaces:
            start = GOLD_TEXT.index(surface, cursor)
            symptom_entities.append(NerEntity(
                surface, "TRIỆU_CHỨNG", [], (start, start + len(surface)), score=0.9,
            ))
            cursor = start + len(surface)

        entities, logs = apply_clinical_rules(GOLD_TEXT, symptom_entities)
        drugs = [entity for entity in entities if entity.type == "THUỐC"]
        symptoms = [entity for entity in entities if entity.type == "TRIỆU_CHỨNG"]

        self.assertEqual(11, len(drugs))
        self.assertTrue(all(entity.assertions == ["isHistorical"] for entity in drugs))
        self.assertEqual(
            symptom_surfaces,
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

    def test_boundary_repairs_do_not_delete_by_memorized_surface(self):
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

        self.assertEqual(["sốt", "vàng da", "Thiếu men G6PD", "chụp ct sọ não", "ăn ngủ"],
                         [entity.text for entity in cleaned])

    def test_boundary_cleanup_expands_a_span_cut_inside_unicode_word(self):
        raw = "Người bệnh khó thở và tăng huyết áp."
        symptom_start = raw.index("khó thở")
        diagnosis_start = raw.index("tăng huyết áp")
        entities = [
            NerEntity("khó th", "TRIỆU_CHỨNG", [],
                      (symptom_start, symptom_start + len("khó th")), 0.95),
            NerEntity("ăng huyết áp", "CHẨN_ĐOÁN", [],
                      (diagnosis_start + 1, diagnosis_start + len("tăng huyết áp")), 0.95),
        ]

        cleaned, _logs = deterministic_cleanup(raw, entities)

        self.assertEqual(["khó thở", "tăng huyết áp"], [entity.text for entity in cleaned])

    def test_standalone_dose_is_not_a_linkable_drug(self):
        raw = "Dùng 1 gram rồi theo dõi."
        start = raw.index("1 gram")
        entity = NerEntity("1 gram", "THUỐC", [], (start, start + len("1 gram")), 0.9)

        cleaned, logs = deterministic_cleanup(raw, [entity])

        self.assertEqual([], cleaned)
        self.assertTrue(any(log["reason"] == "isolated_measurement" for log in logs))

    def test_structural_boundary_cleanup_does_not_expand_medical_vocabulary(self):
        raw = "bệnh hiếm\nMặc; Cấy mẫu, dịch vị; làm làm xét nghiệm"
        surfaces = ["bệnh hiếm\nMặc", "dịch vị", "làm làm xét nghiệm"]
        types = ["CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "TÊN_XÉT_NGHIỆM"]
        entities = []
        for surface, entity_type in zip(surfaces, types):
            start = raw.index(surface)
            entities.append(NerEntity(surface, entity_type, [],
                                      (start, start + len(surface)), 0.7))

        cleaned, _ = deterministic_cleanup(raw, entities)

        self.assertEqual(["bệnh hiếm", "dịch vị", "làm xét nghiệm"],
                         [entity.text for entity in cleaned])


if __name__ == "__main__":
    unittest.main()
