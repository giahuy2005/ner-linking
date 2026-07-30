import json
import unittest
from collections import Counter
from unittest.mock import patch

from src.data_gen.generate_data import (
    BTC_MEDICATION_GOLD_OUTPUT,
    BTC_MEDICATION_GOLD_TEXT,
    SECTION_TYPES,
    V3_VERY_LONG_FOCUS,
    V4_FOCUS_AREAS,
    V5_FOCUS_AREAS,
    V5_DIRTY_RECORD_PERCENT,
    V5_QA_RECORD_PERCENT,
    V6_ASSERTION_TAXONOMY,
    V6_ERROR_TAXONOMY,
    V6_FOCUS_AREAS,
    build_btc_medication_gold_record,
    build_v5_focus_schedule,
    build_v6_focus_schedule,
    build_generation_messages,
    choose_soft_focus,
    completion_tokens_for_focus,
    forced_long_focus_index,
    load_vihealthqa_style_profile,
    parse_llm_json,
    validate_focus_quality,
)


class GenerateProfileTests(unittest.TestCase):
    def test_v6_schedule_matches_failure_curriculum(self):
        with patch("src.data_gen.generate_data.random.shuffle", side_effect=lambda items: None):
            schedule = build_v6_focus_schedule(600)

        counts = Counter(focus["key"] for focus in schedule)
        self.assertEqual(600, len(schedule))
        self.assertEqual(
            {focus["key"]: focus["quota_weight"] for focus in V6_FOCUS_AREAS},
            dict(counts),
        )
        self.assertIn("truncated_or_short_span", V6_ERROR_TAXONOMY)
        self.assertIn("negation_exception_and_false_cue", V6_ASSERTION_TAXONOMY)

    def test_btc_medication_anchor_is_exact_and_offset_safe(self):
        record = build_btc_medication_gold_record(include_linking=True)
        self.assertEqual(BTC_MEDICATION_GOLD_TEXT, record["input_text"])
        self.assertEqual(len(BTC_MEDICATION_GOLD_OUTPUT), len(record["entities"]))

        drugs = []
        for expected, entity in zip(BTC_MEDICATION_GOLD_OUTPUT, record["entities"]):
            text, entity_type, candidate, position, assertions = expected
            self.assertEqual(text, record["input_text"][slice(*entity["position"])])
            self.assertEqual(entity_type, entity["type"])
            self.assertEqual(list(position), entity["position"])
            self.assertEqual(list(assertions), entity["assertions"])
            if candidate is None:
                self.assertNotIn("candidates", entity)
            else:
                self.assertEqual([candidate], entity["candidates"])
            if entity_type == "THUỐC":
                drugs.append(entity)

        self.assertEqual(11, len(drugs))
        self.assertTrue(all(e["assertions"] == ["isHistorical"] for e in drugs))
        symptoms = [e for e in record["entities"] if e["type"] == "TRIỆU_CHỨNG"]
        self.assertTrue(all(e["assertions"] == [] for e in symptoms))

    def test_v6_medication_prompt_and_quality_follow_btc_contract(self):
        focus = next(f for f in V6_FOCUS_AREAS if f.get("btc_medication_contract"))
        messages = build_generation_messages(
            SECTION_TYPES[0], [], [], [], {}, {}, None, focus_cfg=focus
        )
        prompt = messages[-1]["content"]
        escaped_anchor_text = json.dumps(BTC_MEDICATION_GOLD_TEXT, ensure_ascii=False)[1:-1]
        self.assertIn(escaped_anchor_text, prompt)
        self.assertIn("candidates chỉ là neo linking", prompt)
        self.assertIn("không tự kế thừa isHistorical", prompt)
        self.assertIn("KHÔNG sao chép nguyên văn", prompt)

        record = build_btc_medication_gold_record(include_linking=False)
        self.assertIsNone(validate_focus_quality(record, focus))

    def test_v6_assertion_quality_rejects_missing_or_multiple_assertions(self):
        focus = next(
            f for f in V6_FOCUS_AREAS
            if f["key"] == "assertion_negation_exception_scope"
        )
        valid = {
            "input_text": "Không đau ngực, nhưng vẫn ho và sốt kèm khó thở.",
            "entities": [
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"]},
                {"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "khó thở", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }
        self.assertIsNone(validate_focus_quality(valid, focus))

        missing = {**valid, "entities": [dict(e) for e in valid["entities"]]}
        missing["entities"][0]["assertions"] = []
        self.assertIn("thiếu assertion", validate_focus_quality(missing, focus))

        multiple = {**valid, "entities": [dict(e) for e in valid["entities"]]}
        multiple["entities"][0]["assertions"] = ["isNegated", "isHistorical"]
        self.assertIn("tối đa một assertion", validate_focus_quality(multiple, focus))

    def test_v6_fragment_focus_requires_long_and_short_valid_entities(self):
        focus = next(
            f for f in V6_FOCUS_AREAS if f["key"] == "ner_truncated_and_short_spans"
        )
        valid = {
            "input_text": "Ghi nhận suy hô hấp cấp tiến triển, ho; chỉ định CT.",
            "entities": [
                {
                    "text": "suy hô hấp cấp tiến triển",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": [],
                },
                {"text": "CT", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }
        self.assertIsNone(validate_focus_quality(valid, focus))
        invalid = {**valid, "entities": valid["entities"][:1]}
        self.assertIn("entity ngắn", validate_focus_quality(invalid, focus))

    def test_v5_schedule_matches_600_plan_and_dirty_ratio(self):
        with patch("src.data_gen.generate_data.random.shuffle", side_effect=lambda items: None), \
             patch("src.data_gen.generate_data.random.sample", side_effect=lambda items, k: items[:k]):
            schedule = build_v5_focus_schedule(600)

        counts = {}
        for focus in schedule:
            counts[focus["key"]] = counts.get(focus["key"], 0) + 1
        self.assertEqual(130, counts["contrastive_assertions"])
        self.assertEqual(110, counts["dense_ner_boundaries"])
        self.assertEqual(
            90,
            counts["sparse_zero_entity"]
            + counts["sparse_one_type"]
            + counts["sparse_two_types"],
        )
        self.assertEqual(80, counts["false_cues_and_scope"])
        self.assertEqual(50, counts["dirty_btc_text"])
        self.assertEqual(80, counts["btc_medication_lists"])
        self.assertEqual(60, counts["complete_occurrence_recall"])
        self.assertEqual(
            round(600 * V5_DIRTY_RECORD_PERCENT / 100),
            sum(bool(focus.get("boundary_noise")) for focus in schedule),
        )
        self.assertEqual(
            round(600 * V5_QA_RECORD_PERCENT / 100),
            sum(bool(focus.get("qa_style")) for focus in schedule),
        )

    def test_v5_sparse_and_scope_quality_rules(self):
        sparse_zero = next(f for f in V5_FOCUS_AREAS if f.get("sparse_variant") == "zero")
        self.assertIsNone(validate_focus_quality({"input_text": "Hẹn tái khám.", "entities": []}, sparse_zero))

        sparse_one = next(f for f in V5_FOCUS_AREAS if f.get("sparse_variant") == "one_type")
        valid_one = {
            "input_text": "Người bệnh ho khan và khó thở.",
            "entities": [
                {"text": "ho khan", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "khó thở", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }
        self.assertIsNone(validate_focus_quality(valid_one, sparse_one))

        scope = next(f for f in V5_FOCUS_AREAS if f["key"] == "false_cues_and_scope")
        valid_scope = {
            "input_text": "Cha mắc tiểu đường. Người bệnh từng tăng huyết áp, phủ nhận đau ngực.",
            "entities": [
                {"text": "tiểu đường", "type": "CHẨN_ĐOÁN", "assertions": ["isFamily"]},
                {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "assertions": ["isHistorical"]},
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"]},
            ],
        }
        self.assertIsNone(validate_focus_quality(valid_scope, scope))

    def test_v5_prompt_explicitly_allows_zero_entity_and_fake_cues(self):
        focus = next(f for f in V5_FOCUS_AREAS if f.get("sparse_variant") == "zero")
        messages = build_generation_messages(
            SECTION_TYPES[0], [], [], [], {}, {}, None, focus_cfg=focus
        )
        prompt = messages[-1]["content"]
        self.assertIn("`entities` bắt buộc là []", prompt)
        self.assertIn("Record thưa được phép có 0 entity", prompt)
        self.assertIn("không cải thiện", prompt)

    def test_v5_medication_focus_anchors_btc_boundaries_and_assertions(self):
        focus = next(f for f in V5_FOCUS_AREAS if f["key"] == "btc_medication_lists")
        messages = build_generation_messages(
            SECTION_TYPES[0], [], [], [], {}, {}, None, focus_cfg=focus
        )
        prompt = messages[-1]["content"]
        self.assertIn("tên + strength + dose form + route + frequency", prompt)
        self.assertIn("không tự kế thừa isHistorical", prompt)
        self.assertIn("Không được bỏ item cuối", prompt)

        valid = {
            "input_text": "a b c d e ho",
            "entities": [
                {"text": value, "type": "THUỐC", "assertions": ["isHistorical"]}
                for value in "abcde"
            ] + [{"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": []}],
        }
        self.assertIsNone(validate_focus_quality(valid, focus))

        invalid = dict(valid)
        invalid["entities"] = [dict(entity) for entity in valid["entities"]]
        invalid["entities"][-1]["assertions"] = ["isHistorical"]
        self.assertIn("không được kế thừa", validate_focus_quality(invalid, focus))

    def test_v5_recall_focus_rejects_missing_repeated_occurrence(self):
        focus = next(f for f in V5_FOCUS_AREAS if f["key"] == "complete_occurrence_recall")
        record = {
            "input_text": "ho khan, ho khan và sốt.",
            "entities": [
                {"text": "ho khan", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }
        self.assertIn("occurrence", validate_focus_quality(record, focus))

    def test_v4_covers_observed_btc_long_formats(self):
        self.assertEqual(
            {"qa", "clinical_long", "hybrid", "education", "long_timeline"},
            {focus["format"] for focus in V4_FOCUS_AREAS},
        )
        self.assertTrue(all(focus["mode"] == "v4" for focus in V4_FOCUS_AREAS))

    def test_mixed_v4_selects_only_qa_focus(self):
        with patch(
            "src.data_gen.generate_data.random.random",
            side_effect=[0.0, 0.0, 0.99],
        ):
            focus = choose_soft_focus("mixed_v4", SECTION_TYPES[1])

        self.assertIsNotNone(focus)
        self.assertEqual("v4", focus["mode"])
        self.assertEqual("qa", focus["format"])
        self.assertFalse(focus["boundary_noise"])

    def test_mixed_v4_can_keep_baseline(self):
        with patch("src.data_gen.generate_data.random.random", return_value=0.999):
            focus = choose_soft_focus("mixed_v4", SECTION_TYPES[1])

        self.assertIsNone(focus)

    def test_mixed_v4_can_select_non_qa_v4_focus(self):
        # Lần đầu qua cổng V4; lần hai không qua cổng QA.
        with patch(
            "src.data_gen.generate_data.random.random",
            side_effect=[0.0, 0.99, 0.99],
        ):
            focus = choose_soft_focus("mixed_v4", SECTION_TYPES[1])

        self.assertIsNotNone(focus)
        self.assertEqual("education", focus["format"])

    def test_mixed_v4_can_add_controlled_boundary_noise(self):
        with patch(
            "src.data_gen.generate_data.random.random",
            side_effect=[0.0, 0.0, 0.0],
        ):
            focus = choose_soft_focus("mixed_v4", SECTION_TYPES[1])

        self.assertTrue(focus["boundary_noise"])

        messages = build_generation_messages(
            SECTION_TYPES[1], [], [], [], {}, {}, None, focus,
        )
        self.assertIn("NHIỄU RANH GIỚI CÓ KIỂM SOÁT", messages[-1]["content"])

    def test_boundary_noise_never_allows_entity_to_cross_two_sentences(self):
        focus = dict(V4_FOCUS_AREAS[0], boundary_noise=True)
        valid = {
            "input_text": "Người bệnh đau ngực.Hồi hộp tăng dần.",
            "entities": [
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "Hồi hộp", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }
        self.assertIsNone(validate_focus_quality(valid, focus))

        invalid = dict(valid)
        invalid["entities"] = [
            {"text": "đau ngực.Hồi hộp", "type": "TRIỆU_CHỨNG", "assertions": []}
        ]
        self.assertIn("ranh giới", validate_focus_quality(invalid, focus))

    def test_v4_boundary_noise_is_soft_when_llm_omits_noise(self):
        focus = dict(V4_FOCUS_AREAS[0], boundary_noise=True)
        clean_without_noise = {
            "input_text": "Người bệnh đau ngực và hồi hộp.",
            "entities": [
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "hồi hộp", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }
        self.assertIsNone(validate_focus_quality(clean_without_noise, focus))

    def test_qa_prompt_overrides_clinical_note_format(self):
        messages = build_generation_messages(
            SECTION_TYPES[1],
            seed_examples=[],
            drug_pool=[],
            vitals_pool=[],
            icd10_by_chapter={},
            specialty_pool={},
            force_assertion=None,
            focus_cfg=V4_FOCUS_AREAS[0],
        )
        prompt = messages[-1]["content"]

        self.assertIn("HỎI ĐÁP Y KHOA DÀI, ĐA DẠNG NGUỒN", prompt)
        self.assertIn('"Hỏi/Đáp"', prompt)
        self.assertIn('"Người bệnh/Bác sĩ"', prompt)
        self.assertIn("BẮT BUỘC có một cặp nhãn phân cách rõ", prompt)
        self.assertIn("FREE-FORM DÀI VÀ ĐA DẠNG", prompt)
        self.assertIn("<free-form text V4 format qa>", prompt)
        self.assertNotIn("Heading chỉ xuất hiện MỘT LẦN ở dòng đầu", prompt)
        for public_case_fragment in ("amyloidosis", "G6PD", "160/70", "ctchưa"):
            self.assertNotIn(public_case_fragment, prompt)

    def test_v4_uses_only_aggregate_vihealthqa_metadata(self):
        profile = load_vihealthqa_style_profile()
        self.assertTrue(profile["available"])
        self.assertGreater(profile["rows"], 1000)
        self.assertNotIn("question", profile)
        self.assertNotIn("answer", profile)

        messages = build_generation_messages(
            SECTION_TYPES[1],
            seed_examples=[],
            drug_pool=[],
            vitals_pool=[],
            icd10_by_chapter={},
            specialty_pool={},
            force_assertion=None,
            focus_cfg=V4_FOCUS_AREAS[0],
        )
        prompt = messages[-1]["content"]
        self.assertIn("CHỈ THỐNG KÊ TỔNG HỢP", prompt)
        self.assertNotIn("vnexpress.net", prompt)
        self.assertNotIn("Đang chích ngừa viêm gan B", prompt)

    def test_v3_can_select_very_long_focus_and_get_larger_budget(self):
        with patch("src.data_gen.generate_data.random.random", return_value=0.0):
            focus = choose_soft_focus("mixed_v3", SECTION_TYPES[1])
        self.assertEqual(V3_VERY_LONG_FOCUS, focus)
        self.assertEqual(3200, completion_tokens_for_focus(focus))
        self.assertEqual(1400, completion_tokens_for_focus(None))

    def test_small_audit_batch_guarantees_one_compatible_long_slot(self):
        for profile in ("mixed_v3", "mixed_v4"):
            index = forced_long_focus_index(profile, 20)
            self.assertIsNotNone(index)
            self.assertIn(
                SECTION_TYPES[index % len(SECTION_TYPES)]["key"],
                {"hien_tai", "danh_gia"},
            )
        self.assertIsNone(forced_long_focus_index("mixed_v3", 5))

    def test_long_focus_only_hard_rejects_when_far_too_short(self):
        focus = V3_VERY_LONG_FOCUS
        short = {"input_text": "đau ngực", "entities": []}
        self.assertIn("ít nhất 350", validate_focus_quality(short, focus))

        long_without_repeat = {
            "input_text": "x" * 1500,
            "entities": [{"text": "x", "type": "TRIỆU_CHỨNG", "assertions": []}],
        }
        self.assertIsNone(validate_focus_quality(long_without_repeat, focus))

        long_with_repeat = {
            "input_text": "đau ngực " + "x" * 1450 + " đau ngực",
            "entities": [
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }
        self.assertIsNone(validate_focus_quality(long_with_repeat, focus))

    def test_parse_llm_json_salvages_fenced_or_trailing_content(self):
        expected = {"input_text": "x", "entities": []}
        self.assertEqual(
            expected,
            parse_llm_json('```json\n{"input_text":"x","entities":[]}\n```'),
        )
        self.assertEqual(
            expected,
            parse_llm_json('{"input_text":"x","entities":[]}\nGhi chú thừa'),
        )


if __name__ == "__main__":
    unittest.main()
