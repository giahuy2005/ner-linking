import unittest
from unittest.mock import patch

from src.data_gen.generate_data import (
    SECTION_TYPES,
    V3_VERY_LONG_FOCUS,
    V4_FOCUS_AREAS,
    V5_FOCUS_AREAS,
    V5_DIRTY_RECORD_PERCENT,
    V5_QA_RECORD_PERCENT,
    build_v5_focus_schedule,
    build_generation_messages,
    choose_soft_focus,
    completion_tokens_for_focus,
    forced_long_focus_index,
    load_vihealthqa_style_profile,
    parse_llm_json,
    validate_focus_quality,
)


class GenerateProfileTests(unittest.TestCase):
    def test_v5_schedule_matches_600_plan_and_dirty_ratio(self):
        with patch("src.data_gen.generate_data.random.shuffle", side_effect=lambda items: None), \
             patch("src.data_gen.generate_data.random.sample", side_effect=lambda items, k: items[:k]):
            schedule = build_v5_focus_schedule(600)

        counts = {}
        for focus in schedule:
            counts[focus["key"]] = counts.get(focus["key"], 0) + 1
        self.assertEqual(180, counts["contrastive_assertions"])
        self.assertEqual(150, counts["dense_ner_boundaries"])
        self.assertEqual(
            100,
            counts["sparse_zero_entity"]
            + counts["sparse_one_type"]
            + counts["sparse_two_types"],
        )
        self.assertEqual(100, counts["false_cues_and_scope"])
        self.assertEqual(70, counts["dirty_btc_text"])
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
