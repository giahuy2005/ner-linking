import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data_gen"))

from gen_reject import detect_section, process_record


class GenRejectRuleTests(unittest.TestCase):
    def test_v5_can_keep_zero_entity_without_relaxing_default_minimum(self):
        sparse_record = {
            "input_text": "Người gọi hẹn tái khám vào sáng thứ hai và sẽ mang giấy tờ tùy thân.",
            "entities": [],
            "_require_lab_pair": False,
            "_min_entities": 0,
        }
        status, cleaned, logs = process_record(sparse_record)
        self.assertEqual("keep", status)
        self.assertEqual([], cleaned["entities"])

        default_record = dict(sparse_record)
        default_record.pop("_min_entities")
        status, rejection = process_record(default_record)
        self.assertEqual("reject", status)
        self.assertEqual("too_few_entity_reject", rejection["stage"])

    def test_v5_fake_negation_cues_do_not_negate_real_conditions(self):
        record = {
            "input_text": (
                "Bệnh nhân không nhớ rõ thời điểm bắt đầu đau ngực. "
                "Khó thở không cải thiện sau khi nghỉ. "
                "Bệnh nhân không dùng thuốc điều trị tăng huyết áp đều đặn."
            ),
            "entities": [
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "Khó thở", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "assertions": []},
            ],
            "_require_lab_pair": False,
        }
        status, cleaned, logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertTrue(all(entity["assertions"] == [] for entity in cleaned["entities"]))

    def test_v5_nearby_assertion_cues_remain_local(self):
        record = {
            "input_text": (
                "Mẹ bệnh nhân mắc tăng huyết áp. Bệnh nhân từng bị sỏi thận nhưng hiện "
                "phủ nhận đau ngực, nhập viện vì khó thở tăng dần."
            ),
            "entities": [
                {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "assertions": ["isFamily"]},
                {"text": "sỏi thận", "type": "CHẨN_ĐOÁN", "assertions": ["isHistorical"]},
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"]},
                {"text": "khó thở tăng dần", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
            "_require_lab_pair": False,
        }
        status, cleaned, logs = process_record(record)
        self.assertEqual("keep", status)
        by_text = {entity["text"]: entity["assertions"] for entity in cleaned["entities"]}
        self.assertEqual(["isFamily"], by_text["tăng huyết áp"])
        self.assertEqual(["isHistorical"], by_text["sỏi thận"])
        self.assertEqual(["isNegated"], by_text["đau ngực"])
        self.assertEqual([], by_text["khó thở tăng dần"])

    def test_splits_multi_pair_blood_count_result(self):
        merged = (
            "WBC:12,5; NEUT% (Tỷ lệ % bạch cầu trung tính):78,2; "
            "LYPH% (Tỷ lệ bạch cầu lympho):15,3"
        )
        record = {
            "input_text": f"Kết quả đánh giá ban đầu\nCông thức máu: {merged}.",
            "entities": [
                {"text": "Công thức máu", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": merged, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        result = process_record(record)
        self.assertEqual("keep", result[0], result)
        status, cleaned, logs = result

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("Công thức máu", "TÊN_XÉT_NGHIỆM"),
                ("WBC", "TÊN_XÉT_NGHIỆM"),
                ("12,5", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("NEUT% (Tỷ lệ % bạch cầu trung tính)", "TÊN_XÉT_NGHIỆM"),
                ("78,2", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("LYPH% (Tỷ lệ bạch cầu lympho)", "TÊN_XÉT_NGHIỆM"),
                ("15,3", "KẾT_QUẢ_XÉT_NGHIỆM"),
            ],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-split-multi-lab" in log for log in logs))

    def test_consolidates_holter_finding_and_drops_context(self):
        finding = (
            "Nhịp xoang chiếm ưu thế. Ghi nhận ngoại tâm thu nhĩ và "
            "ngoại tâm thu thất xuất hiện thường xuyên"
        )
        record = {
            "input_text": (
                "Bệnh sử hiện tại\n"
                f"monitor holter cho thấy {finding}.\n"
                "Không có khó chịu vùng ngực khi nghỉ và không liên quan đến gắng sức."
            ),
            "entities": [
                {"text": "Nhịp", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "Nhịp xoang", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "ngoại tâm thu nhĩ", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "ngoại tâm thu thất", "type": "CHẨN_ĐOÁN", "assertions": []},
                {
                    "text": "khó chịu vùng ngực khi",
                    "type": "TRIỆU_CHỨNG",
                    "assertions": ["isNegated"],
                },
                {"text": "gắng sức", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"]},
            ],
        }

        result = process_record(record)
        self.assertEqual("keep", result[0], result)
        status, cleaned, logs = result

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("monitor holter", "TÊN_XÉT_NGHIỆM", []),
                (finding, "KẾT_QUẢ_XÉT_NGHIỆM", []),
                ("khó chịu vùng ngực", "TRIỆU_CHỨNG", ["isNegated"]),
            ],
            [
                (entity["text"], entity["type"], entity["assertions"])
                for entity in cleaned["entities"]
            ],
        )
        self.assertTrue(any("autofix-holter-clause" in log for log in logs))
        self.assertTrue(any("trailing-symptom-connector" in log for log in logs))
        self.assertTrue(any("non-condition" in log for log in logs))

    def test_consolidates_ecg_narrative_result(self):
        record = {
            "input_text": (
                "Tiền căn bệnh lý\nKết quả điện tâm đồ gần nhất ghi nhận "
                "nhịp xoang 85 lần/phút. Không khó thở."
            ),
            "entities": [
                {"text": "điện tâm đồ", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "nhịp xoang", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "85 lần/phút", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "khó thở", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"]},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("điện tâm đồ", "TÊN_XÉT_NGHIỆM"),
                ("nhịp xoang 85 lần/phút", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("khó thở", "TRIỆU_CHỨNG"),
            ],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-ecg-clause" in log for log in logs))

    def test_fixes_echo_name_and_full_valve_finding(self):
        record = {
            "input_text": "Đánh giá tại bệnh viện\nSiêu âm tim ghi nhận hở van động mạch chủ độ 2. CRP 4 mg/L.",
            "entities": [
                {"text": "hở van động mạch chủ độ", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "2", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "CRP", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "4 mg/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                {"text": "Siêu âm tim", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "hở van động mạch chủ độ 2", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "CRP", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "4 mg/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
            cleaned["entities"],
        )
        self.assertTrue(
            any(
                "autofix-echo-valve-finding" in log or "autofix-imaging-clause" in log
                for log in logs
            )
        )

    def test_rejects_allopurinol_for_lipid_disorder(self):
        record = {
            "input_text": "Tiền sử bệnh\nAllopurinol 150 MG Oral Capsule điều trị rối loạn chuyển hóa lipid máu.",
            "entities": [
                {"text": "Allopurinol 150 MG Oral Capsule", "type": "THUỐC", "assertions": ["isHistorical"]},
                {"text": "rối loạn chuyển hóa lipid máu", "type": "CHẨN_ĐOÁN", "assertions": ["isHistorical"]},
            ],
        }

        status, rejection = process_record(record)

        self.assertEqual("reject", status)
        self.assertEqual("medical_inconsistency_reject", rejection["stage"])

    def test_extends_pruritus_with_severity_and_location(self):
        record = {
            "input_text": "Tiền sử bệnh hiện tại\nBệnh nhân ngứa nhiều vùng da mặt và cổ, kèm nổi ban đỏ.",
            "entities": [
                {"text": "ngứa", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "nổi ban đỏ", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual("ngứa nhiều vùng da mặt và cổ", cleaned["entities"][0]["text"])
        self.assertTrue(any("autofix-pruritus-full-span" in log for log in logs))

    def test_removes_family_from_disease_name_but_keeps_real_family_context(self):
        record = {
            "input_text": (
                "Tiền sử bệnh hiện tại\nCó tiền sử liệt chu kỳ gia đình. "
                "Mẹ bệnh nhân từng mắc đái tháo đường type 2."
            ),
            "entities": [
                {
                    "text": "liệt chu kỳ gia đình",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical", "isFamily"],
                },
                {
                    "text": "đái tháo đường type 2",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isFamily", "isHistorical"],
                },
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(["isHistorical"], cleaned["entities"][0]["assertions"])
        self.assertEqual(["isFamily", "isHistorical"], cleaned["entities"][1]["assertions"])
        self.assertTrue(any("autofix-invalid-family" in log for log in logs))

    def test_changes_drug_allergy_to_diagnosis(self):
        record = {
            "input_text": "Tiền sử bệnh\nKhông ghi nhận tiền sử dị ứng thuốc. Đang dùng Vitamin C.",
            "entities": [
                {"text": "dị ứng thuốc", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"]},
                {"text": "Vitamin C", "type": "THUỐC", "assertions": ["isHistorical"]},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual("CHẨN_ĐOÁN", cleaned["entities"][0]["type"])
        self.assertEqual(["isNegated"], cleaned["entities"][0]["assertions"])
        self.assertTrue(any("autofix-allergy-condition-type" in log for log in logs))

    def test_splits_compressed_vitals_consistently(self):
        record = {
            "input_text": "Đánh giá tại bệnh viện\nĐau bụng. VS 98.3 120/80 72 18 98RA.",
            "entities": [
                {"text": "Đau bụng", "type": "TRIỆU_CHỨNG", "assertions": []},
                {
                    "text": "VS 98.3 120/80 72 18 98RA",
                    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "assertions": [],
                },
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual("VS", cleaned["entities"][1]["text"])
        self.assertEqual("TÊN_XÉT_NGHIỆM", cleaned["entities"][1]["type"])
        self.assertEqual("98.3 120/80 72 18 98RA", cleaned["entities"][2]["text"])
        self.assertTrue(any("autofix-split-compressed-vitals" in log for log in logs))

    def test_rejects_riley_day_periodic_paralysis_mix(self):
        record = {
            "input_text": "Tiền sử bệnh hiện tại\nCó tiền sử liệt chu kỳ gia đình (Riley-Day), dùng phenacemide.",
            "entities": [
                {
                    "text": "liệt chu kỳ gia đình (Riley-Day)",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                },
                {"text": "phenacemide", "type": "THUỐC", "assertions": ["isHistorical"]},
            ],
        }

        status, rejection = process_record(record)

        self.assertEqual("reject", status)
        self.assertEqual("medical_inconsistency_reject", rejection["stage"])

    def test_fixes_full_mri_graded_finding(self):
        record = {
            "input_text": (
                "Đánh giá tại bệnh viện\nMRI khớp gối trái: thoái hóa khớp gối độ 2, "
                "tràn dịch khớp nhẹ."
            ),
            "entities": [
                {"text": "MRI khớp gối trái", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "thoái hóa khớp gối độ", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "2", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "tràn dịch khớp nhẹ", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                {"text": "MRI khớp gối trái", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {
                    "text": "thoái hóa khớp gối độ 2, tràn dịch khớp nhẹ",
                    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "assertions": [],
                },
            ],
            cleaned["entities"],
        )
        self.assertTrue(any("autofix-graded-imaging-finding" in log for log in logs))

    def test_recognizes_synonymous_section_headings(self):
        cases = {
            "Tiền căn bệnh lý\nNội dung": "tien_su",
            "Bệnh sử hiện tại\nNội dung": "hien_tai",
            "Đánh giá lâm sàng và cận lâm sàng\nNội dung": "danh_gia",
        }

        for text, expected_key in cases.items():
            with self.subTest(text=text):
                key, _ = detect_section(text)
                self.assertEqual(expected_key, key)

    def test_fixes_quantified_echo_finding(self):
        record = {
            "input_text": (
                "Đánh giá tại bệnh viện\nSiêu âm tim phát hiện rối loạn vận động vùng trước "
                "vách với phân suất tống máu EF 45%."
            ),
            "entities": [
                {"text": "Siêu âm tim", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {
                    "text": "rối loạn vận động vùng trước vách với phân suất tống máu EF",
                    "type": "TÊN_XÉT_NGHIỆM",
                    "assertions": [],
                },
                {"text": "45%", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            "rối loạn vận động vùng trước vách với phân suất tống máu EF 45%",
            cleaned["entities"][1]["text"],
        )
        self.assertEqual("KẾT_QUẢ_XÉT_NGHIỆM", cleaned["entities"][1]["type"])
        self.assertTrue(any("autofix-graded-imaging-finding" in log for log in logs))

    def test_cleans_diagnosis_context_from_spans(self):
        record = {
            "input_text": (
                "Tiền sử bệnh\nBệnh nhân có tiền sử loãng xương không triệu chứng. "
                "Không ghi nhận tiền sử gãy xương trước đây."
            ),
            "entities": [
                {
                    "text": "loãng xương không triệu chứng",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                },
                {
                    "text": "tiền sử gãy xương",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isNegated"],
                },
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual("loãng xương", cleaned["entities"][0]["text"])
        self.assertEqual(["isHistorical"], cleaned["entities"][0]["assertions"])
        self.assertEqual("gãy xương", cleaned["entities"][1]["text"])
        self.assertEqual(["isNegated"], cleaned["entities"][1]["assertions"])
        self.assertTrue(any("autofix-diagnosis-context-span" in log for log in logs))

    def test_fixes_hydronephrosis_finding_after_ultrasound(self):
        record = {
            "input_text": "Bệnh sử hiện tại\nSiêu âm thận: thận phải ứ nước độ 2.",
            "entities": [
                {"text": "Siêu âm thận", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "thận phải ứ nước độ", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "2", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual("Siêu âm thận", cleaned["entities"][0]["text"])
        self.assertEqual("thận phải ứ nước độ 2", cleaned["entities"][1]["text"])
        self.assertEqual("KẾT_QUẢ_XÉT_NGHIỆM", cleaned["entities"][1]["type"])
        self.assertTrue(any("autofix-graded-imaging-finding" in log for log in logs))

    def test_rejects_kidney_stone_typo(self):
        record = {
            "input_text": "Bệnh sử hiện tại\nChưa ghi nhận sót thận trên siêu âm.",
            "entities": [
                {"text": "sót thận", "type": "CHẨN_ĐOÁN", "assertions": ["isNegated"]},
                {"text": "siêu âm", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, rejection = process_record(record)

        self.assertEqual("reject", status)
        self.assertEqual("medical_inconsistency_reject", rejection["stage"])

    def test_rejects_duplicate_heading_and_stuck_text(self):
        record = {
            "input_text": (
                "2. Tiền sử bệnh hiện tại\n"
                "3. Tiền sử bệnh hiện tạiBệnh nhân đau bụng."
            ),
            "entities": [
                {"text": "đau bụng", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "Bệnh nhân", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }

        status, rejection = process_record(record)

        self.assertEqual("reject", status)
        self.assertEqual("text_quality_reject", rejection["stage"])

    def test_propagates_negation_across_symptom_list(self):
        record = {
            "input_text": (
                "Tiền căn bệnh lý\nKhông ghi nhận triệu chứng hạ đường huyết như "
                "vã mồ hôi, run tay hay choáng váng."
            ),
            "entities": [
                {"text": "vã mồ hôi", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "run tay", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "choáng váng", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [["isNegated"], ["isNegated"], ["isNegated"]],
            [entity["assertions"] for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-negation-scope" in log for log in logs))

    def test_negation_scope_stops_at_contrast_word(self):
        record = {
            "input_text": "Bệnh sử hiện tại\nBệnh nhân phủ nhận sốt nhưng có đau đầu.",
            "entities": [
                {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "đau đầu", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }

        status, cleaned, _ = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(["isNegated"], cleaned["entities"][0]["assertions"])
        self.assertEqual([], cleaned["entities"][1]["assertions"])

    def test_drops_procedure_but_keeps_transplant_failure_diagnosis(self):
        record = {
            "input_text": (
                "Tiền sử bệnh\nBệnh nhân từng ghép thận, hiện có ghép thận thất bại "
                "và suy thận mạn giai V."
            ),
            "entities": [
                {"text": "ghép thận", "type": "CHẨN_ĐOÁN", "assertions": ["isHistorical"]},
                {"text": "ghép thận thất bại", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "suy thận mạn giai V", "type": "CHẨN_ĐOÁN", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            ["ghép thận thất bại", "suy thận mạn giai V"],
            [entity["text"] for entity in cleaned["entities"]],
        )
        self.assertTrue(any("reject-entity-procedure" in log for log in logs))

    def test_recovers_transplant_failure_hidden_by_procedure_wording(self):
        record = {
            "input_text": (
                "Tiền sử bệnh\nSuy thận mạn giai V sau khi ghép thận thất bại. "
                "Mục thủ thuật ghi lại ghép thận thất bại."
            ),
            "entities": [
                {
                    "text": "Suy thận mạn giai V",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                }
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            ["Suy thận mạn giai V", "ghép thận thất bại", "ghép thận thất bại"],
            [entity["text"] for entity in cleaned["entities"]],
        )
        self.assertTrue(all("isHistorical" in entity["assertions"] for entity in cleaned["entities"]))
        self.assertTrue(any("autofix-diagnosis-recall" in log for log in logs))

    def test_splits_symptom_due_to_diagnosis(self):
        record = {
            "input_text": "Bệnh sử hiện tại\nBệnh nhân giọng khàn do tổn thương dây thanh quản.",
            "entities": [
                {
                    "text": "giọng khàn do tổn thương dây thanh quản",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": [],
                }
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [("giọng khàn", "TRIỆU_CHỨNG"), ("tổn thương dây thanh quản", "CHẨN_ĐOÁN")],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-symptom-cause" in log for log in logs))

    def test_merges_dynamic_lab_result(self):
        record = {
            "input_text": (
                "Đánh giá tại bệnh viện\nCreatinine tăng từ 5.2 lên 6.3 mg/dl "
                "(460 - 557 umol/l)."
            ),
            "entities": [
                {"text": "Creatinine", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "5.2", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "6.3 mg/dl", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "460 - 557 umol/l", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(2, len(cleaned["entities"]))
        self.assertEqual(
            "tăng từ 5.2 lên 6.3 mg/dl (460 - 557 umol/l)",
            cleaned["entities"][1]["text"],
        )
        self.assertTrue(any("autofix-dynamic-lab" in log for log in logs))

    def test_merges_ure_dynamic_result_with_conversion_parentheses(self):
        record = {
            "input_text": (
                "Đánh giá tại bệnh viện\nUre tăng từ 69 lên 91 mg/dl "
                "( 24.6 -32.5 mmol/l) trong 2 tháng qua."
            ),
            "entities": [
                {"text": "Ure", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "69", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "91 mg/dl", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {
                    "text": "24.6 -32.5 mmol/l)",
                    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "assertions": [],
                },
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(2, len(cleaned["entities"]))
        self.assertEqual(
            "tăng từ 69 lên 91 mg/dl ( 24.6 -32.5 mmol/l)",
            cleaned["entities"][1]["text"],
        )
        self.assertTrue(any("autofix-dynamic-lab" in log for log in logs))

    def test_history_section_adds_historical_to_diagnoses(self):
        record = {
            "input_text": "Tiền căn bệnh lý\n- Rung nhĩ\n- Suy tim",
            "entities": [
                {"text": "Rung nhĩ", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "Suy tim", "type": "CHẨN_ĐOÁN", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [["isHistorical"], ["isHistorical"]],
            [entity["assertions"] for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-history-section" in log for log in logs))

    def test_no_heading_uses_internal_section_hint_but_drops_metadata(self):
        record = {
            "input_text": "Đang theo dõi rung nhĩ nhiều năm, kèm suy tim mạn.",
            "entities": [
                {"text": "rung nhĩ", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "suy tim mạn", "type": "CHẨN_ĐOÁN", "assertions": []},
            ],
            "_section_key_hint": "tien_su",
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertNotIn("_section_key_hint", cleaned)
        self.assertEqual(
            [["isHistorical"], ["isHistorical"]],
            [entity["assertions"] for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-history-section" in log for log in logs))

    def test_previous_admission_drug_is_historical(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\nTrong lần nhập viện trước bệnh nhân đã dùng ciproflagyl. "
                "Hiện còn đau bụng."
            ),
            "entities": [
                {"text": "ciproflagyl", "type": "THUỐC", "assertions": []},
                {"text": "đau bụng", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(["isHistorical"], cleaned["entities"][0]["assertions"])
        self.assertTrue(any("autofix-drug-home-timing" in log for log in logs))

    def test_postposed_previous_admission_marker_is_historical(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\nBệnh nhân sử dụng ciproflagyl trong lần nhập viện trước, "
                "hiện còn đau bụng."
            ),
            "entities": [
                {"text": "ciproflagyl", "type": "THUỐC", "assertions": []},
                {"text": "đau bụng", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(["isHistorical"], cleaned["entities"][0]["assertions"])
        self.assertTrue(any("autofix-drug-home-timing" in log for log in logs))

    def test_recovers_temperature_name_and_inr_result(self):
        record = {
            "input_text": (
                "Khám tại bệnh viện\nNhiệt độ: 36.5 độ C. "
                "INR dưới ngưỡng điều trị 1.7."
            ),
            "entities": [
                {"text": "36.5 độ C", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "INR", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("Nhiệt độ", "TÊN_XÉT_NGHIỆM"),
                ("36.5 độ C", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("INR", "TÊN_XÉT_NGHIỆM"),
                ("dưới ngưỡng điều trị 1.7", "KẾT_QUẢ_XÉT_NGHIỆM"),
            ],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-named-measurement" in log for log in logs))

    def test_recovers_value_before_lab_name(self):
        record = {
            "input_text": "Đánh giá tại bệnh viện\nXét nghiệm ghi nhận 3.2 kali và 80% neutrophil.",
            "entities": [],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("3.2", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("kali", "TÊN_XÉT_NGHIỆM"),
                ("80%", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("neutrophil", "TÊN_XÉT_NGHIỆM"),
            ],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-value-first-lab" in log for log in logs))

    def test_recovers_value_then_parenthesized_lab_name(self):
        record = {
            "input_text": (
                "Đánh giá tại bệnh viện\nIon đồ: 4.1 (K+), 140 (Na+), 98 (Cl-). "
                "Kết quả khác: 8.4 mg/dL (phospho), 0.5 mmol/L (Mg++)."
            ),
            "entities": [],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("Ion đồ", "TÊN_XÉT_NGHIỆM"),
                ("4.1", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("K+", "TÊN_XÉT_NGHIỆM"),
                ("140", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("Na+", "TÊN_XÉT_NGHIỆM"),
                ("98", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("Cl-", "TÊN_XÉT_NGHIỆM"),
                ("8.4 mg/dL", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("phospho", "TÊN_XÉT_NGHIỆM"),
                ("0.5 mmol/L", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("Mg++", "TÊN_XÉT_NGHIỆM"),
            ],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-parenthesized-value-first-lab" in log for log in logs))

    def test_recovers_repeated_full_coordinated_symptom(self):
        phrase = "đau vùng hạ vị bên phải và hạ vị bên trái"
        record = {
            "input_text": f"Bệnh sử hiện tại\nLý do: {phrase}. Hiện tại vẫn {phrase}.",
            "entities": [
                {"text": "đau vùng hạ vị bên phải", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "hạ vị bên trái", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual([phrase, phrase], [entity["text"] for entity in cleaned["entities"]])
        self.assertTrue(any("autofix-full-symptom" in log for log in logs))

    def test_merges_split_xray_test_name(self):
        record = {
            "input_text": "Đánh giá tại bệnh viện\nChụp x-quang ngực không ghi nhận bất thường.",
            "entities": [
                {"text": "Chụp", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "x-quang ngực", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {
                    "text": "không ghi nhận bất thường",
                    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "assertions": [],
                },
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual("Chụp x-quang ngực", cleaned["entities"][0]["text"])
        self.assertTrue(
            any(
                "autofix-imaging-test-name" in log or "autofix-imaging-clause" in log
                for log in logs
            )
        )

    def test_removes_isolated_fragments_and_procedure_volume(self):
        record = {
            "input_text": (
                "Đánh giá tại bệnh viện\nCK: 58 U/L. Cấu trúc giảm âm và lan lên trên. "
                "Cơn khó thở kéo dài 20 giây. Hút 0.5cc dịch mủ."
            ),
            "entities": [
                {"text": "CK", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "58 U/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "âm", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "lên", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "giây", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "0.5", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(["CK", "58 U/L"], [entity["text"] for entity in cleaned["entities"]])
        self.assertTrue(any("reject-entity-fragment" in log for log in logs))
        self.assertTrue(any("reject-entity-procedure-value" in log for log in logs))

    def test_keeps_gold_boundary_for_qualified_crp_result(self):
        record = {
            "input_text": "Đánh giá tại bệnh viện\nCRP tăng cao 15.2 mg/L.",
            "entities": [
                {"text": "CRP", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {
                    "text": "tăng cao 15.2 mg/L",
                    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "assertions": [],
                },
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("CRP", "TÊN_XÉT_NGHIỆM"),
                ("15.2 mg/L", "KẾT_QUẢ_XÉT_NGHIỆM"),
            ],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-lab-gold-boundary" in log for log in logs))

    def test_recovers_second_single_value_lab_trend(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\nKali là 2.4. "
                "Lặp lại xét nghiệm cho thấy kali vẫn giảm xuống 2.2."
            ),
            "entities": [
                {"text": "Kali", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "2.4", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "kali", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertIn(
            ("2.2", "KẾT_QUẢ_XÉT_NGHIỆM"),
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-single-lab-trend" in log for log in logs))

    def test_drops_duplicate_lab_name_when_input_mentions_it_once(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\n"
                "Xét nghiệm: kali lần đầu 3.8 mmol/L, sau giảm còn 3.5 mmol/L."
            ),
            "entities": [
                {"text": "kali", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "3.8 mmol/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "kali", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "3.5 mmol/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            1,
            sum(
                entity["text"] == "kali" and entity["type"] == "TÊN_XÉT_NGHIỆM"
                for entity in cleaned["entities"]
            ),
        )
        self.assertTrue(any("autofix-excess-duplicate" in log for log in logs))

    def test_maps_repeated_lab_names_to_their_real_case(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\n"
                "Kali lần đầu 3.8 mmol/L. Sau 6 giờ kali giảm còn 3.5 mmol/L."
            ),
            "entities": [
                {"text": "kali", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "3.8 mmol/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "kali", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "3.5 mmol/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        lab_names = [
            entity["text"]
            for entity in cleaned["entities"]
            if entity["type"] == "TÊN_XÉT_NGHIỆM"
        ]
        self.assertEqual(["Kali", "kali"], lab_names)
        self.assertTrue(any("autofix-case-mismatch" in log for log in logs))

    def test_splits_repeated_qualitative_stress_tests_and_drops_unknown(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\nLý do: xét nghiệm gắng sức bất thường. "
                "Vị trí: Không rõ. Các sự kiện: xét nghiệm gắng sức bất thường. "
                "Điện tâm đồ bình thường tại phòng cấp cứu."
            ),
            "entities": [
                {
                    "text": "xét nghiệm gắng sức bất thường",
                    "type": "TÊN_XÉT_NGHIỆM",
                    "assertions": [],
                },
                {"text": "Không rõ", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        pairs = [(entity["text"], entity["type"]) for entity in cleaned["entities"]]
        self.assertEqual(2, pairs.count(("xét nghiệm gắng sức", "TÊN_XÉT_NGHIỆM")))
        self.assertEqual(2, pairs.count(("bất thường", "KẾT_QUẢ_XÉT_NGHIỆM")))
        self.assertIn(("Điện tâm đồ", "TÊN_XÉT_NGHIỆM"), pairs)
        self.assertIn(("bình thường", "KẾT_QUẢ_XÉT_NGHIỆM"), pairs)
        self.assertNotIn(("bình thường tại phòng cấp cứu", "KẾT_QUẢ_XÉT_NGHIỆM"), pairs)
        self.assertNotIn(("Không rõ", "KẾT_QUẢ_XÉT_NGHIỆM"), pairs)
        self.assertTrue(any("qualitative-test-clause" in log for log in logs))

    def test_consolidates_long_imaging_finding(self):
        finding = (
            "xuất huyết dưới nhện vùng trán phải, bầm dập nhu mô vùng trán phải, "
            "nghĩ nhiều đến nang màng nhện"
        )
        record = {
            "input_text": f"Bệnh sử hiện tại\nChụp cắt lớp vi tính sọ não cho hình ảnh {finding}.",
            "entities": [
                {
                    "text": "cắt lớp vi tính sọ não",
                    "type": "TÊN_XÉT_NGHIỆM",
                    "assertions": [],
                },
                {
                    "text": "xuất huyết dưới nhện vùng trán phải",
                    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "assertions": [],
                },
                {
                    "text": "nang màng nhện",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": [],
                },
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("Chụp cắt lớp vi tính sọ não", "TÊN_XÉT_NGHIỆM"),
                (finding, "KẾT_QUẢ_XÉT_NGHIỆM"),
            ],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-imaging-clause" in log for log in logs))

    def test_fixes_current_hypoxia_and_diagnosis_timing(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\nLý do nhập viện: ho, thiếu oxy. "
                "Khoảng 1 tháng trước nhập viện, bệnh nhân được chẩn đoán xuất huyết dưới nhện. "
                "Hôm nay đến ED vì suy tim sung huyết cấp."
            ),
            "entities": [
                {"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": []},
                {
                    "text": "thiếu oxy",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                },
                {
                    "text": "xuất huyết dưới nhện",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": [],
                },
                {
                    "text": "suy tim sung huyết cấp",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                },
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        by_text = {entity["text"]: entity for entity in cleaned["entities"]}
        self.assertEqual("TRIỆU_CHỨNG", by_text["thiếu oxy"]["type"])
        self.assertEqual([], by_text["thiếu oxy"]["assertions"])
        self.assertEqual(["isHistorical"], by_text["xuất huyết dưới nhện"]["assertions"])
        self.assertEqual([], by_text["suy tim sung huyết cấp"]["assertions"])
        self.assertTrue(any("autofix-diagnosis-past" in log for log in logs))
        self.assertTrue(any("autofix-diagnosis-current" in log for log in logs))

    def test_recovers_unpunctuated_oxygen_saturation_name(self):
        record = {
            "input_text": (
                "Đánh giá tại bệnh viện\nNhiệt độ 36.7 độ C, "
                "độ bão hòa oxy 100%, SpO2 90-92%. Vị trí: Không rõ."
            ),
            "entities": [
                {"text": "36.7 độ C", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "100%", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "90-92%", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "Không rõ", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        pairs = [(entity["text"], entity["type"]) for entity in cleaned["entities"]]
        self.assertIn(("Nhiệt độ", "TÊN_XÉT_NGHIỆM"), pairs)
        self.assertIn(("độ bão hòa oxy", "TÊN_XÉT_NGHIỆM"), pairs)
        self.assertIn(("SpO2", "TÊN_XÉT_NGHIỆM"), pairs)
        self.assertNotIn(("90", "KẾT_QUẢ_XÉT_NGHIỆM"), pairs)
        self.assertNotIn(("Không rõ", "KẾT_QUẢ_XÉT_NGHIỆM"), pairs)

    def test_rejects_procedure_units_and_bad_ef_unit(self):
        clean_record = {
            "input_text": (
                "Tiền sử bệnh\nTăng huyết áp, đái tháo đường type 2. "
                "Sau CABG bệnh nhân giảm 60 pound."
            ),
            "entities": [
                {
                    "text": "Tăng huyết áp",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                },
                {
                    "text": "đái tháo đường type 2",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                },
                {"text": "CABG", "type": "CHẨN_ĐOÁN", "assertions": ["isHistorical"]},
                {"text": "pound", "type": "THUỐC", "assertions": []},
            ],
        }

        status, cleaned, _logs = process_record(clean_record)
        self.assertEqual("keep", status)
        self.assertEqual(
            ["Tăng huyết áp", "đái tháo đường type 2"],
            [entity["text"] for entity in cleaned["entities"]],
        )

        bad_record = {
            "input_text": "Tiền sử bệnh\nSuy tim, hệ số tống máu 50 inch.",
            "entities": [
                {
                    "text": "Suy tim",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                }
            ],
        }
        status, rejected = process_record(bad_record)
        self.assertEqual("reject", status)
        self.assertEqual("medical_inconsistency_reject", rejected["stage"])

    def test_expands_valid_fever_span_and_rejects_invalid_temperature(self):
        valid_record = {
            "input_text": "Bệnh sử hiện tại\nBệnh nhân sốt cao 39°C và mệt mỏi.",
            "entities": [
                {"text": "sốt cao", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "mệt mỏi", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(valid_record)
        self.assertEqual("keep", status)
        self.assertEqual("sốt cao 39°C", cleaned["entities"][0]["text"])
        self.assertTrue(any("autofix-fever-full-span" in log for log in logs))

        implicit_celsius_record = {
            "input_text": "Bệnh sử hiện tại\nBệnh nhân sốt nhẹ 37.8 và mệt mỏi.",
            "entities": [
                {"text": "sốt nhẹ", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "mệt mỏi", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }
        status, cleaned, logs = process_record(implicit_celsius_record)
        self.assertEqual("keep", status)
        self.assertEqual("sốt nhẹ 37.8", cleaned["entities"][0]["text"])
        self.assertTrue(any("autofix-fever-full-span" in log for log in logs))

        duration_record = {
            "input_text": "Bệnh sử hiện tại\nBệnh nhân sốt 3 ngày và mệt mỏi.",
            "entities": [
                {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "mệt mỏi", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }
        status, _cleaned, _logs = process_record(duration_record)
        self.assertEqual("keep", status)

        invalid_record = {
            "input_text": "Bệnh sử hiện tại\nBệnh nhân sốt 90 và mệt mỏi.",
            "entities": [
                {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "mệt mỏi", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }
        status, rejected = process_record(invalid_record)
        self.assertEqual("reject", status)
        self.assertEqual("medical_inconsistency_reject", rejected["stage"])

    def test_drops_duration_number_mistyped_as_lab_result(self):
        record = {
            "input_text": (
                "Tiền sử bệnh\nBệnh nhân tăng huyết áp 5 năm và đái tháo đường type 2."
            ),
            "entities": [
                {
                    "text": "tăng huyết áp",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                },
                {"text": "5", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {
                    "text": "đái tháo đường type 2",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                },
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertNotIn("5", [entity["text"] for entity in cleaned["entities"]])
        self.assertTrue(any("tuổi/thời lượng" in log for log in logs))

    def test_drops_seconds_and_time_units_from_entities(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\nKhó thở và đánh trống ngực kéo dài 20 giây, "
                "tái diễn sau 30 phút."
            ),
            "entities": [
                {"text": "Khó thở", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "đánh trống ngực", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "20 giây", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "phút", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            ["Khó thở", "đánh trống ngực"],
            [entity["text"] for entity in cleaned["entities"]],
        )
        self.assertTrue(any("tuổi/thời lượng" in log for log in logs))
        self.assertTrue(any("mảnh từ vô nghĩa" in log for log in logs))

    def test_consolidates_negative_ecg_and_fundoscopy_findings(self):
        record = {
            "input_text": (
                "Đánh giá tại bệnh viện\n"
                "Điện tâm đồ không có dấu hiệu thiếu máu cơ tim. "
                "Soi đáy mắt phát hiện xuất huyết võng mạc."
            ),
            "entities": [
                {"text": "Điện tâm đồ", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {
                    "text": "thiếu máu cơ tim",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isNegated"],
                },
                {"text": "Soi đáy mắt", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {
                    "text": "xuất huyết võng mạc",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": [],
                },
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("Điện tâm đồ", "TÊN_XÉT_NGHIỆM"),
                ("không có dấu hiệu thiếu máu cơ tim", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("Soi đáy mắt", "TÊN_XÉT_NGHIỆM"),
                ("xuất huyết võng mạc", "KẾT_QUẢ_XÉT_NGHIỆM"),
            ],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertTrue(any("qualitative-test-clause" in log for log in logs))
        self.assertTrue(any("imaging-clause" in log for log in logs))

    def test_drops_concepts_used_only_as_treatment_goals(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\nBệnh nhân đang dùng thuốc để phòng ngừa huyết khối "
                "và dùng paracetamol để giảm đau. Hiện có mệt mỏi và khó thở."
            ),
            "entities": [
                {"text": "huyết khối", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "paracetamol", "type": "THUỐC", "assertions": []},
                {"text": "đau", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "mệt mỏi", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "khó thở", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }

        result = process_record(record)
        self.assertEqual("keep", result[0], result)
        status, cleaned, logs = result
        texts = [entity["text"] for entity in cleaned["entities"]]
        self.assertNotIn("huyết khối", texts)
        self.assertNotIn("đau", texts)
        self.assertIn("paracetamol", texts)
        self.assertTrue(any("reject-entity-treatment-goal" in log for log in logs))

    def test_expands_drug_span_through_route_and_frequency(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\nThuốc dùng tại nhà: Lactulose 15 ml uống hàng ngày để "
                "điều trị táo bón. Trước đó dùng Ciprofloxacin 500 mg x 2 lần/ngày trong "
                "7 ngày. Hiện bôi Hydrocortisone 1% bôi tại chỗ điều trị viêm da."
                " Có thể dùng Paracetamol 500 mg mỗi 6 giờ khi cần."
            ),
            "entities": [
                {"text": "Lactulose 15 ml", "type": "THUỐC", "assertions": []},
                {"text": "Ciprofloxacin 500 mg", "type": "THUỐC", "assertions": []},
                {"text": "Hydrocortisone 1%", "type": "THUỐC", "assertions": []},
                {"text": "Paracetamol 500 mg", "type": "THUỐC", "assertions": []},
            ],
        }

        status, cleaned, logs = process_record(record)

        self.assertEqual("keep", status)
        self.assertEqual(
            [
                "Lactulose 15 ml uống hàng ngày",
                "Ciprofloxacin 500 mg x 2 lần/ngày trong 7 ngày",
                "Hydrocortisone 1% bôi tại chỗ",
                "Paracetamol 500 mg mỗi 6 giờ khi cần",
            ],
            [entity["text"] for entity in cleaned["entities"]],
        )
        self.assertTrue(any("autofix-drug-regimen-span" in log for log in logs))

    def test_current_drug_does_not_inherit_previous_sentence_timing(self):
        record = {
            "input_text": (
                "Bệnh sử hiện tại\nTrước đó bệnh nhân dùng kem chống nắng không rõ nguồn gốc. "
                "Đang dùng Hydrocortisone 1% bôi tại chỗ 2 lần/ngày. Da vẫn ngứa."
            ),
            "entities": [
                {
                    "text": "Hydrocortisone 1% bôi tại chỗ 2 lần/ngày",
                    "type": "THUỐC",
                    "assertions": ["isHistorical"],
                },
                {"text": "ngứa", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
        }

        result = process_record(record)
        self.assertEqual("keep", result[0], result)
        status, cleaned, logs = result

        self.assertEqual("keep", status)
        self.assertEqual([], cleaned["entities"][0]["assertions"])
        self.assertTrue(any("autofix-drug-active-default" in log for log in logs))

    def test_removes_named_regimen_without_specific_drug(self):
        record = {
            "input_text": (
                "Chẩn đoán u lympho. Điều trị hiện tại bằng Rituximab 375 mg/m2 "
                "và CHOP phác đồ."
            ),
            "entities": [
                {"text": "u lympho", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "Rituximab 375 mg/m2", "type": "THUỐC", "assertions": []},
                {"text": "CHOP phác đồ", "type": "THUỐC", "assertions": []},
            ],
        }

        result = process_record(record)
        self.assertEqual("keep", result[0], result)
        status, cleaned, logs = result

        self.assertEqual("keep", status)
        self.assertEqual(
            ["u lympho", "Rituximab 375 mg/m2"],
            [e["text"] for e in cleaned["entities"]],
        )
        self.assertTrue(any("reject-entity-non-drug" in log for log in logs))

    def test_removes_combined_sex_and_age_entity(self):
        record = {
            "input_text": "BN nam 26t vào viện vì đau ngực. ECG bình thường.",
            "entities": [
                {"text": "nam 26t", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "ECG", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "bình thường", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
        }
        status, cleaned, logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertNotIn("nam 26t", [entity["text"] for entity in cleaned["entities"]])

    def test_orders_same_text_finding_before_later_diagnosis(self):
        record = {
            "input_text": (
                "Soi đáy mắt phát hiện xuất huyết võng mạc. "
                "Đo nhãn áp: 18 mmHg. CĐ: xuất huyết võng mạc."
            ),
            "entities": [
                {"text": "Soi đáy mắt", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "xuất huyết võng mạc", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "Đo nhãn áp", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "18 mmHg", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "xuất huyết võng mạc", "type": "CHẨN_ĐOÁN", "assertions": []},
            ],
        }
        status, cleaned, logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("Soi đáy mắt", "TÊN_XÉT_NGHIỆM"),
                ("xuất huyết võng mạc", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("Đo nhãn áp", "TÊN_XÉT_NGHIỆM"),
                ("18 mmHg", "KẾT_QUẢ_XÉT_NGHIỆM"),
                ("xuất huyết võng mạc", "CHẨN_ĐOÁN"),
            ],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )

    def test_short_entity_maps_to_later_independent_occurrence(self):
        record = {
            "input_text": (
                "Bác sĩ nghĩ đến thiếu vitamin A với đốm Bitot và khô kết mạc. "
                "Người bệnh sẽ làm xét nghiệm máu. Sau đó xác nhận thiếu vitamin A."
            ),
            "entities": [
                {
                    "text": "thiếu vitamin A với đốm Bitot và khô kết mạc",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": [],
                },
                {"text": "thiếu vitamin A", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "xét nghiệm máu", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
            ],
        }
        result = process_record(record)
        self.assertEqual("keep", result[0], result)
        self.assertEqual(
            [
                "thiếu vitamin A với đốm Bitot và khô kết mạc",
                "xét nghiệm máu",
                "thiếu vitamin A",
            ],
            [entity["text"] for entity in result[1]["entities"]],
        )

    def test_v4_metadata_can_disable_legacy_required_lab_pair(self):
        record = {
            "input_text": "Đánh giá tại bệnh viện\nBệnh nhân đau đầu và buồn nôn.",
            "entities": [
                {"text": "đau đầu", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "buồn nôn", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
            "_section_key_hint": "danh_gia",
            "_require_lab_pair": False,
        }
        result = process_record(record)
        self.assertEqual("keep", result[0], result)

    def test_current_treatment_heading_removes_historical_drug_assertion(self):
        record = {
            "input_text": (
                "Khám ghi nhận mảng hồng ban. Đ/trị: Hydrocortisone 1% bôi tại chỗ, "
                "cetirizine uống trước khi ngủ."
            ),
            "entities": [
                {"text": "mảng hồng ban", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "Hydrocortisone 1% bôi tại chỗ", "type": "THUỐC", "assertions": []},
                {"text": "cetirizine uống", "type": "THUỐC", "assertions": ["isHistorical"]},
            ],
        }
        status, cleaned, logs = process_record(record)
        self.assertEqual("keep", status)
        by_text = {entity["text"]: entity for entity in cleaned["entities"]}
        self.assertEqual([], by_text["cetirizine uống"]["assertions"])

    def test_adds_bone_marrow_aspiration_as_test_name_for_its_finding(self):
        record = {
            "input_text": (
                "Chẩn đoán u lympho. Chọc hút tủy xương ghi nhận thâm nhiễm 30% "
                "tế bào lympho ác tính."
            ),
            "entities": [
                {"text": "u lympho", "type": "CHẨN_ĐOÁN", "assertions": []},
                {
                    "text": "thâm nhiễm 30% tế bào lympho ác tính",
                    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "assertions": [],
                },
            ],
        }
        status, cleaned, logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertIn(
            ("Chọc hút tủy xương", "TÊN_XÉT_NGHIỆM"),
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )

    def test_v4_knowledge_disease_is_not_historical_or_test_name(self):
        record = {
            "input_text": (
                "Thiếu men G6PD là một bệnh di truyền. "
                "Thiếu men G6PD có thể gây vàng da ở một số trường hợp."
            ),
            "entities": [
                {
                    "text": "Thiếu men G6PD",
                    "type": "TÊN_XÉT_NGHIỆM",
                    "assertions": ["isHistorical"],
                },
                {
                    "text": "Thiếu men G6PD",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                },
                {"text": "vàng da", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
            "_knowledge_context": True,
            "_require_lab_pair": False,
        }
        status, cleaned, logs = process_record(record)
        self.assertEqual("keep", status)
        diseases = [e for e in cleaned["entities"] if e["text"].lower() == "thiếu men g6pd"]
        self.assertEqual(2, len(diseases))
        self.assertTrue(all(e["type"] == "CHẨN_ĐOÁN" for e in diseases))
        self.assertTrue(all(e["assertions"] == [] for e in diseases))

    def test_drops_generic_concepts_and_non_drug_exposures(self):
        record = {
            "input_text": (
                "Amyloid là một protein. Người bệnh đau đầu và vàng da. "
                "Cần tránh băng phiến và long não."
            ),
            "entities": [
                {"text": "Amyloid", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "protein", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "đau đầu", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "vàng da", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "băng phiến", "type": "THUỐC", "assertions": []},
                {"text": "long não", "type": "THUỐC", "assertions": []},
            ],
            "_knowledge_context": True,
            "_require_lab_pair": False,
        }
        status, cleaned, logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertEqual(
            ["đau đầu", "vàng da"],
            [entity["text"] for entity in cleaned["entities"]],
        )

    def test_repairs_bai_nao_fragments_to_full_span(self):
        record = {
            "input_text": "Biến chứng có thể gồm Bại não và chậm phát triển trí tuệ.",
            "entities": [
                {"text": "Bại", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "não", "type": "CHẨN_ĐOÁN", "assertions": []},
                {
                    "text": "chậm phát triển trí tuệ",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": [],
                },
            ],
            "_knowledge_context": True,
            "_require_lab_pair": False,
        }
        status, cleaned, logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertIn(
            ("Bại não", "CHẨN_ĐOÁN"),
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )
        self.assertNotIn("Bại", [entity["text"] for entity in cleaned["entities"]])
        self.assertNotIn("não", [entity["text"] for entity in cleaned["entities"]])

    def test_keeps_separate_entities_around_missing_sentence_space(self):
        record = {
            "input_text": "Người bệnh đau ngực.Hồi hộp tăng dần.",
            "entities": [
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "Hồi hộp", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
            "_require_lab_pair": False,
        }
        status, cleaned, logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertEqual(
            ["đau ngực", "Hồi hộp"],
            [entity["text"] for entity in cleaned["entities"]],
        )

    def test_rejects_entity_that_crosses_missing_sentence_space(self):
        record = {
            "input_text": "Người bệnh đau ngực.Hồi hộp tăng dần và khó thở.",
            "entities": [
                {
                    "text": "đau ngực.Hồi hộp",
                    "type": "TRIỆU_CHỨNG",
                    "assertions": [],
                },
                {"text": "khó thở", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
            "_require_lab_pair": False,
        }
        status, rejection = process_record(record)
        self.assertEqual("reject", status)
        self.assertEqual("entity_boundary_reject", rejection["stage"])

    def test_repairs_spo2_name_value_and_drops_fragments(self):
        record = {
            "input_text": "Độ bão hòa oxy (SPO2) từ 88-92 % khi thở khí trời.",
            "entities": [
                {"text": "Độ", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "từ", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
                {"text": "88-92 % khi thở khí trời", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
            ],
            "_require_lab_pair": False,
        }
        status, cleaned, _logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertEqual(
            [
                ("Độ bão hòa oxy (SPO2)", "TÊN_XÉT_NGHIỆM"),
                ("88-92 %", "KẾT_QUẢ_XÉT_NGHIỆM"),
            ],
            [(entity["text"], entity["type"]) for entity in cleaned["entities"]],
        )

    def test_current_episode_exam_is_not_historical(self):
        record = {
            "input_text": (
                "Hai tuần trước khi nhập viện bệnh nhân ho và được bác sĩ gia đình "
                "khám vì các triệu chứng nhiễm trùng đường hô hấp trên."
            ),
            "entities": [
                {"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": []},
                {
                    "text": "nhiễm trùng đường hô hấp trên",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                },
            ],
            "_require_lab_pair": False,
        }
        status, cleaned, _logs = process_record(record)
        self.assertEqual("keep", status)
        diagnosis = next(e for e in cleaned["entities"] if e["type"] == "CHẨN_ĐOÁN")
        self.assertEqual([], diagnosis["assertions"])

    def test_diagnosis_duration_and_home_drug_are_historical(self):
        record = {
            "input_text": "TS THA 10năm,đang dùng amlodipine 5 mg.",
            "entities": [
                {"text": "THA", "type": "CHẨN_ĐOÁN", "assertions": []},
                {"text": "amlodipine 5 mg", "type": "THUỐC", "assertions": []},
            ],
            "_require_lab_pair": False,
        }
        status, cleaned, _logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertTrue(all(e["assertions"] == ["isHistorical"] for e in cleaned["entities"]))

    def test_repairs_common_compact_lab_pairs(self):
        record = {
            "input_text": "WBC:12.5;CRP:64mg/L",
            "entities": [
                {"text": "WBC", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
                {"text": "CRP", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
            ],
            "_require_lab_pair": False,
        }
        status, cleaned, _logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertEqual(
            ["WBC", "12.5", "CRP", "64mg/L"],
            [entity["text"] for entity in cleaned["entities"]],
        )

    def test_rejects_entity_crossing_glued_comma_boundary(self):
        record = {
            "input_text": "BN mệt 3ngày,ho khan.",
            "entities": [
                {"text": "3ngày,ho", "type": "TRIỆU_CHỨNG", "assertions": []},
                {"text": "khan", "type": "TRIỆU_CHỨNG", "assertions": []},
            ],
            "_require_lab_pair": False,
        }
        status, rejection = process_record(record)
        self.assertEqual("reject", status)
        self.assertEqual("entity_boundary_reject", rejection["stage"])

    def test_expands_drug_dose_without_name(self):
        record = {
            "input_text": "Đã xử trí ceftriaxone 1 gram. Sau đó uống Tylenol 1 gram.",
            "entities": [
                {"text": "ceftriaxone 1 gram", "type": "THUỐC", "assertions": []},
                {"text": "1 gram", "type": "THUỐC", "assertions": []},
            ],
            "_require_lab_pair": False,
        }
        status, cleaned, _logs = process_record(record)
        self.assertEqual("keep", status)
        self.assertEqual(
            ["ceftriaxone 1 gram", "Tylenol 1 gram"],
            [entity["text"] for entity in cleaned["entities"]],
        )


if __name__ == "__main__":
    unittest.main()
