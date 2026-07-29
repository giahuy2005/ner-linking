import unittest

from src.preprocessing.json_to_bio import JSONToBioConverter


class JSONToBioBoundaryTests(unittest.TestCase):
    def test_ignores_spurious_leading_underscore_from_rdr(self):
        class FakeRdr:
            @staticmethod
            def tokenize(_text):
                return [["_Thống", "kinh"]]

        converter = JSONToBioConverter.__new__(JSONToBioConverter)
        converter.rdr = FakeRdr()
        tokens, offsets = converter.segment_with_offsets("Thống kinh")
        self.assertEqual(["Thống", "kinh"], tokens)
        self.assertEqual([(0, 5), (6, 10)], offsets)

    def test_keeps_literal_underscore_token_mappable(self):
        class FakeRdr:
            @staticmethod
            def tokenize(_text):
                return [["_"]]

        converter = JSONToBioConverter.__new__(JSONToBioConverter)
        converter.rdr = FakeRdr()
        tokens, offsets = converter.segment_with_offsets("_")
        self.assertEqual(["_"], tokens)
        self.assertEqual([(0, 1)], offsets)

    def test_splits_stuck_sentence_punctuation_before_bio(self):
        text = "tôi bị đau ngực.Hồi hộp và khó thở"
        # Mô phỏng đúng dạng token RDR gây lỗi mà không cần khởi động JVM trong test.
        tokens = ["tôi", "bị", "đau", "ngực.Hồi", "hộp", "và", "khó", "thở"]
        offsets = [(0, 3), (4, 6), (7, 10), (11, 19), (20, 23), (24, 26), (27, 30), (31, 34)]

        tokens, offsets = JSONToBioConverter._split_stuck_sentence_punctuation(
            text, tokens, offsets
        )
        self.assertEqual(
            ["tôi", "bị", "đau", "ngực", ".", "Hồi", "hộp", "và", "khó", "thở"],
            tokens,
        )

        spans = [
            {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "char_start": 7, "char_end": 15},
            {"text": "Hồi hộp", "type": "TRIỆU_CHỨNG", "char_start": 16, "char_end": 23},
            {"text": "khó thở", "type": "TRIỆU_CHỨNG", "char_start": 27, "char_end": 34},
        ]
        self.assertEqual(
            [
                "O", "O", "B-TRIỆU_CHỨNG", "I-TRIỆU_CHỨNG", "O",
                "B-TRIỆU_CHỨNG", "I-TRIỆU_CHỨNG", "O",
                "B-TRIỆU_CHỨNG", "I-TRIỆU_CHỨNG",
            ],
            JSONToBioConverter.build_bio_tags(tokens, offsets, spans),
        )

    def test_rejects_unrepresentable_shared_token_instead_of_overwriting(self):
        with self.assertRaisesRegex(ValueError, "overlap nhiều entity"):
            JSONToBioConverter.build_bio_tags(
                ["ngựcHồi"],
                [(0, 8)],
                [
                    {"text": "ngực", "type": "TRIỆU_CHỨNG", "char_start": 0, "char_end": 4},
                    {"text": "Hồi", "type": "TRIỆU_CHỨNG", "char_start": 4, "char_end": 8},
                ],
            )

    def test_does_not_split_decimal_numbers(self):
        text = "CRP 5.2 mg/L, sốt 37.8 độ C"
        tokens = ["CRP", "5.2", "mg", "/", "L", ",", "sốt", "37.8", "độ", "C"]
        offsets = [(0, 3), (4, 7), (8, 10), (10, 11), (11, 12), (12, 13),
                   (14, 17), (18, 22), (23, 25), (26, 27)]
        actual_tokens, actual_offsets = JSONToBioConverter._split_stuck_sentence_punctuation(
            text, tokens, offsets
        )
        self.assertEqual(tokens, actual_tokens)
        self.assertEqual(offsets, actual_offsets)

    def test_splits_dirty_lab_separators_but_keeps_decimal(self):
        text = "WBC:15.2;CRP:64mg/L"
        tokens = [text]
        offsets = [(0, len(text))]
        actual_tokens, actual_offsets = JSONToBioConverter._split_stuck_sentence_punctuation(
            text, tokens, offsets
        )
        self.assertEqual(
            ["WBC", ":", "15.2", ";", "CRP", ":", "64mg/L"],
            actual_tokens,
        )
        self.assertEqual((4, 8), actual_offsets[2])

    def test_explicit_spans_preserve_repeated_occurrence_identity(self):
        text = "Không sốt. Hôm qua từng sốt."
        entities = [
            {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"]},
            {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": ["isHistorical"]},
        ]
        spans = [
            {"char_start": 6, "char_end": 9, "text": "sốt", "type": "TRIỆU_CHỨNG"},
            {"char_start": 24, "char_end": 27, "text": "sốt", "type": "TRIỆU_CHỨNG"},
        ]

        resolved, skipped = JSONToBioConverter.resolve_entity_char_spans(text, entities, spans)

        self.assertEqual([], skipped)
        self.assertEqual([(6, 9), (24, 27)], [
            (entity["char_start"], entity["char_end"]) for entity in resolved
        ])

    def test_explicit_spans_reject_stale_offsets(self):
        with self.assertRaisesRegex(ValueError, "không khớp text"):
            JSONToBioConverter.resolve_entity_char_spans(
                "Không sốt",
                [{"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": []}],
                [{"char_start": 0, "char_end": 3}],
            )


if __name__ == "__main__":
    unittest.main()
