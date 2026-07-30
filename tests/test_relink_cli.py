import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from src.inference.relink_cli import run


class RelinkCliTests(unittest.TestCase):
    def test_relink_revalidates_raw_boundaries_and_discards_old_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            entities_dir = root / "old"
            output_dir = root / "new"
            input_dir.mkdir()
            entities_dir.mkdir()
            raw = "Khó thở do bệnh tim; dùng 1 gram."
            (input_dir / "1.txt").write_text(raw, encoding="utf-8")
            symptom_start = raw.index("Khó thở")
            diagnosis_start = raw.index("bệnh tim")
            dose_start = raw.index("1 gram")
            saved = [
                {
                    "text": "Khó th",
                    "type": "TRIỆU_CHỨNG",
                    "assertions": [],
                    "position": [symptom_start, symptom_start + len("Khó th")],
                },
                {
                    "text": "bệnh tim",
                    "type": "CHẨN_ĐOÁN",
                    "candidates": ["OLD"],
                    "assertions": [],
                    "position": [diagnosis_start, diagnosis_start + len("bệnh tim")],
                },
                {
                    "text": "1 gram",
                    "type": "THUỐC",
                    "candidates": ["WRONG"],
                    "assertions": [],
                    "position": [dose_start, dose_start + len("1 gram")],
                },
            ]
            (entities_dir / "1.json").write_text(
                json.dumps(saved, ensure_ascii=False), encoding="utf-8"
            )
            args = Namespace(
                input_dir=input_dir,
                entities_dir=entities_dir,
                output_dir=output_dir,
                with_rxnorm=False,
                with_icd10=False,
                with_llm_selector=False,
            )

            stats = run(args)
            output = json.loads((output_dir / "1.json").read_text(encoding="utf-8"))

            self.assertEqual(1, stats["records"])
            self.assertEqual(["Khó thở", "bệnh tim"], [item["text"] for item in output])
            self.assertEqual([], output[1]["candidates"])
            self.assertNotIn("candidates", output[0])

    def test_relink_refuses_to_overwrite_source_entities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(
                input_dir=root,
                entities_dir=root,
                output_dir=root,
                with_rxnorm=False,
                with_icd10=False,
                with_llm_selector=False,
            )
            with self.assertRaisesRegex(ValueError, "khác"):
                run(args)


if __name__ == "__main__":
    unittest.main()
