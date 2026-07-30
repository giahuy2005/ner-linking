import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linking.icd10 import build_icd10_faiss_index as build_module
from linking.icd10.build_icd10_faiss_index import load_embedding_terms, write_metadata
from linking.icd10.icd10_linker import (
    Icd10Linker,
    _exact_alias_result,
    _finalize_term_results,
    aggregate_term_results,
)
from linking.sapbert_encoder import clean_query_text, l2_normalize, resolve_model_source


class Icd10LinkingTests(unittest.TestCase):
    def test_l2_normalize_returns_float32_unit_rows(self):
        vectors = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64))
        self.assertEqual(np.float32, vectors.dtype)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])

    def test_l2_normalize_rejects_zero_vector(self):
        with self.assertRaises(ValueError):
            l2_normalize(np.array([[0.0, 0.0]], dtype=np.float32))

    def test_clean_query_text_normalizes_whitespace(self):
        self.assertEqual("Viêm tai giữa", clean_query_text("  Viêm\n tai   giữa "))
        with self.assertRaises(ValueError):
            clean_query_text("  \n ")

    def test_resolves_stale_windows_sapbert_path_to_current_project(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            local_model = project_root / "models" / "sapbert"
            local_model.mkdir(parents=True)

            source, is_local = resolve_model_source(
                r"Z:\build-machine\viettel_ai_ner\models\sapbert",
                project_root=project_root,
            )

            self.assertTrue(is_local)
            self.assertEqual(local_model.resolve(), Path(source))

    def test_missing_foreign_sapbert_path_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "SAPBERT_MODEL_ID"):
                resolve_model_source(
                    r"D:\old-machine\models\sapbert",
                    project_root=directory,
                )

    def test_sapbert_environment_override_has_highest_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            local_model = Path(directory) / "custom-sapbert"
            local_model.mkdir()
            with patch.dict("os.environ", {"SAPBERT_MODEL_ID": str(local_model)}):
                source, is_local = resolve_model_source(
                    "models/sapbert",
                    project_root=Path(directory) / "different-project",
                )
            self.assertTrue(is_local)
            self.assertEqual(local_model.resolve(), Path(source))

    def test_aggregate_uses_max_score_per_code(self):
        term_results = [
            {
                "score": 0.92,
                "term_id": "H65.9|vi|inclusion|0",
                "code": "H65.9",
                "text": "Viêm tai giữa xuất tiết",
                "language": "vi",
                "term_type": "inclusion",
            },
            {
                "score": 0.94,
                "term_id": "H65.9|vi|inclusion|1",
                "code": "H65.9",
                "text": "Viêm tai giữa tiết dịch",
                "language": "vi",
                "term_type": "inclusion",
            },
            {
                "score": 0.88,
                "term_id": "H65.0|vi|preferred|0",
                "code": "H65.0",
                "text": "Viêm tai giữa thanh dịch",
                "language": "vi",
                "term_type": "preferred",
            },
        ]

        results = aggregate_term_results(term_results, top_k_codes=10)

        self.assertEqual(["H65.9", "H65.0"], [row["code"] for row in results])
        self.assertEqual(0.94, results[0]["score"])
        self.assertEqual("Viêm tai giữa tiết dịch", results[0]["matched_term"])

    def test_aggregate_honors_threshold_and_limit(self):
        rows = [
            {
                "score": 0.9,
                "term_id": "A|vi|preferred|0",
                "code": "A",
                "text": "a",
                "language": "vi",
                "term_type": "preferred",
            },
            {
                "score": 0.7,
                "term_id": "B|vi|preferred|0",
                "code": "B",
                "text": "b",
                "language": "vi",
                "term_type": "preferred",
            },
        ]
        self.assertEqual(
            ["A"],
            [
                row["code"]
                for row in aggregate_term_results(rows, top_k_codes=1, min_score=0.8)
            ],
        )

    def test_exact_vietnamese_aliases_prevent_unrelated_icd_candidates(self):
        expected = {
            "ăng huyết áp": "I10",
            "ổ loét trong bao tử": "K25",
            "viêm bao tử": "K29",
            "Bệnh đa xơ cứng": "G35",
            "thiếu men G6PD": "D55.0",
        }
        for mention, code in expected.items():
            with self.subTest(mention=mention):
                result = _exact_alias_result(mention)
                self.assertIsNotNone(result)
                self.assertEqual([code], [row["code"] for row in result])

    def test_icd_finalize_filters_threshold_chapter_and_caps_at_two(self):
        rows = [
            {"score": 0.93, "term_id": "K25|vi|preferred|0", "code": "K25",
             "text": "Loét dạ dày", "language": "vi", "term_type": "preferred"},
            {"score": 0.90, "term_id": "C90|vi|preferred|0", "code": "C90.0",
             "text": "Đa u tủy", "language": "vi", "term_type": "preferred"},
            {"score": 0.88, "term_id": "K26|vi|preferred|0", "code": "K26",
             "text": "Loét tá tràng", "language": "vi", "term_type": "preferred"},
            {"score": 0.80, "term_id": "K27|vi|preferred|0", "code": "K27",
             "text": "Loét tiêu hóa", "language": "vi", "term_type": "preferred"},
            {"score": 0.40, "term_id": "K28|vi|preferred|0", "code": "K28",
             "text": "Loét hỗng tràng", "language": "vi", "term_type": "preferred"},
        ]

        result = _finalize_term_results(
            "ổ loét trong bao tử", rows, top_k_codes=10, min_score=0.55,
        )

        self.assertEqual(["K25", "K26"], [row["code"] for row in result])

    def test_term_loading_and_metadata_preserve_vector_order(self):
        terms = [
            {
                "term_id": "H65.9|vi|preferred|0",
                "code": "H65.9",
                "text": "Viêm tai giữa không mủ, không xác định",
                "language": "vi",
                "term_type": "preferred",
            },
            {
                "term_id": "H65.9|en|preferred|0",
                "code": "H65.9",
                "text": "Nonsuppurative otitis media, unspecified",
                "language": "en",
                "term_type": "preferred",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "terms.jsonl"
            metadata = Path(directory) / "metadata.jsonl"
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in terms),
                encoding="utf-8",
            )

            loaded = load_embedding_terms(source)
            write_metadata(metadata, loaded)
            rows = Icd10Linker._load_metadata(metadata)

        self.assertEqual([0, 1], [row["vector_id"] for row in rows])
        self.assertEqual([term["term_id"] for term in terms], [row["term_id"] for row in rows])

    def test_builder_writes_all_four_consistent_artifacts(self):
        terms = [
            {
                "term_id": "A00|en|preferred|0",
                "code": "A00",
                "text": "Cholera",
                "language": "en",
                "term_type": "preferred",
            },
            {
                "term_id": "A00|vi|preferred|0",
                "code": "A00",
                "text": "Bệnh tả",
                "language": "vi",
                "term_type": "preferred",
            },
        ]

        class FakeEncoder:
            dimension = 2
            resolved_revision = "test-revision"

            def __init__(self, *_args, **_kwargs):
                pass

            def encode(self, texts, **_kwargs):
                self.assert_texts = texts
                return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        class FakeIndexFlatIP:
            def __init__(self, dimension):
                self.d = dimension
                self.ntotal = 0
                self.vectors = None

            def add(self, vectors):
                self.vectors = vectors
                self.ntotal = len(vectors)

        def fake_write_index(index, path):
            Path(path).write_bytes(f"d={index.d};n={index.ntotal}".encode())

        fake_faiss = SimpleNamespace(IndexFlatIP=FakeIndexFlatIP, write_index=fake_write_index)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "terms.jsonl"
            output_dir = root / "index"
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in terms),
                encoding="utf-8",
            )
            args = Namespace(
                input=source,
                output_dir=output_dir,
                model="fake-sapbert",
                revision=None,
                device="cpu",
                max_length=64,
                pooling="cls",
                batch_size=2,
                no_progress=True,
            )

            with patch.dict(sys.modules, {"faiss": fake_faiss}), patch.object(
                build_module, "SapBertEncoder", FakeEncoder
            ):
                config = build_module.build_index(args)

            expected = {
                "icd10_embeddings.npy",
                "icd10_faiss.index",
                "icd10_metadata.jsonl",
                "icd10_index_config.json",
            }
            self.assertEqual(expected, {path.name for path in output_dir.iterdir()})
            self.assertEqual((2, 2), np.load(output_dir / "icd10_embeddings.npy").shape)
            self.assertEqual(2, config["embedding"]["count"])
            self.assertEqual("test-revision", config["model"]["revision"])
            metadata = Icd10Linker._load_metadata(output_dir / "icd10_metadata.jsonl")
            self.assertEqual([0, 1], [row["vector_id"] for row in metadata])


if __name__ == "__main__":
    unittest.main()
