import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing.rxnorm.build_rxnorm_corpus import (
    DEFAULT_OUTPUT_TTYS,
    CANDIDATE_TTYS,
    PRODUCT_TTYS,
    candidate_priority,
    deduplicated_embedding_rows,
    enrich_concept,
    lexical_text,
    pack_item_quantity,
    read_concepts,
    read_relations,
)
from linking.rxnorm.build_rxnorm_faiss_indexes import load_clean_by_rxcui, load_terms, metadata_row


class RxNormPipelineTests(unittest.TestCase):
    def test_lexical_normalization_preserves_drug_information(self):
        self.assertEqual("amlodipine 10 MG oral tablet", lexical_text("  Amlodipine  10 mg Oral Tablet "))

    def test_rrf_relationship_is_semantically_second_to_first(self):
        concepts = {
            "308135": {"tty": "SCD"},
            "329526": {"tty": "SCDC"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "RXNREL.RRF"
            output = root / "relations.jsonl"
            source.write_text(
                "329526||CUI|RO|308135||CUI|consists_of|1||RXNORM|||||4096|\n",
                encoding="utf-8",
            )
            adjacency, stats = read_relations(source, concepts, output)
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual([("consists_of", "329526")], adjacency["308135"])
        self.assertEqual("308135", row["source_rxcui"])
        self.assertEqual("329526", row["target_rxcui"])
        self.assertEqual(1, stats["relations_written"])

    def test_gold_granularity_allows_product_and_support_candidates(self):
        self.assertEqual(CANDIDATE_TTYS, DEFAULT_OUTPUT_TTYS)
        self.assertIn("IN", DEFAULT_OUTPUT_TTYS)
        self.assertIn("SCD", DEFAULT_OUTPUT_TTYS)
        self.assertEqual(0, candidate_priority("SCD", "product"))
        self.assertEqual(1, candidate_priority("GPCK", "product"))
        self.assertEqual(2, candidate_priority("IN", "support"))
        self.assertEqual(3, candidate_priority("SCD", "historical"))

    def test_enrichment_uses_outgoing_semantic_edges(self):
        base = {
            "prescribable_name": None, "names": [], "_active": True,
            "_prescribable": True, "_atoms": [{"suppress": "N"}],
        }
        concepts = {
            "1": {**base, "rxcui": "1", "tty": "SCD", "canonical_name": "drug"},
            "2": {**base, "rxcui": "2", "tty": "SCDC", "canonical_name": "drug 10 MG"},
            "3": {**base, "rxcui": "3", "tty": "IN", "canonical_name": "drug"},
            "4": {**base, "rxcui": "4", "tty": "DF", "canonical_name": "Oral Tablet"},
        }
        adjacency = {"1": [("consists_of", "2"), ("has_dose_form", "4")], "2": [("has_ingredient", "3")]}
        row = enrich_concept(concepts["1"], concepts, adjacency, {"2": {"RXN_STRENGTH": "10 MG"}}, {}, DEFAULT_OUTPUT_TTYS)
        self.assertEqual("2", row["clinical_components"][0]["component_rxcui"])
        self.assertEqual("3", row["clinical_components"][0]["ingredient"]["rxcui"])
        self.assertEqual("4", row["dose_forms"][0]["rxcui"])

    def test_support_component_keeps_metadata_and_can_be_output_when_underspecified(self):
        base = {
            "prescribable_name": None, "names": [], "_active": True,
            "_prescribable": True, "_atoms": [{"suppress": "N"}],
        }
        concepts = {
            "2": {**base, "rxcui": "2", "tty": "SCDC", "canonical_name": "drug 10 MG"},
            "3": {**base, "rxcui": "3", "tty": "IN", "canonical_name": "drug"},
        }
        row = enrich_concept(
            concepts["2"], concepts, {"2": [("has_ingredient", "3")]},
            {"2": {"RXN_STRENGTH": "10 MG"}}, {}, DEFAULT_OUTPUT_TTYS,
        )
        self.assertEqual("3", row["clinical_components"][0]["ingredient"]["rxcui"])
        self.assertEqual("10 MG", row["clinical_components"][0]["strength"]["display"])
        self.assertTrue(row["retrieval"]["output_eligible"])
        self.assertEqual(2, row["retrieval"]["candidate_priority"])

    def test_branded_support_inherits_generic_ingredient_and_strength(self):
        base = {
            "prescribable_name": None, "names": [], "_active": True,
            "_prescribable": True, "_atoms": [{"suppress": "N"}],
        }
        concepts = {
            "1": {**base, "rxcui": "1", "tty": "SBDC", "canonical_name": "drug 10 MG [Brand]"},
            "2": {**base, "rxcui": "2", "tty": "SCDC", "canonical_name": "drug 10 MG"},
            "3": {**base, "rxcui": "3", "tty": "IN", "canonical_name": "drug"},
            "4": {**base, "rxcui": "4", "tty": "BN", "canonical_name": "Brand"},
        }
        adjacency = {
            "1": [("has_ingredient", "4"), ("tradename_of", "2")],
            "2": [("has_ingredient", "3")],
        }
        row = enrich_concept(
            concepts["1"], concepts, adjacency,
            {"2": {"RXN_STRENGTH": "10 MG"}}, {}, DEFAULT_OUTPUT_TTYS,
        )
        component = row["clinical_components"][0]
        self.assertEqual("3", component["ingredient"]["rxcui"])
        self.assertEqual("10 MG", component["strength"]["display"])
        self.assertEqual("4", row["brand"]["rxcui"])

    def test_pack_item_quantity_is_parsed_only_from_matching_item(self):
        name = "{24 (drug A Oral Tablet) / 4 (inert Oral Tablet) } Pack"
        self.assertEqual(24, pack_item_quantity(name, "drug A Oral Tablet"))
        self.assertEqual(4, pack_item_quantity(name, "inert Oral Tablet"))
        self.assertIsNone(pack_item_quantity(name, "missing product"))

    def test_embedding_dedup_prefers_active_over_historical_for_same_rxcui_text(self):
        clean = [{
            "rxcui": "1", "tty": "SCD",
            "names": [{"text": "Drug 10 MG", "normalized_text": "drug 10 MG", "name_type": "canonical", "source_tty": "SCD"}],
            "retrieval": {"index_tier": "product", "output_eligible": True, "candidate_priority": 0},
            "status": {"active": True},
        }]
        history = [{
            "old_rxcui": "1", "archived_names": [{"text": "Drug 10 MG", "tty": "SCD"}],
            "current_rxcuis": [],
        }]
        rows = list(deduplicated_embedding_rows(clean, history, DEFAULT_OUTPUT_TTYS))
        self.assertEqual(1, len(rows))
        self.assertEqual("product", rows[0]["index_tier"])

    def test_index_loader_keeps_tiers_separate(self):
        rows = []
        for tier in ("product", "support", "historical"):
            rows.append({
                "term_id": f"1|{tier}|0", "rxcui": "1", "text": tier,
                "embedding_text": tier, "term_type": "canonical", "source_tty": "SCD",
                "concept_tty": "SCD", "index_tier": tier,
                "output_eligible": True,
                "candidate_priority": 3 if tier == "historical" else 0,
                "active": tier != "historical",
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terms.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            loaded = load_terms(path, {"product", "support"})
        self.assertEqual({"product", "support"}, set(loaded))
        metadata = metadata_row(0, rows[0])
        self.assertEqual("1", metadata["rxcui"])
        self.assertNotIn("ingredients", metadata)
        self.assertNotIn("strengths", metadata)

    def test_clean_lookup_can_load_only_retrieved_rxcuis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.jsonl"
            path.write_text(
                json.dumps({"rxcui": "1", "clinical_components": ["a"]}) + "\n"
                + json.dumps({"rxcui": "2", "clinical_components": ["b"]}) + "\n",
                encoding="utf-8",
            )
            lookup = load_clean_by_rxcui(path, {"2"})
        self.assertEqual({"2"}, set(lookup))


if __name__ == "__main__":
    unittest.main()
