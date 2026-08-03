import sys
import hashlib
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.ner.candidates import CandidateEvidence, build_missing_proposals, build_review_regions
from src.inference.schemas import NerEntity, normalize_entity_schema
from src.inference.selection.candidate_selector import select_candidates_many
from src.linking.icd10.icd10_linker import _query_variants, aggregate_term_results
from src.linking.rxnorm.parser import parse_drug_mention
from src.linking.rxnorm.reranker import RxNormRuleReranker
from src.linking.rxnorm.schemas import RxNormCandidate
from src.llm.response_schemas import CandidateSelection


class _BatchLlm:
    def __init__(self, response): self.response, self.calls, self.batch_size = response, 0, None
    def generate_batch(self, prompts, batch_size=4):
        self.calls += 1; self.batch_size = batch_size
        return [self.response for _ in prompts]


class V3GuardTests(unittest.TestCase):
    def test_regions_are_bounded_and_small(self):
        text = " ".join(f"word{i}" for i in range(200))
        catalog = []
        cursor = 0
        for index in range(12):
            token = f"word{index}"
            start = text.index(token, cursor); cursor = start + len(token)
            catalog.append(CandidateEvidence(
                f"c{index}", token, "CHẨN_ĐOÁN", (start, cursor),
                sources=["span_head"], scores={"span_head": .4}, pre_llm_selected=True,
            ))
        regions = build_review_regions("r", text, catalog)
        self.assertTrue(regions)
        self.assertTrue(all(len(region.candidate_ids) <= 6 for region in regions))
        self.assertTrue(all(len(region.context) <= 900 for region in regions))

    def test_repeat_proposal_does_not_match_substring_or_one_character(self):
        text = "HA cao; HAX; n n; HA"
        seed = CandidateEvidence("c", "HA", "CHẨN_ĐOÁN", (0, 2), scores={"crf": .99})
        proposals = build_missing_proposals("r", text, [seed])
        self.assertEqual([(18, 20)], [item.position for item in proposals])
        one = CandidateEvidence("n", "n", "CHẨN_ĐOÁN", (12, 13), scores={"crf": .99})
        self.assertEqual([], build_missing_proposals("r", text, [one]))

    def test_assertions_removed_after_retype_to_lab_result(self):
        entity = NerEntity("10 mg/L", "KẾT_QUẢ_XÉT_NGHIỆM", ["isNegated"], (0, 7))
        self.assertEqual([], normalize_entity_schema(entity).assertions)

    def test_rxnorm_parser_route_range_quantity_and_liquid_hint(self):
        parsed = parse_drug_mention("acetaminophen 325-650 mg po q6h:prn")
        self.assertEqual("range", parsed.strength_role)
        self.assertEqual("PO", parsed.route)
        self.assertEqual(6, parsed.interval_hours)
        liquid = parse_drug_mention("guaifenesin ml po q6h:prn")
        self.assertEqual("guaifenesin", liquid.ingredient_core)
        self.assertTrue(any(item["source"] == "liquid_form_hint" for item in liquid.query_variants))

    def test_decimal_strength_and_range(self):
        reranker = RxNormRuleReranker()
        candidate = RxNormCandidate("1", "SCD", "product", "x", structured={
            "ingredients": ["x"], "strengths": ["0.5 MG"], "dose_forms": [],
        })
        parsed = parse_drug_mention("x 0.500 mg")
        self.assertEqual("exact", reranker.compare_strength(parsed, candidate))
        ranged = parse_drug_mention("x 0.5-1.5 mg")
        self.assertEqual("range_contains", reranker.compare_strength(ranged, candidate))

    def test_icd_aggregate_preserves_multiple_term_evidence(self):
        rows = [
            {"code": "A", "score": .9, "text": "x", "language": "vi", "term_type": "preferred", "term_id": "1", "query_source": "raw"},
            {"code": "A", "score": .8, "text": "y", "language": "en", "term_type": "inclusion", "term_id": "2", "query_source": "folded"},
        ]
        item = aggregate_term_results(rows, top_k_codes=1)[0]
        self.assertEqual(2, item["independent_term_count"])
        self.assertEqual({"raw", "folded"}, set(item["query_sources"]))
        self.assertGreater(len(_query_variants("bệnh A-B (AB) ở trẻ")), 2)

    def test_selector_v2_whitelist_cache_and_batch_size(self):
        request_id = hashlib.sha1("CHẨN_ĐOÁN|x|".encode("utf-8")).hexdigest()[:16]
        response = '{"request_id":"%s","decision":"SELECT","chosen_codes":["A"],"confidence":"HIGH","reason_code":"CONTEXT_DISAMBIGUATION"}' % request_id
        llm = _BatchLlm(response)
        items = [{"entity_text": "x", "entity_type": "CHẨN_ĐOÁN", "context": "",
                  "candidates": [{"code": "A"}, {"code": "B"}]}]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "selector.jsonl"
            first = select_candidates_many(items, llm, batch_size=7, cache_path=cache)
            second = select_candidates_many(items, llm, batch_size=7, cache_path=cache)
        self.assertEqual([["A"]], first)
        self.assertEqual(first, second)
        self.assertEqual(1, llm.calls)
        self.assertEqual(7, llm.batch_size)

    def test_selector_schema_rejects_too_many_or_bad_decision(self):
        self.assertIsNone(CandidateSelection.from_dict({"decision": "SELECT", "chosen_codes": []}))
        valid = CandidateSelection.from_dict({
            "request_id": "r", "decision": "ABSTAIN", "chosen_codes": [],
            "confidence": "LOW", "reason_code": "INSUFFICIENT_EVIDENCE",
        })
        self.assertIsNotNone(valid)


if __name__ == "__main__": unittest.main()
