import json
import unittest

from src.inference.ner.candidates import CandidateEvidence, MissingProposal
from src.inference.ner.editor_schemas import MissingDecision
from src.inference.ner.qwen_editor import apply_editor_response, apply_missing_decisions
from src.inference.schemas import NerEntity


DIAGNOSIS = "CHẨN_ĐOÁN"
SYMPTOM = "TRIỆU_CHỨNG"


def candidate(raw, text, entity_type=SYMPTOM, *, sources=None, scores=None):
    start = raw.index(text)
    sources = sources or ["crf"]
    scores = scores or {"crf": 0.7}
    return CandidateEvidence(
        "cand_" + str(start), text, entity_type, (start, start + len(text)),
        sources=list(sources), scores=dict(scores), allowed_types=[SYMPTOM, DIAGNOSIS],
        pre_llm_selected=True,
    )


class LockedEditorTests(unittest.TestCase):
    def test_invalid_json_keeps_original(self):
        raw = "Bệnh nhân sốt."
        item = candidate(raw, "sốt")
        result = apply_editor_response(raw, [item], "not-json")
        self.assertEqual(["sốt"], [entity.text for entity in result.entities])

    def test_invalid_json_does_not_promote_audit_only_candidate(self):
        raw = "Bệnh nhân sốt."
        item = candidate(raw, "sốt")
        item.pre_llm_selected = False
        result = apply_editor_response(raw, [item], "not-json")
        self.assertEqual([], result.entities)

    def test_unknown_candidate_id_is_rejected(self):
        raw = "Bệnh nhân sốt."
        item = candidate(raw, "sốt")
        response = {"actions": [{
            "action": "DROP", "candidate_ids": ["invented"], "text": None,
            "type": None, "assertions": [], "local_position": None,
            "confidence": "HIGH", "reason_code": "FUNCTION_WORD_OR_FRAGMENT",
        }]}
        result = apply_editor_response(raw, [item], json.dumps(response))
        self.assertEqual(1, len(result.entities))
        self.assertTrue(result.rejected)

    def test_retype_preserves_exact_position(self):
        raw = "Chẩn đoán Kawasaki."
        item = candidate(raw, "Kawasaki")
        response = {"actions": [{
            "action": "RETYPE", "candidate_ids": [item.candidate_id], "text": None,
            "type": DIAGNOSIS, "assertions": [], "local_position": None,
            "confidence": "HIGH", "reason_code": "WRONG_TYPE",
        }]}
        result = apply_editor_response(raw, [item], json.dumps(response))
        self.assertEqual(item.position, result.entities[0].position)
        self.assertEqual(DIAGNOSIS, result.entities[0].type)

    def test_newline_leakage_is_repaired_to_first_line(self):
        raw = "bệnh Kawasaki\nMặc"
        item = candidate(raw, raw, DIAGNOSIS)
        response = {"actions": [{
            "action": "REPAIR_SPAN", "candidate_ids": [item.candidate_id],
            "text": "bệnh Kawasaki", "type": DIAGNOSIS, "assertions": [],
            "local_position": [0, len("bệnh Kawasaki")], "confidence": "HIGH",
            "reason_code": "WRONG_BOUNDARY",
        }]}
        result = apply_editor_response(raw, [item], json.dumps(response))
        self.assertEqual(["bệnh Kawasaki"], [entity.text for entity in result.entities])

    def test_merge_requires_exact_same_unit_span(self):
        raw = "Sưng đỏ mu bàn tay – chân."
        left = candidate(raw, "Sưng")
        right = candidate(raw, "đỏ mu bàn tay – chân")
        response = {"actions": [{
            "action": "MERGE", "candidate_ids": [left.candidate_id, right.candidate_id],
            "text": raw[:-1], "type": SYMPTOM, "assertions": [],
            "local_position": [0, len(raw) - 1], "confidence": "HIGH",
            "reason_code": "MERGE_REQUIRED",
        }]}
        result = apply_editor_response(raw, [left, right], json.dumps(response))
        self.assertEqual([raw[:-1]], [entity.text for entity in result.entities])

    def test_duplicate_actions_block_only_affected_candidate(self):
        raw = "sốt và ho"
        fever, cough = candidate(raw, "sốt"), candidate(raw, "ho")
        actions = []
        for action, item in [("DROP", fever), ("KEEP", fever), ("DROP", cough)]:
            actions.append({
                "action": action, "candidate_ids": [item.candidate_id], "text": None,
                "type": None, "assertions": [], "local_position": None,
                "confidence": "HIGH", "reason_code": (
                    "FUNCTION_WORD_OR_FRAGMENT" if action == "DROP" else "VALID_ENTITY"
                ),
            })
        result = apply_editor_response(raw, [fever, cough], json.dumps({"actions": actions}))
        self.assertEqual(["sốt"], [entity.text for entity in result.entities])

    def test_strong_consensus_generic_drop_is_blocked(self):
        raw = "bệnh tim"
        item = candidate(raw, raw, DIAGNOSIS, sources=["crf", "span_head"], scores={"crf": .95, "span_head": .96})
        response = {"actions": [{
            "action": "DROP", "candidate_ids": [item.candidate_id], "text": None,
            "type": None, "assertions": [], "local_position": None,
            "confidence": "HIGH", "reason_code": "AMBIGUOUS",
        }]}
        result = apply_editor_response(raw, [item], json.dumps(response))
        self.assertEqual(1, len(result.entities))

    def test_missing_add_is_proposal_id_only(self):
        raw = "Kawasaki"
        proposal = MissingProposal(
            "prop_1", raw, (0, len(raw)), [DIAGNOSIS],
            ["repeated_confirmed_surface"], ["repeated_confirmed_surface"],
            auto_add_eligible=True,
        )
        decision = MissingDecision.from_dict({
            "proposal_id": "prop_1", "decision": "ADD_PROPOSAL", "type": DIAGNOSIS,
            "assertions": [], "confidence": "HIGH", "reason_code": "VALID_MISSING_ENTITY",
        })
        result = apply_missing_decisions(raw, [], [proposal], [decision])
        self.assertEqual([raw], [entity.text for entity in result.entities])

    def test_weak_proposal_and_overlap_are_blocked(self):
        raw = "Kawasaki"
        existing = [NerEntity(raw, DIAGNOSIS, [], (0, len(raw)))]
        proposal = MissingProposal("prop_1", raw, (0, len(raw)), [DIAGNOSIS], ["soft"], [])
        decision = MissingDecision.from_dict({
            "proposal_id": "prop_1", "decision": "ADD_PROPOSAL", "type": DIAGNOSIS,
            "assertions": [], "confidence": "HIGH", "reason_code": "VALID_MISSING_ENTITY",
        })
        result = apply_missing_decisions(raw, existing, [proposal], [decision])
        self.assertEqual(1, len(result.entities))
        self.assertTrue(result.rejected)


if __name__ == "__main__":
    unittest.main()
