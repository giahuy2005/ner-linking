import json
import tempfile
import unittest
from pathlib import Path

from src.inference.ner.candidates import CandidateEvidence
from src.inference.ner.editor_schemas import ReviewRegion
from src.inference.ner.qwen_editor import (
    PROMPT_VERSION, apply_editor_response, build_editor_request,
)
from src.llm.batching import VersionedJsonlCache, generate_with_cache
from src.inference.detailed_artifacts import load_artifacts, save_artifact
from src.inference.ner.evidence import NerDetailedResult
from src.inference.schemas import NerEntity


TYPE = "TRIỆU_CHỨNG"


def item(raw, text, identifier, *, selected=True):
    start = raw.index(text)
    return CandidateEvidence(
        identifier, text, TYPE, (start, start + len(text)),
        sources=["crf"], scores={"crf": 0.75}, allowed_types=[TYPE],
        pre_llm_selected=selected,
    )


class _InterruptedLLM:
    def __init__(self):
        self.calls = 0

    def generate_batches(self, prompts, **_kwargs):
        self.calls += 1
        yield {"indexes": [0], "responses": ["first"], "stats": {}, "batch_total": 2}
        raise RuntimeError("simulated interruption")


class V4SpeedGuards(unittest.TestCase):
    def test_change_only_omission_keeps_targets_and_not_context(self):
        raw = "sốt và ho"
        fever = item(raw, "sốt", "c1")
        cough = item(raw, "ho", "c2", selected=False)
        result = apply_editor_response(
            raw, [fever, cough], json.dumps({
                "request_id": "r", "changes": [], "unresolved_ids": [],
            }), target_candidate_ids=["c1"],
        )
        self.assertEqual(["sốt"], [entity.text for entity in result.entities])
        self.assertEqual([], result.unresolved)

    def test_change_only_drop_does_not_affect_omitted_target(self):
        raw = "sốt và ho"
        fever, cough = item(raw, "sốt", "c1"), item(raw, "ho", "c2")
        response = {"request_id": "r", "changes": [{
            "action": "DROP", "candidate_ids": ["c1"], "text": None,
            "type": None, "assertions": [], "local_position": None,
            "reason_code": "FUNCTION_WORD_OR_FRAGMENT",
        }], "unresolved_ids": []}
        result = apply_editor_response(raw, [fever, cough], json.dumps(response))
        self.assertEqual(["ho"], [entity.text for entity in result.entities])

    def test_prompt_is_v3_compact_and_marks_context_role(self):
        raw = "sốt và ho"
        fever = item(raw, "sốt", "c1")
        cough = item(raw, "ho", "c2", selected=False)
        region = ReviewRegion("r", "record", raw, 0, len(raw), ["c1"], ["c2"])
        _system, user = build_editor_request(region, [fever, cough])
        payload = json.loads(user)
        self.assertEqual(PROMPT_VERSION, payload["schema_version"])
        self.assertEqual(["selected_target", "context_only"], [row["role"] for row in payload["candidates"]])
        self.assertNotIn("global_position", user)
        self.assertNotIn("pre_llm_selected", user)

    def test_incremental_cache_survives_interrupted_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            cache = VersionedJsonlCache(path)
            self.assertTrue(path.exists())
            with self.assertRaises(RuntimeError):
                generate_with_cache(
                    _InterruptedLLM(), [("s", "1"), ("s", "2")],
                    batch_size=1, model_id="m", task="editor",
                    prompt_version="v", cache=cache, max_new_tokens=8,
                )
            self.assertEqual(1, len(VersionedJsonlCache(path).values))

    def test_saved_detailed_artifact_is_portable_and_raw_hash_locked(self):
        raw = "sốt"
        detail = NerDetailedResult(
            len(raw), len(raw), final_entities=[NerEntity(raw, TYPE, [], (0, len(raw)))],
        )
        with tempfile.TemporaryDirectory() as directory:
            save_artifact(directory, "1", raw, detail)
            loaded = load_artifacts(directory, {"1": raw})
            self.assertEqual(raw, loaded["1"].final_entities[0].text)
            with self.assertRaises(ValueError):
                load_artifacts(directory, {"1": raw + "!"})


if __name__ == "__main__":
    unittest.main()
