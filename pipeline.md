# Production NER + linking pipeline

```text
raw text
  -> offset-preserving sectioning/tokenization
  -> ViHealthBERT CRF + trained span head
  -> exact CRF marginals + conservative span lattice
  -> stable candidate catalog + closed missing proposals
  -> Qwen3-8B locked editor
  -> action-level Python guards + structural validation
  -> batched RxNorm/ICD-10 retrieval
  -> deterministic reranking
  -> same Qwen3-8B whitelisted code selector
  -> strict BTC output validation + audit JSON
```

The production path has one optional LLM: `Qwen/Qwen3-8B`. Candidate IDs are
stable and missing-entity recovery is restricted to exact proposals.

## Main modules

- `src/inference/ner/engine.py`: strict CRF/span checkpoint load and detailed inference.
- `src/inference/ner/evidence.py`: typed word/CRF/span/local evidence.
- `src/inference/ner/candidates.py`: stable catalog and exact proposals.
- `src/inference/ner/editor_schemas.py`: closed editor schemas.
- `src/inference/ner/qwen_editor.py`: prompts, cache, validation and application.
- `src/inference/pipeline.py`: batch stage orchestration.
- `src/inference/selection/candidate_selector.py`: whitelisted linking selection.
- `src/linking/rxnorm/`: batched RxNorm retrieval and ranking.
- `src/linking/icd10/`: metadata aliases, query variants and batched ICD retrieval.

## Commands

NER only:

```bash
python -m src.inference.cli --input data/input/1.txt --print
```

Full A40 pipeline:

```bash
python -m src.inference.cli \
  --input-dir data/input \
  --output-dir output \
  --with-llm-8b --with-rxnorm --with-icd10 \
  --llm-dtype bfloat16 --llm-quantization none \
  --llm-cache-path output/qwen_cache.jsonl \
  --llm-audit-dir output/audit
```

Strict relink without changing NER identity:

```bash
python -m src.inference.relink_cli \
  --input-dir data/input --entities-dir output \
  --output-dir output_relinked --with-rxnorm --with-icd10
```
