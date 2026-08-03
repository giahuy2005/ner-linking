#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 || ( "$1" != "one" && "$1" != "30" ) ]]; then
  echo "Usage: bash scripts/benchmark_a40_llm.sh one|30 INPUT SAVED_DETAILED_DIR OUTPUT_DIR" >&2
  exit 2
fi

mode="$1"
input="$2"
details="$3"
output="$4"
mkdir -p "$output"

if [[ "$mode" == "one" ]]; then
  input_args=(--input "$input")
else
  input_args=(--input-dir "$input")
fi

python -m src.inference.cli \
  "${input_args[@]}" \
  --saved-detailed-ner-dir "$details" \
  --output-dir "$output" \
  --with-rxnorm --with-icd10 --with-llm-8b \
  --speed-profile balanced \
  --llm-batch-size 12 \
  --llm-max-batch-tokens 16384 \
  --llm-min-batch-size 1 \
  --llm-device-map single_gpu \
  --require-full-gpu \
  --llm-local-files-only \
  --llm-cache-path "$output/editor_recovery_cache.jsonl" \
  --linking-selector-cache-path "$output/selector_cache.jsonl" \
  --stage-cache-dir "$output/stages" \
  --llm-progress-every 1 \
  --max-recovery-proposals-per-record 24 \
  --max-recovery-requests-per-record 4

echo "Runtime:  $output/.stages/runtime.json"
echo "Workload: $output/.stages/llm_workload.json"
echo "Batches:  $output/.stages/llm_batch_telemetry.json"
