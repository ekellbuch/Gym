#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit official HLE text answers unchanged to the native Gym HLE verifier.
# Usage: bash scripts/run_oracles/hle_text.sh [--prepare-only | positive-task-limit]
# Requires HF_TOKEN for preparation and INFERENCE_API_KEY for the judge.
# Results: results/hle_text_oracle.jsonl; logs: results/hle_text_oracle/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports hle_text
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/hle_text.sh [--prepare-only | positive-task-limit]'
    echo 'Prepare official text-only HLE answers and score them with the native Gym HLE judge.'
    echo 'Requires HF_TOKEN and INFERENCE_API_KEY. Results/logs: results/hle_text_oracle*.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
LIMIT="${1:-}"
if [[ $# -gt 1 || ( -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ) ]]; then
    echo 'Expected a positive task limit or --prepare-only.' >&2
    exit 2
fi
LOG_DIR="results/hle_text_oracle"
OUTPUT_JSONL="results/hle_text_oracle.jsonl"
mkdir -p "$LOG_DIR"
echo "==> Preparing benchmark data..."
uv run --locked python -c 'from benchmarks.hle.prepare import prepare; prepare()' \
    2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi
if [[ -z "${INFERENCE_API_KEY:-}" ]]; then
    echo 'INFERENCE_API_KEY is not set in this shell; restart the session.' >&2
    exit 1
fi
# Reuse the inference credential, as in the post-training HLE judge recipe.
export JUDGE_API_KEY="$INFERENCE_API_KEY"
echo '==> Starting Gym...'
# Keep SIGINT enabled so Gym can stop its worker processes.
set -m
uv run --locked gym env start --config scripts/run_oracles/configs/hle_text.yaml \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    > "$LOG_DIR/server.log" 2>&1 &
GYM_ENV_PID=$!
set +m
cleanup() {
    kill -INT "$GYM_ENV_PID" 2>/dev/null || true
    wait "$GYM_ENV_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo '==> Waiting for Gym readiness...'
SERVER_READY=false
for i in $(seq 1 180); do
    if ! kill -0 "$GYM_ENV_PID" 2>/dev/null; then
        echo "Gym exited; inspect $LOG_DIR/server.log" >&2
        exit 1
    fi
    if uv run --locked gym env status --json "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" 2>/dev/null \
        | uv run --locked python scripts/run_oracles/gym_ready.py; then
        SERVER_READY=true
        break
    fi
    sleep 5
done
if ! "$SERVER_READY"; then echo 'Gym did not become ready within 15 minutes.' >&2; exit 1; fi
LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then LIMIT_ARGS=(+limit="$LIMIT"); fi
echo '==> Verifying official reference answers...'
# The submitter preserves expected_answer verbatim; Gym owns HLE scoring.
uv run --locked python scripts/run_oracles/submit_references.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/hle/data/hle_benchmark.jsonl \
    +resources_server=hle_equivalence_llm_judge_resources_server \
    +reference_field=expected_answer \
    +prompt_config=benchmarks/hle/prompts/default.yaml \
    +output_jsonl="$OUTPUT_JSONL" \
    "${LIMIT_ARGS[@]}" 2>&1 | tee "$LOG_DIR/oracle.log"
echo "==> Results: $OUTPUT_JSONL; logs: $LOG_DIR"
