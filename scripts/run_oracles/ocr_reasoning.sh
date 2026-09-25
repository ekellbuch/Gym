#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Published OCR Reasoning reference chains -> native Gym verifier -> per-row results.
# Usage: bash scripts/run_oracles/ocr_reasoning.sh [--prepare-only] [positive-limit]
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/ocr_reasoning.sh [--prepare-only] [positive-limit]'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if [[ $# -gt 1 || ( $# -eq 1 && ! "$1" =~ ^[1-9][0-9]*$ ) ]]; then
    echo 'Expected one positive task limit or --prepare-only.' >&2
    exit 2
fi
LIMIT_ARGS=()
if [[ $# -eq 1 ]]; then LIMIT_ARGS=(+limit="$1"); fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports ocr_reasoning
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$PWD/scripts/run_oracles/build-constraints.txt}"
mkdir -p results/ocr_reasoning_oracle
echo '==> Preparing benchmark data...'
uv run --locked python benchmarks/ocr_reasoning/prepare.py 2>&1 | tee results/ocr_reasoning_oracle/prepare.log
if "$PREPARE_ONLY"; then exit 0; fi
if [[ -z "${INFERENCE_API_KEY:-}" ]]; then
    echo 'INFERENCE_API_KEY is not set in this shell; restart the session.' >&2
    exit 1
fi
uv run --locked python scripts/run_oracles/check_judge_auth.py benchmarks/ocr_reasoning/config.yaml
echo '==> Starting Gym...'
set -m
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --locked gym env start --config benchmarks/ocr_reasoning/config.yaml \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    ++ocr_reasoning_simple_agent=null ++vlm_eval_kit_simple_agent=null \
    > results/ocr_reasoning_oracle/server.log 2>&1 &
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
        echo 'Gym exited; inspect results/ocr_reasoning_oracle/server.log' >&2
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
echo '==> Submitting official reference answers...'
uv run --locked python scripts/run_oracles/submit_references.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/ocr_reasoning/data/ocr_reasoning_benchmark.jsonl \
    +resources_server=ocr_reasoning_resources_server +reference_field=reasoning \
    +output_jsonl=results/ocr_reasoning_oracle.jsonl "${LIMIT_ARGS[@]}" \
    2>&1 | tee results/ocr_reasoning_oracle/oracle.log
echo '==> Computing native OCR Reasoning accuracy and reasoning metrics...'
uv run --locked gym eval aggregate \
    --input-glob results/ocr_reasoning_oracle.jsonl \
    --output results/ocr_reasoning_oracle.jsonl \
    +merge_shards=false 2>&1 | tee results/ocr_reasoning_oracle/aggregate.log
echo '==> Results: results/ocr_reasoning_oracle.jsonl; aggregate metrics: results/ocr_reasoning_oracle_aggregate_metrics.json'
