#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit official WMT24++ translations unchanged to the native Gym verifier.
# Usage: bash scripts/run_oracles/wmt24pp.sh [--prepare-only | positive-task-limit]
# Full metrics require the native XCOMET-XXL Ray GPU workers (extra_gpu resources).
# Results: results/wmt24pp_oracle.jsonl; logs: results/wmt24pp_oracle/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports wmt24pp
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/wmt24pp.sh [--prepare-only | positive-task-limit]'
    echo 'Prepare official WMT24++ translations and compute native spBLEU, chrF, COMET and language consistency.'
    echo 'Requires native XCOMET-XXL Ray GPU setup. Results/logs: results/wmt24pp_oracle*.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
LIMIT="${1:-}"
if [[ $# -gt 1 || ( -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ) ]]; then
    echo 'Expected a positive task limit or --prepare-only.' >&2
    exit 2
fi
LOG_DIR="results/wmt24pp_oracle"
OUTPUT_JSONL="results/wmt24pp_oracle.jsonl"
mkdir -p "$LOG_DIR"
# Preparation defers the COMET download to native runtime; compute_comet stays enabled.
echo "==> Preparing benchmark data..."
uv run --locked python -c 'from benchmarks.wmt24pp.prepare import prepare; prepare(prefetch_comet=False)' \
    2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi
echo '==> Starting Gym...'
# Keep SIGINT enabled so Gym can stop its worker processes.
set -m
uv run --locked gym env start --config scripts/run_oracles/configs/wmt24pp.yaml \
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
# The submitter preserves the official translation verbatim; Gym owns scoring.
uv run --locked python scripts/run_oracles/submit_references.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/wmt24pp/data/wmt24pp_benchmark.jsonl \
    +resources_server=wmt24pp_wmt_translation_resources_server \
    +reference_field=translation \
    +prompt_config=benchmarks/wmt24pp/prompts/default.yaml \
    +output_jsonl="$OUTPUT_JSONL" \
    "${LIMIT_ARGS[@]}" 2>&1 | tee "$LOG_DIR/oracle.log"
# Aggregation is a separate Gym process; give it the same scorer config and
# head port used for verification so it reaches this run's WMT server.
echo '==> Computing full native translation metrics (including XCOMET-XXL)...'
uv run --locked gym eval aggregate \
    --config scripts/run_oracles/configs/wmt24pp.yaml \
    --input-glob "$OUTPUT_JSONL" --output "$OUTPUT_JSONL" \
    +merge_shards=false "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/aggregate.log"
echo "==> Results: $OUTPUT_JSONL; full metrics: results/wmt24pp_oracle_aggregate_metrics.json; logs: $LOG_DIR"
