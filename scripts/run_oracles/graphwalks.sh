#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit official GraphWalks reference node lists unchanged to the native Gym verifier.
# Usage: bash scripts/run_oracles/graphwalks.sh [--prepare-only | positive-task-limit]
# Node sets come from upstream preparation; no graph-solving algorithm is run.
# Results: results/graphwalks_oracle.jsonl; logs: results/graphwalks_oracle/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports graphwalks
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/graphwalks.sh [--prepare-only | positive-task-limit]'
    echo 'Prepare official GraphWalks reference node lists and score them with the native GraphWalks verifier.'
    echo 'Results/logs: results/graphwalks_oracle*.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
LIMIT="${1:-}"
if [[ $# -gt 1 || ( -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ) ]]; then
    echo 'Expected a positive task limit or --prepare-only.' >&2
    exit 2
fi
LOG_DIR="results/graphwalks_oracle"
OUTPUT_JSONL="results/graphwalks_oracle.jsonl"
mkdir -p "$LOG_DIR"
echo "==> Preparing benchmark data..."
uv run --locked --extra graphwalks python -c 'from benchmarks.graphwalks.prepare import prepare; prepare()' \
    2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi
echo '==> Starting Gym...'
# Keep SIGINT enabled so Gym can stop its worker processes.
set -m
uv run --locked gym env start --config scripts/run_oracles/configs/graphwalks.yaml \
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
# Serialize the published node list in the native Final Answer format; do not solve the graph.
uv run --locked python scripts/run_oracles/submit_references.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/graphwalks/data/graphwalks_benchmark.jsonl \
    +resources_server=graphwalks_benchmark_resources_server \
    +reference_field=expected_answer \
    +reference_encoding=json_string_list \
    '+response_template="Final Answer: {answer}"' \
    +output_jsonl="$OUTPUT_JSONL" \
    "${LIMIT_ARGS[@]}" 2>&1 | tee "$LOG_DIR/oracle.log"
echo "==> Results: $OUTPUT_JSONL; logs: $LOG_DIR"
