#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit the 6,000 OCRBench_v2 English rows whose published annotation is a
# complete response. Five unsupported categories stay in the full prepared data.
# Usage: bash scripts/run_oracles/ocrbench_v2_en.sh [--prepare-only] [positive-limit]
# First preparation needs GITLAB_TOKEN; the public TSV is about 1.4 GB.
# Results: results/ocrbench_v2_en_oracle.jsonl; logs: results/ocrbench_v2_en_oracle/.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/ocrbench_v2_en.sh [--prepare-only] [positive-limit]'
    echo 'Prepare and verify the 6,000-row published-reference English subset with Gym.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
LIMIT_ARGS=()
if [[ $# -gt 1 || ( $# -eq 1 && ! "$1" =~ ^[1-9][0-9]*$ ) ]]; then
    echo 'Expected one positive task limit or --prepare-only.' >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then LIMIT_ARGS=(+limit="$1"); fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports ocrbench_v2_en
REPO_ROOT="$PWD"
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$REPO_ROOT/scripts/run_oracles/build-constraints.txt}"
LOG_DIR=results/ocrbench_v2_en_oracle
OUTPUT_JSONL=results/ocrbench_v2_en_oracle.jsonl
mkdir -p "$LOG_DIR"

echo '==> Preparing benchmark data...'
uv run --locked python benchmarks/ocrbench_v2/prepare_reference_subset.py 2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi

echo '==> Starting Gym...'
set -m
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --locked gym env start --config benchmarks/ocrbench_v2/config_en.yaml \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    ++ocrbench_v2_en_benchmark_simple_agent=null \
    ++vlm_eval_kit_simple_agent=null \
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

echo '==> Submitting complete published references for the English subset...'
uv run --locked python scripts/run_oracles/submit_references.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/ocrbench_v2/data/ocrbench_v2_en_reference_subset_benchmark.jsonl \
    +resources_server=ocrbench_v2_en_benchmark_resources_server \
    +reference_field=oracle_reference \
    +output_jsonl="$OUTPUT_JSONL" \
    "${LIMIT_ARGS[@]}" 2>&1 | tee "$LOG_DIR/oracle.log"
echo '==> Computing the pinned five-group English subset aggregate...'
uv run --locked --project resources_servers/vlm_eval_kit python benchmarks/ocrbench_v2/aggregate_en_subset.py "$OUTPUT_JSONL" \
    2>&1 | tee "$LOG_DIR/aggregate.log"
echo "==> English 6,000-row reference subset results: $OUTPUT_JSONL and results/ocrbench_v2_en_oracle.aggregate.json; logs: $LOG_DIR/"
