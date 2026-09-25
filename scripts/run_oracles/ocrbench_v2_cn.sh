#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify OCRBench_v2 Chinese published answers with Gym's pinned evaluator.
# The full annotation list stays in each prepared row for official scoring.
# Usage: bash scripts/run_oracles/ocrbench_v2_cn.sh [--prepare-only] [positive-limit]
# First preparation needs GITLAB_TOKEN; the public TSV is about 1.4 GB.
# Results: results/ocrbench_v2_cn_oracle.jsonl and .aggregate.json.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/ocrbench_v2_cn.sh [--prepare-only] [positive-limit]'
    echo 'Prepare and verify published OCRBench_v2 Chinese annotations with Gym.'
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
oracle_ports ocrbench_v2_cn
REPO_ROOT="$PWD"
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$REPO_ROOT/scripts/run_oracles/build-constraints.txt}"
LOG_DIR=results/ocrbench_v2_cn_oracle
OUTPUT_JSONL=results/ocrbench_v2_cn_oracle.jsonl
SOURCE_DIR=results/oracle-upstream/VLMEvalKitMcore-6962c8d06b2b7b26a74a73d6212c06562b63e1b7
mkdir -p "$LOG_DIR"

echo '==> Preparing benchmark data...'
uv run --locked python benchmarks/ocrbench_v2/prepare_cn.py 2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi

echo '==> Starting Gym...'
set -m
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --locked gym env start --config benchmarks/ocrbench_v2/config_cn.yaml \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    ++ocrbench_v2_cn_benchmark_simple_agent=null \
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

echo '==> Verifying published Chinese references...'
uv run --locked python scripts/run_oracles/submit_references.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/ocrbench_v2/data/ocrbench_v2_cn_benchmark.jsonl \
    +resources_server=ocrbench_v2_cn_benchmark_resources_server \
    +reference_field=oracle_reference \
    +output_jsonl="$OUTPUT_JSONL" \
    "${LIMIT_ARGS[@]}" 2>&1 | tee "$LOG_DIR/oracle.log"

echo '==> Computing the official five-group Chinese aggregate...'
PYTHONPATH="$REPO_ROOT/$SOURCE_DIR:$REPO_ROOT" \
    uv run --project resources_servers/vlm_eval_kit python benchmarks/ocrbench_v2/aggregate_cn.py "$OUTPUT_JSONL" \
    2>&1 | tee "$LOG_DIR/aggregate.log"
echo "==> Results: $OUTPUT_JSONL; aggregate: results/ocrbench_v2_cn_oracle.aggregate.json; logs: $LOG_DIR/"
