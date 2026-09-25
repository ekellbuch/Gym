#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: ./scripts/run_oracles/deepswe.sh [limit | --prepare-only | --help]
# Prepare upstream data, start Gym, and apply upstream gold patches in parallel.
# This wrapper owns only an optional positive task limit. OpenSandbox credentials
# must be exported. Results and server logs are saved under results/.
# The upstream Python runner owns verification and scoring; no model is called.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports deepswe
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$PWD/scripts/run_oracles/build-constraints.txt}"
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [positive-task-limit | --prepare-only | --help]"
    echo "Prepare upstream data and verify gold patches through Gym; results are under results/."
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
LIMIT="${1:-}"
if [[ $# -gt 1 || ( -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ) ]]; then
    echo "Usage: $0 [positive-task-limit]" >&2
    exit 2
fi
OUTPUT_JSONL="results/deepswe_oracle.jsonl"
LOG_DIR="results/deepswe_oracle"
mkdir -p "$LOG_DIR"

echo "==> Preparing benchmark data..."
uv run --locked gym eval prepare --config benchmarks/deepswe/opencode.yaml \
    2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi
# Count selected rows; native async runners dispatch their verification requests.
TASK_COUNT=$(awk 'NF { count++ } END { print count+0 }' benchmarks/deepswe/data/deepswe_benchmark.jsonl)
if [[ -n "$LIMIT" && "$LIMIT" -lt "$TASK_COUNT" ]]; then TASK_COUNT="$LIMIT"; fi
if [[ "$TASK_COUNT" -eq 0 ]]; then echo "Prepared dataset is empty." >&2; exit 1; fi

echo "==> Starting deepswe_resources_server..."
# Keep SIGINT enabled in the background job so Gym can stop its workers.
set -m
uv run --locked gym env start \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    --config resources_servers/deepswe/configs/deepswe.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    +deepswe_resources_server.resources_servers.deepswe.is_verifying_golden_patch=true \
    > "$LOG_DIR/server.log" 2>&1 &
GYM_ENV_PID=$!
set +m
# Stop the server owned by this invocation, including when verification fails.
cleanup() {
    kill -INT "$GYM_ENV_PID" 2>/dev/null || true
    wait "$GYM_ENV_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "==> Waiting for Gym readiness..."
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
if ! "$SERVER_READY"; then echo "Gym did not become ready within 15 minutes." >&2; exit 1; fi

# Forward the optional limit to the existing upstream runner without changing rows.
LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then LIMIT_ARGS=(+limit="$LIMIT"); fi
echo "==> Verifying upstream gold patches ($TASK_COUNT parallel tasks)..."
uv run --locked python resources_servers/deepswe/validate_golden.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/deepswe/data/deepswe_benchmark.jsonl \
    +concurrency="$TASK_COUNT" \
    +output_jsonl="$OUTPUT_JSONL" \
    "${LIMIT_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/oracle.log"
echo "==> Per-task oracle results: $OUTPUT_JSONL; logs: $LOG_DIR"
