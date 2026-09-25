#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: ./scripts/run_oracles/swebench_verified.sh [limit | --prepare-only | --help]
# Prepare upstream data, start Gym, and apply upstream gold patches in parallel.
# This wrapper owns only an optional positive task limit. OpenSandbox credentials
# must be exported. Results and server logs are saved under results/.
# The upstream Python runner owns verification and scoring; no model is called.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$PWD/scripts/run_oracles/build-constraints.txt}"
# Ray otherwise uploads this checkout, including unrelated worktrees, before Gym becomes ready.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
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
OUTPUT_JSONL="results/swebench_verified_oracle.jsonl"
mkdir -p results

echo "==> Preparing upstream benchmark data..."
uv run --locked gym eval prepare --config benchmarks/swebench/verified/opencode.yaml \
    +head_server.port=11132 +port_range_low=21151 +port_range_high=21170 \
    2>&1 | tee results/swebench_verified_oracle.prepare.log
if "$PREPARE_ONLY"; then exit 0; fi
# Count selected rows; native async runners dispatch their verification requests.
TASK_COUNT=$(awk 'NF { count++ } END { print count+0 }' benchmarks/swebench/data/swebench_verified_benchmark.jsonl)
if [[ -n "$LIMIT" && "$LIMIT" -lt "$TASK_COUNT" ]]; then TASK_COUNT="$LIMIT"; fi
if [[ "$TASK_COUNT" -eq 0 ]]; then echo "Prepared dataset is empty." >&2; exit 1; fi
if [[ -n "$LIMIT" ]]; then OUTPUT_JSONL="results/swebench_verified_oracle_${TASK_COUNT}.jsonl"; fi

# Match benchmarks/swebench/verified/opencode.yaml: Verified repos are already clean.
echo "==> Starting swebench_resources_server..."
# Keep SIGINT enabled for the background job so Gym can shut down its workers.
set -m
uv run --locked gym env start \
    --config resources_servers/swebench/configs/swebench.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    +swebench_resources_server.resources_servers.swebench.is_verifying_golden_patch=true \
    +swebench_resources_server.resources_servers.swebench.apply_anti_cheating=false \
    +head_server.port=11132 +port_range_low=21151 +port_range_high=21170 \
    > "results/swebench_verified_oracle.server.log" 2>&1 &
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
        echo "Gym exited; inspect results/swebench_verified_oracle.server.log" >&2
        exit 1
    fi
    if uv run --locked gym env status --json +head_server.port=11132 2>/dev/null \
        | uv run --locked python scripts/run_oracles/gym_ready.py; then
        SERVER_READY=true
        break
    fi
    sleep 5
done
if ! "$SERVER_READY"; then echo "Gym did not become ready within 15 minutes." >&2; exit 1; fi

# Submit the same official rows through the unchanged upstream runner, eight at a time.
# Completed batches are validated and reused after an interrupted run.
echo "==> Verifying upstream gold patches ($TASK_COUNT tasks, eight concurrent sandboxes)..."
uv run --locked python scripts/run_oracles/swebench_gold_batches.py \
    benchmarks/swebench/data/swebench_verified_benchmark.jsonl "$TASK_COUNT" "$OUTPUT_JSONL" 11132 \
    2>&1 | tee results/swebench_verified_oracle.log
echo "==> Per-task oracle results: $OUTPUT_JSONL"
