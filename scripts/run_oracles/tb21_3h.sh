#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: ./scripts/run_oracles/tb21_3h.sh [limit | --prepare-only | --help]
# Uses the upstream evaluation_timeout field with the requested three-hour limit.
# Prepare upstream data, start Gym, and apply upstream gold patches in parallel.
# This wrapper owns an optional positive task limit and selected input/output
# paths via ORACLE_BENCHMARK_JSONL and ORACLE_OUTPUT_JSONL. OpenSandbox
# credentials must be exported. Results and server logs are saved under results/.
# The upstream Python runner owns verification and scoring; no model is called.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$PWD"
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$REPO_ROOT/scripts/run_oracles/build-constraints.txt}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [positive-task-limit | --prepare-only | --help]"
    echo "Prepare upstream data and verify gold patches through Gym; results are under results/."
    echo "Set ORACLE_BENCHMARK_JSONL and ORACLE_OUTPUT_JSONL to retry selected official rows without replacing the full result."
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
LIMIT="${1:-}"
if [[ $# -gt 1 || ( -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ) ]]; then
    echo "Usage: $0 [positive-task-limit]" >&2
    exit 2
fi
OUTPUT_JSONL="${ORACLE_OUTPUT_JSONL:-results/tb21_3h_oracle.jsonl}"
BENCHMARK_JSONL="${ORACLE_BENCHMARK_JSONL:-benchmarks/terminal_bench_2_1/data/benchmark.jsonl}"
LOG_PREFIX="${OUTPUT_JSONL%.jsonl}"
RUN_DIR="$LOG_PREFIX"
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
BENCHMARK_PATH="$(cd "$(dirname "$BENCHMARK_JSONL")" && pwd -P)/$(basename "$BENCHMARK_JSONL")"

echo "==> Preparing upstream benchmark data..."
uv run --locked gym eval prepare --config benchmarks/terminal_bench_2_1/terminus_2.yaml \
    2>&1 | tee "$LOG_PREFIX.prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi
# Count selected rows; native async runners dispatch their verification requests.
TASK_COUNT=$(awk 'NF { count++ } END { print count+0 }' "$BENCHMARK_PATH")
if [[ -n "$LIMIT" && "$LIMIT" -lt "$TASK_COUNT" ]]; then TASK_COUNT="$LIMIT"; fi
if [[ "$TASK_COUNT" -eq 0 ]]; then echo "Prepared dataset is empty." >&2; exit 1; fi

echo "==> Starting terminal_bench_2_1_resources_server..."
# Preserve SIGINT handling so Gym shuts down its worker processes.
set -m
uv run --locked gym env start \
    --config resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    +terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.is_verifying_golden_patch=true \
    +terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.evaluation_timeout=10800 \
    ++head_server.port=11135 ++port_range_low=21211 ++port_range_high=21230 \
    > "$LOG_PREFIX.server.log" 2>&1 &
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
        echo "Gym exited; inspect $LOG_PREFIX.server.log" >&2
        exit 1
    fi
    if uv run --locked gym env status --json ++head_server.port=11135 2>/dev/null \
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
# The upstream runner writes a fixed temp2.jsonl name. Its own workdir
# keeps this run separate from other Gym oracles without changing the runner.
(
    cd "$RUN_DIR"
    uv run --project "$REPO_ROOT" --locked python \
        "$REPO_ROOT/resources_servers/terminal_bench_2_1/apply_golden_patch.py" \
        +benchmark_jsonl="$BENCHMARK_PATH" \
        ++head_server.port=11135 "${LIMIT_ARGS[@]}"
) 2>&1 | tee "$LOG_PREFIX.log"
mv "$RUN_DIR/temp2.jsonl" "$OUTPUT_JSONL"
echo "==> Per-task oracle results: $OUTPUT_JSONL"
