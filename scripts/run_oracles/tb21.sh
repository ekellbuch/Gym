#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# What this does:
#   Every Terminal-Bench 2.1 task ships its own known-correct solution (a
#   "golden patch"). This script feeds that solution straight to the task's
#   verifier and checks it passes. There's no model and no agent involved —
#   it's just a sanity check that the verifier itself works.
#
# Usage:
#   ./scripts/run_oracles/tb21.sh [limit]
#   (no "limit" = run all tasks; pass a number like "2" for a quick smoke test)
#
# Before running:
#   Install uv; the commands below use the checked-in dependency lock.
#
# Where the results go:
#   results/terminal_bench_2_1_oracle.jsonl — one line per task, pass/fail.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$(pwd)"

# Starting the server below (gym env start) runs "uv pip install" for the first
# time in its own folder, and that install can freeze forever — this is a real
# bug in setuptools-scm 10+, not just a slow download. build-constraints.txt
# pins an older setuptools-scm to avoid it. The path has to be absolute here
# because that "uv pip install" runs from a different folder than this script.
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$REPO_ROOT/scripts/run_oracles/build-constraints.txt}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/tb21.sh [limit] [Gym oracle overrides...]'
    echo 'Use --prepare-only to download upstream tasks without starting Gym.'
    echo 'Set ORACLE_BENCHMARK_JSONL and ORACLE_OUTPUT_JSONL to verify selected official rows without replacing the full result.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then
    PREPARE_ONLY=true
    shift
fi
LIMIT=""
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    LIMIT="$1"
    shift
    if [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
        echo 'Task limit must be a positive integer.' >&2
        exit 2
    fi
fi
OUTPUT_JSONL="${ORACLE_OUTPUT_JSONL:-results/terminal_bench_2_1_oracle.jsonl}"
BENCHMARK_JSONL="${ORACLE_BENCHMARK_JSONL:-benchmarks/terminal_bench_2_1/data/benchmark.jsonl}"
LOG_DIR="${OUTPUT_JSONL%.jsonl}"

mkdir -p "$LOG_DIR"
LOG_DIR_PATH="$(cd "$LOG_DIR" && pwd -P)"
BENCHMARK_PATH="$(cd "$(dirname "$BENCHMARK_JSONL")" && pwd -P)/$(basename "$BENCHMARK_JSONL")"

echo "==> Preparing benchmark data..."
uv run --locked gym eval prepare --config benchmarks/terminal_bench_2_1/terminus_2.yaml \
    2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi

# The upstream runner submits all selected requests concurrently.
# Keep Gym's normal worker count; parallel requests do not need one process each.
CONCURRENCY="$(wc -l < "$BENCHMARK_PATH")"
if [[ -n "$LIMIT" ]] && (( LIMIT < CONCURRENCY )); then CONCURRENCY="$LIMIT"; fi
if (( CONCURRENCY < 1 )); then echo "No tasks selected." >&2; exit 2; fi

# Start the server that actually checks each task's solution. This command
# doesn't return on its own — it keeps running in the foreground forever — so
# we background it with "&" and save its process ID ($!) to shut it down later.
echo "==> Starting terminal_bench_2_1_resources_server (sandbox_provider=opensandbox, concurrency=$CONCURRENCY)..."
# Keep SIGINT enabled so Gym can shut down its worker processes on exit.
set -m
uv run --locked gym env start \
    --config resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    +terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.debug=true \
    +terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.is_verifying_golden_patch=true \
    ++head_server.port=11134 ++port_range_low=21191 ++port_range_high=21210 \
    > "$LOG_DIR/server.log" 2>&1 &
GYM_ENV_PID=$!
set +m

cleanup() {
    echo "==> Stopping terminal_bench_2_1_resources_server (pid $GYM_ENV_PID)..."
    kill -INT "$GYM_ENV_PID" 2>/dev/null || true
    wait "$GYM_ENV_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "==> Waiting for the resources server to be ready (first run builds its venv, can take 10+ min)..."
SERVER_READY=false
for i in $(seq 1 180); do
    if ! kill -0 "$GYM_ENV_PID" 2>/dev/null; then
        echo "Gym exited before readiness." >&2
        exit 1
    fi
    if uv run --locked gym env status --json ++head_server.port=11134 2>/dev/null \
        | uv run --locked python scripts/run_oracles/gym_ready.py; then
        echo "Server ready."
        SERVER_READY=true
        break
    fi
    sleep 5
done
if ! "$SERVER_READY"; then
    echo "Resources server never became ready after 15 minutes; aborting." >&2
    exit 1
fi

echo "==> Running the oracle against every task (parallel, via OpenSandbox)..."
LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARGS=(+limit="$LIMIT")
fi

# The upstream runner writes temp2.jsonl in its current directory. Run it from
# this oracle's log directory so concurrent benchmarks cannot replace its output.
(
    cd "$LOG_DIR_PATH"
    uv run --project "$REPO_ROOT" --locked python \
        "$REPO_ROOT/resources_servers/terminal_bench_2_1/apply_golden_patch.py" \
        +benchmark_jsonl="$BENCHMARK_PATH" ++head_server.port=11134 \
        "${LIMIT_ARGS[@]}" "$@"
) 2>&1 | tee "$LOG_DIR_PATH/oracle.log"

mv "$LOG_DIR_PATH/temp2.jsonl" "$OUTPUT_JSONL"
echo "==> Done. Per-task oracle results: $OUTPUT_JSONL"
