#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Execute official TB4 solution/solve.sh files through this checkout's Gym server.
# Usage: bash scripts/run_oracles/tb4.sh [limit] [Gym oracle overrides...]
# Use --prepare-only to download the pinned packages without running solutions.
# Set TB4_USE_PREPARED=1 to reuse an already prepared local dataset after a failed run.
# Harbor is only the package downloader; no Harbor evaluation is started.
# Packages: results/task-packages; native Gym inputs: benchmarks/terminal_bench_4/data/.
# Results and prepare/server/oracle logs: results/terminal_bench_4_oracle*.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports tb4
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/tb4.sh [limit] [Gym oracle overrides...]'
    echo 'Use --prepare-only to download official tasks without starting Gym.'
    echo 'Set TB4_USE_PREPARED=1 to reuse previously downloaded packages and prepared Gym data.'
    echo 'Overrides are forwarded to the runner; use ++benchmark_jsonl=PATH for a task list.'
    echo 'Example: bash scripts/run_oracles/tb4.sh 2'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
LIMIT=""
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    LIMIT="$1"
    shift
    if [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
        echo 'Task limit must be a positive integer.' >&2
        exit 2
    fi
fi
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$PWD/scripts/run_oracles/build-constraints.txt}"
OUTPUT_JSONL="results/terminal_bench_4_oracle.jsonl"
LOG_DIR="results/terminal_bench_4_oracle"
mkdir -p "$LOG_DIR"

# Read the dataset identity from Gym's own manifest, which also drives its prepare.py.
DATASET_REF="$(jq -r '.dataset + "@" + .ref' benchmarks/terminal_bench_4/manifest.json)"
echo "==> Gym revision: $(git rev-parse HEAD); TB4 dataset: $DATASET_REF"
echo '==> Preparing benchmark data...'
if [[ "${TB4_USE_PREPARED:-0}" == 1 ]]; then
    test -s benchmarks/terminal_bench_4/data/benchmark.jsonl
    test -d results/task-packages
    echo '==> Reusing previously prepared Gym data and official task packages.'
else
    SOURCE_DIR="$PWD/results/oracle-upstream/harbor-cdb76bae6dc88d5bca1c8f0754bbba300d6574b4"
    if [[ ! -d "$SOURCE_DIR/.git" ]]; then git init --quiet "$SOURCE_DIR"; fi
    if ! git -C "$SOURCE_DIR" cat-file -e cdb76bae6dc88d5bca1c8f0754bbba300d6574b4^{commit} 2>/dev/null; then
        git -C "$SOURCE_DIR" fetch --depth=1 https://github.com/laude-institute/harbor.git cdb76bae6dc88d5bca1c8f0754bbba300d6574b4
    fi
    git -C "$SOURCE_DIR" checkout --quiet --detach cdb76bae6dc88d5bca1c8f0754bbba300d6574b4
    uv run --locked --project "$SOURCE_DIR" harbor download "$DATASET_REF" \
        --cache --output-dir "$PWD/results/task-packages" 2>&1 | tee "$LOG_DIR/prepare.log"
    uv run --locked python benchmarks/terminal_bench_4/prepare.py 2>&1 | tee -a "$LOG_DIR/prepare.log"
fi
if "$PREPARE_ONLY"; then exit 0; fi

CONCURRENCY="$(wc -l < benchmarks/terminal_bench_4/data/benchmark.jsonl)"
if [[ -n "$LIMIT" ]] && (( LIMIT < CONCURRENCY )); then CONCURRENCY="$LIMIT"; fi
if (( CONCURRENCY < 1 )); then echo 'No tasks selected.' >&2; exit 2; fi

echo "==> Starting terminal_bench_4 (concurrency=$CONCURRENCY)..."
# The resources-only config retains the pinned CPU, GPU and Compose environments.
# Job control keeps SIGINT enabled for Gym's native shutdown and sandbox cleanup.
set -m
uv run --locked gym env start --config benchmarks/terminal_bench_4/resources.yaml \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    +tb4_concurrency="$CONCURRENCY" \
    +tb4_jobs_dir="$PWD/$LOG_DIR/resources" \
    +terminal_bench_4.resources_servers.terminal_bench_4.task_download_dir="$PWD/results/task-packages" \
    +terminal_bench_4.resources_servers.terminal_bench_4.is_verifying_golden_patch=true \
    > "$LOG_DIR/server.log" 2>&1 &
GYM_ENV_PID=$!
set +m
cleanup() {
    echo "==> Stopping terminal_bench_4 (pid $GYM_ENV_PID)..."
    kill -INT "$GYM_ENV_PID" 2>/dev/null || true
    wait "$GYM_ENV_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo '==> Waiting for the resources server to be ready...'
SERVER_READY=false
for i in $(seq 1 180); do
    if ! kill -0 "$GYM_ENV_PID" 2>/dev/null; then
        echo 'Gym exited before readiness.' >&2
        exit 1
    fi
    if uv run --locked gym env status --json "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" 2>/dev/null \
        | uv run --locked python scripts/run_oracles/gym_ready.py; then
        SERVER_READY=true
        break
    fi
    sleep 5
done
if ! "$SERVER_READY"; then
    echo 'Resources server never became ready after 15 minutes; aborting.' >&2
    exit 1
fi

echo '==> Running official solution scripts against every selected task (parallel, via Gym)...'
LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then LIMIT_ARGS=(+limit="$LIMIT"); fi
uv run --locked python resources_servers/terminal_bench_4/apply_golden_patch.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/terminal_bench_4/data/benchmark.jsonl \
    +output_fpath="$OUTPUT_JSONL" +concurrency="$CONCURRENCY" \
    "${LIMIT_ARGS[@]}" "$@" 2>&1 | tee "$LOG_DIR/oracle.log"
echo "==> Done. Oracle summary: $LOG_DIR/oracle.log"
