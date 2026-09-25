#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run upstream SWE-bench Pro preparation and apply_golden_patch.py through Gym.
# Preparation downloads the pinned public tasks, gold patches, and test assets.
# Usage: bash scripts/run_oracles/swebench_pro.sh [--prepare-only] [limit] [Gym overrides...]
# Example: bash scripts/run_oracles/swebench_pro.sh 2
# The optional positive task limit belongs to this script; remaining arguments
# are forwarded verbatim to upstream apply_golden_patch.py. Its default four
# workers verify selected tasks in parallel. Requires uv and OpenSandbox access.
# Results: results/swebench_pro_oracle.jsonl; logs: results/swebench_pro_oracle/.
# Upstream preparation and verification are unchanged; no local oracle logic.

set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: bash scripts/run_oracles/swebench_pro.sh [--prepare-only] [limit] [Gym overrides...]
Prepare pinned upstream SWE-bench Pro data, start Gym, and verify upstream gold patches.
--prepare-only downloads/prepares data and exits without starting Gym or verification.
The optional positive limit selects tasks; all remaining arguments go to upstream
apply_golden_patch.py. Its default four workers verify tasks in parallel.
Requires uv and OPENSANDBOX_DOMAIN/OPENSANDBOX_API_KEY for verification.
Results: results/swebench_pro_oracle.jsonl; logs: results/swebench_pro_oracle/.
Example: bash scripts/run_oracles/swebench_pro.sh 2
HELP
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then
    PREPARE_ONLY=true
    shift
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$(pwd)"
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$REPO_ROOT/scripts/run_oracles/build-constraints.txt}"
# Ray's uv hook would package this entire local checkout (including worktrees)
# before the head server starts; these Gym workers already run in this checkout.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

LIMIT_ARGS=()
LIMIT=""
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    LIMIT="$1"
    shift
    if [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
        echo "The task limit must be positive." >&2
        exit 2
    fi
    LIMIT_ARGS=(+limit="$LIMIT")
fi
BENCHMARK_JSONL="benchmarks/swebench/data/swebench_pro_benchmark.jsonl"
OUTPUT_JSONL="results/swebench_pro_oracle.jsonl"
LOG_DIR="results/swebench_pro_oracle"
PORT_CONFIG="scripts/run_oracles/swebench_pro_ports.yaml"
mkdir -p "$LOG_DIR"

echo "==> Preparing upstream SWE-bench Pro data..."
uv run --locked gym eval prepare --config benchmarks/swebench/pro/opencode.yaml \
    2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then
    echo "==> Prepared upstream data: $BENCHMARK_JSONL; log: $LOG_DIR/prepare.log"
    exit 0
fi
TASK_COUNT="$(wc -l < "$BENCHMARK_JSONL")"
if [[ -n "$LIMIT" ]] && (( LIMIT < TASK_COUNT )); then
    TASK_COUNT="$LIMIT"
fi
if (( TASK_COUNT == 0 )); then
    echo "Preparation produced no tasks: $BENCHMARK_JSONL" >&2
    exit 1
fi

# Gym owns sandbox creation and the verifier. The golden-patch flag makes it
# use each prepared task's upstream patch instead of an agent-generated patch.
echo "==> Starting swebench_pro_resources_server..."
# Keep SIGINT enabled for the background job so Gym can shut down its workers.
set -m
uv run --locked gym env start \
    --config resources_servers/swebench_pro/configs/swebench_pro.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config "$PORT_CONFIG" \
    +swebench_pro_resources_server.resources_servers.swebench_pro.is_verifying_golden_patch=true \
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

echo "==> Waiting for Gym readiness (first run installs server dependencies)..."
SERVER_READY=false
for i in $(seq 1 180); do
    if ! kill -0 "$GYM_ENV_PID" 2>/dev/null; then
        echo "Gym exited before readiness; see $LOG_DIR/server.log" >&2
        exit 1
    fi
    if uv run --locked gym env status --json "+config_paths=[$PORT_CONFIG]" 2>/dev/null \
        | uv run --locked python scripts/run_oracles/gym_ready.py; then
        SERVER_READY=true
        break
    fi
    sleep 5
done
if ! "$SERVER_READY"; then
    echo "Gym was not ready after 15 minutes; see $LOG_DIR/server.log" >&2
    exit 1
fi

# Upstream owns request construction, verification, result serialization, and
# its default four-worker concurrency bound.
echo "==> Verifying $TASK_COUNT upstream golden patches (four parallel workers)..."
uv run --locked python resources_servers/swebench_pro/apply_golden_patch.py \
    "+config_paths=[$PORT_CONFIG]" \
    +benchmark_jsonl="$BENCHMARK_JSONL" \
    +output_fpath="$OUTPUT_JSONL" \
    "${LIMIT_ARGS[@]}" "$@" \
    2>&1 | tee "$LOG_DIR/oracle.log"
echo "==> Upstream oracle finished. Results: $OUTPUT_JSONL; logs: $LOG_DIR"
