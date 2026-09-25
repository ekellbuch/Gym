#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Validate a selected official OSWorld task JSONL through Gym's existing
# preparation function. Upstream publishes task setup/evaluators but no
# executable reference actions; this command cannot produce an oracle score.
# Usage: bash scripts/run_oracles/osworld.sh [--prepare-only] [input-jsonl]
# The default input is Gym's five-task example, not the full verified split.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/osworld.sh [--prepare-only] [input-jsonl]'
    echo 'Validate a task JSONL (default: five checked-in examples). --prepare-only exits after validation.'
    echo 'No official OSWorld-Verified solution runner or complete reference actions are published.'
    echo 'Convert the pinned upstream test_all.json with benchmarks/osworld/tools/convert_osworld_tasks.py for the full task input.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if [[ $# -gt 1 || "${1:-}" == -* ]]; then
    echo 'Expected at most one input JSONL path after --prepare-only.' >&2
    exit 2
fi
INPUT_JSONL="${1:-benchmarks/osworld/data/example.jsonl}"

# Gym's prepare() validates task setup; it does not download a full split or
# turn evaluator metadata into actions.
echo "==> Validating OSWorld task input: $INPUT_JSONL"
uv run --locked python -c 'import sys; from pathlib import Path; from benchmarks.osworld.prepare import prepare; print(prepare(Path(sys.argv[1])))' "$INPUT_JSONL"
if "$PREPARE_ONLY"; then exit 0; fi

# A model run or third-party harvested script would not be an official gold
# trajectory. Keep the missing oracle explicit after task validation.
echo 'OSWorld oracle unavailable: upstream publishes task setup and evaluators, but no complete official executable reference-action trajectories or model-free solution runner.' >&2
exit 2
