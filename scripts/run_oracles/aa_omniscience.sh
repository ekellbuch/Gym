#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# AA-Omniscience Non-Hallucination: prepare the official public subset only.
# Usage: bash scripts/run_oracles/aa_omniscience.sh [--prepare-only]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Only preparation is supported: the exact 6,000-question set is private, and
# replaying gold answers makes the non-hallucination denominator zero.
# Public source: https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public
# Revision: e4883edbb9f5ccf2b2a8fdc6fb65e01a58e99849
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/aa_omniscience.sh [--prepare-only]'
    echo 'Prepares the official 600-question public subset. The exact AA Non-Hallucination metric uses a private 6,000-question set and has no model-free oracle runner. Gold-answer replay has no non-correct answers, so its hallucination-rate denominator is zero.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if [[ "$#" -ne 0 ]]; then
    echo "Unexpected argument: $1" >&2
    exit 2
fi

echo '==> Preparing benchmark data...'
uv run --locked python -c 'from benchmarks.omniscience.prepare import prepare; print(prepare())'
if "$PREPARE_ONLY"; then exit 0; fi

# The public answer field is a grading reference, not a model trajectory.
# An invented incorrect/abstaining response would manufacture this metric.
echo 'UPSTREAM PREPARATION COMPLETE; ORACLE EXECUTION UNSUPPORTED: Only 600 public questions are available, versus 6,000 in the exact AA Non-Hallucination benchmark. Gold-answer replay has no non-correct responses and cannot establish its hallucination rate. No official model-free oracle runner or score was produced.' >&2
exit 2
