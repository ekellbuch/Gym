#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# bbq: prepare official upstream inputs/references; never synthesize solutions.
# Usage: bash scripts/run_oracles/bbq.sh [--prepare-only] [upstream overrides...]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Preparation is wired below. Later phases await an established upstream oracle command.
# Source: https://github.com/NVIDIA-NeMo/Gym.git @ 1c8261080bdc881b3e9b7f870e6418f160516991
# See SOURCES.json for the exact reference fields, variant, and remaining requirement.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/bbq.sh [--prepare-only] [upstream overrides...]'
    echo 'Current Gym BBQ resource is synthetic training task, not exact BBQ evaluation dataset. Reviewed BBQ source is internal and unpinned; no native preparation or oracle established. Do not substitute fixtures.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
echo 'PREPARATION NOT ESTABLISHED: Current Gym BBQ resource is synthetic training task, not exact BBQ evaluation dataset. Reviewed BBQ source is internal and unpinned; no native preparation or oracle established. Do not substitute fixtures.' >&2
exit 2
