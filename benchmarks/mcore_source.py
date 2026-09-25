# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serialize acquisition of a pinned VLMEvalKitMcore source checkout."""

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def locked_mcore_source(checkout: Path) -> Iterator[None]:
    """Lock the shared cache until its caller has verified commit and file hashes.

    The lock file stays beside the checkout so independent benchmark prepares
    using the same pinned commit serialize their existence check and rename.
    """
    checkout.parent.mkdir(parents=True, exist_ok=True)
    lock_path = checkout.with_name(checkout.name + ".lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
