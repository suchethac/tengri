# SPDX-License-Identifier: BSD-3-Clause
"""Tiny string helpers shared across the tree.

The Levenshtein primitive here is the standard dynamic-programming edit
distance: used to power "Did you mean: <closest>?" suggestions in
``forward/orchestrator.py`` and ``parameters/registry.py``. Keep both
suggestion sites pointed at this single primitive so their behavior
stays in lockstep.
"""

from __future__ import annotations

from collections.abc import Iterable


def levenshtein(a: str, b: str) -> int:
    """Plain Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def closest(target: str, options: Iterable[str], max_distance: int = 2) -> str | None:
    """Closest option by Levenshtein distance, ``None`` if nothing within ``max_distance``."""
    best_name: str | None = None
    best_dist = max_distance + 1
    for k in options:
        d = levenshtein(target, k)
        if d < best_dist:
            best_dist = d
            best_name = k
    return best_name
