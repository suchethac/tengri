# SPDX-License-Identifier: BSD-3-Clause
"""Resident bytes of a precomputed table's stored arrays.

The byte budget for precomputed tables (#3505) reads ``nbytes`` on each table
type. Every such property sums through this one helper, so the rule lives in
one place: a ``None`` field counts 0, every other field must be an array (or a
nested table that carries its own ``nbytes``), and a non-array field fails
loudly instead of counting 0.
"""

from __future__ import annotations

from typing import Any

#: Python scalars a table stores as metadata (a float redshift, an int filter count).
_METADATA_SCALARS = (int, float, bool)


def resident_nbytes(*fields: Any) -> int:
    """Sum of ``.nbytes`` over the given array fields.

    ``None`` and Python scalars count 0. A nested table is passed as one field and
    contributes its own ``nbytes``, so it is counted once however it is reached.
    """
    total = 0
    for field in fields:
        if field is None or isinstance(field, _METADATA_SCALARS):
            continue
        total += int(field.nbytes)
    return total
