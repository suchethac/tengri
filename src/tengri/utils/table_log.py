"""Platform-independent logarithms of tabulated float32 data.

numpy evaluates ``log``/``log10`` of a float32 array with a SIMD kernel whose
results are not correctly rounded and differ between CPU families (x86_64
AVX-512 and Apple ARM disagree on about 2 % of a 6e6-entry table by 1 ulp).
A table built that way shifts every downstream value by the platform's
rounding pattern (#3067). Evaluating in float64 and rounding to float32 once
is deterministic on every platform.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def log_table_f32(
    table: np.ndarray,
    *,
    floor: float,
    base: Literal["e", "10"] = "e",
) -> np.ndarray:
    """Logarithm of ``max(table, floor)`` computed in float64, rounded once to float32.

    Parameters
    ----------
    table : ndarray
        Tabulated values, any shape and float dtype (typically float32 from HDF5).
    floor : float
        Lower clip applied before the logarithm; keeps zeros and underflows finite.
    base : {"e", "10"}
        Natural or decimal logarithm.

    Returns
    -------
    ndarray
        float32, same shape as ``table``. Equals
        ``np.log(np.maximum(table.astype(float64), floor)).astype(float32)``
        (or ``log10``) bit for bit on every platform.

    Notes
    -----
    **JIT-compatible**: no, eager numpy for load-time table construction.
    """
    wide = np.maximum(np.asarray(table, dtype=np.float64), floor)
    out = np.log10(wide) if base == "10" else np.log(wide)
    return out.astype(np.float32)
