# SPDX-License-Identifier: BSD-3-Clause
"""Pre-fit data-hygiene utilities: zero-point corrections, error floors, upper limits.

This module is a namespace seed. Future additions: Milky Way extinction,
photometry/spectroscopy cross-calibration, LSF attachment. Everything here
is pure numpy: safe to import before JAX initialization.
"""

from tengri.preprocessing.error_floor import add_systematic_floor
from tengri.preprocessing.upper_limits import (
    detect_upper_limits,
    sigma_upper_limit_from_flux,
)
from tengri.preprocessing.zeropoints import (
    ZEROPOINT_REGISTRY,
    ZeropointEntry,
    apply_zeropoints,
    lookup_zeropoints,
)

__all__ = [
    "ZEROPOINT_REGISTRY",
    "ZeropointEntry",
    "add_systematic_floor",
    "apply_zeropoints",
    "detect_upper_limits",
    "lookup_zeropoints",
    "sigma_upper_limit_from_flux",
]
