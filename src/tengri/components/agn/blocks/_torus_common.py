# SPDX-License-Identifier: BSD-3-Clause
"""Shared constants for torus block modules."""

from __future__ import annotations

from tengri.utils.physics_constants import L_SUN

_C_AA_PER_S: float = 2.99792458e18
_RV_SMC: float = 2.93  # Pei 1992 SMC R_V (matches polar_dust.py)
_L_SUN_ERG: float = L_SUN  # already in erg/s, see physics_constants.py
