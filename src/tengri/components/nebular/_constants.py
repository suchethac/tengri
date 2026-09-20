# SPDX-License-Identifier: BSD-3-Clause
"""Physical and astrophysical constants for nebular emission module.

This module centralizes constant definitions used across nebular backends.
Fundamental constants (h, c, L_sun) are re-exported from
:mod:`tengri.utils.physics_constants` (SI→CGS conversions documented there;
sources: CODATA 2018, IAU 2015). Nebular-specific constants are defined here.

**Important**: cue.py intentionally uses L_SUN_CUE = 3.839e33 erg/s
(neural-net training convention), not the IAU 2015 value. This constant
is NOT re-exported from this module. See physics_constants.py docstring
for the rationale.
"""

from tengri.utils.physics_constants import (
    AA_TO_CM as _AA_TO_CM,  # noqa: F401  (re-exported for ionizing_spectrum.py)
    C_AA as _C_AA,  # noqa: F401
    C_CGS as _C_CGS,  # noqa: F401
    H_PLANCK as _H_PLANCK,  # noqa: F401
    L_SUN as _LSUN_ERG,  # noqa: F401
    LOG10_ZSUN as _LOG10_ZSUN,
)
from tengri.utils.ssp_anchor import (
    ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR,  # noqa: F401  (re-exported for _shared.py, #2418)
)

# ── Nebular-specific constants ────────────────────────────────────

# Hydrogen Lyman limit [Angstrom]
_LYMAN_LIMIT: float = 911.76

# Oxygen abundance offset for CB19 CLOUDY c17.01 solar scale
# Derived as: log10(O/H)_solar − log10(Z_sun)
#   log10(O/H)_solar = −3.07  (Asplund+2009 Table 1, 12+log(O/H)=8.69
#                              → log(O/H)=−3.31+0.24 for Z/X scaling)
#   offset = −3.07 − (−1.848) = −1.222
_LOG_OH_SOLAR: float = -3.07
_LOG_OH_OFFSET: float = _LOG_OH_SOLAR - _LOG10_ZSUN  # ≈ -1.222

# Optically thin thermal bremsstrahlung spectral index (L_nu ∝ nu^alpha) used
# to continue tabulated nebular continua past their last node (#2346).
# The same value as the radio block's Murphy+2011 `radio_freefree` default
# `alpha_ff`. pcigale's tabulated nebular continuum measures -0.096 over 1 cm
# to 1 m, so -0.1 is a reasonable nominal value for the free-free extension.
NEBULAR_FREEFREE_TAIL_ALPHA_NU: float = -0.1
