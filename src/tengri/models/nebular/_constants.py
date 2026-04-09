"""Physical constants for nebular emission calculations.

Fundamental constants (h, c, L_sun) are imported from
:mod:`tengri.utils.physics`, which documents their SI→CGS derivations and
primary references (CODATA 2018, IAU 2015).

Nebular-specific constants (Lyman limit, solar metallicity, CB19 oxygen
abundance offset) are defined here.

Note: cue.py intentionally uses ``L_SUN_CUE = 3.839e33`` (CUE neural-net
training convention) rather than the IAU 2015 value.  Do not replace that
constant.  See tengri.utils.physics_constants for the full explanation.
"""

from tengri.utils.physics_constants import (
    AA_TO_CM as _AA_TO_CM,  # noqa: F401 — re-exported for ionizing_spectrum.py
    C_AA as _C_AA,  # noqa: F401
    C_CGS as _C_CGS,  # noqa: F401
    H_PLANCK as _H_PLANCK,  # noqa: F401
    L_SUN as _LSUN_ERG,  # noqa: F401
)

# ---------------------------------------------------------------------------
# Nebular-specific constants
# ---------------------------------------------------------------------------

# Hydrogen Lyman limit [Angstrom]
_LYMAN_LIMIT: float = 911.76

# log10(Z_sun) — Asplund+2009, used by DSPS convention
_LOG10_ZSUN: float = -1.8477116556169435

# Oxygen abundance offset for CB19 CLOUDY c17.01 solar scale
# Derived as: log10(O/H)_solar − log10(Z_sun)
#   log10(O/H)_solar = −3.07  (Asplund+2009 Table 1, 12+log(O/H)=8.69
#                              → log(O/H)=−3.31+0.24 for Z/X scaling)
#   offset = −3.07 − (−1.848) = −1.222
_LOG_OH_SOLAR: float = -3.07
_LOG_OH_OFFSET: float = _LOG_OH_SOLAR - _LOG10_ZSUN  # ≈ -1.222
