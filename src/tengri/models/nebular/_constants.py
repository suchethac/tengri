"""Physical constants for nebular emission calculations.

Single source of truth. Import these instead of defining locally.

Note: cue.py intentionally uses _LSUN_ERG = 3.839e33 (Cue neural net convention)
rather than the IAU 2015 value here. Do not import _LSUN_ERG from this module
into cue.py.
"""

# Planck constant [erg s]
_H_PLANCK: float = 6.62607015e-27

# Speed of light [cm/s]
_C_CGS: float = 2.99792458e10

# Speed of light [Angstrom/s]  — for ionizing_spectrum.py
_C_AA: float = _C_CGS * 1e8

# Solar luminosity [erg/s] — IAU 2015 value
# NOTE: cue.py uses 3.839e33 (Cue training convention). Do NOT replace cue.py's value.
_LSUN_ERG: float = 3.828e33

# Hydrogen Lyman limit [Angstrom]
_LYMAN_LIMIT: float = 911.76

# log10(Z_sun) — Asplund+2009, used by DSPS convention
_LOG10_ZSUN: float = -1.8477116556169435

# Oxygen abundance offset for CB19 CLOUDY c17.01 solar scale
# Derived as: log10(O/H)_solar - _LOG10_ZSUN = -3.07 - (-1.848) ≈ -1.222
_LOG_OH_SOLAR: float = -3.07
_LOG_OH_OFFSET: float = _LOG_OH_SOLAR - _LOG10_ZSUN  # ≈ -1.222
