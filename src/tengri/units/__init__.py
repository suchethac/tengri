# SPDX-License-Identifier: BSD-3-Clause
"""Unit conversions and magnitude system helpers.

Pure JAX, JIT/grad/vmap-safe. Re-exports from
:mod:`tengri.utils.conversions` and :mod:`tengri.utils.magnitudes` so
users can write ``from tengri import units`` without reaching into
``utils``.

Conventions
-----------

- Wavelength: Angstrom (vacuum throughout).
- L_nu: erg/s/Hz.
- F_nu: erg/s/cm^2/Hz (cgs); helpers convert to/from Jy, mJy, uJy, nJy,
  maggies, AB magnitudes, Vega magnitudes.
- Luminosity: erg/s and L_sun.

Examples
--------
>>> from tengri import units
>>> units.fnu_to_jy(1e-23)  # 1 erg/s/cm^2/Hz -> Jy
DeviceArray(1.0, dtype=float64)
>>> units.ab_mag_to_fnu(25.0)  # AB mag -> F_nu cgs
DeviceArray(3.6307805e-30, dtype=float64)
"""

from __future__ import annotations

from tengri.utils.conversions import (
    air_to_vacuum,
    attenuation_to_tau,
    erg_per_s_to_lsun,
    flambda_to_fnu,
    fnu_to_flambda,
    fnu_to_jy,
    fnu_to_lnu,
    fnu_to_maggies,
    fnu_to_njy,
    fnu_to_ujy,
    jy_to_fnu,
    llambda_to_lnu,
    lnu_to_fnu,
    lnu_to_llambda,
    lsun_to_erg_per_s,
    maggies_to_fnu,
    njy_to_fnu,
    tau_to_attenuation,
    ujy_to_fnu,
    vacuum_to_air,
)
from tengri.utils.magnitudes import (
    AB_VEGA_OFFSETS,
    ab_mag_to_fnu,
    ab_to_vega,
    absolute_ab_mag_to_lnu,
    absolute_to_apparent,
    apparent_to_absolute,
    cosmological_dimming,
    distance_modulus_from_dl,
    distance_modulus_from_dl_mpc,
    fnu_to_ab_mag,
    lnu_to_absolute_ab_mag,
    mag_to_surface_brightness,
    surface_brightness_to_mag,
    vega_to_ab,
)

__all__ = [
    # ``ab_to_vega``/``vega_to_ab`` take a float offset, not a band name, and
    # this is where the offsets come from. It is re-exported alongside them
    # because the module contract above is "users can write ``from tengri
    # import units`` without reaching into ``utils``" — and until #1613 the one
    # argument those two functions need was reachable only from ``utils``.
    "AB_VEGA_OFFSETS",
    "ab_mag_to_fnu",
    "ab_to_vega",
    "absolute_ab_mag_to_lnu",
    "absolute_to_apparent",
    "air_to_vacuum",
    "apparent_to_absolute",
    "attenuation_to_tau",
    "cosmological_dimming",
    "distance_modulus_from_dl",
    "distance_modulus_from_dl_mpc",
    "erg_per_s_to_lsun",
    "flambda_to_fnu",
    "fnu_to_ab_mag",
    "fnu_to_flambda",
    "fnu_to_jy",
    "fnu_to_lnu",
    "fnu_to_maggies",
    "fnu_to_njy",
    "fnu_to_ujy",
    "jy_to_fnu",
    "llambda_to_lnu",
    "lnu_to_absolute_ab_mag",
    "lnu_to_fnu",
    "lnu_to_llambda",
    "lsun_to_erg_per_s",
    "mag_to_surface_brightness",
    "maggies_to_fnu",
    "njy_to_fnu",
    "surface_brightness_to_mag",
    "tau_to_attenuation",
    "ujy_to_fnu",
    "vacuum_to_air",
    "vega_to_ab",
]

# Physical constants re-exported from their canonical sources.
# These are imported at the end to avoid circular import issues.
from tengri.parameters.translate import LOG10_ZSUN
from tengri.utils.physics_constants import C_AA

__all__ += ["C_AA", "LOG10_ZSUN"]
