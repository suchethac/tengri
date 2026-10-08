# SPDX-License-Identifier: BSD-3-Clause
"""The one air <-> vacuum wavelength converter pair.

Every wavelength inside tengri is a **vacuum** wavelength, matching FSPS, SDSS,
DESI and JWST. A wavelength published in air (the Lick/Lick-IDS index windows,
Cloudy and PyNeb line labels above 2000 Angstrom, the MILES library as
observed) is converted exactly once, at ingestion, with :func:`air_to_vac`.
:func:`vac_to_air` exists for the reverse direction (labels, plots, exports).
No other module may carry its own refractive-index formula; a sweep test
(``tests/regression/bug/test_bug_vacuum_wavelengths_everywhere.py``) enforces
that.

Both functions accept ``numpy`` or JAX arrays and Python scalars. A ``numpy``
or scalar input returns ``numpy`` ``float64``; a JAX array or tracer input
returns a JAX array, traceable under ``jit`` and differentiable.

Notes
-----
IAU standard (Morton 2000, ApJS 130, 403, Eq. 8; Ciddor 1996, Appl. Opt. 35,
1566), the form used by SDSS and ``specutils``. With the vacuum wavenumber
:math:`s = 10^4/\\lambda_{\\rm vac}` (:math:`\\lambda` in Angstrom, so
:math:`s` is in :math:`\\mu{\\rm m}^{-1}`), the refractive index of standard
air is

.. math::
    n = 1 + 8.34254\\times10^{-5}
          + \\frac{2.406147\\times10^{-2}}{130 - s^2}
          + \\frac{1.5998\\times10^{-4}}{38.9 - s^2},

and :math:`\\lambda_{\\rm air} = \\lambda_{\\rm vac}/n(\\lambda_{\\rm vac})`.
This is a closed form in the vacuum wavelength. Its inverse has no closed form
in the air wavelength, so :func:`air_to_vac` solves
:math:`\\lambda_{\\rm vac} = n(\\lambda_{\\rm vac})\\,\\lambda_{\\rm air}` by
fixed-point iteration. The map is a contraction with ratio
:math:`|\\lambda\\,dn/d\\lambda| \\lesssim 10^{-5}`, so four iterations leave
a residual below :math:`10^{-12}` Angstrom; the iteration count is static, so
the function stays traceable and exactly differentiable.

IAU convention: the conversion is applied only for vacuum
:math:`\\lambda \\ge 2000` Angstrom; shorter wavelengths are returned
unchanged, because vacuum-ultraviolet wavelengths are always quoted in
vacuum. The cutoff is defined once, in the vacuum frame, so the pair is an
exact bijection: :func:`vac_to_air` switches on at 2000 Angstrom vacuum and
:func:`air_to_vac` at its image, 1999.35 Angstrom air (:data:`AIR_CUTOFF_AA`).
Below those thresholds both are the identity.

The formula describes standard air (15 C, 101325 Pa, 450 ppm CO2) and is
quoted by Morton (2000) for 2000 Angstrom to 2.5 micron; beyond that it is
the conventional extrapolation. Float64 is required for 1e-6 Angstrom
accuracy (the correction is 3e-4 of the wavelength).

References
----------
Morton, D. C. 2000, ApJS, 130, 403 (Eq. 8; the IAU standard).
Ciddor, P. E. 1996, Appl. Opt., 35, 1566.
Greisen, E. W. et al. 2006, A&A, 446, 747 (FITS WCS Paper III, Sect. 7).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["AIR_CUTOFF_AA", "AIR_VACUUM_CUTOFF_AA", "air_to_vac", "vac_to_air"]

#: IAU cutoff [Angstrom, vacuum]: below this wavelength no conversion is applied.
AIR_VACUUM_CUTOFF_AA: float = 2000.0

# Morton (2000) Eq. 8 / Ciddor (1996) refractivity of standard air.
_N0 = 8.34254e-5
_A1, _B1 = 2.406147e-2, 130.0
_A2, _B2 = 1.5998e-4, 38.9

#: The IAU cutoff expressed in air [Angstrom]: the image of 2000 Angstrom vacuum.
AIR_CUTOFF_AA: float = AIR_VACUUM_CUTOFF_AA / (
    1.0
    + _N0
    + _A1 / (_B1 - (1.0e4 / AIR_VACUUM_CUTOFF_AA) ** 2)
    + _A2 / (_B2 - (1.0e4 / AIR_VACUUM_CUTOFF_AA) ** 2)
)

#: Fixed-point iterations of :func:`air_to_vac`; contraction ratio ~1e-5.
_N_ITER = 4


def _namespace(wavelength):
    """Array namespace for the input: JAX arrays and tracers stay JAX."""
    return jnp if isinstance(wavelength, jax.Array) else np


def _refractive_index(wavelength_vac):
    """Refractive index of standard air at a vacuum wavelength [Angstrom]."""
    s2 = (1.0e4 / wavelength_vac) ** 2
    return 1.0 + _N0 + _A1 / (_B1 - s2) + _A2 / (_B2 - s2)


def vac_to_air(wavelength):
    """Convert vacuum wavelengths to air wavelengths [Angstrom].

    Parameters
    ----------
    wavelength : array_like
        Vacuum wavelengths [Angstrom]. Values below 2000 Angstrom are
        returned unchanged (IAU convention).

    Returns
    -------
    ndarray
        Air wavelengths [Angstrom], same shape as the input.

    Notes
    -----
    Closed form, Morton (2000, ApJS 130, 403, Eq. 8) with the Ciddor (1996)
    refractive index: ``lambda_air = lambda_vac / n(lambda_vac)``. See the
    module docstring for the formula. JAX-traceable and differentiable.

    Examples
    --------
    >>> round(float(vac_to_air(6564.61)), 2)
    6562.8
    """
    xp = _namespace(wavelength)
    wave = xp.asarray(wavelength, dtype=xp.float64)
    converted = wave / _refractive_index(wave)
    return xp.where(wave >= AIR_VACUUM_CUTOFF_AA, converted, wave)


def air_to_vac(wavelength):
    """Convert air wavelengths to vacuum wavelengths [Angstrom].

    Parameters
    ----------
    wavelength : array_like
        Air wavelengths [Angstrom]. Values below 1999.35 Angstrom (the air
        image of the 2000 Angstrom vacuum cutoff) are returned unchanged.

    Returns
    -------
    ndarray
        Vacuum wavelengths [Angstrom], same shape as the input.

    Notes
    -----
    Exact inverse of :func:`vac_to_air`: the root of
    ``lambda_vac = n(lambda_vac) * lambda_air`` found by four fixed-point
    iterations (residual < 1e-12 Angstrom). See the module docstring for the
    formula. JAX-traceable and differentiable.

    Examples
    --------
    >>> round(float(air_to_vac(6562.80)), 2)
    6564.61
    """
    xp = _namespace(wavelength)
    wave_air = xp.asarray(wavelength, dtype=xp.float64)
    above = wave_air >= AIR_CUTOFF_AA
    # Below the cutoff the iteration is evaluated on a safe stand-in so the
    # unused branch never produces a non-finite value (or gradient).
    safe = xp.where(above, wave_air, AIR_CUTOFF_AA)
    wave_vac = safe
    for _ in range(_N_ITER):
        wave_vac = safe * _refractive_index(wave_vac)
    return xp.where(above, wave_vac, wave_air)
