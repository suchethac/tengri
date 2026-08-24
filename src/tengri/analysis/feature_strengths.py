# SPDX-License-Identifier: BSD-3-Clause
r"""Clipped PAH feature strengths (Draine, Li, Hensley et al. 2021).

Implements the ``Fclip`` definition in Section 9.1 of Draine, Li,
Hensley, Hunt, Sandstrom & Smith (2021, ApJ 917, 3, arXiv:2011.07046):
the power in a PAH emission feature is integrated above a linear
baseline drawn between two clip wavelengths :math:`\lambda_1` and
:math:`\lambda_2`,

.. math::

    F_{\rm clip} = \int_{\nu_2}^{\nu_1}
        \big[\nu P_\nu(\lambda) - B(\lambda)\big]\,d\ln\nu

where :math:`B(\lambda)` is the linear interpolation in
:math:`\ln \nu` between the two endpoint values
:math:`\nu_1 P_{\nu}(\lambda_1)` and :math:`\nu_2 P_{\nu}(\lambda_2)`.

Reference
---------
.. [1] Draine, B.T., Li, A., Hensley, B.S., Hunt, L.K., Sandstrom, K.,
   Smith, J.-D.T., 2021, "Excitation of Polycyclic Aromatic Hydrocarbon
   Emission: Dependence on Size Distribution, Ionization, and Starlight
   Spectrum and Intensity", ApJ, 917, 3.  arXiv:2011.07046.
   DOI: 10.3847/1538-4357/abff51.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["TABLE5_FEATURES", "clip_feature", "total_ir_power"]


# Table 5 of Draine+2021: clip points for the 5 selected PAH features.
# Standard model (qpah=0.0379, mMMP, U=1, st ion, std size).
TABLE5_FEATURES: dict[str, dict[str, float]] = {
    "3.3": {"lam1_um": 3.09, "lam2_um": 3.52, "fclip_over_ftir": 0.0044},
    "6.2": {"lam1_um": 5.90, "lam2_um": 6.50, "fclip_over_ftir": 0.0130},
    "7.7": {"lam1_um": 6.90, "lam2_um": 9.70, "fclip_over_ftir": 0.0394},
    "11.2": {"lam1_um": 10.80, "lam2_um": 11.70, "fclip_over_ftir": 0.0103},
    "17": {"lam1_um": 15.50, "lam2_um": 18.50, "fclip_over_ftir": 0.0061},
}


def _validate_pair(
    wavelength_um: NDArray[np.floating],
    nu_pnu: NDArray[np.floating],
) -> None:
    if wavelength_um.shape != nu_pnu.shape:
        raise ValueError(
            f"shape mismatch: wavelength {wavelength_um.shape} vs nu_pnu {nu_pnu.shape}"
        )
    if wavelength_um.ndim != 1:
        raise ValueError("wavelength_um must be 1-D")


def total_ir_power(
    wavelength_um: ArrayLike,
    nu_pnu: ArrayLike,
) -> float:
    r"""Integrate :math:`\nu P_\nu` over :math:`\ln \nu` (the IR power).

    Parameters
    ----------
    wavelength_um : array_like, shape (n_wave,)
        Wavelength grid in microns; strictly increasing.
    nu_pnu : array_like, shape (n_wave,)
        :math:`\nu P_\nu` in any per-frequency units; integrated over
        :math:`d\ln\nu` (equivalent to :math:`\int P_\nu\,d\nu`).

    Returns
    -------
    float
        :math:`\int \nu P_\nu \, d\ln\nu` over the full provided grid.
    """
    wave_um = np.asarray(wavelength_um, dtype=np.float64)
    spec = np.asarray(nu_pnu, dtype=np.float64)
    _validate_pair(wave_um, spec)
    # nu = c/lam, so d(ln nu) = -d(ln lam).  Integrating from
    # nu_min..nu_max maps to lam_max..lam_min, and the two sign flips
    # cancel: int_{nu_min}^{nu_max} f d(ln nu) = int_{lam_min}^{lam_max}
    # f d(ln lam).
    return float(np.trapezoid(spec, np.log(wave_um)))


def clip_feature(
    wavelength_um: ArrayLike,
    nu_pnu: ArrayLike,
    lam1_um: float,
    lam2_um: float,
) -> float:
    r"""Clipped PAH feature strength (Draine+2021 Section 9.1).

    The "clipped" power is the integral of
    :math:`\nu P_\nu(\lambda) - B(\lambda)` over
    :math:`d\ln\nu` between the two clip wavelengths, where
    :math:`B(\lambda)` is the linear-in-:math:`\ln\nu` baseline
    connecting the spectrum at the two endpoints.

    Parameters
    ----------
    wavelength_um : array_like, shape (n_wave,)
        Wavelength grid in microns; strictly increasing.
    nu_pnu : array_like, shape (n_wave,)
        :math:`\nu P_\nu` spectrum on the same grid.
    lam1_um, lam2_um : float
        Clip points in microns, with ``lam1_um < lam2_um``; both must
        lie within the provided grid.

    Returns
    -------
    float
        :math:`F_{\rm clip}` in the same per-frequency units as
        ``nu_pnu`` (the unit of :math:`\nu P_\nu` integrated over
        :math:`\ln\nu`).

    Raises
    ------
    ValueError
        If ``lam1_um >= lam2_um``, if either endpoint is outside the
        grid, or if input shapes are inconsistent.

    Notes
    -----
    **JIT-compatible**: no; uses ``numpy.searchsorted`` with Python
    branches.  Used in tests and analysis only, not in the JAX forward
    model.
    """
    wave_um = np.asarray(wavelength_um, dtype=np.float64)
    spec = np.asarray(nu_pnu, dtype=np.float64)
    _validate_pair(wave_um, spec)

    if not (lam1_um < lam2_um):
        raise ValueError(f"lam1_um must be < lam2_um (got {lam1_um}, {lam2_um})")
    if lam1_um < wave_um[0] or lam2_um > wave_um[-1]:
        raise ValueError(
            f"clip range [{lam1_um}, {lam2_um}] um outside data grid "
            f"[{wave_um[0]:.3f}, {wave_um[-1]:.3f}]"
        )

    # Endpoint values via linear interpolation in lambda.
    y1 = float(np.interp(lam1_um, wave_um, spec))
    y2 = float(np.interp(lam2_um, wave_um, spec))

    # Build a working grid that includes the clip endpoints exactly so
    # the trapezoid integral is well-defined right at the boundaries.
    mask = (wave_um > lam1_um) & (wave_um < lam2_um)
    inner_lam = wave_um[mask]
    inner_y = spec[mask]
    grid_lam = np.concatenate(([lam1_um], inner_lam, [lam2_um]))
    grid_y = np.concatenate(([y1], inner_y, [y2]))

    # Linear-in-ln(nu) baseline = linear-in-ln(lam) since ln nu = -ln lam + C.
    ln_lam = np.log(grid_lam)
    t = (ln_lam - ln_lam[0]) / (ln_lam[-1] - ln_lam[0])
    baseline = (1.0 - t) * y1 + t * y2

    excess = grid_y - baseline
    # See ``total_ir_power``: int f d(ln nu) = int f d(ln lam) for
    # increasing-lambda integration limits.
    return float(np.trapezoid(excess, ln_lam))
