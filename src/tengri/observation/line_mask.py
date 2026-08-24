# SPDX-License-Identifier: BSD-3-Clause
"""Boolean pixel mask for emission lines in observed-frame spectra."""

from __future__ import annotations

import numpy as np

from tengri.utils.physics_constants import C_KM_S as _C_KMS


def build_line_mask(
    wave_obs: np.ndarray,
    line_rest_waves: np.ndarray,
    redshift: float,
    line_sigmas_kms: np.ndarray | float = 150.0,
    n_sigma: float = 2.5,
    min_sigma_kms: float = 50.0,
) -> np.ndarray:
    """Build a boolean pixel mask around emission lines in an observed spectrum.

    Returns ``True`` for pixels within ``n_sigma`` line widths of any emission
    line.  Masked pixels should be excluded from continuum fitting; unmasked
    pixels are line-free continuum windows.

    Parameters
    ----------
    wave_obs: array_like, shape (n_pix,)
        Observed-frame wavelength array [Angstrom].
    line_rest_waves: array_like, shape (n_lines,)
        Rest-frame vacuum wavelengths of emission lines [Angstrom].
    redshift: float
        Galaxy redshift [dimensionless].
    line_sigmas_kms: array_like or float, optional
        Intrinsic line widths [km/s].  Scalar is broadcast to all lines.
        Default is 150.0 km/s.
    n_sigma: float, optional
        Half-width of the mask in units of ``sigma_ang`` per line.
        Default is 2.5.
    min_sigma_kms: float, optional
        Minimum velocity width [km/s] used to set the floor on the mask
        half-width, preventing zero-width windows.  Default is 50.0 km/s.

    Returns
    -------
    ndarray, shape (n_pix,)
        Boolean mask; ``True`` where a pixel overlaps an emission line window.

    Notes
    -----
    Implements the same algorithm as FastSpecFit
    ``LineMasker.linepix_and_contpix()`` (Moustakas et al. 2023 [1]_);
    validated against its output.

    Not JIT-compatible (uses NumPy; designed for pre-inference preprocessing).

    Algorithm: for each line *i* with observed-frame center
    :math:`\\lambda_{z,i} = \\lambda_{\\mathrm{rest},i} \\cdot (1 + z)`, compute

    .. math::

        \\sigma_{\\mathrm{ang},i} =
            \\frac{\\max(\\sigma_{\\mathrm{kms},i},\\, \\sigma_{\\min})}{c}
            \\cdot \\lambda_{z,i}

    and mask all pixels satisfying
    :math:`|\\lambda - \\lambda_{z,i}| \\leq n_\\sigma \\cdot \\sigma_{\\mathrm{ang},i}`.

    References
    ----------
    .. [1] Moustakas, J., Scholte, D., Dey, B., Khederlarian, A., 2023,
           "FastSpecFit: Fast spectral synthesis and emission-line fitting
           of DESI spectra", Astrophysics Source Code Library,
           record ascl:2308.005.
           https://ui.adsabs.harvard.edu/abs/2023ascl.soft08005M

    """
    wave_obs = np.asarray(wave_obs, dtype=float)
    line_rest_waves = np.asarray(line_rest_waves, dtype=float)
    sigmas = np.broadcast_to(np.asarray(line_sigmas_kms, dtype=float), line_rest_waves.shape)

    mask = np.zeros(wave_obs.shape[0], dtype=bool)
    z1 = 1.0 + redshift

    for rest_wave, sigma_kms in zip(line_rest_waves, sigmas):
        z_wave = rest_wave * z1
        sigma_ang = max(float(sigma_kms), min_sigma_kms) / _C_KMS * z_wave
        half_width = n_sigma * sigma_ang
        mask |= np.abs(wave_obs - z_wave) <= half_width

    return mask
