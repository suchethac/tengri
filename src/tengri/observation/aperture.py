# SPDX-License-Identifier: BSD-3-Clause
r"""Aperture-correction preprocessing for catalog photometry.

tengri's forward model assumes inputs are aperture-corrected (i.e., total
fluxes through each filter, integrated to infinity for the relevant source
profile). For catalog data measured in finite apertures (Petrosian, Kron,
fixed-radius), apply the per-band corrections produced by the survey
pipeline *before* feeding the photometry to ``Fitter``::

    flux_corr, noise_corr = apply_aperture_correction(flux_obs, noise, corrections)
    fitter = Fitter(model, flux_corr, noise_corr)

This is a thin utility, purely a per-band multiplication. It exists so
the convention is documented and verified rather than reinvented per
project.

References
----------

- Graham, A. W. & Driver, S. P., 2005, PASA, 22, 118 (Sersic-profile
  Petrosian → total-flux corrections).
- Kron, R. G., 1980, ApJS, 43, 305 (Kron-aperture concept).

"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["apply_aperture_correction"]


def apply_aperture_correction(
    flux: jnp.ndarray,
    noise: jnp.ndarray,
    corrections: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Apply per-band aperture corrections to flux and noise.

    Parameters
    ----------
    flux: array_like, shape (n_bands,)
        Observed flux density per band, before correction.
        [erg/s/cm^2/Hz] or any consistent unit.
    noise: array_like, shape (n_bands,)
        1-sigma noise per band, before correction. Same units as
        ``flux``.
    corrections: float or array_like, shape (n_bands,)
        Per-band multiplicative correction. Must be strictly positive.
        A scalar applies the same factor to all bands.

    Returns
    -------
    flux_corr: ndarray, shape (n_bands,)
        ``flux * corrections``. [same units as input ``flux``]
    noise_corr: ndarray, shape (n_bands,)
        ``noise * corrections``. [same units as input ``noise``]

    Raises
    ------
    ValueError
        If ``corrections`` has a shape incompatible with ``flux``, or
        if any correction value is non-positive.

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp`` arithmetic.

    Because correction is multiplicative on the flux scale and applied
    identically to ``noise``, the per-band signal-to-noise ratio is
    preserved:

    .. math::

        \mathrm{SNR}_{\rm corr} = \frac{c \cdot f}{c \cdot \sigma}
        = \frac{f}{\sigma} = \mathrm{SNR}.

    This means aperture correction shifts the absolute scale of any
    derived stellar-mass / SFR posterior but does *not* change the
    relative information content. For surveys that report aperture
    corrections with their own uncertainty, fold that uncertainty into
    the noise model upstream, the simple multiplicative form here
    assumes the corrections are deterministic.

    Examples
    --------
    >>> flux = jnp.array([1.0, 2.0, 3.0])
    >>> noise = jnp.array([0.1, 0.2, 0.3])
    >>> corr = jnp.array([1.10, 1.05, 1.02])  # per-band Sersic-Petrosian
    >>> flux_corr, noise_corr = apply_aperture_correction(flux, noise, corr)
    """
    flux_arr = jnp.asarray(flux)
    noise_arr = jnp.asarray(noise)
    corr_arr = jnp.asarray(corrections)

    if corr_arr.ndim == 0:
        if float(corr_arr) <= 0.0:
            raise ValueError("aperture correction must be positive (got <= 0)")
    else:
        if corr_arr.shape != flux_arr.shape:
            raise ValueError(f"corrections shape {corr_arr.shape} != flux shape {flux_arr.shape}")
        if not bool(np.all(np.asarray(corr_arr) > 0.0)):
            raise ValueError("aperture corrections must be positive (got <= 0 in some band)")

    return flux_arr * corr_arr, noise_arr * corr_arr
