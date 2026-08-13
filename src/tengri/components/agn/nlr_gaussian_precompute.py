# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for the AGN NLR Gaussian-line composer.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the Richardson+2014 narrow-line region template
(:func:`tengri.components.agn.nlr.compute_nlr_sed_richardson2014`).

Like the BLR adapter (see :mod:`tengri.components.agn.blr_precompute`), this
emitter is fundamentally *runtime-Gaussian*: line widths and the bolometric
disc luminosity feeding the NLR are tunable on every gradient step.  The only
work that can be moved to model build time is the filter projection — i.e. the
``(n_lines, n_filt)`` matrix obtained by evaluating each filter transmission
curve at the redshifted line center.  Runtime then weighs lines by the
Richardson+2014 ``a42`` flux template (normalized to H-beta = 1) and rescales
by ``covering_fraction × line_efficiency × L_disc``.

References
----------
.. [1] C. T. Richardson, J. T. Allen, J. A. Baldwin, P. C. Hewett, and
   G. J. Ferland, "Interpreting the ionization sequence in AGN emission-line
   spectra," MNRAS, 437, 3, 2376-2403 (2014). Table 3, column 'a42'.
   https://doi.org/10.1093/mnras/stt2056
.. [2] B. D. Johnson, et al., "Prospector: Inferring the Star Formation
   Histories of Galaxies from Observed Spectral Energy Distributions,"
   ApJS, 254, 22 (2021). https://doi.org/10.3847/1538-4365/abef67
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn.nlr import (
    _NLR_FWHM_KMS,
    _NLR_LINE_EFFICIENCY_DEFAULT as _NLR_LINE_EFFICIENCY,
    _RICHARDSON_FLUXES,
    _RICHARDSON_WAVES,
)

# NLR Gaussian composer has no grid axes — all parameters are runtime.
AXIS_PARAMS: tuple[str, ...] = ()


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    **kwargs: Any,
) -> dict:
    """Build the NLR line filter-projection matrix.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter observed-frame wavelengths [Angstrom].
    filter_trans : list[ndarray]
        Per-filter transmission (0–1).
    redshift : float
        Source redshift. [dimensionless]
    parameters : Parameters or None
        Unused — the NLR Gaussian composer has no grid axes.
    **kwargs : optional
        Ignored.

    Returns
    -------
    dict
        Keys: ``line_wavelengths_obs`` (n_lines,), ``line_strengths`` (n_lines,),
        ``line_weight_matrix`` (n_lines, n_filt), ``redshift`` (float).

    Notes
    -----
    **JIT-compatible**: no — build-time NumPy.
    """
    line_waves_rest = np.asarray(_RICHARDSON_WAVES, dtype=np.float64)
    line_strengths = np.asarray(_RICHARDSON_FLUXES, dtype=np.float64)
    line_waves_obs = line_waves_rest * (1.0 + float(redshift))

    n_lines = line_waves_obs.size
    n_filt = len(filter_waves)
    line_weight_matrix = np.zeros((n_lines, n_filt), dtype=np.float64)
    for i_line, w_obs in enumerate(line_waves_obs):
        for i_filt, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
            fw_arr = np.asarray(fw)
            ft_arr = np.asarray(ft)
            line_weight_matrix[i_line, i_filt] = np.interp(
                w_obs, fw_arr, ft_arr, left=0.0, right=0.0
            )

    return {
        "line_wavelengths_obs": jnp.asarray(line_waves_obs),
        "line_strengths": jnp.asarray(line_strengths),
        "line_weight_matrix": jnp.asarray(line_weight_matrix),
        "redshift": float(redshift),
    }


def build_lookup(preint: dict, **kwargs: Any):
    """Build the runtime NLR emission lookup.

    The returned closure has signature::

        fn(l_disc_bol_erg, covering_fraction, sigma_nlr_kms,
           line_efficiency=_NLR_LINE_EFFICIENCY) -> photometry

    Parameters
    ----------
    preint : dict
        Output of :func:`precompute`.
    **kwargs : optional
        Ignored.

    Returns
    -------
    callable
        JIT-compiled closure returning ``(n_filt,)`` band fluxes [erg/s/Hz].
        Wavelength integral is replaced by the precomputed projection
        matrix at line centers (delta-function approximation; runtime
        Gaussian widths affect only the spectral shape, not the band
        integrals to first order — exact for narrow lines whose FWHM
        is much smaller than the filter width).

    Notes
    -----
    **JIT-compatible**: yes — pure JAX.

    **Runtime parameters (NOT precomputed)**:

    - ``l_disc_bol_erg``: bolometric disc luminosity [erg/s].
    - ``covering_fraction``: NLR covering fraction [0, 1].
    - ``sigma_nlr_kms``: Gaussian line width [km/s] (default
      ``_NLR_FWHM_KMS / 2.355``).
    - ``line_efficiency``: fraction of intercepted L converted to lines.

    """
    line_strengths = preint["line_strengths"]
    line_weight_matrix = preint["line_weight_matrix"]

    flux_sum = jnp.maximum(jnp.sum(line_strengths), 1e-30)

    @jax.jit
    def predict_nlr_photometry(
        l_disc_bol_erg,
        covering_fraction,
        sigma_nlr_kms=_NLR_FWHM_KMS / 2.355,
        line_efficiency=_NLR_LINE_EFFICIENCY,
    ):
        l_intercepted = covering_fraction * l_disc_bol_erg
        l_lines_total = line_efficiency * l_intercepted
        l_per_flux = l_lines_total / flux_sum
        # Per-line luminosities (erg/s) → projected to filters via delta-function
        # weight matrix → erg/s/Hz at each filter (caller divides by filter width
        # if integrating over filter; matrix already encodes transmission).
        line_lum = line_strengths * l_per_flux
        # Suppress unused-variable lint while keeping signature meaningful.
        _ = sigma_nlr_kms
        return jnp.einsum("l,lf->f", line_lum, line_weight_matrix)

    return predict_nlr_photometry
