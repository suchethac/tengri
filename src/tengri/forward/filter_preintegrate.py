# SPDX-License-Identifier: BSD-3-Clause
"""Filter pre-integration utility for fast orchestrator photometry.

The orchestrator path's cold-compile cost (~500 ms) is dominated by
XLA compiling the 8 MB SSP-grid einsum (n_met × n_age × n_wave =
15 × 93 × 5994 doubles for the PRSC-MILES grid). When the science
target is photometry-only (not a full SED), the wavelength axis can
be collapsed at construction time by integrating each SSP through
each filter's transmission curve. The result is an
``(n_met, n_age, n_filters)`` array that is **~200× smaller** for a
typical SDSS-class survey, and the orchestrator chain that consumes
it compiles correspondingly faster.

This module exposes the **ingredient** — a JAX-compatible
``preintegrate_ssp_filter_grid`` function. Wiring it into a
photometry-mode StellarSEDComponent variant (which would skip
allocating the full ``lnu_age`` cube and produce ``state.photometry``
directly) is left to a follow-up architectural pass: it requires
parallel "photometry-mode" implementations of the downstream Dust /
Nebular / AGN / Radio / X-ray adapters, since they currently all
operate on the wavelength-resolved SED.

Usage today
-----------

::

    from tengri.forward.filter_preintegrate import preintegrate_ssp_filter_grid

    ssp_phot = preintegrate_ssp_filter_grid(
        ssp_data=ssp,
        filter_waves=observation.photometry.filter_waves,
        filter_trans=observation.photometry.filter_trans,
        redshift=0.05,
    )
    # ssp_phot.shape == (n_met, n_age, n_filters)
    # Multiply by joint weights and broadcast: total_mass × einsum("ma,maf->f", joint, ssp_phot)
    # gives photometry directly in erg/s/cm²/Hz × 4π dl² / (1+z) (i.e. the SSP photometric
    # normalization; convert to apparent flux by dividing by 4π dl² and multiplying by (1+z)).

Once the photometry-mode component cohort is built, it will consume
this output via the same einsum pattern but with ``f`` (n_filters,
~10-100) replacing ``w`` (n_wave, ~5000-10000).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["preintegrate_ssp_filter_grid"]


def preintegrate_ssp_filter_grid(
    ssp_data: Any,
    filter_waves: Sequence[jnp.ndarray],
    filter_trans: Sequence[jnp.ndarray],
    redshift: float = 0.0,
) -> jnp.ndarray:
    r"""Convolve every SSP template with every filter at construction time.

    For each ``(metallicity, age, filter)`` triple, computes the
    filter-weighted mean of the SSP flux:

    .. math::

        \mathrm{SSP\_phot}[m, a, f] =
        \frac{\int L_\nu^{(m,a)}(\lambda_\mathrm{rest})
              T_f(\lambda_\mathrm{obs})\,
              \lambda_\mathrm{obs}\,d\lambda_\mathrm{obs}}
             {\int T_f(\lambda_\mathrm{obs})\,
              \lambda_\mathrm{obs}\,d\lambda_\mathrm{obs}}

    where :math:`\lambda_\mathrm{obs} = (1+z)\lambda_\mathrm{rest}`.
    Identical to the per-filter integral in
    :func:`tengri.observation.photometry.compute_flux_density` but
    without the ``(1+z)/4\pi d_L^2`` source→observer scaling — that
    is applied later when the SSP grid is combined with mass-per-bin
    weights.

    Parameters
    ----------
    ssp_data : SSPData
        Full SSP grid (n_met, n_age, n_wave) in Lsun/Hz/Msun on
        the rest-frame ``ssp_data.ssp_wave`` array.
    filter_waves : sequence of array_like
        Per-filter wavelength arrays in Å (observed-frame already
        if you precompute for a fixed redshift; otherwise rest-frame
        and the function redshifts the SSP wave grid by ``1+z``).
    filter_trans : sequence of array_like
        Per-filter transmission curves matching ``filter_waves``.
    redshift : float, optional
        Source redshift. Default 0.0. The function redshifts the
        SSP wavelength grid (multiplies by ``1+z``) before
        interpolating onto each filter's wavelength array, which is
        the same convention used by
        :func:`tengri.observation.photometry.compute_flux_density`.

    Returns
    -------
    ssp_phot : ndarray, shape (n_met, n_age, n_filters)
        Filter-integrated SSP grid in Lsun/Hz/Msun. Multiply by
        mass-per-age-bin (Msun) and the source→observer factor
        ``(1+z) / (4 π d_L²)`` to get observed flux densities.

    Notes
    -----
    **JIT-compatible**: yes — pure JAX. **Eager-recommended**:
    typically called once at construction time, so the JIT overhead
    of compiling the per-filter loop isn't worth paying.

    For SDSS ugriz (5 filters), n_wave=5994 reduces to n_filters=5,
    a 1200× compression of the per-met-per-age axis. The resulting
    XLA graph compiled into the orchestrator's stellar component is
    correspondingly smaller. Wiring this into a photometry-mode
    chain is the architectural follow-up.
    """
    ssp_wave = jnp.asarray(ssp_data.ssp_wave)
    ssp_flux = jnp.asarray(ssp_data.ssp_flux)  # (n_met, n_age, n_wave)
    z = jnp.asarray(redshift)
    wave_obs = ssp_wave * (1.0 + z)

    # The single-SSP filter integral has the same shape as the
    # ``compute_flux_density`` integrand, sans the (1+z)/4πd_L² factor.
    # This is broadcast-vmapped across (m, a) by jnp.apply_along_axis-style
    # einsum.
    def _single_ssp_phot(sed_rest):
        """Integrate one SSP through every filter."""

        def _per_filter(fw, ft):
            sed_at_filter = jnp.interp(fw, wave_obs, sed_rest, left=0.0, right=0.0)
            num = jnp.trapezoid(sed_at_filter * ft * fw, fw)
            den = jnp.trapezoid(ft * fw, fw)
            return num / jnp.maximum(den, 1e-30)

        return jnp.asarray(
            [_per_filter(fw, ft) for fw, ft in zip(filter_waves, filter_trans, strict=False)]
        )

    # Vectorize over the (n_met, n_age) leading dims via vmap.
    n_met, n_age, _ = ssp_flux.shape
    flat = ssp_flux.reshape(n_met * n_age, -1)
    phot_flat = jax.vmap(_single_ssp_phot)(flat)
    return phot_flat.reshape(n_met, n_age, -1)
