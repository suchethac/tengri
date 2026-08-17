# SPDX-License-Identifier: BSD-3-Clause
"""Unified DSPS-backed rest→observer-frame SED kernel (issue #398).

One function. Single source of truth for the rest→observer-frame SED
transform across the forward model. Built on top of DSPS's photometry
primitive at ``dsps.photometry.photometry_kernels._obs_flux_ssp`` and
tengri's own ``luminosity_distance`` wrapper around DSPS's cosmology.

Convention:

.. math::

   f_\\nu^{\\rm obs}(\\lambda_{\\rm obs})
   = L_\\nu^{\\rm rest}(\\lambda_{\\rm obs}/(1+z))
     \\times (1+z) / (4\\pi d_L^2)

The ``(1+z)`` factor on F_ν is the bandwidth (frequency specific-
intensity) correction — see Hogg 1999 (astro-ph/9905116) §4.

For a redshift table (precompute path), use ``jax.vmap`` directly::

    from jax import vmap

    shift_grid = vmap(shift_to_obs_frame, in_axes=(None, None, None, 0, None))
    f_nu_table = shift_grid(wave_rest, L_nu_rest, wave_obs, z_table, cosmo)

The migration is staged across multiple PRs — see
``docs/dev/audits/2026-05-26-redshift-sites-audit.md`` for the full
inventory and the per-component routing plan.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dsps.cosmology.flat_wcdm import CosmoParams

from tengri.cosmology import PLANCK18, luminosity_distance
from tengri.utils.scale import apply_log10_scale, log10_flux_scale

__all__ = ("shift_to_obs_frame",)


@jax.jit
def shift_to_obs_frame(
    wave_rest: jnp.ndarray,
    L_nu_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    z: jnp.ndarray,
    cosmo: CosmoParams = PLANCK18,
) -> jnp.ndarray:
    """Shift a rest-frame L_ν spectrum onto an observed-frame wavelength grid.

    Parameters
    ----------
    wave_rest : ndarray, shape (n_wave_rest,)
        Rest-frame wavelength grid [Ångstrom], strictly increasing.
    L_nu_rest : ndarray, shape (n_wave_rest,)
        Rest-frame specific luminosity [erg/s/Hz].
    wave_obs : ndarray, shape (n_wave_obs,)
        Observed-frame wavelength grid [Ångstrom] on which to return
        the result.
    z : scalar
        Redshift.
    cosmo : CosmoParams, optional
        Flat w₀wₐCDM parameters. Defaults to PLANCK18.

    Returns
    -------
    f_nu_obs : ndarray, shape (n_wave_obs,)
        Observed-frame specific flux [erg/s/cm²/Hz]. Zero outside the
        observed-frame support of ``wave_rest``.

    Notes
    -----
    **JIT/grad/vmap-safe.** Uses the same interp pattern as
    :func:`dsps.photometry.photometry_kernels._obs_flux_ssp` plus a
    range-safe ``(1+z) / (4π d_L²)`` F_ν conversion applied as a log10 offset
    via :func:`tengri.utils.scale.apply_log10_scale`. ``d_L`` comes from
    :func:`tengri.cosmology.luminosity_distance` (which delegates to
    DSPS and applies the standard 10-pc convention at z=0).

    References
    ----------
    .. [1] Hogg, D. W. "Distance measures in cosmology." 1999,
       arXiv:astro-ph/9905116.
    """
    dl_cm = luminosity_distance(z, cosmo=cosmo)
    L_on_obs_grid = jnp.interp(wave_obs, wave_rest * (1.0 + z), L_nu_rest, left=0.0, right=0.0)
    # (1+z)/(4π d_L²) as a log10 offset — never form d_L² (overflow) or
    # flux_scale (underflow) as standalone float32 values.
    return apply_log10_scale(L_on_obs_grid, log10_flux_scale(z, dl_cm))
