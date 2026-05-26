# SPDX-License-Identifier: BSD-3-Clause
"""Unified DSPS-backed rest→observer-frame SED kernel (issue #398).

Single source of truth for the rest→observer-frame SED transform.
Every forward-pass call site that previously rolled its own
``wave_obs = wave_rest * (1+z)`` interp routes through
:func:`shift_to_obs_frame`. The kernel is built on top of DSPS's
photometry primitive at
``dsps.photometry.photometry_kernels._obs_flux_ssp`` — same arithmetic,
single canonical implementation.

The migration is staged across multiple PRs (see
``docs/dev/audits/2026-05-26-redshift-sites-audit.md``). This file
introduces the kernel + the DSPS parity test; downstream PRs migrate
the call sites one component at a time.

Cosmological conventions:

- Wavelength shift:  λ_obs = λ_rest × (1 + z)
- L_ν → F_ν:        f_ν,obs(λ_obs) = L_ν,rest(λ_rest) × (1 + z) / (4π d_L²)

Both are the standard astrophysical conventions; the ``(1+z)`` factor
on F_ν is the bandwidth (specific-intensity in frequency) correction —
see Hogg 1999 (astro-ph/9905116) §4 for the derivation. The kernel
returns the SED on the user-supplied observed-frame wavelength grid by
``jnp.interp`` (zero outside the support), matching what
:func:`dsps.photometry.photometry_kernels._obs_flux_ssp` does at its
core (filter integration is the user's job; this kernel only ships the
SED transform).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dsps.cosmology.flat_wcdm import CosmoParams, luminosity_distance_to_z
from jax import vmap

from tengri.utils.cosmology import DEFAULT_COSMO

__all__ = ("shift_to_obs_frame", "shift_to_obs_frame_grid")


@jax.jit
def _luminosity_distance_cm(z: jnp.ndarray, c: CosmoParams) -> jnp.ndarray:
    """Luminosity distance in cm. At z<=0, returns 10 pc (absolute-mag convention)."""
    dl_mpc = luminosity_distance_to_z(z, c.Om0, c.w0, c.wa, c.h)
    # 10 pc = 1e-5 Mpc; matches utils/cosmology.py convention.
    dl_mpc = jnp.where(z <= 0.0, 1e-5, dl_mpc)
    # 1 Mpc = 3.0856775814913673e24 cm — matches tengri.utils.cosmology.MPC_CM.
    return dl_mpc * 3.0856775814913673e24


@jax.jit
def shift_to_obs_frame(
    wave_rest: jnp.ndarray,
    L_nu_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    z: jnp.ndarray,
    cosmo: CosmoParams = DEFAULT_COSMO,
) -> jnp.ndarray:
    """Shift a rest-frame L_ν spectrum onto an observed-frame wavelength grid.

    Returns the observed-frame F_ν on the supplied ``wave_obs`` grid:

    .. math::

       f_\\nu^{\\rm obs}(\\lambda_{\\rm obs})
       = L_\\nu^{\\rm rest}(\\lambda_{\\rm obs}/(1+z))
         \\times (1+z) / (4\\pi d_L^2)

    Parameters
    ----------
    wave_rest : ndarray, shape (n_wave_rest,)
        Rest-frame wavelength grid [Ångstrom]. Must be strictly
        monotonically increasing.
    L_nu_rest : ndarray, shape (n_wave_rest,)
        Rest-frame specific luminosity at ``wave_rest`` [erg/s/Hz].
    wave_obs : ndarray, shape (n_wave_obs,)
        Observed-frame wavelength grid [Ångstrom] on which to return
        the result. May be a subset of, equal to, or finer/coarser
        than ``wave_rest * (1+z)``.
    z : scalar or shape () jnp.ndarray
        Redshift.
    cosmo : CosmoParams, optional
        Flat w₀wₐCDM parameters. Defaults to :data:`tengri.cosmology.PLANCK18`.

    Returns
    -------
    f_nu_obs : ndarray, shape (n_wave_obs,)
        Observed-frame specific flux at ``wave_obs`` [erg/s/cm²/Hz].
        Zero outside the observed-frame support of ``wave_rest``.

    Notes
    -----
    **JIT-compatible**: yes — pure JAX, fully traceable through
    grad/vmap.

    **Cosmology**: the ``cosmo`` argument is the DSPS
    ``CosmoParams`` dataclass; tengri's ``PLANCK18`` constant is one
    of these. Pass a custom one (e.g. ``Flatw0waCDM``-extracted) for
    non-ΛCDM forward evaluation.

    **Mathematical equivalence to DSPS**: this kernel uses the same
    interp pattern as
    :func:`dsps.photometry.photometry_kernels._obs_flux_ssp`
    (``jnp.interp(target_wave, source_wave*(1+z), L)``), plus an
    explicit ``(1+z) / (4π d_L²)`` factor for the F_ν conversion. DSPS
    bundles the equivalent factor into a magnitude-space dimming term
    (``distance_modulus - 2.5*log10(1+z)``); both paths give the same
    photometric answer at the AB-magnitude level.

    References
    ----------
    .. [1] Hogg, D. W. "Distance measures in cosmology." 1999,
       arXiv:astro-ph/9905116.
    """
    z_arr = jnp.asarray(z)
    dl_cm = _luminosity_distance_cm(z_arr, cosmo)
    flux_scale = (1.0 + z_arr) / (4.0 * jnp.pi * dl_cm * dl_cm)

    # Same interp pattern as dsps.photometry._obs_flux_ssp:
    # query observed-frame wave_obs against the redshifted rest-frame grid.
    L_nu_on_obs = jnp.interp(
        wave_obs,
        wave_rest * (1.0 + z_arr),
        L_nu_rest,
        left=0.0,
        right=0.0,
    )
    return L_nu_on_obs * flux_scale


def shift_to_obs_frame_grid(
    wave_rest: jnp.ndarray,
    L_nu_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    z_table: jnp.ndarray,
    cosmo: CosmoParams = DEFAULT_COSMO,
) -> jnp.ndarray:
    """Vmap :func:`shift_to_obs_frame` over a redshift table.

    Used by precompute paths (``WavePrecomp``) to build the per-z F_ν
    tensor in one vmap'd call. Mirrors the role DSPS's
    :func:`dsps.photometry.photpop.precompute_ssp_obsmag_table` plays
    for the stellar SSP × filter precompute.

    Parameters
    ----------
    wave_rest : ndarray, shape (n_wave_rest,)
    L_nu_rest : ndarray, shape (..., n_wave_rest)
        Rest-frame L_ν. Leading axes are passed through unchanged
        (e.g. an SSP grid of shape (n_met, n_age, n_wave) returns a
        (n_z, n_met, n_age, n_wave_obs) tensor).
    wave_obs : ndarray, shape (n_wave_obs,)
        Single observed-frame wave grid, shared across all z.
    z_table : ndarray, shape (n_z,)
        Redshift grid.
    cosmo : CosmoParams, optional

    Returns
    -------
    f_nu_obs_table : ndarray, shape (n_z, ..., n_wave_obs)
        Observed-frame F_ν on the (z, ..., wave_obs) grid.

    Notes
    -----
    **JIT-compatible**: yes via the inner kernel; not decorated here
    because the vmap shapes depend on caller.
    """
    # vmap over z (leading axis of output) while broadcasting wave/L_nu/wave_obs.
    return vmap(
        lambda zi: shift_to_obs_frame(wave_rest, L_nu_rest, wave_obs, zi, cosmo),
        in_axes=0,
    )(z_table)
