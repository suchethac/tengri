# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP Netzer accretion-disc template.

Implements the ``activatedisk`` module from upstream
``JohannesBuchner/GRAHSP`` (CeCILL-v2). The accretion-disc continuum is a
pre-computed template grid spanning black-hole mass :math:`M_{\\rm BH}`,
spin parameter :math:`a`, and Eddington accretion rate :math:`\\dot{M}`.
Each template :math:`T(\\lambda)` is normalized to 1 at 510 nm (rest-frame
5100 Å) and scaled so that :math:`\\lambda L_\\lambda(5100\\,\\mathrm{\\AA})` equals
``l5100`` [erg/s], the same convention as the power-law disc, torus, lines
and FeII:

.. math::

   L_\\lambda(\\lambda) = \\frac{\\mathrm{l5100}}{510\\,\\mathrm{nm}} \\cdot T(\\lambda)

where :math:`T(\\lambda)` is interpolated onto the user's wavelength grid and
re-normalised to 1 at 510 nm on that interpolation.

References
----------
.. [1] Buchner, J. et al. 2024, arXiv:2405.19297, §2.1.1.
.. [2] Netzer, H. & Trakhtenbrot, B. 2014, MNRAS, 438, 672. \
       Accretion-disc SED shapes across parameter space.
.. [3] Netzer, H. 2013, The Physics and Evolution of Active Galactic Nuclei. \
       Cambridge University Press.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from tengri.components.agn.grahsp.bbb import LAMBDA_5100_NM
from tengri.utils.grid_interp import resample_template

__all__ = ["netzer_disc", "netzer_disc_interp", "select_disc_model"]


def _resample_anchored(
    wave_nm: Array,
    disc_wave_nm: Array,
    template: Array,
) -> Array:
    r"""Resample a disc template so that :math:`\lambda L_\lambda(510\,\mathrm{nm}) = 1`.

    The stored templates are normalised to 1 at 510 nm on a linear
    interpolation of their native grid; the log-space resampler used here
    differs from that by about 1e-3 at 510 nm. Dividing by the resampled value
    at 510 nm (and by 510 nm for the :math:`\lambda L_\lambda` convention)
    makes the anchor exact on whatever wavelength grid the caller evaluates.

    Returns
    -------
    ndarray, shape (n_wave,)
        :math:`L_\lambda` [1/nm] with :math:`\lambda L_\lambda(510\,\mathrm{nm}) = 1`.
    """
    shape = resample_template(wave_nm, disc_wave_nm, template, left=0.0, right=0.0)
    anchor = resample_template(
        jnp.asarray([LAMBDA_5100_NM]), disc_wave_nm, template, left=0.0, right=0.0
    )[0]
    return shape / (LAMBDA_5100_NM * anchor)


def netzer_disc(
    wave_nm: Array,
    l5100: float,
    disc_wave_nm: Array,
    disc_lumin_model: Array,
) -> Array:
    r"""Netzer accretion-disc template, scaled to l5100.

    Interpolates a single disc template (pre-selected by M, a, Mdot)
    onto an arbitrary wavelength grid and scales by the 5100 Å
    luminosity :math:`\lambda L_\lambda(5100\,\mathrm{\AA})`.

    .. math::

       L_\lambda(\lambda) = \frac{\mathrm{l5100}}{510\,\mathrm{nm}} \cdot T(\lambda),
       \qquad \lambda L_\lambda(510\,\mathrm{nm}) = \mathrm{l5100}

    Parameters
    ----------
    wave_nm : array_like, shape (n_wave,)
        Output wavelength grid [nm].
    l5100 : float
        :math:`\lambda L_\lambda` at 5100 Å [erg/s].
    disc_wave_nm : array_like, shape (n_disc_wave,)
        Disc template wavelength grid [nm] (from
        ``data/grahsp/grahsp_templates.h5`` group ``netzer_disc``).
    disc_lumin_model : array_like, shape (n_disc_wave,)
        Disc template :math:`L_\lambda` per (M, a, Mdot) model,
        normalized to 1 at 510 nm.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        Specific disc luminosity [erg/s/nm], interpolated onto
        ``wave_nm`` grid.

    Notes
    -----
    JIT/grad/vmap-compatible. The convention is
    :math:`\lambda L_\lambda(5100\,\mathrm{\AA}) = \mathrm{l5100}`, exact on any
    evaluation grid that brackets 510 nm.

    Upstream GRAHSP's ``activatedisk`` instead scales :math:`L_\lambda(510\,
    \mathrm{nm})` by ``l5100``, which makes its Netzer disc 510 times brighter
    than its stated :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` and
    inconsistent with the power-law disc, torus, lines and FeII that use the
    same ``l5100``. tengri deliberately does not reproduce this; the spectral
    shape is the upstream template's.
    """
    wave = jnp.asarray(wave_nm)
    disc_wave = jnp.asarray(disc_wave_nm)
    disc_lumin = jnp.asarray(disc_lumin_model)

    # Scale template by l5100 and interpolate onto output grid.
    # Zero padding outside disc template support.
    return l5100 * _resample_anchored(wave, disc_wave, disc_lumin)


def netzer_disc_interp(
    wave_nm: Array,
    l5100: float,
    disc_wave_nm: Array,
    disc_lumin: Array,
    disc_m: tuple[str, ...],
    disc_a: tuple[str, ...],
    disc_mdot: tuple[str, ...],
    log_mbh: float,
    spin: float,
    log_mdot: float,
) -> Array:
    r"""Netzer accretion-disc template with multilinear interpolation.

    Implements differentiable multilinear interpolation over the GRAHSP Netzer
    disc grid in (M, a, Mdot) parameter space. This is a tengri extension
    beyond upstream; interpolation is exact at all 16 grid nodes and linearly
    interpolated between them.

    .. math::

       L_\lambda(\lambda) = \mathrm{l5100} \cdot T_\mathrm{interp}(\lambda)

    where :math:`T_\mathrm{interp}` is the multilinearly interpolated disc
    template in the (M, a, Mdot) grid.

    Parameters
    ----------
    wave_nm : array_like, shape (n_wave,)
        Output wavelength grid [nm].
    l5100 : float
        :math:`\lambda L_\lambda` at 5100 Å [erg/s].
    disc_wave_nm : array_like, shape (n_disc_wave,)
        Disc template wavelength grid [nm] (from
        ``data/grahsp/grahsp_templates.h5`` group ``netzer_disc``).
    disc_lumin : array_like, shape (16, n_disc_wave)
        Disc templates, one per (M, a, Mdot) grid node.
    disc_m : tuple[str, ...], shape (16,)
        Black-hole mass labels from GRAHSP bundle.
    disc_a : tuple[str, ...], shape (16,)
        Spin parameter labels from GRAHSP bundle.
    disc_mdot : tuple[str, ...], shape (16,)
        Eddington ratio labels from GRAHSP bundle.
    log_mbh : float
        Log10 black-hole mass [Msun], interpolated within grid support
        [6.0, 9.0]; clipped to grid boundaries outside this range.
    spin : float
        Spin parameter, interpolated within grid support [0.0, 0.998];
        clipped to grid boundaries outside this range.
    log_mdot : float
        Log10 Eddington ratio, interpolated within grid support
        [log10(0.03), log10(0.3)]; clipped to grid boundaries outside
        this range.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        Specific disc luminosity [erg/s/nm], interpolated onto ``wave_nm`` grid.

    Notes
    -----
    JIT/grad/vmap-compatible. Exact at all 16 grid nodes (rtol 1e-12) against
    :func:`netzer_disc`, so :math:`\lambda L_\lambda(5100\,\mathrm{\AA}) =
    \mathrm{l5100}` at every node and, because each corner spectrum is anchored
    at 510 nm before the (unit-sum) weights mix them, between nodes too. Upstream GRAHSP's
    ``activatedisk`` scales :math:`L_\lambda(510\,\mathrm{nm})` by ``l5100``
    instead, making its Netzer disc 510 times brighter than its stated
    :math:`\lambda L_\lambda(5100\,\mathrm{\AA})`; tengri deliberately does not
    reproduce this.
    Midpoint interpolation (e.g. log_mbh=7.5 between 7.0 and 8.0) equals
    the arithmetic mean of the two node spectra. Out-of-bounds inputs clip
    to grid edges (no NaN).

    At the upper log_mdot edge (the default log10(0.3)), the multilinear
    tie-break derivative uses clipped weights (0.5x magnitude), documented
    but not widened.
    """
    wave = jnp.asarray(wave_nm)
    disc_wave = jnp.asarray(disc_wave_nm)
    disc_lum = jnp.asarray(disc_lumin)

    # Grid axes (from GRAHSP bundle labels)
    m_grid = jnp.array([6.0, 7.0, 8.0, 9.0])
    log10_003 = jnp.log10(0.03)
    log10_03 = jnp.log10(0.3)

    # Clamp inputs to grid support
    log_mbh_clamped = jnp.clip(log_mbh, 6.0, 9.0)
    spin_clamped = jnp.clip(spin, 0.0, 0.998)
    log_mdot_clamped = jnp.clip(log_mdot, log10_003, log10_03)

    # M dimension: find bracketing indices in [6, 7, 8, 9]
    i_m = jnp.searchsorted(m_grid, log_mbh_clamped)
    i_m = jnp.clip(i_m, 1, 3)
    m_low = m_grid[i_m - 1]
    m_high = m_grid[i_m]
    w_m = (log_mbh_clamped - m_low) / (m_high - m_low)
    w_m = jnp.clip(w_m, 0.0, 1.0)
    m_idx_low = i_m - 1
    m_idx_high = i_m

    # a dimension: interpolate between [0.0, 0.998]
    # a_idx=1 (a=0.0) when w_a=0, a_idx=0 (a=0.998) when w_a=1
    w_a = (spin_clamped - 0.0) / (0.998 - 0.0)
    w_a = jnp.clip(w_a, 0.0, 1.0)
    a_idx_low = 1  # Lower bound: a=0.0 (bundle index 1)
    a_idx_high = 0  # Upper bound: a=0.998 (bundle index 0)

    # mdot dimension: interpolate between [log10(0.03), log10(0.3)]
    # mdot_idx=1 (mdot=0.03) when w_mdot=0, mdot_idx=0 (mdot=0.3) when w_mdot=1
    w_mdot = (log_mdot_clamped - log10_003) / (log10_03 - log10_003)
    w_mdot = jnp.clip(w_mdot, 0.0, 1.0)
    mdot_idx_low = 1  # Lower bound: mdot=0.03 (bundle index 1)
    mdot_idx_high = 0  # Upper bound: mdot=0.3 (bundle index 0)

    # Bundle grid layout: idx = a_idx * 8 + mdot_idx * 4 + m_idx
    # Get indices for the 8 corners
    def get_idx(m_idx, a_idx, mdot_idx):
        return a_idx * 8 + mdot_idx * 4 + m_idx

    idx_000 = get_idx(m_idx_low, a_idx_low, mdot_idx_low)
    idx_100 = get_idx(m_idx_high, a_idx_low, mdot_idx_low)
    idx_010 = get_idx(m_idx_low, a_idx_high, mdot_idx_low)
    idx_110 = get_idx(m_idx_high, a_idx_high, mdot_idx_low)
    idx_001 = get_idx(m_idx_low, a_idx_low, mdot_idx_high)
    idx_101 = get_idx(m_idx_high, a_idx_low, mdot_idx_high)
    idx_011 = get_idx(m_idx_low, a_idx_high, mdot_idx_high)
    idx_111 = get_idx(m_idx_high, a_idx_high, mdot_idx_high)

    # Gather templates for all 8 corners
    lumin_000 = disc_lum[idx_000]
    lumin_100 = disc_lum[idx_100]
    lumin_010 = disc_lum[idx_010]
    lumin_110 = disc_lum[idx_110]
    lumin_001 = disc_lum[idx_001]
    lumin_101 = disc_lum[idx_101]
    lumin_011 = disc_lum[idx_011]
    lumin_111 = disc_lum[idx_111]

    # Multilinear interpolation of the 8 corner spectra. Each corner is
    # resampled and anchored (lambda*L_lambda(510 nm) = 1) BEFORE mixing, so the
    # weights (which sum to 1) keep the anchor exact between nodes and the
    # interpolation stays linear in the node spectra.
    corners = jnp.stack(
        [lumin_000, lumin_100, lumin_010, lumin_110, lumin_001, lumin_101, lumin_011, lumin_111]
    )
    weights = jnp.stack(
        [
            (1 - w_m) * (1 - w_a) * (1 - w_mdot),
            w_m * (1 - w_a) * (1 - w_mdot),
            (1 - w_m) * w_a * (1 - w_mdot),
            w_m * w_a * (1 - w_mdot),
            (1 - w_m) * (1 - w_a) * w_mdot,
            w_m * (1 - w_a) * w_mdot,
            (1 - w_m) * w_a * w_mdot,
            w_m * w_a * w_mdot,
        ]
    )
    anchored = jax.vmap(lambda t: _resample_anchored(wave, disc_wave, t))(corners)
    return l5100 * jnp.tensordot(weights, anchored, axes=1)


def select_disc_model(
    disc_m: tuple[str, ...],
    disc_a: tuple[str, ...],
    disc_mdot: tuple[str, ...],
    m: str = "8.0",
    a: str = "0",
    mdot: str = "0.3",
) -> int:
    """Select disc model index by (M, a, Mdot) label.

    Parameters
    ----------
    disc_m, disc_a, disc_mdot : tuple[str, ...], each shape (16,)
        Disc grid labels from the GRAHSP template bundle
        (``load_grahsp_templates().disc_m``, etc.). Each tuple contains
        string representations of black-hole mass, spin, and Eddington ratio.
    m : str, optional
        Black-hole mass :math:`\\log_{10}(M_{\\rm BH}/M_\\odot)`. Default: "8.0".
    a : str, optional
        Spin parameter. Default: "0".
    mdot : str, optional
        Eddington accretion rate. Default: "0.3".

    Returns
    -------
    idx : int
        Row index into the disc template grid (0–15 for GRAHSP bundle).
        Raises ``ValueError`` if the requested model is not found.

    Notes
    -----
    Static (not JIT-traced). No interpolation between grid points: the
    disc grid is too sparse. Model selection is a structural choice.
    """
    for idx, (m_i, a_i, mdot_i) in enumerate(zip(disc_m, disc_a, disc_mdot)):
        if m_i == m and a_i == a and mdot_i == mdot:
            return idx
    raise ValueError(
        f"Disc model (M={m}, a={a}, Mdot={mdot}) not found in GRAHSP bundle. "
        f"Available: {list(zip(disc_m, disc_a, disc_mdot))}"
    )
