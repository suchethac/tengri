# SPDX-License-Identifier: BSD-3-Clause
"""Redshift handling for the fast nebular grid's band photometry.

The per-Q_H nebular grid (:mod:`~tengri.components.nebular.nebular_grid_precompute`)
serves the nebular contribution to each photometric band as
:math:`Q_H \\times \\mathrm{table}`. The band value of a redshifted SED depends on
the redshift, so a table projected once at a build-time reference redshift is
wrong at every other one (8 to 22 % of the total band flux on a dust-free Cue
model). The nebular SED is exactly a continuum plus a line catalog, and the two
halves want different treatments:

* **Lines** move through a bandpass as :math:`(1+z)\\lambda_0`, and a line
  crossing a filter edge changes the band by its whole flux, so a table over
  :math:`z` cannot resolve them. They are evaluated at the traced redshift as
  delta lines: :func:`delta_line_band_kernel`.
* **The continuum** is smooth in :math:`z` except at a handful of features (the
  Balmer and Lyman edges). It is tabulated on a grid uniform in
  :math:`\\ln(1+z)` by :func:`continuum_ztable` and interpolated linearly in that
  variable by :func:`continuum_band_at_z`.

Both halves are linear in :math:`Q_H`. Every band value here is the
un-dimmed, pre-IGM, rest-frame band :math:`L_\\nu` that
``nebular_phot_lnu_precomp`` has always carried: the cosmological dimming and
the dust screen are applied downstream, as for the exact path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.utils.filter_convention import FilterConvention, filter_weight, filter_weight_np
from tengri.utils.physics_constants import C_AA
from tengri.utils.scale import pow10

__all__ = [
    "continuum_band_at_z",
    "continuum_band_from_slab",
    "continuum_z_slab",
    "continuum_ztable",
    "delta_line_band_kernel",
    "ln1pz_grid",
    "rendered_line_band_kernel",
]

#: Smallest positive table value kept when a log is taken, relative to the
#: smallest positive entry actually present. Only ever applies to entries that
#: are not exactly zero; exact zeros are carried by a separate mask, never by a
#: floor (a floor would make a partly covered band interpolate through a fake
#: tiny value instead of a true zero).
_LOG_FLOOR_REL = 1e-3


def ln1pz_grid(z_lo: float, z_hi: float, n_z: int) -> np.ndarray:
    """Redshift grid uniform in :math:`\\ln(1+z)`.

    Parameters
    ----------
    z_lo, z_hi : float
        Redshift range covered [dimensionless], ``0 <= z_lo <= z_hi``.
    n_z : int
        Number of nodes. ``1`` (or ``z_lo == z_hi``) gives a single node at
        ``z_lo``: the table of a model whose redshift is a build-time constant.

    Returns
    -------
    ndarray, shape (n_z,)
        :math:`\\ln(1+z)` at the nodes, ascending.
    """
    if n_z < 1:
        raise ValueError(f"n_z must be >= 1; got {n_z}.")
    if n_z == 1 or z_hi <= z_lo:
        return np.array([np.log1p(z_lo)])
    return np.linspace(np.log1p(z_lo), np.log1p(z_hi), int(n_z))


def _union_kernel(wave_obs, fw, ft, convention):
    """Sparse linear functional of an SED that ``lnu_filter_integral`` evaluates.

    The exact projector interpolates the SED linearly onto the sorted union of the
    SED nodes and the filter nodes, then integrates with the trapezoid rule. Every
    step is linear in the SED values, so the band value is ``K . sed``; this
    returns ``K`` (over the SED nodes it touches) so many SEDs can share one
    kernel. Nodes outside the filter support carry zero weight and are dropped
    (one node past each edge is kept, so the straddling segment stays exact).

    Returns
    -------
    lo : int
        First SED node ``K`` touches.
    kernel : ndarray, shape (n_touched,)
        Weights on ``sed[lo : lo + n_touched]``.
    """
    n = wave_obs.size
    grid = np.sort(np.concatenate([wave_obs, fw]))
    g_lo = max(int(np.searchsorted(grid, fw[0])) - 1, 0)
    g_hi = min(int(np.searchsorted(grid, fw[-1], side="right")) + 1, grid.size)
    grid = grid[g_lo:g_hi]
    trans = np.interp(grid, fw, ft, left=0.0, right=0.0)
    tw = trans * filter_weight_np(grid, convention)
    dg = np.diff(grid)
    omega = np.zeros_like(grid)
    omega[:-1] += 0.5 * dg
    omega[1:] += 0.5 * dg
    denom = float(np.sum(omega * tw))
    inside = (grid >= wave_obs[0]) & (grid <= wave_obs[-1])
    idx = np.clip(np.searchsorted(wave_obs, grid, side="right") - 1, 0, n - 2)
    frac = (grid - wave_obs[idx]) / (wave_obs[idx + 1] - wave_obs[idx])
    coef = omega * tw * inside
    touched = coef != 0.0
    if not np.any(touched):
        return 0, np.zeros(1)
    idx, frac, coef = idx[touched], frac[touched], coef[touched]
    lo = int(idx.min())
    hi = int(idx.max()) + 2
    kernel = np.bincount(idx - lo, weights=coef * (1.0 - frac), minlength=hi - lo) + np.bincount(
        idx + 1 - lo, weights=coef * frac, minlength=hi - lo
    )
    return lo, kernel[: hi - lo] / max(denom, 1e-300)


def continuum_ztable(
    cont_nodes,
    wave,
    filter_waves,
    filter_trans,
    lnz_grid,
    convention: FilterConvention = FilterConvention.BESSELL,
):
    """Band-project per-node continuum SEDs at every redshift of a grid.

    For each redshift and filter this is the same quadrature as
    :func:`~tengri.observation.photometry.lnu_filter_integral` (the union grid of
    the redshifted SED nodes and the filter nodes), evaluated for all grid nodes at
    once through the shared sparse kernel.

    Parameters
    ----------
    cont_nodes : array_like, shape (n_nodes, n_wave)
        Continuum :math:`L_\\nu` per grid node on ``wave`` [erg/s/Hz per photon/s].
    wave : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom], ascending.
    filter_waves, filter_trans : sequence of array_like
        Observed-frame filter curves [Angstrom], [dimensionless].
    lnz_grid : array_like, shape (n_z,)
        :math:`\\ln(1+z)` at the nodes (see :func:`ln1pz_grid`).
    convention : FilterConvention, optional
        Bandpass weight (default ``BESSELL``, 1/lambda: what the exact path uses).

    Returns
    -------
    ndarray, shape (n_nodes, n_z, n_filter)
        Rest-frame band :math:`L_\\nu`, float64.

    Notes
    -----
    **JIT-compatible**: no; build-time numpy.
    """
    cont = np.asarray(cont_nodes, dtype=np.float64)
    wave = np.asarray(wave, dtype=np.float64)
    zs = np.expm1(np.asarray(lnz_grid, dtype=np.float64))
    fws = [np.asarray(fw, dtype=np.float64) for fw in filter_waves]
    fts = [np.asarray(ft, dtype=np.float64) for ft in filter_trans]
    out = np.zeros((cont.shape[0], zs.size, len(fws)))
    for iz, z in enumerate(zs):
        wave_obs = wave * (1.0 + z)
        for jf, (fw, ft) in enumerate(zip(fws, fts, strict=True)):
            lo, kernel = _union_kernel(wave_obs, fw, ft, convention)
            out[:, iz, jf] = cont[:, lo : lo + kernel.size] @ kernel
    return out


def rendered_line_band_kernel(profiles, wave, filter_waves, filter_trans, redshift):
    """Band response of each rendered unit-luminosity line at one redshift.

    The exact response of a line to a filter, for a build-time-constant redshift:
    each row of ``profiles`` is the line as the SED renders it (a unit-luminosity
    velocity-width profile), projected with the exact quadrature. A line
    luminosity is linear in the SED, so ``L_line * kernel`` is the line's band
    contribution.

    Parameters
    ----------
    profiles : array_like, shape (n_lines, n_wave)
        Unit-luminosity rendered line profiles [1/Hz] on ``wave``.
    wave : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    filter_waves, filter_trans : sequence of array_like
        Observed-frame filter curves.
    redshift : float
        Redshift of the projection [dimensionless].

    Returns
    -------
    ndarray, shape (n_lines, n_filter)
        Band response [1/Hz] (rest-frame band :math:`L_\\nu` per unit line
        luminosity [erg/s]).
    """
    from tengri.observation.photometry import lnu_filter_integral

    profiles = jnp.asarray(profiles)
    wave = jnp.asarray(wave)

    def _one(profile):
        return jnp.stack(
            [
                lnu_filter_integral(profile, wave, jnp.asarray(fw), jnp.asarray(ft), redshift)
                for fw, ft in zip(filter_waves, filter_trans, strict=True)
            ]
        )

    return jax.vmap(_one)(profiles)


def delta_line_band_kernel(
    redshift,
    line_waves,
    filter_waves,
    filter_trans,
    convention: FilterConvention = FilterConvention.BESSELL,
):
    r"""Band response of a delta line at the evaluation redshift.

    A line of integrated luminosity :math:`L` at rest wavelength
    :math:`\lambda_0` is :math:`L_\nu = L\,\delta(\nu-\nu_0)`. Through the
    observed-frame bandpass :math:`T_b` with weight :math:`w=1/\lambda`
    (:func:`~tengri.observation.photometry.lnu_filter_integral`), substituting
    :math:`\lambda_{\rm obs}=(1+z)\lambda_{\rm rest}`,

    .. math::

        \frac{L_b}{L} = \frac{\lambda_0\,T_b\big((1+z)\lambda_0\big)}
                             {c\,\int T_b\,\mathrm{d}\lambda/\lambda},

    with :math:`\lambda_0` in Angstrom and :math:`c` in Angstrom/s (so the
    result has units of 1/Hz). There is no :math:`d_L` here: the cosmological
    dimming is applied downstream, as for the exact path.

    Parameters
    ----------
    redshift : float or ndarray, shape ()
        Evaluation redshift [dimensionless]; may be traced.
    line_waves : array_like, shape (n_lines,)
        Rest wavelengths of the lines **as the SED renders them** [Angstrom].
    filter_waves, filter_trans : sequence of array_like
        Observed-frame filter curves [Angstrom], [dimensionless].
    convention : FilterConvention, optional
        Bandpass weight; ``BESSELL`` (1/lambda) is what the exact path uses.

    Returns
    -------
    ndarray, shape (n_lines, n_filter)
        Band response [1/Hz].

    Notes
    -----
    **JIT/grad/vmap-safe.** Piecewise linear in :math:`z` (the transmission is
    interpolated linearly), so the redshift gradient is the local transmission
    slope; it is discontinuous only at a filter table node.

    **Approximation**: the exact SED spreads each line over a ~100 km/s profile;
    for a delta line the band value differs only when the profile straddles a
    filter edge or a transmission kink (measured against the rendered profiles:
    median 1.3e-4, p95 1.4e-2 of the nebular band, at most 6e-3 of the total
    flux).
    """
    line_waves = jnp.asarray(line_waves)
    obs = (1.0 + jnp.asarray(redshift)) * line_waves
    cols = []
    for fw, ft in zip(filter_waves, filter_trans, strict=True):
        fw = jnp.asarray(fw)
        ft = jnp.asarray(ft)
        trans = jnp.interp(obs, fw, ft, left=0.0, right=0.0)
        norm = jnp.trapezoid(ft * filter_weight(fw, convention), fw)
        cols.append(line_waves * trans / (C_AA * norm))
    return jnp.stack(cols, axis=-1)


def continuum_z_slab(log_table, lnz_grid, redshift):
    r"""The tabulated redshifts that bracket the evaluation redshift.

    Parameters
    ----------
    log_table : ndarray, shape (*grid_dims, n_z, n_filter)
        log10 continuum band :math:`L_\nu` per :math:`Q_H`.
    lnz_grid : ndarray, shape (n_z,)
        :math:`\ln(1+z)` at the table nodes.
    redshift : float or ndarray, shape ()
        Evaluation redshift [dimensionless]; may be traced.

    Returns
    -------
    slab : ndarray, shape (*grid_dims, m, n_filter)
        The bracketing table rows: ``m = 2``, or ``m = 1`` for a one-node table.
    lower : int or ndarray, shape ()
        Index of the first row of ``slab`` in the table (0 for a one-node table).
    weight : float or ndarray, shape ()
        Weight of the upper row, linear in :math:`\ln(1+z)`, clipped to [0, 1]
        (the table is not extrapolated).
    """
    n_dims = log_table.ndim - 2
    n_z = lnz_grid.shape[0]
    if n_z == 1:
        return jax.lax.slice_in_dim(log_table, 0, 1, axis=n_dims), 0, 0.0
    lnz = jnp.log1p(jnp.asarray(redshift))
    i = jnp.clip(jnp.searchsorted(lnz_grid, lnz, side="right") - 1, 0, n_z - 2)
    span = lnz_grid[i + 1] - lnz_grid[i]
    w = jnp.clip((lnz - lnz_grid[i]) / span, 0.0, 1.0)
    return jax.lax.dynamic_slice_in_dim(log_table, i, 2, axis=n_dims), i, w


def continuum_band_from_slab(log_nion, log_bands, keep, lower, weight):
    r"""Blend the two interpolated table rows into the continuum band.

    The blend is linear because the continuum is exactly zero below the 915
    Angstrom truncation: interpolating a floored log between a zero and a finite
    node would invent flux at the edge. Exponentiation comes first, so a band
    that is zero at one node stays zero there.

    Parameters
    ----------
    log_nion : float
        log10 ionizing photon rate [dex re photons/s].
    log_bands : ndarray, shape (m, n_filter)
        The slab from :func:`continuum_z_slab` after the grid-axis interpolation.
    keep : ndarray, shape (n_z, n_filter), bool
        False where the band is exactly zero at every grid node.
    lower, weight
        From :func:`continuum_z_slab`.

    Returns
    -------
    ndarray, shape (n_filter,)
        Rest-frame continuum band :math:`L_\nu` [erg/s/Hz], pre-IGM.

    Notes
    -----
    **JIT/grad/vmap-safe**: the z gradient is the slope between the two
    bracketing nodes, finite everywhere including where the band is exactly zero.
    """
    if log_bands.shape[0] == 1:
        return pow10(jnp.asarray(log_nion) + log_bands[0]) * keep[0]
    keep2 = jax.lax.dynamic_slice_in_dim(keep, lower, 2, axis=0)
    bands = pow10(jnp.asarray(log_nion) + log_bands) * keep2
    return (1.0 - weight) * bands[0] + weight * bands[1]


def continuum_band_at_z(log_nion, log_table, keep, lnz_grid, redshift, node_interp):
    r"""Continuum band :math:`L_\nu` at the evaluation redshift from the z-table.

    Gathers the two tabulated redshifts that bracket :math:`\ln(1+z)`, runs the
    grid-axis interpolation (``node_interp``) on that slab in log10, and blends
    the two rows in linear space (:func:`continuum_band_from_slab`).

    Parameters
    ----------
    log_nion : float
        log10 ionizing photon rate [dex re photons/s].
    log_table : ndarray, shape (*grid_dims, n_z, n_filter)
        log10 continuum band :math:`L_\nu` per :math:`Q_H`.
    keep : ndarray, shape (n_z, n_filter), bool
        False where the band is exactly zero at every grid node.
    lnz_grid : ndarray, shape (n_z,)
        :math:`\ln(1+z)` at the table nodes.
    redshift : float or ndarray, shape ()
        Evaluation redshift [dimensionless]; may be traced.
    node_interp : callable
        ``node_interp(slab) -> ndarray``: interpolates the leading grid
        dimensions of a ``(*grid_dims, ...)`` array at the query point.

    Returns
    -------
    ndarray, shape (n_filter,)
        Rest-frame continuum band :math:`L_\nu` [erg/s/Hz], pre-IGM.

    Notes
    -----
    **JIT/grad/vmap-safe.**
    """
    slab, lower, weight = continuum_z_slab(log_table, lnz_grid, redshift)
    return continuum_band_from_slab(log_nion, node_interp(slab), keep, lower, weight)
