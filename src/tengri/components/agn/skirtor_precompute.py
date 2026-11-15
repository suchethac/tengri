# SPDX-License-Identifier: BSD-3-Clause
r"""Precompute adapter for SKIRTOR AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
SKIRTOR's 5-D torus template grid: (tau, p, q, oa, cos_inc), with the radius
ratio pinned at its R = 20 node.

The lookup returns the band photometry of the torus SED that
:func:`~tengri.components.agn.skirtor.skirtor_sed` emits (the ``model='skirtor'``
runtime, ``skirtor._skirtor_grid_sed``). It reads the
same template cube (``skirtor._torus_template_cube``,
the v3 dust emission), interpolates it the same way (node-exact PCHIP, ``cos_inc``
extended past its end node), and emits it through the runtime's own normalization
and log-log resampling (``skirtor._emit_normalized_lnu``).
The saving is the wavelength range: the table holds only the template wavelengths
the filters read, and the SED is emitted only on the filter quadrature grid.

Records any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters`` and reads the
table there, keeping the table whole so it is interpolated in the runtime's axis order.

References
----------
.. [1] M. Stalevski, J. Fritz, M. Baes, T. Nakos, and L. C. Popovic,
   "3D radiative transfer modelling of the dusty tori around active galactic
   nuclei as a clumpy two-phase medium," MNRAS, 420, 2756 (2012). arXiv:1109.1286.
   https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski, C. Ricci, Y. Ueda, P. Lira, J. Fritz, and M. Baes,
   "The dust covering factor in active galactic nuclei," MNRAS, 458,
   2288 (2016). arXiv:1602.06954. bibcode:2016MNRAS.458.2288S.
   https://doi.org/10.1093/mnras/stw444
.. [3] F. N. Fritsch and R. E. Carlson, "Monotone Piecewise Cubic Interpolation,"
   SIAM J. Numer. Anal. 17, 238 (1980). https://doi.org/10.1137/0717021
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._phys import L_SUN as _L_SUN
from tengri.components.agn._template_power_table import (
    band_fold,
    quadrature_grid,
    read_at_fixed_axes,
)
from tengri.components.agn.skirtor import (
    _cos_inc_extrapolate,
    _emit_normalized_lnu,
    _load_grid_arrays,
    _torus_template_cube,
)
from tengri.forward.precompute.templates import (
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import interp_nd_pchip, loglog_integral

# Parameter names corresponding to the 5 SKIRTOR grid axes, in order.
AXIS_PARAMS: tuple[str, ...] = (
    "agn_tau_skirtor",
    "agn_p_skirtor",
    "agn_q_skirtor",
    "agn_oa_skirtor",
    "agn_cos_inc",
)

#: The radius-ratio node the 5-D table is sliced at: the runtime's default
#: ``agn_radius_ratio`` (CIGALE's R = 20).
PINNED_RADIUS_RATIO: float = 20.0

#: Records evaluated per batch when the node photometry is built.
_NODE_BATCH: int = 64


def _torus_cube_at_pinned_radius(grid_path: str) -> tuple[np.ndarray, np.ndarray, tuple]:
    """The runtime's torus template cube, its wavelengths and its 5 axes, at R = 20.

    Raises
    ------
    ValueError
        If the grid has a radius-ratio axis without an R = 20 node, where a slice
        could not reproduce the runtime at its default radius ratio.
    """
    raw = _load_grid_arrays(grid_path)
    cube = np.asarray(_torus_template_cube(raw), dtype=np.float64)
    axes = tuple(np.asarray(ax, dtype=np.float64) for ax in raw["axes"])
    # ``_load_grid_arrays`` returns the R axis at position 4 on 6-axis grids.
    if len(axes) == 6:
        r_axis = axes[4]
        r_idx = int(np.argmin(np.abs(r_axis - PINNED_RADIUS_RATIO)))
        if not np.isclose(r_axis[r_idx], PINNED_RADIUS_RATIO, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"SKIRTOR grid {grid_path!r} has radius-ratio nodes {r_axis.tolist()} and no "
                f"R = {PINNED_RADIUS_RATIO:g} node; the 5-D photometry table is a node slice "
                "at the runtime's default radius ratio."
            )
        cube = cube[:, :, :, :, r_idx, :]
        axes = axes[:4] + axes[5:]
    return cube, np.asarray(raw["wave"], dtype=np.float64), axes


def _record_power(cube: np.ndarray, wave: np.ndarray) -> np.ndarray:
    """Bolometric integral of every record, as the runtime normalizes it; validated."""
    flat = jnp.asarray(cube.reshape(-1, cube.shape[-1]))
    wave_j = jnp.asarray(wave)
    power = np.asarray(jax.vmap(lambda t: loglog_integral(wave_j, t))(flat), dtype=np.float64)
    power = power.reshape(cube.shape[:-1])
    bad = ~(np.isfinite(power) & (power > 0.0))
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} SKIRTOR torus record(s) have non-positive or non-finite "
            f"bolometric power (first at grid index {np.argwhere(bad)[0].tolist()}); the "
            "table divides by that power and cannot represent them."
        )
    return power


def _emit_lsun(row, wave_read, wave_quad) -> jnp.ndarray:
    """The SED of one interpolated table row on the quadrature grid, per L_sun [erg/s/Hz]."""
    return _emit_normalized_lnu(row[:-1], wave_read, wave_quad, _L_SUN, row[-1])


def precompute_skirtor_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    r"""Tabulate the SKIRTOR torus templates the filters read, with their quadrature.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_templates_v3.h5`` (or a v2 ``.h5`` / ``.npz`` grid).
    filter_waves : list of array_like
        Wavelength grid per filter [Angstrom], observed frame, ascending.
    filter_trans : list of array_like
        Transmission per filter [dimensionless].
    redshift : float, optional
        Source redshift [dimensionless]. Shifts the rest-frame templates into the
        observed frame of the filters. Default 0.0.

    Returns
    -------
    dict
        ``grid_phot`` : ndarray, shape (n_tau, n_p, n_q, n_oa, n_inc, n_read + 1)
            The template :math:`T_\lambda` at the ``n_read`` wavelengths the filters
            read, over :math:`P_{\rm ref}` [per Angstrom per unit template power],
            then the record power :math:`P/P_{\rm ref}` [dimensionless]; see Notes.
        ``wave_read`` : ndarray, shape (n_read,)
            Those template wavelengths [Angstrom].
        ``wave_quad`` : ndarray, shape (n_quad,)
            Rest-frame quadrature grid [Angstrom].
        ``fold`` : ndarray, shape (n_quad, n_filters)
            The runtime's filter quadrature on ``wave_quad`` [dimensionless].
        ``axes`` : tuple of 5 ndarray
            Grid axes (tau, p, q, oa, cos_inc), ascending.
        ``_preint`` : PreintegratedGrid
            The band photometry at every node, :math:`L_\nu` per :math:`L_\odot`
            [erg/s/Hz], for the Fixed-axis bookkeeping of ``collapse_fixed_axes``.

    Raises
    ------
    ValueError
        If the grid has a radius-ratio axis without an R = 20 node, or a record with
        non-positive bolometric power.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function.

    **What the lookup computes.** The runtime emits, at a parameter point
    :math:`\mathbf{c}`,

    .. math::

        L_\nu(\lambda;\,\mathbf{c}) = L_{\rm bol}\,f_{\rm torus}\,L_\odot\,
            \frac{\mathcal{R}\big[\mathcal{I}[T_\lambda](\mathbf{c})\big](\lambda)}
                 {\int \mathcal{I}[T_\lambda](\lambda';\,\mathbf{c})\,d\lambda'}\,
            \frac{\lambda^2}{c},

    where :math:`T_\lambda` is the stored L_lambda-like cube, :math:`\mathcal{I}` the
    node-exact PCHIP over the axes (Fritsch & Carlson [3]_), :math:`\mathcal{R}` the
    log-log resampling, :math:`L_{\rm bol} = 10^{\rm agn\_log\_lbol}` [L_sun],
    :math:`f_{\rm torus}` = ``agn_torus_frac`` [dimensionless], :math:`\lambda`
    [Angstrom] and :math:`c` [Angstrom/s]; the integral is the log-log one. The lookup
    interpolates :math:`T_\lambda` at the wavelengths the filters read exactly as the
    runtime does (PCHIP acts on each wavelength separately), resamples it onto the
    filter quadrature grid with the runtime's resampling and applies the runtime's
    quadrature (``fold``). Only the power in the denominator is interpolated as a
    channel of its own: at a node it is the record's own, so the lookup is the
    runtime's photometry there to rounding; between nodes the PCHIP of the power
    stands in for the power of the PCHIP.

    **Why the SED and not its band fluxes.** The runtime's band flux is the integral
    of the log-log resampled SED, a nonlinear function of the template values; and
    the PCHIP tangents of a band flux are not those of the wavelengths inside it. A
    table of band fluxes is therefore off the runtime between nodes (7 % at 1500 A
    near edge-on), and a band integral taken by linear interpolation between the
    template's own wavelengths (spaced :math:`R \sim 7`) is off at the nodes (a
    factor 177 at 1500 A).

    **Radius ratio.** The table is 5-D, sliced at the R = 20 node, the runtime's
    default ``agn_radius_ratio``. A model with another radius ratio is not
    represented; the exact path interpolates R (#772).
    """
    cube, wave, axes = _torus_cube_at_pinned_radius(grid_path)
    read, wave_quad = quadrature_grid(wave, filter_waves, redshift)
    power = _record_power(cube, wave)
    p_ref = float(np.median(power))
    table = jnp.asarray(
        np.concatenate([cube[..., read] / p_ref, (power / p_ref)[..., None]], axis=-1)
    )
    wave_read = jnp.asarray(wave[read])
    wq = jnp.asarray(wave_quad)
    fold = band_fold(wave_quad, filter_waves, filter_trans, redshift)

    rows = table.reshape(-1, table.shape[-1])
    node_sed = jax.jit(
        lambda r: jax.lax.map(lambda x: _emit_lsun(x, wave_read, wq), r, batch_size=_NODE_BATCH)
    )(rows)
    preint = precompute_template_photometry(
        templates=np.asarray(node_sed, dtype=np.float64).reshape(*table.shape[:-1], -1),
        wave_rest=wave_quad,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        axes=axes,
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,
    )
    return {
        "grid_phot": table,
        "wave_read": wave_read,
        "wave_quad": wq,
        "fold": fold,
        "axes": tuple(jnp.asarray(ax) for ax in axes),
        "_preint": preint,
    }


def _phot_per_lsun(row, precomp_arrays) -> jnp.ndarray:
    """Band L_nu per L_sun of torus power from one interpolated row [erg/s/Hz]."""
    wave_read, wave_quad, fold = precomp_arrays
    return _emit_lsun(row, wave_read, wave_quad) @ fold


def _quadrature_arrays(precomp: dict) -> tuple:
    return precomp["wave_read"], precomp["wave_quad"], precomp["fold"]


def build_skirtor_photometry_lookup(precomp: dict, grid_arrays_traced: tuple | None = None):
    """Build a JIT-compiled SKIRTOR torus photometry function.

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_skirtor_photometry` or :func:`precompute`.
    grid_arrays_traced : tuple or None, optional
        ``(grid_phot, axes)`` passed to the compiled function as traced arguments
        instead of closure-captured constants (smaller HLO). Default None.

    Returns
    -------
    callable
        ``fn(agn_log_lbol, agn_tau_skirtor, agn_p_skirtor, agn_q_skirtor,
        agn_oa_skirtor, agn_cos_inc, agn_torus_frac) -> ndarray, shape (n_filters,)``,
        the torus band L_nu [erg/s/Hz]. The caller applies
        ``(1 + z) / (4 pi d_L^2)`` for a flux density.

    Notes
    -----
    **JIT-compatible**: yes.

    **Gradient-safe**: yes, node-exact PCHIP is C1 on every axis; ``cos_inc``
    continues its end cubic past the stored ``cos(90 deg) = 6e-17`` node, as the
    runtime does, so ``cos_inc = 0`` keeps a nonzero gradient.
    """

    @jax.jit
    def skirtor_phot(
        agn_log_lbol,
        agn_tau_skirtor,
        agn_p_skirtor,
        agn_q_skirtor,
        agn_oa_skirtor,
        agn_cos_inc,
        agn_torus_frac,
        table,
        axes,
        quadrature,
    ):
        """Torus band L_nu [erg/s/Hz] at one parameter point."""
        point = (agn_tau_skirtor, agn_p_skirtor, agn_q_skirtor, agn_oa_skirtor, agn_cos_inc)
        row = interp_nd_pchip(table, axes, point, extrapolate=_cos_inc_extrapolate(axes))
        return 10.0**agn_log_lbol * agn_torus_frac * _phot_per_lsun(row, quadrature)

    if grid_arrays_traced is not None:
        table, axes = grid_arrays_traced
    else:
        table, axes = precomp["grid_phot"], precomp["axes"]
    quadrature = _quadrature_arrays(precomp)

    def lookup(
        agn_log_lbol,
        agn_tau_skirtor,
        agn_p_skirtor,
        agn_q_skirtor,
        agn_oa_skirtor,
        agn_cos_inc,
        agn_torus_frac,
    ):
        return skirtor_phot(
            agn_log_lbol,
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_cos_inc,
            agn_torus_frac,
            table,
            axes,
            quadrature,
        )

    return lookup


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build the SKIRTOR photometry table, recording Fixed-parameter axes.

    Parameters
    ----------
    filter_waves : list of array_like
        Wavelength grid per filter [Angstrom], observed frame, ascending.
    filter_trans : list of array_like
        Transmission per filter [dimensionless].
    redshift : float
        Source redshift [dimensionless].
    parameters : Parameters or None
        Parameters spec, used to detect Fixed-axis parameters.
    grid_path : str, keyword-only
        Path to ``skirtor_templates_v3.h5`` (or a v2 ``.h5`` / ``.npz`` grid).

    Returns
    -------
    dict
        Same as :func:`precompute_skirtor_photometry`, plus ``_collapsed_axes``
        (axis index to its Fixed coordinate) when any :data:`AXIS_PARAMS` entry is
        Fixed. The table is not contracted: the lookup reads it at the Fixed
        coordinates.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function.
    """
    result = precompute_skirtor_photometry(
        grid_path, filter_waves, filter_trans, redshift=redshift
    )
    _, _, fixed = collapse_fixed_axes(
        result["_preint"], AXIS_PARAMS, parameters, origin="skirtor_precompute"
    )
    if not fixed:
        return result
    # Kept whole and read at the Fixed coordinates (read_at_fixed_axes): contracting a
    # Fixed axis first would interpolate in another axis order than the runtime, and the
    # tensor-product PCHIP depends on that order.
    return {**result, "_collapsed_axes": fixed}


def build_lookup(preint: dict, *, free_param_names: tuple[str, ...] | None = None):
    """Build the runtime SKIRTOR photometry lookup from a preintegrated dict.

    Parameters
    ----------
    preint : dict
        Output of :func:`precompute`.
    free_param_names : tuple of str or None, optional
        Unused; accepted for Protocol uniformity.

    Returns
    -------
    callable
        Without Fixed axes, :func:`build_skirtor_photometry_lookup`. With Fixed
        axes, ``fn(agn_log_lbol, *free_axis_values, agn_torus_frac)``, the free
        coordinates in :data:`AXIS_PARAMS` order. Both return the torus band L_nu
        [erg/s/Hz], shape (n_filters,).

    Notes
    -----
    **JIT-compatible**: yes.

    **Gradient-safe**: yes, node-exact PCHIP is C1 on every axis.
    """
    collapsed = preint.get("_collapsed_axes")
    if not collapsed:
        return build_skirtor_photometry_lookup(preint)

    table = preint["grid_phot"]
    axes = preint["axes"]
    quadrature = _quadrature_arrays(preint)
    extrapolate = _cos_inc_extrapolate(axes)

    @jax.jit
    def skirtor_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """Torus band L_nu [erg/s/Hz] with the Fixed axes read at their values."""
        row = read_at_fixed_axes(table, axes, free_axis_values, collapsed, extrapolate=extrapolate)
        return 10.0**agn_log_lbol * agn_torus_frac * _phot_per_lsun(row, quadrature)

    return skirtor_phot_collapsed
