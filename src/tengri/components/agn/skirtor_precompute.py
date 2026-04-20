"""Precompute adapter for SKIRTOR AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
SKIRTOR's 5D torus template grid: (tau, p, q, oa, cos_inc).

Auto-collapses any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters`` — e.g., a
user who pins ``agn_tau_skirtor`` and ``agn_p_skirtor`` gets a 3D runtime grid
instead of the full 5D one, for free.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.forward.precompute.grid import (
    PreintegratedGrid,
    interp_nd_triweight,
    preintegrate_grid,
    slice_fixed_axes,
)
from tengri.utils.interpolation import edges_for_grid

# Parameter names corresponding to the 5 SKIRTOR grid axes, in order.
AXIS_PARAMS: tuple[str, ...] = (
    "agn_tau_skirtor",
    "agn_p_skirtor",
    "agn_q_skirtor",
    "agn_oa_skirtor",
    "agn_cos_inc",
)


def precompute_skirtor_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
) -> dict:
    """Pre-integrate SKIRTOR templates through filter curves.

    For each 5D grid point (tau, p, q, oa, cos_inc), compute the
    filter-integrated photometry.  Backward-compatible — returns a dict with
    ``grid_phot`` and ``axes``.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_templates.npz``.
    filter_waves, filter_trans : list of array
        Filter curves in Angstrom / relative transmission.

    Returns
    -------
    dict
        ``grid_phot`` : array (n_tau, n_p, n_q, n_oa, n_inc, n_filters)
        ``axes`` : tuple of 5 grid arrays (jnp.ndarray, for legacy callers)
        ``_preint`` : :class:`PreintegratedGrid` (new, used by auto-collapse)
    """
    data = np.load(grid_path)
    grid = data["grid"]  # (n_tau, n_p, n_q, n_oa, n_inc, n_wave)
    wave_grid = data["wavelength"]  # Angstrom
    axes_np = (
        np.asarray(data["tau"]),
        np.asarray(data["p"]),
        np.asarray(data["q"]),
        np.asarray(data["oa"]),
        np.asarray(data["cos_inc"]),
    )

    preint = preintegrate_grid(
        templates=grid,
        wave_rest=np.asarray(wave_grid),
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=0.0,
        dl_cm=1.0,
        axes=axes_np,
        energy_normalize=True,
    )

    axes_jax = tuple(jnp.asarray(ax) for ax in axes_np)
    return {
        "grid_phot": preint.phot,
        "axes": axes_jax,
        "_preint": preint,
    }


def build_skirtor_photometry_lookup(precomp: dict):
    """Build a JIT-compiled SKIRTOR photometry function.

    Uses triweight interpolation for C²-continuous gradients.

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_skirtor_photometry` or :func:`precompute`
        (the Protocol-shaped entry point).

    Returns
    -------
    callable
        ``(agn_log_lbol, agn_tau_skirtor, agn_p_skirtor, agn_q_skirtor,
           agn_oa_skirtor, agn_cos_inc, agn_torus_frac) -> array (n_filters,)``
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]
    edges = tuple(edges_for_grid(ax) for ax in axes)

    @jax.jit
    def skirtor_phot(
        agn_log_lbol,
        agn_tau_skirtor,
        agn_p_skirtor,
        agn_q_skirtor,
        agn_oa_skirtor,
        agn_cos_inc,
        agn_torus_frac,
    ):
        l_bol_lsun = 10.0**agn_log_lbol
        point = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_cos_inc,
        )
        phot_normed = interp_nd_triweight(grid_phot, axes, edges, point)
        return l_bol_lsun * agn_torus_frac * phot_normed

    return skirtor_phot


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build preintegrated SKIRTOR grid, auto-collapsing Fixed-parameter axes.

    Parameters
    ----------
    filter_waves, filter_trans : list
        Filter curves (observed frame).
    redshift : float
        Unused — SKIRTOR templates are rest-frame and the filter grid is
        already observed-frame.  Accepted for Protocol consistency.
    parameters : Parameters | None
        Parameters spec, used to detect Fixed-axis parameters.
    grid_path : str
        Path to ``skirtor_templates.npz``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_skirtor_photometry` but with grid
        axes collapsed for any Fixed :data:`AXIS_PARAMS` entry.
    """
    result = precompute_skirtor_photometry(grid_path, filter_waves, filter_trans)
    if parameters is None:
        return result

    preint: PreintegratedGrid = result["_preint"]
    fixed: dict[int, float] = {}
    for i, pname in enumerate(AXIS_PARAMS):
        if parameters.is_fixed(pname):
            fixed[i] = float(parameters.fixed_value(pname))
    if not fixed:
        return result

    collapsed = slice_fixed_axes(preint, fixed)
    # Rebuild dict view; drop the axes that were collapsed
    remaining_axes = tuple(ax for i, ax in enumerate(result["axes"]) if i not in fixed)
    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def build_lookup(preint: dict, *, free_param_names: tuple[str, ...] | None = None):
    """Build the runtime lookup.

    When no axes are collapsed, this is :func:`build_skirtor_photometry_lookup`.
    When some axes are collapsed, the generated function expects fewer axis
    arguments — caller supplies the remaining free parameters in
    ``free_param_names`` order.  The default behavior assumes no collapse.
    """
    if not preint.get("_collapsed_axes"):
        return build_skirtor_photometry_lookup(preint)

    # Collapsed case: lookup takes (scale, *remaining_axis_values, torus_frac)
    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    edges = tuple(edges_for_grid(ax) for ax in axes)

    @jax.jit
    def skirtor_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        l_bol_lsun = 10.0**agn_log_lbol
        phot_normed = interp_nd_triweight(grid_phot, axes, edges, tuple(free_axis_values))
        return l_bol_lsun * agn_torus_frac * phot_normed

    return skirtor_phot_collapsed
