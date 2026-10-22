# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for the Nenkova+08 ``NK0_mean_3p`` AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the three-parameter (inclination, opening-angle, optical-depth) AGNfitter-rX
torus reduction, keyed on ``(cos(inclination), opening angle, tau_V)``.

Auto-collapses any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters``.

References
----------
.. [1] M. Nenkova, M. M. Sirocky, R. Nikutta, Z. Ivezic, and M. Elitzur,
   "AGN Dusty Tori. II. Observational Implications of Clumpiness," ApJ 685,
   160 (2008). doi:10.1086/590483. arXiv:0806.0512. bibcode:2008ApJ...685..160N.
.. [2] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components._collapsed_lookup import interp_collapsed
from tengri.components.agn._template_power_table import (
    full_coords,
    lnu_per_lsun_from_power_table,
    power_weighted_template_photometry,
)
from tengri.components.agn._torus_inclination import (
    library_anisotropy,
    log10_anisotropy,
)
from tengri.components.agn.nenkova_agnfitter_3p import (
    TORUS_INCLINATION,
    load_nenkova_agnfitter_3p_grid,
)
from tengri.forward.precompute.templates import collapse_fixed_axes
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_pchip,
)
from tengri.utils.scale import pow10

# NK0_mean_3p grid parametrized by inclination (as cos), opening angle, tau_V.
AXIS_PARAMS: tuple[str, ...] = ("agn_cos_inc", "agn_oa_nenkova", "agn_tv_nenkova")


def precompute_nenkova_agnfitter_3p_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate ``NK0_mean_3p`` torus templates through filter curves.

    For each ``(cos_inc, oa, tv)`` grid point, compute the filter-integrated
    photometry. Returns a dict with ``grid_phot`` and ``axes``.

    The lookup reads the table as L_ν [erg/s/Hz] per L_sun of torus power,
    normalized as the runtime ``nenkova_agnfitter_3p.py`` does.

    Parameters
    ----------
    grid_path : str
        Path to ``nenkova_agnfitter_3p_torus_grid.h5``.
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0–1).
    redshift : float, optional
        Source redshift. Default 0.0.

    Returns
    -------
    dict
        ``grid_phot`` : ndarray, shape (n_incl, n_oa, n_tv, n_filters + 1)
            Power-weighted table: raw band L_ν [erg/s/Hz per unit template power]
            and, last, the raw bolometric power; see Notes.
        ``axes`` : tuple of 3 grid arrays (jnp.ndarray)
            Grid axes (cos_inc, oa, tv).
        ``_preint`` : PreintegratedGrid
            Internal preintegration data structure.
        ``_anisotropy`` : AnisotropyGrid
            The library anisotropy table the lookup reads a / mean(a) from.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

    **Build-time operation**: This function performs frequency-domain
    integration via NumPy. The precomputed photometry is grid-independent
    (depends only on filter curves and redshift, not wavelength grid).

    **Normalization**: the runtime kernel normalizes the *interpolated* template by
    its own bolometric integral, so off-node it weights each node by its raw power.
    The table therefore holds the raw band photometry and, as a last channel, the raw
    power (``power_weighted_template_photometry`` in ``_template_power_table``); the
    lookup interpolates both and divides. At a node this is the node-normalized
    photometry exactly.
    The sightline factor a / mean(a) is applied after the division, from the
    library's anisotropy table read as the runtime reads it
    (``_torus_inclination.log10_anisotropy``); it is returned as ``_anisotropy``.

    **Grid**: read through the runtime loader
    ``nenkova_agnfitter_3p.load_nenkova_agnfitter_3p_grid`` (inclination as cos i,
    ascending), so the table and the runtime share axes and node order.
    """
    # The runtime grid: inclination as cos i, ascending, templates reordered to match.
    runtime_grid = load_nenkova_agnfitter_3p_grid(grid_path)
    wave_grid = np.asarray(runtime_grid.wave_grid, dtype=np.float64)
    cos_inc_axis, oa_axis, tv_axis = (np.asarray(a, dtype=np.float64) for a in runtime_grid.axes)
    # The runtime normalizes the interpolated template, so its off-node value is
    # weighted by each node's raw power: tabulate the raw band photometry and the
    # raw power and divide after interpolating (_template_power_table).
    # The sightline power a / mean a multiplies the shape, as in the runtime
    # (_torus_inclination.torus_lnu_anisotropic).
    anisotropy = library_anisotropy(runtime_grid, TORUS_INCLINATION, jnp.asarray(0.0).dtype)
    preint = power_weighted_template_photometry(
        runtime_grid.template,
        wave_grid,
        filter_waves,
        filter_trans,
        redshift,
        axes=(cos_inc_axis, oa_axis, tv_axis),
    )

    axes_jax = (jnp.asarray(cos_inc_axis), jnp.asarray(oa_axis), jnp.asarray(tv_axis))
    return {
        "grid_phot": preint.phot,
        "axes": axes_jax,
        "_preint": preint,
        "_anisotropy": anisotropy,
    }


def build_nenkova_agnfitter_3p_photometry_lookup(precomp: dict):
    """Build a JIT-compiled ``NK0_mean_3p`` torus photometry function.

    Uses node-exact PCHIP interpolation for monotone-cubic accuracy.

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_nenkova_agnfitter_3p_photometry` or
        :func:`precompute` (the Protocol-shaped entry point).

    Returns
    -------
    callable
        Function with signature::

            fn(agn_log_lbol, agn_cos_inc, agn_oa_nenkova, agn_tv_nenkova,
               agn_torus_frac) -> ndarray, shape (n_filters,)

        Returns torus L_ν [erg/s/Hz]. Caller applies
        ``flux_scale = (1+z) / (4π d_L²)`` to get flux density.

    Notes
    -----
    **JIT-compatible**: yes.

    **Gradient-safe**: yes; PCHIP kernel is C¹-continuous.
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]
    anisotropy = precomp["_anisotropy"]

    @jax.jit
    def nenkova_agnfitter_3p_phot(
        agn_log_lbol, agn_cos_inc, agn_oa_nenkova, agn_tv_nenkova, agn_torus_frac
    ):
        """Compute ``NK0_mean_3p`` torus photometry via PCHIP interpolation.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        l_bol_lsun = 10.0**agn_log_lbol
        point = (agn_cos_inc, agn_oa_nenkova, agn_tv_nenkova)
        shape = lnu_per_lsun_from_power_table(interp_nd_pchip(grid_phot, axes, point))
        phot_per_lsun = shape * pow10(log10_anisotropy(anisotropy, point, TORUS_INCLINATION))
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return nenkova_agnfitter_3p_phot


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build preintegrated ``NK0_mean_3p`` grid, auto-collapsing Fixed-parameter axes.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0–1).
    redshift : float
        Source redshift. [dimensionless]
    parameters : Parameters | None
        Parameters spec, used to detect Fixed-axis parameters.
    grid_path : str, keyword-only
        Path to ``nenkova_agnfitter_3p_torus_grid.h5``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_nenkova_agnfitter_3p_photometry` but
        with grid axes collapsed for any Fixed :data:`AXIS_PARAMS` entry.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    """
    result = precompute_nenkova_agnfitter_3p_photometry(
        grid_path, filter_waves, filter_trans, redshift=redshift
    )
    preint: PreintegratedGrid = result["_preint"]
    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint, AXIS_PARAMS, parameters, origin="nenkova_agnfitter_3p_precompute"
    )
    if not fixed:
        return result

    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
        "_anisotropy": result["_anisotropy"],
    }


def build_lookup(preint: dict, *, free_param_names: tuple[str, ...] | None = None):
    """Build the runtime ``NK0_mean_3p`` photometry lookup from a preintegrated dict.

    When no axes are collapsed, delegates to
    :func:`build_nenkova_agnfitter_3p_photometry_lookup`. When some axes are
    collapsed (fixed at preintegration time), the returned function expects
    only the remaining free parameter values.

    Parameters
    ----------
    preint : dict
        Preintegrated data dict with keys ``"grid_phot"``, ``"axes"``,
        and optionally ``"_collapsed_axes"`` and ``"_preint"``.
    free_param_names : tuple of str or None, optional
        Names of remaining free axes in the collapsed case.
        Not used in the default (no-collapse) case.

    Returns
    -------
    callable
        JIT-compiled photometry lookup function returning torus L_ν
        [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, the returned function is fully JAX-native.

    **Gradient-safe**: yes; PCHIP interpolation is fully differentiable.
    """
    if not preint.get("_collapsed_axes"):
        return build_nenkova_agnfitter_3p_photometry_lookup(preint)

    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    anisotropy = preint["_anisotropy"]
    collapsed = preint["_collapsed_axes"]

    @jax.jit
    def nenkova_agnfitter_3p_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """``NK0_mean_3p`` torus photometry with collapsed (fixed) axes via PCHIP."""
        l_bol_lsun = 10.0**agn_log_lbol
        shape = lnu_per_lsun_from_power_table(
            interp_collapsed(grid_phot, axes, free_axis_values, kernel="pchip")
        )
        point = full_coords(free_axis_values, collapsed, 3)
        phot_per_lsun = shape * pow10(log10_anisotropy(anisotropy, point, TORUS_INCLINATION))
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return nenkova_agnfitter_3p_phot_collapsed
