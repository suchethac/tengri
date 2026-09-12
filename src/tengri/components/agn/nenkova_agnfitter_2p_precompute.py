# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for the Nenkova+08 ``NK0_mean_2p`` AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the two-parameter (inclination, opening-angle) AGNfitter-rX torus reduction,
keyed on ``(cos(inclination), opening angle)``.

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
from tengri.components.agn.nenkova_agnfitter_2p import _load_nenkova_agnfitter_2p_arrays
from tengri.forward.precompute.templates import (
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_pchip,
)

# NK0_mean_2p grid parametrized by inclination (as cos) and opening angle.
AXIS_PARAMS: tuple[str, ...] = ("agn_cos_inc", "agn_oa_nenkova")


def precompute_nenkova_agnfitter_2p_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate ``NK0_mean_2p`` torus templates through filter curves.

    For each ``(cos_inc, oa)`` grid point, compute the filter-integrated
    photometry. Returns a dict with ``grid_phot`` and ``axes``.

    Templates are frequency-normalized (matching the runtime normalization
    in ``nenkova_agnfitter_2p.py``) so that
    ``build_nenkova_agnfitter_2p_photometry_lookup`` returns L_ν [erg/s/Hz]
    per L_sun of bolometric luminosity.

    Parameters
    ----------
    grid_path : str
        Path to ``nenkova_agnfitter_2p_torus_grid.h5``.
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0–1).
    redshift : float, optional
        Source redshift. Used to shift rest-frame templates into the
        observed frame before integrating against observed-frame filters.
        Default 0.0.

    Returns
    -------
    dict
        ``grid_phot`` : ndarray, shape (n_incl, n_oa, n_filters)
            Filter-integrated L_ν [erg/s/Hz] per L_sun (unit torus fraction).
        ``axes`` : tuple of 2 grid arrays (jnp.ndarray)
            Grid axes (cos_inc, oa).
        ``_preint`` : PreintegratedGrid
            Internal preintegration data structure.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

    **Build-time operation**: This function performs frequency-domain
    integration via NumPy. The precomputed photometry is grid-independent
    (depends only on filter curves and redshift, not wavelength grid).

    **Normalization**: Templates are frequency-normalized so that the
    integration constant equals L_sun / trapz(template, nu). This matches
    the runtime normalization in ``nenkova_agnfitter_2p.py``.

    **Grid reordering**: The native inclination axis is stored in degrees
    (ascending). This function converts to cos(incl) and reorders templates
    to match, mirroring ``nenkova_agnfitter_2p.load_nenkova_agnfitter_2p_grid``.
    """
    from tengri.components.agn._phys import C_LIGHT as _C_CGS
    from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

    raw = _load_nenkova_agnfitter_2p_arrays(grid_path)
    grid = np.asarray(raw["template"], dtype=np.float64)  # (n_incl, n_oa, n_wave)
    wave_grid = np.asarray(raw["wavelength"], dtype=np.float64)
    incl_deg_axis = np.asarray(raw["incl_axis"], dtype=np.float64)
    oa_axis = np.asarray(raw["oa_axis"], dtype=np.float64)

    cos_inc_axis = np.cos(np.deg2rad(incl_deg_axis))
    order = np.argsort(cos_inc_axis)
    cos_inc_axis = cos_inc_axis[order]
    grid_reordered = grid[order]

    nu_grid = _C_CGS / (wave_grid * 1e-8)  # Hz (decreasing order)
    sort_idx = np.argsort(nu_grid)
    nu_sorted = nu_grid[sort_idx]

    n_incl, n_oa, _ = grid_reordered.shape
    lnu_grid = np.empty_like(grid_reordered)
    for i in range(n_incl):
        for j in range(n_oa):
            template = grid_reordered[i, j]
            integral = np.trapezoid(template[sort_idx], nu_sorted)
            integral_safe = max(abs(integral), 1e-100)
            lnu_grid[i, j] = _LSUN_ERG * template / integral_safe

    preint = precompute_template_photometry(
        templates=lnu_grid,
        wave_rest=wave_grid,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(cos_inc_axis, oa_axis),
        energy_normalize=False,  # templates already normalized per L_sun
        units="lnu",
    )

    axes_jax = (jnp.asarray(cos_inc_axis), jnp.asarray(oa_axis))
    return {
        "grid_phot": preint.phot,
        "axes": axes_jax,
        "_preint": preint,
    }


def build_nenkova_agnfitter_2p_photometry_lookup(precomp: dict):
    """Build a JIT-compiled ``NK0_mean_2p`` torus photometry function.

    Uses node-exact PCHIP interpolation for monotone-cubic accuracy.

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_nenkova_agnfitter_2p_photometry` or
        :func:`precompute` (the Protocol-shaped entry point).

    Returns
    -------
    callable
        Function with signature::

            fn(agn_log_lbol, agn_cos_inc, agn_oa_nenkova, agn_torus_frac)
                -> ndarray, shape (n_filters,)

        Returns torus L_ν [erg/s/Hz]. Caller applies
        ``flux_scale = (1+z) / (4π d_L²)`` to get flux density.

    Notes
    -----
    **JIT-compatible**: yes.

    **Gradient-safe**: yes; PCHIP kernel is C¹-continuous.

    **Interpolation kernel**: PCHIP (monotone-cubic) interpolation is
    node-exact and provides C¹-continuous gradients for autodiff.
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]

    @jax.jit
    def nenkova_agnfitter_2p_phot(agn_log_lbol, agn_cos_inc, agn_oa_nenkova, agn_torus_frac):
        """Compute ``NK0_mean_2p`` torus photometry via PCHIP interpolation.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        l_bol_lsun = 10.0**agn_log_lbol
        point = (agn_cos_inc, agn_oa_nenkova)
        phot_per_lsun = interp_nd_pchip(grid_phot, axes, point)
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return nenkova_agnfitter_2p_phot


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build preintegrated ``NK0_mean_2p`` grid, auto-collapsing Fixed-parameter axes.

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
        Path to ``nenkova_agnfitter_2p_torus_grid.h5``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_nenkova_agnfitter_2p_photometry` but
        with grid axes collapsed for any Fixed :data:`AXIS_PARAMS` entry.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    """
    result = precompute_nenkova_agnfitter_2p_photometry(
        grid_path, filter_waves, filter_trans, redshift=redshift
    )
    preint: PreintegratedGrid = result["_preint"]
    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint, AXIS_PARAMS, parameters, origin="nenkova_agnfitter_2p_precompute"
    )
    if not fixed:
        return result

    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def build_lookup(preint: dict, *, free_param_names: tuple[str, ...] | None = None):
    """Build the runtime ``NK0_mean_2p`` photometry lookup from a preintegrated dict.

    When no axes are collapsed, delegates to
    :func:`build_nenkova_agnfitter_2p_photometry_lookup`. When some axes are
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
        return build_nenkova_agnfitter_2p_photometry_lookup(preint)

    grid_phot = preint["grid_phot"]
    axes = preint["axes"]

    @jax.jit
    def nenkova_agnfitter_2p_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """``NK0_mean_2p`` torus photometry with collapsed (fixed) axes via PCHIP."""
        l_bol_lsun = 10.0**agn_log_lbol
        phot_per_lsun = interp_collapsed(grid_phot, axes, free_axis_values, kernel="pchip")
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return nenkova_agnfitter_2p_phot_collapsed
