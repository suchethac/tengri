# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for Nenkova+08 AGN torus templates (AGNfitter-rX).

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
Nenkova et al. (2008) CLUMPY inclination-averaged torus model from the
AGNfitter-rX ``NK0_mean_1p`` library, keyed on ``cos(inclination)``.

Auto-collapses any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters``.

References
----------
.. [1] M. Nenkova et al., "Revisiting the AGN torus with MIDI and VISIR
   Herschel observations," ApJ 685, 160 (2008). arXiv:0806.1512.
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
from tengri.components.agn.nenkova_agnfitter import _load_nenkova_agnfitter_arrays
from tengri.forward.precompute.templates import (
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_pchip,
)

# Nenkova AGNfitter grid parametrized by inclination only.
AXIS_PARAMS: tuple[str, ...] = ("agn_cos_inc",)


def precompute_nenkova_agnfitter_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate Nenkova AGNfitter torus templates through filter curves.

    For each inclination grid point, compute the filter-integrated photometry.
    Returns a dict with ``grid_phot`` and ``axes``.

    Templates are frequency-normalized (matching the runtime normalization
    in ``nenkova_agnfitter.py``) so that ``build_nenkova_agnfitter_photometry_lookup``
    returns L_ν [erg/s/Hz] per L_sun of bolometric luminosity.

    Parameters
    ----------
    grid_path: str
        Path to ``nenkova_agnfitter_torus_grid.h5``.
    filter_waves: list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans: list[ndarray]
        Transmission per filter (0–1).
    redshift: float, optional
        Source redshift. Used to shift rest-frame templates into the
        observed frame before integrating against observed-frame filters.
        Default 0.0.

    Returns
    -------
    dict
        ``grid_phot``: ndarray, shape (n_incl, n_filters)
            Filter-integrated L_ν [erg/s/Hz] per L_sun (unit torus fraction).
        ``axes``: tuple of 1 grid array (jnp.ndarray)
            Grid axis (cos_inc, ascending).
        ``_preint``: PreintegratedGrid
            Internal preintegration data structure.

    References
    ----------
    .. [1] M. Nenkova et al., "Revisiting the AGN torus with MIDI and VISIR
       Herschel observations," ApJ 685, 160 (2008). arXiv:0806.1512.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

    **Build-time operation**: This function performs frequency-domain
    integration via NumPy. The precomputed photometry is grid-independent
    (depends only on filter curves and redshift, not wavelength grid).

    **Normalization**: Templates are frequency-normalized so that the
    integration constant equals L_sun / trapz(template, nu). This matches
    the runtime normalization in ``nenkova_agnfitter.py``.
    """
    from tengri.components.agn._phys import C_LIGHT as _C_CGS
    from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

    raw = _load_nenkova_agnfitter_arrays(grid_path)
    grid = np.asarray(raw["template"], dtype=np.float64)  # (n_incl, n_wave)
    wave_grid = np.asarray(raw["wavelength"], dtype=np.float64)
    incl_deg_axis = np.asarray(raw["incl_axis"], dtype=np.float64)

    # Convert inclination (deg, ascending) to cos(incl), reorder ascending.
    # This matches the runtime path in nenkova_agnfitter.py.
    cos_inc_axis = np.cos(np.deg2rad(incl_deg_axis))
    order = np.argsort(cos_inc_axis)
    cos_inc_axis = cos_inc_axis[order]
    grid_reordered = grid[order]

    # Normalize each template by its frequency integral.
    # This matches nenkova_agnfitter.py's normalization:
    #   L_ν = L_bol * torus_frac * template / trapz(template, nu)
    # Precomputed: lnu_per_lsun = LSUN_ERG * template / trapz(template, nu)
    # Runtime: L_bol_lsun * torus_frac * lnu_per_lsun → L_ν [erg/s/Hz]
    nu_grid = _C_CGS / (wave_grid * 1e-8)  # Hz (decreasing order)
    sort_idx = np.argsort(nu_grid)
    nu_sorted = nu_grid[sort_idx]

    n_incl, _ = grid_reordered.shape
    lnu_grid = np.empty_like(grid_reordered)

    for i in range(n_incl):
        template = grid_reordered[i]
        integral = np.trapezoid(template[sort_idx], nu_sorted)
        integral_safe = max(abs(integral), 1e-100)
        lnu_grid[i] = _LSUN_ERG * template / integral_safe

    preint = precompute_template_photometry(
        templates=lnu_grid,
        wave_rest=wave_grid,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(cos_inc_axis,),
        energy_normalize=False,  # templates already normalized per L_sun
        units="lnu",
    )

    axes_jax = (jnp.asarray(cos_inc_axis),)
    return {
        "grid_phot": preint.phot,
        "axes": axes_jax,
        "_preint": preint,
    }


def build_nenkova_agnfitter_photometry_lookup(precomp: dict):
    """Build a JIT-compiled Nenkova AGNfitter torus photometry function.

    Uses node-exact PCHIP interpolation for monotone-cubic accuracy.

    Parameters
    ----------
    precomp: dict
        Output of :func:`precompute_nenkova_agnfitter_photometry` or
        :func:`precompute` (the Protocol-shaped entry point).

    Returns
    -------
    callable
        Function with signature::

            fn(agn_log_lbol, agn_cos_inc, agn_torus_frac)
                -> ndarray, shape (n_filters,)

        Returns torus L_ν [erg/s/Hz]. Caller applies
        ``flux_scale = (1+z) / (4π d_L²)`` to get flux density.

    References
    ----------
    .. [1] M. Nenkova et al., "Revisiting the AGN torus with MIDI and VISIR
       Herschel observations," ApJ 685, 160 (2008). arXiv:0806.1512.

    Notes
    -----
    **JIT-compatible**: yes, the returned function uses ``jnp`` and
    PCHIP interpolation, which are JAX-native.

    **Gradient-safe**: yes; PCHIP kernel is C¹-continuous.

    **Interpolation kernel**: PCHIP (monotone-cubic) interpolation is
    node-exact and provides C¹-continuous gradients for autodiff, ensuring
    accurate inference when inclination is fitted via gradient descent.
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]

    @jax.jit
    def nenkova_agnfitter_phot(
        agn_log_lbol,
        agn_cos_inc,
        agn_torus_frac,
    ):
        """Compute Nenkova AGNfitter torus photometry via PCHIP interpolation.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        # grid_phot stores L_ν [erg/s/Hz] per L_sun of L_bol (unit torus fraction)
        # Return: L_bol_lsun [L_sun] * torus_frac * phot [erg/s/Hz/L_sun] = L_ν [erg/s/Hz]
        l_bol_lsun = 10.0**agn_log_lbol
        point = (agn_cos_inc,)
        phot_per_lsun = interp_nd_pchip(grid_phot, axes, point)
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return nenkova_agnfitter_phot


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build preintegrated Nenkova AGNfitter grid, auto-collapsing Fixed-parameter axes.

    Parameters
    ----------
    filter_waves: list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans: list[ndarray]
        Transmission per filter (0–1).
    redshift: float
        Source redshift. [dimensionless]
    parameters: Parameters | None
        Parameters spec, used to detect Fixed-axis parameters.
    grid_path: str, keyword-only
        Path to ``nenkova_agnfitter_torus_grid.h5``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_nenkova_agnfitter_photometry` but with
        grid axes collapsed for any Fixed :data:`AXIS_PARAMS` entry.

    References
    ----------
    .. [1] M. Nenkova et al., "Revisiting the AGN torus with MIDI and VISIR
       Herschel observations," ApJ 685, 160 (2008). arXiv:0806.1512.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    """
    result = precompute_nenkova_agnfitter_photometry(
        grid_path, filter_waves, filter_trans, redshift=redshift
    )
    preint: PreintegratedGrid = result["_preint"]
    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint, AXIS_PARAMS, parameters, origin="nenkova_agnfitter_precompute"
    )
    if not fixed:
        return result

    # Rebuild dict view; drop the axes that were collapsed
    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def build_lookup(preint: dict, *, free_param_names: tuple[str, ...] | None = None):
    """Build the runtime Nenkova AGNfitter photometry lookup from a preintegrated dict.

    When no axes are collapsed, delegates to
    :func:`build_nenkova_agnfitter_photometry_lookup`. When some axes are
    collapsed (fixed at preintegration time), the returned function expects
    only the remaining free parameter values.

    Parameters
    ----------
    preint: dict
        Preintegrated data dict with keys ``"grid_phot"``, ``"axes"``,
        and optionally ``"_collapsed_axes"`` and ``"_preint"``.
    free_param_names: tuple of str or None, optional
        Names of remaining free axes in the collapsed case.
        Not used in the default (no-collapse) case.

    Returns
    -------
    callable
        JIT-compiled photometry lookup function with signature::

            fn(agn_log_lbol, *free_axis_values, agn_torus_frac)
                -> ndarray, shape (n_filters,)

        Returns torus L_ν [erg/s/Hz].

    References
    ----------
    .. [1] M. Nenkova et al., "Revisiting the AGN torus with MIDI and VISIR
       Herschel observations," ApJ 685, 160 (2008). arXiv:0806.1512.

    Notes
    -----
    **JIT-compatible**: yes, the returned function is fully JAX-native.

    **Gradient-safe**: yes; PCHIP interpolation is fully differentiable.
    """
    if not preint.get("_collapsed_axes"):
        return build_nenkova_agnfitter_photometry_lookup(preint)

    # Collapsed case: lookup takes (scale, *remaining_axis_values, torus_frac)
    grid_phot = preint["grid_phot"]
    axes = preint["axes"]

    @jax.jit
    def nenkova_agnfitter_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """Compute Nenkova AGNfitter torus photometry with collapsed (fixed) axes via PCHIP.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        # Same unit convention as build_nenkova_agnfitter_photometry_lookup: L_ν [erg/s/Hz]
        l_bol_lsun = 10.0**agn_log_lbol
        phot_per_lsun = interp_collapsed(grid_phot, axes, free_axis_values, kernel="pchip")
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return nenkova_agnfitter_phot_collapsed
