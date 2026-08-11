# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for Silva+04 AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
Silva, Maiolino & Granato (2004) semi-empirical torus model keyed on hydrogen
column density.

Auto-collapses any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters``.

References
----------
.. [1] L. Silva, R. Maiolino & G. L. Granato, "The nature of the Compton-thick
   AGN in NGC 1068 and implications for the cosmic X-ray background," MNRAS
   355, 973 (2004). arXiv:astro-ph/0403425.
.. [2] G. Calistro Rivera et al., "AGNfitter: a Bayesian MCMC approach to
   fitting spectral energy distributions of AGNs," ApJ 833, 98 (2016).
   arXiv:1606.05648.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components._collapsed_lookup import interp_collapsed
from tengri.components.agn.silva04 import _load_silva04_arrays
from tengri.forward.precompute.templates import (
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_triweight,
    slice_fixed_axes,
)
from tengri.utils.interpolation import edges_for_grid

# Silva+04 grid parametrized by hydrogen column density only.
AXIS_PARAMS: tuple[str, ...] = ("silva04_log_NH",)


def precompute_silva04_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate Silva+04 torus templates through filter curves.

    For each log10(N_H) grid point, compute the filter-integrated photometry.
    Returns a dict with ``grid_phot`` and ``axes``.

    Templates are frequency-normalized (matching the runtime normalization
    in ``silva04.py``) so that ``build_silva04_photometry_lookup`` returns
    L_ν [erg/s/Hz] per L_sun of bolometric luminosity.

    Parameters
    ----------
    grid_path : str
        Path to ``silva04_torus_grid.h5``.
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
        ``grid_phot`` : ndarray, shape (n_nh, n_filters)
            Filter-integrated L_ν [erg/s/Hz] per L_sun (unit torus fraction).
        ``axes`` : tuple of 1 grid array (jnp.ndarray)
            Grid axis (log10(N_H)).
        ``_preint`` : PreintegratedGrid
            Internal preintegration data structure.

    References
    ----------
    .. [1] L. Silva, R. Maiolino & G. L. Granato, "The nature of the Compton-thick
       AGN in NGC 1068 and implications for the cosmic X-ray background," MNRAS
       355, 973 (2004). arXiv:astro-ph/0403425.

    Notes
    -----
    **JIT-compatible**: no — this is a build-time function using NumPy.

    **Build-time operation**: This function performs frequency-domain
    integration via NumPy. The precomputed photometry is grid-independent
    (depends only on filter curves and redshift, not wavelength grid).

    **Normalization**: Templates are frequency-normalized so that the
    integration constant equals L_sun / trapz(template, nu). This matches
    the runtime normalization in ``silva04.py``.
    """
    from tengri.components.agn._phys import C_LIGHT as _C_CGS
    from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

    raw = _load_silva04_arrays(grid_path)
    grid = np.asarray(raw["template"], dtype=np.float64)  # (n_nh, n_wave)
    wave_grid = np.asarray(raw["wavelength"], dtype=np.float64)
    log_nh_axis = np.asarray(raw["log_nh_axis"], dtype=np.float64)

    # Normalize each template by its frequency integral.
    # This matches silva04.py's normalization:
    #   L_ν = L_bol * torus_frac * template / trapz(template, nu)
    # Precomputed: lnu_per_lsun = LSUN_ERG * template / trapz(template, nu)
    # Runtime: L_bol_lsun * torus_frac * lnu_per_lsun → L_ν [erg/s/Hz]
    nu_grid = _C_CGS / (wave_grid * 1e-8)  # Hz (decreasing order)
    sort_idx = np.argsort(nu_grid)
    nu_sorted = nu_grid[sort_idx]

    n_nh, _ = grid.shape
    lnu_grid = np.empty_like(grid)

    for i in range(n_nh):
        template = grid[i]
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
        axes=(log_nh_axis,),
        energy_normalize=False,  # templates already normalized per L_sun
        units="lnu",
    )

    axes_jax = (jnp.asarray(log_nh_axis),)
    return {
        "grid_phot": preint.phot,
        "axes": axes_jax,
        "_preint": preint,
    }


def build_silva04_photometry_lookup(precomp: dict):
    """Build a JIT-compiled Silva+04 torus photometry function.

    Uses triweight interpolation for C²-continuous gradients.

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_silva04_photometry` or :func:`precompute`
        (the Protocol-shaped entry point).

    Returns
    -------
    callable
        Function with signature::

            fn(agn_log_lbol, silva04_log_NH, agn_torus_frac)
                -> ndarray, shape (n_filters,)

        Returns torus L_ν [erg/s/Hz]. Caller applies
        ``flux_scale = (1+z) / (4π d_L²)`` to get flux density.

    References
    ----------
    .. [1] L. Silva, R. Maiolino & G. L. Granato, "The nature of the Compton-thick
       AGN in NGC 1068 and implications for the cosmic X-ray background," MNRAS
       355, 973 (2004). arXiv:astro-ph/0403425.

    Notes
    -----
    **JIT-compatible**: yes — the returned function uses ``jnp`` and
    triweight interpolation, which are JAX-native.

    **Gradient-safe**: yes — triweight kernel is fully differentiable.

    **Interpolation kernel**: Triweight kernel provides C²-continuous
    gradients for autodiff, unlike nearest-neighbor or linear interpolation.
    This is important for robust inference when Silva+04 parameters are
    fitted via gradient descent.
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]
    edges = tuple(edges_for_grid(ax) for ax in axes)

    @jax.jit
    def silva04_phot(
        agn_log_lbol,
        silva04_log_NH,
        agn_torus_frac,
    ):
        """Compute Silva+04 torus photometry via triweight interpolation on 1D grid.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        # grid_phot stores L_ν [erg/s/Hz] per L_sun of L_bol (unit torus fraction)
        # Return: L_bol_lsun [L_sun] * torus_frac * phot [erg/s/Hz/L_sun] = L_ν [erg/s/Hz]
        l_bol_lsun = 10.0**agn_log_lbol
        point = (silva04_log_NH,)
        phot_per_lsun = interp_nd_triweight(grid_phot, axes, edges, point)
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return silva04_phot


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build preintegrated Silva+04 grid, auto-collapsing Fixed-parameter axes.

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
        Path to ``silva04_torus_grid.h5``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_silva04_photometry` but with grid
        axes collapsed for any Fixed :data:`AXIS_PARAMS` entry.

    References
    ----------
    .. [1] L. Silva, R. Maiolino & G. L. Granato, "The nature of the Compton-thick
       AGN in NGC 1068 and implications for the cosmic X-ray background," MNRAS
       355, 973 (2004). arXiv:astro-ph/0403425.

    Notes
    -----
    **JIT-compatible**: no — this is a build-time function using NumPy.
    """
    result = precompute_silva04_photometry(
        grid_path, filter_waves, filter_trans, redshift=redshift
    )
    if parameters is None:
        return result

    preint: PreintegratedGrid = result["_preint"]
    fixed_values = parameters.get_fixed_values()
    fixed: dict[int, float] = {}
    for i, pname in enumerate(AXIS_PARAMS):
        if pname in fixed_values:
            fixed[i] = float(fixed_values[pname])
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
    """Build the runtime Silva+04 photometry lookup from a preintegrated dict.

    When no axes are collapsed, delegates to
    :func:`build_silva04_photometry_lookup`. When some axes are collapsed
    (fixed at preintegration time), the returned function expects only the
    remaining free parameter values.

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
        JIT-compiled photometry lookup function with signature::

            fn(agn_log_lbol, *free_axis_values, agn_torus_frac)
                -> ndarray, shape (n_filters,)

        Returns torus L_ν [erg/s/Hz].

    References
    ----------
    .. [1] L. Silva, R. Maiolino & G. L. Granato, "The nature of the Compton-thick
       AGN in NGC 1068 and implications for the cosmic X-ray background," MNRAS
       355, 973 (2004). arXiv:astro-ph/0403425.

    Notes
    -----
    **JIT-compatible**: yes — the returned function is fully JAX-native.

    **Gradient-safe**: yes — triweight interpolation is fully differentiable.
    """
    if not preint.get("_collapsed_axes"):
        return build_silva04_photometry_lookup(preint)

    # Collapsed case: lookup takes (scale, *remaining_axis_values, torus_frac)
    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    edges = tuple(edges_for_grid(ax) for ax in axes)

    @jax.jit
    def silva04_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """Compute Silva+04 torus photometry with collapsed (fixed) axes via triweight interp.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        # Same unit convention as build_silva04_photometry_lookup: L_ν [erg/s/Hz]
        l_bol_lsun = 10.0**agn_log_lbol
        phot_per_lsun = interp_collapsed(
            grid_phot, axes, free_axis_values, kernel="triweight", edges=edges
        )
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return silva04_phot_collapsed
