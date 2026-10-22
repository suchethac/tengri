# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for Silva+04 AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
Silva, Maiolino & Granato (2004) semi-empirical torus model keyed on hydrogen
column density.

Records any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters`` and reads the
table there, keeping the table whole so it is interpolated in the runtime's axis order.

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

from tengri.components.agn._template_power_table import (
    lnu_per_lsun_from_power_table,
    power_weighted_template_photometry,
    read_at_fixed_axes,
)
from tengri.components.agn.silva04 import _load_silva04_arrays
from tengri.forward.precompute.templates import collapse_fixed_axes
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_pchip,
)

# Silva+04 grid parametrized by hydrogen column density only.
AXIS_PARAMS: tuple[str, ...] = ("agn_log_nh_silva",)


def precompute_silva04_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate Silva+04 torus templates through filter curves.

    For each log10(N_H) grid point, compute the filter-integrated photometry.
    Returns a dict with ``grid_phot`` and ``axes``.

    The lookup reads the table as L_ν [erg/s/Hz] per L_sun of torus power,
    normalized as the runtime ``silva04.py`` does.

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
        ``grid_phot`` : ndarray, shape (n_nh, n_filters + 1)
            Power-weighted table: raw band L_ν [erg/s/Hz per unit template power]
            and, last, the raw bolometric power; see Notes.
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
    """
    raw = _load_silva04_arrays(grid_path)
    grid = np.asarray(raw["template"], dtype=np.float64)  # (n_nh, n_wave)
    wave_grid = np.asarray(raw["wavelength"], dtype=np.float64)
    log_nh_axis = np.asarray(raw["log_nh_axis"], dtype=np.float64)

    # The runtime normalizes the interpolated template, so its off-node value is
    # weighted by each node's raw power: tabulate the raw band photometry and the
    # raw power and divide after interpolating (_template_power_table).
    power_table = power_weighted_template_photometry(
        grid,
        wave_grid,
        filter_waves,
        filter_trans,
        redshift,
        axes=(log_nh_axis,),
    )

    axes_jax = (jnp.asarray(log_nh_axis),)
    return {
        "grid_phot": power_table.table,
        "axes": axes_jax,
        "_preint": power_table.preint,
    }


def build_silva04_photometry_lookup(precomp: dict):
    """Build a JIT-compiled Silva+04 torus photometry function.

    Uses node-exact monotone-cubic (PCHIP) interpolation, matching the exact
    (non-precomputed) path in ``silva04.py`` (task3 fix round 1, RULING R11).

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_silva04_photometry` or :func:`precompute`
        (the Protocol-shaped entry point).

    Returns
    -------
    callable
        Function with signature::

            fn(agn_log_lbol, agn_log_nh_silva, agn_torus_frac)
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
    **JIT-compatible**: yes, the returned function uses ``jnp`` and
    PCHIP interpolation, which are JAX-native.

    **Gradient-safe**: yes, PCHIP (monotone-cubic) interpolation is fully
    differentiable and reproduces the tabulated grid nodes exactly.

    **Interpolation kernel**: PCHIP is node-exact and C¹-continuous, unlike
    the C²-smooth triweight *smoother* this replaced, which averaged
    neighboring nodes and smeared the template shape by ~9% at the boundary
    node (matching the exact path's fix, ``silva04.py``).
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]

    @jax.jit
    def silva04_phot(
        agn_log_lbol,
        agn_log_nh_silva,
        agn_torus_frac,
    ):
        """Compute Silva+04 torus photometry via PCHIP interpolation on 1D grid.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        # grid_phot is power-weighted: the ratio below is L_ν [erg/s/Hz] per L_sun
        # Return: L_bol_lsun [L_sun] * torus_frac * phot [erg/s/Hz/L_sun] = L_ν [erg/s/Hz]
        l_bol_lsun = 10.0**agn_log_lbol
        point = (agn_log_nh_silva,)
        phot_per_lsun = lnu_per_lsun_from_power_table(interp_nd_pchip(grid_phot, axes, point))
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
    """Build preintegrated Silva+04 grid, recording Fixed axes.

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
        Same as :func:`precompute_silva04_photometry`, plus ``_collapsed_axes``
        (axis index to its Fixed coordinate) when any :data:`AXIS_PARAMS` entry is
        Fixed. The table is not contracted: the lookup reads it at the Fixed coordinates.

    References
    ----------
    .. [1] L. Silva, R. Maiolino & G. L. Granato, "The nature of the Compton-thick
       AGN in NGC 1068 and implications for the cosmic X-ray background," MNRAS
       355, 973 (2004). arXiv:astro-ph/0403425.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    """
    result = precompute_silva04_photometry(
        grid_path, filter_waves, filter_trans, redshift=redshift
    )
    preint: PreintegratedGrid = result["_preint"]
    _, _, fixed = collapse_fixed_axes(preint, AXIS_PARAMS, parameters, origin="silva04_precompute")
    if not fixed:
        return result

    # The table is kept whole and read at the Fixed coordinates (read_at_fixed_axes):
    # contracting a Fixed axis first would interpolate in another axis order than
    # the runtime kernel, and the tensor-product PCHIP depends on that order.
    return {**result, "_collapsed_axes": fixed}


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
    **JIT-compatible**: yes, the returned function is fully JAX-native.

    **Gradient-safe**: yes, PCHIP (monotone-cubic) interpolation is fully
    differentiable and reproduces the tabulated grid nodes exactly.
    """
    if not preint.get("_collapsed_axes"):
        return build_silva04_photometry_lookup(preint)

    # Collapsed case: lookup takes (scale, *remaining_axis_values, torus_frac)
    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    collapsed = preint["_collapsed_axes"]

    @jax.jit
    def silva04_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """Compute Silva+04 torus photometry with collapsed (fixed) axes via PCHIP interp.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        # Same unit convention as build_silva04_photometry_lookup: L_ν [erg/s/Hz]
        l_bol_lsun = 10.0**agn_log_lbol
        phot_per_lsun = lnu_per_lsun_from_power_table(
            read_at_fixed_axes(grid_phot, axes, free_axis_values, collapsed)
        )
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return silva04_phot_collapsed
