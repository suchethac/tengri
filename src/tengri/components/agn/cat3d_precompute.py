# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for CAT3D-Wind AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
Hönig & Kishimoto (2017) clumpy-disc-plus-polar-wind torus model.

Auto-collapses any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters``.

References
----------
.. [1] S. F. Hönig & M. Kishimoto, "Dusty winds in active galactic nuclei: reconciling
   observations with models," ApJL 838,
   L20 (2017). arXiv:1703.07781. https://doi.org/10.3847/2041-8213/aa6838
.. [2] L. N. Martínez-Ramírez, G. Calistro Rivera, E. Lusso, et al.,
   "AGNfitter-rx: Modeling the radio-to-X-ray spectral energy
   distributions of AGNs," A&A 688, A46 (2024). arXiv:2405.12111.
   DOI: 10.1051/0004-6361/202449329.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components._collapsed_lookup import interp_collapsed
from tengri.components.agn._template_power_table import (
    lnu_per_lsun_from_power_table,
    power_weighted_template_photometry,
)
from tengri.components.agn.cat3d_wind import _load_cat3d_arrays
from tengri.forward.precompute.templates import collapse_fixed_axes
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_pchip,
)

# CAT3D-Wind grid parametrized by three axes: inclination (as cos),
# radial power-law index, and wind fraction.
AXIS_PARAMS: tuple[str, ...] = (
    "agn_cos_inc",
    "agn_a_cat3d",
    "agn_fwd_cat3d",
)


def precompute_cat3d_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate CAT3D-Wind torus templates through filter curves.

    For each (cos_inc, a, fwd) grid point, compute the filter-integrated
    photometry. Returns a dict with ``grid_phot`` and ``axes``.

    The lookup reads the table as L_ν [erg/s/Hz] per L_sun of torus power,
    normalized as the runtime ``cat3d_wind.py`` does.

    Parameters
    ----------
    grid_path : str
        Path to ``cat3d_wind_torus_grid.h5``.
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
        ``grid_phot`` : ndarray, shape (n_cos_inc, n_a, n_fwd, n_filters + 1)
            Power-weighted table: raw band L_ν [erg/s/Hz per unit template power]
            and, last, the raw bolometric power; see Notes.
        ``axes`` : tuple of 3 grid arrays (jnp.ndarray)
            Grid axes (cos_inc, a, fwd).
        ``_preint`` : PreintegratedGrid
            Internal preintegration data structure.

    References
    ----------
    .. [1] S. F. Hönig & M. Kishimoto, "Dusty winds in active galactic nuclei: reconciling
       observations with models," ApJL 838,
       L20 (2017). arXiv:1703.07781.

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

    **Grid reordering**: The native inclination axis is stored in degrees
    (ascending). This function converts to cos(incl) and reorders templates
    to match, mirroring the upstream ``cat3d_wind.create_cat3d_wind_from_grid``
    logic.
    """
    raw = _load_cat3d_arrays(grid_path)

    # Convert native inclination axis (degrees, ascending) to cos(incl).
    # Template's leading axis must follow suit: reorder and reverse.
    incl_deg = raw["incl_axis"]
    cos_inc_axis = np.cos(np.deg2rad(incl_deg))
    order = np.argsort(cos_inc_axis)
    cos_inc_axis = cos_inc_axis[order]
    template_reordered = raw["template"][order]

    grid = np.asarray(template_reordered, dtype=np.float64)  # (n_cos_inc, n_a, n_fwd, n_wave)
    wave_grid = np.asarray(raw["wavelength"], dtype=np.float64)
    a_axis = np.asarray(raw["a_axis"], dtype=np.float64)
    fwd_axis = np.asarray(raw["fwd_axis"], dtype=np.float64)

    # The runtime normalizes the interpolated template, so its off-node value is
    # weighted by each node's raw power: tabulate the raw band photometry and the
    # raw power and divide after interpolating (_template_power_table).
    preint = power_weighted_template_photometry(
        grid,
        wave_grid,
        filter_waves,
        filter_trans,
        redshift,
        axes=(cos_inc_axis, a_axis, fwd_axis),
    )

    axes_jax = (
        jnp.asarray(cos_inc_axis),
        jnp.asarray(a_axis),
        jnp.asarray(fwd_axis),
    )
    return {
        "grid_phot": preint.phot,
        "axes": axes_jax,
        "_preint": preint,
    }


def build_cat3d_photometry_lookup(precomp: dict):
    """Build a JIT-compiled CAT3D-Wind torus photometry function.

    Uses node-exact monotone-cubic (PCHIP) interpolation for C¹ gradients.

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_cat3d_photometry` or :func:`precompute`
        (the Protocol-shaped entry point).

    Returns
    -------
    callable
        Function with signature::

            fn(agn_log_lbol, agn_cos_inc, agn_a_cat3d, agn_fwd_cat3d,
               agn_torus_frac) -> ndarray, shape (n_filters,)

        Returns torus L_ν [erg/s/Hz]. Caller applies
        ``flux_scale = (1+z) / (4π d_L²)`` to get flux density.

    References
    ----------
    .. [1] S. F. Hönig & M. Kishimoto, "Dusty winds in active galactic nuclei: reconciling
       observations with models," ApJL 838,
       L20 (2017). arXiv:1703.07781.

    Notes
    -----
    **JIT-compatible**: yes, the returned function uses ``jnp`` and
    monotone-cubic interpolation, which are JAX-native.

    **Gradient-safe**: yes, node-exact PCHIP is C¹-differentiable.

    **Interpolation kernel**: node-exact monotone cubic (PCHIP), matching the
    exact-wave-grid path (:mod:`tengri.components.agn.cat3d_wind`) so the
    WavePrecomp photometry does not diverge from the exact SED. The C²-smooth
    triweight smoother previously used here averaged neighbors, smearing the
    torus mid-IR peak; PCHIP reproduces every AGNfitter node exactly while
    keeping continuous gradients for inference.
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]

    @jax.jit
    def cat3d_phot(
        agn_log_lbol,
        agn_cos_inc,
        agn_a_cat3d,
        agn_fwd_cat3d,
        agn_torus_frac,
    ):
        """Compute CAT3D-Wind torus photometry via monotone-cubic interpolation.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        # grid_phot is power-weighted: the ratio below is L_ν [erg/s/Hz] per L_sun
        # Return: L_bol_lsun [L_sun] * torus_frac * phot [erg/s/Hz/L_sun] = L_ν [erg/s/Hz]
        l_bol_lsun = 10.0**agn_log_lbol
        point = (agn_cos_inc, agn_a_cat3d, agn_fwd_cat3d)
        phot_per_lsun = lnu_per_lsun_from_power_table(interp_nd_pchip(grid_phot, axes, point))
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return cat3d_phot


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build preintegrated CAT3D grid, auto-collapsing Fixed-parameter axes.

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
        Path to ``cat3d_wind_torus_grid.h5``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_cat3d_photometry` but with grid
        axes collapsed for any Fixed :data:`AXIS_PARAMS` entry.

    References
    ----------
    .. [1] S. F. Hönig & M. Kishimoto, "Dusty winds in active galactic nuclei: reconciling
       observations with models," ApJL 838,
       L20 (2017). arXiv:1703.07781.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    """
    result = precompute_cat3d_photometry(grid_path, filter_waves, filter_trans, redshift=redshift)

    preint: PreintegratedGrid = result["_preint"]
    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint, AXIS_PARAMS, parameters, origin="cat3d_precompute"
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
    """Build the runtime CAT3D-Wind photometry lookup from a preintegrated dict.

    When no axes are collapsed, delegates to
    :func:`build_cat3d_photometry_lookup`. When some axes are collapsed
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
    .. [1] S. F. Hönig & M. Kishimoto, "Dusty winds in active galactic nuclei: reconciling
       observations with models," ApJL 838,
       L20 (2017). arXiv:1703.07781.

    Notes
    -----
    **JIT-compatible**: yes, the returned function is fully JAX-native.

    **Gradient-safe**: yes, node-exact PCHIP is C¹-differentiable.
    """
    if not preint.get("_collapsed_axes"):
        return build_cat3d_photometry_lookup(preint)

    # Collapsed case: lookup takes (scale, *remaining_axis_values, torus_frac)
    grid_phot = preint["grid_phot"]
    axes = preint["axes"]

    @jax.jit
    def cat3d_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """Compute CAT3D-Wind torus photometry with collapsed (fixed) axes via PCHIP interp.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        # Same unit convention as build_cat3d_photometry_lookup: L_ν [erg/s/Hz]
        l_bol_lsun = 10.0**agn_log_lbol
        phot_per_lsun = lnu_per_lsun_from_power_table(
            interp_collapsed(grid_phot, axes, free_axis_values, kernel="pchip")
        )
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return cat3d_phot_collapsed
