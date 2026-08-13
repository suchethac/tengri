# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for SKIRTOR_mean_3p AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the SKIRTOR_mean_3p three-parameter clumpy torus model (Stalevski et al. 2016)
as packaged by AGNfitter-rX.

Auto-collapses any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters``.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
   torus around AGN — the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski et al., "The dust covering factor in AGN — combining the
   IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
   arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444
.. [3] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components._collapsed_lookup import interp_collapsed
from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.components.agn.skirtor_agnfitter import _load_skirtor_agnfitter_arrays
from tengri.forward.precompute.templates import (
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_pchip,
)
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

# SKIRTOR_mean_3p grid parametrized by three independent axes:
# oa (half-opening angle), incl (inclination), tv (optical depth).
AXIS_PARAMS: tuple[str, ...] = (
    "skirtor_agnfitter_oa",
    "skirtor_agnfitter_incl",
    "skirtor_agnfitter_tv",
)


def precompute_skirtor_agnfitter_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate SKIRTOR_mean_3p torus templates through filter curves.

    For each (oa, incl, tv) grid point, compute the filter-integrated photometry.
    Returns a dict with ``grid_phot`` and ``axes``.

    Templates are frequency-normalized (matching the runtime normalization
    in ``skirtor_agnfitter.py``) so that ``build_skirtor_agnfitter_photometry_lookup``
    returns L_ν [erg/s/Hz] per L_sun of bolometric luminosity.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean3p_torus_grid.h5``.
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
        ``grid_phot`` : ndarray, shape (n_oa, n_incl, n_tv, n_filters)
            Filter-integrated L_ν [erg/s/Hz] per L_sun (unit torus fraction).
        ``axes`` : tuple of 3 grid arrays (jnp.ndarray)
            Grid axes (oa, incl, tv).
        ``_preint`` : PreintegratedGrid
            Internal preintegration data structure.

    References
    ----------
    .. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
       torus around AGN — the influence of clumping," MNRAS, 420, 2756 (2012).
       arXiv:1109.1286.
    .. [2] M. Stalevski et al., "The dust covering factor in AGN — combining the
       IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
       arXiv:1602.01954.

    Notes
    -----
    **JIT-compatible**: no — this is a build-time function using NumPy.

    **Build-time operation**: This function performs frequency-domain
    integration via NumPy. The precomputed photometry is grid-independent
    (depends only on filter curves and redshift, not wavelength grid).

    **Normalization**: Templates are frequency-normalized so that the
    integration constant equals L_sun / trapz(template, nu). This matches
    the runtime normalization in ``skirtor_agnfitter.py``.
    """
    from tengri.components.agn._phys import C_LIGHT as _C_CGS
    from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

    raw = _load_skirtor_agnfitter_arrays(grid_path)
    grid = np.asarray(raw["template"], dtype=np.float64)  # (n_oa, n_incl, n_tv, n_wave)
    wave_grid = np.asarray(raw["wavelength"], dtype=np.float64)
    oa_axis = np.asarray(raw["oa_axis"], dtype=np.float64)
    incl_axis = np.asarray(raw["incl_axis"], dtype=np.float64)
    tv_axis = np.asarray(raw["tv_axis"], dtype=np.float64)

    # Normalize each template by its frequency integral.
    # This matches skirtor_agnfitter.py's normalization:
    #   L_ν = L_bol * torus_frac * template / trapz(template, nu)
    # Precomputed: lnu_per_lsun = LSUN_ERG * template / trapz(template, nu)
    # Runtime: L_bol_lsun * torus_frac * lnu_per_lsun → L_ν [erg/s/Hz]
    nu_grid = _C_CGS / (wave_grid * 1e-8)  # Hz (decreasing order)
    sort_idx = np.argsort(nu_grid)
    nu_sorted = nu_grid[sort_idx]

    n_oa, n_incl, n_tv, _n_wave = grid.shape
    lnu_grid = np.empty_like(grid)

    for i_oa in range(n_oa):
        for i_incl in range(n_incl):
            for i_tv in range(n_tv):
                template = grid[i_oa, i_incl, i_tv]
                integral = np.trapezoid(template[sort_idx], nu_sorted)
                integral_safe = max(abs(integral), 1e-100)
                lnu_grid[i_oa, i_incl, i_tv] = _LSUN_ERG * template / integral_safe

    preint = precompute_template_photometry(
        templates=lnu_grid,
        wave_rest=wave_grid,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(oa_axis, incl_axis, tv_axis),
        energy_normalize=False,  # templates already normalized per L_sun
        units="lnu",
    )

    axes_jax = (
        jnp.asarray(oa_axis),
        jnp.asarray(incl_axis),
        jnp.asarray(tv_axis),
    )
    return {
        "grid_phot": preint.phot,
        "axes": axes_jax,
        "_preint": preint,
    }


def build_skirtor_agnfitter_photometry_lookup(precomp: dict):
    """Build a JIT-compiled SKIRTOR_mean_3p torus photometry function.

    Uses node-exact monotone-cubic (PCHIP) interpolation for C¹ gradients.

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_skirtor_agnfitter_photometry`.

    Returns
    -------
    callable
        ``fn(agn_log_lbol, agn_oa_skirtor, agn_incl_skirtor, agn_tv_skirtor,
        agn_torus_frac, **_) -> photometry [erg/s/Hz]``.
        Output has the same shape as ``precomp["grid_phot"][:, :, :, :]``
        but along the filter axis.

    Notes
    -----
    **JIT-compatible**: yes — pure JAX with no data I/O.
    """
    from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]

    grid_jax = jnp.asarray(grid_phot)

    def skirtor_agnfitter_photometry(
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_oa_skirtor: float = 40.0,
        agn_incl_skirtor: float = 30.0,
        agn_tv_skirtor: float = 7.0,
        agn_torus_frac: float = 0.5,
        **_kwargs,
    ) -> jnp.ndarray:
        r"""SKIRTOR_mean_3p torus photometry at a given (oa, incl, tv).

        Parameters
        ----------
        agn_log_lbol : float
            ``log10(L_bol / L_sun)``. Default 11.0.
        agn_oa_skirtor : float
            Half-opening angle [deg]. Default 40.0.
        agn_incl_skirtor : float
            Inclination [deg]. Default 30.0.
        agn_tv_skirtor : float
            Equatorial optical depth τ_9.7. Default 7.0.
        agn_torus_frac : float
            Torus reprocessing fraction. Default 0.5.

        Returns
        -------
        ndarray, shape (n_filters,)
            Photometry [erg/s/Hz] per filter.

        Notes
        -----
        **JIT-compatible**: yes.

        **Gradient-safe**: yes — node-exact PCHIP is C¹ differentiable.

        See Also
        --------
        precompute_skirtor_agnfitter_photometry
        """
        phot = interp_nd_pchip(
            grid_jax,
            axes,
            (agn_oa_skirtor, agn_incl_skirtor, agn_tv_skirtor),
        )
        l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
        return l_scale * phot

    return skirtor_agnfitter_photometry


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build the preintegrated SKIRTOR_mean_3p grid, auto-collapsing Fixed axes.

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
        Path to ``skirtor_mean3p_torus_grid.h5``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_skirtor_agnfitter_photometry`, with grid
        axes collapsed for any Fixed :data:`AXIS_PARAMS` entry.

    Notes
    -----
    **JIT-compatible**: no — this is a build-time function using NumPy.
    """
    result = precompute_skirtor_agnfitter_photometry(
        grid_path, filter_waves, filter_trans, redshift=redshift
    )
    preint: PreintegratedGrid = result["_preint"]
    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint, AXIS_PARAMS, parameters, origin="skirtor_agnfitter_precompute"
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
    """Build the runtime SKIRTOR_mean_3p photometry lookup from a preintegrated dict.

    When no axes are collapsed, delegates to
    :func:`build_skirtor_agnfitter_photometry_lookup`. When some axes are
    collapsed (fixed at preintegration time), the returned function expects only
    the remaining free parameter values.

    Parameters
    ----------
    preint : dict
        Preintegrated data dict with keys ``"grid_phot"``, ``"axes"``, and
        optionally ``"_collapsed_axes"``.
    free_param_names : tuple of str or None, optional
        Names of the remaining free axes in the collapsed case (unused in the
        default no-collapse case).

    Returns
    -------
    callable
        JIT-compiled photometry lookup returning torus L_ν [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — the returned function is fully JAX-native.

    **Gradient-safe**: yes — node-exact PCHIP is C¹-differentiable.
    """
    if not preint.get("_collapsed_axes"):
        return build_skirtor_agnfitter_photometry_lookup(preint)

    grid_phot = preint["grid_phot"]
    axes = preint["axes"]

    @jax.jit
    def skirtor_agnfitter_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """SKIRTOR_mean_3p torus photometry with collapsed (fixed) axes via PCHIP."""
        l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
        phot = interp_collapsed(grid_phot, axes, free_axis_values, kernel="pchip")
        return l_scale * phot

    return skirtor_agnfitter_phot_collapsed
