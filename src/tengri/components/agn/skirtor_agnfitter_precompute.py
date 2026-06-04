# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for SKIRTOR_mean_3p AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the SKIRTOR_mean_3p three-parameter clumpy torus model (Stalevski et al. 2016)
as packaged by AGNfitter-rX.

Auto-collapses any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters``.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modelling of the dusty
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

import jax.numpy as jnp
import numpy as np

from tengri.components.agn.skirtor_agnfitter import _load_skirtor_agnfitter_arrays
from tengri.forward.precompute.templates import (
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    interp_nd_triweight,
)
from tengri.utils.interpolation import edges_for_grid

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
    .. [1] M. Stalevski et al., "3D radiative transfer modelling of the dusty
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

    Uses triweight interpolation for C²-continuous gradients.

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
    edges = tuple(edges_for_grid(ax) for ax in axes)

    grid_jax = jnp.asarray(grid_phot)

    def skirtor_agnfitter_photometry(
        agn_log_lbol: float = 11.0,
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

        **Gradient-safe**: yes — triweight interpolation is C² differentiable.

        See Also
        --------
        precompute_skirtor_agnfitter_photometry
        """
        phot = interp_nd_triweight(
            grid_jax,
            axes,
            edges,
            (agn_oa_skirtor, agn_incl_skirtor, agn_tv_skirtor),
        )
        l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
        return l_scale * phot

    return skirtor_agnfitter_photometry


def get_skirtor_agnfitter_precompute_config(
    fixed_params: dict[str, Any],
) -> dict[str, Any] | None:
    """Determine which SKIRTOR_mean_3p axes to collapse (fixed parameters).

    Parameters
    ----------
    fixed_params : dict
        Frozen parameters passed from the user model.

    Returns
    -------
    dict or None
        Configuration dict with keys ``to_collapse`` (list of axis indices),
        or None if no axes are fixed.

    Notes
    -----
    If all three parameters (oa, incl, tv) are fixed, returns the grid point
    directly without triweight interpolation. If some are fixed, the
    precomputation automatically collapses those axes via
    :func:`~tengri.utils.grid_interp.slice_fixed_axes`.
    """
    to_collapse = []
    for i, param_key in enumerate(AXIS_PARAMS):
        if param_key in fixed_params:
            to_collapse.append(i)

    if not to_collapse:
        return None

    return {"to_collapse": to_collapse}
