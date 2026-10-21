# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for SKIRTOR_mean_1p AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the one-parameter (inclination-only) SKIRTOR-AGNfitter torus reduction
(Stalevski et al. 2016), as packaged by AGNfitter-rX.

Auto-collapses the axis when its corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters``.

References
----------
.. [1] M. Stalevski, C. Ricci, Y. Ueda, P. Lira, J. Fritz, and M. Baes,
   "The dust covering factor in active galactic nuclei," MNRAS, 458,
   2288 (2016). arXiv:1602.06954. bibcode:2016MNRAS.458.2288S.
   https://doi.org/10.1093/mnras/stw444
.. [2] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components._collapsed_lookup import interp_collapsed
from tengri.components.agn._params import DEFAULT_AGN_COS_INC, DEFAULT_AGN_LOG_LBOL
from tengri.components.agn._skirtor_agnfitter_sightline import (
    accepts_retired_incl_skirtor,
    cos_inc_axis_from_incl,
)
from tengri.components.agn._torus_inclination import node_power_scale
from tengri.components.agn.skirtor_agnfitter_1p import (
    TORUS_INCLINATION,
    _load_skirtor_agnfitter_1p_arrays,
    load_skirtor_agnfitter_1p_grid,
)
from tengri.forward.precompute.templates import (
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_pchip,
)
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

# SKIRTOR_mean_1p grid parametrized by inclination only, keyed by the model's one
# inclination agn_cos_inc (the library's degree axis re-keyed by ascending cos i).
AXIS_PARAMS: tuple[str, ...] = ("agn_cos_inc",)


def precompute_skirtor_agnfitter_1p_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate SKIRTOR_mean_1p torus templates through filter curves.

    For each inclination grid point, compute the filter-integrated
    photometry. Returns a dict with ``grid_phot`` and ``axes``.

    Templates are frequency-normalized (matching the runtime normalization
    in ``skirtor_agnfitter_1p.py``) so that
    ``build_skirtor_agnfitter_1p_photometry_lookup`` returns L_ν [erg/s/Hz]
    per L_sun of bolometric luminosity.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean1p_torus_grid.h5``.
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0–1).
    redshift : float, optional
        Source redshift. Default 0.0.

    Returns
    -------
    dict
        ``grid_phot`` : ndarray, shape (n_incl, n_filters)
            Filter-integrated L_ν [erg/s/Hz] per L_sun (unit torus fraction).
        ``axes`` : tuple of 1 grid array (jnp.ndarray)
            Grid axis (cos i, ascending).
        ``_preint`` : PreintegratedGrid
            Internal preintegration data structure.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

    **Normalization**: Templates are frequency-normalized so that the
    integration constant is L_sun x anisotropy / native_integral(template). This matches
    the runtime normalization in ``skirtor_agnfitter_1p.py``.
    """

    raw = _load_skirtor_agnfitter_1p_arrays(grid_path)
    grid = np.asarray(raw["template"], dtype=np.float64)  # (n_incl, n_wave)
    wave_grid = np.asarray(raw["wavelength"], dtype=np.float64)
    incl_axis = np.asarray(raw["incl_axis"], dtype=np.float64)

    # Each record at unit power times its anisotropy a / mean a (the scalar table),
    # as the runtime path normalizes the records (skirtor_agnfitter_1p.TORUS_INCLINATION).
    runtime_grid = load_skirtor_agnfitter_1p_grid(grid_path)
    scale = node_power_scale(runtime_grid, TORUS_INCLINATION)
    lnu_grid, cos_inc_axis = cos_inc_axis_from_incl(
        _LSUN_ERG * grid * scale[..., None], incl_axis, axis=0
    )

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


def build_skirtor_agnfitter_1p_photometry_lookup(precomp: dict):
    """Build a JIT-compiled SKIRTOR_mean_1p torus photometry function.

    Uses node-exact monotone-cubic (PCHIP) interpolation for C¹ gradients.

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_skirtor_agnfitter_1p_photometry`.

    Returns
    -------
    callable
        ``fn(agn_log_lbol, agn_cos_inc, agn_torus_frac, **_)
        -> photometry [erg/s/Hz]``.

    Notes
    -----
    **JIT-compatible**: yes, pure JAX with no data I/O.
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]

    grid_jax = jnp.asarray(grid_phot)

    @accepts_retired_incl_skirtor
    def skirtor_agnfitter_1p_photometry(
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_cos_inc: float = DEFAULT_AGN_COS_INC,
        agn_torus_frac: float = 0.5,
        **_kwargs,
    ) -> jnp.ndarray:
        r"""SKIRTOR_mean_1p torus photometry at a given inclination.

        Parameters
        ----------
        agn_log_lbol : float
            ``log10(L_bol / L_sun)``. Default 10.0.
        agn_cos_inc : float
            :math:`\cos i`, the model's one inclination (1 = face-on)
            [dimensionless]. Default ``cos(30 deg)``. The grid's inclination axis is
            stored as ascending ``cos i``.
        agn_torus_frac : float
            Torus reprocessing fraction. Default 0.5.

        Returns
        -------
        ndarray, shape (n_filters,)
            Photometry [erg/s/Hz] per filter.

        Notes
        -----
        **JIT-compatible**: yes.

        **Gradient-safe**: yes, node-exact PCHIP is C¹ differentiable.
        """
        phot = interp_nd_pchip(grid_jax, axes, (agn_cos_inc,))
        # grid_phot/phot already carries the _LSUN_ERG factor from the
        # precompute stage (lnu_grid = _LSUN_ERG * template / integral); do
        # not multiply it in again here (fix round 1: this double-counted
        # _LSUN_ERG, over-scaling precompute photometry by ~3.8e33).
        l_scale = 10.0**agn_log_lbol * agn_torus_frac
        return l_scale * phot

    return skirtor_agnfitter_1p_photometry


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build the preintegrated SKIRTOR_mean_1p grid, auto-collapsing Fixed axes.

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
        Path to ``skirtor_mean1p_torus_grid.h5``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_skirtor_agnfitter_1p_photometry`,
        with the grid axis collapsed if Fixed.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    """
    result = precompute_skirtor_agnfitter_1p_photometry(
        grid_path, filter_waves, filter_trans, redshift=redshift
    )
    preint: PreintegratedGrid = result["_preint"]
    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint, AXIS_PARAMS, parameters, origin="skirtor_agnfitter_1p_precompute"
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
    """Build the runtime SKIRTOR_mean_1p photometry lookup from a preintegrated dict.

    When no axis is collapsed, delegates to
    :func:`build_skirtor_agnfitter_1p_photometry_lookup`. When the axis is
    collapsed (fixed at preintegration time), the returned function takes
    no axis argument at all (a single scalar photometry, scaled by
    ``agn_log_lbol``/``agn_torus_frac``).

    Parameters
    ----------
    preint : dict
        Preintegrated data dict with keys ``"grid_phot"``, ``"axes"``, and
        optionally ``"_collapsed_axes"``.
    free_param_names : tuple of str or None, optional
        Names of the remaining free axes in the collapsed case (unused in
        the default no-collapse case).

    Returns
    -------
    callable
        JIT-compiled photometry lookup returning torus L_ν [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, the returned function is fully JAX-native.

    **Gradient-safe**: yes, node-exact PCHIP is C¹-differentiable.
    """
    if not preint.get("_collapsed_axes"):
        return build_skirtor_agnfitter_1p_photometry_lookup(preint)

    grid_phot = preint["grid_phot"]
    axes = preint["axes"]

    @jax.jit
    def skirtor_agnfitter_1p_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """SKIRTOR_mean_1p torus photometry with collapsed (fixed) axis via PCHIP."""
        # grid_phot/phot already carries the _LSUN_ERG factor from the
        # precompute stage (lnu_grid = _LSUN_ERG * template / integral); do
        # not multiply it in again here (fix round 1: this double-counted
        # _LSUN_ERG, over-scaling precompute photometry by ~3.8e33).
        l_scale = 10.0**agn_log_lbol * agn_torus_frac
        phot = interp_collapsed(grid_phot, axes, free_axis_values, kernel="pchip")
        return l_scale * phot

    return skirtor_agnfitter_1p_phot_collapsed
