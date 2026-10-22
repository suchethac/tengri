# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for SKIRTOR_mean_2p AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the two-parameter (opening-angle, inclination) SKIRTOR-AGNfitter torus
reduction (Stalevski et al. 2016), as packaged by AGNfitter-rX.

Auto-collapses any axis whose corresponding parameter is
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
    AXIS_TRANSFORMS,
    accepts_retired_incl_skirtor,
    free_incl_position,
    skirtor_incl_from_cos_inc,
)
from tengri.components.agn._template_power_table import (
    lnu_per_lsun_from_power_table,
    power_weighted_template_photometry,
)
from tengri.components.agn.skirtor_agnfitter_2p import _load_skirtor_agnfitter_2p_arrays
from tengri.forward.precompute.templates import collapse_fixed_axes
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_pchip,
)

# SKIRTOR_mean_2p grid parametrized by two independent axes: oa, incl.
# The inclination axis is tabulated in degrees from the polar axis, as the runtime
# kernel reads it, and governed by the model's one inclination agn_cos_inc: lookups
# and the Fixed-axis collapse read it at i = arccos(agn_cos_inc) (AXIS_TRANSFORMS).
AXIS_PARAMS: tuple[str, ...] = ("agn_oa_skirtor", "agn_cos_inc")


def precompute_skirtor_agnfitter_2p_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate SKIRTOR_mean_2p torus templates through filter curves.

    For each (oa, incl) grid point, compute the filter-integrated
    photometry. Returns a dict with ``grid_phot`` and ``axes``.

    ``build_skirtor_agnfitter_2p_photometry_lookup`` reads the table as L_ν
    [erg/s/Hz] per L_sun of torus power, normalized as the runtime
    ``skirtor_agnfitter_2p.py`` does.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean2p_torus_grid.h5``.
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0–1).
    redshift : float, optional
        Source redshift. Default 0.0.

    Returns
    -------
    dict
        ``grid_phot`` : ndarray, shape (n_oa, n_incl, n_filters + 1)
            Power-weighted table: raw band L_ν [erg/s/Hz per unit template power]
            and, last, the raw bolometric power; see Notes.
        ``axes`` : tuple of 2 grid arrays (jnp.ndarray)
            Grid axes (oa, incl).
        ``_preint`` : PreintegratedGrid
            Internal preintegration data structure.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

    **Normalization**: the runtime kernel normalizes the *interpolated* template by
    its own bolometric integral, so off-node it weights each node by its raw power.
    The table therefore holds the raw band photometry and, as a last channel, the raw
    power (``power_weighted_template_photometry`` in
    ``_template_power_table``);
    the lookup interpolates both and divides. At a node this is the node-normalized
    photometry exactly.
    """

    raw = _load_skirtor_agnfitter_2p_arrays(grid_path)
    grid = np.asarray(raw["template"], dtype=np.float64)  # (n_oa, n_incl, n_wave)
    wave_grid = np.asarray(raw["wavelength"], dtype=np.float64)
    oa_axis = np.asarray(raw["oa_axis"], dtype=np.float64)
    incl_axis = np.asarray(raw["incl_axis"], dtype=np.float64)

    # The runtime normalizes the interpolated template, so its off-node value is
    # weighted by each node's raw power: tabulate the raw band photometry and the
    # raw power and divide after interpolating (#3093).
    preint = power_weighted_template_photometry(
        grid,
        wave_grid,
        filter_waves,
        filter_trans,
        redshift,
        axes=(oa_axis, incl_axis),
    )

    axes_jax = (jnp.asarray(oa_axis), jnp.asarray(incl_axis))
    return {
        "grid_phot": preint.phot,
        "axes": axes_jax,
        "_preint": preint,
    }


def build_skirtor_agnfitter_2p_photometry_lookup(precomp: dict):
    """Build a JIT-compiled SKIRTOR_mean_2p torus photometry function.

    Uses node-exact monotone-cubic (PCHIP) interpolation for C¹ gradients.

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_skirtor_agnfitter_2p_photometry`.

    Returns
    -------
    callable
        ``fn(agn_log_lbol, agn_oa_skirtor, agn_cos_inc, agn_torus_frac,
        **_) -> photometry [erg/s/Hz]``.

    Notes
    -----
    **JIT-compatible**: yes, pure JAX with no data I/O.
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]

    grid_jax = jnp.asarray(grid_phot)

    @accepts_retired_incl_skirtor
    def skirtor_agnfitter_2p_photometry(
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_oa_skirtor: float = 40.0,
        agn_cos_inc: float = DEFAULT_AGN_COS_INC,
        agn_torus_frac: float = 0.5,
        **_kwargs,
    ) -> jnp.ndarray:
        r"""SKIRTOR_mean_2p torus photometry at a given (oa, incl).

        Parameters
        ----------
        agn_log_lbol : float
            ``log10(L_bol / L_sun)``. Default 11.0.
        agn_oa_skirtor : float
            Half-opening angle [deg]. Default 40.0.
        agn_cos_inc : float
            :math:`\cos i`, the model's one inclination (1 = face-on)
            [dimensionless]. Default ``cos(30 deg)``. The grid's inclination axis
            is in degrees from the polar axis and is read at
            :math:`i = \arccos(\cos i)`, as the runtime kernel reads it.
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
        phot = interp_nd_pchip(
            grid_jax, axes, (agn_oa_skirtor, skirtor_incl_from_cos_inc(agn_cos_inc))
        )
        # The table's band channels already carry L_sun: the ratio is L_nu per L_sun.
        l_scale = 10.0**agn_log_lbol * agn_torus_frac
        return l_scale * lnu_per_lsun_from_power_table(phot)

    return skirtor_agnfitter_2p_photometry


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build the preintegrated SKIRTOR_mean_2p grid, auto-collapsing Fixed axes.

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
        Path to ``skirtor_mean2p_torus_grid.h5``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_skirtor_agnfitter_2p_photometry`,
        with grid axes collapsed for any Fixed :data:`AXIS_PARAMS` entry.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    """
    result = precompute_skirtor_agnfitter_2p_photometry(
        grid_path, filter_waves, filter_trans, redshift=redshift
    )
    preint: PreintegratedGrid = result["_preint"]
    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint,
        AXIS_PARAMS,
        parameters,
        axis_transforms=AXIS_TRANSFORMS,
        origin="skirtor_agnfitter_2p_precompute",
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
    """Build the runtime SKIRTOR_mean_2p photometry lookup from a preintegrated dict.

    When no axes are collapsed, delegates to
    :func:`build_skirtor_agnfitter_2p_photometry_lookup`. When some axes are
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
    **JIT-compatible**: yes, the returned function is fully JAX-native.

    **Gradient-safe**: yes, node-exact PCHIP is C¹-differentiable.
    """
    if not preint.get("_collapsed_axes"):
        return build_skirtor_agnfitter_2p_photometry_lookup(preint)

    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    incl_at = free_incl_position(1, preint["_collapsed_axes"])

    @jax.jit
    def skirtor_agnfitter_2p_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """SKIRTOR_mean_2p torus photometry with collapsed (fixed) axes via PCHIP."""
        if incl_at is not None:
            free_axis_values = list(free_axis_values)  # cos i -> library degrees
            free_axis_values[incl_at] = skirtor_incl_from_cos_inc(free_axis_values[incl_at])
            free_axis_values = tuple(free_axis_values)
        # The table's band channels already carry L_sun: the ratio is L_nu per L_sun.
        l_scale = 10.0**agn_log_lbol * agn_torus_frac
        phot = interp_collapsed(grid_phot, axes, free_axis_values, kernel="pchip")
        return l_scale * lnu_per_lsun_from_power_table(phot)

    return skirtor_agnfitter_2p_phot_collapsed
