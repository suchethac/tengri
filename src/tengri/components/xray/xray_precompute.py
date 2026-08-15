# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for analytic X-ray components.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
three X-ray emitters in :mod:`tengri.components.xray.xray`:

1. **xray_xrb** — high- and low-mass X-ray binaries (HMXB+LMXB), Grimm+2003
   and Gilfanov+2004 calibrations.  Free axes: ``xray_gamma_hmxb``,
   ``xray_gamma_lmxb``.
2. **xray_corona** — AGN corona via the α_OX relation (Lusso+2010 / Just+2007).
   Free axes: ``xray_gamma_agn``, ``xray_delta_alpha_ox``.
3. **xray_corona_lopez24** — AGN corona via the α_IRX relation (López+2024,
   Yang+2022 anisotropy).  Free axes: ``xray_gamma_agn``, ``xray_alpha_irx``.

Each model is L-linear in its scalar luminosity input (SFR×stellar_mass for
XRBs; L_agn_bol for corona; l_12um for the López24 corona).  Auto-collapses
Fixed axes via ``slice_fixed_axes``.

.. note::

   The current X-ray models are closed-form power-law-with-cutoff spectra,
   so the runtime path is already cheap. These precompute adapters exist to
   establish the ``(scale, *axes) -> photometry`` contract that tabulated
   future models (e.g. Comptonization tables, reflection grids, ionized
   absorber transmission cubes) will need; benchmark before assuming a
   speedup over the analytic runtime path.  See
   ``scripts/benchmark_precompute_analytic.py``.

References
----------
.. [1] G. Grimm, M. Gilfanov, and R. Sunyaev, "High-mass X-ray binaries as a
   star formation rate indicator in distant galaxies," MNRAS, 339, 793 (2003).
   https://doi.org/10.1046/j.1365-8711.2003.06224.x
.. [2] M. Gilfanov, "Low-mass X-ray binaries as a stellar mass indicator for
   the host galaxy," MNRAS, 349, 146 (2004).
   https://doi.org/10.1111/j.1365-2966.2004.07473.x
.. [3] E. Lusso et al., "The X-ray to optical-UV luminosity ratio of X-ray
   selected type 1 AGN in XMM-COSMOS," A&A, 512, A34 (2010).
   https://doi.org/10.1051/0004-6361/200913298
.. [4] G. López et al., "An α-IRX relation for AGN," 2024 (paper credited in
   xray.py source docstring).
.. [5] G. Yang et al., "Inclination-dependent anisotropy of AGN X-ray
   emission," ApJ, 927, 192 (2022).
   https://doi.org/10.3847/1538-4357/ac4971
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from tengri.components.xray.xray import (
    xray_agn_corona as _xray_corona,
    xray_agn_corona_lopez24 as _xray_corona_lopez24,
    xray_xrb as _xray_xrb,
)
from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import PreintegratedGrid

AXIS_PARAMS_XRB = ("xray_gamma_hmxb", "xray_gamma_lmxb")
AXIS_PARAMS_CORONA = ("xray_gamma_agn", "xray_delta_alpha_ox")
AXIS_PARAMS_CORONA_LOPEZ24 = ("xray_gamma_agn", "xray_alpha_irx")

AXIS_PARAMS: dict[str, tuple[str, ...]] = {
    "xray_xrb": AXIS_PARAMS_XRB,
    "xray_corona": AXIS_PARAMS_CORONA,
    "xray_corona_lopez24": AXIS_PARAMS_CORONA_LOPEZ24,
}

# Standard rest-frame wavelength grid covering 0.01–100 keV plus an optical tail.
# 0.01 keV ≈ 124 Å; 100 keV ≈ 0.124 Å.
_WAVE_REST = np.logspace(-1.0, 4.0, 1024, dtype=np.float64)  # 0.1 to 1e4 Å

# Reference scales chosen to keep the L-linear runtime rescaling well-conditioned.
_SFR_REF = 1.0  # Msun/yr
_MSTAR_REF = 1.0e10  # Msun
_LBOL_REF = 1.0e44  # erg/s
_L12_REF = 1.0e30  # erg/s/Hz


def _build_grid_xrb(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    gamma_h_grid: np.ndarray,
    gamma_l_grid: np.ndarray,
) -> PreintegratedGrid:
    gamma_h_grid = np.asarray(gamma_h_grid, dtype=np.float64)
    gamma_l_grid = np.asarray(gamma_l_grid, dtype=np.float64)
    templates = np.empty((gamma_h_grid.size, gamma_l_grid.size, _WAVE_REST.size), dtype=np.float64)
    for i, gh in enumerate(gamma_h_grid):
        for j, gl in enumerate(gamma_l_grid):
            templates[i, j] = np.asarray(
                _xray_xrb(
                    jnp.asarray(_WAVE_REST),
                    sfr=_SFR_REF,
                    stellar_mass=_MSTAR_REF,
                    gamma_hmxb=float(gh),
                    gamma_lmxb=float(gl),
                )
            )
    return precompute_template_photometry(
        templates=templates,
        wave_rest=_WAVE_REST,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(gamma_h_grid, gamma_l_grid),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,
        units="lnu",
    )


def _build_grid_corona(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    gamma_grid: np.ndarray,
    alpha_ox_grid: np.ndarray,
) -> PreintegratedGrid:
    gamma_grid = np.asarray(gamma_grid, dtype=np.float64)
    alpha_ox_grid = np.asarray(alpha_ox_grid, dtype=np.float64)
    # Convert reference L_bol → L_2500_30deg via Hopkins+2007 BC_2500.
    # The runtime parameter xray_delta_alpha_ox is a delta offset to the Just+2007
    # empirical prior at this L_2500. The grid axis is now indexed by delta
    # values (not absolute), so the grid lookup directly uses xray_delta_alpha_ox
    # as the axis coordinate. Internally, we compute the prior to apply.
    _NU_2500 = 1.199e15  # Hz
    _BC_2500 = 5.15
    L_2500_REF = _LBOL_REF / (_BC_2500 * _NU_2500)  # erg/s/Hz
    alpha_ox_prior = -0.137 * np.log10(L_2500_REF) + 2.638  # Just+2007 at L_2500_REF
    # Interpret the grid axis as delta_alpha_ox (offset to the prior)
    delta_alpha_ox_axis = alpha_ox_grid - alpha_ox_prior
    templates = np.empty((gamma_grid.size, alpha_ox_grid.size, _WAVE_REST.size), dtype=np.float64)
    for i, g in enumerate(gamma_grid):
        for j, daox in enumerate(delta_alpha_ox_axis):
            templates[i, j] = np.asarray(
                _xray_corona(
                    jnp.asarray(_WAVE_REST),
                    l_2500_30deg_erg_hz=L_2500_REF,
                    gamma=float(g),
                    delta_alpha_ox=float(daox),
                    # Templates are anchored at the Yang+2020 30° reference
                    # (anisotropy factor exactly 1). The inclination factor
                    # is a wavelength-independent scalar — consumers fold
                    # xray_anisotropy(1, cos_inc) into the runtime scale
                    # alongside l_2500/L_2500_REF (#980).
                    apply_anisotropy=False,
                )
            )
    return precompute_template_photometry(
        templates=templates,
        wave_rest=_WAVE_REST,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(gamma_grid, delta_alpha_ox_axis),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,
        units="lnu",
    )


def _build_grid_corona_lopez24(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    gamma_grid: np.ndarray,
    alpha_irx_grid: np.ndarray,
) -> PreintegratedGrid:
    gamma_grid = np.asarray(gamma_grid, dtype=np.float64)
    alpha_irx_grid = np.asarray(alpha_irx_grid, dtype=np.float64)
    templates = np.empty((gamma_grid.size, alpha_irx_grid.size, _WAVE_REST.size), dtype=np.float64)
    for i, g in enumerate(gamma_grid):
        for j, ai in enumerate(alpha_irx_grid):
            templates[i, j] = np.asarray(
                _xray_corona_lopez24(
                    jnp.asarray(_WAVE_REST),
                    l_12um_erg_hz=_L12_REF,
                    gamma=float(g),
                    alpha_irx=float(ai),
                    apply_anisotropy=False,  # 30°-anchored, like _build_grid_corona (#980)
                )
            )
    return precompute_template_photometry(
        templates=templates,
        wave_rest=_WAVE_REST,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(gamma_grid, alpha_irx_grid),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,
        units="lnu",
    )


_BUILDERS = {
    "xray_xrb": _build_grid_xrb,
    "xray_corona": _build_grid_corona,
    "xray_corona_lopez24": _build_grid_corona_lopez24,
}

_DEFAULT_GRIDS = {
    "xray_xrb": (
        np.linspace(1.7, 2.3, 5, dtype=np.float64),
        np.linspace(1.4, 1.9, 5, dtype=np.float64),
    ),
    "xray_corona": (
        np.linspace(1.5, 2.3, 6, dtype=np.float64),
        np.linspace(-1.8, -1.0, 6, dtype=np.float64),
    ),
    "xray_corona_lopez24": (
        np.linspace(1.5, 2.3, 6, dtype=np.float64),
        np.linspace(0.0, 0.6, 6, dtype=np.float64),
    ),
}


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    model: str = "xray_corona",
    axis_grids: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Build preintegrated X-ray grid; auto-collapse Fixed-parameter axes.

    Parameters
    ----------
    filter_waves, filter_trans : list[ndarray]
        Per-filter observed-frame wavelengths [Angstrom] and transmission curves.
    redshift : float
        Source redshift. [dimensionless]
    parameters : Parameters or None
        Parameters spec; used to detect Fixed axes via ``is_fixed``/``fixed_value``.
    model : str, keyword-only
        One of ``"xray_xrb"``, ``"xray_corona"``, ``"xray_corona_lopez24"``.
    axis_grids : tuple[ndarray, ndarray], optional
        Custom 2D grid axes; defaults match the documented validity ranges
        of the underlying physics functions.

    Returns
    -------
    dict
        ``"grid_phot"`` (band fluxes), ``"axes"`` (free axes after auto-collapse),
        ``"_preint"`` (PreintegratedGrid).

    Notes
    -----
    **JIT-compatible**: no — build-time NumPy.
    """
    if model not in _BUILDERS:
        raise ValueError(f"Unknown X-ray model: {model!r}. Expected one of {sorted(_BUILDERS)}.")

    if axis_grids is None:
        axis_grids = _DEFAULT_GRIDS[model]

    a0, a1 = axis_grids
    preint = _BUILDERS[model](filter_waves, filter_trans, redshift, a0, a1)
    result = {
        "grid_phot": preint.phot,
        "axes": (jnp.asarray(a0), jnp.asarray(a1)),
        "_preint": preint,
    }

    collapsed, remaining, fixed = collapse_fixed_axes(
        preint, AXIS_PARAMS[model], parameters, origin=f"xray_precompute[{model}]"
    )
    if not fixed:
        return result

    return {
        "grid_phot": collapsed.phot,
        "axes": remaining,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def build_lookup(preint: dict, *, model: str = "xray_corona"):
    """Build the runtime X-ray photometry lookup.

    Returned closure: ``(scale, *grid_params) -> photometry``.
    The runtime ``scale`` is the appropriate luminosity ratio (e.g.
    ``L_agn_bol / L_BOL_REF`` for corona).
    """
    if model not in AXIS_PARAMS:
        raise ValueError(f"Unknown X-ray model: {model!r}")
    return build_template_photometry_lookup(preint["_preint"])
