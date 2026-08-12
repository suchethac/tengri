# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for QSOgen quasar SED model.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the QSOgen empirical quasar SED (Temple, Hewett & Banerji 2021).

QSOgen is a flexible broken-power-law + hot dust model with multiple
physical parameters (power-law slopes, dust temperature, emission-line
strength, dust reddening, etc.).

**Axis selection for precomputation**: In realistic SED fits, the most
frequently varied parameters are ``agn_plslp1`` (UV/optical slope) and
``agn_ebv`` (dust reddening). These are precomputed on a 2D grid.
Other parameters (``agn_plslp2``, ``agn_plbrk``, ``agn_tbb``, ``agn_bbnorm``,
``agn_emline_scale``, ``agn_bcnorm``) are kept at their defaults during
precomputation. If a real fit needs to vary other dimensions, the runtime
path will be used (falling back to per-call wavelength evaluation).

Auto-collapses any axis whose corresponding parameter is ``Fixed`` in the
user's ``Parameters`` — e.g., a user who pins ``agn_plslp1`` gets a 1D
reddening grid instead of 2D.

References
----------
.. [1] G. Temple, P. J. Hewett, and M. Banerji, "QSOgen: a model of the
   UV-to-submillimeter spectral energy distributions of quasars," MNRAS, 508,
   737–754 (2021). https://doi.org/10.1093/mnras/stab2586
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components._collapsed_lookup import interp_collapsed
from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL, DEFAULT_AGN_LUM_RATIO
from tengri.components.agn.qsogen import compute_qsogen_sed as _compute_qsogen_sed
from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    slice_fixed_axes,
)
from tengri.utils.interpolation import edges_for_grid

# ── Axis definitions ──────────────────────────────────────

# Minimal practical axis set for QSOgen: UV slope and dust reddening.
# These are the most frequently varied parameters in real SED fits.
AXIS_PARAMS = ("agn_plslp1", "agn_ebv")


def _build_grid_qsogen(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    plslp1_grid: np.ndarray,
    ebv_grid: np.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    # Fixed parameters (not varied in precomputation):
    agn_plslp2: float = 0.593,
    agn_plbrk: float = 3880.0,
    agn_tbb: float = 1240.0,
    agn_bbnorm: float = 3.96,
    agn_emline_scale: float = 1.0,
    agn_bcnorm: float = 0.0,
) -> PreintegratedGrid:
    """Preintegrate QSOgen over a 2D grid of (plslp1, E(B-V)).

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter wavelength arrays [Angstrom].
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift.
    plslp1_grid : ndarray, shape (n_plslp1,)
        Grid of UV/optical power-law slopes (dimensionless).
    ebv_grid : ndarray, shape (n_ebv,)
        Grid of dust reddening E(B-V) values [mag].
    agn_log_lbol : float
        Reference bolometric luminosity [log10(L_sun)].
    agn_lum_ratio : float
        Fraction of luminosity emitted by QSOgen component.
    agn_plslp2 : float
        Red power-law slope (kept fixed).
    agn_plbrk : float
        Break wavelength [Angstrom] (kept fixed).
    agn_tbb : float
        Hot dust temperature [K] (kept fixed).
    agn_bbnorm : float
        Hot dust normalization (kept fixed).
    agn_emline_scale : float
        Emission-line strength multiplier (kept fixed).
    agn_bcnorm : float
        Balmer continuum normalization (kept fixed).

    Returns
    -------
    PreintegratedGrid
        Preintegrated photometry with shape (n_plslp1, n_ebv, n_filters).
    """
    plslp1_grid = np.asarray(plslp1_grid, dtype=np.float64)
    ebv_grid = np.asarray(ebv_grid, dtype=np.float64)
    n_plslp1 = len(plslp1_grid)
    n_ebv = len(ebv_grid)

    # Standard wavelength grid for QSOgen (912–100000 Angstrom)
    wave_rest = np.logspace(2.96, 5.0, 2000, dtype=np.float64)

    phot_grid = []
    for plslp1 in plslp1_grid:
        for ebv in ebv_grid:
            # Call QSOgen with the given parameters
            l_nu = np.asarray(
                _compute_qsogen_sed(
                    jnp.asarray(wave_rest),
                    agn_plslp1=float(plslp1),
                    agn_plslp2=float(agn_plslp2),
                    agn_plbrk=float(agn_plbrk),
                    agn_tbb=float(agn_tbb),
                    agn_bbnorm=float(agn_bbnorm),
                    agn_emline_scale=float(agn_emline_scale),
                    agn_ebv=float(ebv),
                    agn_log_lbol=agn_log_lbol,
                    agn_lum_ratio=agn_lum_ratio,
                    agn_bcnorm=float(agn_bcnorm),
                )
            )
            phot_grid.append(l_nu)

    templates = np.array(phot_grid, dtype=np.float64).reshape(n_plslp1, n_ebv, len(wave_rest))

    # Preintegrate through filters using template helper
    return precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(plslp1_grid, ebv_grid),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,  # already normalized per L_sun from qsogen
        units="lnu",
    )


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    plslp1_grid: np.ndarray | None = None,
    ebv_grid: np.ndarray | None = None,
) -> dict:
    """Build preintegrated QSOgen grid, auto-collapsing Fixed-parameter axes.

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
    plslp1_grid : ndarray, optional
        Grid for agn_plslp1. If None, uses [-0.8, -0.35, 0.1] (covers
        typical range around default -0.349).
    ebv_grid : ndarray, optional
        Grid for agn_ebv. If None, uses [0.0, 0.05, 0.1, 0.2, 0.3]
        (covers typical reddening from 0 to SMC-like).

    Returns
    -------
    dict
        Keys: "grid_phot" (photometry array), "axes" (free axes), "_preint"
        (PreintegratedGrid), optionally "_collapsed_axes" (if any axes fixed).

    References
    ----------
    .. [1] G. Temple, P. J. Hewett, and M. Banerji, "QSOgen: a model of the
       UV-to-submillimeter spectral energy distributions of quasars," MNRAS, 508,
       737–754 (2021). https://doi.org/10.1093/mnras/stab2586

    Notes
    -----
    **JIT-compatible**: no — this is a build-time function using NumPy.

    **Axis selection rationale**: agn_plslp1 (UV/optical slope) and agn_ebv
    (dust reddening) are the most frequently constrained parameters in real
    AGN SED fits. Other QSOgen parameters are kept at their defaults during
    precomputation to avoid exploding the grid dimensionality.
    """
    if plslp1_grid is None:
        plslp1_grid = np.array([-0.8, -0.35, 0.1], dtype=np.float64)
    if ebv_grid is None:
        ebv_grid = np.array([0.0, 0.05, 0.1, 0.2, 0.3], dtype=np.float64)

    result = {
        "grid_phot": _build_grid_qsogen(
            filter_waves, filter_trans, redshift, plslp1_grid, ebv_grid
        ).phot,
        "axes": (jnp.asarray(plslp1_grid), jnp.asarray(ebv_grid)),
        "_preint": _build_grid_qsogen(filter_waves, filter_trans, redshift, plslp1_grid, ebv_grid),
    }

    # Auto-collapse any Fixed axes — including axes for params the user
    # never declared (treat as Fixed at the qsogen default). This avoids a
    # 0-d-vs-2-d shape mismatch in the runtime triweight interp when, e.g.,
    # the user only declares ``agn_log_lbol`` and leaves ``agn_plslp1`` /
    # ``agn_ebv`` untouched: precompute used to keep both axes free, but
    # the kernel only passes ``agn_log_lbol`` → ``IndexError: tuple index
    # out of range`` (BUG-NSS-03 regression test).
    if parameters is None:
        return result

    from tengri.components.agn.qsogen import _DEFAULT_EBV, _DEFAULT_PLSLP1

    _AXIS_DEFAULTS = {
        "agn_plslp1": _DEFAULT_PLSLP1,
        "agn_ebv": _DEFAULT_EBV,
    }

    preint: PreintegratedGrid = result["_preint"]
    fixed_values = parameters.get_fixed_values()
    free_param_names = set(parameters.free_params)
    fixed: dict[int, float] = {}
    for i, pname in enumerate(AXIS_PARAMS):
        if pname in fixed_values:
            fixed[i] = float(fixed_values[pname])
        elif pname not in free_param_names:
            # Param not in the spec at all — collapse axis at qsogen default.
            fixed[i] = float(_AXIS_DEFAULTS[pname])

    if not fixed:
        return result

    collapsed = slice_fixed_axes(preint, fixed)
    remaining_axes = tuple(ax for i, ax in enumerate(result["axes"]) if i not in fixed)
    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def build_lookup(preint: dict, *, free_param_names: tuple[str, ...] | None = None):
    """Build the runtime QSOgen photometry lookup from a preintegrated dict.

    Delegates to the template helper for triweight interpolation.

    Parameters
    ----------
    preint : dict
        Preintegrated data dict with keys "grid_phot", "axes", optionally
        "_collapsed_axes".
    free_param_names : tuple of str, optional
        Names of remaining free axes in the collapsed case.
        Not used in the default (no-collapse) case.

    Returns
    -------
    callable
        JIT-compiled photometry lookup function with signature::

            fn(agn_log_lbol, *free_axis_values) -> ndarray, shape (n_filters,)

        Returns QSOgen L_ν [erg/s/Hz]. Caller applies flux scaling.

    References
    ----------
    .. [1] G. Temple, P. J. Hewett, and M. Banerji, "QSOgen: a model of the
       UV-to-submillimeter spectral energy distributions of quasars," MNRAS, 508,
       737–754 (2021). https://doi.org/10.1093/mnras/stab2586

    Notes
    -----
    **JIT-compatible**: yes — the returned function uses ``jnp`` and triweight
    interpolation.

    **Gradient-safe**: yes — triweight kernel is fully differentiable.
    """
    if not preint.get("_collapsed_axes"):
        # No axes collapsed: use template helper directly
        return build_template_photometry_lookup(preint["_preint"])

    # Collapsed case: return a wrapped lookup that takes remaining free params
    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    if axes:
        edges = tuple(edges_for_grid(ax) for ax in axes)
    else:
        edges = ()

    @jax.jit
    def qsogen_phot_collapsed(agn_log_lbol, *free_axis_values):
        """Compute QSOgen photometry with some axes collapsed (fixed).

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        l_bol_lsun = 10.0**agn_log_lbol
        normed = interp_collapsed(
            grid_phot, axes, free_axis_values, kernel="triweight", edges=edges
        )
        return l_bol_lsun * normed

    return qsogen_phot_collapsed
