# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for analytic radio components.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for the
three radio components in :mod:`tengri.components.radio.radio`:

1. **radio_synchrotron**: star-forming synchrotron via the Bell+2003 fixed-q
   far-IR/radio correlation.  One free axis: ``radio_alpha_sf`` (spectral index).
2. **radio_freefree**: thermal bremsstrahlung from HII regions, Murphy+2011
   calibration.  One free axis: ``radio_alpha_ff``.
3. **radio_agn_jet**: radio-loudness scaled AGN jet power law.  One free
   axis: ``radio_alpha_agn``.

Each component has a multiplicative scalar runtime input (``L_ir`` for SF
synchrotron and free-free, ``L_agn_bol`` for AGN jet) plus its spectral-index
axis.  Auto-collapses Fixed axes via ``slice_fixed_axes``.

.. note::

   For the present pure power-law radio models the spectrum is closed-form,
   so the runtime path (``radio_sfr_bell2003``, ``radio_freefree``,
   ``radio_agn``) is already cheap. These precompute adapters exist to
   establish the ``(scale, *axes) -> photometry`` contract that tabulated
   future models (e.g. self-absorbed synchrotron + free-free + jet geometry
   from radiative transfer) will need; benchmark before assuming a speedup
   over the analytic runtime path.  See
   ``bench/scripts/benchmark_precompute_analytic.py``.

References
----------
.. [1] E. F. Bell, "Estimating star formation rates from infrared and radio
   luminosities: The origin of the radio-infrared correlation," ApJ, 586, 794
   (2003). https://doi.org/10.1086/367829
.. [2] E. J. Murphy et al., "Calibrating extinction-free star formation rate
   diagnostics with 33 GHz free-free emission in NGC 6946," ApJ, 737, 67
   (2011). https://doi.org/10.1088/0004-637X/737/2/67
.. [3] P. F. Hopkins et al., "An observational determination of the bolometric
   quasar luminosity function," ApJ, 654, 731 (2007).
   https://doi.org/10.1086/509629
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from tengri.components.radio.radio import (
    radio_agn as _radio_agn,
    radio_freefree as _radio_freefree,
    radio_sfr_bell2003 as _radio_synchrotron,
)
from tengri.forward.precompute import reach_axes
from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import PreintegratedGrid
from tengri.utils.host_array import device_table, host_array

# Each radio sub-model precomputes along one parameter axis.
AXIS_PARAMS_SYNCHROTRON = ("radio_alpha_sf",)
AXIS_PARAMS_FREEFREE = ("radio_alpha_ff",)
AXIS_PARAMS_AGN_JET = ("radio_alpha_agn",)

# Protocol-required dict form for multi-model module.
AXIS_PARAMS: dict[str, tuple[str, ...]] = {
    "radio_synchrotron": AXIS_PARAMS_SYNCHROTRON,
    "radio_freefree": AXIS_PARAMS_FREEFREE,
    "radio_agn_jet": AXIS_PARAMS_AGN_JET,
}

# Reference luminosity used to build the grid; runtime ``scale`` factor
# rescales linearly (all three models are L-linear in their respective
# luminosity input).
_L_REF = 1.0e44  # erg/s: convenient unit-luminosity reference

# Standard rest-frame wavelength grid covering radio (1 mm to 1 km) and
# extending blueward enough that the suppression mask in radio.py
# (lambda > _RADIO_WAVE_MIN_AA) is not pathological.
_WAVE_REST = host_array(np.logspace(7, 13, 1024, dtype=np.float64))  # 0.1 um to 100 km


def _build_grid_synchrotron(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    alpha_grid: np.ndarray,
) -> PreintegratedGrid:
    """Preintegrate Bell+2003 synchrotron over a 1D grid of alpha_sf."""
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
    templates = np.empty((alpha_grid.size, _WAVE_REST.size), dtype=np.float64)
    for i, a in enumerate(alpha_grid):
        templates[i] = np.asarray(
            _radio_synchrotron(device_table(_WAVE_REST), L_ir=_L_REF, alpha_sf=float(a))
        )
    return precompute_template_photometry(
        templates=templates,
        wave_rest=np.asarray(_WAVE_REST),
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(alpha_grid,),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,
        units="lnu",
    )


def _build_grid_freefree(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    alpha_grid: np.ndarray,
) -> PreintegratedGrid:
    """Preintegrate Murphy+2011 free-free over a 1D grid of alpha_ff."""
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
    templates = np.empty((alpha_grid.size, _WAVE_REST.size), dtype=np.float64)
    for i, a in enumerate(alpha_grid):
        templates[i] = np.asarray(
            _radio_freefree(device_table(_WAVE_REST), L_ir=_L_REF, alpha_ff=float(a))
        )
    return precompute_template_photometry(
        templates=templates,
        wave_rest=np.asarray(_WAVE_REST),
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(alpha_grid,),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,
        units="lnu",
    )


def _build_grid_agn_jet(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    alpha_grid: np.ndarray,
) -> PreintegratedGrid:
    """Preintegrate radio_agn over a 1D grid of alpha_agn (radio_loudness=0)."""
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
    templates = np.empty((alpha_grid.size, _WAVE_REST.size), dtype=np.float64)
    for i, a in enumerate(alpha_grid):
        templates[i] = np.asarray(
            _radio_agn(
                device_table(_WAVE_REST),
                L_agn_bol=_L_REF,
                radio_loudness=0.0,
                alpha_agn=float(a),
            )
        )
    return precompute_template_photometry(
        templates=templates,
        wave_rest=np.asarray(_WAVE_REST),
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(alpha_grid,),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,
        units="lnu",
    )


_BUILDERS = {
    "radio_synchrotron": _build_grid_synchrotron,
    "radio_freefree": _build_grid_freefree,
    "radio_agn_jet": _build_grid_agn_jet,
}


def _axis(
    param_name: str,
    supplied: Any,
    literal: np.ndarray,
    parameters: Any,
) -> np.ndarray:
    """Node axis of ``param_name``: the supplied one checked against the reach, or the hull.

    The default is ``literal`` widened to the parameter's reach (``reach_axes.hull_axis``).
    """
    literal = np.asarray(literal, dtype=np.float64)
    support = reach_axes.active_support(
        param_name, parameters, (float(literal.min()), float(literal.max()))
    )
    if supplied is None:
        return reach_axes.hull_axis(literal, support)
    axis = np.asarray(supplied, dtype=np.float64)
    reach_axes.check_user_axis(param_name, axis, support)
    return axis


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    model: str = "radio_synchrotron",
    alpha_grid: np.ndarray | None = None,
) -> dict:
    """Build preintegrated radio grid; auto-collapse Fixed-parameter axes.

    Multi-model entry point.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter observed-frame wavelength arrays [Angstrom].
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift. [dimensionless]
    parameters : Parameters or None
        Parameters spec; used to detect Fixed axes.
    model : str, keyword-only
        One of ``"radio_synchrotron"``, ``"radio_freefree"``, ``"radio_agn_jet"``.
    alpha_grid : ndarray, optional
        Custom spectral-index grid. Defaults: synchrotron [0.5, 1.0],
        free-free [-0.2, 0.0], AGN [0.4, 1.2].

    Returns
    -------
    dict
        Keys: ``"grid_phot"`` (band fluxes), ``"axes"`` (free axes after
        auto-collapse), ``"_preint"`` (PreintegratedGrid).

    Notes
    -----
    **JIT-compatible**: no, build-time NumPy.
    """
    if model not in _BUILDERS:
        raise ValueError(f"Unknown radio model: {model!r}. Expected one of {sorted(_BUILDERS)}.")

    literal_grids = {
        "radio_synchrotron": np.linspace(0.5, 1.0, 8, dtype=np.float64),
        "radio_freefree": np.linspace(-0.2, 0.0, 6, dtype=np.float64),
        "radio_agn_jet": np.linspace(0.4, 1.2, 8, dtype=np.float64),
    }
    alpha_grid = _axis(AXIS_PARAMS[model][0], alpha_grid, literal_grids[model], parameters)

    preint = _BUILDERS[model](filter_waves, filter_trans, redshift, alpha_grid)
    result = {
        "grid_phot": preint.phot,
        "axes": (jnp.asarray(alpha_grid),),
        "_preint": preint,
    }

    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint, AXIS_PARAMS[model], parameters, origin=f"radio_precompute[{model}]"
    )
    if not fixed:
        return result

    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def build_lookup(preint: dict, *, model: str = "radio_synchrotron"):
    """Build the runtime radio photometry lookup.

    The returned closure has signature ``(scale, *grid_params) -> photometry``
    where ``scale = L_ir / L_REF`` (synchrotron, free-free) or
    ``L_agn_bol / L_REF`` (AGN jet) and ``L_REF = 1e44 erg/s``.

    Parameters
    ----------
    preint : dict
        Output of :func:`precompute`.
    model : str, keyword-only
        Selects the spectral-axis name used in the lookup (cosmetic; the
        actual interpolation is identical across models).

    Returns
    -------
    callable
        JIT-compilable lookup ``(scale, *axes) -> ndarray, shape (n_filters,)``.
    """
    if model not in AXIS_PARAMS:
        raise ValueError(f"Unknown radio model: {model!r}")
    return build_template_photometry_lookup(preint["_preint"])


#: Shape parameters each radio term's table depends on, per registry key. A free one outside
#: the table's own axis declines the table: its shape would move under a fixed table.
TERM_SHAPE_PARAMS: dict[str, tuple[str, ...]] = {
    "radio_synchrotron": ("radio_alpha_sf",),
    "radio_freefree": ("radio_alpha_ff", "radio_T_e"),
    "radio_agn_jet": ("radio_alpha_agn", "radio_log_nu_cut"),
}


def _term_shape_fn(model: str, fixed: Mapping[str, float], wave: Any):
    """Unit-amplitude shape of one term as a function of its axis value(s).

    The amplitude-carrying inputs are set to 1: the table is normalized at ``lam_ref``, so
    any scale cancels. Shape parameters outside the axis are read from ``fixed``.
    """
    if model == "radio_synchrotron":
        return lambda values: _radio_synchrotron(wave, L_ir=1.0, alpha_sf=float(values[0]))
    if model == "radio_freefree":
        return lambda values: _radio_freefree(
            wave, 1.0, T_e=float(fixed["radio_T_e"]), alpha_ff=float(values[0])
        )
    return lambda values: _radio_agn(
        wave,
        1.0,
        radio_loudness=0.0,
        alpha_agn=float(values[0]),
        log_nu_cut=float(fixed["radio_log_nu_cut"]),
    )


def term_shape_table(
    *,
    wave: Any,
    filter_waves_padded: Any,
    filter_trans_padded: Any,
    redshift: float,
    parameters: Any,
    model: str,
) -> dict | None:
    """Shape-only band table of one radio term, built through the shared normalizer.

    Parameters
    ----------
    wave : ndarray, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom] the term is evaluated on at predict.
    filter_waves_padded, filter_trans_padded : ndarray, shape (n_filters, max_len)
        Padded observed-frame filter curves.
    redshift : float
        The single ``Fixed`` redshift [dimensionless].
    parameters : Parameters
        Model spec; supplies the free set and the fixed values of the non-axis shape parameters.
    model : str
        Registry key: ``"radio_synchrotron"``, ``"radio_freefree"`` or ``"radio_agn_jet"``.

    Returns
    -------
    dict or None
        :func:`tengri.components._term_shape_table.build_term_shape_table` output, or ``None``
        when a shape parameter other than the axis is free (the table would be wrong).

    Notes
    -----
    **JIT-compatible**: no, build-time NumPy and JAX integrals.
    """
    from tengri.components._term_shape_table import build_term_shape_table

    axis = AXIS_PARAMS[model][0]
    free = set(parameters.free_params)
    if free & (set(TERM_SHAPE_PARAMS[model]) - {axis}):
        return None
    fixed = parameters.get_fixed_values()
    literal = {
        "radio_synchrotron": np.linspace(0.5, 1.0, 8, dtype=np.float64),
        "radio_freefree": np.linspace(-0.2, 0.0, 6, dtype=np.float64),
        "radio_agn_jet": np.linspace(0.4, 1.2, 8, dtype=np.float64),
    }[model]
    nodes = _axis(axis, None, literal, parameters)
    return build_term_shape_table(
        _term_shape_fn(model, fixed, jnp.asarray(wave)),
        {axis: nodes},
        wave,
        filter_waves_padded,
        filter_trans_padded,
        redshift,
    )
