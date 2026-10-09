# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for CLOUDY nebular grid.

CLOUDY is unusual: its preintegration happens inside the
:class:`CloudyGridBackend` init path because the grid shape depends on the
loaded HDF5 file.  This module exposes the Protocol surface so the registry
lookup works uniformly; callers should construct the backend via the normal
configuration path, and access its preintegrated state via the backend
attributes.

The auto-collapse for `met_logzsol`, `age`, and `neb_logU` Fixed parameters is
already wired inside `CloudyGridBackend._preintegrate_photometry` (see
cloudy_grid.py:360-434).  This module documents the axis mapping.
"""

from __future__ import annotations

# CLOUDY continuum grid axes: (log_met, log_age, log_U). The line grid uses
# the same axes. Auto-collapse is currently done at backend init: see
# CloudyGridBackend._preintegrate_photometry in cloudy_grid.py.
AXIS_PARAMS: tuple[str, ...] = ("met_logzsol", "log_age", "neb_logU")

# "log_age" is an internal grid-axis label, not a declared user parameter.
# It is intentionally not collapsed by the collapse_fixed_axes logic; the
# CLOUDY backend handles its own axis management. See issue #1827.
INTERNAL_AXES: frozenset[str] = frozenset({"log_age"})


def _span(parameters: object, name: str) -> tuple[float, float] | None:
    """Support of a declared parameter: its prior bounds, or a Fixed value as a point.

    Returns None when the spec does not declare ``name`` (the caller decides
    what an absent parameter means).
    """
    if name in parameters.free_params:
        return tuple(float(b) for b in parameters.get_distribution(name).bounds)
    if name in parameters.fixed_params:
        value = float(parameters.fixed_value(name))
        return (value, value)
    return None


def nebular_axis_ranges(parameters: object) -> tuple[tuple[float, float], tuple[float, float]]:
    """Absolute (log10 Z_gas, log10 U) ranges the spec can reach.

    ``neb_logZ_gas`` is solar-relative and the table takes it absolute. When
    the spec does not declare it, the gas is tied to the stellar metallicity
    (``CloudyGridBackend`` reads ``neb_logZ_gas=None`` as tied), so the
    ``met_logzsol`` support is used.
    """
    from tengri.parameters.translate import LOG10_ZSUN

    gas = _span(parameters, "neb_logZ_gas")
    if gas is None:
        met = _span(parameters, "met_logzsol") or (0.0, 0.0)
        gas = met
    z_abs = (gas[0] + LOG10_ZSUN, gas[1] + LOG10_ZSUN)
    u_span = _span(parameters, "neb_logU")
    if u_span is None:
        # Declared NEB_LOGU_DEFAULT when absent (component.py common kwargs).
        from tengri.components.nebular.nebular_grid_precompute import NEB_LOGU_DEFAULT

        u_span = (float(NEB_LOGU_DEFAULT), float(NEB_LOGU_DEFAULT))
    return z_abs, u_span


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: object = None,
    *,
    backend: object = None,
    wave: object = None,
    line_sigma_kms: float | None = None,
    **kwargs: object,
) -> object:
    """Build the age-resolved CLOUDY band table, or return None (#2324).

    With no ``backend`` this stays the Protocol marker it always was: CLOUDY
    preintegration runs inside :class:`CloudyGridBackend`. With a backend, it
    builds the band table of :mod:`cloudy_band_table` for the fixed filters,
    redshift and line width, and returns it only when the table's (Z_gas, logU)
    axes contain the spec's reachable ranges. Otherwise it returns None, and the
    caller keeps the per-call path.

    Parameters
    ----------
    filter_waves : list
        Filter wavelength arrays [Angstrom] (observed frame).
    filter_trans : list
        Filter transmission curves (unitless).
    redshift : float
        Source redshift (a single Fixed value).
    parameters : Parameters, optional
        Parameter spec; supplies the priors of ``neb_logZ_gas`` and ``neb_logU``.
    backend : CloudyGridBackend, keyword-only, optional
        The nebular backend whose grid, Q_H table and young bins are tabulated.
    wave : array_like, keyword-only, optional
        SED wavelength grid the per-call path projects onto (``state.wave``).
    line_sigma_kms : float, keyword-only, optional
        Fixed line velocity width; required with ``backend``.
    **kwargs
        Ignored, for Protocol consistency.

    Returns
    -------
    CloudyBandTable or None
        None when no backend is given or the axes do not cover the priors.

    Notes
    -----
    **JIT-compatible**: no, build-time only.
    """
    if backend is None:
        return None
    if wave is None or line_sigma_kms is None:
        raise ValueError("cloudy precompute with a backend needs wave= and line_sigma_kms=")
    from tengri.components.nebular.cloudy_band_table import (
        build_cloudy_band_table,
        table_covers,
    )

    table = build_cloudy_band_table(
        backend, wave, filter_waves, filter_trans, float(redshift), float(line_sigma_kms)
    )
    if parameters is not None:
        z_range, u_range = nebular_axis_ranges(parameters)
        if not table_covers(table, z_range, u_range):
            return None
    return table


def build_lookup(preint: object, **kwargs: object) -> object:
    """Protocol marker for CLOUDY runtime lookup (handled internally).

    CLOUDY runtime lookup is internal to CloudyGridBackend; no Protocol-level
    lookup function is exposed here. This function serves as a Protocol marker.

    Parameters
    ----------
    preint : object
        Unused: CLOUDY preintegration is handled inside CloudyGridBackend.
    **kwargs
        Additional arguments (ignored for Protocol consistency).

    Returns
    -------
    None
        CLOUDY runtime lookup is performed directly in CloudyGridBackend.

    Notes
    -----
    **JIT-compatible**: no, returns None (metadata function).

    """
    return None
