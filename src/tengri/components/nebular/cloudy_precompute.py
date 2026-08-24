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


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: object = None,
    **kwargs: object,
) -> object:
    """Protocol marker for CLOUDY preintegration (deferred to backend init).

    CLOUDY preintegration is performed inside CloudyGridBackend.__init__
    because the grid shape is determined by the loaded HDF5 file. This function
    serves as a Protocol marker; SEDModel uses it only to introspect AXIS_PARAMS.

    Parameters
    ----------
    filter_waves: list
        Filter wavelength arrays [Angstrom] (observed frame).
    filter_trans: list
        Filter transmission curves (unitless).
    redshift: float
        Source redshift.
    parameters: Parameters, optional
        Parameter spec (unused: CLOUDY backend handles auto-collapse
        internally). Default: None.
    **kwargs
        Additional arguments (ignored for Protocol consistency).

    Returns
    -------
    None
        CLOUDY preintegration is deferred to CloudyGridBackend.__init__.

    Notes
    -----
    **JIT-compatible**: no, returns None (metadata function).

    Auto-collapse for fixed parameters (met_logzsol, log_age, neb_logU) is
    wired inside CloudyGridBackend._preintegrate_photometry().

    """
    return None


def build_lookup(preint: object, **kwargs: object) -> object:
    """Protocol marker for CLOUDY runtime lookup (handled internally).

    CLOUDY runtime lookup is internal to CloudyGridBackend; no Protocol-level
    lookup function is exposed here. This function serves as a Protocol marker.

    Parameters
    ----------
    preint: object
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
