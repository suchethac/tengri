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
# the same axes. Auto-collapse is currently done at backend init — see
# CloudyGridBackend._preintegrate_photometry in cloudy_grid.py.
AXIS_PARAMS: tuple[str, ...] = ("met_logzsol", "log_age", "neb_logU")


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: object = None,
    **kwargs: object,
) -> object:
    """CLOUDY precompute is performed inside CloudyGridBackend.__init__.

    This function is a Protocol marker — the actual preintegration runs when
    the backend is constructed via the normal config path.  Returns None;
    SEDModel uses this registration only to introspect AXIS_PARAMS.

    Parameters
    ----------
    filter_waves, filter_trans : list
        Filter curves (observed frame).
    redshift : float
        Source redshift.
    parameters : Parameters | None
        Parameter spec (unused — CLOUDY backend handles auto-collapse
        internally).
    **kwargs
        Ignored; accepted for Protocol consistency.

    Returns
    -------
    None
        CLOUDY preintegration is deferred to CloudyGridBackend init.
    """
    return None


def build_lookup(preint: object, **kwargs: object) -> object:
    """CLOUDY runtime lookup is internal to CloudyGridBackend; no Protocol-level
    lookup function is exposed here.

    Parameters
    ----------
    preint : Any
        Unused (CLOUDY preintegration is handled inside the backend).
    **kwargs
        Ignored; accepted for Protocol consistency.

    Returns
    -------
    None
        CLOUDY runtime lookup is performed directly in the CloudyGridBackend.
    """
    return None
