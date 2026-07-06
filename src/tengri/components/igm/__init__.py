# SPDX-License-Identifier: BSD-3-Clause
"""Intergalactic medium (IGM) attenuation models."""

from tengri.components.igm._models import (
    IGM_MODELS,
    IGMRegistryEntry,
    register_igm_model,
)
from tengri.components.igm.dla import (
    dla_transmission,
    dla_transmission_obs,
)
from tengri.components.igm.igm import (
    IGM_TRANSMISSION_MODELS,
    igm_absorption,
    igm_transmission,
    igm_transmission_asada25,
    igm_transmission_madau,
    igm_transmission_patchy,
    resolve_igm_model,
)
from tengri.components.igm.meiksin06 import igm_transmission_meiksin06

# Populate the runtime registry. ``_VALID_IGM_TYPES`` in
# ``parameters/groups.py`` derives from ``IGM_MODELS.keys()``. The
# ``'inoue'`` alias for ``'inoue14'`` (#344) maps to the same callable
# so user-facing strings stay backwards-compatible.
register_igm_model(
    "none",
    short_doc="Disable IGM attenuation (e.g. low-z fits)",
)(None)
register_igm_model(
    "inoue14",
    citation="Inoue+2014 (MNRAS 442, 1805)",
    short_doc="Mean LyC + Lyman-series IGM attenuation; default for z >~ 1",
)(igm_transmission)
register_igm_model(
    "inoue",
    citation="Inoue+2014 (MNRAS 442, 1805)",
    short_doc="Alias of 'inoue14' kept for backwards compatibility",
)(igm_transmission)
register_igm_model(
    "madau",
    citation="Madau 1995 (ApJ 441, 18)",
    short_doc="Original Lyman-forest mean transmission",
)(igm_transmission_madau)
register_igm_model(
    "meiksin06",
    citation="Meiksin 2006 (MNRAS 365, 807)",
    short_doc="Smooth Ly-alpha forest continuum + LLS damping (matches CIGALE)",
)(igm_transmission_meiksin06)
register_igm_model(
    "asada25",
    citation="Asada+2025 (ApJL 983, L2)",
    short_doc="Inoue+2014 IGM + Asada+2025 proximate-CGM damping wing (fixes z>7 photo-z bias)",
)(igm_transmission_asada25)

__all__ = [
    "IGM_MODELS",
    "IGM_TRANSMISSION_MODELS",
    "IGMRegistryEntry",
    "dla_transmission",
    "dla_transmission_obs",
    "igm_absorption",
    "igm_transmission",
    "igm_transmission_asada25",
    "igm_transmission_madau",
    "igm_transmission_meiksin06",
    "igm_transmission_patchy",
    "register_igm_model",
    "resolve_igm_model",
]
