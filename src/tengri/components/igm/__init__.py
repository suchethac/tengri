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
    igm_transmission,
    igm_transmission_madau,
    igm_transmission_patchy,
)

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

__all__ = [
    "IGM_MODELS",
    "IGMRegistryEntry",
    "dla_transmission",
    "dla_transmission_obs",
    "igm_transmission",
    "igm_transmission_madau",
    "igm_transmission_patchy",
    "register_igm_model",
]
