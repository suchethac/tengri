# SPDX-License-Identifier: BSD-3-Clause
"""Intergalactic medium (IGM) attenuation models."""

from tengri.components.igm.dla import (
    dla_transmission,
    dla_transmission_obs,
)
from tengri.components.igm.igm import (
    IGM_TRANSMISSION_MODELS,
    igm_transmission,
    igm_transmission_madau,
    igm_transmission_patchy,
    resolve_igm_model,
)

__all__ = [
    "IGM_TRANSMISSION_MODELS",
    "dla_transmission",
    "dla_transmission_obs",
    "igm_transmission",
    "igm_transmission_madau",
    "igm_transmission_patchy",
    "resolve_igm_model",
]
