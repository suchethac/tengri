# SPDX-License-Identifier: BSD-3-Clause
"""User-facing data containers for observations.

This namespace gathers the dataclass-shaped containers a user constructs
to describe their observations: photometry, spectroscopy, line-flux
constraints, spectral-index measurements, noise models, and the
top-level :class:`Observation` bundle.

The flat ``tengri.observation.X`` and top-level ``tengri.X`` import paths
remain valid — this module is an *additive* sub-namespace that surfaces
the same container objects under a focused, browseable name.

Examples
--------
>>> from tengri.observation.containers import (
...     LineFluxData,
...     LineList,
...     NoiseModel,
...     Observation,
...     Photometry,
...     SpectralIndexData,
...     SpectralIndexDef,
...     Spectroscopy,
... )

See Also
--------
tengri.observation.physics : transformation functions (calibration, LSF, etc.)
tengri.observation.constants : module-level constants (line catalogs, status flags)
"""

from __future__ import annotations

from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_list import LineList
from tengri.observation.noise_model import NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectral_indices import SpectralIndexData, SpectralIndexDef
from tengri.observation.spectroscopy import Spectroscopy

__all__ = [
    "LineFluxData",
    "LineList",
    "NoiseModel",
    "Observation",
    "Photometry",
    "SpectralIndexData",
    "SpectralIndexDef",
    "Spectroscopy",
]
