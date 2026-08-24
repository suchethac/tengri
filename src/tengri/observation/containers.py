# SPDX-License-Identifier: BSD-3-Clause
"""User-facing data containers for observations.

This namespace gathers the dataclass-shaped containers a user constructs
to describe their observations: photometry, spectroscopy, line-flux
constraints, spectral-index measurements, noise models, the :class:`Instrument`
bundle, and the top-level :class:`Observation` composite.

The flat ``tengri.observation.X`` and top-level ``tengri.X`` import paths
remain valid, this module is an *additive* sub-namespace that surfaces
the same container objects under a focused, browseable name.

Examples
--------
>>> from tengri.observation.containers import (
...     Instrument,
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
tengri.observation.physics : transformation functions (LSF, aperture, etc.)
tengri.observation.constants : module-level constants (line catalogs, status flags)
"""

from __future__ import annotations

from tengri.observation.data import Data, ValidatedData
from tengri.observation.instrument import Instrument, list_instruments
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_list import LineList
from tengri.observation.line_ratio_data import LineRatioData
from tengri.observation.noise_model import NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectral_indices import SpectralIndexData, SpectralIndexDef
from tengri.observation.spectroscopy import Spectroscopy

__all__ = [
    "Data",
    "Instrument",
    "LineFluxData",
    "LineList",
    "LineRatioData",
    "NoiseModel",
    "Observation",
    "Photometry",
    "SpectralIndexData",
    "SpectralIndexDef",
    "Spectroscopy",
    "ValidatedData",
    "list_instruments",
]
