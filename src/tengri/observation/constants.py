# SPDX-License-Identifier: BSD-3-Clause
"""Module-level constants for the observation layer.

This namespace gathers the catalogs, status flags, instrument-
resolution registries, and analysis modes used throughout the observation layer:

- Photometry status flags: ``DETECTED``, ``UPPER_LIMIT``, ``LOWER_LIMIT``
- Emission-line fitting modes: ``ELINE_MODES`` (legal modes for spectroscopic
  emission line analysis).
- Default emission-line catalog: ``DEFAULT_LINE_NAMES`` /
  ``DEFAULT_LINE_WAVELENGTHS`` (the 13-line set used by the
  marginalized e-line likelihood when no line list is given).
- Cloudy-grid line catalog: ``CLOUDY_LINE_NAMES`` /
  ``CLOUDY_LINE_WAVELENGTHS``.
- Standard spectral-index definitions: ``STANDARD_INDICES``.
- SSP-library nominal spectral resolution registry: ``SSP_LIBRARY_RESOLUTIONS``.

The flat ``tengri.observation.X`` import path remains valid, this
module is an *additive* sub-namespace that gathers constants apart
from the user-facing containers and physics primitives.

See Also
--------
tengri.observation.containers : user-facing data classes
tengri.observation.physics : transformation functions
"""

from __future__ import annotations

#: Legal emission line fitting modes for spectroscopic fitting.
#:
#: - ``"off"``: No emission line fitting.
#: - ``"marginalized"``: Analytically marginalize line amplitudes.
#: - ``"fitted"``: Line amplitudes as free MCMC parameters.
ELINE_MODES = ("off", "marginalized", "fitted")

from tengri.observation.eline_catalog import (
    CLOUDY_LINE_NAMES,
    CLOUDY_LINE_WAVELENGTHS,
)
from tengri.observation.eline_marginalization import (
    DEFAULT_LINE_NAMES,
    DEFAULT_LINE_WAVELENGTHS,
)
from tengri.observation.noise import DETECTED, LOWER_LIMIT, UPPER_LIMIT
from tengri.observation.spectral_indices import STANDARD_INDICES
from tengri.observation.spectrum import SSP_LIBRARY_RESOLUTIONS

__all__ = [
    "CLOUDY_LINE_NAMES",
    "CLOUDY_LINE_WAVELENGTHS",
    "DEFAULT_LINE_NAMES",
    "DEFAULT_LINE_WAVELENGTHS",
    "DETECTED",
    "ELINE_MODES",
    "LOWER_LIMIT",
    "SSP_LIBRARY_RESOLUTIONS",
    "STANDARD_INDICES",
    "UPPER_LIMIT",
]
