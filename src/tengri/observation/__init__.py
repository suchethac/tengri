# SPDX-License-Identifier: BSD-3-Clause
"""Observation models: photometry, spectroscopy, calibration, and configuration.

Sub-namespaces (additive, browseable groupings of the same objects):

- :mod:`tengri.observation.containers`: user-facing data classes
  (Photometry, Spectroscopy, LineFluxData, LineList, NoiseModel,
  Observation, SpectralIndexData, SpectralIndexDef, Instrument).
- :mod:`tengri.observation.physics`: transformation primitives
  (apply_aperture_correction, apply_lsf, build_eline_design_matrix,
  apply_wavelength_mask, …).
- :mod:`tengri.observation.constants`: catalogs and status flags
  (DETECTED / UPPER_LIMIT / LOWER_LIMIT, DEFAULT_LINE_*, STANDARD_INDICES,
  SSP_LIBRARY_RESOLUTIONS).

Root-level exports include user-facing containers and constants. Calibration
math primitives live in :mod:`tengri.observation.calibration`,
:mod:`tengri.observation.eline_marginalization`, and related sub-modules,
accessible via direct imports (e.g.,
:func:`tengri.observation.calibration.marginalize_calibration`).
"""

from tengri.observation import constants, containers, physics
from tengri.observation.aperture import apply_aperture_correction
from tengri.observation.data import Data, ValidatedData
from tengri.observation.eline_catalog import (
    CLOUDY_LINE_NAMES,
    CLOUDY_LINE_WAVELENGTHS,
)
from tengri.observation.eline_marginalization import (
    DEFAULT_LINE_NAMES,
    DEFAULT_LINE_WAVELENGTHS,
    build_eline_design_matrix,
)
from tengri.observation.eline_priors import (
    cloudy_line_priors,
    marginalize_emission_lines_cloudy,
)
from tengri.observation.instrument import Instrument, list_instruments
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_list import LineList
from tengri.observation.line_mask import build_line_mask
from tengri.observation.line_ratio_data import LineRatioData
from tengri.observation.noise import DETECTED, LOWER_LIMIT, UPPER_LIMIT, apply_zp_floor
from tengri.observation.noise_model import NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    SpectralIndexData,
    SpectralIndexDef,
    measure_index_jax,
)
from tengri.observation.spectroscopy import (
    Spectroscopy,
    apply_wavelength_mask,
)
from tengri.observation.spectrum import (
    SSP_LIBRARY_RESOLUTIONS,
    apply_lsf,
    blend_emission_lines,
    nirspec_g140m_resolution,
    nirspec_prism_resolution,
    velocity_broaden,
)

__all__ = [
    "CLOUDY_LINE_NAMES",
    "CLOUDY_LINE_WAVELENGTHS",
    "DEFAULT_LINE_NAMES",
    "DEFAULT_LINE_WAVELENGTHS",
    "DETECTED",
    "LOWER_LIMIT",
    "SSP_LIBRARY_RESOLUTIONS",
    "STANDARD_INDICES",
    "UPPER_LIMIT",
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
    "apply_aperture_correction",
    "apply_lsf",
    "apply_wavelength_mask",
    "apply_zp_floor",
    "blend_emission_lines",
    "build_eline_design_matrix",
    "build_line_mask",
    "cloudy_line_priors",
    "constants",
    "containers",
    "list_instruments",
    "marginalize_emission_lines_cloudy",
    "measure_index_jax",
    "nirspec_g140m_resolution",
    "nirspec_prism_resolution",
    "physics",
    "velocity_broaden",
]
