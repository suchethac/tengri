"""Observation models: photometry, spectroscopy, calibration, and configuration.

Sub-namespaces (additive, browseable groupings of the same objects re-exported here):

- :mod:`tengri.observation.containers` — user-facing data classes
  (Photometry, Spectroscopy, LineFluxData, LineList, NoiseModel,
  Observation, SpectralIndexData, SpectralIndexDef).
- :mod:`tengri.observation.physics` — transformation primitives
  (apply_calibration, apply_lsf, build_eline_design_matrix, …).
- :mod:`tengri.observation.constants` — catalogs and status flags
  (DETECTED / UPPER_LIMIT / LOWER_LIMIT, DEFAULT_LINE_*, STANDARD_INDICES,
  SSP_LIBRARY_RESOLUTIONS).

Sub-namespace bindings are *identical* to the flat-namespace bindings;
they're additive groupings, not copies. The flat surface
(``tengri.observation.X``) remains the source of truth and continues to
work without deprecation warnings.
"""

from tengri.observation import constants, containers, physics
from tengri.observation.aperture import apply_aperture_correction
from tengri.observation.calibration import (
    apply_calibration,
    apply_double_calibration,
    calibration_polynomial,
    chebyshev_basis,
    double_calibration_polynomial,
    marginalize_calibration,
)
from tengri.observation.eline_catalog import (
    CLOUDY_LINE_NAMES,
    CLOUDY_LINE_WAVELENGTHS,
)
from tengri.observation.eline_marginalization import (
    DEFAULT_LINE_NAMES,
    DEFAULT_LINE_WAVELENGTHS,
    build_eline_design_matrix,
    marginalize_emission_lines,
    predict_with_marginalized_lines,
)
from tengri.observation.eline_priors import (
    cloudy_line_priors,
    marginalize_emission_lines_cloudy,
)
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_list import LineList
from tengri.observation.line_mask import build_line_mask
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
    "LineFluxData",
    "LineList",
    "NoiseModel",
    "Observation",
    "Photometry",
    "SpectralIndexData",
    "SpectralIndexDef",
    "Spectroscopy",
    "apply_aperture_correction",
    "apply_calibration",
    "apply_double_calibration",
    "apply_lsf",
    "apply_wavelength_mask",
    "apply_zp_floor",
    "blend_emission_lines",
    "build_eline_design_matrix",
    "build_line_mask",
    "calibration_polynomial",
    "chebyshev_basis",
    "cloudy_line_priors",
    "constants",
    "containers",
    "double_calibration_polynomial",
    "marginalize_calibration",
    "marginalize_emission_lines",
    "marginalize_emission_lines_cloudy",
    "measure_index_jax",
    "nirspec_g140m_resolution",
    "nirspec_prism_resolution",
    "physics",
    "predict_with_marginalized_lines",
]
