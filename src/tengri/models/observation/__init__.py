"""Observation models: photometry, spectroscopy, calibration, and configuration."""

from tengri.models.observation.calibration import (
    apply_calibration,
    calibration_polynomial,
    chebyshev_basis,
)
from tengri.models.observation.eline_marginalization import (
    DEFAULT_LINE_NAMES,
    DEFAULT_LINE_WAVELENGTHS,
    build_eline_design_matrix,
    marginalize_emission_lines,
    predict_with_marginalized_lines,
)
from tengri.models.observation.noise_config import NoiseConfig
from tengri.models.observation.observation import Observation
from tengri.models.observation.photometry_config import Photometry
from tengri.models.observation.spectroscopy import (
    SSP_LIBRARY_RESOLUTIONS,
    apply_lsf,
    blend_emission_lines,
    nirspec_g140m_resolution,
    nirspec_prism_resolution,
)
from tengri.models.observation.spectroscopy_config import SpectroscopyConfig

__all__ = [
    "DEFAULT_LINE_NAMES",
    "DEFAULT_LINE_WAVELENGTHS",
    "SSP_LIBRARY_RESOLUTIONS",
    "NoiseConfig",
    "Observation",
    "Photometry",
    "SpectroscopyConfig",
    "apply_calibration",
    "apply_lsf",
    "blend_emission_lines",
    "build_eline_design_matrix",
    "calibration_polynomial",
    "chebyshev_basis",
    "marginalize_emission_lines",
    "nirspec_g140m_resolution",
    "nirspec_prism_resolution",
    "predict_with_marginalized_lines",
]
