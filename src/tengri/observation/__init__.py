"""Observation models: photometry, spectroscopy, calibration, and configuration."""

from tengri.observation.calibration import (
    apply_calibration,
    calibration_polynomial,
    chebyshev_basis,
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
from tengri.observation.noise_model import NoiseConfig, NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectroscopy import Spectroscopy, SpectroscopyConfig
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
    "SSP_LIBRARY_RESOLUTIONS",
    "NoiseConfig",
    "NoiseModel",
    "Observation",
    "Photometry",
    "Spectroscopy",
    "SpectroscopyConfig",
    "apply_calibration",
    "apply_lsf",
    "blend_emission_lines",
    "build_eline_design_matrix",
    "calibration_polynomial",
    "chebyshev_basis",
    "cloudy_line_priors",
    "marginalize_calibration",
    "marginalize_emission_lines",
    "marginalize_emission_lines_cloudy",
    "nirspec_g140m_resolution",
    "nirspec_prism_resolution",
    "predict_with_marginalized_lines",
]
