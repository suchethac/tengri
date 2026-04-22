"""Observation models: photometry, spectroscopy, calibration, and configuration."""

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
from tengri.observation.line_mask import build_line_mask
from tengri.observation.noise import DETECTED, LOWER_LIMIT, UPPER_LIMIT
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
    "NoiseModel",
    "Observation",
    "Photometry",
    "SpectralIndexData",
    "SpectralIndexDef",
    "Spectroscopy",
    "apply_calibration",
    "apply_double_calibration",
    "apply_lsf",
    "apply_wavelength_mask",
    "blend_emission_lines",
    "build_eline_design_matrix",
    "build_line_mask",
    "calibration_polynomial",
    "chebyshev_basis",
    "cloudy_line_priors",
    "double_calibration_polynomial",
    "marginalize_calibration",
    "marginalize_emission_lines",
    "marginalize_emission_lines_cloudy",
    "measure_index_jax",
    "nirspec_g140m_resolution",
    "nirspec_prism_resolution",
    "predict_with_marginalized_lines",
]
