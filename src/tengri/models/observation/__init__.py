"""Observation models: photometry, spectroscopy, and calibration."""

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
from tengri.models.observation.spectroscopy import (
    SSP_LIBRARY_RESOLUTIONS,
    apply_lsf,
    blend_emission_lines,
    nirspec_g140m_resolution,
    nirspec_prism_resolution,
)

__all__ = [
    "DEFAULT_LINE_NAMES",
    "DEFAULT_LINE_WAVELENGTHS",
    "SSP_LIBRARY_RESOLUTIONS",
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
