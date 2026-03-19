"""Observation models: photometry, spectroscopy, and calibration."""

from diffsed.models.observation.calibration import (
    apply_calibration,
    calibration_polynomial,
    chebyshev_basis,
)
from diffsed.models.observation.eline_marginalization import (
    DEFAULT_LINE_NAMES,
    DEFAULT_LINE_WAVELENGTHS,
    build_eline_design_matrix,
    marginalize_emission_lines,
    predict_with_marginalized_lines,
)
from diffsed.models.observation.spectroscopy import blend_emission_lines

__all__ = [
    "DEFAULT_LINE_NAMES",
    "DEFAULT_LINE_WAVELENGTHS",
    "apply_calibration",
    "blend_emission_lines",
    "build_eline_design_matrix",
    "calibration_polynomial",
    "chebyshev_basis",
    "marginalize_emission_lines",
    "predict_with_marginalized_lines",
]
