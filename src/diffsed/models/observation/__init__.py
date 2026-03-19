"""Observation models: photometry, spectroscopy, and calibration."""

from diffsed.models.observation.calibration import (
    apply_calibration,
    calibration_polynomial,
    chebyshev_basis,
)
from diffsed.models.observation.spectroscopy import blend_emission_lines

__all__ = [
    "apply_calibration",
    "blend_emission_lines",
    "calibration_polynomial",
    "chebyshev_basis",
]
