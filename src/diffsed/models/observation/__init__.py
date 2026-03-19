"""Observation models: photometry, spectroscopy, and calibration."""

from diffsed.models.observation.calibration import (
    apply_calibration,
    calibration_polynomial,
    chebyshev_basis,
)

__all__ = [
    "apply_calibration",
    "calibration_polynomial",
    "chebyshev_basis",
]
