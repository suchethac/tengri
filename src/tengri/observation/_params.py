"""Parameter declarations for the observation module.

Noise model parameters (calibration floor, Student-t degrees of freedom)
belong to the observation layer and are declared here following ADR-0005
component-owned `_params.py` pattern.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed
from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "noise_frac_cal",
        Fixed(0.0),
        "Fractional calibration noise floor (added in quadrature with obs noise)",
        lambda lo, hi: lo >= 0,
        "noise_frac_cal bounds must have lo >= 0",
    ),
    ParamDeclaration(
        "noise_dof",
        Fixed(0.0),
        "Student-t degrees of freedom for outlier robustness (0=Gaussian)",
        lambda lo, hi: lo >= 0,
        "noise_dof bounds must have lo >= 0",
    ),
)

__all__ = ["PARAMS"]
