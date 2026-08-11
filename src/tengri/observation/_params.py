# SPDX-License-Identifier: BSD-3-Clause
"""Parameter declarations for the observation module.

Noise model parameters (calibration floor, Student-t degrees of freedom)
belong to the observation layer and are declared here following ADR-0005
component-owned `_params.py` pattern.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "noise_frac_cal",
        Fixed(0.0),
        "Fractional calibration noise floor (added in quadrature with obs noise)",
        lambda lo, hi: lo >= 0,
        "noise_frac_cal bounds must have lo >= 0",
        # A fraction of the flux added in quadrature to the reported errors, so
        # 0 (the default) trusts the catalog uncertainties as given. The ceiling
        # is where "calibration floor" stops describing a floor: at 20% the
        # systematic term dominates any well-measured band, and a fit that wants
        # more than that is telling you the photometry, not the model, is the
        # problem.
        free_prior=Uniform(0.0, 0.2, "Fractional calibration floor", default=0.0),
    ),
    ParamDeclaration(
        "noise_dof",
        Fixed(0.0),
        "Student-t degrees of freedom for outlier robustness (0=Gaussian)",
        lambda lo, hi: lo >= 0,
        "noise_dof bounds must have lo >= 0",
        # Deliberately NO free_prior. 0 is a sentinel, not a value: it selects
        # the Gaussian likelihood rather than naming zero degrees of freedom,
        # and the Student-t only becomes meaningful above ~1. A prior spanning
        # the sentinel and the continuum at once would make the choice of
        # likelihood a thing the sampler wanders into. Choosing a heavy-tailed
        # likelihood is a modeling decision, so set it explicitly.
    ),
)

__all__ = ["PARAMS"]
