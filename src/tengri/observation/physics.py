# SPDX-License-Identifier: BSD-3-Clause
"""Transformation primitives for the observation layer.

This namespace gathers the JIT-compatible functions that transform a
predicted SED into observable space — calibration polynomials,
line-spread function convolution, aperture corrections, emission-line
design matrices, marginalisation kernels, and instrument-resolution
helpers.

The flat ``tengri.observation.X`` import path remains valid — this
module is an *additive* sub-namespace that gathers the physics
primitives apart from the user-facing containers and constants.

Examples
--------
>>> from tengri.observation.physics import (
...     apply_calibration,
...     apply_lsf,
...     build_eline_design_matrix,
... )

See Also
--------
tengri.observation.containers : user-facing data classes
tengri.observation.constants : module-level constants (catalogs, status flags)
"""

from __future__ import annotations

from tengri.observation.aperture import apply_aperture_correction
from tengri.observation.calibration import (
    apply_calibration,
    apply_double_calibration,
    calibration_polynomial,
    chebyshev_basis,
    double_calibration_polynomial,
    marginalize_calibration,
)
from tengri.observation.eline_marginalization import (
    build_eline_design_matrix,
    marginalize_emission_lines,
    predict_with_marginalized_lines,
)
from tengri.observation.eline_priors import (
    cloudy_line_priors,
    marginalize_emission_lines_cloudy,
)
from tengri.observation.line_mask import build_line_mask
from tengri.observation.noise import apply_zp_floor
from tengri.observation.spectral_indices import measure_index_jax
from tengri.observation.spectroscopy import apply_wavelength_mask
from tengri.observation.spectrum import (
    apply_lsf,
    blend_emission_lines,
    nirspec_g140m_resolution,
    nirspec_prism_resolution,
)

__all__ = [
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
    "double_calibration_polynomial",
    "marginalize_calibration",
    "marginalize_emission_lines",
    "marginalize_emission_lines_cloudy",
    "measure_index_jax",
    "nirspec_g140m_resolution",
    "nirspec_prism_resolution",
    "predict_with_marginalized_lines",
]
