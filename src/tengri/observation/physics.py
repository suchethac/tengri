# SPDX-License-Identifier: BSD-3-Clause
"""Transformation primitives for the observation layer.

This namespace gathers the JIT-compatible functions that transform a
predicted SED into observable space, line-spread function convolution,
aperture corrections, emission-line design matrices, and instrument-resolution
helpers.

For calibration math primitives, import directly from the sub-modules:

- :func:`tengri.observation.calibration.apply_calibration`
- :func:`tengri.observation.calibration.marginalize_calibration`
- :func:`tengri.observation.calibration.calibration_polynomial`
- :func:`tengri.observation.calibration.chebyshev_basis`

and similarly for emission-line functions from
:mod:`tengri.observation.eline_marginalization` and
:mod:`tengri.observation.eline_priors`.

Examples
--------
>>> from tengri.observation.physics import (
...     apply_lsf,
...     build_eline_design_matrix,
... )
>>> from tengri.observation.calibration import apply_calibration

See Also
--------
tengri.observation.containers: user-facing data classes
tengri.observation.constants: module-level constants (catalogs, status flags)
"""

from __future__ import annotations

from tengri.observation.aperture import apply_aperture_correction
from tengri.observation.eline_marginalization import (
    build_eline_design_matrix,
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
    velocity_broaden,
)

__all__ = [
    "apply_aperture_correction",
    "apply_lsf",
    "apply_wavelength_mask",
    "apply_zp_floor",
    "blend_emission_lines",
    "build_eline_design_matrix",
    "build_line_mask",
    "cloudy_line_priors",
    "marginalize_emission_lines_cloudy",
    "measure_index_jax",
    "nirspec_g140m_resolution",
    "nirspec_prism_resolution",
    "velocity_broaden",
]
