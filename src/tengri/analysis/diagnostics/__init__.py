# SPDX-License-Identifier: BSD-3-Clause
"""Diagnostics enabled by full differentiability.

These tools are only possible because the entire pipeline
θ → SFH → SSP → dust → observables → likelihood
is differentiable end-to-end. They provide:

1. Fisher Information Matrix (FIM): parameter constraints from data
2. Gradient SEDs (saliency); ∂flux/∂θ across wavelength
3. Laplace approximation: cheap posteriors from the Hessian at MAP
4. Autocorrelation time estimation: Sokal method (Behroozi 2025)
"""

from tengri.analysis.diagnostics.autocorrelation import (
    autocorrelation_time,
    autocorrelation_time_combined,
    check_chain_length,
    effective_sample_size,
)
from tengri.analysis.diagnostics.energy_balance import (
    dust_energy_balance,
    integrate_lnu_over_band,
)
from tengri.analysis.diagnostics.fisher import (
    compute_fisher_matrix,
    compute_jacobian,
    fisher_correlation_matrix,
    fisher_parameter_errors,
)
from tengri.analysis.diagnostics.green_functions import (
    compute_green_function,
    compute_time_sensitivity_matrix,
    compute_window_function,
)
from tengri.analysis.diagnostics.lines import (
    compute_equivalent_widths,
    compute_line_fluxes,
    compute_line_moments,
)
from tengri.analysis.diagnostics.saliency import (
    compute_all_gradient_seds,
    compute_gradient_sed,
    compute_photometry_sensitivity,
)
from tengri.analysis.diagnostics.spectral import (
    dn4000,
    irx,
    rest_frame_color,
    rest_frame_luminosity,
    uv_slope_beta,
)

__all__ = [
    "autocorrelation_time",
    "autocorrelation_time_combined",
    "check_chain_length",
    "compute_all_gradient_seds",
    "compute_equivalent_widths",
    "compute_fisher_matrix",
    "compute_gradient_sed",
    "compute_green_function",
    "compute_jacobian",
    "compute_line_fluxes",
    "compute_line_moments",
    "compute_photometry_sensitivity",
    "compute_time_sensitivity_matrix",
    "compute_window_function",
    "dn4000",
    "dust_energy_balance",
    "effective_sample_size",
    "fisher_correlation_matrix",
    "fisher_parameter_errors",
    "integrate_lnu_over_band",
    "irx",
    "rest_frame_color",
    "rest_frame_luminosity",
    "uv_slope_beta",
]
