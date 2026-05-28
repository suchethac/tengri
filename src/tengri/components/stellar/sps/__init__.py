# SPDX-License-Identifier: BSD-3-Clause
"""Stellar population synthesis wrappers (DSPS backend)."""

# Convenience re-exports for `from tengri.sps import ...`
from tengri.components.stellar.sps.dsps_wrapper import (
    SSPData as SSPData,
    compute_csp_weights as compute_csp_weights,
    compute_surviving_mass as compute_surviving_mass,
    effective_metallicity as effective_metallicity,
    interpolate_mass_remaining as interpolate_mass_remaining,
    interpolate_met_alpha as interpolate_met_alpha,
    load_ssp as load_ssp,
    load_ssp_data as load_ssp_data,
    predict_surviving_mass as predict_surviving_mass,
)
from tengri.components.stellar.sps.mass_remaining import (
    compute_mass_remaining_fraction as mass_remaining_fraction,
)
