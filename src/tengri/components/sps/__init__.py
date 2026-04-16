"""Stellar population synthesis wrappers (DSPS backend)."""

# Convenience re-exports for `from tengri.sps import ...`
from tengri.components.sps.dsps_wrapper import (
    SSPData as SSPData,
    compute_csp_weights as compute_csp_weights,
    effective_metallicity as effective_metallicity,
    interpolate_met_alpha as interpolate_met_alpha,
    load_ssp_data as load_ssp_data,
)
