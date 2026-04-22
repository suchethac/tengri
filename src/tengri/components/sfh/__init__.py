"""Star formation history models: PSD kernels, GP generation, mean SFH, registry."""

from tengri.components.sfh.chemical_evolution import (
    chem_evol_metallicity_on_ssp_grid,
    closed_box_metallicity,
    closed_box_metallicity_anchored,
)
from tengri.components.sfh.dense_basis import dense_basis_pure_sfh, dense_basis_sfh
from tengri.components.sfh.gp_sfh import (
    compute_sqrt_power_drw as compute_sqrt_power_drw,
    generate_gp_fourier as generate_gp_fourier,
    gp_from_xi as gp_from_xi,
)
from tengri.components.sfh.mean_sfh import (
    AGEMAX_YR,
    constant_sfh,
    constant_then_exponential_sfh,
    declining_exponential_sfh,
    delayed_exponential_sfh,
    delayed_tau,
    double_powerlaw,
    dpl,
    exponential_sfh,
    gaussian_sfh,
    lnorm,
    lognormal_sfh,
    norm,
    powerlaw_sfh,
    psb_wild2020,
    skewnormal_sfh,
    snorm,
    snorm_burst,
    snorm_burst_sfh,
    snorm_trunc_burst_sfh,
    spline_sfh,
    triweight_burst,
    truncated_skewnormal_sfh,
    tsnorm,
    tsnorm_burst,
)
from tengri.components.sfh.met_registry import MET_REGISTRY, resolve_met
from tengri.components.sfh.metallicity_history import (
    metallicity_bins_continuity_on_ssp_grid,
    metallicity_bins_on_ssp_grid,
    psb_two_step_metallicity,
    tabulated_metallicity_on_ssp_grid,
    two_step_metallicity,
)
from tengri.components.sfh.nonparametric import (
    continuity_prior_logp,
    continuity_sfh,
    dirichlet_sfh,
    make_agebins_from_zred,
    psb_continuity_sfh,
)
from tengri.components.sfh.psd_models import (
    psd_drw as psd_drw,
    psd_extended_regulator as psd_extended_regulator,
    psd_matern as psd_matern,
)
from tengri.components.sfh.registry import (
    FIELD_MODEL_REGISTRY,
    SFH_REGISTRY,
    compute_field_gp,
    resolve_sfh,
)

__all__ = [
    "AGEMAX_YR",
    "FIELD_MODEL_REGISTRY",
    "MET_REGISTRY",
    "SFH_REGISTRY",
    "chem_evol_metallicity_on_ssp_grid",
    # Chemical evolution
    "closed_box_metallicity",
    "closed_box_metallicity_anchored",
    "compute_field_gp",
    "compute_sqrt_power_drw",
    "constant_sfh",
    "constant_then_exponential_sfh",
    "continuity_prior_logp",
    # Nonparametric
    "continuity_sfh",
    "make_agebins_from_zred",
    "declining_exponential_sfh",
    "delayed_exponential_sfh",
    "delayed_tau",
    "dense_basis_pure_sfh",
    # Dense basis
    "dense_basis_sfh",
    "dirichlet_sfh",
    "psb_continuity_sfh",
    "double_powerlaw",
    # Mean SFH models
    "dpl",
    "exponential_sfh",
    "gaussian_sfh",
    # GP and PSD
    "generate_gp_fourier",
    "gp_from_xi",
    # Wrappers for convenience
    "lnorm",
    "lognormal_sfh",
    "metallicity_bins_continuity_on_ssp_grid",
    "metallicity_bins_on_ssp_grid",
    "norm",
    "powerlaw_sfh",
    "psb_two_step_metallicity",
    # PSB SFH
    "psb_wild2020",
    "psd_drw",
    "psd_extended_regulator",
    "psd_matern",
    # Metallicity registry
    "resolve_met",
    # SFH Registry
    "resolve_sfh",
    "skewnormal_sfh",
    "snorm",
    "snorm_burst",
    "snorm_burst_sfh",
    "snorm_trunc_burst_sfh",
    "spline_sfh",
    "tsnorm_burst",
    "tabulated_metallicity_on_ssp_grid",
    "triweight_burst",
    "truncated_skewnormal_sfh",
    "tsnorm",
    # Metallicity history
    "two_step_metallicity",
]
