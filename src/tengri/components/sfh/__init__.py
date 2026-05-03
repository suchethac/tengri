"""Star formation history models: PSD kernels, GP generation, mean SFH, registry."""

from tengri.components.sfh.chemical_evolution import (
    chem_evol_metallicity_on_ssp_grid,
    closed_box_metallicity,
    closed_box_metallicity_anchored,
)
from tengri.components.sfh.dense_basis import dense_basis, dense_basis_pure
from tengri.components.sfh.gp_sfh import (
    compute_sqrt_power_drw as compute_sqrt_power_drw,
    generate_gp_fourier as generate_gp_fourier,
    gp_from_xi as gp_from_xi,
)
from tengri.components.sfh.mean_sfh import (
    AGEMAX_YR,
    constant,
    constant_then_exponential_sfh,
    declining_exponential_sfh,
    delayed_exponential,
    delayed_tau,
    double_powerlaw,
    dpl,
    exponential,
    gaussian,
    lnorm,
    lognormal,
    norm,
    powerlaw,
    psb_wild2020,
    skewnormal,
    snorm,
    snorm_burst,
    snorm_trunc_burst,
    spline,
    triweight_burst,
    truncated_skewnormal,
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
    bursty_continuity_prior_logp,
    continuity,
    continuity_flex,
    continuity_flex_prior_logp,
    continuity_prior_logp,
    dirichlet,
    make_agebins_from_zred,
    psb_continuity,
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

# Backward compatibility aliases — deprecated, will be removed in v1.0
# The `_sfh` suffix was redundant inside the `tengri.components.sfh` namespace
constant_sfh = constant
exponential_sfh = exponential
delayed_exponential_sfh = delayed_exponential
gaussian_sfh = gaussian
lognormal_sfh = lognormal
powerlaw_sfh = powerlaw
skewnormal_sfh = skewnormal
snorm_burst_sfh = snorm_burst
snorm_trunc_burst_sfh = snorm_trunc_burst
spline_sfh = spline
truncated_skewnormal_sfh = truncated_skewnormal
dense_basis_sfh = dense_basis
dense_basis_pure_sfh = dense_basis_pure
dirichlet_sfh = dirichlet
continuity_sfh = continuity
continuity_flex_sfh = continuity_flex
psb_continuity_sfh = psb_continuity

# New short names are canonical; the `_sfh`-suffixed variants remain
# accessible as deprecated aliases (defined just above in this module)
# until v1.0. See docs/dev/api_migration_v0.x.md.
__all__ = [
    "AGEMAX_YR",
    "FIELD_MODEL_REGISTRY",
    "MET_REGISTRY",
    "SFH_REGISTRY",
    "bursty_continuity_prior_logp",
    "chem_evol_metallicity_on_ssp_grid",
    "closed_box_metallicity",
    "closed_box_metallicity_anchored",
    "compute_field_gp",
    "compute_sqrt_power_drw",
    "constant",
    "constant_sfh",
    "constant_then_exponential_sfh",
    "continuity",
    "continuity_flex",
    "continuity_flex_prior_logp",
    "continuity_flex_sfh",
    "continuity_prior_logp",
    "continuity_sfh",
    "declining_exponential_sfh",
    "delayed_exponential",
    "delayed_exponential_sfh",
    "delayed_tau",
    "dense_basis",
    "dense_basis_pure",
    "dense_basis_pure_sfh",
    "dense_basis_sfh",
    "dirichlet",
    "dirichlet_sfh",
    "double_powerlaw",
    "dpl",
    "exponential",
    "exponential_sfh",
    "gaussian",
    "gaussian_sfh",
    "generate_gp_fourier",
    "gp_from_xi",
    "lnorm",
    "lognormal",
    "lognormal_sfh",
    "make_agebins_from_zred",
    "metallicity_bins_continuity_on_ssp_grid",
    "metallicity_bins_on_ssp_grid",
    "norm",
    "powerlaw",
    "powerlaw_sfh",
    "psb_continuity",
    "psb_continuity_sfh",
    "psb_two_step_metallicity",
    "psb_wild2020",
    "psd_drw",
    "psd_extended_regulator",
    "psd_matern",
    "resolve_met",
    "resolve_sfh",
    "skewnormal",
    "skewnormal_sfh",
    "snorm",
    "snorm_burst",
    "snorm_burst_sfh",
    "snorm_trunc_burst",
    "snorm_trunc_burst_sfh",
    "spline",
    "spline_sfh",
    "tabulated_metallicity_on_ssp_grid",
    "triweight_burst",
    "truncated_skewnormal",
    "truncated_skewnormal_sfh",
    "tsnorm",
    "tsnorm_burst",
    "two_step_metallicity",
]
