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
    skewnormal_sfh,
    snorm,
    triweight_burst,
    truncated_skewnormal_sfh,
    tsnorm,
)
from tengri.components.sfh.nonparametric import (
    continuity_prior_logp,
    continuity_sfh,
    dirichlet_sfh,
)
from tengri.components.sfh.psd_models import psd_drw as psd_drw
from tengri.components.sfh.registry import (
    FIELD_MODEL_REGISTRY,
    SFH_REGISTRY,
    compute_field_gp,
    resolve_sfh,
)

__all__ = [
    # Mean SFH models
    "dpl",
    "tsnorm",
    "double_powerlaw",
    "exponential_sfh",
    "delayed_tau",
    "constant_sfh",
    "declining_exponential_sfh",
    "delayed_exponential_sfh",
    "gaussian_sfh",
    "lognormal_sfh",
    "skewnormal_sfh",
    "triweight_burst",
    "truncated_skewnormal_sfh",
    "powerlaw_sfh",
    # Nonparametric
    "continuity_sfh",
    "dirichlet_sfh",
    "continuity_prior_logp",
    # Dense basis
    "dense_basis_sfh",
    "dense_basis_pure_sfh",
    # Chemical evolution
    "closed_box_metallicity",
    "closed_box_metallicity_anchored",
    "chem_evol_metallicity_on_ssp_grid",
    # GP and PSD
    "generate_gp_fourier",
    "gp_from_xi",
    "compute_sqrt_power_drw",
    "psd_drw",
    # Registry
    "resolve_sfh",
    "SFH_REGISTRY",
    "FIELD_MODEL_REGISTRY",
    "compute_field_gp",
    # Wrappers for convenience
    "lnorm",
    "norm",
    "snorm",
    "AGEMAX_YR",
]
