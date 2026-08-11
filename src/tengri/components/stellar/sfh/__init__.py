# SPDX-License-Identifier: BSD-3-Clause
"""Star formation history models: PSD kernels, GP generation, mean SFH, registry."""

from tengri._completion import curated_dir
from tengri.components.stellar.sfh._prior_sampling import (
    DEFAULT_AGE_GRID_YR,
    sample_sfh_prior,
)
from tengri.components.stellar.sfh.chemical_evolution import (
    chem_evol_metallicity_on_ssp_grid,
    closed_box_metallicity,
    closed_box_metallicity_anchored,
)
from tengri.components.stellar.sfh.dense_basis import (
    dense_basis,
    dense_basis_pure,
)
from tengri.components.stellar.sfh.gp_sfh import (
    compute_sqrt_power_drw as compute_sqrt_power_drw,
    generate_gp_fourier as generate_gp_fourier,
    gp_from_xi as gp_from_xi,
)
from tengri.components.stellar.sfh.mean_sfh import (
    AGEMAX_YR,
    constant,
    constant_then_exponential,
    declining_exponential,
    delayed_exponential,
    delayed_tau,
    double_powerlaw,
    dpl,
    exponential,
    gaussian,
    gaussian_burst,
    lnorm,
    lognormal,
    norm,
    powerlaw,
    psb_wild2020,
    sfhdelayed,
    skewnormal,
    snorm,
    snorm_burst,
    snorm_trunc_burst,
    spline,
    top_hat,
    triweight_burst,
    truncated_skewnormal,
    tsnorm,
    tsnorm_burst,
)
from tengri.components.stellar.sfh.met_registry import MET_REGISTRY, resolve_met
from tengri.components.stellar.sfh.metallicity_history import (
    metallicity_bins_continuity_on_ssp_grid,
    metallicity_bins_on_ssp_grid,
    psb_two_step_metallicity,
    tabulated_metallicity_on_ssp_grid,
    two_step_metallicity,
)
from tengri.components.stellar.sfh.nonparametric import (
    DEFAULT_BIN_EDGES_GYR,
    bursty_continuity_prior_logp,
    continuity,
    continuity_flex,
    continuity_flex_prior_logp,
    continuity_prior_logp,
    dirichlet,
    make_agebins_from_zred,
    psb_continuity,
)
from tengri.components.stellar.sfh.psd_models import (
    psd_drw as psd_drw,
    psd_extended_regulator as psd_extended_regulator,
    psd_matern as psd_matern,
)
from tengri.components.stellar.sfh.registry import (
    FIELD_MODEL_REGISTRY,
    SFH_REGISTRY,
    compute_field_gp,
    resolve_sfh,
)

# Canonical short-form names. The historical ``*_sfh``-suffixed variants
# were redundant inside ``tengri.components.stellar.sfh`` and have been
# removed pre-v1.0.
__all__ = [
    "AGEMAX_YR",
    "DEFAULT_AGE_GRID_YR",
    "DEFAULT_BIN_EDGES_GYR",
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
    "constant_then_exponential",
    "continuity",
    "continuity_flex",
    "continuity_flex_prior_logp",
    "continuity_prior_logp",
    "declining_exponential",
    "delayed_exponential",
    "delayed_tau",
    "dense_basis",
    "dense_basis_pure",
    "dirichlet",
    "double_powerlaw",
    "dpl",
    "exponential",
    "gaussian",
    "gaussian_burst",
    "generate_gp_fourier",
    "gp_from_xi",
    "lnorm",
    "lognormal",
    "make_agebins_from_zred",
    "metallicity_bins_continuity_on_ssp_grid",
    "metallicity_bins_on_ssp_grid",
    "norm",
    "powerlaw",
    "psb_continuity",
    "psb_two_step_metallicity",
    "psb_wild2020",
    "psd_drw",
    "psd_extended_regulator",
    "psd_matern",
    "resolve_met",
    "resolve_sfh",
    "sample_sfh_prior",
    "sfhdelayed",
    "skewnormal",
    "snorm",
    "snorm_burst",
    "snorm_trunc_burst",
    "spline",
    "tabulated_metallicity_on_ssp_grid",
    "top_hat",
    "triweight_burst",
    "truncated_skewnormal",
    "tsnorm",
    "tsnorm_burst",
    "two_step_metallicity",
]


__dir__ = curated_dir(__all__)
