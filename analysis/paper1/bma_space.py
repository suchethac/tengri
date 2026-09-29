"""Model space for Bayesian model averaging over SED configurations.

The factorial model space is a Cartesian product of component axes:
- sfh: continuity, dirichlet, delayed, dpl, lnorm (5 options)
- ssp: 5 stellar population grids from configurations I-V
- attenuation: calzetti, smc, kriek_conroy_2c, cf00_2c (4 options)
- dust_emission: dl14 (fixed)
- nebular: cue (fixed)

Total: 5 * 5 * 4 * 1 * 1 = 100 factorial models.

The axes, model keys, enumeration and weight-set membership live in
``_bma_keys`` (jax-free, shared with the evidence runner and the combiner); this
module holds only the jax-dependent model builders.

Component dicts are extracted from configs.py to guarantee exact parity;
no copy-paste of priors or parameters.
"""

from __future__ import annotations

from configs import (
    _continuity_sfh,
    _kriek_conroy_two_component,
    met_prior_for,
)

import tengri
from tengri import DEFAULT, FREE, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.cosmology import age_at_z

from ._bma_keys import ATTENUATION_TYPES, DUST_EMISSION, NEBULAR, SFH_TYPES

LOG10_ZSUN = -1.848
MET_EDGE_INSET_DEX = 0.02


def _calzetti_single_component() -> dict:
    """Calzetti single-screen attenuation (from config_II)."""
    return {
        "type": "single_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_v": Uniform(0.0, 3.0),
    }


def _smc_single_component() -> dict:
    """SMC single-screen attenuation (from config_V)."""
    return {
        "type": "single_component",
        "law": "smc",
        "all_params": Fixed(DEFAULT),
        "tau_v": Uniform(0.0, 3.0),
    }


def _charlot_fall_two_component() -> dict:
    """Charlot & Fall (2000) two-component attenuation (from config_III)."""
    return {
        "type": "two_component",
        "law_bc": "power_law",
        "slope_bc": -1.3,
        "law_diff": "power_law",
        "slope_diff": -0.7,
        "all_params": Fixed(DEFAULT),
        "tau_bc": Uniform(0.0, 3.0),
        "tau_diff": Uniform(0.5, 3.0),
    }


def _dirichlet_sfh(ssp_data: tengri.SSPData, z: float) -> dict:
    """Dirichlet SFH (from config_IV)."""
    N_SFH_BINS = 7
    sfh = {
        "type": "dirichlet",
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": met_prior_for(ssp_data),
        "bin_edges_gyr": tengri.make_agebins_from_zred(z, n_bins=N_SFH_BINS),
    }
    for i in range(N_SFH_BINS - 1):
        sfh[f"z_{i}"] = FREE
    return sfh


def _delayed_sfh(ssp_data: tengri.SSPData, z: float) -> dict:
    """Delayed-tau SFH (from config_III)."""
    return {
        "type": "delayed",
        "all_params": Fixed(DEFAULT),
        "tau_gyr": Uniform(0.1, 20.0),
        "age_gyr": Uniform(1.0, age_at_z(z)),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": met_prior_for(ssp_data),
    }


def _dpl_sfh(ssp_data: tengri.SSPData, z: float) -> dict:
    """Double power-law SFH (from config_II)."""
    tau_upper = age_at_z(z)
    return {
        "type": "dpl",
        "all_params": Fixed(DEFAULT),
        "alpha": Uniform(0.5, 5.0),
        "beta": Uniform(0.3, 3.0),
        "tau_gyr": Uniform(0.5, tau_upper),
        "age_gyr": Uniform(1.0, age_at_z(z)),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": met_prior_for(ssp_data),
    }


def _lnorm_sfh(ssp_data: tengri.SSPData, z: float) -> dict:
    """Log-normal SFH (from config_V)."""
    return {
        "type": "lnorm",
        "all_params": Fixed(DEFAULT),
        "peak_gyr": Uniform(0.1, age_at_z(z)),
        "width_gyr": Uniform(0.1, 5.0),
        "age_gyr": Uniform(1.0, age_at_z(z)),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": met_prior_for(ssp_data),
    }


# Mapping from axis names to dict builders
_SFH_BUILDERS = {
    "continuity": _continuity_sfh,
    "dirichlet": _dirichlet_sfh,
    "delayed": _delayed_sfh,
    "dpl": _dpl_sfh,
    "lnorm": _lnorm_sfh,
}

_ATTENUATION_BUILDERS = {
    "calzetti": _calzetti_single_component,
    "smc": _smc_single_component,
    "kriek_conroy_2c": _kriek_conroy_two_component,
    "cf00_2c": _charlot_fall_two_component,
}

# The builders must cover exactly the axis values the keys enumerate.
if tuple(_SFH_BUILDERS) != SFH_TYPES or tuple(_ATTENUATION_BUILDERS) != ATTENUATION_TYPES:
    raise RuntimeError("bma_space builders are out of step with the _bma_keys axes")


def build_model(
    components: dict[str, str],
    ssp_data: tengri.SSPData,
    observation: tengri.Observation,
    z: float,
) -> SEDModel:
    """Build a SEDModel from component names.

    Args:
        components: dict with "sfh", "ssp", "attenuation", "dust_emission",
            "nebular" keys (all required for factorial models).
        ssp_data: Preloaded SSP data.
        observation: Observation with photometry.
        z: Redshift.

    Returns:
        Compiled SEDModel ready for inference.

    Raises:
        ValueError: if a component name is unrecognized.
    """
    sfh_type = components["sfh"]
    att_type = components["attenuation"]
    dust_em_type = components["dust_emission"]
    neb_type = components["nebular"]

    if sfh_type not in _SFH_BUILDERS:
        raise ValueError(f"Unknown sfh type: {sfh_type}")
    if att_type not in _ATTENUATION_BUILDERS:
        raise ValueError(f"Unknown attenuation type: {att_type}")
    if dust_em_type != DUST_EMISSION:
        raise ValueError(
            f"Only {DUST_EMISSION!r} dust emission is supported; got {dust_em_type!r}"
        )
    if neb_type != NEBULAR:
        raise ValueError(f"Only {NEBULAR!r} nebular is supported; got {neb_type!r}")

    sfh_builder = _SFH_BUILDERS[sfh_type]
    sfh = sfh_builder(ssp_data, z)

    att_builder = _ATTENUATION_BUILDERS[att_type]
    dust_attenuation = att_builder()

    dust_emission = {"type": DUST_EMISSION, "all_params": Fixed(DEFAULT)}
    neb = {"type": NEBULAR, "all_params": Fixed(DEFAULT), "neb_logU": Fixed(-2.5)}

    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh=sfh,
        dust_attenuation=dust_attenuation,
        dust_emission=dust_emission,
        neb=neb,
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )
