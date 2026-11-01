# SPDX-License-Identifier: BSD-3-Clause
"""X-like configurations: external-code parity implementations.

Five configurations designed to match Pacifici et al. (2023) Table 1 codes:
CIGALE, Prospector, BAGPIPES, BEAGLE, and Dense Basis. Each follows the
published code's physics pipeline while using tengri's default priors and
component libraries. Built as free-parameter models for fitting.
"""

from __future__ import annotations

import numpy as np

from tengri import DEFAULT, Fixed, LogUniform, SEDModel, Uniform, WavePrecomp, load_ssp
from tengri.cosmology import age_at_z
from tengri.parameters.translate import LOG10_ZSUN

from .config_metadata import XLIKE_SSP
from .configs import _continuity_sfh, met_prior_for

# BEAGLE priors (Chevallard & Charlot 2016; BEAGLE example parameter files).
# tau is log-uniform over 10^7-10^10.5 yr in BEAGLE; tengri's delayed-tau declaration
# stops at 20 Gyr, indistinguishable from 31.6 Gyr over the age of a z ~ 1 galaxy.
BEAGLE_TAU_GYR = (0.01, 20.0)
# Log-uniform maximum stellar age from 0.1 Gyr to the age of the universe at z.
BEAGLE_MIN_AGE_GYR = 0.1
# CC16 Table 2 gives [-12, -7]; -7 is the pole psi_S * dt = 1 at a 10 Myr window.
BEAGLE_LOG_SSFR = (-12.0, -7.3)
BEAGLE_RECENT_TIMESCALE_MYR = 10.0
# Flat in log tau_V (CC16 sec. 4.2); mu fixed as in the workshop run (every catalog mu = 0.400).
BEAGLE_TAU_V = (1e-3, 5.0)
BEAGLE_MU = 0.4
# Uniform in log10(Z / BEAGLE_ZSUN), with BEAGLE's solar metallicity.
BEAGLE_LOGZ_RANGE_BEAGLE_ZSUN = (-2.2, 0.24)
BEAGLE_ZSUN = 0.01524


def cigale_like(ssp_data, observation, z: float) -> SEDModel:
    """CIGALE-like: delayed-tau SFH, BC03, leitherer02 2-comp, dale2014_cigale, Cue.

    Parameterization follows Pacifici et al. (2023) Table 1 (CIGALE row):
    - SFH: Delayed exponential (delayed-tau)
    - SSP: BC03 (registered grid: bc03_pdva_stelib_chabrier)
    - Attenuation: Two-component leitherer02 (young and diffuse)
    - Dust IR: Dale 2014 CIGALE variant with fixed alpha_dale=2.0
    - Nebular: Cue at fixed logU=-2.0, fesc=0

    Physical differences vs external CIGALE (stated in mismatches):
    - leitherer02 two-component screens (independent tau_bc, tau_diff) vs
      CIGALE's fixed birth-cloud/diffuse ratio (0.44 factor)
    - Cue vs CIGALE's Cloudy 13 grids
    - Stellar metallicity continuous over the grid vs CIGALE's discrete values
    """
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Uniform(0.1, 20.0),
            "age_frac": Uniform(1.0 / age_at_z(z), 1.0),  # formation age >= 1 Gyr
            "log_total_mass": Uniform(8.0, 12.5),
            "met_logzsol": met_prior_for(ssp_data),
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "leitherer02",
            "law_diff": "leitherer02",
            "lyman_cutoff": True,
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 3.0),
            "tau_diff": Uniform(0.0, 3.0),
        },
        dust_emission={
            "type": "dale2014_cigale",
            "alpha_dale": Fixed(2.0),
            "all_params": Fixed(DEFAULT),
        },
        neb={
            "type": "cue",
            "neb_logU": Fixed(-2.0),
            "neb_fesc": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def prospector_like(ssp_data, observation, z: float) -> SEDModel:
    """Prospector-like: continuity SFH, FSPS MIST/MILES, calzetti 2-comp, draine_li2007, Cue.

    Parameterization follows Pacifici et al. (2023) Table 1 (Prospector row):
    - SFH: Continuity (piecewise-constant 7 bins with Student-t log-ratio priors)
    - SSP: FSPS MIST/MILES (registered grid: fsps_mist_miles_chabrier)
    - Attenuation: Two-component Calzetti
    - Dust IR: Draine & Li 2007 with fixed qpah, umin, gamma_dl (fiducial values)
    - Nebular: Cue at fixed logU=-2.0, logZ_gas=0.0 (solar)

    Physical differences vs external Prospector (stated in mismatches):
    - Continuity with Leja et al. (2019) Student-t prior on log-SFR ratios
      vs Prospector's free parameters (exact form not recorded in workshop output)
    - Cue vs FSPS/Cloudy 13 grids (Byler+2017)
    - Metallicity continuous over the grid vs Prospector's discrete values
    - Energy balance excludes lambda<912 Å; far-IR ~11% lower than exact integration
    """
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh=_continuity_sfh(ssp_data, z),
        dust_attenuation={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 3.0),
            "tau_diff": Uniform(0.0, 3.0),
        },
        dust_emission={
            "type": "draine_li2007",
            "qpah": Fixed(2.5),
            "umin": Fixed(1.0),
            "gamma_dl": Fixed(0.05),
            "all_params": Fixed(DEFAULT),
        },
        neb={
            "type": "cue",
            "neb_logU": Fixed(-2.0),
            "neb_logZ_gas": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def bagpipes_like(ssp_data, observation, z: float) -> SEDModel:
    """BAGPIPES-like: double power law SFH, BC03, calzetti 1-comp, draine_li2007, Cue.

    Parameterization follows Pacifici et al. (2023) Table 1 (BAGPIPES row):
    - SFH: Double power law (alpha, beta, tau_gyr, age_frac)
    - SSP: BC03 2003 (registered grid: bc03_pdva_stelib_chabrier)
    - Attenuation: Single-component Calzetti (birth-cloud term off)
    - Dust IR: Draine & Li 2007 with fixed fiducial qpah, umin, gamma_dl
    - Nebular: Cue at fixed logU=-2.0, solar metallicity

    Physical differences vs external BAGPIPES (stated in mismatches):
    - dpl priors Uniform(log space for alpha/beta) vs BAGPIPES's log-uniform
      (observed catalog uses dblplaw:* columns, exact priors not recorded)
    - Cue vs BAGPIPES's Cloudy 17 grids
    - BC03 2003 vs BAGPIPES 2016 update (MILES library variation)
    - Single-component Calzetti (tau_v) vs BAGPIPES two-component (birth-cloud
      term off by fixing tau_bc=0); physically equivalent, different parameterization
    """
    tau_upper = age_at_z(z)
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "alpha": Uniform(0.5, 5.0),
            "beta": Uniform(0.3, 3.0),
            "tau_gyr": Uniform(0.5, tau_upper),
            "age_frac": Uniform(1.0 / age_at_z(z), 1.0),  # formation age >= 1 Gyr
            "log_total_mass": Uniform(8.0, 12.5),
            "met_logzsol": met_prior_for(ssp_data),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Uniform(0.0, 3.0),
        },
        dust_emission={
            "type": "draine_li2007",
            "qpah": Fixed(2.5),
            "umin": Fixed(1.0),
            "gamma_dl": Fixed(0.05),
            "all_params": Fixed(DEFAULT),
        },
        neb={
            "type": "cue",
            "neb_logU": Fixed(-2.0),
            "neb_logZ_gas": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def beagle_met_prior(ssp_data) -> Uniform:
    """BEAGLE metallicity prior in log10(Z/Zsun), clipped to the SSP grid.

    BEAGLE samples log10(Z/BEAGLE_ZSUN) uniformly on ``BEAGLE_LOGZ_RANGE_BEAGLE_ZSUN``.
    That range is shifted into tengri's log10(Z/Zsun) convention and intersected
    with the grid-derived prior from ``met_prior_for``.

    Parameters
    ----------
    ssp_data : SSPData
        SSP library whose metallicity grid bounds the prior.

    Returns
    -------
    Uniform
        Prior on ``met_logzsol`` with lower bound ``max(beagle_lo, grid_lo)`` and
        upper bound ``min(beagle_hi, grid_hi)``.

    Raises
    ------
    ValueError
        If the intersection is empty.
    """
    offset = float(np.log10(BEAGLE_ZSUN) - LOG10_ZSUN)
    beagle_lo, beagle_hi = (edge + offset for edge in BEAGLE_LOGZ_RANGE_BEAGLE_ZSUN)
    grid = met_prior_for(ssp_data)
    lower = max(beagle_lo, grid.lo)
    upper = min(beagle_hi, grid.hi)
    if lower >= upper:
        raise ValueError(
            f"BEAGLE metallicity prior [{beagle_lo:.3f}, {beagle_hi:.3f}] does not "
            f"overlap the grid prior [{grid.lo:.3f}, {grid.hi:.3f}] in log10(Z/Zsun)"
        )
    return Uniform(lower, upper)


def beagle_like(ssp_data, observation, z: float) -> SEDModel:
    """BEAGLE-like: delayed-tau SFH with a recent episode, BC03, CF00 tau_V/mu dust, Gutkin+16 nebular.

    Parameterization follows Chevallard & Charlot (2016) and Gutkin, Charlot &
    Bruzual (2016):
    - SFH: delayed-tau plus a recent episode (``recent_ssfr``) with a fixed
      10 Myr timescale. Priors: ``tau_gyr`` LogUniform(0.01, 20),
      ``age_frac`` LogUniform(0.1 Gyr / age(z), 1),
      ``log_total_mass`` Uniform(8, 12.5), ``log_ssfr`` Uniform(-12, -7.3),
      ``timescale_myr`` Fixed(10)
    - Metallicity: ``beagle_met_prior`` (BEAGLE's uniform prior in
      log10(Z/0.01524) on [-2.2, 0.24], clipped to the grid)
    - SSP: BC03 (registered grid: bc03_pdva_stelib_chabrier)
    - Attenuation: Charlot & Fall (2000) two-component power laws parameterized
      by (tau_V, mu), with slopes -1.3 (birth cloud) and -0.7 (diffuse). Priors:
      ``tau_v`` LogUniform(1e-3, 5); ``mu`` Fixed(0.4)
    - Dust IR: none
    - Nebular: Gutkin+16 with free ``logU`` Uniform(-4, -1); ``xi_d`` Fixed(0.3),
      ``log_nH`` Fixed(2), ``co_ratio`` Fixed(1)
    - IGM: Inoue

    **Parity check: False** — no reproduction notebook available.

    Physical differences vs BEAGLE (stated in mismatches):
    - BC03 2003 instead of BEAGLE's 2016 Charlot & Bruzual update (not public)
    - Gas-phase metallicity fixed at the nebular default instead of tied to the stellar metallicity
    - sSFR prior stops at -7.3 (BEAGLE's -7 is the pole of the conversion at a 10 Myr window)
    - tau prior stops at 20 Gyr instead of 31.6 Gyr (tengri's delayed-tau declaration)
    - A common 5% flux-error floor instead of BEAGLE's 2%
    """
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={
            "type": ["delayed", "recent_ssfr"],
            "all_params": Fixed(DEFAULT),
            "tau_gyr": LogUniform(BEAGLE_TAU_GYR[0], BEAGLE_TAU_GYR[1]),
            "age_frac": LogUniform(BEAGLE_MIN_AGE_GYR / age_at_z(z), 1.0),
            "log_total_mass": Uniform(8.0, 12.5),
            "met_logzsol": beagle_met_prior(ssp_data),
            "log_ssfr": Uniform(BEAGLE_LOG_SSFR[0], BEAGLE_LOG_SSFR[1]),
            "timescale_myr": Fixed(BEAGLE_RECENT_TIMESCALE_MYR),
        },
        dust_attenuation={
            "type": "two_component",
            "parameterization": "tauv_mu",
            "law_bc": "power_law",
            "slope_bc": -1.3,
            "law_diff": "power_law",
            "slope_diff": -0.7,
            "tau_v": LogUniform(BEAGLE_TAU_V[0], BEAGLE_TAU_V[1]),
            "mu": Fixed(BEAGLE_MU),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "none"},
        neb={
            "type": "gutkin16",
            "logU": Uniform(-4.0, -1.0),
            "xi_d": Fixed(0.3),
            "log_nH": Fixed(2.0),
            "co_ratio": Fixed(1.0),
            "other_params": Fixed(DEFAULT),
        },
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def dense_basis_like(ssp_data, observation, z: float) -> SEDModel:
    """Dense Basis-like: GP-SFH, FSPS MIST/MILES, calzetti 1-comp, draine_li2007, Cue.

    Parameterization follows Pacifici et al. (2023) Table 1 (Dense Basis row):
    - SFH: Dense basis (non-parametric GP over mass-time quantiles, Iyer+2019)
    - SSP: FSPS MIST/MILES (registered grid: fsps_mist_miles_chabrier)
    - Attenuation: Single-component Calzetti (tau_v)
    - Dust IR: Draine & Li 2007 with fixed fiducial qpah, umin, gamma_dl
    - Nebular: Cue at fixed logU=-2.0

    **Parity check: False** — no reproduction notebook available.

    Physical differences vs external Dense Basis (stated in mismatches):
    - No parity check: no reproduction notebook with Dense Basis API calls
    - Dense Basis atlas uses a full GP with Dirichlet prior on mass quantiles
      over 256 log-grid points (Iyer & Gawiser 2017); tengri's dense_basis uses
      a Matérn 3/2 + Linear kernel GP with independent Uniform priors on three
      mass-fraction quantile times (tx_frac_0/1/2), plus instantaneous SFR
      constraint points for recent history (Iyer et al. 2019 algorithm)
    - FSPS library version (isochrone/spectral library) not recorded in external
      run metadata
    - Cue vs FSPS default Cloudy (Byler+2017)
    - The reference age for the time quantiles is the age of the universe at the
      galaxy's redshift (Iyer et al. 2019), derived from ``redshift`` and the
      cosmology; there is no separate setting for it
    """
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={
            "type": "dense_basis",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(8.0, 12.5),
            "log_sfr_inst": Uniform(-2.0, 3.0),
            "tx_frac_0": Uniform(0.05, 0.95),
            "tx_frac_1": Uniform(0.05, 0.95),
            "tx_frac_2": Uniform(0.05, 0.95),
            "met_logzsol": met_prior_for(ssp_data),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Uniform(0.0, 3.0),
        },
        dust_emission={
            "type": "draine_li2007",
            "qpah": Fixed(2.5),
            "umin": Fixed(1.0),
            "gamma_dl": Fixed(0.05),
            "all_params": Fixed(DEFAULT),
        },
        neb={
            "type": "cue",
            "neb_logU": Fixed(-2.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


# Mapping from x-like key to builder function
XLIKE_BUILDERS = {
    "cigale_like": cigale_like,
    "prospector_like": prospector_like,
    "bagpipes_like": bagpipes_like,
    "beagle_like": beagle_like,
    "dense_basis_like": dense_basis_like,
}


def load_ssp_for_xlike(key: str):
    """Load the stellar library for X-like configuration key.

    Maps X-like keys to their corresponding SSP grid names via
    ``config_metadata.XLIKE_SSP``, the single declared mapping (an inline
    copy here previously drifted from it in shape, not content: this
    function fed the copy's grid *names* -- e.g. "bc03_pdva_stelib_chabrier"
    -- through ``configs.load_ssp_for``, which expects a *configuration key*
    ("I".."VI") and looks it up in ``SSP_FOR_CONFIG``. Every call raised
    ``KeyError: 'bc03_pdva_stelib_chabrier'`` before a grid ever loaded).

    Parameters
    ----------
    key : str
        X-like configuration key (cigale_like, prospector_like, etc.)

    Returns
    -------
    SSPData
        Loaded stellar population synthesis data.

    Raises
    ------
    KeyError
        If key is not a recognized X-like configuration.
    """
    return load_ssp(XLIKE_SSP[key])
