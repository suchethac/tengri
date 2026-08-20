# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for stellar precomp publish/require interface.

Validates that the stellar SED component correctly registers derived keys,
publishes required outputs at the right times, and respects wave_precomp
configuration flags.
"""

import warnings

import chex
import pytest

from tengri import Parameters, SEDModel, WavePrecomp
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def ssp(synthetic_ssp_wide):
    # #613: run on the shared synthetic SSP so these *structural* contract checks
    # execute on CI instead of skipping when data/ssp_*.h5 is absent.
    return synthetic_ssp_wide


@pytest.fixture(scope="module")
def stellar_only_model(ssp, synthetic_tophat_obs):
    """Stellar-only SED model for contract tests."""
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())


@pytest.fixture(scope="module")
def stellar_only_free_z_model(ssp, synthetic_tophat_obs):
    """Free-redshift variant of stellar_only_model for contract tests."""
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Uniform(0.0, 2.0),  # FREE
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())


_PARAMS = {
    "sfh_tsnorm_log_total_mass": 1.0,
    "sfh_tsnorm_peak_lbt_gyr": 2.0,
    "sfh_tsnorm_width_gyr": 1.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 3.0,
}


# ── Contract: DerivedKey registration and publish/require ───────────────────


def test_lut_only_published_when_wave_precomp_on(ssp, synthetic_tophat_obs):
    """state.derived has no stellar_phot_lnu_precomp when wave_precomp=False (default)."""
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs)  # wave_precomp default False
    state = m.predict_state(_PARAMS)
    assert "stellar_phot_lnu_precomp" not in state.derived


def test_free_z_state_carries_ztable_not_lut(stellar_only_free_z_model):
    """Phase 3c-1: component's _state has ssp_phot_ztable populated, ssp_phot_lut None."""
    m = stellar_only_free_z_model
    chain = m._build_component_chain()
    stellar_comp = chain[0]
    assert stellar_comp._state is not None
    assert stellar_comp._state.ssp_phot_ztable is not None, (
        "Free-z model should populate ssp_phot_ztable"
    )
    assert stellar_comp._state.ssp_phot_lut is None, "Free-z model should have ssp_phot_lut=None"


def test_dust_attenuation_precomps_published(ssp, synthetic_tophat_obs):
    """DustAttenuationSEDComponent publishes A and A' per filter when filter_eff_waves
    is in state.derived (i.e. wave_precomp is on).
    """
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_v=Fixed(0.3),
        dust_model="single_component",
        dust_law_bc="calzetti",
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())
    state = m.predict_state(_PARAMS)
    assert "filter_eff_waves" in state.derived, (
        "Stellar should publish filter_eff_waves when wave_precomp=True"
    )
    assert "dust_attenuation_precomp" in state.derived, (
        "Dust should publish dust_attenuation_precomp when filter_eff_waves is available"
    )
    assert "dust_attenuation_slope_precomp" in state.derived, (
        "Dust should publish dust_attenuation_slope_precomp when filter_eff_waves is available"
    )


def test_two_component_dust_publishes_bc_diff_precomp(ssp, synthetic_tophat_obs):
    """Phase 3c-3c-iv-b: two-component dust publishes A_bc, A_diff and slopes
    when filter_eff_waves is available (i.e. wave_precomp is on).

    Phase 3c-3c-iv-c will consume these to apply the per-age Charlot-Fall
    expansion. For now we only validate the publish.
    """
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.3),
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())
    state = m.predict_state(_PARAMS)

    assert "dust_bc_attenuation_precomp" in state.derived
    assert "dust_bc_attenuation_slope_precomp" in state.derived
    assert "dust_diff_attenuation_precomp" in state.derived
    assert "dust_diff_attenuation_slope_precomp" in state.derived
    assert "dust_young_indicator" in state.derived


def test_predict_via_precomp_handles_bakedin_only_no_neb_precomp(stellar_only_model):
    """BakedIn nebular models do NOT publish nebular_phot_lnu_precomp because
    the emission is already in the stellar SSP grid. ``predict_via_precomp``
    must not raise — the multi-component sum reduces to stellar (which
    includes BakedIn nebular).
    """
    m = stellar_only_model
    state = m.predict_state(_PARAMS)
    assert state.derived.get("nebular_phot_lnu_precomp") is None
    full = {**m.spec.get_fixed_values(), **_PARAMS}
    # Should not raise — BakedIn nebular is handled by stellar LUT.
    m.observation.predict_via_precomp(state, full, observables_type=m.Observables)


def test_agn_phot_lnu_precomp_published(ssp, synthetic_tophat_obs):
    """AGN component publishes agn_phot_lnu_precomp when wave_precomp is on."""
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
        agn_model="qsogen",
        agn_log_lbol=Fixed(11.42),
        agn_lum_ratio=Fixed(0.5),
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())
    state = m.predict_state(_PARAMS)
    assert "agn_phot_lnu_precomp" in state.derived
    chex.assert_tree_all_finite(state.derived["agn_phot_lnu_precomp"])


def test_dust_luts_absent_without_wave_precomp(ssp, synthetic_tophat_obs):
    """No filter_eff_waves publish when wave_precomp=False."""
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_v=Fixed(0.3),
        dust_model="single_component",
        dust_law_bc="power_law",
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs)  # no wave_precomp
    state = m.predict_state(_PARAMS)
    assert state.derived.get("filter_eff_waves") is None
    assert state.derived.get("dust_attenuation_precomp") is None
    assert state.derived.get("dust_attenuation_slope_precomp") is None
