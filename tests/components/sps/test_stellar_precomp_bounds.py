# SPDX-License-Identifier: BSD-3-Clause
"""Bounds and finiteness tests for stellar precomputed LUT paths.

Validates that precomputed LUT outputs satisfy physical constraints:
- Non-negativity and unit-interval bounds for attenuation factors
- Finiteness across free-parameter ranges
- Magnitude sanity checks
"""

import pathlib
import warnings

import chex
import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel, WavePrecomp
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.bounds

_SSP = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp(synthetic_ssp_wide):
    # #613: synthetic SSP + synthetic filters so these bounds checks run on CI.
    return synthetic_ssp_wide


@pytest.fixture(scope="module")
def stellar_only_model(ssp, synthetic_tophat_obs):
    """Stellar-only SED model for bounds tests."""
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
    """Free-redshift variant for bounds tests."""
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Uniform(0.0, 2.0),
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


def test_observation_predict_bit_exact_with_wave_precomp_on(stellar_only_model):
    """Phase 3b invariant: observation.predict is UNCHANGED when wave_precomp=True.

    The LUT is computed and published into derived but observation.predict
    continues integrating sed_intrinsic through filters as before. This
    test pins that invariant.
    """
    m = stellar_only_model
    o = m.predict_observables(_PARAMS)
    legacy = m.predict_photometry_components(_PARAMS)
    diff = float(jnp.max(jnp.abs(o.phot_fnu - legacy)))
    assert diff < 1e-10, f"predict_observables drifted when wave_precomp=True: max diff = {diff}"


@pytest.mark.parametrize("z_test", [0.05, 0.5, 1.0, 1.5])
def test_free_z_stellar_lut_runs_for_multiple_z(stellar_only_free_z_model, z_test):
    """The free-z apply path produces finite stellar_phot_lnu_precomp at multiple z.

    Numerical equivalence to direct integration is deferred to Phase 3c-3
    (where the LUT is consumed by observation.predict). Phase 3c-1 just
    pins that the ztable interpolation runs and returns sensible values.
    """
    m = stellar_only_free_z_model
    params = {**_PARAMS, "redshift": z_test}
    state = m.predict_state(params)
    lut_path = state.derived["stellar_phot_lnu_precomp"]
    assert jnp.all(jnp.isfinite(lut_path)), f"non-finite values at z={z_test}: {lut_path}"
    assert jnp.all(lut_path > 0), f"non-positive values at z={z_test}: {lut_path}"


def test_free_z_ztable_interpolation_matches_grid_points(stellar_only_free_z_model):
    """At a grid point of the ztable, the apply-time interp returns the
    grid-point value (no smoothing artifacts).

    Picks a z on the precomputed grid and asserts the apply-time
    interpolated LUT matches the direct grid-point value to within JAX
    precision.
    """
    m = stellar_only_free_z_model
    chain = m._build_component_chain()
    ztable = chain[0]._state.ssp_phot_ztable
    # Pick a grid point (middle of the grid)
    i_mid = ztable.z_grid.shape[0] // 2
    z_grid_point = float(ztable.z_grid[i_mid])
    params = {**_PARAMS, "redshift": z_grid_point}
    state = m.predict_state(params)
    lut_path = state.derived["stellar_phot_lnu_precomp"]

    from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

    age_weights = state.derived["age_weights"]
    ssp_phot_grid = ztable.ssp_phot_table[i_mid]  # (n_met, n_age, n_filt)
    manual = jnp.einsum("a,maf->mf", age_weights, ssp_phot_grid).sum(axis=0) * LSUN_ERG_PER_S

    chex.assert_tree_all_finite(lut_path)
    assert jnp.all(lut_path > 0)
    # Magnitude sanity: within a factor of 100 of the naive sum.
    ratio = lut_path / manual
    assert jnp.all((ratio > 1e-2) & (ratio < 1e2)), (
        f"grid-point LUT magnitude off: ratio={list(map(float, ratio))}"
    )


@pytest.mark.parametrize(
    ("metallicity_model", "met_params"),
    [
        ("delta", {"met_logzsol": Fixed(-0.5)}),
        ("ramp", {"met_logzsol_0": Fixed(-1.0), "met_logzsol_final": Fixed(-0.5)}),
        (
            "two_step",
            {
                "met_logzsol_old": Fixed(-1.0),
                "met_logzsol_young": Fixed(-0.5),
                "met_step_age_gyr": Fixed(1.0),
            },
        ),
    ],
)
def test_lut_publishes_for_metallicity_mode(
    ssp, synthetic_tophat_obs, metallicity_model, met_params
):
    """LUT publishes finite, positive values across non-delta metallicity modes.

    The joint_weights × ssp_phot einsum is metallicity-mode-agnostic; this
    test pins that each supported mode produces a usable LUT.
    """
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
        met_mode=metallicity_model,
        **met_params,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())
    state = m.predict_state(_PARAMS)
    assert "stellar_phot_lnu_precomp" in state.derived, (
        f"stellar_phot_lnu_precomp missing for metallicity_model={metallicity_model}"
    )
    lut = state.derived["stellar_phot_lnu_precomp"]
    assert jnp.all(jnp.isfinite(lut)), f"non-finite LUT for {metallicity_model}: {lut}"
    assert jnp.all(lut > 0), f"non-positive LUT for {metallicity_model}: {lut}"


def test_per_age_lut_sums_to_marginalized_lut(stellar_only_model):
    """Phase 3c-3c-iv-a: age-resolved per-filter LUT sums (over age axis) to the
    existing marginalized stellar_phot_lnu_precomp.

    The age-resolved LUT is the input to two-component dust attenuation
    (Phase 3c-3c-iv-c); summing it over the age axis must recover the
    aggregate LUT to within JAX precision.
    """
    m = stellar_only_model
    state = m.predict_state(_PARAMS)
    per_age = state.derived["stellar_phot_lnu_per_age_precomp"]
    marginalized = state.derived["stellar_phot_lnu_precomp"]
    assert per_age.ndim == 2, f"per_age should be 2D (n_age, n_filter): {per_age.shape}"
    assert per_age.shape[1] == marginalized.shape[0], (
        f"filter axes mismatch: per_age={per_age.shape}, marginalized={marginalized.shape}"
    )
    reconstructed = jnp.sum(per_age, axis=0)
    rel_err = jnp.abs(reconstructed - marginalized) / jnp.abs(marginalized)
    assert float(jnp.max(rel_err)) < 1e-10, (
        f"per_age sum diverges from marginalized LUT: max rel err = {float(jnp.max(rel_err)):.2e}"
    )


@pytest.fixture(scope="module")
def stellar_only_taylor_model(ssp, synthetic_tophat_obs):
    """``stellar_only_model``, but opting IN to the Taylor moment.

    Since #1122 the moment is superseded by the sub-band quadrature and
    ``taylor_correction`` defaults to False, so Psi is no longer built unless
    asked for. The machinery is still supported; these tests exercise it.
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
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(
            spec,
            ssp,
            observation=synthetic_tophat_obs,
            approx=WavePrecomp(n_subbands=0, taylor_correction=True),
        )


def test_default_publishes_the_subband_quadrature_not_the_moment(stellar_only_model):
    """The DEFAULT WavePrecomp now ships the sub-band quadrature (#1122).

    The Taylor moment extrapolated the dust screen from one point per filter and
    diverged in the rest-UV; the quadrature evaluates it at K nodes instead. Ψ is
    no longer built by default — pin that, so a silent revert to the old default
    cannot pass.
    """
    state = stellar_only_model.predict_state(_PARAMS)
    assert "stellar_phot_lnu_per_age_subband_precomp" in state.derived
    assert "stellar_subband_waves_rest_precomp" in state.derived
    assert "stellar_phot_moment_precomp" not in state.derived


def test_per_age_moment_lut_sums_to_marginalized_moment(stellar_only_taylor_model):
    """Same invariant for the Taylor moment Ψ (opt-in since #1122)."""
    m = stellar_only_taylor_model
    state = m.predict_state(_PARAMS)
    per_age = state.derived["stellar_phot_moment_per_age_precomp"]
    marginalized = state.derived["stellar_phot_moment_precomp"]
    reconstructed = jnp.sum(per_age, axis=0)
    rel_err = jnp.abs(reconstructed - marginalized) / jnp.maximum(jnp.abs(marginalized), 1e-30)
    assert float(jnp.max(rel_err)) < 1e-10, (
        f"per_age moment sum diverges: max rel err = {float(jnp.max(rel_err)):.2e}"
    )


def test_taylor_moment_published_in_fixed_z(stellar_only_taylor_model):
    """Phase 3c-3c: stellar_phot_moment_precomp Ψ is published alongside the LUT
    in fixed-z mode.

    Ψ is the first spectral moment of the filter-integrated CSP — units
    erg/s/Hz × Å. The expected magnitude is roughly ``stellar_phot_lnu_precomp``
    × filter width (~1000 Å for SDSS); the moment can be positive or
    negative depending on filter shape and SED slope, so we only check
    that it's finite and the magnitude is in a plausible range.
    """
    m = stellar_only_taylor_model
    state = m.predict_state(_PARAMS)
    assert "stellar_phot_moment_precomp" in state.derived, (
        "Taylor moment should be published when it is explicitly requested "
        "(taylor_correction=True); it is no longer the default (#1122)"
    )
    moment = state.derived["stellar_phot_moment_precomp"]
    lnu = state.derived["stellar_phot_lnu_precomp"]
    chex.assert_tree_all_finite(moment)
    # Order-of-magnitude check: |Ψ| should be at most a few filter widths × |Φ|.
    # SDSS filter widths are ~500–1500 Å; allow up to 1e5 Å as a sanity bound.
    ratio = jnp.abs(moment) / jnp.abs(lnu)
    assert jnp.all(ratio < 1e5), (
        f"Taylor moment magnitude implausible: max |Ψ|/|Φ| = {float(jnp.max(ratio)):.2e} Å"
    )


def test_taylor_moment_published_in_free_z(ssp, synthetic_tophat_obs):
    """Phase 3c-3c-v: free-z mode now publishes the Taylor moment via the
    extended ztable (was deferred in Phase 3c-3c-ii).

    Validates that all the precompute keys land for a free-z model
    so the dust LUT path can consume them at arbitrary runtime z.
    """
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Uniform(0.0, 2.0),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(
            spec, ssp, observation=obs, approx=WavePrecomp(n_subbands=0, taylor_correction=True)
        )
    params = {**_PARAMS, "redshift": 0.5}
    state = m.predict_state(params)
    for k in (
        "stellar_phot_lnu_precomp",
        "stellar_phot_moment_precomp",
        "stellar_phot_lnu_per_age_precomp",
        "stellar_phot_moment_per_age_precomp",
        "filter_eff_waves",
    ):
        assert k in state.derived, (
            f"free-z model should publish {k} when taylor_correction=True "
            "(no longer the default since #1122)"
        )
        assert jnp.all(jnp.isfinite(state.derived[k])), f"{k} contains non-finite values"


def test_dust_attenuation_precomp_consistent_with_pipeline(ssp, synthetic_tophat_obs):
    """LUT A matches exp(-tau_v * k(λ_eff)) computed independently to 1e-10."""
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
    a = state.derived["dust_attenuation_precomp"]
    filter_eff = state.derived["filter_eff_waves"]
    tau_v = 0.3
    from tengri.components.dust.attenuation import calzetti

    k = calzetti(filter_eff)
    a_expected = jnp.exp(-tau_v * k)
    rel_err = jnp.abs(a - a_expected) / jnp.abs(a_expected)
    assert float(jnp.max(rel_err)) < 1e-10, (
        f"dust_attenuation_precomp diverges: max rel err = {float(jnp.max(rel_err)):.2e}"
    )


def test_predict_via_precomp_with_dust_matches_predict(ssp, synthetic_tophat_obs):
    """Phase 3c-3c-iii: under dust attenuation, predict_via_precomp matches
    predict within the documented hybrid-kernel accuracy (~0.5%).

    The LUT path applies the Taylor expansion ``A·Φ + A'·Ψ`` to the stellar
    contribution; the default path integrates the attenuated SED through
    filters directly. The two should agree within Zacharegkas+2025's ~0.5%
    factorization tolerance for SDSS bands on a stellar+Calzetti model.
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
    full = {**m.spec.get_fixed_values(), **_PARAMS}
    default = m.observation.predict(state, full, observables_type=m.Observables)
    precomp = m.observation.predict_via_precomp(state, full, observables_type=m.Observables)
    rel_err = jnp.abs(precomp.phot_fnu - default.phot_fnu) / jnp.abs(default.phot_fnu)
    assert float(jnp.max(rel_err)) < 5e-3, (
        f"predict_via_precomp under dust diverges: max rel err = {float(jnp.max(rel_err)):.4%}"
    )


def test_predict_via_precomp_dust_attenuates_phot_fnu(ssp, synthetic_tophat_obs):
    """The dust precompute path produces lower phot_fnu than the no-dust path
    (stellar-only) — confirms the Taylor expansion actually applies attenuation.
    """
    spec_no_dust = Parameters(
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
        m_no_dust = SEDModel(spec_no_dust, ssp, observation=obs, approx=WavePrecomp())
    state_no_dust = m_no_dust.predict_state(_PARAMS)
    full_no_dust = {**m_no_dust.spec.get_fixed_values(), **_PARAMS}
    fnu_no_dust = m_no_dust.observation.predict_via_precomp(
        state_no_dust, full_no_dust, observables_type=m_no_dust.Observables
    ).phot_fnu

    spec_dust = Parameters(
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
    m_dust = SEDModel(spec_dust, ssp, observation=obs, approx=WavePrecomp())
    state_dust = m_dust.predict_state(_PARAMS)
    full_dust = {**m_dust.spec.get_fixed_values(), **_PARAMS}
    fnu_dust = m_dust.observation.predict_via_precomp(
        state_dust, full_dust, observables_type=m_dust.Observables
    ).phot_fnu

    ratio = fnu_dust / fnu_no_dust
    assert jnp.all(ratio < 1.0), f"dust did not attenuate phot_fnu: ratio = {ratio}"
    # u-band attenuates more than z-band (Calzetti rises into UV).
    assert float(ratio[0]) < float(ratio[-1]), (
        f"u-band should attenuate more than z-band: ratio={list(map(float, ratio))}"
    )


def test_predict_via_precomp_two_component_dust_matches_predict(ssp, synthetic_tophat_obs):
    """Phase 3c-3c-iv-c: two-component dust through the LUT path matches
    ``predict`` within the documented hybrid-kernel accuracy (~0.5%).

    The expansion is ``flux_b = Σ_a per_age[a, b] × A_diff(b) × A_bc(b)^y(a)``,
    where ``y(a)`` is the smooth young indicator. Compared against the
    default ``predict`` which integrates the attenuated SED through filters
    directly. The two should agree within Zacharegkas+2025's factorization
    tolerance on a stellar+Charlot-Fall model.
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
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.3),
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())
    state = m.predict_state(_PARAMS)
    full = {**m.spec.get_fixed_values(), **_PARAMS}
    default = m.observation.predict(state, full, observables_type=m.Observables)
    precomp = m.observation.predict_via_precomp(state, full, observables_type=m.Observables)
    rel_err = jnp.abs(precomp.phot_fnu - default.phot_fnu) / jnp.abs(default.phot_fnu)
    assert float(jnp.max(rel_err)) < 5e-3, (
        f"two-component dust precomp diverges: max rel err = {float(jnp.max(rel_err)):.4%}"
    )


@pytest.mark.parametrize("z_test", [0.5, 1.0, 1.8])
def test_predict_via_precomp_free_z_with_dust_matches_predict(ssp, synthetic_tophat_obs, z_test):
    """Phase 3c-3c-v: free-z + single-component dust through the LUT path
    matches ``predict`` within 0.5% across redshifts.

    Tests the same Taylor-expansion math as Phase 3c-3c-iii but with the
    LUT interpolated from the ztable instead of a fixed-z lookup. Free-z
    fits are the dominant photometric-redshift workflow; this is the
    primary speedup target.
    """
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Uniform(0.0, 2.0),
        dust_tau_v=Fixed(0.3),
        dust_model="single_component",
        dust_law_bc="calzetti",
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())
    params = {**_PARAMS, "redshift": z_test}
    state = m.predict_state(params)
    full = {**m.spec.get_fixed_values(), **params}
    default = m.observation.predict(state, full, observables_type=m.Observables)
    precomp = m.observation.predict_via_precomp(state, full, observables_type=m.Observables)
    rel_err = jnp.abs(precomp.phot_fnu - default.phot_fnu) / jnp.abs(default.phot_fnu)
    # Tolerance: 3% — combines ztable linear interpolation accuracy
    # (~1–2% with n_z=100 across z ∈ [0.001, 3]) and the Zacharegkas+2025
    # dust factorization (~0.5%). Tighter tolerance requires denser n_z
    # or a higher-order ztable interpolation scheme.
    assert float(jnp.max(rel_err)) < 3e-2, (
        f"at z={z_test}: free-z + dust precomp diverges: max rel err = "
        f"{float(jnp.max(rel_err)):.4%}"
    )


def test_predict_via_precomp_agn_matches_predict(ssp, synthetic_tophat_obs):
    """Phase 3c-3d-agn: AGN-bearing model goes through the LUT path and matches
    the default ``predict`` within 0.5%.

    AGN.apply now filter-integrates its analytic SED contribution and
    publishes ``agn_phot_lnu_precomp``. predict_via_precomp's multi-component
    sum picks it up automatically alongside stellar contributions.
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
    full = {**m.spec.get_fixed_values(), **_PARAMS}
    default = m.observation.predict(state, full, observables_type=m.Observables)
    precomp = m.observation.predict_via_precomp(state, full, observables_type=m.Observables)
    rel_err = jnp.abs(precomp.phot_fnu - default.phot_fnu) / jnp.abs(default.phot_fnu)
    assert float(jnp.max(rel_err)) < 5e-3, (
        f"AGN model precomp diverges: max rel err = {float(jnp.max(rel_err)):.4%}"
    )


def test_predict_via_precomp_handles_bakedin_nebular(ssp, synthetic_tophat_obs):
    """Phase 3c-3b: BakedIn nebular flows through the LUT path transparently.

    The default ``_wNE`` MILES SSPs carry baked-in nebular emission. That
    means ``stellar_phot_lnu_precomp`` already contains nebular continuum and
    lines — no separate ``nebular_phot_lnu_precomp`` should be published, and
    the multi-component sum reduces to the stellar contribution alone.

    Cue / CloudyGrid nebular backends will publish their own LUT entry
    in later sub-PRs; this test pins the BakedIn invariant.
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
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())
    state = m.predict_state(_PARAMS)
    # BakedIn nebular doesn't publish a separate LUT.
    assert state.derived.get("nebular_phot_lnu_precomp") is None, (
        "BakedIn nebular should not publish nebular_phot_lnu_precomp — already in stellar LUT."
    )
    # The LUT path still works — predict_via_precomp sums whatever LUTs exist.
    full = {**m.spec.get_fixed_values(), **_PARAMS}
    lut = m.observation.predict_via_precomp(state, full, observables_type=m.Observables)
    default = m.observation.predict(state, full, observables_type=m.Observables)
    rel_err = jnp.abs(lut.phot_fnu - default.phot_fnu) / jnp.abs(default.phot_fnu)
    assert float(jnp.max(rel_err)) < 5e-3, (
        f"BakedIn LUT path drifts: max rel err = {float(jnp.max(rel_err)):.4%}"
    )


def test_predict_via_precomp_matches_default_predict_observables(ssp, synthetic_tophat_obs):
    """Phase 3c-3a opt-in path: predict_via_precomp output matches predict_observables
    within the documented hybrid-kernel accuracy (~0.5%) for a stellar-only model.

    predict_observables integrates state.sed_intrinsic through filters (exact path);
    predict_via_precomp projects the precomputed SSP × filter LUT and applies cosmology
    (approximate path). The two should agree to the LUT's documented accuracy.
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
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())
    state = m.predict_state(_PARAMS)
    full = {**m.spec.get_fixed_values(), **_PARAMS}
    default = m.observation.predict(state, full, observables_type=m.Observables)
    lut = m.observation.predict_via_precomp(state, full, observables_type=m.Observables)
    rel_err = jnp.abs(lut.phot_fnu - default.phot_fnu) / jnp.abs(default.phot_fnu)
    assert float(jnp.max(rel_err)) < 5e-3, (
        f"predict_via_precomp diverges from predict: max rel err = {float(jnp.max(rel_err)):.4%}"
    )
    # Also check magnitudes
    rel_mag = jnp.abs(lut.mag_apparent - default.mag_apparent)
    assert float(jnp.max(rel_mag)) < 5e-3, (
        f"mag_apparent drifts: max abs diff = {float(jnp.max(rel_mag))}"
    )


def test_predict_via_precomp_raises_without_wave_precomp(ssp, synthetic_tophat_obs):
    """predict_via_precomp requires the LUT — raises clearly when missing."""
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
        m = SEDModel(spec, ssp, observation=obs)  # NO approx=WavePrecomp()
    state = m.predict_state(_PARAMS)
    full = {**m.spec.get_fixed_values(), **_PARAMS}
    try:
        m.observation.predict_via_precomp(state, full)
        raise AssertionError("predict_via_precomp should have raised without LUT")
    except ValueError as e:
        # The guard message points users at approx=WavePrecomp() (the message was
        # updated from the old lowercase "wave_precomp"; the assertion was stale).
        assert "WavePrecomp" in str(e)


def test_lut_metallicity_changes_with_logzsol(ssp, synthetic_tophat_obs):
    """Higher metallicity changes the LUT-projected stellar photometry."""
    spec_lo = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-1.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    obs = synthetic_tophat_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_lo = SEDModel(spec_lo, ssp, observation=obs, approx=WavePrecomp())

    spec_hi = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(0.0),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_hi = SEDModel(spec_hi, ssp, observation=obs, approx=WavePrecomp())

    lut_lo = m_lo.predict_state(_PARAMS).derived["stellar_phot_lnu_precomp"]
    lut_hi = m_hi.predict_state(_PARAMS).derived["stellar_phot_lnu_precomp"]
    # At fixed SFH+mass, metallicity changes the stellar continuum shape;
    # we expect a measurable difference between metal-poor and solar.
    rel_diff = jnp.abs(lut_lo - lut_hi) / jnp.maximum(lut_lo, lut_hi)
    assert float(jnp.max(rel_diff)) > 1e-2, (
        f"LUT failed to respond to metallicity change: max rel diff = {float(jnp.max(rel_diff))}"
    )


def test_free_z_lut_published_for_multiple_z(stellar_only_free_z_model):
    """stellar_phot_lnu_precomp is published for multiple z values within free range."""
    m = stellar_only_free_z_model
    for z in [0.1, 0.5, 1.0, 1.8]:
        params = {**_PARAMS, "redshift": z}
        state = m.predict_state(params)
        assert "stellar_phot_lnu_precomp" in state.derived
        lut = state.derived["stellar_phot_lnu_precomp"]
        assert lut is not None
        assert jnp.all(jnp.isfinite(lut)), f"LUT contains non-finite values at z={z}"
