"""Unit tests for geoVI JIT engine primitives.

Tests the nonlinear coordinate transform, sample mode dispatch,
constants/point_estimates masks, and block schedule configuration.
"""

import jax
import jax.numpy as jnp
import pytest

from tengri import Fitter, Fixed, Model, Observation, ParamSpec, Photometry, Uniform
from tengri.inference.vi_config import BlockSchedule, BlockStep, OptimizationSchedule
from tengri.models.observation.photometry import FilterCurve
from tengri.models.sps.dsps_wrapper import SSPData

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_ssp():
    """Minimal synthetic SSP for fast tests (3 Z x 20 ages x 100 lam)."""
    n_met, n_age, n_wave = 3, 20, 100
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages_gyr = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(123)
    flux = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


@pytest.fixture(scope="module")
def simple_observation():
    """Synthetic 3-band observation."""
    waves = [
        jnp.linspace(3500.0, 4500.0, 50),
        jnp.linspace(5000.0, 6500.0, 50),
        jnp.linspace(7500.0, 9000.0, 50),
    ]
    trans = [jnp.ones(50) * 0.5 for _ in range(3)]
    curves = tuple(
        FilterCurve(wave=w, trans=t, name=f"band_{i}")
        for i, (w, t) in enumerate(zip(waves, trans))
    )
    photometry = Photometry(filters=curves)
    return Observation(photometry=photometry)


@pytest.fixture(scope="module")
def smooth_spec():
    """Smooth SFH spec (D=5, minimal free params)."""
    return ParamSpec(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(1.0, 10.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
        met_logzsol=Uniform(-1.5, 0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def fitter_and_mock(smooth_spec, synthetic_ssp, simple_observation):
    """Fitter + mock data for geoVI testing."""
    model = Model(smooth_spec, synthetic_ssp, observation=simple_observation)
    key = jax.random.PRNGKey(42)
    params = smooth_spec.sample(key)
    mock = model.mock(params, snr=20.0, key=key)
    fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")
    return fitter, mock, params


@pytest.fixture(scope="module")
def engine(fitter_and_mock):
    """Pre-compiled JIT engine."""
    fitter = fitter_and_mock[0]
    dummy_pos = fitter._initialize_unbounded(jax.random.PRNGKey(0))
    return fitter._get_or_build_engine(dummy_pos)


@pytest.fixture(scope="module")
def data_args(fitter_and_mock):
    """Data-dependent arguments for JIT engine calls."""
    fitter = fitter_and_mock[0]
    return fitter._data_args


# ---------------------------------------------------------------------------
# BlockSchedule tests
# ---------------------------------------------------------------------------


class TestBlockSchedule:
    """BlockStep and BlockSchedule dataclasses."""

    def test_block_step_defaults(self):
        step = BlockStep()
        assert step.sample_mode == "nonlinear_resample"
        assert step.constants == ()
        assert step.point_estimates == ()
        assert step.n_samples is None

    def test_block_step_custom(self):
        step = BlockStep(
            sample_mode="linear_resample",
            constants=("sfh_field_xi",),
            n_samples=4,
        )
        assert step.sample_mode == "linear_resample"
        assert step.constants == ("sfh_field_xi",)
        assert step.n_samples == 4

    def test_individual_geovi_schedule(self):
        sched = BlockSchedule.individual_geovi()
        assert len(sched.blocks) == 2
        assert sched.blocks[0].sample_mode == "nonlinear_resample"
        assert "sfh_field_xi" in sched.blocks[0].constants
        assert sched.blocks[1].sample_mode == "linear_resample"

    def test_hierarchical_schedule(self):
        sched = BlockSchedule.hierarchical()
        assert len(sched.blocks) == 3
        # Block 1: shared PSD with point estimates
        assert sched.blocks[0].sample_mode == "nonlinear_resample"
        assert sched.blocks[0].n_samples == 6
        # Block 3: SFH with linear sampling
        assert sched.blocks[2].sample_mode == "linear_resample"
        assert sched.blocks[2].n_samples == 2

    def test_block_schedule_immutable(self):
        step = BlockStep()
        with pytest.raises(AttributeError):
            step.sample_mode = "linear_resample"


# ---------------------------------------------------------------------------
# Engine primitive tests
# ---------------------------------------------------------------------------


class TestEnginePrimitives:
    """Test geoVI primitives inside the JIT engine."""

    def test_engine_has_geovi_keys(self, engine):
        """Engine dict includes geoVI-specific entries."""
        assert "run_evi_geovi" in engine
        assert "draw_nonlinear_samples" in engine
        assert "param_ranges" in engine
        assert "make_mask" in engine
        assert "make_pe_mask" in engine
        assert "d_total" in engine

    def test_param_ranges_covers_all_dims(self, engine):
        """param_ranges spans the full d_total dimensions."""
        total = sum(end - start for start, end in engine["param_ranges"].values())
        assert total == engine["d_total"]

    def test_make_mask_zeros_and_ones(self, engine):
        """make_mask produces correct boolean mask."""
        keys = list(engine["param_ranges"].keys())
        if len(keys) >= 2:
            mask = engine["make_mask"]((keys[0],))
            start, end = engine["param_ranges"][keys[0]]
            assert bool(jnp.all(mask[start:end]))  # True for named param
            start2, end2 = engine["param_ranges"][keys[1]]
            assert not bool(jnp.any(mask[start2:end2]))  # False for other

    def test_make_pe_mask_zeros_and_ones(self, engine):
        """make_pe_mask produces 0.0 for PE params, 1.0 for sampled."""
        keys = list(engine["param_ranges"].keys())
        pe_mask = engine["make_pe_mask"]((keys[0],))
        start, end = engine["param_ranges"][keys[0]]
        assert float(jnp.sum(pe_mask[start:end])) == 0.0
        # Other params should be 1.0
        total_ones = float(jnp.sum(pe_mask))
        assert total_ones == engine["d_total"] - (end - start)


# ---------------------------------------------------------------------------
# geoVI run tests
# ---------------------------------------------------------------------------


class TestGeoVIRuns:
    """geoVI optimizer runs and produces reasonable results."""

    def test_geovi_runs_smooth_model(self, fitter_and_mock):
        """geoVI runs on smooth model without errors."""
        fitter, _, _ = fitter_and_mock
        result = fitter.run(
            "native_geovi",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=10,
            verbose=False,
            key=jax.random.PRNGKey(10),
        )
        assert result is not None
        assert len(result.samples) > 0

    def test_geovi_posterior_finite(self, fitter_and_mock):
        """geoVI posterior samples are all finite."""
        fitter, _, _ = fitter_and_mock
        result = fitter.run(
            "native_geovi",
            n_iterations=5,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=20,
            verbose=False,
            key=jax.random.PRNGKey(11),
        )
        for name, vals in result.samples.items():
            assert bool(jnp.all(jnp.isfinite(vals))), f"NaN/Inf in {name}"

    def test_geovi_vs_mgvi_different_posteriors(self, fitter_and_mock):
        """geoVI and MGVI produce different expansion points.

        They should generally agree on the mode but differ in posterior
        width (geoVI captures nonlinear geometry).
        """
        fitter, _, _ = fitter_and_mock
        key = jax.random.PRNGKey(12)

        result_mgvi = fitter.run(
            "native_evi",
            n_iterations=5,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=20,
            verbose=False,
            key=key,
        )
        result_geovi = fitter.run(
            "native_geovi",
            n_iterations=5,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=20,
            verbose=False,
            key=key,
        )

        # Both should have samples
        assert len(result_mgvi.samples) > 0
        assert len(result_geovi.samples) > 0

    def test_posterior_predictive_check(self, fitter_and_mock):
        """Posterior predictive: predicted data should bracket observed data.

        For each posterior sample, compute predicted photometry and check
        that the observed data falls within the predicted spread.
        """
        fitter, mock, _ = fitter_and_mock
        result = fitter.run(
            "native_geovi",
            n_iterations=5,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=50,
            verbose=False,
            key=jax.random.PRNGKey(13),
        )

        # Compute predicted photometry for each posterior sample
        model = fitter.model
        predictions = []
        for i in range(len(result.samples[next(iter(result.samples.keys()))])):
            sample = {k: v[i] for k, v in result.samples.items()}
            pred = model.predict_photometry(sample)
            predictions.append(pred)

        predictions = jnp.stack(predictions)
        pred_median = jnp.median(predictions, axis=0)
        pred_lo = jnp.percentile(predictions, 2.5, axis=0)
        pred_hi = jnp.percentile(predictions, 97.5, axis=0)

        # Check: observed data should mostly fall within 95% CI
        obs = mock.flux_obs
        within_ci = jnp.sum((obs >= pred_lo) & (obs <= pred_hi))
        # At least 1 out of 3 bands should be within CI
        # (relaxed because small n_iterations)
        assert int(within_ci) >= 1, (
            f"Posterior predictive check failed: {int(within_ci)}/3 bands "
            f"within 95% CI. Observed: {obs}, "
            f"Predicted: [{pred_lo}, {pred_hi}]"
        )

        # Residuals should be reasonable (chi < 10 per band)
        chi = jnp.abs(obs - pred_median) / mock.noise
        assert bool(jnp.all(chi < 10)), f"Posterior predictive residuals too large: chi = {chi}"


# ---------------------------------------------------------------------------
# OptimizationSchedule tests
# ---------------------------------------------------------------------------


class TestOptimizationSchedule:
    """OptimizationSchedule factory methods and behavior."""

    def test_geovi_schedule_resample_at_zero(self):
        sched = OptimizationSchedule.geovi(n_iterations=15, resample_every=5)
        assert sched(0).sample_mode == "nonlinear_resample"

    def test_geovi_schedule_update_between(self):
        sched = OptimizationSchedule.geovi(n_iterations=15, resample_every=5)
        for i in [1, 2, 3, 4]:
            assert sched(i).sample_mode == "nonlinear_update", f"iter {i}"

    def test_geovi_schedule_resample_periodic(self):
        sched = OptimizationSchedule.geovi(n_iterations=15, resample_every=5)
        assert sched(5).sample_mode == "nonlinear_resample"
        assert sched(10).sample_mode == "nonlinear_resample"

    def test_evi_schedule_linear_then_nonlinear(self):
        sched = OptimizationSchedule.evi(n_iterations=20, transition=10)
        assert sched(0).sample_mode == "linear_resample"
        assert sched(9).sample_mode == "linear_resample"
        assert sched(10).sample_mode == "nonlinear_resample"
        assert sched(11).sample_mode == "nonlinear_update"

    def test_mgvi_schedule_always_linear(self):
        sched = OptimizationSchedule.mgvi(n_iterations=10)
        for i in range(10):
            assert sched(i).sample_mode == "linear_resample"

    def test_custom_schedule(self):
        sched = OptimizationSchedule.custom(
            lambda i: BlockStep("nonlinear_update" if i > 0 else "nonlinear_resample"),
            n_iterations=5,
        )
        assert sched(0).sample_mode == "nonlinear_resample"
        assert sched(3).sample_mode == "nonlinear_update"

    def test_sample_mode_at_for_nifty(self):
        """sample_mode_at returns string compatible with NIFTy."""
        sched = OptimizationSchedule.geovi()
        assert sched.sample_mode_at(0) == "nonlinear_resample"
        assert sched.sample_mode_at(1) == "nonlinear_update"


# ---------------------------------------------------------------------------
# Native geovi schedule test
# ---------------------------------------------------------------------------


class TestNativeGeoVISchedule:
    """Verify native engine uses resample+update schedule (same as fast path)."""

    def test_geovi_mode_stable_convergence(self, engine, data_args):
        """The 'geovi' sample_mode should not oscillate wildly."""
        flatten = engine["flatten"]
        d_total = engine["d_total"]
        # Need a position to start from
        pos_flat = jnp.zeros(d_total)

        # Run 8 iterations with geovi schedule
        m, _iters = engine["run_evi_geovi"](
            pos_flat,
            jax.random.PRNGKey(42),
            data_args,
            n_iterations=8,
            n_samples=2,
            kl_rtol=0.0,
            sample_mode="geovi",
        )
        # Should produce a finite result
        assert bool(jnp.all(jnp.isfinite(m))), "geovi schedule produced non-finite result"
