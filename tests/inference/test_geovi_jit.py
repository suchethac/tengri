# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for geoVI JIT engine primitives.

Tests the nonlinear coordinate transform, sample mode dispatch,
constants/point_estimates masks, and block schedule configuration.
"""

import pytest

pytestmark = pytest.mark.contract

import jax
import jax.numpy as jnp
import pytest

from tengri import Fitter, Fixed, Parameters, SEDModel, Uniform
from tengri.inference._backend_registry import _BACKENDS

# ``native_vi_nonlinear`` (geoVI) and ``native_vi_linear`` (MGVI) are registered
# tier="broken" (#1287): they segfault on DPL/dense_basis photometry mocks, so
# ``fitter.run(...)`` on them raises ``BackendError`` by default. Only the
# ``TestGeoVIRuns`` class calls ``fitter.run`` on these; the schedule and direct
# engine-primitive classes below never hit the broken-tier guard and keep
# running. Mirrors the tier-aware skip added in #1324.
skip_if_broken = pytest.mark.skipif(
    _BACKENDS["native_vi_nonlinear"].tier == "broken",
    reason="native_vi_nonlinear is registered tier='broken' (#1287); skip until repaired",
)


# ── Fixtures ──────────────────────────────────────────────────────
# synthetic_ssp and simple_observation are provided by conftest.py (session scope)


@pytest.fixture(scope="module")
def smooth_spec():
    """Smooth SFH spec (D=5, minimal free params)."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(1.0, 10.0),
        sfh_dpl_log_total_mass=Uniform(-1.0, 2.0),
        met_logzsol=Uniform(-1.5, 0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def fitter_and_mock(smooth_spec, synthetic_ssp, simple_observation):
    """Fitter + mock data for geoVI testing."""
    model = SEDModel(smooth_spec, synthetic_ssp, observation=simple_observation)
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


# ── Engine primitive tests ────────────────────────────────────────


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


# ── geoVI run tests ───────────────────────────────────────────────


@skip_if_broken
class TestGeoVIRuns:
    """geoVI optimizer runs and produces reasonable results."""

    def test_geovi_runs_smooth_model(self, fitter_and_mock):
        """geoVI runs on smooth model without errors."""
        fitter, _, _ = fitter_and_mock
        result = fitter.run(
            "native_vi_nonlinear",
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
            "native_vi_nonlinear",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=10,
            verbose=False,
            key=jax.random.PRNGKey(11),
        )
        for name, vals in result.samples.items():
            assert bool(jnp.all(jnp.isfinite(vals))), f"NaN/Inf in {name}"

    def test_geovi_vs_mgvi_different_posteriors(self, fitter_and_mock):
        """geoVI and MGVI produce non-identical posterior widths.

        geoVI uses a nonlinear coordinate transform; MGVI linearizes.
        Even with few iterations they must not produce bit-identical samples —
        if they do, the routing logic has collapsed to a single code path.

        We compare per-parameter posterior stds.  They need not agree within
        any particular tolerance (convergence is not expected at n_iterations=3),
        but they must differ, confirming two distinct algorithms ran.
        """
        fitter, _, _ = fitter_and_mock
        key = jax.random.PRNGKey(12)

        result_mgvi = fitter.run(
            "native_vi_linear",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=10,
            verbose=False,
            key=key,
        )
        result_geovi = fitter.run(
            "native_vi_nonlinear",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=10,
            verbose=False,
            key=key,
        )

        # Both must have samples
        assert len(result_mgvi.samples) > 0
        assert len(result_geovi.samples) > 0

        # Posterior widths must not be bit-identical — if they are, both ran
        # the same algorithm (routing broken) or one path was silently skipped.
        shared = [p for p in result_geovi.samples if p in result_mgvi.samples]
        geovi_stds = jnp.array([jnp.std(result_geovi.samples[p]) for p in shared])
        mgvi_stds = jnp.array([jnp.std(result_mgvi.samples[p]) for p in shared])
        assert not jnp.allclose(geovi_stds, mgvi_stds, rtol=1e-8), (
            "geoVI and MGVI posterior widths are bit-identical — "
            "method routing may have collapsed to a single code path."
        )

    def test_posterior_predictive_check(self, fitter_and_mock):
        """Posterior predictive: predicted data should bracket observed data.

        For each posterior sample, compute predicted photometry and check
        that the observed data falls within the predicted spread.
        """
        fitter, mock, _ = fitter_and_mock
        result = fitter.run(
            "native_vi_nonlinear",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=20,
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


# ── Native geovi schedule test ────────────────────────────────────


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
