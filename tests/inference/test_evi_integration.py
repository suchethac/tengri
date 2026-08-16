# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for EVI inference pipeline.

Validates that the full EVI pipeline (build engine → optimize → sample)
runs without errors and produces reasonable results for a smooth SFH
model with synthetic SSP data.
"""

import pytest

pytestmark = pytest.mark.contract

import jax
import jax.numpy as jnp
import pytest

from tengri import Fitter, Fixed, Parameters, SEDModel, Uniform
from tengri.inference._backend_registry import _BACKENDS

jax.config.update("jax_enable_x64", True)

# ``native_vi_linear`` (MGVI) is registered tier="broken" (#1287): it segfaults
# on DPL/dense_basis photometry mocks — exactly what these tests build — so
# ``fitter.run("native_vi_linear")`` raises ``BackendError`` by default. Skip
# until the backend is repaired, mirroring the tier-aware skip added in #1324;
# do NOT force it with ``allow_unvalidated=True`` (it could crash the runner).
skip_if_broken = pytest.mark.skipif(
    _BACKENDS["native_vi_linear"].tier == "broken",
    reason="native_vi_linear is registered tier='broken' (#1287); skip until repaired",
)


# ── Fixtures ─────────────────────────────────────────────────────
# synthetic_ssp and simple_observation are provided by conftest.py (session scope)


@pytest.fixture(scope="module")
def simple_spec():
    """Simple smooth SFH spec (D=5, minimal free params)."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(1.0, 10.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Uniform(-1.5, 0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def model_and_mock(simple_spec, synthetic_ssp, simple_observation):
    """SEDModel + mock data for EVI testing."""
    model = SEDModel(simple_spec, synthetic_ssp, observation=simple_observation)
    key = jax.random.PRNGKey(42)
    params = simple_spec.sample(key)
    mock = model.mock(params, snr=20.0, key=key)
    return model, mock, params


# ── Tests ─────────────────────────────────────────────────────────


@skip_if_broken
class TestEVIRuns:
    """EVI pipeline runs without errors."""

    def test_evi_runs_and_returns_posterior(self, model_and_mock, simple_spec):
        """EVI produces a Posterior object with samples."""
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        result = fitter.run(
            "native_vi_linear",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=20,
            verbose=False,
            key=jax.random.PRNGKey(0),
        )

        assert result is not None
        assert hasattr(result, "samples")
        assert hasattr(result, "diagnostics")

        # All Uniform-prior samples must be within prior support.
        # A broken transform (wrong unbounding) pushes samples outside [low, high].
        for name in simple_spec.free_params:
            dist = simple_spec.get_distribution(name)
            if hasattr(dist, "low") and hasattr(dist, "high"):
                samples = result.samples[name]
                assert bool(jnp.all(samples >= dist.low)), (
                    f"{name}: samples below prior lower bound {dist.low}; "
                    f"min={float(jnp.min(samples)):.4f}"
                )
                assert bool(jnp.all(samples <= dist.high)), (
                    f"{name}: samples above prior upper bound {dist.high}; "
                    f"max={float(jnp.max(samples)):.4f}"
                )

    def test_evi_samples_have_correct_keys(self, model_and_mock, simple_spec):
        """Posterior samples contain all free parameter names."""
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        result = fitter.run(
            "native_vi_linear",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=20,
            verbose=False,
            key=jax.random.PRNGKey(1),
        )

        for name in simple_spec.free_params:
            assert name in result.samples, f"Missing parameter: {name}"

    def test_evi_samples_finite(self, model_and_mock, simple_spec):
        """All posterior samples are finite (no NaN/inf)."""
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        result = fitter.run(
            "native_vi_linear",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=20,
            verbose=False,
            key=jax.random.PRNGKey(2),
        )

        for name, vals in result.samples.items():
            assert jnp.all(jnp.isfinite(vals)), f"{name} has non-finite samples"

    def test_evi_with_multiseed(self, model_and_mock, simple_spec):
        """EVI with multiple seeds runs without error."""
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        result = fitter.run(
            "native_vi_linear",
            n_iterations=3,
            n_samples=2,
            n_seeds=3,
            n_posterior_samples=10,
            verbose=False,
            key=jax.random.PRNGKey(3),
        )

        assert result is not None


# TestGeoVIMGVIRouting removed: tested the `native_geovi`/`native_mgvi`
# deprecated aliases, which were hard-deleted pre-v1.0. Coverage of the
# canonical names lives in TestEVIRuns above (``native_vi_linear``) and
# in tests/inference/test_geovi_jit.py (``native_vi_nonlinear``).
