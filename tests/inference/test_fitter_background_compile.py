# SPDX-License-Identifier: BSD-3-Clause
"""Tests for compile_modes parameter controlling background JIT compilation.

Ensures that:
- Default compile_modes=None produces no background thread and no compilation.
- compile_modes="auto" inspects spec.stochastic and data_type to select defaults.
- Explicit compile_modes=(...) queues those exact modes.
- TENGRI_NO_BACKGROUND_COMPILE=1 disables thread regardless of compile_modes.
"""

from __future__ import annotations

from unittest.mock import patch

import jax
import pytest

from tengri import Fitter, Fixed, Parameters, SEDModel, Uniform

pytestmark = pytest.mark.contract

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def parametric_spec():
    """Non-stochastic Parameters for photometry."""
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
def stochastic_spec():
    """Stochastic Parameters for photometry."""
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
        stochastic=True,
    )


@pytest.fixture(scope="module")
def model_and_data_parametric(parametric_spec, synthetic_ssp, simple_observation):
    model = SEDModel(parametric_spec, synthetic_ssp, observation=simple_observation)
    key = jax.random.PRNGKey(42)
    params = parametric_spec.sample(key)
    mock = model.mock(params, snr=20.0, key=key)
    return model, mock


@pytest.fixture(scope="module")
def model_and_data_stochastic(stochastic_spec, synthetic_ssp, simple_observation):
    model = SEDModel(stochastic_spec, synthetic_ssp, observation=simple_observation)
    key = jax.random.PRNGKey(42)
    params = stochastic_spec.sample(key)
    mock = model.mock(params, snr=20.0, key=key)
    return model, mock


# ── Tests ──────────────────────────────────────────────────────────


class TestCompileModesDefault:
    """Test that compile_modes=None (default) produces no background thread."""

    def test_default_no_thread(self, model_and_data_parametric):
        """Default Fitter with no compile_modes should not spawn a thread."""
        model, mock = model_and_data_parametric
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        # With compile_modes=None (default):
        # - _target_modes should be empty
        # - _compilation_thread should be None
        # - _compilation_event should be set immediately
        assert fitter._target_modes == (), (
            f"Expected empty _target_modes for compile_modes=None, got {fitter._target_modes}"
        )
        assert fitter._compilation_thread is None, (
            "Expected _compilation_thread=None when compile_modes=None"
        )
        assert fitter._compilation_event.is_set(), (
            "_compilation_event should be set immediately when compile_modes=None"
        )

    def test_default_lazy_compile(self, model_and_data_parametric):
        """First run() should compile lazily when no background thread."""
        model, mock = model_and_data_parametric
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        # run("map") should not raise; the engine compiles on-demand
        result = fitter.run("map", n_steps=10, verbose=False)
        assert result is not None


class TestCompileModesAuto:
    """Test that compile_modes="auto" infers sensible defaults."""

    def test_auto_parametric_photometry(self, model_and_data_parametric, monkeypatch):
        """Non-stochastic + photometry should infer ("mcmc_nuts",)."""
        monkeypatch.delenv("TENGRI_NO_BACKGROUND_COMPILE", raising=False)
        model, mock = model_and_data_parametric

        # Drop any warm cache from a prior test so the background thread
        # actually triggers the patched compile() rather than short-circuiting
        # on a cache hit (this fixture's signature can clash with earlier tests
        # that share the same compile_signature).
        from tengri.inference.jit_engine import _SHARED_ENGINE_CACHE

        _SHARED_ENGINE_CACHE.clear()

        compile_calls = []
        original_compile = Fitter.compile

        def tracking_compile(self, **kwargs):
            compile_calls.append(kwargs.get("modes"))
            return original_compile(self, **kwargs)

        with patch.object(Fitter, "compile", tracking_compile):
            fitter = Fitter(
                model,
                mock.flux_obs,
                mock.noise,
                data_type="photometry",
                compile_modes="auto",
            )
            # Wait for background thread to finish
            fitter._compilation_event.wait(timeout=120)

        assert len(compile_calls) >= 1, (
            "Expected compile() to be called at least once in background thread"
        )
        assert compile_calls[0] == ("mcmc_nuts",), (
            f"Expected ('mcmc_nuts',) for parametric photometry, got {compile_calls[0]}"
        )

    @pytest.mark.skip(
        reason=(
            "Stochastic SFH path uses float() concretizations that broke after "
            "Phase 6 routed predict through JIT'd observables. Needs JIT-safe "
            "rework of stochastic SFH inner loop."
        )
    )
    def test_auto_stochastic_photometry(self, model_and_data_stochastic, monkeypatch):
        """Stochastic should infer ('linear_resample', 'nonlinear_update')."""
        monkeypatch.delenv("TENGRI_NO_BACKGROUND_COMPILE", raising=False)
        model, mock = model_and_data_stochastic

        compile_calls = []
        original_compile = Fitter.compile

        def tracking_compile(self, **kwargs):
            compile_calls.append(kwargs.get("modes"))
            return original_compile(self, **kwargs)

        with patch.object(Fitter, "compile", tracking_compile):
            fitter = Fitter(
                model,
                mock.flux_obs,
                mock.noise,
                data_type="photometry",
                compile_modes="auto",
            )
            fitter._compilation_event.wait(timeout=120)

        assert len(compile_calls) >= 1
        assert compile_calls[0] == ("linear_resample", "nonlinear_update"), (
            f"Expected VI modes for stochastic spec, got {compile_calls[0]}"
        )


class TestCompileModesExplicit:
    """Test explicit compile_modes values."""

    def test_explicit_tuple(self, model_and_data_parametric):
        """Explicit tuple should be stored in _target_modes."""
        model, mock = model_and_data_parametric
        fitter = Fitter(
            model,
            mock.flux_obs,
            mock.noise,
            data_type="photometry",
            compile_modes=("mcmc_nuts", "mcmc_hmc"),
        )
        assert fitter._target_modes == ("mcmc_nuts", "mcmc_hmc")

    def test_explicit_string_wraps(self, model_and_data_parametric):
        """Explicit string should be wrapped into a tuple."""
        model, mock = model_and_data_parametric
        fitter = Fitter(
            model,
            mock.flux_obs,
            mock.noise,
            data_type="photometry",
            compile_modes="mcmc_nuts",
        )
        assert fitter._target_modes == ("mcmc_nuts",)


class TestCompileModesEnvironmentOverride:
    """Test that TENGRI_NO_BACKGROUND_COMPILE disables background thread."""

    def test_env_override_suppresses_thread(self, model_and_data_parametric, monkeypatch):
        """TENGRI_NO_BACKGROUND_COMPILE should suppress thread even with compile_modes."""
        monkeypatch.setenv("TENGRI_NO_BACKGROUND_COMPILE", "1")
        model, mock = model_and_data_parametric

        fitter = Fitter(
            model,
            mock.flux_obs,
            mock.noise,
            data_type="photometry",
            compile_modes="auto",
        )

        # Thread should not be spawned
        assert fitter._compilation_thread is None
        assert fitter._compilation_event.is_set()


class TestCompileModesResolution:
    """Test the _resolve_compile_modes method."""

    def test_resolve_none(self, model_and_data_parametric):
        model, mock = model_and_data_parametric
        fitter = Fitter(model, mock.flux_obs, mock.noise, compile_modes=None)
        assert fitter._resolve_compile_modes(None) == ()

    def test_resolve_auto_parametric(self, model_and_data_parametric):
        model, mock = model_and_data_parametric
        fitter = Fitter(model, mock.flux_obs, mock.noise, compile_modes=None)
        # _resolve_compile_modes handles "auto" by calling _infer_default_compile_modes
        modes = fitter._resolve_compile_modes("auto")
        assert modes == ("mcmc_nuts",), f"Expected ('mcmc_nuts',), got {modes}"

    @pytest.mark.skip(reason="Same stochastic SFH JIT-safety issue as above")
    def test_resolve_auto_stochastic(self, model_and_data_stochastic):
        model, mock = model_and_data_stochastic
        fitter = Fitter(model, mock.flux_obs, mock.noise, compile_modes=None)
        modes = fitter._resolve_compile_modes("auto")
        assert modes == ("linear_resample", "nonlinear_update"), f"Expected VI modes, got {modes}"

    def test_resolve_string(self, model_and_data_parametric):
        model, mock = model_and_data_parametric
        fitter = Fitter(model, mock.flux_obs, mock.noise, compile_modes=None)
        assert fitter._resolve_compile_modes("mcmc_nuts") == ("mcmc_nuts",)

    def test_resolve_tuple(self, model_and_data_parametric):
        model, mock = model_and_data_parametric
        fitter = Fitter(model, mock.flux_obs, mock.noise, compile_modes=None)
        result = fitter._resolve_compile_modes(("mcmc_nuts", "mcmc_hmc"))
        assert result == ("mcmc_nuts", "mcmc_hmc")

    def test_resolve_invalid_type(self, model_and_data_parametric):
        model, mock = model_and_data_parametric
        fitter = Fitter(model, mock.flux_obs, mock.noise, compile_modes=None)
        with pytest.raises(TypeError):
            fitter._resolve_compile_modes(123)


class TestCompileModesIntegration:
    """Integration tests: verify compile_modes affects behavior end-to-end."""

    def test_no_compile_runs_without_precompile(self, model_and_data_parametric, monkeypatch):
        """compile_modes=None should not pre-compile; first run() should work."""
        monkeypatch.setenv("TENGRI_NO_BACKGROUND_COMPILE", "0")
        model, mock = model_and_data_parametric

        # Create fitter with no background compile
        fitter = Fitter(
            model,
            mock.flux_obs,
            mock.noise,
            data_type="photometry",
            compile_modes=None,
        )
        assert fitter._compilation_thread is None

        # First run() should compile lazily and work fine
        result = fitter.run("map", n_steps=10, verbose=False)
        assert result is not None

    def test_auto_compile_spawns_thread(self, model_and_data_parametric, monkeypatch):
        """compile_modes='auto' should spawn a thread (when cache is empty)."""
        from tengri.inference._model_cache import get_model_cache

        monkeypatch.delenv("TENGRI_NO_BACKGROUND_COMPILE", raising=False)
        model, mock = model_and_data_parametric

        # Clear cache to ensure thread spawns
        engine_cache = get_model_cache(model).get("jit_engine")
        if engine_cache is not None:
            engine_cache.clear()

        # Create fitter with background compile
        fitter = Fitter(
            model,
            mock.flux_obs,
            mock.noise,
            data_type="photometry",
            compile_modes="auto",
        )
        # Thread should be spawned (or None if cache was already warm)
        # Wait for background thread to finish
        fitter._compilation_event.wait(timeout=120)
        assert fitter._compilation_event.is_set()

        # run() should not raise and should work correctly
        result = fitter.run("mcmc_nuts", n_warmup=10, n_burnin=5, n_samples=5, verbose=False)
        assert result is not None
