# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for background JIT compilation in Fitter.__init__.

Ensures that:
- The compilation thread is a daemon thread.
- _compilation_event is set after the thread finishes.
- Two Fitters sharing the same SEDModel + cache key do NOT double-compile.
- A compilation error is propagated to run() rather than silently dropped.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import jax
import pytest

from tengri import Fitter, Fixed, Parameters, SEDModel, Uniform

pytestmark = pytest.mark.contract

jax.config.update("jax_enable_x64", True)


# ── Fixtures ─────────────────────────────────────────────────────
# synthetic_ssp and simple_observation are provided by conftest.py (session scope)


@pytest.fixture(scope="module")
def smooth_spec():
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
def model_and_data(smooth_spec, synthetic_ssp, simple_observation):
    model = SEDModel(smooth_spec, synthetic_ssp, observation=simple_observation)
    key = jax.random.PRNGKey(42)
    params = smooth_spec.sample(key)
    mock = model.mock(params, snr=20.0, key=key)
    return model, mock


# ── Tests ─────────────────────────────────────────────────────────


class TestBackgroundCompilation:
    @pytest.fixture(autouse=True)
    def enable_background_compilation(self, monkeypatch):
        """Clear the test-suite suppression flag so the thread actually spawns."""
        monkeypatch.delenv("TENGRI_NO_BACKGROUND_COMPILE", raising=False)

    def test_compilation_event_is_set(self, model_and_data):
        """_compilation_event must be set (thread finished) before we call run."""
        model, mock = model_and_data
        fitter = Fitter(
            model, mock.flux_obs, mock.noise, data_type="photometry", compile_modes="auto"
        )
        # Wait up to 300s — cold-start compilation can take ~60s for two modes;
        # on a warm XLA cache this returns in milliseconds.
        assert fitter._compilation_event.wait(timeout=300), (
            "_compilation_event was never set — background thread may have deadlocked"
        )

    def test_thread_is_daemon(self, model_and_data, monkeypatch):
        """The spawned thread must be a daemon thread."""
        model, mock = model_and_data
        spawned: list[threading.Thread] = []
        original_thread = threading.Thread

        def capturing_thread(*args, **kwargs):
            t = original_thread(*args, **kwargs)
            spawned.append(t)
            return t

        monkeypatch.setattr(threading, "Thread", capturing_thread)
        fitter = Fitter(
            model, mock.flux_obs, mock.noise, data_type="photometry", compile_modes="auto"
        )
        fitter._compilation_event.wait(timeout=120)

        assert spawned, "No thread was created"
        assert spawned[0].daemon, "Background compilation thread must be daemon=True"

    def test_no_double_compile(self, model_and_data):
        """Two Fitters with the same SEDModel + cache key compile only once."""
        model, mock = model_and_data
        # Ensure the cache is populated by the first Fitter.
        fitter1 = Fitter(
            model, mock.flux_obs, mock.noise, data_type="photometry", compile_modes="auto"
        )
        fitter1._compilation_event.wait(timeout=120)

        compile_calls: list[int] = []
        original_compile = Fitter.compile

        def counting_compile(self, **kwargs):
            compile_calls.append(1)
            return original_compile(self, **kwargs)

        with patch.object(Fitter, "compile", counting_compile):
            fitter2 = Fitter(
                model, mock.flux_obs, mock.noise, data_type="photometry", compile_modes="auto"
            )
            fitter2._compilation_event.wait(timeout=120)

        # compile() should NOT have been called again — cache was already warm.
        assert len(compile_calls) == 0, (
            f"compile() was called {len(compile_calls)} time(s) for a second Fitter "
            "with the same cache key — double-compile detected"
        )

    def test_compilation_error_propagated(self, model_and_data):
        """A compile() exception must surface as RuntimeError when run() is called."""
        from tengri.inference._model_cache import get_model_cache

        model, mock = model_and_data

        def failing_compile(self, **kwargs):
            raise ValueError("Simulated XLA OOM")

        with patch.object(Fitter, "compile", failing_compile):
            # Clear the engine cache so the thread actually tries to compile.
            # Uses get_model_cache (the current cache location) rather than the
            # legacy model._jit_engine_cache attribute.
            engine_cache = get_model_cache(model).get("jit_engine")
            if engine_cache is not None:
                engine_cache.clear()

            # Also clear the module-level shared cache used by the background
            # worker — see fitter._start_background_compilation. Without this
            # the worker short-circuits on a warm signature from prior tests
            # in the same process and never sees the patched compile().
            from tengri.inference.jit_engine import _SHARED_ENGINE_CACHE

            _SHARED_ENGINE_CACHE.clear()

            fitter = Fitter(
                model, mock.flux_obs, mock.noise, data_type="photometry", compile_modes="auto"
            )
            fitter._compilation_event.wait(timeout=30)

            assert fitter._compilation_error is not None, (
                "Expected _compilation_error to be set after a failing compile()"
            )
            assert isinstance(fitter._compilation_error, ValueError)

            dummy_pos = fitter._initialize_unbounded(jax.random.PRNGKey(0))
            with pytest.raises(RuntimeError, match="Background JIT compilation failed"):
                fitter._get_or_build_engine(dummy_pos)
