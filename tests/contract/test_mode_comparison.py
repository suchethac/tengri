"""Mode comparison: compositional vs hybrid vs exact.

Tests that:
1. All three prediction modes produce finite, positive photometry.
2. Compositional is bit-identical to exact (0% error).
3. Hybrid agrees with exact to within 0.5% (SDSS worst-case).
4. auto-mode resolves to compositional (not exact) when possible.
5. Relative speed ordering: hybrid >= compositional >> exact.

These are correctness tests, not strict performance gates.
Timing results are printed as diagnostics for manual inspection.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

jax.config.update("jax_enable_x64", True)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILES = sorted(_DATA_DIR.glob("ssp_*.h5"))
_SSP_FILE = _SSP_FILES[0] if _SSP_FILES else None
_SSP_EXISTS = _SSP_FILE is not None and _SSP_FILE.is_file()
_needs_ssp = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")


def _make_model(sfh="dpl", agn=None, redshift=0.1):
    import tengri

    return tengri.SEDModel.from_config(
        ssp=str(_SSP_FILE),
        sfh=sfh,
        agn=agn,
        filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
        redshift=redshift,
    )


def _warmup_and_time(fn, n_warmup=5, n_bench=10):
    """Return median wall-clock µs after JIT warmup."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(times))


@_needs_ssp
class TestModeRouting:
    """Verify auto-mode selects the fastest available kernel."""

    @pytest.fixture(scope="class")
    def model(self):
        return _make_model()

    def test_auto_uses_compositional_not_exact(self, model):
        """auto-mode should route to compositional when filters + SSP are present."""
        assert model._compositional.photometry is not None, (
            "Compositional kernel not built — auto-mode will fall back to exact"
        )

    def test_hybrid_kernel_also_built(self, model):
        """Hybrid kernel should be available as a fallback."""
        assert model._hybrid.photometry is not None, "Hybrid kernel not built"

    def test_explicit_compositional_mode_runs(self, model):
        key = jax.random.PRNGKey(0)
        params = model.spec.sample(key)
        phot = model.predict_photometry(params, mode="compositional")
        assert jnp.all(jnp.isfinite(phot))
        assert jnp.all(phot > 0)

    def test_explicit_hybrid_mode_runs(self, model):
        key = jax.random.PRNGKey(0)
        params = model.spec.sample(key)
        phot = model.predict_photometry(params, mode="hybrid")
        assert jnp.all(jnp.isfinite(phot))
        assert jnp.all(phot > 0)

    def test_explicit_exact_mode_runs(self, model):
        key = jax.random.PRNGKey(0)
        params = model.spec.sample(key)
        phot = model.predict_photometry(params, mode="exact")
        assert jnp.all(jnp.isfinite(phot))
        assert jnp.all(phot > 0)


@_needs_ssp
class TestModeNumericalAgreement:
    """Compositional = bit-identical to exact; hybrid = within 0.5%."""

    @pytest.fixture(scope="class")
    def model_and_params(self):
        model = _make_model()
        key = jax.random.PRNGKey(7)
        params = model.spec.sample(key)
        return model, params

    def test_compositional_matches_exact(self, model_and_params):
        model, params = model_and_params
        comp = model.predict_photometry(params, mode="compositional")
        exact = model.predict_photometry(params, mode="exact")
        # Bit-identical: relative difference should be at floating-point noise level
        rel_diff = jnp.max(jnp.abs(comp - exact) / (jnp.abs(exact) + 1e-40))
        assert float(rel_diff) < 1e-10, (
            f"Compositional vs exact max relative diff = {float(rel_diff):.2e} (expected < 1e-10)"
        )

    def test_hybrid_agrees_with_exact(self, model_and_params):
        model, params = model_and_params
        hybrid = model.predict_photometry(params, mode="hybrid")
        exact = model.predict_photometry(params, mode="exact")
        rel_diff = jnp.max(jnp.abs(hybrid - exact) / (jnp.abs(exact) + 1e-40))
        # 0.5% tolerance — SDSS g-band worst-case dust factorization error
        assert float(rel_diff) < 0.005, (
            f"Hybrid vs exact max relative diff = {float(rel_diff):.4%} (expected < 0.5%)"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_compositional_exact_agreement_multiple_params(self, model_and_params, seed):
        """Compositional-exact agreement holds across a range of prior samples."""
        model, _ = model_and_params
        params = model.spec.sample(jax.random.PRNGKey(seed))
        comp = model.predict_photometry(params, mode="compositional")
        exact = model.predict_photometry(params, mode="exact")
        rel_diff = jnp.max(jnp.abs(comp - exact) / (jnp.abs(exact) + 1e-40))
        assert float(rel_diff) < 1e-10, (
            f"seed={seed}: compositional vs exact rel diff = {float(rel_diff):.2e}"
        )


@_needs_ssp
class TestModeSpeedOrdering:
    """Hybrid >= compositional >> exact in wall-clock time.

    Not a strict gate — prints timing for manual inspection.
    Only asserts the ordering, not absolute thresholds.
    """

    @pytest.fixture(scope="class")
    def model_warmed(self):
        model = _make_model()
        key = jax.random.PRNGKey(0)
        params = model.spec.sample(key)
        # Warm all three paths
        for mode in ("compositional", "hybrid", "exact"):
            for _ in range(3):
                jax.block_until_ready(model.predict_photometry(params, mode=mode))
        return model, params

    def test_speed_ordering_compositional_faster_than_exact(self, model_warmed, capsys):
        model, params = model_warmed

        t_comp = _warmup_and_time(lambda: model.predict_photometry(params, mode="compositional"))
        t_exact = _warmup_and_time(lambda: model.predict_photometry(params, mode="exact"))

        with capsys.disabled():
            print(f"\n  compositional: {t_comp:.0f} µs")
            print(f"  exact:         {t_exact:.0f} µs")
            print(f"  speedup:       {t_exact / t_comp:.1f}×")

        assert t_comp < t_exact, (
            f"Compositional ({t_comp:.0f} µs) should be faster than exact ({t_exact:.0f} µs)"
        )

    def test_speed_ordering_hybrid_faster_than_exact(self, model_warmed, capsys):
        model, params = model_warmed

        t_hybrid = _warmup_and_time(lambda: model.predict_photometry(params, mode="hybrid"))
        t_exact = _warmup_and_time(lambda: model.predict_photometry(params, mode="exact"))

        with capsys.disabled():
            print(f"\n  hybrid:  {t_hybrid:.0f} µs")
            print(f"  exact:   {t_exact:.0f} µs")
            print(f"  speedup: {t_exact / t_hybrid:.1f}×")

        assert t_hybrid < t_exact, (
            f"Hybrid ({t_hybrid:.0f} µs) should be faster than exact ({t_exact:.0f} µs)"
        )

    def test_auto_mode_matches_compositional_output(self, model_warmed):
        """auto-mode should return the same result as explicit compositional."""
        model, params = model_warmed
        auto = model.predict_photometry(params, mode="auto")
        comp = model.predict_photometry(params, mode="compositional")
        assert jnp.allclose(auto, comp, atol=0.0), "auto-mode did not route to compositional"


@_needs_ssp
class TestModeWithStochasticSFH:
    """Modes should all work correctly with the stochastic field SFH."""

    @pytest.fixture(scope="class")
    def stochastic_model(self):
        return _make_model(sfh="dpl+field")

    def test_all_modes_finite_stochastic(self, stochastic_model):
        key = jax.random.PRNGKey(0)
        params = stochastic_model.spec.sample(key)
        for mode in ("compositional", "hybrid", "exact"):
            phot = stochastic_model.predict_photometry(params, mode=mode)
            assert jnp.all(jnp.isfinite(phot)), f"NaN/Inf in mode={mode} with stochastic SFH"

    def test_compositional_exact_agreement_stochastic(self, stochastic_model):
        key = jax.random.PRNGKey(3)
        params = stochastic_model.spec.sample(key)
        comp = stochastic_model.predict_photometry(params, mode="compositional")
        exact = stochastic_model.predict_photometry(params, mode="exact")
        rel_diff = jnp.max(jnp.abs(comp - exact) / (jnp.abs(exact) + 1e-40))
        assert float(rel_diff) < 1e-10, (
            f"Stochastic SFH: compositional vs exact rel diff = {float(rel_diff):.2e}"
        )
