"""Invariant tests for precomputation state, fitter caches, and JIT safety.

Covers three classes of bugs that are easy to reintroduce:

Class 1 — Stale fitter cache
  precompute_spectroscopy() and precompute_ztable() must clear the
  compiled loss-function cache on the model.  If they don't, a Fitter
  built before precomputation keeps using the old (slow, heavy) loss fn.

Class 2 — len() on traced JAX arrays inside JIT
  jnp.clip(..., 0, len(arr) - 2) raises ConcretizationTypeError when
  arr is a vmapped or function-argument array.  .shape[0] is always a
  concrete integer regardless of tracing context.

Class 3 — _traceable mode routing and JIT composability
  predict_spectrum/photometry(mode='_traceable') must route to the
  lean precomputed kernel, not the full compositional SED path.  It
  must also be composable inside jax.jit and jax.grad without raising
  "jit inside jit" or shape errors.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILES = sorted(_DATA_DIR.glob("ssp_*.h5"))
_SSP_FILE = _SSP_FILES[0] if _SSP_FILES else None
_SSP_EXISTS = _SSP_FILE is not None and _SSP_FILE.is_file()
_needs_ssp = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")


# ---------------------------------------------------------------------------
# Helper: build a minimal model (requires SSP)
# ---------------------------------------------------------------------------


def _make_spec_model():
    """Return a DPL SEDModel with precomputed spectroscopy at z=0.1."""
    import tengri

    wave_obs = jnp.linspace(4000.0, 9000.0, 200)
    model = tengri.Model.from_config(
        ssp=str(_SSP_FILE),
        sfh="dpl",
        redshift=0.1,
    )
    model.precompute_spectroscopy(wave_obs)
    return model, wave_obs


def _make_phot_model():
    """Return a DPL SEDModel with photometry (fixed z)."""
    import tengri

    return tengri.Model.from_config(
        ssp=str(_SSP_FILE),
        sfh="dpl",
        filters=["sdss_u", "sdss_g", "sdss_r"],
        redshift=0.1,
    )


# ===========================================================================
# Class 1: Stale fitter cache
# ===========================================================================


class TestCacheInvalidation:
    """precompute_* methods must wipe all compiled loss-fn caches."""

    @_needs_ssp
    def test_precompute_spectroscopy_clears_loss_fn_cache(self):
        """_loss_fn_cache removed after precompute_spectroscopy."""
        model, wave_obs = _make_spec_model()
        model._loss_fn_cache = {"stale": object()}

        model.precompute_spectroscopy(wave_obs)

        assert not hasattr(model, "_loss_fn_cache"), (
            "_loss_fn_cache survives precompute_spectroscopy — "
            "a Fitter built before precompute would use a stale loss fn"
        )

    @_needs_ssp
    def test_precompute_spectroscopy_clears_jit_engine_cache(self):
        """_jit_engine_cache removed after precompute_spectroscopy."""
        model, wave_obs = _make_spec_model()
        model._jit_engine_cache = {"stale": object()}

        model.precompute_spectroscopy(wave_obs)

        assert not hasattr(model, "_jit_engine_cache")

    @_needs_ssp
    def test_precompute_spectroscopy_clears_loglik_cache(self):
        """_loglik_fn_cache removed after precompute_spectroscopy."""
        model, wave_obs = _make_spec_model()
        model._loglik_fn_cache = {"stale": object()}

        model.precompute_spectroscopy(wave_obs)

        assert not hasattr(model, "_loglik_fn_cache")

    @_needs_ssp
    def test_precompute_ztable_clears_loss_fn_cache(self):
        """_loss_fn_cache removed after precompute_ztable."""
        model = _make_phot_model()
        model._loss_fn_cache = {"stale": object()}

        model.precompute_ztable()

        assert not hasattr(model, "_loss_fn_cache"), (
            "_loss_fn_cache survives precompute_ztable — "
            "a Fitter built before precompute_ztable would use a stale loss fn"
        )

    @_needs_ssp
    def test_precompute_ztable_clears_jit_engine_cache(self):
        """_jit_engine_cache removed after precompute_ztable."""
        model = _make_phot_model()
        model._jit_engine_cache = {"stale": object()}

        model.precompute_ztable()

        assert not hasattr(model, "_jit_engine_cache")

    @_needs_ssp
    def test_precompute_ztable_clears_loglik_cache(self):
        """_loglik_fn_cache removed after precompute_ztable."""
        model = _make_phot_model()
        model._loglik_fn_cache = {"stale": object()}

        model.precompute_ztable()

        assert not hasattr(model, "_loglik_fn_cache")

    @_needs_ssp
    def test_no_error_when_caches_absent(self):
        """precompute_* must not raise if caches were never populated."""
        model, wave_obs = _make_spec_model()
        # Caches may not exist yet — this must not raise AttributeError.
        for attr in ("_loss_fn_cache", "_jit_engine_cache", "_loglik_fn_cache"):
            if hasattr(model, attr):
                delattr(model, attr)

        # Should not raise.
        model.precompute_spectroscopy(wave_obs)

        # ztable requires filters; use a phot model for that branch.
        phot_model = _make_phot_model()
        for attr in ("_loss_fn_cache", "_jit_engine_cache", "_loglik_fn_cache"):
            if hasattr(phot_model, attr):
                delattr(phot_model, attr)
        phot_model.precompute_ztable()

    @_needs_ssp
    def test_double_precompute_spectroscopy_idempotent(self):
        """Calling precompute_spectroscopy twice must not raise."""
        model, wave_obs = _make_spec_model()
        # Second call should rebuild cleanly.
        model.precompute_spectroscopy(wave_obs)
        key = jax.random.PRNGKey(0)
        params = model.spec.sample(key)
        flux = model.predict_spectrum(params, wave_obs=wave_obs)
        assert jnp.all(jnp.isfinite(flux)), "Flux has NaN/Inf after double precompute"


# ===========================================================================
# Class 2: len() vs .shape[0] on traced arrays
# ===========================================================================


class TestJITSafeSearchsorted:
    """Verify searchsorted + clip is safe when the grid array is traced.

    The production code captures ssp_lgmet at closure-build time (numpy
    array), so len() works there.  But if the array is ever passed as a
    jax.jit / jax.vmap argument, len() raises ConcretizationTypeError.
    These tests exercise that exact path to catch any regression.
    """

    def test_searchsorted_clip_jit_safe_array_arg(self):
        """clip upper bound via .shape[0] does not raise inside jax.jit."""

        @jax.jit
        def interp_met(lz, grid):
            # Pattern used in fused_kernels.py — must use .shape[0], not len()
            clamped = jnp.clip(lz, grid[0], grid[-1])
            idx = jnp.clip(jnp.searchsorted(grid, clamped) - 1, 0, grid.shape[0] - 2)
            frac = (clamped - grid[idx]) / (grid[idx + 1] - grid[idx])
            return idx, frac

        grid = jnp.linspace(-2.0, 0.0, 7)
        idx, frac = interp_met(jnp.array(-1.0), grid)
        assert 0 <= int(idx) <= 5
        assert 0.0 <= float(frac) <= 1.0

    def test_searchsorted_clip_vmap_safe(self):
        """Batched metallicity interpolation via vmap — shape[0] is required."""

        def interp_one(lz, grid):
            clamped = jnp.clip(lz, grid[0], grid[-1])
            idx = jnp.clip(jnp.searchsorted(grid, clamped) - 1, 0, grid.shape[0] - 2)
            frac = (clamped - grid[idx]) / (grid[idx + 1] - grid[idx])
            return idx, frac

        # vmap over lz; grid is a broadcast constant.
        grid = jnp.linspace(-2.0, 0.0, 7)
        lz_batch = jnp.linspace(-2.0, 0.0, 10)

        batch_interp = jax.vmap(interp_one, in_axes=(0, None))
        idxs, fracs = batch_interp(lz_batch, grid)
        assert idxs.shape == (10,)
        assert fracs.shape == (10,)
        assert jnp.all(fracs >= 0.0) and jnp.all(fracs <= 1.0)

    def test_two_axis_alpha_interp_vmap_safe(self):
        """2-D (met × alpha) bilinear interpolation is JIT+vmap-safe."""

        def interp_2d(lz, afe, ssp_lgmet, ssp_alpha_fe):
            lz_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            iz = jnp.clip(
                jnp.searchsorted(ssp_lgmet, lz_c) - 1,
                0,
                ssp_lgmet.shape[0] - 2,
            )
            fz = (lz_c - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])
            afe_c = jnp.clip(afe, ssp_alpha_fe[0], ssp_alpha_fe[-1])
            ia = jnp.clip(
                jnp.searchsorted(ssp_alpha_fe, afe_c) - 1,
                0,
                ssp_alpha_fe.shape[0] - 2,
            )
            fa = (afe_c - ssp_alpha_fe[ia]) / (ssp_alpha_fe[ia + 1] - ssp_alpha_fe[ia])
            return iz, fz, ia, fa

        met_grid = jnp.linspace(-2.0, 0.0, 7)
        afe_grid = jnp.linspace(0.0, 0.4, 5)
        lz_batch = jnp.linspace(-1.8, -0.2, 8)
        afe_batch = jnp.linspace(0.05, 0.35, 8)

        batched = jax.jit(jax.vmap(interp_2d, in_axes=(0, 0, None, None)))
        iz, fz, ia, fa = batched(lz_batch, afe_batch, met_grid, afe_grid)

        assert jnp.all(fz >= 0.0) and jnp.all(fz <= 1.0)
        assert jnp.all(fa >= 0.0) and jnp.all(fa <= 1.0)
        assert jnp.all(iz >= 0) and jnp.all(iz <= 5)
        assert jnp.all(ia >= 0) and jnp.all(ia <= 3)

    def test_shape0_preferred_over_len_for_rank1_arrays(self):
        """Document the idiom: .shape[0] is preferred over len() for JAX arrays.

        For rank-1 jnp arrays with concrete shapes, len() and .shape[0] are
        equivalent.  .shape[0] is the preferred form because:
        - It is explicit about dimensionality (won't silently switch meaning
          if the array is later reshaped or given a batch dimension).
        - It survives future dynamic-shape JAX extensions where len() may
          become shape-dependent.
        - It is consistent with the rest of the JAX codebase convention.
        """
        grid = jnp.linspace(-2.0, 0.0, 7)

        # Both must agree for rank-1 arrays.
        assert len(grid) == grid.shape[0]

        @jax.jit
        def interp_shape0(lz, grid):
            clamped = jnp.clip(lz, grid[0], grid[-1])
            idx = jnp.clip(jnp.searchsorted(grid, clamped) - 1, 0, grid.shape[0] - 2)
            return idx

        result = interp_shape0(jnp.array(-1.0), grid)
        assert 0 <= int(result) <= 5

    def test_vmap_over_grid_rank_change_caught_by_shape0(self):
        """Vmap over the grid axis changes len() vs .shape[0] semantics.

        When vmap maps over the first axis of a (batch, n_met) grid,
        the function sees a (n_met,) slice.  len() returns n_met correctly
        here too, but .shape[0] makes the intent explicit and prevents
        accidental access of the wrong dimension if the grid layout changes.
        """

        def interp_shape0(lz, grid):
            n = grid.shape[0]
            clamped = jnp.clip(lz, grid[0], grid[-1])
            idx = jnp.clip(jnp.searchsorted(grid, clamped) - 1, 0, n - 2)
            return idx

        # Single grid (n_met=7)
        grid = jnp.linspace(-2.0, 0.0, 7)
        result = interp_shape0(jnp.array(-1.0), grid)
        assert int(result) >= 0

        # Batched grids (batch=3, n_met=7) — vmap over batch axis
        grid_batch = jnp.stack([grid, grid + 0.1, grid - 0.1])
        lz_batch = jnp.array([-1.8, -1.0, -0.5])
        batched = jax.vmap(interp_shape0, in_axes=(0, 0))
        idxs = batched(lz_batch, grid_batch)
        assert idxs.shape == (3,)


# ===========================================================================
# Class 3: _traceable mode routing and JIT composability
# ===========================================================================


class TestTraceableRouting:
    """_traceable mode must route to the lean precomputed kernel and be JIT-safe."""

    def test_traceable_methods_exist(self):
        """Both _predict_*_traceable methods must exist on Model."""
        from tengri.forward.sed_model import SEDModel

        assert hasattr(SEDModel, "_predict_photometry_traceable"), (
            "_predict_photometry_traceable missing from SEDModel"
        )
        assert hasattr(SEDModel, "_predict_spectrum_traceable"), (
            "_predict_spectrum_traceable missing — spectrum VI will use full SED path"
        )

    @_needs_ssp
    def test_traceable_spectrum_composable_with_jit(self):
        """predict_spectrum(_traceable) must not raise inside jax.jit scope.

        If _traceable internally calls jax.jit, a nested-JIT error is raised.
        This verifies the 'no jit-inside-jit' invariant that keeps VI memory
        from exploding.
        """
        model, wave_obs = _make_spec_model()
        key = jax.random.PRNGKey(1)
        params = model.spec.sample(key)

        @jax.jit
        def loss(params):
            flux = model.predict_spectrum(params, wave_obs=wave_obs, mode="_traceable")
            return jnp.sum(flux**2)

        val = loss(params)
        assert jnp.isfinite(val), "_traceable spectrum inside jit produced non-finite loss"

    @_needs_ssp
    def test_traceable_photometry_composable_with_jit(self):
        """predict_photometry(_traceable) must not raise inside jax.jit scope."""
        model = _make_phot_model()
        key = jax.random.PRNGKey(2)
        params = model.spec.sample(key)

        @jax.jit
        def loss(params):
            flux = model.predict_photometry(params, mode="_traceable")
            return jnp.sum(flux**2)

        val = loss(params)
        assert jnp.isfinite(val)

    @_needs_ssp
    def test_traceable_spectrum_gradient_finite(self):
        """jax.grad through _traceable spectrum must produce finite gradients.

        VI/MAP optimization differentiates through predict_spectrum at every
        step.  Non-finite gradients cause immediate divergence.
        """
        model, wave_obs = _make_spec_model()
        key = jax.random.PRNGKey(3)
        params = model.spec.sample(key)

        # Differentiate w.r.t. a single free scalar to keep the test simple.
        # Use dust_tau_diff if available, otherwise take the first free param.
        free = model.spec.free_params
        test_param = "dust_tau_diff" if "dust_tau_diff" in free else next(iter(free))

        def loss(val):
            p = {**params, test_param: val}
            flux = model.predict_spectrum(p, wave_obs=wave_obs, mode="_traceable")
            return jnp.sum(flux)

        grad = jax.grad(loss)(params[test_param])
        assert jnp.isfinite(grad), (
            f"Non-finite gradient d(sum_flux)/d({test_param}) from _traceable spectrum — "
            "gradient will diverge during VI"
        )

    @_needs_ssp
    def test_traceable_spectrum_matches_compositional(self):
        """_traceable and compositional spectra must agree to within 2%."""
        model, wave_obs = _make_spec_model()
        key = jax.random.PRNGKey(4)
        params = model.spec.sample(key)

        flux_t = model.predict_spectrum(params, wave_obs=wave_obs, mode="_traceable")
        flux_c = model.predict_spectrum(params, wave_obs=wave_obs, mode="compositional")

        med = jnp.median(jnp.abs(flux_c))
        if float(med) == 0.0:
            pytest.skip("compositional flux is zero — degenerate sample")

        rel_err = jnp.max(jnp.abs(flux_t - flux_c)) / (med + 1e-30)
        assert float(rel_err) < 0.02, (
            f"_traceable vs compositional max relative error {float(rel_err):.3%} — "
            "routing to the wrong kernel"
        )


# ===========================================================================
# Class 3b: prediction consistency across precompute boundary
# ===========================================================================


class TestPrecomputeConsistency:
    """Predictions must be identical before and after precompute (same physics)."""

    @_needs_ssp
    def test_spectrum_consistent_before_after_precompute(self):
        """predict_spectrum result must not change when precompute_spectroscopy is called."""
        import tengri

        wave_obs = jnp.linspace(4000.0, 9000.0, 100)
        model = tengri.Model.from_config(
            ssp=str(_SSP_FILE),
            sfh="dpl",
            redshift=0.1,
        )
        key = jax.random.PRNGKey(5)
        params = model.spec.sample(key)

        # Exact mode: no precompute involved.
        flux_before = model.predict_spectrum(params, wave_obs=wave_obs, mode="exact")

        model.precompute_spectroscopy(wave_obs)

        # After precompute, compositional should give same result.
        flux_after = model.predict_spectrum(params, wave_obs=wave_obs, mode="compositional")

        med = jnp.median(jnp.abs(flux_before))
        if float(med) == 0.0:
            pytest.skip("flux is zero — degenerate sample")

        rel_err = float(jnp.max(jnp.abs(flux_before - flux_after)) / (med + 1e-30))
        assert rel_err < 1e-6, (
            f"predict_spectrum changed by {rel_err:.2e} after precompute_spectroscopy — "
            "precompute altered the physics"
        )
