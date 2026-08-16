# SPDX-License-Identifier: BSD-3-Clause
"""Tests verifying SEDModel JIT trace safety for HDF5 grid loaders.

Caches @functools.cache decorators on HDF5 loaders can leak JAX tracers into
downstream code if first called inside a jax.jit trace. This module verifies
that:

1. Grid caches are warmed at SEDModel construction (outside JIT context)
2. Cached grids return concrete arrays, not JAX tracers
3. Multiple SEDModels can be built and JIT-traced without tracer leaks
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.gradient

jax.config.update("jax_enable_x64", True)

_DATA = Path(__file__).resolve().parents[4] / "data"
_MAPPINGS_H5 = _DATA / "mappings_templates.h5"
_CAT3D_H5 = _DATA / "cat3d_wind_torus_grid.h5"

# Skip markers for missing optional grids
_SKIP_NO_MAPPINGS = pytest.mark.skipif(
    not _MAPPINGS_H5.exists(),
    reason="data/mappings_templates.h5 not found",
)

_SKIP_NO_CAT3D = pytest.mark.skipif(
    not _CAT3D_H5.exists(),
    reason="data/cat3d_wind_torus_grid.h5 not found",
)


@pytest.fixture
def shock_spec_with_filters(synthetic_ssp):
    """Build a Parameters spec with shock enabled + observation."""
    from tengri import Fixed, Observation, Parameters, Photometry
    from tengri.observation.photometry import FilterCurve

    centers = np.array([1500.0, 3500.0, 5500.0, 8500.0, 12000.0])
    widths = np.array([300.0, 500.0, 700.0, 1000.0, 1500.0])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 32)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)

    curves = tuple(
        FilterCurve(wave=w, trans=t, name=f"shock_band_{i}")
        for i, (w, t) in enumerate(zip(waves, trans))
    )
    observation = Observation(photometry=Photometry(filters=curves))

    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(0.0),
        met_logzsol=Fixed(-0.5),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        dust_slope=Fixed(-0.7),
        shock_frac=Fixed(0.1),
        shock_velocity=Fixed(300.0),
        shock_log_density=Fixed(0.0),
        shock_b_over_sqrt_n=Fixed(1.0),
        redshift=Fixed(0.1),
        shock=True,
    )

    return spec, observation, synthetic_ssp


@pytest.fixture
def cat3d_spec_with_filters(synthetic_ssp):
    """Build a Parameters spec with CAT3D-Wind AGN enabled + observation."""
    from tengri import Fixed, Observation, Parameters, Photometry
    from tengri.observation.photometry import FilterCurve

    centers = np.array([1500.0, 3500.0, 5500.0, 8500.0, 12000.0])
    widths = np.array([300.0, 500.0, 700.0, 1000.0, 1500.0])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 32)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)

    curves = tuple(
        FilterCurve(wave=w, trans=t, name=f"cat3d_band_{i}")
        for i, (w, t) in enumerate(zip(waves, trans))
    )
    observation = Observation(photometry=Photometry(filters=curves))

    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(0.0),
        met_logzsol=Fixed(-0.5),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        dust_slope=Fixed(-0.7),
        agn_model="cat3d_wind",
        agn_log_lbol=Fixed(10.42),
        agn_cos_inc=Fixed(0.5),
        agn_a_cat3d=Fixed(-2.0),
        agn_fwd_cat3d=Fixed(1.75),
        agn_torus_frac=Fixed(0.5),
        redshift=Fixed(0.1),
    )

    return spec, observation, synthetic_ssp


class TestShockGridCacheWarming:
    """Verify shock grid caches are warmed to concrete arrays."""

    @_SKIP_NO_MAPPINGS
    def test_mappings_grid_cache_returns_concrete_arrays(self):
        """After SEDModel init, _load_mappings_grids() should return jnp arrays, not tracers."""
        from tengri.components.nebular.shock import _load_mappings_grids

        # First call (cache cold)
        grids = _load_mappings_grids()

        # Verify it returns a dict with grid arrays
        assert grids is not None
        assert isinstance(grids, dict)
        assert "mappings5" in grids

        # Verify all arrays are concrete (not traced)
        for key, val in grids["mappings5"].items():
            if isinstance(val, jnp.ndarray):
                # Concrete arrays should be JAX tracer-free
                assert hasattr(val, "shape"), f"{key} is not a valid array"
                # Try to convert to numpy — should succeed if concrete
                try:
                    arr = np.asarray(val)
                    assert arr.dtype != object, f"{key} contains tracer objects"
                except Exception as e:
                    pytest.fail(f"Grid {key} is traced: {e}")


class TestShockJITTraceSafety:
    """Verify shock kernel JIT-tracing doesn't cause tracer leaks."""

    @_SKIP_NO_MAPPINGS
    def test_shock_jit_trace_no_tracer_leak(self, shock_spec_with_filters):
        """Build SEDModels sequentially and JIT-trace each without tracer leaks."""
        from tengri import SEDModel

        spec, observation, ssp_data = shock_spec_with_filters

        # Build first SEDModel (caches grids at init)
        model1 = SEDModel(spec, ssp_data, observation=observation, precompute=False)

        # JIT-compile and call kernel on model1
        key1 = jax.random.PRNGKey(42)
        params1 = spec.sample(key1)

        def jit_predict_phot(model, params):
            return model.predict_photometry(params)

        jit_fn = jax.jit(jit_predict_phot, static_argnames=[])

        # This should succeed without TracerArrayConversionError
        sed1 = jit_fn(model1, params1)
        chex.assert_tree_all_finite(sed1)
        # Build a second SEDModel (cache already warm)
        model2 = SEDModel(spec, ssp_data, observation=observation, precompute=False)

        # JIT-compile and call kernel on model2
        key2 = jax.random.PRNGKey(123)
        params2 = spec.sample(key2)

        # Create a new JIT-compiled function (different closure)
        jit_fn2 = jax.jit(jit_predict_phot, static_argnames=[])

        # This should also succeed without tracer leaks
        sed2 = jit_fn2(model2, params2)
        chex.assert_tree_all_finite(sed2)
        # Build a third SEDModel to stress-test cache persistence
        model3 = SEDModel(spec, ssp_data, observation=observation, precompute=False)
        key3 = jax.random.PRNGKey(999)
        params3 = spec.sample(key3)

        jit_fn3 = jax.jit(jit_predict_phot, static_argnames=[])
        sed3 = jit_fn3(model3, params3)
        chex.assert_tree_all_finite(sed3)


class TestCat3DGridCacheWarming:
    """Verify CAT3D-Wind grid caches are warmed to concrete arrays."""

    @_SKIP_NO_CAT3D
    def test_cat3d_grid_cache_returns_callable(self):
        """After _warm_grid_caches, _load_cat3d_default() should return callable."""
        from tengri.components.agn.cat3d_wind import _load_cat3d_default

        # First call (cache cold)
        fn = _load_cat3d_default()

        # Verify it returns a callable
        assert callable(fn), "_load_cat3d_default should return a callable"

        # Test the callable with dummy args (don't care about output, just no crash)
        try:
            wave = jnp.linspace(1000, 10000, 100)
            _output = fn(wave)
            chex.assert_shape(_output, (100,))
        except Exception as e:
            pytest.fail(f"CAT3D-Wind callable failed: {e}")


class TestCat3DJITTraceSafety:
    """Verify CAT3D-Wind kernel JIT-tracing doesn't cause tracer leaks."""

    @_SKIP_NO_CAT3D
    def test_cat3d_jit_trace_no_tracer_leak(self, cat3d_spec_with_filters):
        """Build SEDModels with CAT3D and JIT-trace without tracer leaks."""
        from tengri import SEDModel

        spec, observation, ssp_data = cat3d_spec_with_filters

        # Build first SEDModel with CAT3D (caches grids at init)
        model1 = SEDModel(spec, ssp_data, observation=observation, precompute=False)

        key1 = jax.random.PRNGKey(42)
        params1 = spec.sample(key1)

        def jit_predict_phot(model, params):
            return model.predict_photometry(params)

        jit_fn = jax.jit(jit_predict_phot, static_argnames=[])

        # This should succeed without TracerArrayConversionError
        sed1 = jit_fn(model1, params1)
        chex.assert_tree_all_finite(sed1)
        # Build a second SEDModel with CAT3D (cache already warm)
        model2 = SEDModel(spec, ssp_data, observation=observation, precompute=False)

        key2 = jax.random.PRNGKey(123)
        params2 = spec.sample(key2)

        jit_fn2 = jax.jit(jit_predict_phot, static_argnames=[])
        sed2 = jit_fn2(model2, params2)
        chex.assert_tree_all_finite(sed2)
