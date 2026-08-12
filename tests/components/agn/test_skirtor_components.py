# SPDX-License-Identifier: BSD-3-Clause
"""Tests for SKIRTOR separate disk/dust component loading.

Validates the v3 HDF5 layout with separate disk_emission and dust_emission
grids, backward compatibility with v2 (total-only), and the
SKIRTORComponents named tuple interface.
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tests._grad_parity import assert_grad_matches_fd

jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def v3_grid_path():
    """Create a synthetic v3 SKIRTOR HDF5 with separate disk/dust."""
    import h5py

    n_tau, n_p, n_q, n_oa, n_inc = 2, 2, 2, 2, 3
    n_wave = 50
    rng = np.random.default_rng(42)
    wavelength = np.logspace(1, 7, n_wave)
    shape = (n_tau, n_p, n_q, n_oa, n_inc, n_wave)
    disk = rng.uniform(1e-15, 1e-10, size=shape)
    dust = rng.uniform(1e-12, 1e-8, size=shape)
    total = disk + dust
    tau = np.array([3.0, 7.0])
    p = np.array([0.0, 1.0])
    q = np.array([0.0, 1.0])
    oa = np.array([30.0, 60.0])
    cos_inc = np.array([0.0, 0.5, 1.0])
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    with h5py.File(path, "w") as f:
        f.create_dataset("wavelength", data=wavelength)
        grp = f.create_group("grid")
        grp.create_dataset("tau_97", data=tau)
        grp.create_dataset("p", data=p)
        grp.create_dataset("q", data=q)
        grp.create_dataset("opening_angle", data=oa)
        grp.create_dataset("cos_inclination", data=cos_inc)
        spec = f.create_group("spectra")
        spec.create_dataset("disk_emission", data=disk)
        spec.create_dataset("dust_emission", data=dust)
        spec.create_dataset("torus_emission", data=total)
        meta = f.create_group("metadata")
        meta.attrs["version"] = 3
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture()
def v2_grid_path():
    """Create a synthetic v2 SKIRTOR HDF5 (total-only, no disk/dust)."""
    import h5py

    n_tau, n_p, n_q, n_oa, n_inc = 2, 2, 2, 2, 3
    n_wave = 50
    rng = np.random.default_rng(42)
    wavelength = np.logspace(1, 7, n_wave)
    shape = (n_tau, n_p, n_q, n_oa, n_inc, n_wave)
    total = rng.uniform(1e-12, 1e-8, size=shape)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    with h5py.File(path, "w") as f:
        f.create_dataset("wavelength", data=wavelength)
        grp = f.create_group("grid")
        grp.create_dataset("tau_97", data=np.array([3.0, 7.0]))
        grp.create_dataset("p", data=np.array([0.0, 1.0]))
        grp.create_dataset("q", data=np.array([0.0, 1.0]))
        grp.create_dataset("opening_angle", data=np.array([30.0, 60.0]))
        grp.create_dataset("cos_inclination", data=np.array([0.0, 0.5, 1.0]))
        f.create_group("spectra")
        f["spectra"].create_dataset("torus_emission", data=total)
    yield path
    Path(path).unlink(missing_ok=True)


class TestSKIRTORComponentsV3:
    def test_returns_named_tuple(self, v3_grid_path):
        from tengri.components.agn.skirtor import (
            SKIRTORComponents,
            create_skirtor_components_from_grid,
        )

        fn = create_skirtor_components_from_grid(v3_grid_path)
        wavelength = jnp.logspace(1, 7, 100)
        result = fn(wavelength, agn_log_lbol=44.0)
        assert isinstance(result, SKIRTORComponents)
        assert hasattr(result, "disk")
        assert hasattr(result, "dust")
        assert hasattr(result, "total")

    def test_component_shapes(self, v3_grid_path):
        from tengri.components.agn.skirtor import create_skirtor_components_from_grid

        fn = create_skirtor_components_from_grid(v3_grid_path)
        wavelength = jnp.logspace(1, 7, 80)
        result = fn(wavelength)
        chex.assert_shape(result.disk, (80,))
        chex.assert_shape(result.dust, (80,))
        chex.assert_shape(result.total, (80,))

    def test_components_nonnegative(self, v3_grid_path):
        from tengri.components.agn.skirtor import create_skirtor_components_from_grid

        fn = create_skirtor_components_from_grid(v3_grid_path)
        wavelength = jnp.logspace(1, 7, 100)
        result = fn(wavelength, agn_log_lbol=44.0)
        assert jnp.all(result.disk >= 0.0)
        assert jnp.all(result.dust >= 0.0)
        assert jnp.all(result.total >= 0.0)

    def test_dust_dominates_over_disk(self, v3_grid_path):
        """In our synthetic grid, dust >> disk by construction."""
        from tengri.components.agn.skirtor import create_skirtor_components_from_grid

        fn = create_skirtor_components_from_grid(v3_grid_path)
        wavelength = jnp.logspace(1, 7, 100)
        result = fn(wavelength, agn_log_lbol=44.0)
        assert jnp.sum(result.dust) > jnp.sum(result.disk)

    def test_all_finite(self, v3_grid_path):
        from tengri.components.agn.skirtor import create_skirtor_components_from_grid

        fn = create_skirtor_components_from_grid(v3_grid_path)
        wavelength = jnp.logspace(1, 7, 100)
        result = fn(wavelength)
        chex.assert_tree_all_finite(result.disk)
        chex.assert_tree_all_finite(result.dust)
        chex.assert_tree_all_finite(result.total)

    def test_torus_frac_scales_output(self, v3_grid_path):
        from tengri.components.agn.skirtor import create_skirtor_components_from_grid

        fn = create_skirtor_components_from_grid(v3_grid_path)
        wavelength = jnp.logspace(1, 7, 100)
        r1 = fn(wavelength, agn_torus_frac=0.3)
        r2 = fn(wavelength, agn_torus_frac=0.6)
        ratio = jnp.sum(r2.total) / jnp.maximum(jnp.sum(r1.total), 1e-100)
        np.testing.assert_allclose(ratio, 2.0, rtol=0.1)


class TestV3BackwardCompat:
    def test_total_only_from_v3(self, v3_grid_path):
        """create_skirtor_from_grid works on v3 file (returns total only)."""
        from tengri.components.agn.skirtor import create_skirtor_from_grid

        fn = create_skirtor_from_grid(v3_grid_path)
        wavelength = jnp.logspace(1, 7, 100)
        result = fn(wavelength, agn_log_lbol=44.0)
        chex.assert_shape(result, (100,))
        chex.assert_tree_all_finite(result)

    def test_v2_raises_on_components(self, v2_grid_path):
        """create_skirtor_components_from_grid fails gracefully on v2."""
        from tengri.components.agn.skirtor import create_skirtor_components_from_grid

        with pytest.raises(KeyError, match="separate disk/dust"):
            create_skirtor_components_from_grid(v2_grid_path)

    def test_v2_total_still_works(self, v2_grid_path):
        """create_skirtor_from_grid still works with v2 layout."""
        from tengri.components.agn.skirtor import create_skirtor_from_grid

        fn = create_skirtor_from_grid(v2_grid_path)
        wavelength = jnp.logspace(1, 7, 100)
        result = fn(wavelength)
        chex.assert_shape(result, (100,))
        chex.assert_tree_all_finite(result)


class TestJITAndGradients:
    def test_jit_components(self, v3_grid_path):
        from tengri.components.agn.skirtor import create_skirtor_components_from_grid

        fn = create_skirtor_components_from_grid(v3_grid_path)
        wavelength = jnp.logspace(1, 7, 100)

        @jax.jit
        def compute(w):
            return fn(w, agn_log_lbol=44.0)

        result = compute(wavelength)
        chex.assert_tree_all_finite(result.total)

    def test_grad_through_components(self, v3_grid_path):
        from tengri.components.agn.skirtor import create_skirtor_components_from_grid

        fn = create_skirtor_components_from_grid(v3_grid_path)
        wavelength = jnp.logspace(1, 7, 100)

        def loss(lbol):
            r = fn(wavelength, agn_log_lbol=lbol)
            return jnp.sum(r.total)

        grad = assert_grad_matches_fd(loss, 44.0)
        assert jnp.isfinite(grad)
        assert grad > 0.0
