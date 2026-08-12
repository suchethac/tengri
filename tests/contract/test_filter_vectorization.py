# SPDX-License-Identifier: BSD-3-Clause
"""Tests for vectorized filter integration (pad_filters + compute_flux_density_batch).

Validates that the vmap-based filter integration matches the loop-based version
across different filter configurations, SED shapes, and edge cases.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.bounds

jax.config.update("jax_enable_x64", True)

from tengri.observation.photometry import (
    compute_flux_density,
    compute_flux_density_batch,
    pad_filters,
)
from tests._grad_parity import assert_grad_matches_fd


@pytest.fixture
def flat_sed():
    """Flat SED on a typical optical wavelength grid."""
    wave = jnp.linspace(3000.0, 10000.0, 500)
    sed = jnp.ones_like(wave) * 1e28
    return wave, sed


@pytest.fixture
def sdss_like_filters():
    """Three Gaussian-shaped filters with different lengths."""
    fw1 = jnp.linspace(3500.0, 5500.0, 100)
    ft1 = jnp.exp(-0.5 * ((fw1 - 4500.0) / 300.0) ** 2)
    fw2 = jnp.linspace(5000.0, 7500.0, 80)
    ft2 = jnp.exp(-0.5 * ((fw2 - 6200.0) / 400.0) ** 2)
    fw3 = jnp.linspace(7000.0, 9500.0, 120)
    ft3 = jnp.exp(-0.5 * ((fw3 - 8000.0) / 500.0) ** 2)
    return [fw1, fw2, fw3], [ft1, ft2, ft3]


class TestPadFilters:
    """Tests for pad_filters."""

    def test_output_shapes(self, sdss_like_filters):
        fws, fts = sdss_like_filters
        fw_pad, ft_pad, n_valid = pad_filters(fws, fts)
        max_len = max(len(fw) for fw in fws)
        chex.assert_shape(fw_pad, (3, max_len))
        chex.assert_shape(ft_pad, (3, max_len))
        chex.assert_shape(n_valid, (3,))

    def test_valid_lengths(self, sdss_like_filters):
        fws, fts = sdss_like_filters
        _fw_pad, _ft_pad, n_valid = pad_filters(fws, fts)
        expected = jnp.array([len(fw) for fw in fws])
        assert jnp.array_equal(n_valid, expected)

    def test_padding_is_zero(self, sdss_like_filters):
        fws, fts = sdss_like_filters
        fw_pad, ft_pad, n_valid = pad_filters(fws, fts)
        max_len = fw_pad.shape[1]
        for i in range(3):
            n = int(n_valid[i])
            # Padded region should be zero
            assert jnp.all(fw_pad[i, n:] == 0.0)
            assert jnp.all(ft_pad[i, n:] == 0.0)

    def test_valid_region_matches_input(self, sdss_like_filters):
        fws, fts = sdss_like_filters
        fw_pad, ft_pad, n_valid = pad_filters(fws, fts)
        for i in range(3):
            n = int(n_valid[i])
            assert jnp.allclose(fw_pad[i, :n], fws[i])
            assert jnp.allclose(ft_pad[i, :n], fts[i])

    def test_single_filter(self):
        fw = [jnp.linspace(4000.0, 5000.0, 50)]
        ft = [jnp.ones(50)]
        fw_pad, _ft_pad, n_valid = pad_filters(fw, ft)
        chex.assert_shape(fw_pad, (1, 50))
        assert n_valid[0] == 50

    def test_equal_length_filters(self):
        fw1 = jnp.linspace(4000.0, 5000.0, 60)
        fw2 = jnp.linspace(6000.0, 7000.0, 60)
        fws = [fw1, fw2]
        fts = [jnp.ones(60), jnp.ones(60)]
        fw_pad, _ft_pad, n_valid = pad_filters(fws, fts)
        chex.assert_shape(fw_pad, (2, 60))
        # No padding needed — all valid
        assert jnp.all(n_valid == 60)


class TestComputeFluxDensityBatch:
    """Tests for compute_flux_density_batch vs loop-based compute_flux_density."""

    def test_matches_loop(self, flat_sed, sdss_like_filters):
        wave, sed = flat_sed
        fws, fts = sdss_like_filters
        z, dl_cm = 0.1, 1.4e27

        # Loop version
        loop = jnp.array(
            [compute_flux_density(sed, wave, fw, ft, z, dl_cm) for fw, ft in zip(fws, fts)]
        )

        # Batch version
        fw_pad, ft_pad, _ = pad_filters(fws, fts)
        batch = compute_flux_density_batch(sed, wave, fw_pad, ft_pad, z, dl_cm)

        assert jnp.allclose(loop, batch, rtol=1e-12)

    def test_non_flat_sed(self, sdss_like_filters):
        """Power-law SED (more realistic)."""
        wave = jnp.linspace(3000.0, 10000.0, 500)
        sed = 1e28 * (wave / 5000.0) ** (-2.0)
        fws, fts = sdss_like_filters
        z, dl_cm = 0.3, 4.2e27

        loop = jnp.array(
            [compute_flux_density(sed, wave, fw, ft, z, dl_cm) for fw, ft in zip(fws, fts)]
        )
        fw_pad, ft_pad, _ = pad_filters(fws, fts)
        batch = compute_flux_density_batch(sed, wave, fw_pad, ft_pad, z, dl_cm)

        assert jnp.allclose(loop, batch, rtol=1e-12)

    def test_high_redshift(self, flat_sed, sdss_like_filters):
        """High redshift shifts SED out of some filters."""
        wave, sed = flat_sed
        fws, fts = sdss_like_filters
        z, dl_cm = 2.0, 1.5e28

        loop = jnp.array(
            [compute_flux_density(sed, wave, fw, ft, z, dl_cm) for fw, ft in zip(fws, fts)]
        )
        fw_pad, ft_pad, _ = pad_filters(fws, fts)
        batch = compute_flux_density_batch(sed, wave, fw_pad, ft_pad, z, dl_cm)

        assert jnp.allclose(loop, batch, rtol=1e-12)

    def test_single_filter(self, flat_sed):
        wave, sed = flat_sed
        fw = [jnp.linspace(4000.0, 5000.0, 50)]
        ft = [jnp.exp(-0.5 * ((fw[0] - 4500.0) / 200.0) ** 2)]
        z, dl_cm = 0.1, 1.4e27

        loop = compute_flux_density(sed, wave, fw[0], ft[0], z, dl_cm)
        fw_pad, ft_pad, _ = pad_filters(fw, ft)
        batch = compute_flux_density_batch(sed, wave, fw_pad, ft_pad, z, dl_cm)

        assert jnp.allclose(loop, batch[0], rtol=1e-12)

    def test_gradients_finite(self, flat_sed, sdss_like_filters):
        """Gradients through the batch function should be finite."""
        wave, sed = flat_sed
        fws, fts = sdss_like_filters
        z, dl_cm = 0.1, 1.4e27
        fw_pad, ft_pad, _ = pad_filters(fws, fts)

        def loss(sed_input):
            return jnp.sum(
                compute_flux_density_batch(
                    sed_input,
                    wave,
                    fw_pad,
                    ft_pad,
                    z,
                    dl_cm,
                )
            )

        grad = assert_grad_matches_fd(loss, sed)
        chex.assert_tree_all_finite(grad)

    def test_jit_compatible(self, flat_sed, sdss_like_filters):
        """Batch function works inside jax.jit."""
        wave, sed = flat_sed
        fws, fts = sdss_like_filters
        z, dl_cm = 0.1, 1.4e27
        fw_pad, ft_pad, _ = pad_filters(fws, fts)

        @jax.jit
        def compute(sed_input):
            return compute_flux_density_batch(
                sed_input,
                wave,
                fw_pad,
                ft_pad,
                z,
                dl_cm,
            )

        result = compute(sed)
        chex.assert_shape(result, (3,))
        chex.assert_tree_all_finite(result)
