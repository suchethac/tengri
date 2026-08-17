# SPDX-License-Identifier: BSD-3-Clause
"""Tests for SSP pre-computation at fixed redshift."""

import chex
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sps.precompute import (
    fast_photometry,
    fast_spectrum,
    interpolate_ssp_phot_metallicity,
)
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.bounds


class TestFastPhotometry:
    """Tests for the fast (pre-computed) photometry function."""

    def test_output_shape(self):
        """Output has shape (n_filters,)."""
        n_age, n_filt = 50, 8
        weights = jnp.ones(n_age) / n_age
        ssp_phot = jnp.ones((n_age, n_filt))
        dust = jnp.ones((n_age, n_filt))
        flux = fast_photometry(weights, ssp_phot, dust, 0.0)
        chex.assert_shape(flux, (n_filt,))

    def test_no_dust_equals_ssp_weighted_sum(self):
        """With dust=1 and uniform weights, flux = mean(ssp_phot) * scale."""
        n_age, n_filt = 10, 3
        weights = jnp.ones(n_age) / n_age
        ssp_phot = jnp.arange(n_age * n_filt, dtype=float).reshape(n_age, n_filt)
        dust = jnp.ones((n_age, n_filt))
        # The kernel takes a log10 OFFSET, not a linear factor (#1859): passing
        # 2.0 here would silently ask for 100x.
        scale = 2.0
        log10_scale = jnp.log10(scale)

        flux = fast_photometry(weights, ssp_phot, dust, log10_scale)

        # Expected: scale * Lsun * weighted_sum over ages for each filter
        from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

        expected = scale * LSUN_ERG_PER_S * jnp.mean(ssp_phot, axis=0)
        assert_allclose(flux, expected, rtol=1e-10)

    def test_dust_reduces_flux(self):
        """Dust < 1 always reduces flux."""
        n_age, n_filt = 20, 5
        weights = jnp.ones(n_age) / n_age
        ssp_phot = jnp.ones((n_age, n_filt))

        flux_no_dust = fast_photometry(weights, ssp_phot, jnp.ones((n_age, n_filt)), 0.0)
        flux_dusty = fast_photometry(weights, ssp_phot, 0.5 * jnp.ones((n_age, n_filt)), 0.0)

        assert jnp.all(flux_dusty < flux_no_dust)

    def test_is_jittable(self):
        """fast_photometry is JIT-compiled (already @jax.jit)."""
        n_age, n_filt = 20, 5
        weights = jnp.ones(n_age) / n_age
        ssp_phot = jnp.ones((n_age, n_filt))
        dust = jnp.ones((n_age, n_filt))
        flux = fast_photometry(weights, ssp_phot, dust, 0.0)
        chex.assert_shape(flux, (n_filt,))

    def test_has_gradients(self):
        """Gradients w.r.t. weights and dust are finite."""
        n_age, n_filt = 20, 5
        ssp_phot = jnp.ones((n_age, n_filt))

        def loss(weights, dust):
            return jnp.sum(fast_photometry(weights, ssp_phot, dust, 0.0))

        w = jnp.ones(n_age) / n_age
        d = jnp.ones((n_age, n_filt)) * 0.8

        gw, gd = jax.grad(loss, argnums=(0, 1))(w, d)
        chex.assert_tree_all_finite(gw)
        chex.assert_tree_all_finite(gd)


class TestFastSpectrum:
    """Tests for the fast (pre-rebinned) spectrum function."""

    def test_output_shape(self):
        """Output has shape (n_pix,)."""
        n_age, n_pix = 50, 1000
        weights = jnp.ones(n_age) / n_age
        ssp_pix = jnp.ones((n_age, n_pix))
        dust = jnp.ones((n_age, n_pix))
        spec = fast_spectrum(weights, ssp_pix, dust, 0.0)
        chex.assert_shape(spec, (n_pix,))

    def test_has_gradients(self):
        """Gradients through fast_spectrum are finite."""
        n_age, n_pix = 20, 100
        ssp_pix = jnp.ones((n_age, n_pix))
        dust = jnp.ones((n_age, n_pix)) * 0.8

        def loss(weights):
            return jnp.sum(fast_spectrum(weights, ssp_pix, dust, 0.0) ** 2)

        w = jnp.ones(n_age) / n_age
        g = assert_grad_matches_fd(loss, w)
        chex.assert_tree_all_finite(g)


class TestMetallicityInterpolation:
    """Tests for pre-computed SSP metallicity interpolation."""

    def test_at_grid_point(self):
        """Interpolation at a grid point returns that grid point's data."""
        n_met, n_age, n_filt = 5, 10, 3
        ssp_phot = jax.random.normal(jax.random.PRNGKey(0), (n_met, n_age, n_filt))
        lgmet = jnp.array([-2.0, -1.5, -1.0, -0.5, 0.0])

        result = interpolate_ssp_phot_metallicity(ssp_phot, lgmet, -1.0)
        assert_allclose(result, ssp_phot[2], rtol=1e-5)

    def test_midpoint_interpolation(self):
        """Midpoint between two grid values gives the average."""
        n_met, n_age, n_filt = 3, 5, 2
        ssp_phot = jnp.zeros((n_met, n_age, n_filt))
        ssp_phot = ssp_phot.at[0].set(0.0)  # Z = -2
        ssp_phot = ssp_phot.at[1].set(1.0)  # Z = -1
        ssp_phot = ssp_phot.at[2].set(2.0)  # Z = 0
        lgmet = jnp.array([-2.0, -1.0, 0.0])

        result = interpolate_ssp_phot_metallicity(ssp_phot, lgmet, -0.5)
        assert_allclose(result, 1.5, rtol=1e-5)

    def test_clamps_to_bounds(self):
        """Values outside grid are clamped to nearest edge."""
        n_met, n_age, n_filt = 3, 5, 2
        ssp_phot = jnp.ones((n_met, n_age, n_filt))
        ssp_phot = ssp_phot.at[0].set(10.0)
        ssp_phot = ssp_phot.at[2].set(30.0)
        lgmet = jnp.array([-2.0, -1.0, 0.0])

        # Below grid: should return value at lgmet[0]
        result_low = interpolate_ssp_phot_metallicity(ssp_phot, lgmet, -5.0)
        assert_allclose(result_low, ssp_phot[0], rtol=1e-5)
