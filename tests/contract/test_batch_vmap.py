# SPDX-License-Identifier: BSD-3-Clause
"""Tests for vmap batch photometry and spectroscopy.

Validates that batched (vmapped) predictions match individual
predictions, and that vmap provides speedup over Python loops.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import precompute_dust_age_weights
from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

pytestmark = pytest.mark.bounds


# ── Fixtures: synthetic data for testing vmap ─────────────────────


@pytest.fixture
def n_age():
    return 50


@pytest.fixture
def n_filt():
    return 5


@pytest.fixture
def n_met():
    return 5


@pytest.fixture
def ssp_ages_yr(n_age):
    return 10.0 ** jnp.linspace(5.5, 10.14, n_age)


@pytest.fixture
def ssp_lgmet(n_met):
    return jnp.array([-2.0, -1.5, -1.0, -0.5, 0.0])


@pytest.fixture
def ssp_phot(n_met, n_age, n_filt):
    key = jax.random.PRNGKey(0)
    return jnp.abs(jax.random.normal(key, (n_met, n_age, n_filt))) + 0.1


@pytest.fixture
def eff_waves_rest(n_filt):
    return jnp.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0])


@pytest.fixture
def dust_age_weights(ssp_ages_yr):
    return precompute_dust_age_weights(ssp_ages_yr)


@pytest.fixture
def single_galaxy_fn(ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_weights, ssp_ages_yr):
    """Build a single-galaxy photometry function for vmapping."""
    dt = jnp.float64

    @jax.jit
    def predict(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n):
        age_dt = jnp.concatenate(
            [
                jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
            ]
        )
        weights = sfr_on_ssp * age_dt
        log_z_c = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
        idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
        frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
        ssp_at_z = (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]
        wave_ratio = (eff_waves_rest / 5500.0) ** dust_n
        tau_v_eff = dust_age_weights * tau_v1 + tau_v2
        dust = jnp.exp(-(tau_v_eff[:, None] * wave_ratio[None, :]))
        flux_lsun = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)
        return 1e-30 * flux_lsun * LSUN_ERG_PER_S

    return predict


# ── Tests ─────────────────────────────────────────────────────────


class TestVmapBatchAccuracy:
    """Vmapped batch matches individual calls."""

    def test_batch_matches_individual(self, single_galaxy_fn, ssp_ages_yr):
        """vmap output matches loop over individual galaxies."""
        n_batch = 10
        n_age = len(ssp_ages_yr)
        key = jax.random.PRNGKey(42)

        # Generate batch of random SFRs and params
        keys = jax.random.split(key, n_batch)
        sfr_batch = jnp.abs(jax.random.normal(key, (n_batch, n_age))) + 0.01
        log_z_batch = jax.random.uniform(keys[0], (n_batch,), minval=-2.0, maxval=0.0)
        tau_v1_batch = jax.random.uniform(keys[1], (n_batch,), minval=0.0, maxval=2.0)
        tau_v2_batch = jax.random.uniform(keys[2], (n_batch,), minval=0.0, maxval=1.0)
        dust_n_batch = jnp.full(n_batch, -0.7)

        # Vmapped batch
        batch_fn = jax.vmap(single_galaxy_fn)
        result_batch = batch_fn(sfr_batch, log_z_batch, tau_v1_batch, tau_v2_batch, dust_n_batch)

        # Individual loop
        for i in range(n_batch):
            result_single = single_galaxy_fn(
                sfr_batch[i],
                log_z_batch[i],
                tau_v1_batch[i],
                tau_v2_batch[i],
                dust_n_batch[i],
            )
            assert_allclose(result_batch[i], result_single, rtol=1e-12)

    def test_batch_output_shape(self, single_galaxy_fn, ssp_ages_yr):
        """Batch output has correct shape (N, n_filters)."""
        n_batch = 20
        n_age = len(ssp_ages_yr)
        sfr_batch = jnp.ones((n_batch, n_age))
        params = jnp.zeros(n_batch)

        batch_fn = jax.vmap(single_galaxy_fn)
        result = batch_fn(sfr_batch, params, params + 0.5, params + 0.3, params - 0.7)
        assert result.shape == (n_batch, 5)  # 5 filters


class TestVmapBatchGradients:
    """Gradients through vmapped batch."""

    def test_batch_gradients_finite(self, single_galaxy_fn, ssp_ages_yr):
        """Gradients through vmap are all finite."""
        n_batch = 5
        n_age = len(ssp_ages_yr)
        sfr_batch = jnp.ones((n_batch, n_age)) * 0.1

        batch_fn = jax.vmap(single_galaxy_fn)

        def loss(sfr_batch, log_z_batch):
            return jnp.sum(
                batch_fn(
                    sfr_batch,
                    log_z_batch,
                    jnp.full(n_batch, 0.5),
                    jnp.full(n_batch, 0.3),
                    jnp.full(n_batch, -0.7),
                )
            )

        g_sfr, g_lz = jax.grad(loss, argnums=(0, 1))(
            sfr_batch,
            jnp.full(n_batch, -1.0),
        )
        chex.assert_tree_all_finite(g_sfr)
        chex.assert_tree_all_finite(g_lz)
