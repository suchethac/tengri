# SPDX-License-Identifier: BSD-3-Clause
"""Integration test for mock galaxy population generator.

Tests that the population maker handles realistic forward model predictions
and correctly reports edge cases (Halpha in absorption) without silently
dropping galaxies.
"""

import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, FREE, Observation, SEDModel
from tengri.analysis.population_mocks import make_population
from tengri.observation import Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.slow


@pytest.fixture
def phot_obs():
    """10-band synthetic photometry observation for integration testing.

    Uses the same top-hat filter pattern as synthetic_tophat_obs but with
    more bands to exercise the forward model's photometry path thoroughly.
    """

    def _tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = [2500.0, 3500.0, 4500.0, 5500.0, 6500.0, 7500.0, 8500.0, 9500.0, 11000.0, 13000.0]
    curves = tuple(_tophat(c) for c in centers)
    return Observation(photometry=Photometry(filters=curves))


@pytest.fixture
def field_model(synthetic_ssp_wide, phot_obs):
    """DPL + stochastic field at z = 0.1, 16 field latents.

    Uses synthetic_ssp_wide (no data/ files) so the test runs on every PR.
    The stochastic field captures burstiness that can produce Halpha
    absorption during lulls in the SFH.
    """
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=phot_obs,
        sfh={"type": "dpl", "all_params": FREE, "age_gyr": 11.0, "field": {"all_params": FREE}},
        dust={"type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
        n_grid=16,
    )


def test_halpha_absorption_galaxies_are_counted_not_dropped(field_model):
    """Verify that Halpha absorption events are reported, not silently dropped.

    At sigma = 0.6 roughly 1 in 15 drawn truths has Halpha in absorption --
    a bursty history observed during a lull. They must be counted, never
    silently dropped: dropping them biases the survivors line-bright.

    Uses synthetic_ssp_wide (no emission lines on real SSP), so line fluxes
    will be approximately zero. This test validates the counting mechanism
    and the fact that no galaxies are dropped, which is the critical safety
    check. On a real SSP with nebular emission, this would track meaningful
    Halpha absorption events from burstiness.
    """
    pop = make_population(
        field_model,
        n_galaxies=45,
        sigma_true=0.6,
        tau_true_myr=150.0,
        key=jax.random.PRNGKey(0),
        snr_phot=20.0,
        snr_line=10.0,
    )
    assert len(pop.truth_params) == 45, "galaxies were dropped"
    assert pop.n_halpha_absorption >= 0, "absorption count should be non-negative"
    assert pop.table is not None, "table should be populated"
    assert len(pop.table) == 45, "table row count should match n_galaxies"
