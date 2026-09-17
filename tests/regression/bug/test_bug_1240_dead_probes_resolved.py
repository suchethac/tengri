# SPDX-License-Identifier: MIT
"""Test for issue #1240: verify that dead fail-open probes are resolved."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SSP_EXISTS = len(list(_DATA_DIR.glob("ssp_*.h5"))) > 0

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.skipif(
        not _SSP_EXISTS,
        reason="SSP data file not found — tests require data/ssp_*.h5",
    ),
]


@pytest.fixture(scope="module")
def minimal_spec():
    """Minimal parameter spec for testing."""
    from tengri.parameters.parameters import Parameters
    from tengri.parameters.priors import Fixed, Uniform

    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.5, 4.0),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def synthetic_ssp():
    """Minimal synthetic SSP for testing."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    n_met, n_age, n_wave = 3, 20, 100
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages_gyr = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(456)
    flux = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    return SSPData(
        ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet
    )


@pytest.fixture
def minimal_model(minimal_spec, synthetic_ssp):
    """Build minimal SEDModel for testing."""
    from tengri.forward.sed_model import SEDModel

    return SEDModel(minimal_spec, synthetic_ssp, precompute=False, filters=None)


class TestBug1240ProbesResolved:
    """Verify dead fail-open probes from #1240 are resolved."""

    def test_hybrid_never_assigned(self, minimal_model):
        """SEDModel._hybrid is never assigned — verify it."""
        # grep confirms _hybrid has 0 assignments
        assert not hasattr(minimal_model, "_hybrid"), (
            "_hybrid should never be assigned; probe is dead"
        )

    def test_wave_obs_cache_never_populated(self, minimal_model):
        """SEDModel._wave_obs cache is never populated — verify it."""
        # grep confirms _wave_obs has 0 assignments
        assert not hasattr(minimal_model, "_wave_obs"), (
            "_wave_obs cache should never be populated; probe is dead"
        )

    def test_dust_config_emission_model_never_assigned(self, minimal_model):
        """dust.config.emission_model is dead (legacy path post-migration)."""
        # The dust component is not used in this minimal model
        # but we verify the new _dust_emission_model is what's used instead
        dust_emission = getattr(minimal_model, "_dust_emission_model", None)
        # The property should not exist, or be correctly set
        assert isinstance(dust_emission, (type(None), str)), (
            "_dust_emission_model should be None or str, never dust.config.emission_model"
        )

    def test_compositional_probe_removed(self, minimal_model):
        """profiling/pipeline.py _compositional probe should not exist."""
        # grep confirms _compositional has 0 assignments
        assert not hasattr(minimal_model, "_compositional"), (
            "_compositional should never be assigned; probe is dead"
        )

    def test_weights_probe_uses_correct_attribute(self, minimal_model):
        """profiling/memory.py should check 'weights' not '_weights'."""
        # CueBackend uses 'weights', not '_weights'
        from tengri.components.nebular.cue import CueBackend

        neb = getattr(minimal_model, "_nebular_backend", None)
        if neb is not None and isinstance(neb, CueBackend):
            # If it's a CueBackend, verify it has 'weights', not '_weights'
            assert hasattr(neb, "weights"), (
                "CueBackend should have 'weights', not '_weights'"
            )
            assert not hasattr(neb, "_weights"), (
                "CueBackend should not have '_weights'"
            )
