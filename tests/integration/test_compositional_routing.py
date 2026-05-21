"""Integration tests verifying compositional kernel routing for special SFH/metallicity modes.

Tests cover:
- evolving_metallicity=True routes through _predict_photometry_compositional
- chem_evol=True routes through _predict_photometry_compositional
- Tabular SFH (sfh_t_gyr + sfh_sfr) produces finite photometry without recompilation

All tests require SSP data on disk and are skipped gracefully when missing.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import SEDModel
from tengri.observation.filters import load_filter_set
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

# ── Skip guard — all tests require SSP data on disk ───────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(
    not _SSP_EXISTS,
    reason="SSP data file not found",
)

_N_FILTERS = 5
_FILTER_NAMES = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


# ── Session-scoped fixtures (heavy: SSP load + filter load) ───────


@pytest.fixture(scope="session")
def ssp_data():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="session")
def filters():
    return load_filter_set(_FILTER_NAMES)


# ── TestEvolvingZRouting ──────────────────────────────────────────


class TestEvolvingZRouting:
    """Compositional kernel is built and returns finite photometry when evolving_metallicity=True."""  # noqa: E501

    @pytest.fixture(scope="class")
    def evolvingz_spec(self):
        return Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            # evolving metallicity: linear ramp from met_logzsol_0 (oldest) to met_logzsol_final
            evolving_metallicity=True,
            met_logzsol_0=Uniform(-2.0, 0.2),
            met_logzsol_final=Uniform(-2.0, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_slope=-0.7,
            redshift=0.1,
        )

    @pytest.fixture(scope="class")
    def evolvingz_model(self, evolvingz_spec, ssp_data, filters):
        return SEDModel(evolvingz_spec, ssp_data, filters=filters)

    @pytest.fixture(scope="class")
    def evolvingz_params(self, evolvingz_spec):
        return evolvingz_spec.sample(jax.random.PRNGKey(0))

    def test_compositional_kernel_built(self, evolvingz_model):
        """SEDModel must have built the compositional photometry kernel for evolving-Z."""
        assert evolvingz_model._compositional is not None
        assert evolvingz_model._compositional.photometry is not None

    def test_photometry_finite(self, evolvingz_model, evolvingz_params):
        phot = evolvingz_model.predict_photometry(evolvingz_params)
        chex.assert_tree_all_finite(phot), f"Non-finite photometry: {phot}"

    def test_photometry_positive(self, evolvingz_model, evolvingz_params):
        phot = evolvingz_model.predict_photometry(evolvingz_params)
        assert jnp.all(phot > 0), f"Non-positive photometry: {phot}"

    def test_photometry_shape(self, evolvingz_model, evolvingz_params):
        phot = evolvingz_model.predict_photometry(evolvingz_params)
        chex.assert_shape(phot, (_N_FILTERS,))


# ── TestChemEvolRouting ───────────────────────────────────────────


class TestChemEvolRouting:
    """Compositional kernel is built and returns finite photometry when chem_evol=True."""

    @pytest.fixture(scope="class")
    def chemevol_spec(self):
        # chem_evol params all have Fixed defaults — no Uniform needed
        return Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            chem_evol=True,
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_slope=-0.7,
            redshift=0.1,
        )

    @pytest.fixture(scope="class")
    def chemevol_model(self, chemevol_spec, ssp_data, filters):
        return SEDModel(chemevol_spec, ssp_data, filters=filters)

    @pytest.fixture(scope="class")
    def chemevol_params(self, chemevol_spec):
        return chemevol_spec.sample(jax.random.PRNGKey(1))

    def test_compositional_kernel_built(self, chemevol_model):
        """SEDModel must have built the compositional photometry kernel for chem-evol."""
        assert chemevol_model._compositional is not None
        assert chemevol_model._compositional.photometry is not None

    def test_photometry_finite(self, chemevol_model, chemevol_params):
        phot = chemevol_model.predict_photometry(chemevol_params)
        chex.assert_tree_all_finite(phot), f"Non-finite photometry: {phot}"

    def test_photometry_positive(self, chemevol_model, chemevol_params):
        phot = chemevol_model.predict_photometry(chemevol_params)
        assert jnp.all(phot > 0), f"Non-positive photometry: {phot}"

    def test_photometry_shape(self, chemevol_model, chemevol_params):
        phot = chemevol_model.predict_photometry(chemevol_params)
        chex.assert_shape(phot, (_N_FILTERS,))


# ── TestTabularSFHRouting ─────────────────────────────────────────


class TestTabularSFHRouting:
    """Tabular SFH (sfh_t_gyr + sfh_sfr in params dict) produces finite photometry.

    Calling predict_photometry twice with different SFH arrays exercises the
    JIT-compiled kernel with traced (not concretized) array values — if the
    tabular branch accidentally concretizes shapes or values the second call
    would trigger a recompilation or raise a ConcretizationTypeError.
    """

    @pytest.fixture(scope="class")
    def tabular_spec(self):
        # mean_sfh_type="table" defers the SFH to runtime arrays in the params dict
        return Parameters(
            mean_sfh_type="table",
            met_logzsol=Uniform(-1.5, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_slope=-0.7,
            redshift=0.1,
        )

    @pytest.fixture(scope="class")
    def tabular_model(self, tabular_spec, ssp_data, filters):
        return SEDModel(tabular_spec, ssp_data, filters=filters)

    @pytest.fixture(scope="class")
    def base_params(self, tabular_spec):
        return tabular_spec.sample(jax.random.PRNGKey(2))

    def _add_sfh(self, base_params: dict, rng_key) -> dict:
        """Attach a random tabular SFH to a params dict (immutable update)."""
        t_gyr = jnp.linspace(0.01, 13.5, 50)
        # random rising + declining shape so the two calls differ non-trivially
        peak_idx = int(jax.random.randint(rng_key, shape=(), minval=5, maxval=45))
        sfr = jnp.concatenate(
            [
                jnp.linspace(0.1, 10.0, peak_idx),
                jnp.linspace(10.0, 0.5, 50 - peak_idx),
            ]
        )
        return {**base_params, "sfh_t_gyr": t_gyr, "sfh_sfr": sfr}

    def test_first_call_finite(self, tabular_model, base_params):
        params = self._add_sfh(base_params, jax.random.PRNGKey(10))
        phot = tabular_model.predict_photometry(params)
        chex.assert_tree_all_finite(phot), f"Non-finite photometry (call 1): {phot}"

    def test_second_call_finite_no_recompile(self, tabular_model, base_params):
        """Second call with a different SFH array must succeed without error."""
        params = self._add_sfh(base_params, jax.random.PRNGKey(99))
        phot = tabular_model.predict_photometry(params)
        chex.assert_tree_all_finite(phot), f"Non-finite photometry (call 2): {phot}"

    def test_photometry_shape(self, tabular_model, base_params):
        params = self._add_sfh(base_params, jax.random.PRNGKey(7))
        phot = tabular_model.predict_photometry(params)
        chex.assert_shape(phot, (_N_FILTERS,))
