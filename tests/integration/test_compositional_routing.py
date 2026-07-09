# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for the special SFH/metallicity prediction modes.

Tests cover:
- evolving_metallicity=True: photometry is finite AND the evolving-Z ramp
  measurably changes the prediction (not a silent no-op)
- chem_evol=True: photometry is finite AND chem_yield changes the prediction
  under jit (regression lock for the float()-on-tracer ConcretizationTypeError)
- Tabular SFH (sfh_t_gyr + sfh_sfr): the component chain does not implement it
  yet (orphaned by the kernel deletion) — the loud NotImplementedError gate is
  locked so it can never silently return the registry placeholder's zeros

The pre-ADR-0019 ``_compositional`` kernel attribute these tests once asserted
was removed when the compositional/hybrid/exact modes collapsed into the one
component-chain path; the routing guarantee is now behavioral, not structural.

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
    """Evolving-Z ramp: finite photometry and a measurable (non-no-op) effect."""

    @pytest.fixture(scope="class")
    def evolvingz_spec(self):
        return Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(-1.0, 2.5),
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

    def test_evolving_z_ramp_is_not_a_noop(self, evolvingz_model, evolvingz_params):
        """The evolving-Z ramp must measurably change the photometry.

        A flat ramp (met_logzsol_0 == met_logzsol_final) and a strongly
        evolving one must differ — guards the routing the retired
        ``_compositional`` attribute assertion used to (indirectly) cover.
        """
        flat = {**evolvingz_params, "met_logzsol_0": -0.5, "met_logzsol_final": -0.5}
        ramp = {**evolvingz_params, "met_logzsol_0": -2.0, "met_logzsol_final": 0.2}
        phot_flat = evolvingz_model.predict_photometry(flat)
        phot_ramp = evolvingz_model.predict_photometry(ramp)
        # Ratio-based: AB flux densities are ~1e-27 erg/s/cm2/Hz, far below
        # allclose's default atol=1e-8 — allclose is vacuously True on them.
        max_ratio_dev = float(jnp.max(jnp.abs(phot_ramp / phot_flat - 1.0)))
        assert max_ratio_dev > 1e-3, (
            f"evolving_metallicity=True looks like a silent no-op: flat vs "
            f"strongly evolving Z ramp differ by max {max_ratio_dev:.2e} (ratio)."
        )

    def test_photometry_finite(self, evolvingz_model, evolvingz_params):
        phot = evolvingz_model.predict_photometry(evolvingz_params)
        assert jnp.all(jnp.isfinite(phot)), f"Non-finite photometry: {phot}"

    def test_photometry_positive(self, evolvingz_model, evolvingz_params):
        phot = evolvingz_model.predict_photometry(evolvingz_params)
        assert jnp.all(phot > 0), f"Non-positive photometry: {phot}"

    def test_photometry_shape(self, evolvingz_model, evolvingz_params):
        phot = evolvingz_model.predict_photometry(evolvingz_params)
        chex.assert_shape(phot, (_N_FILTERS,))


# ── TestChemEvolRouting ───────────────────────────────────────────


class TestChemEvolRouting:
    """Chem-evol mode: finite photometry; chem_yield acts through jit."""

    @pytest.fixture(scope="class")
    def chemevol_spec(self):
        # chem_evol params all have Fixed defaults — no Uniform needed
        return Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(-1.0, 2.5),
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

    def test_chem_yield_is_not_a_noop_under_jit(self, chemevol_model, chemevol_params):
        """chem_yield must change the photometry, traced through jit.

        Regression lock: the chem-evol branch cast its params with float(),
        which raises ConcretizationTypeError on traced values — and would
        freeze the closed-box yield at its default if ever eagerly bypassed.
        Two different yields must give different photometry.
        """
        low = {**chemevol_params, "chem_yield": 0.01}
        high = {**chemevol_params, "chem_yield": 0.06}
        phot_low = chemevol_model.predict_photometry(low)
        phot_high = chemevol_model.predict_photometry(high)
        # Ratio-based (see evolving-Z test: allclose is vacuous at ~1e-27).
        max_ratio_dev = float(jnp.max(jnp.abs(phot_high / phot_low - 1.0)))
        assert max_ratio_dev > 1e-3, (
            f"chem_yield looks like a silent no-op: yields 0.01 vs 0.06 differ "
            f"by max {max_ratio_dev:.2e} (ratio) through predict_photometry."
        )

    def test_photometry_finite(self, chemevol_model, chemevol_params):
        phot = chemevol_model.predict_photometry(chemevol_params)
        assert jnp.all(jnp.isfinite(phot)), f"Non-finite photometry: {phot}"

    def test_photometry_positive(self, chemevol_model, chemevol_params):
        phot = chemevol_model.predict_photometry(chemevol_params)
        assert jnp.all(phot > 0), f"Non-positive photometry: {phot}"

    def test_photometry_shape(self, chemevol_model, chemevol_params):
        phot = chemevol_model.predict_photometry(chemevol_params)
        chex.assert_shape(phot, (_N_FILTERS,))


# ── TestTabularSFHRouting ─────────────────────────────────────────


class TestTabularSFHRouting:
    """Tabular SFH (sfh_t_gyr + sfh_sfr) — orphaned feature, gate locked.

    The runtime-table SFH lost its execution path when the legacy kernels
    were deleted (ADR-0019 one-path migration): the registry entry is a
    zeros placeholder ("actual tabulated SFH handled separately") and the
    component chain gates ``sfh_model='table'`` with NotImplementedError.
    Until the port lands, the one behavior worth locking is that the gate
    stays LOUD — a model built with ``mean_sfh_type='table'`` must raise,
    never silently predict from the placeholder's all-zero SFH.

    The three functional tests are preserved, skipped, as the port's
    acceptance criteria (finite photometry, no recompile across different
    tables, correct shape).
    """

    _PORT_SKIP = (
        "tabular SFH awaits its component-chain port (execution path deleted "
        "with the legacy kernels; registry entry is a zeros placeholder) — "
        "these are the port's acceptance tests"
    )

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

    def test_table_gate_is_loud(self, tabular_spec, ssp_data, filters, base_params):
        """The unported table mode must raise, never silently predict zeros."""
        with pytest.raises(NotImplementedError, match="table"):
            model = SEDModel(tabular_spec, ssp_data, filters=filters)
            params = self._add_sfh(base_params, jax.random.PRNGKey(10))
            model.predict_photometry(params)

    @pytest.mark.skip(reason=_PORT_SKIP)
    def test_first_call_finite(self, tabular_model, base_params):
        params = self._add_sfh(base_params, jax.random.PRNGKey(10))
        phot = tabular_model.predict_photometry(params)
        assert jnp.all(jnp.isfinite(phot)), f"Non-finite photometry (call 1): {phot}"

    @pytest.mark.skip(reason=_PORT_SKIP)
    def test_second_call_finite_no_recompile(self, tabular_model, base_params):
        """Second call with a different SFH array must succeed without error."""
        params = self._add_sfh(base_params, jax.random.PRNGKey(99))
        phot = tabular_model.predict_photometry(params)
        assert jnp.all(jnp.isfinite(phot)), f"Non-finite photometry (call 2): {phot}"

    @pytest.mark.skip(reason=_PORT_SKIP)
    def test_photometry_shape(self, tabular_model, base_params):
        params = self._add_sfh(base_params, jax.random.PRNGKey(7))
        phot = tabular_model.predict_photometry(params)
        chex.assert_shape(phot, (_N_FILTERS,))
