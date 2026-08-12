# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for the special SFH/metallicity prediction modes.

Tests cover:
- evolving_metallicity=True: photometry is finite AND the evolving-Z ramp
  measurably changes the prediction (not a silent no-op)
- chem_evol=True: photometry is finite AND chem_yield changes the prediction
  under jit (regression lock for the float()-on-tracer ConcretizationTypeError)
- Tabular SFH (sfh_t_gyr + sfh_sfr) and tabular metallicity (met_history):
  runtime simulation tables through the CIC path (#996) — finite photometry,
  no recompile across tables, equivalence with the parametric twin, and loud
  guards for the silently-ignored-key combinations

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
    """Runtime tabular SFH + metallicity through the component chain (#996).

    The tables are runtime params (traced values, static shapes): cosmic-time
    grid ``sfh_t_gyr`` [Gyr], ``sfh_sfr`` [Msun/yr], and optionally
    ``met_history`` = log10(Z/Zsun) at the same nodes (met_mode='table').
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
        assert jnp.all(jnp.isfinite(phot)), f"Non-finite photometry (call 1): {phot}"

    def test_second_call_finite_no_recompile(self, tabular_model, base_params):
        """Second call with a different SFH array must succeed without error."""
        params = self._add_sfh(base_params, jax.random.PRNGKey(99))
        phot = tabular_model.predict_photometry(params)
        assert jnp.all(jnp.isfinite(phot)), f"Non-finite photometry (call 2): {phot}"

    def test_photometry_shape(self, tabular_model, base_params):
        params = self._add_sfh(base_params, jax.random.PRNGKey(7))
        phot = tabular_model.predict_photometry(params)
        chex.assert_shape(phot, (_N_FILTERS,))

    def test_table_matches_parametric_twin(self, ssp_data, filters):
        """A densely tabulated delayed-tau SFH reproduces the parametric model.

        Code-independent equivalence: evaluate the registry's delayed-tau on a
        dense lookback grid, convert to the table's cosmic-time convention,
        and feed it through sfh_model='table'. Both routes then share the CIC
        machinery, so the only difference is the table interp — which the
        dense sampling makes negligible.
        """
        from tengri.components.stellar.sfh.registry import SFH_REGISTRY
        from tengri.cosmology import age_at_z

        common = dict(
            met_logzsol=-0.3,
            dust_tau_bc=0.3,
            dust_tau_diff=0.2,
            dust_slope=-0.7,
            redshift=0.1,
        )
        spec_par = Parameters(
            mean_sfh_type="delayed",
            sfh_delayed_log_total_mass=10.0,
            sfh_delayed_tau_gyr=2.0,
            sfh_delayed_age_gyr=8.0,
            **common,
        )
        m_par = SEDModel(spec_par, ssp_data, filters=filters)
        phot_par = m_par.predict_photometry({})

        # Tabulate the same SFH densely: registry fn takes lookback [yr].
        t_obs_gyr = float(age_at_z(common["redshift"]))
        t_gyr = jnp.linspace(1e-4, t_obs_gyr, 4000)  # cosmic time
        lbt_yr = (t_obs_gyr - t_gyr) * 1e9
        delayed = SFH_REGISTRY["delayed"]
        kw = {}
        for public, (internal, scale, offset) in delayed.internal_param_map.items():
            val = {
                "sfh_delayed_log_total_mass": 10.0,
                "sfh_delayed_tau_gyr": 2.0,
                "sfh_delayed_age_gyr": 8.0,
            }[public]
            kw[internal] = jnp.asarray(val) * scale + offset
        sfr = delayed.fn(lbt_yr, **kw)
        # The registry fn returns an unnormalized shape (the parametric path
        # renormalizes to 10**log_total_mass downstream); the table path
        # correctly preserves the table's ABSOLUTE normalization, so scale
        # the tabulated shape to the twin's total mass here.
        mass = jnp.trapezoid(sfr[::-1], lbt_yr[::-1])
        sfr = sfr * (10.0**10.0 / mass)

        spec_tab = Parameters(mean_sfh_type="table", **common)
        m_tab = SEDModel(spec_tab, ssp_data, filters=filters)
        phot_tab = m_tab.predict_photometry({"sfh_t_gyr": t_gyr, "sfh_sfr": sfr})

        ratio_dev = float(jnp.max(jnp.abs(phot_tab / phot_par - 1.0)))
        assert ratio_dev < 5e-3, (
            f"tabulated delayed-tau deviates from the parametric twin by "
            f"{ratio_dev:.2e} (max band ratio)"
        )

    def test_met_history_constant_reduces_to_delta(self, ssp_data, filters):
        """Constant met_history == delta metallicity (degenerate reduction).

        A flat Z(t) table must reproduce the delta-metallicity model at the
        same log10(Z/Zsun) — the same degeneracy contract the per-age modes
        satisfy (#964)."""
        common = dict(
            dust_tau_bc=0.3,
            dust_tau_diff=0.2,
            dust_slope=-0.7,
            redshift=0.1,
        )
        t_gyr = jnp.linspace(0.01, 13.4, 200)
        sfr = jnp.exp(-((t_gyr - 5.0) ** 2) / 8.0) * 5.0
        table = {"sfh_t_gyr": t_gyr, "sfh_sfr": sfr}

        spec_delta = Parameters(mean_sfh_type="table", met_logzsol=-0.3, **common)
        m_delta = SEDModel(spec_delta, ssp_data, filters=filters)
        phot_delta = m_delta.predict_photometry(table)

        spec_hist = Parameters(mean_sfh_type="table", met_mode="table", **common)
        m_hist = SEDModel(spec_hist, ssp_data, filters=filters)
        phot_hist = m_hist.predict_photometry(
            {**table, "met_history": jnp.full(t_gyr.shape, -0.3)}
        )

        ratio_dev = float(jnp.max(jnp.abs(phot_hist / phot_delta - 1.0)))
        assert ratio_dev < 1e-3, (
            f"constant met_history deviates from delta metallicity by "
            f"{ratio_dev:.2e} (max band ratio)"
        )

    def test_table_keys_on_parametric_model_raise_loud(self, ssp_data, filters):
        """sfh_t_gyr on a non-table model must raise, not be silently ignored."""
        spec = Parameters(
            mean_sfh_type="delayed",
            sfh_delayed_log_total_mass=10.0,
            sfh_delayed_tau_gyr=2.0,
            sfh_delayed_age_gyr=8.0,
            met_logzsol=-0.3,
            # Fix dust to the off-state so the model has no free parameters: the
            # table-key rejection below must be what raises, not the missing-
            # parameter guard fielding an unset dust_tau_bc/dust_tau_diff.
            dust_tau_bc=0.0,
            dust_tau_diff=0.0,
            redshift=0.1,
        )
        m = SEDModel(spec, ssp_data, filters=filters)
        with pytest.raises(NotImplementedError, match="silently"):
            m.predict_photometry(
                {
                    "sfh_t_gyr": jnp.linspace(0.1, 13.0, 20),
                    "sfh_sfr": jnp.ones(20),
                }
            )

    def test_met_history_without_table_met_mode_raises_loud(self, tabular_model, base_params):
        """met_history under delta metallicity must raise, not silently drop."""
        params = self._add_sfh(base_params, jax.random.PRNGKey(4))
        with pytest.raises(NotImplementedError, match="met_history"):
            tabular_model.predict_photometry({**params, "met_history": jnp.full((50,), -0.3)})
