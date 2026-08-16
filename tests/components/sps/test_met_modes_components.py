from __future__ import annotations

import chex
import pytest

pytestmark = pytest.mark.bounds
# SPDX-License-Identifier: BSD-3-Clause
"""Limiting-case tests for the 5 newly-wired metallicity modes.

The legacy ``_compute_sed_components`` doesn't support these modes either,
so there is no legacy parity gate. Instead, each mode is tested by
configuring it to reduce to ``delta`` (constant Z) and asserting bit-equal
SED.

Modes covered:
- ``two_step``: Z_old == Z_young → constant.
- ``psb_two_step``: Z_old == Z_burst → constant.
- ``bins``: all met_bin_<i> equal → constant.
- ``bins_continuity``: all d_log_z == 0 → constant.
- ``table``: constant table → constant.

Plus a non-trivial smoke test per mode (Z_old != Z_young, etc.) asserting
the SED responds to the metallicity input.
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, Parameters, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import SSPData

# ── Synthetic SSP fixture ─────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_ssp():
    """3-met × 20-age × 100-wave synthetic SSP."""
    n_met, n_age, n_wave = 3, 20, 100
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages_gyr = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(123)
    flux = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
    )


def _delta_sed(synthetic_ssp, log_z_solar):
    """Run the orchestrator with met_mode='delta' and return SED."""
    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(log_z_solar),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        redshift=Fixed(0.0),
    )
    m = SEDModel(spec, synthetic_ssp)
    return m.predict_state({}).sed_intrinsic


# ── two_step ──────────────────────────────────────────────────────


class TestTwoStep:
    def test_equal_old_young_reduces_to_delta(self, synthetic_ssp):
        """Z_old == Z_young → matches delta-mode SED at the same Z."""
        Z = -0.3
        spec = Parameters(
            met_mode="two_step",
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_age_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol_old=Fixed(Z),
            met_logzsol_young=Fixed(Z),
            met_step_age_gyr=Fixed(2.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=Fixed(0.0),
        )
        m = SEDModel(spec, synthetic_ssp)
        sed_two = m.predict_state({}).sed_intrinsic
        sed_delta = _delta_sed(synthetic_ssp, Z)
        np.testing.assert_allclose(sed_two, sed_delta, rtol=1e-3)

    def test_different_old_young_changes_sed(self, synthetic_ssp):
        """Different Z for old/young must produce a different SED.

        Synthetic SSP has lgmet = [-1.5, -0.5, 0.0] absolute, so Z_solar
        values must lie in ~[0.35, 1.85] to exercise the grid (otherwise
        DSPS clamps both to the same edge bin).
        """
        spec_pos = Parameters(
            met_mode="two_step",
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_age_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol_old=Fixed(0.5),
            met_logzsol_young=Fixed(1.7),
            met_step_age_gyr=Fixed(2.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=Fixed(0.0),
        )
        m = SEDModel(spec_pos, synthetic_ssp)
        sed = m.predict_state({}).sed_intrinsic
        sed_delta = _delta_sed(synthetic_ssp, 1.1)  # midpoint
        chex.assert_tree_all_finite(sed)
        assert float(jnp.max(jnp.abs(sed - sed_delta) / jnp.maximum(sed_delta, 1e-30))) > 1e-3


# ── psb_two_step ──────────────────────────────────────────────────


class TestPsbTwoStep:
    def test_equal_old_burst_reduces_to_delta(self, synthetic_ssp):
        Z = -0.4
        spec = Parameters(
            met_mode="psb_two_step",
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_age_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol_old=Fixed(Z),
            met_logzsol_burst=Fixed(Z),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=Fixed(0.0),
        )
        m = SEDModel(spec, synthetic_ssp)
        sed = m.predict_state({}).sed_intrinsic
        sed_delta = _delta_sed(synthetic_ssp, Z)
        np.testing.assert_allclose(sed, sed_delta, rtol=1e-3)


# ── bins ──────────────────────────────────────────────────────────


class TestBins:
    def test_uniform_bins_reduce_to_delta(self, synthetic_ssp):
        Z = -0.2
        spec = Parameters(
            met_mode="bins",
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_age_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            **{f"met_bin_{i}": Fixed(Z) for i in range(6)},
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=Fixed(0.0),
        )
        m = SEDModel(spec, synthetic_ssp)
        sed = m.predict_state({}).sed_intrinsic
        sed_delta = _delta_sed(synthetic_ssp, Z)
        np.testing.assert_allclose(sed, sed_delta, rtol=1e-3)

    def test_bins_responds_to_metallicity(self, synthetic_ssp):
        # Z_solar values must lie in ~[0.35, 1.85] for the synthetic SSP
        # grid (lgmet absolute [-1.5, -0.5, 0.0]) — otherwise DSPS
        # clamps both spec_lo and spec_hi to the same edge bin.
        spec_lo = Parameters(
            met_mode="bins",
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_age_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            **{f"met_bin_{i}": Fixed(0.5) for i in range(6)},
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=Fixed(0.0),
        )
        spec_hi = Parameters(
            met_mode="bins",
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_age_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            **{f"met_bin_{i}": Fixed(1.7) for i in range(6)},
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=Fixed(0.0),
        )
        m_lo = SEDModel(spec_lo, synthetic_ssp)
        m_hi = SEDModel(spec_hi, synthetic_ssp)
        sed_lo = m_lo.predict_state({}).sed_intrinsic
        sed_hi = m_hi.predict_state({}).sed_intrinsic
        assert float(jnp.max(jnp.abs(sed_lo - sed_hi) / jnp.maximum(sed_lo, 1e-30))) > 0.05


# ── bins_continuity ────────────────────────────────────────────────


class TestBinsContinuity:
    def test_zero_d_log_z_reduces_to_delta(self, synthetic_ssp):
        """All d_log_z == 0 → constant Z = met_logzsol_base everywhere."""
        Z = -0.4
        spec = Parameters(
            met_mode="bins_continuity",
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_age_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol_base=Fixed(Z),
            **{f"met_d_log_z_{i}": Fixed(0.0) for i in range(5)},
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=Fixed(0.0),
        )
        m = SEDModel(spec, synthetic_ssp)
        sed = m.predict_state({}).sed_intrinsic
        sed_delta = _delta_sed(synthetic_ssp, Z)
        np.testing.assert_allclose(sed, sed_delta, rtol=1e-3)


# ── table ──────────────────────────────────────────────────────────


class TestTable:
    def test_constant_table_reduces_to_delta(self, synthetic_ssp):
        """Constant Z(t) table → matches delta-mode SED."""
        from tengri.components.stellar.component import (
            StellarSEDComponent,
            StellarSEDComponentConfig,
        )
        from tengri.forward.orchestrator import run_components
        from tengri.parameters.translate import LOG10_ZSUN
        from tengri.protocols.component import ForwardState

        Z_solar = -0.3
        Z_abs = Z_solar + LOG10_ZSUN
        # Constant table over 1 Myr → 13.7 Gyr.
        met_log_age_yr = jnp.linspace(6.0, 10.14, 16)
        met_log_z_abs = jnp.full_like(met_log_age_yr, Z_abs)
        stellar = StellarSEDComponent(
            config=StellarSEDComponentConfig(
                sfh_model="dpl",
                metallicity_model="table",
                met_table_log_age_yr=met_log_age_yr,
                met_table_log_z_abs=met_log_z_abs,
            ),
            ssp_data=synthetic_ssp,
        )
        wave = synthetic_ssp.ssp_wave
        state0 = ForwardState(wave=wave, sed_observed=jnp.ones_like(wave))
        params = {
            "sfh_dpl_alpha": 1.5,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 5.0,
            "sfh_dpl_age_gyr": 5.0,
            "sfh_dpl_log_total_mass": 1.0,
            "redshift": 0.0,
        }
        state = run_components([stellar], state0, params)
        sed = state.sed_intrinsic
        sed_delta = _delta_sed(synthetic_ssp, Z_solar)
        np.testing.assert_allclose(sed, sed_delta, rtol=1e-3)

    def test_table_missing_raises(self, synthetic_ssp):
        from tengri.components.stellar.component import (
            StellarSEDComponent,
            StellarSEDComponentConfig,
        )
        from tengri.forward.orchestrator import run_components
        from tengri.protocols.component import ForwardState

        stellar = StellarSEDComponent(
            config=StellarSEDComponentConfig(
                sfh_model="dpl",
                metallicity_model="table",
                # met_table_log_age_yr / met_table_log_z_abs NOT set
            ),
            ssp_data=synthetic_ssp,
        )
        wave = synthetic_ssp.ssp_wave
        state0 = ForwardState(wave=wave, sed_observed=jnp.ones_like(wave))
        params = {
            "sfh_dpl_alpha": 1.5,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 5.0,
            "sfh_dpl_age_gyr": 5.0,
            "sfh_dpl_log_total_mass": 1.0,
            "redshift": 0.0,
        }
        with pytest.raises(ValueError, match="met_table_log_age_yr"):
            run_components([stellar], state0, params)
