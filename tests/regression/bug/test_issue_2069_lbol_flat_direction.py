# SPDX-License-Identifier: BSD-3-Clause
"""Tests for issue #2069: free agn_log_lbol under cigale_joint is flat when no blocks consume it.

When agn_norm='cigale_joint' (default), the disc/torus/polar are tied to a single
agn_power reference via fixed SKIRTOR template ratios. Under this policy, moving
agn_log_lbol has no effect when agn_ir_frac>0 (the CIGALE coupling is active) AND
no other blocks (polar, NLR, BLR, FeII) consume agn_log_lbol. This creates a flat
likelihood direction. Blocks that consume agn_log_lbol (polar, NLR, BLR, FeII)
keep the direction live even under cigale_joint + skirtor.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri import Fixed, Parameters, SEDModel, Uniform
from tengri.config.exceptions import ConfigError

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def simple_filters():
    """Synthetic 3-band filter set: optical (where polar/AGN blocks show)."""
    from tengri.observation.photometry import FilterCurve

    waves = [
        jnp.linspace(3500.0, 4500.0, 50),
        jnp.linspace(5000.0, 6500.0, 50),
        jnp.linspace(7500.0, 9000.0, 50),
    ]
    trans = [jnp.ones(50) * 0.5 for _ in range(3)]
    names = ["synth_blue", "synth_green", "synth_red"]
    curves = [FilterCurve(wave=w, trans=t, name=n) for n, w, t in zip(names, waves, trans)]
    return (waves, trans, curves)


@pytest.fixture(scope="module")
def issue_spec():
    """Issue's spec: skirtor with free agn_log_lbol, no polar/NLR/BLR/FeII blocks."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.3),
        agn_model="skirtor",
        agn_log_lbol=Uniform(9.0, 12.0),
        agn_tau_skirtor=Fixed(3.0),
        agn_p_skirtor=Fixed(1.0),
        agn_q_skirtor=Fixed(1.0),
        agn_oa_skirtor=Fixed(40.0),
        agn_cos_inc=Fixed(0.7),
        agn_torus_frac=Fixed(0.5),
        agn_alpha=Fixed(-1.0),
        agn_polar_ebv=Fixed(0.0),  # Polar block inert (key!)
        agn_polar_oa=Fixed(60.0),
    )


class TestLBolFlatDirectionRefusal:
    """Regression for #2069: flat agn_log_lbol direction when all consumers inactive."""

    def test_issue_spec_raises_with_message(self, issue_spec, synthetic_ssp_wide, simple_filters):
        """Issue's spec (free lbol, no active blocks) raises ConfigError."""
        with pytest.raises(ConfigError) as exc_info, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SEDModel(issue_spec, synthetic_ssp_wide, filters=simple_filters)

        error_msg = str(exc_info.value)
        assert "agn_log_lbol" in error_msg
        assert "agn_ir_frac" in error_msg
        assert "polar" in error_msg.lower() or "block" in error_msg.lower()

    def test_polar_block_active_allows_build(self, issue_spec, synthetic_ssp_wide, simple_filters):
        """With active polar block (agn_polar_ebv>0), model builds and lbol moves photometry."""
        import jax

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.3),
            agn_model="skirtor",
            agn_log_lbol=Uniform(9.0, 12.0),
            agn_tau_skirtor=Fixed(3.0),
            agn_p_skirtor=Fixed(1.0),
            agn_q_skirtor=Fixed(1.0),
            agn_oa_skirtor=Fixed(40.0),
            agn_cos_inc=Fixed(0.7),
            agn_torus_frac=Fixed(0.5),
            agn_alpha=Fixed(-1.0),
            agn_polar_ebv=Fixed(0.3),  # Polar block active!
            agn_polar_oa=Fixed(60.0),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(spec, synthetic_ssp_wide, filters=simple_filters)

        key = jax.random.PRNGKey(42)
        params = spec.sample(key)
        params_low = {**params, "agn_log_lbol": 9.0}
        params_high = {**params, "agn_log_lbol": 12.0, "agn_ir_frac": 0.122}

        phot_low = model.predict_photometry(params_low)
        phot_high = model.predict_photometry(params_high)

        # Polar block reads agn_log_lbol, so it should respond
        assert jnp.any(phot_high != phot_low), (
            f"Polar block active; agn_log_lbol should move photometry. "
            f"Low: {phot_low}, High: {phot_high}"
        )

    def test_fixed_agn_ir_frac_allows_build(self, issue_spec, synthetic_ssp_wide, simple_filters):
        """With agn_ir_frac=Fixed(0.0), CIGALE tie disabled, model builds."""
        import jax

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.3),
            agn_model="skirtor",
            agn_log_lbol=Uniform(9.0, 12.0),
            agn_tau_skirtor=Fixed(3.0),
            agn_p_skirtor=Fixed(1.0),
            agn_q_skirtor=Fixed(1.0),
            agn_oa_skirtor=Fixed(40.0),
            agn_cos_inc=Fixed(0.7),
            agn_torus_frac=Fixed(0.5),
            agn_alpha=Fixed(-1.0),
            agn_polar_ebv=Fixed(0.0),
            agn_polar_oa=Fixed(60.0),
            agn_ir_frac=Fixed(0.0),  # Disable coupling
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(spec, synthetic_ssp_wide, filters=simple_filters)

        key = jax.random.PRNGKey(42)
        params = spec.sample(key)
        params_low = {**params, "agn_log_lbol": 9.0}
        params_high = {**params, "agn_log_lbol": 12.0}

        phot_low = model.predict_photometry(params_low)
        phot_high = model.predict_photometry(params_high)

        assert jnp.all(phot_high > phot_low), (
            f"agn_ir_frac=0.0 disables tie; lbol should move. Low: {phot_low}, High: {phot_high}"
        )

    def test_independent_norm_allows_build(self, issue_spec, synthetic_ssp_wide, simple_filters):
        """With agn_norm='independent', model builds."""
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.3),
            agn_model="skirtor",
            agn_log_lbol=Uniform(9.0, 12.0),
            agn_tau_skirtor=Fixed(3.0),
            agn_p_skirtor=Fixed(1.0),
            agn_q_skirtor=Fixed(1.0),
            agn_oa_skirtor=Fixed(40.0),
            agn_cos_inc=Fixed(0.7),
            agn_torus_frac=Fixed(0.5),
            agn_alpha=Fixed(-1.0),
            agn_polar_ebv=Fixed(0.0),
            agn_polar_oa=Fixed(60.0),
            agn_norm="independent",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(spec, synthetic_ssp_wide, filters=simple_filters)

        key = jax.random.PRNGKey(42)
        params = spec.sample(key)
        phot = model.predict_photometry(params)
        assert jnp.all(jnp.isfinite(phot))

    def test_fixed_agn_log_lbol_allows_build(self, issue_spec, synthetic_ssp_wide, simple_filters):
        """With agn_log_lbol=Fixed(10.5), model builds."""
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.3),
            agn_model="skirtor",
            agn_log_lbol=Fixed(10.5),
            agn_tau_skirtor=Fixed(3.0),
            agn_p_skirtor=Fixed(1.0),
            agn_q_skirtor=Fixed(1.0),
            agn_oa_skirtor=Fixed(40.0),
            agn_cos_inc=Fixed(0.7),
            agn_torus_frac=Fixed(0.5),
            agn_alpha=Fixed(-1.0),
            agn_polar_ebv=Fixed(0.0),
            agn_polar_oa=Fixed(60.0),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(spec, synthetic_ssp_wide, filters=simple_filters)

        key = jax.random.PRNGKey(42)
        params = spec.sample(key)
        phot = model.predict_photometry(params)
        assert jnp.all(jnp.isfinite(phot))
