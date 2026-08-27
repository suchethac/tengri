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
        """Issue's spec (free lbol, no active blocks) raises ConfigError with measured message."""
        with pytest.raises(ConfigError) as exc_info, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SEDModel(issue_spec, synthetic_ssp_wide, filters=simple_filters)

        error_msg = str(exc_info.value)
        assert "agn_log_lbol" in error_msg
        assert "agn_ir_frac" in error_msg
        assert "identical" in error_msg, (
            "Guard should report the measured result: SED is identical at bounds"
        )

    def test_nlr_block_active_allows_build(self, synthetic_ssp_wide):
        """With active NLR block, model builds and lbol moves photometry.

        The NLR block reads agn_log_lbol, so the direction is live even under
        cigale_joint + skirtor. Uses a rest-frame top-hat filter covering H-beta
        and [OIII] at z=0.3 where emission-line photometry is sensitive to lbol.
        """
        from tengri.observation.photometry import FilterCurve

        # H-beta (4861 A) and [OIII] (4959, 5007 A) at z=0.3 rest-frame coverage
        # Top-hat from 6300 to 6600 A (rest-frame) to span the doublet
        rest_center = 6450.0  # Angstrom, rest-frame
        wave_rest = jnp.linspace(6300.0, 6600.0, 50)
        filters = [FilterCurve(wave=wave_rest, trans=jnp.ones(50) * 0.7, name="hbeta_oiii")]

        # Spec with active NLR (analytic block): free agn_log_lbol, fixed agn_ir_frac
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(10.5),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.3),
            agn_model="composable",
            agn_disc_block="multicolor",
            agn_torus_block="skirtor",
            agn_nlr_block="analytic",  # NLR block is active
            agn_blr_block="none",
            agn_feii_block="none",
            agn_log_lbol=Uniform(9.0, 12.0),
            agn_tau_skirtor=Fixed(3.0),
            agn_p_skirtor=Fixed(1.0),
            agn_q_skirtor=Fixed(1.0),
            agn_oa_skirtor=Fixed(40.0),
            agn_cos_inc=Fixed(0.7),
            agn_torus_frac=Fixed(0.5),
            agn_alpha=Fixed(-1.0),
            agn_polar_ebv=Fixed(0.0),  # Polar block inactive
            agn_polar_oa=Fixed(60.0),
            agn_ir_frac=Fixed(0.2),  # Fixed to disable CIGALE tie
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(spec, synthetic_ssp_wide, filters=filters)

        # NLR block reads agn_log_lbol; photometry should change
        params = spec.sample(jax.random.PRNGKey(42))
        params_low = {**params, "agn_log_lbol": 9.0}
        params_high = {**params, "agn_log_lbol": 12.0}

        phot_low = model.predict_photometry(params_low)
        phot_high = model.predict_photometry(params_high)

        # NLR reads lbol and should produce different line flux
        rel_change = jnp.max(jnp.abs(phot_high - phot_low)) / (
            jnp.max(jnp.abs(phot_high)) + 1e-300
        )
        assert rel_change > 1e-3, (
            "NLR block active; agn_log_lbol should move photometry by >0.1%. "
            f"Relative change: {rel_change:.6e}\n"
            f"Low (lbol=9.0): {phot_low}\n"
            f"High (lbol=12.0): {phot_high}"
        )

    def test_fixed_agn_ir_frac_allows_build(self, issue_spec, synthetic_ssp_wide, simple_filters):
        """agn_ir_frac=Fixed(0.0) disables CIGALE tie; model builds and lbol moves photometry."""
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
            agn_ir_frac=Fixed(0.0),  # Disable CIGALE coupling
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(spec, synthetic_ssp_wide, filters=simple_filters)

        params = spec.sample(jax.random.PRNGKey(42))
        params_low = {**params, "agn_log_lbol": 9.0}
        params_high = {**params, "agn_log_lbol": 12.0}

        phot_low = model.predict_photometry(params_low)
        phot_high = model.predict_photometry(params_high)

        # agn_ir_frac=0.0 disables the CIGALE tie, so lbol should affect photometry
        assert jnp.all(phot_high > phot_low), (
            f"agn_ir_frac=Fixed(0.0) disables CIGALE tie; "
            f"agn_log_lbol should move all bands. Low: {phot_low}, High: {phot_high}"
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

    def test_issue_spec_filterless_raises_with_measured_error(
        self, issue_spec, synthetic_ssp_wide
    ):
        """Filterless build (filters=None) raises ConfigError (flat), not ValueError."""
        with pytest.raises(ConfigError) as exc_info, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SEDModel(issue_spec, synthetic_ssp_wide, filters=None, observation=None)

        error_msg = str(exc_info.value)
        assert "bit-identical" in error_msg or "identical to within 1e-10" in error_msg, (
            "Guard must raise ConfigError (flat direction), not ValueError (no filters). "
            f"Got: {error_msg}"
        )
        assert "agn_log_lbol" in error_msg

    def test_filterless_nlr_block_with_fixed_ir_frac_builds(self, synthetic_ssp_wide):
        """Filterless build (filters=None) with NLR block and fixed agn_ir_frac builds."""
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(10.5),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.3),
            agn_model="composable",
            agn_disc_block="multicolor",
            agn_torus_block="skirtor",
            agn_nlr_block="analytic",
            agn_blr_block="none",
            agn_feii_block="none",
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
            agn_ir_frac=Fixed(0.2),  # Fixed to disable CIGALE tie
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(spec, synthetic_ssp_wide, filters=None, observation=None)

        # Model builds successfully; NLR block keeps direction live
        params = spec.sample(jax.random.PRNGKey(42))
        sed_result = model._predict_rest_sed(params)
        assert sed_result.sed is not None
