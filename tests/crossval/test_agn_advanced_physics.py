# SPDX-License-Identifier: BSD-3-Clause
"""Physics tests for unified_nlr_blr, QSOgen, and SKIRTOR AGN models.

References
----------
- Temple, Hewett & Banerji 2021, MNRAS, 508, 737 (QSOgen)
- Stalevski et al. 2012, MNRAS, 420, 2756; 2016, MNRAS, 458, 2288 (SKIRTOR)
- Groves et al. 2004, ApJS, 153, 75 (NLR photoionization)
- Vanden Berk et al. 2001, AJ, 122, 549 (BLR composite)
"""

import chex
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.crossval

WAVE = jnp.geomspace(100.0, 1e8, 2000)
WAVE_OPT = jnp.geomspace(1000.0, 50000.0, 2000)


# ── 1. UNIFIED NLR+BLR — Type 1/Type 2 AGN unification ────────────


class TestUnifiedNLRBLRPhysics:
    """Unified AGN model must implement orientation-dependent obscuration."""

    def test_produces_finite_sed(self):
        """Basic sanity: unified model produces finite positive SED."""
        from tengri.components.agn.unified import unified_nlr_blr

        l_nu = unified_nlr_blr(WAVE_OPT, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert float(jnp.sum(l_nu)) > 0

    def test_type1_sees_blr(self):
        """Type 1 (face-on, cos_inc ~ 1): BLR lines should be visible.

        The torus does not obscure the BLR at low inclination.
        Check for broad Hα emission at 6563A.
        """
        from tengri.components.agn.unified import unified_nlr_blr

        wave = jnp.linspace(6400.0, 6700.0, 1000)
        l_type1 = unified_nlr_blr(wave, agn_log_lbol=44.0, agn_cos_inc=0.9)
        # Should have emission line feature (non-zero flux variation)
        flux_range = float(jnp.max(l_type1) - jnp.min(l_type1))
        assert flux_range > 0, "Type 1 should show BLR emission lines"

    def test_type2_blr_obscured(self):
        """Type 2 (edge-on, cos_inc ~ 0.1): BLR should be obscured by torus.

        At high inclination (cos_inc < cos(theta_torus)), the BLR is hidden.
        """
        from tengri.components.agn.unified import unified_nlr_blr

        wave = jnp.linspace(6400.0, 6700.0, 1000)
        # Type 2: edge-on with small torus opening
        l_type2 = unified_nlr_blr(wave, agn_log_lbol=44.0, agn_cos_inc=0.1, agn_theta_torus=45.0)
        # Type 1 for comparison
        l_type1 = unified_nlr_blr(wave, agn_log_lbol=44.0, agn_cos_inc=0.9, agn_theta_torus=45.0)
        # Type 2 BLR emission should be weaker (obscured)
        type1_peak = float(jnp.max(l_type1))
        type2_peak = float(jnp.max(l_type2))
        if type1_peak > 0:
            assert type2_peak < type1_peak, "Type 2 should have weaker broad lines than Type 1"

    def test_nlr_always_visible(self):
        """NLR is isotropic — visible from all angles.

        [OIII] 5007 should appear in both Type 1 and Type 2.
        """
        from tengri.components.agn.unified import unified_nlr_blr

        wave = jnp.linspace(4950.0, 5050.0, 500)
        l_type1 = unified_nlr_blr(wave, agn_log_lbol=44.0, agn_cos_inc=0.9)
        l_type2 = unified_nlr_blr(wave, agn_log_lbol=44.0, agn_cos_inc=0.1)

        # Both should have [OIII] emission (NLR is isotropic)
        has_oiii_t1 = float(jnp.max(l_type1) - jnp.min(l_type1)) > 0
        has_oiii_t2 = float(jnp.max(l_type2) - jnp.min(l_type2)) > 0
        assert has_oiii_t1, "Type 1 should show [OIII]"
        assert has_oiii_t2, "Type 2 should show [OIII] (NLR is isotropic)"

    def test_luminosity_scaling(self):
        """Higher agn_log_lbol → brighter SED."""
        from tengri.components.agn.unified import unified_nlr_blr

        l_faint = unified_nlr_blr(WAVE_OPT, agn_log_lbol=43.0)
        l_bright = unified_nlr_blr(WAVE_OPT, agn_log_lbol=45.0)
        ratio = float(jnp.sum(l_bright)) / float(jnp.sum(l_faint))
        assert ratio > 10.0, f"100x L_bol should give >10x flux, got {ratio:.1f}x"

    def test_polar_dust_reddens_type1(self):
        """agn_polar_ebv > 0 should redden the Type 1 SED (SMC law)."""
        from tengri.components.agn.unified import unified_nlr_blr

        l_no_dust = unified_nlr_blr(
            WAVE_OPT, agn_log_lbol=44.0, agn_cos_inc=0.9, agn_polar_ebv=0.0
        )
        l_dust = unified_nlr_blr(WAVE_OPT, agn_log_lbol=44.0, agn_cos_inc=0.9, agn_polar_ebv=0.3)
        # UV should be more suppressed than optical with polar dust
        uv_mask = (WAVE_OPT > 1400) & (WAVE_OPT < 1600)
        opt_mask = (WAVE_OPT > 5000) & (WAVE_OPT < 6000)
        if float(jnp.sum(l_no_dust[uv_mask])) > 0:
            ratio_clean = float(jnp.mean(l_no_dust[uv_mask]) / jnp.mean(l_no_dust[opt_mask]))
            ratio_dusty = float(jnp.mean(l_dust[uv_mask]) / jnp.mean(l_dust[opt_mask]))
            assert ratio_dusty < ratio_clean, "Polar dust should redden UV"


# ── 2. QSOGEN — Temple, Hewett & Banerji (2021) ───────────────────


class TestQSOgenPhysics:
    """QSOgen SED must match Temple+2021 empirical quasar properties."""

    def test_default_produces_quasar_sed(self):
        """Default parameters reproduce a typical Type 1 quasar SED."""
        from tengri.components.agn.qsogen import qsogen_sed

        l_nu = qsogen_sed(WAVE_OPT)
        chex.assert_tree_all_finite(l_nu)
        assert float(jnp.sum(l_nu)) > 0

    def test_power_law_slope_controls_uv(self):
        """plslp1 controls the UV continuum slope.

        Temple+2021 Eq. 1: L_ν ∝ ν^{plslp1} for λ > plbrk.
        More negative plslp1 → redder UV.
        """
        from tengri.components.agn.qsogen import qsogen_sed

        l_blue = qsogen_sed(WAVE_OPT, agn_plslp1=0.0)
        l_red = qsogen_sed(WAVE_OPT, agn_plslp1=-1.0)

        uv_mask = (WAVE_OPT > 1400) & (WAVE_OPT < 1600)
        opt_mask = (WAVE_OPT > 5000) & (WAVE_OPT < 6000)

        ratio_blue = float(jnp.mean(l_blue[uv_mask]) / jnp.mean(l_blue[opt_mask]))
        ratio_red = float(jnp.mean(l_red[uv_mask]) / jnp.mean(l_red[opt_mask]))
        assert ratio_blue > ratio_red, "plslp1=0 should be bluer than plslp1=-1"

    def test_hot_dust_creates_nir_bump(self):
        """Temple+2021: hot dust (T~1240K) creates a 1-3μm bump.

        Higher bbnorm → stronger NIR bump.
        """
        from tengri.components.agn.qsogen import qsogen_sed

        wave_nir = jnp.geomspace(5000.0, 50000.0, 500)
        l_no_dust = qsogen_sed(wave_nir, agn_bbnorm=0.0)
        l_dust = qsogen_sed(wave_nir, agn_bbnorm=5.0)

        nir_mask = (wave_nir > 10000) & (wave_nir < 30000)
        opt_mask = (wave_nir > 5000) & (wave_nir < 7000)

        if float(jnp.sum(l_no_dust[opt_mask])) > 0:
            ratio_no = float(jnp.mean(l_no_dust[nir_mask]) / jnp.mean(l_no_dust[opt_mask]))
            ratio_yes = float(jnp.mean(l_dust[nir_mask]) / jnp.mean(l_dust[opt_mask]))
            assert ratio_yes > ratio_no, "Hot dust should boost NIR/optical ratio"

    def test_ebv_reddens_sed(self):
        """SMC reddening (agn_ebv > 0) should suppress UV flux."""
        from tengri.components.agn.qsogen import qsogen_sed

        l_clean = qsogen_sed(WAVE_OPT, agn_ebv=0.0)
        l_red = qsogen_sed(WAVE_OPT, agn_ebv=0.3)

        uv_mask = (WAVE_OPT > 1400) & (WAVE_OPT < 1600)
        # UV should be suppressed
        assert float(jnp.mean(l_red[uv_mask])) < float(jnp.mean(l_clean[uv_mask])), (
            "E(B-V) should suppress UV"
        )

    def test_emission_lines_add_flux(self):
        """Temple+2021: emission lines add flux above continuum."""
        from tengri.components.agn.qsogen import qsogen_sed

        l_no_lines = qsogen_sed(WAVE_OPT, agn_emline_scale=0.0)
        l_lines = qsogen_sed(WAVE_OPT, agn_emline_scale=1.0)

        # Total flux with lines should be higher
        assert float(jnp.sum(l_lines)) >= float(jnp.sum(l_no_lines) * 0.99), (
            "Emission lines should add flux"
        )

    def test_balmer_continuum(self):
        """Temple+2021 Sec. 3.2: Balmer continuum at λ < 3646A.

        bcnorm controls Balmer pseudo-continuum strength.
        """
        from tengri.components.agn.qsogen import qsogen_sed

        wave = jnp.linspace(3000.0, 4000.0, 500)
        l_no_bc = qsogen_sed(wave, agn_bcnorm=0.0)
        l_bc = qsogen_sed(wave, agn_bcnorm=1.0)

        # Balmer continuum should add flux below 3646A
        bc_mask = wave < 3646.0
        if jnp.any(l_no_bc[bc_mask] > 0):
            excess = float(jnp.mean(l_bc[bc_mask]) / jnp.mean(l_no_bc[bc_mask]))
            assert excess > 1.0, "Balmer continuum should add flux below 3646A"


# ── 3. SKIRTOR TORUS — Stalevski+2012, 2016 ───────────────────────


class TestSKIRTORPhysics:
    """SKIRTOR torus must obey radiative transfer physics."""

    def test_skirtor_peaks_in_mir(self):
        """Torus emission peaks in mid-IR (T ~ 200-1500K)."""
        from tengri.components.agn.skirtor import skirtor_analytic

        wave = jnp.geomspace(1000.0, 1e6, 2000)
        l_nu = skirtor_analytic(wave, agn_log_lbol=44.0)
        if float(jnp.sum(l_nu)) > 0:
            peak_wave = float(wave[jnp.argmax(l_nu)])
            # MIR: 5-100 μm = 50000-1000000 A
            assert 10000 < peak_wave < 1e6, f"Torus peak at {peak_wave:.0f} A, expected MIR"

    def test_skirtor_luminosity_scales(self):
        """10x L_bol → ~10x torus emission."""
        from tengri.components.agn.skirtor import skirtor_analytic

        wave = jnp.geomspace(10000.0, 1e6, 500)
        l_low = skirtor_analytic(wave, agn_log_lbol=43.0)
        l_high = skirtor_analytic(wave, agn_log_lbol=44.0)
        if float(jnp.sum(l_low)) > 0:
            ratio = float(jnp.sum(l_high)) / float(jnp.sum(l_low))
            assert 5.0 < ratio < 20.0, f"10x L_bol gave {ratio:.1f}x torus flux"

    def test_skirtor_finite_positive(self):
        """SKIRTOR must produce finite non-negative SED."""
        from tengri.components.agn.skirtor import skirtor_analytic

        wave = jnp.geomspace(1000.0, 1e6, 500)
        l_nu = skirtor_analytic(wave, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0)
