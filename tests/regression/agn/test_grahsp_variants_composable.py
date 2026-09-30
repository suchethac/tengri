# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for GRAHSP Balmer continuum block on blr:grahsp (Item A, T1).

Tests composable-block implementation against the component API (evaluate_grahsp_agn).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.agn.blocks import composable_agn_l_nu
from tengri.components.agn.grahsp.model import GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.utils.physics_constants import C_AA
from tests._data_skip import requires_grahsp


@requires_grahsp
class TestBalmercontinuumBlrGrahsp:
    """Balmer continuum (a_bc parameter) on blr:grahsp block (Item A)."""

    @pytest.fixture
    def params(self):
        """Base GRAHSPParams with a_bc=0.0 (default)."""
        return GRAHSPParams(l5100=1e44, a_bc=0.0, a_lines=1.0, linewidth_kms=5000.0)

    @pytest.fixture
    def wave_nm(self):
        """Wavelength grid [nm] including nodes for 5100 Å."""
        return np.union1d(np.logspace(2.5, 3.5, 100), [510.0])  # 5100 A = 510 nm

    @pytest.fixture
    def templates(self):
        """Cached GRAHSP templates."""
        return load_grahsp_templates()

    def test_block_equality_a_bc_zero(self, wave_nm, templates):
        """Block at a_bc=0 equals component API (Balmer absent)."""
        wave_aa = wave_nm * 10.0
        params_obj = GRAHSPParams(l5100=1e44, a_bc=0.0, a_lines=1.0, linewidth_kms=5000.0)

        # Component API reference
        sed_ref = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_obj, templates)
        l_nu_ref = np.asarray(sed_ref.bbb_attenuated + sed_ref.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        # Composable runner (all-grahsp)
        l_nu_got = np.asarray(composable_agn_l_nu(
            jnp.asarray(wave_aa),
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_a_bc=0.0,
            agn_grahsp_a_lines=1.0,
            agn_grahsp_linewidth_kms=5000.0,
            agn_type=1,
        ))

        np.testing.assert_allclose(l_nu_got, l_nu_ref, rtol=1e-8)

    def test_block_equality_a_bc_nonzero(self, wave_nm, templates):
        """Block at a_bc=0.3 includes Balmer and matches component API."""
        wave_aa = wave_nm * 10.0
        a_bc = 0.3
        params_obj = GRAHSPParams(l5100=1e44, a_bc=a_bc, a_lines=1.0, linewidth_kms=5000.0)

        # Component API reference
        sed_ref = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_obj, templates)
        l_nu_ref = np.asarray(sed_ref.bbb_attenuated + sed_ref.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        # Composable runner with a_bc=0.3
        l_nu_got = np.asarray(composable_agn_l_nu(
            jnp.asarray(wave_aa),
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_a_bc=a_bc,
            agn_grahsp_a_lines=1.0,
            agn_grahsp_linewidth_kms=5000.0,
            agn_type=1,
        ))

        np.testing.assert_allclose(l_nu_got, l_nu_ref, rtol=1e-8)

    def test_liveness_a_bc_measurable_change(self, wave_nm, templates):
        """Parameter must move the SED by > 1e-2 relative (liveness test)."""
        wave_aa = wave_nm * 10.0

        # Evaluate at a_bc=0
        sed_a0 = evaluate_grahsp_agn(
            jnp.asarray(wave_nm),
            GRAHSPParams(l5100=1e44, a_bc=0.0, a_lines=1.0, linewidth_kms=5000.0),
            templates,
        )
        l_nu_a0 = np.asarray(sed_a0.bbb_attenuated + sed_a0.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        # Evaluate at a_bc=0.5
        sed_a5 = evaluate_grahsp_agn(
            jnp.asarray(wave_nm),
            GRAHSPParams(l5100=1e44, a_bc=0.5, a_lines=1.0, linewidth_kms=5000.0),
            templates,
        )
        l_nu_a5 = np.asarray(sed_a5.bbb_attenuated + sed_a5.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        rel_change = np.abs((l_nu_a5 - l_nu_a0) / np.maximum(np.abs(l_nu_a0), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-2, f"a_bc change too small: {max_rel:.2e}; parameter is inert"

    def test_agn_type_gate(self, wave_nm, templates):
        """Balmer must be suppressed for agn_type=2 (type-2 AGN)."""
        wave_aa = wave_nm * 10.0
        a_bc = 0.3

        # Type 1 (broad lines visible)
        params_t1 = GRAHSPParams(l5100=1e44, a_bc=a_bc, a_lines=1.0, linewidth_kms=5000.0, agn_type=1)
        sed_t1 = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_t1, templates)
        l_nu_t1 = np.asarray(sed_t1.bbb_attenuated + sed_t1.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        # Type 2 (Balmer suppressed)
        params_t2 = GRAHSPParams(l5100=1e44, a_bc=a_bc, a_lines=1.0, linewidth_kms=5000.0, agn_type=2)
        sed_t2 = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_t2, templates)
        l_nu_t2 = np.asarray(sed_t2.bbb_attenuated + sed_t2.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        # Type 2 should be less (Balmer removed)
        assert np.all(l_nu_t2 <= l_nu_t1), "Type 2 should suppress Balmer continuum"

    def test_monolithic_parity_a_bc_change(self, wave_nm, templates):
        """Monolithic must show expected change (~0.246 for a_bc 0 -> 0.5)."""
        wave_aa = wave_nm * 10.0

        # Reference: monolithic at a_bc=0
        params_a0 = GRAHSPParams(l5100=1e44, a_bc=0.0, a_lines=1.0, linewidth_kms=5000.0)
        sed_a0 = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_a0, templates)
        l_nu_a0 = np.asarray(sed_a0.bbb_attenuated + sed_a0.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        # Reference: monolithic at a_bc=0.5
        params_a5 = GRAHSPParams(l5100=1e44, a_bc=0.5, a_lines=1.0, linewidth_kms=5000.0)
        sed_a5 = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_a5, templates)
        l_nu_a5 = np.asarray(sed_a5.bbb_attenuated + sed_a5.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        rel_change = np.abs((l_nu_a5 - l_nu_a0) / np.maximum(np.abs(l_nu_a0), 1e-300))
        max_rel = np.max(rel_change)

        # Design: 0.246 (per probe p5_rootcause/r4b.py)
        assert 0.15 < max_rel < 0.35, (
            f"Monolithic a_bc change {max_rel:.3f} outside expected 0.15–0.35 range"
        )


@requires_grahsp
class TestMN12TorusGrahsp:
    """Mor & Netzer 2012 torus on torus:grahsp_mn12 block (Item B)."""

    @pytest.fixture
    def wave_nm(self):
        """Wavelength grid [nm] including nodes for 5100 Å."""
        return np.union1d(np.logspace(2.5, 5.0, 150), [510.0])

    @pytest.fixture
    def templates(self):
        """Cached GRAHSP templates."""
        return load_grahsp_templates()

    def test_block_equality_mn12_default(self, wave_nm, templates):
        """Block at default tor_temp=0, tor_cutoff_um=1.2 matches component API."""
        wave_aa = wave_nm * 10.0
        params_obj = GRAHSPParams(
            l5100=1e44,
            torus_model="mn12",
            tor_temp=0.0,
            tor_cutoff_um=1.2,
            fcov=0.4,
            si=0.0,
        )

        # Component API reference
        sed_ref = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_obj, templates)
        l_nu_ref = np.asarray(sed_ref.bbb_attenuated + sed_ref.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        # Composable runner with MN12 torus
        l_nu_got = np.asarray(composable_agn_l_nu(
            jnp.asarray(wave_aa),
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp_mn12",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_fcov=0.4,
            agn_grahsp_si=0.0,
            agn_grahsp_tor_temp=0.0,
            agn_grahsp_tor_cutoff_um=1.2,
            agn_type=1,
        ))

        np.testing.assert_allclose(l_nu_got, l_nu_ref, rtol=1e-8)

    def test_block_equality_mn12_varied_params(self, wave_nm, templates):
        """Block with varied tor_temp, tor_cutoff_um, si parameters."""
        wave_aa = wave_nm * 10.0
        test_cases = [
            (0.4, 1.5, 0.5),    # All nonzero
            (-0.7, 1.0, -0.3),  # Negative si (exercises clipping)
        ]

        for tor_temp, tor_cutoff_um, si in test_cases:
            params_obj = GRAHSPParams(
                l5100=1e44,
                torus_model="mn12",
                tor_temp=tor_temp,
                tor_cutoff_um=tor_cutoff_um,
                fcov=0.4,
                si=si,
            )

            # Component API reference
            sed_ref = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_obj, templates)
            l_nu_ref = np.asarray(sed_ref.bbb_attenuated + sed_ref.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

            # Composable runner
            l_nu_got = np.asarray(composable_agn_l_nu(
                jnp.asarray(wave_aa),
                agn_log_lbol=44.0,
                agn_disc_block="grahsp_sbpl",
                agn_nlr_block="grahsp",
                agn_blr_block="grahsp",
                agn_feii_block="grahsp",
                agn_torus_block="grahsp_mn12",
                agn_attenuation_block="grahsp_biatten",
                agn_grahsp_log_l5100=44.0,
                agn_grahsp_fcov=0.4,
                agn_grahsp_si=si,
                agn_grahsp_tor_temp=tor_temp,
                agn_grahsp_tor_cutoff_um=tor_cutoff_um,
                agn_type=1,
            ))

            np.testing.assert_allclose(
                l_nu_got, l_nu_ref, rtol=1e-8,
                err_msg=f"Failed for tor_temp={tor_temp}, tor_cutoff_um={tor_cutoff_um}, si={si}"
            )

    def test_liveness_tor_temp(self, wave_nm, templates):
        """tor_temp parameter must move the SED by > 1e-6 relative (liveness)."""
        wave_aa = wave_nm * 10.0

        # Evaluate at tor_temp=0
        sed_t0 = evaluate_grahsp_agn(
            jnp.asarray(wave_nm),
            GRAHSPParams(l5100=1e44, torus_model="mn12", tor_temp=0.0, tor_cutoff_um=1.2, fcov=0.4, si=0.0),
            templates,
        )
        l_nu_t0 = np.asarray(sed_t0.bbb_attenuated + sed_t0.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        # Evaluate at tor_temp=0.4
        sed_t4 = evaluate_grahsp_agn(
            jnp.asarray(wave_nm),
            GRAHSPParams(l5100=1e44, torus_model="mn12", tor_temp=0.4, tor_cutoff_um=1.2, fcov=0.4, si=0.0),
            templates,
        )
        l_nu_t4 = np.asarray(sed_t4.bbb_attenuated + sed_t4.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        rel_change = np.abs((l_nu_t4 - l_nu_t0) / np.maximum(np.abs(l_nu_t0), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-6, f"tor_temp change too small: {max_rel:.2e}; parameter is inert"

    def test_self_contained_no_type_mask(self, wave_nm, templates):
        """MN12 torus is self-contained: no Type-1/2 gray mask applied."""
        wave_aa = wave_nm * 10.0

        # Same SED at different agn_cos_inc (torus inclination) should be equal
        # if the torus is self-contained (no mask)
        l_nu_inc90 = np.asarray(composable_agn_l_nu(
            jnp.asarray(wave_aa),
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp_mn12",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_fcov=0.4,
            agn_grahsp_si=0.0,
            agn_grahsp_tor_temp=0.0,
            agn_grahsp_tor_cutoff_um=1.2,
            agn_cos_inc=0.9,
            agn_type=1,
        ))

        l_nu_inc10 = np.asarray(composable_agn_l_nu(
            jnp.asarray(wave_aa),
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp_mn12",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_fcov=0.4,
            agn_grahsp_si=0.0,
            agn_grahsp_tor_temp=0.0,
            agn_grahsp_tor_cutoff_um=1.2,
            agn_cos_inc=0.1,
            agn_type=1,
        ))

        # Same output (torus is self-contained)
        np.testing.assert_allclose(l_nu_inc90, l_nu_inc10, rtol=1e-10)


@requires_grahsp
class TestVetroncettyFeiiGrahsp:
    """Véron-Cetty 2004 FeII on feii:grahsp_veroncetty block (Item C)."""

    @pytest.fixture
    def wave_nm(self):
        """Wavelength grid [nm] including nodes for 5100 Å."""
        return np.union1d(np.logspace(2.5, 3.5, 100), [510.0])

    @pytest.fixture
    def templates(self):
        """Cached GRAHSP templates."""
        return load_grahsp_templates()

    def test_block_equality_veroncetty(self, wave_nm, templates):
        """Block at VC04 template matches component API."""
        wave_aa = wave_nm * 10.0
        params_obj = GRAHSPParams(
            l5100=1e44,
            feii_template="veroncetty2004",
            a_feii=5.0,
            a_lines=1.0,
        )

        # Component API reference
        sed_ref = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_obj, templates)
        l_nu_ref = np.asarray(sed_ref.bbb_attenuated + sed_ref.torus_attenuated) * 0.1 * wave_aa ** 2 / C_AA

        # Composable runner with VC04 FeII
        l_nu_got = np.asarray(composable_agn_l_nu(
            jnp.asarray(wave_aa),
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp_veroncetty",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_a_feii=5.0,
            agn_grahsp_a_lines=1.0,
            agn_type=1,
        ))

        np.testing.assert_allclose(l_nu_got, l_nu_ref, rtol=1e-8)

    def test_differs_from_bruhweiler_verner(self, wave_nm, templates):
        """VC04 template differs from Bruhweiler+Verner (B&V) by > 1e-2 relative."""
        wave_aa = wave_nm * 10.0

        # B&V (grahsp block)
        l_nu_bv = np.asarray(composable_agn_l_nu(
            jnp.asarray(wave_aa),
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_a_feii=5.0,
            agn_grahsp_a_lines=1.0,
            agn_type=1,
        ))

        # VC04 (grahsp_veroncetty block)
        l_nu_vc = np.asarray(composable_agn_l_nu(
            jnp.asarray(wave_aa),
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp_veroncetty",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_a_feii=5.0,
            agn_grahsp_a_lines=1.0,
            agn_type=1,
        ))

        rel_change = np.abs((l_nu_vc - l_nu_bv) / np.maximum(np.abs(l_nu_bv), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-2, f"VC04 and B&V too similar: {max_rel:.2e}"

    def test_liveness_a_feii(self, wave_nm, templates):
        """a_feii parameter must move the SED by > 1e-6 relative (liveness)."""
        wave_aa = wave_nm * 10.0

        # Evaluate at a_feii=0
        l_nu_a0 = np.asarray(composable_agn_l_nu(
            jnp.asarray(wave_aa),
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp_veroncetty",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_a_feii=0.0,
            agn_grahsp_a_lines=1.0,
            agn_type=1,
        ))

        # Evaluate at a_feii=5
        l_nu_a5 = np.asarray(composable_agn_l_nu(
            jnp.asarray(wave_aa),
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp_veroncetty",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_a_feii=5.0,
            agn_grahsp_a_lines=1.0,
            agn_type=1,
        ))

        rel_change = np.abs((l_nu_a5 - l_nu_a0) / np.maximum(np.abs(l_nu_a0), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-6, f"a_feii change too small: {max_rel:.2e}; parameter is inert"
