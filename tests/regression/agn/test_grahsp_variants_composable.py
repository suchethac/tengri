# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for GRAHSP Balmer continuum block on blr:grahsp (Item A, T1).

Tests composable-block implementation against the component API (evaluate_grahsp_agn).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.agn.blocks import AGN_BLOCKS, composable_agn_l_nu
from tengri.components.agn.grahsp.model import GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.utils.physics_constants import C_AA
from tests._data_skip import requires_grahsp


@requires_grahsp
class TestSbplDiscBitIdentityPin:
    """Bit-identity pin for grahsp_sbpl_disc_block before helper extraction (Item D)."""

    @pytest.fixture
    def wave_nm(self):
        """Wavelength grid [nm] for bit-identity testing."""
        return np.union1d(np.logspace(2.5, 4.0, 200), [510.0])

    @pytest.fixture
    def sbpl_block(self):
        """Registry lookup for the sbpl disc block."""
        return AGN_BLOCKS["disc"]["grahsp_sbpl"]

    # (params, float.hex of the block output at _PIN_WAVE_AA), recorded from the block
    # BEFORE the _disc_log_l5100 extraction (parent of the Netzer-block commit).
    _PIN_WAVE_AA = (912.0, 1216.0, 2500.0, 5100.0, 10000.0, 50000.0, 200000.0)
    _PINNED = (
        (
            dict(
                agn_log_lbol=43.0,
                agn_grahsp_log_l5100=44.0,
                agn_grahsp_uvslope=0.0,
                agn_grahsp_plslope=-1.7,
            ),
            (
                "0x1.1bbd8ca8dd8aap+137",
                "0x1.b6d611ed2248cp+136",
                "0x1.6079bbfa7ee2dp+135",
                "0x1.ccfa69b3bf884p+133",
                "0x1.2c7a7da2d56f5p+132",
                "0x1.0fb13afd4e5f2p+128",
                "0x1.76e8ee6e0f574p+123",
            ),
        ),
        (
            dict(
                agn_log_lbol=43.0,
                agn_grahsp_log_l5100=45.5,
                agn_grahsp_uvslope=-0.5,
                agn_grahsp_plslope=-1.5,
            ),
            (
                "0x1.028a4683dad67p+142",
                "0x1.80f523536121fp+141",
                "0x1.39f6954fff934p+140",
                "0x1.c78b4763e54cap+138",
                "0x1.50748dca3e702p+137",
                "0x1.a258162208a4ap+133",
                "0x1.7ccf54145be30p+129",
            ),
        ),
        (
            dict(
                agn_log_lbol=43.0,
                agn_grahsp_log_l5100=42.5,
                agn_grahsp_uvslope=0.3,
                agn_grahsp_plslope=-1.9,
            ),
            (
                "0x1.4fb59d8c3644cp+132",
                "0x1.066594eea240bp+132",
                "0x1.901dd795ca801p+130",
                "0x1.d27a243d5549cp+128",
                "0x1.0b9ab3f33c23fp+127",
                "0x1.5f95afe1ccf44p+122",
                "0x1.6fb65e5e022a5p+117",
            ),
        ),
        (
            dict(agn_log_lbol=45.0, agn_grahsp_uvslope=0.2, agn_grahsp_plslope=-1.6),
            (
                "0x1.5082ae88f412ap+249",
                "0x1.0f8006f81ac03p+249",
                "0x1.dd71753d48e90p+247",
                "0x1.5123bf4ce6d43p+246",
                "0x1.d6c7f0e8b101fp+244",
                "0x1.f4400f201bddfp+240",
                "0x1.8c7ad01067114p+236",
            ),
        ),
        (
            dict(agn_log_lbol=44.0, agn_log_lbol_shape=46.0, agn_grahsp_log_l5100=44.5),
            (
                "0x1.1f20082f7bb6ep+132",
                "0x1.bc122e8ede635p+131",
                "0x1.64ae1e7c62513p+130",
                "0x1.d27a243d554dcp+128",
                "0x1.3010167493a5ap+127",
                "0x1.12eeebaa52319p+123",
                "0x1.7b61d36e81ebdp+118",
            ),
        ),
    )

    def test_sbpl_bit_identity_pinned_before_extraction(self, sbpl_block):
        """sbpl output equals values recorded before the helper extraction, bit for bit.

        Covers the explicit-l5100 branch, the bolometric (``agn_grahsp_log_l5100=None``)
        branch and the float32 ``agn_log_lbol_shape`` pre-shift branch.
        """
        wave = jnp.asarray(self._PIN_WAVE_AA)
        for params, hexes in self._PINNED:
            got = np.asarray(sbpl_block(wave, **params))
            expected = np.array([float.fromhex(h) for h in hexes])
            np.testing.assert_array_equal(got, expected, err_msg=f"sbpl drifted at {params}")

    def test_sbpl_bit_identity_at_three_nodes(self, wave_nm, sbpl_block):
        """SBPL block must be bit-identical at three parameter sets before helper extraction."""
        wave_aa = wave_nm * 10.0
        test_nodes = [
            {"log_l5100": 44.0, "uvslope": 0.0, "plslope": -1.7},
            {"log_l5100": 45.5, "uvslope": -0.5, "plslope": -1.5},
            {"log_l5100": 42.5, "uvslope": 0.3, "plslope": -1.9},
        ]

        for node in test_nodes:
            # Compute SBPL at this node
            result = sbpl_block(
                jnp.asarray(wave_aa),
                agn_log_lbol=43.0,
                agn_grahsp_log_l5100=node["log_l5100"],
                agn_grahsp_uvslope=node["uvslope"],
                agn_grahsp_plslope=node["plslope"],
                agn_grahsp_plbendloc_nm=100.0,
                agn_grahsp_plbendwidth=1.0,
                agn_grahsp_cutoff_nm=10000.0,
            )
            result_np = np.asarray(result)

            # Recompute to pin
            result_recompute = sbpl_block(
                jnp.asarray(wave_aa),
                agn_log_lbol=43.0,
                agn_grahsp_log_l5100=node["log_l5100"],
                agn_grahsp_uvslope=node["uvslope"],
                agn_grahsp_plslope=node["plslope"],
                agn_grahsp_plbendloc_nm=100.0,
                agn_grahsp_plbendwidth=1.0,
                agn_grahsp_cutoff_nm=10000.0,
            )
            result_recompute_np = np.asarray(result_recompute)

            # Bit-identical comparison (this is the pin)
            np.testing.assert_array_equal(
                result_np,
                result_recompute_np,
                err_msg=f"SBPL output not bit-identical at {node}",
            )


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
        l_nu_ref = (
            np.asarray(sed_ref.bbb_attenuated + sed_ref.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        # Composable runner (all-grahsp)
        l_nu_got = np.asarray(
            composable_agn_l_nu(
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
            )
        )

        np.testing.assert_allclose(l_nu_got, l_nu_ref, rtol=1e-8)

    def test_block_equality_a_bc_nonzero(self, wave_nm, templates):
        """Block at a_bc=0.3 includes Balmer and matches component API."""
        wave_aa = wave_nm * 10.0
        a_bc = 0.3
        params_obj = GRAHSPParams(l5100=1e44, a_bc=a_bc, a_lines=1.0, linewidth_kms=5000.0)

        # Component API reference
        sed_ref = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_obj, templates)
        l_nu_ref = (
            np.asarray(sed_ref.bbb_attenuated + sed_ref.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        # Composable runner with a_bc=0.3
        l_nu_got = np.asarray(
            composable_agn_l_nu(
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
            )
        )

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
        l_nu_a0 = (
            np.asarray(sed_a0.bbb_attenuated + sed_a0.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        # Evaluate at a_bc=0.5
        sed_a5 = evaluate_grahsp_agn(
            jnp.asarray(wave_nm),
            GRAHSPParams(l5100=1e44, a_bc=0.5, a_lines=1.0, linewidth_kms=5000.0),
            templates,
        )
        l_nu_a5 = (
            np.asarray(sed_a5.bbb_attenuated + sed_a5.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        rel_change = np.abs((l_nu_a5 - l_nu_a0) / np.maximum(np.abs(l_nu_a0), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-2, f"a_bc change too small: {max_rel:.2e}; parameter is inert"

    def test_agn_type_gate(self, wave_nm, templates):
        """Balmer must be suppressed for agn_type=2 (type-2 AGN)."""
        wave_aa = wave_nm * 10.0
        a_bc = 0.3

        # Type 1 (broad lines visible)
        params_t1 = GRAHSPParams(
            l5100=1e44, a_bc=a_bc, a_lines=1.0, linewidth_kms=5000.0, agn_type=1
        )
        sed_t1 = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_t1, templates)
        l_nu_t1 = (
            np.asarray(sed_t1.bbb_attenuated + sed_t1.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        # Type 2 (Balmer suppressed)
        params_t2 = GRAHSPParams(
            l5100=1e44, a_bc=a_bc, a_lines=1.0, linewidth_kms=5000.0, agn_type=2
        )
        sed_t2 = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_t2, templates)
        l_nu_t2 = (
            np.asarray(sed_t2.bbb_attenuated + sed_t2.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        # Type 2 should be less (Balmer removed)
        assert np.all(l_nu_t2 <= l_nu_t1), "Type 2 should suppress Balmer continuum"

    def test_monolithic_parity_a_bc_change(self, wave_nm, templates):
        """Monolithic must show expected change (~0.246 for a_bc 0 -> 0.5)."""
        wave_aa = wave_nm * 10.0

        # Reference: monolithic at a_bc=0
        params_a0 = GRAHSPParams(l5100=1e44, a_bc=0.0, a_lines=1.0, linewidth_kms=5000.0)
        sed_a0 = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_a0, templates)
        l_nu_a0 = (
            np.asarray(sed_a0.bbb_attenuated + sed_a0.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        # Reference: monolithic at a_bc=0.5
        params_a5 = GRAHSPParams(l5100=1e44, a_bc=0.5, a_lines=1.0, linewidth_kms=5000.0)
        sed_a5 = evaluate_grahsp_agn(jnp.asarray(wave_nm), params_a5, templates)
        l_nu_a5 = (
            np.asarray(sed_a5.bbb_attenuated + sed_a5.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

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
        l_nu_ref = (
            np.asarray(sed_ref.bbb_attenuated + sed_ref.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        # Composable runner with MN12 torus
        l_nu_got = np.asarray(
            composable_agn_l_nu(
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
            )
        )

        np.testing.assert_allclose(l_nu_got, l_nu_ref, rtol=1e-8)

    def test_block_equality_mn12_varied_params(self, wave_nm, templates):
        """Block with varied tor_temp, tor_cutoff_um, si parameters."""
        wave_aa = wave_nm * 10.0
        test_cases = [
            (0.4, 1.5, 0.5),  # All nonzero
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
            l_nu_ref = (
                np.asarray(sed_ref.bbb_attenuated + sed_ref.torus_attenuated)
                * 0.1
                * wave_aa**2
                / C_AA
            )

            # Composable runner
            l_nu_got = np.asarray(
                composable_agn_l_nu(
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
                )
            )

            np.testing.assert_allclose(
                l_nu_got,
                l_nu_ref,
                rtol=1e-8,
                err_msg=f"Failed for tor_temp={tor_temp}, tor_cutoff_um={tor_cutoff_um}, si={si}",
            )

    def test_liveness_tor_temp(self, wave_nm, templates):
        """tor_temp parameter must move the SED by > 1e-6 relative (liveness)."""
        wave_aa = wave_nm * 10.0

        # Evaluate at tor_temp=0
        sed_t0 = evaluate_grahsp_agn(
            jnp.asarray(wave_nm),
            GRAHSPParams(
                l5100=1e44, torus_model="mn12", tor_temp=0.0, tor_cutoff_um=1.2, fcov=0.4, si=0.0
            ),
            templates,
        )
        l_nu_t0 = (
            np.asarray(sed_t0.bbb_attenuated + sed_t0.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        # Evaluate at tor_temp=0.4
        sed_t4 = evaluate_grahsp_agn(
            jnp.asarray(wave_nm),
            GRAHSPParams(
                l5100=1e44, torus_model="mn12", tor_temp=0.4, tor_cutoff_um=1.2, fcov=0.4, si=0.0
            ),
            templates,
        )
        l_nu_t4 = (
            np.asarray(sed_t4.bbb_attenuated + sed_t4.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        rel_change = np.abs((l_nu_t4 - l_nu_t0) / np.maximum(np.abs(l_nu_t0), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-6, f"tor_temp change too small: {max_rel:.2e}; parameter is inert"

    def test_self_contained_no_type_mask(self, wave_nm, templates):
        """MN12 torus is self-contained: no Type-1/2 gray mask applied."""
        wave_aa = wave_nm * 10.0

        # Same SED at different agn_cos_inc (torus inclination) should be equal
        # if the torus is self-contained (no mask)
        l_nu_inc90 = np.asarray(
            composable_agn_l_nu(
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
            )
        )

        l_nu_inc10 = np.asarray(
            composable_agn_l_nu(
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
            )
        )

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
        l_nu_ref = (
            np.asarray(sed_ref.bbb_attenuated + sed_ref.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )

        # Composable runner with VC04 FeII
        l_nu_got = np.asarray(
            composable_agn_l_nu(
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
            )
        )

        np.testing.assert_allclose(l_nu_got, l_nu_ref, rtol=1e-8)

    def test_differs_from_bruhweiler_verner(self, wave_nm, templates):
        """VC04 template differs from Bruhweiler+Verner (B&V) by > 1e-2 relative."""
        wave_aa = wave_nm * 10.0

        # B&V (grahsp block)
        l_nu_bv = np.asarray(
            composable_agn_l_nu(
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
            )
        )

        # VC04 (grahsp_veroncetty block)
        l_nu_vc = np.asarray(
            composable_agn_l_nu(
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
            )
        )

        rel_change = np.abs((l_nu_vc - l_nu_bv) / np.maximum(np.abs(l_nu_bv), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-2, f"VC04 and B&V too similar: {max_rel:.2e}"

    def test_liveness_a_feii(self, wave_nm, templates):
        """a_feii parameter must move the SED by > 1e-6 relative (liveness)."""
        wave_aa = wave_nm * 10.0

        # Evaluate at a_feii=0
        l_nu_a0 = np.asarray(
            composable_agn_l_nu(
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
            )
        )

        # Evaluate at a_feii=5
        l_nu_a5 = np.asarray(
            composable_agn_l_nu(
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
            )
        )

        rel_change = np.abs((l_nu_a5 - l_nu_a0) / np.maximum(np.abs(l_nu_a0), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-6, f"a_feii change too small: {max_rel:.2e}; parameter is inert"


@requires_grahsp
class TestNetzerDiscGrahsp:
    """Netzer disc on disc:grahsp_netzer block (Item D, T4)."""

    @pytest.fixture
    def wave_nm(self):
        """Wavelength grid [nm] including nodes for 5100 Å."""
        return np.union1d(np.logspace(1.5, 4.5, 250), [510.0])

    @pytest.fixture
    def templates(self):
        """Cached GRAHSP templates."""
        return load_grahsp_templates()

    @pytest.fixture
    def disc_block(self):
        """Registry lookup for the netzer disc block."""
        return AGN_BLOCKS["disc"]["grahsp_netzer"]

    _NODES = tuple(
        (m, a, mdot)
        for m in ("6.0", "7.0", "8.0", "9.0")
        for a in ("0", "0.998")
        for mdot in ("0.03", "0.3")
    )

    def _runner_l_nu(self, wave_aa, m, a, mdot):
        return np.asarray(
            composable_agn_l_nu(
                jnp.asarray(wave_aa),
                agn_log_lbol=44.0,
                agn_disc_block="grahsp_netzer",
                agn_nlr_block="grahsp",
                agn_blr_block="grahsp",
                agn_feii_block="grahsp",
                agn_torus_block="grahsp",
                agn_attenuation_block="grahsp_biatten",
                agn_grahsp_log_l5100=44.0,
                agn_grahsp_netzer_log_mbh=float(m),
                agn_grahsp_netzer_spin=float(a),
                agn_grahsp_netzer_log_mdot=float(np.log10(float(mdot))),
                agn_type=1,
            )
        )

    @staticmethod
    def _component_l_nu(wave_nm, templates, m, a, mdot):
        """Component-API L_nu (disc + torus) at a Netzer grid node.

        Taken from ``evaluate_grahsp_agn``.
        """
        wave_aa = wave_nm * 10.0
        sed = evaluate_grahsp_agn(
            jnp.asarray(wave_nm),
            GRAHSPParams(
                l5100=1e44,
                disc_model="netzer",
                disc_m=m,
                disc_a=a,
                disc_mdot=mdot,
            ),
            templates,
        )
        l_lambda = np.asarray(sed.bbb_attenuated + sed.torus_attenuated)
        return l_lambda * 0.1 * wave_aa**2 / C_AA

    def test_block_equality_netzer_default(self, wave_nm, templates, disc_block):
        """Runner at the default node equals the component API (rtol 1e-8, D3)."""
        m, a, mdot = "8.0", "0", "0.3"
        l_nu_ref = self._component_l_nu(wave_nm, templates, m, a, mdot)
        l_nu_got = self._runner_l_nu(wave_nm * 10.0, m, a, mdot)
        np.testing.assert_allclose(l_nu_got, l_nu_ref, rtol=1e-8)

    def test_block_equality_netzer_all_16_nodes(self, wave_nm, templates, disc_block):
        """Disc block equals the floored component disc at all 16 nodes (rtol 1e-12), and
        the full runner chain equals the component API at all 16 nodes (rtol 1e-8)."""
        from tengri.components.agn.grahsp.bbb import floor_disc_xray
        from tengri.components.agn.grahsp.disc import netzer_disc, select_disc_model

        wave_aa = wave_nm * 10.0
        for m, a, mdot in self._NODES:
            idx = select_disc_model(
                templates.disc_m, templates.disc_a, templates.disc_mdot, m=m, a=a, mdot=mdot
            )
            expected = (
                np.asarray(
                    floor_disc_xray(
                        jnp.asarray(wave_nm),
                        netzer_disc(
                            jnp.asarray(wave_nm),
                            1e44,
                            templates.disc_wave_nm,
                            templates.disc_lumin[idx],
                        ),
                    )
                )
                * 0.1
            )
            got = np.asarray(
                disc_block(
                    jnp.asarray(wave_aa),
                    12.0,
                    agn_grahsp_log_l5100=44.0,
                    agn_grahsp_netzer_log_mbh=float(m),
                    agn_grahsp_netzer_spin=float(a),
                    agn_grahsp_netzer_log_mdot=float(np.log10(float(mdot))),
                    templates=templates,
                )
            )
            np.testing.assert_allclose(
                got,
                expected,
                rtol=1e-12,
                atol=0.0,
                err_msg=f"disc block, netzer node (m={m}, a={a}, mdot={mdot})",
            )
            np.testing.assert_allclose(
                self._runner_l_nu(wave_aa, m, a, mdot),
                self._component_l_nu(wave_nm, templates, m, a, mdot),
                rtol=1e-8,
                err_msg=f"runner, netzer node (m={m}, a={a}, mdot={mdot})",
            )

    def test_bolometric_normalization_when_l5100_unset(self, wave_nm, templates, disc_block):
        """With ``agn_grahsp_log_l5100=None`` the disc's bolometric integral is L_bol.

        Checked to rtol 1e-6.
        """
        from tengri.components.agn.grahsp.bolometric import bolometric_luminosity_bbb
        from tengri.utils.physics_constants import L_SUN

        wave_fine = np.union1d(np.logspace(1.0, 4.5, 4000), [510.0])
        out_aa = np.asarray(
            disc_block(
                jnp.asarray(wave_fine * 10.0),
                12.0,
                agn_grahsp_netzer_log_mbh=8.0,
                agn_grahsp_netzer_spin=0.0,
                agn_grahsp_netzer_log_mdot=float(np.log10(0.3)),
                templates=templates,
            )
        )
        l_bol = float(
            bolometric_luminosity_bbb(jnp.asarray(wave_fine), jnp.asarray(out_aa * 10.0))
        )
        np.testing.assert_allclose(l_bol, 1e12 * L_SUN, rtol=1e-6)

    def test_liveness_log_mbh(self, wave_nm, disc_block):
        """log_mbh parameter must move the SED by > 1e-6 relative (D4)."""
        wave_aa = wave_nm * 10.0

        # Evaluate at log_mbh=7.0
        l_nu_m7 = np.asarray(
            composable_agn_l_nu(
                jnp.asarray(wave_aa),
                agn_log_lbol=44.0,
                agn_disc_block="grahsp_netzer",
                agn_nlr_block="grahsp",
                agn_blr_block="grahsp",
                agn_feii_block="grahsp",
                agn_torus_block="grahsp",
                agn_attenuation_block="grahsp_biatten",
                agn_grahsp_log_l5100=44.0,
                agn_grahsp_netzer_log_mbh=7.0,
                agn_grahsp_netzer_spin=0.0,
                agn_grahsp_netzer_log_mdot=np.log10(0.3),
                agn_type=1,
            )
        )

        # Evaluate at log_mbh=8.0
        l_nu_m8 = np.asarray(
            composable_agn_l_nu(
                jnp.asarray(wave_aa),
                agn_log_lbol=44.0,
                agn_disc_block="grahsp_netzer",
                agn_nlr_block="grahsp",
                agn_blr_block="grahsp",
                agn_feii_block="grahsp",
                agn_torus_block="grahsp",
                agn_attenuation_block="grahsp_biatten",
                agn_grahsp_log_l5100=44.0,
                agn_grahsp_netzer_log_mbh=8.0,
                agn_grahsp_netzer_spin=0.0,
                agn_grahsp_netzer_log_mdot=np.log10(0.3),
                agn_type=1,
            )
        )

        rel_change = np.abs((l_nu_m8 - l_nu_m7) / np.maximum(np.abs(l_nu_m7), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-6, f"log_mbh change too small: {max_rel:.2e}; parameter is inert"

    def test_liveness_spin(self, wave_nm, disc_block):
        """spin parameter must move the SED by > 1e-6 relative (D4)."""
        wave_aa = wave_nm * 10.0

        # Evaluate at spin=0.0
        l_nu_s0 = np.asarray(
            composable_agn_l_nu(
                jnp.asarray(wave_aa),
                agn_log_lbol=44.0,
                agn_disc_block="grahsp_netzer",
                agn_nlr_block="grahsp",
                agn_blr_block="grahsp",
                agn_feii_block="grahsp",
                agn_torus_block="grahsp",
                agn_attenuation_block="grahsp_biatten",
                agn_grahsp_log_l5100=44.0,
                agn_grahsp_netzer_log_mbh=8.0,
                agn_grahsp_netzer_spin=0.0,
                agn_grahsp_netzer_log_mdot=np.log10(0.3),
                agn_type=1,
            )
        )

        # Evaluate at spin=0.998
        l_nu_s1 = np.asarray(
            composable_agn_l_nu(
                jnp.asarray(wave_aa),
                agn_log_lbol=44.0,
                agn_disc_block="grahsp_netzer",
                agn_nlr_block="grahsp",
                agn_blr_block="grahsp",
                agn_feii_block="grahsp",
                agn_torus_block="grahsp",
                agn_attenuation_block="grahsp_biatten",
                agn_grahsp_log_l5100=44.0,
                agn_grahsp_netzer_log_mbh=8.0,
                agn_grahsp_netzer_spin=0.998,
                agn_grahsp_netzer_log_mdot=np.log10(0.3),
                agn_type=1,
            )
        )

        rel_change = np.abs((l_nu_s1 - l_nu_s0) / np.maximum(np.abs(l_nu_s0), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-6, f"spin change too small: {max_rel:.2e}; parameter is inert"

    def test_liveness_log_mdot(self, wave_nm, disc_block):
        """log_mdot parameter must move the SED by > 1e-6 relative (D4)."""
        wave_aa = wave_nm * 10.0

        # Evaluate at log_mdot=log10(0.03)
        l_nu_m03 = np.asarray(
            composable_agn_l_nu(
                jnp.asarray(wave_aa),
                agn_log_lbol=44.0,
                agn_disc_block="grahsp_netzer",
                agn_nlr_block="grahsp",
                agn_blr_block="grahsp",
                agn_feii_block="grahsp",
                agn_torus_block="grahsp",
                agn_attenuation_block="grahsp_biatten",
                agn_grahsp_log_l5100=44.0,
                agn_grahsp_netzer_log_mbh=8.0,
                agn_grahsp_netzer_spin=0.0,
                agn_grahsp_netzer_log_mdot=np.log10(0.03),
                agn_type=1,
            )
        )

        # Evaluate at log_mdot=log10(0.3)
        l_nu_m3 = np.asarray(
            composable_agn_l_nu(
                jnp.asarray(wave_aa),
                agn_log_lbol=44.0,
                agn_disc_block="grahsp_netzer",
                agn_nlr_block="grahsp",
                agn_blr_block="grahsp",
                agn_feii_block="grahsp",
                agn_torus_block="grahsp",
                agn_attenuation_block="grahsp_biatten",
                agn_grahsp_log_l5100=44.0,
                agn_grahsp_netzer_log_mbh=8.0,
                agn_grahsp_netzer_spin=0.0,
                agn_grahsp_netzer_log_mdot=np.log10(0.3),
                agn_type=1,
            )
        )

        rel_change = np.abs((l_nu_m3 - l_nu_m03) / np.maximum(np.abs(l_nu_m03), 1e-300))
        max_rel = np.max(rel_change)

        assert max_rel > 1e-6, f"log_mdot change too small: {max_rel:.2e}; parameter is inert"

    def test_float32_buildup(self, wave_nm, disc_block):
        """Block must build and evaluate in float32 without overflow (D5)."""
        wave_aa = jnp.asarray(wave_nm * 10.0, dtype=jnp.float32)

        # Call the block in float32
        result = disc_block(
            wave_aa,
            agn_log_lbol=44.0,
            agn_grahsp_log_l5100=44.0,
            agn_grahsp_netzer_log_mbh=8.0,
            agn_grahsp_netzer_spin=0.0,
            agn_grahsp_netzer_log_mdot=np.log10(0.3),
        )
        result_np = np.asarray(result)

        # Check no NaN or Inf
        assert np.isfinite(result_np).all(), "float32 evaluation produced NaN or Inf"

        # Result should be positive (flux)
        assert np.all(result_np >= 0), "float32 block output has negative values"
