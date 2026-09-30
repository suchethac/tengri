# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for GRAHSP variant blocks: Balmer continuum, MN12 torus, VC04 FeII.

Tests the composable-block implementations of each variant against:
1. The component API (``evaluate_grahsp_agn`` and ``compute_grahsp_sed``)
2. The monolithic reference values (upstream parity)

Item A: Balmer continuum on blr:grahsp (T1)
Item B: MN12 torus (T2)
Item C: VC04 FeII (T2)
Item D: Netzer disc (T3+T4)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.agn.blocks import AGN_BLOCKS, composable_agn_l_nu
from tengri.components.agn.grahsp.model import evaluate_grahsp_agn, compute_grahsp_sed
from tengri.components.agn.grahsp.lines import gaussian_lines
from tengri.components.agn.grahsp.balmer import balmer_continuum
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tests._data_skip import requires_grahsp


# ──────────────────────────────────────────────────────────────────────
# Item A: Balmer continuum on blr:grahsp (T1)
# ──────────────────────────────────────────────────────────────────────


@requires_grahsp
class TestBalmercontinuumBlrGrahsp:
    """Balmer continuum (a_bc parameter) on blr:grahsp block."""

    @pytest.fixture
    def wave_nm(self):
        """Wavelength grid [nm] spanning the Balmer continuum."""
        return np.logspace(2.5, 3.5, 50)  # 316 nm to 3162 nm

    @pytest.fixture
    def l5100_disc(self):
        """Reference 5100Å luminosity [erg/s]."""
        return 1e44

    def test_block_equality_with_balmer_a_bc_zero(self, wave_nm, l5100_disc):
        """Block output at a_bc=0 equals gaussian_lines broad component only."""
        templates = load_grahsp_templates()
        wave_aa = wave_nm * 10.0

        # Direct component call: just gaussian broad lines.
        broad, _ = gaussian_lines(
            wave_nm=wave_nm,
            line_wave_nm=templates.line_wave_nm,
            line_broad=templates.line_broad,
            line_narrow_sy2=templates.line_narrow_sy2,
            line_narrow_liner=templates.line_narrow_liner,
            l5100=l5100_disc,
            a_lines=1.0,
            linewidth_kms=5000.0,
            agn_type=1,
        )
        expected = broad * 0.1  # nm -> Å

        # Block call with a_bc=0
        block_result = AGN_BLOCKS["blr"]["grahsp"](
            wave_aa,
            agn_log_lbol=44.0,
            l5100_disc=l5100_disc,
            agn_grahsp_a_lines=1.0,
            agn_grahsp_linewidth_kms=5000.0,
            agn_grahsp_a_bc=0.0,
            agn_type=1,
            templates=templates,
        )

        # Should be bit-identical when a_bc=0
        np.testing.assert_array_equal(block_result, expected)

    def test_block_equality_with_balmer_a_bc_nonzero(self, wave_nm, l5100_disc):
        """Block output includes Balmer continuum when a_bc > 0."""
        templates = load_grahsp_templates()
        wave_aa = wave_nm * 10.0
        a_bc = 0.3

        # Direct component calls
        broad, _ = gaussian_lines(
            wave_nm=wave_nm,
            line_wave_nm=templates.line_wave_nm,
            line_broad=templates.line_broad,
            line_narrow_sy2=templates.line_narrow_sy2,
            line_narrow_liner=templates.line_narrow_liner,
            l5100=l5100_disc,
            a_lines=1.0,
            linewidth_kms=5000.0,
            agn_type=1,
        )
        balmer = balmer_continuum(
            wave_nm=wave_nm,
            l5100=l5100_disc,
            a_bc=a_bc,
            linewidth_kms=5000.0,
        )
        expected = (broad + balmer) * 0.1  # nm -> Å

        # Block call with a_bc=0.3
        block_result = AGN_BLOCKS["blr"]["grahsp"](
            wave_aa,
            agn_log_lbol=44.0,
            l5100_disc=l5100_disc,
            agn_grahsp_a_lines=1.0,
            agn_grahsp_linewidth_kms=5000.0,
            agn_grahsp_a_bc=a_bc,
            agn_type=1,
            templates=templates,
        )

        np.testing.assert_allclose(block_result, expected, rtol=1e-12)

    def test_runner_equality_balmer_with_component_api(self, wave_nm, l5100_disc):
        """Composable runner (blr:grahsp) matches evaluate_grahsp_agn(a_bc=0.3)."""
        wave_aa = wave_nm * 10.0
        a_bc = 0.3

        # Component API reference
        ref_sed = evaluate_grahsp_agn(
            wave_nm=wave_nm,
            agn_log_lbol=44.0,
            agn_type=1,
            a_bc=a_bc,
            a_lines=1.0,
            linewidth_kms=5000.0,
        )

        # Composable runner with all-grahsp recipe
        got_l_nu = composable_agn_l_nu(
            wave_aa,
            agn_log_lbol=44.0,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_log_l5100=44.0,  # parametric normalization
            agn_grahsp_a_bc=a_bc,
            agn_type=1,
        )
        # Runner returns L_nu [erg/s/Hz]; component API returns L_lambda [erg/s/nm]
        # Convert: L_nu = L_lambda * lambda^2 / c (in CGS units, no Å conversion)
        c_aa = 2.99792458e18  # c in Å/s
        got_sed = got_l_nu * (wave_aa**2) / c_aa

        np.testing.assert_allclose(got_sed, ref_sed, rtol=1e-8)

    def test_liveness_a_bc_changes_sed(self):
        """Varying a_bc must change the SED measurably (>1e-2 relative)."""
        from tengri.models import SEDModel
        from jax import grad

        # Build a minimal model with blr:grahsp and a_bc as free parameter
        spec = {
            "wavelength": {"min_aa": 1000, "max_aa": 1e5},
            "agn": {
                "disc": "grahsp_sbpl",
                "blr": {"type": "grahsp"},
                "feii": "grahsp",
                "torus": "grahsp",
            },
            "agn_grahsp_a_bc": "Uniform(0.0, 2.0)",
        }
        model = SEDModel.from_dict(spec)

        # Evaluate at a_bc=0 and a_bc=0.5
        sed_a0 = model.forward(
            wavelengths=np.array([5100.0]),
            params={"agn_grahsp_a_bc": 0.0, "agn_log_lbol": 11.0, "agn_lum_ratio": 0.5},
        )
        sed_a5 = model.forward(
            wavelengths=np.array([5100.0]),
            params={"agn_grahsp_a_bc": 0.5, "agn_log_lbol": 11.0, "agn_lum_ratio": 0.5},
        )

        rel_change = np.abs((sed_a5 - sed_a0) / np.maximum(np.abs(sed_a0), 1e-30))
        assert np.max(rel_change) > 1e-2, (
            f"a_bc=0 vs a_bc=0.5 produces negligible change {np.max(rel_change):.2e}; "
            "parameter is inert."
        )

        # Gradient should be nonzero w.r.t. a_bc
        def predict_sed(a_bc_val):
            return model.forward(
                wavelengths=np.array([5100.0]),
                params={"agn_grahsp_a_bc": a_bc_val, "agn_log_lbol": 11.0, "agn_lum_ratio": 0.5},
            )

        grad_a_bc = grad(predict_sed)(0.5)
        assert np.any(np.abs(grad_a_bc) > 0), "Gradient w.r.t. a_bc is zero; parameter is not differentiable."

    def test_agn_type_gate_suppresses_balmer_for_type2(self, wave_nm, l5100_disc):
        """Block must not emit Balmer for agn_type=2 (type-2 AGN, edge-on)."""
        templates = load_grahsp_templates()
        wave_aa = wave_nm * 10.0
        a_bc = 0.3

        # Block call with agn_type=2
        block_type2 = AGN_BLOCKS["blr"]["grahsp"](
            wave_aa,
            agn_log_lbol=44.0,
            l5100_disc=l5100_disc,
            agn_grahsp_a_lines=1.0,
            agn_grahsp_linewidth_kms=5000.0,
            agn_grahsp_a_bc=a_bc,
            agn_type=2,
            templates=templates,
        )

        # Block call with agn_type=1 (for comparison of line component)
        block_type1 = AGN_BLOCKS["blr"]["grahsp"](
            wave_aa,
            agn_log_lbol=44.0,
            l5100_disc=l5100_disc,
            agn_grahsp_a_lines=1.0,
            agn_grahsp_linewidth_kms=5000.0,
            agn_grahsp_a_bc=a_bc,
            agn_type=1,
            templates=templates,
        )

        # Type 2 should be less than Type 1 (Balmer removed)
        assert np.all(block_type2 <= block_type1)

        # Should match the component API call with agn_type=2
        ref_type2 = evaluate_grahsp_agn(
            wave_nm=wave_nm,
            agn_log_lbol=44.0,
            agn_type=2,
            a_bc=a_bc,
            a_lines=1.0,
            linewidth_kms=5000.0,
        )
        np.testing.assert_allclose(block_type2, ref_type2, rtol=1e-12)

    def test_monolith_unchanged_a_bc_on_top_level(self):
        """Monolithic path must still respond to top-level agn_grahsp_a_bc."""
        wave_nm = np.logspace(2.5, 4.0, 50)
        wave_aa = wave_nm * 10.0

        # Monolithic reference (component API)
        ref_a0 = compute_grahsp_sed(
            wave_nm=wave_nm,
            agn_log_lbol=44.0,
            agn_type=1,
            a_bc=0.0,
            a_lines=1.0,
            linewidth_kms=5000.0,
        )
        ref_a5 = compute_grahsp_sed(
            wave_nm=wave_nm,
            agn_log_lbol=44.0,
            agn_type=1,
            a_bc=0.5,
            a_lines=1.0,
            linewidth_kms=5000.0,
        )

        # Monolithic change should be ~0.246 (as per design §4 Item A.5)
        rel_change = np.abs((ref_a5 - ref_a0) / np.maximum(np.abs(ref_a0), 1e-30))
        max_rel = np.max(rel_change)

        # Expect around 0.246 (24.6% change) for 0.0 -> 0.5
        assert 0.15 < max_rel < 0.35, (
            f"Monolithic a_bc=0 vs a_bc=0.5 expected ~0.246 change, got {max_rel:.3f}; "
            "monolithic physics may be broken."
        )
