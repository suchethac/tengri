# SPDX-License-Identifier: BSD-3-Clause
"""Tests for kubota_done_full_agn: full 3-zone K&D disc + two-temperature torus."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.bounds


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


class TestKubotaDoneFullAgn:
    """Tests for kubota_done_full_agn (K&D 3-zone disc + 2T torus)."""

    def test_finite_nonneg(self, wavelength):
        """kubota_done_full_agn produces finite, non-negative SED."""
        from tengri.components.agn.unified import kubota_done_full_agn

        l_nu = kubota_done_full_agn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0.0)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_kubota_done_full(self):
        """'kubota_done_full' appears in AGN_MODELS."""
        from tengri.components.agn.unified import AGN_MODELS

        assert "kubota_done_full" in AGN_MODELS

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_frac multiplies the whole SED linearly."""
        from tengri.components.agn.unified import kubota_done_full_agn

        l1 = kubota_done_full_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.1)
        l2 = kubota_done_full_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.2)
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_has_xray_emission(self, wavelength):
        """Full 3-zone disc produces X-ray emission from the hot corona.

        kubota_done_full includes a hard X-ray power law (hot corona).
        At λ < 100 Å, the full model should have non-negligible flux
        while a torus-only comparison has zero disc contribution.
        """
        from tengri.components.agn.unified import kubota_done_full_agn

        xray_mask = (wavelength > 1.0) & (wavelength < 100.0)
        if not jnp.any(xray_mask):
            pytest.skip("wavelength grid does not cover X-ray regime")

        l_nu = kubota_done_full_agn(
            wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_f_hard=0.1,
            agn_torus_frac=0.0,
        )
        assert float(jnp.sum(l_nu[xray_mask])) > 0.0

    def test_f_hard_changes_sed_shape(self, wavelength):
        """Changing agn_f_hard from 0 to 0.1 alters the SED.

        With fixed L_bol, increasing f_hard routes more power to the
        corona power law and less to the disc. The two SEDs must differ.
        """
        from tengri.components.agn.unified import kubota_done_full_agn

        l_no_corona = kubota_done_full_agn(
            wavelength, agn_log_lbol=44.0, agn_frac=1.0, agn_f_hard=0.0, agn_torus_frac=0.0
        )
        l_corona = kubota_done_full_agn(
            wavelength, agn_log_lbol=44.0, agn_frac=1.0, agn_f_hard=0.1, agn_torus_frac=0.0
        )
        # The SEDs must differ somewhere (not identical arrays)
        assert not jnp.allclose(l_corona, l_no_corona, rtol=1e-6)

    def test_jit_compatible(self, wavelength):
        """kubota_done_full_agn is JIT-compilable."""
        from tengri.components.agn.unified import kubota_done_full_agn

        @jax.jit
        def _run(wave):
            return kubota_done_full_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))
