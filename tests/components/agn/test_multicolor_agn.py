# SPDX-License-Identifier: BSD-3-Clause
"""Tests for multicolor_agn: spin-dependent disc + two-temperature torus."""

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


@pytest.fixture()
def optical_wavelength():
    """Optical/UV wavelength grid."""
    return jnp.logspace(2.5, 5.0, 200)  # 316 A to 100,000 A


class TestMulticolorAgn:
    """Tests for multicolor_agn (spin-dependent K&D outer disc + 2T torus)."""

    def test_finite_nonneg(self, wavelength):
        """multicolor_agn produces finite, non-negative SED."""
        from tengri.components.agn.unified import multicolor_agn

        l_nu = multicolor_agn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0.0)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_multicolor_agn(self):
        """'multicolor_agn' appears in AGN_MODELS."""
        from tengri.components.agn.unified import AGN_MODELS

        assert "multicolor_agn" in AGN_MODELS

    def test_kubota_done_alias_is_same_function(self):
        """AGN_MODELS['kubota_done'] wraps the same function as multicolor_agn."""
        from tengri.components.agn.unified import AGN_MODELS

        # kubota_done and multicolor_agn are registry entries; check they wrap same function
        assert AGN_MODELS["kubota_done"] is AGN_MODELS["multicolor_agn"]

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_frac multiplies the whole SED linearly."""
        from tengri.components.agn.unified import multicolor_agn

        l1 = multicolor_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.1)
        l2 = multicolor_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.2)
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_higher_spin_more_far_uv(self, optical_wavelength):
        """Higher BH spin → smaller ISCO → hotter inner disc → more far-UV flux.

        At maximal spin (a=0.998), the ISCO shrinks to ~1.2 R_g vs 6 R_g at a=0,
        allowing the disc to reach temperatures ~3x higher. This shifts the
        Wien peak into the far-UV/EUV (< 500 Å), boosting flux there.
        """
        from tengri.components.agn.unified import multicolor_agn

        far_uv = (optical_wavelength > 300.0) & (optical_wavelength < 500.0)
        # Physical, sub-Eddington L_bol (log10 L_sun): under the luminosity-first
        # parameterization (ADR-0020) the shape is driven by L_bol + M_BH + spin,
        # and agn_log_lbol=44 would clip at the Eddington limit and wash out the
        # spin effect. At a sub-Eddington L_bol the far-UV (300-500 A) sits on the
        # Wien tail, where the hotter (smaller-ISCO) high-spin disc clearly wins.
        l_nospin = multicolor_agn(
            optical_wavelength, agn_log_lbol=12.0, agn_a_spin=0.0, agn_torus_frac=0.0
        )
        l_spin = multicolor_agn(
            optical_wavelength, agn_log_lbol=12.0, agn_a_spin=0.99, agn_torus_frac=0.0
        )
        assert jnp.any(far_uv), "No wavelengths in far-UV window"
        assert float(jnp.sum(l_spin[far_uv])) > float(jnp.sum(l_nospin[far_uv]))

    def test_jit_compatible(self, wavelength):
        """multicolor_agn is JIT-compilable."""
        from tengri.components.agn.unified import multicolor_agn

        @jax.jit
        def _run(wave):
            return multicolor_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))
