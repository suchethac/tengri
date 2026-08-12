# SPDX-License-Identifier: BSD-3-Clause
"""Tests for unified_nlr_blr: NLR/BLR decomposition with geometric masking."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.bounds


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


@pytest.fixture()
def optical_wavelength():
    """Optical/UV wavelength grid."""
    return jnp.logspace(2.5, 5.0, 200)  # 316 A to 100,000 A


class TestUnifiedNlrBlr:
    """Tests for unified_nlr_blr: disc + torus + NLR/BLR with sigmoid masking."""

    def test_finite_nonneg(self, wavelength):
        """unified_nlr_blr produces finite, non-negative SED."""
        from tengri.components.agn.unified import unified_nlr_blr

        l_nu = unified_nlr_blr(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0.0)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_unified_nlr_blr(self):
        """'unified_nlr_blr' resolves via resolve_agn_model."""
        import warnings

        from tengri.components.agn.unified import resolve_agn_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fn = resolve_agn_model("unified_nlr_blr")
        assert callable(fn)

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_lum_ratio multiplies the whole SED linearly."""
        from tengri.components.agn.unified import unified_nlr_blr

        l1 = unified_nlr_blr(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.1)
        l2 = unified_nlr_blr(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.2)
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_type1_more_uv_than_type2(self, optical_wavelength):
        """Face-on (Type 1) has more disc UV emission than edge-on (Type 2).

        The disc is masked by the torus at high inclinations. Type 2 (edge-on,
        cos_inc=0) has the disc fully obscured, so the UV (disc-dominated)
        band carries much less flux.
        """
        from tengri.components.agn.unified import unified_nlr_blr

        uv_mask = (optical_wavelength > 1000.0) & (optical_wavelength < 4000.0)
        l_type1 = unified_nlr_blr(
            optical_wavelength,
            agn_log_lbol=44.0,
            agn_cos_inc=1.0,
            agn_theta_torus=30.0,
        )
        l_type2 = unified_nlr_blr(
            optical_wavelength,
            agn_log_lbol=44.0,
            agn_cos_inc=0.0,
            agn_theta_torus=30.0,
        )
        assert jnp.any(uv_mask), "No wavelengths in UV window"
        assert float(jnp.sum(l_type1[uv_mask])) > float(jnp.sum(l_type2[uv_mask]))

    def test_nlr_always_visible(self, optical_wavelength):
        """NLR emission appears even in edge-on (Type 2) views.

        NLR is isotropic — the mask is not applied to l_nlr. Even when
        cos_inc=0 fully obscures disc+BLR, the NLR contributes flux.
        """
        from tengri.components.agn.unified import unified_nlr_blr

        l_type2 = unified_nlr_blr(
            optical_wavelength,
            agn_log_lbol=44.0,
            agn_cos_inc=0.0,
            agn_theta_torus=30.0,
            agn_nlr_cf=0.3,
        )
        assert float(jnp.sum(l_type2)) > 0.0

    def test_polar_dust_reddens_type1_uv(self, optical_wavelength):
        """Polar E(B-V) > 0 suppresses UV more than optical for face-on views.

        SMC extinction law rises steeply toward UV, so the UV/optical ratio
        decreases when polar dust is applied to the disc + BLR.
        """
        from tengri.components.agn.unified import unified_nlr_blr

        uv_mask = (optical_wavelength > 1000.0) & (optical_wavelength < 3000.0)
        opt_mask = (optical_wavelength > 4500.0) & (optical_wavelength < 6000.0)

        l_nodust = unified_nlr_blr(
            optical_wavelength, agn_log_lbol=44.0, agn_cos_inc=1.0, agn_polar_ebv=0.0
        )
        l_dusty = unified_nlr_blr(
            optical_wavelength, agn_log_lbol=44.0, agn_cos_inc=1.0, agn_polar_ebv=0.5
        )

        assert jnp.any(uv_mask) and jnp.any(opt_mask)
        uv_ratio = float(jnp.mean(l_dusty[uv_mask] / jnp.maximum(l_nodust[uv_mask], 1e-100)))
        opt_ratio = float(jnp.mean(l_dusty[opt_mask] / jnp.maximum(l_nodust[opt_mask], 1e-100)))
        # UV is attenuated more than optical (SMC law steeper at short λ)
        assert uv_ratio < opt_ratio

    def test_jit_compatible(self, wavelength):
        """unified_nlr_blr is JIT-compilable."""
        from tengri.components.agn.unified import unified_nlr_blr

        @jax.jit
        def _run(wave):
            return unified_nlr_blr(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))
