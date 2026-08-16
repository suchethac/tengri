# SPDX-License-Identifier: BSD-3-Clause
"""Tests for _sigmoid_mask: smooth geometric visibility function."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.bounds


class TestSigmoidMask:
    """Tests for the _sigmoid_mask visibility function."""

    def test_face_on_high_visibility(self):
        """Face-on (cos_inc=1) gives visibility close to 1."""
        from tengri.components.agn.unified import _sigmoid_mask

        mask = _sigmoid_mask(cos_inc=1.0, theta_torus=30.0)
        assert float(mask) > 0.9

    def test_edge_on_low_visibility(self):
        """Edge-on (cos_inc=0) gives visibility close to 0 for a wide torus."""
        from tengri.components.agn.unified import _sigmoid_mask

        mask = _sigmoid_mask(cos_inc=0.0, theta_torus=30.0)
        assert float(mask) < 0.1

    def test_output_in_unit_interval(self):
        """Mask value is always in [0, 1]."""
        from tengri.components.agn.unified import _sigmoid_mask

        for cos_inc in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = float(_sigmoid_mask(cos_inc=cos_inc, theta_torus=30.0))
            assert 0.0 <= val <= 1.0, f"Mask out of [0,1] at cos_inc={cos_inc}: {val}"

    def test_monotone_increasing_with_cos_inc(self):
        """Larger cos_inc (more face-on) → larger visibility."""
        from tengri.components.agn.unified import _sigmoid_mask

        mask_edge = float(_sigmoid_mask(cos_inc=0.0, theta_torus=30.0))
        mask_mid = float(_sigmoid_mask(cos_inc=0.5, theta_torus=30.0))
        mask_face = float(_sigmoid_mask(cos_inc=1.0, theta_torus=30.0))
        assert mask_edge < mask_mid < mask_face

    def test_narrower_torus_increases_visibility(self):
        """Narrower torus (theta_torus = torus half-angle) → disc more visible.

        `theta_torus` is the torus HALF-ANGLE (dusty region), so
        inc_crit = 90 - theta_torus.  A narrow torus (theta=20°) has
        inc_crit=70°; a wide torus (theta=60°) has inc_crit=30°.

        At 45° inclination (cos_inc≈0.707):
          - narrow torus: 45° < 70° → disc visible (mask ≈ 1)
          - wide torus:   45° > 30° → disc blocked (mask ≈ 0)
        """
        from tengri.components.agn.unified import _sigmoid_mask

        # cos(45 deg) ≈ 0.707
        mask_narrow = float(_sigmoid_mask(cos_inc=0.707, theta_torus=20.0))
        mask_wide = float(_sigmoid_mask(cos_inc=0.707, theta_torus=60.0))
        assert mask_narrow > mask_wide

    def test_jit_compatible(self):
        """_sigmoid_mask is JIT-compilable."""
        from tengri.components.agn.unified import _sigmoid_mask

        @jax.jit
        def _run(cos_inc):
            return _sigmoid_mask(cos_inc, theta_torus=30.0)

        result = _run(jnp.array(0.5))
        assert jnp.isfinite(result)

    def test_gradient_wrt_cos_inc(self):
        """_sigmoid_mask has a finite, non-zero gradient w.r.t. cos_inc near transition."""
        from tengri.components.agn.unified import _sigmoid_mask

        # Near the transition (inc ~ 90 - theta_torus = 60 deg, cos ~ 0.5)
        g = float(jax.grad(_sigmoid_mask)(0.5, theta_torus=30.0))
        assert jnp.isfinite(jnp.array(g))
        assert g != 0.0, "Gradient of sigmoid mask should be non-zero near transition"
