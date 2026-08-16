# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for disc.py ring area π factor bug.

Bug: disc.py:298/618/639/858 — ring area missing pi from hemisphere integral.

Rybicki & Lightman 1979, Eq. 1.6: dL_nu = pi * B_nu * dA * cos(i).
Since the disc is renormalized to L_bol, the shape is unaffected, but
l_warm_bol used in the Beloborodov energy budget must include the pi.
"""

import chex
import jax.numpy as jnp
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.regression_bug

_WAVE = jnp.logspace(2.5, 8.0, 500)  # 316 A to 10 cm, broad grid


class TestRingAreaPi:
    """Bug: disc.py:298/618/639/858 — ring area missing pi from hemisphere integral."""

    def test_multicolor_disc_finite_positive(self):
        """multicolor_disc should return finite, positive SED at all wavelengths."""
        from tengri.components.agn.disc import multicolor_disc

        l_nu = multicolor_disc(
            _WAVE, agn_log_lbol=12.0, agn_lum_ratio=1.0, agn_log_mbh=8.0, agn_cos_inc=0.5
        )
        chex.assert_tree_all_finite(l_nu)
        assert_non_negative(l_nu, name="l_nu")

    def test_kubota_done_disc_finite(self):
        """kubota_done_disc should return finite, positive SED."""
        from tengri.components.agn.disc import kubota_done_disc

        l_nu = kubota_done_disc(
            _WAVE, agn_log_lbol=12.0, agn_lum_ratio=1.0, agn_log_mbh=8.0, agn_log_ledd=-1.0
        )
        chex.assert_tree_all_finite(l_nu)
        assert_non_negative(l_nu, name="l_nu")

    def test_adaf_disc_finite(self):
        """adaf_disc should return finite, positive SED."""
        from tengri.components.agn.disc import adaf_disc

        l_nu = adaf_disc(
            _WAVE, agn_log_lbol=10.0, agn_lum_ratio=0.1, agn_log_mbh=8.0, agn_log_ledd=-3.0
        )
        chex.assert_tree_all_finite(l_nu)
        assert_non_negative(l_nu, name="l_nu")
