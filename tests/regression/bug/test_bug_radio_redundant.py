# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for radio.py redundant *_LSUN/_LSUN removed bug.

Bug: radio.py:113 — L_B = L_agn_bol * _LSUN / (...) / _LSUN; *_LSUN/_LSUN canceled.
"""

import chex
import jax.numpy as jnp
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.regression_bug


class TestRadioAGNSimplified:
    """Bug: radio.py:113 — redundant *_LSUN/_LSUN factor."""

    def test_radio_agn_finite(self):
        """radio_agn should return finite values with simplified formula."""
        from tengri.components.radio import radio_agn

        wave = jnp.logspace(7.0, 9.0, 100)  # radio wavelengths
        l_nu = radio_agn(wave, L_agn_bol=1e11, radio_loudness=2.0)
        chex.assert_tree_all_finite(l_nu)
        assert_non_negative(l_nu, name="l_nu")
