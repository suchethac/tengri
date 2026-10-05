# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-06: Balmer continuum tau direction.

See ADR / docs/known_bugs.md for full context.
"""

import pytest

pytestmark = pytest.mark.regression_bug


class TestBug06BalmerTau:
    """qsogen.py — tau must be largest at the Balmer edge and fall toward the blue.

    Grandi 1982: sigma_bf(nu) ~ nu^-3, so tau ~ nu^-3 ~ lambda^3. An earlier
    version of this test asserted the inverse (tau ~ lambda^-3), i.e. it pinned a
    transcription error of upstream QSOGen's ``taube * (nuzero/nu)**3``; it also
    only evaluated local arithmetic and never called the model. It now measures
    tau(lambda) through ``_balmer_continuum``.
    """

    def test_tau_decreases_shortward(self):
        """Grandi 1982: sigma(nu) ~ nu^-3, so tau decreases toward shorter lambda."""
        import jax.numpy as jnp
        import numpy as np

        from tengri.components.agn.qsogen import _balmer_continuum

        wave = jnp.asarray([2000.0, 2500.0])
        # tau << 1: component = B_lambda * tau, so component / B_lambda ~ tau(lambda).
        bc = np.asarray(_balmer_continuum(wave, jnp.ones_like(wave), 1.0, 15000.0, 1e-6, 3646.0))
        w = np.asarray(wave)
        b_lam = w ** (-3.0) / np.expm1(1.43877735e8 / (15000.0 * w))
        tau_short, tau_long = bc / b_lam
        assert tau_short < tau_long
        assert tau_long / tau_short == pytest.approx((2500.0 / 2000.0) ** 3, rel=2e-3)
