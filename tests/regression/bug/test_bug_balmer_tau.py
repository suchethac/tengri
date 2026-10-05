# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for Balmer continuum tau direction bug.

sigma_bf(nu) ~ nu^{-3} (Grandi 1982; Osterbrock & Ferland AGN^2 Eq. 2.4) gives
tau(nu) = tau_BE * (nu_BE/nu)^3 = tau_BE * (lambda/lambda_BE)^3: largest at the Balmer
edge, falling toward the blue. This file previously asserted the inverse,
(lambda_BE/lambda)^3, by evaluating local arithmetic only (never the model).
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


class TestBalmerContinuumTauDirection:
    """tau direction: largest at the edge, smaller to the blue (upstream QSOGen form)."""

    def test_tau_decreases_toward_shorter_wavelengths(self):
        """The model's tau(lambda) rises toward the Balmer edge (3646 A) from the blue."""
        from tengri.components.agn.qsogen import _balmer_continuum

        wave = jnp.asarray([1500.0, 2000.0, 2500.0])  # well blueward of the edge
        # tau << 1: component / B_lambda ~ tau(lambda) (scale is wavelength independent)
        bc = np.asarray(_balmer_continuum(wave, jnp.ones_like(wave), 1.0, 15000.0, 1e-6, 3646.0))
        w = np.asarray(wave)
        b_lam = w ** (-3.0) / np.expm1(1.43877735e8 / (15000.0 * w))
        tau_rel = bc / b_lam
        assert np.all(np.diff(tau_rel) > 0.0), "tau must rise toward the Balmer edge"
        np.testing.assert_allclose(tau_rel[1:] / tau_rel[0], (w[1:] / w[0]) ** 3, rtol=2e-3)

    def test_qsogen_balmer_continuum_shape(self):
        """Balmer continuum in qsogen should peak near the edge and fall at longer wavelengths."""
        pytest.importorskip("tengri.components.agn.qsogen")
        from tengri.components.agn.qsogen import _balmer_continuum

        wave = jnp.linspace(2500.0, 5000.0, 200)
        # Use a flat continuum for the test
        continuum = jnp.ones_like(wave)
        bc = _balmer_continuum(wave, continuum, tbc=2.0, taube=1.0, wavbe=3646.0)
        # Find peak: should be near or at the Balmer edge
        peak_wave = wave[jnp.argmax(bc)]
        assert peak_wave < 4000.0, f"Balmer continuum peak at {peak_wave:.0f} A, expected < 4000 A"
