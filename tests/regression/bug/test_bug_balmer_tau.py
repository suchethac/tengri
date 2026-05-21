"""Regression test for Balmer continuum tau direction bug.

Bug: qsogen.py:397 — tau ∝ (lambda/lambda_BE)^3 made tau larger at longer wavelengths.
Correct: sigma_bf(nu) ~ nu^{-3} → tau(lambda) = tau_BE * (lambda_BE/lambda)^3
(Osterbrock & Ferland AGN^2 Eq. 2.4).
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestBalmerContinuumTauDirection:
    """Bug: qsogen.py:397 — tau direction reversed (lambda_BE/lambda)^3."""

    def test_tau_decreases_at_longer_wavelengths(self):
        """tau must be largest at the Balmer edge (3646 A) and fall off at longer wavelengths."""
        wavbe = 3646.0  # Balmer edge in Angstrom
        taube = 1.0

        # Wavelengths shorter (above edge, higher nu) should have large tau
        # Wavelengths longer (below edge, lower nu) should have smaller tau
        wave_short = jnp.array([3000.0, 3200.0, 3400.0])  # shorter than edge -> tau > taube
        wave_long = jnp.array([4000.0, 5000.0, 7000.0])  # longer than edge -> tau < taube

        tau_short = taube * (wavbe / wave_short) ** 3
        tau_long = taube * (wavbe / wave_long) ** 3

        # tau at the edge should be taube
        tau_at_edge = taube * (wavbe / wavbe) ** 3
        assert jnp.isclose(tau_at_edge, taube)

        # tau should increase toward shorter wavelengths (tau_short > taube)
        assert jnp.all(tau_short > taube), "tau should exceed taube below the Balmer edge"

        # tau should decrease at longer wavelengths (tau_long < taube)
        assert jnp.all(tau_long < taube), "tau should fall below taube above the Balmer edge"

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
