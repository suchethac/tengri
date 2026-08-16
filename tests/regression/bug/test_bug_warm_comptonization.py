# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for warm Comptonization UV boost bug.

Bug: disc.py:321-362 — warm zone used kT_warm (soft X-rays) as seed frequency,
so the enhancement was never triggered at optical/UV.  K&D 2018 Eq. 3 prescribes
the local disc temperature as the seed frequency.
"""

import chex
import jax.numpy as jnp
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.regression_bug

_WAVE = jnp.logspace(2.5, 8.0, 500)  # 316 A to 10 cm, broad grid


class TestWarmComptonization:
    """Bug: disc.py:321-362 — warm zone used kT_warm as seed frequency at optical/UV."""

    def test_warm_comp_exceeds_outer_disc_at_uv(self):
        """With warm Comptonization, the warm zone SED should exceed a pure blackbody
        at intermediate UV/soft-X-ray wavelengths.
        """
        from tengri.components.agn.disc import (
            _planck_lnu,
            _warm_comptonization_lnu,
            _wavelength_to_nu,
        )

        wave_uv = jnp.logspace(2.5, 6.0, 200)  # 316 A - 1 mm
        nu = _wavelength_to_nu(wave_uv)
        temperature = 1e5  # K  — representative warm zone ring temperature
        kt_warm_kev = 0.2  # keV
        _KEV_TO_ERG = 1.602176634e-9
        _H_PLANCK = 6.626e-27
        nu_warm = kt_warm_kev * _KEV_TO_ERG / _H_PLANCK

        b_nu_plain = _planck_lnu(nu, temperature)
        b_nu_comp = _warm_comptonization_lnu(nu, temperature, nu_warm, gamma_warm=2.5)

        # The Comptonized spectrum should have MORE power than a plain blackbody
        # at intermediate frequencies between nu_seed and nu_warm.
        _K_BOLTZ = 1.38e-16
        nu_seed = _K_BOLTZ * temperature / _H_PLANCK  # ~2e15 Hz for T=1e5 K
        mid_mask = (nu > nu_seed) & (nu < nu_warm)
        if jnp.sum(mid_mask) > 5:
            denom = jnp.maximum(jnp.mean(b_nu_plain[mid_mask]), 1e-300)
            ratio = jnp.mean(b_nu_comp[mid_mask]) / denom
            assert ratio > 1.0, f"Comptonized SED not enhanced over plain BB: ratio={ratio:.3f}"

    def test_warm_comp_finite_positive(self):
        """_warm_comptonization_lnu must return finite, positive values."""
        from tengri.components.agn.disc import _warm_comptonization_lnu, _wavelength_to_nu

        nu = _wavelength_to_nu(_WAVE)
        _KEV_TO_ERG = 1.602176634e-9
        _H_PLANCK = 6.626e-27
        nu_warm = 0.2 * _KEV_TO_ERG / _H_PLANCK
        b_nu = _warm_comptonization_lnu(nu, 1e5, nu_warm, 2.5)
        chex.assert_tree_all_finite(b_nu)
        assert_non_negative(b_nu, name="b_nu")
