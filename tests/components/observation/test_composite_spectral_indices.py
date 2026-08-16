# SPDX-License-Identifier: BSD-3-Clause
"""Composite spectral indices ([MgFe]', <Fe>, …) — closes #505."""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectral_indices import (
    STANDARD_COMPOSITE_INDICES,
    STANDARD_INDICES,
    CompositeIndexDef,
    measure_index_jax,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _toy_lick_spectrum():
    """Synthetic continuum with absorption at Mgb / Fe5270 / Fe5335."""
    wave = jnp.linspace(4000.0, 5600.0, 2000)
    flux = (
        jnp.ones_like(wave)
        - 0.30 * jnp.exp(-(((wave - 5175.0) / 8.0) ** 2))
        - 0.10 * jnp.exp(-(((wave - 5270.0) / 8.0) ** 2))
        - 0.10 * jnp.exp(-(((wave - 5335.0) / 8.0) ** 2))
    )
    return wave, flux


class TestStandardComposites:
    def test_mean_fe_is_average_of_atomic(self):
        wave, flux = _toy_lick_spectrum()
        fe1 = float(measure_index_jax(wave, flux, STANDARD_INDICES["Fe5270"]))
        fe2 = float(measure_index_jax(wave, flux, STANDARD_INDICES["Fe5335"]))
        mean_fe = float(measure_index_jax(wave, flux, STANDARD_COMPOSITE_INDICES["<Fe>"]))
        np.testing.assert_allclose(mean_fe, 0.5 * (fe1 + fe2), rtol=1e-6)

    def test_mgfe_prime_matches_thomas2003(self):
        """[MgFe]' = sqrt(Mgb * (0.72*Fe5270 + 0.28*Fe5335)) per Thomas+2003."""
        wave, flux = _toy_lick_spectrum()
        mgb = float(measure_index_jax(wave, flux, STANDARD_INDICES["Mgb"]))
        fe1 = float(measure_index_jax(wave, flux, STANDARD_INDICES["Fe5270"]))
        fe2 = float(measure_index_jax(wave, flux, STANDARD_INDICES["Fe5335"]))
        expected = np.sqrt(max(mgb * (0.72 * fe1 + 0.28 * fe2), 0.0))
        actual = float(measure_index_jax(wave, flux, STANDARD_COMPOSITE_INDICES["[MgFe]'"]))
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_higher_balmer_sums(self):
        wave = jnp.linspace(4000.0, 4300.0, 1500)
        flux = jnp.ones_like(wave) - 0.10 * jnp.exp(-(((wave - 4101.7) / 5.0) ** 2))
        hda = float(measure_index_jax(wave, flux, STANDARD_INDICES["HdA"]))
        hga = float(measure_index_jax(wave, flux, STANDARD_INDICES["HgA"]))
        composite = float(measure_index_jax(wave, flux, STANDARD_COMPOSITE_INDICES["HdA+HgA"]))
        np.testing.assert_allclose(composite, hda + hga, rtol=1e-6)


class TestCompositeIndexDef:
    def test_wave_min_max_aggregate_over_components(self):
        comp = STANDARD_COMPOSITE_INDICES["[MgFe]'"]
        expected_min = min(c.wave_min for c in comp.components)
        expected_max = max(c.wave_max for c in comp.components)
        assert comp.wave_min == expected_min
        assert comp.wave_max == expected_max

    def test_user_combiner(self):
        """Users can supply their own combiner — JAX-compatible callable."""
        wave, flux = _toy_lick_spectrum()
        custom = CompositeIndexDef(
            name="Mgb-Fe5270",
            components=(STANDARD_INDICES["Mgb"], STANDARD_INDICES["Fe5270"]),
            combiner=lambda a, b: a - b,
        )
        mgb = float(measure_index_jax(wave, flux, STANDARD_INDICES["Mgb"]))
        fe = float(measure_index_jax(wave, flux, STANDARD_INDICES["Fe5270"]))
        np.testing.assert_allclose(
            float(measure_index_jax(wave, flux, custom)), mgb - fe, rtol=1e-6
        )
