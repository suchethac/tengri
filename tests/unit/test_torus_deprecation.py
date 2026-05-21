"""Regression tests for toy torus models warning.

Both simple_torus and two_temperature_torus are MBB approximations that
should not be used for science. They must warn loudly (once per process)
so users know to switch to skirtor_analytic.
"""

import warnings

import chex
import jax.numpy as jnp
import pytest

from tengri.components.agn import torus as torus_module
from tengri.components.agn.torus import simple_torus, two_temperature_torus


@pytest.fixture
def wave():
    return jnp.linspace(1e4, 1e6, 200)  # 1 um – 100 um in Angstrom


@pytest.fixture(autouse=True)
def reset_torus_warnings():
    """Reset the warning guard before each test."""
    torus_module._WARNED.clear()
    yield
    torus_module._WARNED.clear()


class TestSimpleTorusDeprecation:
    def test_emits_deprecation_warning(self, wave):
        with pytest.warns(UserWarning, match="simple_torus"):
            simple_torus(wave, agn_log_lbol=12.0)

    def test_warning_mentions_skirtor(self, wave):
        with pytest.warns(UserWarning) as record:
            simple_torus(wave, agn_log_lbol=12.0)
        assert "SKIRTOR" in str(record[0].message)

    def test_still_returns_finite_sed(self, wave):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = simple_torus(wave, agn_log_lbol=12.0)
        chex.assert_equal_shape([result, wave])
        chex.assert_tree_all_finite(result)
        assert jnp.all(result >= 0.0)

    def test_output_not_plain_mbb(self, wave):
        """Silicate opacity must modify the shape — output != featureless MBB."""
        from tengri.components.dust.emission import modified_blackbody

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            torus_sed = simple_torus(wave, agn_log_lbol=12.0, agn_T_torus=1000.0)

        l_abs = float(jnp.trapezoid(torus_sed, jnp.flip(jnp.sort(1e10 / wave))))
        plain_mbb = modified_blackbody(wave, L_absorbed=max(l_abs, 1.0), dust_T=1000.0)

        # Normalized peak-normalized shapes must differ (silicate opacity creates a dip)
        torus_norm = torus_sed / jnp.max(torus_sed)
        mbb_norm = plain_mbb / jnp.max(plain_mbb)
        max_diff = float(jnp.max(jnp.abs(torus_norm - mbb_norm)))
        assert max_diff > 0.01, "simple_torus and plain MBB should differ (silicate opacity)"


class TestTwoTemperatureTorusDeprecation:
    def test_emits_deprecation_warning(self, wave):
        with pytest.warns(UserWarning, match="two_temperature_torus"):
            two_temperature_torus(wave, agn_log_lbol=12.0)

    def test_warning_mentions_skirtor(self, wave):
        with pytest.warns(UserWarning) as record:
            two_temperature_torus(wave, agn_log_lbol=12.0)
        assert "SKIRTOR" in str(record[0].message)

    def test_still_returns_finite_sed(self, wave):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = two_temperature_torus(wave, agn_log_lbol=12.0)
        chex.assert_equal_shape([result, wave])
        chex.assert_tree_all_finite(result)
        assert jnp.all(result >= 0.0)

    def test_differs_from_simple_torus(self, wave):
        """Two-temperature model should produce a different shape than single-T."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sed_simple = simple_torus(wave, agn_log_lbol=12.0, agn_T_torus=1000.0)
            sed_two = two_temperature_torus(
                wave, agn_log_lbol=12.0, agn_T_hot=1200.0, agn_T_warm=300.0
            )

        simple_norm = sed_simple / jnp.max(sed_simple)
        two_norm = sed_two / jnp.max(sed_two)
        assert float(jnp.max(jnp.abs(simple_norm - two_norm))) > 1e-3
