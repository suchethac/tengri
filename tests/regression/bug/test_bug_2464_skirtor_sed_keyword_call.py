# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2464: skirtor_sed raises IndexError when called by keyword.

Mechanism: ``skirtor_sed(*args, **kwargs)`` unpacks ``args[0]`` directly as the
wavelength grid without checking if ``args`` is empty. When called with all keyword
arguments (e.g., ``skirtor_sed(wavelength=..., agn_log_lbol=...)``), ``args == ()``
and ``args[0]`` raises ``IndexError: tuple index out of range``. The same defect
affects the deprecated alias ``skirtor_analytic``.

Fix: resolve ``wavelength`` from ``args[0]`` when present, else
``kwargs.pop("wavelength")``;
if absent, raise ``TypeError`` naming the missing parameter explicitly.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


class TestBug2464SkirTorSedKeywordCall:
    """skirtor_sed must accept wavelength by keyword argument."""

    def test_skirtor_sed_positional_call(self):
        """Control: positional call works."""
        wave = np.array([1000.0, 5000.0, 10000.0])
        # Call with positional wavelength argument
        sed = __import__("tengri").components.agn.skirtor_sed(wave, agn_log_lbol=10.0)
        assert sed.shape == wave.shape
        assert np.all(np.isfinite(sed))

    def test_skirtor_sed_keyword_call_equals_positional(self):
        """Test (1): skirtor_sed called with keyword wavelength equals positional."""
        wave = np.array([1000.0, 5000.0, 10000.0])

        # Positional call
        sed_pos = __import__("tengri").components.agn.skirtor_sed(wave, agn_log_lbol=10.0)

        # Keyword call
        sed_kw = __import__("tengri").components.agn.skirtor_sed(
            wavelength=wave, agn_log_lbol=10.0
        )

        # Must be exactly equal
        np.testing.assert_array_equal(sed_kw, sed_pos)

    def test_skirtor_analytic_keyword_call_with_deprecation_warning(self):
        """Test (2): skirtor_analytic (deprecated alias) works by keyword.

        Expects DeprecationWarning.
        """
        wave = np.array([1000.0, 5000.0, 10000.0])

        # skirtor_analytic is a deprecated alias; expect DeprecationWarning
        with pytest.warns(DeprecationWarning, match="skirtor_analytic"):
            sed_pos = __import__("tengri").components.agn.skirtor_analytic(wave, agn_log_lbol=10.0)

        with pytest.warns(DeprecationWarning, match="skirtor_analytic"):
            sed_kw = __import__("tengri").components.agn.skirtor_analytic(
                wavelength=wave, agn_log_lbol=10.0
            )

        # Must be exactly equal
        np.testing.assert_array_equal(sed_kw, sed_pos)

    def test_skirtor_sed_missing_wavelength_raises_type_error(self):
        """Test (3): calling skirtor_sed without wavelength raises TypeError."""
        # Call with neither positional nor keyword wavelength argument
        with pytest.raises(TypeError) as exc_info:
            __import__("tengri").components.agn.skirtor_sed(agn_log_lbol=10.0)

        # Error message must name 'wavelength'
        assert "wavelength" in str(exc_info.value).lower()
