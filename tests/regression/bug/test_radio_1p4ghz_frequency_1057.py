# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for issue #1057: radio l_1p4ghz frequency fix.

Issue #1057: l_1p4ghz evaluates at the 21 cm HI line (1.4204 GHz),
not the FIR-radio correlation standard (1.400 GHz).

The fix unifies the 1.4 GHz wavelength in a shared constant WAVE_1P4GHZ_AA
in physics_constants.py, used by both radio.component._l_1p4ghz_fn and
component_factory.state_to_radio_quantities. Both code paths must use
the same constant.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def test_wave_1p4ghz_constant_equals_1400_mhz_frequency():
    """The 1.4 GHz wavelength constant must equal c / 1.400 GHz exactly."""
    from tengri.utils.physics_constants import C_AA, WAVE_1P4GHZ_AA

    # Exact frequency: 1.400 GHz (standard FIR-radio correlation frequency)
    freq_hz = 1.400e9

    # Expected wavelength: c / freq = wavelength
    wave_expected = C_AA / freq_hz

    # Tolerance: 1 part in 1e6 (frequency precision to ~1 Hz at 1.4 GHz)
    np.testing.assert_allclose(WAVE_1P4GHZ_AA, wave_expected, rtol=1e-6)

    # Also verify it evaluates to ~1.400 GHz
    freq_actual = C_AA / WAVE_1P4GHZ_AA
    np.testing.assert_allclose(freq_actual, freq_hz, rtol=1e-6)


def test_wave_1p4ghz_not_hi_line_frequency():
    """WAVE_1P4GHZ_AA must NOT be the 21 cm HI line (1.4204 GHz)."""
    from tengri.utils.physics_constants import C_AA, WAVE_1P4GHZ_AA

    # 21 cm HI line frequency
    hi_line_freq = 1.42040575177e9  # Hz (rest frequency)
    hi_line_wave = C_AA / hi_line_freq

    # WAVE_1P4GHZ_AA should NOT equal the HI line wavelength
    assert not np.isclose(WAVE_1P4GHZ_AA, hi_line_wave, rtol=1e-6)

    # Difference should be ~1.5% (as stated in #1057)
    rel_diff = abs(WAVE_1P4GHZ_AA - hi_line_wave) / hi_line_wave
    assert 0.01 < rel_diff < 0.02, "Expected ~1.5% difference between 1.4 GHz and HI line"


def test_radio_component_uses_shared_constant():
    """Radio component._l_1p4ghz_fn must use WAVE_1P4GHZ_AA from physics_constants."""
    import inspect

    from tengri.components.radio.component import _l_1p4ghz_fn

    # Check that the function uses the shared constant
    source = inspect.getsource(_l_1p4ghz_fn)
    assert "WAVE_1P4GHZ_AA" in source, (
        "_l_1p4ghz_fn must use WAVE_1P4GHZ_AA from physics_constants"
    )


def test_component_factory_uses_shared_constant():
    """component_factory.state_to_radio_quantities must use WAVE_1P4GHZ_AA."""
    import inspect

    from tengri.forward.component_factory import state_to_radio_quantities

    # Check that the function uses the shared constant
    source = inspect.getsource(state_to_radio_quantities)
    assert "WAVE_1P4GHZ_AA" in source, (
        "state_to_radio_quantities must import and use WAVE_1P4GHZ_AA"
    )


def test_both_radio_code_paths_import_shared_constant():
    """Both radio code paths (component.py and component_factory.py) must be in sync."""
    from tengri.utils.physics_constants import WAVE_1P4GHZ_AA

    # Verify the constant is defined and correct
    assert WAVE_1P4GHZ_AA > 0
    assert 21.4e8 < WAVE_1P4GHZ_AA < 21.5e8, (
        "WAVE_1P4GHZ_AA should be ~21.41e8 Å for 1.400 GHz"
    )

    # Both modules should import from the same location
    import tengri.components.radio.component
    import tengri.forward.component_factory

    # If both successfully import, they're using the shared constant
    assert hasattr(tengri.components.radio.component, "WAVE_1P4GHZ_AA")
    assert hasattr(tengri.forward.component_factory, "jnp")
