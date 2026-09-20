# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2191: eline_mode legal set consistency.

The issue: Spectroscopy accepted 'fixed' which no consumer reads, and
two validators disagree on the legal set (settings.py rejects 'fitted').
The fix: one legal set defined once, imported by both validators.
"""

from __future__ import annotations

import pytest

from tengri.config.settings import NebularConfig
from tengri.observation.constants import ELINE_MODES
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.regression_bug


def test_spectroscopy_eline_mode_fixed_retired() -> None:
    """Spectroscopy('fixed') is rejected with a helpful error."""
    wave = pytest.importorskip("jax.numpy").linspace(4000.0, 9000.0, 100)
    with pytest.raises(ValueError) as exc_info:
        Spectroscopy(wave_obs=wave, eline_mode="fixed")
    error_msg = str(exc_info.value)
    assert "fixed" in error_msg
    assert "modes" in error_msg.lower()


def test_settings_eline_mode_fitted_accepted() -> None:
    """NebularConfig accepts 'fitted' (not rejected)."""
    cfg = NebularConfig(eline_mode="fitted")
    assert cfg.eline_mode == "fitted"


def test_eline_modes_constant_exported() -> None:
    """ELINE_MODES constant is defined and exported."""
    assert isinstance(ELINE_MODES, tuple)
    assert "off" in ELINE_MODES
    assert "marginalized" in ELINE_MODES
    assert "fitted" in ELINE_MODES
    assert "fixed" not in ELINE_MODES


def test_validators_agree_on_eline_modes() -> None:
    """Both Spectroscopy and NebularConfig validators accept the same set."""
    wave = pytest.importorskip("jax.numpy").linspace(4000.0, 9000.0, 100)

    # Test set: ELINE_MODES plus three intentionally bad strings
    test_strings = [*ELINE_MODES, "bad_mode", "FITTED", "Fixed"]

    for mode_str in test_strings:
        is_valid_eline_modes = mode_str in ELINE_MODES

        # Test Spectroscopy acceptance
        spectroscopy_accepted = False
        try:
            Spectroscopy(wave_obs=wave, eline_mode=mode_str)
            spectroscopy_accepted = True
        except ValueError:
            spectroscopy_accepted = False

        # Test NebularConfig acceptance
        config_accepted = False
        try:
            NebularConfig(eline_mode=mode_str)
            config_accepted = True
        except ValueError:
            config_accepted = False

        # Both should agree: accept if in ELINE_MODES, reject otherwise
        assert spectroscopy_accepted == is_valid_eline_modes, (
            f"Spectroscopy and ELINE_MODES disagree on '{mode_str}': "
            f"Spectroscopy {spectroscopy_accepted} vs constant {is_valid_eline_modes}"
        )
        assert config_accepted == is_valid_eline_modes, (
            f"NebularConfig and ELINE_MODES disagree on '{mode_str}': "
            f"NebularConfig {config_accepted} vs constant {is_valid_eline_modes}"
        )


def test_fitter_spectroscopy_config_read_path() -> None:
    """Verify dead _spectroscopy_config read path was removed from Fitter._init_emission_lines."""
    import inspect

    from tengri.inference.fitter import Fitter

    # Read the source code of _init_emission_lines
    source = inspect.getsource(Fitter._init_emission_lines)

    # The dead probe getattr(model, "_spectroscopy_config", None) should NOT appear
    # The code should directly get spectroscopy from model.observation.spectroscopy
    assert 'getattr(model, "_spectroscopy_config"' not in source, (
        "Dead probe read of _spectroscopy_config should be removed; "
        "spectroscopy config is obtained via model.observation.spectroscopy"
    )
