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
    """Fitter reads eline_mode from model.observation.spectroscopy, not dead probes."""
    jnp = pytest.importorskip("jax.numpy")
    import types
    import warnings

    from tengri.inference.fitter import Fitter
    from tengri.observation.spectroscopy import Spectroscopy
    from tengri.parameters.parameters import Parameters
    from tengri.parameters.priors import Fixed, Uniform

    # Build minimal Parameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model_spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.1),
            dust_tau_diff=Fixed(0.1),
            redshift=Fixed(0.1),
        )

    # Create Spectroscopy with eline_mode='off'
    wave = jnp.linspace(4000.0, 7000.0, 300)
    spec = Spectroscopy(wave_obs=wave, eline_mode="off")

    # Create mock model with spectroscopy
    continuum = jnp.ones(len(wave)) * 10.0
    model = types.SimpleNamespace(
        spec=model_spec,
        observation=types.SimpleNamespace(spectroscopy=spec),
        predict_spectrum=lambda params, w=None, **kwargs: continuum,
    )

    # Set a stray _spectroscopy_config attribute on the model.
    # The dead probe getattr(model, "_spectroscopy_config", None) would pick
    # this up; the correct code should ignore it and read from
    # model.observation.spectroscopy instead.
    model._spectroscopy_config = object()

    # Construct the Fitter. This calls _init_emission_lines which should
    # read model.observation.spectroscopy.eline_mode, NOT the stray attribute.
    # Pass spectroscopy data (continuum and noise arrays).
    noise = jnp.ones(len(wave)) * 0.1
    fitter = Fitter(model, continuum, noise, data_type="spectroscopy")

    # With eline_mode="off", both flags should be False
    assert fitter._eline_marginalize is False
    assert fitter._eline_fitted is False
