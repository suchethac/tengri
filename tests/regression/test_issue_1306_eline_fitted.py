# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1306: eline_mode='fitted' registers amplitude free parameters.

When eline_mode='fitted', emission line amplitudes should be registered as free
parameters in the Fitter and accessible via _free_names. The root cause was that
amplitudes were merged into the spec in _init_eline_arrays, but then the fit-time
approx policy (line 540-541 in Fitter.__init__) reassigned self.spec to the
approx-resolved model's spec, which lacked the merged amplitude priors. The fix
re-applies the merge after the reassignment (lines 551-552).
"""

from __future__ import annotations

import types
import warnings

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def test_eline_fitted_amplitude_names_in_free_names():
    """Amplitude parameters registered in eline_mode='fitted' appear in _free_names.

    Regression for issue #1306: the fit-time approx policy reassigns self.spec,
    dropping amplitude priors merged in _init_eline_arrays. The fix re-applies
    the merge after the reassignment, ensuring amplitude names are in _free_names.
    """
    from tengri.inference.fitter import Fitter
    from tengri.observation.spectroscopy import Spectroscopy
    from tengri.parameters.parameters import Parameters
    from tengri.parameters.priors import Fixed, Uniform

    # Build minimal Parameters with one free SFH param
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

    # Create Spectroscopy with eline_mode='fitted'
    wave = jnp.linspace(4000.0, 7000.0, 300)
    cfg = Spectroscopy(wave_obs=wave, eline_mode="fitted")

    # Create mock model
    continuum = jnp.ones(len(wave)) * 10.0
    model = types.SimpleNamespace(
        spec=model_spec,
        wave_obs=wave,
        _spectral_resolution=2000.0,
        _spectroscopy_config=cfg,
        predict_spectrum=lambda params, w=None, **kwargs: continuum,
        observation=None,
    )

    # Create Fitter without running fit — just construct
    fitter = Fitter(
        model, jnp.ones(len(wave)) * 10.0, jnp.ones(len(wave)) * 0.1, data_type="spectroscopy"
    )

    # Assert amplitude names are in _free_names
    for amp_name in fitter._eline_amplitude_names:
        assert amp_name in fitter._free_names, (
            f"Amplitude parameter {amp_name!r} missing from _free_names; "
            f"this suggests the fit-time approx reassignment dropped the merged priors. "
            f"Present names: {sorted(fitter._free_names)[:10]}"
        )
