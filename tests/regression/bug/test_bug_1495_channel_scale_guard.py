# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #1495: likelihood channels have no scale guard.

Tests that CompositeLikelihood checks channel scales at construction time
to catch units mismatches before the fit begins.

Cites: #1495
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_SSP_FILE = "data/fsps_prsc_miles_chabrier.h5"


def _load_ssp():
    """Load SSP data, skipping test if not found."""
    if not Path(_SSP_FILE).is_file():
        pytest.skip(f"SSP data not found: {_SSP_FILE}")
    from tengri import load_ssp_data

    return load_ssp_data(_SSP_FILE)


def test_misscaled_line_channel_raises():
    """Test that a line channel with ~29-order scale mismatch raises at construction.

    Cites: #1495

    Verifies that _check_channel_scales catches pathological units mismatches
    (e.g., line fluxes in erg/s instead of erg/s/cm^2) and raises a ValueError
    during Fitter construction, before the fit begins.
    """
    from tengri import FREE, Observation, Photometry, SEDModel
    from tengri.inference import Fitter
    from tengri.observation.line_flux_data import LineFluxData

    ssp = _load_ssp()

    # Build a minimal model: photometry + lines
    line_wavelengths = np.array([1215.67, 1549.06])
    line_names = ("Lya", "HeII")

    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]),
        line_fluxes=LineFluxData(
            names=line_names,
            wavelengths=line_wavelengths,
            fluxes=np.array([1e-14, 1e-15]),  # Placeholder, will be replaced
            errors=np.array([1e-15, 1e-16]),  # Placeholder
        ),
    )

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": FREE},
        dust={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FREE,
        },
        redshift=0.05,
    )

    # Catastrophic units mismatch: line fluxes supplied ~22 orders too large
    # (erg/s-like numbers instead of erg/s/cm^2) while the ERRORS stay on the
    # correct measurement scale — the real-world shape of the #1495 incident.
    # chi2 ~ ((obs - pred)/err)^2 ~ (1e6/1e-17)^2 ~ 1e46: unrepresentable, and
    # it silently absorbs the photometry channel through rounding.
    phot_data = np.ones(3) * 1e-29  # Reasonable photometry [erg/s/cm2/Hz]
    phot_err = phot_data * 0.05
    line_fluxes = np.array([1e6, 1e6])  # WAY too large for a flux
    line_errors = np.array([1e-17, 1e-18])  # correct measurement scale

    # The guard runs at loss build time (once, eagerly, outside JIT) — before
    # any sampling. Expect a ValueError naming the channel.
    fitter = Fitter(
        model,
        data=phot_data,
        noise=phot_err,
        line_flux_data=LineFluxData(
            names=line_names,
            wavelengths=line_wavelengths,
            fluxes=line_fluxes,
            errors=line_errors,
        ),
    )
    from tengri.inference.loss_functions import build_loss_fn

    with pytest.raises(ValueError, match=r"channel"):
        build_loss_fn(fitter)


def test_healthy_multichannel_fit_constructs_silently():
    """Test that a healthy photometry + line-flux fit constructs and JITs successfully.

    Cites: #1495

    Verifies that _check_channel_scales does NOT interfere with healthy
    multi-channel likelihoods and that the composite likelihood remains
    JIT-compilable after construction.
    """
    import jax

    from tengri import FREE, Observation, Photometry, SEDModel
    from tengri.inference import Fitter
    from tengri.observation.line_flux_data import LineFluxData

    ssp = _load_ssp()

    # Build a minimal model: photometry + lines
    line_wavelengths = np.array([1215.67, 1549.06])
    line_names = ("Lya", "HeII")

    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]),
        line_fluxes=LineFluxData(
            names=line_names,
            wavelengths=line_wavelengths,
            fluxes=np.array([1e-14, 1e-15]),  # Placeholder, will be replaced
            errors=np.array([1e-15, 1e-16]),  # Placeholder
        ),
    )

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": FREE},
        dust={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FREE,
        },
        redshift=0.05,
    )

    # Create mock data with reasonable scales
    phot_data = np.ones(3) * 1e-29  # [erg/s/cm2/Hz]
    phot_err = phot_data * 0.05
    line_fluxes = np.array([1e-14, 1e-15])  # Reasonable line fluxes [erg/s/cm2]
    line_errors = line_fluxes * 0.1

    # This should construct without raising
    fitter = Fitter(
        model,
        data=phot_data,
        noise=phot_err,
        line_flux_data=LineFluxData(
            names=line_names,
            wavelengths=line_wavelengths,
            fluxes=line_fluxes,
            errors=line_errors,
        ),
    )

    # The loss builds silently (the eager pre-check passed) and the built loss
    # — the surface the guard protects — evaluates finite at the prior center.
    from tengri.inference.loss_functions import build_loss_fn

    loss_fn = build_loss_fn(fitter)

    params_unbounded = {name: jnp.asarray(0.0) for name in model.spec.free_params}
    result = jax.jit(loss_fn)(params_unbounded, fitter._data_args)
    assert jnp.isfinite(result), f"loss at prior center not finite: {result}"
