# SPDX-License-Identifier: BSD-3-Clause
"""Spectroscopy fits must default to the exact path, not the SpectrumPrecomp LUT (#1688).

``SpectrumPrecomp`` evaluates the forward model at pixel centers. On a
high-SNR spectrum that bias moves the posterior by about one sigma against
the exact path (#1688, the spectroscopy sibling of #1671). ``approx="auto"``
takes the pixel LUT for spectroscopy only where it serves the pixels exactly
(#2958, ``SEDModel._spectrum_lut_serves_exactly``): a fit with a free
``sigma_v_kms`` stays exact, and point-sampled pixels the table declines stay
exact. Coarse pixels that need the bin integral are served by the table
(#3500). An explicit ``approx=SpectrumPrecomp()`` still engages it. The
emission-line ``FeaturePrecomp`` is a separate mechanism: a spectroscopy fit
with a measured line channel keeps it under ``"auto"``.
"""

import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    ForwardModel,
    Observation,
    SEDModel,
    Spectroscopy,
    SpectrumPrecomp,
)
from tengri.inference.fitter import Fitter, _inner_sed
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.regression_bug

_LINES = ("Halpha", "Hbeta", "OIII_5007")
_WAVES = np.array([6564.61, 4862.71, 5008.24])


def _spectroscopy_fixture(ssp, *, line_fluxes=None, coarse=False, sigma_v_kms=None):
    """64 point-sampled pixels, or with ``coarse`` 50 pixels that need the bin integral.

    ``sigma_v_kms`` declares a Fixed stellar dispersion on the model when given.
    """
    if coarse:
        wave_obs = np.linspace(4500.0, 7500.0, 50)
        spectro = Spectroscopy(resample="auto", wave_obs=wave_obs)
    else:
        wave_obs = np.linspace(4500.0, 7500.0, 64)
        spectro = Spectroscopy(resample="point", wave_obs=wave_obs)
    obs = Observation(spectroscopy=spectro, line_fluxes=line_fluxes)
    model = SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.3,
            "tau_bc": 0.2,
        },
        dust_emission=None,
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )
    if sigma_v_kms is not None:
        spec = model.spec.merge_observation_params(sigma_v_kms=Fixed(sigma_v_kms))
        model = SEDModel(spec, ssp, observation=obs)
    fwd = ForwardModel.build(sed=model, observation=obs)
    n_pix = wave_obs.shape[0]
    return fwd, np.full(n_pix, 1.0), np.full(n_pix, 0.05)


def _line_flux_channel():
    fluxes = np.array([3.0, 1.0, 0.5])
    return LineFluxData(
        names=_LINES,
        fluxes=fluxes,
        errors=fluxes * 0.05,
        wavelengths=_WAVES,
    )


def test_auto_serves_coarse_conserving_pixels_from_the_table(synthetic_ssp_wide):
    """Coarse pixels that need the bin integral are served by the table under auto.

    The table serves a flux-conserving pixel integral at any kernel width (#3500), and
    the undeclared sigma_v_kms is Fixed at 0, so auto takes the table. It must agree with
    the exact path within 1e-4 of the peak.
    """
    fwd, flux, noise = _spectroscopy_fixture(synthetic_ssp_wide, coarse=True)
    fitter = Fitter(fwd, flux, noise, data_type="spectroscopy")  # approx="auto" default
    assert fitter.model._has_modern_approx()
    exact = Fitter(fwd, flux, noise, data_type="spectroscopy", approx=None)
    # The inner SED models: ``ForwardModel.predict_spectrum`` is the exact spectrum on both.
    table_sed, exact_sed = _inner_sed(fitter.model), _inner_sed(exact.model)
    assert table_sed._approx.get("spectrum_precomp") and not exact_sed._approx.get(
        "spectrum_precomp"
    )
    f_auto = np.asarray(table_sed.predict_spectrum({}))
    f_exact = np.asarray(exact_sed.predict_spectrum({}))
    peak = float(np.max(np.abs(f_exact)))
    rel = float(np.max(np.abs(f_auto - f_exact)) / peak)
    # Measured: 6.1e-6 of peak. Not zero: the two are different routes.
    assert 0.0 < rel < 1e-4, "auto table disagrees with exact"


def test_auto_keeps_feature_precomp_for_a_spectroscopy_line_channel(synthetic_ssp_wide):
    """A measured line channel still gets FeaturePrecomp under auto; the pixel LUT stays off."""
    fwd, flux, noise = _spectroscopy_fixture(
        synthetic_ssp_wide, line_fluxes=_line_flux_channel(), coarse=True
    )
    fitter = Fitter(fwd, flux, noise, data_type="spectroscopy")  # approx="auto" default
    state = fitter.model.approx
    assert state.feature_precomp, "a spectroscopy line-channel fit lost FeaturePrecomp"
    assert state.spectrum_precomp, "auto did not serve the conserving pixels from the table"
    assert not state.wave_precomp, "auto engaged WavePrecomp on a spectroscopy fit"


def test_explicit_spectrum_precomp_still_engages(synthetic_ssp_wide):
    fwd, flux, noise = _spectroscopy_fixture(synthetic_ssp_wide, sigma_v_kms=150.0)
    fitter = Fitter(fwd, flux, noise, data_type="spectroscopy", approx=SpectrumPrecomp())
    assert fitter.model._has_modern_approx()
    inner = fitter.model._inner_sed_for_delegation()
    assert inner._approx.get("spectrum_precomp")
