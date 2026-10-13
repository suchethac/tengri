# SPDX-License-Identifier: BSD-3-Clause
"""Spectroscopy fits must default to the exact path, not the SpectrumPrecomp LUT (#1688).

``SpectrumPrecomp`` evaluates the forward model at pixel centers. On a
high-SNR spectrum that bias moves the posterior by about one sigma against
the exact path (#1688, the spectroscopy sibling of #1671). Until a sub-pixel
LUT exists, ``approx="auto"`` takes no pixel LUT for spectroscopy, and an
explicit ``approx=SpectrumPrecomp()`` still engages it. The emission-line
``FeaturePrecomp`` is a separate mechanism: a spectroscopy fit with a measured
line channel keeps it under ``"auto"``, with the pixel LUT still off.
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
from tengri.inference.fitter import Fitter
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.regression_bug

_LINES = ("Halpha", "Hbeta", "OIII_5007")
_WAVES = np.array([6564.61, 4862.71, 5008.24])


def _spectroscopy_fixture(ssp, *, line_fluxes=None):
    wave_obs = np.linspace(4500.0, 7500.0, 64)
    obs = Observation(
        spectroscopy=Spectroscopy(resample="point", wave_obs=wave_obs),
        line_fluxes=line_fluxes,
    )
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


def test_auto_resolves_spectroscopy_to_the_exact_path(synthetic_ssp_wide):
    fwd, flux, noise = _spectroscopy_fixture(synthetic_ssp_wide)
    fitter = Fitter(fwd, flux, noise, data_type="spectroscopy")  # approx="auto" default
    assert not fitter.model._has_modern_approx()
    assert fitter.model is fwd


def test_auto_keeps_feature_precomp_for_a_spectroscopy_line_channel(synthetic_ssp_wide):
    """A measured line channel still gets FeaturePrecomp under auto; the pixel LUT stays off."""
    fwd, flux, noise = _spectroscopy_fixture(synthetic_ssp_wide, line_fluxes=_line_flux_channel())
    fitter = Fitter(fwd, flux, noise, data_type="spectroscopy")  # approx="auto" default
    state = fitter.model.approx
    assert state.feature_precomp, "a spectroscopy line-channel fit lost FeaturePrecomp"
    assert not state.spectrum_precomp, "auto engaged the biased pixel LUT"
    assert not state.wave_precomp, "auto engaged WavePrecomp on a spectroscopy fit"


def test_explicit_spectrum_precomp_still_engages(synthetic_ssp_wide):
    fwd, flux, noise = _spectroscopy_fixture(synthetic_ssp_wide)
    fitter = Fitter(fwd, flux, noise, data_type="spectroscopy", approx=SpectrumPrecomp())
    assert fitter.model._has_modern_approx()
    inner = fitter.model._inner_sed_for_delegation()
    assert inner._approx.get("spectrum_precomp")
