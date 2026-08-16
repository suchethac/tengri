# SPDX-License-Identifier: BSD-3-Clause
"""Integration: non-baked (Cue) nebular emission is included AND correctly
attenuated under the precompute paths.

``NebularSEDComponent`` publishes a spectrum LUT family (``nebular_spec_lnu_precomp``),
and both precompute consumers redden the nebular emission by the **young-limit**
screen (birth cloud + diffuse, ``T_bc·T_diff``) — matching the exact path, where
the nebular SED is reddened by both screens. Regressions guarded:

* SpectrumPrecomp previously **dropped** Cue nebular entirely (emission lines
  vanished); a naive un-attenuated re-add then over-counted the lines ~2.3x
  because the consumer applied diffuse-only attenuation.
* WavePrecomp left nebular-line-dominated bands ~18 % (τ=0.5) – ~37 % (τ=1) too
  bright for the same diffuse-only reason.

Requires the FSPS SSP + Cue weights; skips otherwise.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, SpectrumPrecomp, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectroscopy import Spectroscopy

_DATA = Path(__file__).resolve().parents[2] / "data"
_SSP = _DATA / "fsps_prsc_miles_chabrier.h5"
_CUE = _DATA / "cue_weights.npz"
pytestmark = [
    pytest.mark.contract,
    pytest.mark.skipif(
        not (_SSP.is_file() and _CUE.is_file()),
        reason="needs data/fsps_prsc_miles_chabrier.h5 + data/cue_weights.npz",
    ),
]


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_SSP))


def _sfh_dust_neb():
    return dict(
        sfh={
            "type": "tsnorm",
            "*": FIXED,
            "log_total_mass": Fixed(10.0),
            "peak_lbt_gyr": Fixed(0.1),
            "width_gyr": Fixed(0.3),
        },
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "cue", "*": FIXED},
    )


@pytest.mark.parametrize("tau", [0.5, 1.0])
def test_spectrum_precomp_nebular_line_matches_exact(ssp, tau):
    """The Cue Hα line flux under SpectrumPrecomp matches the exact path."""
    wave = jnp.linspace(6400.0, 6700.0, 300)
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave))
    build = lambda approx: SEDModel.build(  # noqa: E731
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(0.0),
        apply_igm=False,
        approx=approx,
        **_sfh_dust_neb(),
    )
    sf = {**build(None).spec.get_fixed_values(), "dust_tau_bc": tau, "dust_tau_diff": tau}
    w = np.asarray(wave)
    line, cont = (w > 6540) & (w < 6590), (w < 6520) | (w > 6610)
    lf = lambda s: float(np.trapezoid(s[line] - np.median(s[cont]), w[line]))  # noqa: E731
    s_exact = np.asarray(build(None).predict_spectrum(sf))
    s_lut = np.asarray(build(SpectrumPrecomp()).predict_spectrum(sf))
    assert lf(s_exact) > 0  # there is a real emission line to match
    assert abs(lf(s_lut) / lf(s_exact) - 1.0) < 0.02


@pytest.mark.parametrize("tau", [0.5, 1.0])
def test_wave_precomp_nebular_band_matches_exact(ssp, tau):
    """A Cue Hα-dominated band under WavePrecomp matches the exact path
    (BC+diffuse attenuation), within the #617 continuum Taylor residual."""

    def th(c, n):
        wv = jnp.linspace(c * 0.96, c * 1.04, 40)
        return FilterCurve(wave=wv, trans=jnp.sin(jnp.linspace(0, jnp.pi, 40)) * 0.6, name=n)

    obs = Observation(photometry=Photometry(filters=(th(6563.0, "Ha"),)))
    build = lambda approx: SEDModel.build(  # noqa: E731
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(0.0),
        apply_igm=False,
        approx=approx,
        **_sfh_dust_neb(),
    )
    sf = {**build(None).spec.get_fixed_values(), "dust_tau_bc": tau, "dust_tau_diff": tau}
    p_exact = float(np.asarray(build(None).predict_photometry(sf))[0])
    p_lut = float(np.asarray(build(WavePrecomp()).predict_photometry(sf))[0])
    # Was 1.18 (τ=0.5) / 1.37 (τ=1) with diffuse-only; now within the residual.
    assert abs(p_lut / p_exact - 1.0) < 0.05
