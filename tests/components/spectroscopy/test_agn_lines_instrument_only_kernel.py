# SPDX-License-Identifier: BSD-3-Clause
"""AGN emission lines receive the instrument kernel alone (#2565).

Every emission line an AGN component paints (the composable NLR / BLR / FeII
blocks, GRAHSP's Gaussians, QSOGen's line template) is painted at its own
intrinsic width and never passes through the stellar library. The observed
width is therefore ``sqrt(sigma_line**2 + sigma_inst**2)`` and must not
depend on the stellar velocity dispersion ``sigma_v``. Before the fix the
lines were broadened by the stellar kernel
``sqrt(sigma_v**2 + sigma_inst**2 - sigma_lib(lambda)**2)``, so a 300 km/s
narrow line read 354.7 km/s at ``sigma_v = 200`` km/s (+18 %).

The expected width is derived, not fitted: the Gaussian ``sigma`` follows from
the declared FWHM (``sigma = FWHM / 2 sqrt(2 ln 2)``) and the instrument term
from the declared resolving power (``sigma_inst = c / (2 sqrt(2 ln 2) R)``).
The fit leaves the line center free (the catalog carries vacuum wavelengths)
and a linear continuum, on an observed grid much finer than the line.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import curve_fit

from tengri import DEFAULT, FREE, Fixed, Observation, SEDModel, Uniform, load_ssp_data
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.regression_bug

_C_KMS = 299792.458
_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
_MILES_BARE = "data/fsps_prsc_miles_chabrier.h5"
_SIGMA_INST_KMS = 50.0
_SIGMA_V_GRID = (0.0, 200.0, 400.0)
_Z = 0.05
#: Stellar mass low enough that the host continuum is negligible next to the AGN.
_LOG_MASS = 6.0

_DISC = {"type": "powerlaw", "all_params": Fixed(DEFAULT)}
_TWO_COMPONENT = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
_SINGLE = {"type": "single_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}


def _composable(**blocks):
    return {
        "type": "composable",
        "norm": "independent",
        "disc": _DISC,
        **{k: {"type": v, "all_params": Fixed(DEFAULT)} for k, v in blocks.items()},
        "all_params": Fixed(DEFAULT),
    }


# id -> (agn config, dust config, rest-frame vacuum center [A], line FWHM [km/s],
#        half-window [A])
_CASES = {
    "nlr_two_component": (
        _composable(nlr="analytic"),
        _TWO_COMPONENT,
        5008.24,
        500.0,
        15.0,
    ),
    "nlr_agn_screen_diffuse": (
        _composable(nlr="analytic"),
        {**_TWO_COMPONENT, "agn_screen": "diffuse"},
        5008.24,
        500.0,
        15.0,
    ),
    "nlr_agn_screen_birth_cloud": (
        _composable(nlr="analytic"),
        {**_TWO_COMPONENT, "agn_screen": "birth_cloud"},
        5008.24,
        500.0,
        15.0,
    ),
    "nlr_single_component": (
        _composable(nlr="analytic"),
        _SINGLE,
        5008.24,
        500.0,
        15.0,
    ),
    "blr_two_component": (
        _composable(blr="analytic"),
        _TWO_COMPONENT,
        6564.61,
        5000.0,
        250.0,
    ),
    "grahsp_monolithic": (
        {
            "type": "grahsp",
            "agn_grahsp_linewidth_kms": Fixed(1000.0),
            "agn_grahsp_a_feii": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        _TWO_COMPONENT,
        4862.68,
        1000.0,
        60.0,
    ),
}


def _resolution_for_sigma_inst(sigma_inst_kms: float) -> float:
    return _C_KMS / (_FWHM_TO_SIGMA * sigma_inst_kms)


def _gaussian_width_kms(wave: np.ndarray, flux: np.ndarray) -> float:
    """Least-squares Gaussian (free amplitude, center, sigma; linear continuum)."""
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux = (flux - np.median(flux)) / (np.max(flux) - np.median(flux))  # O(1) for the optimizer
    j = int(np.argmax(flux))
    slope0 = (flux[-1] - flux[0]) / (wave[-1] - wave[0])

    def model(x, amp, mu, sig, c0, c1):
        return c0 + c1 * (x - mu) + amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    p0 = [1.0, wave[j], 1.0e-3 * wave[j], 0.0, slope0]
    popt, _ = curve_fit(model, wave, flux, p0=p0, maxfev=20000)
    return float(abs(popt[2]) / popt[1] * _C_KMS)


def _ssp_or_skip():
    if not Path(_MILES_BARE).is_file():
        pytest.skip(f"missing SSP grid {_MILES_BARE}")
    return load_ssp_data(_MILES_BARE)


def _build_model(case: str, wave_obs):
    agn, dust, *_ = _CASES[case]
    ssp = _ssp_or_skip()
    obs = Observation(
        spectroscopy=Spectroscopy(
            wave_obs=wave_obs,
            resolution=_resolution_for_sigma_inst(_SIGMA_INST_KMS),
            sigma_lib_kms=70.0,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation=dust,
            agn=agn,
            redshift=Fixed(_Z),
        )
        merged = base.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0))
        return SEDModel(merged, ssp, observation=obs)


def _measure(case: str, sigma_v: float) -> float:
    _, _, center_rest, _, half = _CASES[case]
    center_obs = center_rest * (1.0 + _Z)
    wave_obs = jnp.linspace(center_obs - half, center_obs + half, 3000)
    model = _build_model(case, wave_obs)
    p = dict(model.spec.sample(jax.random.PRNGKey(1)))
    p["sfh_dpl_log_total_mass"] = jnp.asarray(_LOG_MASS)
    p["sigma_v_kms"] = jnp.asarray(sigma_v)
    flux = np.asarray(model.predict_spectrum(p))
    return _gaussian_width_kms(np.asarray(wave_obs), flux)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_agn_line_width_is_instrument_only_and_flat_in_sigma_v(case):
    """Measured width = sqrt(sigma_line^2 + sigma_inst^2) within 2 %, flat in sigma_v (< 1 %)."""
    _, _, _, fwhm, _ = _CASES[case]
    sigma_line = fwhm / _FWHM_TO_SIGMA
    expected = float(np.hypot(sigma_line, _SIGMA_INST_KMS))

    widths = {sv: _measure(case, sv) for sv in _SIGMA_V_GRID}

    # Linear interpolation of a line on the 0.9 Å rest grid (~54 km/s at
    # 5000 Å) adds a variance of (grid step)^2 / 6 to the measured width:
    # +0.5 % for a 212 km/s line, less for broader ones. 1 % covers it.
    for sv, w in widths.items():
        assert w == pytest.approx(expected, rel=1e-2), (
            f"{case}: sigma_v={sv} km/s measured {w:.2f} km/s, expected {expected:.2f} km/s "
            f"(all: {widths})"
        )
    spread = (max(widths.values()) - min(widths.values())) / expected
    assert spread < 1e-2, f"{case}: width varies {100 * spread:.2f} % across sigma_v: {widths}"
