# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #2662 (item 1): CIGALE's ``lambda_fracAGN`` window for ``agn_ir_frac``.

CIGALE ``skirtor2016`` scales the AGN dust power by the fraction of the light in a
wavelength window, ``agn_power = scale * f / (1 - f)`` with ``scale = L_host(window) /
AGNlumin(window)`` (``pcigale/sed_modules/skirtor2016.py:473-498``). ``"0/0"`` is the
total-dust case that tengri has always implemented. ``agn={'ir_frac_window': (lmin, lmax)}``
(or CIGALE's ``'5/20'``) selects a window in micrometers.

Tier: regression_bug.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.config.exceptions import ParameterError
from tengri.parameters.groups import _parse_ir_frac_window

pytestmark = pytest.mark.regression_bug

_C_AA = 2.99792458e18  # speed of light [Angstrom/s]
_SFH = {
    "type": "declining_exp",
    "sfh_declining_exp_tau_gyr": Fixed(0.3),
    "sfh_declining_exp_age_gyr": Fixed(1.0),
    "log_total_mass": Fixed(10.0),
}


def _agn(window=None, frac=0.3, **extra):
    agn = {
        "type": "composable",
        "disc": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "nlr": {"type": "none"},
        "blr": {"type": "none"},
        "atten": {"type": "none"},
        "agn_ir_frac": frac if not isinstance(frac, float) else Fixed(frac),
        "all_params": Fixed(DEFAULT),
        "norm": "cigale_joint",
        **extra,
    }
    if window is not None:
        agn["ir_frac_window"] = window
    return agn


def _build(ssp, *, window=None, frac=None, with_agn=True):
    kwargs = {}
    if with_agn:
        kwargs["agn"] = _agn(window=window, frac=Fixed(0.3) if frac is None else frac)
    return SEDModel.build(
        ssp_data=ssp,
        sfh=_SFH,
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.0),
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        **kwargs,
    )


@pytest.fixture(scope="module")
def ssp(ssp_data_wne):
    return ssp_data_wne


def _pieces(model):
    """Rest wavelength [A], total AGN L_nu, and the host L_nu (everything but the AGN)."""
    pred = model.predict({})
    comps = pred.sed.components
    wave = np.asarray(comps["wavelength"], float)
    agn = np.asarray(comps["sed_agn"], float)
    host = np.asarray(comps["sed_total"], float) - agn
    return wave, agn, host


def _window_integral(wave, lnu, lo, hi):
    """pcigale's window integral: L_lambda = L_nu c / lambda^2 over [lo, hi] with edges interp."""
    lam = np.concatenate(([lo], wave[(wave > lo) & (wave < hi)], [hi]))
    f = np.interp(lam, wave, lnu / wave**2)
    return float(np.trapezoid(f, lam))


def _lum_at(wave, lnu, lam):
    return float(np.interp(lam, wave, lnu))


def test_a_zero_window_and_absent_key_are_bit_identical(ssp):
    """(a) (0, 0) and no key give the same photometry and AGN SED, bit for bit."""
    absent = _build(ssp).predict({})
    zero = _build(ssp, window=(0.0, 0.0)).predict({})
    np.testing.assert_array_equal(np.asarray(absent.photometry()), np.asarray(zero.photometry()))
    np.testing.assert_array_equal(
        np.asarray(absent.sed.components["sed_agn"]),
        np.asarray(zero.sed.components["sed_agn"]),
    )


def test_b_window_ratio_is_f_over_one_minus_f(ssp):
    """(b) Window (5, 20) um: AGN/host window luminosity ratio is f/(1-f) to 1e-6."""
    frac = 0.3
    model = _build(ssp, window=(5.0, 20.0), frac=Fixed(frac))
    wave, agn, host = _pieces(model)
    ratio = _window_integral(wave, agn, 5.0e4, 2.0e5) / _window_integral(wave, host, 5.0e4, 2.0e5)
    np.testing.assert_allclose(ratio, frac / (1.0 - frac), rtol=1e-6)


def test_c_monochromatic_ratio_is_f_over_one_minus_f(ssp):
    """(c) Monochromatic (10, 10) um: AGN/host L_lambda at 10 um is f/(1-f) to 1e-6."""
    frac = 0.3
    model = _build(ssp, window=(10.0, 10.0), frac=Fixed(frac))
    wave, agn, host = _pieces(model)
    ratio = _lum_at(wave, agn, 1.0e5) / _lum_at(wave, host, 1.0e5)
    np.testing.assert_allclose(ratio, frac / (1.0 - frac), rtol=1e-6)


def test_d_formula_matches_pcigale_on_synthetic_host():
    """(d) Flat L_lambda host and one-piece AGN template: pcigale's agn_power, computed here.

    pcigale (skirtor2016.py:473-498): scale = int_w L_host dlam / AGNlumin, with
    AGNlumin = int_w L_agn dlam / D_unit and D_unit the template's dust luminosity;
    agn_power = scale * f / (1 - f) (skirtor2016.py:498). Here D = int L_lambda dlam over the grid.
    """
    from tengri.components.agn._ir_frac_window import agn_power_from_window

    wave = np.geomspace(1.0e3, 1.0e6, 4000)  # rest grid [A]
    k_host, k_agn = 3.0e-5, 7.0e-5  # flat L_lambda [erg/s/A]
    host_nu = k_host * wave**2 / _C_AA
    agn_nu = k_agn * wave**2 / _C_AA
    frac, lo, hi = 0.3, 5.0e4, 2.0e5

    h_w = k_host * (hi - lo)
    a_w = k_agn * (hi - lo)
    # Flat L_lambda: the grid's total dust luminosity is exactly k_agn times the grid span.
    dust_total = k_agn * (wave[-1] - wave[0])
    expected = h_w * (frac / (1.0 - frac)) * dust_total / a_w

    got = agn_power_from_window(
        jnp.asarray(wave),
        jnp.asarray(host_nu),
        jnp.asarray(agn_nu),
        jnp.asarray(agn_nu),
        frac,
        lo,
        hi,
    )
    np.testing.assert_allclose(float(got), expected, rtol=1e-6)


@pytest.mark.parametrize(
    "bad, match",
    [
        ((-1.0, 5.0), r"-1(\.0)?"),
        ((20.0, 5.0), r"20(\.0)?.*5(\.0)?"),
    ],
)
def test_e_refuses_bad_window_numbers(bad, match):
    """(e) Negative or inverted windows are refused and the message names the numbers."""
    with pytest.raises(ParameterError, match=match):
        _parse_ir_frac_window(bad)


def test_e_refuses_nonzero_window_without_fraction(ssp):
    """(e) A non-zero window with agn_ir_frac absent or Fixed(0) has nothing to scale."""
    with pytest.raises(ParameterError, match="agn_ir_frac"):
        _build(ssp, window=(5.0, 20.0), frac=Fixed(0.0))


def test_e_refuses_window_beyond_rest_grid(ssp):
    """(e) A window outside the model's rest wavelength grid is refused with its numbers."""
    with pytest.raises(ParameterError, match=r"1000000"):
        _build(ssp, window=(5.0, 1.0e6), frac=Fixed(0.3))


def test_e_cigale_string_equals_tuple():
    """(e) CIGALE's '5/20' is the same option as (5.0, 20.0)."""
    assert _parse_ir_frac_window("5/20") == (5.0, 20.0)
    assert _parse_ir_frac_window((5, 20)) == (5.0, 20.0)


def test_f_gradient_through_window_is_finite_and_nonzero(ssp):
    """(f) d(photometry)/d(agn_ir_frac) with a window is finite and non-zero."""
    model = _build(ssp, window=(5.0, 20.0), frac=Uniform(0.05, 0.9))

    def band_flux(f):
        phot = model.predict_photometry({"agn_ir_frac": f})
        return jnp.asarray(phot)[0]

    grad = float(jax.grad(band_flux)(jnp.asarray(0.3)))
    assert np.isfinite(grad)
    assert grad != 0.0
