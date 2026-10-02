# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2596: graybody closure carries extra (nu/nu_ref)^beta factor.

The graybody opacity factor (1 - exp(-(lambda0/lambda)^beta)) was being multiplied by an
extra optically-thin emissivity factor (nu/nu_ref)^beta, which is correct for
modified_blackbody but not for the general-opacity graybody model (Casey 2012
Eq. 1, CIGALE mbb.py:78-80, Synthesizer Greybody(optically_thin=False)).

Fix: remove the emissivity factor from the graybody closure, so the shape is
opacity * B_nu only, matching Casey (2012), CIGALE, and Synthesizer.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

pytestmark = pytest.mark.regression_bug


def _bnu(T, wave):
    """Planck function nu^3 / (exp(h*nu/k*T) - 1)."""
    h = 6.62607015e-27
    k = 1.380649e-16
    c = 2.99792458e10
    nu = c / (wave * 1e-8)
    x = h * nu / (k * T)
    return nu**3 / np.expm1(x)


def _reference_graybody(T, beta, lam0_um, wave):
    """Reference: (1 - exp(-(lambda0/lambda)^beta)) * B_nu, normalized to unit power.

    Casey (2012) Eq. 1, second term (general opacity, no extra nu^beta factor).
    Matches CIGALE mbb.py:78-80 and Synthesizer Greybody(optically_thin=False).
    """
    tau = (lam0_um * 1e4 / wave) ** beta
    bnu_val = _bnu(T, wave)
    s = -np.expm1(-tau) * bnu_val
    c = 2.99792458e10
    nu = c / (wave * 1e-8)
    return s / -np.trapezoid(s, nu)


@pytest.mark.parametrize(
    "T,beta,lam0",
    [
        (50.0, 1.5, 200.0),
        (35.0, 1.6, 200.0),
        (25.0, 2.0, 100.0),
        (20.0, 1.5, 100.0),
        (60.0, 1.2, 300.0),
    ],
)
def test_graybody_equals_reference_formula(T, beta, lam0):
    """graybody matches independent numpy expression (1 - exp(-(lambda0/lambda)^beta)) * B_nu.

    Max relative error < 1e-2 in float64 over 8-1000 um.
    References Casey 2012 Eq. 1, CIGALE mbb.py:78-80, Synthesizer Greybody(optically_thin=False).
    """
    wave = np.logspace(4, 7.3, 4000)  # 1 um .. 2 mm in Angstrom

    # Compute graybody via closure
    tg = np.asarray(
        M["graybody"](
            jnp.asarray(wave),
            1.0,
            dust_T=T,
            dust_beta_ir=beta,
            dust_lambda_0_um=lam0,
            dust_epsilon_mbb=1.0,
        )
    )
    c = 2.99792458e10
    nu = c / (wave * 1e-8)
    tg_norm = tg / -np.trapezoid(tg, nu)

    # Reference formula
    ref = _reference_graybody(T, beta, lam0, wave)

    # Select 8-1000 um
    sel = (wave >= 8e4) & (wave <= 1e7)

    # Relative error
    rel_err = np.abs((tg_norm[sel] - ref[sel]) / (ref[sel] + 1e-30))
    max_rel_err = np.max(rel_err)

    assert max_rel_err < 1e-2, (
        f"Relative error {max_rel_err:.3e} exceeds 1e-2 at T={T}, beta={beta}, lam0={lam0}"
    )


def test_graybody_peak_position():
    """nu*L_nu peak for (50 K, 1.5, 200 um) lies in 71.5-72.5 um."""
    T, beta, lam0 = 50.0, 1.5, 200.0
    wave = np.logspace(4, 7.3, 4000)

    tg = np.asarray(
        M["graybody"](
            jnp.asarray(wave),
            1.0,
            dust_T=T,
            dust_beta_ir=beta,
            dust_lambda_0_um=lam0,
            dust_epsilon_mbb=1.0,
        )
    )
    c = 2.99792458e10
    nu = c / (wave * 1e-8)

    peak_idx = np.argmax(tg * nu)
    peak_um = wave[peak_idx] / 1e4

    assert 71.5 <= peak_um <= 72.5, f"Peak position {peak_um:.1f} um outside 71.5-72.5 um"


def test_graybody_ratio_to_modified_blackbody():
    """graybody / modified_blackbody equals (1 − e^{−(λ0/λ)^β}) / (ν/ν_ref)^β up to constant.

    Cell c: At fixed (T=50 K, β=1.5, λ0=200 µm), the shape ratio graybody / mbb divided
    by the expected ratio (opacity / emissivity) should be constant to relative scatter < 1e-6
    over 8–1000 µm. ν_ref = c/250 µm as in the thin closure.
    """
    T, beta, lam0 = 50.0, 1.5, 200.0
    wave = np.logspace(4, 7.3, 4000)

    # Graybody shape: (1 - exp(-(lambda0/lambda)^beta)) * B_nu
    tg = np.asarray(
        M["graybody"](
            jnp.asarray(wave),
            1.0,
            dust_T=T,
            dust_beta_ir=beta,
            dust_lambda_0_um=lam0,
            dust_epsilon_mbb=1.0,
        )
    )

    # Modified blackbody shape: (nu/nu_ref)^beta * B_nu
    tmbb = np.asarray(
        M["modified_blackbody"](
            jnp.asarray(wave),
            1.0,
            dust_T=T,
            dust_beta_ir=beta,
            dust_epsilon_mbb=1.0,
        )
    )

    # Reference formula: ratio = opacity / emissivity
    # = (1 - exp(-(lambda0/lambda)^beta)) / (nu/nu_ref)^beta
    c = 2.99792458e10
    nu = c / (wave * 1e-8)
    nu_ref = c / (250.0 * 1e-4)  # c/250 µm in CGS
    tau = (lam0 * 1e4 / wave) ** beta
    opacity = -np.expm1(-tau)
    emissivity = (nu / nu_ref) ** beta
    ratio_ref = opacity / (emissivity + 1e-30)

    # Compute observed ratio
    obs_ratio = tg / (tmbb + 1e-30)

    # Normalize by expected ratio
    normalized = obs_ratio / (ratio_ref + 1e-30)

    # Select 8-1000 um
    sel = (wave >= 8e4) & (wave <= 1e7)

    # The normalized ratio should be approximately constant (the constant is the relative scale)
    normalized_sel = normalized[sel]
    normalized_sel = normalized_sel / np.mean(normalized_sel)

    # Scatter about the mean
    scatter = np.std(normalized_sel)
    assert scatter < 1e-6, f"Normalized ratio scatter {scatter:.3e} exceeds 1e-6 over 8-1000 um"


# -- public path: SEDModel.build -> sed_dust_ir --------------------------------------

_SSP_NAME = "fsps_prsc_miles_chabrier"
_PUBLIC_T, _PUBLIC_BETA, _PUBLIC_LAM0 = 50.0, 1.5, 200.0


def _require_ssp():
    if not Path(f"data/{_SSP_NAME}.h5").is_file():
        pytest.skip(f"data/{_SSP_NAME}.h5 not available")


def _public_dust_sed(*, x64):
    """``(rest wavelength [A], sed_dust_ir)`` from ``SEDModel.build`` at one precision."""
    with jax.enable_x64(x64):
        ssp = tengri.load_ssp(_SSP_NAME, download=False)
        model = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "tau_v": Fixed(1.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={
                "type": "graybody",
                "dust_T": Fixed(_PUBLIC_T),
                "dust_beta_ir": Fixed(_PUBLIC_BETA),
                "dust_lambda_0_um": Fixed(_PUBLIC_LAM0),
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "ssp"},
            redshift=Fixed(0.0),
        )
        state = model.predict_state({})
        wave = np.asarray(state.wave, dtype=np.float64)
        sed = np.asarray(state.derived["sed_dust_ir"], dtype=np.float64)
    return wave, sed


@pytest.fixture(scope="module")
def public_dust_sed():
    _require_ssp()
    return _public_dust_sed(x64=True)


def test_public_path_shape_matches_reference(public_dust_sed):
    """sed_dust_ir over 8-500 um is a constant multiple of (1 - e^-tau) B_nu (Casey 2012).

    The window stops at 500 um: the closure's z = 0 CMB heating/contrast terms
    (da Cunha+2013) depart from the pure Planck reference by 1.7e-3 at 1000 um.
    """
    wave, sed = public_dust_sed
    sel = (wave >= 8e4) & (wave <= 5e6)
    opacity = -np.expm1(-((_PUBLIC_LAM0 * 1e4 / wave[sel]) ** _PUBLIC_BETA))
    ratio = sed[sel] / (opacity * _bnu(_PUBLIC_T, wave[sel]))
    assert ratio.max() / ratio.min() - 1.0 < 1e-3


def test_public_path_peak_position(public_dust_sed):
    """nu L_nu of the published graybody peaks in 71.5-72.5 um for (50 K, 1.5, 200 um)."""
    wave, sed = public_dust_sed
    nu = 2.99792458e10 / (wave * 1e-8)
    sel = (wave >= 8e4) & (wave <= 1e7)
    peak_um = wave[sel][np.argmax((sed * nu)[sel])] / 1e4
    assert 71.5 <= peak_um <= 72.5, f"peak {peak_um:.2f} um"


def test_public_path_float32_matches_float64(public_dust_sed):
    """The published graybody agrees between float32 and float64 builds to 1e-4."""
    wave64, sed64 = public_dust_sed
    wave32, sed32 = _public_dust_sed(x64=False)
    sel = (wave64 >= 8e4) & (wave64 <= 1e7)
    np.testing.assert_allclose(wave32[sel], wave64[sel], rtol=1e-6)
    np.testing.assert_allclose(sed32[sel], sed64[sel], rtol=1e-4)


# -- precompute path: dust_analytic_precompute ---------------------------------------

_FIR_BANDS_UM = ((70.0, 160.0), (160.0, 500.0), (500.0, 1000.0))
_BAND_IDS = [f"{lo:g}-{hi:g}_um" for lo, hi in _FIR_BANDS_UM]
# Node of each axis grid at which the lookup is queried (the middle node).
# The lookup is a triweight kernel average over the neighboring nodes, so it is a smoothed
# reading of the grid; 1 per cent node spacing keeps that smoothing far below the 1e-3
# tolerances (10 per cent spacing biases a steep far-IR band by ~1 per cent).
_PRECOMP_GRIDS = {
    "T_grid": np.array([49.5, 50.0, 50.5]),
    "beta_grid": np.array([1.485, 1.5, 1.515]),
    "lambda_0_um_grid": np.array([198.0, 200.0, 202.0]),
}
_EDGE_EPS = 1e-6


def _fir_tophats():
    """Top-hat filters with near-vertical edges (zero-transmission nodes 1e-6 outside)."""
    waves, trans = [], []
    for lo, hi in _FIR_BANDS_UM:
        inner = np.geomspace(lo * 1e4, hi * 1e4, 400)
        waves.append(
            np.concatenate([[inner[0] * (1 - _EDGE_EPS)], inner, [inner[-1] * (1 + _EDGE_EPS)]])
        )
        trans.append(np.concatenate([[0.0], np.ones_like(inner), [0.0]]))
    return waves, trans


def _band_average(wave_sed, sed, filter_waves, filter_trans):
    """Filter average of L_nu with the 1/lambda weight used by ``preintegrate_grid``."""
    return np.array(
        [
            np.trapezoid(np.interp(fw, wave_sed, sed) * ft / fw, fw) / np.trapezoid(ft / fw, fw)
            for fw, ft in zip(filter_waves, filter_trans)
        ]
    )


@pytest.fixture(scope="module")
def graybody_precompute_and_exact():
    """Far-IR top-hat photometry: precompute lookup, exact closure, and numpy reference."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    waves, trans = _fir_tophats()
    result = adapter.precompute(waves, trans, 0.0, None, model="graybody", **_PRECOMP_GRIDS)
    lookup = adapter.build_lookup(result, model="graybody")
    phot_precompute = np.asarray(lookup(1.0, 50.0, 1.5, 200.0))

    wide = np.geomspace(1e2, 10**7.8, 40000)  # 0.01 um .. 6.3 mm
    sed = np.asarray(
        M["graybody"](
            jnp.asarray(wide),
            1.0,
            dust_T=50.0,
            dust_beta_ir=1.5,
            dust_lambda_0_um=200.0,
        )
    )
    phot_exact = _band_average(wide, sed, waves, trans)

    nu = 2.99792458e10 / (wide * 1e-8)
    shape = -np.expm1(-((200.0 * 1e4 / wide) ** 1.5)) * _bnu(50.0, wide)
    shape = shape / -np.trapezoid(shape, nu)
    phot_reference = _band_average(wide, shape, waves, trans)
    return phot_precompute, phot_exact, phot_reference


@pytest.mark.parametrize("band", range(len(_FIR_BANDS_UM)), ids=_BAND_IDS)
def test_precompute_matches_exact_in_far_ir_bands(graybody_precompute_and_exact, band):
    """The precomputed graybody photometry reproduces the exact closure to 1e-3 per band."""
    phot_precompute, phot_exact, _ = graybody_precompute_and_exact
    assert phot_exact[band] > 0.0
    np.testing.assert_allclose(phot_precompute[band], phot_exact[band], rtol=1e-3)


@pytest.mark.parametrize("band", range(len(_FIR_BANDS_UM)), ids=_BAND_IDS)
def test_precompute_matches_numpy_reference_in_far_ir_bands(graybody_precompute_and_exact, band):
    """The precomputed photometry equals the numpy (1 - e^-tau) B_nu band average to 2e-3."""
    phot_precompute, _, phot_reference = graybody_precompute_and_exact
    np.testing.assert_allclose(phot_precompute[band], phot_reference[band], rtol=2e-3)


# Every thermal-continuum precompute builder integrates on the same rest-frame grid, so
# the far-IR coverage is checked for each: (model, node-grid kwargs, lookup point, closure kwargs).
_CONTINUUM_MODELS = {
    "modified_blackbody": (
        {"T_grid": np.array([34.65, 35.0, 35.35]), "beta_grid": np.array([1.584, 1.6, 1.616])},
        (35.0, 1.6),
        {"dust_T": 35.0, "dust_beta_ir": 1.6},
    ),
    "casey2012": (
        {**_PRECOMP_GRIDS, "alpha_mir_grid": np.array([1.98, 2.0, 2.02])},
        (50.0, 1.5, 2.0, 200.0),
        {"dust_T": 50.0, "dust_beta_ir": 1.5, "dust_alpha_mir": 2.0, "dust_lambda_0_um": 200.0},
    ),
    "graybody": (
        _PRECOMP_GRIDS,
        (50.0, 1.5, 200.0),
        {"dust_T": 50.0, "dust_beta_ir": 1.5, "dust_lambda_0_um": 200.0},
    ),
}


@pytest.mark.parametrize("model", sorted(_CONTINUUM_MODELS))
@pytest.mark.parametrize("band", range(len(_FIR_BANDS_UM)), ids=_BAND_IDS)
def test_continuum_precompute_covers_far_ir(model, band):
    """Each thermal-continuum precompute reproduces its closure in 70-1000 um bands to 1e-3."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    grids, query, closure_kwargs = _CONTINUUM_MODELS[model]
    waves, trans = _fir_tophats()
    result = adapter.precompute(waves, trans, 0.0, None, model=model, **grids)
    lookup = adapter.build_lookup(result, model=model)
    phot_precompute = np.asarray(lookup(1.0, *query))

    wide = np.geomspace(1e2, 10**7.8, 40000)
    sed = np.asarray(M[model](jnp.asarray(wide), 1.0, **closure_kwargs))
    phot_exact = _band_average(wide, sed, waves, trans)

    assert phot_exact[band] > 0.0
    np.testing.assert_allclose(phot_precompute[band], phot_exact[band], rtol=1e-3)
