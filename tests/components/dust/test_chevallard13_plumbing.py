# SPDX-License-Identifier: BSD-3-Clause
"""``chevallard13`` plumbing: a two-component screen offers its own depth to the law (#3178).

``chevallard13`` reads the screen's V-band optical depth ``tau_v`` (the
curve index depends on it), so every screen that evaluates it must hand it
the depth it applies, not the spec's fixed value and not the default 1.0.
The tests build models through ``SEDModel.build`` with the Cue nebular
backend and check the stellar channel, the shape of the curve in
wavelength, the nebular and line channels, the WavePrecomp LUT, float32 and
gradients.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.dust.attenuation import chevallard13
from tengri.components.dust.laws._registry import resolve_dust_law
from tests.conftest import fd_grad

pytestmark = pytest.mark.regression_paper

_BANDS = ["galex_fuv", "sdss_r", "Herschel_Pacs_green"]
_HALPHA = 6564.72
_HBETA = 4862.68
_V_BAND = 5500.0
_LYMAN_LIMIT = 912.0
#: The SFH ended 0.2 Gyr ago, so every SSP node is older than the 10 Myr birth cloud.
_OLD_STARS_END_GYR = 0.2
#: The WavePrecomp nebular-channel bound of tests/contract/test_precomp_channel_drift.py.
_WAVE_PRECOMP_BOUND = 2e-3


@pytest.fixture(scope="module")
def ssp():
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:  # pragma: no cover - depends on checkout
        pytest.skip(f"default bare-stellar SSP not available: {exc}")


def _build(ssp, dust, *, sfh_end=0.0, approx=None, redshift=0.0):
    kwargs = dict(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "start_gyr": 10.0,
            "end_gyr": sfh_end,
        },
        dust_attenuation=dust,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(redshift),
        approx=approx,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**kwargs)


def _two(law_bc, law_diff, tau_bc, tau_diff, **extra):
    return {
        "type": "two_component",
        "law_bc": law_bc,
        "law_diff": law_diff,
        "tau_bc": tau_bc,
        "tau_diff": tau_diff,
        **extra,
    }


def _full(model, **values):
    """Free-parameter dict: every free key at 0.5 unless ``values`` names it."""
    return {**{k: 0.5 for k in model.spec.free_params}, **values}


def _old_stars_ratio(ssp, law_diff, tau):
    """Attenuated / intrinsic stellar SED of an old-stars-only model, per wavelength.

    Returns ``(wave, ratio, mask)``; ``mask`` drops wavelengths with no stellar flux and
    the ionizing range below the Lyman limit, where the gas takes the photons.
    """
    model = _build(
        ssp,
        _two("power_law", law_diff, Fixed(0.0), Fixed(tau)),
        sfh_end=_OLD_STARS_END_GYR,
    )
    state = model.predict_state(_full(model))
    attenuated = np.asarray(state.derived["sed_dust_attenuated"])
    intrinsic = np.sum(np.asarray(state.derived["lnu_age"]), axis=0)
    mask = (intrinsic > 1e-12 * np.max(intrinsic)) & (np.asarray(state.wave) > _LYMAN_LIMIT)
    ratio = np.where(mask, attenuated / np.where(mask, intrinsic, 1.0), 1.0)
    return np.asarray(state.wave), ratio, mask


@pytest.mark.parametrize("tau", [0.3, 2.0])
def test_old_stars_only_transmission_equals_exp_minus_tau_k_of_own_depth(ssp, tau):
    """T = exp(-tau_diff k(lambda, tau_diff)): the law sees the screen's own depth."""
    wave, ratio, mask = _old_stars_ratio(ssp, "chevallard13", tau)
    expected = np.exp(-tau * np.asarray(chevallard13(jnp.asarray(wave), tau_v=tau)))
    assert_allclose(ratio[mask], expected[mask], rtol=1e-10)


def test_tau_diff_changes_curve_shape_not_only_amplitude(ssp):
    """ln T(tau2) / ln T(tau1) depends on wavelength for chevallard13, not for power_law."""
    _, r1, m1 = _old_stars_ratio(ssp, "chevallard13", 0.3)
    _, r2, m2 = _old_stars_ratio(ssp, "chevallard13", 2.0)
    mask = m1 & m2 & (r1 > 1e-3) & (r2 > 1e-3)
    shape = np.log(r2[mask]) / np.log(r1[mask])
    spread = (np.max(shape) - np.min(shape)) / np.mean(shape)
    assert spread > 1e-2

    _, p1, pm1 = _old_stars_ratio(ssp, "power_law", 0.3)
    _, p2, pm2 = _old_stars_ratio(ssp, "power_law", 2.0)
    pmask = pm1 & pm2 & (p1 > 1e-3) & (p2 > 1e-3)
    pshape = np.log(p2[pmask]) / np.log(p1[pmask])
    assert (np.max(pshape) - np.min(pshape)) / np.mean(pshape) < 1e-8


def test_at_tau_one_equals_power_law_minus_0p7_at_v_band_only():
    """At tau = 1 the curve matches the power law of slope -0.7 at V, and only there."""
    v = jnp.asarray([_V_BAND])
    power_law = resolve_dust_law("power_law")
    assert_allclose(
        np.asarray(chevallard13(v, tau_v=1.0)),
        np.asarray(power_law(v, dust_slope=-0.7)),
        rtol=1e-12,
    )
    h = 1e-4
    slope_wave = jnp.asarray([_V_BAND * np.exp(-h), _V_BAND * np.exp(h)])
    k = np.asarray(chevallard13(slope_wave, tau_v=1.0))
    assert abs((np.log(k[1]) - np.log(k[0])) / (2.0 * h) + 0.7) < 1e-4
    uv = jnp.asarray(np.geomspace(1000.0, 25000.0, 200))
    assert not np.allclose(
        np.asarray(chevallard13(uv, tau_v=1.0)),
        np.asarray(power_law(uv, dust_slope=-0.7)),
        rtol=1e-2,
        atol=0.0,
    )


def test_wave_precomp_matches_exact_with_free_tau_diff(ssp):
    """The WavePrecomp LUT tracks the exact path while tau_diff is free; node-exact at 3.0."""
    dust = _two("power_law", "chevallard13", Fixed(0.3), Uniform(0.0, 3.0))
    exact = _build(ssp, dust)
    lut = _build(ssp, dust, approx=WavePrecomp())
    p = _full(exact, dust_tau_diff=1.7)
    a = np.asarray(exact.predict_photometry(p))
    b = np.asarray(lut.predict_photometry(p))
    assert float(np.max(np.abs(b - a) / np.abs(a))) < _WAVE_PRECOMP_BOUND

    p3 = _full(exact, dust_tau_diff=3.0)
    l_exact = np.asarray(exact.predict_state(p3).derived["L_absorbed"])
    l_lut = np.asarray(lut.predict_state(p3).derived["L_absorbed"])
    assert_allclose(l_lut, l_exact, rtol=1e-8)


def _diffuse_nebular_model(ssp, tau):
    """Every nebular and line photon passes the diffuse screen (``nebular_screen='diffuse'``)."""
    dust = _two(
        "power_law",
        "chevallard13",
        Fixed(0.0),
        Fixed(tau),
        nebular_screen="diffuse",
    )
    return _build(ssp, dust)


def test_nebular_and_line_channels_see_the_depth_dependent_curve(ssp):
    """The nebular continuum and the lines take exp(-tau k(lambda, tau)) at the screen's own depth.

    ``chevallard13`` is a diffuse-ISM law, so the galaxy-wide screen is the diffuse one:
    ``nebular_screen='diffuse'`` puts the nebular continuum and the lines behind the same
    depth-dependent curve. The tau = 0 model supplies the intrinsic continuum and lines.
    """
    targets = jnp.asarray([_HALPHA, _HBETA])
    models = {tau: _diffuse_nebular_model(ssp, tau) for tau in (0.0, 0.3, 2.0)}
    base = models[0.0].predict_state({})
    wave = np.asarray(base.wave)
    neb0 = np.asarray(base.derived["sed_nebular"])
    line_waves = np.asarray(base.derived["line_waves"])
    lines0 = np.asarray(models[0.0].predict_line_fluxes({}, target_wavelengths=targets))
    mask = (neb0 > 1e-12 * np.max(neb0)) & (wave > _LYMAN_LIMIT)
    # Each target is matched to the nearest catalog line; the curve is evaluated there.
    lines = jnp.asarray([line_waves[np.argmin(np.abs(line_waves - t))] for t in (_HALPHA, _HBETA)])
    for tau in (0.3, 2.0):
        state = models[tau].predict_state({})
        transmission = np.asarray(state.derived["sed_nebular"]) / neb0
        expected = np.exp(-tau * np.asarray(chevallard13(jnp.asarray(wave), tau_v=tau)))
        assert_allclose(transmission[mask], expected[mask], rtol=1e-10)
        flux = np.asarray(models[tau].predict_line_fluxes({}, target_wavelengths=targets))
        expected_lines = np.exp(-tau * np.asarray(chevallard13(lines, tau_v=tau)))
        assert_allclose(flux / lines0, expected_lines, rtol=1e-10)


def test_float32_matches_float64(ssp):
    """The float32 build agrees with the float64 build to 1e-4."""
    dust = _two("power_law", "chevallard13", Fixed(0.5), Fixed(1.5))
    a = np.asarray(_build(ssp, dust).predict_photometry({}))
    with jax.enable_x64(False):
        b = np.asarray(_build(ssp, dust).predict_photometry({}))
    assert_allclose(b, a, rtol=1e-4)


def test_gradient_wrt_tau_diff_finite_matches_fd(ssp):
    """d(photometry)/d(tau_diff) through chevallard13 is finite and matches finite differences."""
    model = _build(ssp, _two("power_law", "chevallard13", Fixed(0.3), Uniform(0.0, 3.0)))

    def total(x):
        return jnp.sum(model.predict_photometry(_full(model, dust_tau_diff=x)))

    grad = float(jax.grad(total)(jnp.asarray(1.0)))
    assert np.isfinite(grad)
    assert grad != 0.0, (
        "an identically zero d(photometry)/d(tau_diff) would mean the screen is not read"
    )
    fd = fd_grad(lambda x: float(total(jnp.asarray(x))), 1.0)
    assert_allclose(grad, fd, rtol=1e-3)
