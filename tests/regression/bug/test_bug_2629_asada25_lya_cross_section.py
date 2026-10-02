# SPDX-License-Identifier: BSD-3-Clause
"""#2629: the asada25 CGM Lyα damping-wing cross-section carried f twice.

The Lorentzian Lyα wing is (Miralda-Escudé 1998, ApJ 501, 15, Eq. 1; Totani et al.
2006, PASJ 58, 485)

    sigma(nu) = [3 lam^2 A / (8 pi)] * A r^4 / [4 pi^2 (nu - nu_a)^2 + A^2 r^6 / 4],
    r = nu / nu_a,

and the prefactor 3 lam^2 A/(8 pi) equals the sum-rule value pi e^2 f/(m_e c) with
g2/g1 = 3, so the oscillator strength is already inside A. The code multiplied f in a
second time (0.0046 against 0.01105 cm^2 Hz).

Every expected value below is computed here from the formula and CODATA constants
written in this file. Nothing is imported from ``tengri.components.igm.dla`` except the
DLA Voigt cross-section that the wing-ratio test compares against.

Constants: lam_a = 1215.6701 Angstrom is the rest wavelength in ``dla.py`` (Morton 2003).
A 4-digit 1215.67 differs by 8e-9 relative, which shifts nu - nu_a at |dnu|/nu_a = 1e-4 by
8e-5 and the cross-section by 1.6e-4, so the 1e-6 comparisons use the 7-digit value.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.components.igm.dla import _deltanu_doppler, _sigma_lya
from tengri.components.igm.igm import (
    _cgm_damping_wing_tau,
    igm_transmission,
    igm_transmission_asada25,
)
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

# CODATA, cgs
_E = 4.80320471e-10  # esu
_ME = 9.1093837015e-28  # g
_C = 2.99792458e10  # cm/s
_C_KMS = 2.99792458e5  # km/s
# Lyα atomic data
_LAM_ANG = 1215.6701  # Angstrom
_A = 6.265e8  # s^-1
_F = 0.4164  # sum-rule oscillator strength
_NU_A = _C / (_LAM_ANG * 1e-8)
_PREFACTOR = 3.0 * (_LAM_ANG * 1e-8) ** 2 * _A / (8.0 * np.pi)
_SUM_RULE = np.pi * _E**2 * _F / (_ME * _C)


def _n_hi(z):
    """Asada et al. (2025) N_HI(z), the sigmoid in the ``_cgm_damping_wing_tau`` docstring."""
    return 10.0 ** (3.592 / (1.0 + np.exp(-1.841 * (z - 6.0))) + 18.001)


def _sigma_formula(wave_rest_ang):
    """Miralda-Escudé (1998) Eq. 1 Lorentzian wing [cm^2] at rest wavelengths [Angstrom]."""
    nu = _C / (np.asarray(wave_rest_ang, dtype=float) * 1e-8)
    r = nu / _NU_A
    return _PREFACTOR * _A * r**4 / (4.0 * np.pi**2 * (nu - _NU_A) ** 2 + _A**2 * r**6 / 4.0)


def _rest_at_velocity(dv_kms):
    """Rest wavelength [Angstrom] a velocity dv redward of Lyα."""
    return _LAM_ANG * (1.0 + dv_kms / _C_KMS)


def _rest_at_frac(x):
    """Rest wavelength [Angstrom] with nu = nu_a (1 - x), i.e. redward by |dnu|/nu_a = x."""
    return _LAM_ANG / (1.0 - x)


# ── the code's cross-section against the formula ──────────────────────────────


@pytest.mark.parametrize("x", [1e-4, 1e-3, 1e-2, 3e-2, 1e-1])
def test_code_cross_section_equals_lorentzian_formula(x):
    """tau/N_HI from the code equals the formula to 1e-11 (same A, c and lam_a)."""
    z = 8.0
    rest = _rest_at_frac(x)
    tau = float(_cgm_damping_wing_tau(jnp.asarray([rest * (1.0 + z)]), z)[0])
    np.testing.assert_allclose(tau / _n_hi(z), _sigma_formula(rest), rtol=1e-11)


# ── sum rule ───────────────────────────────────────────────────────────────────


def test_sum_rule_identity():
    """3 lam^2 A/(8 pi) = pi e^2 f/(m_e c) to 1e-3: why no second f belongs there."""
    np.testing.assert_allclose(_PREFACTOR, _SUM_RULE, rtol=1e-3)


@pytest.mark.parametrize("x", [1e-3, 1e-2])
def test_code_prefactor_equals_sum_rule(x):
    """The code's prefactor, recovered from its tau, equals pi e^2 f/(m_e c) to 1e-3."""
    z = 8.0
    rest = _rest_at_frac(x)
    nu = _C / (rest * 1e-8)
    r = nu / _NU_A
    tau = float(_cgm_damping_wing_tau(jnp.asarray([rest * (1.0 + z)]), z)[0])
    denominator = 4.0 * np.pi**2 * (nu - _NU_A) ** 2 + _A**2 * r**6 / 4.0
    numerator = _A * r**4
    recovered = tau * denominator / numerator / _n_hi(z)
    np.testing.assert_allclose(recovered, _SUM_RULE, rtol=1e-3)


# ── public transmission ──────────────────────────────────────────────────────


@pytest.mark.parametrize("z", [6.0, 8.0, 10.0])
@pytest.mark.parametrize("dv", [1000.0, 2000.0, 5000.0])
def test_transmission_is_inoue14_times_wing(z, dv):
    """T_asada25 = T_inoue14 * exp(-N_HI sigma_formula) to 1e-3, redward of the line."""
    rest = _rest_at_velocity(dv)
    wave_obs = jnp.asarray([rest * (1.0 + z)])
    expected = float(igm_transmission(wave_obs, z)[0]) * np.exp(
        -_n_hi(z) * _sigma_formula(rest)[()]
    )
    got = float(igm_transmission_asada25(wave_obs, z)[0])
    np.testing.assert_allclose(got, expected, rtol=1e-3)


@pytest.mark.parametrize(
    "z, rest, literal, literal_rtol",
    [
        # 0.00073 is 0.000734 to two significant figures: half a unit in the last digit
        # is 0.5/73 = 0.7 %, so the literal is pinned at 1e-2 (the formula value, at 1e-3).
        (8.0, 1220.0, 0.00073, 1e-2),
        (8.0, 1230.0, 0.523, 2e-3),
        (8.0, 1240.0, 0.8015, 2e-3),
        (10.0, 1225.0, 0.1527, 2e-3),
    ],
)
def test_issue_literals(z, rest, literal, literal_rtol):
    """Four tabulated transmissions: formula value at 1e-3, printed literal at its precision."""
    wave_obs = jnp.asarray([rest * (1.0 + z)])
    got = float(igm_transmission_asada25(wave_obs, z)[0])
    t_formula = float(igm_transmission(wave_obs, z)[0]) * np.exp(
        -_n_hi(z) * _sigma_formula(rest)[()]
    )
    np.testing.assert_allclose(got, t_formula, rtol=1e-3)
    np.testing.assert_allclose(got, literal, rtol=literal_rtol)


# ── CGM Lorentzian wing against the DLA Voigt wing ────────────────────────────


@pytest.mark.parametrize(
    "dv, measured_ratio",
    [(3000.0, 0.9449), (10000.0, 0.8297)],
)
def test_cgm_to_dla_wing_ratio(dv, measured_ratio):
    """sigma_CGM / sigma_DLA(T = 1e4 K) at the same column, to the measured ratio +- 1 %.

    With the double f the ratio was 0.41 x these (0.39 / 0.34 near 3000 / 10000 km/s), so
    a double-f cross-section fails it. The residual from unity is the line-shape difference,
    not f: the CGM form carries the Rayleigh factor r^4 = (nu/nu_a)^4 (0.96 at 3000 km/s,
    0.875 at 10000 km/s) while the DLA form carries the Lee (2013) asymmetry factor
    1 - 1.792 x dnu_D/nu_a, which is 1.018 / 1.060 redward; 0.96/1.018 = 0.943 and
    0.875/1.060 = 0.826. What is left (about 0.2 %) is the 4-digit atomic data
    (f = 0.4162 in ``dla.py`` against 0.4164, A, lam_a) and the Voigt core term.
    """
    z = 8.0
    rest = _rest_at_velocity(dv)
    tau = float(_cgm_damping_wing_tau(jnp.asarray([rest * (1.0 + z)]), z)[0])
    sigma_cgm = tau / _n_hi(z)
    dnu_d = float(_deltanu_doppler(1e4, 0.0))
    x = (_C / (rest * 1e-8) - _NU_A) / dnu_d
    sigma_dla = float(_sigma_lya(jnp.asarray([x]), 1e4, 0.0)[0])
    np.testing.assert_allclose(sigma_cgm / sigma_dla, measured_ratio, rtol=1e-2)


# ── public path: photometry through a model built with the asada25 IGM ─────────

_Z = 8.0
_BAND = (1220.0 * (1.0 + _Z), 1260.0 * (1.0 + _Z))  # observed Angstrom, rest 1220-1260


def _ssp():
    """Smooth synthetic SSP; flux scaled so L_nu stays below float32 overflow."""
    wave = jnp.logspace(2.0, 7.0, 1600)
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=(jnp.abs(flux) + 1e-12) * 1e-16,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
    )


def _filter():
    wave = jnp.linspace(_BAND[0], _BAND[1], 41)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, 41))
    return wave, trans


def _build(igm, *, free_z=False):
    wave, trans = _filter()
    obs = Observation(
        photometry=Photometry(filters=(FilterCurve(wave=wave, trans=trans, name="lya_red"),))
    )
    return SEDModel.build(
        ssp_data=_ssp(),
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Uniform(7.5, 8.5) if free_z else Fixed(_Z),
        igm={"type": igm},
    )


def _band_flux(model, key=1):
    params = model.spec.sample(jax.random.PRNGKey(key))
    return params, model.predict(params)


def test_photometry_matches_band_average_of_inoue_times_wing():
    """pred.photometry() of the asada25 model equals the test's band average to 2e-3.

    The test's average: the Inoue+2014 model's observed SED times exp(-N_HI sigma_formula)
    on the model wavelength grid, weighted by transmission/lambda^2 (the default Bessell
    filter convention), over the same top-hat filter; the denominator is the same average
    without the wing. Measured agreement is below 1e-3 (the grid has only seven nodes in
    the band, so the quadrature differences are the residual).
    """
    model_inoue = _build("inoue14")
    model_asada = _build("asada25")
    params, pred_inoue = _band_flux(model_inoue)
    pred_asada = model_asada.predict(params)

    lam = np.asarray(pred_inoue.wave_obs)
    sed = np.asarray(pred_inoue.obs_sed())
    fwave, ftrans = (np.asarray(a) for a in _filter())
    weight = np.interp(lam, fwave, ftrans, left=0.0, right=0.0) / lam**2

    tau = _n_hi(_Z) * _sigma_formula(lam / (1.0 + _Z))
    tau = np.where(lam > _LAM_ANG * (1.0 + _Z), tau, 0.0)
    expected = np.trapezoid(sed * np.exp(-tau) * weight, lam) / np.trapezoid(sed * weight, lam)

    measured = float(pred_asada.photometry()[0] / pred_inoue.photometry()[0])
    assert 0.3 < expected < 0.95  # a real attenuation, so the comparison is not vacuous
    np.testing.assert_allclose(measured, expected, rtol=2e-3)


def test_float32_agrees_with_float64():
    """The asada25 band flux in float32 equals float64 to 1e-4, and is float32."""
    model64 = _build("asada25")
    params64, pred64 = _band_flux(model64)
    flux64 = float(pred64.photometry()[0])
    with jax.enable_x64(False):
        model32 = _build("asada25")
        # Same parameter values as the float64 run (a float32 PRNG draw is a different sample).
        params32 = {k: jnp.asarray(np.asarray(v)) for k, v in params64.items()}
        phot32 = model32.predict(params32).photometry()
        assert phot32.dtype == jnp.float32
        flux32 = float(phot32[0])
    np.testing.assert_allclose(flux32, flux64, rtol=1e-4)


def test_redshift_gradient_is_finite_and_nonzero():
    """d(band flux)/dz is finite and non-zero, and matches a central difference."""
    model = _build("asada25", free_z=True)
    params = {**model.spec.sample(jax.random.PRNGKey(1)), "redshift": jnp.asarray(_Z)}

    def band_flux(z):
        return model.predict({**params, "redshift": z}).photometry()[0]

    grad = float(jax.grad(band_flux)(jnp.asarray(_Z)))
    h = 1e-3
    fd = float(band_flux(jnp.asarray(_Z + h)) - band_flux(jnp.asarray(_Z - h))) / (2.0 * h)
    assert np.isfinite(grad)
    assert grad != 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-2)


# ── float32 autodiff through the wing ───────────────────────────────────────────


def _tau_and_transmission(z, wave_obs):
    return (
        _cgm_damping_wing_tau(wave_obs, z)[0],
        igm_transmission_asada25(wave_obs, z)[0],
    )


@pytest.mark.parametrize("output", [0, 1], ids=["tau", "transmission"])
def test_float32_redshift_gradient_matches_float64_finite_difference(output):
    """d/dz at fixed observed wavelength (rest 1225 Angstrom at z = 8), float32 AD vs float64 FD.

    Differentiating with the observed wavelength held fixed moves the rest-frame offset from
    Lyα, so the backward pass goes through (nu - nu_a)^2 ~ 1e26 squared once more in the
    denominator derivative; that term overflows float32 unless the offset is carried as
    x = (nu - nu_a)/nu_a.
    """
    wave_obs = jnp.asarray([1225.0 * (1.0 + _Z)])

    def f(z):
        return _tau_and_transmission(z, wave_obs)[output]

    h = 1e-4
    fd64 = (float(f(_Z + h)) - float(f(_Z - h))) / (2.0 * h)
    with jax.enable_x64(False):
        z32 = jnp.asarray(_Z, dtype=jnp.float32)
        value32 = f(z32)
        grad32 = jax.grad(f)(z32)
        assert value32.dtype == jnp.float32
        assert grad32.dtype == jnp.float32
    np.testing.assert_allclose(float(grad32), fd64, rtol=1e-3)
