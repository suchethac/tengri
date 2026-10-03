# SPDX-License-Identifier: BSD-3-Clause
"""Line-flux upper and lower limits are scored as censored likelihoods (#2665, #2666).

Per line, with sigma the flux uncertainty, F the tabulated flux (the limit
value for a limit) and m the model flux: a detection scores
``ln L = -0.5 ((F - m)/sigma)^2 - ln sigma - 0.5 ln 2 pi``, an upper limit
``ln L = ln Phi((F - m)/sigma)`` and a lower limit
``ln L = ln Phi((m - F)/sigma)``, with Phi evaluated by a log-CDF (no floor,
no clamp). ``LineFluxData.log_likelihood`` and the line-flux term of the
joint/spectroscopy loss must both follow these forms. Every expected value is
computed here with numpy/scipy in float64.

https://github.com/suchethac/tengri/issues/2665
https://github.com/suchethac/tengri/issues/2666
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

import tengri
from tengri import DEFAULT, Fitter, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.regression_bug

_NAMES = ("Ha", "Hb")
_WAVES = np.array([6564.61, 4862.68])
_FLUX = 1.0
_SIGMA = 0.1
_KIND_FLAGS = {
    "detection": {},
    "upper": {"is_upper_limit": np.array([False, True])},
    "lower": {"is_lower_limit": np.array([False, True])},
}


def _detection_norm(sigma):
    """Normalization -ln sigma - 0.5 ln 2 pi of a detection with zero residual."""
    return -np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


def _container(kind, dtype=None):
    """Two-line container: line 1 a detection, line 2 of the given kind."""
    return LineFluxData(
        names=_NAMES,
        fluxes=jnp.full(2, _FLUX, dtype=dtype),
        errors=jnp.full(2, _SIGMA, dtype=dtype),
        wavelengths=jnp.asarray(_WAVES, dtype=dtype),
        **{k: jnp.asarray(v) for k, v in _KIND_FLAGS[kind].items()},
    )


def _line2_closed_form(kind, flux, model, sigma):
    """Closed-form ln L of one line from float64 numpy scalars."""
    r = (flux - model) / sigma
    if kind == "detection":
        return -0.5 * r**2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
    if kind == "upper":
        return special.log_ndtr(r)
    return special.log_ndtr(-r)


@pytest.mark.parametrize("kind", ["detection", "upper", "lower"])
@pytest.mark.parametrize("d", [-3.0, 0.0, 3.0])
def test_log_likelihood_matches_closed_form(kind, d):
    """Cell (a): ln L equals the closed form to 1e-10 for model line 2 at d = (m - F)/sigma."""
    model = np.array([_FLUX, _FLUX + d * _SIGMA])
    expected = _detection_norm(_SIGMA) + _line2_closed_form(kind, _FLUX, model[1], _SIGMA)
    got = float(_container(kind).log_likelihood(jnp.asarray(model)))
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("z", [-3.0, 3.0])
def test_lower_limit_gradient(z):
    """Cell (b): d ln L / dm = +phi(z)/Phi(z)/sigma, z = (m - F)/sigma, violated and satisfied."""
    lfd = _container("lower")
    m = _FLUX + z * _SIGMA
    z_np = (m - _FLUX) / _SIGMA
    expected = np.exp(stats.norm.logpdf(z_np) - special.log_ndtr(z_np)) / _SIGMA

    def total(x):
        return lfd.log_likelihood(jnp.stack([jnp.asarray(_FLUX), x]))

    grad = float(jax.grad(total)(jnp.asarray(m)))
    assert expected > 0.0
    assert grad == pytest.approx(expected, rel=1e-8)


def test_lower_limit_value_increases_when_satisfied():
    """Cell (b): the lower-limit value at m = F + 3 sigma exceeds the value at m = F - 3 sigma."""
    lfd = _container("lower")
    satisfied = float(lfd.log_likelihood(jnp.array([_FLUX, _FLUX + 3 * _SIGMA])))
    violated = float(lfd.log_likelihood(jnp.array([_FLUX, _FLUX - 3 * _SIGMA])))
    assert satisfied > violated


@pytest.mark.parametrize("z", [-15.0, -30.0, -100.0])
def test_upper_limit_strongly_violated_value_and_gradient(z):
    """Cell (c): ln L = log_ndtr(z) (rel 1e-10); d ln L / dm = -phi/Phi/sigma (rel 1e-8)."""
    lfd = _container("upper")
    m = _FLUX - z * _SIGMA
    z_np = (_FLUX - m) / _SIGMA
    norm = _detection_norm(_SIGMA)

    def second_line(x):
        return lfd.log_likelihood(jnp.stack([jnp.asarray(_FLUX), x])) - norm

    value = float(second_line(jnp.asarray(m)))
    assert value == pytest.approx(float(special.log_ndtr(z_np)), rel=1e-10)
    grad = float(jax.grad(second_line)(jnp.asarray(m)))
    expected = -np.exp(stats.norm.logpdf(z_np) - special.log_ndtr(z_np)) / _SIGMA
    assert np.isfinite(grad), (
        "a non-finite gradient would mean the log-CDF was evaluated outside its stable range"
    )
    assert grad != 0.0
    assert grad == pytest.approx(expected, rel=1e-8)


def test_chi2_sums_detections_only():
    """Cell (d): chi2 of a detection, an upper and a lower limit is the detection's alone."""
    fluxes = np.array([1.0, 1.0, 1.0])
    model = np.array([1.2, 1.3, 0.8])
    lfd = LineFluxData(
        names=("Ha", "Hb", "Hg"),
        fluxes=jnp.asarray(fluxes),
        errors=jnp.full(3, _SIGMA),
        wavelengths=jnp.array([6564.61, 4862.68, 4341.68]),
        is_upper_limit=jnp.array([False, True, False]),
        is_lower_limit=jnp.array([False, False, True]),
    )
    expected = ((fluxes[0] - model[0]) / _SIGMA) ** 2
    assert float(lfd.chi2(jnp.asarray(model))) == pytest.approx(expected, rel=1e-10)


# -- Cell (e): joint/spectroscopy path ---------------------------------

_LOG_TOTAL_MASS = 11.8
_LINE_WAVES = jnp.array([6564.61, 4862.71])
_LINE_ERR_FRAC = 0.05


@functools.lru_cache(maxsize=1)
def _joint_setup():
    ssp = tengri.load_ssp()
    curves = tuple(
        FilterCurve(
            wave=np.linspace(c - 500, c + 500, 32),
            trans=np.ones(32),
            name=f"b{i}",
        )
        for i, c in enumerate([4000.0, 5500.0, 7000.0, 9000.0])
    )
    phot = Photometry(filters=curves)
    spec = Spectroscopy(wave_obs=np.linspace(4000.0, 9000.0, 40))

    def build(lfd):
        return SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=phot, spectroscopy=spec, line_fluxes=lfd),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(8.0, 12.0),
            },
            dust_attenuation={"type": "none"},
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )

    params = {"sfh_dpl_log_total_mass": _LOG_TOTAL_MASS}
    base = build(None)
    model_lf = np.asarray(base.predict_line_fluxes(params, target_wavelengths=_LINE_WAVES))
    data = np.concatenate(
        [np.asarray(base.predict_photometry(params)), np.asarray(base.predict_spectrum(params))]
    )
    return build, model_lf, data


@functools.cache
def _joint_energy(kind, scale):
    """Joint-path loss with line 1 at its model flux and line 2 at ``scale`` times its.

    ``kind`` is "detection", "upper" or "lower" (the flag on line 2). Returns
    the loss, the ``data_args`` keys, and the line fluxes that
    ``predict_line_fluxes`` returns on the Fitter's own model at the evaluated
    parameters (the model the loss scores the lines against).
    """
    build, base_lf, data = _joint_setup()
    lfd = LineFluxData(
        names=("Halpha", "Hbeta"),
        fluxes=jnp.asarray(base_lf * np.array([1.0, scale])),
        errors=jnp.asarray(np.abs(base_lf) * _LINE_ERR_FRAC),
        wavelengths=_LINE_WAVES,
        **{k: jnp.asarray(v) for k, v in _KIND_FLAGS[kind].items()},
    )
    model = build(lfd)
    fitter = Fitter(
        model,
        data=data,
        noise=0.05 * np.abs(data),
        data_type="joint",
        data_mask=np.zeros(data.size, int),
    )
    loss_fn = fitter._get_or_build_loss_fn()
    data_args = fitter._build_data_args(model)
    u = fitter._initialize_unbounded(jax.random.PRNGKey(0))
    key = next(k for k in u if "log_total_mass" in k)
    u = {
        **u,
        key: fitter.spec.get_distribution(key).standardize(jnp.asarray(_LOG_TOTAL_MASS)),
    }
    energy = float(loss_fn(u, data_args))
    # The Fitter resolves its own approximation on the model it scores
    # against; its ``predict_line_fluxes`` returns the fluxes the loss compares
    # with the data.
    model_lf = np.asarray(
        fitter.model.predict_line_fluxes(
            {"sfh_dpl_log_total_mass": _LOG_TOTAL_MASS}, target_wavelengths=_LINE_WAVES
        )
    )
    return energy, frozenset(data_args), model_lf


def _residuals(scale, model_lf):
    """(F - m)/sigma per line for the observed fluxes of ``_joint_energy``."""
    _, base_lf, _ = _joint_setup()
    sigma = np.abs(base_lf) * _LINE_ERR_FRAC
    return (base_lf * np.array([1.0, scale]) - model_lf) / sigma, sigma


@pytest.mark.parametrize("kind", ["upper", "lower"])
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_joint_path_limit_energy_difference(kind, scale):
    """Cell (e): energy(kind) - energy(detection) equals the closed form to 1e-8 relative.

    The detection energy is the inlined 0.5 r^2 per line; the censored energy
    adds ln sigma_1 for detected line 1 (it carries ln sigma, the inlined
    chi-square does not) and, for limit line 2, -ln Phi(z) - 0.5 r_2^2, with
    r_2 = (F_2 - m_2)/sigma_2 and z = r_2 (upper) or -r_2 (lower).
    """
    e_kind, keys, model_lf = _joint_energy(kind, scale)
    e_det, _, _ = _joint_energy("detection", scale)
    r, sigma = _residuals(scale, model_lf)
    z = r[1] if kind == "upper" else -r[1]
    expected = np.log(sigma[0]) - special.log_ndtr(z) - 0.5 * r[1] ** 2
    assert "line_flux_limit_mask" in keys
    assert (e_kind - e_det) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_joint_path_energies_differ(scale):
    """Cell (e): detection, upper-limit and lower-limit energies are three different values."""
    energies = {k: _joint_energy(k, scale)[0] for k in ("detection", "upper", "lower")}
    assert len(set(energies.values())) == 3


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_joint_path_without_limits_is_inlined_chi_square(scale):
    """Cell (e): with no limit flags the energy moves by 0.5 sum r^2 relative to scale 1.

    Both energies run through the inlined chi-square line term, so their
    difference is 0.5 (sum r(scale)^2 - sum r(1)^2) computed from the model
    line fluxes.
    """
    e_s, keys, lf_s = _joint_energy("detection", scale)
    e_1, _, lf_1 = _joint_energy("detection", 1.0)
    chi2_s = np.sum(_residuals(scale, lf_s)[0] ** 2)
    chi2_1 = np.sum(_residuals(1.0, lf_1)[0] ** 2)
    assert "line_flux_limit_mask" not in keys
    assert (e_s - e_1) == pytest.approx(0.5 * (chi2_s - chi2_1), rel=1e-8)


# -- Cell (f): float32 -------------------------------------------------


@pytest.mark.parametrize("kind", ["upper", "lower"])
@pytest.mark.parametrize("z", [-3.0, 3.0])
def test_float32_limit_value_matches_float64_closed_form(kind, z):
    """Cell (f): float32 ln L is within four float32 ulp of the float64 closed form.

    The tolerance is four float32 ulp of the expected value (the float32
    evaluation of Phi and of the sum of the two line terms each round at the
    ulp level); the closed form uses the float32 inputs actually passed in.
    """
    sign = -1.0 if kind == "upper" else 1.0
    with jax.enable_x64(False):
        lfd = _container(kind, dtype=jnp.float32)
        model = jnp.array([_FLUX, _FLUX + sign * z * _SIGMA], dtype=jnp.float32)
        assert model.dtype == jnp.float32
        got = float(lfd.log_likelihood(model))
        f, m, s = (np.asarray(v, dtype=np.float64) for v in (lfd.fluxes, model, lfd.errors))
    expected = _detection_norm(s[0]) + _line2_closed_form(kind, f[1], m[1], s[1])
    atol = 4 * float(np.spacing(np.float32(abs(expected))))
    assert got == pytest.approx(expected, rel=0.0, abs=atol)


def test_float32_strongly_violated_upper_limit_is_finite_with_gradient():
    """Cell (f): at z = -30 float32 gives log_ndtr(-30) (rel 1e-5), finite gradient."""
    with jax.enable_x64(False):
        lfd = _container("upper", dtype=jnp.float32)
        m = jnp.float32(_FLUX + 30.0 * _SIGMA)

        def total(x):
            return lfd.log_likelihood(jnp.stack([jnp.float32(_FLUX), x]))

        value = float(total(m))
        grad = float(jax.grad(total)(m))
        f, s = (np.asarray(v, dtype=np.float64) for v in (lfd.fluxes, lfd.errors))
        m64 = float(m)
    z_np = (f[1] - m64) / s[1]
    assert np.isfinite(value)
    assert (value - _detection_norm(s[0])) == pytest.approx(
        float(special.log_ndtr(z_np)), rel=1e-5
    )
    assert np.isfinite(grad)
    assert grad != 0.0
