# SPDX-License-Identifier: BSD-3-Clause
"""Model-centred emission-line prior: unit mapping, normalization and the exact limits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.eline_marginalization import (
    ModelLinePrior,
    build_eline_design_matrix,
    match_model_lines,
    model_line_prior,
    unit_model_line_prior,
)
from tengri.observation.linear_marginal import marginalize_linear_nuisance
from tengri.utils.cosmology import luminosity_distance

pytestmark = pytest.mark.contract

_HBETA_REST_AA = 4862.68
_Z = 0.1
_PARAMS = {"sfh_delayed_log_total_mass": 10.0, "sfh_delayed_tau_gyr": 1.0}
_WAVE = np.linspace(4800.0, 5400.0, 1201)  # observed frame, 0.5 A pixels
_RESOLUTION = 2000.0


def _fine_ssp(ssp):
    """The fixture's age and metallicity axes on a 1 A rest grid that resolves the lines."""
    wave = np.arange(100.0, 10000.0 + 0.5, 1.0)
    shape = (ssp.ssp_lgmet.shape[0], ssp.ssp_lg_age_gyr.shape[0], wave.size)
    flux = np.broadcast_to((wave / 5000.0) ** -1.0, shape) * 1.0e-17
    return SSPData(
        ssp_wave=jnp.asarray(wave),
        ssp_flux=jnp.asarray(flux),
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )


def _model(ssp, nebular_lines):
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_WAVE),
        resolution=_RESOLUTION,
        nebular_lines=nebular_lines,
    )
    return SEDModel.build(
        ssp_data=_fine_ssp(ssp),
        observation=Observation(spectroscopy=spectroscopy),
        redshift=Fixed(_Z),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": Uniform(0.5, 3.0),
            "age_gyr": Fixed(5.0),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "publish_line_fraction": True},
    )


def test_hbeta_mapping_matches_spectrum_line_flux(synthetic_ssp_wide):
    """The helper's mu_Hbeta equals the lines-on minus lines-off integral to 1 %."""
    with jax.enable_x64(True):
        on = _model(synthetic_ssp_wide, True)
        off = _model(synthetic_ssp_wide, False)
        f_on = np.asarray(on.predict_spectrum(_PARAMS), dtype=np.float64)
        f_off = np.asarray(off.predict_spectrum(_PARAMS), dtype=np.float64)
        derived = on.predict_state(_PARAMS).derived
        waves = np.asarray(derived["line_waves"], dtype=np.float64)
        log_att = jnp.asarray(derived["log_line_lums_attenuated"], dtype=jnp.float64)
        dl_cm = float(luminosity_distance(_Z))
        index, matched = match_model_lines(waves, [_HBETA_REST_AA])
        prior = model_line_prior(
            log_att, index, matched, np.array([_HBETA_REST_AA]), _Z, dl_cm, 1.0, 0.3
        )
    assert bool(matched[0])
    mu = float(prior.prior_mean[0])
    lam_obs = _HBETA_REST_AA * (1.0 + _Z)
    sigma_obs = lam_obs / (2.355 * _RESOLUTION)
    win = np.abs(_WAVE - lam_obs) <= 5.0 * sigma_obs
    integral = np.trapezoid((f_on - f_off)[win], _WAVE[win])
    ratio = integral / mu
    print(f"mu={mu:.6e} integral={integral:.6e} ratio={ratio:.6f}")
    assert ratio == pytest.approx(1.0, rel=1e-2)


def _toy_problem(seed=0):
    """Two Gaussian lines on a 60-pixel grid with a positive model amplitude each."""
    rng = np.random.default_rng(seed)
    wave = np.linspace(4800.0, 5400.0, 60)
    catalog = np.array([4900.0, 5100.0])
    design = np.asarray(
        build_eline_design_matrix(jnp.asarray(wave), jnp.asarray(catalog), 500.0, 0.0)
    )
    sigma = np.full(wave.size, 0.05)
    data = 0.8 * design[:, 0] * 10.0 + 0.4 * design[:, 1] * 10.0 + rng.normal(0.0, 0.05, wave.size)
    return data, design, sigma


def _marginal(data, design, sigma, prior):
    unit = unit_model_line_prior(jnp.asarray(design), prior, noise=jnp.asarray(sigma))
    return marginalize_linear_nuisance(
        jnp.asarray(data),
        jnp.zeros(design.shape[0]),
        unit.design,
        sigma=jnp.asarray(sigma),
        prior_mean=unit.prior_mean,
        prior_precision=unit.prior_precision,
        scale=unit.scale,
    ).log_like


def test_zero_luminosity_line_equals_deleted_column():
    """A line forced to zero luminosity contributes exactly nothing: value = deleted column."""
    data, design, sigma = _toy_problem()
    log_lums = jnp.array([np.log10(3.0e-3), -jnp.inf])
    index, matched = match_model_lines(np.array([4900.0, 5100.0]), [4900.0, 5100.0])
    lam = np.array([4900.0, 5100.0])
    full = model_line_prior(log_lums, index, matched, lam, 0.0, 1.0, 1.0, 0.3)
    assert float(full.prior_mean[1]) == 0.0
    single = ModelLinePrior(full.prior_mean[:1], full.prior_width[:1], full.gaussian[:1])
    with_zero = _marginal(data, design, sigma, full)
    deleted = _marginal(data, design[:, :1], sigma, single)
    np.testing.assert_allclose(float(with_zero), float(deleted), rtol=1e-10, atol=0)


def test_flat_line_gets_zero_precision_and_unit_width():
    """A catalog line with no model counterpart is flat: zero precision, unit width."""
    log_lums = jnp.array([np.log10(3.0e-3)])
    index, matched = match_model_lines(np.array([4900.0]), [4900.0, 9999.0])
    prior = model_line_prior(
        log_lums, index, matched, np.array([4900.0, 9999.0]), 0.0, 1.0, 1.0, 0.3
    )
    assert list(np.asarray(prior.gaussian)) == [1.0, 0.0]
    assert float(prior.prior_width[1]) == 1.0
    assert float(prior.prior_mean[1]) == 0.0


def test_normalization_offset_is_minus_log_width_per_gaussian_line():
    """Gaussian minus flat at the same design equals -ln W for a line with W*gg >> 1.

    The primitive has no (2 pi)^(-1/2) per Gaussian prior normalization, so the offset
    is -ln W, not -ln(sqrt(2 pi) W). The residual is O(1 / (W^2 g^T g)).
    """
    data, design, sigma = _toy_problem()
    sigma = np.full(design.shape[0], 0.002)
    mu = jnp.array([8.0e-3, 4.0e-3])
    data = np.asarray(design) @ np.asarray(mu)
    width = (10.0**5.0 - 1.0) * mu
    gauss = ModelLinePrior(mu, width, jnp.array([1.0, 1.0]))
    flat = ModelLinePrior(jnp.zeros(2), jnp.ones(2), jnp.zeros(2))
    offset = _marginal(data, design, sigma, gauss) - _marginal(data, design, sigma, flat)
    expected = -float(np.sum(np.log(np.asarray(width))))
    np.testing.assert_allclose(float(offset), expected, rtol=1e-6, atol=0)


def test_aperture_fraction_gradient_is_finite_and_nonzero():
    """The aperture factor moves the prior mean: d(marginal)/d(f_ap) is finite and nonzero."""
    data, design, sigma = _toy_problem()
    lums = jnp.array([np.log10(3.0e-3), np.log10(1.0e-3)])
    index, matched = match_model_lines(np.array([4900.0, 5100.0]), [4900.0, 5100.0])
    lam = np.array([4900.0, 5100.0])

    def value(aperture):
        prior = model_line_prior(lums, index, matched, lam, 0.0, 1.0, aperture, 0.3)
        return _marginal(data, design, sigma, prior)

    grad = float(jax.grad(value)(jnp.asarray(0.7)))
    assert np.isfinite(grad)
    assert grad != 0.0
